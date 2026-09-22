"""Plan an NWB file's ingestion without inserting any of it.

Parses every ingestion table against one open file, checks the result for the
integrity failures that would otherwise surface mid-transaction, and reports
what it found -- including whether any of it is new.

The check that matters most is prospective: an entry's parent must exist
either in the database *or* elsewhere in the same plan. Without that, tables
that emit a parent's rows alongside their own would appear broken, and one
missing object would be reported once per dependent table rather than once.
"""

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from spyglass.data_import.ingestion_plan import (
    IngestionPlan,
    PlannedEntries,
    Problem,
    ReportResult,
    TablePlan,
    row_key,
)
from spyglass.data_import.plan_cache import (
    load_plan,
    plan_provenance,
    save_plan,
)
from spyglass.utils.logging import logger


class VirtualKeySpace:
    """The primary keys a table would hold once a plan is inserted.

    `existing ∪ planned`: the union of what the database holds now and what
    the plan intends to add. Lazily read, and never written -- the only
    database access the planning pass makes.
    """

    def __init__(self):
        self._existing: Dict[str, set] = {}
        self._planned: Dict[str, set] = {}
        self._planned_rows: Dict[str, Dict[tuple, dict]] = {}

    def existing_keys(self, table) -> set:
        """Return the primary keys this table already holds.

        Takes an instance -- see `_as_table`, which normalizes once at the
        boundary rather than in every method here.
        """
        name = table.full_table_name

        if name not in self._existing:
            try:
                self._existing[name] = {
                    row_key(table, key)
                    for key in table.fetch("KEY", as_dict=True)
                }
            except Exception as err:  # table not declared, or unreadable
                logger.debug(f"Could not read keys for {name}: {err}")
                self._existing[name] = set()

        return self._existing[name]

    def add_planned(self, table, rows) -> None:
        """Record rows a plan intends to insert into a table."""
        name = table.full_table_name
        planned = self._planned.setdefault(name, set())
        by_key = self._planned_rows.setdefault(name, {})

        for row in rows:
            key = row_key(table, row)
            planned.add(key)
            by_key.setdefault(key, row)

    def holds(self, table, key: tuple) -> bool:
        """Whether a key is present, in the database or in the plan."""
        return key in self.existing_keys(table) or key in self._planned.get(
            table.full_table_name, set()
        )

    def planned_keys(self, table) -> set:
        """Return the keys planned for a table in this pass."""
        return self._planned.get(table.full_table_name, set())


def _as_table(target):
    """Return a plan target as a table instance.

    Ingestion targets are Spyglass tables, which answer `as_instance`
    whichever side they are asked from. A `FreeTable` -- how DataJoint hands
    back a parent, and how config-declared part entries are keyed -- carries
    no such descriptor but is already an instance.

    Normalizing here, once, keeps every function below able to assume it was
    handed an instance.
    """
    return getattr(target, "as_instance", target)


def plan_nwbfile(
    nwb_file_name: str,
    config: dict = None,
    tables: Optional[List] = None,
    use_cache: bool = False,
) -> IngestionPlan:
    """Plan the ingestion of one NWB file, writing nothing.

    Parsing is the expensive half of ingestion and depends only on the file,
    the config and the Spyglass version. The *verdict* does not: novelty,
    divergence and foreign-key resolution are all read from the database, so
    a plan describes a moment in it.

    `use_cache` therefore defaults to False. Opt in only when the database
    cannot have changed since the plan was built -- planning to read the
    report, then planning again to insert what it approved. Ingest anything
    in between and the cached verdict is stale: it will still report those
    entries as novel. Caching the parse alone, and re-checking against the
    database every time, is the fix; it is not what this does yet.

    Parameters
    ----------
    nwb_file_name : str
        The copy file registered in Nwbfile.
    config : dict, optional
        Per-table config, as `populate_all_common` assembles it.
    tables : list, optional
        Tables to plan. Default: every ingestion table, in dependency order.
    use_cache : bool, optional
        Read and write the plan cache. Default False; see above for when it
        is safe. A subset of `tables` does not describe the whole file, so
        those runs are never cached.

    Returns
    -------
    IngestionPlan
        Entries, problems, and the novelty verdict for the whole file.
    """
    from spyglass.common.common_nwbfile import Nwbfile

    config = config or dict()
    nwb_key = {"nwb_file_name": nwb_file_name}

    # Only a whole-file plan is cacheable: one built for some tables would be
    # served later as though it covered all of them.
    cacheable = use_cache and tables is None
    nwb_hash = config_hash = version = None
    if cacheable:
        nwb_hash, config_hash, version = plan_provenance(
            Nwbfile.get_abs_path(nwb_file_name), config
        )
        cached = load_plan(nwb_file_name, nwb_hash, config_hash, version)
        if cached is not None:
            return cached

    if not (query := Nwbfile & nwb_key):
        return IngestionPlan(
            nwb_file_name=nwb_file_name,
            fatal=(
                Problem(
                    severity="fatal",
                    code="file_not_registered",
                    message=f"{nwb_file_name} is not in the Nwbfile table",
                ),
            ),
        )

    try:  # one open file, shared by every table
        nwb_file = query.fetch_nwb()[0]
    except Exception as err:
        return IngestionPlan(
            nwb_file_name=nwb_file_name,
            fatal=(
                Problem(
                    severity="fatal",
                    code="file_unreadable",
                    message=str(err),
                    exc_type=type(err).__name__,
                ),
            ),
        )

    if tables is None:
        from spyglass.common.populate_all_common import ingestion_table_list

        tables = ingestion_table_list()

    # One hashing pass over the file yields a digest per object, so each
    # table's read-set can be fingerprinted without re-reading anything.
    # Roughly 14x cheaper than parsing the same file, so it earns its place
    # even when nothing turns out to be reusable.
    hasher = _object_hasher(nwb_file_name)

    key_space = VirtualKeySpace()
    table_plans: List[TablePlan] = []
    failed_tables: set = set()
    novel: Dict[str, int] = {}

    for table in tables:
        instance = table.as_instance
        table_config = config.get(instance.camel_name, dict())

        blocked_by = _blocking_parents(instance, failed_tables)
        if blocked_by:
            table_plans.append(
                TablePlan(
                    table_name=instance.full_table_name,
                    entries=PlannedEntries().freeze(),
                    status="blocked",
                    problems=(),
                )
            )
            continue

        plan = instance.plan_from_nwbfile(
            nwb_file_name, config=table_config, nwb_file=nwb_file
        )

        problems = list(plan.problems)
        for target, rows in plan.entries:
            table = _as_table(target)  # the one normalization point
            name = table.full_table_name

            # The table that planned the rows answers for them, whichever
            # table they are destined for -- a target may be an ordinary
            # table, or a FreeTable part declared through config.
            problems.extend(
                instance.check_planned_rows(rows, key_space, table=table)
            )

            # Counted before the rows join the key space, so "new" means new
            # to the database rather than new to this loop.
            existing = key_space.existing_keys(table)
            novel[name] = novel.get(name, 0) + sum(
                1 for row in rows if row_key(table, row) not in existing
            )
            key_space.add_planned(table, rows)

        status = plan.status
        if any(problem.severity in ("hard", "fatal") for problem in problems):
            status = "failed"
            failed_tables.add(instance.full_table_name)

        table_plans.append(
            TablePlan(
                table_name=plan.table_name,
                entries=plan.entries,
                status=status,
                problems=tuple(problems),
                reads=plan.reads,
                read_set_digest=(
                    hasher.read_set_digest(plan.reads) if hasher else None
                ),
            )
        )

    plan = IngestionPlan(
        nwb_file_name=nwb_file_name,
        table_plans=tuple(table_plans),
        novel=novel,
        nwb_hash=nwb_hash,
        config_hash=config_hash,
        spyglass_version=version,
    )

    if cacheable:
        save_plan(plan)

    return plan


def _object_hasher(nwb_file_name: str):
    """Return a hasher indexing the file's objects, or None.

    A file that cannot be hashed is not a failure: the plan is still correct,
    it just cannot say whether a later run could reuse any of it. Returning
    None rather than raising keeps that distinction -- every `read_set_digest`
    is then None, which reads as "unknown", not "unchanged".

    Parameters
    ----------
    nwb_file_name : str
        The copy file registered in Nwbfile.

    Returns
    -------
    NwbfileHasher or None
    """
    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.utils.nwb_hash import NwbfileHasher

    try:
        return NwbfileHasher(
            Nwbfile.get_abs_path(nwb_file_name), object_ids=True
        )
    except Exception as err:  # unreadable, or a dtype the hasher chokes on
        logger.debug(f"Read-set digests unavailable for {nwb_file_name}: {err}")
        return None


def _blocking_parents(table, failed_tables: set) -> Tuple[str, ...]:
    """Return the failed tables this one depends on.

    A table whose parent could not be planned cannot be planned meaningfully
    either; reporting it separately would turn one root cause into many.
    """
    if not failed_tables:
        return ()
    try:
        parents = set(table.parents())
    except Exception:  # pragma: no cover - undeclared table
        return ()
    return tuple(sorted(parents & failed_tables))


def insert_plan(
    plan: IngestionPlan,
    allow_partial: bool = False,
    on_divergence: str = "interactive",
    rollback_on_miss: bool = False,
) -> ReportResult:
    """Insert what a plan worked out, re-deriving nothing.

    The plan already holds every intended row, already checked. This applies
    the divergence policy, then writes those rows in dependency order. No
    file is reopened and no mapping is re-run: if a row is wrong here, the
    plan was wrong, which is a planner gap rather than an ingestion error.

    Freshness is judged per entry, never by comparing the file's hash to the
    one recorded when the plan was made. The expected workflow edits the file
    between attempts, so a whole-file comparison would reject every plan it
    was meant to preserve.

    Parameters
    ----------
    plan : IngestionPlan
        As returned by `plan_nwbfile`, with live table targets.
    allow_partial : bool, optional
        Insert the tables that planned cleanly even though others did not.
        Default False: a plan with blocking problems inserts nothing, so a
        half-ingested file is a choice rather than an accident.
    on_divergence : str, optional
        What to do when the file disagrees with a stored row (D7).
        `interactive` prompts once per divergence, outside any transaction;
        `accept` keeps the stored value and inserts the rest; `raise` fails
        with the report attached. Default `interactive`.
    rollback_on_miss : bool, optional
        Delete the session when a `planner_miss` leaves the file part
        inserted. Default False. This is the *only* case a rollback is for
        now: a plan is checked before anything is written, so a failure here
        means the planner was wrong, not the file. Everything the old
        `rollback_on_fail` guarded against is now caught at plan time, and a
        blanket rollback would throw away good rows to undo a bug.

    Returns
    -------
    ReportResult
        Falsy when everything asked for was inserted.
    """
    from spyglass.common.common_usage import IngestionPlanLog

    if on_divergence not in ("interactive", "accept", "raise"):
        raise ValueError(
            f"Unknown on_divergence {on_divergence!r}. "
            + "Expected interactive, accept or raise."
        )

    if plan.verdict == "no_op":
        # Nothing to do to the *data*; the staging area still needs closing.
        # Leaving it open would keep a payload for every entry that is
        # already stored, which is the one thing the log must not do.
        logger.info(f"{plan.nwb_file_name}: already ingested, nothing to do")
        IngestionPlanLog().mark_inserted(
            plan, inserted=(), existing=list(_targets(plan)), complete=True
        )
        return ReportResult(plan)

    divergences = [p for p in plan.problems if p.code == "divergence"]
    if divergences and not _divergence_accepted(divergences, on_divergence):
        logger.error(plan.report(log=False))
        return ReportResult(plan)

    blocking = plan.hard_failures
    if blocking and not allow_partial:
        logger.error(
            f"{plan.nwb_file_name}: {len(blocking)} blocking problems, "
            + "nothing inserted. Fix them, or pass allow_partial=True."
        )
        logger.error(plan.report(log=False))
        return ReportResult(plan)

    inserted, misses, existing = [], [], []
    for table_plan in plan.table_plans:
        if table_plan.status != "ok":
            continue  # failed or blocked: its rows were never validated

        for target, rows in table_plan.entries.entries:
            if not rows:
                continue
            table = target.as_instance
            # The *target's* name, not the owning plan's: a table routinely
            # emits rows for another, and the staged entry is keyed by where
            # the row is going. Using the owner's name here matched nothing
            # for every secondary target, silently leaving them staged.
            name = getattr(target, "full_table_name", str(target))

            try:
                # A plan describes what the file holds, not what is missing
                # from the database, so a partial re-run legitimately
                # restages rows that already landed. Inserting those would
                # raise a duplicate for work already done -- the very noise
                # this replaces. Inside the try: deciding what to insert can
                # fail too, and that is no less a planner gap than the
                # insert itself.
                novel = _novel_rows(table, rows)
                if stored := [row for row in rows if row not in novel]:
                    existing.append((name, target, stored))
                if not novel:
                    continue

                # A target may be a plain SpyglassMixin -- Task, say, which
                # TaskEpoch plans rows for but which ingests nothing itself.
                insert = getattr(table, "_insert_plan", None)
                if insert is None:
                    table.insert(
                        novel, skip_duplicates=False, allow_direct_insert=True
                    )
                else:
                    insert(novel, nwb_file_name=plan.nwb_file_name)
                inserted.append((name, target, novel))
            except Exception as err:
                # The plan said these rows were insertable and they were not.
                # That is a gap in the planner, not a user error, so it is
                # coded distinctly to stay findable.
                misses.append(
                    Problem(
                        severity="hard",
                        code="planner_miss",
                        message=f"{type(err).__name__}: {err}",
                        table=table_plan.table_name,
                        exc_type=type(err).__name__,
                    )
                )
                logger.error(f"planner_miss in {table_plan.table_name}: {err}")
                break

    if misses and rollback_on_miss:
        _rollback(plan.nwb_file_name)

    if skipped := sum(len(rows) for _, _, rows in existing):
        logger.info(f"{plan.nwb_file_name}: {skipped} entries already stored")

    IngestionPlanLog().mark_inserted(
        plan, inserted, existing=existing, complete=not misses
    )

    if misses:  # attach them, so the caller sees what the plan missed
        plan = replace(plan, fatal=plan.fatal + tuple(misses))

    return ReportResult(plan)


def _rollback(nwb_file_name: str) -> None:
    """Delete a session after a planner miss left it part inserted.

    The fallback of last resort, and deliberately not the default: it throws
    away rows that inserted correctly in order to undo the ones that did not.
    Worth it only when a validated plan failed halfway, because then the
    partial state is not something the user chose.

    Parameters
    ----------
    nwb_file_name : str
        The file to roll back.
    """
    from spyglass.common.common_nwbfile import Nwbfile

    query = Nwbfile & {"nwb_file_name": nwb_file_name}
    if not query:
        return

    logger.error(
        f"Rolling back {nwb_file_name} after a planner miss. "
        + "This deletes rows that inserted correctly; the miss is a bug "
        + "worth reporting."
    )
    query.super_delete(warn=False)


def _targets(plan):
    """Yield `(table_name, target, rows)` for every target a plan holds.

    Keyed by the *target*, not by the plan that emitted it: a table
    routinely plans rows for another, and a staged entry belongs to where
    the row is going.

    Parameters
    ----------
    plan : IngestionPlan

    Yields
    ------
    tuple of (str, dj.Table, tuple of dict)
    """
    for table_plan in plan.table_plans:
        for target, rows in table_plan.entries.entries:
            if rows:
                yield (
                    getattr(target, "full_table_name", str(target)),
                    target,
                    rows,
                )


def _novel_rows(table, rows) -> List[dict]:
    """Return the rows this table does not already hold.

    One query per table, not per row: the primary keys already stored are
    fetched once and compared in memory. Restricted to the file where the
    table is keyed by one, so a shared table like `Task` is still answered
    correctly without reading every row in it.

    Parameters
    ----------
    table : dj.Table
        An instanced table.
    rows : sequence of dict
        Planned entries for it.

    Returns
    -------
    list of dict
        Those whose primary key is not present.
    """
    restriction = True
    if "nwb_file_name" in table.primary_key:
        names = {
            row.get("nwb_file_name") for row in rows if row.get("nwb_file_name")
        }
        if names:
            restriction = [{"nwb_file_name": name} for name in names]

    try:
        stored = {
            row_key(table, existing)
            for existing in (table & restriction).fetch(as_dict=True)
        }
    except Exception as err:  # unreadable: insert and let the table object
        logger.debug(f"Could not read existing keys for {table}: {err}")
        return list(rows)

    return [row for row in rows if row_key(table, row) not in stored]


def _divergence_accepted(divergences, on_divergence: str) -> bool:
    """Apply the divergence policy, outside any transaction.

    Prompting mid-transaction is what the plan pass exists to avoid: a
    question asked with rows half-written holds a lock open on an answer
    nobody is there to give.

    Parameters
    ----------
    divergences : list of Problem
        The divergence problems this plan recorded.
    on_divergence : str
        `interactive`, `accept` or `raise`.

    Returns
    -------
    bool
        Whether to go on and insert.
    """
    from spyglass.settings import test_mode

    if on_divergence == "accept":
        logger.info(
            f"Keeping the stored values for {len(divergences)} divergences"
        )
        return True

    if on_divergence == "raise":
        logger.error(
            f"{len(divergences)} entries disagree with stored rows. "
            + "Nothing inserted."
        )
        return False

    # interactive, which never runs unattended: a suite that blocked on
    # stdin would hang rather than fail, so test mode declines instead.
    # This is the same short-circuit `accept_divergence` has always had.
    if test_mode:
        logger.error(
            f"{len(divergences)} divergences, and test mode does not prompt. "
            + "Nothing inserted."
        )
        return False

    for problem in divergences:
        logger.warning(str(problem))
    answer = input(
        f"{len(divergences)} entries disagree with stored rows. "
        + "Keep the stored values and insert the rest? [y/N] "
    )
    return answer.strip().lower() in ("y", "yes")
