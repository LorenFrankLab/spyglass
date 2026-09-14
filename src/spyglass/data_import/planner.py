"""Plan an NWB file's ingestion without inserting any of it.

Parses every ingestion table against one open file, checks the result for the
integrity failures that would otherwise surface mid-transaction, and reports
what it found -- including whether any of it is new.

The check that matters most is prospective: an entry's parent must exist
either in the database *or* elsewhere in the same plan. Without that, tables
that emit a parent's rows alongside their own would appear broken, and one
missing object would be reported once per dependent table rather than once.
"""

from typing import Dict, List, Optional, Tuple

from spyglass.data_import.ingestion_plan import (
    IngestionPlan,
    PlannedEntries,
    Problem,
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
