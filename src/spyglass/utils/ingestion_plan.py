"""Types carrying ingestion work between parsing and insertion.

`PlannedEntries` is what a table produces while parsing an NWB file: rows
destined for one or more tables, keyed by stable identity and ordered by
dependency rather than by the order a caller happened to add them.

`FileContext` is what a table is given while parsing: the file, its config,
scratch space for file-level lookups, the set of NWB objects the table read,
and somewhere to record problems instead of raising.

Both exist to make a class of mistake unrepresentable rather than merely
discouraged. `PlannedEntries` keys a table by name, so a class, an instance
and a FreeTable resolve to one target and merging never has to guess;
`FileContext` carries per-file state explicitly, so it cannot outlive the
ingestion that created it.

**Imports nothing from spyglass, and must not.** This is the vocabulary
`utils.mixins.ingestion` speaks, imported there at module scope while
`spyglass.utils` is still initializing. It therefore has to sit below every
other spyglass module, which is also why it lives here rather than under
`data_import`, whose package `__init__` reaches `spyglass.common`.
"""

import textwrap
from dataclasses import dataclass, field
from hashlib import md5
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Severity of a problem found while parsing. `fatal` stops the file, `hard`
# stops one table and blocks its dependents, `soft` is an absence that is not
# an error, `info` is a resolution worth reporting.
SEVERITIES = ("fatal", "hard", "soft", "info")

# Report lines wrap here. A problem message carries an exception's text, which
# has no length limit of its own.
REPORT_WIDTH = 79


@dataclass(frozen=True)
class Problem:
    """Something that went wrong, or was resolved, while parsing."""

    severity: str
    code: str
    message: str
    table: Optional[str] = None
    nwb_object_id: Optional[str] = None
    exc_type: Optional[str] = None
    traceback: Optional[str] = None
    # For a divergence: the secondary-key values that would align the planned
    # entry with the row already stored.
    suggested_revision: Optional[dict] = None
    # For a divergence: which row. Carried in-run so the insert pass can stage
    # the row as `conflict` and keep its planned value; not persisted, since
    # the Problem part names its own columns.
    primary_key: Optional[dict] = None

    def __post_init__(self):
        """Reject a severity outside the taxonomy."""
        if self.severity not in SEVERITIES:
            raise ValueError(
                f"Unknown severity {self.severity!r}. "
                + f"Expected one of {SEVERITIES}."
            )

    def __str__(self):
        where = f"{display_name(self.table)}: " if self.table else ""
        # The object id is how a user finds the thing in the file, so it
        # belongs on the line rather than only in the stored record.
        which = f" (object {self.nwb_object_id})" if self.nwb_object_id else ""
        # Stripped: a `parse_error` carries an exception's text, which often
        # ends in a newline and would otherwise split the remedy off its line.
        message = str(self.message).strip()
        return f"[{self.severity}] {where}{self.code}: {message}{which}"


def _target_key(table) -> str:
    """Return a stable, hashable identity for a target table.

    A table names itself, so a class, an instance of it, a FreeTable, and a
    table rebuilt from a stored plan all resolve to one target.

    Parameters
    ----------
    table : type or dj.Table
        Target table, as a class or an instance.

    Returns
    -------
    str
        The target's full table name.

    Raises
    ------
    TypeError
        If the target does not name itself, and so is not a table.
    """
    if full_name := getattr(table, "full_table_name", None):
        return full_name
    raise TypeError(
        f"Not a table: {table!r}. Entries are planned for tables, which "
        "identify themselves by `full_table_name`."
    )


def display_name(full_table_name: str) -> str:
    """Return a DataJoint table name as `schema.CamelCase` for display.

    `` `common_ephys`.`_electrode` `` reads as `common_ephys.Electrode`, and a
    part reads as `common_behav.RawPosition.PosObject`. Display only: what is
    stored and compared is always the full table name.

    Parameters
    ----------
    full_table_name : str
        A DataJoint full table name, backquoted.

    Returns
    -------
    str
        The camel-case form, or the input unchanged if it does not parse.
    """
    if not full_table_name or "." not in full_table_name:
        return full_table_name

    schema, _, table = full_table_name.partition(".")
    schema = schema.strip("`")
    table = table.strip("`")

    # A part table is `_master__part`; the tier prefix is ~, #, _ or __.
    master, sep, part = table.partition("__")
    camel = _camel(master)
    if sep and part:
        camel = f"{camel}.{_camel(part)}"
    return f"{schema}.{camel}"


def wrapped(text: str, indent: str = "", hanging: str = None) -> List[str]:
    """Return `text` wrapped to the report width, as indented lines.

    Parameters
    ----------
    text : str
        One logical line, unwrapped.
    indent : str, optional
        Prefix for the first line. Default none.
    hanging : str, optional
        Prefix for continuation lines. Default: `indent` plus two spaces.

    Returns
    -------
    list of str
    """
    return textwrap.wrap(
        text,
        width=REPORT_WIDTH,
        initial_indent=indent,
        subsequent_indent=indent + " " * 2 if hanging is None else hanging,
        break_long_words=False,  # an object id or a path stays one token
        break_on_hyphens=False,
    ) or [indent + text]


def _camel(name: str) -> str:
    """Return one DataJoint name component as CamelCase."""
    stripped = name.lstrip("_~#")
    if not stripped:
        return name
    return "".join(part.title() for part in stripped.split("_"))


def coerced(table, attr: str, value):
    """Return a value as the column would store it.

    Parameters
    ----------
    table : dj.Table
        The table the value is destined for.
    attr : str
        Column name.
    value : Any
        The planned or fetched value.

    Returns
    -------
    Any
        The value coerced to the column's declared type, or unchanged if it
        cannot be.
    """
    if value is None:
        return None

    declared = getattr(table.heading.attributes.get(attr), "type", "") or ""

    try:
        if declared.startswith(("int", "tinyint", "smallint", "bigint")):
            return int(value)
        if declared.startswith(("float", "double", "decimal")):
            return float(value)
    except (TypeError, ValueError):
        return value

    return value


def hashable(value):
    """Return a hashable stand-in for a key value.

    Values fetched from the database arrive as numpy scalars while planned
    values are plain Python, so both are reduced to the same thing. Getting
    this wrong makes every key comparison fail silently: `np.int64(0)` and
    `0` would otherwise hash differently.
    """
    if isinstance(value, (list, dict, set)):
        return str(value)
    if getattr(value, "ndim", None) == 0 and hasattr(value, "item"):
        return value.item()  # numpy scalar -> python scalar
    if hasattr(value, "tobytes"):  # array
        return value.tobytes()
    return value


def entry_digest(entry: dict) -> str:
    """Digest over a whole entry, faithful to what arrays actually hold.

    Not `datajoint.hash.key_hash`: it hashes `str(value)`, and numpy
    abbreviates a large array to `[0. 1. ... 9998. 9999.]`, so an edit in the
    middle of an `IntervalList.valid_times` would hash identically.

    Parameters
    ----------
    entry : dict
        A planned entry.

    Returns
    -------
    str
        32-character hex digest.
    """
    hashed = md5()
    for name, value in sorted(entry.items()):
        hashed.update(str(name).encode())
        reduced = hashable(value)
        hashed.update(
            reduced if isinstance(reduced, bytes) else str(reduced).encode()
        )
    return hashed.hexdigest()


def rows_digest(rows) -> str:
    """Digest over a set of rows, order-independent.

    Order-independent because a fetch promises no order.

    Parameters
    ----------
    rows : iterable of dict

    Returns
    -------
    str
        32-character hex digest.
    """
    rolled = md5()
    for digest in sorted(entry_digest(row) for row in rows):
        rolled.update(digest.encode())
    return rolled.hexdigest()


def config_digest(config) -> Optional[str]:
    """Digest over the config a parse was given, or None when there is none.

    Not `datajoint.hash.key_hash`, which hashes values and uses keys only for
    ordering: a config is keyed by table name, so moving an override from one
    table to another would leave the digest unchanged.

    Reordering keys within one entry changes the digest, since `hashable`
    reduces a list to `str(value)`. That re-parses a file for an edit that
    changed nothing, which is the safe direction to err in.

    Parameters
    ----------
    config : dict or None
        The merged config, as `merged_config` assembles it.

    Returns
    -------
    str or None
        None for an empty config, which reads as "nothing to compare": a
        digest of nothing would not match a plan staged without one.
    """
    return entry_digest(config) if config else None


def collated(value):
    """Reduce a string key value to Spyglass's notion of one identity.

    A Spyglass rule, not a mirror of the server's collation, and deliberately
    *stricter*: what Spyglass calls one key the database always does too, never
    the reverse. Being laxer is the direction that fails.

    Parameters
    ----------
    value : Any
        A primary-key value. Returned unchanged unless it is a string.

    Returns
    -------
    Any
    """
    if not isinstance(value, str):
        return value

    # Accents are *not* folded: `Müller` and `Muller` stay two identities here
    # while the default collation calls them one, so that pair reaches the
    # insert and fails. A known gap, not an oversight.
    #
    # `strip` stays: a PAD SPACE collation ignores trailing spaces, so without
    # it Spyglass would be laxer than that backend.
    return value.strip().casefold()


def normalized(table, attr: str, value):
    """Return one attribute value as Spyglass compares it.

    The single definition of a comparable value: declared type via `coerced`,
    representation via `hashable`, case and surrounding whitespace via
    `collated`.

    Parameters
    ----------
    table : dj.Table
        The table the value belongs to.
    attr : str
        Column name.
    value : Any
        A planned or fetched value.

    Returns
    -------
    Any
        An identity, not a record of what the row said. Show a reader the
        row's own value, never this.
    """
    return collated(hashable(coerced(table, attr, value)))


def row_key(table, row) -> tuple:
    """Return a row's primary key as Spyglass identifies it.

    Each value through `normalized`. This is the single definition of row
    identity. Novelty, foreign-key presence, same-key collapsing and the
    staged `key_hash` all come through here, so none of them can drift from
    the others.

    Parameters
    ----------
    table : dj.Table
        The table whose primary key defines the tuple.
    row : dict
        A planned or fetched entry. Missing attributes read as None.

    Returns
    -------
    tuple
        (attribute, value) pairs, sorted by attribute.
    """
    return tuple(
        (attr, normalized(table, attr, row.get(attr)))
        for attr in sorted(table.primary_key)
    )


class PlannedEntries:
    """Rows destined for one or more tables, keyed by stable identity.

    Adding to a target that is already present appends to it; merging accepts
    targets the receiver has never seen. Neither raises: a table's entries
    arrive one source object at a time, and any object may be the first to
    name a given target.
    """

    def __init__(self):
        # key -> (target, rows). dict preserves first-seen order.
        self._entries: Dict[Any, Tuple[Any, List[dict]]] = {}

    def add(self, table, rows) -> "PlannedEntries":
        """Add rows for a target table.

        Parameters
        ----------
        table : type or dj.Table
            Target table. A class and an instance of it are the same target.
        rows : iterable of dict
            Entries to insert into that table.

        Returns
        -------
        PlannedEntries
            self, for chaining.
        """
        rows = list(rows)
        if not rows:
            return self

        key = _target_key(table)
        if key not in self._entries:
            # The class is the canonical form: it is what callers name when
            # adding, and what they look up by. A target with no class form
            # such as a FreeTable, or a plan rebuilt from storage, is kept as
            # given, since its class cannot reproduce it.
            self._entries[key] = (getattr(table, "as_class", table), [])
        self._entries[key][1].extend(rows)

        return self

    def extend(self, other: "PlannedEntries") -> "PlannedEntries":
        """Merge another collection into this one.

        Parameters
        ----------
        other : PlannedEntries
            Collection whose rows are appended to this one's.

        Returns
        -------
        PlannedEntries
            self, for chaining.
        """
        for target, rows in other:
            self.add(target, rows)
        return self

    def rows_for(self, table) -> Tuple[dict, ...]:
        """Return the rows planned for one target, empty if it has none."""
        entry = self._entries.get(_target_key(table))
        return tuple(entry[1]) if entry else ()

    def in_dependency_order(self) -> Tuple[Tuple[Any, Tuple[dict, ...]], ...]:
        """Return (target, rows) pairs sorted so parents precede children.

        Ranked by DataJoint's own topological sort, so a table emitting its
        parent's entries need not put the parent first. Targets the graph does
        not know keep their first-seen order, after the ranked ones.
        """
        pairs = list(self)
        ranks = self._topological_ranks(pairs)
        if not ranks:
            return tuple(pairs)

        return tuple(
            pair
            for _, pair in sorted(
                enumerate(pairs),
                key=lambda item: (
                    ranks.get(_target_key(item[1][0]), len(ranks)),
                    item[0],  # stable within a rank, and for unranked targets
                ),
            )
        )

    @staticmethod
    def _topological_ranks(pairs) -> Dict[str, int]:
        """Return {full table name: position} from DataJoint's graph.

        Empty when no target can supply a connection, or the graph cannot be
        read, so callers then keep first-seen order.
        """
        for target, _ in pairs:
            connection = getattr(target, "connection", None)
            if connection is None:
                continue
            try:
                dependencies = connection.dependencies
                dependencies.load()
                return {
                    name: rank
                    for rank, name in enumerate(dependencies.topo_sort())
                }
            except Exception:  # graph unavailable; order is the caller's
                return {}
        return {}

    def as_dict(self) -> dict:
        """Return the legacy `{table: [rows]}` mapping, dependency-ordered."""
        return {
            target: list(rows) for target, rows in self.in_dependency_order()
        }

    @classmethod
    def from_dict(cls, mapping) -> "PlannedEntries":
        """Build from the legacy `{table: [rows]}` mapping."""
        entries = cls()
        for table, rows in (mapping or {}).items():
            entries.add(table, rows)
        return entries

    def __iter__(self) -> Iterator[Tuple[Any, Tuple[dict, ...]]]:
        for target, rows in self._entries.values():
            yield target, tuple(rows)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return any(rows for _, rows in self._entries.values())

    def __repr__(self) -> str:
        summary = ", ".join(
            f"{getattr(t, '__name__', t)}: {len(r)}" for t, r in self
        )
        return f"PlannedEntries({summary})"


# One line per problem code, saying what to do about it. Keyed by code so a
# report stays useful to someone who has not read the parser: the message says
# what happened, this says what to change.
REMEDIES = {
    "file_not_registered": "Insert the file into Nwbfile before planning it.",
    "file_unreadable": "Check the file opens with pynwb; it may be truncated.",
    "parse_error": (
        "A table raised while parsing. Fetch `error_raw` from "
        "`IngestionPlanLog.Problem` for the traceback."
    ),
    "missing_attribute": (
        "The NWB file does not supply a required column. Add it to the file, "
        "or declare it in the file's _spyglass_config.yaml."
    ),
    "missing_parent": (
        "Nothing in the file or the database supplies the referenced row. "
        "Ingest the parent first, or fix the reference."
    ),
    "duplicate_key": (
        "Two planned entries share a primary key. Usually one source object "
        "is described twice in the file."
    ),
    "value_too_long": "Shorten the value in the file, or widen the column.",
    "file_already_registered": (
        "This file is already in Nwbfile, so it was skipped. Pass "
        "reinsert=True to delete the existing session and ingest it again."
    ),
    "extension_unmet": (
        "This table needs an NWB extension the file does not declare. Convert "
        "the file with that extension, or ingest without this table."
    ),
    "missing_camera": (
        "No CameraDevice supplies the camera this video names. Add the camera "
        "to the file or its _spyglass_config.yaml, or the video stays unplaced."
    ),
    "planner_miss": (
        "The plan said these rows were insertable and they were not, which is "
        "a gap in the planner rather than in the file. Please report it."
    ),
    "after_insert_failed": (
        "The rows were inserted; follow-on work for that table was not. "
        "A re-run reaches the no-op path, so run that step directly."
    ),
    "divergence": (
        "Edit the file to match, or update the rows if the file is right. "
        "Re-run to accept the stored values and ingest the rest."
    ),
}

# Ordered worst-first, so the first match is the severity that matters.
_SEVERITY_RANK = {name: rank for rank, name in enumerate(SEVERITIES)}

# Severities that stop work rather than merely describing it.
BLOCKING = ("fatal", "hard")


@dataclass(frozen=True)
class TablePlan:
    """What one table would insert for one file, and what went wrong.

    Produced by parsing, consumed by inserting. The dataclass is frozen, so
    nothing rebinds a plan's fields once it is built. The entries themselves
    are an ordinary `PlannedEntries`: nothing mutates them between validation
    and insertion, and `tests/utils/test_ingestion_contract.py` asserts it.
    """

    table_name: str
    entries: "PlannedEntries"
    status: str = "ok"  # ok | skipped | failed | blocked
    problems: Tuple[Problem, ...] = ()
    reads: Tuple[str, ...] = ()
    # Digest over the objects this table read, from the file as it was when
    # the plan was made. Equal digest, same inputs: this table's parse can be
    # reused. None when the file could not be hashed, which means "unknown",
    # never "unchanged".
    read_set_digest: Optional[str] = None
    # The rest of the reuse key: what the parse read from other tables, which
    # the digest above cannot cover. See `FileContext.xrefs`.
    xrefs: Tuple[dict, ...] = ()

    @property
    def entry_count(self) -> int:
        """Number of rows this plan would insert, across all its targets."""
        return sum(len(rows) for _, rows in self.entries)

    @property
    def severity(self) -> Optional[str]:
        """The worst severity among this plan's problems, if any."""
        if not self.problems:
            return None
        return min(
            (problem.severity for problem in self.problems),
            key=lambda name: _SEVERITY_RANK[name],
        )

    @property
    def is_clean(self) -> bool:
        """Whether nothing here blocks insertion."""
        return not any(
            problem.severity in BLOCKING for problem in self.problems
        )


@dataclass(frozen=True)
class IngestionPlan:
    """What a whole file would insert, and everything wrong with it.

    Also what an ingestion returns. It behaves as a collection of the problems
    that blocked it: falsy when nothing did, so `if plan:` reads as "is there
    a problem", and iterable and measurable over `blocking`.
    """

    nwb_file_name: str
    table_plans: Tuple[TablePlan, ...] = ()
    fatal: Tuple[Problem, ...] = ()
    nwb_hash: Optional[str] = None
    spyglass_version: Optional[str] = None
    # Digest over the file's set of NWB object ids: the whole-file gate on
    # reuse, since no per-table digest can notice an object appearing. None
    # means unknown.
    id_set_digest: Optional[str] = None
    config_digest: Optional[str] = None
    # {table_name: count} for entries not already in the database
    novel: Dict[str, int] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        """Whether this file holds anything new.

        Returns
        -------
        str
            One of `fatal` (could not be planned at all), `no_op` (every
            entry already present and matching), `all_new`, or `partial_new`.

        Notes
        -----
        `fatal` is answered before the count, and must stay that way: a file
        that could not be read plans no entries, so nothing is novel, and
        counting alone would call it `no_op`, meaning already ingested.
        """
        if any(problem.severity == "fatal" for problem in self.problems):
            return "fatal"

        novel = sum(self.novel.values())
        if not novel:
            return "no_op"
        return "all_new" if novel == self.entry_count else "partial_new"

    def new_entries_by_table(self) -> Dict[str, int]:
        """Return the count of novel entries per table, omitting zeroes."""
        return {name: n for name, n in self.novel.items() if n}

    def report(self, verbose: bool = False) -> str:
        """Render the plan as text, leading with the verdict.

        A clean plan is one line to avoid empty sections. Renders only;
        `log_report` is the one that emits.

        Parameters
        ----------
        verbose : bool, optional
            Include every problem, not only the blocking ones. Default False.

        Returns
        -------
        str
            The report.
        """
        headline = {
            "fatal": "could not be planned",
            "no_op": "already ingested, nothing to do",
            "all_new": f"{self.entry_count} entries, all new",
            "partial_new": f"{sum(self.novel.values())} new entries",
        }[self.verdict]

        lines = [f"{self.nwb_file_name}: {self.verdict}, {headline}"]

        if novel := self.new_entries_by_table():
            lines.append("")
            lines.append("New entries:")
            lines.extend(
                f"  {count:>5}  {display_name(name)}"
                for name, count in sorted(novel.items())
            )

        lines.extend(self._divergence_lines())

        # Soft problems are shown by default: an absence that is not an error
        # is still something the reader has to know about. `verbose` adds
        # `info`, which is resolution noise rather than news.
        shown = [
            p
            for p in self.problems
            if p.code != "divergence"  # reported in its own section above
            and (verbose or p.severity != "info")
        ]
        if shown:
            by_severity = {}
            for problem in shown:
                by_severity.setdefault(problem.severity, []).append(problem)

            for severity in sorted(by_severity, key=_SEVERITY_RANK.get):
                group = sorted(
                    by_severity[severity],
                    key=lambda p: (p.table or "", p.code, p.message),
                )
                lines.append("")
                lines.append(f"{severity.capitalize()} ({len(group)}):")
                seen_codes = set()
                for problem in group:
                    lines.extend(wrapped(str(problem), indent=" " * 2))
                    # The remedy is per code, so print it once per group
                    # rather than repeating it under every occurrence.
                    if problem.code not in seen_codes and (
                        remedy := REMEDIES.get(problem.code)
                    ):
                        lines.extend(
                            wrapped(
                                f"-> {remedy}",
                                indent=" " * 6,
                                hanging=" " * 9,
                            )
                        )
                        seen_codes.add(problem.code)

            if suggestions := [
                p for p in shown if p.suggested_revision is not None
            ]:
                lines.append("")
                lines.append("Suggested revisions, to apply as-is:")
                for problem in suggestions:
                    lines.append(f"  # {display_name(problem.table)}")
                    lines.append(f"  {problem.suggested_revision!r}")

        if blocked := [
            plan.table_name
            for plan in self.table_plans
            if plan.status == "blocked"
        ]:
            lines.append("")
            lines.extend(
                wrapped(
                    f"Blocked by the above ({len(blocked)}): "
                    + ", ".join(sorted(display_name(name) for name in blocked))
                )
            )

        return "\n".join(lines)

    def log_report(self, verbose: bool = False) -> str:
        """Emit the report through `logger`, and return it.

        Parameters
        ----------
        verbose : bool, optional
            Include every problem, not only the blocking ones. Default False.

        Returns
        -------
        str
            The report, as `report` renders it.
        """
        from spyglass.utils.logging import logger

        text = self.report(verbose=verbose)
        # Anything beyond `info` is news, so a file that only disagrees with
        # stored rows still warns rather than whispering.
        notable = [p for p in self.problems if p.severity != "info"]
        emit = logger.warning if notable else logger.info
        emit(text)

        return text

    def _divergence_lines(self) -> List[str]:
        """Render the rows the file disagrees with, grouped by table.

        Its own section, whatever severity the divergence carries, so the
        rows and their revisions read together. Grouped and counted so a
        re-ingest touching many rows stays skimmable.
        """
        divergences = [p for p in self.problems if p.code == "divergence"]
        if not divergences:
            return []

        by_table: Dict[str, List[Problem]] = {}
        for problem in divergences:
            by_table.setdefault(problem.table or "", []).append(problem)

        lines = ["", f"Disagrees with stored rows ({len(divergences)}):"]
        for table in sorted(by_table):
            group = by_table[table]
            lines.append(f"  {display_name(table)} ({len(group)}):")
            for problem in sorted(group, key=lambda p: p.message):
                lines.extend(wrapped(problem.message, indent=" " * 4))
                if problem.suggested_revision:
                    # The dict literal, not a prose summary: the point of a
                    # suggested revision is that it can be pasted rather than
                    # retyped from the message.
                    lines.append(
                        f"      stored values: {problem.suggested_revision!r}"
                    )
        lines.extend(
            wrapped(
                f"-> {REMEDIES['divergence']}",
                indent=" " * 2,
                hanging=" " * 5,
            )
        )
        return lines

    @property
    def problems(self) -> Tuple[Problem, ...]:
        """Every problem, file-level first, then per table."""
        found = list(self.fatal)
        for table_plan in self.table_plans:
            found.extend(table_plan.problems)
        return tuple(found)

    @property
    def entry_count(self) -> int:
        """Number of rows this plan would insert, across all tables."""
        return sum(plan.entry_count for plan in self.table_plans)

    @property
    def blocking(self) -> Tuple[Problem, ...]:
        """The problems that stop work, rather than merely describe it.

        `fatal` and `hard`, never `soft` or `info`. The single definition of
        "blocking": anything asking whether a plan may be inserted asks here,
        so that a narrower second answer cannot come into existence.
        """
        return tuple(p for p in self.problems if p.severity in BLOCKING)

    @property
    def is_clean(self) -> bool:
        """Whether the file can be ingested with nothing left unresolved."""
        return not self.blocking

    def status_by_table(self) -> Dict[str, str]:
        """Return each table's status, keyed by table name."""
        return {plan.table_name: plan.status for plan in self.table_plans}

    def __bool__(self) -> bool:
        """True when something blocks this plan."""
        return not self.is_clean

    def __iter__(self) -> Iterator[Problem]:
        """Yield the blocking problems, not every problem noticed."""
        return iter(self.blocking)

    def __len__(self) -> int:
        """Count the blocking problems, not every problem noticed."""
        return len(self.blocking)

    def __str__(self) -> str:
        return self.report()

    def __repr__(self) -> str:
        verdict = "clean" if self.is_clean else "blocked"
        return (
            f"IngestionPlan({self.nwb_file_name}, {verdict}, "
            + f"{self.entry_count} entries, {len(self.problems)} problems)"
        )


@dataclass
class FileContext:
    """What a table is given while parsing one NWB file.

    Holds what would otherwise be cached on the table object, which outlives
    any one ingestion: the read-set a later attempt needs, and the problems
    this parse hit, so it accumulates a report rather than stopping at the
    first failure.
    """

    nwb_file_name: str
    nwb_file: Any = None
    config: dict = field(default_factory=dict)
    base_key: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)
    reads: List[str] = field(default_factory=list)
    # What this parse resolved out of other tables, as `{table, restriction,
    # attr, value}`. Part of the reuse key: `reads` covers NWB objects, so
    # without these a parse is replayable after a row it read has changed.
    xrefs: List[dict] = field(default_factory=list)
    problems: List[Problem] = field(default_factory=list)
    # The object `_parse` is reading, so a raise can name where it happened.
    # Only the parse loop sets this; a row-level problem has no object.
    current_source_id: Optional[str] = None
    # The keys and rows the whole plan will hold, when one is being built.
    # None on a plain insert, where the database is authoritative because
    # rows are written table by table as the pass proceeds.
    key_space: Any = None

    @property
    def file_restr(self) -> dict:
        """Restriction selecting this file."""
        return {"nwb_file_name": self.nwb_file_name}

    def rows_for(self, table, restriction: dict = None) -> tuple:
        """Return the rows a table will hold, for a cross-reference.

        Ask this rather than querying, whenever a table needs a value from
        another table the same ingestion fills: during planning those rows are
        not written yet, so a query would report a failure or plan something
        different from what the insert does. Answers `planned | stored` with a
        plan in progress, and queries without one.

        Parameters
        ----------
        table : dj.Table
            The table to read, instanced.
        restriction : dict, optional
            Attribute equalities every returned row must satisfy.

        Returns
        -------
        tuple of dict
        """
        rows = self._resolve(table, restriction)
        # Digested whole: a caller handed rows may read any column of them.
        self._record_xref(table, restriction, None, rows_digest(rows))
        return rows

    def value_for(self, table, attr: str, restriction: dict = None):
        """Return one attribute of one row a table will hold, or None.

        The common shape of the lookups above: a single value, from a row this
        file owns. None when no such row is planned or stored, which the
        caller decides what to do about. Silently different behaviour on a
        missing cross-reference is the failure mode this exists to prevent.

        Parameters
        ----------
        table : dj.Table
        attr : str
            Attribute to read.
        restriction : dict, optional

        Returns
        -------
        Any or None
        """
        rows = self._resolve(table, restriction)
        value = rows[0].get(attr) if rows else None

        # The value, not the rows it came from: a lookup of one attribute
        # must not be re-run because a column it never read moved.
        self._record_xref(
            table, restriction, attr, normalized(table, attr, value)
        )

        return value

    def _resolve(self, table, restriction: dict = None) -> tuple:
        """Return a table's rows, from the plan where there is one.

        Shared by `rows_for` and `value_for`, so neither records the other's
        cross-reference on top of its own.
        """
        if self.key_space is not None:
            return self.key_space.rows_for(table, restriction)

        return tuple((table & (restriction or True)).fetch(as_dict=True))

    def _record_xref(self, table, restriction, attr, value) -> None:
        """Note what a lookup into another table resolved to.

        Parameters
        ----------
        table : dj.Table
            The table read. Ignored if it does not name itself.
        restriction : dict or None
            The equalities the lookup was narrowed by.
        attr : str or None
            The attribute read, or None when whole rows were.
        value : Any
            What it resolved to: a normalized value, or a rows digest.
        """
        name = getattr(table, "full_table_name", None)
        if name is None:  # pragma: no cover - not a table
            return

        restriction = dict(restriction) if restriction else None
        if any(
            xref["table"] == name
            and xref["attr"] == attr
            and xref["restriction"] == restriction
            for xref in self.xrefs
        ):
            return  # the same lookup twice in one parse resolves the same

        self.xrefs.append(
            dict(table=name, restriction=restriction, attr=attr, value=value)
        )

    def record_read(self, nwb_object) -> None:
        """Note that an NWB object was read while parsing.

        Parameters
        ----------
        nwb_object : str or object
            An object id, or an object carrying one.
        """
        object_id = getattr(nwb_object, "object_id", nwb_object)
        if object_id is not None and object_id not in self.reads:
            self.reads.append(object_id)

    def problem(self, severity: str, code: str, message: str, **kwargs) -> None:
        """Record a problem without raising.

        Parameters
        ----------
        severity : str
            One of `fatal`, `hard`, `soft`, `info`.
        code : str
            Short machine-readable identifier for the kind of problem.
        message : str
            Human-readable detail.
        **kwargs
            Further Problem fields: table, nwb_object_id, exc_type, traceback.
        """
        self.problems.append(
            Problem(severity=severity, code=code, message=message, **kwargs)
        )
