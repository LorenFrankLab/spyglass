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
"""

import json
from dataclasses import asdict, dataclass, field
from hashlib import md5
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Severity of a problem found while parsing. `fatal` stops the file, `hard`
# stops one table and blocks its dependents, `soft` is an absence that is not
# an error, `info` is a resolution worth reporting.
SEVERITIES = ("fatal", "hard", "soft", "info")


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

    def __post_init__(self):
        """Reject a severity outside the taxonomy."""
        if self.severity not in SEVERITIES:
            raise ValueError(
                f"Unknown severity {self.severity!r}. "
                + f"Expected one of {SEVERITIES}."
            )

    def __str__(self):
        where = f"{self.table}: " if self.table else ""
        return f"[{self.severity}] {where}{self.code}: {self.message}"


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


def row_key(table, row) -> tuple:
    """Return a row's primary key as a hashable tuple.

    Values are coerced to the column's declared type first. The database
    coerces on insert -- a planned `'0'` for an integer column is stored as
    `0` -- so comparing raw values would report an entry as new, or a parent
    as missing, when it is neither.

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
        (attr, hashable(coerced(table, attr, row.get(attr))))
        for attr in sorted(table.primary_key)
    )


class PlannedEntries:
    """Rows destined for one or more tables, keyed by stable identity.

    Adding to a target that is already present appends to it; merging accepts
    targets the receiver has never seen. Neither raises, which is what the
    two hand-rolled merge sites this replaces used to do.
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
            # -- a FreeTable, or a plan rebuilt from storage -- is kept as
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

    def targets(self) -> Tuple[Any, ...]:
        """Return the target tables, in first-seen order."""
        return tuple(target for target, _ in self._entries.values())

    def in_dependency_order(self) -> Tuple[Tuple[Any, Tuple[dict, ...]], ...]:
        """Return (target, rows) pairs sorted so parents precede children.

        Ranked by DataJoint's own topological sort of the dependency graph,
        so a table that emits its parent's entries alongside its own need not
        remember to put the parent first.

        Targets the graph does not know -- a plan rebuilt from storage, say --
        keep their first-seen order, after the ranked ones.
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
        read -- callers then keep first-seen order.
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

    def freeze(self) -> "FrozenPlannedEntries":
        """Return an immutable snapshot of this collection."""
        return FrozenPlannedEntries(
            tuple(
                (target, tuple(rows)) for target, rows in self._entries.values()
            )
        )

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


@dataclass(frozen=True)
class FrozenPlannedEntries:
    """An immutable snapshot of a PlannedEntries collection."""

    entries: Tuple[Tuple[Any, Tuple[dict, ...]], ...]

    def rows_for(self, table) -> Tuple[dict, ...]:
        """Return the rows planned for one target, empty if it has none."""
        key = _target_key(table)
        for target, rows in self.entries:
            if _target_key(target) == key:
                return rows
        return ()

    def __iter__(self) -> Iterator[Tuple[Any, Tuple[dict, ...]]]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return any(rows for _, rows in self.entries)


# Ordered worst-first, so the first match is the severity that matters.
_SEVERITY_RANK = {name: rank for rank, name in enumerate(SEVERITIES)}

# Severities that stop work rather than merely describing it.
BLOCKING = ("fatal", "hard")


@dataclass(frozen=True)
class TablePlan:
    """What one table would insert for one file, and what went wrong.

    Produced by parsing, consumed by inserting. Immutable: a plan records a
    decision already made, rather than a buffer still being filled.
    """

    table_name: str
    entries: FrozenPlannedEntries
    status: str = "ok"  # ok | skipped | failed
    problems: Tuple[Problem, ...] = ()
    reads: Tuple[str, ...] = ()

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

    Falsy when nothing blocks it, so `if plan:` reads as "is there a problem".
    """

    nwb_file_name: str
    table_plans: Tuple[TablePlan, ...] = ()
    fatal: Tuple[Problem, ...] = ()
    nwb_hash: Optional[str] = None
    spyglass_version: Optional[str] = None
    config_hash: Optional[str] = None
    # {table_name: count} for entries not already in the database
    novel: Dict[str, int] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        """Whether this file holds anything new.

        One answer in place of a wall of duplicate errors on a re-run:

        - `conflict` — an entry exists with different values, which is a
          disagreement to resolve rather than work to do
        - `no_op` — every planned entry is already present and matches
        - `all_new` — none of it is in the database yet
        - `partial_new` — some of it is
        """
        if any(problem.code == "divergence" for problem in self.problems):
            return "conflict"

        novel = sum(self.novel.values())
        if not novel:
            return "no_op"
        return "all_new" if novel == self.entry_count else "partial_new"

    def new_entries_by_table(self) -> Dict[str, int]:
        """Return the count of novel entries per table, omitting zeroes."""
        return {name: n for name, n in self.novel.items() if n}

    def report(self, verbose: bool = False) -> str:
        """Render the plan as text, leading with the verdict.

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
            "no_op": "already ingested, nothing to do",
            "all_new": f"{self.entry_count} entries, all new",
            "partial_new": f"{sum(self.novel.values())} new entries",
            "conflict": "conflicts with what is already stored",
        }[self.verdict]

        lines = [f"{self.nwb_file_name}: {self.verdict} — {headline}"]

        if novel := self.new_entries_by_table():
            lines.append("")
            lines.append("New entries:")
            lines.extend(
                f"  {count:>5}  {name}" for name, count in sorted(novel.items())
            )

        shown = (
            self.problems
            if verbose
            else [p for p in self.problems if p.severity in BLOCKING]
        )
        if shown:
            lines.append("")
            lines.append("Problems:")
            lines.extend(f"  {problem}" for problem in shown)
            if suggestions := [
                p for p in shown if p.suggested_revision is not None
            ]:
                lines.append("")
                lines.append("Suggested revisions:")
                lines.extend(
                    f"  {p.table}: {p.suggested_revision}" for p in suggestions
                )

        if blocked := [
            plan.table_name
            for plan in self.table_plans
            if plan.status == "blocked"
        ]:
            lines.append("")
            lines.append(
                f"Blocked by the above ({len(blocked)}): "
                + ", ".join(sorted(blocked))
            )

        return "\n".join(lines)

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
    def hard_failures(self) -> Tuple[Problem, ...]:
        """Problems that block a table."""
        return tuple(p for p in self.problems if p.severity == "hard")

    @property
    def is_clean(self) -> bool:
        """Whether the file can be ingested with nothing left unresolved."""
        return not any(p.severity in BLOCKING for p in self.problems)

    def status_by_table(self) -> Dict[str, str]:
        """Return each table's status, keyed by table name."""
        return {plan.table_name: plan.status for plan in self.table_plans}

    @property
    def plan_hash(self) -> str:
        """Content hash, stable across runs of identical plans."""
        return md5(
            json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        ).hexdigest()

    def to_dict(self) -> dict:
        """Return a JSON-safe mapping of this plan."""
        return {
            "nwb_file_name": self.nwb_file_name,
            "nwb_hash": self.nwb_hash,
            "spyglass_version": self.spyglass_version,
            "config_hash": self.config_hash,
            "fatal": [asdict(p) for p in self.fatal],
            "table_plans": [
                {
                    "table_name": plan.table_name,
                    "status": plan.status,
                    "entry_count": plan.entry_count,
                    "problems": [asdict(p) for p in plan.problems],
                    "reads": list(plan.reads),
                    "entries": [
                        {
                            "table": getattr(
                                target, "full_table_name", str(target)
                            ),
                            "rows": [_jsonable(row) for row in rows],
                        }
                        for target, rows in plan.entries
                    ],
                }
                for plan in self.table_plans
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IngestionPlan":
        """Rebuild a plan from `to_dict` output.

        Entries come back as the stored rows keyed by table *name*: the plan
        is a record, and the tables it names may not be importable here.
        """
        table_plans = []
        for plan in data.get("table_plans", []):
            entries = PlannedEntries()
            for target in plan.get("entries", []):
                entries.add(_NamedTable(target["table"]), target["rows"])
            table_plans.append(
                TablePlan(
                    table_name=plan["table_name"],
                    entries=entries.freeze(),
                    status=plan.get("status", "ok"),
                    problems=tuple(
                        Problem(**p) for p in plan.get("problems", [])
                    ),
                    reads=tuple(plan.get("reads", [])),
                )
            )

        return cls(
            nwb_file_name=data["nwb_file_name"],
            table_plans=tuple(table_plans),
            fatal=tuple(Problem(**p) for p in data.get("fatal", [])),
            nwb_hash=data.get("nwb_hash"),
            spyglass_version=data.get("spyglass_version"),
            config_hash=data.get("config_hash"),
        )

    def __bool__(self) -> bool:
        """True when something blocks this plan."""
        return not self.is_clean

    def __repr__(self) -> str:
        verdict = "clean" if self.is_clean else "blocked"
        return (
            f"IngestionPlan({self.nwb_file_name}, {verdict}, "
            + f"{self.entry_count} entries, {len(self.problems)} problems)"
        )


@dataclass(frozen=True)
class _NamedTable:
    """A table known only by name, as rebuilt from a stored plan."""

    full_table_name: str


def _jsonable(value):
    """Coerce a value into something JSON can hold."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):  # numpy array or scalar
        return value.tolist()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass
class FileContext:
    """What a table is given while parsing one NWB file.

    Replaces caching file-level state on the table object, which outlives any
    one ingestion. Also collects the objects a table read -- the read-set that
    lets a later attempt re-parse only what actually changed -- and the
    problems it hit, so parsing accumulates a report rather than stopping at
    the first failure.
    """

    nwb_file_name: str
    nwb_file: Any = None
    config: dict = field(default_factory=dict)
    base_key: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)
    reads: List[str] = field(default_factory=list)
    problems: List[Problem] = field(default_factory=list)

    @property
    def file_restr(self) -> dict:
        """Restriction selecting this file."""
        return {"nwb_file_name": self.nwb_file_name}

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
