"""Generate a SQL script to drop an ancestor table + the part and intermediate
tables that block DataJoint from performing the drop normally.

This is helpful for dropping drafted pipelines that terminate in a Merge part
table (e.g. a PositionOutput merge) where the Master must be preserved but
the Part and Ancestor must be dropped to reset the pipeline

Problem structure
-----------------
DataJoint's dependency graph looks like::

    Ancestor  <── [Intermediate …]  <── Master  <── Part

Arrows show the FK reference direction (``A → B`` means A has a FK column
pointing at B's primary key).  DataJoint will refuse to drop ``Ancestor``
unless every downstream table is also dropped — including ``Master``, which
must be preserved.

For **merge tables** (e.g. a PositionOutput merge in DataJoint), the Master
itself has no upstream FK — all upstream FKs live in the Part table::

    Ancestor  <── [Intermediate …]  <── Part
                                         └── Master  (no upstream FK; preserved)

Strategy
--------
1. **Drop Part** first — dropping a child (FK holder) is always allowed in
   MySQL without touching the parent (Master).  No ``ALTER TABLE`` needed.
2. **Drop intermediate tables** in order (closest to Part first, each one
   is a child of the next).
3. **Drop Ancestor** — nothing references it anymore.

``Master`` is never touched and its rows are fully preserved.

This script **only queries the database and writes a ``.sql`` file**.
Nothing is executed.  Inspect the file, confirm the plan, then run it
manually.

Usage
-----
::

    python drop_pipeline.py \\
        --ancestor my_schema.ancestor_table \\
        --part     my_schema.master__part \\
        [--output  temp_drop_plan_YYYYMMDD_HHMMSS.sql]

Table names are supplied as plain ``schema.table`` (no backticks — bash
interprets backticks as command substitution).  Backticks are added
automatically in the generated SQL.
"""

import argparse
import re
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import datajoint as dj

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

# (schema_name, table_name)
TableRef = tuple[str, str]


@dataclass
class FkInfo:
    """One foreign-key constraint between two tables."""

    from_schema: str
    from_table: str
    constraint_name: str
    to_schema: str
    to_table: str

    @property
    def from_full(self) -> str:
        """Full table name of the FK source (the referencing table)."""
        return f"`{self.from_schema}`.`{self.from_table}`"

    @property
    def to_full(self) -> str:
        """Full table name of the FK target (the referenced table)."""
        return f"`{self.to_schema}`.`{self.to_table}`"


@dataclass
class DropPlan:
    """Everything needed to generate the SQL script."""

    # Input tables
    part: TableRef
    master: TableRef  # informational only — preserved, never dropped
    ancestor: TableRef

    # Original shortest FK-reference path: [Part, hop1, …, Ancestor]
    chain: list[TableRef]

    # Full topologically-sorted drop list (leaf-first).
    # Superset of chain — includes all transitive FK blockers.
    drop_order: list[TableRef] = field(default_factory=list)

    # Tables that were added to drop_order solely because they blocked a
    # chain member (not part of the original Part→Ancestor path).
    blocker_tables: set[TableRef] = field(default_factory=set)


# ─────────────────────────────────────────────────────────────────────────────
# Name parsing
# ─────────────────────────────────────────────────────────────────────────────

_BACKTICK_RE = re.compile(r"^`(?P<schema>[^`]+)`\.`(?P<table>[^`]+)`$")


def _parse_full_name(name: str) -> TableRef:
    """Parse a full table name (schema.table or `schema`.`table`)."""
    name = name.strip()
    m = _BACKTICK_RE.match(name)
    if m:
        return m.group("schema"), m.group("table")
    # Accept plain schema.table as well
    parts = name.split(".")
    if len(parts) == 2:
        return parts[0].strip("`"), parts[1].strip("`")
    raise ValueError(
        f"Expected `schema`.`table` or schema.table, got: {name!r}"
    )


def _master_of_part(table_name: str) -> str:
    """Derive the DataJoint master table name from a part table name."""
    if "__" not in table_name:
        raise ValueError(
            f"{table_name!r} does not look like a DataJoint part table "
            "(expected a '__' separator)."
        )
    return table_name.split("__")[0]


# ─────────────────────────────────────────────────────────────────────────────
# Database introspection (read-only)
# ─────────────────────────────────────────────────────────────────────────────


def _table_exists(conn, schema: str, table: str) -> bool:
    """Return True if (schema, table) exists in information_schema."""
    r = conn.query(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_schema = %s AND table_name = %s",
        (schema, table),
    )
    return next(r)[0] > 0


def _outgoing_fks(conn, schema: str, table: str) -> list[FkInfo]:
    """FK constraints that originate FROM (schema, table)."""
    rows = conn.query(
        """
        SELECT   kcu.constraint_name,
                 kcu.referenced_table_schema,
                 kcu.referenced_table_name
        FROM     information_schema.key_column_usage  kcu
        JOIN     information_schema.table_constraints tc
               ON  kcu.constraint_name = tc.constraint_name
               AND kcu.table_schema    = tc.table_schema
               AND kcu.table_name      = tc.table_name
        WHERE    tc.constraint_type  = 'FOREIGN KEY'
          AND    kcu.table_schema    = %s
          AND    kcu.table_name      = %s
        GROUP BY kcu.constraint_name,
                 kcu.referenced_table_schema,
                 kcu.referenced_table_name
        """,
        (schema, table),
    )
    return [FkInfo(schema, table, r[0], r[1], r[2]) for r in rows]


def _incoming_fks(conn, schema: str, table: str) -> list[FkInfo]:
    """FK constraints from OTHER tables that point TO (schema, table)."""
    rows = conn.query(
        """
        SELECT   kcu.table_schema,
                 kcu.table_name,
                 kcu.constraint_name
        FROM     information_schema.key_column_usage  kcu
        JOIN     information_schema.table_constraints tc
               ON  kcu.constraint_name = tc.constraint_name
               AND kcu.table_schema    = tc.table_schema
               AND kcu.table_name      = tc.table_name
        WHERE    tc.constraint_type          = 'FOREIGN KEY'
          AND    kcu.referenced_table_schema = %s
          AND    kcu.referenced_table_name   = %s
        GROUP BY kcu.table_schema, kcu.table_name, kcu.constraint_name
        """,
        (schema, table),
    )
    return [FkInfo(r[0], r[1], r[2], schema, table) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Path finding via BFS over information_schema FK edges
# ─────────────────────────────────────────────────────────────────────────────


def _find_fk_path(
    conn, start: TableRef, end: TableRef, max_depth: int = 20
) -> Optional[list[TableRef]]:
    """BFS: shortest FK-reference path from start to end.

    Follows *outgoing* FKs: start → X → Y → … → end.
    Returns the full path including both endpoints, or None if unreachable.

    Parameters
    ----------
    conn
        DataJoint connection.
    start
        (schema, table) to begin traversal from.
    end
        (schema, table) to search for.
    max_depth
        Maximum path length before giving up (guards against cycles).
    """
    queue: deque[list[TableRef]] = deque([[start]])
    visited: set[TableRef] = set()

    while queue:
        path = queue.popleft()
        current = path[-1]

        if current == end:
            return path
        if current in visited or len(path) > max_depth:
            continue
        visited.add(current)

        for fk in _outgoing_fks(conn, *current):
            nxt: TableRef = (fk.to_schema, fk.to_table)
            queue.append(path + [nxt])

    return None


def _reachable_tables(
    conn, start: TableRef, max_depth: int = 20
) -> set[TableRef]:
    """Return all tables reachable from start via outgoing FKs."""
    visited: set[TableRef] = set()
    queue: deque[TableRef] = deque([start])

    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        if len(visited) > max_depth * 10:
            break
        for fk in _outgoing_fks(conn, *current):
            nxt: TableRef = (fk.to_schema, fk.to_table)
            if nxt not in visited:
                queue.append(nxt)

    return visited


def _expand_and_sort_drop_set(
    conn, initial_chain: list[TableRef], master: TableRef
) -> tuple[list[TableRef], set[TableRef]]:
    """Expand initial_chain to include all transitive FK blockers, then sort.

    Any table that holds a FK pointing at a table we plan to drop must itself
    be dropped first.  This function collects them all (recursively) and
    returns a topologically-sorted drop order so every DROP TABLE can execute
    without a FK violation.

    Parameters
    ----------
    conn
        DataJoint connection.
    initial_chain
        The original BFS path [Part, hop1, …, Ancestor].
    master
        The Merge master table — never added to the drop set.

    Returns
    -------
    (drop_order, blocker_tables)
        drop_order      : full list of tables in safe drop order (leaf-first).
        blocker_tables  : subset of drop_order that were *not* in initial_chain.
    """
    # ── BFS to collect transitive blockers ────────────────────────────────
    drop_set: set[TableRef] = set(initial_chain)
    frontier: set[TableRef] = set(initial_chain)

    while frontier:
        new_tables: set[TableRef] = set()
        for tbl in frontier:
            for fk in _incoming_fks(conn, *tbl):
                blocker: TableRef = (fk.from_schema, fk.from_table)
                if blocker != master and blocker not in drop_set:
                    drop_set.add(blocker)
                    new_tables.add(blocker)
        frontier = new_tables

    blocker_tables: set[TableRef] = drop_set - set(initial_chain)

    # ── Kahn's topological sort (leaf-first drop order) ───────────────────
    # Edge A→B (A references B, both in drop_set): drop A before B.
    outgoing: dict[TableRef, list[TableRef]] = {t: [] for t in drop_set}
    in_degree: dict[TableRef, int] = {t: 0 for t in drop_set}

    for tbl in drop_set:
        for fk in _outgoing_fks(conn, *tbl):
            ref: TableRef = (fk.to_schema, fk.to_table)
            if ref in drop_set:
                outgoing[tbl].append(ref)
                in_degree[ref] += 1

    queue: deque[TableRef] = deque(
        tbl for tbl in drop_set if in_degree[tbl] == 0
    )
    order: list[TableRef] = []
    while queue:
        tbl = queue.popleft()
        order.append(tbl)
        for ref in outgoing[tbl]:
            in_degree[ref] -= 1
            if in_degree[ref] == 0:
                queue.append(ref)

    # Guard: if a cycle exists (shouldn't in a healthy DB) append remainder
    if len(order) < len(drop_set):
        order.extend(sorted(drop_set - set(order)))

    return order, blocker_tables


# ─────────────────────────────────────────────────────────────────────────────
# Plan assembly
# ─────────────────────────────────────────────────────────────────────────────


def build_drop_plan(conn, part: TableRef, ancestor: TableRef) -> DropPlan:
    """Query the DB and assemble a complete DropPlan."""
    # ── Validate supplied tables exist ────────────────────────────────────
    for ref, label in [(part, "Part"), (ancestor, "Ancestor")]:
        if not _table_exists(conn, *ref):
            sys.exit(
                f"ERROR: {label} table `{ref[0]}`.`{ref[1]}` does not exist."
            )

    master_name = _master_of_part(part[1])
    master: TableRef = (part[0], master_name)

    if not _table_exists(conn, *master):
        sys.exit(
            f"ERROR: Master table `{master[0]}`.`{master[1]}` does not exist."
        )

    # ── Find the FK chain from Part to Ancestor ───────────────────────────
    # Start from Part (not Master): for merge tables the master has no
    # upstream FKs — all upstream references live in the Part.
    chain = _find_fk_path(conn, part, ancestor)
    if chain is None:
        # Helpful diagnosis: show tables with the same name in reachable schemas
        reachable = _reachable_tables(conn, part)
        matches = [
            f"`{s}`.`{t}`" for s, t in sorted(reachable) if t == ancestor[1]
        ]
        hint = (
            (
                f"\n  Tables named '{ancestor[1]}' reachable from Part:\n"
                + "\n".join(f"    {m}" for m in matches)
            )
            if matches
            else ""
        )
        sys.exit(
            f"ERROR: No FK path found from "
            f"`{part[0]}`.`{part[1]}` to "
            f"`{ancestor[0]}`.`{ancestor[1]}`.\n"
            f"  Verify that the part table really does reference "
            f"the ancestor (directly or transitively).{hint}"
        )

    # chain = [Part, hop1, …, Ancestor] — all will be dropped in this order.

    # ── Master-in-chain safety check ──────────────────────────────────────
    if master in chain:
        sys.exit(
            f"ERROR: Master table `{master[0]}`.`{master[1]}` appears in the "
            f"FK chain between Part and Ancestor.  Dropping it would violate "
            f"the requirement to preserve Master.  Aborting."
        )

    # ── Expand chain with blockers and compute safe drop order ─────────────
    drop_order, blocker_tables = _expand_and_sort_drop_set(conn, chain, master)

    return DropPlan(
        part=part,
        master=master,
        ancestor=ancestor,
        chain=chain,
        drop_order=drop_order,
        blocker_tables=blocker_tables,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SQL generation
# ─────────────────────────────────────────────────────────────────────────────


def _fmt(ref: TableRef) -> str:
    """Format a TableRef as a backtick-quoted full table name."""
    return f"`{ref[0]}`.`{ref[1]}`"


def _step_label(ref: TableRef, plan: "DropPlan") -> str:
    """Return an annotation suffix for a table's role in the plan."""
    if ref == plan.part:
        return "  ← part table"
    if ref == plan.ancestor:
        return "  ← ancestor"
    if ref in plan.blocker_tables:
        return "  ← blocker"
    return "  ← intermediate"


def write_sql(plan: DropPlan, output_path: str) -> None:
    """Render the drop plan as a SQL file."""
    lines: list[str] = []
    a = lines.append

    def hr() -> None:
        a("-- " + "─" * 68)

    # ── Header ────────────────────────────────────────────────────────────
    a("-- " + "=" * 68)
    a(
        f"-- DROP PLAN  |  generated "
        f"{datetime.now().isoformat(timespec='seconds')}"
    )
    a("-- " + "=" * 68)
    a("--")
    a("-- REVIEW THIS FILE CAREFULLY BEFORE EXECUTING.")
    a("-- This script cannot be undone without restoring from a backup.")
    a("--")
    a("-- Tables to DROP (leaf-first order):")
    for i, ref in enumerate(plan.drop_order, 1):
        a(f"--   {i}. {_fmt(ref)}{_step_label(ref, plan)}")
    a("--")
    a(f"-- Table PRESERVED (not dropped): {_fmt(plan.master)}")
    a("--   Master holds a FK to Part; it disappears automatically when")
    a("--   Part is dropped.  No ALTER TABLE needed.")
    a("--")
    a("-- Original dependency chain (Part → … → Ancestor):")
    a("--   " + " → ".join(_fmt(ref) for ref in plan.chain))
    if plan.blocker_tables:
        a("--")
        a("-- Additional tables dropped to clear FK blockers:")
        for ref in plan.drop_order:
            if ref in plan.blocker_tables:
                a(f"--   {_fmt(ref)}")
    a("--")

    # ── DROP statements ───────────────────────────────────────────────────
    for step, ref in enumerate(plan.drop_order, 1):
        hr()
        label = _step_label(ref, plan).strip().lstrip("← ").capitalize()
        a(f"-- Step {step}: Drop {label} table {_fmt(ref)}.")
        if ref == plan.part:
            a("--")
            a("--   DJ normally requires dropping Master alongside a Part.")
            a("--   In MySQL the child (FK holder) can always be dropped")
            a("--   without touching the parent (Master).")
        a("")
        a(f"DROP TABLE {_fmt(ref)};")
        a("")

    # ── Footer ────────────────────────────────────────────────────────────
    hr()
    a("-- Done.")
    a(f"-- {_fmt(plan.master)} is intact with all its rows and columns.")

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ancestor",
        default="`cbroz_position_v2_video`.`vid_file_group`",
        metavar="SCHEMA.TABLE",
        help=(
            "Full table name as schema.table (no backticks — bash treats "
            "backticks as command substitution). "
            "e.g. my_schema.ancestor_table"
        ),
    )
    parser.add_argument(
        "--part",
        default="`position_merge`.`position_output__pose_v2`",
        metavar="SCHEMA.TABLE",
        help=(
            "Full table name as schema.table (no backticks — bash treats "
            "backticks as command substitution). "
            "e.g. my_schema.master__part"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help=(
            "Path for the generated SQL file "
            "(default: temp_drop_plan_<timestamp>.sql in the current dir)."
        ),
    )
    args = parser.parse_args()

    ancestor = _parse_full_name(args.ancestor)
    part = _parse_full_name(args.part)

    conn = dj.conn()

    print("Querying information_schema for FK graph …")
    plan = build_drop_plan(conn, part, ancestor)

    output_path = (
        args.output
        or f"temp_drop_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
    )
    write_sql(plan, output_path)

    print(f"\nSQL plan written to: {output_path}")
    print("  Chain  : " + " → ".join(_fmt(ref) for ref in plan.chain))
    if plan.blocker_tables:
        print(
            "  Blockers added: "
            + ", ".join(
                _fmt(r) for r in plan.drop_order if r in plan.blocker_tables
            )
        )
    print(f"  Drop steps : {len(plan.drop_order)}")
    print(f"  Preserved  : {_fmt(plan.master)}")
    print("\nReview the SQL file carefully before executing it.")


if __name__ == "__main__":
    main()
