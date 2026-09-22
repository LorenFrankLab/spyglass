"""Specification for the plan objects and the parse-only pass.

Written before the implementation. A plan is what parsing produces *instead
of* inserting: the entries a table would write, the problems it hit, and
enough provenance to decide later whether it can still be trusted.

The invariant these exist to enforce: parsing an NWB file writes nothing to a
data table, however badly the file is malformed. Every failure becomes a
`Problem` on the plan rather than an exception out of it.
"""

import pytest

from spyglass.data_import.ingestion_plan import (
    IngestionPlan,
    PlannedEntries,
    Problem,
    TablePlan,
    _NamedTable,
)

# ---------------------------------------------------------------------------
# TablePlan -- one table's share of the work
# ---------------------------------------------------------------------------


TABLE_NAME = "`common_task`.`_task_epoch`"


def _table_plan(**kwargs):
    """Build a TablePlan with sensible defaults.

    Targets the table by name, as a plan rebuilt from storage does -- these
    tests are about the plan objects, not about any live table.
    """
    entries = PlannedEntries()
    entries.add(_NamedTable(TABLE_NAME), [{"a": 1}, {"a": 2}])
    defaults = dict(
        table_name=TABLE_NAME,
        entries=entries.freeze(),
        status="ok",
        problems=(),
    )
    defaults.update(kwargs)
    return TablePlan(**defaults)


def test_table_plan_counts_its_entries():
    """A plan knows how much work it holds, without unpacking it."""
    assert _table_plan().entry_count == 2


def test_table_plan_is_immutable():
    """A plan is a record of a decision, not a working buffer."""
    plan = _table_plan()

    with pytest.raises((AttributeError, TypeError)):
        plan.status = "failed"


def test_table_plan_reports_its_worst_problem():
    """The severity that matters is the highest one present."""
    plan = _table_plan(
        status="failed",
        problems=(
            Problem("soft", "absent", "no optional object"),
            Problem("hard", "bad_fk", "parent missing"),
            Problem("info", "duplicate", "already present"),
        ),
    )

    assert plan.severity == "hard"


def test_table_plan_without_problems_has_no_severity():
    """A clean plan reports nothing rather than a sentinel severity."""
    assert _table_plan().severity is None


# ---------------------------------------------------------------------------
# IngestionPlan -- the whole file
# ---------------------------------------------------------------------------


def _plan(*table_plans, **kwargs):
    """Build an IngestionPlan with sensible defaults."""
    defaults = dict(
        nwb_file_name="mini_.nwb",
        table_plans=tuple(table_plans),
    )
    defaults.update(kwargs)
    return IngestionPlan(**defaults)


def test_plan_is_clean_when_nothing_blocks():
    """Soft and info problems do not make a plan dirty.

    An absent optional object is not a failure; it is the file saying it has
    no such data.
    """
    plan = _plan(
        _table_plan(problems=(Problem("soft", "absent", "no such object"),)),
        _table_plan(problems=(Problem("info", "duplicate", "already there"),)),
    )

    assert plan.is_clean
    assert not plan, "A clean plan is falsy -- nothing blocks it"


def test_plan_is_not_clean_with_a_hard_problem():
    """A hard problem blocks its table, so the plan is not clean."""
    plan = _plan(
        _table_plan(status="failed", problems=(Problem("hard", "e", "m"),))
    )

    assert not plan.is_clean
    assert plan, "A plan with blocking problems is truthy"
    assert len(plan.hard_failures) == 1


def test_plan_surfaces_fatal_problems_separately():
    """A fatal problem stops the file, not just one table."""
    plan = _plan(_table_plan(), fatal=(Problem("fatal", "unreadable", "io"),))

    assert plan.fatal
    assert not plan.is_clean


def test_plan_counts_entries_across_tables():
    """Entry count spans every table's plan."""
    assert _plan(_table_plan(), _table_plan()).entry_count == 4


def test_plan_hash_is_stable_and_content_sensitive():
    """The same plan hashes the same; a different one does not.

    The hash is what lets a later attempt recognise the work it already did.
    """
    first = _plan(_table_plan())
    second = _plan(_table_plan())
    third = _plan(_table_plan(status="failed"))

    assert first.plan_hash == second.plan_hash
    assert first.plan_hash != third.plan_hash


def test_plan_round_trips_through_a_dict():
    """A plan can be stored and rebuilt without losing its verdict."""
    plan = _plan(
        _table_plan(problems=(Problem("hard", "bad_fk", "parent missing"),))
    )

    rebuilt = IngestionPlan.from_dict(plan.to_dict())

    assert rebuilt.nwb_file_name == plan.nwb_file_name
    assert rebuilt.entry_count == plan.entry_count
    assert len(rebuilt.hard_failures) == len(plan.hard_failures)
    assert rebuilt.plan_hash == plan.plan_hash


def test_plan_reports_per_table_status():
    """A plan can say which tables succeeded and which did not."""
    plan = _plan(
        _table_plan(table_name="`s`.`a`"),
        _table_plan(table_name="`s`.`b`", status="failed"),
    )

    assert plan.status_by_table() == {"`s`.`a`": "ok", "`s`.`b`": "failed"}


# ---------------------------------------------------------------------------
# plan_from_nwbfile -- parsing without inserting
# ---------------------------------------------------------------------------


def test_planning_produces_entries_without_inserting(
    common, mini_copy_name, mini_insert
):
    """Planning yields the entries a table would write, and writes none."""
    table = common.SampleCount()
    restr = {"nwb_file_name": mini_copy_name}

    (table & restr).super_delete(warn=False, safemode=False)
    try:
        plan = table.plan_from_nwbfile(mini_copy_name)

        assert plan.entry_count == 1, "Plan should hold the entry to insert"
        assert not len(table & restr), "Planning must not write to a data table"
    finally:
        table.insert_from_nwbfile(mini_copy_name)


def test_planning_a_missing_source_is_soft_not_fatal(
    common, mini_copy_name, mini_insert
):
    """A table with no source object in the file plans cleanly and empty."""
    plan = common.StateScriptFile().plan_from_nwbfile(mini_copy_name)

    assert plan.entry_count == 0
    assert plan.severity != "hard", "An absent source is not a hard failure"


def test_planning_converts_a_raise_into_a_problem(
    common, mini_copy_name, mini_insert, monkeypatch
):
    """An exception while parsing becomes a hard problem, not a traceback.

    The point of the pass is to report everything wrong with a file in one
    go, which a raise out of the first bad table would prevent.
    """
    table = common.SampleCount()

    def _boom(self, row, ctx):
        raise RuntimeError("mapping blew up")

    monkeypatch.setattr(type(table), "entries_for_row", _boom, raising=False)

    plan = table.plan_from_nwbfile(mini_copy_name)

    assert plan.severity == "hard", "A raise should be recorded, not raised"
    assert any(
        "mapping blew up" in problem.message for problem in plan.problems
    ), f"Expected the error's message, saw {plan.problems}"
    assert plan.problems[0].exc_type == "RuntimeError"


def test_plan_records_what_it_read(common, mini_copy_name, mini_insert):
    """The plan carries the objects the table read, for later reuse checks."""
    plan = common.SampleCount().plan_from_nwbfile(mini_copy_name)

    assert plan.reads, "Planning should record the source objects it read"


def test_insert_from_nwbfile_still_returns_entries(
    common, mini_copy_name, mini_insert
):
    """The public return shape is unchanged by the plan/insert split."""
    table = common.SampleCount()
    restr = {"nwb_file_name": mini_copy_name}

    # Cleared first: this table has no duplicate handling, so re-inserting an
    # already-ingested file raises rather than no-opping.
    (table & restr).super_delete(warn=False, safemode=False)
    entries = table.insert_from_nwbfile(mini_copy_name)

    assert isinstance(entries, dict), "Callers still receive a table->rows map"
    assert len(table & restr), "The entries should have been inserted"
