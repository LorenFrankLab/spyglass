"""Specification for the plan objects and the parse-only pass.

Written before the implementation. A plan is what parsing produces *instead
of* inserting: the entries a table would write, the problems it hit, and
enough provenance to decide later whether it can still be trusted.

The invariant these exist to enforce: parsing an NWB file writes nothing to a
data table, however badly the file is malformed. Every failure becomes a
`Problem` on the plan rather than an exception out of it.
"""

import pytest

from dataclasses import dataclass

from spyglass.utils.ingestion_plan import (
    IngestionPlan,
    PlannedEntries,
    Problem,
    TablePlan,
)


@dataclass(frozen=True)
class _NamedTable:
    """A target known only by its name.

    These tests are about the plan objects, so they need a target that names
    a table without one being declared or reachable.
    """

    full_table_name: str


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
        entries=entries,
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
    assert len(plan.blocking) == 1


def test_plan_surfaces_fatal_problems_separately():
    """A fatal problem stops the file, not just one table."""
    plan = _plan(_table_plan(), fatal=(Problem("fatal", "unreadable", "io"),))

    assert plan.fatal
    assert not plan.is_clean


def test_plan_counts_entries_across_tables():
    """Entry count spans every table's plan."""
    assert _plan(_table_plan(), _table_plan()).entry_count == 4


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


# ---------------------------------------------------------------------------
# Key identity -- a Spyglass rule, strictly coarser than any backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b",
    [
        ("Sleep", "sleep"),  # case
        ("Sleep", " Sleep "),  # surrounding whitespace
        ("Sleep", "SLEEP "),  # both
    ],
)
def test_keys_differing_only_in_case_or_space_are_one_identity(a, b):
    """Spyglass names one thing once, however it was typed.

    Stricter than the backend, so it never offers the database a row the
    insert will reject.
    """
    from spyglass.utils.ingestion_plan import collated

    assert collated(a) == collated(b), f"{a!r} and {b!r} must be one identity"


def test_accents_are_a_known_and_accepted_gap():
    """`Müller` and `Muller` stay two identities; the server calls them one.

    Not a bug to fix but a line drawn: folding accents needs NFKD
    normalisation for a case that effectively does not arise, so the pair
    reaches the insert and fails as a `planner_miss`. Pinned so the gap stays
    deliberate -- if this ever starts biting, this test is where the decision
    is recorded.
    """
    from spyglass.utils.ingestion_plan import collated

    assert collated("Müller") != collated("Muller")


@pytest.mark.parametrize("a, b", [("alpha", "beta"), ("task1", "task2")])
def test_genuinely_different_keys_stay_different(a, b):
    """Strict is not the same as collapsing everything."""
    from spyglass.utils.ingestion_plan import collated

    assert collated(a) != collated(b), f"{a!r} and {b!r} are not one identity"


def test_collated_leaves_non_strings_alone():
    """Only strings carry case; everything else passes through untouched."""
    from spyglass.utils.ingestion_plan import collated

    for value in (0, 1.5, None, b"bytes", True):
        assert collated(value) is value


def test_identity_needs_no_knowledge_of_the_backend():
    """`collated` takes a value and nothing else.

    The first implementation read each table's collation and folded to match,
    which made row identity depend on parsing collation names and on PAD vs
    NO PAD. Pinned here because the pull toward re-adding it is real: it is
    the "accurate" answer, and it is the wrong one -- Spyglass owns this rule.
    """
    import inspect

    from spyglass.utils import ingestion_plan

    assert list(inspect.signature(ingestion_plan.collated).parameters) == [
        "value"
    ], "collated must depend on the value alone"
    assert not hasattr(
        ingestion_plan, "table_collation"
    ), "backend collation lookup should not come back"


def _lettered_task_name(table):
    """A stored task_name containing a letter, so swapcase changes it."""
    for name in sorted(table.fetch("task_name")):
        if name != name.swapcase():
            return name
    return None


def test_spyglass_identity_is_at_least_as_strict_as_the_server(
    common, mini_copy_name, mini_insert
):
    """The invariant that keeps the planner and the insert in agreement.

    If the server calls two keys equal, Spyglass must too -- otherwise a
    stored row reads as novel and the insert rejects it (`planner_miss`), or a
    stored parent reads as missing and `missing_parent` blocks the table and
    everything beneath it. The converse is allowed: Spyglass may be stricter.

    Asserted as an implication rather than an equality, so it holds on a
    case-sensitive deployment too, where the server's answer flips and ours
    does not.
    """
    from spyglass.utils.ingestion_plan import row_key

    task = common.Task()
    stored = _lettered_task_name(task)
    if stored is None:
        pytest.skip("no stored task_name contains a letter to vary")
    varied = stored.swapcase()

    if not (task & {"task_name": varied}):
        pytest.skip("this deployment's collation distinguishes case")

    assert row_key(task, {"task_name": stored}) == row_key(
        task, {"task_name": varied}
    ), (
        f"the server treats {stored!r} and {varied!r} as one key, so Spyglass "
        "must not treat them as two"
    )


def test_a_row_the_server_already_holds_is_not_offered_as_novel(
    common, mini_copy_name, mini_insert
):
    """Novelty must not offer a row the insert will reject.

    Deciding identity with Python's `==` returned a differently-cased row as
    new; the insert then raised `DuplicateError`, which `_write_plan` records
    as `planner_miss` -- whose own remedy text asks the user to report it as a
    planner gap. It was one.
    """
    from spyglass.data_import.planner import _novel_rows

    task = common.Task()
    stored = _lettered_task_name(task)
    if stored is None:
        pytest.skip("no stored task_name contains a letter to vary")
    varied = stored.swapcase()

    if not (task & {"task_name": varied}):
        pytest.skip("this deployment's collation distinguishes case")

    assert not _novel_rows(
        task, [{"task_name": varied, "task_description": "x"}]
    ), f"{varied!r} is already stored as {stored!r} and must not be offered"


def test_a_stored_parent_is_not_reported_missing_under_another_case(
    common, mini_copy_name, mini_insert
):
    """The foreign-key check must find what the insert would find.

    A miss here is `missing_parent` at severity `hard`, which blocks the table
    and every table beneath it -- so a capitalisation difference in a shared
    key took out a whole branch of the plan.
    """
    from spyglass.data_import.planner import VirtualKeySpace
    from spyglass.utils.ingestion_plan import row_key

    task = common.Task()
    stored = _lettered_task_name(task)
    if stored is None:
        pytest.skip("no stored task_name contains a letter to vary")
    varied = stored.swapcase()

    if not (task & {"task_name": varied}):
        pytest.skip("this deployment's collation distinguishes case")

    key_space = VirtualKeySpace()
    assert key_space.holds(
        task, row_key(task, {"task_name": varied})
    ), f"{varied!r} resolves to the stored {stored!r} and must read as present"
