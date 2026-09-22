"""Specification for whole-file planning and prospective integrity.

Written before the implementation. Planning a file answers three questions
without writing anything:

  1. what would be inserted, table by table
  2. what would fail, and which tables are blocked as a consequence rather
     than being separate failures
  3. whether any of it is new -- the novelty verdict that replaces a wall of
     duplicate errors on a re-run

Measured on a file with one object removed: `Session` failed and six
downstream tables each reported an IntegrityError. One root cause should read
as one problem.
"""

import pytest

from spyglass.data_import.planner import plan_nwbfile
from spyglass.data_import.ingestion_plan import IngestionPlan


@pytest.fixture
def clean_plan(common, mini_copy_name, mini_insert):
    """A plan for the already-ingested test file."""
    return plan_nwbfile(mini_copy_name)


# ---------------------------------------------------------------------------
# One pass, one open file, dependency order
# ---------------------------------------------------------------------------


def test_planning_covers_the_ingestion_tables(clean_plan):
    """Every table ingestion touches gets a plan of its own."""
    planned = set(clean_plan.status_by_table())

    assert len(planned) > 20, f"Expected the full table set, saw {len(planned)}"
    assert any("session" in name for name in planned)
    assert any("electrode" in name for name in planned)


def test_planning_orders_parents_before_children(clean_plan):
    """Tables are planned in dependency order, not a hand-kept list."""
    order = list(clean_plan.status_by_table())
    names = [name.split(".")[-1].strip("`_") for name in order]

    assert names.index("session") < names.index(
        "electrode_group"
    ), "Session must be planned before tables that depend on it"


def test_planning_writes_nothing(common, mini_copy_name, mini_insert):
    """The pass is read-only, however the file is shaped.

    The invariant the whole design rests on: planning touches log tables
    only, never a data table.
    """
    counts_before = {
        name: len(getattr(common, name)())
        for name in ("Session", "Electrode", "IntervalList", "TaskEpoch")
    }

    plan_nwbfile(mini_copy_name)

    counts_after = {
        name: len(getattr(common, name)()) for name in counts_before
    }

    assert counts_after == counts_before, "Planning wrote to a data table"


def test_planning_returns_an_ingestion_plan(clean_plan):
    """The result is the plan object, carrying the file it describes."""
    assert isinstance(clean_plan, IngestionPlan)
    assert clean_plan.nwb_file_name


# ---------------------------------------------------------------------------
# Integrity, and blocking rather than cascading
# ---------------------------------------------------------------------------


def test_missing_parent_is_reported_once_not_per_child(
    common, mini_copy_name, mini_insert, monkeypatch
):
    """One missing parent is one problem, plus blocked children.

    The alternative was measured before this pass existed: a missing subject
    produced eight InsertError rows describing a single root cause.
    """

    def _no_entries(self, source, ctx):
        raise RuntimeError("pretend this table cannot be parsed")

    monkeypatch.setattr(
        type(common.Session()), "entries_for_row", _no_entries, raising=False
    )

    plan = plan_nwbfile(mini_copy_name)

    hard = [p for p in plan.problems if p.severity == "hard"]
    blocked = [
        name
        for name, status in plan.status_by_table().items()
        if status == "blocked"
    ]

    assert len(hard) == 1, f"Expected one root cause, saw {hard}"
    assert blocked, "Tables depending on the failed one should be blocked"
    assert not plan.is_clean


def test_planned_parents_satisfy_their_children(
    common, mini_copy_name, mini_insert
):
    """An entry whose parent is planned in the same pass is not a failure.

    Several tables emit their parent's rows alongside their own -- the
    IntervalList pattern. Those parents are not in the database yet, so the
    check has to consider the plan as well as the database.
    """
    plan = plan_nwbfile(mini_copy_name)

    interval_problems = [
        problem
        for problem in plan.problems
        if problem.code == "missing_parent"
        and "interval" in (problem.message or "")
    ]

    assert not interval_problems, (
        "Entries whose parent is planned in the same pass were reported as "
        + f"missing: {interval_problems}"
    )


def test_duplicate_primary_keys_within_the_plan_are_caught(
    common, mini_copy_name, mini_insert, monkeypatch
):
    """Two planned entries with one primary key is a problem, not a crash.

    This is the mid-transaction DuplicateError, moved before the transaction.
    """

    def _twice(self, source, ctx):
        from spyglass.data_import.ingestion_plan import PlannedEntries

        entries = PlannedEntries()
        row = dict(ctx.base_key, sample_count_object_id="x" * 8)
        entries.add(self, [row, dict(row)])
        return entries

    monkeypatch.setattr(
        type(common.SampleCount()), "entries_for_row", _twice, raising=False
    )

    plan = plan_nwbfile(mini_copy_name)

    assert any(
        problem.code == "duplicate_key" for problem in plan.problems
    ), f"Expected a duplicate_key problem, saw {plan.problems}"


# ---------------------------------------------------------------------------
# The novelty verdict
# ---------------------------------------------------------------------------


def test_verdict_is_no_op_for_an_already_ingested_file(clean_plan):
    """A complete re-run says so in one word, not in N duplicate errors."""
    assert clean_plan.verdict == "no_op"
    assert clean_plan.is_clean, "Nothing to do is not a failure"


def test_verdict_is_partial_new_when_one_table_is_missing(
    common, mini_copy_name, mini_insert
):
    """A file with some new entries reports which tables they are in."""
    table = common.SampleCount()
    restr = {"nwb_file_name": mini_copy_name}

    (table & restr).super_delete(warn=False, safemode=False)
    try:
        plan = plan_nwbfile(mini_copy_name)

        assert plan.verdict == "partial_new"
        assert any(
            "sample_count" in name for name in plan.new_entries_by_table()
        ), f"Expected sample_count among new entries, saw {plan.new_entries_by_table()}"
    finally:
        table.insert_from_nwbfile(mini_copy_name)


def test_verdict_reports_conflict_over_novelty(
    common, mini_copy_name, mini_insert, monkeypatch
):
    """An entry that exists with different values is a conflict, not new."""

    def _changed(self, source, ctx):
        from spyglass.data_import.ingestion_plan import PlannedEntries

        entries = PlannedEntries()
        entries.add(
            self,
            [dict(ctx.base_key, sample_count_object_id="deadbeef" * 5)],
        )
        return entries

    monkeypatch.setattr(
        type(common.SampleCount()), "entries_for_row", _changed, raising=False
    )

    plan = plan_nwbfile(mini_copy_name)

    assert plan.verdict == "conflict", f"Saw {plan.verdict}"
    assert any(problem.code == "divergence" for problem in plan.problems)


def test_report_leads_with_the_verdict(clean_plan):
    """The report is readable, and says the verdict first."""
    report = clean_plan.report()

    assert isinstance(report, str)
    assert "no_op" in report or "already ingested" in report.lower()


# --- divergence policy (D7) -------------------------------------------------
# A divergence is the file disagreeing with a row already stored. The policy
# decides what a *real* run does about it; a dry run only ever records it.


def _divergence(plan_types=None):
    """One divergence problem, enough to exercise a policy."""
    from spyglass.data_import.ingestion_plan import Problem

    return [
        Problem(
            severity="hard",
            code="divergence",
            message="subject 54321 exists with different values",
            table="`common_subject`.`subject`",
            suggested_revision={"sex": "M"},
        )
    ]


def test_divergence_accept_keeps_the_stored_value():
    """`accept` takes what is already stored and inserts the rest."""
    from spyglass.data_import.planner import _divergence_accepted

    assert _divergence_accepted(
        _divergence(), "accept"
    ), "accept should proceed"


def test_divergence_raise_inserts_nothing():
    """`raise` refuses the run rather than choosing for the user."""
    from spyglass.data_import.planner import _divergence_accepted

    assert not _divergence_accepted(
        _divergence(), "raise"
    ), "raise should decline"


def test_divergence_interactive_does_not_prompt_in_test_mode():
    """An unattended run must fail, not block on stdin.

    A suite that prompted would hang rather than fail, which is worse than
    either outcome. This is the short-circuit `accept_divergence` has always
    had.
    """
    from spyglass.data_import.planner import _divergence_accepted

    assert not _divergence_accepted(
        _divergence(), "interactive"
    ), "interactive must decline under test mode rather than prompt"


def test_divergence_interactive_asks_and_obeys(monkeypatch):
    """Outside test mode it asks once, and takes the answer."""
    import spyglass.settings
    from spyglass.data_import import planner

    monkeypatch.setattr(spyglass.settings, "test_mode", False)

    asked = []

    def fake_input(prompt):
        asked.append(prompt)
        return answer

    monkeypatch.setattr("builtins.input", fake_input)

    answer = "y"
    assert planner._divergence_accepted(_divergence(), "interactive")
    answer = ""
    assert not planner._divergence_accepted(_divergence(), "interactive")

    assert len(asked) == 2, "One prompt per run, not one per divergence"
    assert "disagree" in asked[0]


def test_unknown_divergence_policy_is_refused(common, mini_copy_name):
    """A misspelled policy must not silently fall through to a default."""
    import pytest as _pytest

    from spyglass.data_import.ingestion_plan import IngestionPlan
    from spyglass.data_import.planner import insert_plan

    with _pytest.raises(ValueError, match="on_divergence"):
        insert_plan(
            IngestionPlan(nwb_file_name=mini_copy_name),
            on_divergence="ignore",
        )


def test_rollback_is_off_by_default_and_scoped_to_a_miss(
    common, mini_copy_name, monkeypatch
):
    """A rollback undoes good rows to fix a bug, so it must be asked for.

    Everything the old blanket `rollback_on_fail` guarded against is caught
    at plan time now. The only state worth undoing is a `planner_miss`: a
    plan that validated and then failed halfway, leaving a partial file the
    user never chose.
    """
    from spyglass.data_import import planner
    from spyglass.data_import.ingestion_plan import (
        IngestionPlan,
        PlannedEntries,
        TablePlan,
    )

    rolled = []
    monkeypatch.setattr(planner, "_rollback", lambda name: rolled.append(name))

    # A plan with one novel row, so the insert path is actually entered.
    entries = PlannedEntries()
    entries.add(common.Institution, [{"institution_name": "_miss test"}])
    plan = IngestionPlan(
        nwb_file_name=mini_copy_name,
        novel={"`common_lab`.`institution`": 1},
        table_plans=(
            TablePlan(
                table_name="`common_lab`.`institution`",
                entries=entries.freeze(),
            ),
        ),
    )

    monkeypatch.setattr(
        planner,
        "_novel_rows",
        lambda table, rows: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    result = planner.insert_plan(plan, on_divergence="accept")

    assert any(
        problem.code == "planner_miss" for problem in result
    ), f"A failure inserting a validated plan is a planner_miss: {result!r}"
    assert not rolled, "rollback_on_miss defaults to False"

    planner.insert_plan(plan, on_divergence="accept", rollback_on_miss=True)
    assert rolled == [mini_copy_name], "Asked for, it rolls back that file"
