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
from spyglass.utils.ingestion_plan import IngestionPlan


@pytest.fixture
def clean_plan(common, mini_copy_name, mini_insert):
    """A plan for a file this fixture has *made sure* is fully ingested.

    `mini_insert` ingests once per session, so a fixture that merely depended
    on it was asserting the suite's history: any test in between may leave a
    table short of rows -- deliberately, as several do -- and the verdict then
    reads `partial_new` for reasons that have nothing to do with the test.

    So establish the premise rather than assume it: plan, insert whatever is
    still outstanding, and re-plan. Only rows the file itself describes are
    inserted, which is the state the suite expects anyway, so this converges
    toward the shared baseline rather than away from it. Extra rows left by
    another test do not matter here -- novelty counts what is planned and
    missing, not what is stored and unplanned.
    """
    from spyglass.data_import.planner import insert_plan

    plan = plan_nwbfile(mini_copy_name)

    if plan.verdict != "no_op":  # repair, then look again
        insert_plan(plan, on_divergence="report", allow_partial=True)
        plan = plan_nwbfile(mini_copy_name)

    assert plan.verdict == "no_op", (
        "Fixture could not reach a fully-ingested file; still outstanding: "
        + f"{plan.new_entries_by_table()}"
    )

    return plan


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

    plan = plan_nwbfile(mini_copy_name, force_replan=True)

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
    IntervalList pattern -- and those parents are not in the database yet, so
    the check has to consider the plan as well as the database.

    A smoke check only, and worth saying why: this file is already ingested,
    so its IntervalList rows exist and the foreign-key check resolves against
    them whatever order the plan is in. The ordering this guards is only
    observable when those parents are novel, which is the clean-database
    golden-run comparison -- where exactly this bug did surface, as 18
    missing_parent problems naming parents the plan held two entries later.
    Forcing an adverse order here does not reproduce it; that was tried.
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


def test_conflicting_entries_for_one_key_are_reported(
    common, mini_copy_name, mini_insert, monkeypatch
):
    """Same key, different values is a problem, not a crash.

    The mid-transaction DuplicateError, moved before the transaction. Neither
    row is stored, so there is no stored value to defer to.
    """

    def _conflicting(self, source, ctx):
        from spyglass.utils.ingestion_plan import PlannedEntries

        entries = PlannedEntries()
        row = dict(ctx.base_key, sample_count_object_id="x" * 8)
        entries.add(self, [row, dict(row, sample_count_object_id="y" * 8)])
        return entries

    monkeypatch.setattr(
        type(common.SampleCount()),
        "entries_for_row",
        _conflicting,
        raising=False,
    )

    plan = plan_nwbfile(mini_copy_name, force_replan=True)

    assert any(
        problem.code == "duplicate_key" for problem in plan.problems
    ), f"Expected a duplicate_key problem, saw {plan.problems}"


def test_identical_repeats_of_one_key_collapse(
    common, mini_copy_name, mini_insert, monkeypatch
):
    """Same key, same values is normal content and must not block.

    Two task epochs naming one task emit that task twice. The direct insert
    path has always collapsed these; blocking them instead would refuse files
    that ingested fine before planning existed.
    """

    def _twice(self, source, ctx):
        from spyglass.utils.ingestion_plan import PlannedEntries

        entries = PlannedEntries()
        row = dict(ctx.base_key, sample_count_object_id="x" * 8)
        entries.add(self, [row, dict(row)])
        return entries

    monkeypatch.setattr(
        type(common.SampleCount()), "entries_for_row", _twice, raising=False
    )

    plan = plan_nwbfile(mini_copy_name, force_replan=True)

    duplicates = [p for p in plan.problems if p.code == "duplicate_key"]
    assert not duplicates, f"Identical repeats must collapse, got {duplicates}"

    counts = {
        tp.table_name: tp.entry_count
        for tp in plan.table_plans
        if tp.table_name == common.SampleCount().full_table_name
    }
    assert counts, "SampleCount should still plan its row"
    assert all(
        n == 1 for n in counts.values()
    ), f"Expected one row after collapsing, got {counts}"


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


def test_a_divergence_is_reported_without_changing_the_verdict(
    common, mini_copy_name, mini_insert, monkeypatch
):
    """A disagreement is a warning; the verdict still says what will happen.

    It used to answer `conflict` ahead of counting novelty, so one trivial
    mismatch headlined a file that was otherwise entirely new (D7).
    """

    def _changed(self, source, ctx):
        from spyglass.utils.ingestion_plan import PlannedEntries

        entries = PlannedEntries()
        entries.add(
            self,
            [dict(ctx.base_key, sample_count_object_id="deadbeef" * 5)],
        )
        return entries

    monkeypatch.setattr(
        type(common.SampleCount()), "entries_for_row", _changed, raising=False
    )

    # force_replan: reuse would serve the staged plan and the patched parse
    # above would never run, so the test would assert on an empty problem list.
    plan = plan_nwbfile(mini_copy_name, force_replan=True)

    assert plan.verdict != "conflict", "conflict left the vocabulary"
    divergences = [p for p in plan.problems if p.code == "divergence"]
    assert divergences, "The disagreement must still be reported"
    assert all(
        p.severity == "soft" for p in divergences
    ), "A divergence is a warning, not a blocker"
    assert not plan.blocking, "It must not block"
    assert "Disagrees with stored rows" in plan.report(
        log=False
    ), "A soft divergence must still appear in the default report"


def test_report_leads_with_the_verdict(clean_plan):
    """The report is readable, and says the verdict first."""
    report = clean_plan.report()

    assert isinstance(report, str)
    assert "no_op" in report or "already ingested" in report.lower()


# --- divergence policy (D7) -------------------------------------------------
# A divergence is the file disagreeing with a row already stored. The policy
# decides what a *real* run does about it; a dry run only ever records it.


def test_a_fatal_plan_does_not_close_its_staging_area(common, mini_copy_name):
    """A file that could not be read has not "already been ingested".

    A fatal plan holds no table plans, so nothing is novel, so the verdict
    used to read `no_op` -- and `insert_plan` checked that first. The result
    was an unreadable file logging "already ingested, nothing to do" and
    marking its staging area complete. The return value was truthy, so a
    caller testing it still saw the failure; the log and the plan record did
    not.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import insert_plan, plan_nwbfile

    unregistered = "_no_such_file_.nwb"
    plan = plan_nwbfile(unregistered)

    assert plan.verdict == "fatal", f"Expected fatal, got {plan.verdict}"

    result = insert_plan(plan, on_divergence="report")

    assert result, "A file that could not be planned is not a success"
    assert not (
        IngestionPlanLog & {"nwb_file_name": unregistered}
    ), "Nothing to stage, and certainly nothing to mark complete"


def test_unknown_divergence_policy_is_refused(common, mini_copy_name):
    """A misspelled policy must not silently fall through to a default."""
    import pytest as _pytest

    from spyglass.utils.ingestion_plan import IngestionPlan
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
    from spyglass.utils.ingestion_plan import (
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
                entries=entries,
            ),
        ),
    )

    monkeypatch.setattr(
        planner,
        "_novel_rows",
        lambda table, rows: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    result = planner.insert_plan(plan, on_divergence="report")

    assert any(
        problem.code == "planner_miss" for problem in result
    ), f"A failure inserting a validated plan is a planner_miss: {result!r}"
    assert not rolled, "rollback_on_miss defaults to False"

    planner.insert_plan(plan, on_divergence="report", rollback_on_miss=True)
    assert rolled == [mini_copy_name], "Asked for, it rolls back that file"


def _divergent_plan(common, nwb_file_name):
    """A plan whose only problem is one divergence."""
    from spyglass.utils.ingestion_plan import (
        IngestionPlan,
        PlannedEntries,
        Problem,
        TablePlan,
    )

    entries = PlannedEntries()
    entries.add(common.Institution, [{"institution_name": "_divergence test"}])
    return IngestionPlan(
        nwb_file_name=nwb_file_name,
        novel={"`common_lab`.`institution`": 1},
        table_plans=(
            TablePlan(
                table_name="`common_lab`.`institution`",
                entries=entries,
                problems=(
                    Problem(
                        severity="soft",
                        code="divergence",
                        message="stored with different values",
                        table="`common_lab`.`institution`",
                        suggested_revision={"institution_name": "other"},
                        primary_key={"institution_name": "_divergence test"},
                    ),
                ),
            ),
        ),
    )


def test_a_divergence_does_not_stop_the_run(common, mini_copy_name):
    """Reporting a disagreement must not stop the rest of the file.

    This is the babysit behaviour removed in D7: a divergence used to be
    `hard`, so the gate refused an unattended run over a mismatch as small as
    a hyphen. Pinned on the gate rather than a whole ingestion: `_refuse`
    returning None is what "go on and insert" means.
    """
    from spyglass.data_import import planner
    from spyglass.utils.ingestion_plan import (
        IngestionPlan,
        PlannedEntries,
        Problem,
        TablePlan,
    )

    entries = PlannedEntries()
    entries.add(common.Institution, [{"institution_name": "_divergence test"}])
    plan = IngestionPlan(
        nwb_file_name=mini_copy_name,
        novel={"`common_lab`.`institution`": 1},
        table_plans=(
            TablePlan(
                table_name="`common_lab`.`institution`",
                entries=entries,
                problems=(
                    Problem(
                        severity="soft",
                        code="divergence",
                        message="stored with different values",
                        table="`common_lab`.`institution`",
                        suggested_revision={"institution_name": "other"},
                    ),
                ),
            ),
        ),
    )

    assert not plan.blocking, "Premise: a divergence is soft, so never blocking"

    assert (
        planner._refuse(plan, allow_partial=False, on_divergence="report")
        is None
    ), "reporting a divergence lets the run proceed"

    assert (
        planner._refuse(plan, allow_partial=False, on_divergence="raise")
        is not None
    ), "raise still declines, so the two are not the same outcome"


def test_accepting_a_divergence_does_not_excuse_other_failures(
    common, mini_copy_name
):
    """A divergence stops blocking; a real failure does not.

    The narrow reading matters: making a disagreement non-blocking must not
    become a way to insert a file with a missing parent or an unset required
    column.
    """
    from spyglass.data_import import planner
    from spyglass.utils.ingestion_plan import (
        IngestionPlan,
        PlannedEntries,
        Problem,
        TablePlan,
    )

    entries = PlannedEntries()
    entries.add(common.Institution, [{"institution_name": "_mixed test"}])
    plan = IngestionPlan(
        nwb_file_name=mini_copy_name,
        novel={"`common_lab`.`institution`": 1},
        table_plans=(
            TablePlan(
                table_name="`common_lab`.`institution`",
                entries=entries,
                problems=(
                    Problem(
                        severity="hard",
                        code="divergence",
                        message="stored with different values",
                        table="`common_lab`.`institution`",
                    ),
                    Problem(
                        severity="hard",
                        code="missing_parent",
                        message="no such parent row",
                        table="`common_lab`.`institution`",
                    ),
                ),
            ),
        ),
    )

    assert (
        planner._refuse(plan, allow_partial=False, on_divergence="report")
        is not None
    ), "A missing parent still blocks, whatever was decided about divergence"


def test_a_soft_problem_is_visible_in_the_default_report(
    common, mini_copy_name
):
    """A warning nobody sees is not a warning.

    D7 replaces the divergence prompt with a line in the report, which only
    works if the default report shows it. `report(verbose=False)` used to show
    blocking problems only, so making divergence `soft` without this would
    have hidden it completely -- quieter than the behaviour it replaced.
    """
    from spyglass.utils.ingestion_plan import (
        IngestionPlan,
        PlannedEntries,
        Problem,
        TablePlan,
    )

    plan = IngestionPlan(
        nwb_file_name=mini_copy_name,
        table_plans=(
            TablePlan(
                table_name="`common_lab`.`institution`",
                entries=PlannedEntries(),
                problems=(
                    Problem(
                        severity="soft",
                        code="divergence",
                        message="{'institution_name': 'x'} exists with "
                        + "different values for ['institution_name']",
                        table="`common_lab`.`institution`",
                        suggested_revision={"institution_name": "stored"},
                        primary_key={"institution_name": "x"},
                    ),
                ),
            ),
        ),
    )

    report = plan.report(log=False)

    assert "Disagrees with stored rows (1)" in report
    assert "institution_name" in report
    assert not plan.blocking, "A soft problem must not block"
    assert not plan, "A plan with only warnings is falsy"


def test_on_divergence_report_proceeds_and_raise_declines(
    common, mini_copy_name
):
    """The two policies left, pinned on the gate.

    `interactive` and `accept` are gone: accept became the default, so the
    name stopped meaning anything, and the prompt is the behaviour D7
    removes. What remains is report -- warn and insert the rest -- and raise,
    for a caller that wants a disagreement to be an error.
    """
    from spyglass.data_import import planner

    plan = _divergent_plan(common, mini_copy_name)

    assert (
        planner._refuse(plan, allow_partial=False, on_divergence="report")
        is None
    ), "report proceeds"

    refused = planner._refuse(plan, allow_partial=False, on_divergence="raise")
    assert refused is not None, "raise declines"
    assert any(
        p.code == "divergence" for p in refused.problems
    ), "and hands back the plan carrying why"


def test_nothing_prompts_on_a_divergence(common, mini_copy_name, monkeypatch):
    """No path may read stdin over a disagreement.

    The regression this guards is the whole point of D7: an ingest that stops
    to ask cannot run unattended, and a suite that prompts hangs rather than
    fails.
    """
    from spyglass.data_import import planner

    def explode(prompt=""):  # pragma: no cover - must not be reached
        raise AssertionError(f"something prompted: {prompt!r}")

    monkeypatch.setattr("builtins.input", explode)

    plan = _divergent_plan(common, mini_copy_name)
    for policy in ("report", "raise"):
        planner._refuse(plan, allow_partial=False, on_divergence=policy)
