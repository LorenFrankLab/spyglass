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
        insert_plan(plan, allow_partial=True)
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
    mismatch headlined a file that was otherwise entirely new.
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
    assert (
        "Disagrees with stored rows" in plan.report()
    ), "A soft divergence must still appear in the default report"


def test_report_leads_with_the_verdict(clean_plan):
    """The report is readable, and says the verdict first."""
    report = clean_plan.report()

    assert isinstance(report, str)
    assert "no_op" in report or "already ingested" in report.lower()


# --- divergence policy --------------------------------------------------
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

    result = insert_plan(plan)

    assert result, "A file that could not be planned is not a success"
    assert not (
        IngestionPlanLog & {"nwb_file_name": unregistered}
    ), "Nothing to stage, and certainly nothing to mark complete"


def test_a_planner_miss_leaves_nothing_behind(
    common, mini_copy_name, monkeypatch
):
    """A validated plan that fails partway through writes nothing.

    Two targets, with the second insert raising, so there is a row already
    written when the failure arrives; `calls` pins that premise, because an
    assertion that two rows are absent is satisfied by never inserting either.
    """
    from spyglass.data_import import planner
    from spyglass.utils.ingestion_plan import (
        IngestionPlan,
        PlannedEntries,
        TablePlan,
    )

    institution, lab = "_ffi txn institution", "_ffi txn lab"
    planned = []
    for table, row in (
        (common.Institution, {"institution_name": institution}),
        (common.Lab, {"lab_name": lab}),
    ):
        entries = PlannedEntries()
        entries.add(table, [row])
        planned.append(
            TablePlan(table_name=table().full_table_name, entries=entries)
        )

    plan = IngestionPlan(
        nwb_file_name=mini_copy_name,
        novel={tp.table_name: 1 for tp in planned},
        table_plans=tuple(planned),
    )

    wrote = (common.Institution & {"institution_name": institution}) or (
        common.Lab & {"lab_name": lab}
    )
    assert not wrote, "Premise: neither row is stored before this runs"

    calls = []
    insert_rows = planner._insert_rows

    def failing(target, rows, nwb_file_name):
        calls.append(target)
        if len(calls) > 1:  # whichever target the order puts second
            raise RuntimeError("boom")
        return insert_rows(target, rows, nwb_file_name)

    monkeypatch.setattr(planner, "_insert_rows", failing)

    try:
        result = planner.insert_plan(plan)

        assert any(
            problem.code == "planner_miss" for problem in result
        ), f"A failure inserting a validated plan is a planner_miss: {result!r}"
        assert len(calls) > 1, "Premise: one target inserted before the failure"
        assert not (common.Institution & {"institution_name": institution}), (
            "A row written before the failure must be rolled back with it, "
            + "not left for the caller to delete"
        )
        assert not (common.Lab & {"lab_name": lab})
    finally:
        for query in (
            common.Institution & {"institution_name": institution},
            common.Lab & {"lab_name": lab},
        ):
            if query:  # only reachable if the rollback did not happen
                query.delete_quick()


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
        planner._refuse(plan, allow_partial=False) is None
    ), "reporting a divergence lets the run proceed"


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
        planner._refuse(plan, allow_partial=False) is not None
    ), "A missing parent still blocks, divergence alongside it or not"


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

    report = plan.report()

    assert "Disagrees with stored rows (1)" in report
    assert "institution_name" in report
    assert not plan.blocking, "A soft problem must not block"
    assert not plan, "A plan with only warnings is falsy"


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
    planner._refuse(plan, allow_partial=False)


def test_a_table_is_handed_the_whole_config_not_its_own_slice(
    common, mini_copy_name, mini_insert
):
    """Every config consumer indexes the mapping by a name it chooses.

    The generic hook reads `config[self.camel_name]` and its parts; an
    override reads whatever key it declares, so `TaskEpoch` reads `Tasks`.
    Handing a table `config[camel_name]` instead gave `Electrode` its own row
    list -- `list.get` raised, and the table planned as `failed` with zero
    entries, blocking everything beneath it -- while `TaskEpoch` got an empty
    dict and dropped config-declared tasks without a word.

    Asserted on the shape each table receives, not on the entries it yields,
    because the two defects had opposite symptoms and only the shape is
    common to both.
    """
    from spyglass.utils.mixins.ingestion import IngestionMixin

    config = {
        "Electrode": [{"electrode_id": 0}],
        "Tasks": [{"task_name": "whole_config_probe", "task_epochs": [99]}],
    }

    received = {}
    original = IngestionMixin._parse

    def capture(self, ctx):
        received[self.camel_name] = ctx.config
        return original(self, ctx)

    IngestionMixin._parse = capture
    try:
        plan_nwbfile(
            mini_copy_name,
            config=config,
            tables=[common.Electrode, common.TaskEpoch],
            force_replan=True,
        )
    finally:
        IngestionMixin._parse = original

    for name in ("Electrode", "TaskEpoch"):
        assert name in received, f"{name} never parsed"
        assert received[name] == config, (
            f"{name} was handed {received[name]!r} rather than the whole "
            "config; it indexes the mapping by its own key"
        )


def test_electrode_config_reaches_the_plan(common, mini_copy_name, mini_insert):
    """A config override must land on the planned row, not only on an insert.

    `Electrode` held the config on `self` via an `insert_from_nwbfile`
    override, which the plan path never calls -- so on the only ingest path
    there is, every override was dropped. Nothing failed; the rows were just
    quietly wrong.
    """
    target = 0
    config = {"Electrode": [{"electrode_id": target, "filtering": "probe"}]}

    plan = plan_nwbfile(
        mini_copy_name,
        config=config,
        tables=[common.Electrode],
        force_replan=True,
    )

    electrode = common.Electrode().full_table_name
    rows = [
        row
        for table_plan in plan.table_plans
        for tbl, planned in table_plan.entries
        if getattr(tbl, "full_table_name", None) == electrode
        for row in planned
        if row.get("electrode_id") == target
    ]

    assert rows, f"Electrode planned no row for electrode_id {target}"
    assert rows[0]["filtering"] == "probe", (
        "the config override did not reach the planned row: "
        f"{rows[0].get('filtering')!r}"
    )


def test_planning_never_reads_a_file_keyed_table_whole(
    common, mini_copy_name, mini_insert
):
    """Every read of a file-keyed table is narrowed to the file being planned.

    `existing_keys` used to issue an unrestricted `fetch("KEY")` for each
    table the plan touched *and each parent of each* — 29 tables on a median
    file, scaling with the whole database rather than the file, and paid by a
    dry run too. On a shared server that is the dominant cost of planning.

    Asserted on the restriction each read carries rather than on timing, which
    says nothing on a near-empty test database: a read of a file-keyed table
    that carries no restriction is unbounded however fast it happens to be
    here.

    The shared metadata tables (`Task`, `Institution`, `ProbeType`, …) are
    expected to be read whole — they carry no `nwb_file_name` to narrow by and
    stay small by nature — so the assertion is scoped to tables that do.
    """
    from spyglass.data_import import planner

    unrestricted = []
    original_keys = planner.stored_keys
    original_rows = planner.VirtualKeySpace.rows_for

    def watched_keys(table, restriction=True):
        if restriction is True and "nwb_file_name" in table.primary_key:
            unrestricted.append(f"stored_keys({table.full_table_name})")
        return original_keys(table, restriction)

    def watched_rows(self, table, restriction=None):
        if (
            self._scope(table) is True
            and not restriction
            and "nwb_file_name" in table.primary_key
        ):
            unrestricted.append(f"rows_for({table.full_table_name})")
        return original_rows(self, table, restriction)

    planner.stored_keys = watched_keys
    planner.VirtualKeySpace.rows_for = watched_rows
    try:
        plan = planner.plan_nwbfile(
            mini_copy_name, config={}, force_replan=True
        )
    finally:
        planner.stored_keys = original_keys
        planner.VirtualKeySpace.rows_for = original_rows

    assert plan.entry_count, "premise: the plan read something"
    assert not unrestricted, (
        "planning read a file-keyed table without narrowing to the file: "
        + ", ".join(sorted(set(unrestricted)))
    )
