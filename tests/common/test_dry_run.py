"""Dry runs: report what a file would insert, and insert none of it.

The flag is opt-in. The default path is unchanged, so these tests are as much
about what a dry run leaves alone as about what it reports.
"""

import pytest

# Every data table an ingestion touches would show a dry run's writes. Chosen
# to span the shapes: a master, a part, a blob-carrying table, and one fed by
# a table other than itself (Task, from TaskEpoch).
WATCHED = (
    "Session",
    "Electrode",
    "ElectrodeGroup",
    "IntervalList",
    "TaskEpoch",
    "Task",
    "Subject",
    "Institution",
    "Raw",
)


@pytest.fixture
def counts(common):
    """Return a callable giving the current row count of each watched table."""

    def _counts():
        return {name: len(getattr(common, name)()) for name in WATCHED}

    return _counts


@pytest.fixture
def missing_leaf(common, mini_copy_name, mini_insert):
    """Remove one leaf table's rows, so a real insert would be visible.

    Without this, a dry run of a fully-ingested file cannot be distinguished
    from a real one: both write nothing, because there is nothing left to
    write. `super_delete` rather than `delete` -- the cautious path follows a
    row delete by removing the NWB file from disk, which would take the test
    corpus with it. `safemode=False` because `warn=False` silences only the
    bypass warning: without it the delete asks "Commit deletes?" on a stdin
    pytest has captured, and the suite hangs rather than fails.

    Establishes its precondition rather than asserting it. The suite runs
    against a container that persists between runs, so a previous run killed
    between the delete and the restore would otherwise leave every later run
    erroring in setup.
    """
    table = common.SampleCount()
    key = {"nwb_file_name": mini_copy_name}

    if not (table & key):
        table.insert_from_nwbfile(mini_copy_name)

    (table & key).super_delete(warn=False, safemode=False)
    yield table, key

    if not (table & key):  # restore, so later tests see a complete file
        table.insert_from_nwbfile(mini_copy_name)


def test_dry_run_does_not_insert_what_it_reports_as_new(
    common, mini_copy_name, missing_leaf
):
    """The test with teeth: something *is* missing, and stays missing.

    A dry run of a complete file writes nothing whatever the flag does, so
    the flag has to be exercised against a file with work outstanding.
    """
    from spyglass.common.populate_all_common import populate_all_common

    table, key = missing_leaf

    plan = populate_all_common(mini_copy_name, dry_run=True)

    assert plan.verdict == "partial_new", f"Expected work to do: {plan.verdict}"
    assert plan.new_entries_by_table(), "The plan should name what is new"
    assert not (table & key), "A dry run inserted what it only reported"


def test_dry_run_writes_to_no_data_table(
    common, mini_copy_name, missing_leaf, counts
):
    """The invariant the whole design rests on (D2).

    A dry run touches log tables only. Asserted across every watched table
    rather than the one a bug would most likely hit -- and with work
    outstanding, so a run that quietly inserted would move a count.
    """
    from spyglass.common.populate_all_common import populate_all_common

    before = counts()
    populate_all_common(mini_copy_name, dry_run=True)

    assert counts() == before, "A dry run wrote to a data table"


def test_dry_run_returns_a_plan_that_reads_as_the_old_error_list(
    common, mini_copy_name, mini_insert
):
    """Falsy when clean, and printable as the report.

    An already-ingested file is clean: there is nothing to do, which is not a
    failure. A caller written against the old empty-list-means-success
    contract keeps working.
    """
    from spyglass.common.populate_all_common import populate_all_common
    from spyglass.utils.ingestion_plan import IngestionPlan

    plan = populate_all_common(mini_copy_name, dry_run=True)

    assert isinstance(plan, IngestionPlan)
    assert not plan, "Nothing blocks an already-ingested file"
    assert len(plan) == 0
    assert mini_copy_name in str(plan), "str() gives the report"


def test_dry_run_stages_its_plan(common, mini_copy_name, mini_insert):
    """The plan is kept, so a later attempt can see what this one worked out.

    Asserted on the attempt counter rather than on a row existing: the file
    may well have been planned before, and `stage` advances the counter, so
    this pins that *this* call staged rather than that some call once did.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.common.populate_all_common import populate_all_common

    key = {"nwb_file_name": mini_copy_name}
    before = (
        (IngestionPlanLog & key).fetch1("attempt")
        if (IngestionPlanLog & key)
        else 0
    )

    populate_all_common(mini_copy_name, dry_run=True)
    staged = IngestionPlanLog & key

    assert staged, "A dry run writes to the log tables -- that is the point"
    assert staged.fetch1("attempt") == before + 1, "This run did not stage"
    assert staged.fetch1("status") == "open", "Nothing was inserted"


def test_dry_run_reports_every_problem_not_just_the_first(
    common, mini_copy_name, mini_insert, monkeypatch
):
    """The headline reason for planning: one report, not one error per table.

    Two unrelated tables are made to fail. The old path logged them one at a
    time as each raised, and stopped writing at the first; a plan names both
    before anything is written.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.common.populate_all_common import populate_all_common
    from spyglass.utils.mixins.ingestion import IngestionMixin

    # Nothing staged, so nothing is reusable: per-table reuse would skip the
    # parse this test induces a failure in, and the file would report clean.
    IngestionPlanLog().clear(mini_copy_name)

    broken = {"`common_session`.`_session`", "`common_ephys`.`_electrode`"}
    original = IngestionMixin._parse

    def _parse(self, ctx):
        if self.full_table_name in broken:
            raise ValueError(f"induced failure in {self.camel_name}")
        return original(self, ctx)

    monkeypatch.setattr(IngestionMixin, "_parse", _parse)

    plan = populate_all_common(mini_copy_name, dry_run=True)
    failed = {
        problem.table
        for problem in plan.blocking
        if problem.code == "parse_error"
    }

    assert plan, "A file with two broken tables must not report as clean"
    assert failed == broken, f"Both failures should be reported, got {failed}"


def test_a_real_run_is_unchanged_by_the_flag_existing(
    common, mini_copy_name, mini_insert
):
    """D4: the flag is opt-in, so the default path behaves as before.

    `populate_all_common` on an ingested file still returns the old value --
    None, or a list of InsertError keys -- rather than a plan.
    """
    from spyglass.common.populate_all_common import populate_all_common
    from spyglass.utils.ingestion_plan import IngestionPlan

    result = populate_all_common(mini_copy_name)

    assert not isinstance(
        result, IngestionPlan
    ), "The default path must not start returning a plan"
    assert result is None or isinstance(result, (list, tuple))


def test_insert_sessions_dry_run_copies_nothing_and_registers_nothing(
    common, mini_path, mini_copy_name, mini_insert, counts
):
    """8.1: a dry run through the top-level entry point writes nothing.

    `insert_sessions` normally copies the raw file to `_.nwb` and registers it
    in `Nwbfile` before populating. Both are writes, so a dry run does
    neither -- including the `reinsert` delete, which would remove a session
    to report on it.
    """
    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.data_import import insert_sessions

    nwbfiles_before = len(Nwbfile())
    before = counts()

    results = insert_sessions(mini_path.name, dry_run=True)

    assert len(results) == 1, "One result per file, even when already present"
    assert not results[0], "The ingested file plans cleanly"
    assert len(Nwbfile()) == nwbfiles_before, "A dry run registered a file"
    assert counts() == before, "A dry run wrote to a data table"


def test_dry_run_of_an_unregistered_file_reports_rather_than_registering(
    common,
):
    """The scope limit, pinned so it stays deliberate.

    Planning reads the `_.nwb` copy and the tables keyed by `nwb_file_name`
    refer to the `Nwbfile` row for it, so a file Spyglass has never seen
    cannot be planned without first writing both. A dry run says so instead
    of quietly registering the file — which would be a data-table write, the
    one thing the flag promises not to do.
    """
    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.common.populate_all_common import populate_all_common

    unknown = "_never_ingested_dry_run_.nwb"
    before = len(Nwbfile())

    plan = populate_all_common(unknown, dry_run=True)

    assert plan, "An unplannable file is not a clean result"
    assert plan.verdict == "fatal"
    assert {p.code for p in plan.blocking} == {"file_not_registered"}
    assert len(Nwbfile()) == before, "A dry run registered the file"


def test_insert_sessions_dry_run_plans_an_already_ingested_file(
    common, mini_path, mini_copy_name, mini_insert
):
    """A real run skips a registered file; a dry run reports on it.

    Reporting on a file that is already in `Nwbfile` is the common reason to
    ask, so the skip that protects a real run from double-inserting must not
    also suppress the report.
    """
    from spyglass.data_import import insert_sessions
    from spyglass.utils.ingestion_plan import IngestionPlan

    results = insert_sessions(mini_path.name, dry_run=True)

    assert len(results) == 1, "The registered file was skipped, not planned"
    assert isinstance(results[0], IngestionPlan)
    assert results[0].nwb_file_name == mini_copy_name
