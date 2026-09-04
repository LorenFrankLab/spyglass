"""Baseline behavior of NWB ingestion, pinned ahead of the planner refactor.

These tests describe how `populate_all_common` behaves *today* so the move to
a parse-then-insert pipeline can be shown to preserve it. Each assertion here
is a claim about observable behavior, not about implementation, so it should
survive the refactor unchanged -- and any that cannot survive marks a
deliberate behavior change worth stating in the PR that makes it.
"""

import pytest


@pytest.fixture
def insert_error(common):
    """InsertError, cleared before the test and again afterward.

    Ingestion state persists across runs under `--no-teardown`, so a test
    that inspects the error log has to establish its own starting point.
    """
    from spyglass.common.common_usage import InsertError

    InsertError().delete(safemode=False)
    yield InsertError()
    InsertError().delete(safemode=False)


def test_clean_file_ingests_into_session(mini_insert, common, mini_dict):
    """A well-formed NWB file reaches the Session table."""
    assert common.Session & mini_dict, "Session not populated"


def test_repopulation_adds_no_rows_but_logs_duplicates(
    mini_insert, common, mini_copy_name, insert_error
):
    """Re-populating an ingested file inserts nothing and logs duplicates.

    Baseline behavior, not desired behavior. Row counts hold steady, but the
    tables with a custom `make` raise DuplicateError on the second pass and
    each one lands in InsertError -- so a user re-running ingestion sees a
    wall of failures describing work that was already done.

    The planner should classify these as `info` ("already present"), which
    will make this test's second assertion change. That is expected: update
    it in the commit that makes the change.
    """
    from spyglass.common.populate_all_common import populate_all_common

    counted = ("Session", "Electrode", "ElectrodeGroup", "IntervalList")
    before = {name: len(getattr(common, name)()) for name in counted}

    populate_all_common(mini_copy_name, raise_err=False)

    after = {name: len(getattr(common, name)()) for name in counted}
    assert (
        after == before
    ), f"Re-population changed row counts: {before} -> {after}"

    logged = insert_error.fetch(as_dict=True)
    types = {err["error_type"] for err in logged}
    assert types <= {
        "DuplicateError"
    }, f"Re-population logged more than duplicates: {types}"


def test_absent_source_object_is_skipped_not_raised(mini_insert, common):
    """A file with no source object for a table leaves it empty, quietly.

    The mini file has no optogenetics data. Today that is a silent skip; in
    the planner this becomes a `soft` problem that is reported but does not
    block ingestion.
    """
    assert (
        len(common.OptogeneticProtocol()) == 0
    ), "OptogeneticProtocol populated from a file with no optogenetics data"
    assert (
        len(common.StateScriptFile()) == 0
    ), "StateScriptFile populated from a file with no associated_files"


def test_interval_list_populated_by_dependent_tables(
    mini_insert, common, mini_dict
):
    """IntervalList rows arrive as a side effect of the tables that need them.

    Several ingestion tables emit an IntervalList entry alongside their own.
    The planner has to model those entries explicitly -- parent before child
    -- so this pins that they exist and that dependents resolve to them.
    """
    intervals = set(
        (common.IntervalList & mini_dict).fetch("interval_list_name")
    )
    assert intervals, "No IntervalList entries created during ingestion"

    for table_name in ("Raw", "TaskEpoch"):
        table = getattr(common, table_name)()
        referenced = (table & mini_dict).fetch("interval_list_name")
        for name in referenced:
            assert (
                name in intervals
            ), f"{table_name} references missing interval {name!r}"


@pytest.mark.skip(reason="Phase 2: IngestionPlan not yet implemented")
def test_plan_pass_writes_no_data_rows():
    """A dry run writes to log tables only -- never to a data table.

    The D2 invariant. Asserted against every table reached by the planner,
    not only the ones under review.
    """


@pytest.mark.skip(reason="Phase 3: prospective integrity not yet implemented")
def test_blocked_downstream_reported_once():
    """A hard failure blocks dependents instead of re-reporting each one.

    Baseline evidence (Phase 0.4): deleting `general/subject` makes Session
    fail with ValueError, after which six downstream tables each record an
    IntegrityError. The planner should report one hard problem plus blocked
    dependents.
    """


@pytest.mark.skip(reason="Phase 6: per-object plan reuse not yet implemented")
def test_unrelated_edit_does_not_invalidate_plans():
    """Editing an object no table read leaves every stored table plan valid."""
