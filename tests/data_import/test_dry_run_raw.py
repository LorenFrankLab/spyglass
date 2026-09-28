"""Checking a file before anything at all is written.

`insert_sessions(dry_run=True)` can only report on a file Spyglass already
registered. `insert_sessions` plans the raw file instead, so the answer is
available before the copy and the `Nwbfile` row exist.
"""

from pathlib import Path

import pytest

WATCHED = ("Session", "Electrode", "IntervalList", "TaskEpoch", "Subject")


@pytest.fixture
def unregistered_raw(mini_path, raw_dir):
    """A raw file under a name Spyglass has never seen.

    A symlink rather than a copy: the point is a name with no `Nwbfile` row,
    not a second body of data, and `NWBHDF5IO` reads through it. Removed
    afterwards so the substring matching in `_resolve_raw_path` cannot later
    find two candidates for one name.
    """
    target = Path(raw_dir) / "checkonly20230622.nwb"

    if not target.exists():
        target.symlink_to(mini_path)

    yield target

    if target.is_symlink():
        target.unlink()


def test_insert_sessions_plans_a_file_it_has_never_seen(
    common, unregistered_raw
):
    """The whole point: a report without a copy, a row, or an ingestion."""
    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.data_import import insert_sessions
    from spyglass.utils.ingestion_plan import IngestionPlan

    copy_name = "checkonly20230622_.nwb"
    assert not (
        Nwbfile & {"nwb_file_name": copy_name}
    ), "Fixture file must not be registered"

    counts_before = {name: len(getattr(common, name)()) for name in WATCHED}
    nwbfiles_before = len(Nwbfile())

    plans = insert_sessions(unregistered_raw.name, dry_run=True)

    assert len(plans) == 1
    plan = plans[0]
    assert isinstance(plan, IngestionPlan)
    assert (
        plan.nwb_file_name == copy_name
    ), "Entries are keyed by the copy name a real ingestion would use"

    assert len(Nwbfile()) == nwbfiles_before, "It registered the file"
    assert {
        name: len(getattr(common, name)()) for name in WATCHED
    } == counts_before, "It wrote to a data table"
    assert not Path(
        Nwbfile.get_abs_path(copy_name, new_file=True)
    ).exists(), "It created the _.nwb copy"


def test_insert_sessions_resolves_the_nwbfile_foreign_key(
    common, unregistered_raw
):
    """The obstacle that makes this more than opening a file.

    Every session-keyed table refers to the `Nwbfile` row, and ingestion
    creates it before reading anything. Unless the plan accounts for the row
    it intends to create, the report is one `missing_parent` for `Session`
    plus a blocked list of everything beneath it -- a complaint about the
    absence of a row the caller was about to add.
    """
    from spyglass.data_import import insert_sessions

    plan = insert_sessions(unregistered_raw.name, dry_run=True)[0]

    missing = [
        problem
        for problem in plan.problems
        if problem.code == "missing_parent" and "nwbfile" in str(problem.table)
    ]
    assert not missing, f"Nwbfile FK reported as missing: {missing}"

    assert any(
        problem.code == "file_will_be_registered" for problem in plan.problems
    ), "The assumption should be on the record, not silent"

    blocked = [p.table_name for p in plan.table_plans if p.status == "blocked"]
    assert not blocked, f"Nothing should be blocked on a good file: {blocked}"


def test_planning_a_raw_file_resolves_every_cross_reference(
    common, unregistered_raw
):
    """A cross-reference resolves against the plan, not the database.

    Five tables need a value from a table this same ingestion fills --
    SensorData and DIOEvents want `Raw`'s interval, TaskEpoch and VideoFile
    want `IntervalList`/`TaskEpoch`, ImportedLFP needs an
    `LFPElectrodeGroup`. Querying for those rows finds nothing on a file that
    is not ingested yet, so each must read them from the plan instead.

    Asserted against the file's own content rather than against a plan of its
    registered copy. Comparing the two found this bug and is a bad way to pin
    it: the copy has database history, so its plan legitimately varies with
    whatever else the suite did to it, and the test then fails for reasons that
    have nothing to do with the invariant.
    """
    from spyglass.data_import import insert_sessions

    plan = insert_sessions(unregistered_raw.name, dry_run=True)[0]

    # Each table's own rows, not the parents it emits alongside them.
    own = {
        tp.table_name: len(rows)
        for tp in plan.table_plans
        for target, rows in tp.entries
        if getattr(target, "full_table_name", None) == tp.table_name
    }

    for fragment, least in (
        ("_sensor_data", 1),
        ("_d_i_o_events", 3),
        ("_task_epoch", 2),
        ("_video_file", 2),
        ("_imported_l_f_p", 1),
    ):
        found = next((n for name, n in own.items() if fragment in name), 0)
        assert found >= least, (
            f"{fragment} planned {found} rows, expected at least {least} -- "
            + "a cross-reference resolved against the database instead of "
            + "the plan"
        )

    # The sharpest of the five: absent `Raw`, DIOEvents used to fall back to an
    # interval of its own, so it planned a different `interval_list_name` and
    # an extra IntervalList row -- a plan disagreeing with its own insert.
    dio_rows = [
        row
        for tp in plan.table_plans
        for target, rows in tp.entries
        if "_d_i_o_events" in getattr(target, "full_table_name", "")
        for row in rows
    ]
    assert dio_rows, "Premise: the file has DIO events"
    assert {row["interval_list_name"] for row in dio_rows} == {
        "raw data valid times"
    }, "DIOEvents must take Raw's planned interval, not its own fallback"


def test_planning_a_raw_file_leaves_no_table_unchecked(
    common, unregistered_raw
):
    """No table is skipped, and nothing blocks, on a good file.

    The point of dropping the partial check: every table is planned, so the
    report speaks for the whole file rather than for the part of it that
    happened to be answerable.
    """
    from spyglass.data_import import insert_sessions

    plan = insert_sessions(unregistered_raw.name, dry_run=True)[0]

    assert not plan.blocking, f"Nothing should block: {list(plan.blocking)}"
    # `skipped` is not a shortfall: it means the file holds no source object
    # for that table. `failed` and `blocked` are what a partial check produced.
    assert not [
        tp.table_name
        for tp in plan.table_plans
        if tp.status in ("failed", "blocked")
    ], "No table should fail or be blocked on a good file"


def test_insert_sessions_stages_its_plan(common, unregistered_raw):
    """A checked file is on the record, so a later attempt can reuse it."""
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import import insert_sessions

    key = {"nwb_file_name": "checkonly20230622_.nwb"}
    before = (
        (IngestionPlanLog & key).fetch1("attempt")
        if (IngestionPlanLog & key)
        else 0
    )

    insert_sessions(unregistered_raw.name, dry_run=True)

    staged = IngestionPlanLog & key
    assert staged, "The plan should be staged"
    assert staged.fetch1("attempt") == before + 1, "This run did not stage"


def test_insert_sessions_accepts_a_list(common, unregistered_raw):
    """One plan per file, in the order given."""
    from spyglass.data_import import insert_sessions

    plans = insert_sessions(
        [unregistered_raw.name, unregistered_raw.name], dry_run=True
    )

    assert len(plans) == 2, "One result per name, duplicates included"
    assert all(p.nwb_file_name == "checkonly20230622_.nwb" for p in plans)


def test_insert_sessions_reports_a_missing_file_rather_than_planning(common):
    """A name matching nothing is the caller's error, and is raised."""
    from spyglass.data_import import insert_sessions

    with pytest.raises(FileNotFoundError):
        insert_sessions("_no_such_raw_file_at_all.nwb", dry_run=True)
