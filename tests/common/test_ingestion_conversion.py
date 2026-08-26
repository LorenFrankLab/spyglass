"""Acceptance tests for moving ingestion tables onto SpyglassIngestion.

Every table reached by `populate_all_common` should end up on one declarative
code path. That migration must not change *what* gets ingested, only how the
entries are produced -- so these tests pin the rows a clean insert of the
`minirec20230622` test file produces.

Two shapes of assertion, because the tables come in two shapes:

- Tables keyed by `nwb_file_name` are asserted by exact count, restricted to
  the mini file.
- Tables without it (Subject, Lab, devices, ...) are shared across every file
  the suite ingests, so a count is environment-dependent. Those are asserted
  by identity: the entries the mini file contributes must be present, with
  the values it specifies.

Log output is deliberately not asserted anywhere: the migration may change
ingestion's verbosity, only not its consequences. When a conversion changes
one of the values below, that *is* a behavior change -- update it in the same
commit and say why in the PR description.
"""

from types import SimpleNamespace

import pytest

# Rows each file-scoped table holds after a clean insert of the mini file.
EXPECTED_COUNTS = {
    "DIOEvents": 3,
    "Electrode": 128,
    "ElectrodeGroup": 32,
    "PositionSource": 2,
    "Raw": 1,
    "RawPosition": 2,
    "SampleCount": 1,
    "SensorData": 1,
    "Session": 1,
    "TaskEpoch": 2,
    "VideoFile": 2,
}

# Entries the mini file contributes to tables that accumulate rows from
# elsewhere. Asserted by presence, not by count: other files add their own
# subjects and devices, and other tests add their own intervals for this file
# (LFP, position, and caution-check entries all land in IntervalList).
EXPECTED_ENTRIES = {
    "CameraDevice": ("camera_name", {"test camera 1", "test camera 2"}),
    "DataAcquisitionDevice": (
        "data_acquisition_device_name",
        {"dataacq_device0"},
    ),
    "DataAcquisitionDeviceAmplifier": (
        "data_acquisition_device_amplifier",
        {"Intan"},
    ),
    "DataAcquisitionDeviceSystem": (
        "data_acquisition_device_system",
        {"Main Control Unit"},
    ),
    "Institution": ("institution_name", {"UCSF"}),
    "IntervalList": (
        "interval_list_name",
        {
            "01_s1",
            "02_s2",
            "imported lfp 0 valid times",
            "pos 0 valid times",
            "pos 1 valid times",
            "raw data valid times",
        },
    ),
    "Lab": ("lab_name", {"Loren Frank Lab"}),
    "LabMember": (
        "lab_member_name",
        {"firstname lastname", "firstname2 lastname2"},
    ),
    "LabTeam": ("team_name", {"firstname lastname", "firstname2 lastname2"}),
    "Probe": ("probe_id", {"tetrode_12.5"}),
    "ProbeType": ("probe_type", {"tetrode_12.5"}),
    "Subject": ("subject_id", {"54321"}),
    "Task": ("task_name", {"Sleep", "wtrack"}),
}

# Tables the mini file legitimately does not populate. A conversion that
# starts inserting rows here is as much a regression as one that stops.
EXPECTED_EMPTY = [
    "OpticalFiberDevice",
    "OpticalFiberImplant",
    "OptogeneticProtocol",
    "RawCompassDirection",
    "StateScriptFile",
    "Virus",
    "VirusInjection",
]

# Part tables, checked separately: their masters must exist first.
EXPECTED_PART_COUNTS = {
    ("PositionSource", "SpatialSeries"): 4,
    ("Probe", "Electrode"): 4,
    ("Probe", "Shank"): 1,
    ("RawPosition", "PosObject"): 4,
    ("Session", "DataAcquisitionDevice"): 1,
    ("Session", "Experimenter"): 2,
}


@pytest.mark.parametrize("table_name", sorted(EXPECTED_COUNTS))
def test_ingested_row_counts(mini_insert, common, mini_restr, table_name):
    """Each file-scoped table holds the expected number of rows."""
    table = getattr(common, table_name)()
    expected = EXPECTED_COUNTS[table_name]
    found = len(table & mini_restr)

    assert found == expected, (
        f"{table_name} has {found} rows for the mini file, expected "
        + f"{expected}. If a conversion changed this deliberately, update "
        + "EXPECTED_COUNTS."
    )


@pytest.mark.parametrize("table_name", sorted(EXPECTED_ENTRIES))
def test_shared_table_entries(mini_insert, common, table_name):
    """Shared tables carry the entries the mini file contributes."""
    attr, expected = EXPECTED_ENTRIES[table_name]
    found = set(getattr(common, table_name)().fetch(attr))

    assert expected <= found, (
        f"{table_name} is missing {expected - found}. Other files in the "
        + "suite may add entries here, so only absence is a failure."
    )


@pytest.mark.parametrize("table_name", EXPECTED_EMPTY)
def test_tables_not_populated(mini_insert, common, mini_restr, table_name):
    """Tables with no source data in the mini file stay empty."""
    table = getattr(common, table_name)()

    if "nwb_file_name" not in table.heading:
        pytest.skip(
            f"{table_name} is shared across files; emptiness cannot be "
            + "attributed to the mini file"
        )

    found = len(table & mini_restr)

    assert not found, (
        f"{table_name} unexpectedly has {found} rows for the mini file, "
        + "which has no source object for it."
    )


@pytest.mark.parametrize("master,part", sorted(EXPECTED_PART_COUNTS))
def test_part_table_counts(mini_insert, common, mini_restr, master, part):
    """Part tables are populated alongside their masters."""
    table = getattr(getattr(common, master), part)()
    expected = EXPECTED_PART_COUNTS[(master, part)]
    found = (
        len(table & mini_restr)
        if "nwb_file_name" in table.heading
        else len(table)
    )

    assert (
        found == expected
    ), f"{master}.{part} has {found} rows, expected {expected}"


def test_reingest_logs_no_new_failures(mini_insert, common, mini_copy_name):
    """Ingesting the file again surfaces no failure beyond duplicates.

    A conversion that starts raising something other than DuplicateError on
    an already-ingested file has changed behavior. Clears the log first:
    ingestion state persists across runs under `--no-teardown`.
    """
    from spyglass.common.common_usage import InsertError
    from spyglass.common.populate_all_common import populate_all_common

    InsertError().delete(safemode=False)
    try:
        populate_all_common(mini_copy_name, raise_err=False)
        unexpected = [
            f"{err['table']}: {err['error_type']}: {err['error_message']}"
            for err in InsertError().fetch(as_dict=True)
            if err["error_type"] != "DuplicateError"
        ]
    finally:
        InsertError().delete(safemode=False)

    assert not unexpected, f"Re-ingestion logged new failures: {unexpected}"


def test_electrode_group_links_probe(mini_insert, common, mini_restr):
    """Electrodes resolve to their group and probe, not just to a count.

    Guards the conversions most likely to break silently: a mapping that
    produces the right number of rows with the wrong foreign keys.
    """
    electrodes = (common.Electrode & mini_restr).fetch(
        "electrode_group_name", "probe_id", as_dict=True
    )
    assert electrodes, "No electrodes ingested"

    groups = set(
        (common.ElectrodeGroup & mini_restr).fetch("electrode_group_name")
    )
    probes = set(common.Probe().fetch("probe_id"))

    for row in electrodes:
        assert (
            row["electrode_group_name"] in groups
        ), f"Electrode references unknown group {row['electrode_group_name']}"
        assert (
            row["probe_id"] in probes
        ), f"Electrode references unknown probe {row['probe_id']}"


def test_object_ids_reference_real_nwb_objects(mini_insert, common, mini_restr):
    """Stored NWB object ids resolve to objects of the expected kind.

    Row counts cannot catch a mapping that picks up the wrong object -- an
    enclosing container, say, rather than the series inside it. This checks
    the ids actually point at what the column claims.
    """
    import pynwb

    from spyglass.common.common_nwbfile import Nwbfile

    nwb = (Nwbfile & mini_restr).fetch_nwb()[0]

    expected_types = {
        ("Raw", "raw_object_id"): pynwb.ecephys.ElectricalSeries,
        ("SampleCount", "sample_count_object_id"): pynwb.base.TimeSeries,
        ("SensorData", "sensor_data_object_id"): pynwb.base.TimeSeries,
    }

    for (table_name, attr), expected_type in expected_types.items():
        for object_id in (getattr(common, table_name)() & mini_restr).fetch(
            attr
        ):
            assert (
                object_id in nwb.objects
            ), f"{table_name}.{attr} {object_id} is not in the NWB file"
            found = nwb.objects[object_id]
            assert isinstance(found, expected_type), (
                f"{table_name}.{attr} points at {type(found).__name__}, "
                + f"expected {expected_type.__name__}"
            )


def test_statescript_expands_one_entry_per_epoch(
    mini_insert, common, mini_copy_name
):
    """A state script naming several epochs yields an entry for each.

    The mini file's `associated_files` module is empty, so ingestion of it
    exercises only the absent-source path. This builds the object the file
    lacks and checks the mapping directly: epochs are parsed from the
    comma-separated string, restricted to epochs that have a TaskEpoch row,
    and every entry points at the associated file's own object id.
    """
    from ndx_franklab_novela import AssociatedFiles

    known = set(
        (common.TaskEpoch & {"nwb_file_name": mini_copy_name}).fetch("epoch")
    )
    assert known, "Mini file has no TaskEpoch rows to attach a script to"

    unknown_epoch = max(known) + 97
    associated = AssociatedFiles(
        name="statescript_test",
        description="Statescript for several epochs",
        content="callback()\n",
        task_epochs=",".join(str(e) for e in sorted(known) + [unknown_epoch]),
    )

    generated = common.StateScriptFile().generate_entries_from_nwb_object(
        associated, {"nwb_file_name": mini_copy_name}
    )
    entries = next(iter(generated.values()))

    assert {entry["epoch"] for entry in entries} == known, (
        "Expected one entry per known epoch; an epoch with no TaskEpoch row "
        + "must not produce one"
    )
    assert all(
        entry["file_object_id"] == associated.object_id for entry in entries
    ), "Entries should carry the associated file's object id"


def test_statescript_ignores_non_script_files(
    mini_insert, common, mini_copy_name
):
    """Associated files that are not state scripts are not selected.

    The description filter lives in `get_nwb_objects` via the mixin's
    `_source_nwb_object_description`, so selection is what to assert on.
    """
    from ndx_franklab_novela import AssociatedFiles

    script = AssociatedFiles(
        name="statescript_test",
        description="Statescript for epoch 1",
        content="callback()\n",
        task_epochs="1",
    )
    notes = AssociatedFiles(
        name="notes_test",
        description="Experimenter notes, not a script of any kind",
        content="the animal was sleepy\n",
        task_epochs="1",
    )
    nwb_file = SimpleNamespace(
        objects={script.object_id: script, notes.object_id: notes}
    )

    selected = common.StateScriptFile().get_nwb_objects(
        nwb_file, mini_copy_name
    )

    assert selected == [
        script
    ], "Only the associated file describing a state script should be selected"


def test_task_epoch_config_cameras_map_by_id(common):
    """Config-declared cameras resolve by id, matching NWB-declared ones.

    Tasks reference a camera by id, so the lookup must be keyed by id. The
    previous implementation zipped the config's scalar `camera_name` and
    `camera_id`, which both inverted the mapping and raised TypeError on the
    int -- a config-declared camera never resolved.
    """

    class _NoDevices:
        devices = {}

    table = common.TaskEpoch()
    table._file_config = {
        "CameraDevice": [
            {"camera_id": 7, "camera_name": "cam seven"},
            {"camera_id": 8, "camera_name": "cam eight"},
        ]
    }

    mapping = table._camera_name_map(_NoDevices())  # NWB file with no cameras

    assert mapping == {
        7: "cam seven",
        8: "cam eight",
    }, "Config cameras should map camera_id -> camera_name"

    assert table._get_valid_camera_names([7], mapping) == [
        {"camera_name": "cam seven"}
    ], "A task referencing a config-declared camera should resolve it"


def test_task_epoch_camera_less_epoch_is_kept(common, mini_copy_name):
    """An epoch naming no camera stores an empty list, not nothing.

    `camera_names` is required, so omitting it drops the whole TaskEpoch
    entry -- and with it the VideoFile, StateScriptFile and
    OptogeneticProtocol rows that reference the epoch.
    """
    table = common.TaskEpoch()

    assert (
        table._get_valid_camera_names([], {1: "cam one"}) == []
    ), "A camera-less epoch should resolve to an empty list"

    key = dict(
        nwb_file_name=mini_copy_name,
        epoch=1,
        task_name="Sleep",
        interval_list_name="01_s1",
        camera_names=[],
    )

    assert table._adjust_keys_for_entry([key]) == [
        key
    ], "A camera-less epoch entry should survive key adjustment"


def test_task_epoch_unresolved_camera_raises(common):
    """A camera id with no device is a data error, not something to skip."""
    with pytest.raises(ValueError):
        common.TaskEpoch()._get_valid_camera_names([9], {1: "cam one"})


def test_validate_duplicates_collapses_repeated_keys(common):
    """Two objects naming the same novel parent yield one insert.

    Entries are checked against the database, not against each other, so a
    repeated novel key used to reach `insert(skip_duplicates=False)` twice
    and abort the whole file with a DuplicateError.
    """
    task_tbl = common.Task()
    task = dict(task_name="_dedup test task", task_description="repeated")

    validated = common.TaskEpoch().validate_duplicates(
        {task_tbl: [dict(task), dict(task)]}
    )

    assert validated[task_tbl] == [
        task
    ], "Repeated identical entries should collapse to one"


def test_validate_duplicates_conflicting_keys_raise(common):
    """Same key, different values, neither stored: nothing to defer to."""
    import datajoint as dj

    with pytest.raises(dj.errors.DuplicateError):
        common.TaskEpoch().validate_duplicates(
            {
                common.Task(): [
                    dict(task_name="_conflict test", task_description="a"),
                    dict(task_name="_conflict test", task_description="b"),
                ]
            }
        )


def test_video_partial_import_counts_source_series(common, monkeypatch):
    """A multi-epoch video must not mask a series that placed nowhere.

    The report used to compare source-series count with generated-row count.
    One video spanning two epochs yields two rows, so a second video that
    placed nowhere left the two equal and its failure went unreported.
    """
    from collections import defaultdict

    table = common.VideoFile()
    table._epoch_cache = {"fake_.nwb": {1: None, 2: None}}
    table._failed_videos = defaultdict(list)
    table._video_count = 0
    table._placed_videos = 0

    def fake_validate(video_obj, valid_times, key):
        if video_obj.name == "spans two epochs":
            return [dict(key, video_file_num=0)], None, 1.0
        return [], "no timestamp overlap with epoch", 0.0

    monkeypatch.setattr(table, "_validate_video_timestamps", fake_validate)

    base_key = {"nwb_file_name": "fake_.nwb"}
    placed = table.generate_entries_from_nwb_object(
        SimpleNamespace(name="spans two epochs"), dict(base_key)
    )
    table.generate_entries_from_nwb_object(
        SimpleNamespace(name="placed nowhere"), dict(base_key)
    )

    assert len(placed[table]) == 2, "The first video should land in two epochs"
    assert table._video_count == 2, "Both source series should be counted"
    assert table._placed_videos == 1, "Only one source series was placed"
    assert (
        table._placed_videos < table._video_count
    ), "A failed series must trigger the partial-import report"

    # Only the video that landed nowhere is a failure. A placed video fails
    # the overlap check for every epoch it does not belong to, and those are
    # not diagnostics anyone should see.
    # The unplaced video failed in both epochs, so both are recorded.
    reported = {
        item["name"] for item in table._failed_videos["timestamp_mismatch"]
    }
    assert reported == {
        "placed nowhere"
    }, "A placed video must not report its non-owning epochs as mismatches"
