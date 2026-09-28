"""Tests for TaskEpoch epoch tag format handling (Issue #1443)."""

from pathlib import Path

import pytest
from ndx_franklab_novela import CameraDevice
from pynwb import NWBHDF5IO
from pynwb.core import DynamicTable
from pynwb.device import DeviceModel
from pynwb.testing.mock.file import mock_NWBFile, mock_Subject


def create_nwb_with_epoch_tags(identifier, epoch_tags):
    """Helper function to create NWB file with specified epoch tags.

    Parameters
    ----------
    identifier : str
        Unique identifier for the NWB file
    epoch_tags : list of str
        List of epoch tag strings (e.g., ["1", "02", "baseline"])

    Returns
    -------
    pynwb.NWBFile
        NWB file with epochs and task tables configured
    """
    nwbfile = mock_NWBFile(
        identifier=identifier,
        session_description=f"Test epoch tags: {epoch_tags}",
        lab="Test Lab",
        institution="Test Institution",
        experimenter=["Test Experimenter"],
    )

    # Add subject (required for session insertion)
    nwbfile.subject = mock_Subject()

    # Add behavior processing module (required)
    nwbfile.create_processing_module(
        name="behavior", description="Behavioral data"
    )

    # Add epochs with specified tags
    for i, tag in enumerate(epoch_tags):
        start_time = float(i + 1)
        stop_time = float(i + 2)
        nwbfile.add_epoch(
            start_time=start_time, stop_time=stop_time, tags=[tag]
        )

    # Add camera device (required for TaskEpoch)
    # Note: name must be "camera_device <number>" format for Spyglass
    camera_model = DeviceModel(
        name="test_model", manufacturer="test_manufacturer"
    )
    camera_device = CameraDevice(
        name="camera_device 1",
        meters_per_pixel=1.0,
        model=camera_model,
        lens="test_lens",
        camera_name="test_camera_name",
    )
    nwbfile.add_device_model(camera_model)
    nwbfile.add_device(camera_device)

    # Create tasks module with task tables
    tasks_module = nwbfile.create_processing_module(
        name="tasks", description="tasks module"
    )

    # Create task table for each epoch
    # Use identifier prefix to make task names unique per test
    task_prefix = identifier.replace("test_", "").replace("_", "")
    for i in range(len(epoch_tags)):
        task_table = DynamicTable(
            name=f"task_table_{i}", description=f"task table {i}"
        )
        task_table.add_column(name="task_name", description="Name of the task.")
        task_table.add_column(
            name="task_description", description="Description of the task."
        )
        task_table.add_column(name="camera_id", description="Camera ID.")
        task_table.add_column(name="task_epochs", description="Task epochs.")

        task_table.add_row(
            task_name=f"{task_prefix}_task{i+1}",
            task_description=f"{task_prefix} task{i+1} description",
            camera_id=[1],
            task_epochs=[i + 1],
        )
        tasks_module.add(task_table)

    return nwbfile


@pytest.fixture(scope="session")
def epoch_tag_nwb(raw_dir, common):
    """Create NWB file with various epoch tag formats.

    Creates epochs with tags:
    - "1" (single digit, non-zero-padded)
    - "02" (zero-padded, 2 digits)
    - "003" (zero-padded, 3 digits)
    """
    nwbfile = create_nwb_with_epoch_tags(
        identifier="test_epoch_tag_format",
        epoch_tags=["1", "02", "003"],
    )

    file_name = "test_epoch_tag_format.nwb"
    nwb_path = Path(raw_dir) / file_name
    if nwb_path.exists():
        nwb_path.unlink()

    with NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)

    yield file_name

    (common.Nwbfile & "nwb_file_name LIKE 'test_epoch_tag%'").delete(
        safemode=False
    )


def test_interval_list_accepts_all_tag_formats(
    epoch_tag_nwb, common, data_import
):
    """Test that IntervalList accepts all epoch tag formats."""

    # Insert the session
    _ = data_import.insert_sessions(
        epoch_tag_nwb, raise_err=True, rollback_on_fail=True
    )
    nwb_copy_file_name = epoch_tag_nwb.replace(".", "_.")

    # Fetch interval list names
    intervals = (
        common.IntervalList & {"nwb_file_name": nwb_copy_file_name}
    ).fetch("interval_list_name")

    # All epoch tags should be present as intervals
    assert "1" in intervals, "Single digit '1' should be in IntervalList"
    assert "02" in intervals, "Zero-padded '02' should be in IntervalList"
    assert "003" in intervals, "Zero-padded '003' should be in IntervalList"

    # Verify TaskEpoch also accepts these formats
    task_epochs = (
        common.TaskEpoch & {"nwb_file_name": nwb_copy_file_name}
    ).fetch("epoch")

    assert 1 in task_epochs, "TaskEpoch should accept epoch 1 with tag '1'"
    assert 2 in task_epochs, "TaskEpoch should accept epoch 2 with tag '02'"
    assert 3 in task_epochs, "TaskEpoch should accept epoch 3 with tag '003'"


def test_task_epoch_get_epoch_interval_name(common):
    """Test get_epoch_interval_name with single digit tags."""
    get_epoch = common.TaskEpoch.get_epoch_interval_name
    msg_template = "get_epoch_interval_name should find '{}' when epoch is {}"

    session_intervals = ["1", "02", "003", "baseline", "task_05", "trial_05"]

    for epoch, expected in [
        (1, "1"),  # match single digit
        (2, "02"),  # match zero-padded two digit
        (3, "003"),  # match zero-padded three digit
        (4, None),  # no match for missing epoch
        (5, None),  # multiple matches for epoch 5 (task_05, trial_05)
        ("baseline", "baseline"),  # match descriptive tag as-is
    ]:
        result = get_epoch(epoch, session_intervals)
        assert result == expected, msg_template.format(expected, epoch)


def test_franklab_task_epoch_tags(common):
    """Test task epoch tags in the franklab format are handled correctly."""
    epoch = 1
    session_intervals = ["01_s1", "02_r1", "03_s2", "04_r2"]
    interval_name = common.TaskEpoch.get_epoch_interval_name(
        epoch, session_intervals
    )
    assert (
        interval_name == "01_s1"
    ), "Failed to prioritize 2-digit zero-padded format"


# --- epoch matching against a live database ---------------------------------
# Interval names observed in a test database once LFP and spikesorting have
# run, rather than invented ones.

LIVE_DB_INTERVALS = [
    "01_s1",
    "01_s1_first9",
    "02_s2",
    "77c053e5-0b1d-4013-881c-ad4cb76309d1",
    "1ae82410-f27d-4a30-8c08-e1aa8845c24e",
    "lfp_test_01_s1_first9_valid times",
    "pos 0 valid times",
    "pos 1 valid times",
    "raw data valid times",
]


def test_epoch_does_not_claim_an_unrelated_uuid_interval(common):
    """An epoch must not match a UUID that happens to contain it.

    `"5" in "77c053e5-0b1d-..."` is true, so matching on a bare substring gave
    an epoch with no session interval a spikesorting interval instead, and
    `TaskEpoch` gained a phantom row -- which `VideoFile`, keyed on TaskEpoch,
    then turned into phantom video rows.
    """
    get_epoch = common.TaskEpoch.get_epoch_interval_name

    for epoch in (3, 5):
        assert (
            get_epoch(epoch, LIVE_DB_INTERVALS) is None
        ), f"Epoch {epoch} has no session interval and must match nothing"


def test_epoch_resolves_despite_derived_intervals(common):
    """Derived intervals must not make a real epoch ambiguous.

    Once a database holds `01_s1_first9` and
    `lfp_test_01_s1_first9_valid times`, a bare substring match found several
    names for epoch 1, the uniqueness check gave up, and the epoch silently
    produced no row -- invisible on a first ingestion, waiting for any later
    re-plan.
    """
    get_epoch = common.TaskEpoch.get_epoch_interval_name

    assert get_epoch(1, LIVE_DB_INTERVALS) == "01_s1"
    assert get_epoch(2, LIVE_DB_INTERVALS) == "02_s2"


def test_epoch_prefers_the_session_interval_over_its_descendants(common):
    """The session interval is the stem the derived ones extend."""
    get_epoch = common.TaskEpoch.get_epoch_interval_name

    assert (
        get_epoch(1, ["01_s1_first9", "01_s1", "01_s1 lfp band 100Hz"])
        == "01_s1"
    )


def test_epoch_declines_to_guess_between_distinct_intervals(common):
    """Two unrelated intervals sharing the epoch token is still ambiguous.

    Guessing is what produced the phantom rows, so a real ambiguity returns
    None rather than picking the shorter or the first.
    """
    get_epoch = common.TaskEpoch.get_epoch_interval_name

    assert get_epoch(1, ["01_s1", "01_r1"]) is None


def test_epoch_as_a_trailing_token_still_matches(common):
    """`epoch_01` and `task_05` name their epoch; they are just not leading.

    Requiring the epoch to *lead* the name would reject `epoch_01`, an
    ordinary convention. Trailing tokens are tier 3: consulted only when
    nothing leads with the epoch, so they never outrank an `01_s1`.
    """
    get_epoch = common.TaskEpoch.get_epoch_interval_name

    assert get_epoch(1, ["epoch_01"]) == "epoch_01"
    assert get_epoch(5, ["task_05"]) == "task_05"

    # Two of them is a real ambiguity, and is not guessed at.
    assert get_epoch(5, ["task_05", "trial_05"]) is None


def test_a_leading_epoch_outranks_a_trailing_one(common):
    """Tiering is what keeps tier 3 from re-creating the ambiguity.

    `pos 1 valid times` holds `1` as a token, so without tiers it would tie
    with `01_s1` for epoch 1, and the tie would yield no row at all.
    """
    get_epoch = common.TaskEpoch.get_epoch_interval_name

    assert (
        get_epoch(1, ["pos 1 valid times", "01_s1", "epoch_01"]) == "01_s1"
    ), "A name leading with the epoch wins outright"
