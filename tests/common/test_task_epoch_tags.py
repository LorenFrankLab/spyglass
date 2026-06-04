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
