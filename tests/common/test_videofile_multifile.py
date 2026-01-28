"""Test VideoFile import with multi-file ImageSeries.

This module verifies that:

- VideoFile entries are correctly created for each external file in a
    multi-file ImageSeries.
- Each entry is associated with the correct epoch and references the same
    ImageSeries object.
- Camera device and task information are properly linked.

Related to issue #1445.
"""

import numpy as np
import pytest
from hdmf.common.table import DynamicTable, VectorData
from ndx_franklab_novela import CameraDevice
from pynwb import NWBHDF5IO
from pynwb.device import DeviceModel
from pynwb.image import ImageSeries
from pynwb.testing.mock.file import mock_NWBFile, mock_Subject


@pytest.fixture(scope="module")
def multifile_video_nwb(raw_dir, common, data_import, teardown):
    """Create an NWB file with ImageSeries containing multiple external files.

    This fixture creates a mock NWB file that simulates the scenario described
    in issue #1445: a single ImageSeries with 3 external video files, each
    corresponding to a different epoch.
    """
    nwb_filename = "mock_multifile_video.nwb"
    nwb_path = raw_dir / nwb_filename

    # Create mock NWB file
    nwbfile = mock_NWBFile(
        identifier="multifile_video_test",
        session_description="Test file for multi-file ImageSeries import",
    )
    nwbfile.subject = mock_Subject()

    # Add 3 epochs, each 10 seconds long
    nwbfile.add_epoch(start_time=0.0, stop_time=10.0, tags=["01"])
    nwbfile.add_epoch(start_time=10.0, stop_time=20.0, tags=["02"])
    nwbfile.add_epoch(start_time=20.0, stop_time=30.0, tags=["03"])

    # Create camera device
    # Use minimal parameters to avoid API compatibility issues
    camera_model = DeviceModel(
        name="Test Camera Model", manufacturer="Test Camera Manufacturer"
    )
    camera_device = CameraDevice(
        name="camera_device 1",
        meters_per_pixel=0.001,
        lens="test_lens",
        model=camera_model,
        camera_name="test_camera",
    )
    nwbfile.add_device_model(camera_model)
    nwbfile.add_device(camera_device)

    # Create behavior processing module (required for some Spyglass imports)
    nwbfile.create_processing_module(
        name="behavior", description="Behavioral data including video"
    )

    # Add task information for each epoch
    tasks_module = nwbfile.create_processing_module(
        name="tasks", description="Task information for each epoch"
    )

    for epoch_num in range(1, 4):
        task_name = VectorData(
            name="task_name",
            description="Name of the task",
            data=[f"task{epoch_num}"],
        )
        task_description = VectorData(
            name="task_description",
            description="Description of the task",
            data=[f"Test task for epoch {epoch_num}"],
        )
        camera_id = VectorData(
            name="camera_id",
            description="Camera ID",
            data=[[1]],
        )
        task_epochs = VectorData(
            name="task_epochs",
            description="Task epochs",
            data=[[epoch_num]],
        )
        task_environment = VectorData(
            name="task_environment",
            description="Environment description",
            data=["test_env"],
        )

        task_table = DynamicTable(
            name=f"task_{epoch_num}",
            description=f"Task for epoch {epoch_num}",
            columns=[
                task_name,
                task_description,
                camera_id,
                task_epochs,
                task_environment,
            ],
        )
        tasks_module.add(task_table)

    # Create ImageSeries with 3 external files
    video_files = [
        "video_epoch1.mp4",
        "video_epoch2.mp4",
        "video_epoch3.mp4",
    ]
    timestamps = np.linspace(0, 30, 900, endpoint=False)
    starting_frame = [0, 300, 600]

    image_series = ImageSeries(
        name="video_files",
        description="Video recordings across multiple epochs",
        unit="n.a.",
        external_file=video_files,
        format="external",
        timestamps=timestamps,
        starting_frame=starting_frame,
        device=camera_device,
    )
    nwbfile.add_acquisition(image_series)

    # Write NWB file
    if nwb_path.exists():
        nwb_path.unlink()

    with NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)

    # Import into Spyglass
    data_import.insert_sessions(nwb_filename, raise_err=True)

    # Get the copied filename
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    nwb_copy_name = get_nwb_copy_filename(nwb_filename)
    nwb_dict = {"nwb_file_name": nwb_copy_name}
    video_files = (common.VideoFile & nwb_dict).fetch(as_dict=True)

    yield video_files

    # Cleanup
    if teardown:
        (common.Nwbfile & nwb_dict).delete(safemode=False)
        if nwb_path.exists():
            nwb_path.unlink()


def test_multifile_video_import(common, multifile_video_nwb):
    """Test that VideoFile creates entries for each external file."""
    video_files = multifile_video_nwb

    # Assert that we have 3 VideoFile entries (one per external file)
    n_found = len(video_files)
    assert (
        n_found == 3
    ), f"Expected 3 VideoFile entries for multi-ImageSeries, found {n_found}"

    # Verify each entry corresponds to a different epoch
    epochs = sorted([vf["epoch"] for vf in video_files])
    expect = [1, 2, 3]
    assert (
        epochs == expect
    ), f"Expected VideoFile epochs {expect}, found epochs {epochs}"


def test_multifile_video_inserts(common, multifile_video_nwb):
    """Test that each VideoFile entry has correct object_id.

    When multiple VideoFile entries are created from a single ImageSeries,
    they should all reference the same NWB object (the ImageSeries), but
    be associated with different epochs.
    """
    video_files = multifile_video_nwb

    if len(video_files) == 0:
        pytest.skip("No VideoFile entries found - primary test is failing")

    # All video files should reference the same ImageSeries object
    object_ids = [vf["video_file_object_id"] for vf in video_files]
    assert (
        len(set(object_ids)) == 1
    ), "Expected all VideoFile entries to reference the same object ID"

    camera_names = [vf["camera_name"] for vf in video_files]
    assert all(name == "test_camera" for name in camera_names), (
        f"Expected all VideoFile entries to have camera_name='test_camera', "
        f"but found: {set(camera_names)}"
    )
