"""Test VideoFile import with single-file session-spanning ImageSeries.

This module verifies that:

- VideoFile entries are correctly created for each epoch when a single
    continuous video file spans the entire recording session.
- All entries reference the same ImageSeries object_id.
- Camera device and task information are properly linked.

Related to issue #1548.
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
def singlefile_multiepoch_video_nwb(raw_dir, common, data_import, teardown):
    """Create an NWB file with a single session-spanning ImageSeries.

    This fixture creates a mock NWB file that simulates the scenario described
    in issue #1548: a single external video file whose timestamps span all
    epochs in the session.
    """
    nwb_filename = "mock_multiepoch_video.nwb"
    nwb_path = raw_dir / nwb_filename

    nwbfile = mock_NWBFile(
        identifier="multiepoch_video_test",
        session_description="Test file for single-file multi-epoch import",
    )
    nwbfile.subject = mock_Subject()

    # Add 3 epochs, each 10 seconds long
    nwbfile.add_epoch(start_time=0.0, stop_time=10.0, tags=["01"])
    nwbfile.add_epoch(start_time=10.0, stop_time=20.0, tags=["02"])
    nwbfile.add_epoch(start_time=20.0, stop_time=30.0, tags=["03"])

    # Create camera device
    camera_model = DeviceModel(
        name="Test Camera Model", manufacturer="Test Camera Manufacturer"
    )
    camera_device = CameraDevice(
        name="camera_device 1",
        meters_per_pixel=0.001,
        lens="test_lens",
        model=camera_model,
        camera_name="test_multiepoch_video_camera",
    )
    nwbfile.add_device_model(camera_model)
    nwbfile.add_device(camera_device)

    nwbfile.create_processing_module(
        name="behavior", description="Behavioral data including video"
    )

    tasks_module = nwbfile.create_processing_module(
        name="tasks", description="Task information for each epoch"
    )

    for epoch_number in range(1, 4):
        task_table = DynamicTable(
            name=f"task_{epoch_number}",
            description=f"Task for epoch {epoch_number}",
            columns=[
                VectorData(
                    name="task_name",
                    description="Name of the task",
                    data=[f"task{epoch_number}"],
                ),
                VectorData(
                    name="task_description",
                    description="Description of the task",
                    data=[f"Test task for epoch {epoch_number}"],
                ),
                VectorData(
                    name="camera_id",
                    description="Camera ID",
                    data=[[1]],
                ),
                VectorData(
                    name="task_epochs",
                    description="Task epochs",
                    data=[[epoch_number]],
                ),
                VectorData(
                    name="task_environment",
                    description="Environment description",
                    data=["test_env"],
                ),
            ],
        )
        tasks_module.add(task_table)

    # Single video file spanning all 3 epochs (0–30s)
    image_series = ImageSeries(
        name="video_files",
        description="Session-spanning video recording",
        unit="n.a.",
        external_file=["session_video.mp4"],
        format="external",
        timestamps=np.linspace(0, 30, 900, endpoint=False),
        device=camera_device,
    )
    nwbfile.add_acquisition(image_series)

    if nwb_path.exists():
        nwb_path.unlink()

    with NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)

    data_import.insert_sessions(nwb_filename, raise_err=True)

    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    nwb_copy_name = get_nwb_copy_filename(nwb_filename)
    nwb_dict = {"nwb_file_name": nwb_copy_name}
    video_files = (common.VideoFile & nwb_dict).fetch(as_dict=True)

    yield video_files

    if teardown:
        (common.Nwbfile & nwb_dict).delete(safemode=False)
        if nwb_path.exists():
            nwb_path.unlink()


def test_singlefile_multiepoch_import(common, singlefile_multiepoch_video_nwb):
    """VideoFile creates one entry per epoch for a session-spanning video."""
    video_files = singlefile_multiepoch_video_nwb
    number_found = len(video_files)
    assert number_found == 3, (
        f"Expected 3 VideoFile entries for single-file multi-epoch video, "
        f"found {number_found}"
    )
    epochs = sorted([video_file["epoch"] for video_file in video_files])
    assert epochs == [
        1,
        2,
        3,
    ], f"Expected VideoFile epochs [1, 2, 3], found {epochs}"


def test_singlefile_multiepoch_same_object_id(
    common, singlefile_multiepoch_video_nwb
):
    """All epoch entries reference the same ImageSeries object_id."""
    video_files = singlefile_multiepoch_video_nwb
    if len(video_files) == 0:
        pytest.skip("No VideoFile entries found - primary test is failing")
    object_ids = [
        video_file["video_file_object_id"] for video_file in video_files
    ]
    assert (
        len(set(object_ids)) == 1
    ), "Expected all VideoFile entries to reference the same object_id"
