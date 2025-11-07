"""#1444: Video files fail to import without task information."""

from pathlib import Path

import numpy as np
import pynwb
import pytest
from ndx_franklab_novela import CameraDevice
from pynwb import NWBHDF5IO
from pynwb.image import ImageSeries
from pynwb.testing.mock.file import mock_NWBFile, mock_Subject


@pytest.fixture(scope="function")
def nwb_with_video_no_task(raw_dir, common):
    """Create an NWB file with ImageSeries but no task information.

    This simulates a real-world scenario where video data is recorded
    independently of task/behavioral paradigms.
    """
    nwbfile = mock_NWBFile(
        identifier="video_only_test",
        session_description="Session with video but no task",
        lab="Test Lab",
        institution="Test Institution",
        experimenter=["Test Experimenter"],
    )
    nwbfile.subject = mock_Subject()
    camera_device = CameraDevice(
        name="camera_device 0",
        meters_per_pixel=0.001,
        manufacturer="Test Camera Co",
        model="TestCam 3000",
        lens="50mm",
        camera_name="test_camera",
    )
    nwbfile.add_device(camera_device)

    # Create ImageSeries (video data) with timestamps
    timestamps = np.linspace(0, 10, 100)  # 10 seconds, 100 frames
    video_data = np.random.randint(0, 255, (100, 480, 640, 3), dtype=np.uint8)

    image_series = ImageSeries(
        name="test_video",
        data=video_data,
        unit="n.a.",
        format="raw",
        timestamps=timestamps,
        device=camera_device,
        description="Test video without associated task",
    )
    nwbfile.add_acquisition(image_series)

    # Write the NWB file
    from spyglass.settings import raw_dir as settings_raw_dir

    file_name = "video_no_task.nwb"
    nwbfile_path = Path(settings_raw_dir) / file_name
    with NWBHDF5IO(nwbfile_path, "w") as io:
        io.write(nwbfile)

    yield file_name

    # Cleanup
    (common.Nwbfile & {"nwb_file_name": file_name}).delete(safemode=False)
    if nwbfile_path.exists():
        nwbfile_path.unlink()


def test_video_import_without_task_silent_failure(
    nwb_with_video_no_task, common, caplog
):
    """Test that videos without task info now generate warnings"""
    from spyglass.common import populate_all_common

    common.Nwbfile.insert_from_relative_file_name(nwb_with_video_no_task)
    populate_all_common(nwb_with_video_no_task)
    nwb_dict = dict(nwb_file_name=nwb_with_video_no_task)

    # Check that a warning was logged
    warn_msgs = [
        record.message
        for record in caplog.records
        if record.levelname == "WARNING"
        and "TaskEpoch Import" in record.message
    ]

    assert any(warn_msgs), "Expected warning about #1444"

    # Check that TaskEpoch and VideoFile are not populated
    assert len(common.TaskEpoch & nwb_dict) == 0
    assert len(common.VideoFile & nwb_dict) == 0
