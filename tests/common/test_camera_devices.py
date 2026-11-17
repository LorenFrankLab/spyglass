"""Test camera devices other than the ndx-franklab-novela CameraDevice.

Core NWB has no camera type, so a file can record a video with a plain
``Device`` next to a typed camera. The first file has a plain-Device camera, a
novela ``CameraDevice`` and a plain Device that records no video, and its task
table names both cameras by ID. It verifies that:

- a plain Device that an ImageSeries records with is ingested in CameraDevice;
- a plain Device that records no video is not;
- the task's camera IDs resolve to both cameras in TaskEpoch;
- both videos are ingested in VideoFile.

The second file (ndx-pose >= 0.4, pynwb >= 4) has an ndx-pose
``CalibratedCamera`` without a ``DeviceModel``, recording a rate-based external
video. It verifies that the camera and its video are ingested.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from hdmf.common import DynamicTable, VectorData, VectorIndex
from ndx_franklab_novela import CameraDevice as NovelaCameraDevice
from pynwb import NWBHDF5IO, NWBFile
from pynwb.device import Device, DeviceModel
from pynwb.file import Subject
from pynwb.image import ImageSeries

N_FRAMES = 50
RATE = 50.0
PLAIN_CAMERA = "DeviceTestCamera1"  # plain Device, camera id 1
NOVELA_CAMERA = "DeviceTestCamera2"  # novela "camera_device 2"
NOT_A_CAMERA = "DeviceTestStimulator"
CALIBRATED_CAMERA = "DeviceTestCamera3"  # ndx-pose CalibratedCamera


def _task_table(camera_ids):
    """A tasks module table with one task in epoch 1 naming the cameras."""
    camera_id = VectorData(
        name="camera_id", description="Camera IDs.", data=list(camera_ids)
    )
    task_epochs = VectorData(
        name="task_epochs", description="Task epochs.", data=[1]
    )
    return DynamicTable(
        name="task_table",
        description="task of the session",
        columns=[
            VectorData(
                name="task_name",
                description="Name of the task.",
                data=["camera_device_test_task"],
            ),
            VectorData(
                name="task_description",
                description="Description of the task.",
                data=["task recorded by the test cameras"],
            ),
            camera_id,
            VectorIndex(
                name="camera_id_index", data=[len(camera_ids)], target=camera_id
            ),
            task_epochs,
            VectorIndex(name="task_epochs_index", data=[1], target=task_epochs),
        ],
    )


def _new_nwbfile(identifier):
    nwbfile = NWBFile(
        session_description="test camera devices",
        identifier=identifier,
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    nwbfile.subject = Subject(
        subject_id=f"{identifier}_subject", species="Rattus norvegicus"
    )
    nwbfile.add_epoch(
        start_time=0.0, stop_time=(N_FRAMES - 1) / RATE, tags=["01_session"]
    )
    return nwbfile


def _insert(nwbfile, raw_file_name, cleanup_context):
    """Write the file, insert it, and return its key and cleanup callback."""
    from spyglass.common import Nwbfile
    from spyglass.data_import import insert_sessions
    from spyglass.settings import raw_dir
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    nwb_path = Path(raw_dir) / raw_file_name
    key = dict(nwb_file_name=get_nwb_copy_filename(raw_file_name))
    if (Nwbfile & key) or nwb_path.exists():
        (Nwbfile & key).delete(safemode=False)
        nwb_path.unlink(missing_ok=True)
    with NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)

    insert_sessions(raw_file_name, raise_err=True)

    def cleanup():
        with cleanup_context:
            (Nwbfile & key).delete(safemode=False)
            nwb_path.unlink(missing_ok=True)

    return key, cleanup


@pytest.fixture(scope="module")
def plain_camera_nwb(verbose_context):
    nwbfile = _new_nwbfile("test_plain_device_cameras")
    camera_model = DeviceModel(name="test camera model", manufacturer="maker")
    nwbfile.add_device_model(camera_model)
    cameras = [
        nwbfile.add_device(
            Device(name=PLAIN_CAMERA, description="camera without a type")
        ),
        nwbfile.add_device(
            NovelaCameraDevice(
                name="camera_device 2",
                camera_name=NOVELA_CAMERA,
                meters_per_pixel=0.001,
                model=camera_model,
                lens="lens",
            )
        ),
    ]
    nwbfile.add_device(Device(name=NOT_A_CAMERA, description="no video"))
    timestamps = np.arange(N_FRAMES) / RATE
    for camera in cameras:
        nwbfile.add_acquisition(
            ImageSeries(
                name=f"Video{camera.name.replace(' ', '')}",
                description=f"video of {camera.name}",
                unit="n.a.",
                format="external",
                external_file=[f"{camera.name}.mp4"],
                starting_frame=[0],
                timestamps=timestamps,
                device=camera,
            )
        )
    nwbfile.create_processing_module("tasks", "tasks").add(_task_table([1, 2]))

    key, cleanup = _insert(nwbfile, "test_camera_devices.nwb", verbose_context)
    yield key
    cleanup()


def test_plain_device_camera_ingested(common, plain_camera_nwb):
    cameras = (
        common.CameraDevice
        & [{"camera_name": n} for n in (PLAIN_CAMERA, NOVELA_CAMERA)]
    ).fetch("camera_name", "camera_id", as_dict=True, order_by="camera_name")
    assert [(c["camera_name"], c["camera_id"]) for c in cameras] == [
        (PLAIN_CAMERA, 1),
        (NOVELA_CAMERA, 2),
    ]
    assert not (common.CameraDevice & {"camera_name": NOT_A_CAMERA})


def test_task_epoch_resolves_plain_device_camera(common, plain_camera_nwb):
    epoch = (common.TaskEpoch & plain_camera_nwb).fetch1()
    assert [c["camera_name"] for c in epoch["camera_names"]] == [
        PLAIN_CAMERA,
        NOVELA_CAMERA,
    ]


def test_plain_device_video_ingested(common, plain_camera_nwb):
    videos = (common.VideoFile & plain_camera_nwb).fetch(
        "camera_name", "video_file_num", as_dict=True, order_by="video_file_num"
    )
    assert [(v["camera_name"], v["video_file_num"]) for v in videos] == [
        (PLAIN_CAMERA, 1),
        (NOVELA_CAMERA, 2),
    ]


@pytest.fixture(scope="module")
def calibrated_camera_nwb(verbose_context):
    # CalibratedCamera and ImageSeries.num_samples need ndx-pose >= 0.4 and
    # pynwb >= 4.
    ndx_pose = pytest.importorskip("ndx_pose")
    if not hasattr(ndx_pose, "CalibratedCamera"):
        pytest.skip("ndx-pose < 0.4 has no CalibratedCamera")

    nwbfile = _new_nwbfile("test_calibrated_camera")
    camera = nwbfile.add_device(
        ndx_pose.CalibratedCamera(
            name=CALIBRATED_CAMERA,
            intrinsic_matrix=np.eye(3),
            rotation_matrix=np.eye(3),
            translation_vector=np.zeros(3),
            distortion_coefficients=np.zeros(4),
        )
    )
    nwbfile.add_acquisition(
        ImageSeries(
            name="VideoCalibrated",
            description="rate-based external video",
            unit="n.a.",
            format="external",
            external_file=[f"{CALIBRATED_CAMERA}.mp4"],
            starting_frame=[0],
            rate=RATE,  # no timestamps
            num_samples=N_FRAMES,
            device=camera,
        )
    )
    nwbfile.create_processing_module("tasks", "tasks").add(_task_table([3]))

    key, cleanup = _insert(
        nwbfile, "test_calibrated_camera.nwb", verbose_context
    )
    yield key
    cleanup()


def test_calibrated_camera_ingested(common, calibrated_camera_nwb):
    camera = (common.CameraDevice & {"camera_name": CALIBRATED_CAMERA}).fetch1()
    assert camera["camera_id"] == 3
    assert camera["model"] == ""  # no DeviceModel

    epoch = (common.TaskEpoch & calibrated_camera_nwb).fetch1()
    assert [c["camera_name"] for c in epoch["camera_names"]] == [
        CALIBRATED_CAMERA
    ]
    videos = (common.VideoFile & calibrated_camera_nwb).fetch("camera_name")
    assert list(videos) == [CALIBRATED_CAMERA]
