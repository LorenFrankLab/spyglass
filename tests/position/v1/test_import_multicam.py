"""Test ingestion of standard ndx-pose 0.4 multi-camera 3D pose.

A minimal NWB file shaped like a DANNCE conversion: one ``CalibratedCamera``
with a ``DeviceModel`` and one plain ``Device`` camera without calibration, a
rate-based external video per camera, a task table naming both cameras, and a
``MultiCameraPoseEstimation`` holding 3D series and one series-less
``PoseEstimation`` child per camera (linking the camera and its video).

It verifies that:

- the calibrated camera gets a ``CameraCalibration`` row and the plain one
  does not;
- the pose is ingested with its body parts and one ``.Camera`` row per video,
  with a null calibration for the uncalibrated camera
  (``ImportedMultiCameraPose``), and its fetchers return the stored data;
- the per-camera children are not ingested as single-camera ``ImportedPose``.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from hdmf.common import DynamicTable, VectorData, VectorIndex
from pynwb import NWBHDF5IO, NWBFile
from pynwb.device import Device, DeviceModel
from pynwb.file import Subject
from pynwb.image import ImageSeries

ndx_pose = pytest.importorskip("ndx_pose")
if not hasattr(ndx_pose, "MultiCameraPoseEstimation"):
    pytest.skip(
        "ndx-pose < 0.4 has no MultiCameraPoseEstimation",
        allow_module_level=True,
    )

N_FRAMES = 50
RATE = 50.0
CALIBRATED = "MulticamTestCamera1"  # ndx-pose CalibratedCamera, id 1
UNCALIBRATED = "MulticamTestCamera2"  # plain Device, id 2
CAMERAS = (CALIBRATED, UNCALIBRATED)
NODES = ("Snout", "TailBase")


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
                data=["multicam_test_task"],
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


@pytest.fixture(scope="module")
def multicam_nwb(verbose_context):
    from spyglass.common import Nwbfile
    from spyglass.data_import import insert_sessions
    from spyglass.settings import raw_dir
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    rng = np.random.default_rng(0)
    nwbfile = NWBFile(
        session_description="test multi-camera pose",
        identifier="test_multicam_pose",
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    subject = Subject(
        subject_id="test_multicam_subject", species="Rattus norvegicus"
    )
    nwbfile.subject = subject
    nwbfile.add_epoch(
        start_time=0.0, stop_time=(N_FRAMES - 1) / RATE, tags=["01_session"]
    )

    camera_model = DeviceModel(
        name="test camera model", manufacturer="test maker"
    )
    nwbfile.add_device_model(camera_model)
    cameras = {
        CALIBRATED: nwbfile.add_device(
            ndx_pose.CalibratedCamera(
                name=CALIBRATED,
                intrinsic_matrix=np.eye(3) * 2,
                rotation_matrix=np.eye(3),
                translation_vector=np.array([0.0, 0.0, 20.0]),
                distortion_coefficients=np.zeros(4),
                model=camera_model,
            )
        ),
        UNCALIBRATED: nwbfile.add_device(
            Device(name=UNCALIBRATED, description="camera without calibration")
        ),
    }
    camera_poses = []
    for name, camera in cameras.items():
        video = ImageSeries(
            name=f"Video{name}",
            description=f"video of {name}",
            unit="n.a.",
            format="external",
            external_file=[f"{name}.mp4"],
            starting_frame=[0],
            rate=RATE,  # rate-based: no timestamps
            num_samples=N_FRAMES,
            device=camera,
        )
        nwbfile.add_acquisition(video)
        camera_poses.append(
            ndx_pose.PoseEstimation(
                name=f"{name}PoseEstimation",
                description=f"camera {name} of the pose",
                device=camera,
                source_video=video,
            )
        )
    nwbfile.create_processing_module("tasks", "tasks").add(_task_table([1, 2]))

    skeleton = ndx_pose.Skeleton(
        name="skeleton", nodes=list(NODES), edges=[[0, 1]], subject=subject
    )
    data = rng.random((N_FRAMES, 3))
    confidence = rng.random(N_FRAMES)
    series = [
        ndx_pose.PoseEstimationSeries(
            name=f"PoseEstimationSeries{node}",
            description=f"{node} position",
            data=data + i,
            confidence=confidence,
            unit="millimeters",
            reference_frame="arena center",
            rate=RATE,
            starting_time=0.0,
        )
        for i, node in enumerate(NODES)
    ]
    pose = ndx_pose.MultiCameraPoseEstimation(
        name="PoseEstimation3D",
        description="test 3D pose",
        pose_estimation_series=series,
        pose_estimations=camera_poses,
        skeleton=skeleton,
        scorer="test scorer",
        source_software="DANNCE",
        source_software_version="1.0",
    )
    behavior = nwbfile.create_processing_module("behavior", "behavior")
    behavior.add(ndx_pose.Skeletons(skeletons=[skeleton]))
    behavior.add(pose)

    raw_file_name = "test_import_multicam.nwb"
    nwb_path = Path(raw_dir) / raw_file_name
    key = dict(nwb_file_name=get_nwb_copy_filename(raw_file_name))
    if (Nwbfile & key) or nwb_path.exists():
        (Nwbfile & key).delete(safemode=False)
        nwb_path.unlink(missing_ok=True)
    with NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)

    insert_sessions(raw_file_name, raise_err=True)

    yield dict(key=key, data=data, confidence=confidence)

    with verbose_context:
        (Nwbfile & key).delete(safemode=False)
        nwb_path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def multicam_pose(multicam_nwb):
    from spyglass.position.v1.imported_multicam_pose import (
        ImportedMultiCameraPose,
    )

    return ImportedMultiCameraPose & multicam_nwb["key"]


def test_calibration_only_for_calibrated_camera(multicam_nwb):
    from spyglass.position.v1.imported_multicam_pose import CameraCalibration

    calibrations = CameraCalibration & multicam_nwb["key"]
    assert list(calibrations.fetch("camera_name")) == [CALIBRATED]
    calibration = calibrations.fetch_calibration()
    np.testing.assert_array_equal(calibration["intrinsic_matrix"], np.eye(3) * 2)
    np.testing.assert_array_equal(calibration["translation_vector"], [0, 0, 20])


def test_multicam_pose_entries(multicam_pose):
    from spyglass.position.v1.imported_multicam_pose import (
        ImportedMultiCameraPose,
    )

    entry = multicam_pose.fetch1()
    assert entry["pose_estimation_name"] == "PoseEstimation3D"
    assert entry["source_software"] == "DANNCE"
    assert entry["source_software_version"] == "1.0"
    assert entry["scorer"] == "test scorer"
    assert (
        entry["interval_list_name"] == "pose_PoseEstimation3D_valid_intervals"
    )

    parts = (ImportedMultiCameraPose.BodyPart & multicam_pose).fetch(
        "part_name"
    )
    assert sorted(parts) == sorted(f"PoseEstimationSeries{n}" for n in NODES)

    # One row per video; only the calibrated camera links a calibration
    cameras = (ImportedMultiCameraPose.Camera & multicam_pose).fetch(
        as_dict=True, order_by="video_file_num"
    )
    assert [c["video_file_num"] for c in cameras] == [1, 2]
    assert [c["camera_name"] for c in cameras] == [CALIBRATED, None]


def test_no_single_camera_pose(multicam_nwb):
    from spyglass.position.v1.imported_pose import ImportedPose

    # The series-less per-camera children are not single-camera poses
    assert not (ImportedPose & multicam_nwb["key"])


def test_fetch_pose_dataframe(multicam_nwb, multicam_pose):
    df = multicam_pose.fetch_pose_dataframe()
    assert df.shape == (N_FRAMES, 4 * len(NODES))
    np.testing.assert_allclose(df.index, np.arange(N_FRAMES) / RATE)
    snout = df["PoseEstimationSeriesSnout"]
    np.testing.assert_array_equal(snout[["x", "y", "z"]], multicam_nwb["data"])
    np.testing.assert_array_equal(
        snout["likelihood"], multicam_nwb["confidence"]
    )


def test_fetch_skeleton_calibrations_videos(multicam_pose):
    skeleton = multicam_pose.fetch_skeleton()
    assert skeleton["nodes"] == list(NODES)
    assert skeleton["edges"] == [list(NODES)]

    calibrations = multicam_pose.fetch_calibrations()
    assert list(calibrations) == [CALIBRATED]
    np.testing.assert_array_equal(
        calibrations[CALIBRATED]["intrinsic_matrix"], np.eye(3) * 2
    )

    # Videos of all cameras, calibrated or not
    assert multicam_pose.fetch_video_paths() == {
        name: f"{name}.mp4" for name in CAMERAS
    }
