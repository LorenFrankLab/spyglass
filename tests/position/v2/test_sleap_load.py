"""SL04 — SLEAP NWB import smoke test.

Tests that PoseEstim.load_from_nwb() can load a SLEAP-exported NWB file
(ndx-pose PoseEstimation with source_software="SLEAP") with no code changes.
"""

from datetime import datetime

import ndx_pose
import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile


@pytest.fixture
def sleap_nwb_path(tmp_path):
    """Minimal NWB file mimicking SLEAP's ndx-pose export format.

    Creates a PoseEstimation object with source_software="SLEAP" and
    three bodypart series so we can verify load_from_nwb returns the
    correct metadata without running any inference.
    """
    nwb_path = tmp_path / "sleap_output.nwb"

    nwbfile = NWBFile(
        session_description="Synthetic SLEAP output for testing",
        identifier="sleap_smoke_test_001",
        session_start_time=datetime(2025, 1, 1),
    )

    skeleton = ndx_pose.Skeleton(
        name="rat",
        nodes=["nose", "earL", "tailBase"],
        edges=np.array([[0, 1], [0, 2]], dtype="uint8"),
    )

    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="Behavioral data"
    )
    skeletons = ndx_pose.Skeletons(skeletons=[skeleton])
    behavior_module.add_data_interface(skeletons)

    n_frames = 50
    timestamps = np.linspace(0, 5, n_frames)

    series_list = []
    for bodypart in ["nose", "earL", "tailBase"]:
        series_list.append(
            ndx_pose.PoseEstimationSeries(
                name=bodypart,
                description=f"Pose for {bodypart}",
                data=np.random.rand(n_frames, 2) * 100,
                unit="pixels",
                reference_frame="(0,0) top-left",
                timestamps=timestamps,
                confidence=np.random.rand(n_frames),
            )
        )

    pose_estimation = ndx_pose.PoseEstimation(
        name="PoseEstimation",
        pose_estimation_series=series_list,
        description="SLEAP single-animal pose estimation",
        original_videos=["test_video.mp4"],
        dimensions=np.array([[640, 480]], dtype="uint16"),
        skeleton=skeleton,
        source_software="SLEAP",
        source_software_version="1.3.3",
    )

    behavior_module.add(pose_estimation)

    with NWBHDF5IO(str(nwb_path), mode="w") as io:
        io.write(nwbfile)

    return nwb_path


def test_load_from_nwb_sleap_source_software(sleap_nwb_path):
    """load_from_nwb returns source_software='SLEAP' for SLEAP NWB files."""
    from spyglass.position.v2.estim import PoseEstim

    metadata = PoseEstim.load_from_nwb(sleap_nwb_path)

    assert metadata["source_software"] == "SLEAP"


def test_load_from_nwb_returns_correct_bodyparts(sleap_nwb_path):
    """load_from_nwb returns all bodypart names from the NWB file."""
    from spyglass.position.v2.estim import PoseEstim

    metadata = PoseEstim.load_from_nwb(sleap_nwb_path)

    assert set(metadata["bodyparts"]) == {"nose", "earL", "tailBase"}


def test_load_from_nwb_returns_valid_metadata(sleap_nwb_path):
    """load_from_nwb returns a complete metadata dict for a SLEAP NWB file."""
    from spyglass.position.v2.estim import PoseEstim

    metadata = PoseEstim.load_from_nwb(sleap_nwb_path)

    assert isinstance(metadata, dict)
    required_keys = {
        "nwb_file_path",
        "pose_estimation_name",
        "bodyparts",
        "n_frames",
        "scorer",
        "source_software",
    }
    assert required_keys.issubset(metadata.keys())
    assert metadata["n_frames"] == 50


def test_load_from_nwb_multimodel_selects_sleap_by_name(
    mock_ndx_pose_nwb_multimodel,
):
    """load_from_nwb selects SLEAP PoseEstimation by name in a multi-model NWB.

    Uses the existing mock_ndx_pose_nwb_multimodel fixture which creates a
    file with pose_model_0 (DeepLabCut) and pose_model_1 (SLEAP).
    """
    from spyglass.position.v2.estim import PoseEstim

    metadata = PoseEstim.load_from_nwb(
        mock_ndx_pose_nwb_multimodel,
        pose_estimation_name="pose_model_1",
    )
    assert metadata["source_software"] == "SLEAP"
