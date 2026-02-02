"""Test fixtures for position v2 tests."""

from datetime import datetime

import ndx_pose
import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject

# ----------------------------- Class Fixtures ---------------------------------


@pytest.fixture(scope="session")
def position_v2():
    from spyglass.position import v2

    yield v2


@pytest.fixture(scope="session")
def pv2_train(position_v2):
    """Fixture for position.v2.train module."""
    yield position_v2.train


@pytest.fixture(scope="session")
def bodypart(pv2_train):
    """Fixture for BodyPart class."""
    yield pv2_train.BodyPart()


@pytest.fixture(scope="session")
def skeleton(pv2_train):
    """Fixture for Skeleton class."""
    yield pv2_train.Skeleton()


@pytest.fixture(scope="session")
def model_params(pv2_train):
    """Fixture for ModelParams class."""
    yield pv2_train.ModelParams()


@pytest.fixture(scope="session")
def model_sel(pv2_train):
    """Fixture for ModelSelection class."""
    yield pv2_train.ModelSelection()


@pytest.fixture(scope="session")
def model(pv2_train):
    """Fixture for Model class."""
    yield pv2_train.Model()


# ----------------------------- NWB Fixtures -----------------------------------


@pytest.fixture
def mock_ndx_pose_nwb_file(tmp_path):
    """Create a mock NWB file with ndx-pose data.

    This fixture creates a complete NWB file with:
    - ndx_pose.Skeletons container with skeleton graph
    - ndx_pose.PoseEstimation with pose estimation data
    - Proper metadata (subject, session, etc.)

    Returns
    -------
    Path
        Path to the created NWB file
    """
    nwb_path = tmp_path / "test_pose_model.nwb"

    # Create NWB file with metadata
    nwbfile = NWBFile(
        session_description="Test session for pose estimation",
        identifier="test_pose_001",
        session_start_time=datetime(2025, 1, 1, 0, 0, 0),
        subject=Subject(
            subject_id="test_subject",
            species="Rattus norvegicus",
            age="P90D",
            sex="M",
        ),
    )

    # Create skeleton using ndx-pose (use actual BodyPart table entries)
    skeleton = ndx_pose.Skeleton(
        name="test_skeleton",
        nodes=["nose", "earL", "earR", "tailBase"],
        # Edges define connectivity: (node1_idx, node2_idx)
        edges=np.array([[0, 1], [0, 2], [0, 3]], dtype="uint8"),
    )

    # Create behavior processing module
    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="Behavioral data"
    )

    # Create Skeletons container and add to behavior module
    skeletons = ndx_pose.Skeletons(skeletons=[skeleton])
    behavior_module.add_data_interface(skeletons)

    # Create pose estimation data
    # Note: PoseEstimationSeries expects data per series, shape (n_frames, 2) or (n_frames, 3)
    n_frames = 100

    # Mock pose data for all nodes: [frames, (x, y)]
    pose_data = np.random.rand(n_frames, 2) * 100

    # Mock confidence data: [frames]
    confidence_data = np.random.rand(n_frames) * 0.3 + 0.7  # 0.7-1.0

    # Create pose estimation series
    pose_estimation_series = ndx_pose.PoseEstimationSeries(
        name="pose_estimation",
        description="Test pose estimation data",
        data=pose_data,
        unit="pixels",
        reference_frame="(0,0) is top-left corner",
        timestamps=np.linspace(0, 10, n_frames),
        confidence=confidence_data,
        confidence_definition="Softmax output from neural network",
    )

    # Create pose estimation container
    pose_estimation = ndx_pose.PoseEstimation(
        name="test_pose_estimation",
        pose_estimation_series=[pose_estimation_series],
        description="Test pose estimation",
        original_videos=["test_video.mp4"],
        labeled_videos=["test_video_labeled.mp4"],
        dimensions=np.array([[640, 480]], dtype="uint16"),
        skeleton=skeleton,
        # Add source software metadata
        source_software="DeepLabCut",
        source_software_version="3.0.0",
    )

    # Add pose estimation to behavior module
    behavior_module.add(pose_estimation)

    # Write NWB file
    with NWBHDF5IO(str(nwb_path), mode="w") as io:
        io.write(nwbfile)

    return nwb_path


@pytest.fixture
def mock_ndx_pose_nwb_multimodel(tmp_path):
    """Create a mock NWB file with multiple pose estimation models.

    Returns
    -------
    Path
        Path to the created NWB file
    """
    nwb_path = tmp_path / "test_multimodel_pose.nwb"

    nwbfile = NWBFile(
        session_description="Multi-model test session",
        identifier="test_pose_002",
        session_start_time=datetime(2025, 1, 1, 0, 0, 0),
    )

    # Create two different skeletons
    skeleton1 = ndx_pose.Skeleton(
        name="skeleton_head",
        nodes=["nose", "leftear", "rightear"],
        edges=np.array([[0, 1], [0, 2]], dtype="uint8"),
    )

    skeleton2 = ndx_pose.Skeleton(
        name="skeleton_body",
        nodes=["spine1", "spine2", "spine3", "tailbase"],
        edges=np.array([[0, 1], [1, 2], [2, 3]], dtype="uint8"),
    )

    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="Behavioral data"
    )

    skeletons = ndx_pose.Skeletons(skeletons=[skeleton1, skeleton2])
    behavior_module.add_data_interface(skeletons)

    # Add pose estimation for each skeleton
    for i, skeleton in enumerate([skeleton1, skeleton2]):
        n_frames = 50

        # Data shape: (n_frames, 2) for 2D or (n_frames, 3) for 3D
        pose_data = np.random.rand(n_frames, 2) * 100
        confidence_data = np.random.rand(n_frames) * 0.3 + 0.7

        pose_series = ndx_pose.PoseEstimationSeries(
            name=f"pose_estimation_{i}",
            description=f"Pose estimation for {skeleton.name}",
            data=pose_data,
            unit="pixels",
            reference_frame="(0,0) is top-left corner",
            timestamps=np.linspace(0, 5, n_frames),
            confidence=confidence_data,
            confidence_definition="Model confidence",
        )

        pose_estimation = ndx_pose.PoseEstimation(
            name=f"pose_model_{i}",
            pose_estimation_series=[pose_series],
            description=f"Model {i} pose estimation",
            original_videos=[f"video_{i}.mp4"],
            dimensions=np.array([[640, 480]], dtype="uint16"),
            skeleton=skeleton,
            source_software="DeepLabCut" if i == 0 else "SLEAP",
            source_software_version="3.0.0" if i == 0 else "1.3.0",
        )

        behavior_module.add(pose_estimation)

    with NWBHDF5IO(str(nwb_path), mode="w") as io:
        io.write(nwbfile)

    return nwb_path


@pytest.fixture
def mock_nwb_file_for_parent(tmp_path):
    """Create a minimal NWB file to serve as a parent file.

    This simulates a session NWB file that can be used as
    the parent for derived analysis files.

    Returns
    -------
    Path
        Path to the created NWB file
    """
    nwb_path = tmp_path / "parent_session.nwb"

    nwbfile = NWBFile(
        session_description="Parent session for testing",
        identifier="parent_001",
        session_start_time=datetime(2025, 1, 1, 0, 0, 0),
        subject=Subject(
            subject_id="test_subject_001",
            species="Rattus norvegicus",
        ),
    )

    with NWBHDF5IO(str(nwb_path), mode="w") as io:
        io.write(nwbfile)

    return nwb_path


# ----------------------------- Inference Fixtures -----------------------------


@pytest.fixture
def mock_video_file(tmp_path):
    """Create a mock video file for testing inference.

    Creates a simple test video using OpenCV if available,
    otherwise creates a placeholder file.

    Returns
    -------
    Path
        Path to the created video file
    """
    video_path = tmp_path / "test_video.avi"

    try:
        import cv2

        # Create a simple 10-frame video (640x480, 30fps)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

        for i in range(10):
            # Create a blank frame with frame number
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                f"Frame {i}",
                (250, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            out.write(frame)

        out.release()
    except ImportError:
        # If cv2 not available, create a placeholder file
        video_path.write_text("MOCK_VIDEO_FILE")

    return video_path


@pytest.fixture
def mock_dlc_inference_output(tmp_path):
    """Create mock DLC inference output files (h5 and csv).

    Returns
    -------
    dict
        Dictionary with keys 'h5' and 'csv' pointing to the output files
    """
    import pandas as pd

    # Create mock DLC output structure
    # DLC outputs have multi-level columns: [scorer, bodypart, coords]
    scorer = "DLC_resnet50_TESTSep8shuffle1_6"
    bodyparts = ["bodypart1", "bodypart2", "bodypart3", "objectA"]
    coords = ["x", "y", "likelihood"]

    # Create MultiIndex columns
    columns = pd.MultiIndex.from_product(
        [[scorer], bodyparts, coords], names=["scorer", "bodyparts", "coords"]
    )

    # Create mock data (10 frames)
    n_frames = 10
    data = np.random.rand(n_frames, len(bodyparts) * len(coords)) * 100

    # Set likelihood column to reasonable values (0.7-1.0)
    for i, bp in enumerate(bodyparts):
        likelihood_col = (scorer, bp, "likelihood")
        col_idx = columns.get_loc(likelihood_col)
        data[:, col_idx] = np.random.rand(n_frames) * 0.3 + 0.7

    df = pd.DataFrame(data, columns=columns)

    # Save as h5 and csv
    h5_path = tmp_path / "test_video_dlc_output.h5"
    csv_path = tmp_path / "test_video_dlc_output.csv"

    df.to_hdf(str(h5_path), key="df_with_missing", mode="w")
    df.to_csv(str(csv_path))

    return {"h5": h5_path, "csv": csv_path, "dataframe": df}
