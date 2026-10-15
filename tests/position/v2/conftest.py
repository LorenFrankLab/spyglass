"""Test fixtures for position v2 tests."""

from datetime import datetime

import datajoint as dj
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


@pytest.fixture(autouse=True)
def _neutralize_model_reuse_guard(monkeypatch, pv2_train):
    """Disable the redundant-model guard by default across the v2 suite.

    Most tests declare models as setup, not to exercise the guard (which is
    unit-tested directly in ``test_model_reuse.py``). Patch the reuse lookup to
    find no candidates so ``ModelSelection.insert1`` never blocks; tests that
    verify the guard override ``Model.reusable_for`` within their own body.
    """
    monkeypatch.setattr(
        pv2_train.Model,
        "reusable_for",
        classmethod(lambda cls, skeleton_id, subjects=None, exclude=None: []),
    )


@pytest.fixture(scope="session")
def bodypart(pv2_train):
    """Fixture for BodyPart class."""
    yield pv2_train.BodyPart()


@pytest.fixture
def skeleton(pv2_train):
    """Fixture for Skeleton class.

    Function-scoped with always-run post-yield teardown: skeletons the test
    inserts are removed afterward so they do not accumulate across the session
    and collide on the topology-hash unique index. This matters under
    ``--no-teardown``, where the database persists between tests.
    """
    tbl = pv2_train.Skeleton()
    before = set(tbl.fetch("skeleton_id"))
    yield tbl
    new = set(tbl.fetch("skeleton_id")) - before
    if new:
        (tbl & [{"skeleton_id": s} for s in new]).delete(safemode=False)


@pytest.fixture(scope="session")
def model_params(pv2_train):
    """Fixture for ModelParams class."""
    yield pv2_train.ModelParams()


@pytest.fixture(scope="session")
def model_sel(pv2_train):
    """Fixture for ModelSelection class."""
    yield pv2_train.ModelSelection()


@pytest.fixture
def model(pv2_train):
    """Fixture for Model class.

    Function-scoped with always-run post-yield teardown: models the test
    creates are removed afterward (cascading to their downstream pose
    entries) so a trained model does not leak into a later test — e.g. making
    a "weightless" inference unexpectedly succeed. Matters under
    ``--no-teardown``, where the database persists between tests.
    """
    tbl = pv2_train.Model()
    before = set(tbl.fetch("model_id"))
    yield tbl
    new = set(tbl.fetch("model_id")) - before
    if new:
        (tbl & [{"model_id": m} for m in new]).delete(safemode=False)


@pytest.fixture(scope="session")
def pv2_estim(position_v2):
    """Fixture for position.v2.estim module."""
    yield position_v2.estim


@pytest.fixture(scope="session")
def PoseV2(pv2_estim):
    """Fixture for PoseV2 class."""
    yield pv2_estim.PoseV2


@pytest.fixture(scope="session")
def pose_v2_instance(pv2_estim):
    """Fixture for PoseV2 instance."""
    yield pv2_estim.PoseV2()


@pytest.fixture(scope="session")
def PoseParams(pv2_estim):
    """Fixture for PoseParams class."""
    yield pv2_estim.PoseParams


@pytest.fixture(scope="session")
def PoseEstim(pv2_estim):
    """Fixture for PoseEstim class."""
    yield pv2_estim.PoseEstim


@pytest.fixture(scope="session")
def PoseEstimParams(pv2_estim):
    """Fixture for PoseEstimParams class."""
    yield pv2_estim.PoseEstimParams


@pytest.fixture(scope="session")
def PoseEstimSelection(pv2_estim):
    """Fixture for PoseEstimSelection class."""
    yield pv2_estim.PoseEstimSelection


@pytest.fixture(scope="session")
def PoseSelection(pv2_estim):
    """Fixture for PoseSelection class."""
    yield pv2_estim.PoseSelection


@pytest.fixture(scope="session")
def pv2_video(position_v2):
    """Fixture for position.v2.video module."""
    yield position_v2.video


@pytest.fixture(scope="session")
def VidFileGroup(pv2_video):
    """Fixture for VidFileGroup class."""
    yield pv2_video.VidFileGroup


@pytest.fixture(scope="session")
def VideoGroupParams(pv2_video):
    """Fixture for VideoGroupParams class."""
    yield pv2_video.VideoGroupParams


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
    df.index = pd.Index(np.arange(n_frames) / 30.0, name="time")

    # Save as h5 and csv
    h5_path = tmp_path / "test_video_dlc_output.h5"
    csv_path = tmp_path / "test_video_dlc_output.csv"

    df.to_hdf(str(h5_path), key="df_with_missing", mode="w")
    df.to_csv(str(csv_path))

    return {"h5": h5_path, "csv": csv_path, "dataframe": df}


# ----------------------------- DLC Fixtures -----------------------------------
# ``skip_if_no_dlc`` / ``skip_if_no_sleap`` are defined in the shared
# ``tests/position/conftest.py`` so v1, v2, and utils suites share one gate.


@pytest.fixture(scope="session")
def dlc_project_config(tmp_path_factory):
    """Session-scoped DLC project created from tests/_data/deeplabcut/.

    Builds a minimal DLC project directory (config.yaml, fake trained model,
    labeled frames) using ``make_dlc_project()``.  No database or DLC
    installation required.

    Returns
    -------
    Path
        Absolute path to the project's ``config.yaml``.
    """
    from tests.position.v2.make_example_dlc_project import make_dlc_project

    project_dir = tmp_path_factory.mktemp("dlc_project", numbered=False)
    return make_dlc_project(project_dir)


@pytest.fixture(scope="session")
def dlc_bootstrapped_session(dlc_project_config):
    """Register minimal Spyglass DB entries for the test DLC project.

    Calls ``bootstrap_dlc_session()`` so that
    ``VidFileGroup.create_from_dlc_config()`` can match the project's video
    paths to an existing ``Nwbfile`` entry.

    Returns
    -------
    str
        The ``nwb_file_name`` inserted into ``Nwbfile``.
    """
    from tests.position.v2.make_example_dlc_project import bootstrap_dlc_session

    return bootstrap_dlc_session(dlc_project_config)


# ------------------------- Multi-session Fixtures ---------------------------
# `VidFileGroup` may legitimately span sessions (training), while the inference
# path requires exactly one. The test DB ships a single session (minirec), so
# these fixtures register synthetic ones to exercise both sides.
#
# They are deliberately FUNCTION-scoped with teardown, never session-scoped.
# A synthetic Session/Nwbfile row that outlives its test pollutes the shared
# database for the rest of the run: unrelated suites assume the minirec session
# is the only one and break in confusing ways (`fetch1 requires exactly one
# tuple` out of `get_position_interval_epoch`, `IndexError` on `.fetch()[0]`,
# "No valid ..." in burst detection). Session-scoping these once cost ~7 extra
# failures across `tests/position/v1` and `tests/spikesorting`. Keep the rows
# alive only as long as the test that needs them.


def _register_sessions(stems, videos_per_session=1):
    """Register synthetic sessions and return their names and VideoFile keys.

    Parameters
    ----------
    stems : iterable of str
        NWB stems; each becomes ``f"{stem}_.nwb"``.
    videos_per_session : int, optional
        ``VideoFile`` rows to create per session, numbered from 0.

    Returns
    -------
    tuple[tuple[str, ...], tuple[dict, ...]]
        ``(names, keys)`` -- one name per stem, and one ``VideoFile`` primary
        key per (session, video).
    """
    from tests.position.v2.make_example_dlc_project import (
        bootstrap_minimal_session,
    )

    names, keys = [], []
    for stem in stems:
        name = bootstrap_minimal_session(
            stem,
            [
                f"/nonexistent/{stem}/video{i}.mp4"
                for i in range(videos_per_session)
            ],
            task_name=f"pv2_task_{stem}",
            camera_name="pv2_test_cam",
        )
        names.append(name)
        keys.extend(
            {"nwb_file_name": name, "epoch": 1, "video_file_num": i}
            for i in range(videos_per_session)
        )
    return tuple(names), tuple(keys)


def _drop_sessions(names):
    """Delete synthetic sessions, cascading to everything built on them."""
    from spyglass.common import Nwbfile

    query = Nwbfile & [{"nwb_file_name": n} for n in names]
    if query:
        query.super_delete(warn=False, safemode=False)


@pytest.fixture
def two_sessions(mini_insert):
    """Two synthetic single-epoch sessions, one ``VideoFile`` row each.

    Registers ``Nwbfile`` → ``Session`` → ``TaskEpoch`` → ``VideoFile`` rows
    for two distinct NWB files. The NWB placeholders are copies of minirec, so
    ``AnalysisNwbfile.create()`` can open them.

    The video paths recorded in ``VideoFile.path`` are synthetic and are never
    opened -- these fixtures support schema/query tests, not inference.

    Teardown deletes both ``Nwbfile`` rows, which cascades to every group,
    model, and pose entry built on them. See the module note above on why this
    must not be session-scoped.

    Returns
    -------
    dict
        ``{"names": (a, b), "keys": (key_a, key_b)}`` where each key is a
        ``VideoFile`` primary key dict for ``video_file_num=0``.
    """
    names, keys = _register_sessions(("pv2_sess_a", "pv2_sess_b"))
    yield {"names": names, "keys": keys}
    _drop_sessions(names)


@pytest.fixture
def make_vid_group(VidFileGroup):
    """Factory building a ``VidFileGroup`` from ``VideoFile`` keys.

    Function-scoped with always-run teardown: groups created through the
    factory are removed afterward so they do not accumulate across the session
    or collide on ``vid_group_id``. This matters under ``--no-teardown``, where
    the database persists between tests.

    Returns
    -------
    callable
        ``_make(vid_group_id, video_keys, description=...) -> str``
    """
    created = []

    def _make(vid_group_id, video_keys, description="multi-session test group"):
        VidFileGroup().insert1(
            {
                "vid_group_id": vid_group_id,
                "description": description,
                "files": list(video_keys),
            }
        )
        created.append(vid_group_id)
        return vid_group_id

    yield _make

    for gid in created:
        (VidFileGroup() & {"vid_group_id": gid}).super_delete(
            warn=False, safemode=False
        )


@pytest.fixture
def multi_session_group(two_sessions, make_vid_group):
    """A ``VidFileGroup`` spanning both synthetic sessions.

    Legal to create (training groups may span sessions); the inference path
    must reject it.

    Returns
    -------
    str
        The ``vid_group_id``.
    """
    return make_vid_group("pv2_multi_sess_grp", two_sessions["keys"])


@pytest.fixture
def single_session_group(two_sessions, make_vid_group):
    """A ``VidFileGroup`` holding one video from one session.

    Positive control for the inference-path session guard.

    Returns
    -------
    str
        The ``vid_group_id``.
    """
    return make_vid_group("pv2_single_sess_grp", two_sessions["keys"][:1])


@pytest.fixture
def multicam_session(mini_insert):
    """One synthetic session holding three videos, one per camera.

    The 3-D shape: several ``VidFileGroup.File`` rows sharing a single NWB
    parent. ``get_nwb_file`` must collapse them to that one name rather than
    counting once per camera.

    Function-scoped with teardown -- see the module note above.

    Returns
    -------
    dict
        ``{"name": str, "keys": tuple[dict, ...]}`` -- three ``VideoFile``
        primary keys, ``video_file_num`` 0/1/2.
    """
    names, keys = _register_sessions(("pv2_sess_3cam",), videos_per_session=3)
    yield {"name": names[0], "keys": keys}
    _drop_sessions(names)


@pytest.fixture
def multicam_group(multicam_session, VidFileGroup):
    """A three-camera ``VidFileGroup`` within one session.

    Built directly rather than through ``make_vid_group`` so ``camera_index``
    can be supplied. Torn down after each test.

    Returns
    -------
    str
        The ``vid_group_id``.
    """
    gid = "pv2_multicam_grp"
    VidFileGroup().insert1(
        {
            "vid_group_id": gid,
            "description": "three-camera single-session group",
            "files": list(multicam_session["keys"]),
            "camera_indices": [0, 1, 2],
        }
    )
    yield gid
    (VidFileGroup() & {"vid_group_id": gid}).super_delete(
        warn=False, safemode=False
    )


@pytest.fixture
def training_group(two_sessions, pv2_video):
    """A ``VidFileGroup`` spanning both sessions; backs ``stub_model``.

    Distinct from ``multi_session_group`` only in intent.

    Teardown deletes the group *master*. The ``two_sessions`` cascade reaches
    only ``VidFileGroup.File`` (via ``VideoFile``), not the master row -- so
    without this the group would survive as an empty shell, and the next test's
    insert would short-circuit on "already exists" and leave it fileless.
    Deleting the master also cascades to ``ModelSelection`` → ``Model``.

    Returns
    -------
    str
        The ``vid_group_id``.
    """
    gid = "pv2_training_grp"
    tbl = pv2_video.VidFileGroup()
    tbl.insert1(
        {
            "vid_group_id": gid,
            "description": "multi-session training group",
            "files": list(two_sessions["keys"]),
        }
    )
    yield gid
    (tbl & {"vid_group_id": gid}).super_delete(warn=False, safemode=False)


@pytest.fixture
def stub_model(pv2_train, training_group):
    """A ``Model`` row with no trained weights, for selection-level tests.

    Builds the minimal Skeleton → ModelParams → ModelSelection → Model chain by
    direct insert, so tests that only exercise the *selection* layer do not pay
    for training. The backing video group deliberately spans two sessions,
    which is legal for training and is what
    ``TestMultiSessionTrainingAllowed`` asserts.

    ``ModelSelection`` and ``Model`` are removed by the ``two_sessions``
    cascade; the ``Skeleton`` / ``ModelParams`` lookups are idempotent and
    reused across tests.

    Returns
    -------
    dict
        ``{"model_id", "vid_group_id", "model_params_id", "skeleton_id"}``
    """
    skeleton_id = "pv2_sess_guard_skel"
    # 2-LED set so the stock ``default`` PoseParams (2pt centroid, two_pt
    # orientation on redLED_C→greenLED) matches and validation stays quiet.
    pv2_train.Skeleton().insert1(
        {
            "skeleton_id": skeleton_id,
            "bodyparts": ["greenLED", "redLED_C", "tailBase"],
            "edges": [("greenLED", "redLED_C"), ("redLED_C", "tailBase")],
        },
        check_duplicates=False,
        skip_duplicates=True,
    )

    mp_id = pv2_train.ModelParams().insert1(
        {
            "model_params_id": "pv2_sess_guard_mp",
            "tool": "DLC",
            "params": {
                "project_path": "/nonexistent/pv2_sess_guard_project",
                "shuffle": 1,
                "trainingsetindex": 0,
            },
            "skeleton_id": skeleton_id,
        },
        skip_duplicates=True,
    )["model_params_id"]

    sel_id = "pv2_sess_guard_sel"
    sel_key = {
        "model_params_id": mp_id,
        "tool": "DLC",
        "vid_group_id": training_group,
        "model_selection_id": sel_id,
    }
    pv2_train.ModelSelection().insert1(
        sel_key, allow_redundant_model=True, skip_duplicates=True
    )

    model_id = "pv2_sess_guard_model"
    pv2_train.Model().insert1(
        {**sel_key, "model_id": model_id, "model_path": "stub_scorer"},
        allow_direct_insert=True,
        skip_duplicates=True,
    )

    yield {
        "model_id": model_id,
        "vid_group_id": training_group,
        "model_params_id": mp_id,
        "skeleton_id": skeleton_id,
    }


def _make_pose_chain(pv2_estim, model_id, vid_group_id, nwb_file_name):
    """Build PoseEstimSelection → PoseEstim → PoseSelection → PoseV2 rows.

    Direct inserts only -- no inference, no ``compute_pose_outputs``. The
    ``PoseV2`` row carries a real ``AnalysisNwbfile`` but no stored object ids,
    which is enough for selection/restriction tests.

    Returns
    -------
    tuple[dict, dict]
        ``(pose_selection_key, selection_key)`` -- the second is the
        ``PoseEstimSelection`` key, whose deletion cascades to the rest.
    """
    from spyglass.common import AnalysisNwbfile

    pv2_estim.PoseParams().insert_default(skip_duplicates=True)

    sel = {
        "model_id": model_id,
        "vid_group_id": vid_group_id,
        "pose_estim_params_id": "default",
    }
    pv2_estim.PoseEstimSelection().insert1(
        {**sel, "task_mode": "load", "output_dir": ""}
    )

    analysis_file = AnalysisNwbfile().create(nwb_file_name)
    AnalysisNwbfile().add(nwb_file_name, analysis_file)
    pv2_estim.PoseEstim().insert1({**sel, "analysis_file_name": analysis_file})

    key = {**sel, "pose_params_id": "default"}
    pv2_estim.PoseSelection().insert1(key)
    pv2_estim.PoseV2().insert1(
        {**key, "analysis_file_name": analysis_file},
        allow_direct_insert=True,
    )
    return key, sel


@pytest.fixture
def pose_v2_row(pv2_estim, stub_model, single_session_group, two_sessions):
    """One ``PoseV2`` entry whose videos come from the first test session.

    Torn down after each test: deleting the ``PoseEstimSelection`` row cascades
    to everything built on it.

    Returns
    -------
    dict
        The ``PoseV2`` primary key, plus ``nwb_file_name`` for assertions.
    """
    key, sel = _make_pose_chain(
        pv2_estim,
        stub_model["model_id"],
        single_session_group,
        two_sessions["names"][0],
    )
    yield {**key, "nwb_file_name": two_sessions["names"][0]}
    (pv2_estim.PoseEstimSelection() & sel).super_delete(
        warn=False, safemode=False
    )


@pytest.fixture
def multicam_pose_v2_row(
    pv2_estim, stub_model, multicam_group, multicam_session
):
    """One ``PoseV2`` entry backed by a three-camera group in one session.

    Exercises the fan-out ``fetch_by_epoch`` must collapse: three
    ``VidFileGroup.File`` rows all matching the same epoch restriction.

    Returns
    -------
    dict
        The ``PoseV2`` primary key.
    """
    key, sel = _make_pose_chain(
        pv2_estim,
        stub_model["model_id"],
        multicam_group,
        multicam_session["name"],
    )
    yield key
    (pv2_estim.PoseEstimSelection() & sel).super_delete(
        warn=False, safemode=False
    )
