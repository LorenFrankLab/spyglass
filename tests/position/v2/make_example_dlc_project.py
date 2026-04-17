"""Create a minimal DLC project structure for V2 integration tests.

The project is designed to exercise:
  - Model._get_latest_dlc_model_info() (fake trained-model layout)
  - Model.load() via DLC config.yaml
  - VidFileGroup.create_from_dlc_config() (session matching)
  - Model.get_training_history() (fake learning_stats.csv)

Usage as a script (filesystem only — no DB):
    python make_example_dlc_project.py [output_dir]

Usage as a module (filesystem only):
    from tests.position.v2.make_example_dlc_project import make_dlc_project
    config_path = make_dlc_project("/tmp/my_dlc_project")

Session bootstrap (requires Spyglass DB connection, run from pytest fixture):
    from tests.position.v2.make_example_dlc_project import bootstrap_dlc_session
    nwb_file_name = bootstrap_dlc_session(config_path)

The video paths in config.yaml contain ``_NWB_STEM`` so that
``create_from_dlc_config()`` matches a bootstrapped Spyglass session.
"""

import shutil
from pathlib import Path

import numpy as np
import yaml

# Test data directory
_DATA_DIR = Path(__file__).parent.parent.parent / "_data" / "deeplabcut"

# Body parts matching BodyPart.contents and tests/_data/deeplabcut/
_BODYPARTS = ["whiteLED", "tailBase", "tailMid", "tailTip"]
_SKELETON = [
    ["whiteLED", "tailBase"],
    ["tailBase", "tailMid"],
    ["tailMid", "tailTip"],
]

# NWB stem embedded in video paths for create_from_dlc_config() matching.
# Must NOT match the mini test file (minirec20230622_.nwb) to avoid
# conftest.mini_insert skipping insert_sessions() prematurely.
_NWB_STEM = "test_dlc_v2_sess_"

# DLC project identifiers
_TASK = "TESTv2"
_SCORER = "sc_eb"
_DATE = "Nov12"


def make_dlc_project(output_dir, overwrite=False):
    """Create a minimal DLC project directory for integration tests.

    Creates a self-contained project with config.yaml, a fake trained
    model (pose_cfg.yaml + learning_stats.csv), and labeled frames copied
    from ``tests/_data/deeplabcut/``.  No database connection required.

    Parameters
    ----------
    output_dir : str or Path
        Parent directory in which to create the project subdirectory.
    overwrite : bool, optional
        If True, remove and recreate an existing project. Default False.

    Returns
    -------
    Path
        Absolute path to the created ``config.yaml``.
    """
    output_dir = Path(output_dir)
    project_name = f"{_TASK}-{_SCORER}-{_DATE}"
    project_dir = output_dir / project_name
    config_path = project_dir / "config.yaml"

    if config_path.exists() and not overwrite:
        return config_path

    if project_dir.exists() and overwrite:
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    # ---- Video placeholder (path must contain _NWB_STEM for session match) ----
    video_dir = project_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    video_path = video_dir / f"{_NWB_STEM}.avi"
    video_path.touch()  # placeholder; content not needed for import tests

    # ---- config.yaml ----
    config = {
        "Task": _TASK,
        "scorer": _SCORER,
        "date": _DATE,
        "multianimalproject": False,
        "engine": "pytorch",
        "project_path": str(project_dir.resolve()),
        "video_sets": {
            str(video_path.resolve()): {"crop": "0, 640, 0, 480"},
        },
        "bodyparts": _BODYPARTS,
        "skeleton": _SKELETON,
        "TrainingFraction": [0.8],
        "iteration": 0,
        "default_net_type": "resnet_50",
        "snapshotindex": -1,
        "batch_size": 8,
    }
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    # ---- Fake trained model (iteration-0, shuffle 1) ----
    model_dir = (
        project_dir
        / "dlc-models"
        / "iteration-0"
        / f"{_TASK}{_DATE}-trainset80shuffle1"
        / "train"
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    # Minimal pose_cfg.yaml so _get_latest_dlc_model_info() finds this dir
    pose_cfg = {
        "project_path": str(project_dir.resolve()),
        "iteration": 0,
        "init_weights": "resnet_v1_50",
    }
    with open(model_dir / "pose_cfg.yaml", "w") as f:
        yaml.safe_dump(pose_cfg, f)

    # learning_stats.csv: one row per snapshot (iter, loss, lr)
    stats = np.array([[2, 1.035, 0.005], [4, 0.230, 0.005], [6, 0.058, 0.005]])
    np.savetxt(
        model_dir / "learning_stats.csv", stats, delimiter=",", fmt="%.5f"
    )

    # ---- Labeled frames (copy from shared test data if available) ----
    labeled_dir = project_dir / "labeled-data" / _NWB_STEM
    labeled_dir.mkdir(parents=True, exist_ok=True)
    for fname in [
        "img000.png",
        "img001.png",
        "CollectedData_sc_eb.csv",
        "CollectedData_sc_eb.h5",
    ]:
        src = _DATA_DIR / fname
        if src.exists():
            shutil.copy(src, labeled_dir / fname)

    return config_path.resolve()


def bootstrap_dlc_session(config_path):
    """Register minimal Spyglass DB entries so create_from_dlc_config() works.

    Inserts Nwbfile, Session, Task, IntervalList, TaskEpoch, and VideoFile
    rows derived from the DLC config's video paths. Mirrors the tutorial's
    ``_tutorial_bootstrap_dlc_session`` for use in pytest fixtures.

    Requires an active Spyglass database connection.

    Parameters
    ----------
    config_path : str or Path
        Path to the DLC ``config.yaml`` created by ``make_dlc_project()``.

    Returns
    -------
    str
        The ``nwb_file_name`` inserted into ``Nwbfile``.
    """
    import uuid
    from datetime import datetime

    from spyglass.common import (
        IntervalList,
        Nwbfile,
        Session,
        Task,
        TaskEpoch,
        VideoFile,
    )
    from spyglass.settings import raw_dir

    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    nwb_file_name = f"{_NWB_STEM}.nwb"
    nwb_file_path = Path(raw_dir) / nwb_file_name
    nwb_file_path.touch()  # satisfy Nwbfile path constraints

    video_paths = list(cfg.get("video_sets", {}).keys())
    now = datetime.now()

    Nwbfile().insert1(
        {
            "nwb_file_name": nwb_file_name,
            "nwb_file_abs_path": str(nwb_file_path.resolve()),
        },
        skip_duplicates=True,
    )
    Session().insert1(
        {
            "nwb_file_name": nwb_file_name,
            "session_description": "DLC integration test session",
            "session_start_time": now,
            "timestamps_reference_time": now,
        },
        allow_direct_insert=True,
        skip_duplicates=True,
    )
    Task().insert1({"task_name": "test_dlc_v2"}, skip_duplicates=True)
    IntervalList().insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "test_epoch_1",
            "valid_times": np.array([[0.0, 1.0]]),
        },
        skip_duplicates=True,
    )
    TaskEpoch().insert1(
        {
            "nwb_file_name": nwb_file_name,
            "epoch": 1,
            "task_name": "test_dlc_v2",
            "interval_list_name": "test_epoch_1",
            "camera_names": [],
        },
        allow_direct_insert=True,
        skip_duplicates=True,
    )
    for i, vp in enumerate(video_paths):
        VideoFile().insert1(
            {
                "nwb_file_name": nwb_file_name,
                "epoch": 1,
                "video_file_num": i,
                "camera_name": "test_dlc_v2",
                "video_file_object_id": str(uuid.uuid4())[:40],
                "path": str(Path(vp).resolve()),
            },
            allow_direct_insert=True,
            skip_duplicates=True,
        )

    return nwb_file_name


if __name__ == "__main__":
    import sys

    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("/tmp/dlc_v2_test_project")
    )
    config = make_dlc_project(out, overwrite=True)
    print(f"DLC project created: {config}")
    print(
        "To register a Spyglass session, call "
        "bootstrap_dlc_session(config_path) from a Python session with DB access."
    )
