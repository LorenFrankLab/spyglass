"""Create a minimal DLC project structure for V2 integration tests.

The project is designed to exercise:
  - DLCStrategy.get_latest_model_info() (fake trained-model layout)
  - Model.load() via DLC config.yaml
  - VidFileGroup.create_from_dlc_config() (session matching)
  - Model.get_training_history() (fake learning_stats.csv)

Usage as a script (filesystem only — no DB):
    python make_example_dlc_project.py [output_dir]

Usage as a module (filesystem only):
    from tests.position.v2.make_example_dlc_project import make_dlc_project
    config_path = make_dlc_project("/tmp/my_dlc_project")

Session bootstrap from an existing DLC config (requires Spyglass DB connection):
    from tests.position.v2.make_example_dlc_project import bootstrap_dlc_session
    nwb_file_name = bootstrap_dlc_session(config_path)

Session bootstrap from raw video paths — no config.yaml required (tutorial use):
    from tests.position.v2.make_example_dlc_project import *
    nwb_file_name, inf_vid_path = bootstrap_from_video_paths(video_paths)

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

    # ---- Video placeholder ----
    video_dir = project_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    video_path = video_dir / f"{_NWB_STEM}.avi"
    # Create a minimal real AVI so DLC's create_new_project() can open it.
    if not video_path.exists() or video_path.stat().st_size == 0:
        import subprocess

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:size=4x4:rate=5",
                "-t",
                "10",
                str(video_path),
            ],
            check=True,
            capture_output=True,
        )

    # ---- config.yaml ----
    config_dict = {
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
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config_dict, f, default_flow_style=False, sort_keys=False
        )

    # ---- Fake trained model (iteration-0, shuffle 1) ----
    model_dir = (
        project_dir
        / "dlc-models"
        / "iteration-0"
        / f"{_TASK}{_DATE}-trainset80shuffle1"
        / "train"
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    # Minimal pose_cfg.yaml so get_latest_model_info() finds this dir
    pose_cfg = {
        "project_path": str(project_dir.resolve()),
        "iteration": 0,
        "init_weights": "resnet_v1_50",
    }
    with open(model_dir / "pose_cfg.yaml", "w", encoding="utf-8") as f:
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
    with open(config_path, encoding="utf-8") as f:
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
        pk = {"nwb_file_name": nwb_file_name, "epoch": 1, "video_file_num": i}
        fresh_path = str(Path(vp).resolve())
        if VideoFile & pk:
            # Row exists; just refresh the path
            VideoFile.update1({**pk, "path": fresh_path})
        else:
            VideoFile().insert1(
                {
                    **pk,
                    "camera_name": "test_dlc_v2",
                    "video_file_object_id": str(uuid.uuid4())[:40],
                    "path": fresh_path,
                },
                allow_direct_insert=True,
            )

    return nwb_file_name


def bootstrap_from_video_paths(
    video_paths,
    nwb_stem=None,
    task_name="tutorial_dlc",
    camera_name="tutorial_dlc",
):
    """Register minimal Spyglass DB entries from a list of video paths.

    Unlike ``bootstrap_dlc_session()``, this function does not require a DLC
    ``config.yaml`` to already exist — pass raw video paths directly (e.g.
    from a DLC project's ``video_sets``, or videos you intend to train on).

    A 1-second inference clip is created from the first video in
    *video_paths* (via ffmpeg) and registered as an additional
    ``VideoFile`` entry.  The NWB placeholder is a copy of
    ``minirec20230622_.nwb`` so that ``AnalysisNwbfile.create()`` can open
    it when storing pose results.

    For tutorial / development use only.  In production, register sessions
    with ``insert_sessions()`` before calling ``Model.create_project()`` or
    ``Model.load()``.

    If you use this on a shared database, **please delete the entries** when
    you are done.

    Parameters
    ----------
    video_paths : list of str or Path
        Absolute paths to training videos.
    nwb_stem : str, optional
        Stem for the NWB filename (e.g. ``"my_session"``).  If None,
        derived from the parent directory name of the first video.
    task_name : str, optional
        Task name for ``Task`` / ``TaskEpoch`` rows.
        Default ``"tutorial_dlc"``.
    camera_name : str, optional
        Camera name for ``TaskEpoch`` / ``VideoFile`` rows.
        Default ``"tutorial_dlc"``.

    Returns
    -------
    tuple[str, Path]
        ``(nwb_file_name, inf_vid_path)`` — the registered NWB filename and
        path to the 1-second inference clip written into ``video_dir``.
    """
    import subprocess
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

    video_paths = [str(Path(vp).resolve()) for vp in video_paths]
    if not video_paths:
        raise ValueError("video_paths must not be empty")

    # Derive NWB stem from the first video's grandparent directory (project
    # folder) if the caller did not supply one.
    if nwb_stem is None:
        nwb_stem = Path(video_paths[0]).parent.parent.name

    nwb_file_name = f"{nwb_stem}_.nwb"
    nwb_file_path = Path(raw_dir) / nwb_file_name

    # Copy minirec as a valid NWB placeholder so AnalysisNwbfile.create()
    # can open it when storing pose estimation results.
    if not nwb_file_path.exists() or nwb_file_path.stat().st_size == 0:
        minirec = Path(raw_dir) / "minirec20230622_.nwb"
        if minirec.exists():
            shutil.copy2(str(minirec), str(nwb_file_path))
        else:
            nwb_file_path.touch()  # fallback for environments without minirec

    # Create a 1-second inference clip in video_dir.  Only runs ffmpeg when
    # the source is a real (non-empty) video file.
    src_video = Path(video_paths[0])
    inf_vid_name = f"example_inference{src_video.suffix}"
    inf_vid_path = src_video.parent / inf_vid_name
    if not inf_vid_path.exists():
        if src_video.exists() and src_video.stat().st_size > 0:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    str(src_video),
                    "-t",
                    "1",
                    "-c",
                    "copy",
                    str(inf_vid_path),
                ],
                check=True,
                capture_output=True,
            )
        else:
            inf_vid_path.touch()  # placeholder for non-real/test videos

    all_video_paths = video_paths + [str(inf_vid_path)]

    nwb_key = dict(nwb_file_name=nwb_file_name)
    interval_list_name = f"{task_name}_epoch_1"
    interval_key = dict(nwb_key, interval_list_name=interval_list_name)
    now = datetime.now()
    ins = dict(allow_direct_insert=True, skip_duplicates=True)

    if not Nwbfile() & nwb_key:
        Nwbfile().insert1(
            {**nwb_key, "nwb_file_abs_path": str(nwb_file_path.resolve())},
            allow_direct_insert=True,
        )
    Session().insert1(
        {
            **nwb_key,
            "session_description": f"Tutorial dummy: {nwb_stem}",
            "session_start_time": now,
            "timestamps_reference_time": now,
        },
        **ins,
    )
    Task().insert1({"task_name": task_name}, **ins)
    IntervalList().insert1(
        {**interval_key, "valid_times": np.array([[0.0, 1.0]])}, **ins
    )
    TaskEpoch().insert1(
        {
            **interval_key,
            "epoch": 1,
            "task_name": task_name,
            "camera_names": [],
        },
        **ins,
    )
    for i, video_path in enumerate(all_video_paths):
        VideoFile().insert1(
            {
                **nwb_key,
                "epoch": 1,
                "video_file_num": i,
                "camera_name": camera_name,
                "video_file_object_id": str(uuid.uuid4())[:40],
                "path": str(Path(video_path).resolve()),
            },
            **ins,
        )

    print(
        f"Tutorial session ready: {nwb_file_name}"
        f" ({len(all_video_paths)} video(s))"
    )
    return nwb_file_name, inf_vid_path


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
        "bootstrap_dlc_session(config_path) from a Python session"
    )
