# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: pv2
#     language: python
#     name: pv2
# ---

# %% [markdown]
# ## Position Pipeline V2 — 3D Multi-Camera Calibration
#

# %% [markdown]
# ## Overview

# %% [markdown]
# ### Notes

# %% [markdown]
# This is one notebook in a multi-part series on Spyglass.
#
# <details><summary>For contributors</summary>
#
# If you may make a PR in the future, be sure to copy this notebook, and use
# the `gitignore` prefix `temp` to avoid future conflicts.
#
# </details>
#
# - To set up your Spyglass environment and database, see
#   [the Setup notebook](./00_Setup.ipynb)
# - For the single-camera (2D) Position V2 DLC workflow — training, model
#   import, pose processing — see
#   [the Position V2 DLC notebook](./23_PositionV2_DLC_2D.ipynb)
#
# Most pose pipelines track an animal in a single camera's image plane (2D).
# Given two or more **calibrated, synchronized** cameras of the same scene,
# Spyglass V2 can **triangulate** the 2D detections into real-world **3D**
# coordinates. This notebook covers the extra machinery for that:
#
# - **Generate** a camera-rig calibration (intrinsics + extrinsics) with an
#   external tool, and **load** it into the V2 `CameraRig` / `Calibration`
#   tables.
# - Group the per-camera videos in a `VidFileGroup`, tagging each with a
#   `camera_index` and **pairing the group with its calibration**.
# - Run `PoseEstim` in **3D mode**, which triangulates the per-camera 2D pose
#   and stores it in NWB.
# - Fetch and visualize the 3D trajectory.
#
# **Example data.** We use the two-camera mouse-reaching dataset from the
# [Anipose paper](https://doi.org/10.5061/dryad.nzs7h44s4)
# (Karashchuk et al. 2021). It ships a real camera calibration, per-camera 2D
# detections, the raw videos, and Anipose's own 3D output — so we can load a
# real calibration and check our triangulation against a reference.
#
# This notebook is **self-contained**: it downloads the example data and creates
# its own (dummy) upstream session/model entries so it can run start to finish.

# %% [markdown]
# ### Table of Contents
#
# - [Setup](#Setup) — environment + database connection
# - [The 3D tables](#Tables) — how calibration plugs into V2
# - [Get example data](#Data) — download/extract the Anipose subset
# - [Generate a calibration](#Generate) — produce a `calibration.toml`
# - [Load the calibration](#Load) — into `CameraRig` / `Calibration`
# - [Register videos](#Videos) — clip with DLC, pair with the calibration
# - [3D pose estimation](#Pose) — triangulate via `PoseEstim`
# - [Visualize](#Viz) — fetch and plot the 3D trajectory
# - [Video](#Video) — reproject the 3D pose onto the camera videos
# - [Cleanup](#Cleanup) — remove the tutorial entries

# %% [markdown]
# ## Setup <a id="Setup"></a>

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import shutil
import sys
import uuid
import warnings
from datetime import datetime
from pathlib import Path

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ignore datajoint+jupyter async warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# Suppress noisy TF/ABSL C++ logs. Must be set before TensorFlow is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# DataJoint resolves your connection config when the first connection is made —
# from `dj_local_conf.json` in the working directory, `~/.datajoint_config.json`,
# or `DJ_*` environment variables. See the Setup notebook (00_Setup) if you have
# not configured one yet.
print(dj.conn())

# %%
from spyglass.common import (
    IntervalList,
    Nwbfile,
    Session,
    Task,
    TaskEpoch,
    VideoFile,
)
from spyglass.common.common_device import CameraDevice
from spyglass.position.v2 import (
    Calibration,
    CameraRig,
    Model,
    ModelParams,
    ModelSelection,
    PoseEstim,
    PoseEstimParams,
    PoseEstimSelection,
    Skeleton,
    VidFileGroup,
    estim,
    train,
    video,
)
from spyglass.settings import raw_dir

print("All imports successful!")

# %%
# This notebook needs DeepLabCut (video clipping + inference), OpenCV, and a
# TOML reader. Check now and fail fast rather than deep inside the pipeline.
missing_packages = []
for module_name, install_hint in [
    ("deeplabcut", "install deeplabcut in this env"),
    ("cv2", "pip install opencv-python"),
]:
    try:
        __import__(module_name)
    except ImportError:
        missing_packages.append(f"  - {module_name} ({install_hint})")
try:
    import tomllib  # noqa: F401  (py3.11+)
except ModuleNotFoundError:
    try:
        import tomli  # noqa: F401
    except ModuleNotFoundError:
        missing_packages.append("  - tomli (pip install tomli; py<3.11 only)")
if missing_packages:
    raise ImportError(
        "Missing packages required by this notebook:\n"
        + "\n".join(missing_packages)
    )
print("Requirements OK — DeepLabCut, OpenCV, and TOML are available.")

# %% [markdown]
# ## The 3D tables <a id="Tables"></a>
#
# The 3D path reuses the V2 pipeline and adds a small set of calibration tables:
#
# - **`CameraDevice`** (common) — one row per physical camera (the source of
#   truth for camera identity).
# - **`CameraRig`** + **`CameraRig.Camera`** — a rig groups cameras; each slot
#   maps a zero-based `camera_index` to a `CameraDevice`.
# - **`Calibration`** + **`Calibration.Camera`** — per-camera `intrinsics`
#   (`fx, fy, cx, cy, dist_coeffs`) and `extrinsics` (`R`, `t` camera→rig).
# - **`VidFileGroup`** — groups the per-camera videos. Each `VidFileGroup.File`
#   carries a `camera_index`, and `VidFileGroup.Calibration` links the group to
#   a `Calibration`.
#
# When `PoseEstim` sees a video group with **≥2 cameras and a linked
# calibration**, it runs the 3D path: combining each camera's 2D pose into a
# single triangulated 3D pose, stored in NWB.

# %%
# The video + calibration tables, then training, estimation, and the pipeline.
# Passing the schema modules (video / train / estim) makes the diagram label
# each table with its schema-module prefix, showing where it is defined.
dj.Diagram(video) + dj.Diagram(train) + dj.Diagram(estim)

# %% [markdown]
# ## Get example data <a id="Data"></a>
#
# The Anipose dataset is a single ~942 MB zip on Zenodo, and using it takes a
# few dataset-specific steps: download once, extract only the members one
# reaching trial needs, and reshape Anipose's multi-candidate 2D detections
# into the tidy DLC-style `.h5` the pipeline loads. None of that is
# Spyglass-specific, so it lives in a support helper — `fetch_anipose_example` —
# that returns the resolved paths this notebook consumes. **Everything after
# this cell is a user-facing step you would swap for your own data.**
#
# > `fetch_anipose_example(dest_dir=...)` (or the `ANIPOSE_DIR` environment
# > variable) chooses where the data lives; default is
# > `~/spyglass_data/anipose`. Point it at an existing copy to skip the
# > download. The helper ships in the Spyglass source checkout under
# > `tests/position/v2/`, alongside the other tutorial bootstrap helpers.

# %%
import spyglass  # noqa: E402

tests_v2_dir = Path(spyglass.__file__).parents[2] / "tests" / "position" / "v2"
if not tests_v2_dir.exists():
    raise FileNotFoundError(
        f"Tutorial helper directory not found: {tests_v2_dir}\n"
        "This tutorial's example-data helper lives in the Spyglass source "
        "checkout, which is not included in the pip package. Clone the repo "
        "(https://github.com/LorenFrankLab/spyglass) and run from there, or "
        "substitute your own calibration, videos, and 2D detections."
    )
if str(tests_v2_dir) not in sys.path:
    sys.path.insert(0, str(tests_v2_dir))

from anipose_example import fetch_anipose_example  # noqa: E402

example = fetch_anipose_example()
session_dir = example["session_dir"]
print("Session dir:", session_dir)

# %% [markdown]
# ## Generate a calibration <a id="Generate"></a>
#
# A 3D calibration has two parts per camera:
#
# - **Intrinsics** — a single camera's optics: focal lengths, principal point,
#   lens distortion.
# - **Extrinsics** — how the cameras sit relative to one another: the rotation
#   `R` and translation `t` placing each camera in a shared rig coordinate frame.
#
# **You generate these with a calibration tool, not in Spyglass** — record a
# known calibration board (checkerboard / ChArUco) moving through the shared
# field of view, then run a calibration routine. Two common options:
#
# - **DeepLabCut 3D** — see the
#   [DLC 3D overview](https://deeplabcut.github.io/DeepLabCut/docs/Overviewof3D.html)
#   (`deeplabcut.calibrate_cameras`).
# - **Anipose** — see the
#   [Anipose tutorial](https://anipose.readthedocs.io/en/latest/tutorial.html);
#   it writes a `calibration.toml` (the format our example dataset ships).
#
# Below we **load the calibration that ships with the Anipose example** and
# translate it into the dicts the `Calibration` table expects. The two tools use
# slightly different conventions and units, which the helper below converts.
#
# <details><summary>Conversion details</summary>
#
# Anipose stores each camera's orientation as a compact "rotation vector" plus a
# translation in millimetres, oriented world→camera. The V2 tables store the
# rotation matrix `R` and translation `t` oriented camera→rig, in metres, so the
# helper inverts the transform and rescales.
#
# </details>

# %%
try:
    import tomllib as toml_lib  # py3.11+
except ModuleNotFoundError:
    import tomli as toml_lib  # py3.10

import cv2


def anipose_calibration_to_v2(toml_path):
    """Parse an Anipose calibration.toml into V2 calibration dicts.

    Returns ``{camera_index: {"name", "image_size", "intrinsics",
    "extrinsics"}}`` with extrinsic translations in metres.
    """
    with open(toml_path, "rb") as fh:
        raw = toml_lib.load(fh)

    cams = {}
    for key, cam in raw.items():
        if not key.startswith("cam_"):
            continue
        idx = int(key.split("_")[1])
        K = np.asarray(cam["matrix"], dtype=float)
        rvec = np.asarray(cam["rotation"], dtype=float).reshape(3, 1)
        tvec_mm = np.asarray(cam["translation"], dtype=float)
        R_wc, _ = cv2.Rodrigues(rvec)  # world(rig) -> camera
        cams[idx] = {
            "name": cam.get("name", key),
            "image_size": cam.get("size", [1024, 768]),
            "intrinsics": {
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
                "dist_coeffs": list(cam["distortions"]),
            },
            "extrinsics": {
                "R": R_wc.T.tolist(),  # camera -> rig
                "t": ((-R_wc.T @ tvec_mm) / 1000.0).tolist(),  # mm -> metres
            },
        }
    return dict(sorted(cams.items()))


calib = anipose_calibration_to_v2(example["calibration_toml"])
assert len(calib) >= 2, (
    f"3D triangulation needs at least 2 calibrated cameras; the calibration "
    f"parsed only {len(calib)}. Check that calibration.toml has multiple "
    f"`cam_*` sections."
)
for ci, cam in calib.items():
    t = np.asarray(cam["extrinsics"]["t"])
    print(
        f"camera_index {ci}  name={cam['name']}  "
        f"fx={cam['intrinsics']['fx']:.1f}  |t|={np.linalg.norm(t) * 1000:.1f} mm"
    )

# %% [markdown]
# ## Load the calibration <a id="Load"></a>
#
# Each table depends on the one before it, so we insert in order:
# `CameraDevice → CameraRig → CameraRig.Camera → Calibration →
# Calibration.Camera`. Every tutorial row is prefixed with `demo` for easy
# removal later (see [Cleanup](#Cleanup)).

# %%
demo = "ap3d_nb"  # identifier prefix for every row this notebook creates
rig_id = f"{demo}_rig"
cal_id = f"{demo}_cal"
cal_key = {"camera_rig_id": rig_id, "calibration_id": cal_id}
today = datetime.now().date().isoformat()

# 1. One CameraDevice per physical camera (source of truth for identity).
for ci, cam in calib.items():
    CameraDevice.insert1(
        {"camera_name": f"{demo}_{cam['name']}", "meters_per_pixel": 0.0},
        skip_duplicates=True,
    )

# 2. The rig + one slot per camera, mapping camera_index -> CameraDevice.
CameraRig.insert1(
    {
        "camera_rig_id": rig_id,
        "description": "Anipose mouse two-camera rig (tutorial)",
        "n_cameras": len(calib),
    },
    skip_duplicates=True,
)
for ci, cam in calib.items():
    CameraRig.Camera.insert1(
        {
            "camera_rig_id": rig_id,
            "camera_index": ci,
            "camera_name": f"{demo}_{cam['name']}",
        },
        skip_duplicates=True,
    )

# 3. The calibration header + per-camera intrinsics/extrinsics.
Calibration.insert1(
    {
        **cal_key,
        "calibration_date": today,
        "notes": "Loaded from Anipose calibration.toml (tutorial)",
    },
    skip_duplicates=True,
)
for ci, cam in calib.items():
    Calibration.Camera.insert1(
        {
            **cal_key,
            "camera_index": ci,
            "intrinsics": cam["intrinsics"],
            "extrinsics": cam["extrinsics"],
            "image_size": list(cam["image_size"]),
        },
        skip_duplicates=True,
    )

# Read it back to confirm the round-trip.
(Calibration.Camera & cal_key)

# %% [markdown]
# ## Register videos <a id="Videos"></a>
#
# Next we register the per-camera videos and group them. Two ideas matter here:
#
# 1. **`camera_index` must agree with the calibration.** The video at
#    `camera_index = i` is triangulated using the calibration at
#    `camera_index = i`. We pair them **by camera name**: each index's
#    calibration entry carries a name token (`cam1`/`cam2`) that must appear in
#    the matching video's filename.
# 2. **The video group is linked to its calibration** via
#    `VidFileGroup.Calibration` — the link that flips `PoseEstim` into 3D mode.
#
# We use Spyglass's dependency-light `ffmpeg_clip` helper to clip a short demo
# segment from each raw reach video by time. It shells out to `ffmpeg` directly
# (no DeepLabCut needed) and writes `<stem>short.mp4` into the destination.

# %%
from deeplabcut.utils.auxfun_videos import VideoReader

from spyglass.position.utils.make_video import ffmpeg_clip

clips_dir = session_dir / "tutorial_clips"
clips_dir.mkdir(exist_ok=True)

# Clip one short video per camera, paired to its calibrated camera_index.
clip_paths = {}
for ci, cam in calib.items():
    token = cam["name"]
    src = example["videos"][token]
    clip = ffmpeg_clip(src, clips_dir)
    assert (
        token in clip.name
    ), f"clip {clip.name} not paired with calib '{token}'"
    clip_paths[ci] = clip
    print(f"camera_index {ci} ({token}): {clip.name}")

# Frame count of the lowest-index camera's clip drives the shared timeline.
n_frames = VideoReader(str(clip_paths[min(clip_paths)])).get_n_frames()
print(f"\nShared frame count: {n_frames}")

# %% [markdown]
# #### Register a session and the videos
#
# `PoseEstim` stores results in an NWB analysis file derived from the session
# NWB the videos belong to. In production you would register your real recording
# with `insert_sessions()`. For this tutorial we create a minimal dummy session
# (a copy of the bundled `minirec` NWB) and register each clip as a `VideoFile`.

# %%
nwb_file_name = f"{demo}_.nwb"
nwb_path = Path(raw_dir) / nwb_file_name
minirec_src = Path(raw_dir) / "minirec20230622_.nwb"
if not nwb_path.exists():
    if not minirec_src.exists():
        raise FileNotFoundError(
            f"Bundled example NWB not found at {minirec_src}. It ships with the "
            "Setup notebook (00_Setup) test data — run that first, or point "
            "raw_dir at a directory that contains it."
        )
    shutil.copy2(str(minirec_src), str(nwb_path))

ins = dict(allow_direct_insert=True, skip_duplicates=True)
now = datetime.now()

if not (Nwbfile() & {"nwb_file_name": nwb_file_name}):
    Nwbfile().insert1(
        {"nwb_file_name": nwb_file_name, "nwb_file_abs_path": str(nwb_path)},
        allow_direct_insert=True,
    )
Session().insert1(
    {
        "nwb_file_name": nwb_file_name,
        "session_description": "3D calibration tutorial",
        "session_start_time": now,
        "timestamps_reference_time": now,
    },
    **ins,
)
Task().insert1({"task_name": f"{demo}_task"}, **ins)
IntervalList().insert1(
    {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": f"{demo}_epoch_1",
        "valid_times": np.array([[0.0, 1.0]]),
    },
    **ins,
)
TaskEpoch().insert1(
    {
        "nwb_file_name": nwb_file_name,
        "epoch": 1,
        "task_name": f"{demo}_task",
        "interval_list_name": f"{demo}_epoch_1",
        "camera_names": [],
    },
    **ins,
)

vf_keys = []
for ci, clip in clip_paths.items():
    vf_pk = {"nwb_file_name": nwb_file_name, "epoch": 1, "video_file_num": ci}
    if not (VideoFile & vf_pk):
        VideoFile().insert1(
            {
                **vf_pk,
                "camera_name": f"{demo}_{calib[ci]['name']}",
                "video_file_object_id": str(uuid.uuid4())[:40],
                "path": str(clip.resolve()),
            },
            allow_direct_insert=True,
        )
    vf_keys.append(vf_pk)

print(f"Registered {len(vf_keys)} VideoFile rows for {nwb_file_name}")

# %% [markdown]
# #### Group the videos and link the calibration
#
# `camera_indices` aligns each video with its calibrated camera slot, and the
# `VidFileGroup.Calibration` insert is the link that enables 3D triangulation.

# %%
vid_group = f"{demo}_grp"
VidFileGroup().insert1(
    {
        "vid_group_id": vid_group,
        "description": "Anipose two-camera reach (tutorial)",
        "files": vf_keys,
        "camera_indices": list(clip_paths.keys()),
    }
)
VidFileGroup.Calibration().insert1(
    {"vid_group_id": vid_group, **cal_key},
    skip_duplicates=True,
)

VidFileGroup.File & {"vid_group_id": vid_group}

# %% [markdown]
# ## 3D pose estimation <a id="Pose"></a>
#
# In a full workflow you would have a trained DLC model and either run inference
# per camera (`task_mode='trigger'`) or load existing DLC `.h5` output
# (`task_mode='load'`). The Anipose dataset already ships per-camera 2D
# detections, so we use **`task_mode='load'`**.
#
# Two preparation steps:
#
# 1. **Place the 2D detections where `PoseEstim` looks.** The helper already
#    reshaped Anipose's multi-candidate detections into clean DLC-style `.h5`;
#    here we only truncate each to its short demo clip's frame count and name it
#    `{clip_stem}DLC_*.h5`. **With your own data**, drop your model's per-camera
#    DLC output into `output_dir` under that name instead.
# 2. **Register a model.** We're loading detections rather than training, so we
#    register a lightweight placeholder `Skeleton`/`ModelParams`/`Model`. (With
#    your own data, import a real model as in the
#    [DLC notebook](./23_PositionV2_DLC_2D.ipynb) — where you can also continue
#    training an existing model via `Model().train({'model_id': ...}, epochs=N)`.)

# %%
# The body parts + skeleton edges this dataset ships. Swap these two lists for
# your own model's keypoints when adapting the notebook.
bodyparts = example["bodyparts"]
edges = example["edges"]
output_dir = session_dir / "tutorial_dlc_outputs"
output_dir.mkdir(exist_ok=True)

# Truncate each camera's cleaned detections to the clip's frame count and name
# it to match, so PoseEstim (task_mode='load') finds it next to the clip.
for ci, clip in clip_paths.items():
    token = calib[ci]["name"]
    clean = pd.read_hdf(example["pose_2d"][token]).iloc[:n_frames]
    out = output_dir / f"{clip.stem}DLC_resnet50_mouse.h5"
    clean.to_hdf(str(out), key="df_with_missing", mode="w")
print(f"Wrote cleaned DLC h5 files to {output_dir}")

# %%
# Register the placeholder skeleton + model. The Anipose keypoints are already
# canonical entries in the BodyPart reference table, so no extra flag is needed.
skeleton_id = f"{demo}_skel"
if not (Skeleton() & {"skeleton_id": skeleton_id}):
    Skeleton().insert1(
        {"skeleton_id": skeleton_id, "bodyparts": bodyparts, "edges": edges},
    )

mp = ModelParams().insert1(
    {
        "model_params_id": f"{demo}_mp",
        "tool": "DLC",
        "params": {
            "project_path": str(output_dir),
            "shuffle": 1,
            "trainingsetindex": 0,
        },
        "skeleton_id": skeleton_id,
    },
    skip_duplicates=True,
)
sel_key = {
    "model_params_id": mp["model_params_id"],
    "tool": "DLC",
    "vid_group_id": vid_group,
    "model_selection_id": f"{demo}_sel",
}
ModelSelection().insert1(sel_key, skip_duplicates=True)
Model().insert1(
    {
        **sel_key,
        "model_id": f"{demo}_model",
        "model_path": "anipose-mouse (tutorial placeholder)",
    },
    allow_direct_insert=True,
    skip_duplicates=True,
)
print("Placeholder model registered.")

# %% [markdown]
# #### Run triangulation
#
# `PoseEstimParams` carries the triangulation thresholds (here matched to the
# Anipose project: confidence ≥ 0.3, reprojection error ≤ 5 px). Because the
# video group has two cameras **and** a linked calibration,
# `PoseEstim.populate()` automatically takes the 3D path.

# %%
params_key = PoseEstimParams.insert_params(
    {"min_confidence": 0.3, "max_reproj_error": 5.0},
    params_id=f"{demo}_pep",
    skip_duplicates=True,
)
estim_key = PoseEstimSelection().insert_estimation_task(
    {
        "model_id": f"{demo}_model",
        "vid_group_id": vid_group,
        "pose_estim_params_id": params_key["pose_estim_params_id"],
    },
    task_mode="load",
    output_dir=str(output_dir),
    skip_duplicates=True,
)
estim_key = {
    k: estim_key[k]
    for k in ("model_id", "vid_group_id", "pose_estim_params_id")
}

PoseEstim.populate(estim_key, display_progress=True)
print("PoseEstim populated:", bool(PoseEstim & estim_key))

# %% [markdown]
# ## Visualize <a id="Viz"></a>
#
# `fetch1_dataframe()` returns a table indexed by time, with `x`/`y`/`z`/
# `likelihood` columns per body part (coordinates in centimetres). Columns are
# grouped in layers (source, body part, coordinate), so you can pull a single
# body part's `x`/`y`/`z` at once.

# %%
df = (PoseEstim & estim_key).fetch1_dataframe()
has_z = any(c[-1] == "z" for c in df.columns)
if not has_z:
    raise RuntimeError(
        "3D estimation produced no `z` columns — the pipeline fell back to the "
        "2D path. Confirm the video group has >=2 cameras and a linked "
        "Calibration (VidFileGroup.Calibration)."
    )
print("Confirmed 3D output: every body part has a `z` column.")
df.head()

# %%
# 3D trajectory of a well-tracked body part.
bp = "r-middle"
xyz = df["triangulated"][bp][["x", "y", "z"]].to_numpy()
valid = ~np.isnan(xyz[:, 0])

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(
    xyz[valid, 0],
    xyz[valid, 1],
    xyz[valid, 2],
    c=np.arange(valid.sum()),
    cmap="viridis",
    s=6,
)
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_zlabel("Z (cm)")
ax.set_title(f"3D trajectory — {bp} (colored by frame)")
fig.colorbar(sc, label="frame", shrink=0.6)
plt.show()

# %% [markdown]
# #### Optional: compare to Anipose's reference 3D
#
# The dataset ships Anipose's own triangulation. The V2 path uses the same
# standard triangulation math, so the two should agree to within rounding
# (Anipose stores millimetres; V2 stores centimetres, so we rescale V2 by 10).

# %%
ref = pd.read_csv(example["reference_3d"]).iloc[:n_frames]
diffs = []
for b in bodyparts:
    v2 = df["triangulated"][b][["x", "y", "z"]].to_numpy() * 10.0  # cm -> mm
    r = ref[[f"{b}_x", f"{b}_y", f"{b}_z"]].to_numpy()
    m = ~np.isnan(v2[:, 0]) & ~np.isnan(r[:, 0])
    if m.any():
        diffs.append(np.linalg.norm(v2[m] - r[m], axis=1))
alld = np.concatenate(diffs)
print(
    f"V2 vs Anipose 3D: {alld.size} points, "
    f"median {np.median(alld):.4f} mm, p95 {np.percentile(alld, 95):.4f} mm"
)

# %% [markdown]
# ## Video: reproject the 3D pose onto the cameras <a id="Video"></a>
#
# Plots confirm the numbers; a video confirms the *result*. The most direct
# validation is to **reproject** the triangulated 3D points back into each
# camera image (through the same calibration) and overlay them on the real
# video. If the reprojected markers stay on the animal, both the 3D
# reconstruction and the calibration are sound — closing the 2D → 3D → 2D loop.
#
# We render with DeepLabCut's built-in `create_video`, feeding it a small h5 of
# reprojected points. DLC's native (OpenCV-backed) writer is faster and lighter
# than a per-frame matplotlib renderer.

# %%
from deeplabcut.utils.make_labeled_video import create_video

from spyglass.position.v2.utils.triangulation import reproject_pose_to_camera

labeled_dir = session_dir / "tutorial_labeled"
labeled_dir.mkdir(exist_ok=True)

# ``reproject_pose_to_camera`` maps rig-frame metres to pixels, so the 3D
# coordinates (in centimetres) are rescaled with ``scale=100.0``. Marker
# confidence carries over from the triangulation likelihood, and the DLC-style
# h5 (written when ``out_h5`` is given) feeds DLC's ``create_video``.
labeled_videos = {}
for ci, clip in clip_paths.items():
    reproj_h5 = labeled_dir / f"{clip.stem}_reproj.h5"
    reproject_pose_to_camera(
        df, calib[ci], bodyparts=bodyparts, scale=100.0, out_h5=reproj_h5
    )
    out_mp4 = labeled_dir / f"{clip.stem}_reproj3d.mp4"
    out_mp4.unlink(missing_ok=True)
    create_video(
        str(clip),
        str(reproj_h5),
        pcutoff=0.5,
        dotsize=7,
        skeleton_edges=edges,
        output_path=str(out_mp4),
    )
    labeled_videos[ci] = out_mp4
    print(f"camera_index {ci}: {out_mp4.name}")

# %% [markdown]
# Show one overlaid frame inline so the reprojected 3D pose is visible without
# opening the video file:

# %%
cap = cv2.VideoCapture(str(labeled_videos[min(labeled_videos)]))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.5))
ok, frame = cap.read()
cap.release()
if ok:
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Reprojected 3D pose overlaid on camera 0 (mid-clip frame)")
    plt.axis("off")
    plt.show()

# %% [markdown]
# ## Cleanup <a id="Cleanup"></a>
#
# This tutorial wrote rows to a shared database. Set `CLEANUP = True` and run
# the cell below to remove everything it created (the `demo`-prefixed entries).

# %%
CLEANUP = False  # set True to delete all tutorial rows

if CLEANUP:
    # DataJoint deletes cascade to every downstream/part table, so removing a
    # handful of roots clears everything this notebook created:
    #   Nwbfile      -> Session, IntervalList, TaskEpoch, VideoFile,
    #                   AnalysisNwbfile -> PoseEstim
    #   VidFileGroup -> ModelSelection -> Model -> PoseEstimSelection -> PoseEstim
    #   Skeleton     -> ModelParams (-> the Model/PoseEstim chain)
    #   CameraRig    -> CameraRig.Camera, Calibration(.Camera)
    safe = dict(safemode=False)
    (Nwbfile & {"nwb_file_name": nwb_file_name}).delete(**safe)
    (VidFileGroup & {"vid_group_id": vid_group}).delete(**safe)
    (Skeleton & {"skeleton_id": skeleton_id}).delete(**safe)
    (PoseEstimParams & {"pose_estim_params_id": f"{demo}_pep"}).delete(**safe)
    (CameraRig & {"camera_rig_id": rig_id}).delete(**safe)
    (Task & {"task_name": f"{demo}_task"}).delete(**safe)
    for ci, cam in calib.items():
        (CameraDevice & {"camera_name": f"{demo}_{cam['name']}"}).delete(**safe)
    print("Tutorial rows removed.")
else:
    print("CLEANUP is False — tutorial rows left in place.")

# %% [markdown]
# ### What's next
#
# - For per-bodypart smoothing, orientation, centroid, and velocity on the
#   triangulated pose, continue to `PoseV2` (see the
#   [DLC notebook](./23_PositionV2_DLC_2D.ipynb) — it reads 3D input transparently).
# - With your own rig: generate a `calibration.toml` (DLC 3D or Anipose), then
#   repeat [Load the calibration](#Load) onward with your videos.
