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
# _Developer Note:_ if you may make a PR in the future, be sure to copy this
# notebook, and use the `gitignore` prefix `temp` to avoid future conflicts.
#
# This is one notebook in a multi-part series on Spyglass.
#
# - To set up your Spyglass environment and database, see
#   [the Setup notebook](./00_Setup.ipynb)
# - For the single-camera (2D) Position V2 DLC workflow — training, model
#   import, pose processing — see
#   [the Position V2 DLC notebook](./23_PositionV2_DLC_2D.ipynb)
#
# Most pose pipelines track an animal in a single camera's image plane (2D).
# When two or more **calibrated, synchronized** cameras view the same scene,
# Spyglass V2 can **triangulate** the 2D detections into real-world **3D**
# coordinates. This notebook covers the extra machinery that makes that work:
#
# - **Generate** a camera-rig calibration (intrinsics + extrinsics) with an
#   external tool, and **load** it into the V2 `CameraRig` / `Calibration`
#   tables.
# - Group the per-camera videos in a `VidFileGroup`, tagging each with a
#   `camera_index` and **pairing the group with its calibration**.
# - Run `PoseEstim` in **3D mode**, which triangulates the per-camera 2D pose
#   into 3D and stores it in NWB.
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
# - [Cleanup](#Cleanup) — remove the tutorial entries

# %% [markdown]
# ## Setup <a id="Setup"></a>

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import shutil
import uuid
import warnings
import zipfile
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

# Edit this to point at your own DataJoint config if needed.
dj.config.load("../dj_local_conf_prod.json")
print(dj.conn(reset=True))

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

# %% [markdown]
# ## The 3D tables <a id="Tables"></a>
#
# The 3D path reuses the V2 pipeline and adds a small calibration branch:
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
# calibration**, it runs the 3D path: per-camera 2D pose → DLT triangulation →
# 3D pose stored in NWB.

# %%
# The video + calibration tables, then training, estimation, and the pipeline.
dj.Diagram(video) + dj.Diagram(train) + dj.Diagram(estim)

# %% [markdown]
# ## Get example data <a id="Data"></a>
#
# The Anipose dataset is distributed as a single ~942 MB zip on Zenodo. We
# download it once (skip if present) and extract only the few files we need for
# one reaching trial: the calibration, the per-camera 2D detections, the raw
# videos, and Anipose's reference 3D output.
#
# > Set `ANIPOSE_DIR` (or the `ANIPOSE_DIR` environment variable) to choose
# > where the data lives. If you already have a copy, point it there to skip the
# > download.

# %%
ZENODO_URL = (
    "https://zenodo.org/api/records/5733431/files/mouse-anipose.zip/content"
)
ANIPOSE_DIR = Path(
    os.environ.get("ANIPOSE_DIR", Path.home() / "spyglass_data" / "anipose")
)
MOUSE_ZIP = ANIPOSE_DIR / "mouse-anipose.zip"
EXTRACT_ROOT = ANIPOSE_DIR / "extracted"

# One reaching trial from one session of the two-camera mouse project.
ZIP_SESS = "mouse-testing/020820_preDTreaches_day1/020820_JiDT13"
REACH = "preDT1_JiDT13_reach1_hit"
CAM_TOKENS = ["cam1", "cam2"]  # camera-name tokens used in file names

# The minimal set of members we need from the zip.
NEEDED = [
    f"{ZIP_SESS}/calibration/calibration.toml",
    f"{ZIP_SESS}/pose-3d/{REACH}.csv",
]
for tok in CAM_TOKENS:
    NEEDED += [
        f"{ZIP_SESS}/pose-2d/{REACH}_{tok}.h5",
        f"{ZIP_SESS}/videos-raw/{REACH}_{tok}.mp4",
    ]

SESSION_DIR = EXTRACT_ROOT / ZIP_SESS


def fetch_example_data():
    """Download (once) and extract only the files this notebook needs."""
    missing = [m for m in NEEDED if not (EXTRACT_ROOT / m).exists()]
    if not missing:
        print("Example data already present — skipping download.")
        return

    ANIPOSE_DIR.mkdir(parents=True, exist_ok=True)
    if not MOUSE_ZIP.exists() or MOUSE_ZIP.stat().st_size == 0:
        import urllib.request

        print(f"Downloading mouse-anipose.zip (~942 MB) to {MOUSE_ZIP} ...")
        print("This is a one-time download; re-runs reuse the extracted files.")
        urllib.request.urlretrieve(ZENODO_URL, MOUSE_ZIP)

    print(f"Extracting {len(missing)} needed file(s) ...")
    with zipfile.ZipFile(MOUSE_ZIP) as zf:
        for member in missing:
            zf.extract(member, EXTRACT_ROOT)
    print("Done.")


fetch_example_data()
print("Session dir:", SESSION_DIR)

# %% [markdown]
# ## Generate a calibration <a id="Generate"></a>
#
# A 3D calibration has two parts per camera:
#
# - **Intrinsics** — focal lengths, principal point, lens distortion. These
#   describe a single camera's optics.
# - **Extrinsics** — the rotation `R` and translation `t` placing each camera in
#   a shared rig coordinate frame. These describe how the cameras sit relative
#   to one another.
#
# **You generate these with a calibration tool, not in Spyglass.** You record a
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
# Below we **load the calibration that ships with the Anipose example** rather
# than re-deriving it, and translate it into the dicts the `Calibration` table
# expects. The Anipose `.toml` stores each camera's pose as a Rodrigues
# **rotation vector + translation (world→camera, millimetres)**; the V2 tables
# store **camera→rig `R`/`t` in metres**, so we invert and rescale.

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


calib = anipose_calibration_to_v2(
    SESSION_DIR / "calibration" / "calibration.toml"
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
# Insert order is fixed by the foreign keys:
# `CameraDevice → CameraRig → CameraRig.Camera → Calibration →
# Calibration.Camera`. We prefix every tutorial row with `DEMO` so it is easy to
# remove later (see [Cleanup](#Cleanup)).

# %%
DEMO = "ap3d_nb"  # identifier prefix for every row this notebook creates
RIG_ID = f"{DEMO}_rig"
CAL_ID = f"{DEMO}_cal"
today = datetime.now().date().isoformat()

# 1. One CameraDevice per physical camera (source of truth for identity).
for ci, cam in calib.items():
    CameraDevice.insert1(
        {"camera_name": f"{DEMO}_{cam['name']}", "meters_per_pixel": 0.0},
        skip_duplicates=True,
    )

# 2. The rig + one slot per camera, mapping camera_index -> CameraDevice.
CameraRig.insert1(
    {
        "camera_rig_id": RIG_ID,
        "description": "Anipose mouse two-camera rig (tutorial)",
        "n_cameras": len(calib),
    },
    skip_duplicates=True,
)
for ci, cam in calib.items():
    CameraRig.Camera.insert1(
        {
            "camera_rig_id": RIG_ID,
            "camera_index": ci,
            "camera_name": f"{DEMO}_{cam['name']}",
        },
        skip_duplicates=True,
    )

# 3. The calibration header + per-camera intrinsics/extrinsics.
Calibration.insert1(
    {
        "camera_rig_id": RIG_ID,
        "calibration_id": CAL_ID,
        "calibration_date": today,
        "notes": "Loaded from Anipose calibration.toml (tutorial)",
    },
    skip_duplicates=True,
)
for ci, cam in calib.items():
    Calibration.Camera.insert1(
        {
            "camera_rig_id": RIG_ID,
            "calibration_id": CAL_ID,
            "camera_index": ci,
            "intrinsics": cam["intrinsics"],
            "extrinsics": cam["extrinsics"],
            "image_size": list(cam["image_size"]),
        },
        skip_duplicates=True,
    )

# Read it back to confirm the round-trip.
(Calibration.Camera & {"camera_rig_id": RIG_ID, "calibration_id": CAL_ID})

# %% [markdown]
# ## Register videos <a id="Videos"></a>
#
# Next we register the per-camera videos and group them. Two ideas matter here:
#
# 1. **`camera_index` must agree with the calibration.** The video at
#    `camera_index = i` is triangulated using the calibration stored at
#    `camera_index = i`. We pair them **by camera name**: the calibration entry
#    for each index carries a name token (`cam1`/`cam2`) that must appear in the
#    matching video's filename.
# 2. **The video group is linked to its calibration** via
#    `VidFileGroup.Calibration`. That link is what flips `PoseEstim` into 3D
#    mode.
#
# To keep things light we use **DeepLabCut's video writer**
# (`VideoWriter.shorten`) to clip a short demo segment from each raw reach video.

# %%
from deeplabcut.utils.auxfun_videos import VideoReader, VideoWriter

CLIPS_DIR = SESSION_DIR / "tutorial_clips"
CLIPS_DIR.mkdir(exist_ok=True)


def clip_demo_video(src, dest_folder, end="00:00:01"):
    """Write a short clip from ``src`` using DLC's VideoWriter; return its path."""
    out = VideoWriter(str(src)).shorten(
        "00:00:00", end, dest_folder=str(dest_folder)
    )
    if out is None:  # older DLC returns None — locate the produced clip
        out = next(Path(dest_folder).glob(f"{Path(src).stem}*short*.mp4"))
    return Path(out)


# Clip one short video per camera, paired to its calibrated camera_index.
clip_paths = {}
for ci, cam in calib.items():
    token = cam["name"]
    src = SESSION_DIR / "videos-raw" / f"{REACH}_{token}.mp4"
    clip = clip_demo_video(src, CLIPS_DIR)
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
# `PoseEstim` stores its results in an NWB analysis file derived from the
# session NWB that the videos belong to. In production you would register your
# real recording with `insert_sessions()`. For this tutorial we create a minimal
# dummy session (a copy of the bundled `minirec` NWB) and register each clip as
# a `VideoFile`.

# %%
nwb_file_name = f"{DEMO}_.nwb"
nwb_path = Path(raw_dir) / nwb_file_name
if not nwb_path.exists():
    shutil.copy2(str(Path(raw_dir) / "minirec20230622_.nwb"), str(nwb_path))

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
Task().insert1({"task_name": f"{DEMO}_task"}, **ins)
IntervalList().insert1(
    {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": f"{DEMO}_epoch_1",
        "valid_times": np.array([[0.0, 1.0]]),
    },
    **ins,
)
TaskEpoch().insert1(
    {
        "nwb_file_name": nwb_file_name,
        "epoch": 1,
        "task_name": f"{DEMO}_task",
        "interval_list_name": f"{DEMO}_epoch_1",
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
                "camera_name": f"{DEMO}_{calib[ci]['name']}",
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
VID_GROUP = f"{DEMO}_grp"
VidFileGroup().insert1(
    {
        "vid_group_id": VID_GROUP,
        "description": "Anipose two-camera reach (tutorial)",
        "files": vf_keys,
        "camera_indices": list(clip_paths.keys()),
    }
)
VidFileGroup.Calibration().insert1(
    {
        "vid_group_id": VID_GROUP,
        "camera_rig_id": RIG_ID,
        "calibration_id": CAL_ID,
    },
    skip_duplicates=True,
)

VidFileGroup.File & {"vid_group_id": VID_GROUP}

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
# 1. **Write the 2D detections as DLC-style `.h5`.** Anipose stores up to 20
#    candidate detections per body part; we keep the top one (`x`/`y`/
#    `likelihood`) and write a file named `{clip_stem}DLC_*.h5` next to where
#    `PoseEstim` looks for it.
# 2. **Register a model.** We're loading detections rather than training, so we
#    register a lightweight placeholder `Skeleton`/`ModelParams`/`Model`. (With
#    your own data, import a real model as in the
#    [DLC notebook](./23_PositionV2_DLC_2D.ipynb).)

# %%
BODYPARTS = ["l-base", "l-edge", "l-middle", "r-base", "r-edge", "r-middle"]
EDGES = [
    ("l-base", "l-edge"),
    ("l-edge", "l-middle"),
    ("l-middle", "l-base"),
    ("r-base", "r-edge"),
    ("r-edge", "r-middle"),
    ("r-middle", "r-base"),
]
OUTPUT_DIR = SESSION_DIR / "tutorial_dlc_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def write_clean_dlc_h5(pose2d_h5, clip_stem, n_rows):
    """Keep the top detection per body part and write a tidy DLC h5."""
    raw = pd.read_hdf(pose2d_h5)
    scorer = raw.columns.get_level_values("scorer")[0]
    flat = raw[scorer]
    cols, data = [], {}
    for bp in BODYPARTS:
        for coord in ("x", "y", "likelihood"):
            cols.append((scorer, bp, coord))
            data[(scorer, bp, coord)] = flat[(bp, coord)].values
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(
        cols, names=["scorer", "bodyparts", "coords"]
    )
    out = OUTPUT_DIR / f"{clip_stem}DLC_resnet50_mouse.h5"
    df.iloc[:n_rows].to_hdf(str(out), key="df_with_missing", mode="w")
    return out


for ci, clip in clip_paths.items():
    token = calib[ci]["name"]
    write_clean_dlc_h5(
        SESSION_DIR / "pose-2d" / f"{REACH}_{token}.h5", clip.stem, n_frames
    )
print(f"Wrote cleaned DLC h5 files to {OUTPUT_DIR}")

# %%
# Register the placeholder skeleton + model. `accept_new_bodyparts=True` adds
# any body parts not already in the BodyPart reference table.
SKELETON_ID = f"{DEMO}_skel"
if not (Skeleton() & {"skeleton_id": SKELETON_ID}):
    Skeleton().insert1(
        {"skeleton_id": SKELETON_ID, "bodyparts": BODYPARTS, "edges": EDGES},
        accept_new_bodyparts=True,
    )

mp = ModelParams().insert1(
    {
        "model_params_id": f"{DEMO}_mp",
        "tool": "DLC",
        "params": {
            "project_path": str(OUTPUT_DIR),
            "shuffle": 1,
            "trainingsetindex": 0,
        },
        "skeleton_id": SKELETON_ID,
    },
    skip_duplicates=True,
)
ModelSelection().insert1(
    {
        "model_params_id": mp["model_params_id"],
        "tool": "DLC",
        "vid_group_id": VID_GROUP,
        "model_selection_id": f"{DEMO}_sel",
    },
    skip_duplicates=True,
)
Model().insert1(
    {
        "model_id": f"{DEMO}_model",
        "model_params_id": mp["model_params_id"],
        "tool": "DLC",
        "vid_group_id": VID_GROUP,
        "model_selection_id": f"{DEMO}_sel",
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
# video group has two cameras **and** a linked calibration, `PoseEstim.make()`
# automatically takes the 3D path.

# %%
params_key = PoseEstimParams.insert_params(
    {"min_confidence": 0.3, "max_reproj_error": 5.0},
    params_id=f"{DEMO}_pep",
    skip_duplicates=True,
)
estim_key = PoseEstimSelection().insert_estimation_task(
    {
        "model_id": f"{DEMO}_model",
        "vid_group_id": VID_GROUP,
        "pose_estim_params_id": params_key["pose_estim_params_id"],
    },
    task_mode="load",
    output_dir=str(OUTPUT_DIR),
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
# `fetch1_dataframe()` returns a `(scorer, bodypart, coord)` MultiIndex with
# `x`/`y`/`z`/`likelihood` per body part (in centimetres), indexed by time.

# %%
df = (PoseEstim & estim_key).fetch1_dataframe()
print("Columns are 3D:", any(c[-1] == "z" for c in df.columns))
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
# The dataset ships Anipose's own triangulation. Because the V2 path uses the
# same linear DLT, the two should agree to within rounding (Anipose stores
# millimetres; V2 stores centimetres, so we rescale V2 by 10).

# %%
ref = pd.read_csv(SESSION_DIR / "pose-3d" / f"{REACH}.csv").iloc[:n_frames]
diffs = []
for b in BODYPARTS:
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
# ## Cleanup <a id="Cleanup"></a>
#
# This tutorial wrote rows to a shared database. Set `CLEANUP = True` and run the
# cell below to remove everything it created (the `DEMO`-prefixed entries).

# %%
CLEANUP = False  # set True to delete all tutorial rows

if CLEANUP:
    safe = dict(safemode=False)
    (PoseEstim & {"model_id": f"{DEMO}_model"}).delete(**safe)
    (PoseEstimSelection & {"model_id": f"{DEMO}_model"}).delete(**safe)
    (PoseEstimParams & {"pose_estim_params_id": f"{DEMO}_pep"}).delete(**safe)
    (Model & {"model_id": f"{DEMO}_model"}).delete(**safe)
    (ModelSelection & {"model_selection_id": f"{DEMO}_sel"}).delete(**safe)
    (ModelParams & {"model_params_id": f"{DEMO}_mp"}).delete(**safe)
    (VidFileGroup & {"vid_group_id": VID_GROUP}).delete(**safe)
    (Calibration & {"camera_rig_id": RIG_ID}).delete(**safe)
    (CameraRig & {"camera_rig_id": RIG_ID}).delete(**safe)
    (Skeleton & {"skeleton_id": SKELETON_ID}).delete(**safe)
    (Session & {"nwb_file_name": nwb_file_name}).delete(**safe)
    (Nwbfile & {"nwb_file_name": nwb_file_name}).delete(**safe)
    for ci, cam in calib.items():
        (CameraDevice & {"camera_name": f"{DEMO}_{cam['name']}"}).delete(**safe)
    print("Tutorial rows removed.")
else:
    print("CLEANUP is False — tutorial rows left in place.")

# %% [markdown]
# ### What's next
#
# - For per-bodypart smoothing, orientation, centroid, and velocity on the
#   triangulated pose, continue to `PoseV2` (see the
#   [DLC notebook](./23_PositionV2_DLC_2D.ipynb) — it reads 3D input transparently).
# - With your own rig: generate a `calibration.toml` (DLC 3D or Anipose),
#   then repeat [Load the calibration](#Load) onward with your videos.
