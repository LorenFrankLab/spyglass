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
# ## Position Pipeline V2
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
# - For additional info on DataJoint syntax, including table definitions and
#   inserts, see
#   [the Insert Data notebook](./02_Insert_Data.ipynb)
# - For the legacy V1 DLC pipeline, see
#   [the DLC notebook](./21_DLC.ipynb)
#
# **Position V2** is designed to expand the functionality of the V1 pipeline
# while simplifying the number of tables. The V2 pipeline:
#
# - **Reduces complexity**: just a few main tables
# - **Multi-tool support**: Works with DLC with planned expansion for SLEAP
# - **Flexible workflows**: Train models in Spyglass or import pre-trained ones
# - **NWB-native storage**: Uses ndx-pose extension for standardized data
# - **Simplified processing**: Single PoseV2 table handles all post-processing
#
# This tutorial assumes you have already ingested an NWB session file.
# It covers:
#
# - **Primary path**: Training a model from scratch within Spyglass
# - **Alternative path**: Importing a pre-trained model (DLC, SLEAP, or NWB)
# - Running pose estimation on videos
# - Processing pose data (orientation, centroid, smoothing)
# - Retrieving and visualizing results
#

# %% [markdown]
# ### Table of Contents

# %% [markdown]
# #### Core Tutorial
#
# - [Setup](#Setup) - Environment configuration
#     - Load packages & configure environment
#     - Connect to database
# - [Which path?](#DecisionTree) - Decision tree
# - [Path A: Train a New Model](#PathA) - Create a DLC project & train
#     - Choose training videos from `VideoFile`
#     - Define body parts and skeleton
#     - `Model.create_project()` → label frames → train
#     - Training loss curve visualization
# - [Path B: Import a Pre-Trained Model](#PathB) - Import existing model
#     - Find or download a pretrained model (DLC Model Zoo, DANDI)
#     - Import from a V1 Spyglass model (`Model.import_from_v1()`)
#     - Import any DLC project directly (`config.yaml`)
#     - SLEAP / ndx-pose NWB ingestion
# - [Pose Estimation](#PoseEstim) - Run inference on videos
#     - Configure inference parameters (e.g., device, batch size)
#     - Set up estimation task
#     - Run inference and validation
# - [Parameters](#PoseParams) - Configure processing settings
#     - Define orientation calculation
#     - Set centroid method
#     - Configure smoothing parameters
# - [Data Processing](#PoseV2) - Calculate final pose data
#     - Run pose processing (velocity, orientation, centroid, smoothing)
#     - Validate results
# - [Analysis](#FetchData) - Retrieve and visualize results
#     - Fetch processed data
#     - Generate trajectory plots
#     - Analyze time series
#
# #### Advanced Features
#
# - [Model Evaluation](#ModelEvaluation) - Training curves & performance metrics
# - [Video Generation](#VideoGeneration) - Create annotated outputs
#
# #### Reference
#
# - [Troubleshooting](#Troubleshooting) - Common issues & solutions
# - [V1→V2 Migration](#Migration) - Upgrade guide
# - [External Resources](#Resources) - Documentation links
# - [Multi-Tool Support](#MultiTool) - SLEAP integration status
# - [JSON Parameters](#json-parameters) - `blob` search functionality

# %% [markdown]
# # Core Tutorial

# %% [markdown]
# ## Setup

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import warnings
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

dj.config.load("../dj_local_conf_prod.json")
# dj.config.load("dj_local_conf.json")
print(dj.conn(reset=True))

# %%
from spyglass.common import Session, VideoFile
from spyglass.position.v2 import (
    BodyPart,
    Model,
    ModelParams,
    ModelSelection,
    PoseEstim,
    PoseEstimParams,
    PoseEstimSelection,
    PoseParams,
    PoseSelection,
    PoseV2,
    Skeleton,
    VidFileGroup,
    check_environment,
    estim,
    train,
    video,
)

_ = (
    video,
    train,
    estim,
    BodyPart,
    Session,
    VideoFile,
    Model,
    ModelSelection,
    ModelParams,
    PoseEstim,
    PoseEstimParams,
    PoseEstimSelection,
    PoseParams,
    PoseSelection,
    PoseV2,
    Skeleton,
    VidFileGroup,
)

print("All imports successful!")

# %% [markdown]
# ### Check the environment
#
# Catches common dependency issues before you train or run inference. The most
# frequent one for users coming from **Position V1** is a leftover TensorFlow
# install (V1's DeepLabCut backend) colliding with V2's jax stack on the GPU.
# If flagged, follow the printed fix (or rebuild from
# `environments/environment_dlc.yml`), then restart the kernel.

# %%
check_environment()

# %% [markdown]
# ### Diagram

# %%
# Full V2 pipeline: video groups → training → estimation → pose processing
dj.Diagram(video) + dj.Diagram(train) + dj.Diagram(estim) + dj.Diagram(PoseV2)

# %% [markdown]
# For a refresher on reading diagrams, see
# [this doc](https://docs.datajoint.com/how-to/read-diagrams/)
#
# A few key points before diving in:
#
# 1. Training starts with a skeleton, representing a collection of body parts.
# 2. A skeleton is specified with model training parameters.
# 3. A video group is a collection of one or more files or calibrations.
# 4. Training takes place on a video group and results in a Model.
# 5. Pose estimation applies a given model to a given video group.
# 6. The final `PoseV2` table incorporates all secondary calculations, like
#     orientation and smoothing.

# %% [markdown]
# ## Which path is right for you? <a id="DecisionTree"></a>
#
# | Situation | Path |
# |---|---|
# | You want to **train a new model** from videos already in Spyglass | **Path A** — [Train a new model](#PathA) |
# | You already have a **pre-trained DLC or SLEAP model** | **Path B** — [Import a pre-trained model](#PathB) |
#
# Both paths converge at the [Pose Estimation](#PoseEstim) section below.

# %% [markdown]
# ### Shared state for both paths
#
# The Pose Estimation and pose-processing sections below reference the variables
# initialized here **regardless of which path you run** — so you can run either
# Path A or Path B without the other. Each path fills in `model_key`; the
# tutorial bootstrap fills in the video/session variables.

# %%
# Tutorial-only scaffolding: import the bootstrap helper from the test suite so
# it lives in one place (and is exercised by the test suite). NOT for production.
import sys

import spyglass

_tests_v2 = Path(spyglass.__file__).parents[2] / "tests" / "position" / "v2"
if str(_tests_v2) not in sys.path:
    sys.path.insert(0, str(_tests_v2))
from make_example_dlc_project import bootstrap_from_video_paths  # noqa: E402

# Shared state — set by whichever path you run (A: train, B: import). Both the
# Pose Estimation and pose-processing sections read these, so they are defined
# up front to keep the two paths independent.
model_key = None
config_path = None
nwb_file_name = None
inf_vid_path = None
training_vid_group_id = None
skeleton_id = None
DEMO_OUTPUT_DIR = None

# %% [markdown]
# ## Path A: Train a New Model <a id="PathA"></a>
#
# **🎯 Goal**: Create a DLC project from videos in Spyglass, label frames, and
# train a pose estimation model.
#
# **Steps**:
# 1. Choose training videos from the `VideoFile` table
# 2. Define body parts and skeleton
# 3. Call `Model.create_project()` to create the DLC project and extract frames
# 4. Label frames externally (DLC GUI / napari)
# 5. Train the model
# 6. Visualize training curves

# %% [markdown]
# ### Step 1 — Choose training videos
#
# Browse the `VideoFile` table to find recordings you want to train on.
# Each row corresponds to one video file registered in Spyglass.

# %%
# Inspect available videos — pick the ones you want to use for training
VideoFile()

# %% [markdown]
# Select your training videos.  You can use `nwb_file_name` + `epoch` dicts
# to reference videos by session, or supply absolute paths:

# %%
# ── Choose your training videos ──────────────────────────────────────────────
# Reference videos by session/epoch — VideoFile.get_abs_paths() resolves them.
# Partial keys (nwb_file_name + epoch, no video_file_num) expand to all
# camera angles recorded for that epoch.
# training_video_list = [
#     {"nwb_file_name": "subject_20240101_.nwb", "epoch": 1},
#     {"nwb_file_name": "subject_20240101_.nwb", "epoch": 3},
# ]

# Tutorial default: set after bootstrapping a session below.
training_video_list = []  # overridden in the bootstrap block below

# %% [markdown]
# ### Step 2 — Define body parts and skeleton

# %%
# List the body parts your model will track.
# Every name must already exist in the BodyPart table (admins can add new ones).
training_bodyparts = ["whiteLED", "tailBase"]

# %% [markdown]
# ### Step 3 — Bootstrap tutorial session & create project
#
# <details>
# <summary><b>Tutorial helper — not for production use</b></summary>
#
# `Model.create_project()` calls DLC to create a project folder and extract
# frames.  Later, `Model.load()` calls `VidFileGroup.create_from_dlc_config()`
# internally, which needs a matching `Session` in the Spyglass database.
#
# **In production**, register your NWB session with `insert_sessions()` and
# ensure `VideoFile` rows exist *before* calling `Model.create_project()` or
# `Model.load()`.
#
# The `bootstrap_from_video_paths()` helper (from
# `tests/position/v2/make_example_dlc_project.py`) creates minimal dummy
# entries so this tutorial works without a recorded session.  It is maintained
# alongside the test suite to stay in sync with the production API.
#
# **If you run this on a shared database, please delete the dummy entries when
# done.**
#
# </details>

# %%
# Path-A tutorial imports. The bootstrap helper and shared-state variables are
# defined once in "Shared state for both paths" above (Setup), so Path A and
# Path B stay independent; here we only add the training-specific helpers.
import yaml
from make_example_dlc_project import make_dlc_project  # noqa: E402

# ── Tutorial: create a minimal example DLC project if none exists ─────────
_demo_dlc_dir = Path.home() / "DeepLabCut" / "examples"
_demo_config = (
    _demo_dlc_dir / "tutorial_dlc-tutorial_dlc-2025-01-01" / "config.yaml"
)

print("_demo_config:", _demo_config)

if not _demo_config.exists():
    print("Demo config not exist")
    _demo_config = make_dlc_project(_demo_dlc_dir)
    print(f"Created example DLC project: {_demo_config}")
else:
    print(f"Using existing DLC project: {_demo_config}")

config_path = Path(_demo_config)

# ── Read the DLC config to get video paths ───────────────────────────────────
with open(config_path) as _f:
    _cfg = yaml.safe_load(_f)
_training_videos = list(_cfg.get("video_sets", {}).keys())
_project_name = Path(_cfg.get("project_path", str(config_path.parent))).name

# ── Register a Spyglass session for those videos (tutorial only) ──────────────
print("Bootstrapping tutorial Spyglass session...")
nwb_file_name, inf_vid_path = bootstrap_from_video_paths(
    _training_videos, nwb_stem=_project_name
)
print(f"  nwb_file_name  : {nwb_file_name}")
print(f"  inf_vid_path   : {inf_vid_path}")

# Use the registered videos as our training list
training_video_list = [{"nwb_file_name": nwb_file_name, "epoch": 1}]


# %% [markdown]
# Now call `Model.create_project()`.  This:
# 1. Validates body parts are in the `BodyPart` table
# 2. Resolves video paths from `VideoFile` (or uses absolute paths directly)
# 3. Calls `deeplabcut.create_new_project()` to create the project folder
# 4. Calls `deeplabcut.extract_frames()` with uniform sampling
# 5. Inserts (or retrieves) a `Skeleton` entry
# 6. Returns `{"config_path": ..., "skeleton_id": ...}`

# %%
project_info = Model().create_project(
    project_name="tutorial_dlc",
    bodyparts=training_bodyparts,
    video_list=training_video_list,
    # Keep this modest for short tutorial videos to avoid oversampling.
    frames_per_video=5,
)

config_path = Path(project_info["config_path"])
skeleton_id = project_info["skeleton_id"]

print(f"DLC project created : {config_path}")
print(f"Skeleton ID         : {skeleton_id}")
print()
print("Next: label the extracted frames, then return to train below.")

# ── Tutorial shortcut: seed synthetic labels so training can run ──────────────
# In a real workflow you would label frames with the DLC GUI or napari.
# Here we write dummy x/y annotations so Model.populate() does not error.
from make_example_dlc_project import seed_labeled_data  # noqa: E402

seed_labeled_data(config_path)
print("Synthetic labels written (tutorial only - replace with real labels).")

# %% [markdown]
# <details>
# <summary><b>Why no <code>DLCProject</code> table in V2?</b></summary>
#
# V1 maintained a `DLCProject` table to track project state and training files,
# keeping a copy of every config and video path in the database.  In practice
# this created bookkeeping overhead without adding scientific value, because the
# on-disk DLC project folder already serves as the ground truth.
#
# V2 treats the `config.yaml` on disk as the record of truth.  The config is
# stored in `ModelParams` only *after* training is complete (via `Model.load()`),
# keeping the database focused on results rather than intermediate state.
#
# </details>

# %% [markdown]
# ### Step 4 — Label frames (manual step)
#
# After `create_project()` finishes, DLC has extracted frames into
# `labeled-data/` inside the project folder.  Label them using the DLC GUI or
# the napari plugin before training:
#
# ```bash
# # DLC GUI (requires a display)
# python -m deeplabcut
#
# # Or programmatically (works in headless environments with napari installed)
# ipython -c "import deeplabcut; deeplabcut.label_frames('$config_path')"
# ```
#
# Return here once labeling is complete.

# %% [markdown]
# ### Step 5 — Train the model
#
# After labeling, insert `ModelParams` and `ModelSelection`, then call
# `Model.populate()`:

# %%
# Training parameters — use epochs=1/save_epochs=1 for the fastest possible demo.
# DLC 3.x PyTorch backend: `epochs` overrides `maxiters` and controls epochs
# directly; `save_epochs` controls checkpoint frequency.
train_params = {
    "trainingsetindex": 0,
    "shuffle": 1,
    "gputouse": None,
    "TFGPUinference": False,
    "net_type": "resnet_50",
    "augmenter_type": "imgaug",
    "epochs": 1,
    "save_epochs": 1,
    "project_path": str(config_path.parent),
}
TRAIN_PARAMS_ID = "path_a_demo_1epoch"

ModelParams.insert1(
    {
        "model_params_id": TRAIN_PARAMS_ID,
        "params": train_params,
        "tool": "DLC",
        "skeleton_id": skeleton_id,
    },
    skip_duplicates=True,
)

# Create a VidFileGroup from the DLC config so ModelSelection can reference it.
# Package-level matching now includes explicit basename->full-key fallback.
training_vid_group_key = VidFileGroup.create_from_dlc_config(config_path)
training_vid_group_id = training_vid_group_key["vid_group_id"]

_sel_key = {
    "model_params_id": TRAIN_PARAMS_ID,
    "tool": "DLC",
    "vid_group_id": training_vid_group_id,
    "model_selection_id": TRAIN_PARAMS_ID,  # reuse params id as selection id
    "parent_id": None,
}
ModelSelection.insert1(_sel_key, skip_duplicates=True)

if len(Model & _sel_key) > 0:
    print(f"Model '{TRAIN_PARAMS_ID }' already exists — skipping training")
else:
    print("Starting training (10 iterations) ...")
    Model.populate(_sel_key, display_progress=True)
    print(f"Model '{TRAIN_PARAMS_ID }' trained and saved")

model_key = (Model & _sel_key).fetch1()
print(f"model_key: {model_key['model_id']}")

# %% [markdown]
# ### Step 6 — Visualize training curves

# %%
if model_key:
    _history = Model().get_training_history({"model_id": model_key["model_id"]})
    if len(_history) > 0:
        Model().plot_training_history({"model_id": model_key["model_id"]})
    else:
        print("No training history yet (expected for 10-iteration demo).")

# %% [markdown]
# ## Path B: Import a Pre-Trained Model <a id="PathB"></a>
#
# **Goal**: Load an existing DLC, SLEAP, or ndx-pose model into Spyglass.
#
# **What you'll accomplish**:
# - Import a DeepLabCut project (`config.yaml`), SLEAP file, or DLC h5 output
# - Understand skeleton and bodypart organization
# - Create video file groups for analysis
#
# > **Note**: Skip this section if you just completed Path A above.

# %% [markdown]
# Position V2 supports different import methods:
#
# 1. **DLC config.yaml**: Import models trained with DeepLabCut
# 2. **SLEAP / external tools**: Import via ndx-pose NWB file
# 3. **DLC h5 output**: Load inference results from pre-run DLC

# %% [markdown]
# #### From DeepLabCut Project
#
# If you have a DeepLabCut model already trained, you can import it by changing
# `config_path` below to your `config.yaml`. If not, obtain one from the
# [DLC Model Zoo](https://www.mackenziemathislab.org/dlc-modelzoo):
#
# ```python
# import deeplabcut
# deeplabcut.create_project_from_modelzoo(
#     modelname="full_cat",
#     working_directory="/path/to/save",
#     videos=["/path/to/your/video.mp4"],
# )
# ```
#
# Or clone the DLC examples:
# ```bash
# git clone https://github.com/DeepLabCut/DeepLabCut/
# python ./DeepLabCut/examples/testscript.py
# ```

# %% [markdown]
# > **Session prerequisite** — `Model.load()` calls
# > `VidFileGroup.create_from_dlc_config()` internally. In production, run
# > `insert_sessions('your_training_session.nwb')` before this step.
# >
# > For this tutorial, `bootstrap_from_video_paths()` creates minimal dummy
# > entries so the import can proceed without a recorded session.

# %%
# Path B: default — import from a Spyglass V1 DLC model
# ────────────────────────────────────────────────────────────────────────────
# If you completed Path A, model_key is already set — this block is skipped.
#
# By default we import the pre-trained Wtrack_WhiteLED model from the V1
# schema.  Swap _DEFAULT_V1_KEY for any other V1 model key as needed.
# To import a raw config.yaml instead, see the commented fallback below.

_DEFAULT_V1_KEY = {
    "project_name": "Wtrack_WhiteLED",
    "dlc_model_name": "Wtrack_WhiteLED_ms_stim_wtrack_00",
    "dlc_model_params_name": "default",
}

if model_key is None:
    try:
        model_key = Model().import_from_v1(_DEFAULT_V1_KEY)
        print(f"Imported V1 model: {model_key['model_id']}")
    except (ImportError, KeyError, FileNotFoundError) as _exc:
        print(f"V1 import unavailable ({type(_exc).__name__}): {_exc}")
        print(
            "Falling back to direct config.yaml import.\n"
            "Uncomment and set _b_config_path in the block below."
        )

# ── Fallback: import from a raw config.yaml ──────────────────────────────────
# Uncomment if import_from_v1 is unavailable or you want to import a model
# that is not in the V1 schema.
#
# _b_config_path = Path("/path/to/your/config.yaml")
# if model_key is None and _b_config_path.exists():
#     with open(_b_config_path) as _f:
#         _b_cfg = yaml.safe_load(_f)
#     _b_videos = list(_b_cfg.get("video_sets", {}).keys())
#     _b_project_name = Path(
#         _b_cfg.get("project_path", str(_b_config_path.parent))
#     ).name
#     try:
#         model_key = Model().load(_b_config_path)
#     except ValueError:
#         print("Bootstrapping tutorial session for DLC import...")
#         nwb_file_name, inf_vid_path = bootstrap_from_video_paths(
#             _b_videos, nwb_stem=_b_project_name
#         )
#         model_key = Model().load(_b_config_path)
#     config_path = _b_config_path
#     print(f"Imported model: {model_key}")

# %% [markdown]
# The `Model.load()` import process:
#
# 1. Detects the latest trained model snapshot
# 2. Extracts the skeleton (bodyparts and connections)
# 3. Creates entries in Skeleton and ModelParams tables
# 4. Links the DLC videos to a registered Spyglass session via VidFileGroup
# 5. Stores model metadata in an NWB file
# 6. Creates a Model entry for inference
#
# Body-part names are reconciled with the curated `BodyPart` table on import,
# so spelling variants (e.g. `green_led` vs `greenLED`) are handled
# automatically. To also rewrite your DLC project's `config.yaml` to the
# canonical spelling, pass `Model().load(config_path, normalize_names=True)`;
# the original config is saved to a timestamped `config.yaml.<ts>.bak` first.
# Body parts that are not in `BodyPart` and have no canonical match still
# require an admin to add them.
#

# %% [markdown]
# #### Validate model (Path B)

# %%
if model_key and not Model() & model_key:
    raise ValueError(f"❌ Model entry not found : {model_key}")

if model_key:
    model_params = (
        ModelParams() & {"model_params_id": model_key["model_params_id"]}
    ).fetch1()
    skeleton_id = model_params.get("skeleton_id")
    if not (Skeleton() & {"skeleton_id": skeleton_id}).fetch1("KEY"):
        raise ValueError(f"❌ Skeleton not found for model: {model_key}")
    if vid_group_id := model_key.get("vid_group_id"):
        if not VidFileGroup() & {"vid_group_id": vid_group_id}:
            raise ValueError(f"❌ Video group not found: {vid_group_id}")
    training_vid_group_id = model_key["vid_group_id"]
    print("✅ Path B model import validated")

# %% [markdown]
# #### Inference video — auto-generated from training data
#
# If `inf_vid_path` is still `None` after Path B, the cell below fetches the
# first training video from the V1 model, calls `bootstrap_from_video_paths()`
# to create a 1-second inference clip via ffmpeg, and registers it in
# `VideoFile`.
#
# To use your own video instead, set `inf_vid_path` before running this cell:
#
# ```python
# inf_vid_path = Path("/path/to/your/video.mp4")
# ```

# %%
if inf_vid_path is None and model_key is not None:
    _v1_videos = []
    try:
        from spyglass.position.v1.position_dlc_model import (
            DLCModel as _DLCModel,
        )

        _v1_cfg = (_DLCModel & _DEFAULT_V1_KEY).fetch1("config_template")
        _v1_videos = list(_v1_cfg.get("video_sets", {}).keys())
    except Exception as _exc:
        print(f"Could not fetch V1 training videos: {_exc}")

    if _v1_videos:
        _v1_project_name = Path(_v1_videos[0]).parent.parent.name
        print("Creating 1-second inference clip from V1 training video...")
        nwb_file_name, inf_vid_path = bootstrap_from_video_paths(
            _v1_videos, nwb_stem=_v1_project_name
        )
        print(f"  nwb_file_name : {nwb_file_name}")
        print(f"  inf_vid_path  : {inf_vid_path}")
    else:
        print(
            "No training videos found in V1 model.  "
            "Set inf_vid_path manually before running Pose Estimation."
        )

# %%
if skeleton_id:
    fig = (Skeleton & {"skeleton_id": skeleton_id}).show_skeleton()
    plt.show()

# %% [markdown]
# #### From SLEAP / External Tools (via ndx-pose NWB)
#
# SLEAP and other tools (e.g. DLC2NWB) can export results as NWB files using
# the ndx-pose extension.  The canonical Spyglass entry point for this data is
# `ImportedPose`.
#
# **Basic ingestion**
#
# ```python
# # 1. Register the session NWB (must be in the Nwbfile table)
# from spyglass.data_import import insert_sessions
# insert_sessions("sleap_output.nwb")
#
# # 2. Ingest the pose data — this is all you need for most workflows
# from spyglass.position.v1.imported_pose import ImportedPose
# ImportedPose().insert_from_nwbfile("sleap_output_.nwb")
# ```
#
# **Optional: also register in the V2 pipeline**
#
# If you need the pose data to be accessible through V2 inference tables
# (Skeleton, ModelParams, Model, PoseEstimSelection), pass `import_to_v2=True`:
#
# ```python
# ImportedPose().insert_from_nwbfile("sleap_output_.nwb", import_to_v2=True)
# ```
#
# > **Note**: `Model.load()` no longer accepts NWB files.  Use
# > `ImportedPose.insert_from_nwbfile()` as shown above.

# %%
# Generate an example ndx-pose NWB file using the test-suite helper.
import sys

import spyglass

_tests_v2 = Path(spyglass.__file__).parents[2] / "tests" / "position" / "v2"
sys.path.insert(0, str(_tests_v2))
from make_example_ndx_pose import make_ndx_pose_nwb

ndx_pose_path = config_path.parent / "example_ndx_pose.nwb"
make_ndx_pose_nwb(ndx_pose_path)
print(f"ndx-pose NWB file ready: {ndx_pose_path}")

# %%
# To ingest an ndx-pose NWB, first register it in Spyglass, then call
# ImportedPose().insert_from_nwbfile().  The block below shows the intended
# workflow using a locally generated example file (production use requires the
# NWB to already be inserted via insert_sessions()).
#
# Example (uncomment and adapt for real use):
#   from spyglass.data_import import insert_sessions
#   from spyglass.position.v1.imported_pose import ImportedPose
#   insert_sessions("sleap_output.nwb")
#   ImportedPose().insert_from_nwbfile("sleap_output_.nwb")
#   # Also register skeleton/model in V2 tables:
#   ImportedPose().insert_from_nwbfile("sleap_output_.nwb", import_to_v2=True)
print(
    "ndx-pose NWB ingestion goes through ImportedPose.insert_from_nwbfile().\n"
    "See the markdown cell above for the full workflow."
)

# %% [markdown]
# #### From DLC h5 output
#
# If you have run DLC inference and have `.h5` pose output files,
# `deeplabcut.analyze_videos_converth5_to_nwb` (requires `dlc2nwb`) converts
# them to an ndx-pose NWB with real `original_videos` paths.  Register the
# resulting NWB with `insert_sessions()` and then call
# `ImportedPose().insert_from_nwbfile()` to ingest the pose data.  Pass
# `import_to_v2=True` if you also want skeleton/model metadata in the V2 tables.

# %%
try:
    import deeplabcut as _dlc
    import dlc2nwb  # noqa: F401

    # Set config_path and video_folder to your project paths, then:
    # config_path = "your/config/path/config.yaml"
    # video_folder = "your/video/folder"
    # _dlc.analyze_videos_converth5_to_nwb(config_path, video_folder)
    #
    # Then register and ingest:
    # from spyglass.data_import import insert_sessions
    # from spyglass.position.v1.imported_pose import ImportedPose
    # insert_sessions("your_dlc_output.nwb")
    # ImportedPose().insert_from_nwbfile("your_dlc_output_.nwb", import_to_v2=True)

    print("dlc2nwb available — uncomment the lines above to run.")
except ImportError:
    print(
        "dlc2nwb not installed. Install with: pip install dlc2nwb\n"
        "Required by deeplabcut.analyze_videos_converth5_to_nwb."
    )

# %% [markdown]
# ## Pose Estimation <a id="PoseEstim"></a>
#

# %%
# Guard: a model must be available from either Path A or Path B above.
if model_key is None:
    raise ValueError(
        "Complete Path A (train a new model) or Path B (import a pre-trained "
        "model) before running pose estimation."
    )

# %% [markdown]
#
# **🎯 Goal**: Run pose inference on videos using the imported model
#
# **🔍 What you'll accomplish**:
# - Configure inference parameters (device, batch size)
# - Set up estimation tasks with video groups
# - Run inference and validate results
# - Handle common errors gracefully

# %% [markdown]
# Now that we have a model, let's run pose estimation on a video.
#

# %% [markdown]
# #### Load existing DLC output

# %% [markdown]
# If you've already run DLC inference and have the output files:
#

# %%
# Example: Load existing DLC h5/csv files into NWB
# dlc_output = "/path/to/video_DLC_resnet50_project100000.h5"
# nwb_file = "analysis_20250106.nwb"
#
# PoseEstim.load_dlc_output(
#     dlc_output_path=dlc_output,
#     nwb_file_name=nwb_file
# )

# %% [markdown]
# #### Running Inference via the Pipeline
#

# %% [markdown]
# Inference in V2 follows a three-step Spyglass pattern:
#
# 1. **`PoseEstimParams`** — name a set of inference parameters
#     (device, batch size, etc.)
# 2. **`PoseEstimSelection`** — pair a model with a video group and choose
#     `task_mode='trigger'` (run inference) or `'load'` (read existing output)
# 3. **`PoseEstim.populate()`** — executes inference and stores results in an
#     NWB file via ndx-pose
#
# > **`task_mode='load'` vs `ImportedPose`**: Use `task_mode='load'` when
# > DLC/SLEAP has already written output files on disk and you want to read them
# > into Spyglass. Use `ImportedPose` when your results already exist in NWB
# > format from another pipeline.

# %% [markdown]
# ##### Step 1 — Inference parameters (`PoseEstimParams`)
#
# In this table, `params_hash` is a unique identifier for the set of params
# used.  # This will raise an error if you attempt to insert a new entry with
# `params` matching an existing row.

# %%
# Define custom params for your hardware:
params_result = PoseEstimParams.insert_params(
    params={"device": "cuda", "batch_size": 8},
    params_id="gpu_batch8",
    skip_duplicates=True,
)
print(f"Inserted PoseEstimParams: {params_result}")

PoseEstimParams & {"params.device": "cuda", "params.batch_size": 8}

# %% [markdown]
# ##### Step 2 — Estimation task (`PoseEstimSelection`)
#
# Pose estimation uses **two separate `VidFileGroup` entries**:
#
# - **Training group** (`training_vid_group_id`): created by `Model.load` and
#   linked to `ModelSelection`. Contains the original labeled videos used for
#   training. Used by `get_nwb_file()` to resolve the parent session.
# - **Inference group** (`inf_vid_group_id`): created here for the video(s) you
#   want to run inference on. Linked to `PoseEstimSelection`.
#
# This structure supports multi-camera recordings.

# %%
# Create video groups and estimation selection
inf_vid_group_key = None
estim_key = None

if not (inf_vid_path and model_key):
    raise ValueError("⚠️ Missing video or model - skipping PoseEstimSelection")

# Check/create required params entry
PARAMS_ID = "cpu_batch8"  # or "gpu_batch8" depending on your hardware
PoseEstimParams.insert_params(
    params={"device": "cpu", "batch_size": 8},
    params_id=PARAMS_ID,
    skip_duplicates=True,
)

# Create inference video group
inf_vid_group_key = VidFileGroup().insert1(
    {
        "description": f"Inference video for {model_key['model_id']}",
        "files": [inf_vid_path],
    },
    skip_duplicates=True,
)

# Create estimation selection
estim_selection_key = {
    "model_id": model_key["model_id"],
    "vid_group_id": inf_vid_group_key["vid_group_id"],
    "pose_estim_params_id": PARAMS_ID,
    "task_mode": "load" if DEMO_OUTPUT_DIR else "trigger",
    "output_dir": str(DEMO_OUTPUT_DIR) if DEMO_OUTPUT_DIR else "",
}

PoseEstimSelection().insert1(estim_selection_key, skip_duplicates=True)

# Key for populate
estim_key = {
    k: v
    for k, v in estim_selection_key.items()
    if k in ["model_id", "vid_group_id", "pose_estim_params_id"]
}
print(f"✅ Created estimation selection: {estim_key}")

# %%
# Inspect video groups if available
if training_vid_group_id:
    print(
        "Training videos:",
        len(VidFileGroup().File() & {"vid_group_id": training_vid_group_id}),
    )
if inf_vid_group_key:
    print(
        "Inference videos:",
        len(
            VidFileGroup().File()
            & {"vid_group_id": inf_vid_group_key["vid_group_id"]}
        ),
    )

# %% [markdown]
# ##### Step 3 — Run inference (`PoseEstim.populate()`)
#

# %%
if not estim_key:
    raise ValueError("No estim_key available - check previous steps")

# Check if this is demo mode vs. real inference
selection_entry = (PoseEstimSelection() & estim_key).fetch1()
task_mode = selection_entry.get("task_mode", "trigger")
output_dir = selection_entry.get("output_dir", "")
pose_df = None

PoseEstim.populate(estim_key, display_progress=True)

# Fetch the results
pose_df = (PoseEstim() & estim_key).fetch1_dataframe()
print(pose_df.head())

# %% [markdown]
# <details>
# <summary>🚨 <b>Troubleshooting Pose Estimation</b></summary>
#
# ### Common Issues & Solutions
#
# #### 🔴 **Error**: "No h5 output files found"
# **Cause**: Demo mode vs. real inference mismatch
# **Solution**:
# - This is expected in tutorial demo mode
# - For real analysis, ensure `task_mode='trigger'` for automatic inference
# - Or provide existing DLC `.h5` files with `task_mode='load'`
#
# #### 🔴 **Error**: "CUDA out of memory"
# **Cause**: GPU memory insufficient
# **Solution**:
# ```python
# # Use CPU instead
# PoseEstimParams.insert_params(
#     params={"device": "cpu", "batch_size": 4},
#     params_id="cpu_batch4"
# )
# ```
#
# #### 🔴 **Error**: "Model not found" or "VidFileGroup not found"
# **Cause**: Model import incomplete
# **Solution**: Run the validation checkpoint above to diagnose
#
# #### 🔴 **Error**: "KeyError: analysis_file_name"
# **Cause**: NWB file path resolution issue
# **Solution**: Ensure video groups are linked to registered sessions
#
# #### 🔴 **Warning**: "Low likelihood values"
# **Cause**: Model doesn't generalize to your videos
# **Solution**:
# - Adjust `likelihood_thresh` in processing parameters
# - Consider fine-tuning the model with additional labeled frames
#
# ### Diagnostic Commands
#
# ```python
# # Check table states
# print("Models:", len(Model()))
# print("Video Groups:", len(VidFileGroup()))
# print("Pose Estimates:", len(PoseEstim()))
#
# # Inspect your specific entries
# print("\nYour model:", Model() & model_key)
# vid_group = VidFileGroup() & inf_vid_group_key if inf_vid_group_key else None
# print(f"Your video group: {vid_group}")
# print("Your estimation:", PoseEstim() & estim_key if estim_key else "None")
# ```
#
# </details>

# %% [markdown]
# ## Processing Parameters <a id="PoseParams"></a>
#

# %% [markdown]
#
# **Goal**: Configure how raw pose data is processed into final trajectories
#
# **What you'll accomplish**:
# - Understand orientation, centroid, and smoothing parameters
# - Use default parameters or create custom configurations
# - Match parameter settings to your tracking setup

# %% [markdown]
# Before processing pose data, we need to define processing parameters.
# PoseParams stores configuration for:
#
# - **Orientation**: How to calculate head direction
# - **Centroid**: How to combine bodyparts into a single position
# - **Smoothing**: How to interpolate and smooth the trajectory
#
# > **Tip**: Each `pose_params_id` name is human-readable but the full parameter
# > set lives in a JSON blob. Use
# > `(PoseParams() & {"pose_params_id": name}).fetch1()` to inspect all fields, or
# > query by value with `PoseParams & {"params.smoothing.likelihood_thresh": 0.1}`.

# %% [markdown]
# #### Processing-option use-case guide
#
# Choose your settings based on the downstream analysis:
#
# | Use case | Orient | Centroid | Smoothing | Notes |
# |---|---|---|---|---|
# | **Standard 2-LED navigation** | `two_pt` (green→red) | `2pt` | `moving_avg`, 50 ms | Classic Frank Lab LED tracking |
# | **4-LED arrays** | `bisector` (L/R/apex) | `4pt` | `moving_avg`, 50 ms | `greenLED`, `redLED_C/L/R` required |
# | **Single bodypart** | `none` | `1pt` | `savgol` | Head-fixed or whole-body centroid |
# | **DLC / SLEAP skeleton** | `two_pt` (nose→tail) | `1pt` nose | `gaussian` | Any two keypoints for direction |
# | **MoSeq input** | `none` | `1pt` | `smooth=False` | MoSeq expects raw, unsmoothed pose |
# | **Downstream RL / decoding** | `two_pt` or `none` | match model | `moving_avg` | Smooth to match neural bin width |
# | **Visualization only** | any | any | light smoothing | `savgol` or `gaussian` for clean plots |
#
# **Smoothing methods at a glance**:
#
# - `moving_avg` — causal boxcar; good general-purpose choice, little distortion
# - `savgol` — Savitzky-Golay polynomial fit; preserves peak velocities better
# - `gaussian` — symmetric Gaussian; best for display / offline analysis
#
# **likelihood_thresh** controls which frames are treated as low-confidence and
# interpolated over before smoothing.  Typical values: `0.9` (strict) → `0.5`
# (permissive).  Start with `0.9` and lower if too many frames are dropped.

# %% [markdown]
# #### View Available Parameter Sets
#

# %%
PoseParams()

# %% [markdown]
# Inspect the full JSON contents of any named parameter set:

# %%
# Helper: pretty-print the contents of a named PoseParams entry.
# Change the name to any entry shown in the table above.
import json

PARAMS_TO_INSPECT = "default"

if PoseParams() & {"pose_params_id": PARAMS_TO_INSPECT}:
    _row = (PoseParams() & {"pose_params_id": PARAMS_TO_INSPECT}).fetch1()
    print("--- orient ---")
    print(json.dumps(_row["orient"], indent=2))
    print("\n--- centroid ---")
    print(json.dumps(_row["centroid"], indent=2))
    print("\n--- smoothing ---")
    print(json.dumps(_row["smoothing"], indent=2))
else:
    print(
        f"No entry named '{PARAMS_TO_INSPECT}'. "
        "Run PoseParams.insert_default() first, or change _params_to_inspect."
    )

# %% [markdown]
# Search all entries that match a specific sub-field value (DataJoint JSON):

# %%
# Example: find entries where likelihood_thresh > 0.5 (adjust as needed)
# DataJoint dot-notation works for equality and IN-list queries:
#   PoseParams & {"orient.method": "two_pt"}
#   PoseParams & {"smoothing.smooth": True}
#   PoseParams & {"centroid.method": ["1pt", "2pt"]}
PoseParams & {"orient.method": "two_pt"}

# %% [markdown]
# #### Use Default Parameters
#

# %% [markdown]
# For 2-LED tracking:
#

# %%
PoseParams.insert_default(skip_duplicates=True)
params_key = {"pose_params_id": "default"}  # or '4LED_default' or 'single_LED
PoseParams()

# %% [markdown]
# #### Create Custom Parameters
#

# %%
# Create custom pose parameters based on model bodyparts
if not model_key:
    raise ValueError("No model key available - cannot create custom PoseParams")

model_params = (ModelParams() & model_key).fetch1()
skeleton_id = model_params["skeleton_id"]
bp_tbl = Skeleton.BodyPart() & {"skeleton_id": skeleton_id}
skeleton_parts = bp_tbl.fetch("bodypart")

# Multi-bodypart custom parameters using standard DataJoint interface
bodypart1, bodypart2 = skeleton_parts[0], skeleton_parts[1]

# Use standard DataJoint insert1 interface with automatic validation
PoseParams.insert1(
    {
        "pose_params_id": "tutorial_custom",
        "orient": {
            "method": "two_pt",
            "bodypart1": bodypart1,
            "bodypart2": bodypart2,
            "interpolate": True,
            "smooth": True,
        },
        "centroid": {
            "method": "1pt",
            "points": {"point1": skeleton_parts[0]},
        },
        "smoothing": {
            "interpolate": True,
            "interp_params": {
                "max_pts_to_interp": 10,
                "max_cm_to_interp": 15.0,
            },
            "smooth": True,
            "smoothing_params": {
                "method": "moving_avg",
                "smoothing_duration": 0.3,
            },
            "likelihood_thresh": 0.1,
        },
    },
    skip_duplicates=True,
)

print("✅ Created pose parameters using standard DataJoint interface")

# %% [markdown]
# Inspect the parameters:
#

# %%
(PoseParams() & {"pose_params_id": "tutorial_custom"}).fetch1()

# %% [markdown]
# Search the table:

# %%
PoseParams & {"params.smoothing.interp_params": "max_pts_to_interp"}

# %% [markdown]
# ## Data Processing <a id="PoseV2"></a>
#

# %% [markdown]
#
# **🎯 Goal**: Process raw pose estimates into final position trajectories
#
# **🔍 What you'll accomplish**:
# - Link pose estimation results to processing parameters
# - Run the position processing pipeline
# - Generate orientation, centroid, and velocity data
# - Validate the final processed dataset

# %% [markdown]
# #### Create Processing Selection
#
# Insert a `PoseSelection` row that links a `PoseEstim` result to a set of
# `PoseParams`, then call `PoseV2.populate()` to run the processing pipeline.
#
# > **Prerequisite** — `PoseEstim` must be populated before inserting
# > `PoseSelection`. Both `PoseEstim` and `PoseV2` require the inference
# > `VidFileGroup` to be linked to a registered `Nwbfile`. Verify with:
# > ```python
# > VidFileGroup().get_nwb_file(vid_group_id)  # raises if not linked
# > ```

# %%
# Create pose processing selection and run processing
pose_selection_key = None
processed_df = None

if not estim_key:
    raise ValueError("No estim_key available - cannot run PoseV2 processing")

# Use custom params if available, otherwise default
params_name = "tutorial_custom"
if not PoseParams() & {"pose_params_id": params_name}:
    PoseParams.insert_default(skip_duplicates=True)
    params_name = "default"

pose_selection_key = {**estim_key, "pose_params_id": params_name}
PoseSelection().insert1(  # This will warn about not using optimal params
    pose_selection_key, skip_duplicates=True, ignore_extra_fields=True
)

print("")
print(f"Processing with params: {params_name}")
PoseV2.populate(pose_selection_key)

processed_df = (PoseV2() & pose_selection_key).fetch1_dataframe()
print(f"✅ Processed data: {processed_df.shape[0]} timepoints")
print(processed_df.head())

# %% [markdown]
# The PoseV2.make() pipeline performs:
#
# 1. **Likelihood filtering**: Remove low-confidence detections
# 2. **Orientation calculation**: Compute head direction
# 3. **Centroid calculation**: Combine bodyparts into position
# 4. **Interpolation**: Fill gaps in tracking
# 5. **Smoothing**: Remove jitter from trajectories
# 6. **Velocity calculation**: Compute speed
# 7. **NWB storage**: Save results in standardized format
#

# %% [markdown]
# ### ✅ Validation Checkpoint: Data Processing
#
# Let's verify the position processing worked correctly:

# %%
if processed_df is None:
    raise ValueError(
        "❌ No processed data - ensure PoseV2.populate() completed"
    )

required = ["position_x", "position_y", "orientation", "speed"]
missing = [col for col in required if col not in processed_df.columns]

if missing:
    raise ValueError(f"❌ Missing columns: {missing}")

# Data summary
time_range = processed_df.index[-1] - processed_df.index[0]
mean_speed = processed_df["speed"].mean()

if not pose_selection_key or not (PoseV2() & pose_selection_key).fetch1("KEY"):
    raise ValueError("❌ PoseV2 entry not found for selection key")

print("✅ Validation passed")
print(f"Duration: {time_range:.1f}s, Mean speed: {mean_speed:.1f} cm/s")

# %% [markdown]
# ## Data Analysis & Retrieval <a id="FetchData"></a>

# %% [markdown]
# **🎯 Goal**: Access processed position data for analysis and visualization
#
# **🔍 What you'll accomplish**:
# - Retrieve data as pandas DataFrames or raw NWB objects
# - Generate trajectory and time series visualizations
# - Understand data structure and coordinate systems
# - Export results for further analysis

# %% [markdown]
# ### [Visualization](#TableOfContents) <a id="Visualization"></a>
#

# %% [markdown]
# Let's visualize the processed position data:
#

# %% [markdown]
# #### Trajectory Plot
#

# %%
if processed_df is None:
    raise ValueError("No processed data available for plotting")

fig, ax = plt.subplots(figsize=(10, 10))

# Plot trajectory colored by speed
scatter = ax.scatter(
    processed_df["position_x"],
    processed_df["position_y"],
    c=processed_df["speed"],
    cmap="viridis",
    s=5,
    alpha=0.6,
)

ax.set_xlabel("X position (cm)")
ax.set_ylabel("Y position (cm)")
ax.set_title("Animal trajectory (colored by speed)")
ax.axis("equal")
ax.invert_yaxis()  # Match video coordinates
plt.colorbar(scatter, label="Speed (cm/s)", ax=ax)
plt.show()

# %% [markdown]
# #### Time Series Plot
#

# %%
if processed_df is None:
    raise ValueError("No processed data available for plotting")

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Position over time
axes[0].plot(
    processed_df.index, processed_df["position_x"], label="X", alpha=0.7
)
axes[0].plot(
    processed_df.index, processed_df["position_y"], label="Y", alpha=0.7
)
axes[0].set_ylabel("Position (cm)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Orientation over time
axes[1].plot(processed_df.index, np.rad2deg(processed_df["orientation"]))
axes[1].set_ylabel("Orientation (degrees)")
axes[1].grid(True, alpha=0.3)

# Speed over time
axes[2].plot(processed_df.index, processed_df["speed"])
axes[2].set_ylabel("Speed (cm/s)")
axes[2].set_xlabel("Time (s)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# # Advanced Features
#

# %% [markdown]
# The following sections cover optional advanced functionality. Most users can
# skip these sections and return later as needed.

# %% [markdown]
# ## New Models <a id="TrainingWorkflow"></a>
#
# The complete workflow for training a new model from scratch — including
# video selection, skeleton definition, frame extraction, and training —
# is covered in **[Path A](#PathA)** at the top of this notebook.
#
# Return to Path A if you need a refresher.  The steps below (Model
# Evaluation, Video Generation) assume you have a `model_key` from either
# Path A or Path B.

# %% [markdown]
# ## Model Evaluation <a id="ModelEvaluation"></a>
#

# %% [markdown]
#
# **🎯 Goal**: Evaluate model performance and visualize training progress
#
# **🔍 What you'll accomplish**:
# - Assess model accuracy on test data
# - Generate comprehensive training and validation loss curves
# - Understand model convergence and potential overfitting
# - Monitor learning rate schedules and optimization progress
# - Compare multiple models or training configurations
#
# ### Background
#
# Model evaluation helps you understand:
# - **Training Progress** - Loss curves show how well the model learned
# - **Generalization** - Test accuracy indicates real-world performance
# - **Overfitting Detection** - Training vs. validation divergence
# - **Hyperparameter Optimization** - Compare different configurations
#
# ### Check Model Evaluation Availability

# %%
# Check model evaluation availability
if model_key is None:
    raise ValueError("No model key available - cannot check evaluation support")

model_params = (ModelParams() & model_key).fetch1()
model_tool = model_params.get("tool", "Unknown")
training_history = Model().get_training_history(model_key)
has_training_history = training_history is not None
evaluation_supported = model_tool.upper() == "DLC"

# %% [markdown]
# ### Generate Training Curves
#
# Visualize training progress and detect potential overfitting:

# %%
# Enhanced training curves visualization
if not (has_training_history and evaluation_supported):
    raise ValueError("⚠️ Model evaluation unavailable.")

if len(training_history) < 1:
    raise ValueError("❌ Not enough training history data available.")

# Use built-in plotting method with detailed diagnostics enabled
fig = Model().plot_training_history(
    model_key,
    save_path=None,
    detailed=True,
)

# %% [markdown]
# ## Video Generation <a id="VideoGeneration"></a>
#

# %% [markdown]
#
# ### Prerequisites Check
#
# Before generating videos, ensure you have completed pose estimation:
#
# ```python
# # Check video generation prerequisites
# can_make_video = False
# if (
#     processed_df is not None
#     and len(processed_df) > 0
#     and pose_selection_key is not None
# ):
#     can_make_video = True
# ```
#
# ### Generate Pose Videos
#
# Create annotated videos showing keypoints and pose estimation results:
#
# ```python
# if can_make_video:
#     # Configure video generation parameters
#     video_params = {
#         "duration": 30.0,  # seconds
#         "fps": 30,
#         "show_skeleton": True,
#         "keypoint_radius": 3,
#         "line_thickness": 2,
#     }
#
#     # Generate video with pose overlay
#     video_path = (PoseV2() & pose_selection_key).make_video(**video_params)
#     print(f"✅ Video generated: {video_path}")
# else:
#     print("⚠️ Complete pose estimation first before generating videos")
# ```
#
# ### Video Quality Optimization
#
# ```python
# # For best quality/size balance
# optimal_params = {
#     "fps": 30,                    # Standard framerate
#     "compression": "h264",        # Standard codec
#     "quality": "medium",          # Balance size/quality
#     "resolution": (1280, 720)     # HD ready
# }
#
# # For fastest generation
# fast_params = {
#     "fps": 15,                    # Lower framerate
#     "resolution": (640, 480),     # Lower resolution
#     "compression": "fast"         # Fast encoding
# }
# ```
#
# **💡 Tip**: Test with short durations (5-10 seconds) before generating full-length videos.

# %% [markdown]
# # Reference

# %% [markdown]
# ## Next Steps
#
# - **Linearization**: Convert 2D position to 1D track position (notebook 24)
# - **Decoding**: Use position for neural decoding (notebooks 41-42)
# - **Custom Analysis**: Work directly with fetched DataFrames
#
# For questions, see the [Spyglass documentation](https://lorenfranklab.github.io/spyglass/)
# or open a discussion on [GitHub](https://github.com/LorenFrankLab/spyglass/discussions).
#

# %% [markdown]
# ## Troubleshooting <a id="Troubleshooting"></a>
#

# %% [markdown]
# <details>
# <summary><b>Model Import</b> (Click to expand)</summary>
#
# #### **Model Import Issues**
#
# **"Permission denied" or "Access forbidden"**
# - Verify database user permissions - Ask admin to add body parts
# - Check if you're connected to the right database
# - Ensure you have insert/update privileges
# - Contact your database administrator
#
# **"No sessions found matching video paths"**
# - Register your session first: `insert_sessions('your_file.nwb')`
# - Check video file paths in DLC config match VideoFile entries
# - Use bootstrap function for tutorials (NOT production)
#
# **"Model import failed" - DLC projects**
# - Verify the DLC project has completed training
# - Check that your models directory exists with snapshots
# - Ensure `config.yaml` path is correct
# - Try: `ls path/to/your/project/dlc-models/`
#
# **"Model import failed" - NWB files**
# - Verify NWB file contains ndx-pose data
# - Check file permissions and accessibility
# - Ensure ndx-pose extension is installed: `pip install ndx-pose`
#
# </details>
#
# <details>
# <summary><b>Pose Estimation</b> (Click to expand)</summary>
#
# #### **Pose Estimation Issues**
#
# **"CUDA out of memory"**
# ```python
# # Switch to CPU processing
# PoseEstimParams.insert_params(
#     params={"device": "cpu", "batch_size": 4},
#     params_id="cpu_small"
# )
# ```
#
# **"No h5 output files found"**
# - For tutorials: This is expected in demo mode
# - For real analysis: Use `task_mode="trigger"` for automatic inference
# - Check `output_dir` exists and contains `.h5` files if using `task_mode="load"`
#
# **"Inference taking too long"**
# - Reduce batch size: `batch_size: 4` or `batch_size: 1`
# - Use GPU if available: `device: "cuda"`
# - Consider shorter video clips for testing
#
# </details>
#
# <details>
# <summary><b>Parameters</b> (Click to expand)</summary>
#
# #### **Parameter Configuration Issues**
#
# **"Bodypart not found"**
# - Check available bodyparts:
#   `(Skeleton.BodyPart() & {"skeleton_id": your_id}).fetch()`
# - Verify bodypart names match exactly (case-sensitive)
# - Use `PoseParams.insert_default()` as fallback
#
# **"PoseParams insertion failed"**
# - Parameter set may already exist (check with `PoseParams()`)
# - Verify JSON parameter format is valid
# - Check that referenced bodyparts exist
# - Use `skip_duplicates=True` to avoid conflicts
#
# </details>
#
# <details>
# <summary><b>Data Processing</b> (Click to expand)</summary>
#
# #### **Data Processing Issues**
#
# **"PoseV2 processing failed"**
# - Ensure PoseEstim data exists first
# - Check that video group is linked to a valid session
# - Verify processing parameters are valid
# - Try with default parameters first
#
# **"Empty or invalid DataFrame"**
# - Check pose estimation completed successfully
# - Verify likelihood thresholds aren't too strict
# - Look for data in PoseEstim table: `PoseEstim() & your_key`
# - Check time ranges and video duration
#
# </details>

# %% [markdown]
# ## Guides

# %% [markdown]
# <details>
# <summary><b>V1 → V2 Migration Guide</b> (Click to expand)</summary>
#
# ### Migration Guide
#
# #### Table and Naming Changes
#
# Position V2 significantly streamlines the table structure compared to V1. i
# Here's a comprehensive migration mapping:
#
#
# | V1 | V2 | Notes |
# |---|---|---|
# | `BodyPart` | `BodyPart` | Reclassed from `Manual` → `Lookup` |
# | `DLCProject` | `Skeleton` | Project's body part set → explicit skeleton graph |
# | — | `Skeleton.BodyPart` | Part table; no V1 equivalent |
# | `DLCModelTrainingParams` | `ModelParams` | |
# | `DLCModelTrainingSelection` | `ModelSelection` | |
# | `DLCModelTraining` | `Model` | |
# | `DLCModelInput` | `ModelParams` | Merged into params |
# | `DLCModelSource` | `ModelParams` | Merged into params |
# | `DLCModelParams` | `ModelParams` | |
# | `DLCModelSelection` | `ModelSelection` | |
# | `DLCModel` | `Model` | |
# | `DLCModelEvaluation` | `Model` |  |
# | `DLCPoseEstimationSelection` | `PoseEstimSelection` | |
# | `DLCPoseEstimation` | `PoseEstim` | |
# | — | `PoseEstimParams` | New; separates inference params (device, batch_size) |
# | `DLCSmoothInterpParams` | `PoseParams` | Consolidated into `smoothing` sub-dict |
# | `DLCCentroidParams` | `PoseParams` | Consolidated into `centroid` sub-dict |
# | `DLCOrientationParams` | `PoseParams` | Consolidated into `orient` sub-dict |
# | `DLCSmoothInterpSelection` | `PoseSelection` | |
# | `DLCCentroidSelection` | `PoseSelection` | |
# | `DLCOrientationSelection` | `PoseSelection` | |
# | `DLCSmoothInterpCohortSelection` | `PoseSelection` | Cohort concept eliminated |
# | `DLCSmoothInterpCohort` | `PoseV2` | Cohort concept eliminated |
# | `DLCSmoothInterp` | `PoseV2` | |
# | `DLCCentroid` | `PoseV2` | |
# | `DLCOrientation` | `PoseV2` | |
# | `DLCPosSelection` | `PoseSelection` | |
# | `DLCPosV1` | `PoseV2` | |
# | `DLCPosVideoParams` | `VidFileGroup` | |
# | `DLCPosVideoSelection` | `VidFileGroup` | |
# | `DLCPosVideo` | `PoseV2.make_video` | No longer stored as table |
# | `TrodesPosParams` | — | No V2 equivalent |
# | `TrodesPosSelection` | — | No V2 equivalent |
# | `TrodesPosV1` | — | No V2 equivalent |
# | `TrodesPosVideo` | — | No V2 equivalent |
# | `ImportedPose` | `ImportedPose` | Poses ingested from external NWB files |
#
#
# #### Key consolidations in V2
#
# - `DLCCentroidParams` + `DLCOrientationParams` + `DLCSmoothInterpParams`
#     → single `PoseParams` with three sub-dicts
# - `DLCModelInput` + `DLCModelSource` + `DLCModelParams` → single `ModelParams`
# - Cohort pattern (`DLCSmoothInterpCohort*`) eliminated; `PoseV2`
#     handles multi-part poses directly
# - Trodes position and video output tables have no V2 counterparts as V2 is
#     focused on pose estimation from video, not direct position data
#
# #### 🔧 **Key Field Changes**
#
#
# | V1 Field | V2 Field | Type Change |
# |----------|----------|-------------|
# | `dlc_model_name` | `model_id` | More generic naming |
# | `dlc_model_params_name` | `model_params_id` | Consistent ID pattern |
# | `pose_estimation_task` | `task_mode` | Clearer terminology |
# | `pose_estimation_output_dir` | `output_dir` | Simplified naming |
# | `smooth_interp_params_name` | `pose_params_id` | Unified parameter naming |
#
#
# #### 📁 **File Organization**
#
# ```
# V1 Structure:
# src/spyglass/position/v1/
# ├── position_dlc_project.py         # Project management
# ├── position_dlc_training.py        # Model training
# ├── position_dlc_pose_estimation.py # Inference
# ├── position_dlc_cohort.py          # Batch processing
# ├── dlc_utils.py                    # Utilities
# └── dlc_reader.py                   # File I/O
#
# V2 Structure:
# src/spyglass/position/v2/
# ├── train.py                     # Model training & management
# ├── estim.py                     # Pose estimation & processing
# └── video.py                     # Video file management
# src/spyglass/position/utils/     # Shared utilities
# ├── dlc_io.py                    # DLC file parsing
# ├── validation.py                # Shared validation utilities
# ├── tool_strategies.py           # Multi-tool support
# └── ...                          # Other shared utilities
# ```
#
# #### 🔍 **Method Equivalents**
#
#
# | V1 Method | V2 Method | Notes |
# |-----------|-----------|-------|
# | `DLCModel.create_dlc_model()` | `Model.train` | Unified training interface |
# | `DLCModel.import_dlc_model()` | `Model.load` | Tool-agnostic import |
# | `DLCPoseEstimation.run_estimation()` | `PoseEstim.run_inference()` | Simplified execution |
# | `DLCSmoothInterp.get_position()` | `PoseV2.fetch1_dataframe()` | Direct DataFrame access |
#
#
# #### ⚠️ **Breaking Changes**
#
# - **No backward compatibility**: V1 and V2 use separate schemas
# - **Different output formats**: V2 uses ndx-pose standardization
# - **Schema changes**: Field names and types may differ
# - **Workflow differences**: Simplified but non-interchangeable processes
#
# </details>
#
# <details><summary><b>JSON Parameter Support</b> (Click to Expand)</summary>
#
# ### JSON Parameters
#
# Position V2 supports **native JSON columns** for enhanced parameter querying
# capabilities:
#
# ```python
# # Example: ModelParams with JSON column (DataJoint v0.14+)
# @schema
# class ModelParams(SpyglassMixin, dj.Lookup):
#     definition = """
#     model_params_id: varchar(32)
#     ---
#     params: json  # ← Native JSON support!
#     """
#
# # Enhanced querying with dot notation (automatic):
# filtered = ModelParams & {'params.learning_rate': 0.001}
# batch_filtered = ModelParams & {'params.batch_size': [4, 8, 16]}
# nested_query = ModelParams & {'params.training.max_epochs': 100}
# ```
#
# **Benefits:**
# - 🔍 **Native database querying** - No custom iteration needed
# - 🚀 **Better performance** - Database-level filtering and indexing
# - 🛠️ **Built-in DataJoint support** - Automatic dot notation translation
# - 🌐 **Cross-language compatibility** - JSON is universally supported
#
# **Migration Note:** Existing blob parameter tables continue to work unchanged. i
# Consider JSON columns for new parameter tables requiring complex querying.
#
# </details>
#
# <details>
# <summary><b>SLEAP Support Status</b> (Click to expand)</summary>
#
# ### SLEAP Integration Roadmap
#
# Position V2 includes preliminary SLEAP support architecture but **SLEAP
# training is not yet functional**. Here's the current status:
#
# #### ✅ **Available Now**
#
# - **Import Support**: Can import pre-trained SLEAP models via `Model.load()`,
#     but not yet run inference.
# - **Data Loading**: Can load existing SLEAP NWB files using
#     `PoseEstim.load_from_nwb()`
#
# #### 🔄 **In Development**
#
# - **Training Pipeline**: The `train_model()` method raises
#    `NotImplementedError`
# - **Inference Integration**: SLEAP analysis integration with Position V2
#     workflows
#
# **Timeline**: Native SLEAP training support is planned for a future release.
#
# </details>

# %% [markdown]
# ## External Resources <a id="Resources"></a>
#
# ### Documentation & Guides
# - [Spyglass Documentation](https://lorenfranklab.github.io/spyglass/)
# - [Position V2 API Reference](https://lorenfranklab.github.io/spyglass/api/position/v2/)
# - [DeepLabCut Documentation](https://deeplabcut.github.io/DeepLabCut/)
# - [ndx-pose Extension](https://github.com/rly/ndx-pose)
# - [DataJoint Documentation](https://docs.datajoint.com/)
#
# ### Getting Help
# - [GitHub Issues](https://github.com/LorenFrankLab/spyglass/issues) \-
#     Bug reports and feature requests
# - [GitHub Discussions](https://github.com/LorenFrankLab/spyglass/discussions)
#     \- Questions and community support
# - [Frank Lab Website](https://franklab.ucsf.edu/) - Lab resources and contact
#     information
#
# ### Related Notebooks
# - [00_Setup.ipynb](./00_Setup.ipynb) - Initial Spyglass configuration
# - [02_Insert_Data.ipynb](./02_Insert_Data.ipynb) - DataJoint basics
# - [21_DLC.ipynb](./21_DLC.ipynb) - Legacy Position V1 pipeline
# - [26_Linearization.ipynb](./26_Linearization.ipynb) - Convert 2D → 1D position
# - [41_Decoding_Clusterless.ipynb](./41_Decoding_Clusterless.ipynb) \- Use
#     position for decoding
#

# %% [markdown]
#
# **🎉 Tutorial Complete!**
#
# You've learned the fundamentals of the Position V2 pipeline. For questions or feedback, please use the resources above.
