# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
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
# This is one notebook in a multi-part series on Spyglass.
#
# - To set up your Spyglass environment and database, see
#   [the Setup notebook](./00_Setup.ipynb)
# - For additional info on DataJoint syntax, including table definitions and
#   inserts, see
#   [the Insert Data notebook](./02_Insert_Data.ipynb)
# - For pose tracking with **SLEAP**, see
#   [the SLEAP notebook](./25_PositionV2_SLEAP_2D.ipynb) (runs in its own
#   `spyglass-sleap` environment)
#
# <details>
# <summary><b>For contributors</b></summary>
#
# If you may make a PR in the future, be sure to copy this notebook and use
# the `gitignore` prefix `temp` to avoid future conflicts.
#
# </details>
#
# **Position V2** is a streamlined pose-estimation pipeline built around a few
# main tables. It:
#
# - **Supports multiple tools**: DeepLabCut here, and SLEAP in notebook 25
# - **Flexible workflows**: train models in Spyglass or import pre-trained ones
# - **NWB-native storage**: uses the ndx-pose extension for standardized data
# - **Simplified processing**: a single PoseV2 table handles all post-processing
#
# This tutorial assumes you have already ingested an NWB session file. It covers:
#
# - **Primary path**: training a model from scratch within Spyglass
# - **Alternative path**: importing a pre-trained DeepLabCut model, or ingesting
#   results from another tool (e.g. SLEAP) via an ndx-pose NWB file
# - Running pose estimation, processing pose data (orientation, centroid,
#   smoothing), and retrieving/visualizing results
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
#     - Import any DLC project directly (`config.yaml`)
#     - Ingest results from other tools via ndx-pose NWB (`ImportedPose`)
# - [Pose Estimation](#PoseEstim) - Run inference on videos
#     - Configure inference parameters (e.g., batch size)
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
# - [External Resources](#Resources) - Documentation links
# - [Multi-Tool Support](#MultiTool) - SLEAP is supported (see notebook 25)
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

# See the Setup notebook (./00_Setup.ipynb) if the following fails:
print(dj.conn())

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

print("All imports successful!")

# %% [markdown]
# ### Check the environment
#
# Catches common dependency issues before training or inference — e.g. a
# conflicting deep-learning backend on the GPU. If anything is flagged, follow
# the printed fix (or rebuild from `environments/environment_dlc.yml`), then
# restart the kernel.

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
# 1. A skeleton is a collection of body parts, paired with training parameters.
# 2. A video group is a collection of one or more files or calibrations.
# 3. Training runs on a video group and produces a Model.
# 4. Pose estimation applies a model to a video group.
# 5. The final `PoseV2` table incorporates all secondary calculations, like
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
# The Pose Estimation and pose-processing sections below read the variables
# initialized here **regardless of which path you run**, so you can run either
# path without the other. Each path fills in `model_key`; the tutorial bootstrap
# fills in the video/session variables.

# %%
# The tutorial creates a few dummy Spyglass entries so it can run without a
# recorded session. That helper ships with the Spyglass source checkout (not the
# pip package), so this step needs a git clone of the repo. It is NOT needed for
# real analysis — register your own session instead (see the Setup notebook).
import sys

import spyglass

tests_dir = Path(spyglass.__file__).parents[2] / "tests" / "position" / "v2"
if not tests_dir.exists():
    raise FileNotFoundError(
        f"Tutorial helper directory not found: {tests_dir}\n"
        "This tutorial's example-data helpers live in the Spyglass source "
        "checkout, which is not included in the pip package. Clone the repo "
        "(https://github.com/LorenFrankLab/spyglass) and run from there, or "
        "substitute your own registered session and videos."
    )
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))
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
demo_output_dir = None

# %% [markdown]
# ## Path A: Train a New Model <a id="PathA"></a>
#
# **Goal**: Create a DLC project from videos in Spyglass, label frames, and
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
# **Before you start**: Path A trains a DeepLabCut model, so DeepLabCut must be
# installed. The check below fails early with a clear message rather than deep
# inside training.

# %%
try:
    import deeplabcut  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "DeepLabCut is required for Path A (training a new model) but is not "
        "installed in this environment. Install it (e.g. rebuild from "
        "`environments/environment_dlc.yml`) and restart the kernel, or use "
        "Path B to import a pre-trained model instead."
    ) from exc

# %% [markdown]
# ### Step 1 — Choose training videos
#
# Browse the `VideoFile` table to find recordings to train on. Each row is one
# video file registered in Spyglass.

# %%
# Inspect available videos — pick the ones to train on
VideoFile()

# %% [markdown]
# Select your training videos. Use `nwb_file_name` + `epoch` dicts to reference
# videos by session, or supply absolute paths:

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
# Body parts your model will track. Each name must already exist in the
# BodyPart table (admins can add new ones).
training_bodyparts = ["whiteLED", "tailBase"]

# %% [markdown]
# ### Step 3 — Bootstrap tutorial session & create project
#
# <details>
# <summary><b>Tutorial helper — not for production use</b></summary>
#
# `Model.create_project()` calls DLC to create a project folder and extract
# frames. Later, `Model.load()` calls `VidFileGroup.create_from_dlc_config()`
# internally, which needs a matching `Session` in the Spyglass database.
#
# **In production**, register your NWB session with `insert_sessions()` and
# ensure `VideoFile` rows exist *before* calling `Model.create_project()` or
# `Model.load()`.
#
# The `bootstrap_from_video_paths()` helper (from
# `tests/position/v2/make_example_dlc_project.py`) creates minimal dummy entries
# so this tutorial works without a recorded session.
#
# **On a shared database, please delete the dummy entries when done.**
#
# </details>

# %%
# Path-A tutorial imports. The bootstrap helper and shared-state variables come
# from "Shared state for both paths" above; here we add only the
# training-specific helpers.
import yaml
from make_example_dlc_project import make_dlc_project  # noqa: E402

# ── Tutorial: create a minimal example DLC project if none exists ─────────
demo_dlc_dir = Path.home() / "DeepLabCut" / "examples"
demo_config = (
    demo_dlc_dir / "tutorial_dlc-tutorial_dlc-2025-01-01" / "config.yaml"
)

print("demo_config:", demo_config)

if not demo_config.exists():
    print("Demo config not exist")
    demo_config = make_dlc_project(demo_dlc_dir)
    print(f"Created example DLC project: {demo_config}")
else:
    print(f"Using existing DLC project: {demo_config}")

config_path = Path(demo_config)

# ── Read the DLC config to get video paths ───────────────────────────────────
with open(config_path) as config_file:
    config = yaml.safe_load(config_file)
training_videos = list(config.get("video_sets", {}).keys())
project_name = Path(config.get("project_path", str(config_path.parent))).name

# ── Register a Spyglass session for those videos (tutorial only) ──────────────
print("Bootstrapping tutorial Spyglass session...")
nwb_file_name, inf_vid_path = bootstrap_from_video_paths(
    training_videos, nwb_stem=project_name
)
print(f"  nwb_file_name  : {nwb_file_name}")
print(f"  inf_vid_path   : {inf_vid_path}")

# Use the registered videos as our training list
training_video_list = [{"nwb_file_name": nwb_file_name, "epoch": 1}]


# %% [markdown]
# Now call `Model.create_project()` with a project name, your body parts, and
# the training videos. It sets up the DLC project on disk (creating the folder
# and extracting frames to label) and returns `config_path` and `skeleton_id`
# for the next steps.

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
# For the full V1 → V2 migration guide (including why V2 has no `DLCProject`
# table), see the Position Pipelines page:
# https://lorenfranklab.github.io/spyglass/latest/GettingStarted/POSITION/

# %% [markdown]
# ### Step 4 — Label frames (manual step)
#
# After `create_project()` finishes, DLC has extracted frames into
# `labeled-data/` inside the project folder. Label them with the DLC GUI or the
# napari plugin before training:
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
    "net_type": "resnet_50",
    "augmenter_type": "imgaug",
    "epochs": 1,
    "save_epochs": 1,
    "project_path": str(config_path.parent),
}
train_params_id = "path_a_demo_1epoch"

ModelParams.insert1(
    {
        "model_params_id": train_params_id,
        "params": train_params,
        "tool": "DLC",
        "skeleton_id": skeleton_id,
    },
    skip_duplicates=True,
)

# Create a VidFileGroup from the DLC config so ModelSelection can reference it.
training_vid_group_key = VidFileGroup.create_from_dlc_config(config_path)
training_vid_group_id = training_vid_group_key["vid_group_id"]

selection_key = {
    "model_params_id": train_params_id,
    "tool": "DLC",
    "vid_group_id": training_vid_group_id,
    "model_selection_id": train_params_id,  # reuse params id as selection id
    "parent_id": None,
}
ModelSelection.insert1(selection_key, skip_duplicates=True)

if len(Model & selection_key) > 0:
    print(f"Model '{train_params_id}' already exists — skipping training")
else:
    print("Starting training (1 epoch) ...")
    # `Model().train()` is the unified training verb: given a fresh
    # ModelSelection it runs training; given an existing Model it resumes
    # (see "Continue / resume training" below).
    Model().train(selection_key)
    print(f"Model '{train_params_id}' trained and saved")

model_key = (Model & selection_key).fetch1()
print(f"model_key: {model_key['model_id']}")

# %% [markdown]
# <details>
# <summary><b>Choosing the compute device</b> (Click to Expand)</summary>
#
# The best available device (GPU if present, otherwise CPU) is selected
# automatically for both training and inference. `device` is a **runtime**
# choice, not a stored/hashed parameter, so it never forks an
# otherwise-identical parameter set.
#
# On a machine with several GPUs, a bare `"cuda"` is resolved to the
# *least-loaded* device, preferring one that is completely unused. This
# matters on a shared cluster node: PyTorch reads a plain `"cuda"` as
# `cuda:0` specifically, so without this step every job on the machine
# would pile onto GPU 0 and fail with an out-of-memory error while the
# other cards sat idle. The chosen device is logged.
#
# To force a specific device, pass it to `populate` via `make_kwargs`:
#
# ```python
# Model().populate(selection_key, make_kwargs={"device": "cuda"})  # training
# PoseEstim().populate(estim_key, make_kwargs={"device": "cuda"})  # inference
# ```
#
# Prefer `"cuda"` over a hardcoded `"cuda:0"`. An explicit index is
# honored as given — which is the point of asking for one — but *which*
# GPU is free changes from run to run, so a pinned index silently
# reintroduces the same crowding problem. Name an index only when you
# genuinely need that card.
#
# </details>

# %% [markdown]
# ### Step 6 — Visualize training curves

# %%
if model_key:
    history = Model().get_traininghistory({"model_id": model_key["model_id"]})
    if len(history) > 0:
        Model().plot_traininghistory({"model_id": model_key["model_id"]})
    else:
        print("No training history yet (expected for a 1-epoch demo).")

# %% [markdown]
# ### Step 7 — Continue / resume training (optional) <a id="ContinueTraining"></a>
#
# A model that underfit need not be retrained from scratch. The same
# `Model().train()` verb resumes an existing model: pass a trained model's key
# plus how many additional `epochs` to run. Since a `Model` row already matches
# the key, `train()` loads the saved weights, trains further, inserts a new
# parent-linked model, and returns its key.

# %%
if model_key:
    resumed_key = Model().train(
        {"model_id": model_key["model_id"]},
        epochs=50,  # additional epochs to train beyond the current snapshot
    )
    print(f"Resumed training → new model_id: {resumed_key['model_id']}")

# %% [markdown]
# ## Path B: Import a Pre-Trained Model <a id="PathB"></a>
#
# **Goal**: Load an existing pre-trained model into Spyglass — import a
# DeepLabCut project from its `config.yaml` and create video file groups for
# analysis.
#
# > **Note**: Skip this section if you just completed Path A above.

# %% [markdown]
# Position V2 supports two import routes:
#
# 1. **DLC `config.yaml`** — import a DeepLabCut-trained model (the default
#    below), then run inference through the pipeline.
# 2. **Pre-computed pose (ndx-pose NWB)** — ingest results from another tool
#    (including **SLEAP**, or DLC `.h5` output) with `ImportedPose`; see the
#    pointer at the end of this section.

# %% [markdown]
# #### From DeepLabCut Project
#
# To import an already-trained DeepLabCut model, set `config_path` below to your
# `config.yaml`. If you don't have one, obtain it from the
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
# Path B default — import a pre-trained DeepLabCut project from its config.yaml.
# ────────────────────────────────────────────────────────────────────────────
# If you completed Path A, model_key is already set and this block is skipped.
#
# To import YOUR model, set `pathb_config_path` to your project's config.yaml.
# Otherwise the tutorial builds a small example DLC project so the import runs
# end to end.
import yaml
from make_example_dlc_project import make_dlc_project  # noqa: E402

if model_key is None:
    # ── Point this at your own trained DLC project to import it ──
    pathb_config_path = None  # e.g. Path("/path/to/your/config.yaml")

    if pathb_config_path is None:  # tutorial: build a small example project
        pathb_dlc_dir = Path.home() / "DeepLabCut" / "import_example"
        pathb_config_path = make_dlc_project(pathb_dlc_dir)

    pathb_config_path = Path(pathb_config_path)

    # Register a Spyglass session for the project's videos (tutorial only). In
    # production, run insert_sessions() on the real training session first.
    with open(pathb_config_path) as config_file:
        pathb_config = yaml.safe_load(config_file)
    pathb_videos = list(pathb_config.get("video_sets", {}).keys())
    pathb_project_name = Path(
        pathb_config.get("project_path", str(pathb_config_path.parent))
    ).name
    nwb_file_name, inf_vid_path = bootstrap_from_video_paths(
        pathb_videos, nwb_stem=pathb_project_name
    )

    model_key = Model().load(pathb_config_path)
    config_path = pathb_config_path
    print(f"Imported DLC model: {model_key['model_id']}")

# %% [markdown]
# `Model.load()` reads the DLC project's config and latest trained snapshot and
# registers a ready-to-use `Model` (plus its `Skeleton`, `ModelParams`, and a
# `VidFileGroup` linking the project's videos to a Spyglass session).
#
# Body-part names are reconciled with the curated `BodyPart` table on import, so
# spelling variants (e.g. `green_led` vs `greenLED`) are handled automatically.
# To also rewrite your DLC project's `config.yaml` to the canonical spelling,
# pass `Model().load(config_path, normalize_names=True)`; the original config is
# saved to a timestamped `config.yaml.<ts>.bak` first. Body parts not in
# `BodyPart` with no canonical match still require an admin to add them.
#

# %% [markdown]
# <details>
# <summary><b>Migrating a model from Position V1</b></summary>
#
# If your pre-trained model already lives in the legacy Position **V1** schema,
# import it into V2 with `Model.import_from_v1()`. It looks up the V1 model's
# project path and delegates to `Model.load()` on the resolved `config.yaml`:
#
# ```python
# v1_key = {
#     "project_name": "Wtrack_WhiteLED",
#     "dlc_model_name": "Wtrack_WhiteLED_ms_stim_wtrack_00",
#     "dlc_model_params_name": "default",
# }
# model_key = Model().import_from_v1(v1_key)
# ```
#
# This is only for models originally trained through the V1 pipeline. For any
# other pre-trained DLC project, use the `Model.load()` default above.
#
# </details>
#

# %% [markdown]
# #### Validate model (Path B)

# %%
if model_key and not Model() & model_key:
    raise ValueError(f"Model entry not found : {model_key}")

if model_key:
    model_params = (
        ModelParams() & {"model_params_id": model_key["model_params_id"]}
    ).fetch1()
    skeleton_id = model_params.get("skeleton_id")
    if not (Skeleton() & {"skeleton_id": skeleton_id}).fetch1("KEY"):
        raise ValueError(f"Skeleton not found for model: {model_key}")
    if vid_group_id := model_key.get("vid_group_id"):
        if not VidFileGroup() & {"vid_group_id": vid_group_id}:
            raise ValueError(f"Video group not found: {vid_group_id}")
    training_vid_group_id = model_key["vid_group_id"]
    print("Path B model import validated")

# %% [markdown]
# #### Inference video
#
# The Path B default already registered an inference clip (from the imported
# project's videos) into `inf_vid_path`. If it is still `None`, the cell below
# derives one from the imported project's videos.
#
# To use your own video instead, set `inf_vid_path` before running this cell:
#
# ```python
# inf_vid_path = Path("/path/to/your/video.mp4")
# ```

# %%
if inf_vid_path is None and config_path is not None:
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    videos = list(config.get("video_sets", {}).keys())
    if videos:
        print("Creating a short inference clip from the project's videos...")
        nwb_file_name, inf_vid_path = bootstrap_from_video_paths(
            videos, nwb_stem=Path(config_path).parent.name
        )
        print(f"  nwb_file_name : {nwb_file_name}")
        print(f"  inf_vid_path  : {inf_vid_path}")
    else:
        print(
            "No videos found in the imported project. "
            "Set inf_vid_path manually before running Pose Estimation."
        )

# %%
if skeleton_id:
    fig = (Skeleton & {"skeleton_id": skeleton_id}).show_skeleton()
    plt.show()

# %% [markdown]
# #### Pointer: pose already computed by another tool (ndx-pose NWB)
#
# If your results already exist as an **ndx-pose NWB file** (from SLEAP, DLC, or
# any tool), skip training/inference here and use `ImportedPose` — the canonical
# entry point for pre-computed pose:
# `insert_sessions("pose.nwb")` then `ImportedPose().insert_from_nwbfile("pose.nwb")`
# (add `import_to_v2=True` to also register the V2 model tables). DLC `.h5`
# output first needs converting with `deeplabcut.analyze_videos_converth5_to_nwb`
# (requires `dlc2nwb`). For SLEAP end-to-end, see
# [notebook 25](./25_PositionV2_SLEAP_2D.ipynb). Note `Model.load()` does not
# accept NWB files.

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
# **Goal**: Run pose inference on a video using the model, configuring inference
# parameters (batch size), setting up an estimation task, and validating
# results.

# %% [markdown]
# #### Pointer: already ran DLC inference?
#
# If you have `.h5`/`.csv` output on disk, set `task_mode='load'` in
# `PoseEstimSelection` (Step 2 below) to read it in rather than re-running
# inference. Otherwise, run inference through the pipeline as follows.

# %% [markdown]
# #### Running Inference via the Pipeline
#
# Inference in V2 follows a three-step Spyglass pattern:
#
# 1. **`PoseEstimParams`** — name a set of inference parameters
#     (batch size, etc.)
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
# Here `params_hash` uniquely identifies the parameter set; inserting a new
# entry whose `params` match an existing row raises an error. The compute device
# (GPU if present, else CPU — and the least-loaded GPU when there are several)
# is chosen automatically at run time — you don't set it here. These parameters
# cover scientific settings like `batch_size`.

# %%
params_id = "batch8"

params_result = PoseEstimParams.insert_params(
    params={"batch_size": 8},
    params_id=params_id,
    skip_duplicates=True,
)
print(f"Inserted PoseEstimParams '{params_id}'")

PoseEstimParams & {"pose_estim_params_id": params_id}

# %% [markdown]
# ##### Step 2 — Estimation task (`PoseEstimSelection`)
#
# Pose estimation uses **two separate `VidFileGroup` entries**:
#
# - **Training group** (`training_vid_group_id`): created by `Model.load` and
#   linked to `ModelSelection`. Contains the original labeled videos used for
#   training. Used by `get_nwb_file()` to resolve the parent session.
# - **Inference group** (`inf_vid_group_id`): created here for the video(s) to
#   run inference on. Linked to `PoseEstimSelection`.
#
# This structure supports multi-camera recordings.

# %%
inf_vid_group_key = None
estim_key = None

if not (inf_vid_path and model_key):
    raise ValueError("Missing video or model - skipping PoseEstimSelection")

# Create the inference video group (reusing params_id from Step 1).
inf_vid_group_key = VidFileGroup().insert1(
    {
        "description": f"Inference video for {model_key['model_id']}",
        "files": [inf_vid_path],
    },
    skip_duplicates=True,
)

estim_selection_key = {
    "model_id": model_key["model_id"],
    "vid_group_id": inf_vid_group_key["vid_group_id"],
    "pose_estim_params_id": params_id,
    "task_mode": "load" if demo_output_dir else "trigger",
    "output_dir": str(demo_output_dir) if demo_output_dir else "",
}

PoseEstimSelection().insert1(estim_selection_key, skip_duplicates=True)

# Restrict to the primary-key fields needed for populate.
estim_key = {
    k: v
    for k, v in estim_selection_key.items()
    if k in ["model_id", "vid_group_id", "pose_estim_params_id"]
}
print(f"Created estimation selection: {estim_key}")

# %%
# Inspect the video groups
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

selection_entry = (PoseEstimSelection() & estim_key).fetch1()
task_mode = selection_entry.get("task_mode", "trigger")
output_dir = selection_entry.get("output_dir", "")
pose_df = None

PoseEstim.populate(estim_key, display_progress=True)

pose_df = (PoseEstim() & estim_key).fetch1_dataframe()
print(pose_df.head())

# %% [markdown]
# <details>
# <summary><b>Troubleshooting Pose Estimation</b></summary>
#
# ### Common Issues & Solutions
#
# #### **Error**: "No h5 output files found"
# **Cause**: Demo mode vs. real inference mismatch
# **Solution**:
# - This is expected in tutorial demo mode
# - For real analysis, ensure `task_mode='trigger'` for automatic inference
# - Or provide existing DLC `.h5` files with `task_mode='load'`
#
# #### **Error**: "CUDA out of memory"
# **Cause**: GPU memory insufficient
# **Solution**: Reduce `batch_size` so each inference step uses less GPU memory:
# ```python
# PoseEstimParams.insert_params(
#     params={"batch_size": 4},
#     params_id="batch4"
# )
# ```
#
# #### **Error**: "Model not found" or "VidFileGroup not found"
# **Cause**: Model import incomplete
# **Solution**: Run the validation checkpoint above to diagnose
#
# #### **Error**: "KeyError: analysis_file_name"
# **Cause**: NWB file path resolution issue
# **Solution**: Ensure video groups are linked to registered sessions
#
# #### **Warning**: "Low likelihood values"
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
# **Goal**: Configure how raw pose data is processed into final trajectories —
# using default parameters or a custom configuration matched to your setup.
#
# `PoseParams` stores configuration for:
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
# interpolated over before smoothing. Typical values: `0.9` (strict) → `0.5`
# (permissive). Start with `0.9` and lower if too many frames are dropped.

# %% [markdown]
# #### View Available Parameter Sets
#

# %%
PoseParams()

# %% [markdown]
# Inspect the full JSON contents of any named parameter set:

# %%
# Pretty-print a named PoseParams entry (change to any entry shown above).
params_to_inspect = "default"

if PoseParams() & {"pose_params_id": params_to_inspect}:
    PoseParams().print_params(params_to_inspect)
else:
    print(
        f"No entry named '{params_to_inspect}'. "
        "Run PoseParams.insert_default() first, or change params_to_inspect."
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
# Build a custom parameter set from two of the model's body parts.
if not model_key:
    raise ValueError("No model key available - cannot create custom PoseParams")

model_params = (ModelParams() & model_key).fetch1()
skeleton_id = model_params["skeleton_id"]
bp_tbl = Skeleton.BodyPart() & {"skeleton_id": skeleton_id}
skeleton_parts = bp_tbl.fetch("bodypart")
bodypart1, bodypart2 = skeleton_parts[0], skeleton_parts[1]

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

print("Created pose parameters 'tutorial_custom'")

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
# **Goal**: Process raw pose estimates into final position trajectories — link
# estimation results to processing parameters, run the pipeline (orientation,
# centroid, velocity), and validate the result.

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
print(f"Processed data: {processed_df.shape[0]} timepoints")
print(processed_df.head())

# %% [markdown]
# ### Finding entries by recording session
#
# `PoseV2`'s key carries `vid_group_id`, not `nwb_file_name`/`epoch` — a video
# group is a *set* of videos, so the session does not ride down the primary
# key the way it does for `DLCPosV1`. Use `fetch_by_epoch()`, which takes any
# restriction valid on `TaskEpoch`:
#
# ```python
# # A whole session
# PoseV2().fetch_by_epoch({"nwb_file_name": "my_session_.nwb"})
#
# # One epoch, straight to a dataframe
# PoseV2().fetch_by_epoch(
#     {"nwb_file_name": "my_session_.nwb", "epoch": 1}
# ).fetch1_dataframe()
#
# # Composes with other restrictions
# (PoseV2 & {"pose_params_id": "default"}).fetch_by_epoch(
#     {"nwb_file_name": "my_session_.nwb"}
# )
# ```
#
# Equivalent to the underlying join `PoseV2 * (VidFileGroup.File & restriction)`,
# but deduplicated: a multi-camera group has one `File` row per camera, so the
# raw join repeats each entry once per matching video.
#
# > **Careful:** `PoseV2 & {"nwb_file_name": ...}` does *not* work and does not
# > raise — DataJoint silently ignores a restriction on an attribute the table
# > does not have, so you get back *every* row.

# %% [markdown]
# `PoseV2.populate()` runs the processing pipeline, which performs:
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
# ### Validation Checkpoint: Data Processing
#
# Verify the processing worked correctly:

# %%
if processed_df is None:
    raise ValueError("No processed data - ensure PoseV2.populate() completed")

# `PoseV2.fetch1_dataframe` builds a fixed set of columns (position_x/y,
# orientation, velocity_x/y, speed) by construction, so no manual column
# check is needed here — the method's contract (and its unit tests) guarantee
# them.

# Data summary
time_range = processed_df.index[-1] - processed_df.index[0]
mean_speed = processed_df["speed"].mean()

if not pose_selection_key or not (PoseV2() & pose_selection_key).fetch1("KEY"):
    raise ValueError("PoseV2 entry not found for selection key")

print("Validation passed")
print(f"Duration: {time_range:.1f}s, Mean speed: {mean_speed:.1f} cm/s")

# %% [markdown]
# ## Data Analysis & Retrieval <a id="FetchData"></a>

# %% [markdown]
# **Goal**: Access processed position data for analysis and visualization —
# retrieve it as pandas DataFrames or raw NWB objects and generate trajectory
# and time-series plots.

# %% [markdown]
# ### [Visualization](#TableOfContents) <a id="Visualization"></a>
#

# %% [markdown]
# #### Trajectory Plot
#

# %%
if processed_df is None:
    raise ValueError("No processed data available for plotting")

# `PoseV2.plot_trajectory()` is a reusable helper: it fetches the processed
# dataframe and scatter-plots the trajectory colored by speed (a 3D scatter
# if the data has a position_z column). It returns the Matplotlib axes for
# further customization.
ax = (PoseV2() & pose_selection_key).plot_trajectory()
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
# The full workflow for training a new model from scratch (video selection,
# skeleton definition, frame extraction, training) is covered in
# **[Path A](#PathA)** at the top of this notebook. The steps below (Model
# Evaluation, Video Generation) assume you have a `model_key` from either path.

# %% [markdown]
# ## Model Evaluation <a id="ModelEvaluation"></a>
#

# %% [markdown]
#
# **Goal**: Evaluate model performance and visualize training progress. Loss
# curves reveal how well the model learned, whether it is overfitting (training
# vs. validation divergence), and how different configurations compare.
#
# ### Check Model Evaluation Availability

# %%
if model_key is None:
    raise ValueError("No model key available - cannot check evaluation support")

model_params = (ModelParams() & model_key).fetch1()
model_tool = model_params.get("tool", "Unknown")
traininghistory = Model().get_traininghistory(model_key)
has_traininghistory = traininghistory is not None
evaluation_supported = model_tool.upper() == "DLC"

# %% [markdown]
# ### Generate Training Curves
#
# Visualize training progress and detect potential overfitting:

# %%
if not (has_traininghistory and evaluation_supported):
    raise ValueError("Model evaluation unavailable.")

if len(traininghistory) < 1:
    raise ValueError("Not enough training history data available.")

# Built-in plotting method with detailed diagnostics enabled
fig = Model().plot_traininghistory(
    model_key,
    save_path=None,
    detailed=True,
)

# %% [markdown]
# If the loss curve suggests the model underfit (still decreasing, or accuracy
# too low), you don't need to start over — resume training with more epochs. See
# [Step 7 — Continue / resume training](#ContinueTraining):
# `Model().train({"model_id": model_key["model_id"]}, epochs=50)`.

# %% [markdown]
# ## Video Generation <a id="VideoGeneration"></a>
#

# %% [markdown]
# Once `PoseV2` is populated, render an annotated video (keypoints, centroid,
# and orientation overlaid on the source video) with
# `(PoseV2() & pose_selection_key).make_video()`. Use `percent_frames` to render
# only a fraction of the video — start small to preview before rendering the
# whole clip.

# %%
if processed_df is not None and pose_selection_key is not None:
    # Render the first 10% of frames as a quick preview.
    video_path = (PoseV2() & pose_selection_key).make_video(percent_frames=0.1)
    print(f"Video generated: {video_path}")
else:
    print("⚠️ Complete pose estimation first before generating videos")

# %% [markdown]
# # Reference

# %% [markdown]
# ## Next Steps
#
# - **Linearization**: Convert 2D position to 1D track position (notebook 26)
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
# # Reduce batch_size to use less GPU memory
# PoseEstimParams.insert_params(
#     params={"batch_size": 4},
#     params_id="batch4"
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
# - Run on a machine with a GPU (the best available device is used
#   automatically; on a multi-GPU host the least-loaded card is picked)
# - Consider shorter video clips for testing
#
# **"CUDA out of memory"**
# - If you pinned a device (`"cuda:0"`), drop the index and pass `"cuda"` so
#   Spyglass can pick a free card — a hardcoded index does not move when that
#   GPU is busy
# - If every GPU is genuinely full, populate raises before inference starts,
#   listing free memory per device: wait for another job, or use a `device:
#   "cpu"` parameter set
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
# For the full V1 → V2 migration guide (table mapping, consolidations, and
# intentional differences), see the Position Pipelines page:
# https://lorenfranklab.github.io/spyglass/latest/GettingStarted/POSITION/
#
# <details><summary><b>JSON Parameter Support</b> (Click to Expand)</summary>
#
# ### JSON Parameters
#
# Parameter tables (`ModelParams`, `PoseParams`, `PoseEstimParams`) store their
# settings in native JSON columns, so you can query any sub-field directly with
# dot notation — no custom iteration needed:
#
# ```python
# ModelParams & {'params.learning_rate': 0.001}
# ModelParams & {'params.batch_size': [4, 8, 16]}   # IN-list query
# PoseParams & {'smoothing.likelihood_thresh': 0.1}
# ```
#
# The database does the filtering (and can index it), and JSON is portable
# across languages. Existing blob parameter tables continue to work unchanged.
#
# </details>
#
# <details>
# <summary><b>Multi-Tool Support: SLEAP</b> (Click to expand)</summary>
#
# ### SLEAP is supported
#
# SLEAP is fully supported in Position V2 — training, continued training, and
# inference — through a dedicated `SLEAPStrategy`. Because SLEAP and DeepLabCut
# have conflicting dependencies, SLEAP work lives in its own notebook and
# environment:
#
# - **Notebook**: [`25_PositionV2_SLEAP_2D`](./25_PositionV2_SLEAP_2D.ipynb)
# - **Environment**: `spyglass-sleap` (separate from this notebook's DLC env)
#
# The V2 table structure (`Skeleton`, `ModelParams`, `Model`,
# `PoseEstimSelection`, `PoseEstim`, `PoseV2`) is tool-agnostic and identical
# across DLC and SLEAP; only the training/inference backend differs. Pose data
# from any tool can also be ingested as an ndx-pose NWB via `ImportedPose`.
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
# **Tutorial Complete!**
#
# You've learned the fundamentals of the Position V2 pipeline. For questions or
# feedback, use the resources above.
