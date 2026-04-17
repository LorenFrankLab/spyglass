# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: pv2
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Position V2 - Streamlined Pose Estimation Pipeline
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
# - **Multi-tool support**: Works with DLC with planned expansion to include SLEAP
# - **Flexible workflows**: Import pre-trained models or train new ones
# - **NWB-native storage**: Uses ndx-pose extension for standardized data
# - **Simplified processing**: Single PoseV2 table handles all post-processing
#
# This tutorial will walk through:
#
# - Importing a trained model (DLC or ndx-pose)
# - Running pose estimation on videos
# - Processing pose data (orientation, centroid, smoothing)
# - Retrieving and visualizing results
#

# %% [markdown]
# ### Table of Contents

# %% [markdown]
# #### Core Tutorial (Essential)
#
# - [Setup](#Setup) - Environment configuration
#     - Load packages & configure environment
#     - Connect to database
# - [Model Import](#Model) - Import pre-trained models
#     - Import DLC project (`config.yaml`) or NWB file (`ndx-pose`)
#     - Verify skeleton and bodyparts
#     - Create video file groups
# - [Pose Estimation](#PoseEstim) - Run inference on videos
#     - Configure inference parameters (e.g., device, batch size)
#     - Set up estimation task
#     - Run inference and validation
# - [Parameters](#PoseParams) - Configure processing settings
#     - Define orientation calculation
#     - Set centroid method
#     - Configure smoothing parameters
# - [Data Processing](#PoseV2) - Calculate final pose data
#     - Run pose processing pipeline (velocity, orientation, centroid, smoothing)
#     - Validate results
# - [Analysis](#FetchData) - Retrieve and visualize results
#     - Fetch processed data
#     - Generate trajectory plots
#     - Analyze time series
# - [MVP Training Demo](#MVPTraining) - Quick training demonstration
#     - Minimal model training setup
#     - Training loss curve visualization
#     - Understand training workflow
#
# #### ADVANCED FEATURES (Optional)
#
# - [Training New Models](#TrainingWorkflow) - Custom model development
# - [Model Evaluation](#ModelEvaluation) - Training curves and performance metrics
# - [Video Generation](#VideoGeneration) - Create annotated outputs
#
# #### REFERENCE
#
# - [🚨 Troubleshooting](#Troubleshooting) - Common issues & solutions
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
import datajoint as dj
import os
import warnings

from pathlib import Path

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

dj.config.load("../dj_local_conf_pv2.json")  # TODO: CHANGE BEFORE MERGE

from spyglass.common import Session, VideoFile
from spyglass.position.v2 import video, train, estim

from spyglass.position.v2 import (
    BodyPart,
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
# ### Diagram

# %%
dj.Diagram(video) + dj.Diagram(train) + dj.Diagram(estim)

# %% [markdown]
# For a refresher on reading diagrams, see [this doc](https://docs.datajoint.com/how-to/read-diagrams/)
#
# A few key points before diving in:
#
# 1. Training starts with a skeleton, representing a collection of body parts.
# 2. A skeleton in specified with model training parameters.
# 3. A video group is a collection of one or more files or calibrations
# 4. Training takes place on a video group and results in a Model
# 5. Pose estimation applies a given model to a given video group
# 6. The final 'PoseV2' table incorporates all secondary calculations, like
#     orientation and smoothing.

# %% [markdown]
# ## Model Import - Existing Models <a id="Model"></a>

# %% [markdown]
# **Goal**: Load a pre-trained pose estimation model into Spyglass
#
# **What you'll accomplish**:
# - Import a DeepLabCut project or NWB file
# - Understand skeleton and bodypart organization
# - Create video file groups for analysis

# %% [markdown]
# For most experiments, you'll start with an existing trained model rather than
# training from scratch. Position V2 supports different import methods:
#
# 1. **DLC config.yaml**: Import models trained with DeepLabCut
# 2. **NWB file**: Import models from any ndx-pose compatible tool
# 3. **SLEAP config**: `NotYetImplemented`

# %% [markdown]
# Let's start by looking at the Model table:
#

# %%
Model()

# %% [markdown]
# #### From DeepLabCut Project
#
# If you have a DeepLabCut model already trained, you can import it by changing
# the following to the path to your `config.yaml`. If not, use the following
# codeblock to download and set up an example project.
#
# ```bash
# cd /your/desired/path
# git clone https://github.com/DeepLabCut/DeepLabCut/
# python ./DeepLabCut/examples/testscript.py
# ```

# %% [markdown]
# > **Session prerequisite** — `import_model()` calls
# > `VidFileGroup.create_from_dlc_config()` internally, which requires a
# > Spyglass `Session` whose `nwb_file_name` stem appears in the DLC config's
# > video paths. In production, run `insert_sessions('your_training_session.nwb')`
# > first.
# >
# > For this tutorial, `_tutorial_bootstrap_dlc_session()` (defined below)
# > creates minimal dummy entries so the import can proceed without a recorded
# > session.
# >
# > If you use this on a shared database **PLEASE DELETE ENTRIES** when you're done.
#

# %%
# NOT IMPORTANT, JUST A HELPER FOR THE TUTORIAL.

import uuid
from datetime import datetime

import yaml


def _tutorial_bootstrap_dlc_session(config_path):
    """Create minimal dummy Spyglass session entries for a DLC project.

    Inserts Nwbfile, Session, Task, IntervalList, TaskEpoch, and VideoFile
    entries derived from the DLC config so that create_from_dlc_config()
    can find a session match and link the VidFileGroup to real VideoFile rows.

    For tutorial / development use only. In production, register your
    session with insert_sessions() before calling Model.load().
    """
    import shutil
    import subprocess

    from spyglass.common import (
        IntervalList,
        Nwbfile,
        Session,
        Task,
        TaskEpoch,
        VideoFile,
    )
    from spyglass.settings import raw_dir, video_dir

    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    project_path = Path(cfg.get("project_path", str(config_path.parent)))
    nwb_stem = project_path.name  # e.g. 'TEST-Alex-2025-09-08'
    nwb_file_name = f"{nwb_stem}_.nwb"
    training_video_paths = list(cfg.get("video_sets", {}).keys())
    nwb_file_path = Path(raw_dir) / nwb_file_name

    # Copy minirec as a valid NWB template so AnalysisNwbfile.create() can
    # open it when storing pose estimation results.
    if not nwb_file_path.exists() or nwb_file_path.stat().st_size == 0:
        minirec = Path(raw_dir) / "minirec20230622_.nwb"
        shutil.copy2(str(minirec), str(nwb_file_path))
        print(f"Copied minirec as placeholder NWB: {nwb_file_path}")

    # Create a 1-second inference clip directly in video_dir.
    # Inference videos are assumed to live in video_dir; training videos
    # stay in the DLC project folder (their absolute paths are stored in
    # the VideoFile.path column so get_abs_path() can find them).
    src_video = Path(training_video_paths[0])
    inf_vid_name = f"example_inference{src_video.suffix}"
    inf_vid_path = Path(video_dir) / inf_vid_name
    if not inf_vid_path.exists():
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
        print(f"Created inference clip: {inf_vid_path}")

    # All video paths: training (DLC project folder) + inference (video_dir).
    # path= is set explicitly so VideoFile.get_abs_path() short-circuits
    # via the stored path without needing an NWB lookup.
    all_video_paths = training_video_paths + [str(inf_vid_path)]

    nwb_key = dict(nwb_file_name=nwb_file_name)
    interval_key = dict(nwb_key, interval_list_name="tutorial_epoch_1")
    now = datetime.now()
    ins = dict(allow_direct_insert=True, skip_duplicates=True)

    # Check existence before insert: DataJoint uploads external files before
    # the duplicate check, so skip_duplicates alone will raise if the file
    # content changed since the first upload.
    if not (Nwbfile() & nwb_key):
        Nwbfile().insert1(
            {**nwb_key, "nwb_file_abs_path": str(nwb_file_path.resolve())},
            allow_direct_insert=True,
        )
    Session().insert1(
        {
            **nwb_key,
            "session_description": f"Tutorial dummy: {project_path.name}",
            "session_start_time": now,
            "timestamps_reference_time": now,
        },
        **ins,
    )
    Task().insert1({"task_name": "tutorial_dlc"}, **ins)
    IntervalList().insert1(
        {**interval_key, "valid_times": np.array([[0.0, 1.0]])}, **ins
    )
    TaskEpoch().insert1(
        {
            **interval_key,
            "epoch": 1,
            "task_name": "tutorial_dlc",
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
                "camera_name": "tutorial_dlc",
                "video_file_object_id": str(uuid.uuid4())[:40],
                "path": str(Path(video_path).resolve()),
            },
            **ins,
        )

    print(
        f"Tutorial session ready: {nwb_file_name} ({len(all_video_paths)} video(s))"
    )
    return nwb_file_name, inf_vid_path


def _get_or_create_inference_video(config_path):
    """Get or create inference video path for existing DLC projects."""
    from spyglass.settings import video_dir

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    training_videos = list(cfg.get("video_sets", {}).keys())
    if training_videos:
        src_video = Path(training_videos[0])
        inf_vid_name = f"example_inference{src_video.suffix}"
        inf_vid_path = Path(video_dir) / inf_vid_name

        if inf_vid_path.exists():
            return inf_vid_path

    return None


# %% [markdown]
# Now, we'll add your DLC project.

# %%
# Point to your DLC project config file.
# If this path does not exist, a self-contained demo runs automatically.
dlc_path = Path.home() / "DeepLabCut"
dlc_path = Path.home() / "wrk/alt/DeepLabCut"  # TODO: REMOVE BEFORE MERGE
config_path = dlc_path / "examples" / "TEST-Alex-2025-09-08" / "config.yaml"

model_key = None
nwb_file_name = None
inf_vid_path = None  # Inference video path (set below)
_demo_output_dir = None  # Set to a pre-computed h5 dir when in demo mode

if config_path.exists():
    # ── Real data path ────────────────────────────────────────────────────
    try:
        model_key = Model().load(config_path)
        # Get inference video path for existing projects
        inf_vid_path = _get_or_create_inference_video(config_path)
    except ValueError:
        # No Spyglass Session matched the DLC video paths.
        # Tutorial: create minimal dummy entries and retry.
        # Production: run insert_sessions('your_session.nwb') instead.
        print("Bootstrapping tutorial session for DLC import...")
        nwb_file_name, inf_vid_path = _tutorial_bootstrap_dlc_session(
            config_path
        )
        model_key = Model().load(config_path)

    print(f"Imported model: {model_key}")
    if inf_vid_path:
        print(f"Inference video: {inf_vid_path}")
    else:
        print("No inference video found - some tutorial steps may be skipped")

# %% [markdown]
# The import process:
#
# 1. Detects the latest trained model snapshot
# 2. Extracts the skeleton (bodyparts and connections)
# 3. Creates entries in Skeleton and ModelParams tables
# 4. Links the DLC videos to a registered Spyglass session via VidFileGroup
# 5. Stores model metadata in an NWB file
# 6. Creates a Model entry for inference
#

# %% [markdown]
# ##### ✅ Model Validation
#
# Let's verify the model import was successful:

# %%
if not (Model() & model_key):
    raise ValueError(f"❌ Model entry not found : {model_key}")

if not (model_params := (ModelParams() & model_key).fetch1()):
    raise ValueError(f"❌ Model parameters not found : {model_key}")

skeleton_id = model_params.get("skeleton_id")
if not (Skeleton() & {"skeleton_id": skeleton_id}).fetch1("KEY"):
    raise ValueError(f"❌ Skeleton details not found : {model_key}")

if vid_group_id := model_key.get("vid_group_id"):
    if not (VidFileGroup() & {"vid_group_id": vid_group_id}):
        raise ValueError(f"❌ Video group not found: {vid_group_id}")

print("✅ Model import successful")

# %%
# Check the model entry
Model() & model_key

# %% [markdown]
# This import process generates a video group based on the config.

# %%
VideoFile()

# %% [markdown]
# We can see the skeleton with this helper method

# %%
fig = (Skeleton & {"skeleton_id": skeleton_id}).show_skeleton()
plt.show()  # Explicit display for testing

# %% [markdown]
# #### From NWB File (ndx-pose)
#
# ndx-pose NWB files from DLC, SLEAP, or other tools can be imported with
# `Model.load()` to store model metadata (skeleton, bodyparts),
# but **cannot feed run inferences**.
#
# ndx-pose imports are useful for registering models from external tools for
# reference. To run the full pipeline, use a DLC model imported from
# `config.yaml` with video files registered in `VideoFile`.

# %%
# Generate an example ndx-pose NWB file using the test-suite helper.
# The same function is used by the pytest fixtures, so this doubles as
# a smoke-test of the NWB generation path.
import sys
import spyglass

_tests_v2 = Path(spyglass.__file__).parents[2] / "tests" / "position" / "v2"
sys.path.insert(0, str(_tests_v2))
from make_example_ndx_pose import make_ndx_pose_nwb

ndx_pose_path = config_path.parent / "example_ndx_pose.nwb"
make_ndx_pose_nwb(ndx_pose_path)
print(f"ndx-pose NWB file ready: {ndx_pose_path}")

# %%
# Import the ndx-pose model into Spyglass
ndx_model_key = None
if ndx_pose_path.exists():
    ndx_model_key = Model().load(ndx_pose_path)
    print(f"Imported ndx-pose model: {ndx_model_key}")

# %% [markdown]
# #### From DLC h5 output
#
# If you have run DLC inference and have `.h5` pose output files,
# `deeplabcut.analyze_videos_converth5_to_nwb` (requires `dlc2nwb`) converts
# them to an ndx-pose NWB with real `original_videos` paths. Any
# `VideoFile` entries in Spyglass whose paths match those videos are then
# automatically linked to `VidFileGroup.File` during `Model.load()`.
#
# Register your session with `insert_sessions()` before importing, so that
# `VideoFile` entries exist for the matching videos.

# %%
try:
    import dlc2nwb  # noqa: F401
    import deeplabcut as _dlc

    # Set config_path and video_folder to your project paths, then:
    # config_path = "your/config/path/config.yaml"
    # video_folder = "your/video/folder"
    # _dlc.analyze_videos_converth5_to_nwb(config_path, video_folder)
    #
    # Locate the produced NWB file and import:
    # nwb_files = list(Path(video_folder).glob("*.nwb"))
    # ndx_model_key = Model().import_model(nwb_files[0])

    print("dlc2nwb available — uncomment the lines above to run.")
except ImportError:
    print(
        "dlc2nwb not installed. Install with: pip install dlc2nwb\n"
        "Required by deeplabcut.analyze_videos_converth5_to_nwb."
    )

# %% [markdown]
# ## 🎯 Pose Estimation <a id="PoseEstim"></a>
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
# 1. **`PoseEstimParams`** — name a set of inference parameters (device, batch size, etc.)
# 2. **`PoseEstimSelection`** — pair a model with a video group and choose `task_mode='trigger'` (run inference) or `'load'` (read existing output)
# 3. **`PoseEstim.populate()`** — executes inference and stores results in an NWB file via ndx-pose
#

# %% [markdown]
# ##### Step 1 — Inference parameters (`PoseEstimParams`)
#
# In this table, `params_hash` is a unique identifier for the set of params used.
# This will raise an error if you attempt to insert a new entry with `params`
# matching an existing row.

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
# This structure supports multi-camera recordings but is not designed for
# batch processing across many sessions.

# %%
# Create video groups and estimation selection
training_vid_group_id = model_key["vid_group_id"] if model_key else None
inf_vid_group_key = None
estim_key = None

if not (inf_vid_path and model_key):
    raise ValueError("⚠️ Missing video or model - skipping PoseEstimSelection")

# Check/create required params entry
params_id = "cpu_batch8"  # or "gpu_batch8" depending on your hardware
PoseEstimParams.insert_params(
    params={"device": "cpu", "batch_size": 8},
    params_id=params_id,
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
    "pose_estim_params_id": params_id,
    "task_mode": "load" if _demo_output_dir else "trigger",
    "output_dir": str(_demo_output_dir) if _demo_output_dir else "",
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
# <summary>🚨 **Troubleshooting Pose Estimation** (Click if you encountered errors)</summary>
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
# print("Your video group:", VidFileGroup() & inf_vid_group_key if inf_vid_group_key else "None")
# print("Your estimation:", PoseEstim() & estim_key if estim_key else "None")
# ```
#
# </details>

# %% [markdown]
# ## Processing Parameters <a id="PoseParams"></a>
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

# %% [markdown]
# #### View Available Parameter Sets
#

# %%
PoseParams()

# %% [markdown]
# #### Use Default Parameters
#

# %% [markdown]
# For 2-LED tracking:
#

# %%
PoseParams.insert_default(skip_duplicates=True)
params_key = {"pose_params": "default"}  # or '4LED_default' or 'single_LED
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
        "pose_params": "tutorial_custom",
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
(PoseParams() & {"pose_params": "tutorial_custom"}).fetch1()

# %% [markdown]
# Search the table:

# %%
PoseParams & {"params.smoothing.interp_params": "max_pts_to_interp"}

# %% [markdown]
# ## 🔄 Data Processing <a id="PoseV2"></a>
#
# **🎯 Goal**: Process raw pose estimates into final position trajectories
#
# **🔍 What you'll accomplish**:
# - Link pose estimation results to processing parameters
# - Run the position processing pipeline
# - Generate orientation, centroid, and velocity data
# - Validate the final processed dataset

# %% [markdown]
# ## Pose Processing <a id="PoseEstim"></a>
#
# Now we can process the pose estimation to get cleaned, smoothed position data.

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
# >
#
# > **⚠️ Note**: Unlike other Spyglass pipelines, `PoseParams` primary keys
# > are automatically generated. You cannot specify custom `pose_params` IDs
# > when using `insert1()` - the system will create unique identifiers for you.

# %%
# Create pose processing selection and run processing
pose_selection_key = None
processed_df = None

if not estim_key:
    raise ValueError("No estim_key available - cannot run PoseV2 processing")

# Use custom params if available, otherwise default
params_name = "tutorial_custom"
if not (PoseParams() & {"pose_params": params_name}):
    PoseParams.insert_default(skip_duplicates=True)
    params_name = "default"

pose_selection_key = {**estim_key, "pose_params": params_name}
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

required = ["position_x", "position_y", "orientation", "velocity"]
missing = [col for col in required if col not in processed_df.columns]

if missing:
    raise ValueError(f"❌ Missing columns: {missing}")

# Data summary
time_range = processed_df.index[-1] - processed_df.index[0]
mean_vel = processed_df["velocity"].mean()

if not pose_selection_key or not (PoseV2() & pose_selection_key).fetch1("KEY"):
    raise ValueError("❌ PoseV2 entry not found for selection key")

print("✅ Validation passed")
print(f"Duration: {time_range:.1f}s, Mean velocity: {mean_vel:.1f} cm/s")

# %% [markdown]
# ## 📊 Data Analysis & Retrieval <a id="FetchData"></a>

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
    c=processed_df["velocity"],
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
axes[2].plot(processed_df.index, processed_df["velocity"])
axes[2].set_ylabel("Speed (cm/s)")
axes[2].set_xlabel("Time (s)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# **🎯 Goal**: Access processed position data for analysis and visualization
#
# **🔍 What you'll accomplish**:
# - Retrieve data as pandas DataFrames or raw NWB objects
# - Generate trajectory and time series visualizations
# - Understand data structure and coordinate systems
# - Export results for further analysis

# %% [markdown]
# ## 🏃‍♂️ MVP Training Demo <a id="MVPTraining"></a>
#
# **🎯 Goal**: Understand training workflow through position v1 test examples
#
# **🔍 What you'll learn**:
# - Examine real training configurations used in tests
# - See minimal parameter sets for demonstration purposes
# - Understand training workflow without running full training
# - Review test-based training examples with actual DLC integration
#
# ### Training Examples in Position V1 Tests
#
# The Position V1 test suite includes comprehensive examples of MVP training with minimal configurations. These tests demonstrate:
#
# - **Minimal Training Parameters**: Short iteration counts for CI/testing
# - **Real DLC Integration**: Actual DeepLabCut training workflows
# - **Training Pipeline**: Complete end-to-end model development
# - **Validation Patterns**: How to verify training completed successfully
#
# **Key test files to examine**:
# - `tests/position/v1/test_dlc_training.py` - Core training workflows
# - `tests/position/v1/conftest.py` - Training configuration fixtures
# - Test training uses ~100-500 iterations vs production 10,000+
#
# ```python
# # Example from position v1 tests - minimal training configuration
# training_params = {
#     "max_iters": 100,           # Very low for testing
#     "display_iters": 50,        # Show progress
#     "save_iters": 100,          # Save at end only
#     "net_type": "resnet_50",    # Standard architecture
#     "batch_size": 1,            # Minimal batch
# }
# ```
#
# 💡 **For production training**: Use the [Training New Models](#TrainingWorkflow) section with 10,000+ iterations and proper validation datasets.

# %% [markdown]
# ### Exploring Test-Based Training Examples
#
# You can examine the actual training configurations used in the test suite:

# %%
# Examine training parameters from position v1 tests
from pathlib import Path

# Show path to training test examples
test_dir = Path("tests/position/v1/")
training_tests = ["test_dlc_training.py", "test_dlc_project.py"]

print("📋 Position V1 Training Test Examples:")
print(f"Location: {test_dir}")
print("\nKey files:")
for test_file in training_tests:
    test_path = test_dir / test_file
    if test_path.exists():
        print(f"✅ {test_file} - {test_path}")
    else:
        print(f"📁 {test_file} - (check repository for location)")

print("\n🔍 To explore training configurations:")
print("1. Look at test fixtures in conftest.py")
print("2. Examine training parameter dictionaries")
print("3. See minimal iteration counts used for testing")
print("4. Understand validation patterns")

print("\n💡 Key differences from production:")
print("   • Test iterations: 100-500 vs Production: 10,000+")
print("   • Test batch size: 1-2 vs Production: 8-32")
print("   • Test duration: 1-5 minutes vs Production: hours")
print(
    "   • Test datasets: Synthetic/minimal vs Production: thousands of frames"
)

# %% [markdown]
# # 🔧 ADVANCED FEATURES
#
# The following sections cover optional advanced functionality. Most users can skip these sections and return later as needed.

# %% [markdown]
# ## 💡 Training New Models <a id="TrainingWorkflow"></a>
#
# **🎯 Goal**: Train custom pose estimation models from scratch
#
# **🔍 What you'll accomplish**:
# - Create skeletal structure and bodypart definitions
# - Set up training parameters and video datasets
# - Train DLC models with custom configurations
# - Enable model evaluation and performance metrics
#
# <details>
# <summary><b>Model Training</b></summary>
#
# ```python
# # 1. Create skeleton
# skeleton_id = Skeleton().insert1({
#     "skeleton_name": "my_tracking",
#     "bodyparts": ["nose", "tail_base", "left_ear", "right_ear"],
#     "edges": [[0, 1], [0, 2], [0, 3]],  # nose connects to all
# })
#
# # 2. Define training parameters
# params_key = ModelParams().insert1({
#     "tool": "DLC",
#     "params": {
#         "dlc": {
#             "train_params": {
#                 "training_params": {
#                     "net_type": "resnet_50",
#                     "augmentation_method": "default",
#                     "solver": "adam",
#                 },
#                 "model_params": {"backbone": "resnet50"},
#             }
#         }
#     },
# })
#
# # 3. Create training VidFileGroup
# training_vid_group = VidFileGroup.create_from_files({
#     "vid_group_id": "training_data",
#     "video_files": [list_of_training_videos],
# })
#
# # 4. Create model training task
# model_key = ModelSelection().insert1({
#     "model_params": params_key,
#     "vid_group_id": "training_data",
#     "skeleton_id": skeleton_id,
# })
#
# # 5. Run training
# Model().populate([model_key])
# ```
#
# **Note**: Training typically requires 10,000+ iterations for good performance.
# For demo purposes, use minimal iterations:
#
# ```python
# # Minimal training for testing evaluation features
# params_key = ModelParams().insert1({
#     "tool": "DLC",
#     "params": {
#         "dlc": {
#             "train_params": {
#                 "training_params": {
#                     "max_iters": 100,  # Very low for demo only
#                     "save_iters": 50,
#                     "display_iters": 25,
#                 }
#             }
#         }
#     }
# })
# ```
# </details>
#
# ---

# %% [markdown]
# ## 📊 Model Evaluation & Training Curves <a id="ModelEvaluation"></a>
#
# **🎯 Goal**: Evaluate model performance and visualize training progress
#
# **🔍 What you'll accomplish**:
# - Assess model accuracy on test data
# - Generate training and validation loss curves
# - Understand model convergence and potential overfitting
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
model_tool = model_params.get("params", {}).get("tool", "Unknown")
training_history = Model().get_training_history(model_key)
has_training_history = training_history is not None
evaluation_supported = model_tool.upper() == "DLC"

# %% [markdown]
# ### Generate Training Curves
#
# Visualize training progress and detect potential overfitting:

# %%
if not (has_training_history and evaluation_supported):
    raise ValueError("⚠️ Model evaluation unavailable. Train a model first.")

fig = Model().plot_training_history(model_key, save_path=None, figsize=(12, 8))
fig.show()

# %% [markdown]
# ### Model Performance Evaluation
#
# Assess accuracy on test data and compare models:

# %%
if not evaluation_supported:
    print(f"⚠️ Model evaluation not supported for tool: {model_tool}")
    # Early return pattern - no further processing needed

print("📊 Model evaluation methods coming soon...")
# Future: Model.evaluate(), comparison metrics, etc.

# %% [markdown]
# ---

# %% [markdown]
# ### Prerequisites Check
#
# Before generating videos, ensure you have completed pose estimation:

# %%
# Check video generation prerequisites
can_make_video = False
if (
    processed_df is not None
    and len(processed_df) > 0
    and pose_selection_key is not None
):
    can_make_video = True

# %% [markdown]
# ### Basic Video Generation
#
# Create a basic annotated video with pose keypoints:

# %%
if not can_make_video:
    raise ValueError("Cannot venerate video")

# Basic video parameters
video_params = {
    "start_time": 0.0,
    "duration": 10.0,
    "fps": 30,
    "show_keypoints": True,
    "show_skeleton": True,
    "keypoint_radius": 3,
    "line_thickness": 2,
}

# In tutorial mode, simulate video generation
video_path = (PoseV2() & pose_selection_key).make_video(**video_params)

# %% [markdown]
# <details>
# <summary>🚨 **Video Generation Troubleshooting** (Click if you encounter issues)</summary>
#
# ### Common Video Generation Issues
#
# #### 🔴 **"Original video file not found"**
# **Cause**: Video paths in database don't match current file locations
# **Solution**:
# ```python
# # Check video file paths
# (VidFileGroup.File() & inf_vid_group_key).fetch("video_file_path")
# # Update paths if files moved
# ```
#
# #### 🔴 **"Video generation very slow"**
# **Cause**: High resolution or long duration
# **Solution**:
# ```python
# # Test with shorter clips first
# video_params = {
#     "duration": 5.0,    # Just 5 seconds
#     "fps": 15,          # Lower framerate
#     "resolution": (640, 480)  # Lower resolution
# }
# ```
#
# #### 🔴 **"Memory error during generation"**
# **Cause**: Large videos exceeding available RAM
# **Solution**:
# - Process shorter segments separately
# - Lower video resolution/quality
# - Close other applications to free memory
# - Use frame-by-frame processing if supported
#
# #### 🔴 **"Keypoints not visible"**
# **Cause**: Keypoint styling too small or transparent
# **Solution**:
# ```python
# # Make keypoints more visible
# video_params = {
#     "keypoint_radius": 5,     # Larger dots
#     "keypoint_alpha": 1.0,    # Fully opaque
#     "line_thickness": 3       # Thicker skeleton lines
# }
# ```
#
# #### 🔴 **"Video output corrupted"**
# **Cause**: Write permissions or disk space issues
# **Solution**:
# - Check available disk space (videos can be large)
# - Verify write permissions in output directory
# - Try different output format/codec
# - Test with very short duration first
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
# </details>
#
# ---

# %% [markdown]
# PoseV2 provides two methods for retrieving processed data:
#

# %% [markdown]
# #### Method 1: fetch1_dataframe()
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
    c=processed_df["velocity"],
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
axes[2].plot(processed_df.index, processed_df["velocity"])
axes[2].set_ylabel("Speed (cm/s)")
axes[2].set_xlabel("Time (s)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# # 📚 REFERENCE

# %% [markdown]
# ## 🐾 Next Steps
#
# - **Linearization**: Convert 2D position to 1D track position (notebook 24)
# - **Decoding**: Use position for neural decoding (notebooks 41-42)
# - **Custom Analysis**: Work directly with fetched DataFrames
#
# For questions, see the [Spyglass documentation](https://lorenfranklab.github.io/spyglass/)
# or open a discussion on [GitHub](https://github.com/LorenFrankLab/spyglass/discussions).
#

# %% [markdown]
# ### 🚨 Troubleshooting <a id="Troubleshooting"></a>
#
# <details>
# <summary><b>Model Import</b> (Click to expand)</summary>
#
# #### **📥 Model Import Issues**
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
# #### **🎯 Pose Estimation Issues**
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
# #### **⚙️ Parameter Configuration Issues**
#
# **"Bodypart not found"**
# - Check available bodyparts: `(Skeleton.BodyPart() & {"skeleton_id": your_id}).fetch()`
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
# #### **🔄 Data Processing Issues**
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
# ## 🎓 Guides

# %% [markdown]
# <details>
# <summary><b>V1 → V2 Migration Guide</b> (Click to expand)</summary>
#
# ### Migration Guide
#
# #### Table and Naming Changes
#
# Position V2 significantly streamlines the table structure compared to V1. Here's
# a comprehensive migration mapping:
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
# | `DLCPosVideo` | `PoseV2.make_video` | No longer stred as table |
# | `TrodesPosParams` | — | No V2 equivalent |
# | `TrodesPosSelection` | — | No V2 equivalent |
# | `TrodesPosV1` | — | No V2 equivalent |
# | `TrodesPosVideo` | — | No V2 equivalent |
# | `ImportedPose` | — | No V2 equivalent |
#
#
# #### Key consolidations in V2
#
# - `DLCCentroidParams` + `DLCOrientationParams` + `DLCSmoothInterpParams` → single `PoseParams` with three sub-dicts
# - `DLCModelInput` + `DLCModelSource` + `DLCModelParams` → single `ModelParams`
# - Cohort pattern (`DLCSmoothInterpCohort*`) eliminated; `PoseV2` handles multi-part poses directly
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
# ├── position_dlc_project.py      # Project management
# ├── position_dlc_training.py     # Model training
# ├── position_dlc_pose_estimation.py  # Inference
# ├── position_dlc_cohort.py       # Batch processing
# ├── dlc_utils.py                 # Utilities
# └── dlc_reader.py                # File I/O
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
# Position V2 supports **native JSON columns** for enhanced parameter querying capabilities:
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
# **Migration Note:** Existing blob parameter tables continue to work unchanged. Consider JSON columns for new parameter tables requiring complex querying.
#
# </details>
#
# <details>
# <summary><b>SLEAP Support Status</b> (Click to expand)</summary>
#
# ### SLEAP Integration Roadmap
#
# Position V2 includes preliminary SLEAP support architecture but **SLEAP training is not yet functional**. Here's the current status:
#
# #### ✅ **Available Now**
#
# - **Import Support**: Can import pre-trained SLEAP models via `Model.load()`, but not yet run inference.
# - **Data Loading**: Can load existing SLEAP NWB files using `PoseEstim.load_from_nwb()`
#
# #### 🔄 **In Development**
#
# - **Training Pipeline**: The `train_model()` method raises `NotImplementedError`
# - **Inference Integration**: SLEAP analysis integration with Position V2 workflows
#
# **Timeline**: Native SLEAP training support is targeted for Q3 2026.
#
# </details>

# %% [markdown]
# ## 🔗 External Resources <a id="Resources"></a>
#
# ### Documentation & Guides
# - [Spyglass Documentation](https://lorenfranklab.github.io/spyglass/)
# - [Position V2 API Reference](https://lorenfranklab.github.io/spyglass/api/position/v2/)
# - [DeepLabCut Documentation](https://deeplabcut.github.io/DeepLabCut/)
# - [ndx-pose Extension](https://github.com/rly/ndx-pose)
# - [DataJoint Documentation](https://docs.datajoint.com/)
#
# ### Getting Help
# - [GitHub Issues](https://github.com/LorenFrankLab/spyglass/issues) - Bug reports and feature requests
# - [GitHub Discussions](https://github.com/LorenFrankLab/spyglass/discussions) - Questions and community support
# - [Frank Lab Website](https://franklab.ucsf.edu/) - Lab resources and contact information
#
# ### Related Notebooks
# - [00_Setup.ipynb](./00_Setup.ipynb) - Initial Spyglass configuration
# - [02_Insert_Data.ipynb](./02_Insert_Data.ipynb) - DataJoint basics
# - [21_DLC.ipynb](./21_DLC.ipynb) - Legacy Position V1 pipeline
# - [24_Linearization.ipynb](./24_Linearization.ipynb) - Convert 2D → 1D position
# - [41_Decoding_Clusterless.ipynb](./41_Decoding_Clusterless.ipynb) - Use position for decoding
#

# %% [markdown]
#
# **🎉 Tutorial Complete!**
#
# You've learned the fundamentals of the Position V2 pipeline. For questions or feedback, please use the resources above.
