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
#

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
# - **Simplified processing**: Single PoseV2 table handles all position processing
#
# This tutorial will walk through:
#
# - Importing a trained model (DLC or ndx-pose)
# - Running pose estimation on videos
# - Processing pose data (orientation, centroid, smoothing)
# - Retrieving and visualizing results
#

# %% [markdown]
# <details>
# <summary><b>V1 → V2 Migration Guide</b> (Click to expand)</summary>
#
# ### Table and Naming Changes
#
# Position V2 significantly streamlines the table structure compared to V1. Here's
# a comprehensive migration mapping:
#
#
# | V1 | V2 | Notes |
# |---|---|---|
# | `BodyPart` | `BodyPart` | Renamed from `Manual` → `Lookup` |
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
# ├── validation.py                # Parameter validation
# ├── tool_strategies.py           # Multi-tool support
# └── ...                          # Other shared utilities
# ```
#
# #### 🔍 **Method Equivalents**
#
#
# | V1 Method | V2 Method | Notes |
# |-----------|-----------|-------|
# | `DLCModel.create_dlc_model()` | `Model.train_model()` | Unified training interface |
# | `DLCModel.import_dlc_model()` | `Model.import_model()` | Tool-agnostic import |
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

# %% [markdown]
# <details><summary><b>JSON Parameter Support</b> (Click to Expand)</summary>
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

# %% [markdown]
# <details>
# <summary><b>SLEAP Support Status</b> (Click to expand)</summary>
#
# ### SLEAP Integration Roadmap
#
# Position V2 includes preliminary SLEAP support architecture but **SLEAP training is not yet functional**. Here's the current status:
#
# #### ✅ **Available Now**
#
# - **Parameter Validation**: Full parameter specification and validation for SLEAP models
# - **Import Support**: Can import pre-trained SLEAP models via `Model.import_model()`, but not yet run inference.
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
#
# ---

# %% [markdown]
# ### Imports
#

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
# ### Table of Contents<a id='TableOfContents'></a>
#
# - [`BodyParts`](#BodyParts)
# - [`Model`](#Model)
# - [`PoseEstim`](#PoseEstim)
# - [`PoseParams`](#PoseParams)
# - [`PoseV2`](#PoseV2)
# - [`Fetching Data`](#FetchData)
# - [`Visualization`](#Visualization)
# - [`V1 vs V2 Comparison`](#Comparison)
#

# %% [markdown]
# ## Path 1: Import Existing Model
#

# %% [markdown]
# ### [Model](#TableOfContents) <a id="Model"></a>
#

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
# #### Import from DeepLabCut Project
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
    session with insert_sessions() before calling Model.import_model().
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
        model_key = Model().import_model(config_path)
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
        model_key = Model().import_model(config_path)

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

# %%
Model().import_model(config_path)

# %%
VideoFile()

# %%
# Check the model entry
Model() & model_key

# %%
skeleton_id = (ModelParams() & model_key).fetch1("skeleton_id")
Skeleton & {"skeleton_id": skeleton_id}

# %% [markdown]
# #### Import from NWB File (ndx-pose)
#
# ndx-pose NWB files from DLC, SLEAP, or other tools can be imported with
# `Model.import_model()`. The entry stores model metadata (skeleton, bodyparts),
# but **cannot feed into `PoseEstim.populate()`**: the placeholder `VidFileGroup`
# created during import has no `VideoFile` entries, so `VidFileGroup.get_nwb_file()`
# cannot resolve a parent `Nwbfile`.
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
if ndx_pose_path.exists():
    try:
        ndx_model_key = Model().import_model(ndx_pose_path)
        print(f"Imported ndx-pose model: {ndx_model_key}")
    except Exception as e:
        print(f"Failed to import ndx-pose model: {e}")
        ndx_model_key = None
else:
    print(f"ndx-pose file not found: {ndx_pose_path}")
    ndx_model_key = None

# %% [markdown]
# #### Importing from DLC h5 output (production workflow)
#
# If you have run DLC inference and have `.h5` pose output files,
# `deeplabcut.analyze_videos_converth5_to_nwb` (requires `dlc2nwb`) converts
# them to an ndx-pose NWB with real `original_videos` paths. Any
# `VideoFile` entries in Spyglass whose paths match those videos are then
# automatically linked to `VidFileGroup.File` during `Model.import_model()`.
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
# ### [PoseEstim](#TableOfContents) <a id="PoseEstim"></a>
#

# %% [markdown]
# Now that we have a model, let's run pose estimation on a video.
#

# %% [markdown]
# #### Option A: Load existing DLC output
#

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
# Inference in V2 follows a three-step DataJoint pattern:
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
try:
    params_result = PoseEstimParams.insert_params(
        params={"device": "cuda", "batch_size": 8},
        params_id="gpu_batch8",
        skip_duplicates=True,
    )
    print(f"Inserted PoseEstimParams: {params_result}")
except Exception as e:
    print(f"PoseEstimParams insertion failed: {e}")
    print("Using default parameters instead...")

PoseEstimParams()

# %% [markdown]
# ##### Step 2 — Estimation task (`PoseEstimSelection`)
#
# Pose estimation uses **two separate `VidFileGroup` entries**:
#
# - **Training group** (`training_vid_group_id`): created by `import_model()` and
#   linked to `ModelSelection`. Contains the original labeled videos used for
#   training. Used by `get_nwb_file()` to resolve the parent session.
# - **Inference group** (`inf_vid_group_id`): created here for the video(s) you
#   want to run inference on. Linked to `PoseEstimSelection`.
#
# This structure supports multi-camera recordings but is not designed for
# batch processing across many sessions.

# %%
# Video group 1: training videos (created by import_model)
training_vid_group_id = model_key["vid_group_id"] if model_key else None

# Video group 2: inference video (created here for PoseEstimSelection)
inf_vid_group_key = None
estim_key = None

if inf_vid_path and model_key:
    try:
        # Check that we have the required params entry
        params_table_entry = PoseEstimParams() & {
            "pose_estim_params_id": "gpu_batch8"
        }
        if not params_table_entry:
            print("Creating required PoseEstimParams entry...")
            PoseEstimParams.insert_params(
                params={"device": "cpu", "batch_size": 8},
                params_id="cpu_batch8",
                skip_duplicates=True,
            )
            params_id = "cpu_batch8"
        else:
            params_id = "gpu_batch8"

        inf_vid_group_key = VidFileGroup().insert1(
            {
                "description": f"Inference video for {model_key['model_id']}",
                "files": [inf_vid_path],
            },
            skip_duplicates=True,
        )
        print(f"Created inference VidFileGroup: {inf_vid_group_key}")

        # Create the complete key with all required fields
        estim_selection_key = {
            "model_id": model_key["model_id"],
            "vid_group_id": inf_vid_group_key["vid_group_id"],
            "pose_estim_params_id": params_id,
            "task_mode": "load" if _demo_output_dir else "trigger",
            "output_dir": str(_demo_output_dir) if _demo_output_dir else "",
        }

        # Insert the selection entry
        PoseEstimSelection().insert1(estim_selection_key, skip_duplicates=True)

        # Return the key for populate
        estim_key = {
            k: v
            for k, v in estim_selection_key.items()
            if k in ["model_id", "vid_group_id", "pose_estim_params_id"]
        }

        print(f"Selection ready: {estim_key}")

    except Exception as e:
        print(f"Failed to create PoseEstimSelection: {e}")
        print(f"Model key: {model_key}")
        print(f"Inf video path: {inf_vid_path}")
        import traceback

        traceback.print_exc()
        estim_key = None

else:
    print("No inference video available — skipping PoseEstimSelection.")
    if not inf_vid_path:
        print("  - inf_vid_path is missing")
    if not model_key:
        print("  - model_key is missing")

# %% [markdown]
# ##### Step 3 — Run inference (`PoseEstim.populate()`)
#

# %%
# Inspect both video groups
if training_vid_group_id:
    print("Training VidFileGroup:")
    print(VidFileGroup().File() & {"vid_group_id": training_vid_group_id})
if inf_vid_group_key:
    print("Inference VidFileGroup:")
    print(
        VidFileGroup().File()
        & {"vid_group_id": inf_vid_group_key["vid_group_id"]}
    )

# %%
# PoseEstim.populate() runs an inference then stores the data as an ndx-pose NWB
if estim_key:
    try:
        PoseEstim.populate(estim_key)
        pose_df = (PoseEstim() & estim_key).fetch1_dataframe()
        print(pose_df.head())
    except Exception as e:
        print(f"PoseEstim.populate() failed: {e}")
        print(f"estim_key: {estim_key}")
        pose_df = None
else:
    print("Cannot run PoseEstim.populate() - estim_key is None")
    pose_df = None

# %% [markdown]
# ### [PoseParams](#TableOfContents) <a id="PoseParams"></a>
#

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
# For 2-LED tracking (common Frank Lab setup):
#

# %%
PoseParams.insert_default(skip_duplicates=True)
params_key = {"pose_params": "default"}

# %% [markdown]
# For 4-LED tracking:
#
# ```python
# PoseParams.insert_4LED_default(skip_duplicates=True)
# params_key = {"pose_params": "4LED_default"}
# ```
#
# For single marker tracking:
#
# ```python
# PoseParams.insert_single_LED(skip_duplicates=True)
# params_key = {"pose_params": "single_LED"}
# ```
#

# %% [markdown]
# #### Create Custom Parameters
#

# %% [markdown]
# For custom tracking scenarios:
#

# %%
# Get actual bodyparts from the imported model
if model_key:
    try:
        model_params = (ModelParams() & model_key).fetch1()
        skeleton_id = model_params["skeleton_id"]

        # Check if skeleton has bodyparts in the part table
        skeleton_parts = (
            Skeleton.BodyPart() & {"skeleton_id": skeleton_id}
        ).fetch("bodypart")
        print(f"Available bodyparts in model: {list(skeleton_parts)}")

        if len(skeleton_parts) >= 2:
            # Use actual bodyparts from the model
            bodypart1, bodypart2 = skeleton_parts[0], skeleton_parts[1]
            centroid_bodypart = skeleton_parts[0]

            try:
                PoseParams.insert_custom(
                    params_name="tutorial_custom",
                    orient={
                        "method": "two_pt",  # Use two points to define orientation
                        "bodypart1": bodypart1,
                        "bodypart2": bodypart2,
                        "interpolate": True,
                        "smooth": True,
                    },
                    centroid={
                        "method": "1pt",  # Use single point as centroid
                        "points": {"point1": centroid_bodypart},
                    },
                    smoothing={
                        "interpolate": True,
                        "interp_params": {
                            "max_pts_to_interp": 10,
                            "max_cm_to_interp": 15.0,
                        },
                        "smooth": True,
                        "smoothing_params": {
                            "method": "moving_avg",
                            "smoothing_duration": 0.3,  # 300ms window
                        },
                        "likelihood_thresh": 0.1,  # Low threshold; tutorial data has ~0.45 likelihoods
                    },
                )
                print("Successfully created custom PoseParams")
            except Exception as e:
                print(f"Failed to create custom PoseParams: {e}")
                print("Using default PoseParams instead...")

        elif len(skeleton_parts) == 1:
            print(
                f"Model has only 1 bodypart ({skeleton_parts[0]}) - creating single-point parameters..."
            )

            try:
                PoseParams.insert_custom(
                    params_name="tutorial_custom_single",
                    orient={
                        "method": "none",  # No orientation calculation
                    },
                    centroid={
                        "method": "1pt",  # Use single point as centroid
                        "points": {"point1": skeleton_parts[0]},
                    },
                    smoothing={
                        "interpolate": True,
                        "interp_params": {
                            "max_pts_to_interp": 10,
                            "max_cm_to_interp": 15.0,
                        },
                        "smooth": True,
                        "smoothing_params": {
                            "method": "moving_avg",
                            "smoothing_duration": 0.3,  # 300ms window
                        },
                        "likelihood_thresh": 0.1,  # Low threshold; tutorial data has ~0.45 likelihoods
                    },
                )
                print("Successfully created single-point custom PoseParams")
            except Exception as e:
                print(f"Failed to create single-point PoseParams: {e}")

        else:
            print("Model has no bodyparts found in Skeleton.BodyPart table")
            # Try getting bodyparts from the skeleton itself
            skeleton_entry = (
                Skeleton() & {"skeleton_id": skeleton_id}
            ).fetch1()
            bodyparts_list = skeleton_entry.get("bodyparts")
            if bodyparts_list:
                print(f"Found bodyparts in Skeleton table: {bodyparts_list}")
                if len(bodyparts_list) >= 2:
                    bodypart1, bodypart2 = bodyparts_list[0], bodyparts_list[1]
                    centroid_bodypart = bodyparts_list[0]

                    try:
                        PoseParams.insert_custom(
                            params_name="tutorial_custom_skeleton",
                            orient={
                                "method": "two_pt",
                                "bodypart1": bodypart1,
                                "bodypart2": bodypart2,
                                "interpolate": True,
                                "smooth": True,
                            },
                            centroid={
                                "method": "1pt",
                                "points": {"point1": centroid_bodypart},
                            },
                            smoothing={
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
                        )
                        print(
                            "Successfully created custom PoseParams from Skeleton bodyparts"
                        )
                    except Exception as e:
                        print(f"Failed to create PoseParams from Skeleton: {e}")

    except Exception as e:
        print(f"Error processing skeleton bodyparts: {e}")
        import traceback

        traceback.print_exc()

else:
    print("No model available - cannot determine bodypart names")

# %% [markdown]
# Inspect the parameters:
#

# %%
(PoseParams() & {"pose_params": "tutorial_custom"}).fetch1()

# %% [markdown]
# ### [PoseV2](#TableOfContents) <a id="PoseV2"></a>
#

# %% [markdown]
# Now we can process the pose estimation to get cleaned, smoothed position data.
#

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
#

# %%
# +
pose_selection_key = None
processed_df = None

if estim_key:
    try:
        # Try with tutorial_custom first, fallback to default if needed
        params_name = "tutorial_custom"

        # Check if custom params exist, otherwise use default
        if not (PoseParams() & {"pose_params": params_name}):
            print(f"Custom params '{params_name}' not found, using default...")
            PoseParams.insert_default(skip_duplicates=True)
            params_name = "default"

        pose_selection_key = {**estim_key, "pose_params": params_name}
        PoseSelection().insert1(
            pose_selection_key, skip_duplicates=True, ignore_extra_fields=True
        )
        print(f"PoseSelection: {pose_selection_key}")

        PoseV2.populate(pose_selection_key)
        processed_df = (PoseV2() & pose_selection_key).fetch1_dataframe()
        print(processed_df.head())
        print(
            f"\nTime range: {processed_df.index[0]:.2f}"
            f" – {processed_df.index[-1]:.2f} s"
        )
        print(f'Mean speed: {processed_df["velocity"].mean():.2f} cm/s')

    except Exception as e:
        print(f"PoseV2 processing failed: {e}")
        print(f"estim_key: {estim_key}")
        processed_df = None
        pose_selection_key = None

else:
    print("No estim_key available — skipping PoseV2.")
    print("Make sure PoseEstim.populate() completed successfully first.")
# -

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
# #### Verify Results
#

# %%
# Show table structure
PoseV2()

# %% [markdown]
# ### [Fetching Data](#TableOfContents) <a id="FetchData"></a>
#

# %% [markdown]
# PoseV2 provides two methods for retrieving processed data:
#

# %% [markdown]
# #### Method 1: fetch1_dataframe()
#

# %% [markdown]
# Get cleaned data as a pandas DataFrame:
#

# %%
if processed_df is not None:
    print(processed_df.head())
    print(f"\nColumns: {list(processed_df.columns)}")
    print(
        f"Time range: {processed_df.index[0]:.2f} - {processed_df.index[-1]:.2f} s"
    )
    print(f"Mean speed: {processed_df['velocity'].mean():.2f} cm/s")

# %% [markdown]
# #### Method 2: fetch_obj()
#

# %% [markdown]
# Get raw pynwb objects for advanced analysis:
#

# %%
# Production: fetch raw pynwb objects via fetch_obj()
# objs = (PoseV2() & pose_selection_key).fetch_obj()
# Returns dict with: 'orient', 'centroid', 'velocity', 'smoothed_pose'
# as BehavioralTimeSeries / Position pynwb objects for custom NWB access.
#
# orient_obj = (PoseV2() & pose_selection_key).fetch_obj("orient")
# centroid_obj = (PoseV2() & pose_selection_key).fetch_obj(["centroid", "velocity"])

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
if processed_df is not None:
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
if processed_df is not None:
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
# ## Path 2: Train New Model
#

# %% [markdown]
# For completeness, here's how to train a new model with V2:
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
#         "project_path": "/path/to/dlc/project",
#         "net_type": "resnet_50",
#         "maxiters": 100000,
#         "shuffle": 1,
#     },
#     "skeleton_id": skeleton_id
# })
#
# # 3. Create video group
# vid_group_key = VidFileGroup().create_from_files([
#     "/path/to/video1.mp4",
#     "/path/to/video2.mp4",
# ])
#
# # 4. Train model
# sel_key = {**params_key, **vid_group_key}
# ModelSelection().insert1(sel_key)
# Model().populate(sel_key)  # This runs training
#
# # 5. Evaluate model
# model_key = (Model() & sel_key).fetch1("KEY")
# results = Model().evaluate(model_key, plotting=True)
# print(f"Test error: {results['test_error']:.2f} pixels")
#
# # 6. Plot training history
# Model().plot_training_history(model_key, save_path="training_curve.png")
# ```
#

# %% [markdown]
# ## Troubleshooting
#

# %% [markdown]
# ### Common Issues
#
# **Import fails - "Model not found"**:
# - Ensure DLC project has completed training
# - Check that model snapshots exist in `dlc-models/` directory
# - Verify config.yaml path is correct
#
# **Inference fails - "CUDA out of memory"**:
# - Use `device="cpu"` instead of `device="cuda"`
# - Process videos in smaller batches
# - Reduce video resolution
#
# **Low likelihood values**:
# - Model may not generalize to your videos
# - Consider fine-tuning with a few labeled frames
# - Adjust `likelihood_thresh` in PoseParams
#
# **Jerky trajectories**:
# - Increase `smoothing_duration` in PoseParams
# - Try different smoothing methods (savgol, gaussian)
# - Check for tracking failures (NaN values)
#
# **Missing bodyparts in output**:
# - Verify bodyparts in PoseParams match those in model
# - Check BodyPart table has required entries
# - Review DLC output files for completeness
#

# %% [markdown]
# ## Next Steps
#
# - **Linearization**: Convert 2D position to 1D track position (notebook 24)
# - **Decoding**: Use position for neural decoding (notebooks 41-42)
# - **Custom Analysis**: Work directly with fetched DataFrames
#
# For questions, see the [Spyglass documentation](https://lorenfranklab.github.io/spyglass/)
# or open an issue on [GitHub](https://github.com/LorenFrankLab/spyglass).
#
