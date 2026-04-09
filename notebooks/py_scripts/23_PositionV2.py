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
# > video paths. In production, run `insert_sessions('your_session.nwb')` first.
# >
# > For this tutorial, `_tutorial_bootstrap_dlc_session()` (defined below)
# > creates minimal dummy entries so the import can proceed without a recorded
# > session.
#

# %%
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
    from spyglass.common import (
        Nwbfile,
        Session,
        VideoFile,
        IntervalList,
        Task,
        TaskEpoch,
    )
    from spyglass.settings import raw_dir

    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    project_path = Path(cfg.get("project_path", str(config_path.parent)))
    nwb_stem = project_path.name  # e.g. 'TEST-Alex-2025-09-08'
    nwb_file_name = f"{nwb_stem}_.nwb"
    video_paths = list(cfg.get("video_sets", {}).keys())
    nwb_file_path = Path(raw_dir) / nwb_file_name

    with open(nwb_file_path, "w") as f:
        f.write("")  # create empty file to satisfy FK constraints

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
            "session_description": f"Tutorial dummy: {project_path.name}",
            "session_start_time": now,
            "timestamps_reference_time": now,
        },
        allow_direct_insert=True,
        skip_duplicates=True,
    )
    Task().insert1({"task_name": "tutorial_dlc"}, skip_duplicates=True)
    IntervalList().insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "tutorial_epoch_1",
            "valid_times": np.array([[0.0, 1.0]]),
        },
        skip_duplicates=True,
    )
    TaskEpoch().insert1(
        {
            "nwb_file_name": nwb_file_name,
            "epoch": 1,
            "task_name": "tutorial_dlc",
            "interval_list_name": "tutorial_epoch_1",
            "camera_names": [],
        },
        allow_direct_insert=True,
        skip_duplicates=True,
    )
    for i, video_path in enumerate(video_paths):
        VideoFile().insert1(
            {
                "nwb_file_name": nwb_file_name,
                "epoch": 1,
                "video_file_num": i,
                "camera_name": "tutorial_dlc",
                "video_file_object_id": str(uuid.uuid4())[:40],
                "path": str(Path(video_path).resolve()),
            },
            allow_direct_insert=True,
            skip_duplicates=True,
        )

    print(
        f"Tutorial session ready: {nwb_file_name} ({len(video_paths)} video(s))"
    )
    return nwb_file_name


# %%
# Point to your DLC project config file
dlc_path = Path.home() / "DeepLabCut"
dlc_path = (
    Path.home() / "wrk" / "alt" / "DeepLabCut"
)  # TODO: REMOVE BEFORE MERGE
config_path = dlc_path / "examples" / "TEST-Alex-2025-09-08" / "config.yaml"

# Import the model; bootstrap a tutorial session if none is registered yet.
model_key = None
if config_path.exists():
    try:
        model_key = Model().import_model(config_path)
    except ValueError:
        # No Spyglass Session matched the DLC video paths.
        # Tutorial: create minimal dummy entries and retry.
        # Production: run insert_sessions('your_session.nwb') instead.
        _tutorial_bootstrap_dlc_session(config_path)
        model_key = Model().import_model(config_path)
    print(f"Imported model: {model_key}")
else:
    print(f"Config not found at {config_path}")
    print("Adjust the path or run testscript.py to create example project")

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
# Trim a training video from the DLC project to 1 second for fast inference.
# video_path defaults to None; update it here to point to any video you prefer.
import subprocess

video_path = None
if config_path.exists():
    with open(config_path) as f:
        dlc_config = yaml.safe_load(f)
    src_videos = list(dlc_config.get("video_sets", {}).keys())
    if src_videos:
        src_video = Path(src_videos[0])
        example_video = (
            src_video.parent / f"example_inference{src_video.suffix}"
        )
        if not example_video.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    str(src_video),
                    "-t",
                    "1",
                    "-c",
                    "copy",
                    str(example_video),
                ],
                check=True,
                capture_output=True,
            )
            print(f"Created 1-second clip: {example_video}")
        else:
            print(f"Example video already exists: {example_video}")
        video_path = example_video
    else:
        print("No training videos found in DLC config")

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
    ndx_model_key = Model().import_model(ndx_pose_path)
    print(f"Imported ndx-pose model: {ndx_model_key}")
else:
    print(f"ndx-pose file not found: {ndx_pose_path}")

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

# %%
# The 'default' parameter set ships with Spyglass (empty dict → tool defaults).
# Define custom params for your hardware:
PoseEstimParams.insert_params(
    params={"device": "cuda", "batch_size": 8},
    params_id="gpu_batch8",
    skip_duplicates=True,
)

PoseEstimParams()

# %% [markdown]
# ##### Step 2 — Estimation task (`PoseEstimSelection`)
#

# %%
# vid_group_id is created during Model.import_model() and stored
# as a secondary attribute on the Model table via ModelSelection.
vid_group_id = None
estim_key = None
if model_key:
    vid_group_id = (Model() & model_key).fetch1("vid_group_id")

    estim_key = PoseEstimSelection().insert_estimation_task(
        key={
            "model_id": model_key["model_id"],
            "vid_group_id": vid_group_id,
        },
        task_mode="trigger",  # 'load' to read existing output
        params={"device": "cpu"},  # swap to "cuda" for GPU
    )
    print(f"Selection ready: {estim_key}")
else:
    print("No model_key; skipping selection.")

# %% [markdown]
# ##### Step 3 — Run inference (`PoseEstim.populate()`)
#

# %%
# PoseEstim.populate() calls self.run_inference() internally,
# then stores the result as an ndx-pose NWB file.
pose_df = None
if estim_key:
    PoseEstim.populate(estim_key)
    pose_df = (PoseEstim() & estim_key).fetch1_dataframe()
    print(pose_df.head())
else:
    # Tutorial fallback: load from the bundled DLC example dataset.
    _dlc_example_dir = Path(
        "~/wrk/alt/DeepLabCut/examples/TEST-Alex-2025-09-08/videos"
    ).expanduser()
    _h5_files = sorted(_dlc_example_dir.glob("*DLC*.h5"))
    if _h5_files:
        pose_df = pd.read_hdf(str(_h5_files[0]))
        pose_df.index = np.arange(len(pose_df)) / 30.0
        print(
            f"Loaded DLC example: {_h5_files[0].name} ({len(pose_df)} frames)"
        )
        print(pose_df.head())
    else:
        print("No inference output or DLC example found")

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
PoseParams.insert_custom(
    params_name="tutorial_custom",
    orient={
        "method": "two_pt",  # Use two points to define orientation
        "bodypart1": "nose",
        "bodypart2": "tail_base",
        "interpolate": True,
        "smooth": True,
    },
    centroid={
        "method": "1pt",  # Use single point as centroid
        "points": {"point1": "nose"},
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
        "likelihood_thresh": 0.95,
    },
    skip_duplicates=True,
)

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
# > **Prerequisite** — `PoseEstim.populate()` and `PoseV2.populate()` both require
# > the model's `VidFileGroup` to be linked to a registered `Nwbfile` via
# > `VideoFile → TaskEpoch → Session`. Both will raise a `ValueError` if the link
# > is missing. Verify with:
# > ```python
# > VidFileGroup().get_nwb_file(vid_group_id)  # raises if not linked
# > ```

# %%
# Direct processing using PoseV2 helper methods.
# Production workflow: PoseEstimSelection → PoseEstim.populate()
#                    → PoseSelection → PoseV2.populate()
# See the Prerequisite note above for the full production example.
processed_df = None

if pose_df is not None:
    bodyparts = pose_df.columns.get_level_values(1).unique().tolist()
    print(f"Detected bodyparts: {bodyparts}")
    bp0 = bodyparts[0]
    bp1 = bodyparts[1] if len(bodyparts) > 1 else bodyparts[0]

    params = (PoseParams() & {"pose_params": "tutorial_custom"}).fetch1()
    orient_params = {**params["orient"], "bodypart1": bp0, "bodypart2": bp1}
    centroid_params = {**params["centroid"], "points": {"point1": bp0}}
    smooth_params = params["smoothing"]

    timestamps = pose_df.index.values.astype(float)
    sampling_rate = 1 / np.median(np.diff(timestamps))

    pv2 = PoseV2()
    thresh = smooth_params.get("likelihood_thresh", 0.95)
    filtered_df = pv2._apply_likelihood_threshold(pose_df.copy(), thresh)
    orientation = pv2._calculate_orientation(
        filtered_df, orient_params, timestamps, sampling_rate
    )
    centroid = pv2._calculate_centroid(filtered_df, centroid_params)
    centroid_smooth = pv2._smooth_position(
        centroid, timestamps, sampling_rate, smooth_params
    )
    velocity = pv2._calculate_velocity(
        centroid_smooth, timestamps, sampling_rate
    )

    processed_df = pd.DataFrame(
        {
            "position_x": centroid_smooth[:, 0],
            "position_y": centroid_smooth[:, 1],
            "orientation": orientation,
            "velocity": velocity,
        },
        index=timestamps,
    )

    print(processed_df.head())
    print(
        f"\nTime range: {processed_df.index[0]:.2f} "
        f"– {processed_df.index[-1]:.2f} s"
    )
    print(f'Mean speed: {processed_df["velocity"].mean():.2f} cm/s')
else:
    print("No pose data available; skipping PoseV2 processing.")

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
    # Production equivalent: (PoseV2() & pose_v2_key).fetch1_dataframe()

# %% [markdown]
# #### Method 2: fetch_obj()
#

# %% [markdown]
# Get raw pynwb objects for advanced analysis:
#

# %%
# Production: fetch raw pynwb objects via fetch_obj()
# objs = (PoseV2() & pose_v2_key).fetch_obj()
# Returns dict with: 'orient', 'centroid', 'velocity', 'smoothed_pose'
# as BehavioralTimeSeries / Position pynwb objects for custom NWB access.
#
# orient_obj = (PoseV2() & pose_v2_key).fetch_obj("orient")
# centroid_obj = (PoseV2() & pose_v2_key).fetch_obj(["centroid", "velocity"])

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
# #### Quiver Plot (Orientation Arrows)
#

# %%
if processed_df is not None:
    # Subsample for clarity
    step = len(processed_df) // 50
    subsampled = processed_df.iloc[::step]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot trajectory
    ax.plot(
        processed_df["position_x"],
        processed_df["position_y"],
        "gray",
        alpha=0.3,
        linewidth=0.5,
    )

    # Add orientation arrows
    arrow_length = 10  # cm
    ax.quiver(
        subsampled["position_x"],
        subsampled["position_y"],
        arrow_length * np.cos(subsampled["orientation"]),
        arrow_length * np.sin(subsampled["orientation"]),
        subsampled["velocity"],
        cmap="viridis",
        alpha=0.8,
        scale=200,
    )

    ax.set_xlabel("X position (cm)")
    ax.set_ylabel("Y position (cm)")
    ax.set_title("Trajectory with orientation (colored by speed)")
    ax.axis("equal")
    ax.invert_yaxis()
    plt.colorbar(label="Speed (cm/s)", ax=ax)
    plt.show()

# %% [markdown]
# ### [V1 vs V2 Comparison](#TableOfContents) <a id="Comparison"></a>
#

# %% [markdown]
# | Feature | V1 | V2 |
# |---------|----|----|
# | **Tables** | 10+ tables | 4 main tables |
# | **Tools** | DLC only | DLC, SLEAP, ndx-pose |
# | **Model Import** | Multi-step | Single method |
# | **Processing** | 6 separate tables | 1 PoseV2 table |
# | **Parameters** | 4 param tables | 1 PoseParams table |
# | **Storage** | Custom format | NWB via ndx-pose |
# | **Multi-camera** | Limited | Native support |
# | **Flexibility** | Fixed pipeline | Configurable workflows |
#
# **Migration Path**: V1 and V2 can coexist. Both write to `PositionOutput`,
# so downstream analyses work with either version.
#

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
