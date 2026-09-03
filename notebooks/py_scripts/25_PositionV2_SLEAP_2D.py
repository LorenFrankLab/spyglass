# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: spyglass-sleap
#     language: python
#     name: spyglass-sleap
# ---

# %% [markdown]
# ## Position Pipeline V2 — SLEAP
#

# %% [markdown]
# ## Overview

# %% [markdown]
# ### Notes

# %% [markdown]
# > **⚠️ Environment**: SLEAP requires its **own conda environment**,
# > `spyglass-sleap` (see `environments/environment_sleap.yml`).  It cannot
# > share the DeepLabCut environment — SLEAP's backend needs NumPy 2 /
# > Python ≥ 3.11, while DeepLabCut pins `numpy<2`.  **Run this notebook in the
# > `spyglass-sleap` environment** (if you registered the kernel under another
# > name locally, e.g. `sl2`, select that — the environment contents are what
# > matter).  For the DeepLabCut workflow, see the
# > [DLC notebook](./23_PositionV2_DLC_2D.ipynb).
#
# This is one notebook in a multi-part Spyglass series. See also
# [Setup](./00_Setup.ipynb) (environment & database),
# [Insert Data](./02_Insert_Data.ipynb) (DataJoint syntax), and the
# [DLC notebook](./23_PositionV2_DLC_2D.ipynb) (DeepLabCut Position V2).
#
# **Position V2** turns tracked video into clean position, orientation, and
# velocity data through a small set of tables:
#
# - **Reduces complexity**: just a few main tables
# - **Multi-tool support**: Works with both DeepLabCut and SLEAP
# - **Flexible workflows**: Train models, import pre-trained ones, or ingest
#   pose already computed elsewhere (ndx-pose NWB / SLEAP `.analysis.h5`)
# - **NWB-native storage**: Uses ndx-pose extension for standardized data
# - **Simplified processing**: Single PoseV2 table handles all post-processing
#
# This tutorial assumes you have already ingested an NWB session file. It covers
# the **SLEAP** workflow:
#
# - **Primary path**: Using a pre-trained SLEAP model (registered manually)
# - **Alternative path**: Training a SLEAP model from external `.slp` labels
# - Running SLEAP inference (sleap-nn / PyTorch backend) on videos
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
# - [Path A: Train a SLEAP Model](#PathA) - Train from external `.slp` labels
#     - Create labels in the SLEAP GUI (external)
#     - Define `ModelParams(tool="SLEAP")`
#     - `ModelSelection(training_labels_path=...)` → `Model.populate()`
# - [Path B: Use a Pre-Trained SLEAP Model](#PathB) - The runnable default
#     - Bootstrap a session & video from the bundled SLEAP clip
#     - Manually register a pre-trained SLEAP model (no `Model.load()` yet)
#     - Alternative entries: ndx-pose NWB import / load existing output
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
# - [Model Evaluation](#ModelEvaluation) - SLEAP `sleap-eval` (OKS / mAP)
# - [Video Generation](#VideoGeneration) - Create annotated outputs
#
# #### Reference
#
# - [Troubleshooting](#Troubleshooting) - Common issues & solutions
# - [External Resources](#Resources) - Documentation links
# - [Multi-Tool Support](#MultiTool) - SLEAP integration status
# - [JSON Parameters](#json-parameters) - `blob` search functionality

# %% [markdown]
# # Core Tutorial

# %% [markdown]
# ## Setup

# %% [markdown]
# > **Run this notebook in the `spyglass-sleap` environment.**  SLEAP and its
# > sleap-nn (PyTorch) backend are installed there; the DeepLabCut environment
# > is incompatible (NumPy / Python version conflicts).

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# Fail fast if not in the SLEAP environment. The sleap-nn / PyTorch backend
# lives only in `spyglass-sleap`; without it inference would fail much later
# with a confusing error.
try:
    import sleap_nn  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "SLEAP is not importable in this kernel. Run this notebook in the "
        "SLEAP environment (see environments/environment_sleap.yml) — the "
        "DeepLabCut environment is incompatible."
    ) from exc

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
    estim,
    train,
    video,
)

print("All imports successful!")

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
# 1. A skeleton (collection of body parts) plus training params define a model.
# 2. A video group is a collection of one or more files or calibrations.
# 3. Training takes place on a video group and results in a Model.
# 4. Pose estimation applies a given model to a given video group.
# 5. The final `PoseV2` table incorporates all secondary calculations, like
#     orientation and smoothing.

# %% [markdown]
# ## Which path is right for you? <a id="DecisionTree"></a>
#
# | Situation | Path |
# |---|---|
# | You have a **pre-trained SLEAP model** you want to run | **Path B** — [Use a pre-trained model](#PathB) (the runnable default) |
# | You want to **train a SLEAP model** from labeled `.slp` data | **Path A** — [Train a SLEAP model](#PathA) (advanced) |
#
# **SLEAP labeling and training projects are created in the SLEAP GUI**, not in
# Spyglass — there is no in-Spyglass project creation for SLEAP (unlike DLC's
# `Model.create_project()`).  This tutorial ships a pre-trained example model,
# so **Path B is the default that actually executes**.  Path A is shown for
# reference (it needs an external `.slp` labels file you do not have here).
#
# Both paths converge at the [Pose Estimation](#PoseEstim) section below.

# %% [markdown]
# ### Shared state for both paths
#
# The Pose Estimation and pose-processing sections read the variables
# initialized here **regardless of which path you run**. Defining them up front
# keeps Path A and Path B independent — neither leaves the others undefined.

# %%
# Shared state — set by whichever path you run (A: train, B: import).
model_key = None
nwb_file_name = None
inf_vid_path = None
training_vid_group_id = None
vid_group_id = None
config_path = None
skeleton_id = None
# Optional: point at a folder of existing *.analysis.h5 to load instead of
# triggering inference (set in the Pose Estimation section below).
sleap_output_dir = None

# %% [markdown]
# ## Path A: Train a SLEAP Model <a id="PathA"></a>
#
# **Goal**: Train a SLEAP model in Spyglass from a labeled `.slp` file.
#
# **Steps**:
# 1. Create and label a project in the **SLEAP GUI** (external to Spyglass),
#    exporting a `.slp` labels file.
# 2. Define `ModelParams(tool="SLEAP", params={...})` (model type, backbone,
#    epochs, batch size).
# 3. Insert `ModelSelection` with `training_labels_path="/path/to/labels.slp"`.
# 4. Call `Model.populate()`, which invokes the `sleap-train` CLI strategy.
#
# > **Why is this advanced?**  SLEAP frame extraction and labeling happen in the
# > SLEAP GUI, which produces the `.slp` labels file — Spyglass does **not**
# > create SLEAP projects or extract frames.  This tutorial bundles no `.slp`
# > labels file, so the code below is an **explanatory, non-executing example**:
# > `model_key` stays `None` and the notebook falls through to [Path B](#PathB).

# %% [markdown]
# ### Step 1 — Create labeled data (SLEAP GUI, external)
#
# Label frames in the SLEAP GUI and export a labels file:
#
# ```bash
# # Launch the SLEAP GUI (requires a display)
# sleap-label
# # ... create a project, label frames, then save as labels.slp
# ```
#
# See https://sleap.ai for the labeling workflow.

# %% [markdown]
# ### Step 2 — Define body parts and skeleton
#
# The skeleton's bodyparts and edges must match the keypoints and connections
# in your `.slp` file.  Edit both together — `training_edges` references the
# names in `training_bodyparts`.

# %%
# List the body parts your SLEAP model will track and the skeleton edges
# (pairs of connected bodyparts).  Both must match your .slp labels file.
training_bodyparts = ["A", "B"]
training_edges = [("A", "B")]

# %% [markdown]
# ### Step 3 — Configure params, selection, and train
#
# The training calls below only run when `run_sleap_training` is `True` (default
# `False`), so the notebook runs end-to-end without a `.slp` labels file. Set it
# `True` and point `training_labels_path` at a real `.slp` file to train.

# %%
# ── Path A (explanatory, non-executing) ──────────────────────────────────────
# Flip to True and provide a real .slp labels file to train a SLEAP model.
run_sleap_training = False

if run_sleap_training:
    # External labels file created in the SLEAP GUI.
    training_labels_path = "/path/to/your/labels.slp"

    # A skeleton describing the keypoints in the .slp file. SLEAP keypoints
    # are often not in the curated BodyPart table, so accept_new_bodyparts.
    skeleton_key = Skeleton().insert1(
        {
            "skeleton_id": "sleap_train_AB",
            "bodyparts": training_bodyparts,
            "edges": training_edges,
        },
        check_duplicates=False,
        skip_duplicates=True,
        accept_new_bodyparts=True,
    )
    skeleton_id = skeleton_key["skeleton_id"]

    # SLEAP training hyperparameters. The sleap-train CLI strategy consumes
    # these; model_type / backbone select the SLEAP architecture.
    train_params = {
        "model_type": "single_instance",
        "backbone": "unet",
        "max_epochs": 50,
        "batch_size": 4,
    }
    train_params_id = "sleap_train_demo"
    ModelParams.insert1(
        {
            "model_params_id": train_params_id,
            "params": train_params,
            "tool": "SLEAP",
            "skeleton_id": skeleton_id,
        },
        skip_duplicates=True,
    )

    # A VidFileGroup for the training video(s) referenced by the labels.
    training_vid_group_key = VidFileGroup.create_from_files(
        video_files=["/path/to/training_video.mp4"],
        description="sleap_training_videos",
    )
    training_vid_group_id = training_vid_group_key["vid_group_id"]

    # ModelSelection carries the external .slp path via training_labels_path.
    train_sel_key = {
        "model_params_id": train_params_id,
        "tool": "SLEAP",
        "vid_group_id": training_vid_group_id,
        "model_selection_id": train_params_id,
        "parent_id": None,
        "training_labels_path": training_labels_path,
    }
    ModelSelection.insert1(train_sel_key, skip_duplicates=True)

    # Model.populate() invokes the sleap-train CLI.
    Model.populate(train_sel_key, display_progress=True)
    model_key = (Model & train_sel_key).fetch1()
    print(f"Trained SLEAP model: {model_key['model_id']}")
else:
    print("Path A skipped (no .slp labels file). Falling through to Path B.")

# %% [markdown]
# ### Continue / resume training
#
# To train an existing model for more epochs, call `Model().train()` with the
# model's key. For SLEAP this resumes from the saved checkpoint
# (`sleap-train --base_checkpoint`). The cell below only runs when Path A
# produced a real trained model above.

# %%
if run_sleap_training and model_key is not None:
    new_model_key = Model().train(
        {"model_id": model_key["model_id"]}, epochs=50
    )
    print(f"Resumed training -> {new_model_key}")
else:
    print("Skipping resume (train a model in Path A first).")

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
# Model().populate(train_sel_key, make_kwargs={"device": "cuda"})  # training
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
# <details>
# <summary><b>Why no <code>Model.create_project()</code> for SLEAP?</b></summary>
#
# DeepLabCut creates and manages its project folder (frame extraction, labeling
# scaffolding) through `deeplabcut.create_new_project()`, which Spyglass wraps in
# `Model.create_project()`.  SLEAP handles project creation and labeling entirely
# in the **SLEAP GUI**, producing a `.slp` labels file.
#
# So for SLEAP there is no `Model.create_project()` and no in-Spyglass frame
# extraction: create the project and labels in SLEAP, then hand the `.slp` file
# to Spyglass via `ModelSelection.training_labels_path`.  Pre-trained models are
# registered manually (see [Path B](#PathB)).
#
# </details>

# %% [markdown]
# ## Path B: Use a Pre-Trained SLEAP Model <a id="PathB"></a>
#
# **Goal**: Register an existing SLEAP model in Spyglass and run inference.
#
# **What you'll accomplish**:
# - Bootstrap a tutorial session + video from the bundled SLEAP clip
# - **Manually register** a pre-trained SLEAP model (Skeleton + ModelParams +
#   VidFileGroup + ModelSelection + `Model().insert1(...)`)
# - Build the inference video group
#
# > **Why register manually?**  There is **no `Model.load()` for SLEAP yet** —
# > Spyglass cannot auto-import a SLEAP model the way it imports a DLC
# > `config.yaml`.  Instead we insert the `Skeleton`, `ModelParams(tool="SLEAP")`,
# > a `VidFileGroup`, a `ModelSelection`, and finally the `Model` row directly,
# > pointing `model_path` at the trained-model directory.
#
# > **Note**: Skip this section if you successfully trained a model in Path A.

# %% [markdown]
# ### Step 0 — Bootstrap a tutorial session & video
#
# <details>
# <summary><b>Tutorial helper — not for production use</b></summary>
#
# SLEAP inference and `PoseV2` processing both need the inference video's
# `VidFileGroup` to resolve to a registered Spyglass `Session` / `Nwbfile`.
#
# **In production**, register your NWB session with `insert_sessions()` and
# ensure `VideoFile` rows exist *before* registering a model.
#
# The `bootstrap_from_video_paths()` helper (from the tool-agnostic
# `tests/position/v2/make_example_dlc_project.py`) creates minimal dummy entries
# so this tutorial works without a recorded session, and is maintained alongside
# the test suite to stay in sync with the production API.
#
# **On a shared database, please delete the dummy entries when done.**
#
# </details>

# %%
# Path B reuses a bootstrap helper from the Spyglass test suite, so it needs a
# source checkout — pip-only installs omit the tests/ directory. (Not for
# production; bring your own registered Session / VideoFile instead.)
import sys

import spyglass

tests_v2_dir = Path(spyglass.__file__).parents[2] / "tests" / "position" / "v2"
if not tests_v2_dir.exists():
    raise FileNotFoundError(
        f"Tutorial helper not found at {tests_v2_dir}. Path B needs a Spyglass "
        "source checkout (the tests/ directory is absent from a pip install). "
        "Clone the repository, or register your own Session / VideoFile and "
        "skip the bootstrap step."
    )
if str(tests_v2_dir) not in sys.path:
    sys.path.insert(0, str(tests_v2_dir))

# %%
from make_example_dlc_project import bootstrap_from_video_paths  # noqa: E402

# ── Bundled SLEAP example data ────────────────────────────────────────────────
# Legacy UNet model (best_model.h5 + training_config.json) tracking
# bodyparts "A" and "B"; plus a small 3-frame video for inference.
repo_root = Path(spyglass.__file__).parents[2]
sleap_data_dir = repo_root / "tests" / "_data" / "sleap"
sleap_model_dir = sleap_data_dir / "model"
sleap_video = sleap_data_dir / "small_robot_3_frame.mp4"
bodyparts = ["A", "B"]

# Check that the bundled example data is present (e.g. a partial checkout).
if not sleap_video.exists():
    raise FileNotFoundError(
        f"Bundled SLEAP video not found: {sleap_video}. Expected it under "
        "tests/_data/sleap/ in a Spyglass source checkout."
    )
if not (sleap_model_dir / "best_model.h5").exists():
    raise FileNotFoundError(
        f"Bundled SLEAP model not found: {sleap_model_dir / 'best_model.h5'}. "
        "Expected the trained model under tests/_data/sleap/model/."
    )

if model_key is None:
    print("Bootstrapping tutorial Spyglass session from SLEAP video...")
    nwb_file_name, inf_vid_path = bootstrap_from_video_paths(
        [sleap_video],
        nwb_stem="sleap_tutorial_sess",
        task_name="sleap_tutorial",
        camera_name="sleap_tutorial",
    )
    print(f"  nwb_file_name  : {nwb_file_name}")
    print(f"  inf_vid_path   : {inf_vid_path}")

# %% [markdown]
# ### Step 1 — Register the SLEAP model manually
#
# Insert each piece by hand: a `Skeleton`, a `ModelParams` row with
# `tool="SLEAP"`, a `VidFileGroup` for the model's training video, a
# `ModelSelection`, and finally the `Model` row pointing at the trained-model
# directory on disk.

# %%
from spyglass.position.v2.train import default_pk_name  # noqa: E402

if model_key is None:
    # 1a. Skeleton with bodyparts A and B (match the model). These bodyparts
    # are not in the curated BodyPart table, so accept_new_bodyparts=True.
    skeleton_key = Skeleton().insert1(
        {
            "skeleton_id": "sleap_AB",
            "bodyparts": bodyparts,
            "edges": [("A", "B")],
        },
        check_duplicates=False,
        skip_duplicates=True,
        accept_new_bodyparts=True,
    )
    skeleton_id = skeleton_key["skeleton_id"]
    print(f"Skeleton: {skeleton_id} -> {Skeleton().get_bodyparts(skeleton_id)}")

    # 1b. ModelParams for the SLEAP tool, linked to the skeleton.
    model_params_key = ModelParams().insert1(
        {
            "tool": "SLEAP",
            "model_params_id": "sleap_tutorial",
            "params": {
                "model_type": "single_instance",
                "backbone": "unet",
            },
            "skeleton_id": skeleton_id,
        },
        skip_duplicates=True,
    )
    print(f"ModelParams: {model_params_key}")

    # 1c. VidFileGroup for the model's "training" video (reuse the clip).
    model_vid_group = VidFileGroup.create_from_files(
        video_files=[str(inf_vid_path)],
        description="sleap_tutorial_model_train_video",
        vid_group_id="sleap_train_grp",
    )
    training_vid_group_id = model_vid_group["vid_group_id"]
    print(f"Model VidFileGroup: {model_vid_group}")

    # 1d. ModelSelection pairing params + train video group.
    # default_pk_name builds a stable, human-readable id from the inputs.
    model_sel_key = {
        **model_params_key,
        **model_vid_group,
        "model_selection_id": default_pk_name(
            "ms-sleap",
            {"model_params_id": model_params_key["model_params_id"]},
        ),
    }
    ModelSelection().insert1(model_sel_key, skip_duplicates=True)
    print(f"ModelSelection: {model_sel_key['model_selection_id']}")

    # 1e. Register the model row. Since there is no Model.load() for SLEAP yet,
    # we add the row directly, pointing model_path at the pre-trained model
    # directory on disk.
    model_id = "sleap_tutorial_model"
    stored_path = str(sleap_model_dir.resolve())
    if Model() & {"model_id": model_id}:
        model_key = (Model() & {"model_id": model_id}).fetch1("KEY")
    else:
        model_key = {
            **model_sel_key,
            "model_id": model_id,
            "model_path": stored_path,
        }
        Model().insert1(model_key, allow_direct_insert=True)
    print(f"Model: {model_id} -> {stored_path}")

# %% [markdown]
# #### Validate model (Path B)

# %%
if model_key and not Model() & model_key:
    raise ValueError(f"Model entry not found : {model_key}")

if model_key:
    # Read the full Model row to get its params id and tool.
    model_row = (Model() & model_key).fetch1()
    model_params = (
        ModelParams()
        & {
            "model_params_id": model_row["model_params_id"],
            "tool": model_row["tool"],
        }
    ).fetch1()
    skeleton_id = model_params.get("skeleton_id")
    if not (Skeleton() & {"skeleton_id": skeleton_id}).fetch1("KEY"):
        raise ValueError(f"Skeleton not found for model: {model_key}")
    if training_vid_group_id and not (
        VidFileGroup() & {"vid_group_id": training_vid_group_id}
    ):
        raise ValueError(f"Video group not found: {training_vid_group_id}")
    print("Path B model registration validated")

# %% [markdown]
# ### Step 2 — Build the inference video group
#
# Pose estimation uses **two separate `VidFileGroup` entries** (a structure that
# supports multi-camera recordings):
#
# - **Training group** (`training_vid_group_id`): the model's labeled videos,
#   linked to `ModelSelection`.  Used by `get_nwb_file()` to resolve the parent
#   session.
# - **Inference group** (`vid_group_id`): the video(s) to run inference on,
#   linked to `PoseEstimSelection`.
#
# Below we build the inference group and verify it resolves to a registered NWB
# session.

# %%
if model_key and inf_vid_path:
    inf_vid_group = VidFileGroup.create_from_files(
        video_files=[str(inf_vid_path)],
        description="sleap_tutorial_inference",
        vid_group_id="sleap_infer_grp",
    )
    vid_group_id = inf_vid_group["vid_group_id"]
    # Verify the group resolves to a registered NWB session.
    resolved = VidFileGroup().get_nwb_file(vid_group_id)
    print(f"Inference VidFileGroup: {vid_group_id} -> {resolved}")

# %%
if skeleton_id:
    fig = (Skeleton & {"skeleton_id": skeleton_id}).show_skeleton()
    plt.show()

# %% [markdown]
# #### Alternative entry — pose already in NWB (ndx-pose)
#
# Pose already exported to NWB via the ndx-pose extension (e.g. SLEAP through
# `sleap_io`'s `Labels.export_nwb()`) skips inference entirely: register the
# session NWB with `insert_sessions(...)`, then ingest with `ImportedPose` — the
# canonical V2 entry point for pre-computed pose (despite living under the
# `spyglass.position.v1` module path):
#
# ```python
# ImportedPose().insert_from_nwbfile("sleap_output.nwb")
# # add import_to_v2=True to also expose it via the V2 inference tables
# ```
#
# Path B below runs live inference instead, so there is nothing to ingest here.

# %% [markdown]
# #### Alternative entry — load existing SLEAP output
#
# Already ran SLEAP and have `.analysis.h5` (or `.slp`) predictions on disk? Set
# `sleap_output_dir` below to that folder; the Pose Estimation step then uses
# `task_mode='load'` (parsed by `parse_sleap_analysis_h5`) instead of triggering
# inference.

# %%
# Point at a folder of existing *.analysis.h5 to load instead of running
# inference; leave as None to trigger inference below.
sleap_output_dir = None
# sleap_output_dir = "/path/to/sleap/output"  # contains *.analysis.h5

# %% [markdown]
# ## Pose Estimation <a id="PoseEstim"></a>
#

# %%
# A model must be available from either Path A or Path B above.
if model_key is None:
    raise ValueError(
        "Complete Path A (train a SLEAP model) or Path B (register a "
        "pre-trained model) before running pose estimation."
    )

# %% [markdown]
#
# **Goal**: Run SLEAP pose inference on videos using the registered model
#
# **What you'll accomplish**:
# - Configure inference parameters (batch size)
# - Set up estimation tasks with video groups
# - Run inference (sleap-nn / PyTorch backend) and validate results
# - Handle common errors gracefully

# %% [markdown]
# #### Running Inference via the Pipeline
#
# Inference in V2 follows a three-step Spyglass pattern:
#
# 1. **`PoseEstimParams`** — name a set of inference parameters
#     (batch size, etc.)
# 2. **`PoseEstimSelection`** — pair a model with a video group and choose
#     `task_mode='trigger'` (run SLEAP inference) or `'load'` (read existing
#     `.analysis.h5` output)
# 3. **`PoseEstim.populate()`** — executes inference via the sleap-nn backend
#     and stores results in an NWB file via ndx-pose
#
# > **`task_mode='load'` vs `ImportedPose`**: Use `task_mode='load'` when SLEAP
# > has already written `.analysis.h5` output on disk and you want to read it
# > into Spyglass. Use `ImportedPose` when your results already exist in NWB
# > format from another pipeline.

# %% [markdown]
# ##### Step 1 — Inference parameters (`PoseEstimParams`)
#
# Here `params_hash` uniquely identifies a param set; inserting a new entry
# whose `params` match an existing row raises an error.

# %%
# Name a set of inference parameters. The best available device (GPU if present,
# else CPU — and the least-loaded GPU when there are several) is selected
# automatically at run time — you don't set it here.
batch_size = 4
params_result = PoseEstimParams.insert_params(
    params={"batch_size": batch_size},
    params_id=f"batch{batch_size}",
    skip_duplicates=True,
)
print(f"Inserted PoseEstimParams: {params_result}")

PoseEstimParams & {"params.batch_size": batch_size}

# %% [markdown]
# ##### Step 2 — Estimation task (`PoseEstimSelection`)
#
# Pair the model with the inference video group built in Path B.

# %%
# Create estimation selection
estim_key = None

if not (inf_vid_path and model_key and vid_group_id):
    raise ValueError("Missing video or model - skipping PoseEstimSelection")

# Check/create required params entry
params_id = "batch4"
estim_params_key = PoseEstimParams.insert_params(
    params={"batch_size": 4},
    params_id=params_id,
    skip_duplicates=True,
)

# Create estimation selection. task_mode='trigger' runs sleap-nn inference;
# 'load' reads an existing SLEAP .analysis.h5 from output_dir.
estim_key = {
    "model_id": model_key["model_id"],
    "vid_group_id": vid_group_id,
    **estim_params_key,
}
PoseEstimSelection().insert_estimation_task(
    estim_key,
    task_mode="load" if sleap_output_dir else "trigger",
    output_dir=str(sleap_output_dir) if sleap_output_dir else "",
    skip_duplicates=True,
)
print(f"Created estimation selection: {estim_key}")

# %%
# Inspect video groups if available
if training_vid_group_id:
    print(
        "Training videos:",
        len(VidFileGroup().File() & {"vid_group_id": training_vid_group_id}),
    )
if vid_group_id:
    print(
        "Inference videos:",
        len(VidFileGroup().File() & {"vid_group_id": vid_group_id}),
    )

# %% [markdown]
# ##### Step 3 — Run inference (`PoseEstim.populate()`)
#
# `task_mode='trigger'` runs SLEAP inference on the modern **sleap-nn (PyTorch)**
# backend.

# %%
if not estim_key:
    raise ValueError("No estim_key available - check previous steps")

PoseEstim.populate(estim_key, display_progress=True)

# Fetch the results
pose_df = (PoseEstim() & estim_key).fetch1_dataframe()
print(pose_df.head())

# %% [markdown]
# <details>
# <summary><b>Troubleshooting Pose Estimation</b></summary>
#
# ### Common Issues & Solutions
#
# #### **Error**: "No .analysis.h5 output files found"
# **Cause**: `task_mode='load'` but SLEAP output not present
# **Solution**:
# - For automatic inference, use `task_mode='trigger'` (sleap-nn backend)
# - For `task_mode='load'`, ensure `output_dir` contains the
#   `*.analysis.h5` produced by `sleap-track` / `sleap-export`
#
# #### **Error**: "CUDA out of memory"
# **Cause**: GPU memory insufficient
# **Solution**: lower the batch size so less data fits in GPU memory at once:
# ```python
# PoseEstimParams.insert_params(
#     params={"batch_size": 1},
#     params_id="batch1"
# )
# ```
#
# #### **Error**: "Model not found" or "VidFileGroup not found"
# **Cause**: Model registration incomplete
# **Solution**: Run the validation checkpoint in Path B to diagnose
#
# #### **Error**: "KeyError: analysis_file_name"
# **Cause**: NWB file path resolution issue
# **Solution**: Ensure the inference video group is linked to a registered
# session (`VidFileGroup().get_nwb_file(vid_group_id)`)
#
# #### **Warning**: "Low likelihood values"
# **Cause**: Model doesn't generalize to your videos
# **Solution**:
# - Adjust `likelihood_thresh` in processing parameters
# - Consider retraining / fine-tuning the SLEAP model with more labels
#
# #### **Note on device**
# - The best available device (GPU if present, otherwise CPU) is selected
#   automatically at run time — you don't set it here.
# - On a multi-GPU host, a bare `"cuda"` resolves to the least-loaded card,
#   preferring one that is completely unused. Prefer it over a hardcoded
#   `"cuda:0"`, which does not move when that GPU is busy.
# - If every GPU is full, populate raises before inference starts and lists
#   free memory per device, rather than dying with an out-of-memory error
#   part-way through a long run.
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
# - Match parameter settings to your SLEAP skeleton

# %% [markdown]
# Before processing pose data, define processing parameters. PoseParams stores
# configuration for:
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
# | **SLEAP 2-keypoint skeleton** | `two_pt` (A→B) | `2pt` (A, B) | `moving_avg`, 50 ms | This tutorial's bundled A/B model |
# | **SLEAP head→tail skeleton** | `two_pt` (nose→tail) | `1pt` nose | `gaussian` | Any two keypoints for direction |
# | **Single keypoint** | `none` | `1pt` | `savgol` | Head-fixed or whole-body centroid |
# | **Standard 2-LED navigation** | `two_pt` (green→red) | `2pt` | `moving_avg`, 50 ms | Classic Frank Lab LED tracking |
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
# interpolated over before smoothing.  Typical range: `0.9` (strict) → `0.5`
# (permissive); start at `0.9` and lower if too many frames are dropped.

# %% [markdown]
# #### View Available Parameter Sets
#

# %%
PoseParams()

# %% [markdown]
# Inspect the full JSON contents of any named parameter set:

# %%
# Pretty-print a named PoseParams entry (any name from the table above).
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
# Example: find entries by sub-field value.
# DataJoint dot-notation works for equality and IN-list queries:
#   PoseParams & {"orient.method": "two_pt"}
#   PoseParams & {"smoothing.smooth": True}
#   PoseParams & {"centroid.method": ["1pt", "2pt"]}
PoseParams & {"orient.method": "two_pt"}

# %% [markdown]
# #### Use Default Parameters
#

# %% [markdown]
# The defaults target LED tracking. The bundled SLEAP A/B skeleton uses the
# custom parameters below, but it's still useful to insert the defaults:
#

# %%
PoseParams.insert_default(skip_duplicates=True)
params_key = {"pose_params_id": "default"}  # or "4LED_default" / "single_LED"
PoseParams()

# %% [markdown]
# #### Create Custom Parameters
#
# A 2-point centroid/orientation on the SLEAP skeleton's keypoints.  The
# bundled model tracks bodyparts **A** and **B**.
#
# > **⚠️ Calibration note**: This tutorial has **no camera calibration**, so
# > `meters_per_pixel` falls back to `1.0` and coordinates are scaled
# > pixels-times-100.  The A–B separation is ~100 px → ~10,000 in these units,
# > so `max_LED_separation` and `max_cm_to_interp` must be very generous
# > (`60000.0`) to avoid rejecting every frame.  With real calibration these
# > would be physical centimeters at typical values (e.g. ~15 cm).

# %%
# Create custom pose parameters from the model's bodyparts.
if not model_key:
    raise ValueError("No model key available - cannot create custom PoseParams")

# Read the full Model row, then look up its ModelParams.
model_row = (Model() & model_key).fetch1()
model_params = (
    ModelParams()
    & {
        "model_params_id": model_row["model_params_id"],
        "tool": model_row["tool"],
    }
).fetch1()
skeleton_id = model_params["skeleton_id"]
bp_tbl = Skeleton.BodyPart() & {"skeleton_id": skeleton_id}
skeleton_parts = list(bp_tbl.fetch("bodypart"))

# Two-keypoint custom parameters using PoseParams.insert_custom.
bodypart1, bodypart2 = skeleton_parts[0], skeleton_parts[1]

pose_params_id = "sleap_AB_2pt"
PoseParams().insert_custom(
    params_name=pose_params_id,
    orient={
        "method": "two_pt",
        "bodypart1": bodypart1,
        "bodypart2": bodypart2,
        "interpolate": True,
        "smooth": False,
    },
    centroid={
        "method": "2pt",
        "points": {"point1": bodypart1, "point2": bodypart2},
        # No camera calibration -> generous separation threshold (see note).
        "max_LED_separation": 60000.0,
    },
    smoothing={
        "interpolate": True,
        "interp_params": {
            "max_pts_to_interp": 10,
            "max_cm_to_interp": 60000.0,
        },
        "smooth": True,
        "smoothing_params": {
            "method": "moving_avg",
            "smoothing_duration": 0.05,
        },
        "likelihood_thresh": 0.0,
        "velocity_smoothing_std_dev": 0.1,
    },
    skip_duplicates=True,
)

print("Created SLEAP pose parameters via insert_custom")

# %% [markdown]
# Inspect the parameters:
#

# %%
(PoseParams() & {"pose_params_id": pose_params_id}).fetch1()

# %% [markdown]
# Search the table:

# %%
PoseParams & {"smoothing.smooth": True}

# %% [markdown]
# ## Data Processing <a id="PoseV2"></a>
#

# %% [markdown]
#
# **Goal**: Process raw pose estimates into final position trajectories
#
# **What you'll accomplish**:
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

# Use the SLEAP A/B params if available, otherwise default
params_name = pose_params_id
if not PoseParams() & {"pose_params_id": params_name}:
    PoseParams.insert_default(skip_duplicates=True)
    params_name = "default"

pose_selection_key = {**estim_key, "pose_params_id": params_name}
PoseSelection().insert1(
    pose_selection_key, skip_duplicates=True, ignore_extra_fields=True
)

print("")
print(f"Processing with params: {params_name}")
PoseV2.populate(pose_selection_key)

processed_df = (PoseV2() & pose_selection_key).fetch1_dataframe()
print(f"Processed data: {processed_df.shape[0]} timepoints")
print(processed_df.head())

# %% [markdown]
# The PoseV2.populate() pipeline performs:
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
# Verify the position processing worked correctly:

# %%
if processed_df is None:
    raise ValueError("No processed data - ensure PoseV2.populate() completed")

required = ["position_x", "position_y", "orientation", "speed"]
missing = [col for col in required if col not in processed_df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

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
# **Goal**: Access processed position data for analysis and visualization
#
# **What you'll accomplish**:
# - Retrieve data as pandas DataFrames or raw NWB objects
# - Generate trajectory and time series visualizations
# - Understand data structure and coordinate systems
# - Export results for further analysis

# %% [markdown]
# ### [Visualization](#TableOfContents) <a id="Visualization"></a>
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
# Training a SLEAP model from external `.slp` labels — model params,
# `ModelSelection.training_labels_path`, and `Model.populate()` (via the
# `sleap-train` CLI) — is covered in **[Path A](#PathA)** at the top of this
# notebook. The steps below (Model Evaluation, Video Generation) assume a
# `model_key` from either Path A or Path B.
#
# **Continue / resume training** — to train an existing model for more epochs,
# call `Model().train({"model_id": model_key["model_id"]}, epochs=50)`. For
# SLEAP this resumes from the saved checkpoint (`sleap-train --base_checkpoint`)
# rather than starting fresh. See the "Continue / resume training" cell in
# [Path A](#PathA); it needs a real trained `model_key`.

# %% [markdown]
# ## Model Evaluation <a id="ModelEvaluation"></a>
#

# %% [markdown]
#
# **Goal**: Evaluate SLEAP model performance
#
# **What you'll accomplish**:
# - Understand how SLEAP model accuracy is assessed
# - Learn which metrics SLEAP reports (OKS, mAP)
#
# ### Background
#
# SLEAP evaluation uses the **`sleap-eval`** CLI, which compares predicted
# keypoints against held-out labeled frames and reports object keypoint
# similarity (**OKS**) and mean average precision (**mAP**) — unlike DeepLabCut,
# which reports per-iteration train/test pixel-error curves.
#
# Because this tutorial **imports a pre-trained model** (Path B) with no
# training history in Spyglass, the evaluation below is explanatory. Train a
# model via Path A (or run `sleap-eval` directly on your model + labels) to
# obtain real metrics.
#
# ```bash
# # Evaluate a trained SLEAP model against labeled frames:
# sleap-eval /path/to/model_dir --labels /path/to/labels.slp
# # Reports OKS / mAP for the predicted keypoints.
# ```

# %% [markdown]
# ### Check Model Evaluation Availability

# %%
if model_key is None:
    raise ValueError("No model key available - cannot check evaluation support")

# Read the model's tool from its full row.
model_tool = (Model() & model_key).fetch1("tool")
evaluation_supported = model_tool.upper() == "SLEAP"

if not evaluation_supported:
    print(f"Model tool is '{model_tool}', not SLEAP - skipping SLEAP eval.")
else:
    print(
        "SLEAP model detected. Use the `sleap-eval` CLI on the model dir and "
        "a labeled .slp file to compute OKS / mAP. This tutorial's pre-trained "
        "model has no bundled labels, so no metrics are computed here."
    )

# %% [markdown]
# ## Video Generation <a id="VideoGeneration"></a>
#

# %% [markdown]
# Once a `PoseV2` entry is populated, `make_video()` renders the source video
# with the tracked pose overlaid. Call it on a single populated entry:
#
# ```python
# output = (PoseV2() & pose_selection_key).make_video()
# print(output)  # path to the rendered <video>_pose.mp4
# ```
#
# Useful arguments: `output_path` (where to write), `cm_to_pixels` (scale
# factor), and `percent_frames` (render a fraction of frames for a quick
# preview, e.g. `percent_frames=0.1`).
#
# **Tip**: Preview with a small `percent_frames` before rendering the full
# clip.

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
# <summary><b>Model Registration</b> (Click to expand)</summary>
#
# #### **Model Registration Issues**
#
# **"Permission denied" or "Access forbidden"**
# - Verify database user permissions - Ask admin to add body parts
# - Check if you're connected to the right database
# - Ensure you have insert/update privileges
# - Contact your database administrator
#
# **"No sessions found matching video paths"**
# - Register your session first: `insert_sessions('your_file.nwb')`
# - Check video file paths match VideoFile entries
# - Use the bootstrap function for tutorials (NOT production)
#
# **"Model registration failed" - SLEAP models**
# - Verify the SLEAP model directory exists and is complete
#   (e.g. `best_model.h5` + `training_config.json` for the legacy backend)
# - Ensure `model_path` points at the trained-model directory
# - Confirm the skeleton bodyparts match the model's keypoints
#
# **"Model registration failed" - NWB files**
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
# # Lower the batch size so less data fits in GPU memory at once
# PoseEstimParams.insert_params(
#     params={"batch_size": 1},
#     params_id="batch1"
# )
# ```
#
# **"No .analysis.h5 output files found"**
# - For automatic inference: use `task_mode="trigger"` (sleap-nn backend)
# - For `task_mode="load"`: check `output_dir` exists and contains the
#   `*.analysis.h5` files written by SLEAP
#
# **"Inference taking too long"**
# - Reduce batch size: `batch_size: 4` or `batch_size: 1`
# - Run on a machine with a GPU (the GPU is used automatically when present)
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
# **"Every frame rejected / empty trajectory"**
# - Without camera calibration, coordinates are scaled pixels; raise
#   `max_LED_separation` and `max_cm_to_interp` (see the calibration note)
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
# For the full V1 → V2 migration guide, see the Position Pipelines page:
# https://lorenfranklab.github.io/spyglass/latest/GettingStarted/POSITION/
#
# <details><summary><b>JSON Parameter Support</b> (Click to Expand)</summary>
#
# ### JSON Parameters
#
# Position V2 stores parameters in **native JSON columns**, so you can query
# nested fields directly with dot notation:
#
# ```python
# filtered = ModelParams & {'params.learning_rate': 0.001}
# batch_filtered = ModelParams & {'params.batch_size': [4, 8, 16]}
# nested_query = ModelParams & {'params.training.max_epochs': 100}
# ```
#
# **Benefits:** database-level filtering/indexing (no custom iteration),
# built-in DataJoint dot-notation translation, and universal JSON support.
#
# **Migration Note:** Existing blob parameter tables continue to work unchanged.
# Consider JSON columns for new parameter tables requiring complex querying.
#
# </details>
#
# <details>
# <summary><b>SLEAP Support Status</b> <a id="MultiTool"></a> (Click to expand)</summary>
#
# ### SLEAP Integration Status
#
# Position V2 has working SLEAP support:
#
# #### **Available Now**
#
# - **Inference**: SLEAP inference runs through `PoseEstim.populate(
#     task_mode='trigger')` on the modern **sleap-nn (PyTorch)** backend.
# - **Training**: Train via the `sleap-train` CLI strategy by passing an
#     externally-created `.slp` labels file as
#     `ModelSelection.training_labels_path` (SLEAP labels are made in the
#     SLEAP GUI, not in Spyglass).
# - **Pre-trained models**: Registered **manually** — there is no
#     `Model.load()` for SLEAP yet.  Insert `Skeleton`, `ModelParams(
#     tool="SLEAP")`, `VidFileGroup`, `ModelSelection`, then a `Model` row with
#     `model_path` pointing at the trained-model directory (see Path B).
# - **Load existing output**: Read SLEAP `.analysis.h5` via
#     `PoseEstimSelection(task_mode='load')` (parsed by
#     `parse_sleap_analysis_h5`), or ingest ndx-pose NWB via
#     `ImportedPose.insert_from_nwbfile()`.
#
# #### **Environment requirement**
#
# - SLEAP requires its **own conda environment**, `spyglass-sleap`
#     (`environments/environment_sleap.yml`).  It is incompatible with the
#     DeepLabCut environment (NumPy 2 / Python ≥ 3.11 vs `numpy<2`).
#
# #### **Not yet available**
#
# - **`Model.load()` for SLEAP** (auto-import a pre-trained model directory).
# - **In-Spyglass project creation / frame extraction** (use the SLEAP GUI).
#
# </details>

# %% [markdown]
# ## External Resources <a id="Resources"></a>
#
# ### Documentation & Guides
# - [Spyglass Documentation](https://lorenfranklab.github.io/spyglass/)
# - [Position V2 API Reference](https://lorenfranklab.github.io/spyglass/api/position/v2/)
# - [SLEAP Documentation](https://sleap.ai)
# - [sleap-nn (PyTorch backend)](https://nn.sleap.ai)
# - [sleap-io (file I/O & NWB export)](https://io.sleap.ai)
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
# - [23_PositionV2_DLC_2D.ipynb](./23_PositionV2_DLC_2D.ipynb) - Position V2 with DeepLabCut
# - [21_DLC.ipynb](./21_DLC.ipynb) - Legacy Position V1 pipeline
# - [26_Linearization.ipynb](./26_Linearization.ipynb) - Convert 2D → 1D position
# - [41_Decoding_Clusterless.ipynb](./41_Decoding_Clusterless.ipynb) \- Use
#     position for decoding
#

# %% [markdown]
#
# **Tutorial Complete!**
#
# You've learned the fundamentals of the Position V2 SLEAP pipeline. For
# questions or feedback, please use the resources above.
