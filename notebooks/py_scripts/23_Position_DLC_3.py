# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
# ---

# # Position - DeepLabCut Estimation

# ## Overview
#

# _Developer Note:_ if you may make a PR in the future, be sure to copy this
# notebook, and use the `gitignore` prefix `temp` to avoid future conflicts.
#
# This is one notebook in a multi-part series on Spyglass.
#
# - To set up your Spyglass environment and database, see
#   [the Setup notebook](./00_Setup.ipynb)
# - For additional info on DataJoint syntax, including table definitions and
#   inserts, see
#   [the Insert Data notebook](./01_Insert_Data.ipynb)
#
# This tutorial will extract position via DeepLabCut (DLC). It will walk through...
# - executing pose estimation
# - processing the pose estimation output to extract a centroid and orientation
# - inserting the resulting information into the `IntervalPositionInfo` table
#
# This tutorial assumes you already have a model in your database. If that's not
# the case, you can either [train one from scratch](./21_Position_DLC_1.ipynb)
# or [load an existing project](./22_Position_DLC_2.ipynb).

# Here is a schematic showing the tables used in this pipeline.
#
# ![dlc_scratch.png|2000x900](./../notebook-images/dlc_scratch.png)

# ### Table of Contents<a id='TableOfContents'></a>
#
# - [Imports](#imports)
# - [GPU](#gpu)
# - [`DLCPoseEstimation`](#DLCPoseEstimation1)
# - [`DLCSmoothInterp`](#DLCSmoothInterp1)
# - [`DLCCentroid`](#DLCCentroid1)
# - [`DLCOrientation`](#DLCOrientation1)
# - [`DLCPos`](#DLCPos1)
# - [`DLCPosVideo`](#DLCPosVideo1)
# - [`PosSource`](#PosSource1)
# - [`IntervalPositionInfo`](#IntervalPositionInfo1)
#
# __You can click on any header to return to the Table of Contents__

# ### [Imports](#TableOfContents)
#

# +
import os
import datajoint as dj
from pprint import pprint

import spyglass.common as sgc
import spyglass.position.v1 as sgp

# change to the upper level folder to detect dj_local_conf.json
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
dj.config.load("dj_local_conf.json")  # load config for database connection info

# ignore datajoint+jupyter async warnings
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
# -

# ### [GPU](#TableOfContents)

# For longer videos, we'll need GPU support. The cell below determines which core
# has space and set the `gputouse` variable accordingly.

sgp.dlc_utils.get_gpu_memory()

# Set GPU core:

gputouse = 1  ## 1-9

# #### [DLCPoseEstimation](#TableOfContents) <a id='DLCPoseEstimation1'></a>
#
# With our trained model in place, we're ready to set up Pose Estimation on a
# behavioral video of your choice. We can select a video with `nwb_file_name` and
# `epoch`, making sure there's an entry in the `VideoFile` table.

nwb_file_name = "J1620210604_.nwb"
epoch = 14
sgc.VideoFile() & {"nwb_file_name": nwb_file_name, "epoch": epoch}

# Using `insert_estimation_task` will convert out video to be in .mp4 format (DLC
# struggles with .h264) and determine the directory in which we'll store the pose
# estimation results.
#
# - `task_mode` (trigger or load) determines whether or not populating
#   `DLCPoseEstimation` triggers a new pose estimation, or loads an existing.
# - `video_file_num` will be 0 in almost all
#   cases.
# - `gputouse` was already set during training. It may be a good idea to make sure
#   that core is still free before moving forward.

pose_estimation_key = sgp.DLCPoseEstimationSelection.insert_estimation_task(
    {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "video_file_num": 0,
        **model_key,
    },
    task_mode="trigger",
    params={"gputouse": gputouse, "videotype": "mp4"},
)

# _Note:_ Populating `DLCPoseEstimation` may take some time for full datasets

sgp.DLCPoseEstimation().populate(pose_estimation_key)

# Let's visualize the output from Pose Estimation

(sgp.DLCPoseEstimation() & pose_estimation_key).fetch_dataframe()

# #### [DLCSmoothInterp](#TableOfContents) <a id='DLCSmoothInterp1'></a>

# After pose estimation, we can interpolate over low likelihood periods and smooth
# the resulting position.
#
# First we define some parameters. We can see the default parameter set below.

pprint(sgp.DLCSmoothInterpParams.get_default())
si_params_name = "default"

# To change any of these parameters, one would do the following:
#
# ```python
# si_params_name = "your_unique_param_name"
# params = {
#     "smoothing_params": {
#         "smoothing_duration": 0.00,
#         "smooth_method": "moving_avg",
#     },
#     "interp_params": {"likelihood_thresh": 0.00},
#     "max_plausible_speed": 0,
#     "speed_smoothing_std_dev": 0.000,
# }
# sgp.DLCSmoothInterpParams().insert1(
#     {"dlc_si_params_name": si_params_name, "params": params},
#     skip_duplicates=True,
# )
# ```

# We'll create a dictionary with the correct set of keys for the `DLCSmoothInterpSelection` table

si_key = pose_estimation_key.copy()
fields = list(sgp.DLCSmoothInterpSelection.fetch().dtype.fields.keys())
si_key = {key: val for key, val in si_key.items() if key in fields}
si_key

# We can insert all of the bodyparts we want to process into
# `DLCSmoothInterpSelection`. Here are the bodyparts we have available to us:

pprint((sgp.DLCPoseEstimation.BodyPart & pose_estimation_key).fetch("bodypart"))

# We can use `insert1` to insert a single bodypart, but would suggest using `insert` to insert a list of keys with different bodyparts.

# We'll set a list of bodyparts and then insert them into
# `DLCSmoothInterpSelection`.

bodyparts = ["greenLED", "redLED_C"]
sgp.DLCSmoothInterpSelection.insert(
    [
        {
            **si_key,
            "bodypart": bodypart,
            "dlc_si_params_name": si_params_name,
        }
        for bodypart in bodyparts
    ],
    skip_duplicates=True,
)

# And verify the entry:

sgp.DLCSmoothInterpSelection() & si_key

# Now, we populate `DLCSmoothInterp`, which will perform smoothing and
# interpolation on all of the bodyparts specified.

sgp.DLCSmoothInterp().populate(si_key)

# And let's visualize the resulting position data using a scatter plot

(
    sgp.DLCSmoothInterp() & {**si_key, "bodypart": bodyparts[0]}
).fetch1_dataframe().plot.scatter(x="x", y="y", s=1, figsize=(5, 5))

# #### [DLCSmoothInterpCohort](#TableOfContents) <a id='DLCSmoothInterpCohort1'></a>

# After smoothing/interpolation, we need to select bodyparts from which we want to
# derive a centroid and orientation, which is performed by the
# `DLCSmoothInterpCohort` table.

# First, let's make a key that represents the 'cohort', using
# `dlc_si_cohort_selection_name`. We'll need a bodypart dictionary using bodypart
# keys and smoothing/interpolation parameters used as value.

cohort_key = si_key.copy()
if "bodypart" in cohort_key:
    del cohort_key["bodypart"]
if "dlc_si_params_name" in cohort_key:
    del cohort_key["dlc_si_params_name"]
cohort_key["dlc_si_cohort_selection_name"] = "green_red_led"
cohort_key["bodyparts_params_dict"] = {
    "greenLED": si_params_name,
    "redLED_C": si_params_name,
}
print(cohort_key)

# We'll insert the cohort into `DLCSmoothInterpCohortSelection` and populate `DLCSmoothInterpCohort`, which collates the separately smoothed and interpolated bodyparts into a single entry.

sgp.DLCSmoothInterpCohortSelection().insert1(cohort_key, skip_duplicates=True)
sgp.DLCSmoothInterpCohort.populate(cohort_key)

# And verify the entry:

sgp.DLCSmoothInterpCohort.BodyPart() & cohort_key

# #### [DLCCentroid](#TableOfContents) <a id='DLCCentroid1'></a>

# With this cohort, we can determine a centroid using another set of parameters.

# Here is the default set
print(sgp.DLCCentroidParams.get_default())
centroid_params_name = "default"

# Here is the syntax to add your own parameters:
#
# ```python
# centroid_params = {
#     "centroid_method": "two_pt_centroid",
#     "points": {
#         "greenLED": "greenLED",
#         "redLED_C": "redLED_C",
#     },
#     "speed_smoothing_std_dev": 0.100,
# }
# centroid_params_name = "your_unique_param_name"
# sgp.DLCCentroidParams.insert1(
#     {
#         "dlc_centroid_params_name": centroid_params_name,
#         "params": centroid_params,
#     },
#     skip_duplicates=True,
# )
# ```

# We'll make a key to insert into `DLCCentroidSelection`.

centroid_key = cohort_key.copy()
fields = list(sgp.DLCCentroidSelection.fetch().dtype.fields.keys())
centroid_key = {key: val for key, val in centroid_key.items() if key in fields}
centroid_key["dlc_centroid_params_name"] = centroid_params_name
pprint(centroid_key)

# After inserting into the selection table, we can populate `DLCCentroid`

sgp.DLCCentroidSelection.insert1(centroid_key, skip_duplicates=True)
sgp.DLCCentroid.populate(centroid_key)

# Here we can visualize the resulting centroid position

(sgp.DLCCentroid() & centroid_key).fetch1_dataframe().plot.scatter(
    x="position_x",
    y="position_y",
    c="speed",
    colormap="viridis",
    alpha=0.5,
    s=0.5,
    figsize=(10, 10),
)

# #### [DLCOrientation](#TableOfContents) <a id='DLCOrientation1'></a>

# We'll go through a similar process for orientation.

pprint(sgp.DLCOrientationParams.get_default())
dlc_orientation_params_name = "default"

# We'll prune the `cohort_key` we used above and add our
# `dlc_orientation_params_name` to make it suitable for `DLCOrientationSelection`.

fields = list(sgp.DLCOrientationSelection.fetch().dtype.fields.keys())
orient_key = {key: val for key, val in cohort_key.items() if key in fields}
orient_key["dlc_orientation_params_name"] = dlc_orientation_params_name
print(orient_key)

# We'll insert into `DLCOrientationSelection` and then populate `DLCOrientation`

sgp.DLCOrientationSelection().insert1(orient_key, skip_duplicates=True)
sgp.DLCOrientation().populate(orient_key)

# We can fetch the orientation as a dataframe as quality assurance.

(sgp.DLCOrientation() & orient_key).fetch1_dataframe()

# #### [DLCPos](#TableOfContents) <a id='DLCPos1'></a>

# After processing the position data, we have to do a few table manipulations to standardize various outputs.
#
# To summarize, we brought in a pretrained DLC project, used that model to run pose estimation on a new behavioral video, smoothed and interpolated the result, formed a cohort of bodyparts, and determined the centroid and orientation of this cohort.
#
# Now we'll populate `DLCPos` with our centroid/orientation entries above.

fields = list(sgp.DLCPos.fetch().dtype.fields.keys())
dlc_key = {key: val for key, val in centroid_key.items() if key in fields}
dlc_key["dlc_si_cohort_centroid"] = centroid_key["dlc_si_cohort_selection_name"]
dlc_key["dlc_si_cohort_orientation"] = orient_key[
    "dlc_si_cohort_selection_name"
]
dlc_key["dlc_orientation_params_name"] = orient_key[
    "dlc_orientation_params_name"
]
pprint(dlc_key)

# Now we can insert into `DLCPosSelection` and populate `DLCPos` with our `dlc_key`

sgp.DLCPosSelection().insert1(dlc_key, skip_duplicates=True)
sgp.DLCPos().populate(dlc_key)

# Fetched as a dataframe, we expect the following 8 columns:
#
# - time
# - video_frame_ind
# - position_x
# - position_y
# - orientation
# - velocity_x
# - velocity_y
# - speed

(sgp.DLCPos() & dlc_key).fetch1_dataframe()

# We can also fetch the `pose_eval_result`, which contains the percentage of
# frames that each bodypart was below the likelihood threshold of 0.95.

(sgp.DLCPos() & dlc_key).fetch1("pose_eval_result")

# #### [DLCPosVideo](#TableOfContents) <a id='DLCPosVideo1'></a>

# We can create a video with the centroid and orientation overlaid on the original
# video. This will also plot the likelihood of each bodypart used in the cohort.
# This is optional, but a good quality assurance step.

sgp.DLCPosVideoParams.insert_default()

params = {
    "percent_frames": 0.05,
    "incl_likelihood": True,
}
sgp.DLCPosVideoParams.insert1(
    {"dlc_pos_video_params_name": "five_percent", "params": params},
    skip_duplicates=True,
)

sgp.DLCPosVideoSelection.insert1(
    {**dlc_key, "dlc_pos_video_params_name": "five_percent"},
    skip_duplicates=True,
)

sgp.DLCPosVideo().populate(dlc_key)

# #### [PositionOutput](#TableOfContents) <a id='PositionOutput1'></a>

# `PositionOutput` is the final table of the pipeline and is automatically
# populated when we populate `DLCPosV1`

sgp.PositionOutput() & dlc_key

# `PositionOutput` also has a part table, similar to the `DLCModelSource` table above. Let's check that out as well.

PositionOutput.DLCPosV1() & dlc_key

(PositionOutput.DLCPosV1() & dlc_key).fetch1_dataframe()

# #### [PositionVideo](#TableOfContents)<a id='PositionVideo1'></a>

# We can use the `PositionVideo` table to create a video that overlays just the
# centroid and orientation on the video. This table uses the parameter `plot` to
# determine whether to plot the entry deriving from the DLC arm or from the Trodes
# arm of the position pipeline. This parameter also accepts 'all', which will plot
# both (if they exist) in order to compare results.

sgp.PositionVideoSelection().insert1(
    {
        "nwb_file_name": "J1620210604_.nwb",
        "interval_list_name": "pos 13 valid times",
        "trodes_position_id": 0,
        "dlc_position_id": 1,
        "plot": "DLC",
        "output_dir": "/home/dgramling/Src/",
    }
)

sgp.PositionVideo.populate({"plot": "DLC"})

# CONGRATULATIONS!! Please treat yourself to a nice tea break :-)

# ### [Return To Table of Contents](#TableOfContents)<br>
