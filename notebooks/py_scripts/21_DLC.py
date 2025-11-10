# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Position- DeepLabCut from Scratch
#

# ### Overview
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
#
# - creating a DLC project
# - extracting and labeling frames
# - training your model
# - executing pose estimation on a novel behavioral video
# - processing the pose estimation output to extract a centroid and orientation
# - inserting the resulting information into the `PositionOutput` table
#
# **Note 2: Make sure you are running this within the spyglass-dlc Conda environment (instructions for install are in the environment_dlc.yml)**
#

# Here is a schematic showing the tables used in this pipeline.
#
# ![dlc_scratch.png|2000x900](./../notebook-images/dlc_scratch.png)
#

# ### Table of Contents<a id='TableOfContents'></a>
#
# [`DLCProject`](#DLCProject1)<br>
# [`DLCModelTraining`](#DLCModelTraining1)<br>
# [`DLCModel`](#DLCModel1)<br>
# [`DLCPoseEstimation`](#DLCPoseEstimation1)<br>
# [`DLCSmoothInterp`](#DLCSmoothInterp1)<br>
# [`DLCCentroid`](#DLCCentroid1)<br>
# [`DLCOrientation`](#DLCOrientation1)<br>
# [`DLCPosV1`](#DLCPosV1-1)<br>
# [`DLCPosVideo`](#DLCPosVideo1)<br>
# [`PositionOutput`](#PositionOutput1)<br>
#

# **You can click on any header to return to the Table of Contents**
#

# ### Imports
#

# %load_ext autoreload
# %autoreload 2

# +
import os
import datajoint as dj

import spyglass.common as sgc
import spyglass.position.v1 as sgp

import numpy as np
import pandas as pd
import pynwb
from spyglass.position import PositionOutput

# change to the upper level folder to detect dj_local_conf.json
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
dj.config.load("dj_local_conf.json")  # load config for database connection info

# ignore datajoint+jupyter async warnings
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
# -

# #### [DLCProject](#TableOfContents) <a id="DLCProject1"></a>
#

# <div class="alert alert-block alert-info">
#     <b>Notes:</b><ul>
#     <li>
#         The cells within this <code>DLCProject</code> step need to be performed
#         in a local Jupyter notebook to allow for use of the frame labeling GUI.
#     </li>
#     <li>
#         Please do not add to the <code>BodyPart</code> table in the production
#         database unless necessary.
#     </li>
#     </ul>
# </div>
#

# ### Body Parts
#

# We'll begin by looking at the `BodyPart` table, which stores standard names of body parts used in DLC models throughout the lab with a concise description.
#

sgp.BodyPart()

# If the bodyparts you plan to use in your model are not yet in the table, here is code to add bodyparts:
#
# ```python
# sgp.BodyPart.insert(
#     [
#         {"bodypart": "bp_1", "bodypart_description": "concise descrip"},
#         {"bodypart": "bp_2", "bodypart_description": "concise descrip"},
#     ],
#     skip_duplicates=True,
# )
# ```
#

# ### Define videos and camera name (optional) for training set
#

# To train a model, we'll need to extract frames, which we can label as training data. We can construct a list of videos from which we'll extract frames.
#
# The list can either contain dictionaries identifying behavioral videos for NWB files that have already been added to Spyglass, or absolute file paths to the videos (in .h264 format) you want to use.
#
# For this tutorial, we'll use two videos for which we already have frames labeled.
#

# Defining camera name is optional: it should be done in cases where there are multiple cameras streaming per epoch, but not necessary otherwise. <br>
# example:
# `camera_name = "HomeBox_camera"
#    `
#

# _NOTE:_ The official release of Spyglass does not yet support multicamera
# projects. You can monitor progress on the effort to add this feature by checking
# [this PR](https://github.com/LorenFrankLab/spyglass/pull/684) or use
# [this experimental branch](https://github.com/dpeg22/spyglass/tree/add-multi-camera),
# which takes the keys nwb_file_name and epoch, and camera_name in the video_list variable.
#

video_list = [
    {"nwb_file_name": "J1620210529_.nwb", "epoch": 2},
    {"nwb_file_name": "peanut20201103_.nwb", "epoch": 4},
]

# ### Path variables
#
# The position pipeline also keeps track of paths for project, video, and output.
# Just like we saw in [Setup](./00_Setup.ipynb), you can manage these either with
# environmental variables...
#
# ```bash
# export DLC_PROJECT_DIR="/nimbus/deeplabcut/projects"
# export DLC_VIDEO_DIR="/nimbus/deeplabcut/video"
# export DLC_OUTPUT_DIR="/nimbus/deeplabcut/output"
# ```
#
# <!-- NOTE: HDF5_USE_FILE_LOCKING now automatically set to 'FALSE' -->
#
# Or these can be set in your datajoint config:
#
# ```json
# {
#   "custom": {
#     "dlc_dirs": {
#       "base": "/nimbus/deeplabcut/",
#       "project": "/nimbus/deeplabcut/projects",
#       "video": "/nimbus/deeplabcut/video",
#       "output": "/nimbus/deeplabcut/output"
#     }
#   }
# }
# ```
#
# _NOTE:_ If only `base` is specified as shown above, spyglass will assume the
# relative directories shown.
#
# You can check the result of this setup process with...
#

# +
from spyglass.settings import config

config
# -

# Before creating our project, we need to define a few variables.
#
# - A team name, as shown in `LabTeam` for setting permissions. Here, we'll
#   use "LorenLab".
# - A `project_name`, as a unique identifier for this DLC project. Here, we'll use
#   **"tutorial_scratch_yourinitials"**
# - `bodyparts` is a list of body parts for which we want to extract position.
#   The pre-labeled frames we're using include the bodyparts listed below.
# - Number of frames to extract/label as `frames_per_video`. Note that the DLC creators recommend having 200 frames as the minimum total number for each project.
#

team_name = sgc.LabTeam.fetch("team_name")[0]  # If on lab DB, "LorenLab"
project_name = "tutorial_scratch_DG"
frames_per_video = 100
bodyparts = ["redLED_C", "greenLED", "redLED_L", "redLED_R", "tailBase"]
project_key = sgp.DLCProject.insert_new_project(
    project_name=project_name,
    bodyparts=bodyparts,
    lab_team=team_name,
    frames_per_video=frames_per_video,
    video_list=video_list,
    skip_duplicates=True,
)

# Now that we've initialized our project we'll need to extract frames which we will then label.
#

# comment this line out after you finish frame extraction for each project
sgp.DLCProject().run_extract_frames(project_key)

# This is the line used to label the frames you extracted, if you wish to use the DLC GUI on the computer you are currently using.
#
# ```#comment this line out after frames are labeled for your project
# sgp.DLCProject().run_label_frames(project_key)
# ```
#

# Otherwise, it is best/easiest practice to label the frames on your local computer (like a MacBook) that can run DeepLabCut's GUI well. Instructions: <br>
#
# 1. Install DLC on your local (preferably into a 'Src' folder): https://deeplabcut.github.io/DeepLabCut/docs/installation.html
# 2. Upload frames extracted and saved in nimbus (should be `/nimbus/deeplabcut/<YOUR_PROJECT_NAME>/labeled-data`) AND the project's associated config file (should be `/nimbus/deeplabcut/<YOUR_PROJECT_NAME>/config.yaml`) to Box (we get free with UCSF)
# 3. Download labeled-data and config files on your local from Box
# 4. Create a 'projects' folder where you installed DeepLabCut; create a new folder with your complete project name there; save the downloaded files there.
# 5. Edit the config.yaml file: line 9 defining `project_path` needs to be the file path where it is saved on your local (ex: `/Users/lorenlab/Src/DeepLabCut/projects/tutorial_sratch_DG-LorenLab-2023-08-16`)
# 6. Open the DLC GUI through terminal
#    <br>(ex: `conda activate miniconda/envs/DEEPLABCUT_M1`
#    <br>`pythonw -m deeplabcut`)
# 7. Load an existing project; choose the config.yaml file
# 8. Label frames; labeling tutorial: https://www.youtube.com/watch?v=hsA9IB5r73E.
# 9. Once all frames are labeled, you should re-upload labeled-data folder back to Box and overwrite it in the original nimbus location so that your completed frames are ready to be used in the model.
#

# Now we can check the `DLCProject.File` part table and see all of our training files and videos there!
#

sgp.DLCProject.File & project_key

# <div class="alert alert-block alert-warning">
#     This step and beyond should be run on a GPU-enabled machine.</b>
# </div>
#

# #### [DLCModelTraining](#ToC)<a id='DLCModelTraining1'></a>
#
# Please make sure you're running this notebook on a GPU-enabled machine.
#
# Now that we've imported existing frames, we can get ready to train our model.
#
# First, we'll need to define a set of parameters for `DLCModelTrainingParams`, which will get used by DeepLabCut during training. Let's start with `gputouse`,
# which determines which GPU core to use.
#
# The cell below determines which core has space and set the `gputouse` variable
# accordingly.
#

sgp.dlc_utils.get_gpu_memory()

# Set GPU core:
#

gputouse = 1  # 1-9

# Now we'll define the rest of our parameters and insert the entry.
#
# To see all possible parameters, try:
#
# ```python
# sgp.DLCModelTrainingParams.get_accepted_params()
# ```
#

training_params_name = "tutorial"
sgp.DLCModelTrainingParams.insert_new_params(
    paramset_name=training_params_name,
    params={
        "trainingsetindex": 0,
        "shuffle": 1,
        "gputouse": gputouse,
        "net_type": "resnet_50",
        "augmenter_type": "imgaug",
    },
    skip_duplicates=True,
)

# Next we'll modify the `project_key` from above to include the necessary entries for `DLCModelTraining`
#

# project_key['project_path'] = os.path.dirname(project_key['config_path'])
if "config_path" in project_key:
    del project_key["config_path"]

# We can insert an entry into `DLCModelTrainingSelection` and populate `DLCModelTraining`.
#
# _Note:_ You can stop training at any point using `I + I` or interrupt the Kernel.
#
# The maximum total number of training iterations is 1030000; you can end training before this amount if the loss rate (lr) and total loss plateau and are very close to 0.
#

sgp.DLCModelTrainingSelection.heading

sgp.DLCModelTrainingSelection().insert1(
    {
        **project_key,
        "dlc_training_params_name": training_params_name,
        "training_id": 0,
        "model_prefix": "",
    }
)
model_training_key = (
    sgp.DLCModelTrainingSelection
    & {
        **project_key,
        "dlc_training_params_name": training_params_name,
    }
).fetch1("KEY")
sgp.DLCModelTraining.populate(model_training_key)

# Here we'll make sure that the entry made it into the table properly!
#

sgp.DLCModelTraining() & model_training_key

# Populating `DLCModelTraining` automatically inserts the entry into
# `DLCModelSource`, which is used to select between models trained using Spyglass
# vs. other tools.
#

sgp.DLCModelSource() & model_training_key

# The `source` field will only accept _"FromImport"_ or _"FromUpstream"_ as entries. Let's checkout the `FromUpstream` part table attached to `DLCModelSource` below.
#

sgp.DLCModelSource.FromUpstream() & model_training_key

# #### [DLCModel](#TableOfContents) <a id='DLCModel1'></a>
#
# Next we'll populate the `DLCModel` table, which holds all the relevant
# information for all trained models.
#
# First, we'll need to determine a set of parameters for our model to select the
# correct model file. Here is the default:
#

sgp.DLCModelParams.get_default()

# Here is the syntax to add your own parameter set:
#
# ```python
# dlc_model_params_name = "make_this_yours"
# params = {
#     "params": {},
#     "shuffle": 1,
#     "trainingsetindex": 0,
#     "model_prefix": "",
# }
# sgp.DLCModelParams.insert1(
#     {"dlc_model_params_name": dlc_model_params_name, "params": params},
#     skip_duplicates=True,
# )
# ```
#

# We can insert sets of parameters into `DLCModelSelection` and populate
# `DLCModel`.
#

temp_model_key = (sgp.DLCModelSource & model_training_key).fetch1("KEY")

# comment these lines out after successfully inserting, for each project
sgp.DLCModelSelection().insert1(
    {**temp_model_key, "dlc_model_params_name": "default"}, skip_duplicates=True
)

model_key = (sgp.DLCModelSelection & temp_model_key).fetch1("KEY")
sgp.DLCModel.populate(model_key)

# Again, let's make sure that everything looks correct in `DLCModel`.
#

sgp.DLCModel() & model_key

# #### [DLCPoseEstimation](#TableOfContents) <a id='DLCPoseEstimation1'></a>
#
# Alright, now that we've trained model and populated the `DLCModel` table, we're ready to set-up Pose Estimation on a behavioral video of your choice.<br><br>For this tutorial, you can choose to use an epoch of your choice, we can also use the one specified below. If you'd like to use your own video, just specify the `nwb_file_name` and `epoch` number and make sure it's in the `VideoFile` table!
#

nwb_file_name = "J1620210604_.nwb"
sgc.VideoFile() & {"nwb_file_name": nwb_file_name}

epoch = 14  # change based on VideoFile entry
video_file_num = 0  # change based on VideoFile entry

# Using `insert_estimation_task` will convert out video to be in .mp4 format (DLC
# struggles with .h264) and determine the directory in which we'll store the pose
# estimation results.
#
# - `task_mode` (trigger or load) determines whether or not populating
#   `DLCPoseEstimation` triggers a new pose estimation, or loads an existing.
# - `gputouse` was already set during training. It may be a good idea to make sure
#   that core is still free before moving forward.
#

# The `DLCPoseEstimationSelection` insertion step will convert your .h264 video to an .mp4 first and save it in `/nimbus/deeplabcut/video`. If this video already exists here, the insertion will never complete.
#
# We first delete any .mp4 that exists for this video from the nimbus folder.
# Remove the `#` to run this line. The `!` tells the notebook that this is
# a system command to be run with a shell script instead of python.
# Be sure to change the string based on date and rat with which you are training the model
#

# +
key = {
    "nwb_file_name": nwb_file_name,
    "epoch": epoch,
    "video_file_num": video_file_num,
    **model_key,
}

dlc_pose_estimation = sgp.DLCPoseEstimationSelection()  # Create an instance

pose_estimation_key = dlc_pose_estimation.insert_estimation_task(
    key,
    task_mode="trigger",  # trigger or load
    params={"gputouse": gputouse, "videotype": "mp4"},
)
# -

# If the above insertion step fails in either trigger or load mode for an epoch, run the following lines:
#
# ```
# (pose_estimation_key = dlc_pose_estimation.insert_estimation_task(
#     key,
#     task_mode="trigger", #trigger or load
#     params={"gputouse": gputouse, "videotype": "mp4"},
#     )).delete()
# ```
#

# And now we populate `DLCPoseEstimation`! This might take some time for full datasets.
#

sgp.DLCPoseEstimation().populate(pose_estimation_key)

# Let's visualize the output from Pose Estimation
#

(sgp.DLCPoseEstimation() & pose_estimation_key).fetch_dataframe()

# #### [DLCSmoothInterp](#TableOfContents) <a id='DLCSmoothInterp1'></a>
#

# Now that we've completed pose estimation, it's time to identify NaNs and optionally interpolate over low likelihood periods and smooth the resulting positions.<br>First we need to define some parameters for smoothing and interpolation. We can see the default parameter set below.<br>**Note**: it is recommended to use the `just_nan` parameters here and save interpolation and smoothing for the centroid step as this provides for a better end result.
#

# The default parameter set to interpolate and smooth over each LED individually
print(sgp.DLCSmoothInterpParams.get_default())

# The just_nan parameter set that identifies NaN indices and leaves smoothing and interpolation to the centroid step
print(sgp.DLCSmoothInterpParams.get_nan_params())
si_params_name = "just_nan"  # could also use "default"

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
#

# We'll create a dictionary with the correct set of keys for the `DLCSmoothInterpSelection` table
#

si_key = pose_estimation_key.copy()
fields = list(sgp.DLCSmoothInterpSelection.fetch().dtype.fields.keys())
si_key = {key: val for key, val in si_key.items() if key in fields}
si_key

# We can insert all of the bodyparts we want to process into `DLCSmoothInterpSelection`<br>
# First lets visualize the bodyparts we have available to us.<br>
#

print((sgp.DLCPoseEstimation.BodyPart & pose_estimation_key).fetch("bodypart"))

# We can use `insert1` to insert a single bodypart, but would suggest using `insert` to insert a list of keys with different bodyparts.
#

# To insert a single bodypart, one would do the following:
#
# ```python
# sgp.DLCSmoothInterpSelection.insert1(
#     {
#         **si_key,
#         'bodypart': 'greenLED',
#         'dlc_si_params_name': si_params_name,
#     },
#     skip_duplicates=True)
# ```
#

# We'll see a list of bodyparts and then insert them into `DLCSmoothInterpSelection`.
#

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
#

sgp.DLCSmoothInterpSelection() & si_key

# Now, we populate `DLCSmoothInterp`, which will perform smoothing and
# interpolation on all of the bodyparts specified.
#

sgp.DLCSmoothInterp().populate(si_key)

# And let's visualize the resulting position data using a scatter plot
#

(
    sgp.DLCSmoothInterp() & {**si_key, "bodypart": bodyparts[0]}
).fetch1_dataframe().plot.scatter(x="x", y="y", s=1, figsize=(5, 5))

# #### [DLCSmoothInterpCohort](#TableOfContents) <a id='DLCSmoothInterpCohort1'></a>
#

# After smoothing/interpolation, we need to select bodyparts from which we want to
# derive a centroid and orientation, which is performed by the
# `DLCSmoothInterpCohort` table.
#

# First, let's make a key that represents the 'cohort', using
# `dlc_si_cohort_selection_name`. We'll need a bodypart dictionary using bodypart
# keys and smoothing/interpolation parameters used as value.
#

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
#

sgp.DLCSmoothInterpCohortSelection().insert1(cohort_key, skip_duplicates=True)
sgp.DLCSmoothInterpCohort.populate(cohort_key)

# And verify the entry:
#

sgp.DLCSmoothInterpCohort.BodyPart() & cohort_key

# #### [DLCCentroid](#TableOfContents) <a id='DLCCentroid1'></a>
#

# With this cohort, we can determine a centroid using another set of parameters.
#

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
#

# We'll make a key to insert into `DLCCentroidSelection`.
#

centroid_key = cohort_key.copy()
fields = list(sgp.DLCCentroidSelection.fetch().dtype.fields.keys())
centroid_key = {key: val for key, val in centroid_key.items() if key in fields}
centroid_key["dlc_centroid_params_name"] = centroid_params_name
print(centroid_key)

# After inserting into the selection table, we can populate `DLCCentroid`
#

sgp.DLCCentroidSelection.insert1(centroid_key, skip_duplicates=True)
sgp.DLCCentroid.populate(centroid_key)

# Here we can visualize the resulting centroid position
#

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
#

# We'll now go through a similar process to identify the orientation.
#

print(sgp.DLCOrientationParams.get_default())
dlc_orientation_params_name = "default"

# We'll prune the `cohort_key` we used above and add our `dlc_orientation_params_name` to make it suitable for `DLCOrientationSelection`.
#

fields = list(sgp.DLCOrientationSelection.fetch().dtype.fields.keys())
orient_key = {key: val for key, val in cohort_key.items() if key in fields}
orient_key["dlc_orientation_params_name"] = dlc_orientation_params_name
print(orient_key)

# We'll insert into `DLCOrientationSelection` and populate `DLCOrientation`
#

sgp.DLCOrientationSelection().insert1(orient_key, skip_duplicates=True)
sgp.DLCOrientation().populate(orient_key)

# We can fetch the orientation as a dataframe as quality assurance.
#

(sgp.DLCOrientation() & orient_key).fetch1_dataframe()

# #### [DLCPosV1](#TableOfContents) <a id='DLCPosV1-1'></a>
#

# After processing the position data, we have to do a few table manipulations to standardize various outputs.
#
# To summarize, we brought in a pretrained DLC project, used that model to run pose estimation on a new behavioral video, smoothed and interpolated the result, formed a cohort of bodyparts, and determined the centroid and orientation of this cohort.
#
# Now we'll populate `DLCPos` with our centroid/orientation entries above.
#

fields = list(sgp.DLCPosV1.fetch().dtype.fields.keys())
dlc_key = {key: val for key, val in centroid_key.items() if key in fields}
dlc_key["dlc_si_cohort_centroid"] = centroid_key["dlc_si_cohort_selection_name"]
dlc_key["dlc_si_cohort_orientation"] = orient_key[
    "dlc_si_cohort_selection_name"
]
dlc_key["dlc_orientation_params_name"] = orient_key[
    "dlc_orientation_params_name"
]
print(dlc_key)

# Now we can insert into `DLCPosSelection` and populate `DLCPos` with our `dlc_key`
#

sgp.DLCPosSelection().insert1(dlc_key, skip_duplicates=True)
sgp.DLCPosV1().populate(dlc_key)

# We can also make sure that all of our data made it through by fetching the dataframe attached to this entry.<br>We should expect 8 columns:
#
# > time<br>video_frame_ind<br>position_x<br>position_y<br>orientation<br>velocity_x<br>velocity_y<br>speed
#

(sgp.DLCPosV1() & dlc_key).fetch1_dataframe()

# And even more, we can fetch the `pose_eval_result` that is calculated during this step. This field contains the percentage of frames that each bodypart was below the likelihood threshold of 0.95 as a means of assessing the quality of the pose estimation.
#

(sgp.DLCPosV1() & dlc_key).fetch1("pose_eval_result")

# #### [DLCPosVideo](#TableOfContents) <a id='DLCPosVideo1'></a>
#

# We can create a video with the centroid and orientation overlaid on the original
# video. This will also plot the likelihood of each bodypart used in the cohort.
# This is optional, but a good quality assurance step.
#

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

# <details><summary>On editing parameters</summary>
#
# The presence of existing parameters in many tables makes it easy to tweak them
# for your needs. You can fetch, edit, and re-insert new params - but the process
# will look a little different if the table has a `=BLOB=` field.
#
# (These example assumes only one primary key. If multiple, `{'primary_key': 'x'}`
# and `['primary_key']` will need to be adjusted accordingly.)
#
# No blob means that all parameters are fields in the table.
#
# ```python
# existing_params = (MyParamsTable & {'primary_key':'x'}).fetch1()
# new_params = {**existing_params, 'primary_key': 'y', 'my_variable': 'a', 'other_variable':'b'}
# MyParamsTable.insert1(new_params)
# ```
#
# A blob means that the params are stored as an embedded dictionary. We'll assume
# this column is called `params`
#
# ```python
# existing_params = (MyParamsTable & {'primary_key':'x'}).fetch1()
# new_params = {**existing_params, 'primary_key': 'y'}
# print(existing_params['params']) # check existing values
# new_params['params'] = {**existing_params['params'], 'my_variable': 'a', 'other_variable':'b'}
# ```
#
# </details>

# #### [PositionOutput](#TableOfContents) <a id='PositionOutput1'></a>
#

# `PositionOutput` is the final table of the pipeline and is automatically
# populated when we populate `DLCPosV1`
#

sgp.PositionOutput.merge_get_part(dlc_key)

# `PositionOutput` also has a part table, similar to the `DLCModelSource` table above. Let's check that out as well.
#

PositionOutput.DLCPosV1() & dlc_key

(PositionOutput.DLCPosV1() & dlc_key).fetch1_dataframe()

# ### _CONGRATULATIONS!!_
#
# Please treat yourself to a nice tea break :-)
#

# ### [Return To Table of Contents](#TableOfContents)<br>
#
