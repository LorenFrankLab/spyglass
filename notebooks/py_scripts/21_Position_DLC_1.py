# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python [conda env:spyglass-position] *
#     language: python
#     name: conda-env-spyglass-position-py
# ---

# # Position - DeepLabCut from Scratch
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
# - creating a DLC project
# - extracting and labeling frames
# - training your model
#
# If you have a pre-trained project, you can either skip to the
# [next tutorial](./22_Position_DLC_2.ipynb) to load it into the database, or skip
# to the [following tutorial](./23_Position_DLC_3.ipynb) to start pose estimation
# with a model that is already inserted.

# Here is a schematic showing the tables used in this pipeline.
#
# ![dlc_scratch.png|2000x900](./../notebook-images/dlc_scratch.png)

# ### Table of Contents<a id='TableOfContents'></a>
#

# - [Imports](#imports)
# - [`DLCProject`](#DLCProject1)
# - [`DLCModelTraining`](#DLCModelTraining1)
# - [`DLCModel`](#DLCModel1)
#
# __You can click on any header to return to the Table of Contents__

# ### Imports
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

# #### [DLCProject](#TableOfContents) <a id="DLCProject1"></a>
#

# <div class="alert alert-block alert-info">
#     <b>Notes:</b><ul>
#     <li>
#         The cells within this <code>DLCProject</code> step need to be performed
#         in a local Jupyter notebook to allow for use of the frame labeling GUI
#     </li>
#     <li>
#         Please do not add to the <code>BodyPart</code> table in the production
#         database unless necessary.
#     </li>
#     </ul>
# </div>

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

# To train a model, we'll need to extract frames, which we can label as training data. We can construct a list of videos from which we'll extract frames.
#
# The list can either contain dictionaries identifying behavioral videos for NWB files that have already been added to Spyglass, or absolute file paths to the videos you want to use.
#
# For this tutorial, we'll use two videos for which we already have frames labeled.

video_list = [
    {"nwb_file_name": "J1620210529_.nwb", "epoch": 2},
    {"nwb_file_name": "peanut20201103_.nwb", "epoch": 4},
]

# Before creating our project, we need to define a few variables.
#
# - A team name, as shown in `LabTeam` for setting permissions. Here, we'll
#  use "LorenLab".
# - A `project_name`, as a unique identifier for this DLC project. Here, we'll use
#  __"tutorial_scratch_yourinitials"__
# - `bodyparts` is a list of body parts for which we want to extract position.
#   The pre-labeled frames we're using include the bodyparts listed below.
# - Number of frames to extract/label as `frames_per_video`. A true project might
#   use 200, but we'll use 100 for efficiency.

team_name = "LorenLab"
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

# After initializing our project, we would typically extract and label frames. This has already been done for this tutorial data, using the following commands to pull up the DLC GUI:
#
# ```python
# sgp.DLCProject().run_extract_frames(project_key)
# sgp.DLCProject().run_label_frames(project_key)
# ```

# In order to use pre-labeled frames, you'll need to change the values in the
# labeled-data files. You can do that using the `import_labeled_frames` method,
# which expects:
#
# - `project_key` from your new project.
# - The absolute path to the project directory from which we'll import labeled
#   frames.
# - The filenames, without extension, of the videos from which we want frames.

sgp.DLCProject.import_labeled_frames(
    project_key.copy(),
    import_project_path="/nimbus/deeplabcut/projects/tutorial_model-LorenLab-2022-07-15/",
    video_filenames=["20201103_peanut_04_r2", "20210529_J16_02_r1"],
    skip_duplicates=True,
)

# <div class="alert alert-block alert-warning">
#     This step and beyond should be run on a GPU-enabled machine.
# </div>

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

sgp.dlc_utils.get_gpu_memory()

# Set GPU core:

gputouse = 1  ## 1-9

# Now we'll define the rest of our parameters and insert the entry.
#
# To see all possible parameters, try:
#
# ```python
# sgp.DLCModelTrainingParams.get_accepted_params()
# ```

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

# Next we'll modify the `project_key` to include the entries for
# `DLCModelTraining`

# project_key['project_path'] = os.path.dirname(project_key['config_path'])
if "config_path" in project_key:
    del project_key["config_path"]

# We can insert an entry into `DLCModelTrainingSelection` and populate `DLCModelTraining`.
#
# _Note:_ You can stop training at any point using `I + I` or interrupt the Kernel

sgp.DLCModelTrainingSelection.heading

# + jupyter={"outputs_hidden": true}
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
# -

# Here we'll make sure that the entry made it into the table properly!

sgp.DLCModelTraining() & model_training_key

# Populating `DLCModelTraining` automatically inserts the entry into
# `DLCModelSource`, which is used to select between models trained using Spyglass
# vs. other tools.

sgp.DLCModelSource() & model_training_key

# The `source` field will only accept _"FromImport"_ or _"FromUpstream"_ as entries. Let's checkout the `FromUpstream` part table attached to `DLCModelSource` below.

sgp.DLCModelSource.FromUpstream() & model_training_key

# #### [DLCModel](#TableOfContents) <a id='DLCModel1'></a>
#
# Next we'll populate the `DLCModel` table, which holds all the relevant
# information for all trained models.
#
# First, we'll need to determine a set of parameters for our model to select the
# correct model file. Here is the default:

pprint(sgp.DLCModelParams.get_default())

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

# We can insert sets of parameters into `DLCModelSelection` and populate
# `DLCModel`.

temp_model_key = (sgp.DLCModelSource & model_training_key).fetch1("KEY")
sgp.DLCModelSelection().insert1(
    {**temp_model_key, "dlc_model_params_name": "default"}, skip_duplicates=True
)
model_key = (sgp.DLCModelSelection & temp_model_key).fetch1("KEY")
sgp.DLCModel.populate(model_key)

# Again, let's make sure that everything looks correct in `DLCModel`.

sgp.DLCModel() & model_key

# ### Next Steps
#
# With our trained model in place, we're ready to move on to pose estimation
# (notebook coming soon!).
#
# <!-- [pose estimation](./23_Position_DLC_3.ipynb). -->

# ### [Return To Table of Contents](#TableOfContents)<br>
