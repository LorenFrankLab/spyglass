# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: spy
#     language: python
#     name: python3
# ---

# # Position - DeepLabCut PreTrained
#

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
# This is a tutorial will cover how to extract position given a pre-trained DeepLabCut (DLC) model. It will walk through adding your DLC model to Spyglass.
#
# If you already have a model in the database, skip to the
# [next tutorial](./23_Position_DLC_3.ipynb).

# ## Imports
#

# +
import os
import datajoint as dj
from pprint import pprint

# change to the upper level folder to detect dj_local_conf.json
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
dj.config.load("dj_local_conf.json")  # load config for database connection info

from spyglass.settings import load_config

load_config(base_dir="/home/cb/wrk/zOther/data/")

import spyglass.common as sgc
import spyglass.position.v1 as sgp
from spyglass.position import PositionOutput

# ignore datajoint+jupyter async warnings
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
# -

# #### Here is a schematic showing the tables used in this notebook.<br>
# ![dlc_existing.png|2000x900](./../notebook-images/dlc_existing.png)

# ## Table of Contents<a id='ToC'></a>
#
# - [`DLCProject`](#DLCProject)
# - [`DLCModel`](#DLCModel)
# <!--
# - [`DLCPoseEstimation`](#DLCPoseEstimation)
# - [`DLCSmoothInterp`](#DLCSmoothInterp)
# - [`DLCCentroid`](#DLCCentroid)
# - [`DLCOrientation`](#DLCOrientation)
# - [`DLCPos`](#DLCPos)
# - [`DLCPosVideo`](#DLCPosVideo)
# - [`PositionOutput`](#PositionOutput) -->
#
# You can click on any header to return to the Table of Contents

# ## [DLCProject](#ToC) <a id='DLCProject'></a>

# We'll look at the BodyPart table, which stores standard names of body parts used within DLC models.

# <div class="alert alert-block alert-info">
#     <b>Notes:</b><ul>
#     <li>
#         Please do not add to the <code>BodyPart</code> table in the production
#         database unless necessary.
#     </li>
#     </ul>
# </div>

sgp.BodyPart()

# We can `insert_existing_project` into the `DLCProject` table using:
#
# - `project_name`: a short, unique, descriptive project name to reference
#   throughout the pipeline
# - `lab_team`: team name from `LabTeam`
# - `config_path`: string path to a DLC `config.yaml`
# - `bodyparts`: optional list of bodyparts used in the project
# - `frames_per_video`: optional, number of frames to extract for training from
#   each video

project_name = "tutorial_DG"
lab_team = "LorenLab"
project_key = sgp.DLCProject.insert_existing_project(
    project_name=project_name,
    lab_team=lab_team,
    config_path="/nimbus/deeplabcut/projects/tutorial_model-LorenLab-2022-07-15/config.yaml",
    bodyparts=["redLED_C", "greenLED", "redLED_L", "redLED_R", "tailBase"],
    frames_per_video=200,
    skip_duplicates=True,
)

sgp.DLCProject() & {"project_name": project_name}

# ## [DLCModel](#ToC) <a id='DLCModel'></a>

# The `DLCModelInput` table has `dlc_model_name` and `project_name` as primary keys and `project_path` as a secondary key.

sgp.DLCModelInput()

# We can modify the `project_key` to replace `config_path` with `project_path` to
# fit with the fields in `DLCModelInput`

print(f"current project_key:\n{project_key}")
if not "project_path" in project_key:
    project_key["project_path"] = os.path.dirname(project_key["config_path"])
    del project_key["config_path"]
    print(f"updated project_key:\n{project_key}")

# After adding a unique `dlc_model_name` to `project_key`, we insert into
# `DLCModelInput`.

dlc_model_name = "tutorial_model_DG"
sgp.DLCModelInput().insert1(
    {"dlc_model_name": dlc_model_name, **project_key}, skip_duplicates=True
)
sgp.DLCModelInput()

# Inserting into `DLCModelInput` will also populate `DLCModelSource`, which
# records whether or not a model was trained with Spyglass.

sgp.DLCModelSource() & project_key

# The `source` field will only accept _"FromImport"_ or _"FromUpstream"_ as entries. Let's checkout the `FromUpstream` part table attached to `DLCModelSource` below.

sgp.DLCModelSource.FromImport() & project_key

# Next we'll get ready to populate the `DLCModel` table, which holds all the relevant information for both pre-trained models and models trained within Spyglass.<br>First we'll need to determine a set of parameters for our model to select the correct model file.<br>We can visualize a default set below:

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

# We can insert sets of parameters into `DLCModelSelection` and populate
# `DLCModel`.

temp_model_key = (sgp.DLCModelSource.FromImport() & project_key).fetch1("KEY")
sgp.DLCModelSelection().insert1(
    {**temp_model_key, "dlc_model_params_name": "default"}, skip_duplicates=True
)
model_key = (sgp.DLCModelSelection & temp_model_key).fetch1("KEY")
sgp.DLCModel.populate(model_key)

# And of course make sure it populated correctly

sgp.DLCModel() & model_key

# ## Next Steps
#
# With our trained model in place, we're ready to move on to
# pose estimation (notebook coming soon!).
# <!-- [pose estimation](./23_Position_DLC_3.ipynb). -->

# ### [`Return To Table of Contents`](#ToC)<br>
