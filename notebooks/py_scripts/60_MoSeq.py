# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: spyglass2025-moseq-gpu
#     language: python
#     name: python3
# ---

# # MoSeq Pipeline Tutorial
#
# This notebook provides a tutorial on how to use the MoSeq pipeline to analyze behavioral data. The pipeline is a tool for taking keypoint pose estimations and extracting behavioral syllables.
#
# *Note: Moseq is an optional dependency within the spyglass package. For installation
# instructions, see the [setup tutorial](./00_Setup.ipynb)
#
# Here is a schematic showing the tables used in this pipeline. The Basic steps are:
# > **Model Training**
# > - Define training data in `PoseGroup`
# >  - Define Moseq model and training parameters in `MoseqModelParams`
# >  - Combine a set of training parameters and training data in `MoseqModelSelection`
# >  - Populate `MoseqModel` to train
# >
# > **Convert pose data to behavioral syllables**
# >  - Combine a trained model from `MoseqModel` and a pose data from `PositionOutput` in
# `MoseqSyllableSelection`
# >  - Populate  `MoseqSyllable` to apply the trained model to the selected data
#
# ![moseq_outline.png|2000x900](./../notebook-images/moseq_outline.png)
#

#  # Accessing the keypoint (pose) data
#
# In the spyglass architecture, keypoint tracking is performed in the `Position` module,
# and can be accessed through `PositionOutput.fetch_pose_dataframe()`. In this tutorial,
# we are using a set of unpublished data from the Frank lab. For a tutorial on
# running keypoint extraction in spyglass, see [the DLC tutorial notebook](21_DLC.ipynb).
#
# We can access an example set of keypoint pose data here:

# +
# %load_ext autoreload
# %autoreload 2
from spyglass.position.position_merge import PositionOutput

# Key defining the DLC data we are using
pose_key = {
    "nwb_file_name": "SC100020230912_.nwb",
    "epoch": 9,
    "video_file_num": 14,
    "project_name": "sideHomeOfficial",
    "dlc_model_name": "sideHomeOfficial_tutorial_00",
    "dlc_model_params_name": "default",
    "task_mode": "trigger",
}

# Fetch the pose data for demo purposes
merge_key = (PositionOutput.DLCPosV1 & pose_key).fetch1("KEY")
pose_df = (PositionOutput & merge_key).fetch_pose_dataframe()
pose_df
# -

# To train a moseq model, we first need to define the epochs of pose data we will train on
# as well as the bodyparts to use within the model. We define this in the `PoseGroup`
# table below.
#
# Note that training can be run using data from multiple epochs by passing a list of
# merge ids to `create_group`

# +
from spyglass.behavior.v1.core import PoseGroup

# Define the group name and bodyparts to include in the Moseq model
group_name = "tutorial_group"
merge_ids = [(PositionOutput & merge_key).fetch("merge_id")[0]]
bodyparts = [
    "forelimbL",
    "forelimbR",
    "nose",
    "spine1",
    "spine3",
    "spine5",
    "tailBase",
]

# Create the group
PoseGroup().create_group(
    group_name,
    merge_ids,
    bodyparts,
)

# Look at the group in the database
group_key = {"pose_group_name": group_name}
PoseGroup() & group_key
# -

# ## Defining the Moseq Model
#
# Next, we make an entry intpo the `MoseqModelParams` table.  The information in this
# is used to initialize the moseq model and includes hyperparameters for model training
# as well as allows you to begin training from an existing model in the database
# (discussed more below). Relevant parameters can be found in the [Moseq documentation](
# https://keypoint-moseq.readthedocs.io/en/latest/modeling.html#model-fitting)
#
# ** Note: All bodyparts in the `PoseGroup` entry will be used in the model

# +
from spyglass.behavior.v1.moseq import (
    MoseqModel,
    MoseqModelParams,
    MoseqModelSelection,
)

model_params_name = "tutorial_kappa4_mini"
params = {}
# the skeleton list defines pairs of bodyparts that are linked by and edge
params["skeleton"] = [
    ["nose", "spine1"],
    ["spine1", "forelimbL"],
    ["spine1", "forelimbR"],
    ["spine1", "spine3"],
    ["spine3", "spine5"],
    ["spine5", "tailBase"],
]
# kappa affects the distribution of syllable durations, likely needs tuning for each dataset
params["kappa"] = 1e4
# num_ar_iters is the number of iterations of the autoregressive model for warm-up
params["num_ar_iters"] = 50
# num_epochs is the number of epochs to train the model
params["num_epochs"] = 50
# anteror and posterior bodyparts are used to define the orientation of the animal
params["anterior_bodyparts"] = ["nose"]
params["posterior_bodyparts"] = ["tailBase"]

MoseqModelParams().insert1(
    {"model_params_name": model_params_name, "model_params": params},
    skip_duplicates=True,
)

MoseqModelParams() & {"model_params_name": model_params_name}
# -

# To train the model, we link a set of model params with training data in `PoseGroup`
# using the `MoseqModelSelection` table.

# +
MoseqModelSelection().insert1(
    {
        "model_params_name": model_params_name,
        "pose_group_name": group_name,
    },
    skip_duplicates=True,
)

MoseqModelSelection() & group_key
# -

# We can then train the model by populating the corresponding `MoseqModel` entry.  This
# will load the keypoint data, format it for moseq, and then train according to the
# setting in the `MoseqModelParams` entry

model_key = {
    "model_params_name": model_params_name,
    "pose_group_name": group_name,
}
MoseqModel().populate(model_key)

# The model is now trained and accessible through the the `MoseqModel` table.

trained_model = MoseqModel().fetch_model(model_key)
trained_model

# We can also analyze components of the trained model;
# ie. the pca breakdown of the pose skeleton

table = MoseqModel() & model_key
table.analyze_pca()

# as well as average trajectories for each syllable

table.generate_trajectory_plots()

# And example videos of each syllable. These are saved as mp4 files in the passed `output_dir`

output_dir = "/path/to/save/videos/"
table.generate_grid_movies(output_dir=output_dir)

# ## Run data through the trained model
#
# Now that we have a trained model, we can use it to convert pose data into behavioral
# syllables. We do so by combining a trained model with an epoch of pose data, and then
# applying the populate command

# +
# %load_ext autoreload
# %autoreload 2
from spyglass.behavior.v1.moseq import MoseqSyllableSelection, MoseqSyllable

# Make a selection table entry defining the pose data and moseq model to use
pose_key = (
    PoseGroup().Pose().fetch("pose_merge_id", as_dict=True)[0]
)  # can also use data outside of the training epochs
key = {**model_key, **pose_key, "num_iters": 3}
MoseqSyllableSelection().insert1(key, skip_duplicates=True)

# Run populate to apply the model to the pose data
MoseqSyllable().populate(key)
MoseqSyllable()

# +
import matplotlib.pyplot as plt

moseq_df = (MoseqSyllable() & key).fetch1_dataframe()
moseq_df
ind = slice(1000, 3000)

fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
t = moseq_df.index.values[ind]
ax[0].plot(t, moseq_df["centroid x"].values[ind], label="x")
ax[0].plot(t, moseq_df["centroid y"].values[ind], label="y")
ax[1].scatter(
    t,
    moseq_df.syllable.values[ind],
    s=10,
    c=moseq_df.syllable.values[ind],
    cmap="tab20",
)
ax[0].set_ylabel("centroid")
ax[1].set_ylabel("syllable")
ax[1].set_xlabel("time (s)")
# -

# This concludes the tutorial for basic usage of the Moseq pipeline. Next, we will
# look at usage for extending training from a base model and leveraging spyglass's
# relational database to easily sweep through model hyperparameters

# # Extend model training
#
# There are many cases where you may want to begin trining from an existing model
# rather than begin from scratch. Examples include continuing training an incompletely
# converged entry, or using a pre-trained base model and refining it to a particular set
# of animals/ imaging conditions (ie. transfer learning).
#
# The spyglass moseq pipeline allows for this style of iterative training. To do so,
# we will define a new entry in `MoseqModelParams` using the `make_training_extension_params`
# method. This entry will have the same params as those used in model_key, except it
# will point to the model_key entry for the `initial_model`

# Insert a training extension model entry
extension_params = MoseqModelParams().make_training_extension_params(
    model_key, num_epochs=100, skip_duplicates=True
)
print("initial model: ", extension_params["model_params"]["initial_model"])
new_params_key = {
    "model_params_name": extension_params["model_params_name"],
}
MoseqModelParams() & "model_params_name LIKE '%tutorial_kappa4%'"

# This extension model can then be trained following the same steps as before

new_model_key = {
    **new_params_key,
    "pose_group_name": model_key["pose_group_name"],
}
MoseqModelSelection().insert1(new_model_key, skip_duplicates=True)
MoseqModel().populate(new_model_key)

# # Hyperparameter search (kappa)
#
# The relational database structure makes it relatively easy to train and organize
# multiple models on the same data. Here we demonstrate leveraging this architecture to
# test values of `kappa` in the moseq model. The `kappa` value determines the rate
# of syllable transitions, with larger values corresponding to longer syllables
# ([moseq docs](https://keypoint-moseq.readthedocs.io/en/latest/advanced.html#automatic-kappa-scan)).
# This value will likely need tuned for your specific data to achieve a syllable
# distribution at appropriate timescales.
#
# To do so we will make a set of parameter entries with varying values of kappa and then
# training an initial model for each. Looking at the results above, we see a shorter
# median distribution (~3 frames = 100ms) than we would like
# ([recommended ~400ms](https://keypoint-moseq.readthedocs.io/en/latest/FAQs.html#modeling))
# . We will therefore try several parameter sets with larger kappa values.
#
#

# +
original_params = (MoseqModelParams() & model_key).fetch1("model_params")

new_params_key_list = []
for i in [5, 6, 7, 8]:
    new_params = original_params.copy()
    new_params["kappa"] = 10**i
    new_params["num_epochs"] = 100
    new_model_params_name = f"tutorial_kappa{i}_mini"
    new_params_key = {
        "model_params_name": new_model_params_name,
        "model_params": new_params,
    }
    new_params_key_list.append(new_params_key)
    MoseqModelParams().insert1(new_params_key, skip_duplicates=True)

MoseqModelParams() & "model_params_name LIKE '%tutorial_kappa%'"
# -

# We can now train a model for each of these entries. We are training several different
# models here, so depending on your hardware, now may be a good time for a coffee break.

for new_params_key in new_params_key_list:
    new_model_key = {
        "model_params_name": new_params_key["model_params_name"],
        "pose_group_name": model_key["pose_group_name"],
    }
    MoseqModelSelection().insert1(new_model_key, skip_duplicates=True)
MoseqModel().populate()

# You can then choose the model that best matches your syllable duration of interest and
# continue training it using the training extension described above

# print out link to the pdf of training results (includes syllable durations)
for new_params_key in new_params_key_list:
    new_model_key = {
        "model_params_name": new_params_key["model_params_name"],
        "pose_group_name": model_key["pose_group_name"],
    }
    training_results_path = MoseqModel().get_training_progress_path(
        new_model_key
    )
    print(
        f"{new_model_key['model_params_name']} training results: {training_results_path}"
    )
