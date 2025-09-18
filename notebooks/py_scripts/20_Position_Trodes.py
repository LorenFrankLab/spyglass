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

# # Trodes Position
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
# In this tutorial, we'll process position data extracted with Trodes Tracking by
#
# - Defining parameters
# - Processing raw position
# - Extracting centroid and orientation
# - Insert the results into the `TrodesPosV1` table
# - Plotting the head position/direction results for quality assurance
#
# The pipeline takes the 2D video pixel data of green/red LEDs, and computes:
#
# - head position (in cm)
# - head orientation (in radians)
# - head velocity (in cm/s)
# - head speed (in cm/s)
#

# ## Imports
#

# +
import os
import datajoint as dj
import matplotlib.pyplot as plt

# change to the upper level folder to detect dj_local_conf.json
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
dj.config.load("dj_local_conf.json")  # load config for database connection info

import spyglass.common as sgc
import spyglass.position as sgp

# ignore datajoint+jupyter async warnings
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
# -

# ## Loading the data
#

# First, we'll grab let us make sure that the session we want to analyze is inserted into the `RawPosition` table
#

# +
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

# Define the name of the file that you copied and renamed
nwb_file_name = "minirec20230622.nwb"
nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
# -

sgc.common_behav.RawPosition() & {"nwb_file_name": nwb_copy_file_name}

# ## Setting parameters
#

# Parameters are set by the `TrodesPosParams` table, with a `default` set
# available. To adjust the default, insert a new set into this table. The
# parameters are...
#
# - `max_separation`, default 9 cm: maximum acceptable distance between red and
#   green LEDs.
#   - If exceeded, the times are marked as NaNs and inferred by interpolation.
#   - Useful when the inferred LED position tracks a reflection instead of the
#     true position.
# - `max_speed`, default 300.0 cm/s: maximum speed the animal can move.
#   - If exceeded, times are marked as NaNs and inferred by interpolation.
#   - Useful to prevent big jumps in position.
# - `position_smoothing_duration`, default 0.100 s: LED position smoothing before
#   computing average position to get head position.
# - `speed_smoothing_std_dev`, default 0.100 s: standard deviation of the Gaussian
#   kernel used to smooth the head speed.
# - `front_led1`, default 1 (True), use `xloc`/`yloc`: Which LED is the front LED
#   for calculating the head direction.
#   - 1: LED corresponding to `xloc`, `yloc` in the `RawPosition` table is the
#     front, `xloc2`, `yloc2` as the back.
#   - 0: LED corresponding to `xloc2`, `yloc2` in the `RawPosition` table is the
#     front, `xloc`, `yloc` as the back.
#
# We can see these defaults with `TrodesPosParams().default_params`.
#

# +
from pprint import pprint

parameters = sgp.v1.TrodesPosParams().default_params
pprint(parameters)
# -

# For the `minirec` demo file, only one LED is moving. The following paramset will
# allow us to process this data.

trodes_params_name = "single_led"
trodes_params = {
    "max_separation": 10000.0,
    "max_speed": 300.0,
    "position_smoothing_duration": 0.125,
    "speed_smoothing_std_dev": 0.1,
    "orient_smoothing_std_dev": 0.001,
    "led1_is_front": 1,
    "is_upsampled": 0,
    "upsampling_sampling_rate": None,
    "upsampling_interpolation_method": "linear",
}
sgp.v1.TrodesPosParams.insert1(
    {
        "trodes_pos_params_name": trodes_params_name,
        "params": trodes_params,
    },
    skip_duplicates=True,
)
sgp.v1.TrodesPosParams()

# ## Select interval
#

# Later, we'll pair the above parameters with an interval from our NWB file and
# insert into `TrodesPosSelection`.
#
# First, let's select an interval from the `IntervalList` table.
#

sgc.IntervalList & {"nwb_file_name": nwb_copy_file_name}

# The raw position in pixels is in the `RawPosition` table is extracted from the
# video data by the algorithm in Trodes. We have timepoints available for the
# duration when position tracking was turned on and off, which may be a subset of
# the video itself.
#
# `fetch1_dataframe` returns the position of the LEDs as a pandas dataframe where
# time is the index.
#

interval_list_name = "pos 0 valid times"  # pos # is epoch # minus 1
raw_position_df = (
    sgc.RawPosition()
    & {
        "nwb_file_name": nwb_copy_file_name,
        "interval_list_name": interval_list_name,
    }
).fetch1_dataframe()
raw_position_df

# Let's just quickly plot the two LEDs to get a sense of the inputs to the pipeline:
#

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(raw_position_df.xloc1, raw_position_df.yloc1, color="green")
# Uncomment for multiple LEDs
# ax.plot(raw_position_df.xloc2, raw_position_df.yloc2, color="red")
ax.set_xlabel("x-position [pixels]", fontsize=18)
ax.set_ylabel("y-position [pixels]", fontsize=18)
ax.set_title("Raw Position", fontsize=28)

# ## Pairing interval and parameters
#

# To associate a set of parameters with a given interval, insert them into the
# `TrodesPosSelection` table.
#

trodes_s_key = {
    "nwb_file_name": nwb_copy_file_name,
    "interval_list_name": interval_list_name,
    "trodes_pos_params_name": trodes_params_name,
}
sgp.v1.TrodesPosSelection.insert1(
    trodes_s_key,
    skip_duplicates=True,
)

# Now let's check to see if we've inserted the parameters correctly:
#

trodes_key = (sgp.v1.TrodesPosSelection() & trodes_s_key).fetch1("KEY")

# ## Running the pipeline
#

# We can run the pipeline for our chosen interval/parameters by using the
# `TrodesPosV1.populate`.
#

sgp.v1.TrodesPosV1.populate(trodes_key)

# Each NWB file, interval, and parameter set is now associated with a new analysis file and object ID.
#

sgp.v1.TrodesPosV1 & trodes_key

# When we populatethe `TrodesPosV1` table, we automatically create an entry in the `PositionOutput` merge table.
# Since this table supports position information from multiple methods, it's best practive to access data through here.
#
# We can view the entry in this table:

# +
from spyglass.position import PositionOutput

PositionOutput.TrodesPosV1 & trodes_key
# -

# To retrieve the results as a pandas DataFrame with time as the index, we use `PositionOutput.fetch1_dataframe`.
# When doing so, we need to restric the merge table by the
#
# This dataframe has the following columns:
#
# - `position_{x,y}`: X or Y position of the head in cm.
# - `orientation`: Direction of the head relative to the bottom left corner
#   in radians
# - `velocity_{x,y}`: Directional change in head position over time in cm/s
# - `speed`: the magnitude of the change in head position over time in cm/s
#

# get the merge id corresponding to our inserted trodes_key
merge_key = (PositionOutput.merge_get_part(trodes_key)).fetch1("KEY")
# use this to restrict PositionOutput and fetch the data
position_info = (PositionOutput & merge_key).fetch1_dataframe()
position_info

# `.index` on the pandas dataframe gives us timestamps.
#

position_info.index

# ## Examine results
#

# We should always spot check our results to verify that the pipeline worked correctly.
#

#
# ### Plots
#

#
# Let's plot some of the variables first:
#

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(position_info.position_x, position_info.position_y)
ax.set_xlabel("x-position [cm]", fontsize=16)
ax.set_ylabel("y-position [cm]", fontsize=16)
ax.set_title("Position", fontsize=20)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(position_info.velocity_x, position_info.velocity_y)
ax.set_xlabel("x-velocity [cm/s]", fontsize=16)
ax.set_ylabel("y-velocity [cm/s]", fontsize=16)
ax.set_title("Velocity", fontsize=20)

fig, ax = plt.subplots(1, 1, figsize=(16, 3))
ax.plot(position_info.index, position_info.speed)
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel("Speed [cm/s]", fontsize=16)
ax.set_title("Head Speed", fontsize=20)
ax.set_xlim((position_info.index.min(), position_info.index.max()))

# ### Video
#
# To keep `minirec` small, the download link does not include videos by default.
#
# If it is available, you can uncomment the code, populate the  `TrodesPosVideo` table, and plot the results on the video using the `make_video` function, which will appear in the current working directory.
#

# +
# sgp.v1.TrodesPosVideo().populate(
#     {
#         "nwb_file_name": nwb_copy_file_name,
#         "interval_list_name": interval_list_name,
#         "position_info_param_name": trodes_params_name,
#     }
# )

# +
# sgp.v1.TrodesPosVideo()
# -

# ## Upsampling position
#
# To get position data in smaller in time bins, we can upsample using the
# following parameters
#
# - `is_upsampled`, default 0 (False): If 1, perform upsampling.
# - `upsampling_sampling_rate`, default None: the rate to upsample to (e.g.,
#   33 Hz video might be upsampled to 500 Hz).
# - `upsampling_interpolation_method`, default linear: interpolation method. See
#   [pandas.DataFrame.interpolate](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html)
#   for alternate methods.
#

# +
trodes_params_up_name = trodes_params_name + "_upsampled"
trodes_params_up = {
    **trodes_params,
    "is_upsampled": 1,
    "upsampling_sampling_rate": 500,
}
sgp.v1.TrodesPosParams.insert1(
    {
        "trodes_pos_params_name": trodes_params_up_name,
        "params": trodes_params_up,
    },
    skip_duplicates=True,
)

sgp.v1.TrodesPosParams()
# -

trodes_s_up_key = {
    "nwb_file_name": nwb_copy_file_name,
    "interval_list_name": interval_list_name,
    "trodes_pos_params_name": trodes_params_up_name,
}
sgp.v1.TrodesPosSelection.insert1(
    trodes_s_up_key,
    skip_duplicates=True,
)
sgp.v1.TrodesPosV1.populate(trodes_s_up_key)

merge_key = (PositionOutput.merge_get_part(trodes_s_up_key)).fetch1("KEY")
upsampled_position_info = (PositionOutput & merge_key).fetch1_dataframe()
upsampled_position_info

# +
fig, axes = plt.subplots(
    1, 2, figsize=(16, 8), sharex=True, sharey=True, constrained_layout=True
)
axes[0].plot(position_info.position_x, position_info.position_y)
axes[0].set_xlabel("x-position [cm]", fontsize=16)
axes[0].set_ylabel("y-position [cm]", fontsize=16)
axes[0].set_title("Position", fontsize=20)

axes[1].plot(
    upsampled_position_info.position_x,
    upsampled_position_info.position_y,
)
axes[1].set_xlabel("x-position [cm]", fontsize=16)
axes[1].set_ylabel("y-position [cm]", fontsize=16)
axes[1].set_title("Upsampled Position", fontsize=20)

# +
fig, axes = plt.subplots(
    2, 1, figsize=(16, 6), sharex=True, sharey=True, constrained_layout=True
)
axes[0].plot(position_info.index, position_info.speed)
axes[0].set_xlabel("Time", fontsize=16)
axes[0].set_ylabel("Speed [cm/s]", fontsize=16)
axes[0].set_title("Speed", fontsize=20)
axes[0].set_xlim((position_info.index.min(), position_info.index.max()))

axes[1].plot(upsampled_position_info.index, upsampled_position_info.speed)
axes[1].set_xlabel("Time", fontsize=16)
axes[1].set_ylabel("Speed [cm/s]", fontsize=16)
axes[1].set_title("Upsampled Speed", fontsize=20)

# +
fig, axes = plt.subplots(
    1, 2, figsize=(16, 8), sharex=True, sharey=True, constrained_layout=True
)
axes[0].plot(position_info.velocity_x, position_info.velocity_y)
axes[0].set_xlabel("x-velocity [cm/s]", fontsize=16)
axes[0].set_ylabel("y-velocity [cm/s]", fontsize=16)
axes[0].set_title("Velocity", fontsize=20)

axes[1].plot(
    upsampled_position_info.velocity_x,
    upsampled_position_info.velocity_y,
)
axes[1].set_xlabel("x-velocity [cm/s]", fontsize=16)
axes[1].set_ylabel("y-velocity [cm/s]", fontsize=16)
axes[1].set_title("Upsampled Velocity", fontsize=20)
# -

# ## Up Next

# In the [next notebook](./21_Position_DLC_1.ipynb), we'll explore using DeepLabCut to generate position data from video.
#
