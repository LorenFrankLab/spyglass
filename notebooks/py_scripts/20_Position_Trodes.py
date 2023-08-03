# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Trodes Position

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
# - Insert the results into the `IntervalPositionInfo` table
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
import spyglass.utils as sgu

# ignore datajoint+jupyter async warnings
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
# -

# ## Loading the data
#
# First, we'll grab let us make sure that the session we want to analyze is inserted into the `RawPosition` table

nwb_file_name = "chimi20200216_new.nwb"
nwb_copy_file_name = sgu.nwb_helper_fn.get_nwb_copy_filename(nwb_file_name)
sgc.common_behav.RawPosition() & {"nwb_file_name": nwb_copy_file_name}

# ## Setting parameters 
#
# Parameters are set by the `TrodesPosParams` table, with a `default` set
# available. To adjust the default, insert a new set into this table. The
# parameters are...
#
# - `max_separation`, default 9 cm: maxmium acceptable distance between red and
#   green LEDs. 
#     - If exceeded, the times are marked as NaNs and inferred by interpolation. 
#     - Useful when the inferred LED position tracks a reflection instead of the
#       true position.
# - `max_speed`, default 300.0 cm/s: maximum speed the animal can move. 
#     - If exceeded, times are marked as NaNs and inferred by interpolation. 
#     - Useful to prevent big jumps in position. 
# - `position_smoothing_duration`, default 0.100 s: LED position smoothing before
#   computing average position to get head position. 
# - `speed_smoothing_std_dev`, default 0.100 s: standard deviation of the Gaussian
#   kernel used to smooth the head speed.
# - `front_led1`, default 1 (True), use `xloc`/`yloc`: Which LED is the front LED
#   for calculating the head direction.
#     - 1: LED corresponding to `xloc`, `yloc` in the `RawPosition` table is the
#       front, `xloc2`, `yloc2` as the back.
#     - 0: LED corresponding to `xloc2`, `yloc2` in the `RawPosition` table is the
#       front, `xloc`, `yloc` as the back.
#
# We can see these defaults with `TrodesPosParams.get_default`.

# +
from pprint import pprint

parameters = sgp.v1.TrodesPosParams.get_default()["params"]
pprint(parameters)
# -

parameters["led1_is_front"] = 0
sgp.v1.TrodesPosParams.insert1(
    {"trodes_pos_params_name": "default_led0", "params": parameters},
    skip_duplicates=True,
)
sgp.v1.TrodesPosParams()

# ## Select interval

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

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(raw_position_df.xloc, raw_position_df.yloc, color="green")
ax.plot(raw_position_df.xloc2, raw_position_df.yloc2, color="red")
ax.set_xlabel("x-position [pixels]", fontsize=18)
ax.set_ylabel("y-position [pixels]", fontsize=18)
ax.set_title("Raw Position", fontsize=28)

# ## Pairing interval and parameters

# To associate a set of parameters with a given interval, insert them into the
# `TrodesPosSelection` table.

sgp.TrodesPosSelection.insert1(
    {
        "nwb_file_name": nwb_copy_file_name,
        "interval_list_name": interval_list_name,
        "trodes_pos_params_name": "default",
    },
    skip_duplicates=True,
)

# Now let's check to see if we've inserted the parameters correctly:

sgp.TrodesPosSelection()
trodes_key = (
    sgp.TrodesPosSelection()
    & {
        "nwb_file_name": nwb_copy_file_name,
        "interval_list_name": interval_list_name,
        "trodes_pos_params_name": "default",
    }
).fetch1("KEY")

# ## Running the pipeline
#

# We can run the pipeline for our chosen interval/parameters by using the
# `TrodesPos.populate`.

sgp.TrodesPos.populate(trodes_key)

# Each NWB file, interval, and parameter set is now associated with a new analysis file and object ID.
#

sgp.TrodesPos()

# To retrieve the results as a pandas DataFrame with time as the index, we use `IntervalPositionInfo.fetch1_dataframe`.
#
# This dataframe has the following columns:
# - `head_position_{x,y}`: X or Y position of the head in cm.
# - `head_orientation`: Direction of the head relative to the bottom left corner
#   in radians
# - `head_velocity_{x,y}`: Directional change in head position over time in cm/s
# - `head_speed`: the magnitude of the change in head position over time in cm/s

position_info = (
    sgc.IntervalPositionInfo()
    & {
        "nwb_file_name": nwb_copy_file_name,
        "interval_list_name": "pos 1 valid times",
        "position_info_param_name": "default",
    }
).fetch1_dataframe()
position_info

# `.index` on the pandas dataframe gives us timestamps.

position_info.index

# ## Examine results
#

# We should always spot check our results to verify that the pipeline worked correctly.
#
# ### Plots
#
# Let's plot some of the variables first:

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(position_info.head_position_x, position_info.head_position_y)
ax.set_xlabel("x-position [cm]", fontsize=18)
ax.set_ylabel("y-position [cm]", fontsize=18)
ax.set_title("Head Position", fontsize=28)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(position_info.head_velocity_x, position_info.head_velocity_y)
ax.set_xlabel("x-velocity [cm/s]", fontsize=18)
ax.set_ylabel("y-velocity [cm/s]", fontsize=18)
ax.set_title("Head Velocity", fontsize=28)

fig, ax = plt.subplots(1, 1, figsize=(25, 3))
ax.plot(position_info.index, position_info.head_speed)
ax.set_xlabel("Time", fontsize=18)
ax.set_ylabel("Speed [cm/s]", fontsize=18)
ax.set_title("Head Speed", fontsize=28)
ax.set_xlim((position_info.index.min(), position_info.index.max()))

# ### Video
#
# These look reasonable but we can visualize further by plotting the results on
#  the video, which will appear in the current working directory.

# +
from spyglass.common.common_position import PositionVideo

PositionVideo().make(
    {
        "nwb_file_name": nwb_copy_file_name,
        "interval_list_name": "pos 1 valid times",
        "position_info_param_name": "default",
    }
)
# -

# ## Upsampling position
#
# Sometimes we need the position data in smaller in time bins, which can be 
# achieved with upsampling using the following parameters.
#
# - `is_upsampled`, default 0 (False): If 1, perform upsampling.
# - `upsampling_sampling_rate`, default None: the rate to upsample to (e.g.,
#   33 Hz video might be upsampled to 500 Hz).
# - `upsampling_interpolation_method`, default linear: interpolation method. See
#   [pandas.DataFrame.interpolate](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html)
#   for alternate methods.

# +
sgc.PositionInfoParameters.insert1(
    {
        "position_info_param_name": "default_decoding",
        "is_upsampled": 1,
        "upsampling_sampling_rate": 500,
    },
    skip_duplicates=True,
)

sgc.PositionInfoParameters()

# +
sgc.IntervalPositionInfoSelection.insert1(
    {
        "nwb_file_name": nwb_copy_file_name,
        "interval_list_name": "pos 1 valid times",
        "position_info_param_name": "default_decoding",
    },
    skip_duplicates=True,
)

sgc.IntervalPositionInfoSelection()
# -

sgc.IntervalPositionInfo.populate()

# +
upsampled_position_info = (
    sgc.IntervalPositionInfo()
    & {
        "nwb_file_name": nwb_copy_file_name,
        "interval_list_name": "pos 1 valid times",
        "position_info_param_name": "default_decoding",
    }
).fetch1_dataframe()

upsampled_position_info

# +
fig, axes = plt.subplots(
    1, 2, figsize=(20, 10), sharex=True, sharey=True, constrained_layout=True
)
axes[0].plot(position_info.head_position_x, position_info.head_position_y)
axes[0].set_xlabel("x-position [cm]", fontsize=18)
axes[0].set_ylabel("y-position [cm]", fontsize=18)
axes[0].set_title("Head Position", fontsize=28)

axes[1].plot(
    upsampled_position_info.head_position_x,
    upsampled_position_info.head_position_y,
)
axes[1].set_xlabel("x-position [cm]", fontsize=18)
axes[1].set_ylabel("y-position [cm]", fontsize=18)
axes[1].set_title("Upsampled Head Position", fontsize=28)

# +
fig, axes = plt.subplots(
    2, 1, figsize=(25, 6), sharex=True, sharey=True, constrained_layout=True
)
axes[0].plot(position_info.index, position_info.head_speed)
axes[0].set_xlabel("Time", fontsize=18)
axes[0].set_ylabel("Speed [cm/s]", fontsize=18)
axes[0].set_title("Head Speed", fontsize=28)
axes[0].set_xlim((position_info.index.min(), position_info.index.max()))

axes[1].plot(upsampled_position_info.index, upsampled_position_info.head_speed)
axes[1].set_xlabel("Time", fontsize=18)
axes[1].set_ylabel("Speed [cm/s]", fontsize=18)
axes[1].set_title("Upsampled Head Speed", fontsize=28)

# +
fig, axes = plt.subplots(
    1, 2, figsize=(20, 10), sharex=True, sharey=True, constrained_layout=True
)
axes[0].plot(position_info.head_velocity_x, position_info.head_velocity_y)
axes[0].set_xlabel("x-velocity [cm/s]", fontsize=18)
axes[0].set_ylabel("y-velocity [cm/s]", fontsize=18)
axes[0].set_title("Head Velocity", fontsize=28)

axes[1].plot(
    upsampled_position_info.head_velocity_x,
    upsampled_position_info.head_velocity_y,
)
axes[1].set_xlabel("x-velocity [cm/s]", fontsize=18)
axes[1].set_ylabel("y-velocity [cm/s]", fontsize=18)
axes[1].set_title("Upsampled Head Velocity", fontsize=28)
# -


