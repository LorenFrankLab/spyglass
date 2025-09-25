# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: spyglass
#     language: python
#     name: spyglass
# ---

# # MUA Detection

# ## Overview

# _Developer Note:_ if you may make a PR in the future, be sure to copy this
# notebook, and use the `gitignore` prefix `temp` to avoid future conflicts.
#
# This is one notebook in a multi-part series on Spyglass.
#
# - To set up your Spyglass environment and database, see
#   [the Setup notebook](./00_Setup.ipynb).
# - For additional info on DataJoint syntax, including table definitions and
#   inserts, see
#   [the Insert Data notebook](./01_Insert_Data.ipynb).
# - Prior to running, please generate sorted spikes with the [spike sorting
#   pipeline](./02_Spike_Sorting.ipynb) and generate input position data with
#   either the [Trodes](./20_Position_Trodes.ipynb) or DLC notebooks
#   ([1](./21_Position_DLC_1.ipynb), [2](./22_Position_DLC_2.ipynb),
#   [3](./23_Position_DLC_3.ipynb)).
#
# The goal of this notebook is to populate the `MuaEventsV1` table, which depends `SortedSpikesGroup` and `PositionOutput`.

# # Imports

# +
import datajoint as dj
from pathlib import Path

dj.config.load(
    Path("../dj_local_conf.json").absolute()
)  # load config for database connection info

from spyglass.mua.v1.mua import MuaEventsV1, MuaEventsParameters

# -

# ## Select Position Data

# +
from spyglass.position import PositionOutput

# First, select the file of interest
nwb_copy_file_name = "mediumnwb20230802_.nwb"

# Then, get position data
trodes_s_key = {
    "nwb_file_name": nwb_copy_file_name,
    "interval_list_name": "pos 0 valid times",
    "trodes_pos_params_name": "single_led_upsampled",
}

pos_merge_id = (PositionOutput.TrodesPosV1 & trodes_s_key).fetch1("merge_id")
pos_merge_id
# -

# ## Select Sorted Spikes Data

# +
from spyglass.spikesorting.analysis.v1.group import (
    SortedSpikesGroup,
)

# Select sorted spikes data
sorted_spikes_group_key = {
    "nwb_file_name": nwb_copy_file_name,
    "sorted_spikes_group_name": "test_group",
    "unit_filter_params_name": "default_exclusion",
}

SortedSpikesGroup & sorted_spikes_group_key
# -

# # Setting MUA Parameters

MuaEventsParameters()

# Here are the default parameters:

(MuaEventsParameters() & {"mua_param_name": "default"}).fetch1()

# Putting everything together: create a key and populate the MuaEventsV1 table

# +
mua_key = {
    "mua_param_name": "default",
    **sorted_spikes_group_key,
    "pos_merge_id": pos_merge_id,
    "detection_interval": "pos 0 valid times",
}

MuaEventsV1().populate(mua_key)
MuaEventsV1 & mua_key
# -

# Now we can use `fetch1_dataframe` for mua data, including start times, end times, and speed.

mua_times = (MuaEventsV1 & mua_key).fetch1_dataframe()
mua_times

# ## Plotting

# From this, we can plot MUA firing rate and speed together.

# +
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 4))
speed = MuaEventsV1.get_speed(mua_key)  # get speed from MuaEventsV1 table
time = speed.index.to_numpy()
speed = speed.to_numpy()
multiunit_firing_rate = MuaEventsV1.get_firing_rate(
    mua_key, time
)  # get firing rate from MuaEventsV1 table

time_slice = slice(
    np.searchsorted(time, mua_times.loc[10].start_time) - 1_000,
    np.searchsorted(time, mua_times.loc[10].start_time) + 5_000,
)

axes[0].plot(
    time[time_slice],
    multiunit_firing_rate[time_slice],
    color="black",
)
axes[0].set_ylabel("firing rate (Hz)")
axes[0].set_title("multiunit activity")
axes[1].fill_between(time[time_slice], speed[time_slice], color="lightgrey")
axes[1].set_ylabel("speed (cm/s)")
axes[1].set_xlabel("time (s)")

for id, mua_time in mua_times.loc[
    np.logical_and(
        mua_times["start_time"] > time[time_slice].min(),
        mua_times["end_time"] < time[time_slice].max(),
    )
].iterrows():
    axes[0].axvspan(
        mua_time["start_time"], mua_time["end_time"], color="red", alpha=0.5
    )
# -

# We can also create a figurl to visualize the data.

(MuaEventsV1 & mua_key).create_figurl(
    zscore_mua=True,
)
