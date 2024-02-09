# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: spyglass
#     language: python
#     name: python3
# ---

# +
import datajoint as dj
from pathlib import Path

dj.config.load(
    Path("../dj_local_conf.json").absolute()
)  # load config for database connection info

from spyglass.mua.v1.mua import MuaEventsV1, MuaEventsParameters

# -

MuaEventsParameters()

MuaEventsV1()

# +
from spyglass.position import PositionOutput

nwb_copy_file_name = "mediumnwb20230802_.nwb"

trodes_s_key = {
    "nwb_file_name": nwb_copy_file_name,
    "interval_list_name": "pos 0 valid times",
    "trodes_pos_params_name": "single_led_upsampled",
}

pos_merge_id = (PositionOutput.TrodesPosV1 & trodes_s_key).fetch1("merge_id")
pos_merge_id

# +
from spyglass.spikesorting.analysis.v1.group import (
    SortedSpikesGroup,
)

sorted_spikes_group_key = {
    "nwb_file_name": nwb_copy_file_name,
    "sorted_spikes_group_name": "test_group",
    "unit_filter_params_name": "default_exclusion",
}

SortedSpikesGroup & sorted_spikes_group_key

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

mua_times = (MuaEventsV1 & mua_key).fetch1_dataframe()
mua_times

# +
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 4))
speed = MuaEventsV1.get_speed(mua_key).to_numpy()
time = speed.index.to_numpy()
multiunit_firing_rate = MuaEventsV1.get_firing_rate(mua_key, time)

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
axes[0].set_title("multiunit")
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

(MuaEventsV1 & mua_key).create_figurl(
    zscore_mua=True,
)
