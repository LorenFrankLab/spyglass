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
from pathlib import Path
import datajoint as dj

dj.config.load(
    Path("../dj_local_conf.json").absolute()
)  # load config for database connection info
# -

# # MUA Analysis and Detection
#
# NOTE: This notebook is a work in progress. It is not yet complete and may contain errors.

# +
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
import spyglass.spikesorting.v1 as sgs


nwb_copy_file_name = "mediumnwb20230802_.nwb"

sorter_keys = {
    "nwb_file_name": nwb_copy_file_name,
    "sorter": "clusterless_thresholder",
    "sorter_param_name": "default_clusterless",
}

(sgs.SpikeSortingSelection & sorter_keys) * SpikeSortingOutput.CurationV1

# +
spikesorting_merge_ids = (
    (sgs.SpikeSortingSelection & sorter_keys) * SpikeSortingOutput.CurationV1
).fetch("merge_id")

spikesorting_merge_ids

# +
from spyglass.spikesorting.unit_inclusion_merge import (
    ImportedUnitInclusionV1,
    UnitInclusionOutput,
)

ImportedUnitInclusionV1().insert_all_units(spikesorting_merge_ids)

UnitInclusionOutput.ImportedUnitInclusionV1() & [
    {"spikesorting_merge_id": id} for id in spikesorting_merge_ids
]

# +
from spyglass.spikesorting.unit_inclusion_merge import (
    ImportedUnitInclusionV1,
    UnitInclusionOutput,
)

ImportedUnitInclusionV1().insert_all_units(spikesorting_merge_ids)

UnitInclusionOutput.ImportedUnitInclusionV1() & [
    {"spikesorting_merge_id": id} for id in spikesorting_merge_ids
]

# +
from spyglass.spikesorting.unit_inclusion_merge import SortedSpikesGroup

unit_inclusion_merge_ids = (
    UnitInclusionOutput.ImportedUnitInclusionV1
    & [{"spikesorting_merge_id": id} for id in spikesorting_merge_ids]
).fetch("merge_id")

SortedSpikesGroup().create_group(
    group_name="test_group",
    nwb_file_name=nwb_copy_file_name,
    unit_inclusion_merge_ids=unit_inclusion_merge_ids,
)

group_key = {
    "nwb_file_name": nwb_copy_file_name,
    "sorted_spikes_group_name": "test_group",
}

SortedSpikesGroup & group_key
# -

SortedSpikesGroup.Units() & group_key

# An example of how to get spike times

spike_times = SortedSpikesGroup.fetch_spike_data(group_key)
spike_times[0]

# +
from spyglass.position import PositionOutput

position_merge_id = (
    PositionOutput.TrodesPosV1
    & {
        "nwb_file_name": nwb_copy_file_name,
        "interval_list_name": "pos 0 valid times",
        "trodes_pos_params_name": "default_decoding",
    }
).fetch1("merge_id")

position_info = (
    (PositionOutput & {"merge_id": position_merge_id})
    .fetch1_dataframe()
    .dropna()
)
position_info

# +
time_ind_slice = slice(63_000, 70_000)
time = position_info.index[time_ind_slice]

SortedSpikesGroup.get_spike_indicator(group_key, time)

# +
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 4))
multiunit_firing_rate = SortedSpikesGroup.get_firing_rate(
    group_key, time, multiunit=True
)
axes[0].plot(
    time,
    multiunit_firing_rate,
)
axes[0].set_ylabel("firing rate (Hz)")
axes[0].set_title("multiunit")
axes[1].fill_between(
    time, position_info["speed"].iloc[time_ind_slice], color="lightgrey"
)
axes[1].set_ylabel("speed (cm/s)")
axes[1].set_xlabel("time (s)")

# +
from spyglass.mua.v1.mua import MuaEventsParameters, MuaEventsV1

MuaEventsParameters().insert_default()
MuaEventsParameters()

# +
selection_key = {
    "mua_param_name": "default",
    "nwb_file_name": nwb_copy_file_name,
    "sorted_spikes_group_name": "test_group",
    "pos_merge_id": position_merge_id,
    "artifact_interval_list_name": "test_artifact_times",
}

MuaEventsV1.populate(selection_key)
# -

MuaEventsV1 & selection_key

mua_times = (MuaEventsV1 & selection_key).fetch1_dataframe()
mua_times

# +
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 4))
axes[0].plot(
    time,
    multiunit_firing_rate,
)
axes[0].set_ylabel("firing rate (Hz)")
axes[0].set_title("multiunit")
axes[1].fill_between(
    time, position_info["speed"].iloc[time_ind_slice], color="lightgrey"
)
axes[1].set_ylabel("speed (cm/s)")
axes[1].set_xlabel("time (s)")

in_bounds = np.logical_and(
    mua_times.start_time >= time[0], mua_times.end_time <= time[-1]
)

for mua_time in mua_times.loc[in_bounds].itertuples():
    axes[0].axvspan(
        mua_time.start_time, mua_time.end_time, color="red", alpha=0.3
    )
    axes[1].axvspan(
        mua_time.start_time, mua_time.end_time, color="red", alpha=0.3
    )
axes[1].set_ylim((0, 80))
axes[1].axhline(4, color="black", linestyle="--")
axes[1].set_xlim((time[0], time[-1]))

# +
from spyglass.common import IntervalList

IntervalList() & {
    "nwb_file_name": nwb_copy_file_name,
    "pipeline": "spikesorting_artifact_v1",
}
# -

(
    sgs.ArtifactDetectionParameters
    * sgs.SpikeSortingRecording
    * sgs.ArtifactDetectionSelection
)

SpikeSortingOutput.CurationV1() * (
    sgs.ArtifactDetectionParameters
    * sgs.SpikeSortingRecording
    * sgs.ArtifactDetectionSelection
)

(
    IntervalList()
    & {
        "nwb_file_name": nwb_copy_file_name,
        "pipeline": "spikesorting_artifact_v1",
    }
).proj(artifact_id="interval_list_name")

sgs.SpikeSortingRecording() * sgs.ArtifactDetectionSelection()

SpikeSortingOutput.CurationV1() * sgs.SpikeSortingRecording()

IntervalList.insert1(
    {
        "nwb_file_name": nwb_copy_file_name,
        "interval_list_name": "test_artifact_times",
        "valid_times": [],
    }
)
