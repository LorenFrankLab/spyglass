# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3.10.5 64-bit
#     language: python
#     name: python3
# ---

# # LFP Extraction
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

# ## Imports
#

# +
import os
import copy
import datajoint as dj
import numpy as np
import pandas as pd

# change to the upper level folder to detect dj_local_conf.json
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
dj.config.load("dj_local_conf.json")  # load config for database connection info

import spyglass.common as sgc
import spyglass.lfp as lfp

# ignore datajoint+jupyter async warnings
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
# -

# ## Select data
#

# First, we select the NWB file, which corresponds to the dataset we want to extract LFP from.
#

nwb_file_name = "minirec20230622_.nwb"

# ## Create Filters
#

# Next, we create the standard LFP Filters. This only needs to be done once.
#

sgc.FirFilterParameters().create_standard_filters()
sgc.FirFilterParameters()

# ## Electrode Group
#

# Now, we create an LFP electrode group, or the set of electrodes we want to
# filter for LFP data. We can grab all electrodes and brain regions as a data
# frame.
#

electrodes_df = (
    pd.DataFrame(
        (sgc.Electrode & {"nwb_file_name": nwb_file_name, "probe_electrode": 0})
        * sgc.BrainRegion
    )
    .loc[:, ["nwb_file_name", "electrode_id", "region_name"]]
    .sort_values(by="electrode_id")
)
electrodes_df

# For a larger dataset, we might want to filter by region, but our example
# data only has one electrode.
#
# ```python
# lfp_electrode_ids = electrodes_df.loc[
#     electrodes_df.region_name == "ca1"
# ].electrode_id
# ```
#

# +
lfp_electrode_ids = [0]
lfp_electrode_group_name = "test"
lfp_eg_key = {
    "nwb_file_name": nwb_file_name,
    "lfp_electrode_group_name": lfp_electrode_group_name,
}

lfp.lfp_electrode.LFPElectrodeGroup.create_lfp_electrode_group(
    nwb_file_name=nwb_file_name,
    group_name=lfp_electrode_group_name,
    electrode_list=lfp_electrode_ids,
)
# -

# We can verify the electrode list as follows
#

lfp.lfp_electrode.LFPElectrodeGroup.LFPElectrode() & {
    "nwb_file_name": nwb_file_name
}

# ## `IntervalList`
#

# Recall from the [previous notebook](./02_Spike_Sorting.ipynb) that
# `IntervalList` selects time frames from the experiment. We can select the
# interval and subset to the first `n` seconds...
#

sgc.IntervalList & {"nwb_file_name": nwb_file_name}

# +
n = 9
orig_interval_list_name = "01_s1"
interval_list_name = orig_interval_list_name + f"_first{n}"

valid_times = (
    sgc.IntervalList
    & {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": orig_interval_list_name,
    }
).fetch1("valid_times")

interval_key = {
    "nwb_file_name": nwb_file_name,
    "interval_list_name": interval_list_name,
    "valid_times": np.asarray([[valid_times[0, 0], valid_times[0, 0] + n]]),
}

sgc.IntervalList.insert1(
    interval_key,
    skip_duplicates=True,
)
# -

sgc.IntervalList() & {
    "nwb_file_name": nwb_file_name,
    "interval_list_name": interval_list_name,
}

# ## `LFPSelection`
#

# LFPSelection combines the data, interval and filter
#

# +
lfp_s_key = copy.deepcopy(lfp_eg_key)

lfp_s_key.update(
    {
        "target_interval_list_name": interval_list_name,
        "filter_name": "LFP 0-400 Hz",
        "filter_sampling_rate": 30_000,  # sampling rate of the data (Hz)
        "target_sampling_rate": 1_000,  # smpling rate of the lfp output (Hz)
    }
)

lfp.v1.LFPSelection.insert1(lfp_s_key, skip_duplicates=True)
# -

lfp.v1.LFPSelection() & lfp_s_key

# ## Populate LFP
#

# `LFPV1` has a similar `populate` command as we've seen before.
#
# _Notes:_
#
# - For full recordings, this takes ~2h when done locally, for all electrodes
# - This `populate` also inserts into the LFP _Merge Table_, `LFPOutput`. For more
#   on Merge Tables, see our documentation. In short, this collects different LFP
#   processing streams into one table.
#

lfp.v1.LFPV1().populate(lfp_s_key)

# We can now look at the LFP table to see the data we've extracted
#

lfp.LFPOutput.LFPV1() & lfp_s_key

lfp_key = {"merge_id": (lfp.LFPOutput.LFPV1() & lfp_s_key).fetch1("merge_id")}
lfp_key

lfp.LFPOutput & lfp_key

# From the Merge Table, we can get the keys for the LFP data we want to see
#

lfp_df = (lfp.LFPOutput & lfp_key).fetch1_dataframe()
lfp_df

# ## LFP Band
#
# Now that we've created the LFP object we can perform a second level of filtering for a band of interest, in this case the theta band. We first need to create the filter.
#

# +
lfp_sampling_rate = lfp.LFPOutput.merge_get_parent(lfp_key).fetch1(
    "lfp_sampling_rate"
)

filter_name = "Theta 5-11 Hz"

sgc.common_filter.FirFilterParameters().add_filter(
    filter_name,
    lfp_sampling_rate,
    "bandpass",
    [4, 5, 11, 12],
    "theta filter for 1 Khz data",
)

sgc.common_filter.FirFilterParameters() & {
    "filter_name": filter_name,
    "filter_sampling_rate": lfp_sampling_rate,
}
# -

sgc.IntervalList()

# We can specify electrodes of interest, and desired sampling rate.
#

# +
from spyglass.lfp.analysis.v1 import lfp_band

lfp_band_electrode_ids = [0]  # assumes we've filtered these electrodes
lfp_band_sampling_rate = 100  # desired sampling rate

lfp_band.LFPBandSelection().set_lfp_band_electrodes(
    nwb_file_name=nwb_file_name,
    lfp_merge_id=lfp_key["merge_id"],
    electrode_list=lfp_band_electrode_ids,
    filter_name=filter_name,  # defined above
    interval_list_name=interval_list_name,  # Defined in IntervalList above
    reference_electrode_list=[-1],  # -1 means no ref electrode for all channels
    lfp_band_sampling_rate=lfp_band_sampling_rate,
)

lfp_band.LFPBandSelection()
# -

# Next we add an entry for the LFP Band and the electrodes we want to filter
#

lfp_band_key = (
    lfp_band.LFPBandSelection
    & {
        "lfp_merge_id": lfp_key["merge_id"],
        "filter_name": filter_name,
        "lfp_band_sampling_rate": lfp_band_sampling_rate,
    }
).fetch1("KEY")
lfp_band_key

# Check to make sure it worked
#

lfp_band.LFPBandSelection() & lfp_band_key

lfp_band.LFPBandV1().populate(lfp_band.LFPBandSelection() & lfp_band_key)
lfp_band.LFPBandV1() & lfp_band_key

# ## Plotting
#

# Now we can plot the original signal, the LFP filtered trace, and the theta
# filtered trace together. Get the three electrical series objects and the indices
# of the electrodes we band pass filtered
#
# _Note:_ Much of the code below could be replaced by a function calls that would
# return the data from each electrical series.
#
# _Note:_ If you see an error `Qt: Session Management Error`, try running the
# following unix command: `export -n SESSION_MANAGER`.
# [See also](https://stackoverflow.com/questions/986964/qt-session-management-error)
#

orig_eseries = (sgc.Raw() & {"nwb_file_name": nwb_file_name}).fetch_nwb()[0][
    "raw"
]
orig_elect_indices = sgc.get_electrode_indices(
    orig_eseries, lfp_band_electrode_ids
)
orig_timestamps = np.asarray(orig_eseries.timestamps)

lfp_eseries = lfp.LFPOutput().fetch_nwb(lfp_key)[0]["lfp"]
lfp_elect_indices = sgc.get_electrode_indices(
    lfp_eseries, lfp_band_electrode_ids
)
lfp_timestamps = np.asarray(lfp_eseries.timestamps)

lfp_band_eseries = (lfp_band.LFPBandV1 & lfp_band_key).fetch_nwb()[0][
    "lfp_band"
]
lfp_band_elect_indices = sgc.get_electrode_indices(
    lfp_band_eseries, lfp_band_electrode_ids
)
lfp_band_timestamps = np.asarray(lfp_band_eseries.timestamps)

# Get a list of times for the first run epoch and then select a 2 second interval
# 100 seconds from the beginning
#

plottimes = [valid_times[0][0] + 1, valid_times[0][0] + 8]

# +
# get the time indices for each dataset
orig_time_ind = np.where(
    np.logical_and(
        orig_timestamps > plottimes[0], orig_timestamps < plottimes[1]
    )
)[0]

lfp_time_ind = np.where(
    np.logical_and(lfp_timestamps > plottimes[0], lfp_timestamps < plottimes[1])
)[0]
lfp_band_time_ind = np.where(
    np.logical_and(
        lfp_band_timestamps > plottimes[0],
        lfp_band_timestamps < plottimes[1],
    )
)[0]

# +
import matplotlib.pyplot as plt

plt.plot(
    orig_eseries.timestamps[orig_time_ind],
    orig_eseries.data[orig_time_ind, orig_elect_indices[0]],
    "k-",
)
plt.plot(
    lfp_eseries.timestamps[lfp_time_ind],
    lfp_eseries.data[lfp_time_ind, lfp_elect_indices[0]],
    "b-",
)
plt.plot(
    lfp_band_eseries.timestamps[lfp_band_time_ind],
    lfp_band_eseries.data[lfp_band_time_ind, lfp_band_elect_indices[0]],
    "r-",
)
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (AD units)")

# Uncomment to see plot
# plt.show()
# -

# ## Next Steps
#
# Next, we'll use look at [Theta](./14_Theta.ipynb) bands within LFP data.
#
