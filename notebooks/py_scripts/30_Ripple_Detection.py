# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python [conda env:spyglass-position3] *
#     language: python
#     name: conda-env-spyglass-position3-py
# ---

# # Ripple Detection
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
# Ripple detection depends on a set of LFPs, the parameters used for detection and the speed of the animal. You will need `RippleLFPSelection`, `RippleParameters`, and `PositionOutput` to be populated accordingly.

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
import spyglass.position as sgp
import spyglass.lfp as lfp
import spyglass.lfp.analysis.v1 as lfp_analysis
from spyglass.lfp import LFPOutput
import spyglass.lfp.v1 as sglfp
from spyglass.position import PositionOutput
import spyglass.ripple.v1 as sgrip
import spyglass.ripple.v1 as sgr

# ignore datajoint+jupyter async warnings
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
# -

# ## Selecting Electrodes
#

# First, we'll pick the electrodes on which we'll run ripple detection on, using
# `RippleLFPSelection.set_lfp_electrodes`

# ?sgr.RippleLFPSelection.set_lfp_electrodes

# We'll need the `nwb_file_name`, an `electrode_list`, and to a `group_name`.
#
# - By default, `group_name` is set to CA1 for ripple detection, but we could
#   alternatively use PFC.
# - We use `nwb_file_name` to explore which electrodes are available for the
#   `electrode_list`.

nwb_file_name = "tonks20211103_.nwb"
interval_list_name = "test interval"
filter_name = "Ripple 150-250 Hz"

# Now we can look at `electrode_id` in the `Electrode` table:

electrodes = (
    (sgc.Electrode() & {"nwb_file_name": nwb_file_name})
    * (
        lfp_analysis.LFPBandSelection.LFPBandElectrode()
        & {
            "nwb_file_name": nwb_file_name,
            "filter_name": filter_name,
            "target_interval_list_name": interval_list_name,
        }
    )
    * sgc.BrainRegion
).fetch(format="frame")
electrodes

# For ripple detection, we want only tetrodes, and only the first good wire on each tetrode. We will assume that is the first wire on each tetrode. I will do this using pandas syntax but you could use datajoint to filter this table as well. Here is the filtered table.

hpc_names = ["ca1", "hippocampus", "CA1", "Hippocampus"]
electrodes.loc[
    (electrodes.region_name.isin(hpc_names)) & (electrodes.probe_electrode == 0)
]

# We only want the electrode_id to put in the `electrode_list`:

# +
electrode_list = np.unique(
    (
        electrodes.loc[
            (electrodes.region_name.isin(hpc_names))
            & (electrodes.probe_electrode == 0)
        ]
        .reset_index()
        .electrode_id
    ).tolist()
)

electrode_list.sort()
# -

# By default, `set_lfp_electrodes` will use all the available electrodes from `LFPBandV1`.
#
# We can insert into `RippleLFPSelection` and the `RippleLFPElectrode` part table,
# passing the key for the entry from `LFPBandV1`, our `electrode_list`, and the
# `group_name` into `set_lfp_electrodes`

# +
group_name = "CA1_test"

lfp_band_key = (
    lfp_analysis.LFPBandV1()
    & {"filter_name": filter_name, "nwb_file_name": nwb_file_name}
).fetch1("KEY")

sgr.RippleLFPSelection.set_lfp_electrodes(
    lfp_band_key,
    electrode_list=electrode_list,
    group_name=group_name,
)
# -

sgr.RippleLFPSelection.RippleLFPElectrode()

# Here's the ripple selection key we'll use downstream

rip_sel_key = (sgrip.RippleLFPSelection & lfp_band_key).fetch1("KEY")

# ## Setting Ripple Parameters
#

sgr.RippleParameters()

# Here are the default ripple parameters:

(sgrip.RippleParameters() & {"ripple_param_name": "default"}).fetch1()

# - `filter_name`: which bandpass filter is used
# - `speed_name`: the name of the speed parameters in `IntervalPositionInfo`
#
# For the `Kay_ripple_detector` (options are currently Kay and Karlsson, see `ripple_detection` package for specifics) the parameters are:
#
# - `speed_threshold` (cm/s): maximum speed the animal can move
# - `minimum_duration` (s): minimum time above threshold
# - `zscore_threshold` (std): minimum value to be considered a ripple, in standard
#   deviations from mean
# - `smoothing_sigma` (s): how much to smooth the signal in time
# - `close_ripple_threshold` (s): exclude ripples closer than this amount
#

# ## Check interval speed
#
# The speed for this interval should exist under the default position parameter
# set and for a given interval.

pos_key = sgp.PositionOutput.merge_get_part(
    {
        "nwb_file_name": nwb_file_name,
        "position_info_param_name": "default",
        "interval_list_name": "pos 1 valid times",
    }
).fetch1("KEY")
(sgp.PositionOutput & pos_key).fetch1_dataframe()

# We'll use the `head_speed` above as part of `RippleParameters`.

# ## Run Ripple Detection
#

#
# Now we can put everything together.

key = {
    "ripple_param_name": "default",
    **rip_sel_key,
    "pos_merge_id": pos_key["merge_id"],
}
sgrip.RippleTimesV1().populate(key)

# And then `fetch1_dataframe` for ripple times

ripple_times = (sgrip.RippleTimesV1() & key).fetch1_dataframe()
ripple_times

# ## Up Next
#
# Next, we'll [extract mark indicator](./31_Extract_Mark_Indicators.ipynb).
