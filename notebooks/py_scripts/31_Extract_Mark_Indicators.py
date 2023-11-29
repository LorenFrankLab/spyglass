# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3.10.5 64-bit
#     language: python
#     name: python3
# ---

# ## Mark Indicators

# ## Overview
#

# _Developer Note:_ if you may make a PR in the future, be sure to copy this
# notebook, and use the `gitignore` prefix `temp` to avoid future conflicts.
#
# This is one notebook in a multi-part series on clusterless decoding in Spyglass
#
# - To set up your Spyglass environment and database, see
#   [the Setup notebook](./00_Setup.ipynb)
# - For additional info on DataJoint syntax, including table definitions and
#   inserts, see
#   [the Insert Data notebook](./01_Insert_Data.ipynb)
# - Prior to running, please familiarize yourself with the [spike sorting
#   pipeline](./02_Spike_Sorting.ipynb) and generate input position data with
#   either the [Trodes](./20_Position_Trodes.ipynb) or DLC notebooks
#   ([1](./21_Position_DLC_1.ipynb), [2](./22_Position_DLC_2.ipynb),
#   [3](./23_Position_DLC_3.ipynb)).
#
# The goal of this notebook is to populate the `UnitMarksIndicator` table, which depends on a series of tables in the spike sorting pipeline:
#
# - `SpikeSorting` -> `CuratedSpikeSorting` -> `UnitMarks` -> `UnitMarkIndicators`
#
# While clusterless decoding avoids actual spike sorting and curation, we need to pass through these tables to maintain (relative) pipeline simplicity. Pass-through tables keep spike sorting and clusterless mark extraction as similar as possible, by using shared steps. Here, "spike sorting" involves simple thresholding (sorter: clusterless_thresholder). The  `populate_mark_indicators` will run each of these steps provided we have data in  `SpikeSortingSelection` and `IntervalPositionInfo`.
#
# `SpikeSortingSelection` depends on:
# - `SpikeSortingRecording`
# - `SpikeSorterParameters`
# - `ArtifactRemovedIntervalList`.
#
# `SpikeSortingRecording` depends on:
# - `SortGroup`
# - `SortInterval`
# - `SpikeSortingPreprocessingParameters`
# - `LabTeam`
#

# ## Imports
#

# +
import os
import datajoint as dj

# change to the upper level folder to detect dj_local_conf.json
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
dj.config.load("dj_local_conf.json")  # load config for database connection info

import spyglass.common as sgc
import spyglass.spikesorting as sgs
import spyglass.decoding as sgd
import spyglass.utils as sgu

# ignore datajoint+jupyter async warnings
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
warnings.simplefilter("ignore", category=UserWarning)
# -

# ## Select Data

nwb_file_name = "J1620210531.nwb"
nwb_copy_file_name = sgu.nwb_helper_fn.get_nwb_copy_filename(nwb_file_name)

# ## Spike Sorting Selection

# We can investigate the `populate_mark_indicators` function we want to use with
# `?`:

# ?sgd.clusterless.populate_mark_indicators

# From the docstring, we see that we need to supply `spikesorting_selection_keys`,
# which is a list of `SpikeSorting` keys as dictionaries. We provide a list to
# extract marks for many electrode at once.
#
# Here are the primary keys required by `SpikeSorting`:

sgs.SpikeSorting.primary_key

# Here is an example of what `spikesorting_selection_keys` should look like:

spikesorting_selections = [
    {
        "nwb_file_name": "J1620210531_.nwb",
        "sort_group_id": 0,
        "sort_interval_name": "raw data valid times no premaze no home",
        "preproc_params_name": "franklab_tetrode_hippocampus",
        "team_name": "JG_DG",
        "sorter": "clusterless_thresholder",
        "sorter_params_name": "clusterless_fixed",
        "artifact_removed_interval_list_name": "J1620210531_.nwb_0_raw data valid times no premaze no home_franklab_tetrode_hippocampus_JG_DG_group_0.8_2000_8_1_artifact_removed_valid_times",
    },
    {
        "nwb_file_name": "J1620210531_.nwb",
        "sort_group_id": 1,
        "sort_interval_name": "raw data valid times no premaze no home",
        "preproc_params_name": "franklab_tetrode_hippocampus",
        "team_name": "JG_DG",
        "sorter": "clusterless_thresholder",
        "sorter_params_name": "clusterless_fixed",
        "artifact_removed_interval_list_name": "J1620210531_.nwb_1_raw data valid times no premaze no home_franklab_tetrode_hippocampus_JG_DG_group_0.8_2000_8_1_artifact_removed_valid_times",
    },
]

# _WARNING:_ This process relies on both `SpikeSortingSelection` and
# `IntervalPositionInfo`. You can check your database with the following:

sgs.SpikeSortingSelection & {
    "nwb_file_name": nwb_copy_file_name,
    "sorter": "clusterless_thresholder",
}

# Remember to replace the nwb_file_name with your own nwb file name

sgc.IntervalPositionInfo & {
    "nwb_file_name": nwb_copy_file_name,
    "position_info_param_name": "default_decoding",
}

# ## Populate Mark Indicators

# Now that we've checked those, we can run the function:

sgd.clusterless.populate_mark_indicators(spikesorting_selections)

# We can verify that this worked:

plt.plot(position_info.head_position_x, position_info.head_position_y)

position_info.shape, marks.shape

# +
from spyglass.common.common_interval import interval_list_intersect
from spyglass.common import IntervalList

key = {}
key["interval_list_name"] = "02_r1"
key["nwb_file_name"] = nwb_copy_file_name

interval = (
    IntervalList
    & {
        "nwb_file_name": key["nwb_file_name"],
        "interval_list_name": key["interval_list_name"],
    }
).fetch1("valid_times")

valid_ephys_times = (
    IntervalList
    & {
        "nwb_file_name": key["nwb_file_name"],
        "interval_list_name": "raw data valid times",
    }
).fetch1("valid_times")
position_interval_names = (
    IntervalPositionInfo
    & {
        "nwb_file_name": key["nwb_file_name"],
        "position_info_param_name": "default_decoding",
    }
).fetch("interval_list_name")
valid_pos_times = [
    (
        IntervalList
        & {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": pos_interval_name,
        }
    ).fetch1("valid_times")
    for pos_interval_name in position_interval_names
]

intersect_interval = interval_list_intersect(
    interval_list_intersect(interval, valid_ephys_times), valid_pos_times[0]
)
valid_time_slice = slice(intersect_interval[0][0], intersect_interval[0][1])
valid_time_slice

# +
from replay_trajectory_classification import ClusterlessClassifier
from replay_trajectory_classification.environments import Environment
from replay_trajectory_classification.continuous_state_transitions import (
    RandomWalk,
    Uniform,
)
from spyglass.decoding.clusterless import ClusterlessClassifierParameters

marks = marks.sel(time=valid_time_slice)
position_info = position_info.loc[valid_time_slice]

parameters = (
    ClusterlessClassifierParameters()
    & {"classifier_param_name": "default_decoding_gpu"}
).fetch1()

parameters["classifier_params"]["clusterless_algorithm_params"] = {
    "mark_std": 24.0,
    "position_std": 3.0,
    "block_size": int(2**13),
    "disable_progress_bar": False,
    "use_diffusion": False,
}
parameters["classifier_params"]["environments"][0] = Environment(
    place_bin_size=3.0
)


import cupy as cp

with cp.cuda.Device(0):
    classifier = ClusterlessClassifier(**parameters["classifier_params"])
    classifier.fit(
        position=position_info[["head_position_x", "head_position_y"]].values,
        multiunits=marks.values,
        **parameters["fit_params"],
    )
    results = classifier.predict(
        multiunits=marks.values,
        time=position_info.index,
        **parameters["predict_params"],
    )
    logging.info("Done!")

# +
from spyglass.decoding.visualization import (
    create_interactive_2D_decoding_figurl,
)


view = create_interactive_2D_decoding_figurl(
    position_info,
    marks,
    results,
    classifier.environments[0].place_bin_size,
    position_name=["head_position_x", "head_position_y"],
    head_direction_name="head_orientation",
    speed_name="head_speed",
    posterior_type="acausal_posterior",
    sampling_frequency=500,
    view_height=800,
)
# -

# ## Up Next
#
# Next, we'll start the process of decoding representations of position with
# ephys data. This can be done either with [GPUs](./32_Decoding_with_GPUs.ipynb)
# or [clusterless](./33_Decoding_Clusterless.ipynb).
