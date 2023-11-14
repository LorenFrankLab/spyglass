# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: spyglass
#     language: python
#     name: python3
# ---

# # Clusterless Decoding

# ## Overview
#

# _Developer Note:_ if you may make a PR in the future, be sure to copy this
# notebook, and use the `gitignore` prefix `temp` to avoid future conflicts.
#
# This is one notebook in a multi-part series on Spyglass.
#
# - To set up your Spyglass environment and database, see
#   [the Setup notebook](./00_Setup.ipynb)
# - This tutorial assumes you've already
#   [extracted marks](./31_Extract_Mark_Indicators.ipynb), as well as loaded
#   position data. If 1D decoding, this data should also be
#   [linearized](./24_Linearization.ipynb).
# - This tutorial also assumes you're familiar with how to run processes on GPU,
#   as presented in [this notebook](./32_Decoding_with_GPUs.ipynb)
#
# Clusterless decoding can be performed on either 1D or 2D data. A few steps in
# this notebook will refer to a `decode_1d` variable set in
# [select data](#select-data) to include these steps.
#

# ## Imports
#

# %reload_ext autoreload
# %autoreload 2

# +
import os
import datajoint as dj
import matplotlib.pyplot as plt
import logging
import cupy as cp

from pprint import pprint


logging.basicConfig(
    level="INFO", format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)

# change to the upper level folder to detect dj_local_conf.json
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
dj.config.load("dj_local_conf.json")  # load config for database connection info

import spyglass.common as sgc
import spyglass.common.common_position as sgc_pos
import spyglass.common.interval as sgc_int
import spyglass.decoding.clusterless as sgd_clusterless
import spyglass.decoding.visualization as sgd_viz
import replay_trajectory_classification as rtc
import replay_trajectory_classification.environments as rtc_env


# ignore datajoint+jupyter async warnings
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
# -

# ## Select data

nwb_copy_file_name = "chimi20200216_new_.nwb"
decode_1d = True

# ## `UnitMarksIndicator`
#

# First, we'll fetch marks with `fetch_xarray`, which provides a labeled array of
# shape (n_time, n_mark_features, n_electrodes). Time is in 2 ms bins with either
# `NaN` if no spike occurred or the value of the spike features.
#
# If there is >1 spike per time bin per tetrode, we take an an average of the
# marks. Ideally, we would use all the marks, this is a rare occurrence and
# decoding is generally robust to the averaging.

# +
marks = (
    sgd_clusterless.UnitMarksIndicator
    & {
        "nwb_file_name": nwb_copy_file_name,
        "sort_interval_name": "runs_noPrePostTrialTimes raw data valid times",
        "filter_parameter_set_name": "franklab_default_hippocampus",
        "unit_inclusion_param_name": "all2",
        "mark_param_name": "default",
        "interval_list_name": "pos 1 valid times",
        "sampling_rate": 500,
    }
).fetch_xarray()

marks
# -

# We'll use `UnitMarksIndicator.plot_all_marks` to make sure our marks look right.
# This will plot each mark feature against the other for each electrode. We check
# for items that look overly correlated (strong diagonal on the off-diagonal
# plots) and extreme amplitudes.
#
# For tutorial purposes, we only look at the first 2 plots, but removing this
# argument will show all plots.

sgd_clusterless.UnitMarksIndicator.plot_all_marks(marks, plot_limit=2)

# ## Position
#

# ### Get position

# Next, we'll grab the 2D position data from `IntervalPositionInfo` table.
#
# _Note:_ Position will need to be upsampled to our decoding frequency (500 Hz).
# See [this notebook](./20_Position_Trodes.ipynb#upsampling-position) for more
# information.

# +
position_key = {
    "nwb_file_name": nwb_copy_file_name,
    "interval_list_name": "pos 1 valid times",
    "position_info_param_name": "default_decoding",
}

position_info = (
    sgc_pos.IntervalPositionInfo() & position_key
).fetch1_dataframe()

position_info
# -

# ### Plot position

# It is important to visualize the 2D position and identify outliers.

plt.figure(figsize=(7, 6))
plt.plot(position_info.head_position_x, position_info.head_position_y)

# For 1D decoding, we load the linearized position tables.

if decode_1d:
    linearization_key = {
        "position_info_param_name": "default_decoding",
        "nwb_file_name": nwb_copy_file_name,
        "interval_list_name": "pos 1 valid times",
        "track_graph_name": "6 arm",
        "linearization_param_name": "default",
    }

    linear_position_df = (
        sgc_pos.IntervalLinearizedPosition() & linearization_key
    ).fetch1_dataframe()

    linear_position_df
else:
    linear_position_df = position_info

# We'll also sanity check linearized values by plotting the 2D position projected
# to its corresponding 1D segment.

if decode_1d:
    plt.figure(figsize=(7, 6))
    plt.scatter(
        linear_position_df.projected_x_position,
        linear_position_df.projected_y_position,
        c=linear_position_df.track_segment_id,
        cmap="tab20",
        s=1,
    )

# And then the linearized position itself:

if decode_1d:
    plt.figure(figsize=(20, 10))
    plt.scatter(
        linear_position_df.index,
        linear_position_df.linear_position,
        s=1,
        c=linear_position_df.track_segment_id,
        cmap="tab20",
    )

# And then, we'll verify that all our data is the same size. It may not be due to
# the valid intervals of the neural and position data.

position_info.shape, marks.shape, linear_position_df.shape

# ## Validate data

# We'll also validate the ephys and position data for decoding. If we had more
# than one time interval, we would decode on each separately.

# +
key = {}
key["interval_list_name"] = "02_r1"
key["nwb_file_name"] = nwb_copy_file_name

interval = (
    sgc.IntervalList
    & {
        "nwb_file_name": key["nwb_file_name"],
        "interval_list_name": key["interval_list_name"],
    }
).fetch1("valid_times")

valid_ephys_times = (
    sgc.IntervalList
    & {
        "nwb_file_name": key["nwb_file_name"],
        "interval_list_name": "raw data valid times",
    }
).fetch1("valid_times")
position_interval_names = (
    sgc_pos.IntervalPositionInfo
    & {
        "nwb_file_name": key["nwb_file_name"],
        "position_info_param_name": "default_decoding",
    }
).fetch("interval_list_name")
valid_pos_times = [
    (
        sgc.IntervalList
        & {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": pos_interval_name,
        }
    ).fetch1("valid_times")
    for pos_interval_name in position_interval_names
]

intersect_interval = sgc_int.interval_list_intersect(
    sgc_int.interval_list_intersect(interval, valid_ephys_times),
    valid_pos_times[0],
)
valid_time_slice = slice(intersect_interval[0][0], intersect_interval[0][1])
valid_time_slice
# -

linear_position_df = linear_position_df.loc[valid_time_slice]
marks = marks.sel(time=valid_time_slice)
position_info = position_info.loc[valid_time_slice]

position_info.shape, marks.shape, linear_position_df.shape

# ## Decoding
#

# After sanity checks, we can finally get to decoding.
#
# _Note:_ Portions of the code below have been integrated into
# `spyglass.decoding`, but are presented here in full.
#
# We'll fetch the default parameters and modify them. For 1D decoding, we'll also pass the track graph and parameters from linearization to handle the random walk properly. `position_std` and `mark_std` set the amount of smoothing in the position and mark dimensions. `block_size` controls how many samples get processed at a time so that we don't run out of GPU memory.

# +
parameters = (
    sgd_clusterless.ClusterlessClassifierParameters()
    & {"classifier_param_name": "default_decoding_gpu"}
).fetch1()

algorithm_params = (
    {
        "mark_std": 24.0,
        "position_std": 6.0,
        "block_size": 2**12,  # in merging nbs, changed from 2**13 for 2d
    },
)

if decode_1d:
    track = sgc_pos.TrackGraph() & {"track_graph_name": "6 arm"}
    track_graph = track.get_networkx_track_graph()
    track_graph_params = track.fetch1()

    parameters["classifier_params"] = {
        "environments": [
            rtc_env.Environment(
                track_graph=track_graph,
                edge_order=track_graph_params["linear_edge_order"],
                edge_spacing=track_graph_params["linear_edge_spacing"],
            )
        ],
        "clusterless_algorithm": "multiunit_likelihood_integer_gpu",
        "clusterless_algorithm_params": algorithm_params,
    }
else:
    parameters["classifier_params"] = {
        "environments": [rtc_env.Environment(place_bin_size=3.0)],
        "clusterless_algorithm_params": {
            **algorithm_params,
            "disable_progress_bar": False,
            "use_diffusion": False,
        },
    }

pprint(parameters)
# -

# To run decoding on the first GPU device, we use `cp.dua.Device(0)`.
# For more information, see [this notebook](./32_Decoding_with_GPUs.ipynb).

# +
if decode_1d:
    position = linear_position_df.linear_position.values
    time = linear_position_df.index
else:
    position = position_info[["head_position_x", "head_position_y"]].values
    time = position_info.index

with cp.cuda.Device(0):
    classifier = rtc.ClusterlessClassifier(**parameters["classifier_params"])
    classifier.fit(
        position=position,
        multiunits=marks.values,
        **parameters["fit_params"],
    )
    results = classifier.predict(
        multiunits=marks.values,
        time=time,
        **parameters["predict_params"],
    )
    logging.info("Done!")
# -

# ## Visualization
#

# Finally, we can sanity check plot the decodes in an interactive figure with `create_interactive_1D_decoding_figurl`, which will return a URL
#
# _Note:_ For this figure, that you need to be running an interactive sorting view backend.
#
# <!-- What is that? -->

# +
df = linear_position_df if decode_1d else position_info

view = sgd_viz.create_interactive_1D_decoding_figurl(
    position_info,
    linear_position_df,
    marks,
    results,
    position_name="linear_position",
    speed_name="head_speed",
    posterior_type="acausal_posterior",
    sampling_frequency=500,
    view_height=800,
)
# -

# To view the decode in the notebook, simply run `view` in a cell.
#
# To create shareable visualization in the cloud, call `url`

dim = "1D" if decode_1d else "2D"
view.url(label=f"{dim} Decoding Example")
