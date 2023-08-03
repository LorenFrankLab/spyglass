# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3.10.5 64-bit
#     language: python
#     name: python3
# ---

# # Position - Linearization

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
# This pipeline takes 2D position data from the `IntervalPositionInfo` table and
# "linearizes" it to 1D position. If you haven't already done so, please generate
# input data with either the [Trodes](./20_Position_Trodes.ipynb) or DLC notebooks
# ([1](./21_Position_DLC_1.ipynb), [2](./22_Position_DLC_2.ipynb),
# [3](./23_Position_DLC_3.ipynb)).

# ## Imports
#

# %reload_ext autoreload
# %autoreload 2

# +
import os
import pynwb
import numpy as np
import datajoint as dj
import matplotlib.pyplot as plt

# change to the upper level folder to detect dj_local_conf.json
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
dj.config.load("dj_local_conf.json")  # load config for database connection info

import spyglass.common as sgc
import spyglass.position.v1 as sgp
import spyglass as nd

# ignore datajoint+jupyter async warnings
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
# -

# ## Retrieve 2D position
#

# To retrieve 2D position data, we'll specify an nwb file, a position time
# interval, and the set of parameters used to compute the position info.

nwb_file_name = "chimi20200216_new.nwb"
nwb_copy_file_name = sgc.nwb_helper_fn.get_nwb_copy_filename(nwb_file_name)
nwb_copy_file_name

# We will fetch the pandas dataframe from the `IntervalPositionInfo`.

position_info = (
    sgc.common_position.IntervalPositionInfo()
    & {
        "nwb_file_name": nwb_copy_file_name,
        "interval_list_name": "pos 1 valid times",
        "position_info_param_name": "default",
    }
).fetch1_dataframe()
position_info

# Before linearizing, plotting the head position will help us understand the data.

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(
    position_info.head_position_x,
    position_info.head_position_y,
    color="lightgrey",
)
ax.set_xlabel("x-position [cm]", fontsize=18)
ax.set_ylabel("y-position [cm]", fontsize=18)
ax.set_title("Head Position", fontsize=28)

# ## Specifying the track
#

# To linearize, we need a graph of nodes and edges to represent track geometry in
# the `TrackGraph` table with four variables:
#
# - `node_positions` (cm): the 2D positions of the graph
# - `edges`: how the nodes are connected, as pairs of node indices, labeled by
#   their respective index in `node_positions`.
# - `linear_edge_order`: layout of edges in linear space in *order*, as tuples.
# - `linear_edge_spacing`: spacing between each edge, as either a single number
# for all gaps or an array with a number for each gap. Gaps may be important for
# edges not connected in 2D space.
#
# For example, (79.910, 216.720) is the 2D position of node 0 and (183.784,
# 45.375) is the 2D position of node 8. Edge (0, 8) means there is an edge between
# node 0 and node 8. Nodes order controls order in 1D space. Edge (0, 1) connects
# from node 0 to 1. Edge (1, 0) would connect from node 1 to 0, reversing the
# linear positions for that edge.
#
# For more examples, see
# [this notebook](https://github.com/LorenFrankLab/track_linearization/blob/master/notebooks/).

# +
node_positions = np.array(
    [
        (79.910, 216.720),  # top left well 0
        (132.031, 187.806),  # top middle intersection 1
        (183.718, 217.713),  # top right well 2
        (132.544, 132.158),  # middle intersection 3
        (87.202, 101.397),  # bottom left intersection 4
        (31.340, 126.110),  # middle left well 5
        (180.337, 104.799),  # middle right intersection 6
        (92.693, 42.345),  # bottom left well 7
        (183.784, 45.375),  # bottom right well 8
        (231.338, 136.281),  # middle right well 9
    ]
)

edges = np.array(
    [
        (0, 1),
        (1, 2),
        (1, 3),
        (3, 4),
        (4, 5),
        (3, 6),
        (6, 9),
        (4, 7),
        (6, 8),
    ]
)

linear_edge_order = [
    (3, 6),
    (6, 8),
    (6, 9),
    (3, 1),
    (1, 2),
    (1, 0),
    (3, 4),
    (4, 5),
    (4, 7),
]
linear_edge_spacing = 15
# -

# With these variables, we then add a `track_graph_name` and the corresponding
# `environment`.

# +
sgc.common_position.TrackGraph.insert1(
    {
        "track_graph_name": "6 arm",
        "environment": "6 arm",
        "node_positions": node_positions,
        "edges": edges,
        "linear_edge_order": linear_edge_order,
        "linear_edge_spacing": linear_edge_spacing,
    },
    skip_duplicates=True,
)

graph = sgc.common_position.TrackGraph & {"track_graph_name": "6 arm"}
graph
# -

# `TrackGraph` has several methods for visualizing in 2D and 1D space.
# `plot_track_graph` plots in 2D to make sure our layout makes sense.

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(
    position_info.head_position_x,
    position_info.head_position_y,
    color="lightgrey",
    alpha=0.7,
    zorder=0,
)
ax.set_xlabel("x-position [cm]", fontsize=18)
ax.set_ylabel("y-position [cm]", fontsize=18)
graph.plot_track_graph(ax=ax)

# `plot_track_graph_as_1D` shows what this looks like in 1D.

fig, ax = plt.subplots(1, 1, figsize=(20, 1))
graph.plot_track_graph_as_1D(ax=ax)

# ## Parameters
#

#
# By default, linearization assigns each 2D position to its nearest point on the
# track graph. This is then translated into 1D space.
#
# If `use_hmm` is set to `true`, a Hidden Markov model is used to assign points.
# The HMM takes into account the prior position and edge, and can keep the
# position from suddenly jumping to another. Position jumping like this may occur
# at intersections or the head position swings closer to a non-target reward well
# while on another edge.

sgc.common_position.LinearizationParameters.insert1(
    {"linearization_param_name": "default"}, skip_duplicates=True
)
sgc.common_position.LinearizationParameters()

# ## Linearization

# With linearization parameters, we specify the position interval we wish to
# linearize.

# +
sgc.common_position.IntervalLinearizationSelection.insert1(
    {
        "position_info_param_name": "default",
        "nwb_file_name": nwb_copy_file_name,
        "interval_list_name": "pos 1 valid times",
        "track_graph_name": "6 arm",
        "linearization_param_name": "default",
    },
    skip_duplicates=True,
)

sgc.common_position.IntervalLinearizationSelection()
# -

# And then run linearization by populating `IntervalLinearizedPosition`.

sgc.common_position.IntervalLinearizedPosition().populate()
sgc.common_position.IntervalLinearizedPosition()

# ## Examine data
#

# Running `fetch1_dataframe` will retrieve the linear position data, including...
#
# - `time`: datafame index
# - `linear_position`: 1D linearized position
# - `track_segment_id`: index number of the edges given to track graph
# - `projected_{x,y}_position`: 2D position projected to the track graph

linear_position_df = (
    IntervalLinearizedPosition()
    & {
        "position_info_param_name": "default",
        "nwb_file_name": nwb_copy_file_name,
        "interval_list_name": "pos 1 valid times",
        "track_graph_name": "6 arm",
        "linearization_param_name": "default",
    }
).fetch1_dataframe()
linear_position_df

# We'll plot the 1D position over time, colored by edge, and use the 1D track
# graph layout on the y-axis.

# +
fig, ax = plt.subplots(figsize=(20, 13))
ax.scatter(
    linear_position_df.index,
    linear_position_df.linear_position,
    c=linear_position_df.track_segment_id,
    s=1,
)
graph.plot_track_graph_as_1D(
    ax=ax, axis="y", other_axis_start=linear_position_df.index[-1] + 10
)

ax.set_xlabel("Time [s]", fontsize=18)
ax.set_ylabel("Linear Position [cm]", fontsize=18)
ax.set_title("Linear Position", fontsize=28)
# -

# We can also plot the 2D position projected to the track graph

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(
    position_info.head_position_x,
    position_info.head_position_y,
    color="lightgrey",
    alpha=0.7,
    zorder=0,
)
ax.set_xlabel("x-position [cm]", fontsize=18)
ax.set_ylabel("y-position [cm]", fontsize=18)
ax.plot(
    linear_position_df.projected_x_position,
    linear_position_df.projected_y_position,
)

# ## Interactive selection
#

#
# _Note:_ Work in Progress
#
# ### `NodePicker`
#
# Linearization heavily depends on the track graph is specified. To help simplify
# setting the nodes/edges, we can use the `NodePicker` to interactively set node
# positions and edges based on the video.

# +
# %matplotlib widget

key = {
    "nwb_file_name": nwb_copy_file_name,
    "interval_list_name": "pos 1 valid times",
}

epoch = (
    int(
        key["interval_list_name"]
        .replace("pos ", "")
        .replace(" valid times", "")
    )
    + 1
)
video_info = (
    nd.common.common_behav.VideoFile()
    & {"nwb_file_name": key["nwb_file_name"], "epoch": epoch}
).fetch1()

io = pynwb.NWBHDF5IO("/stelmo/nwb/raw/" + video_info["nwb_file_name"], "r")
nwb_file = io.read()
nwb_video = nwb_file.objects[video_info["video_file_object_id"]]
video_filename = nwb_video.external_file.value[0]

fig, ax = plt.subplots(figsize=(8, 8))
picker = sgc.common_position.NodePicker(ax=ax, video_filename=video_filename)
# -

# After selection, we can retrieve the data using the `node_positions` and `edges`
# attributes

picker.node_positions

picker.edges

# ### Selector
#
# We can also draw a 2d polygon around the track and attempt to recover the graph.

# +
# %matplotlib widget

fig, ax = plt.subplots(figsize=(8, 8))
selector = sgc.common_position.SelectFromCollection(ax, video_filename)

print("Select points in the figure by enclosing them within a polygon.")
print("Press the 'esc' key to start a new polygon.")
print("Try holding the 'shift' key to move all of the vertices.")
print("Try holding the 'ctrl' key to move a single vertex.")
