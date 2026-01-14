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
#     name: python3
# ---

# # Clusterless Decoding
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
# - This tutorial assumes you've already
#   [extracted waveforms](./41_Extracting_Clusterless_Waveform_Features.ipynb), as well as loaded
#   [position data](./20_Position_Trodes.ipynb). If 1D decoding, this data should also be
#   [linearized](./24_Linearization.ipynb).
#
# Clusterless decoding can be performed on either 1D or 2D data. We will start with 2D data.
#
# ## Elements of Clusterless Decoding
# - **Position Data**: This is the data that we want to decode. It can be 1D or 2D.
# - **Spike Waveform Features**: These are the features that we will use to decode the position data.
# - **Decoding Model Parameters**: This is how we define the model that we will use to decode the position data.
#
# ## Grouping Data
# An important concept will be groups. Groups are tables that allow use to specify collections of data. We will use groups in two situations here:
# 1. Because we want to decode from more than one tetrode (or probe), so we will create a group that contains all of the tetrodes that we want to decode from.
# 2. Similarly, we will create a group for the position data that we want to decode, so that we can decode from position data from multiple sessions.
#
# ### Grouping Waveform Features
# Let's start with grouping the Waveform Features. We will first inspect the waveform features that we have extracted to figure out the primary keys of the data that we want to decode from. We need to use the tables `SpikeSortingSelection` and `SpikeSortingOutput` to figure out the `merge_id` associated with `nwb_file_name` to get the waveform features associated with the NWB file of interest.
#

# +
from pathlib import Path
import datajoint as dj

dj.config.load(
    Path("../dj_local_conf.json").absolute()
)  # load config for database connection info

# +
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
import spyglass.spikesorting.v1 as sgs
from spyglass.decoding.v1.waveform_features import (
    UnitWaveformFeaturesSelection,
    UnitWaveformFeatures,
)


nwb_copy_file_name = "mediumnwb20230802_.nwb"

sorter_keys = {
    "nwb_file_name": nwb_copy_file_name,
    "sorter": "clusterless_thresholder",
    "sorter_param_name": "default_clusterless",
}

feature_key = {"features_param_name": "amplitude"}

(
    UnitWaveformFeaturesSelection.proj(merge_id="spikesorting_merge_id")
    * SpikeSortingOutput.CurationV1
    * sgs.SpikeSortingSelection
) & SpikeSortingOutput().get_restricted_merge_ids(
    sorter_keys, sources=["v1"], as_dict=True
)

# +
from spyglass.decoding.v1.waveform_features import UnitWaveformFeaturesSelection

# find the merge ids that correspond to the sorter key restrictions
merge_ids = SpikeSortingOutput().get_restricted_merge_ids(
    sorter_keys, sources=["v1"], as_dict=True
)

# find the previously populated waveform selection keys that correspond to these sorts
waveform_selection_keys = (
    UnitWaveformFeaturesSelection().proj(merge_id="spikesorting_merge_id")
    & merge_ids
    & feature_key
).fetch(as_dict=True)
for key in waveform_selection_keys:
    key["spikesorting_merge_id"] = key.pop("merge_id")

UnitWaveformFeaturesSelection & waveform_selection_keys
# -

# We will create a group called `test_group` that contains all of the tetrodes that we want to decode from. We will use the `create_group` function to create this group. This function takes two arguments: the name of the group, and the keys of the tables that we want to include in the group.

# +
from spyglass.decoding.v1.clusterless import UnitWaveformFeaturesGroup

UnitWaveformFeaturesGroup().create_group(
    nwb_file_name=nwb_copy_file_name,
    group_name="test_group",
    keys=waveform_selection_keys,
)
UnitWaveformFeaturesGroup & {"waveform_features_group_name": "test_group"}
# -

# We can see that we successfully associated "test_group" with the tetrodes that we want to decode from by using the `get_group` function.

UnitWaveformFeaturesGroup.UnitFeatures & {
    "nwb_file_name": nwb_copy_file_name,
    "waveform_features_group_name": "test_group",
}

# ### Grouping Position Data
#
# We will now create a group called `02_r1` that contains all of the position data that we want to decode from. As before, we will use the `create_group` function to create this group. This function takes two arguments: the name of the group, and the keys of the tables that we want to include in the group.
#
# We use the the `PositionOutput` table to figure out the `merge_id` associated with `nwb_file_name` to get the position data associated with the NWB file of interest. In this case, we only have one position to insert, but we could insert multiple positions if we wanted to decode from multiple sessions.
#
# Note that we can use the `upsample_rate` parameter to define the rate to which position data will be upsampled to to for decoding in Hz. This is useful if we want to decode at a finer time scale than the position data sampling frequency. In practice, a value of 500Hz is used in many analyses. Skipping or providing a null value for this parameter will default to using the position sampling rate.
#
# You will also want to specify the name of the position variables if they are different from the default names. The default names are `position_x` and `position_y`.

# +
from spyglass.position import PositionOutput
import spyglass.position as sgp


sgp.v1.TrodesPosParams.insert1(
    {
        "trodes_pos_params_name": "default_decoding",
        "params": {
            "max_LED_separation": 9.0,
            "max_plausible_speed": 300.0,
            "position_smoothing_duration": 0.125,
            "speed_smoothing_std_dev": 0.100,
            "orient_smoothing_std_dev": 0.001,
            "led1_is_front": 1,
            "is_upsampled": 1,
            "upsampling_sampling_rate": 250,
            "upsampling_interpolation_method": "linear",
        },
    },
    skip_duplicates=True,
)

trodes_s_key = {
    "nwb_file_name": nwb_copy_file_name,
    "interval_list_name": "pos 0 valid times",
    "trodes_pos_params_name": "default_decoding",
}
sgp.v1.TrodesPosSelection.insert1(
    trodes_s_key,
    skip_duplicates=True,
)
sgp.v1.TrodesPosV1.populate(trodes_s_key)

PositionOutput.TrodesPosV1 & trodes_s_key

# +
from spyglass.decoding.v1.core import PositionGroup

position_merge_ids = (
    PositionOutput.TrodesPosV1
    & {
        "nwb_file_name": nwb_copy_file_name,
        "interval_list_name": "pos 0 valid times",
        "trodes_pos_params_name": "default_decoding",
    }
).fetch("merge_id")

PositionGroup().create_group(
    nwb_file_name=nwb_copy_file_name,
    group_name="test_group",
    keys=[{"pos_merge_id": merge_id} for merge_id in position_merge_ids],
    upsample_rate=500,
)

PositionGroup & {
    "nwb_file_name": nwb_copy_file_name,
    "position_group_name": "test_group",
}
# -

(
    PositionGroup
    & {"nwb_file_name": nwb_copy_file_name, "position_group_name": "test_group"}
).fetch1("position_variables")

PositionGroup.Position & {
    "nwb_file_name": nwb_copy_file_name,
    "position_group_name": "test_group",
}

# ## Decoding Model Parameters
#
# We will use the `non_local_detector` package to decode the data. This package is highly flexible and allows several different types of models to be used. In this case, we will use the `ContFragClusterlessClassifier` to decode the data. This has two discrete states: Continuous and Fragmented, which correspond to different types of movement models. To read more about this model, see:
# > Denovellis, E.L., Gillespie, A.K., Coulter, M.E., Sosa, M., Chung, J.E., Eden, U.T., and Frank, L.M. (2021). Hippocampal replay of experience at real-world speeds. eLife 10, e64505. [10.7554/eLife.64505](https://doi.org/10.7554/eLife.64505).
#
# Let's first look at the model and the default parameters:
#

# +
from non_local_detector.models import ContFragClusterlessClassifier

ContFragClusterlessClassifier()
# -

# You can change these parameters like so:

# +
from non_local_detector.models import ContFragClusterlessClassifier

ContFragClusterlessClassifier(
    clusterless_algorithm_params={
        "block_size": 10000,
        "position_std": 12.0,
        "waveform_std": 24.0,
    },
)
# -

# This is how to insert the model parameters into the database:

# +
from spyglass.decoding.v1.core import DecodingParameters


DecodingParameters.insert1(
    {
        "decoding_param_name": "contfrag_clusterless",
        "decoding_params": ContFragClusterlessClassifier(),
        "decoding_kwargs": dict(),
    },
    skip_duplicates=True,
)

DecodingParameters & {"decoding_param_name": "contfrag_clusterless"}
# -

# We can retrieve these parameters and rebuild the model like so:

# +
model_params = (
    DecodingParameters & {"decoding_param_name": "contfrag_clusterless"}
).fetch1()

ContFragClusterlessClassifier(**model_params["decoding_params"])
# -

# ### 1D Decoding
#
# If you want to do 1D decoding, you will need to specify the `track_graph`, `edge_order`, and `edge_spacing` in the `environments` parameter. You can read more about these parameters in the [linearization notebook](./24_Linearization.ipynb). You can retrieve these parameters from the `TrackGraph` table if you have stored them there. These will then go into the `environments` parameter of the `ContFragClusterlessClassifier` model.

# +
from non_local_detector.environment import Environment

# ?Environment
# -

# ## Decoding
#
# Now that we have grouped the data and defined the model parameters, we have finally set up the elements in tables that we need to decode the data. We now need to use the `ClusterlessDecodingSelection` to fully specify all the parameters and data that we want.
#
# This has:
# - `waveform_features_group_name`: the name of the group that contains the waveform features that we want to decode from
# - `position_group_name`: the name of the group that contains the position data that we want to decode from
# - `decoding_param_name`: the name of the decoding parameters that we want to use
# - `nwb_file_name`: the name of the NWB file that we want to decode from
# - `encoding_interval`: the interval of time that we want to train the initial model on
# - `decoding_interval`: the interval of time that we want to decode from
# - `estimate_decoding_params`: whether or not we want to estimate the decoding parameters
#
#
# The first three parameters should be familiar to you.
#
#
# ### Decoding and Encoding Intervals
# The `encoding_interval` is the interval of time that we want to train the initial model on. The `decoding_interval` is the interval of time that we want to decode from. These two intervals can be the same, but they do not have to be. For example, we may want to train the model on a long interval of time, but only decode from a short interval of time. This is useful if we want to decode from a short interval of time that is not representative of the entire session. In this case, we will train the model on a longer interval of time that is representative of the entire session.
#
# These keys come from the `IntervalList` table. We can see that the `IntervalList` table contains the `nwb_file_name` and `interval_name` that we need to specify the `encoding_interval` and `decoding_interval`. We will specify a short decoding interval called `test decoding interval` and use that to decode from.
#
#
# ### Estimating Decoding Parameters
# The last parameter is `estimate_decoding_params`. This is a boolean that specifies whether or not we want to estimate the decoding parameters. If this is `True`, then we will estimate the initial conditions and discrete transition matrix from the data.
#
# NOTE: If estimating parameters, then we need to treat times outside decoding interval as missing. this means that times outside the decoding interval will not use the spiking data and only the state transition matrix and previous time step will be used. This may or may not be desired depending on the length of this missing interval.
#

# +
from spyglass.decoding.v1.clusterless import ClusterlessDecodingSelection

ClusterlessDecodingSelection()

# +
from spyglass.common import IntervalList

IntervalList & {"nwb_file_name": nwb_copy_file_name}

# +
decoding_interval_valid_times = [
    [1625935714.6359036, 1625935714.6359036 + 15.0]
]

IntervalList.insert1(
    {
        "nwb_file_name": "mediumnwb20230802_.nwb",
        "interval_list_name": "test decoding interval",
        "valid_times": decoding_interval_valid_times,
    },
    skip_duplicates=True,
)
# -

# Once we have figured out the keys that we need, we can insert the `ClusterlessDecodingSelection` into the database.

# +
selection_key = {
    "waveform_features_group_name": "test_group",
    "position_group_name": "test_group",
    "decoding_param_name": "contfrag_clusterless",
    "nwb_file_name": nwb_copy_file_name,
    "encoding_interval": "pos 0 valid times",
    "decoding_interval": "test decoding interval",
    "estimate_decoding_params": False,
}

ClusterlessDecodingSelection.insert1(
    selection_key,
    skip_duplicates=True,
)

ClusterlessDecodingSelection & selection_key
# -

ClusterlessDecodingSelection()

# To run decoding, we simply populate the `ClusterlessDecodingOutput` table. This will run the decoding and insert the results into the database. We can then retrieve the results from the database.

# +
from spyglass.decoding.v1.clusterless import ClusterlessDecodingV1

ClusterlessDecodingV1.populate(selection_key)
# -

# We can now see it as an entry in the `DecodingOutput` table.

# +
from spyglass.decoding.decoding_merge import DecodingOutput

DecodingOutput.ClusterlessDecodingV1 & selection_key
# -

# We can load the results of the decoding:

decoding_results = (ClusterlessDecodingV1 & selection_key).fetch_results()
decoding_results

# Finally, if we deleted the results, we can use the `cleanup` function to delete the results from the file system:

DecodingOutput().cleanup()

# ## Visualization of decoding output.
#
# The output of decoding can be challenging to visualize with static graphs, especially if the decoding is performed on 2D data.
#
# We can interactively visualize the output of decoding using the [figurl](https://github.com/flatironinstitute/figurl) package. This package allows to create a visualization of the decoding output that can be viewed in a web browser. This is useful for exploring the decoding output over time and sharing the results with others.
#
# **NOTE**: You will need a kachery cloud instance to use this feature. If you are a member of the Frank lab, you should have access to the Frank lab kachery cloud instance. If you are not a member of the Frank lab, you can create your own kachery cloud instance by following the instructions [here](https://github.com/flatironinstitute/kachery-cloud/blob/main/doc/create_kachery_zone.md).
#
# For each user, you will need to run `kachery-cloud-init` in the terminal and follow the instructions to associate your computer with your GitHub user on the kachery-cloud network.
#

# +
# from non_local_detector.visualization import (
#     create_interactive_2D_decoding_figurl,
# )

# (
#     position_info,
#     position_variable_names,
# ) = ClusterlessDecodingV1.fetch_position_info(selection_key)
# results_time = decoding_results.acausal_posterior.isel(intervals=0).time.values
# position_info = position_info.loc[results_time[0] : results_time[-1]]

# env = ClusterlessDecodingV1.fetch_environments(selection_key)[0]
# spike_times, _ = ClusterlessDecodingV1.fetch_spike_data(selection_key)


# create_interactive_2D_decoding_figurl(
#     position_time=position_info.index.to_numpy(),
#     position=position_info[position_variable_names],
#     env=env,
#     results=decoding_results,
#     posterior=decoding_results.acausal_posterior.isel(intervals=0)
#     .unstack("state_bins")
#     .sum("state"),
#     spike_times=spike_times,
#     head_dir=position_info["orientation"],
#     speed=position_info["speed"],
# )
# -

# ## GPUs
# We can use GPUs for decoding which will result in a significant speedup. This is achieved using the [jax](https://jax.readthedocs.io/en/latest/) package.
#
# ### Ensuring jax can find a GPU
#  Assuming you've set up a GPU, we can use `jax.devices()` to make sure the decoding code can see the GPU. If a GPU is available, it will be listed.
#
# In the following instance, we do not have a GPU:

# +
import jax

jax.devices()
# -

# ### Selecting a GPU
# If you do have multiple GPUs, you can use the `jax` package to set the device (GPU) that you want to use. For example, if you want to use the second GPU, you can use the following code (uncomment first):

# +
# device_id = 2
# device = jax.devices()[device_id]
# jax.config.update("jax_default_device", device)
# device
# -

# ### Monitoring GPU Usage
#
# You can see which GPUs are occupied (if you have multiple GPUs) by running the command `nvidia-smi` in
# a terminal (or `!nvidia-smi` in a notebook). Pick a GPU with low memory usage.
#
# We can monitor GPU use with the terminal command `watch -n 0.1 nvidia-smi`, will
# update `nvidia-smi` every 100 ms. This won't work in a notebook, as it won't
# display the updates.
#
# Other ways to monitor GPU usage are:
#
# - A
#   [jupyter widget by nvidia](https://github.com/rapidsai/jupyterlab-nvdashboard)
#   to monitor GPU usage in the notebook
# - A [terminal program](https://github.com/peci1/nvidia-htop) like nvidia-smi
#   with more information about  which GPUs are being utilized and by whom.
