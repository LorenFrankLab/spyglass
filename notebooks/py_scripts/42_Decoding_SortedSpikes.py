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

# # Sorted Spikes Decoding
#
# The mechanics of decoding with sorted spikes are largely similar to those of decoding with unsorted spikes. You should familiarize yourself with the [clusterless decoding tutorial](./41_Decoding_Clusterless.ipynb) before proceeding with this one.
#
# The elements we will need to decode with sorted spikes are:
# - `PositionGroup`
# - `SortedSpikesGroup`
# - `DecodingParameters`
# - `encoding_interval`
# - `decoding_interval`
#
# This time, instead of extracting waveform features, we can proceed directly from the SpikeSortingOutput table to specify which units we want to decode. The rest of the decoding process is the same as before.
#
#

# +
from pathlib import Path
import datajoint as dj

dj.config.load(
    Path("../dj_local_conf.json").absolute()
)  # load config for database connection info
# -

# ## SortedSpikesGroup
#
# `SortedSpikesGroup` is a child table of `SpikeSortingOutput` in the spikesorting pipeline. It allows us to group the spikesorting results from multiple
# sources (e.g. multiple tetrode groups or intervals) into a single entry. Here we will group together the spiking of multiple tetrode groups to use for decoding.
#
#
# This table allows us filter units by their annotation labels from curation (e.g only include units labeled "good", exclude units labeled "noise") by defining parameters from `UnitSelectionParams`. When accessing data through `SortedSpikesGroup` the table will include only units with at least one label in `include_labels` and no labels in `exclude_labels`. We can look at those here:
#

# +
from spyglass.spikesorting.analysis.v1.group import UnitSelectionParams

UnitSelectionParams().insert_default()

# look at the filter set we'll use here
unit_filter_params_name = "default_exclusion"
print(
    (
        UnitSelectionParams()
        & {"unit_filter_params_name": unit_filter_params_name}
    ).fetch1()
)
# look at full table
UnitSelectionParams()
# -

# Now we can make our sorted spikes group with this unit selection parameter

# +
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
import spyglass.spikesorting.v1 as sgs

nwb_copy_file_name = "mediumnwb20230802_.nwb"

sorter_keys = {
    "nwb_file_name": nwb_copy_file_name,
    "sorter": "mountainsort4",
    "curation_id": 1,
}
# check the set of sorting we'll use
(
    sgs.SpikeSortingSelection & sorter_keys
) * SpikeSortingOutput.CurationV1 & sorter_keys
# -

# Finding the merge id's corresponding to an interpretable restriction such as `merge_id` or `interval_list` can require several join steps with upstream tables.  To simplify this process we can use the included helper function `SpikeSortingOutput().get_restricted_merge_ids()` to perform the necessary joins and return the matching merge id's

# +
# get the merge_ids for the selected sorting
spikesorting_merge_ids = SpikeSortingOutput().get_restricted_merge_ids(
    sorter_keys, restrict_by_artifact=False
)

# create a new sorted spikes group
unit_filter_params_name = "default_exclusion"
SortedSpikesGroup().create_group(
    group_name="test_group",
    nwb_file_name=nwb_copy_file_name,
    keys=[
        {"spikesorting_merge_id": merge_id}
        for merge_id in spikesorting_merge_ids
    ],
    unit_filter_params_name=unit_filter_params_name,
)
# check the new group
SortedSpikesGroup & {
    "nwb_file_name": nwb_copy_file_name,
    "sorted_spikes_group_name": "test_group",
}
# -

# look at the sorting within the group we just made
SortedSpikesGroup.Units & {
    "nwb_file_name": nwb_copy_file_name,
    "sorted_spikes_group_name": "test_group",
    "unit_filter_params_name": unit_filter_params_name,
}

# ## Model parameters
#
# As before we can specify the model parameters. The only difference is that we will use the `ContFragSortedSpikesClassifier` instead of the `ContFragClusterlessClassifier`.

# +
from spyglass.decoding.v1.core import DecodingParameters
from non_local_detector.models import ContFragSortedSpikesClassifier


DecodingParameters.insert1(
    {
        "decoding_param_name": "contfrag_sorted",
        "decoding_params": ContFragSortedSpikesClassifier(),
        "decoding_kwargs": dict(),
    },
    skip_duplicates=True,
)

DecodingParameters()
# -

# ### 1D Decoding
#
# As in the clusterless notebook, we can decode 1D position if we specify the `track_graph`, `edge_order`, and `edge_spacing` parameters in the `Environment` class constructor. See the [clusterless decoding tutorial](./42_Decoding_Clusterless.ipynb) for more details.

# ## Decoding
#
# Now we can decode the position using the sorted spikes using the `SortedSpikesDecodingSelection` table. Here we assume that `PositionGroup` has been specified as in the clusterless decoding tutorial.

# +
selection_key = {
    "sorted_spikes_group_name": "test_group",
    "unit_filter_params_name": "default_exclusion",
    "position_group_name": "test_group",
    "decoding_param_name": "contfrag_sorted",
    "nwb_file_name": "mediumnwb20230802_.nwb",
    "encoding_interval": "pos 0 valid times",
    "decoding_interval": "test decoding interval",
    "estimate_decoding_params": False,
}

from spyglass.decoding import SortedSpikesDecodingSelection

SortedSpikesDecodingSelection.insert1(
    selection_key,
    skip_duplicates=True,
)

# +
from spyglass.decoding.v1.sorted_spikes import SortedSpikesDecodingV1

SortedSpikesDecodingV1.populate(selection_key)
# -

# We verify that the results have been inserted into the `DecodingOutput` merge table.

# +
from spyglass.decoding.decoding_merge import DecodingOutput

DecodingOutput.SortedSpikesDecodingV1 & selection_key
# -

# We can load the results as before:

results = (SortedSpikesDecodingV1 & selection_key).fetch_results()
results
