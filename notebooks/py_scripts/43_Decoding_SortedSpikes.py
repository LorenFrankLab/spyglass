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

# # Sorted Spikes Decoding
#
# The mechanics of decoding with sorted spikes are largely similar to those of decoding with unsorted spikes. You should familiarize yourself with the [clusterless decoding tutorial](./42_Decoding_Clusterless.ipynb) before proceeding with this one.
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
# ## SortedSpikesGroup

# +
from pathlib import Path
import datajoint as dj

dj.config.load(
    Path("../dj_local_conf.json").absolute()
)  # load config for database connection info

# +
from spyglass.spikesorting.merge import SpikeSortingOutput
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
from spyglass.decoding.v1.sorted_spikes import SortedSpikesGroup

SortedSpikesGroup()

# +
SortedSpikesGroup().create_group(
    group_name="test_group",
    nwb_file_name=nwb_copy_file_name,
    keys=[
        {"spikesorting_merge_id": merge_id}
        for merge_id in spikesorting_merge_ids
    ],
)

SortedSpikesGroup & {
    "nwb_file_name": nwb_copy_file_name,
    "sorted_spikes_group_name": "test_group",
}
# -

SortedSpikesGroup.SortGroup & {
    "nwb_file_name": nwb_copy_file_name,
    "sorted_spikes_group_name": "test_group",
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
from spyglass.decoding.v1.sorted_spikes import SortedSpikesDecodingSelection

SortedSpikesDecodingSelection()

# +
selection_key = {
    "sorted_spikes_group_name": "test_group",
    "position_group_name": "test_group",
    "decoding_param_name": "contfrag_sorted",
    "nwb_file_name": "mediumnwb20230802_.nwb",
    "encoding_interval": "pos 0 valid times",
    "decoding_interval": "test decoding interval",
    "estimate_decoding_params": False,
}

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

results = (SortedSpikesDecodingV1 & selection_key).load_results()
results
