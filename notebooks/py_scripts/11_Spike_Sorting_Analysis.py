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

# # Spike Sorting Analysis
#
# Sorted spike times are a starting point of many analysis pipelines. Spyglass provides
# several tools to aid in organizing spikesorting results and tracking annotations
# across multiple analyses depending on this data.
#
# For practical examples see [Sorted Spikes Decoding](./42_Decoding_SortedSpikes.ipynb)

# ## SortedSpikesGroup
#
# In practice, downstream analyses of spikesorting will often need to combine results
# from multiple sorts (e.g. across tetrodes groups in a single interval). To make
# this simple with spyglass's relational database, we use the `SortedSpikesGroup` table.
#
# `SortedSpikesGroup` is a child table of `SpikeSortingOutput` in the spikesorting pipeline.
# It allows us to group the spikesorting results from multiple sources into a single
# entry for downstream reference, and provides tools for easily
# accessing the compiled data. Here we will group together the spiking of multiple
# tetrode groups.
#
#
# This table allows us filter units by their annotation labels from curation (e.g only
# include units labeled "good", exclude units labeled "noise") by defining parameters
# from `UnitSelectionParams`. When accessing data through `SortedSpikesGroup` the table
# will include only units with at least one label in `include_labels` and no labels in
# `exclude_labels`. We can look at those here:
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

# We then define the set of curated sorting results to include in the group
#
# Finding the merge id's corresponding to an interpretable restriction such as `merge_id` or `interval_list` can require several join steps with upstream tables.  To simplify this process we can use the included helper function `SpikeSortingOutput().get_restricted_merge_ids()` to perform the necessary joins and return the matching merge id's

# +
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

nwb_file_name = "mediumnwb20230802_.nwb"

sorter_keys = {
    "nwb_file_name": nwb_file_name,
    "sorter": "mountainsort4",
    "curation_id": 1,
}

# get the merge_ids for the selected sorting
spikesorting_merge_ids = SpikeSortingOutput().get_restricted_merge_ids(
    sorter_keys, restrict_by_artifact=True
)

keys = [{"merge_id": merge_id} for merge_id in spikesorting_merge_ids]
(SpikeSortingOutput.CurationV1 & keys)
# -

# We can now combine this information to make a spike sorting group

# +
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

# create a new sorted spikes group
unit_filter_params_name = "default_exclusion"
SortedSpikesGroup().create_group(
    group_name="demo_group",
    nwb_file_name=nwb_file_name,
    keys=[
        {"spikesorting_merge_id": merge_id}
        for merge_id in spikesorting_merge_ids
    ],
    unit_filter_params_name=unit_filter_params_name,
)
# check the new group
group_key = {
    "nwb_file_name": nwb_file_name,
    "sorted_spikes_group_name": "demo_group",
}
SortedSpikesGroup & group_key
# -

SortedSpikesGroup.Units & group_key

# We can access the spikesorting results for this data using `SortedSpikesGroup.fetch_spike_data()`
#

# get the complete key
group_key = (SortedSpikesGroup & group_key).fetch1("KEY")
# get the spike data, returns a list of unit spike times
SortedSpikesGroup().fetch_spike_data(group_key)

# ## Unit Annotation
#
# Many neuroscience applications are interested in the properties of individual neurons
# or units. For example, one set of custom analysis may classify each unit as a cell type
# based on firing properties, and a second analysis step may want to compare additional
# features based on this classification.
#
# Doing so requires a consistent manner of identifying a unit, and a location to track annotations
#
# Spyglass uses the unit identification system:
# `{"spikesorting_merge_id" : merge_id, "unit_id" : unit_id}"`,
# where `unit_id` is the index of a units in the saved nwb file. `fetch_spike_data`
# can return these identifications by setting `return_unit_ids = True`

spike_times, unit_ids = SortedSpikesGroup().fetch_spike_data(
    group_key, return_unit_ids=True
)
print(unit_ids[0])
print(spike_times[0])

# Further analysis may assign annotations to individual units. These can either be a
# string `label` (e.g. "pyridimal_cell", "thirst_sensitive"), or a float `quantification`
# (e.g. firing_rate, signal_correlation).
#
# The `UnitAnnotation` table can be used to track and cross reference these annotations
# between analysis pipelines. Each unit has a single entry in `UnitAnnotation`, which
# can be connected to multiple entries in the `UnitAnnotation.Annotation` part table.
#
# An `Annotation` entry should include an `annotation` describing the originating analysis,
# along with a `label` and/or `quantification` with the analysis result.
#
# Here, we demonstrate adding quantification and label annotations to the units in
# the spike group we created using the `add_annotation` function.

# +
from spyglass.spikesorting.analysis.v1.unit_annotation import UnitAnnotation

for spikes, unit_key in zip(spike_times, unit_ids):
    # add a quantification annotation for the number of spikes
    annotation_key = {
        **unit_key,
        "annotation": "spike_count",
        "quantification": len(spikes),
    }
    UnitAnnotation().add_annotation(annotation_key, skip_duplicates=True)
    # add a label annotation for the unit id
    annotation_key = {
        **unit_key,
        "annotation": "cell_type",
        "label": "pyridimal" if len(spikes) < 1000 else "interneuron",
    }
    UnitAnnotation().add_annotation(annotation_key, skip_duplicates=True)

annotations = UnitAnnotation().Annotation() & unit_ids
annotations
# -

# Subsets of the the spikesorting data can then be accessed by calling `fetch_unit_spikes`
# on a restricted instance of the table. This allows the user to perform further analysis
# based on these labels.
#
# *Note:* This function will return the spike times for all units in the restricted table

# +
# restrict to units from our sorted spikes group
annotations = UnitAnnotation.Annotation & (SortedSpikesGroup.Units & group_key)
# restrict to units with more than 3000 spikes
annotations = annotations & {"annotation": "spike_count"}
annotations = annotations & "quantification > 3000"

selected_spike_times, selected_unit_ids = annotations.fetch_unit_spikes(
    return_unit_ids=True
)
print(selected_unit_ids[0])
print(selected_spike_times[0])
