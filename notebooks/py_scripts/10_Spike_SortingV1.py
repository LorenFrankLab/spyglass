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

# # Spike Sorting: pipeline version 1

# This is a tutorial for Spyglass spike sorting pipeline version 1 (V1). This pipeline coexists with [version 0](./10_Spike_SortingV0.ipynb) but differs in that:
# - it stores more of the intermediate results (e.g. filtered and referenced recording) in the NWB format
# - it has more streamlined curation pipelines
# - it uses UUIDs as the primary key for important tables (e.g. `SpikeSorting`) to reduce the number of keys that make up the composite primary key
#
# The output of both versions of the pipeline are saved in a [merge table](./03_Merge_Tables.ipynb) called `SpikeSortingOutput`.

# To start, connect to the database. See instructions in [Setup](./00_Setup.ipynb).

# +
import os
import datajoint as dj
import numpy as np

# change to the upper level folder to detect dj_local_conf.json
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
dj.config["enable_python_native_blobs"] = True
dj.config.load("dj_local_conf.json")  # load config for database connection info
# -

# ## Insert Data and populate pre-requisite tables

# First, import the pipeline and other necessary modules.

import spyglass.common as sgc
import spyglass.spikesorting.v1 as sgs
import spyglass.data_import as sgi

# We will be using `minirec20230622.nwb` as our example. As usual, first insert the NWB file into `Session` (can skip if you have already done so).

nwb_file_name = "minirec20230622.nwb"
nwb_file_name2 = "minirec20230622_.nwb"
sgi.insert_sessions(nwb_file_name)
sgc.Session() & {"nwb_file_name": nwb_file_name2}

# All spikesorting results are linked to a team name from the `LabTeam` table. If you haven't already inserted a team for your project do so here.

# Make a lab team if doesn't already exist, otherwise insert yourself into team
team_name = "My Team"
if not sgc.LabTeam() & {"team_name": team_name}:
    sgc.LabTeam().create_new_team(
        team_name=team_name,  # Should be unique
        team_members=[],
        team_description="test",  # Optional
    )

# ## Define sort groups and extract recordings

# Each NWB file will have multiple electrodes we can use for spike sorting. We
# commonly use multiple electrodes in a `SortGroup` selected by what tetrode or
# shank of a probe they were on. Electrodes in the same sort group will then be
# sorted together.

sgs.SortGroup.set_group_by_shank(nwb_file_name=nwb_file_name2)

# The next step is to filter and reference the recording so that we isolate the spike band data. This is done by combining the data with the parameters in `SpikeSortingRecordingSelection`. For inserting into this table, use `insert_selection` method. This automatically generates a UUID for a recording.
#

# define and insert a key for each sort group and interval you want to sort
key = {
    "nwb_file_name": nwb_file_name2,
    "sort_group_id": 0,
    "preproc_param_name": "default",
    "interval_list_name": "01_s1",
    "team_name": "My Team",
}
sgs.SpikeSortingRecordingSelection.insert_selection(key)

# Next we will call `populate` method of `SpikeSortingRecording`.

# +
# Assuming 'key' is a dictionary with fields that you want to include in 'ssr_key'
ssr_key = {
    "recording_id": (sgs.SpikeSortingRecordingSelection() & key).fetch1(
        "recording_id"
    ),
} | key

ssr_pk = (sgs.SpikeSortingRecordingSelection & key).proj()
sgs.SpikeSortingRecording.populate(ssr_pk)
sgs.SpikeSortingRecording() & ssr_key
# -

key = (sgs.SpikeSortingRecordingSelection & key).fetch1()

# ## Artifact Detection

# Sometimes the recording may contain artifacts that can confound spike sorting. For example, we often have artifacts when the animal licks the reward well for milk during behavior. These appear as sharp transients across all channels, and sometimes they are not adequately removed by filtering and referencing. We will identify the periods during which this type of artifact appears and set them to zero so that they won't interfere with spike sorting.

sgs.ArtifactDetectionSelection.insert_selection(
    {"recording_id": key["recording_id"], "artifact_param_name": "default"}
)
sgs.ArtifactDetection.populate()

sgs.ArtifactDetection()

# The output of `ArtifactDetection` is actually stored in `IntervalList` because it is another type of interval. The UUID however can be found in both.

# ## Run Spike Sorting

# Now that we have prepared the recording, we will pair this with a spike sorting algorithm and associated parameters. This will be inserted to `SpikeSortingSelection`, again via `insert_selection` method.

# The spike sorting pipeline is powered by `spikeinterface`, a community-developed Python package that enables one to easily apply multiple spike sorters to a single recording. Some spike sorters have special requirements, such as GPU. Others need to be installed separately from spyglass. In the Frank lab, we have been using `mountainsort4`, though the pipeline have been tested with `mountainsort5`, `kilosort2_5`, `kilosort3`, and `ironclust` as well.
#
# When using `mountainsort5`, make sure to run `pip install mountainsort5`. `kilosort2_5`, `kilosort3`, and `ironclust` are MATLAB-based, but we can run these without having to install MATLAB thanks to `spikeinterface`. It does require downloading additional files (as singularity containers) so make sure to do `pip install spython`. These sorters also require GPU access, so also do ` pip install cuda-python` (and make sure your computer does have a GPU).

# +
sorter = "mountainsort4"

common_key = {
    "recording_id": key["recording_id"],
    "sorter": sorter,
    "nwb_file_name": nwb_file_name2,
    "interval_list_name": str(
        (
            sgs.ArtifactDetectionSelection
            & {"recording_id": key["recording_id"]}
        ).fetch1("artifact_id")
    ),
}

if sorter == "mountainsort4":
    key = {
        **common_key,
        "sorter_param_name": "franklab_tetrode_hippocampus_30KHz",
    }
else:
    key = {
        **common_key,
        "sorter_param_name": "default",
    }
# -

sgs.SpikeSortingSelection.insert_selection(key)
sgs.SpikeSortingSelection() & key

# Once `SpikeSortingSelection` is populated, let's run `SpikeSorting.populate`.

# +
sss_pk = (sgs.SpikeSortingSelection & key).proj()

sgs.SpikeSorting.populate(sss_pk)
# -

# The spike sorting results (spike times of detected units) are saved in an NWB file. We can access this in two ways. First, we can access it via the `fetch_nwb` method, which allows us to directly access the spike times saved in the `units` table of the NWB file. Second, we can access it as a `spikeinterface.NWBSorting` object. This allows us to take advantage of the rich APIs of `spikeinterface` to further analyze the sorting.

sorting_nwb = (sgs.SpikeSorting & key).fetch_nwb()
sorting_si = sgs.SpikeSorting.get_sorting(key)

# Note that the spike times of `fetch_nwb` is in units of seconds aligned with the timestamps of the recording. The spike times of the `spikeinterface.NWBSorting` object is in units of samples (as is generally true for sorting objects in `spikeinterface`).

# ## Automatic Curation

# Next step is to curate the results of spike sorting. This is often necessary because spike sorting algorithms are not perfect;
# they often return clusters that are clearly not biological in origin, and sometimes oversplit clusters that should have been merged.
# We have two main ways of curating spike sorting: by computing quality metrics followed by thresholding, and manually applying curation labels.
# To do either, we first insert the spike sorting to `CurationV1` using `insert_curation` method.
#

sgs.SpikeSortingRecording & key
sgs.CurationV1.insert_curation(
    sorting_id=(
        sgs.SpikeSortingSelection & {"recording_id": key["recording_id"]}
    ).fetch1("sorting_id"),
    description="testing sort",
)

sgs.CurationV1()

# We will first do an automatic curation based on quality metrics. Under the hood, this part again makes use of `spikeinterface`. Some of the quality metrics that we often compute are the nearest neighbor isolation and noise overlap metrics, as well as SNR and ISI violation rate. For computing some of these metrics, the waveforms must be extracted and projected onto a feature space. Thus here we set the parameters for waveform extraction as well as how to curate the units based on these metrics (e.g. if `nn_noise_overlap` is greater than 0.1, mark as `noise`).

key = {
    "sorting_id": (
        sgs.SpikeSortingSelection & {"recording_id": key["recording_id"]}
    ).fetch1("sorting_id"),
    "curation_id": 0,
    "waveform_param_name": "default_not_whitened",
    "metric_param_name": "franklab_default",
    "metric_curation_param_name": "default",
}

sgs.MetricCurationSelection.insert_selection(key)
sgs.MetricCurationSelection() & key

sgs.MetricCuration.populate(key)
sgs.MetricCuration() & key

# to do another round of curation, fetch the relevant info and insert back into CurationV1 using `insert_curation`
#

key = {
    "metric_curation_id": (
        sgs.MetricCurationSelection & {"sorting_id": key["sorting_id"]}
    ).fetch1("metric_curation_id")
}
labels = sgs.MetricCuration.get_labels(key)
merge_groups = sgs.MetricCuration.get_merge_groups(key)
metrics = sgs.MetricCuration.get_metrics(key)
sgs.CurationV1.insert_curation(
    sorting_id=(
        sgs.MetricCurationSelection
        & {"metric_curation_id": key["metric_curation_id"]}
    ).fetch1("sorting_id"),
    parent_curation_id=0,
    labels=labels,
    merge_groups=merge_groups,
    metrics=metrics,
    description="after metric curation",
)

sgs.CurationV1()

# ## Manual Curation (Optional)

# Next we will do manual curation. this is done with figurl. to incorporate info from other stages of processing (e.g. metrics) we have to store that with kachery cloud and get curation uri referring to it. it can be done with `generate_curation_uri`.
#
# _Note_: This step is dependent on setting up a kachery sharing system as described in [02_Data_Sync.ipynb](02_Data_Sync.ipynb)
# and will likely not work correctly on the spyglass-demo server.
#

curation_uri = sgs.FigURLCurationSelection.generate_curation_uri(
    {
        "sorting_id": (
            sgs.MetricCurationSelection
            & {"metric_curation_id": key["metric_curation_id"]}
        ).fetch1("sorting_id"),
        "curation_id": 1,
    }
)
key = {
    "sorting_id": (
        sgs.MetricCurationSelection
        & {"metric_curation_id": key["metric_curation_id"]}
    ).fetch1("sorting_id"),
    "curation_id": 1,
    "curation_uri": curation_uri,
    "metrics_figurl": list(metrics.keys()),
}
sgs.FigURLCurationSelection()

sgs.FigURLCurationSelection.insert_selection(key)
sgs.FigURLCurationSelection()

sgs.FigURLCuration.populate()
sgs.FigURLCuration()

# or you can manually specify it if you already have a `curation.json`
#

# +
gh_curation_uri = (
    "gh://LorenFrankLab/sorting-curations/main/khl02007/test/curation.json"
)

key = {
    "sorting_id": key["sorting_id"],
    "curation_id": 1,
    "curation_uri": gh_curation_uri,
    "metrics_figurl": [],
}
sgs.FigURLCurationSelection.insert_selection(key)
# -

sgs.FigURLCuration.populate()
sgs.FigURLCuration()

# once you apply manual curation (curation labels and merge groups) you can store them as nwb by inserting another row in CurationV1. And then you can do more rounds of curation if you want.
#

labels = sgs.FigURLCuration.get_labels(gh_curation_uri)
merge_groups = sgs.FigURLCuration.get_merge_groups(gh_curation_uri)
sgs.CurationV1.insert_curation(
    sorting_id=key["sorting_id"],
    parent_curation_id=1,
    labels=labels,
    merge_groups=merge_groups,
    metrics=metrics,
    description="after figurl curation",
)

sgs.CurationV1()

# ## Downstream usage (Merge table)
#
# Regardless of Curation method used, to make use of spikeorting results in downstream pipelines like Decoding, we will need to insert it into the `SpikeSortingOutput` merge table.

# +
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

SpikeSortingOutput()
# -

# insert the automatic curation spikesorting results
curation_key = sss_pk.fetch1("KEY")
curation_key["curation_id"] = 1
merge_insert_key = (sgs.CurationV1 & curation_key).fetch("KEY", as_dict=True)
SpikeSortingOutput.insert(merge_insert_key, part_name="CurationV1")
SpikeSortingOutput.merge_view()

# Finding the merge id's corresponding to an interpretable restriction such as `merge_id` or `interval_list` can require several join steps with upstream tables.  To simplify this process we can use the included helper function `SpikeSortingOutput().get_restricted_merge_ids()` to perform the necessary joins and return the matching merge id's

selection_key = {
    "nwb_file_name": nwb_file_name2,
    "sorter": "mountainsort4",
    "interval_list_name": "01_s1",
    "curation_id": 0,
}  # this function can use restrictions from throughout the spikesorting pipeline
spikesorting_merge_ids = SpikeSortingOutput().get_restricted_merge_ids(
    selection_key, as_dict=True
)
spikesorting_merge_ids

# With the spikesorting merge_ids we want we can also use the method `get_sort_group_info` to get a table linking the merge id to the electrode group it is sourced from.  This can be helpful for restricting to just electrodes from a brain area of interest

merge_keys = [{"merge_id": str(id)} for id in spikesorting_merge_ids]
SpikeSortingOutput().get_sort_group_info(merge_keys)
