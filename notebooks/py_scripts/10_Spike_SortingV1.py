# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: spyglass-2024-02-07
#     language: python
#     name: spyglass-ds
# ---

# Connect to db. See instructions in [Setup](./00_Setup.ipynb).
#

# +
import os
import datajoint as dj
import numpy as np

# change to the upper level folder to detect dj_local_conf.json
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
dj.config["enable_python_native_blobs"] = True
dj.config.load("dj_local_conf.json")  # load config for database connection info

# %load_ext autoreload
# %autoreload 2
# -

# import
#

import spyglass.common as sgc
import spyglass.spikesorting.v1 as sgs
import spyglass.data_import as sgi

# insert LabMember and Session
#

nwb_file_name = "mediumnwb20230802.nwb"
nwb_file_name2 = "mediumnwb20230802_.nwb"

sgi.insert_sessions(nwb_file_name)

sgc.Session()

# insert SortGroup
#

sgs.SortGroup.set_group_by_shank(nwb_file_name=nwb_file_name2)

# insert SpikeSortingRecordingSelection. use `insert_selection` method. this automatically generates a unique recording id
#

key = {
    "nwb_file_name": nwb_file_name2,
    "sort_group_id": 0,
    "preproc_param_name": "default",
}

sgs.SpikeSortingRecordingSelection() & key

sgs.SpikeSortingRecordingSelection.insert_selection(key)

# Assuming 'key' is a dictionary with fields that you want to include in 'ssr_key'
ssr_key = {
    "recording_id": (sgs.SpikeSortingRecordingSelection() & key).fetch1(
        "recording_id"
    ),
} | key

# preprocess recording (filtering and referencing)
#

# +
# sgs.SpikeSortingRecording.populate()
ssr_pk = (sgs.SpikeSortingRecordingSelection & key).proj()


sgs.SpikeSortingRecording.populate(ssr_pk)
# -

sgs.SpikeSortingRecording() & ssr_key

key = (sgs.SpikeSortingRecordingSelection & key).fetch1()

# insert ArtifactDetectionSelection
#

sgs.ArtifactDetectionSelection.insert_selection(
    {"recording_id": key["recording_id"], "artifact_param_name": "default"}
)

# detect artifact; note the output is stored in IntervalList
#

sgs.ArtifactDetection.populate()

sgs.ArtifactDetection()

# insert SpikeSortingSelection. again use `insert_selection` method.
#
# We tested mountainsort4, mountainsort5, kilosort2_5, kilosort3, and ironclust.
# when using mountainsort5, pip install 'mountainsort5'
# when using Kilosorts and ironclust -- make sure to pip install 'cuda-python' and 'spython'
# For sorting with Kilosort, make sure to use a machine with GPU and put the whole probe not a sliced individual shank.
#

# Install mountainsort 4 if you haven't done it.

# #!pip install pybind11
# !pip install mountainsort4

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

# run spike sorting
#

# +
sss_pk = (sgs.SpikeSortingSelection & key).proj()

sgs.SpikeSorting.populate(sss_pk)
# -

# we have two main ways of curating spike sorting: by computing quality metrics and applying threshold; and manually applying curation labels. to do so, we first insert CurationV1. use `insert_curation` method.
#

sgs.SpikeSortingRecording & key

sgs.CurationV1.insert_curation(
    sorting_id=(
        sgs.SpikeSortingSelection & {"recording_id": key["recording_id"]}
    ).fetch1("sorting_id"),
    description="testing sort",
)

sgs.CurationV1()

# we will first do an automatic curation based on quality metrics
#

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

sgs.MetricCurationSelection()

sgs.MetricCuration.populate()

sgs.MetricCuration()

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

# next we will do manual curation. this is done with figurl. to incorporate info from other stages of processing (e.g. metrics) we have to store that with kachery cloud and get curation uri referring to it. it can be done with `generate_curation_uri`.
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
# -

sgs.FigURLCurationSelection.insert_selection(key)

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

# We now insert the curated spike sorting to a `Merge` table for feeding into downstream processing pipelines.
#

from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

SpikeSortingOutput()

key

SpikeSortingOutput.insert([key], part_name="CurationV1")

SpikeSortingOutput.merge_view()

SpikeSortingOutput.CurationV1()

SpikeSortingOutput.CuratedSpikeSorting()
