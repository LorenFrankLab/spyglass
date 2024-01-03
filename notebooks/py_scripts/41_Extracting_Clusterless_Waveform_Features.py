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
# The goal of this notebook is to populate the `UnitWaveformFeatures` table, which depends `SpikeSortingOutput`. This table contains the features of the waveforms of each unit.
#
# While clusterless decoding avoids actual spike sorting, we need to pass through these tables to maintain (relative) pipeline simplicity. Pass-through tables keep spike sorting and clusterless waveform extraction as similar as possible, by using shared steps. Here, "spike sorting" involves simple thresholding (sorter: clusterless_thresholder).
#
# Let's start with the following nwb file and time interval:

# +
from pathlib import Path
import datajoint as dj

dj.config.load(
    Path("../dj_local_conf.json").absolute()
)  # load config for database connection info

nwb_copy_file_name = "mediumnwb20230802_.nwb"
interval_list_name = "pos 0 valid times"
# -

# If you haven't already, run the [Insert Data notebook](./01_Insert_Data.ipynb) to populate the tables.
#
# These next steps are the same as in the [Spike Sorting notebook](./10_Spike_SortingV1.ipynb), but we'll repeat them here for clarity. These are pre-processing steps that are shared between spike sorting and clusterless decoding.
#
# We first set the `SortGroup` to define which contacts are sorted together.
#
# We then setup for spike sorting by bandpass filtering and whitening the data via the `SpikeSortingRecording` table.

# +
import spyglass.spikesorting.v1 as sgs

sgs.SortGroup.set_group_by_shank(nwb_file_name=nwb_copy_file_name)

sort_group_ids = (sgs.SortGroup & {"nwb_file_name": nwb_copy_file_name}).fetch(
    "sort_group_id"
)
for sort_group_id in sort_group_ids:
    key = {
        "nwb_file_name": nwb_copy_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": interval_list_name,
        "preproc_param_name": "default",
        "team_name": "Alison Comrie",
    }
    sgs.SpikeSortingRecordingSelection.insert_selection(key)

sgs.SpikeSortingRecording.populate()
# -

# Next we do artifact detection. Here we skip it by setting the `artifact_param_name` to `None`, but in practice you should detect artifacts as it will affect the decoding.

# +
recording_ids = (
    sgs.SpikeSortingRecordingSelection & {"nwb_file_name": nwb_copy_file_name}
).fetch("recording_id")

for recording_id in recording_ids:
    key = {
        "recording_id": recording_id,
        "artifact_param_name": "none",
    }
    sgs.ArtifactDetectionSelection.insert_selection(key)

sgs.ArtifactDetection.populate()
# -

# Now we run the "spike sorting", which in our case is simply thresholding the signal to find spikes. We use the `SpikeSorting` table to store the results. Note that `sorter_param_name` defines the parameters for thresholding the signal.

# +
for recording_id in recording_ids:
    key = {
        "recording_id": recording_id,
        "sorter": "clusterless_thresholder",
        "sorter_param_name": "default_clusterless",
        "nwb_file_name": nwb_copy_file_name,
        "interval_list_name": str(
            (
                sgs.ArtifactDetectionSelection & {"recording_id": recording_id}
            ).fetch1("artifact_id")
        ),
    }
    sgs.SpikeSortingSelection.insert_selection(key)

sgs.SpikeSorting.populate()
# -

# For clusterless decoding we do not need any manual curation, but for the sake of the pipeline, we need to store the output of the thresholding in the `CurationV1` table and insert this into the `SpikeSortingOutput` table.

# +
from spyglass.spikesorting.merge import SpikeSortingOutput

sorting_ids = (
    sgs.SpikeSortingSelection & {"nwb_file_name": nwb_copy_file_name}
).fetch("sorting_id")

for sorting_id in sorting_ids:
    try:
        sgs.CurationV1.insert_curation(sorting_id=sorting_id)
    except KeyError as e:
        pass

SpikeSortingOutput.insert(
    sgs.CurationV1().fetch("KEY"),
    part_name="CurationV1",
    skip_duplicates=True,
)
# -

# Finally, we extract the waveform features of each SortGroup. This is done by the `UnitWaveformFeatures` table.
#
# To set this up, we use the `WaveformFeaturesParams` to define the time around the spike that we want to use for feature extraction, and which features to extract. Here is an example of the parameters used for extraction the amplitude of the negative peak of the waveform:
# ```python
#
# waveform_extraction_params = {
#     "ms_before": 0.5,
#     "ms_after": 0.5,
#     "max_spikes_per_unit": None,
#     "n_jobs": 5,
#     "total_memory": "5G",
# }
# waveform_feature_params = {
#     "amplitude": {
#         "peak_sign": "neg",
#         "estimate_peak_time": False,
#     }
# }
# ```
#
# We see that we want 0.5 ms of time before and after the peak of the negative spike. We also see that we want to extract the amplitude of the negative peak, and that we do not want to estimate the peak time (since we know it is at 0 ms).
#
#
# You can define other features to extract such as spatial location of the spike:
# ```python
# waveform_extraction_params = {
#     "ms_before": 0.5,
#     "ms_after": 0.5,
#     "max_spikes_per_unit": None,
#     "n_jobs": 5,
#     "total_memory": "5G",
# }
# waveform_feature_params = {
#     "amplitude": {
#         "peak_sign": "neg",
#         "estimate_peak_time": False,
#     },
#     "spike location": {}
# }
#
# ```
#

# +
from spyglass.decoding.v1.waveform_features import WaveformFeaturesParams

waveform_extraction_params = {
    "ms_before": 0.5,
    "ms_after": 0.5,
    "max_spikes_per_unit": None,
    "n_jobs": 5,
    "total_memory": "5G",
}
waveform_feature_params = {
    "amplitude": {
        "peak_sign": "neg",
        "estimate_peak_time": False,
    }
}

WaveformFeaturesParams.insert1(
    {
        "features_param_name": "amplitude",
        "params": {
            "waveform_extraction_params": waveform_extraction_params,
            "waveform_feature_params": waveform_feature_params,
        },
    },
    skip_duplicates=True,
)

WaveformFeaturesParams()
# -

# Now that we've inserted the waveform features parameters, we need to define which parameters to use for each SortGroup. This is done by the `UnitWaveformFeaturesSelection` table. We need to link the primary key `merge_id` from the `SpikeSortingOutput` table to a features parameter set.

# +
from spyglass.decoding.v1.waveform_features import UnitWaveformFeaturesSelection

UnitWaveformFeaturesSelection()
# -

# First we find the units we need:

# +
from spyglass.spikesorting.merge import SpikeSortingOutput

merge_ids = (
    (SpikeSortingOutput.CurationV1 * sgs.SpikeSortingSelection)
    & {
        "nwb_file_name": nwb_copy_file_name,
        "sorter": "clusterless_thresholder",
        "sorter_param_name": "default_clusterless",
    }
).fetch("merge_id")
merge_ids
# -

# Then we link them with the features parameters:

# +
selection_keys = [
    {
        "spikesorting_merge_id": merge_id,
        "features_param_name": "amplitude",
    }
    for merge_id in merge_ids
]
UnitWaveformFeaturesSelection.insert(selection_keys, skip_duplicates=True)

UnitWaveformFeaturesSelection & selection_keys
# -

# Finally, we extract the waveform features, by populating the `UnitWaveformFeatures` table:

# +
from spyglass.decoding.v1.waveform_features import UnitWaveformFeatures

UnitWaveformFeatures.populate(selection_keys)
# -

UnitWaveformFeatures & selection_keys

# Now that we've extracted the data, we can inspect the results. Let's fetch the data:

spike_times, spike_waveform_features = (
    UnitWaveformFeatures & selection_keys
).fetch_data()

# Let's look at the features shape. This is a list corresponding to tetrodes, with each element being a numpy array of shape (n_spikes, n_features). The features in this case are the amplitude of each tetrode wire at the negative peak of the waveform.

for features in spike_waveform_features:
    print(features.shape)

# We can plot the amplitudes to see if there is anything that looks neural and to look for outliers:

# +
import matplotlib.pyplot as plt

tetrode_ind = 1
plt.scatter(
    spike_waveform_features[tetrode_ind][:, 0],
    spike_waveform_features[tetrode_ind][:, 1],
    s=1,
)
# -