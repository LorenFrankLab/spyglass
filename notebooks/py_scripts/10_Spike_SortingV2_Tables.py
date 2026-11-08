# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (spyglass_spikesorting_v2)
#     language: python
#     name: python3
# ---

# # Spike Sorting v2: the stage tables
#
# `run_v2_pipeline` in the [first-sort walkthrough](./10_Spike_SortingV2.ipynb)
# is the primary path: one call fills every table below. This notebook runs the
# same stages one table at a time, for when no preset fits or to inspect
# preprocessing and artifacts before sorting.
#
# v1 users drove each table. v2 tables work the same way: fill a `*Selection`
# table with `insert_selection`, then `populate()` its computed table.
# Parameter tables provide `insert_default()`. A later pipeline call with the
# same inputs reuses these rows.
#
# Assumes a configured DataJoint connection and an ingested session (see
# [Setup](./00_Setup.ipynb) / [Insert Data](./02_Insert_Data.ipynb)).

# +
import datajoint as dj
from IPython.display import display

from spyglass.common import LabTeam
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.spikesorting.v2 import initialize_v2_defaults
from spyglass.spikesorting.v2.artifact import (
    RecordingArtifactDetection,
    RecordingArtifactSelection,
)
from spyglass.spikesorting.v2.curation import CurationV2
from spyglass.spikesorting.v2.metric_curation import (
    CurationEvaluation,
    CurationEvaluationSelection,
)
from spyglass.spikesorting.v2.pipeline import describe_sort_groups
from spyglass.spikesorting.v2.recording import (
    Recording,
    RecordingSelection,
    SortGroupV2,
)
from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

dj.config["display.limit"] = 12
# -

# Use row names from `describe_pipeline_presets()`. These are the rows of the
# `franklab_probe_hippocampus_30khz_ms5_2026_06` preset (MountainSort5).

# + tags=["parameters"]
nwb_file_name = "your_session.nwb"  # replace with your ingested session
team_name = "my_team"
interval_list_name = "raw data valid times"
sort_group_id = 0  # choose from the sort-group table below
preprocessing_params_name = "franklab_hippocampus_2026_06"
artifact_detection_params_name = "franklab_100uv_p07_2026_06"
sorter = "mountainsort5"
sorter_params_name = "franklab_30khz_ms5_2026_06"
metric_params_name = "franklab_default"
auto_curation_rules_name = "franklab_default_auto_curation_2026_09"
# -

# ## 1. Defaults, team, and sort groups
#
# The same one-time setup as the first-sort walkthrough, which also shows how
# to review channels and references before creating groups.

initialize_v2_defaults()
LabTeam.insert1(
    {"team_name": team_name, "team_description": "spike sorting"},
    skip_duplicates=True,
)
if not (SortGroupV2 & {"nwb_file_name": nwb_file_name}):
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
describe_sort_groups(nwb_file_name)

# ## 2. Recording
#
# `RecordingSelection` names the session, sort group, interval, preprocessing
# recipe, and team; `Recording` writes the preprocessed traces.

recording_key = RecordingSelection.insert_selection(
    {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": interval_list_name,
        "preprocessing_params_name": preprocessing_params_name,
        "team_name": team_name,
    }
)
Recording.populate(recording_key)
Recording & recording_key

# Preprocessed traces before masking: the first second by default, or pass
# `time_range=(start, stop)` in recording seconds.

Recording().plot_traces(recording_key)

# ## 3. Artifact detection
#
# Detection on one recording. `SharedArtifactGroup` lets several recordings of
# one session share one artifact result (e.g. chewing artifacts on every
# probe), through `SharedGroupArtifactSelection` /
# `SharedGroupArtifactDetection` instead. Masked samples read 0 µV. If a
# masked source has a nonzero channel offset (only with the `no_filter` recipe
# and `reference_mode="none"`), it is first converted to float32 µV.

artifact_detection_key = RecordingArtifactSelection.insert_selection(
    {
        "recording_id": recording_key["recording_id"],
        "artifact_detection_params_name": artifact_detection_params_name,
    }
)
RecordingArtifactDetection.populate(artifact_detection_key)
# The retained (artifact-free) intervals, shape (n_intervals, 2).
RecordingArtifactDetection().get_artifact_removed_intervals(
    artifact_detection_key
)

# ## 4. Sorting
#
# `Sorting.Unit` stores per-unit summaries (`n_spikes`, `peak_amplitude_uv`).

sorting_key = SortingSelection.insert_selection(
    {
        "recording_id": recording_key["recording_id"],
        "sorter": sorter,
        "sorter_params_name": sorter_params_name,
        "artifact_detection_id": artifact_detection_key["artifact_detection_id"],
    }
)
Sorting.populate(sorting_key)
display(Sorting.Unit & sorting_key)
Sorting().get_sorting(sorting_key, as_dataframe=True)  # spike_times in s

# ## 5. Root curation and its merge entry
#
# `insert_curation` with `parent_curation_id=-1` writes the uncurated root. A
# second default root call returns the existing key with a warning. A
# single-recording curation registers on `SpikeSortingOutput.CurationV2`, so
# downstream code keys off its `merge_id`. `get_spike_times` returns
# **every** unit, labels ignored; use it for inspection, not analysis.

curation_key = CurationV2.insert_curation(
    sorting_key=sorting_key, parent_curation_id=-1
)
merge_id = (SpikeSortingOutput.CurationV2 & curation_key).fetch1("merge_id")
spike_times = SpikeSortingOutput().get_spike_times({"merge_id": merge_id})
print(f"merge_id {merge_id}: {len(spike_times)} unit(s)")
CurationV2.summarize_curation(curation_key)

# ## 6. Curation evaluation
#
# `CurationEvaluation` scores a committed curation with a metric recipe and a
# rule set. Its proposals are suggestions: committing them as a child is the
# Curation notebook's job (Appendix A), and the analysis handoff is
# `select_units_for_analysis` (first-sort walkthrough, section 9).

evaluation_key = CurationEvaluationSelection.insert_selection(
    {
        **curation_key,
        "metric_params_name": metric_params_name,
        "auto_curation_rules_name": auto_curation_rules_name,
    }
)
CurationEvaluation.populate(evaluation_key)
display(CurationEvaluation.get_labels(evaluation_key))  # proposed labels
CurationEvaluation.get_metrics(evaluation_key)  # one row per unit
