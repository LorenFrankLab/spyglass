# Spike Sorting v2

## Why

The v0 and v1 spike-sorting pipelines depend on SpikeInterface 0.99 and an
older `WaveformExtractor` API that SpikeInterface no longer ships. Running
them under SpikeInterface 0.104 raises a clear `RuntimeError` pointing
callers at either the legacy SI 0.99 environment (for existing rows) or the
v2 pipeline (for new processing).

`spyglass.spikesorting.v2` is a from-scratch rewrite of the sorting stack
on the SI 0.104 `SortingAnalyzer` API. It preserves Spyglass's DataJoint
contracts (Selection / make / merge dispatch, cascade-safe cautious deletes,
analysis-NWB lifecycle) while replacing the v1 internals with the modern
SpikeInterface objects.

## What

Phase 1 of the rewrite ships the single-session sorting chain:

```
SortGroupV2
   |
   v
RecordingSelection --> Recording          (bandpass + common reference)
                          |
                          v
                       ArtifactSelection --> ArtifactDetection
                                                  |
                                                  v
                                              SortingSelection --> Sorting
                                                                       |
                                                                       v
                                                                   CurationV2 -+--> SpikeSortingOutput.CurationV2
```

All v2 tables live in dedicated DataJoint schemas (`spikesorting_v2_recording`,
`spikesorting_v2_artifact`, `spikesorting_v2_sorting`, `spikesorting_v2_curation`),
so the v0/v1 schemas are untouched. `CurationV2` registers as a new part on the
existing `SpikeSortingOutput` merge table, so v0, v1, imported, and v2
curations all coexist under one merge surface.

### Tables

- **`SortGroupV2`** -- per-session electrode grouping. Constructors
    `set_group_by_shank` and `set_group_by_electrode_table_column` follow
    the inspect-before-destroy contract: passing
    `delete_existing_entries=True, confirm=False` returns a `DeletionPreview`
    so the caller can review cascade impact before committing.
- **`PreprocessingParameters`, `ArtifactDetectionParameters`, `SharedArtifactGroup`,
    `SorterParameters`** -- Pydantic-validated parameter Lookup rows.
    `insert_default()` on each loads a default row; user params validate the
    `params` blob on insert.
- **`RecordingSelection` / `Recording`** -- preprocessed recording materialization.
    Bandpass + common-reference referencing only; whitening is deferred to the
    sort stage so motion correction never sees whitened data. The make body
    validates timestamp coverage and raises `RecordingTruncatedError` if the
    raw timestamps array does not span the requested interval.
- **`ArtifactSelection` / `ArtifactDetection`** -- amplitude-threshold artifact
    intervals. Uses the source-part pattern (`SharedArtifactGroupSource`) so
    multiple recordings can share a single artifact-detection result.
- **`SortingSelection` / `Sorting`** -- runs the configured sorter through
    SpikeInterface. Dispatches `clusterless_thresholder` (peak detection
    only) vs the SI sorter registry (`mountainsort4`, `mountainsort5`, ...).
    The Unit part table stores per-unit summary stats (n_spikes,
    peak_amplitude_uv) so quick filtering does not require loading the NWB.
- **`CurationV2`** -- versioned curation rows (labels + merge groups) chained
    by `parent_curation_id`. `insert_curation` is the single entry point;
    every row is automatically registered on `SpikeSortingOutput.CurationV2`
    so downstream consumers can key off `merge_id`.

### Pipeline orchestrator

`spyglass.spikesorting.v2.pipeline.run_v2_pipeline` chains the per-stage
`insert_selection` + `populate` calls into one call. Three presets ship today:

- `franklab_tetrode_mountainsort4`
- `franklab_tetrode_mountainsort5` (default)
- `franklab_tetrode_clusterless_thresholder`

The orchestrator is idempotent: re-running with the same inputs returns the
same manifest (same `merge_id`, same intermediate PKs) without duplicating
rows.

## How

### Single-session sort

```python
from spyglass.common.common_lab import LabTeam
from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
from spyglass.spikesorting.v2.recording import (
    PreprocessingParameters,
    SortGroupV2,
)
from spyglass.spikesorting.v2.sorting import SorterParameters

# One-time setup of the default Lookup rows the preset references.
PreprocessingParameters.insert_default()
ArtifactDetectionParameters.insert_default()
SorterParameters.insert_default()
LabTeam.insert1(
    {"team_name": "my_team", "team_description": "..."},
    skip_duplicates=True,
)

# Build the sort groups (one per shank).
SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

# End-to-end populate + register on the merge table.
manifest = run_v2_pipeline(
    nwb_file_name=nwb_file_name,
    sort_group_id=0,
    interval_list_name="raw data valid times",
    team_name="my_team",
    preset="franklab_tetrode_mountainsort5",
)
merge_id = manifest["merge_id"]  # key off this downstream
```

### Stage-by-stage (custom preset)

`run_v2_pipeline` is a convenience wrapper. The underlying stages can be
driven directly when a preset does not apply:

```python
from spyglass.spikesorting.v2.recording import (
    Recording, RecordingSelection,
)
from spyglass.spikesorting.v2.artifact import (
    ArtifactDetection, ArtifactSelection,
)
from spyglass.spikesorting.v2.sorting import (
    Sorting, SortingSelection,
)
from spyglass.spikesorting.v2.curation import CurationV2

rec_pk = RecordingSelection.insert_selection({
    "nwb_file_name": nwb_file_name,
    "sort_group_id": 0,
    "interval_list_name": "raw data valid times",
    "preproc_params_name": "default_franklab",
    "team_name": "my_team",
})
Recording.populate(rec_pk)

art_pk = ArtifactSelection.insert_selection({
    "recording_id": rec_pk["recording_id"],
    "artifact_params_name": "default",
})
ArtifactDetection.populate(art_pk)

sort_pk = SortingSelection.insert_selection({
    "recording_id": rec_pk["recording_id"],
    "sorter": "mountainsort5",
    "sorter_params_name": "franklab_tetrode_hippocampus_30kHz_ms5",
    "artifact_id": art_pk["artifact_id"],
})
Sorting.populate(sort_pk)

curation_pk = CurationV2.insert_curation(
    sorting_key=sort_pk,
    labels={},
    parent_curation_id=-1,
    description="first pass",
)
```

### Downstream consumers

Both v1 (`CurationV1`) and v2 (`CurationV2`) curations register on the same
`SpikeSortingOutput` merge table, so existing downstream code (decoding,
ripple detection, etc.) keeps working unchanged:

```python
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

# v2 rows are dispatched alongside v1 rows
SpikeSortingOutput.get_spike_times({"merge_id": merge_id})
SpikeSortingOutput.get_unit_brain_regions({"merge_id": merge_id})
```

### Environment

The v2 pipeline requires SpikeInterface 0.104+ and (for MountainSort) the
`spikesorting-v2` optional extra:

```bash
pip install "spyglass-neuro[spikesorting-v2]"
```

The legacy v0/v1 workflows (`Waveforms`, `MetricCuration`, `BurstPair`,
v1 `ArtifactDetection`) still require the SI 0.99 Spyglass environment;
calling them under SI 0.104 raises a clear `RuntimeError`.

## Status

Phase 1 (single-session sort) is the first landed slice. Later phases extend
the same tables with:

- metrics + auto-curation (Phase 2)
- session-group sorting + cross-session unit matching (Phase 3)

The Phase 3 tables are declared (final-shape) in Phase 1 with gated `make()`
bodies, so there are no schema migrations between phases.
