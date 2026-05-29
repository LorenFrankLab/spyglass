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

v2 currently ships the single-session sorting chain:

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
from spyglass.spikesorting.v2 import initialize_v2_defaults
from spyglass.spikesorting.v2.pipeline import list_presets, run_v2_pipeline
from spyglass.spikesorting.v2.recording import SortGroupV2

# Replace with the session you've already ingested via insert_sessions.
nwb_file_name = "your_session_.nwb"

# One-shot install of every required default Lookup row
# (PreprocessingParameters + ArtifactDetectionParameters + SorterParameters).
initialize_v2_defaults()
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

Available presets:

- `franklab_tetrode_mountainsort4` -- legacy MountainSort4 (parity with v1)
- `franklab_tetrode_mountainsort5` -- **recommended**, current MS5 defaults
- `franklab_tetrode_clusterless_thresholder` -- peak-detection only (no
    clustering), feeds the clusterless decoding pipeline

`list_presets()` returns the same list at runtime.

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

nwb_file_name = "your_session_.nwb"  # same session as above

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
SpikeSortingOutput().get_spike_times({"merge_id": merge_id})
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

The single-session sort chain (above) is available now. Not yet available:

- metrics + auto-curation
- session-group sorting + cross-session unit matching

The tables for those capabilities are already declared in their final shape
with gated `make()` bodies, so enabling them later needs no schema migration.

## Streaming, parallel populate, and v1 parity

v2's `Recording` write path is built for production-scale data and
concurrent use:

- **Streaming Recording writes.** `Recording.make` now streams the
  preprocessed `ElectricalSeries` to NWB via HDMF's
  `GenericDataChunkIterator` (`buffer_gb=5`, matching v1's production
  choice). The full trace array is never materialized in RAM; chronic
  recordings (30 kHz × 128 ch × 1 h ≈ 110 GB float64) populate on any
  lab workstation. The chunked-write helpers live in
  `spikesorting.v2._nwb_iterators` (port of v1's
  `SpikeInterfaceRecordingDataChunkIterator` and
  `TimestampsDataChunkIterator`).
- **Tri-part `make` + `_parallel_make = True`** on `Recording`,
  `ArtifactDetection`, and `Sorting`. The compute step runs outside
  DataJoint's framework transaction, so a 20-minute sort no longer
  holds the row locks that would block other users from declaring or
  modifying tables on the same database. Set
  `dj.config["custom"]["spikesorting_v2_job_kwargs"] = {"n_jobs": N}`
  to thread N workers through every compute stage (the resolver is
  wired into Recording, ArtifactDetection, and Sorting; v1's pattern
  applied only on the sorter call).

v2 also matches v1 behavior on a long list of points; see the v0.5.6
CHANGELOG for the full list. Key user-visible items:

- The `CurationV2.MergeGroup` part table records every merge group's
  `(kept_unit_id, contributor_unit_id)` rows (contributor ids are
  validated against the sorting's units at insert time);
  `CurationV2.get_merge_groups(key)` returns a
  `{kept: [contributors]}` dict, and `CurationV2.get_merged_sorting`
  applies merges lazily at fetch regardless of the `merges_applied`
  flag (matching v1 semantics where a curation created with
  `apply_merge=False` can still be inspected as merged).
- `CurationV2.insert_curation` is idempotent on root curations
  (`parent_curation_id=-1`): a second call for the same `sorting_id`
  returns the existing key + emits a `logger.warning` instead of
  staging a duplicate NWB + new row.
- The `apply_merge` kwarg name is back (v1 spelling); `labels=None`
  is accepted (semantically equivalent to `{}`).
- `Sorting.get_sorting(key, as_dataframe=True)` and
  `CurationV2.get_sorting(key, as_dataframe=True)` both return a
  pandas DataFrame with `unit_id` + `spike_times` (seconds) columns;
  the CurationV2 form also joins the `curation_label` list column
  from `UnitLabel`.
- `get_spiking_sorting_v2_merge_ids(restriction, as_dict=False)` in
  `spyglass.spikesorting.v2.utils` is the notebook-discoverable
  parallel of v1's `get_spiking_sorting_v1_merge_ids`.
- `SpikeSortingOutput.get_restricted_merge_ids` defaults to every
  available source (`v0`/`v1`, plus `v2` when the v2 module is
  importable), so v2 users copying v1 notebook patterns see v2
  merge_ids without an explicit `sources=` arg, while v0/v1-only
  deployments are unaffected. Unknown restriction keys raise
  `ValueError` instead of silently dropping.

## Rerunning fixtures + tests against an existing v2 database

`SortGroupV2.set_group_by_shank` does not honor v1's `test_mode=True`
short-circuit -- v1 would silently no-op if existing sort-group rows for the
session were already present, which masked re-run idempotency bugs. v2
treats every call as authoritative: if rows already exist and you want them
replaced, pass `delete_existing_entries=True, confirm=True` (the existing
v2 kwargs). If you only want to add rows for previously-unseen sort groups,
supply explicit `sort_group_ids=` so v2 knows which to insert.

Test fixtures that previously relied on the v1 short-circuit must opt into
the explicit flow above -- there is no v2 equivalent of `test_mode`.
