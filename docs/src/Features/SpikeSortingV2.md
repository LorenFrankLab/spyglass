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
    Optional ADC phase-shift, then bandpass, then common-reference referencing;
    whitening is deferred to the sort stage so motion correction never sees
    whitened data. The make body validates timestamp coverage and raises
    `RecordingTruncatedError` if the raw timestamps array does not span the
    requested interval. See [ADC phase-shift](#adc-phase-shift-neuropixels) below.
- **`DriftEstimate`** -- per-`Recording` probe-motion QC estimate
    (`compute_motion`), populated **on demand**. Stores the displacement field
    plus a `max_abs_displacement_um` summary so high-drift sessions can be
    flagged. It is **never applied** to the traces or the sort -- drift
    correction stays deferred to the sorter. See
    [Drift QC](#drift-qc-motion-estimate-never-applied) below.
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

### ADC phase-shift (Neuropixels)

Multiplexed ADCs (e.g. Neuropixels) sample the channels of a shank at slightly
different times within each sample period. The optional `phase_shift`
preprocessing parameter compensates these per-channel sub-sample delays. When
enabled, it runs **first** -- before the bandpass -- and only when the recording
carries an `inter_sample_shift` property; on a recording without that property
(any non-multiplexed acquisition, including Frank-lab polymer probes) it logs a
skip and is a **no-op**, so enabling it never fails.

It is **off in `default_franklab`** (the headline default is unchanged) and
**on in the `default_neuropixels` preset** -- a blessed Neuropixels recipe
(`bandpass 300-6000 Hz` + phase-shift, `margin_ms=100`). Because Frank-lab smoke
recordings carry no `inter_sample_shift`, `default_neuropixels` materializes
**identically** to `default_franklab` on them (the phase-shift is skipped); it
only does work once an acquisition system that ingests `inter_sample_shift` is
used. To use it:

```python
rec_pk = RecordingSelection.insert_selection({
    "nwb_file_name": nwb_file_name,
    "sort_group_id": 0,
    "interval_list_name": "raw data valid times",
    "preproc_params_name": "default_neuropixels",
    "team_name": "my_team",
})
```

The persisted `ElectricalSeries.filtering` provenance lists `phase-shift (ADC)`
only when the step actually ran (not when it was requested but skipped).

### Automated bad-channel detection

`suggest_bad_channels` proposes — and, on a second confirmed call, persists —
the `Electrode.bad_channel` flags for a session. It loads the raw recording,
bandpass-filters it, and runs SpikeInterface's `detect_bad_channels`
(`coherence+psd`, the IBL coherence/PSD method) **per full shank** (the
coherence method is spatially local, so each physical shank is scanned on its
own). It is **suggest-then-confirm**: the default `write=False` changes nothing
and just returns a report.

```python
from spyglass.spikesorting.v2._bad_channels import suggest_bad_channels

# 1. Review (default): mutates nothing, returns one dict per flagged electrode.
report = suggest_bad_channels(nwb_file_name, write=False)
for entry in report:
    print(entry)  # {"electrode_group_name", "electrode_id", "probe_shank", "label"}

# 2. Confirm: persist exactly the report you reviewed (no re-detection).
suggest_bad_channels(nwb_file_name, write=True, report=report)
```

Pass the reviewed `report` back to the confirm call so it persists precisely what
you saw. A bare `suggest_bad_channels(nwb_file_name, write=True)` re-detects, and
since the method samples random chunks (`seed=None`) it may flag a slightly
different set than the review returned — fine for a one-shot run, but pass
`report=` (or a fixed `detection_params={"seed": ...}` to both calls) when the
reviewed and persisted sets must match.

Each flagged electrode carries its **label** (`dead`, `noise`, or `out`) so you
can see what kind of bad it is before persisting. `write=True` sets
`Electrode.bad_channel='True'` for **`dead`/`noise` only** and is **additive** —
it never clears a flag you have already curated. `out` (outside-brain) channels
are **report-only**: they are surfaced for awareness but **never** written,
because `Electrode.bad_channel='True'` means a *quality-bad* (dead/noise-class)
channel that is safe to interpolate or remove — it must not mark an
outside-brain channel. To keep an `out` channel out of a sort, omit it from the
group's membership (e.g.
`SortGroupV2.set_group_by_electrode_table_column(nwb_file_name,
column="electrode_id", groups=[[...in-brain electrode_ids...]])`).

The detection thresholds are SpikeInterface defaults (Neuropixels-derived);
pass `detection_params=` (e.g. `{"dead_channel_threshold": -0.4}`) to recalibrate
for other probe geometries such as polymer probes. You can also scope the scan
with `electrode_group_names=` and change the band with `bandpass=`. The method
estimates from random chunks with SpikeInterface's `seed=None`, so the flagged
set can vary run-to-run; pass `detection_params={"seed": ...}` for a reproducible
result (it is Neuropixels-density-tuned, so small shanks such as tetrodes are
unreliable — treat a small-shank "no bad channels" with skepticism).

**Ordering contract:** run this helper and finalize the `bad_channel` flags
**before** creating sort groups. `SortGroupV2.set_group_by_*` excludes flagged
channels at group creation, so a flag added *after* a group already exists does
not retroactively drop its members — recreate the group to apply later flags.

### Bad-channel handling (remove vs interpolate)

The `bad_channel_handling` preprocessing parameter chooses what happens to the
curated `Electrode.bad_channel='True'` channels at materialization:

- **`"remove"` (default)** — byte-identical to before the option existed. A sort
  group is its declared members, and the curated-bad channels that grouping
  already excluded stay excluded. Use this for tetrodes and sparse/custom groups
  (the coherence-style geometry has no meaning there), and whenever you want the
  sorter to see only the good channels.
- **`"interpolate"`** — re-includes the group's **pitch-adjacent interior**
  curated-bad channels and fills them from good neighbours
  (`interpolate_bad_channels`, distance-weighted kriging) so a geometry-aware
  sorter (Kilosort, MountainSort5) sees a complete probe. Only bad channels
  physically embedded among the group's good channels (≥2 good neighbours within
  ~1.5× the probe pitch) are filled; an isolated bad channel, or one in the gap
  of a non-contiguous custom group, is left out (kriging has nothing local to
  fill it from). Interpolation needs probe geometry — if the probe carries no
  contact positions, `interpolate` raises a clear error (use `remove`).

```python
rec_pk = RecordingSelection.insert_selection({
    "nwb_file_name": nwb_file_name,
    "sort_group_id": 0,
    "interval_list_name": "raw data valid times",
    "preproc_params_name": "my_interpolate_preset",  # bad_channel_handling="interpolate"
    "team_name": "my_team",
})
```

The handling runs **between** the bandpass filter and the reference (matching the
IBL/AIND destripe order). The persisted `ElectricalSeries.filtering` provenance
lists `interpolate N bad channels` only when N > 0, so the default `remove` path
is unchanged.

This consumes the **curated** flags only — it does **no** detection (that is
`suggest_bad_channels`, above). The same ordering contract applies: the flags are
honoured at **group creation** (exclusion) and by `interpolate` (re-inclusion of
the excluded interior ones); `remove` honours the declared membership and
re-reads nothing, so curate flags **before** creating the sort group. Convention
boundary: `Electrode.bad_channel='True'` means a *quality-bad* (dead/noise-class)
channel only — a manually set flag on an outside-brain channel must use `remove`,
never `interpolate` (which would invent signal). The `specific` reference
electrode is never a handling target; a `bad_channel='True'` reference (e.g. a
dedicated ground) materializes exactly as before.

### Drift QC (motion estimate, never applied)

`DriftEstimate` estimates probe motion (drift) on a materialized `Recording` and
stores it as a queryable QC artifact. It is **computed, never applied** — nothing
in the pipeline corrects the traces or the sort with it (drift correction stays
deferred to the sorter, exactly as without this table). The point is to *flag*
high-drift sessions, not to change any sort output.

It is a `dj.Computed` table populated **on demand** — the expensive estimation
runs only when you call `.populate()`, never eagerly alongside `Recording`:

```python
from spyglass.spikesorting.v2.recording import DriftEstimate, Recording

# rec_key selects a materialized Recording, e.g. {"recording_id": ...}
DriftEstimate.populate(rec_key)

# Flag high-drift sessions by the summary metric.
(DriftEstimate & "max_abs_displacement_um > 20").fetch("recording_id")
max_um = (DriftEstimate & rec_key).fetch1("max_abs_displacement_um")

# Rehydrate the full SpikeInterface Motion for plotting / inspection.
motion = DriftEstimate().get_motion(rec_key)  # .displacement, .temporal_bins_s, ...
```

The estimate uses a single default preset (`dredge_fast`, stored on the row for
provenance); there is deliberately no parameters Lookup. `dredge_fast` requires
`torch`, so the `spikesorting-v2` extra now installs it. `compute_motion`
localizes peaks spatially, so it consumes the recording's channel locations (the
cached `Recording` carries probe geometry).

To be explicit: populating `DriftEstimate` leaves the upstream `Recording`
untouched — its `cache_hash` and the traces from `get_recording` are unchanged.
Applying motion correction is out of scope by design.

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

Clusterless decoding works for v2 sorts under SpikeInterface 0.104:
`UnitWaveformFeatures` (the decoding input) extracts per-spike amplitudes for a
v2 `merge_id` from a freshly built `SortingAnalyzer` — it no longer requires the
legacy SI 0.99 environment or the removed `extract_waveforms`. The `amplitude`
feature (used by clusterless decoding) and `full_waveform` are supported;
`spike_location` is not yet wired for v2 sources. A zero-unit v2 curation yields
an empty-but-valid features row. Note that v2 amplitudes are in microvolts,
whereas the legacy v0/v1 path used raw ADC counts, so v2 and v1 feature
magnitudes are not directly comparable — retrain decoders per pipeline version.

### Paper export

A v2 `merge_id` exports the same way a v1 one does — there is no v2-specific
export step. Start an export, fetch the sort through `SpikeSortingOutput`,
then populate the `Export`:

```python
from spyglass.common.common_usage import Export, ExportSelection
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

ExportSelection().start_export(paper_id="my_paper", analysis_id=1)
SpikeSortingOutput().fetch_nwb({"merge_id": merge_id})
ExportSelection().stop_export()

Export().populate_paper(paper_id="my_paper")
```

The resulting `Export.File` contains **both** the curated units NWB and the
upstream preprocessed-recording cache (plus the intermediate sort NWB),
so the export is reproducible. The recording cache is pulled in
automatically by `Export.populate_paper`'s foreign-key cascade — exactly
as it is for v1 — so you do **not** need to call `get_recording` /
`get_sorting` during the export to capture it. (Those accessors read their
files directly and do not themselves log export events, matching v1's
`CurationV1` accessors.) A zero-unit curation (the `require_units=False`
path) exports the same way; its empty-but-real units NWB is captured.

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
