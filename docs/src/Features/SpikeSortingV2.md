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
                       ArtifactDetectionSelection --> ArtifactDetection
                                                  |
                                                  v
                                              SortingSelection --> Sorting
                                                                       |
                                                                       v
                                                                   CurationV2 -+--> SpikeSortingOutput.CurationV2
                                                                        |
                                                                        v
                                                        CurationEvaluationSelection --> CurationEvaluation
```

All v2 tables live in dedicated DataJoint schemas (`spikesorting_v2_recording`,
`spikesorting_v2_artifact`, `spikesorting_v2_sorting`,
`spikesorting_v2_curation`, `spikesorting_v2_recompute`), so the v0/v1 schemas
are untouched. `CurationV2` registers as a new part on the existing
`SpikeSortingOutput` merge table, so v0, v1, imported, and v2 curations all
coexist under one merge surface.

### Tables

- **`SortGroupV2`** -- per-session electrode grouping. Constructors
    `set_group_by_shank` and `set_group_by_electrode_table_column` follow
    the inspect-before-destroy contract: passing
    `delete_existing_entries=True, confirm=False` returns a `DeletionPreview`
    so the caller can review cascade impact before committing.
- **`PreprocessingParameters`, `ArtifactDetectionParameters`, `SharedArtifactGroup`,
    `SorterParameters`, `QualityMetricParameters`, `AutoCurationRules`** --
    Pydantic-validated parameter Lookup rows.
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
- **`ArtifactDetectionSelection` / `ArtifactDetection`** -- amplitude-threshold artifact
    intervals. Uses the source-part pattern (`SharedGroupSource`) so
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
- **`CurationEvaluationSelection` / `CurationEvaluation`** -- post-sort SI
    analyzer extension growth, quality metrics, auto-curation labels, merge
    suggestions, and BurstPair-style plots over a **committed** `CurationV2` row,
    scored in that curation's own unit namespace (a merged unit's metrics are
    recomputed over the merged template, not inherited). Proposals are persisted
    to NWB; committing them to a child curation is an explicit
    `create_curation` / `use_evaluation_labels` step. Preview/draft curations are
    rejected. Replaces the removed `AnalyzerCuration`. See
    [Quality metrics, evaluation, and acceptance](#quality-metrics-evaluation-and-acceptance-curationevaluation).
- **`RecordingArtifactRecompute*` / `SortingAnalyzerRecompute*`** -- v2 storage
    verification families for safely reclaiming preprocessed recording/artifact
    NWBs and analyzer folders after a current-environment content match.
- **`SessionGroup`** -- a named bundle of sorting members analyzed together
    (chronic concatenation *and* cross-session matching reuse it). See
    [Cross-session unit tracking](#cross-session-unit-tracking).
- **`MatcherParameters`** -- registry-validated cross-session matcher
    configuration. `insert1` rejects an unregistered `matcher` name and
    Pydantic-validates `params` against that matcher's schema.
- **`UnitMatchSelection` / `UnitMatch`** -- pin one curation per `SessionGroup`
    member, then match units across sessions via the chosen matcher backend.
    The `Pair` part records each cross-session match (FK-validated against the
    pinned `CurationV2.Unit`).
- **`TrackedUnit`** -- biological-unit identities across sessions: a strict
    partition of the curated units into groups via a greedy maximal-clique cover
    of the match graph (one identity per unit). `get_unit_brain_regions`
    resolves each tracked unit's per-session brain regions.

### Pipeline orchestrator

`spyglass.spikesorting.v2.pipeline.run_v2_pipeline` chains the per-stage
`insert_selection` + `populate` calls into one call. The shipped presets are
the dated June-2026 Frank Lab production recipes: a MountainSort4 family keyed
by target region (hippocampus 600 Hz / cortex 300 Hz high-pass) and sampling
rate (30 / 20 kHz), plus a MountainSort5 preset and a clusterless preset.
The default is the MountainSort5 recipe
`franklab_probe_hippocampus_30khz_ms5_2026_06` (the probe-labeled twin of the
tetrode-labeled MS5 preset; both resolve to the same parameter rows): it runs
under the v2 `numpy>=2` baseline out of the box. MountainSort4 is the
scientifically-preferred polymer-probe recipe, but its `ml_ms4alg` backend needs
`numpy<2`, so it is not the default -- run it on a modern (`numpy>=2`) host via
the containerized `franklab_probe_hippocampus_30khz_ms4_singularity_2026_06`
preset (the recommended-science MS4 path when Docker/Singularity is available),
or on a `numpy<2` host via the local `franklab_probe_hippocampus_30khz_ms4_2026_06`
preset (preflight reports an unrunnable MS4 path via its
`sorter_runtime_available` / `container_runtime_available` checks). Call
`describe_pipeline_presets()` for the catalog and `list_pipeline_presets()` for
the names.

The orchestrator is idempotent: re-running with the same inputs returns the
same run summary (same `merge_id`, same intermediate PKs) without duplicating
rows.

## Security & trust model

Spike Sorting v2 assumes a **trusted compute-operator** deployment, the same
model as the rest of Spyglass:

- **Whoever can write `SorterParameters` (or ingest sessions) is a trusted
  operator.** A `SorterParameters` row's `execution_params` can, by design,
  pull and run a container image (Docker / Singularity) to execute a sorter.
  That is a deliberate capability for reproducible containerized sorting, not a
  vulnerability — but it means inserting parameter rows is equivalent to running
  code on the compute host. Restrict write access accordingly.
- **The database is not internet-facing.** v2 hard-refuses to register or write
  its schemas against a non-localhost database host (the import-time
  `_assert_v2_db_safe` guard) while the pipeline is pre-production; deployment
  is on a lab-internal DB reachable only by trusted operators.
- **`team_name` is a provenance tag, not access enforcement.** It records which
  team owns a selection; it does not gate reads or writes. Because sort groups
  in one session can belong to different teams, overwriting a session's sort
  groups can cascade-delete another team's downstream rows — the overwrite
  preview (`SortGroupV2.preview_existing_entries`) enumerates that cross-team
  blast radius so the operator can review it before confirming, but it does not
  block.

Within that model, v2 still defaults to least surprise: materialized analysis
artifacts are written owner-writable (`0o644`), the sorter scratch is only made
world-writable for a container backend (the container-UID case), and
caller-supplied NWB file names are confined to a bare basename before any
directory join.

## How

### Run your first single-session sort

The fastest way to learn the pipeline is to run the notebook
[`notebooks/10_Spike_SortingV2.ipynb`](../notebooks/10_Spike_SortingV2.ipynb),
which walks the first-sort happy path on one already-ingested session; the
deeper how-tos are split into companion notebooks —
[`10_Spike_SortingV2_Curation.ipynb`](../notebooks/10_Spike_SortingV2_Curation.ipynb)
(browser + step-by-step curation),
[`10_Spike_SortingV2_Presets.ipynb`](../notebooks/10_Spike_SortingV2_Presets.ipynb)
(customize a preset, sort a whole session), and
[`10_Spike_SortingV2_CrossSession.ipynb`](../notebooks/10_Spike_SortingV2_CrossSession.ipynb)
(concatenate + cross-session matching). In prose, the first-sort path is:

1. **Defaults** -- `initialize_v2_defaults()` seeds every parameter row.
2. **Sort group** -- `SortGroupV2.set_group_by_shank(nwb_file_name=...)`,
   then inspect `describe_sort_groups(nwb_file_name)` before choosing the
   `sort_group_id`.
3. **Preflight** -- `preflight_v2_pipeline(...)` confirms the session, team,
   parameter rows, and sorter binary are present in ~1 s, *before* any
   `populate`, returning a structured report with the exact fix for any missing
   prerequisite.
4. **Pipeline** -- `run_v2_pipeline(...)` returns the run summary (key off
   `merge_id`).
5. **Summary** -- `CurationV2.summarize_curation(run_summary)`.
6. **Fetch** -- `SpikeSortingOutput().get_spike_times({"merge_id": ...})`.

Each step is detailed below.

### Single-session sort

```python
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_lab import LabTeam
from spyglass.spikesorting.v2 import initialize_v2_defaults
from spyglass.spikesorting.v2.pipeline import (
    describe_pipeline_presets,
    describe_run,
    describe_sort_groups,
    describe_units,
    plot_sort_group_geometry,
    preflight_v2_pipeline,
    run_v2_pipeline,
)
from spyglass.spikesorting.v2.recording import SortGroupV2

# Replace with the session you've already ingested via insert_sessions.
nwb_file_name = "your_session.nwb"

# One-shot install of every required default Lookup row
# (PreprocessingParameters + ArtifactDetectionParameters + SorterParameters).
initialize_v2_defaults()
LabTeam.insert1(
    {"team_name": "my_team", "team_description": "..."},
    skip_duplicates=True,
)

# Intervals available for this session (the valid interval_list_name values):
IntervalList & {"nwb_file_name": nwb_file_name}

# Build the sort groups (one per shank), then choose one DELIBERATELY after
# reviewing the table + geometry plot -- don't default to the first row.
# (For "sort every shank", use run_v2_pipeline_session below instead.)
SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
sort_groups = describe_sort_groups(nwb_file_name)
plot_sort_group_geometry(nwb_file_name)
sort_groups  # inspect membership, brain_region, and geometry, then:
sort_group_id = ...  # e.g. the hippocampal shank you want to sort

describe_pipeline_presets()

# End-to-end populate + register on the merge table.
run_summary = run_v2_pipeline(
    nwb_file_name=nwb_file_name,
    sort_group_id=sort_group_id,
    interval_list_name="raw data valid times",
    team_name="my_team",
    pipeline_preset="franklab_probe_hippocampus_30khz_ms5_2026_06",
)
# run_summary["merge_id"] is the UNCURATED root curation -- fine for a quick
# look, but for downstream science curate first (auto_curate=True or by hand)
# and key off the curated curation's merge_id (see "Downstream consumers").
merge_id = run_summary["merge_id"]

# Receipt: stages + warnings as explicit rows (a zero-unit sort can't hide in a
# print), then the per-unit sort-time snapshot (n_spikes, firing_rate_hz over
# the observed/artifact-removed duration, peak amplitude, peak channel, region).
describe_run(run_summary)
describe_units(run_summary["sorting_id"])
```

`describe_run(run_summary)` renders the run as a receipt table: a summary row
(`n_units`, `merge_id`), one row per stage (status + `seconds`), and one row per
`warning` — so an easily-missed zero-unit advisory is its own row, not a value
buried in the dict. The underlying `run_summary` dict carries the same data.
Besides the stable keys (`pipeline_preset` / `recording_id` /
`artifact_detection_id` / `sorting_id` / `curation_id` / `merge_id` /
`n_units`), it carries per-stage observability:
`recording_status` / `artifact_detection_status` / `sorting_status` /
`curation_status` (`"computed"` if the stage did work this call, `"reused"` if
its row already existed, or `"skipped"` if the preset configured no such stage —
e.g. `artifact_detection_status` for a no-artifact preset), a `stage_seconds`
dict of wall-clock per stage **this
call** (keys `recording` / `artifact_detection` / `sorting` / `curation`; ≈0
on an idempotent re-run, not cumulative compute), and a `warnings` list (e.g.
a zero-unit advisory). A failed stage raises `PipelineStageError`, which names
the stage and carries the partial run summary of the stages that completed
before it.

### Sort a whole session

A real session has one sort group per shank. Rather than hand-writing the loop,
`run_v2_pipeline_session` runs every (or selected) sort group and returns one
entry per group; `preflight_v2_pipeline_session` is the read-only whole-session
check to run first. Both require an explicit `pipeline_preset` (a whole-session
run infers no default):

```python
from spyglass.spikesorting.v2.pipeline import (
    preflight_v2_pipeline_session,
    run_v2_pipeline_session,
)

# Read-only: one PreflightReport per sort group, aggregated.
report = preflight_v2_pipeline_session(
    nwb_file_name=nwb_file_name,
    interval_list_name="raw data valid times",
    team_name="my_team",
    pipeline_preset="franklab_probe_hippocampus_30khz_ms5_2026_06",
)
assert report.ok, report.errors

results = run_v2_pipeline_session(
    nwb_file_name=nwb_file_name,
    interval_list_name="raw data valid times",
    team_name="my_team",
    pipeline_preset="franklab_probe_hippocampus_30khz_ms5_2026_06",
    sort_group_ids=None,        # None = every sort group; or pass a subset
    continue_on_error=True,     # record per-group failures instead of stopping
)

# One receipt for the whole batch: a summary row with the ok / failed /
# zero-unit / with-warnings counts, then a row per group (and per warning).
describe_run(results)
```

Each entry is the single-group run summary plus `sort_group_id` and an
`outcome` of `"ok"`; a failed group (with `continue_on_error=True`) is
`{"sort_group_id", "pipeline_preset", "outcome": "failed", "error_type",
"error", "partial_run_summary"}`. The runner loops sequentially (`run_v2_pipeline`
already parallelizes the heavy populate internally) and, with
`preflight=True` (default), runs the whole-session preflight once up front:
with `continue_on_error=False` a failed-preflight group raises `PreflightError`
before any compute; with `continue_on_error=True` it is recorded and the
preflight-passing groups still run. `continue_on_error` makes the batch
resilient to per-group preflight/sort failures only — an unexpected error (a
missing Lookup row, a DB-state change) still stops the run.

### Choosing a sort group

`set_group_by_shank` creates the rows; `describe_sort_groups` and
`plot_sort_group_geometry` help you decide which one to run:

```python
from spyglass.spikesorting.v2.pipeline import describe_sort_groups, plot_sort_group_geometry

describe_sort_groups(nwb_file_name)
plot_sort_group_geometry(nwb_file_name)
```

Each row is one `SortGroupV2` group. Check `n_electrodes`, `electrode_ids`,
`electrode_group_names`, `probe_shanks`, `brain_regions`, `bad_channel_count`,
and the reference fields before sorting. The geometry plot colors contacts by
`sort_group_id` using Spyglass probe/electrode metadata, overlays bad-channel
members with red `x` markers, and marks `reference_mode="specific"` electrodes
with a star. Multi-probe sessions are laid out side-by-side along x (one column
per probe, annotated with the `probe_id`), since `Probe.Electrode` coordinates
are per-probe. For real analyses, choose `sort_group_id` intentionally from this
table and plot rather than assuming `0` is the scientifically relevant shank.

Available pipeline presets (all dated `_2026_06`):

- `franklab_probe_hippocampus_30khz_ms5_2026_06` -- **default**,
    MountainSort5 (hippocampus 600 Hz preproc, 30 kHz). It is the shipped
    default because it runs under the v2 `numpy>=2` baseline;
    `recommendation_status` stays `"alternative"` (the default is a
    runnability-driven choice, separate from the scientific tier).
    `franklab_tetrode_hippocampus_30khz_ms5_2026_06` is the same recipe under a
    tetrode label (`probe_type` is informational; both resolve to the same rows).
- `franklab_probe_hippocampus_30khz_ms4_singularity_2026_06` -- the
    **recommended-science** MS4 path on modern (`numpy>=2`) hosts: the same
    MountainSort4 polymer-probe science run inside a pinned Singularity
    container, so the host stays on `numpy>=2` while MS4's `ml_ms4alg` runtime
    lives in the image. Preflight gates it on the container runtime
    (`container_runtime_available`) and never silently falls back to local. A
    Docker row (and other rates) is a user-insertable `SorterParameters` row away
    via the same tracked `execution_params` mechanism.
- `franklab_tetrode_hippocampus_30khz_ms4_2026_06` -- production MountainSort4
    (hippocampus 600 Hz preproc, 30 kHz). **Requires `numpy<2`** for LOCAL
    execution: MS4's `ml_ms4alg` backend does not install under the v2 `numpy>=2`
    baseline, so preflight fails it (`sorter_runtime_available`) unless
    `ml_ms4alg` is present (or use the containerized preset above).
- `franklab_probe_{hippocampus,cortex}_{30khz,20khz}_ms4_2026_06` -- the
    production MS4 family by region (600/300 Hz high-pass) and rate. The local
    polymer recipe `franklab_probe_hippocampus_30khz_ms4_2026_06` stays available
    for compatible local (`numpy<2`) MS4 runtimes; on modern hosts prefer the
    containerized Singularity preset above.
- `franklab_clusterless_2026_06` -- peak-detection only (no clustering), feeds
    the clusterless decoding pipeline
- `franklab_neuropixels_ks4_2026_06` -- **experimental** Neuropixels Kilosort4
    recipe matched to the
    [AIND `aind-ephys-spikesort-kilosort4`](https://github.com/AllenNeuralDynamics/aind-ephys-spikesort-kilosort4)
    config (`nblocks=5` non-rigid drift; KS4 does its own high-pass + CAR +
    whitening, so the signal is whitened exactly once). Community-grounded,
    not Frank-lab-attested; KS4 needs a GPU and is non-deterministic. Because
    KS4 common-references internally, set the sort group's
    `reference_mode="none"` to avoid double-referencing.

For Frank-lab polymer/tetrode rows, the scientific defaults mirror the v1
workflow: sort one group at a time, use a 600 Hz hippocampal high-pass (300 Hz
for cortex), pass already-filtered recordings to MountainSort (`filter=False`),
whiten inside the sorter, use a 100 um adjacency radius, and default to
bidirectional `detect_sign=0` because extracellular polarity can flip with
geometry. The analyzer uses separate waveform rows for display and metrics:
unwhitened waveforms preserve the visible shape/amplitude, while the whitened
metric analyzer supports PC/nearest-neighbour metrics. Hippocampal analyzer
rows intentionally keep the v1-like 0.5/0.5 ms window and sample up to 20000
spikes per unit.

The tetrode- and probe-hippocampus 30 kHz presets resolve to the **same**
parameter rows (the recipe is set by region + rate; `probe_type` is
informational). `list_pipeline_presets()` returns the names at runtime;
`describe_pipeline_presets()` returns a table of what each pipeline preset does
(`recommendation_status`, `target_region`, `sampling_rate_hz`,
`adjacency_radius_um`, sorter, parameter rows, intended use, and the
detection-threshold units) so you can choose one without reading the module
source:

```python
from spyglass.spikesorting.v2.pipeline import describe_pipeline_presets

presets = describe_pipeline_presets()  # one row per pipeline preset
presets

# Discover the shipped Neuropixels / Kilosort4 rows by filtering the catalog,
# rather than hardcoding a name that may be re-dated:
presets[presets["sorter_family"] == "kilosort4"]
```

### Parameter names and fingerprints

Shipped parameter-row names are **stable provenance**, not just labels. The
`*_2026_06` suffix dates the recipe: a row named `franklab_hippocampus_2026_06`
is the June 2026 Frank Lab hippocampus preprocessing recipe, and a future
change ships under a **new** dated name rather than mutating the existing
blob -- so a `recording_id` / `sorting_id` derived from a name stays
reproducible. Two guards keep names honest:

- **Content fingerprints.** Each row has a content fingerprint (the validated
  `params` blob + schema version + job kwargs, with the row *name* excluded;
  `SorterParameters` is scoped per sorter). `describe_parameter_rows()` shows
  every row in the database with its fingerprint, whether it is a shipped
  catalog default, which pipeline presets use it, and -- if its content
  duplicates another row -- the name it duplicates:

  ```python
  from spyglass.spikesorting.v2.pipeline import describe_parameter_rows

  describe_parameter_rows()  # table, parameter_name, fingerprint, usage, ...
  ```

- **Duplicate-content guard.** Inserting a second name for content that already
  ships under a different name raises `DuplicateParameterContentError` (a second
  name for the same blob forks provenance). Pass `allow_duplicate_params=True`
  to opt in -- the row then shows a `duplicate_of` in
  `describe_parameter_rows()`.

### Debugging cookbook

- **Preflight fails before any work starts.** Inspect `report.errors` for the
  blocking fixes and `report.checks` for the full pass/fail list:

  ```python
  report = preflight_v2_pipeline(
      nwb_file_name=nwb_file_name,
      sort_group_id=sort_group_id,
      interval_list_name="raw data valid times",
      team_name="my_team",
      pipeline_preset="franklab_probe_hippocampus_30khz_ms5_2026_06",
  )
  for check in report.checks:
      print(check.name, check.ok, check.fix)
  ```

- **A compute stage fails.** Catch `PipelineStageError`; `err.stage` names the
  failed stage and `err.partial_run_summary` shows which IDs were already
  created.

  ```python
  from spyglass.spikesorting.v2.exceptions import PipelineStageError

  try:
      run_summary = run_v2_pipeline(...)
  except PipelineStageError as err:
      print(err.stage)
      print(err.partial_run_summary)
      raise
  ```

- **The sorter binary is missing.** `preflight_v2_pipeline` checks
  `spikeinterface.sorters.installed_sorters()` and tells you whether to install
  the sorter runtime or pick another pipeline preset.
- **The chosen sort group looks suspicious.** Re-run
  `describe_sort_groups(nwb_file_name)` and verify the electrode count, shank,
  brain region, and reference fields. Recreate groups only after reviewing
  `SortGroupV2.preview_existing_entries(nwb_file_name)`.
- **The sort returns zero units.** By default, `run_v2_pipeline` writes an
  empty-but-real curation and `merge_id`; this is valid for quiet shanks. Pass
  `require_units=True` only when zero units should abort the run.
- **The output is unexpectedly sparse.** Check `run_summary["warnings"]`, the
  chosen pipeline preset's threshold units in `describe_pipeline_presets()`,
  and whether artifact masking removed the interval you expected to sort.

### Curation: quick path vs expert path

`run_v2_pipeline` already creates the initial (root) curation for you. To
curate further, reach for the intent-first wrappers on `CurationV2` instead of
the ten-parameter `insert_curation`:

```python
from spyglass.spikesorting.v2.curation import CurationV2

# Inspect what the pipeline produced (accepts the run summary directly).
CurationV2.summarize_curation(run_summary)

# Record proposed merges WITHOUT applying them (reviewable; units keep ids).
# Branch off the pipeline's root curation via parent_curation_id (unit ids
# 3/7 are illustrative).
prev = CurationV2.propose_merge_curation(
    {"sorting_id": run_summary["sorting_id"]},
    merge_groups=[[3, 7]],
    parent_curation_id=run_summary["curation_id"],
)

# Commit the merges into a new curation (merged unit set is final).
CurationV2.create_merged_curation(
    {"sorting_id": run_summary["sorting_id"]},
    merge_groups=[[3, 7]],
    parent_curation_id=run_summary["curation_id"],
)
```

`create_initial_curation` / `propose_merge_curation` / `create_merged_curation`
are thin sugar over `insert_curation` (the expert API, still available for full
control); they pre-fill `parent_curation_id` / `apply_merge` by name.
`summarize_curation` returns a plain dict (`n_units`, `labels`, `merge_groups`,
`merges_applied`, `is_merge_preview`, `merge_id`, ...) for notebook printing.

### Quality metrics, evaluation, and acceptance (`CurationEvaluation`)

`CurationEvaluation` replaces v1's `MetricCuration` + `BurstPair`. It scores a
**committed** `CurationV2` row in that curation's **own** unit namespace: it
walks the curation's `SortingAnalyzer` extensions to compute SpikeInterface
quality metrics, propose merge suggestions, and propose auto-curation labels. A
merged unit gets SNR / ISI-violation / PC-NN separation recomputed over its
**merged** template -- never inherited from the highest-amplitude contributor.
The proposals are written to NWB; turning them into a committed child
`CurationV2` is an explicit step (`create_curation` / `use_evaluation_labels`).

```python
from spyglass.spikesorting.v2.metric_curation import (
    CurationEvaluation,
    CurationEvaluationSelection,
    QualityMetricParameters,
    AutoCurationRules,
)
from spyglass.spikesorting.v2.curation import CurationV2

# Default Lookup rows are installed by initialize_v2_defaults().
QualityMetricParameters().show_available_metrics()  # SI metric names you can request

# Evaluate a COMMITTED curation (root, label-only, or applied-merge). The root
# curation run_v2_pipeline returns is committed.
curation = {
    "sorting_id": run_summary["sorting_id"],
    "curation_id": run_summary["curation_id"],
}
sel = CurationEvaluationSelection.insert_selection(
    {
        **curation,
        # snr/isi/firing/num_spikes/presence_ratio/amplitude_cutoff/nn_advanced(PCA)
        "metric_params_name": "franklab_default",
        # Frank-lab default: nn_noise_overlap > 0.1 -> noise, isi_violation > 0.02
        # -> reject (the lab's ~2% refractory policy). 'v1_default_nn_noise' (the
        # nn-only rules) and 'similarity_merge' / 'none' remain available.
        "auto_curation_rules_name": "franklab_default_auto_curation_2026_06",
    }
)
CurationEvaluation.populate(sel)

metrics = CurationEvaluation.get_metrics(sel)        # DataFrame, indexed by the curation's unit ids
labels = CurationEvaluation.get_labels(sel)          # {unit_id: [label, ...]}
merges = CurationEvaluation.get_merge_groups(sel)    # [[unit_id, ...], ...] suggestions

# Accept the proposals into a COMMITTED child CurationV2
# (curation_source='curation_evaluation'). Labels are two VISIBLY DIFFERENT
# choices, not a quiet flag:
child = CurationEvaluation().use_evaluation_labels(sel)   # USE the evaluation verdict:
#   child labels == the evaluation's labels; a unit the rules no longer flag
#   loses its stale reject/noise (the default final-metrics path).
child = CurationEvaluation().overlay_evaluation_labels(sel)   # KEEP current labels + add
#   the proposed ones (the manual-curation path).
# Merges are never applied implicitly -- accept them with the action methods:
child = CurationEvaluation().accept_merges(sel, merge_groups=[[u0, u1]])  # commit chosen merges
child = CurationEvaluation().accept_all_suggested_merges(sel)             # commit every suggestion
draft = CurationEvaluation().preview_merges(sel, merge_groups=[[u0, u1]]) # unapplied draft to review
```

**Use the action methods for the normal workflow.** `accept_merges` /
`accept_all_suggested_merges` commit the merged unit set and **inherit** the
curation's existing labels -- they deliberately do NOT apply the pre-merge
evaluation labels (those are in the pre-merge namespace; a label on an absorbed
unit cannot attach to the fresh merged unit). The recommended flow is
accept-merge-then-evaluate: accept the merge, RE-EVALUATE the merged child, then
write final labels with `use_evaluation_labels` / `overlay_evaluation_labels`.
`preview_merges` drafts an unapplied merge for review (a preview row downstream
consumers reject until committed); every merge action requires at least one real
>=2-member group.

`use_evaluation_labels` (default verdict) and `overlay_evaluation_labels` (keep +
add) are deliberately separate methods so the label choice is explicit at the
call site -- `use_evaluation_labels` clears labels the evaluation does not
propose (v1's "final auto-curation writes the full label state", a no-longer-flagged
unit is not silently excluded downstream), while `overlay_evaluation_labels`
retains the curation's current labels. (In a UI these are the "Use Evaluation
Labels" and "Overlay Evaluation Labels" actions.)

The lower-level `create_curation` is the **expert/combined** API (merges + labels
in one call); note its `labels=None` default fetches and applies the
evaluation's pre-merge labels, so prefer the action methods unless you are
deliberately combining surviving-unit labels with a merge. `create_preview_curation`
is the explicit expert draft opt-in behind `preview_merges`.

#### Saving a manual FigURL-style payload

Curation need not come from an evaluation. `CurationV2.save_manual_curation` is
the payload-oriented entry point a manual / web-UI workflow posts to: it takes a
v1/FigURL payload (`labelsByUnit` / `mergeGroups`), a v2 payload
(`labels_by_unit` / `merge_groups`), or already-unpacked `labels=` /
`merge_groups=`, and writes the next `CurationV2` row. `merge_action` makes the
review/commit choice explicit (`"preview"` drafts unapplied merges, `"commit"`
applies them); a v1 association map (`{"1": ["2"], "2": ["3"]}`) is unioned
transitively into full groups (`[[1, 2, 3]]`), matching v1.

```python
from spyglass.spikesorting.v2.curation import CurationV2

child = CurationV2.save_manual_curation(
    {"sorting_id": sid},
    parent_curation_id=root_id,                 # branch off a committed curation
    payload={
        "labelsByUnit": {"3": ["mua"]},         # FigURL spellings
        "mergeGroups": {"5": ["6"]},
    },
    merge_action="commit",                      # apply the merge (vs "preview")
)
```

This is the same labels / merge-groups payload that
`FigPackCuration.fetch_curation_from_uri(uri)` extracts from a FigPack bundle
edited in the browser, so a browser round-trip and a hand-built dict both reach
`CurationV2` through this one entry point.

Notes:

- `metric_names` is validated against the installed SpikeInterface at insert.
  The 0.99 names `nn_isolation` / `nn_noise_overlap` are gone -- request the
  `nn_advanced` PCA metric (with `skip_pc_metrics=False`) and threshold its
  `nn_noise_overlap` output column in a rule.
- `isi_violation` is Spyglass's bounded `count / (n_spikes - 1)` fraction, not
  SI's unbounded `isi_violations_ratio`.
- `AutoCurationRules` is inserted via `insert_rules(master, rule_rows)` (direct
  `insert1` is blocked) so the master row and its ordered rule rows validate
  together.

#### Population QC plot and burst-pair views

`CurationEvaluation.plot_units_qc(sel)` renders the population QC overview --
one histogram per quality metric (NaN dropped) plus a unit-depth scatter
colored by a chosen metric -- the "do these units look reasonable?" companion
to the per-unit `describe_units` table. Pass `axes=...` to embed it in a
custom matplotlib layout; the method returns the axes it drew into. The v1
`BurstPair` notebook workflow is ported onto `CurationEvaluation` as
`plot_correlograms`, `investigate_pair_xcorrel`, `investigate_pair_peaks`, and
`plot_peak_over_time` (reading the analyzer's `correlograms` / `waveforms`
extensions; no separate `BurstPair` table).

**These analyzer-backed plots are for RAW-unit-equivalent curations** (the
root, or a label-only child of one) -- they read the raw sort's display
analyzer, whose unit namespace is the raw sort. On a **merged** curation (or a
label-only child of one) they **raise** rather than silently render the wrong
units; inspect a merged curation through the routed `get_metrics(sel)` /
`get_merge_groups(sel)` accessors (which carry the curation's own namespace), or
run the burst-pair plots on the pre-merge curation. Building these plots over
curation-scoped (merged) analyzers is deferred (it needs the persistent
curation-scoped analyzer cache).

#### The evaluate -> accept -> merge curation flow

Curation is iterative; each child edits its **parent's committed state** (see
"Parent-state composition" below), so a merged-parent unit id is a valid input
and the absorbed raw units are never resurrected:

1. **Evaluate + label.** Run `CurationEvaluation` on the root curation, inspect
   `get_metrics(sel)` / `plot_units_qc(sel)`, then `use_evaluation_labels(sel)` to
   commit the proposed labels into a child.
2. **Manually merge.** Oversplit clusters (MS4/MS5 oversplit and do not track
   drift) need a human merge. Find burst pairs with `plot_by_sort_group_ids` /
   `investigate_pair_xcorrel` / `investigate_pair_peaks`, then commit the merge
   with `CurationV2.create_merged_curation(..., parent_curation_id=child)` (the
   child inherits the parent's labels by default).
3. **Re-evaluate the final child.** Run `CurationEvaluation` on the committed
   merged curation for final metrics -- merged units are scored over their
   merged templates in the curation's own namespace.

A committed root or label-only curation (unit set unchanged from the raw sort)
reuses the cached raw-sort analyzer; a merged curation builds a curation-scoped
temporary analyzer over the merged sorting (cleaned immediately, never cached).

A **preview** curation (`apply_merge=False` with a proposed-but-unapplied merge
group) is a draft, not a final state: `CurationEvaluation` **rejects** it
(evaluating it would score the unmerged preview units). Commit the merge first
(`create_merged_curation` / `insert_curation(..., apply_merge=True)`), then
evaluate that curation.

#### Parent-state composition and label inheritance

A child curation (`parent_curation_id != -1`) composes from its **parent
`CurationV2`** state, not the raw sort: its unit rows, spike trains, labels, and
merge namespace all come from the parent. So a child of a merged parent can
reference the parent's fresh merged unit ids (a further merge picks up from
where the parent left off) and the absorbed raw units are not resurrected.

- **Raw provenance stays queryable.** `CurationV2.MergeGroup` always records the
  original `Sorting.Unit` contributors (a child's parent-namespace contributors
  are expanded through the parent's `MergeGroup`), so "which raw units made this
  kept unit?" is one restriction. The immediate parent operation ("which parent
  units were merged here?") is recorded separately in
  `CurationV2.ParentMergeGroup`.
- **Labels inherit by default.** `insert_curation(..., label_policy="inherit")`
  (the default) starts a child from its parent's labels and overlays the
  supplied `labels` per unit; a committed merge inherits the **union** of its
  contributors' labels, so labels on absorbed contributors do not vanish. Pass
  `label_policy="replace"` to make the supplied labels the whole child state.

### Stage-by-stage (custom pipeline preset)

`run_v2_pipeline` is a convenience wrapper. The underlying stages can be
driven directly when a built-in pipeline preset does not apply:

```python
from spyglass.spikesorting.v2.recording import (
    Recording, RecordingSelection,
)
from spyglass.spikesorting.v2.artifact import (
    ArtifactDetection, ArtifactDetectionSelection,
)
from spyglass.spikesorting.v2.sorting import (
    Sorting, SortingSelection,
)
from spyglass.spikesorting.v2.curation import CurationV2

nwb_file_name = "your_session.nwb"  # same session as above

recording_key = RecordingSelection.insert_selection({
    "nwb_file_name": nwb_file_name,
    "sort_group_id": 0,
    "interval_list_name": "raw data valid times",
    "preprocessing_params_name": "franklab_hippocampus_2026_06",
    "team_name": "my_team",
})
Recording.populate(recording_key)

artifact_detection_key = ArtifactDetectionSelection.insert_selection({
    "recording_id": recording_key["recording_id"],
    "artifact_detection_params_name": "default",
})
ArtifactDetection.populate(artifact_detection_key)

sorting_key = SortingSelection.insert_selection({
    "recording_id": recording_key["recording_id"],
    "sorter": "mountainsort5",
    "sorter_params_name": "franklab_30khz_ms5_2026_06",
    "artifact_detection_id": artifact_detection_key["artifact_detection_id"],
})
Sorting.populate(sorting_key)

curation_key = CurationV2.insert_curation(
    sorting_key=sorting_key,
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

It is **off in the `default` and region preproc rows** and **on in the
`default_neuropixels` preset** -- a blessed Neuropixels recipe
(`bandpass 300-6000 Hz` + phase-shift, `margin_ms=100`). Because Frank-lab smoke
recordings carry no `inter_sample_shift`, `default_neuropixels` materializes
**identically** to the `default` preproc row on them (the phase-shift is
skipped); it
only does work once an acquisition system that ingests `inter_sample_shift` is
used. To use it:

```python
recording_key = RecordingSelection.insert_selection({
    "nwb_file_name": nwb_file_name,
    "sort_group_id": 0,
    "interval_list_name": "raw data valid times",
    "preprocessing_params_name": "default_neuropixels",
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
from spyglass.spikesorting.v2.bad_channels import suggest_bad_channels

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
recording_key = RecordingSelection.insert_selection({
    "nwb_file_name": nwb_file_name,
    "sort_group_id": 0,
    "interval_list_name": "raw data valid times",
    "preprocessing_params_name": "my_interpolate_preset",  # bad_channel_handling="interpolate"
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

# recording_key selects a materialized Recording, e.g. {"recording_id": ...}
DriftEstimate.populate(recording_key)

# Flag high-drift sessions by the summary metric.
(DriftEstimate & "max_abs_displacement_um > 20").fetch("recording_id")
max_drift_um = (DriftEstimate & recording_key).fetch1("max_abs_displacement_um")

# Rehydrate the full SpikeInterface Motion for plotting / inspection.
motion = DriftEstimate().get_motion(recording_key)  # .displacement, .temporal_bins_s, ...
```

The estimate uses a single default preset (`dredge_fast`, stored on the row for
provenance); there is deliberately no parameters Lookup. `dredge_fast` requires
`torch`, so the `spikesorting-v2` extra now installs it. `compute_motion`
localizes peaks spatially, so it consumes the recording's channel locations (the
cached `Recording` carries probe geometry).

To be explicit: populating `DriftEstimate` leaves the upstream `Recording`
untouched — its `content_hash` and the traces from `get_recording` are unchanged.
Applying motion correction is out of scope by design.

### Chronic same-day recordings

When a chronic implant is recorded across several files on the **same day**
(e.g. a run split into multiple epochs, or several short sessions), you can
concatenate the per-member recordings into one continuous, motion-corrected
recording and sort them together. This recovers units that a per-file sort
would split, and it is the default chronic path. For **days/weeks-apart**
sessions the recommended path is *sort-then-match* (sort each session
independently, then match units across them) rather than concatenation;
multi-day concatenation is supported but experimental and gated behind an
explicit opt-in.

The workflow groups *sorting members* — a member is a
`(nwb_file_name, sort_group_id, interval_list_name, team_name)` tuple, not a
whole NWB — and materializes one `ConcatenatedRecording` cache before sorting:

```python
from spyglass.spikesorting.v2.recording import RecordingSelection, Recording
from spyglass.spikesorting.v2.session_group import (
    SessionGroup,
    ConcatenatedRecordingSelection,
    ConcatenatedRecording,
)
from spyglass.spikesorting.v2.sorting import SortingSelection, Sorting

# 1. Materialize each member's Recording first (the concat reuses these caches;
#    it never re-preprocesses raw NWB). All members share ONE preprocessing
#    recipe.
members = [
    {"nwb_file_name": nwb_a, "sort_group_id": 0,
     "interval_list_name": "raw data valid times"},
    {"nwb_file_name": nwb_b, "sort_group_id": 0,
     "interval_list_name": "raw data valid times"},
]
for m in members:
    rec_key = RecordingSelection.insert_selection(
        {**m, "preprocessing_params_name": "default", "team_name": "my_team"}
    )
    Recording.populate(rec_key)

# 2. Name the group. session_group_owner namespaces the group name, so two
#    teams can both use "day1". Same-day is the default; multi-day members
#    require allow_multi_day=True (recording dates are derived from
#    Session.session_start_time, never stored).
SessionGroup.create_group("my_team", "day1", members)

# 3. Materialize the motion-corrected, unwhitened concat cache. preset="auto"
#    maps to rigid_fast for same-day groups; for multi-day it is rejected and
#    you must pick an explicit preset (e.g. dredge_fast).
concat_key = ConcatenatedRecordingSelection.insert_selection({
    "session_group_owner": "my_team",
    "session_group_name": "day1",
    "preprocessing_params_name": "default",
    "motion_correction_params_name": "auto_default",
})
ConcatenatedRecording.populate(concat_key)

# 4. Sort the concatenated recording. The SortingSelection takes a
#    concat_recording_id source instead of a recording_id.
sort_key = SortingSelection.insert_selection({
    "concat_recording_id": concat_key["concat_recording_id"],
    "sorter": "mountainsort5",
    "sorter_params_name": "franklab_30khz_ms5_2026_06",
})
Sorting.populate(sort_key)

# 5. (optional) Back-map the concatenated sort into per-session sortings.
sorting = Sorting().get_analyzer(sort_key).sorting
per_session = ConcatenatedRecording().split_sorting_by_session(sorting, concat_key)
# -> {(nwb_file_name, sort_group_id, interval_list_name, team_name): si.BaseSorting}
#    each member's own sample frame, unit ids preserved.
```

Key behaviors and caveats:

- **Whitening stays at the sorter/analyzer boundary.** The concat cache is
  motion-corrected but *unwhitened*, exactly like a single-session `Recording`;
  MS4/MS5 external whitening and analyzer whitening are unchanged.
- **Parent anchoring.** A concat sort's analysis NWB and each unit's `Electrode`
  FK anchor to the **first** `SessionGroup.Member`. Because of that,
  `get_unit_brain_regions` on a concat sort raises `ConcatBrainRegionAmbiguousError`
  by default; pass `allow_anchor_member=True` to get anchor-member regions
  (labeled `region_resolution="anchor_member"`). For per-session brain regions,
  match the sessions and use `TrackedUnit.get_unit_brain_regions` (see
  [Cross-session unit tracking](#cross-session-unit-tracking)). Downstream
  provenance that is session-scoped — `CurationV2.get_sort_group_info` /
  `SpikeSortingOutput.get_sort_group_info`, and the `(sorter, nwb_file_name)`
  decoding metadata from `CurationV2.get_sort_metadata` — resolves through the
  same anchor member rather than raising.
- **No concat artifact detection.** A concat `SortingSelection` may not carry an
  artifact-detection pass; artifact detection remains a single-recording (or
  shared-recording-group) input.
- **Merge restriction.** `SpikeSortingOutput.get_restricted_merge_ids` accepts
  `concat_recording_id` / `session_group_owner` / `session_group_name` for v2
  concat sorts, alongside the usual sorter / curation fields.

### Cross-session unit tracking

For sessions recorded **days or weeks apart**, the recommended workflow is
*sort-then-match*: sort and curate each session independently, then match units
across sessions to recover the same biological unit over time. This is the
cross-day complement to same-day concatenation — both reuse `SessionGroup`, but
matching pins one curation **per member** and never concatenates the raw data.

**Chronic electrode-space contract.** Matched members should share one physical
electrode space. UnitMatch **hard-rejects** a channel-**geometry** mismatch
across members (the lab-agnostic check), and **warns** when members differ in
electrode *identity* — each sort group's
`(electrode_group_name, electrode_id, brain_region)` signature — since two
distinct probes can share a layout. The identity divergence is a *warning, not a
rejection*: electrode-group names / ids come from each NWB file's
`ElectrodeGroup` and are not guaranteed stable across labs' ingestion, so
blocking on them would reject legitimate chronic matches. If you keep a **stable
electrode-group name across sessions** for a chronic implant the warning stays
quiet; a genuine distinct-probe mix-up also shows up as poor matcher AUC / few
pairs. (Concatenation is stricter — it reads members in one electrode frame, so
a mismatched electrode space is rejected outright.)

Matching is pluggable behind a `MatcherProtocol`. The shipped backend is
[UnitMatch](https://github.com/EnnyvanBeest/UnitMatch) (`matcher="unitmatch"`),
installed via the optional extra:

```bash
pip install -e ".[spikesorting-v2-matching]"   # UnitMatchPy + mat73
```

The **validated path is the 128-channel LLNL polymer probe** (the current
Frank-lab implant); a ground-truth MEArec gate
(`test_v2_unitmatch_polymer_mearec_ground_truth`) requires AUC > 0.85 on a
two-session polymer recording with planted cross-session correspondences. This
gate is verified locally; in CI it runs in the matching-extra environment and is
enforced once the two-session polymer fixtures are uploaded (it skips cleanly
until then — see the fixture URLs in
`tests/spikesorting/v2/fixtures/_fetch.py`).

```python
from spyglass.spikesorting.v2 import initialize_v2_defaults
from spyglass.spikesorting.v2.session_group import SessionGroup
from spyglass.spikesorting.v2.unit_matching import (
    UnitMatchSelection,
    UnitMatch,
    TrackedUnit,
)

# Each session is already sorted + curated (a CurationV2 row per session).
initialize_v2_defaults()  # installs the unitmatch_default MatcherParameters row

# 1. Group the sorted sessions as members (one member per session). For
#    days-apart sessions pass allow_multi_day=True.
members = [
    {"nwb_file_name": nwb_day1, "sort_group_id": 0,
     "interval_list_name": "raw data valid times"},
    {"nwb_file_name": nwb_day2, "sort_group_id": 0,
     "interval_list_name": "raw data valid times"},
]
SessionGroup.create_group("my_team", "implant_week1", members,
                          allow_multi_day=True)

# 2. Pin the EXACT curation used for each member (no implicit "latest"). The
#    keys are member_index -> {"sorting_id": ..., "curation_id": ...}.
selection_key = UnitMatchSelection.insert_selection(
    "my_team", "implant_week1", "unitmatch_default",
    {0: curation_day1, 1: curation_day2},
)

# 3. Run the matcher; UnitMatch.Pair holds the cross-session matches.
UnitMatch.populate(selection_key)
pairs = UnitMatch().get_pairs(selection_key)  # DataFrame of matched unit pairs

# 4. Derive biological-unit identities (one TrackedUnit per matched group).
TrackedUnit.populate(selection_key)
regions = TrackedUnit().get_unit_brain_regions(
    {**selection_key, "tracked_unit_id": 0}
)  # per-session sorting_id / unit_id / region_name for that tracked unit
```

Key behaviors and caveats:

- **Explicit, reproducible curations.** `UnitMatchSelection` pins one
  `(sorting_id, curation_id)` per member via its `MemberCuration` part; there is
  no implicit "latest curation" lookup, so a match run is reproducible even if a
  source session gains new curations later. `insert_selection` verifies each
  pinned curation actually belongs to its member, and `UnitMatch.make()`
  re-checks that provenance (raising `UnitMatchSelectionIntegrityError`) so a
  direct-insert bypass cannot silently match the wrong units.
- **The matcher never sees Spyglass internals.** `UnitMatch.make()` extracts a
  dense split-half waveform bundle per session from the curated recording +
  sorting and hands the matcher self-contained directories — never a recording,
  a `SortingAnalyzer`, or a table key. A new backend implements `MatcherProtocol`
  and registers via `register_matcher()`.
- **Tracked units are a strict partition.** `TrackedUnit` groups units that
  match *every* other member of the group, derived as a greedy maximal-clique
  cover of the pair graph (largest clique first, ties broken by highest median
  edge probability) so each curated unit belongs to exactly one tracked unit —
  overlapping cliques never duplicate a unit across identities. If A↔B and B↔C
  match but A↔C does not, A and C land in different tracked units. A unit with no
  matches surfaces as a singleton (`n_sessions_observed == 1`,
  `median_match_probability` NULL). The graph size is bounded by
  `max_strict_nodes` (default 2000); a larger universe raises
  `TrackedUnitBudgetExceededError`.
- **Per-session brain regions.** `TrackedUnit.get_unit_brain_regions` resolves
  each member unit's `Electrode -> BrainRegion` and labels it by session — the
  per-session resolver the concat-sort guard points to.
- **Degenerate single-session.** A one-member group produces zero pairs and no
  matcher call.

### Downstream consumers

Both v1 (`CurationV1`) and v2 (`CurationV2`) curations register on the same
`SpikeSortingOutput` merge table, so existing downstream code (decoding,
ripple detection, etc.) keeps working unchanged. **`run_summary["merge_id"]`
is the uncurated root curation** — for downstream science, curate first and
carry the curated curation's `merge_id` forward: `auto_summary["auto_merge_id"]`
from an `auto_curate=True` run, or the `merge_id` of the curation you build by
hand (see the [evaluate → accept → merge curation
flow](#the-evaluate-accept-merge-curation-flow)). Pass whichever `merge_id` you
choose to the accessors below:

#### What do I call next?

| Goal | Call |
| --- | --- |
| Spike times | `SpikeSortingOutput().get_spike_times({"merge_id": merge_id})` |
| Recording | `SpikeSortingOutput().get_recording({"merge_id": merge_id})` |
| Sorting | `SpikeSortingOutput().get_sorting({"merge_id": merge_id})` |
| Unit brain regions | `SpikeSortingOutput.get_unit_brain_regions({"merge_id": merge_id})` |
| Curation summary | `CurationV2.summarize_curation(run_summary)` |
| Analyzer/debug internals | `Sorting().get_analyzer({"sorting_id": run_summary["sorting_id"]})` |

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
feature (used by clusterless decoding), `full_waveform`, and `spike_location`
are supported for v2 sources; any other feature is rejected with a clear
`NotImplementedError`. The `spike_location` row is v2-only; legacy v0/v1
clusterless workflows should keep using the `amplitude` row. A zero-unit v2
curation yields
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

### Provenance in each v2 NWB

Every v2 analysis NWB is **self-describing**: it embeds the lineage needed to
interpret it without the DataJoint database, so a shared / DANDI'd file stands
on its own. The provenance lives in NWB **scratch** under stable, name-addressed
containers — read them with `nwbfile.get_scratch(name)`, which returns a
DataFrame (a scalar header is a two-column `key` / `value_json` table whose
values are JSON-encoded; a relational table has one row per member / unit / pair).
The large
arrays are **not** duplicated: the recording's content fingerprint, the motion
displacement field, and waveform templates stay DB-derivable; only the producing
params are written.

| Artifact | Container(s) | Carries |
| --- | --- | --- |
| Recording | `spyglass_v2_recording_provenance` | raw source `object_id`, `recording_id`, preprocessing recipe, sort group, resolved reference mode, bad-channel handling, SpikeInterface version |
| Sorting | `spyglass_v2_sorting_provenance` + per-unit Units columns | `peak_amplitude_uv` / `peak_electrode_id` / `n_spikes` / `brain_region` columns (matching `Sorting.Unit`), and a header with the recording/concat id, sorter + params, `artifact_detection_id`, display recipe, effective seed, SI + sorter versions |
| Curated units | `spyglass_v2_curation_provenance` + `spyglass_v2_curation_merge_lineage` | curation header (sorting/curation id, parent, source, `merges_applied`, description) and the kept→contributor merge lineage mirroring `CurationV2.MergeGroup` (raw contributors; proposed-vs-applied is the header's `merges_applied`) |
| UnitMatch | `spyglass_v2_unitmatch_provenance` + `spyglass_v2_unitmatch_members` | run/group/matcher header (matcher backend + versions) and the per-member `(sorting_id, curation_id, session_start_time)` map |
| CurationEvaluation | `spyglass_v2_curation_evaluation_provenance` | metric set + recipe names, auto-merge preset/rules, evaluated curation, the `source_analyzer_hashes` manifest, SI version, upstream recording/concat `content_hash` |
| ConcatenatedRecording | `spyglass_v2_concat_provenance` + `spyglass_v2_concat_members` | resolved motion preset **+ kwargs** (not the displacement field) and the ordered member map with per-member frame boundaries (`split_sorting_by_session` is reconstructable from these) |

```python
import pynwb
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.v2.sorting import Sorting

abs_path = AnalysisNwbfile.get_abs_path(
    (Sorting & {"sorting_id": sorting_id}).fetch1("analysis_file_name")
)
with pynwb.NWBHDF5IO(abs_path, "r", load_namespaces=True) as io:
    prov = io.read().get_scratch("spyglass_v2_sorting_provenance")
    # prov is a DataFrame of (key, value_json); values are JSON-encoded.
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

## Capabilities at a glance

The single-session sort chain, analyzer-driven curation (metrics +
auto-curation, above), same-day chronic concatenate-and-sort (the
[Chronic same-day recordings](#chronic-same-day-recordings) section below), and
cross-session unit matching ([Cross-session unit tracking](#cross-session-unit-tracking))
all run end-to-end through `run_v2_pipeline` / `run_v2_unit_match` and the
underlying tables.

FigPack curation is offline by default: `FigPackCuration` (and
`run_v2_pipeline(..., figpack=True)`, which forces `upload=False`) builds a
self-contained local bundle you open in a browser to label and merge units.
Edits to a local bundle are not written back automatically —
`FigPackCuration.fetch_curation_from_uri(uri)` reads the edited labels / merge
groups out of it, and `CurationV2.save_manual_curation` ingests them into the
next curation (see
[Saving a manual / FigURL-style payload](#saving-a-manual-figurl-style-payload)).
The lower-level `FigPackCurationSelection.insert_selection(..., upload=True)`
can instead publish a hosted figpack.org figure (needs `FIGPACK_API_KEY`, or
`ephemeral=True` for a temporary one); hosted publish is for sharing a view of
an uncurated root curation, while the local bundle is the path that round-trips
edited labels back into Spyglass. FigPack curation needs the
`spikesorting-v2-curation` extra
(`pip install -e ".[spikesorting-v2-curation]"`); cross-session matching needs
the `spikesorting-v2-matching` extra.

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
  pandas DataFrame indexed by `unit_id` with a `spike_times`
  (seconds) column; the CurationV2 form also joins the
  `curation_label` list column from `UnitLabel`.
- `get_spiking_sorting_v2_merge_ids(restriction, as_dict=False)` in
  `spyglass.spikesorting.v2.utils` is the notebook-discoverable
  parallel of v1's `get_spiking_sorting_v1_merge_ids`.
- `SpikeSortingOutput.get_restricted_merge_ids` defaults to every
  available source (`v0`/`v1`, plus `v2` when the v2 module is
  importable), so v2 users copying v1 notebook patterns see v2
  merge_ids without an explicit `sources=` arg, while v0/v1-only
  deployments are unaffected. With an explicit `sources=` list the v2
  resolver is strict — an unknown restriction key raises `ValueError`;
  the default (no `sources=`) stays lenient, since a key meant for v0/v1
  is not a v2 typo.

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
