# Spike Sorting v2

## Why

`spyglass.spikesorting.v2` is the spike-sorting pipeline for SpikeInterface
0.104, built on the SI `SortingAnalyzer` API. It keeps Spyglass's DataJoint
contracts (Selection / make / merge dispatch, cascade-safe cautious deletes,
analysis-NWB lifecycle). Under SpikeInterface 0.104 the v0/v1 pipelines read
existing rows but raise a `RuntimeError` when asked to produce new output; see
[Environment](#environment).

## What

v2 ships the single-session sorting chain plus same-day concatenation and
cross-session unit matching:

```
Recording preparation:
  SortGroupV2 -> RecordingSelection -> Recording (bandpass + reference)

Artifact choices for single-recording sorting:
  Recording -> RecordingArtifactSelection -> RecordingArtifactDetection
  SharedArtifactGroup -> SharedGroupArtifactSelection -> SharedGroupArtifactDetection
  Both detection types register in ArtifactDetectionOutput.

Single-recording path:
  Recording + optional ArtifactDetectionOutput
    -> SortingSelection -> Sorting -> CurationV2
    -> SpikeSortingOutput.CurationV2

Concat path:
  Member Recordings + explicit per-member RecordingArtifactDetection choices
    -> ConcatenatedRecordingSelection.MemberSnapshot
    -> ConcatenatedRecording (mask -> concatenate; no motion correction)
    -> SortingSelection (concat source) -> Sorting -> CurationV2
    -> ConcatMemberCuration (original session timestamps) -> SpikeSortingOutput

Optional motion stage (either path's Recording or ConcatenatedRecording; see
"Optional motion correction" below):
  Recording | ConcatenatedRecording [+ ArtifactDetectionOutput, single source only]
    -> MotionEstimateSelection -> MotionEstimate
    -> MotionCorrectedRecordingSelection -> MotionCorrectedRecording
    -> SortingSelection.MotionCorrectionSource

Both curation paths:
  CurationV2 -> CurationEvaluationSelection -> CurationEvaluation
```

`SortingSelection` takes its recording source through one of two
mutually-exclusive source part tables -- `RecordingSource` (a single-session
`Recording`) or `ConcatenatedRecordingSource` (a same-day chronic
`ConcatenatedRecording`). Single-recording sorts can add an artifact-detection
pass through the internal `ArtifactDetectionOutput` merge. Concat sources
already contain their frozen member masks, so they accept no additional
sorting-stage artifact input. A sort can optionally add a third source part,
`MotionCorrectionSource`, pointing at a `MotionCorrectedRecording` computed from
the same base source and mask -- see
[Optional motion correction](#optional-motion-correction).

Recording access stays `SpikeSortingOutput.get_recording`, whose meaning depends
on the sort. Two accessors have one meaning each, on `CurationV2`,
`ConcatMemberCuration` and `SpikeSortingOutput`: `get_source_recording` is the
original `Recording` cache (unmasked, uncorrected; a concat-backed `CurationV2`
raises and points at each member's), and `get_sorting_input_recording` is the
traces the sorter read (masked, corrected when selected; a concatenation member
gets the parent's traces over its frames on the member's own timestamps).

| Sort                        | `get_recording` returns       | Masked | Corrected | Clock            |
| --------------------------- | ----------------------------- | ------ | --------- | ---------------- |
| single recording            | `get_source_recording`        | no     | no        | acquisition      |
| single recording, corrected | `get_sorting_input_recording` | yes    | yes       | acquisition      |
| concatenation               | `get_sorting_input_recording` | yes    | no        | synthetic concat |
| concatenation, corrected    | `get_sorting_input_recording` | yes    | yes       | synthetic concat |
| concatenation member (any)  | `get_source_recording`        | no     | no        | member recording |

A single-recording sort that pins an artifact detection is still unmasked
through `get_recording`.

All v2 tables live in dedicated `spikesorting_v2_*` DataJoint schemas
(`recording`, `artifact`, `artifact_output`, `sorting`, `curation`,
`metric_curation`, `review_profile`, `concat_curation`, `figpack_curation`,
`session_group`, `unit_matching`, `unit_annotation`, `recompute`, `motion`), so
the v0/v1 schemas are untouched. `CurationV2` and its session-aligned
`ConcatMemberCuration` outputs register as parts on the existing
`SpikeSortingOutput` merge table, so v0, v1, imported, and v2 curations coexist
under one merge surface.

### Tables

For import paths, see the [API map](./SpikeSortingV2_API.md#tables).

- **`SortGroupV2`** -- per-session electrode grouping. `set_group_by_shank` and
    `set_group_by_electrode_table_column` refuse to overwrite existing sort
    groups unless called with `delete_existing_entries=True, confirm=True`.
    Review `SortGroupV2.preview_existing_entries(nwb_file_name)` (the
    `DeletionPreview` of the cascade) first; `confirm=False` raises a
    `ValueError` that embeds the same preview. There is no `test_mode`
    short-circuit: to add groups next to existing ones, pass explicit,
    non-overlapping `sort_group_ids=`.
- **`PreprocessingParameters`, `ArtifactDetectionParameters`,
    `SharedArtifactGroup`, `SorterParameters`, `QualityMetricParameters`,
    `AutoCurationRules`** -- Pydantic-validated parameter Lookup rows.
    `insert_default()` on each loads a default row; user params validate the
    `params` blob on insert.
- **`RecordingSelection` / `Recording`** -- preprocessed recording
    materialization. Stage order: optional ADC phase-shift and bandpass, both on
    the continuous channel-sliced source (so the filter's margin comes from real
    adjacent samples, not from the joins between selected intervals), then the
    restriction to the selected intervals, bad-channel interpolation, and
    referencing. Whitening is deferred to the sort stage, so motion correction
    never sees whitened data. Electrode geometry is normalized to the first
    coordinate plane in which every contact is distinct (x-y, else x-z, else
    y-z; including the `tetrode_12.5` repair) and persisted in the artifact's
    `rel_x`/`rel_y`/`rel_z` electrode rows; a sort group whose contacts still
    share a 2D position raises. `make` raises `RecordingTruncatedError` if the
    raw timestamps do not span the requested interval. Specific and global-median
    referencing require uniform finite channel offsets and uniform finite positive
    gains across the contributing channels, including a specific reference
    electrode. Referencing raises when calibration differs between channels:
    subtracting raw counts would change the result in physical units. This check
    uses the current preprocessing calibration; a bandpass filter already clears
    offsets before the reference stage.
- **`DriftEstimate`** -- per-`Recording` probe-motion QC estimate, populated on
    demand; never applied. See
    [Drift QC](#drift-qc-motion-estimate-never-applied).
- **`RecordingArtifactSelection` / `RecordingArtifactDetection`** and
    **`SharedGroupArtifactSelection` / `SharedGroupArtifactDetection`** --
    amplitude-threshold artifact intervals over a single `Recording` or a
    `SharedArtifactGroup` (several recordings in a session sharing one result,
    e.g. chewing artifacts visible on every probe). Both write their
    artifact-removed valid times to `common.IntervalList` (owned through a
    `RemovedInterval` part) and register themselves in `ArtifactDetectionOutput`
    at materialization.
- **`ArtifactDetectionOutput`** -- internal merge over the two artifact result
    tables; `SortingSelection.ArtifactDetectionSource` carries one optional
    foreign key to it, and user code never imports it. Deleting a result is
    refused while a `SortingSelection` or a frozen concat member references it
    (use `cascade_delete` to remove the dependent outputs too). Concat members
    reference `RecordingArtifactDetection` directly.
- **`SortingSelection` / `Sorting`** -- runs the configured sorter over the
    selection's recording source part. Dispatches `clusterless_thresholder`
    (peak detection only) vs the SI sorter registry (`mountainsort4`,
    `mountainsort5`, ...). Masked samples read 0 µV: when the sort masks frames,
    a source with a nonzero channel offset (only possible with the `no_filter`
    recipe and `reference_mode="none"`) is first converted to float32 µV (gain
    1, offset 0); otherwise traces keep their stored units. The `Unit` part
    stores per-unit summary stats (`n_spikes`, `peak_amplitude_uv`).
- **`MotionEstimationParameters` / `MotionInterpolationParameters` /
    `MotionCorrectionParameters`, `MotionEstimateSelection` / `MotionEstimate`,
    `MotionCorrectedRecordingSelection` / `MotionCorrectedRecording`** -- the
    optional motion stage. See
    [Optional motion correction](#optional-motion-correction).
- **`CurationV2`** -- versioned curation rows (labels + merge groups) chained by
    `parent_curation_id`. `insert_curation` is the single entry point; each
    inserted generation has an immutable, database-unique `curation_uuid` (the
    numeric `curation_id` can be reused after deletion). Single-recording
    curations register on `SpikeSortingOutput.CurationV2`; concat curations
    instead produce `ConcatMemberCuration` outputs with one merge ID per member.
- **`CurationEvaluationSelection` / `CurationEvaluation`** -- quality metrics,
    auto-curation labels, merge suggestions, and burst-pair views over a
    **committed** `CurationV2` row, scored in that curation's own unit
    namespace. Preview/draft curations are rejected. See
    [Quality metrics](#quality-metrics-and-the-scripted-evaluatemerge-loop).
- **`CurationReviewProfile`** -- one immutable name binding the exact
    quality-metric and auto-curation recipes to an ordered property display,
    label palette, and explicit `replace`/`overlay` import mode.
    `initialize_v2_defaults()` installs `franklab_hippocampus_2026_09_17`.
- **`RecordingArtifactRecompute*` / `SortingAnalyzerRecompute*`** -- storage
    verification for reclaiming recording/artifact NWBs and analyzer folders;
    see [Storage Management](./SpikeSortingV2StorageManagement.md).
- **`SessionGroup`** -- a named bundle of sorting members, used by concatenation
    and cross-session matching.
- **`MatcherParameters`** -- validated cross-session matcher configuration.
- **`UnitMatchSelection` / `UnitMatch` / `TrackedUnit`** -- pin an ordered set
    of independently curated matching inputs, match units across them, and
    derive biological-unit identities. See
    [Cross-session unit tracking](#cross-session-unit-tracking).

### Pipeline orchestrator

`spyglass.spikesorting.v2.pipeline.run_v2_pipeline` chains each stage's
`insert_selection` + `populate` into one call. It is idempotent: re-running with
the same inputs returns the same run summary (same `root_merge_id`, same
intermediate keys) without duplicating rows. `list_pipeline_presets()` returns
the preset names; `describe_pipeline_presets()` returns a table of what each
preset does (`recommendation_status`, `target_region`, `sampling_rate_hz`,
`adjacency_radius_um`, sorter, parameter rows, intended use, and the
detection-threshold units):

```python
from spyglass.spikesorting.v2.pipeline import describe_pipeline_presets

presets = describe_pipeline_presets()  # one row per pipeline preset
presets

# Filter the catalog rather than hardcoding a name that may be re-dated:
presets[presets["sorter_family"] == "kilosort4"]
```

The shipped presets (all dated `_2026_06`):

- `franklab_probe_hippocampus_30khz_ms4_2026_06` -- **default**, native
    MountainSort4 (hippocampus 600 Hz high-pass, 30 kHz), with
    `recommendation_status="production"`. The tetrode-labeled MS4 preset uses
    the same parameter rows (`probe_type` is informational).
- `franklab_probe_hippocampus_30khz_ms5_2026_06` -- MountainSort5
    alternative for the same region and rate, with
    `recommendation_status="alternative"`.
- `franklab_probe_hippocampus_30khz_ms4_singularity_2026_06` -- the
    container execution option for the production MountainSort4 recipe: MS4
    runs in a pinned Singularity container. Preflight gates it on
    `container_runtime_available` and never falls back to a local run. A Docker
    row (or other rates) is a user-inserted `SorterParameters` row using the
    same `execution_params` mechanism.
- `franklab_tetrode_hippocampus_30khz_ms4_2026_06` and
    `franklab_probe_{hippocampus,cortex}_{30khz,20khz}_ms4_2026_06` --
    production MountainSort4 by region (600/300 Hz high-pass) and rate. Native
    execution is included in `spyglass-neuro[spikesorting-v2]` and supports the
    standard NumPy-2 environment. The package bundles the algorithm; a separate
    `ml_ms4alg` install is unnecessary. V2 handles the old `spikeextractors`
    `np.Inf` alias during sorting, and preflight imports `mountainsort4` to
    check its runtime dependencies.
- `franklab_clusterless_2026_06` -- peak detection only (no clustering), for the
    clusterless decoding pipeline.
- `franklab_neuropixels_ks4_2026_06` -- **experimental** Neuropixels Kilosort4
    recipe matched to the
    [AIND `aind-ephys-spikesort-kilosort4`](https://github.com/AllenNeuralDynamics/aind-ephys-spikesort-kilosort4)
    config (`nblocks=5` non-rigid drift; KS4 does its own high-pass + CAR +
    whitening, so the signal is whitened exactly once). Not Frank-lab-attested;
    KS4 needs a GPU and is non-deterministic. Set the sort group's
    `reference_mode="none"` to avoid double-referencing. It selects no Spyglass
    artifact masking, and Kilosort's internal drift correction is not artifact
    rejection.

The Frank-lab polymer/tetrode presets mirror the v1 workflow: sort one group at
a time, 600 Hz hippocampal (300 Hz cortical) high-pass, already-filtered input
to MountainSort (`filter=False`), whitening inside the sorter, a 100 µm
adjacency radius, and downward-only `detect_sign=-1`. For bidirectional
detection, clone the preset with `clone_pipeline_preset` and set the sorter
override `detect_sign` to `0`. The analyzer keeps separate waveform rows for
display (unwhitened, preserving shape/amplitude) and metrics (whitened, for
PC/nearest-neighbour metrics); hippocampal rows use a 0.5/0.5 ms window and
sample up to 20000 spikes per unit.

#### Reading `recommendation_status`

- **`production`** -- Frank Lab validated and recommended; the default choice
    for its probe / target region / sampling rate.
- **`alternative`** -- a sound substitute when the production recipe does not
    fit (e.g. MountainSort5 as an alternative to the MountainSort4 recipe).
- **`experimental`** -- not yet validated on Frank Lab data; inspect the output
    before relying on it (e.g. multi-day concatenation, Neuropixels Kilosort4).

`describe_recommendation_status()` returns this legend as a table.

## Security & trust model

Spike Sorting v2 assumes a **trusted compute-operator** deployment, the same
model as the rest of Spyglass:

- **Whoever can write `SorterParameters` (or ingest sessions) is a trusted
    operator.** A `SorterParameters` row's `execution_params` can pull and run a
    container image (Docker / Singularity) to execute a sorter, so inserting
    parameter rows is equivalent to running code on the compute host. Restrict
    write access accordingly.
- **The database is not internet-facing.**
- **`team_name` is a provenance tag, not access enforcement.** Sort groups in
    one session can belong to different teams, so overwriting a session's sort
    groups can cascade-delete another team's downstream rows.
    `SortGroupV2.preview_existing_entries` enumerates that blast radius for
    review; it does not block.

Materialized analysis artifacts are written owner-writable (`0o644`), the sorter
scratch is world-writable only for a container backend, and caller-supplied NWB
file names are confined to a bare basename before any directory join.

## How

### Run your first single-session sort

Start with the [Quickstart](./SpikeSortingV2_Quickstart.md) or the notebook
[`10_Spike_SortingV2.ipynb`](../notebooks/10_Spike_SortingV2.ipynb) (the first
sort on one ingested session). The companion notebooks cover
[curation](../notebooks/10_Spike_SortingV2_Curation.ipynb),
[presets and whole-session sorting](../notebooks/10_Spike_SortingV2_Presets.ipynb),
and
[concatenation and cross-session matching](../notebooks/10_Spike_SortingV2_CrossSession.ipynb).
The sections below are the reference for each step.

### Single-session sort

The examples below assume `initialize_v2_defaults()` has run, a `LabTeam` named
`my_team` exists, and `nwb_file_name` / `sort_group_id` were chosen after
reviewing bad channels, references and sort groups as in
[Quickstart step 1](./SpikeSortingV2_Quickstart.md#1-review-channels-and-references-then-create-sort-groups)
(see also [Choosing a sort group](#choosing-a-sort-group)).

```python
from spyglass.spikesorting.v2.pipeline import (
    describe_run,
    describe_units,
    preflight_v2_pipeline,
    run_v2_pipeline,
)

run_kwargs = dict(
    nwb_file_name=nwb_file_name,
    sort_group_id=sort_group_id,
    interval_list_name="raw data valid times",
    team_name="my_team",
    pipeline_preset="franklab_probe_hippocampus_30khz_ms4_2026_06",
)
report = preflight_v2_pipeline(**run_kwargs)
print(report.summary())  # blockers, warnings, stages to compute/reuse

run_summary = run_v2_pipeline(**run_kwargs)
# root_merge_id is the UNCURATED root, for quick inspection only. No merge id
# is a filtered unit set: hand a curation to analysis with
# select_units_for_analysis (see "Downstream consumers").
root_merge_id = run_summary["root_merge_id"]

describe_run(run_summary)  # stages + warnings as rows
describe_units(run_summary["sorting_id"])  # per-unit sort-time snapshot
```

`describe_run(run_summary)` renders the run as a receipt: a summary row
(`n_units`, `root_merge_id`, `auto_labeled_merge_id`, `"root only"` /
`"auto-curated"` status), one row per stage (status + `seconds`), and one row
per warning, so a zero-unit advisory cannot hide. It and preflight also show
reference settings, preprocessing parameters, artifact settings, and motion
treatment alongside the effective sorter configuration. The underlying dict
carries the stable ids (`pipeline_preset` / `recording_id` /
`artifact_detection_id` / `sorting_id` / `root_curation_id` / `root_merge_id` /
`auto_labeled_curation_id` / `auto_labeled_merge_id` / `n_units`), per-stage
`*_status` (`"computed"`, `"reused"`, or `"skipped"` when the preset configures
no such stage), `stage_seconds` for **this call** (≈0 on an idempotent re-run),
and `warnings`. `describe_units` uses the observed (artifact-removed) duration
for its firing-rate denominator.

### Choosing a sort group

`describe_sort_groups` returns one row per `SortGroupV2` group. Check
`n_electrodes`, `electrode_ids`, `electrode_group_names`, `probe_shanks`,
`brain_regions`, `bad_channel_count`, and the reference fields before sorting.
`plot_sort_group_geometry` colors contacts by `sort_group_id`, marks bad
channels with red `x` markers and `reference_mode="specific"` electrodes with a
star, and lays out multi-probe sessions side by side (one column per probe).
Choose `sort_group_id` intentionally rather than assuming `0` is the relevant
shank.

### Sort a whole session

`run_v2_pipeline_session` runs every (or selected) sort group and returns one
entry per group; `preflight_v2_pipeline_session` is the read-only whole-session
check. Both require an explicit `pipeline_preset`. Run the preflight first,
freeze its resolved list
(`[row["sort_group_id"] for row in report.group_reports]`) and pass it as
`sort_group_ids=` so the checked and executed groups match, as in section 3 of
the [Presets notebook](../notebooks/10_Spike_SortingV2_Presets.ipynb).
`describe_run(results)` renders the batch as one receipt (a summary row with ok
/ failed / zero-unit / with-warnings counts, then a row per group and warning).

Each entry is the single-group run summary plus `sort_group_id` and
`outcome="ok"`; a failed group (with `continue_on_error=True`) is
`{"sort_group_id", "pipeline_preset", "outcome": "failed", "error_type", "error", "partial_run_summary"}`.
Groups run sequentially. With `preflight=True` (default) the whole-session
preflight runs once up front: with `continue_on_error=False` a failing group
raises `PreflightError` before any compute; with `True` it is recorded and the
passing groups still run. `continue_on_error` covers per-group preflight/sort
failures only; an unexpected error (a missing Lookup row, a DB-state change)
still stops the run.

### Parameter names and fingerprints

Shipped parameter-row names are stable provenance. The `*_2026_06` suffix dates
the recipe; a change ships under a **new** dated name rather than mutating the
existing blob, so a `recording_id` / `sorting_id` derived from a name stays
reproducible.

- **Content fingerprints.** Each row has a fingerprint of its validated `params`
    blob + schema version + job kwargs (name excluded; `SorterParameters` is
    scoped per sorter). `describe_parameter_rows()` lists every row with its
    fingerprint, whether it is a shipped default, which presets use it, and the
    name it duplicates, if any.
- **Duplicate-content guard.** Inserting a second name for content that already
    exists raises `DuplicateParameterContentError`. Pass
    `allow_duplicate_params=True` to opt in; the row then shows a `duplicate_of`
    in `describe_parameter_rows()`.

### Debugging cookbook

- **Preflight fails before any work starts.** `report.errors` lists the blocking
    fixes and `report.checks` the full pass/fail list (`check.name`, `check.ok`,
    `check.fix`).

- **A compute stage fails.** `PipelineStageError` (from
    `spyglass.spikesorting.v2.exceptions`) names the stage (`err.stage`) and
    carries the partial run summary (`err.partial_run_summary`). Correct the
    cause and rerun with the same inputs to reuse completed stages. A failed
    sorter restarts its stage; it does not resume an internal checkpoint. The
    same applies to a whole-session batch.

- **The sorter binary is missing.** Preflight checks
    `spikeinterface.sorters.installed_sorters()` and says whether to install the
    sorter runtime or pick another preset.

- **The chosen sort group looks suspicious.** Re-run
    `describe_sort_groups(nwb_file_name)`; recreate groups only after reviewing
    `SortGroupV2.preview_existing_entries(nwb_file_name)`.

- **The sort returns zero units.** By default `run_v2_pipeline` writes an
    empty-but-real curation and `merge_id`, which is valid for quiet shanks.
    Pass `require_units=True` only when zero units should abort the run.

- **The output is unexpectedly sparse.** Check `run_summary["warnings"]`, the
    preset's threshold units in `describe_pipeline_presets()`, and whether
    artifact masking removed the interval you expected to sort.

### Browser-first curation review

The normal hands-on workflow is one profile-backed review. The profile binds the
exact evaluation recipes, ordered metric columns, label palette, and label
import mode. The review evaluates or reuses the requested curation, seeds its
current labels, and builds a FigPack view over that curation's actual analyzer.
Its **unit table** carries the profile's metrics, any selected annotation
columns, the rule set's `proposed_labels` / `proposed_merge_groups`, and
`merged_from`; selecting a row selects that unit for curation. These values
describe the **committed curation under review**: a merge proposed in the
browser has no merged metrics until it is committed and the continued review
shows them. The browser walkthrough is in the
[Quickstart](./SpikeSortingV2_Quickstart.md#3-review-it-in-the-browser-and-reopen-the-review-later);
the equivalent scripted calls are:

```python
review = run_summary.start_review(
    source="root",  # use "auto_labeled" only when the run produced one
    profile="franklab_hippocampus_2026_09_17",
    upload=False,  # local seeded bundle; True publishes the same bundle
)
url = review.open()  # serves review.uri at http://localhost:<port>/bundles/<id>/

# In the browser: select units -> labels / Merge Selected -> Save draft.
# Local browser: Preview and commit -> inspect child -> record review, then
# final_curation = review.result(). Notebook alternative: review.commit_panel().

# Preview is a pure read of the saved annotations.json: no rows or files change.
changes = review.preview_import()
print(changes.summary())  # counts, label +/- per unit, merges, conflicts
changes.changed_units()  # one row per changed / merged unit

# If contributors have incompatible labels, resolve every predicted merged id.
receipt = changes.commit(
    conflict_resolutions={12: ("accept",)},
)

# A merge is re-evaluated with the same profile; the merged child is NOT the
# result until you have inspected it. open() does not wait: stop here.
pending_verification = None
if receipt.needs_merge_verification:
    pending_verification = receipt.continue_review()
    pending_verification.open()
    final_curation = None  # pending
else:
    final_curation = receipt.curation
```

Commit the verification in a later cell, after inspecting; if that commit
imports another merge, the result stays pending and the block is run again:

```python
from spyglass.spikesorting.v2.pipeline import FigPackReview

if pending_verification is not None:
    continuation = FigPackReview.resume(pending_verification.review_id)
    verification = continuation.preview_import()
    verification_receipt = verification.commit(
        confirm_no_changes=not verification.has_changes
    )
    if verification_receipt.needs_merge_verification:
        pending_verification = verification_receipt.continue_review()
        pending_verification.open()  # inspect, then run this block again
    else:
        pending_verification = None
        final_curation = verification_receipt.curation
```

Only once nothing is pending (`final_curation is not None`) is the result usable
downstream:

```python
final_merge_id = final_curation.merge_id
member_merge_ids = final_curation.member_merge_ids  # concat-backed sorts
```

`review.result()` follows the recorded verification chain and never guesses from
the latest child. In the local browser, **Review parent branch** reopens the
exact pre-merge review for recovering a mistaken merge (see below).

**Local serving.** `review.open()` starts (or reuses) one loopback server per
Python process. Each saved bundle has its own URL path and `annotations.json`,
so a browser save writes the exact draft the importer reads; the server accepts
writes only to that file. The port is process state, never persisted: after a
kernel restart, `FigPackReview.resume(review_id).open()` serves the same files
with every saved edit. `open(open_browser=False, port=...)` returns the URL for
a notebook, a test, or a remote kernel; remote use (port forwarding,
`localhost`-only editing) is described in the Quickstart. A missing bundle
raises with the recovery step (start the review again). If another tab saved the
same local draft, an older tab's save is rejected without discarding its edits;
use **Open latest draft in a new tab** and reapply them there. A reopened local
bundle gets the installed Spyglass review controls; its scientific data and
saved annotations stay as saved.

**Browser operations** run one at a time in a worker with its own DataJoint
connection. Progress and committed identities persist in the local bundle, so
reloading reconnects and ownership survives the launcher exiting while its
worker is alive. After a restart, resume the review: a live worker keeps
running; retry only an interrupted action. A failed reevaluation retains the
child and a retry reuses it. Do not edit or discard the bundle while an
operation runs. Hosted figures have no compute service: save with FigPack **Save
Annotations**, then import with `panel = review.commit_panel()` (which exposes
`panel.receipt` and `panel.verification_review`).

**Labels.** The browser offers only the profile's
`CurationReviewProfile.label_options` palette (nonempty strings of at most 32
characters); labels already on a parent are preserved during import.

#### Where am I, and how do I undo a merge?

`changes.next_step()` and `receipt.next_step()` say which of four states applies
-- *saved browser edits differ from the reviewed parent* / *no saved edits
differ* / *merged result awaiting verification* / *result available for
analysis* -- and what to do next. The preview compares the bundle with the
reviewed parent only; a diff you already committed still "differs", and
committing it again reuses that child.

**A pending merge proposal** (saved, not committed): select its units in the
browser, **Undo selected merge**, **Save draft**; `preview_import()` then shows
no merge.

**A committed merge that was wrong** is a branch, not an edit: the merged child
(and any verification child under it) stays as history, and the fix is a
replacement sibling from the **same parent as the mistaken merge** -- which is
the parent of *that* merge, not necessarily the root (a merge committed during a
verification review has the earlier merged child as its parent). Going back to
the root would discard earlier valid merges and every edit committed in between.
The review the mistaken merge came from is on its receipt
(`receipt.changes.review`); undo the proposal there and commit again:

```python
bad_merge_receipt = receipt  # or the verification_receipt that made the bad merge
bad_merge = bad_merge_receipt.curation
recovery_review = bad_merge_receipt.changes.review
recovery_review.open()  # its bundle still holds your edits
```

Stop here: select only the mistaken merge's units, **Undo selected merge**, then
**Save draft**. Keep any other valid merges and label edits. After saving,
preview the correction in a separate cell:

```python
recovery_changes = recovery_review.preview_import()
print(recovery_changes.summary())  # includes the abandoned sibling
recovery_changes.changed_units()
```

Inspect the preview, resolve any listed label conflicts using its merged unit
IDs, then commit in another cell:

```python
recovery_conflict_resolutions = {}  # use the IDs from recovery_changes
replacement_receipt = recovery_changes.commit(
    conflict_resolutions=recovery_conflict_resolutions,
    # unmerging alone restores the parent exactly: a no-change commit that
    # must be confirmed
    confirm_no_changes=not recovery_changes.has_changes,
)
# Switch to the replacement branch so nothing downstream resumes or approves
# the abandoned merge.
if replacement_receipt.needs_merge_verification:
    pending_verification = replacement_receipt.continue_review()
    final_curation = None
    pending_verification.open()
else:
    pending_verification = None
    final_curation = replacement_receipt.curation
```

If a verification view opens, inspect the replacement's merged units before
running the verification block above. Edits made in the merged child's own
verification review live on the abandoned branch; redo them on the replacement.
Once nothing downstream refers to the abandoned branch,
`bad_merge.preview_curation_delete()` lists it leaf-first and
`bad_merge.delete_subtree()` removes it (a `SortedSpikesGroup` built from it
must be deleted first).

**Finding a review without its handle.** `FigPackReview.resume(review_id)`
rebuilds a handle from an id printed earlier. `FigPackReview.find(parent)`
returns every built profile-backed review of that curation, oldest first
(`profile=` narrows it); there may be several, because each `start_review` after
a commit starts a new one. `preview_import().next_step()` says whether a
review's saved edits differ from its parent:

```python
for candidate in FigPackReview.find(bad_merge.parent, profile=profile):
    print(candidate.review_id, candidate.uri)
    print("  ", candidate.preview_import().next_step())
recovery_review = FigPackReview.resume(chosen_review_id)
```

Before any child is committed, `parent.start_review(profile)` also returns the
existing review. `RunResult.start_review(source="auto_labeled")` never falls
back to root: if no analysis curation exists it raises.

**Import safety.** Every import is pinned to the parent's immutable
`curation_uuid` and the exact figure/profile configuration. `preview_import()`
reports children created by other reviewers after this review began; commit
still creates or reuses a sibling from the pinned parent and never silently
rebases. It re-reads the annotations and refuses a figure changed after preview.
A zero-diff review is refused unless `confirm_no_changes=True`, and conflicting
contributor labels require explicit `conflict_resolutions` -- no contributor
wins by precedence.

**Delivery.** Local delivery is the default. A persistent hosted review requires
a FigPack API key (`dj.config["custom"]["figpack_api_key"]` or the
`FIGPACK_API_KEY` environment variable); `ephemeral=True` creates a temporary
hosted figure. Both publish the same prebuilt bundle, including seeded
annotations and the Spyglass identity sidecar. FigPack needs the
`spikesorting-v2-curation` extra. The table-level `FigPackCurationSelection` /
`FigPackCuration` layer remains available for expert composition.

#### Display budgets and focused inspection

The unit selector stays visible across **Waveforms**, **Spike amplitudes**,
**Autocorrelograms**, **Cross-correlograms**, **Electrode geometry**, and
**Raster**. Raster and amplitude budgets default to
`floor(recording_duration_s * 50)` points per unit (3,000 for a minute, 180,000
for an hour); low-rate trains keep every point, and higher-rate samples span the
recording reproducibly. `max_raster_spikes_per_unit` and
`max_amplitudes_per_unit` impose smaller caps. Budgets affect display only, not
waveform sampling or metrics, so absence from a display is not evidence.
`review.summary()` reports unavailable rule inputs and display budgets.

Time plots exceeding `max_initial_points` (default 1,000,000 per view) are
marked **not loaded** in the initial bundle. **Inspect selected units / pairs**
loads the requested units with their full display budget; a raster window
includes every spike in `[start, stop)`, all selected CCG pairs are included,
and a window of at most 10 seconds adds a spikes-on-traces figure. A
hosted/static bundle needs the Python form (or a smaller overview cap):

```python
view = review.inspect_units([1, 4], time_range=(100, 110), include_traces=True)
view.show(title="Selected units", upload=False, ephemeral=False)
```

Displayed times are recording-relative seconds (the synthetic timeline for
concatenated recordings); **Time and sampling** maps them back to original
session seconds, including gaps and member boundaries. Red bands mark excluded
time. Manual exclusions use original session seconds. Inspection opens
separately and preserves the active draft and official evaluation.

### Custom unit annotations

Computed unit properties that are not built-in quality metrics live in typed,
immutable annotation sets (`UnitAnnotationDefinition`,
`CurationUnitAnnotationSet.from_dataframe`). A set belongs to one exact curation
namespace and is selected explicitly; Spyglass never chooses the "latest" set.
Custom columns are namespaced by definition version and full `set_hash`, so two
sets -- or a set and a built-in metric -- cannot overwrite each other. A worked
example is in the
[curation notebook](../notebooks/10_Spike_SortingV2_Curation.ipynb) (section
"3-annotations"). Read or review a set explicitly:

```python
properties = read_unit_properties(
    root, evaluation=None, annotation_sets=[annotation_set]
)
review = root.start_review(
    "franklab_hippocampus_2026_09_17", annotation_sets=[annotation_set]
)  # the set's column joins the unit table; its set_hash joins figure identity
```

Definitions support scalar `float` (stored as `double`), `int`, `bool`, and
`text`. Definitions are immutable by version; changed parameters or values
create a new set. Annotation sets are **not curation labels**: they never write
`CurationV2.UnitLabel` and do not change a curation's UUID, identity, or
`merge_id`. Parent sets are not carried into a continuation review, because a
committed child is a new curation; compute child-scoped sets explicitly.

### Scripted curation facade (automation and debugging)

`run_v2_pipeline` returns a mapping-compatible `RunResult`. Its `root_curation`
and `auto_labeled_curation` attributes are generation-pinned `CurationRef`s to
pass on directly. `auto_labeled_curation` is `None` until an analysis curation
exists; it never falls back to the root.

```python
from spyglass.spikesorting.v2.curation_api import save_manual_curation
from spyglass.spikesorting.v2.curation import CurationV2

root = run_summary.root_curation
CurationV2.summarize_curation(root.as_key())

preview = root.preview_merges([[3, 7]])  # proposed, not applied
merged = root.commit_merges([[3, 7]])  # applied, no re-evaluation

manual = save_manual_curation(
    parent_curation=root,
    labels={3: ["mua"]},
)

print(merged.commit_status, merged.operation_type, merged.merge_id)
print(merged.visualize_lineage())
```

`commit_status` is `"preview"` or `"committed"`; `is_root`, `is_leaf`, and
`has_committed_children` are independent booleans; `operation_type` reports the
stored producer and the change kind. There is no inferred "superseded" state in
a branching graph. Use `preview_curation_delete()` before `delete_subtree()`;
`health_report()` audits lineage and analyzer-cache orphans. `created_at` and
`created_by` expose the generation's creation metadata.

### Quality metrics and the scripted evaluate/merge loop

`CurationEvaluation` scores a **committed** `CurationV2` row in that curation's
**own** unit namespace: SpikeInterface quality metrics, merge suggestions, and
auto-curation labels. A merged unit's SNR / ISI-violation / PC-NN separation is
recomputed over its **merged** template, never inherited from a contributor.
Results are written to NWB and returned as one `EvaluationResult` snapshot whose
`metrics`, `suggested_merges`, and `proposed_labels` are copies. Turning
proposals into a child is explicit.

```python
evaluation = run_summary.root_curation.evaluate(
    metric_params_name="franklab_default",
    auto_curation_rules_name="franklab_default_auto_curation_2026_09",
)
display(evaluation.metrics)
print(evaluation.proposed_labels, evaluation.suggested_merges)
evaluation.plots.units_qc()

# Commits/reuses the merge and evaluates the merged child with evaluation.spec.
receipt = evaluation.merge_and_evaluate([[u0, u1]])
display(receipt.evaluation.metrics)
receipt.evaluation.plots.correlograms()
print(receipt.stage_statuses)

# replace = complete evaluation verdict; overlay = keep current labels + add.
final = receipt.evaluation.accept_labels(mode="replace")
final_merge_id = final.merge_id
```

`merge_and_evaluate(groups)` is idempotent and resumable; it must run outside a
caller-owned DataJoint transaction because `populate` manages its own.
`evaluation.commit_merges(groups)` commits without re-evaluation.

The browser review profile binds `franklab_default` metrics to
`franklab_default_auto_curation_2026_09` with `label_import_mode="replace"`;
publishing location, credentials, ephemeral mode, and annotation sets are
supplied per review:

```python
from spyglass.spikesorting.v2.review_profile import CurationReviewProfile

profile = (
    CurationReviewProfile & {"review_profile_name": "franklab_hippocampus_2026_09_17"}
).fetch1()
```

#### Expert table-method appendix

The facade delegates to table methods. `CurationEvaluation.accept_merges` /
`accept_all_suggested_merges` commit the merged unit set and **inherit** the
curation's existing labels; they do not apply pre-merge evaluation labels (a
label on an absorbed unit cannot attach to the merged unit). The recommended
flow is accept merge, re-evaluate the merged child, then label with
`use_evaluation_labels` (complete verdict: clears labels the evaluation does not
propose, so a no-longer-flagged unit is not silently excluded) or
`overlay_evaluation_labels` (keeps current labels and adds). The combined
`accept_evaluation_outputs` applies merges + labels in one call, and its
`labels=None` default applies the evaluation's pre-merge labels; prefer the
action methods unless that is intended. `preview_merges` drafts an unapplied
merge (a preview row downstream consumers reject until committed). Every merge
action requires at least one group of two or more units.

#### Saving a manual payload

`save_manual_curation` also accepts a payload: a v1/FigURL payload
(`labelsByUnit` / `mergeGroups`), a v2 payload (`labels_by_unit` /
`merge_groups`), or unpacked `labels=` / `merge_groups=`. `merge_action` is
`"preview"` (draft) or `"commit"` (apply); a v1 association map
(`{"1": ["2"], "2": ["3"]}`) is unioned transitively into `[[1, 2, 3]]`.

```python
from spyglass.spikesorting.v2.curation_api import save_manual_curation

child = save_manual_curation(
    parent_curation=root,
    payload={
        "labelsByUnit": {"3": ["mua"]},  # FigURL spellings
        "mergeGroups": {"5": ["6"]},
    },
    merge_action="commit",
)
```

`FigPackCuration.save_curation_from_uri(uri, parent_curation_key)` verifies the
figure's embedded `curation_uuid` before writing a child; identity-less figures
fail closed.
`import_legacy_figpack_curation(..., asserted_parent=..., confirm_unverified_identity=True)`
is only for a parent the operator verified independently.

#### Metric and rule semantics

- `metric_names` is validated against the installed SpikeInterface at insert.
    SpikeInterface 0.104 has no `nn_isolation` / `nn_noise_overlap` metric
    names: request the `nn_advanced` PCA metric (with `skip_pc_metrics=False`)
    and threshold its `nn_noise_overlap` output column in a rule.
- `isi_violation` is Spyglass's bounded `count / (n_spikes - 1)` fraction with
    the evaluation recipe's refractory window -- not SI's
    `isi_violations_ratio`, and not a contamination estimate.
- `AutoCurationRules` is inserted via `insert_rules(master, rule_rows)` (direct
    `insert1` is blocked) so the master and its ordered rules validate together.
- Each rule's `missing_policy` governs units SpikeInterface cannot assess (a NaN
    it leaves on purpose): `error` raises, `fail` applies the rule's label, and
    `pass` leaves the unit unlabelled by that rule. These are Spyglass
    semantics, not SI's `nan_policy`. The expected-NaN conditions per column are
    in `expected_missing_units` (`spyglass.spikesorting.v2._metric_curation`):
    `nn_isolation` / `nn_noise_overlap` below `nn_advanced`'s `min_spikes` or
    `min_fr`; `presence_ratio` for a recording shorter than one bin or a unit
    with no spikes; `amplitude_cutoff` below
    `num_histogram_bins * amplitudes_bins_min_ratio` spikes (500 by default);
    `isi_violation` at one spike or fewer; `firing_rate` with no spikes; `snr`
    and `num_spikes` never.
- Any other non-finite rule value raises `ValueError` regardless of
    `missing_policy`, including any non-finite value on a column with no
    registered conditions (a template metric, a custom metric, or an
    `observed_*` column). If SpikeInterface catches an error inside a metric
    that a rule references, the evaluation aborts naming the metric; an error in
    an unreferenced metric is logged and its columns stay NaN.
- The shipped rule sets use `pass`: NaN there means "not assessable for this
    unit", and `error` would abort a sort over a legitimately short train.
    `pass` and `fail` warn when a metric is non-finite for every unit.
    `missing_policy="pass"` means unflagged, not good. The review's selectable
    `unavailable_qc` column names missing inputs for the enabled rules (disabled
    metrics are not failures).
- Metric persistence accepts only zero-dimensional numeric scalars (scalar NaN
    included); arrays and non-numeric objects raise
    `UnsupportedMetricValueError`.
- Deny labels take precedence over `accept` in the shipped analysis policies.

#### Population QC plot and burst-pair views

`evaluation.plots.units_qc()` (table form `CurationEvaluation.plot_units_qc`)
draws one histogram per quality metric plus a unit-depth scatter; pass `axes=`
to embed it. Pair views are `plot_correlograms`, `investigate_pair_xcorrel`,
`investigate_pair_peaks`, and `plot_peak_over_time`.
`evaluation.burst_pair_metrics(pairs=[...])` (table form
`CurationEvaluation().get_burst_pair_metrics(key)`) returns a DataFrame indexed
by `(unit1, unit2)` with `wf_similarity`, `isi_violation`, `xcorrel_asymm`, and
`unit_distance`, computed from the display analyzer on each call (restrict with
`pairs=`). Pair `isi_violation` uses violating intervals / (`spikes - 1`) with
the evaluation's refractory window (override with `isi_threshold_ms=`); pairs
with fewer than two combined spikes are `NaN`. Commit and re-evaluate to score
the actual merged train after duplicate-spike handling. These plots render a
committed merged curation's own unit namespace and never fall back to the raw
analyzer; preview curations are unscorable until their merges are committed.

#### The scripted evaluate -> merge -> evaluate -> label flow

Each child edits its **parent's committed state** (see below), so a
merged-parent unit id is a valid input and absorbed raw units are never
resurrected:

1. **Evaluate.** `root.evaluate(...)`, then inspect `evaluation.metrics`,
    `evaluation.proposed_labels`, and `evaluation.plots.*`.
2. **Manually merge.** `evaluation.merge_and_evaluate(groups)` commits or reuses
    the merge and evaluates the merged child with the same immutable
    `EvaluationSpec`.
3. **Label the evaluated namespace.** Inspect `receipt.evaluation`, then call
    `accept_labels(mode="replace")` (or the additive `"overlay"`) and use the
    returned `CurationRef.merge_id`.

A root or label-only curation reuses the raw-sort analyzer; a merged curation
resolves a generation-pinned, read-only analyzer over the merged sorting. A
**preview** curation (`apply_merge=False` with an unapplied merge group) is
rejected by `CurationEvaluation`; commit the merge first
(`create_merged_curation` / `insert_curation(..., apply_merge=True)`).

#### Parent-state composition and label inheritance

A child curation (`parent_curation_id != -1`) composes from its **parent
`CurationV2`** state, not the raw sort: unit rows, spike trains, labels, and
merge namespace all come from the parent.

- **Raw provenance stays queryable.** `CurationV2.MergeGroup` always records the
    original `Sorting.Unit` contributors (expanded through the parent's
    `MergeGroup`); the immediate parent operation is in
    `CurationV2.ParentMergeGroup`. `CurationV2.get_unit_contributor_groups(key)`
    returns `{kept: [contributors]}`.
- **Labels inherit by default.** `insert_curation(..., label_policy="inherit")`
    starts a child from its parent's labels and overlays the supplied `labels`
    per unit; a committed merge inherits the **union** of its contributors'
    labels. `label_policy="replace"` makes the supplied labels the whole child
    state.
- **Root idempotency.** A second default-content root `insert_curation`
    (`parent_curation_id=-1`, no labels / merge groups / description /
    `apply_merge`, default `curation_source`) returns the existing key with a
    warning. A second root call that carries such content raises `ValueError`
    unless you pass `reuse_existing=True` or curate a child of the existing
    root.
- `CurationV2.get_merged_sorting` applies merges at fetch regardless of
    `merges_applied`; `Sorting.get_sorting(key, as_dataframe=True)` and
    `CurationV2.get_sorting(key, as_dataframe=True)` return a DataFrame indexed
    by `unit_id` with `spike_times` (seconds), plus `curation_label` for the
    curation form.

### Stage-by-stage (custom pipeline preset)

Drive the stages directly when no preset applies, or to inspect preprocessing
and artifact detection before sorting. Use the parameter-row names from
`describe_pipeline_presets()`; a later pipeline call with the same inputs reuses
these stages.

```python
from spyglass.spikesorting.v2.recording import (
    Recording,
    RecordingSelection,
)
from spyglass.spikesorting.v2.artifact import (
    RecordingArtifactDetection,
    RecordingArtifactSelection,
)
from spyglass.spikesorting.v2.sorting import (
    Sorting,
    SortingSelection,
)
from spyglass.spikesorting.v2.curation import CurationV2

nwb_file_name = "your_session.nwb"  # same session as above

recording_key = RecordingSelection.insert_selection(
    {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,  # reviewed above
        "interval_list_name": "raw data valid times",
        "preprocessing_params_name": "franklab_hippocampus_2026_06",
        "team_name": "my_team",
    }
)
Recording.populate(recording_key)

# Cross-recording detection uses SharedArtifactGroup +
# SharedGroupArtifactSelection / SharedGroupArtifactDetection instead.
artifact_detection_key = RecordingArtifactSelection.insert_selection(
    {
        "recording_id": recording_key["recording_id"],
        "artifact_detection_params_name": "default",
    }
)
RecordingArtifactDetection.populate(artifact_detection_key)
```

Inspect a trace window and the **retained valid intervals**. `plot_traces` shows
the preprocessed recording before masking; it defaults to the first second (pass
`time_range=(start, stop)` in recording seconds).

```python
Recording().plot_traces(recording_key)
valid_intervals = RecordingArtifactDetection().get_artifact_removed_intervals(
    artifact_detection_key
)
valid_intervals
```

Then sort and create the root curation:

```python
sorting_key = SortingSelection.insert_selection(
    {
        "recording_id": recording_key["recording_id"],
        "sorter": "mountainsort5",
        "sorter_params_name": "franklab_30khz_ms5_2026_06",
        "artifact_detection_id": artifact_detection_key["artifact_detection_id"],
    }
)
Sorting.populate(sorting_key)

curation_key = CurationV2.insert_curation(
    sorting_key=sorting_key,
    labels={},
    parent_curation_id=-1,
    description="first pass",
)
```

### ADC phase-shift (Neuropixels)

Multiplexed ADCs (e.g. Neuropixels) sample a shank's channels at slightly
different times. The optional `phase_shift` preprocessing parameter compensates
these sub-sample delays. It runs first (see the stage order under
[Tables](#tables)) and only when the recording carries an `inter_sample_shift`
property; otherwise (including Frank-lab polymer probes) it logs a skip and is a
no-op. It is off in the `default` and region rows and on in
`default_neuropixels` (`bandpass 300-6000 Hz` + phase-shift, `margin_ms=100`),
which materializes identically to `default` on recordings without
`inter_sample_shift`:

```python
recording_key = RecordingSelection.insert_selection(
    {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": 0,
        "interval_list_name": "raw data valid times",
        "preprocessing_params_name": "default_neuropixels",
        "team_name": "my_team",
    }
)
```

`ElectricalSeries.filtering` lists `phase-shift (ADC)` only when the step ran.

### Automated bad-channel detection

`suggest_bad_channels` proposes -- and, on a second confirmed call, persists --
`Electrode.bad_channel` flags. It bandpass-filters the raw recording and runs
SpikeInterface's `detect_bad_channels` (`coherence+psd`) per full shank. The
default `persist=False` changes nothing and returns a report.

```python
from spyglass.spikesorting.v2.bad_channels import suggest_bad_channels

# 1. Review: mutates nothing, returns one dict per flagged electrode.
reviewed_report = suggest_bad_channels(nwb_file_name, persist=False)
for entry in reviewed_report:
    print(entry)  # {"electrode_group_name", "electrode_id", "probe_shank", "label"}

# 2. Confirm: persist exactly the report you reviewed (no re-detection).
suggest_bad_channels(nwb_file_name, persist=True, reviewed_report=reviewed_report)
```

The method samples random chunks (`seed=None`), so a bare
`suggest_bad_channels(nwb_file_name, persist=True)` re-detects and may flag a
different set; pass `reviewed_report=` (or `detection_params={"seed": ...}` to
both calls) when the reviewed and persisted sets must match.

Each flagged electrode carries a **label** (`dead`, `noise`, or `out`).
`persist=True` sets `Electrode.bad_channel='True'` for `dead`/`noise` only and
is additive (it never clears an existing flag). `out` (outside-brain) channels
are report-only, because `bad_channel='True'` means a quality-bad channel that
is safe to interpolate or remove. To keep an `out` channel out of a sort, omit
it from the group (e.g.
`SortGroupV2.set_group_by_electrode_table_column(nwb_file_name, electrode_column="electrode_id", value_groups=[[...in-brain electrode_ids...]])`).

The thresholds are SpikeInterface's Neuropixels-derived defaults; pass
`detection_params=` (e.g. `{"dead_channel_threshold": -0.4}`) to recalibrate for
other geometries. Scope with `electrode_group_names=` and change the band with
`bandpass=`. Results on small shanks such as tetrodes are unreliable; treat a
small-shank "no bad channels" with skepticism.

**Ordering contract:** finalize `bad_channel` flags **before** creating sort
groups. `SortGroupV2.set_group_by_*` excludes flagged channels at creation; a
flag added later does not change an existing group -- recreate it.

### Bad-channel handling (remove vs interpolate)

The `bad_channel_handling` preprocessing parameter chooses what happens to
curated `Electrode.bad_channel='True'` channels at materialization:

- **`"remove"` (default)** -- no bad channel is added back; the group is its
    declared members. Use it for tetrodes and sparse/custom groups, and whenever
    the sorter should see only good channels.
- **`"interpolate"`** -- re-includes the group's pitch-adjacent interior
    curated-bad channels (≥2 good neighbours within ~1.5× the probe pitch) and
    fills them by kriging (`interpolate_bad_channels`), so a geometry-aware
    sorter sees a complete probe. Isolated bad channels, or ones in the gap of a
    non-contiguous group, stay out. It raises if the probe has no contact
    positions.

```python
recording_key = RecordingSelection.insert_selection(
    {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": 0,
        "interval_list_name": "raw data valid times",
        "preprocessing_params_name": "my_interpolate_preset",  # bad_channel_handling="interpolate"
        "team_name": "my_team",
    }
)
```

Handling runs between the bandpass filter and the reference.
`ElectricalSeries.filtering` lists `interpolate N bad channels` only when N > 0.
It consumes curated flags only (no detection), so the ordering contract above
applies. A manually flagged outside-brain channel must use `remove`, never
`interpolate` (which would invent signal). The `specific` reference electrode is
never a handling target; a `bad_channel='True'` reference (e.g. a dedicated
ground) is still used as the reference.

### Drift QC (motion estimate, never applied)

`DriftEstimate` estimates probe motion on a materialized `Recording` and stores
it as a queryable QC artifact to **flag** high-drift sessions. Nothing applies
it: the `Recording`'s `content_hash` and traces are unchanged. To correct
motion, use the [optional motion stage](#optional-motion-correction). It is
populated only on demand:

```python
from spyglass.spikesorting.v2.recording import DriftEstimate, Recording

DriftEstimate.populate(recording_key)  # e.g. {"recording_id": ...}

(DriftEstimate & "max_abs_displacement_um > 20").fetch("recording_id")
max_drift_um = (DriftEstimate & recording_key).fetch1("max_abs_displacement_um")

motion = DriftEstimate().get_motion(recording_key)  # SI Motion object
```

It always uses `dredge_fast` (stored on the row; no parameters Lookup), which
requires `torch` (installed by the `spikesorting-v2` extra), and applies **no
artifact mask**. Its numbers are therefore **not comparable** to the motion
stage's masked, spans-aware `MotionEstimate`.

### Optional motion correction

**EXPERIMENTAL: no motion recipe here is validated for a probe.** Inspect the
saved estimate rather than trusting a sort's improvement. On a simulated
32-contact, single-column, 26 µm-pitch polymer shank, `rigid_fast` was sometimes
worse than no correction, interpolation cost sorting quality even with the true
motion, and a held-out benchmark (cases in
`tests/spikesorting/v2/motion_acceptance_held_out.json`) failed for both shipped
recipes; no real lab recording with drift has been tested. That benchmark ran
against an earlier estimator, so it does not cover the current estimator on
discontinuous inputs (acquisition gaps, concatenations) or on sources that are
not unit-calibrated float microvolts.

`motion_mode` on `run_v2_pipeline` / `run_v2_pipeline_session` (and the matching
preflight helpers) works for a single-session **or** a concat run:

- `"off"` (default) -- no motion stage; the ordinary uncorrected sort.
- `"estimate"` -- saves a `MotionEstimate` of the recording (or
    `ConcatenatedRecording`) under its mask, but still sorts the uncorrected
    traces (same `sorting_id` as `"off"`).
- `"apply"` -- also saves a `MotionCorrectedRecording` and sorts it (a distinct
    `sorting_id`).

`motion_correction_params_name` (a `MotionCorrectionParameters` row pairing an
estimation recipe with an interpolation recipe) is required for `"estimate"` and
`"apply"` and rejected for `"off"`; a contradictory pair raises
`PipelineInputError` before any query. A failed motion stage raises
`PipelineStageError` and nothing downstream is sorted -- it never falls back to
an uncorrected sort.

**The source must be filtered.** Motion is estimated on filtered, unwhitened
traces. A source whose preprocessing recipe applies no temporal filter
(`bandpass_filter=None`, the shipped `no_filter` row; for a concat, the
concatenation's recipe) is refused by
`MotionEstimateSelection.insert_selection`, and preflight fails its
`motion_source_filtered` check (the concat preflight raises `PreflightError`).

**Estimate, inspect, then apply that estimate.** `estimate_motion` takes the
same source arguments as `run_v2_pipeline` plus the recipe, runs preflight,
recording and artifact-detection stages, and saves only the `MotionEstimate`.
Pass its `motion_estimate_id` to an `"apply"` run, which corrects and sorts with
exactly that estimate (never recomputing it):

```python
from spyglass.spikesorting.v2.motion import MotionEstimate
from spyglass.spikesorting.v2.pipeline import estimate_motion, run_v2_pipeline

source = dict(
    nwb_file_name=nwb_file_name,
    sort_group_id=sort_group_id,
    interval_list_name="raw data valid times",
    team_name="my_team",
    pipeline_preset="franklab_probe_hippocampus_30khz_ms4_2026_06",
)

# Concat: pass concat_session_group_owner / concat_session_group_name instead.
receipt = estimate_motion(**source, motion_correction_params_name="dredge_fast_v1")
receipt["motion_estimate_id"], receipt["motion_estimation_preset"]
receipt["motion_diagnostics"]  # peaks detected/kept, max displacement, bins
receipt["motion_spans_without_evidence"]  # [] when every span kept a peak

fig, report = MotionEstimate().report(receipt)  # see "Inspecting an estimate"

summary = run_v2_pipeline(
    **source,
    motion_mode="apply",
    motion_correction_params_name="dredge_fast_v1",
    motion_estimate_id=receipt["motion_estimate_id"],
)
summary["motion_estimate_supplied"]  # True; motion_estimate_status "reused"
```

The supplied estimate must be a populated estimate of this run's source and
artifact mask (for concat: the same session group, recipe, members, member
recordings and masks), made with the recipe's **estimation** row, on traces
unchanged since it was selected; any mismatch is an error before anything is
corrected or sorted (from preflight, or with `preflight=False` from the
`motion_estimate` stage). `motion_estimate_id` with `"off"` or `"estimate"` is a
`PipelineInputError`. `estimate_motion` skips the sorter-only preflight checks;
the `"apply"` run checks them.

Without an explicit id, `"estimate"` and `"apply"` runs with the same source,
mask and estimation row resolve to the same `motion_estimate_id`, so an
`"apply"` after an `"estimate"` reuses it (two recipes sharing an estimation row
reuse it too). An `"apply"` summary reports `motion_corrected_recording_id` and
`motion_removed_channel_ids`.

The equivalent table-level calls:

```python
from spyglass.spikesorting.v2.motion import (
    MotionEstimateSelection,
    MotionEstimate,
    MotionCorrectedRecordingSelection,
    MotionCorrectedRecording,
)
from spyglass.spikesorting.v2.sorting import SortingSelection, Sorting

# One of recording_id / concat_recording_id; artifact_detection_id is optional
# and single-recording only (a concat carries its own frozen member masks).
estimate_key = MotionEstimateSelection.insert_selection(
    {
        "recording_id": recording_key["recording_id"],
        "artifact_detection_id": artifact_detection_key["artifact_detection_id"],
        "motion_estimation_params_name": "dredge_fast_v1",
    }
)
MotionEstimate.populate(estimate_key)

corrected_key = MotionCorrectedRecordingSelection.insert_selection(
    {
        "motion_estimate_id": estimate_key["motion_estimate_id"],
        "motion_interpolation_params_name": "kriging_force_extrapolate_v1",
    }
)
MotionCorrectedRecording.populate(corrected_key)

# The correction must come from the SAME source and mask as this sort; a
# mismatch raises at insert (ValueError) and at compute (SchemaBypassError).
sorting_key = SortingSelection.insert_selection(
    {
        "recording_id": recording_key["recording_id"],
        "artifact_detection_id": artifact_detection_key["artifact_detection_id"],
        "motion_corrected_recording_id": corrected_key["motion_corrected_recording_id"],
        "sorter": "mountainsort5",
        "sorter_params_name": "franklab_30khz_ms5_2026_06",
    }
)
Sorting.populate(sorting_key)
```

**Inspecting an estimate.** `MotionEstimate.report` draws one figure and returns
a summary dict from the stored arrays. Pass an `estimate_motion` receipt, a
`motion_estimate_id` or a restriction; optionally add a corrected recording and
a short window on the source's own clock (seconds) to compare traces:

```python
estimate_key = {"motion_estimate_id": receipt["motion_estimate_id"]}
t_start = MotionEstimate().get_estimation_clock(estimate_key).source_start_s[0]
t_start += 60.0  # 60 s after the source's first sample
fig, report = MotionEstimate().report(
    receipt,
    corrected_key=corrected_key,  # optional MotionCorrectedRecording
    trace_window_s=(t_start, t_start + 0.2),  # optional; needs corrected_key
)
report["spans_without_evidence"]  # spans corrected from the prior alone
report["max_abs_displacement_um"], report["rms_displacement_um"]
report["masked_fraction"], report["gaps"], report["capped_gap_s"]
report["border_channel_ids"], report["removed_channel_ids"]
```

The panels show (a) displacement over source time (a heatmap over depth when
nonrigid); (b) a timeline of masked intervals, acquisition gaps, member joins
(hatched where shortened to `max_gap_s`) and spans without evidence; (c) kept
peaks per continuity span; (d) border channels moved past the probe's ends
(circled when extrapolated, crossed when removed); and (e) with
`trace_window_s`, original vs corrected traces for four central channels
(`trace_channel_ids` to choose). A span with **no evidence** kept no peak, so
its displacement, and any correction there, comes from the estimator's temporal
prior alone; it is not refused, but `run_v2_pipeline` warns and lists it in
`motion_spans_without_evidence`. Check these before applying an estimate.

**Reading a saved estimate.** `MotionEstimate` stores the SpikeInterface
`Motion`, its resolved configuration, the spans it estimated from, and
peak-count diagnostics -- never a raw peak array:

```python
motion = MotionEstimate().get_motion(estimate_key)  # bins on the estimation clock
clock = MotionEstimate().get_estimation_clock(estimate_key)
mapped = MotionEstimate().get_displacement_on_source_clock(estimate_key)
MotionEstimate().get_spans_without_evidence(estimate_key)

row = (MotionEstimate & estimate_key).fetch1()
row["n_peaks_detected"], row["n_peaks_kept"]  # on the masked recording
row["peaks_per_temporal_bin"], row["peaks_per_continuity_span"]
row["max_abs_displacement_um"], row["noise_levels"]
```

`get_displacement_on_source_clock` maps bins back to source time for inspection;
a bin inside a capped gap is flagged (`in_gap=True`, `source_time_s=NaN`).

**Resolved presets.** Every recipe is DREDge's AP registration (SpikeInterface's
`estimate_motion(..., method="dredge_ap")`):

| recipe                                                    | estimator                             | peak localization         | interpolation border mode                            |
| --------------------------------------------------------- | ------------------------------------- | ------------------------- | ---------------------------------------------------- |
| `dredge_v1` (default row)                                 | `dredge_ap`, nonrigid (`rigid=False`) | `monopolar_triangulation` | `force_extrapolate` (`kriging_force_extrapolate_v1`) |
| `dredge_fast_v1` (default row)                            | `dredge_ap`, nonrigid (`rigid=False`) | `grid_convolution`        | `force_extrapolate` (`kriging_force_extrapolate_v1`) |
| `rigid_fast` (allowed; insert explicitly, no default row) | `dredge_ap`, **rigid=True, 5 s bins** | `center_of_mass`          | `remove_channels` (`kriging_remove_channels_v1`)     |

`MotionEstimationParameters` persists the fully resolved SpikeInterface
configuration (preset defaults, signature defaults, your overrides), the
SpikeInterface version, and the estimation algorithm version, all in the
estimate's identity, so an upgrade or a changed override selects a new estimate.
`MotionInterpolationParameters` states every `interpolate_motion` argument
(`spatial_interpolation_method`, `sigma_um`, `p`, `num_closest`). Estimation and
interpolation act on microvolts: a source that is not float µV with gain 1 and
offset 0 is first scaled with `scale_to_uV` (float32), masked samples are
exactly 0 µV after scaling, and the corrected recording is stored with gain 1
and offset 0. Only `remove_channels` and `force_extrapolate` border modes are
allowed; `force_zeros` is rejected.

**Gap policy.** Each source is estimated **once**, on an *estimation clock*:
within a continuity span (an uninterrupted acquisition stretch, or one
concatenation member) time advances at `1 / fs`, and a real gap between spans is
kept up to `max_gap_s` (required on every `MotionEstimationParameters` row, 30 s
in the shipped rows, part of the estimate's identity) and shortened beyond it.
One estimation therefore gives one common reference frame for every span.
Concatenation members must be in acquisition-time order; out-of-order or
overlapping members raise.

**Masks and statistics spans.** Estimation excludes masked samples from noise
and peak statistics: a peak is kept only if its localization window lies inside
one artifact-free statistics span and its detection window does not cross a join
between continuity spans. A `MotionCorrectedRecording` reuses its estimate's
statistics and continuity spans (checked against the source at compute and sort
time) and re-silences masked samples after interpolation.

**Correction ownership relative to the sorter.**
`SortingSelection.insert_selection` rejects a `motion_corrected_recording_id`
paired with a `SorterParameters` row whose sorter would also correct motion
internally (SpykingCircus2's / Tridesclous2's `apply_motion_correction`,
Kilosort's `do_correction`, resolved against SpikeInterface's default when the
row omits the key), and re-checks at compute. A sorter with unknown motion
behavior is refused too. To sort a corrected recording with one of these
sorters, insert a `SorterParameters` row with the internal-correction key
explicitly `False`; the error names the sorter, row, and key.

**Selecting a corrected vs. uncorrected sort.** `motion_corrected_recording_id`
follows the `artifact_detection_id` restriction convention: `None` matches only
sorts of the source's own (uncorrected) traces, an id matches only that
correction's sorts, and an **absent** key matches both. Pass the key explicitly
(to `CurationV2.resolve_restriction`, `SpikeSortingOutput`'s v2 restriction
dispatch, etc.) when you need exactly one of them.

**Database privileges.** Importing `spyglass.spikesorting.v2.sorting` also
declares the `spikesorting_v2_motion` schema (through
`SortingSelection.MotionCorrectionSource`), so users need insert/create
privileges on it, as on any other v2 schema. Storage costs of the motion
artifacts are in
[Storage Management](./SpikeSortingV2StorageManagement.md#motion-correction-artifacts).

#### Using the `rigid_fast` estimator

`rigid_fast` has no default row and is not recommended (see the warning above).
To compare it, insert a `MotionEstimationParameters` row with
`params={"preset": "rigid_fast", "max_gap_s": 30.0}`, pair it with the shipped
`kriging_remove_channels_v1` interpolation row
(`MotionInterpolationParameters.insert_default()`) in a new
`MotionCorrectionParameters` row, and run with `motion_mode="apply"` and that
`motion_correction_params_name`.

### Chronic same-day recordings

When a chronic implant is recorded across several files on the **same day**, you
can concatenate the per-member recordings into one masked recording and sort
them together; this recovers units a per-file sort would split and is the
default chronic path. Concatenation itself never corrects motion; apply the
[optional motion stage](#optional-motion-correction) to the
`ConcatenatedRecording` for a corrected sort. For **days/weeks-apart** sessions,
sort each session and match units across them
([Cross-session unit tracking](#cross-session-unit-tracking)); multi-day
concatenation is experimental and needs an explicit opt-in. For an existing
`SessionGroup`,
`run_v2_pipeline(concat_session_group_owner=..., concat_session_group_name=..., pipeline_preset=...)`
runs the member, concat, sort and curation stages (see the
[Cross-Session notebook](../notebooks/10_Spike_SortingV2_CrossSession.ipynb));
the table-level steps are:

```python
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.spikesorting.v2.artifact import (
    RecordingArtifactDetection,
    RecordingArtifactSelection,
)
from spyglass.spikesorting.v2.concat_member_curation import ConcatMemberCuration
from spyglass.spikesorting.v2.recording import RecordingSelection, Recording
from spyglass.spikesorting.v2.session_group import (
    SessionGroup,
    ConcatenatedRecordingSelection,
    ConcatenatedRecording,
)
from spyglass.spikesorting.v2.sorting import SortingSelection, Sorting

# 1. Materialize each member's Recording (one shared preprocessing recipe) and
#    its artifact detection. A member is (nwb_file_name, sort_group_id,
#    interval_list_name, team_name), not a whole NWB.
members = [
    {
        "nwb_file_name": f,
        "sort_group_id": 0,
        "interval_list_name": "raw data valid times",
    }
    for f in (nwb_a, nwb_b)
]
artifact_ids = {}
for member_index, m in enumerate(members):
    rec_key = RecordingSelection.insert_selection(
        {**m, "preprocessing_params_name": "default", "team_name": "my_team"}
    )
    Recording.populate(rec_key)
    detection = RecordingArtifactSelection.insert_selection(
        {
            **rec_key,
            "artifact_detection_params_name": "default",
        }
    )
    RecordingArtifactDetection.populate(detection)
    artifact_ids[member_index] = detection["artifact_detection_id"]
    print(
        member_index,
        RecordingArtifactDetection().get_artifact_removed_intervals(detection),
    )

# 2. Name the group (namespaced by owner). Multi-day members require
#    allow_multi_day=True.
SessionGroup.create_group("my_team", "day1", members)

# 3. Materialize the masked, unwhitened concat cache.
concat_key = ConcatenatedRecordingSelection.insert_selection(
    {
        "session_group_owner": "my_team",
        "session_group_name": "day1",
        "preprocessing_params_name": "default",
    },
    artifact_detection_ids=artifact_ids,
)
ConcatenatedRecording.populate(concat_key)

# 4. Sort the concatenated recording.
sort_key = SortingSelection.insert_selection(
    {
        "concat_recording_id": concat_key["concat_recording_id"],
        "sorter": "mountainsort5",
        "sorter_params_name": "franklab_30khz_ms5_2026_06",
    }
)
Sorting.populate(sort_key)

# 5. After curating, materialize the chosen curation once per member session.
curation_key = {"sorting_id": sort_key["sorting_id"], "curation_id": ...}
ConcatMemberCuration.populate(curation_key)
member_merge_ids = {
    row["member_index"]: row["merge_id"]
    for row in (
        SpikeSortingOutput.ConcatMemberCuration * ConcatMemberCuration & curation_key
    ).fetch(as_dict=True)
}
# Every member carries the same curated unit ids, including empty spike
# trains when a unit did not fire there.
```

Key behaviors and caveats:

- **Downstream merge gate.** The concat `CurationV2` row is never registered in
    `SpikeSortingOutput`: its synthetic gap-free timeline is unsafe for
    session-scoped consumers. `ConcatMemberCuration` registers one
    wall-clock-aligned merge row per frozen `member_index` (labels and unit IDs
    shared across members); `run_v2_pipeline` returns them as
    `member_merge_ids`, keyed by that index.
- **Artifacts are detected and masked per member before concatenation.** The
    selection freezes each detection by foreign key in its identity, so changing
    a mask creates a new concat and sort; no mask is inherited from an earlier
    standalone sort. Direct callers must supply every member index in
    `artifact_detection_ids` (an explicit `None` means no mask). If any member
    keeps a nonzero channel offset (only with `no_filter` and
    `reference_mode="none"`), every member is converted to float32 µV before
    masking. Sample counts and boundaries never change.
- **Observation intervals survive curation and member export.** The concat NWB
    stores kept intervals in synthetic seconds; each exported member carries
    them in original session time. Detection deletion is blocked while a concat
    selection references it (unless cascade deleting). `describe_run` reports
    member detection IDs/status and masked durations.
- **Valid time needs an explicit analysis choice.** SI `firing_rate` and
    `presence_ratio` use the analyzer sample timeline, including masked time
    (see [Observed-time metrics](#observed-time-metrics-and-downstream-analysis)).
    A population built with `select_units_for_analysis` freezes each unit's
    observed intervals, mapped to original session seconds, and
    `SortedSpikesGroup` spike indicators and sorted-spikes decoding through it
    honor them (see
    [Observed-time metrics](#observed-time-metrics-and-downstream-analysis)). A
    group built without that snapshot has unknown coverage and is not
    restricted: intersect its analysis windows with the stored intervals
    yourself. Do not interpret masked periods as neural silence.
- **Parent anchoring.** A concat sort's analysis NWB and each unit's `Electrode`
    FK anchor to the **first** `SessionGroup.Member`, so
    `get_unit_brain_regions` on a concat sort raises
    `ConcatBrainRegionAmbiguousError` unless you pass `allow_anchor_member=True`
    (labeled `region_resolution="anchor_member"`). For per-session regions,
    match the sessions and use `TrackedUnit.get_unit_brain_regions`.
    `get_sort_group_info` and `CurationV2.get_sort_metadata` resolve through the
    anchor member rather than raising.
- **Whitening stays at the sorter/analyzer boundary**, as for a single-session
    `Recording`.

Concat cache identity, rebuild, and deletion are covered in
[Storage Management](./SpikeSortingV2StorageManagement.md#concatenated-recordings).

### Cross-session unit tracking

For sessions recorded **days or weeks apart**, sort and curate each session
independently, then match units across sessions. Matching never concatenates raw
data: it pins one committed curation per **matching input** and reads only the
traces that sort's sorter read. A matching input is one curated sort -- a
single-recording `CurationV2` row or a same-day concatenation's parent curation
-- so a two-block daily concatenation contributes one input, not two.

**Overlap restrictions**, enforced by `UnitMatchSelection.insert_inputs` and
re-checked by `UnitMatch.make`:

- **No two inputs may share a recording session** (`nwb_file_name`):
    `SameSessionMatchError` rejects two sorts of one session, a concatenation
    matched with a sort of one of its own members, and two concatenations
    sharing a session. Matching within one session is therefore out of scope.
- **A concatenation input must lie within one day.**

**Chronological order and identity.** Inputs are numbered `input_index` `0..n-1`
by their earliest frozen `session_start_time`, regardless of the order named, so
the same inputs in any order resolve to the same selection. Each input pins an
exact `(sorting_id, curation_id)` and `curation_uuid` (no implicit "latest"),
plus its recordings' identity, content hash, session start time, frame span and
kept intervals, frozen in `UnitMatchSelection.Input` / `InputRecording` and
covered by `input_set_hash`. `UnitMatch.make` refuses a run whose frozen
`nwb_file_name` or `session_start_time` differs from the live rows, so after a
`Session.session_start_time` correction, select again. A `SessionGroup` passed
to `insert_selection` is provenance only, not a foreign key. A run of one input
writes the frozen matchable-unit universe and an empty `Pair` table without
calling the matcher.

**Chronic electrode-space contract.** The built-in UnitMatch backend **rejects**
a channel-geometry mismatch across inputs. The pipeline **warns** when electrode identity
(`(electrode_group_name, electrode_id, brain_region)`) differs, since ingestion
naming is not guaranteed stable across labs. Keep a stable electrode-group name
across sessions of one implant; a genuine probe mix-up also shows as poor
matcher AUC / few pairs. (Concatenation rejects mismatched electrode spaces
outright.)

Matching uses [UnitMatch](https://github.com/EnnyvanBeest/UnitMatch)
(`matcher="unitmatch"`, the built-in backend); its reference probe is the
128-channel LLNL polymer:

```bash
pip install -e ".[spikesorting-v2-matching]"   # UnitMatchPy + mat73
```

Custom matchers register through `matcher_protocol.register_matcher(backend,
schema, input_preparer=preparer)`. Preparation and inference have separate
contracts: `MatcherInputPreparer.prepare(source, directory, params, job_kwargs)`
receives a `MatcherInputSource` containing the artifact-masked recording,
curated sorting, frozen statistics spans, curation identity and date. It returns
a `PreparedMatcherInput` containing the `SessionMatcherInput` consumed by the
backend and any excluded unit IDs. Preparation must preserve the frozen identity
and date; excluded units remain in the matchable universe as unmatched units.
Returned pairs must use integer curation and unit IDs and can reference only
units retained by preparation; invalid pairs fail before writing the pairs table.
Prepared files live through inference and are then removed, including on failure.
Omitting `input_preparer` uses the shared dense split-half waveform layout, which
requires SpikeInterface and NumPy but does not import UnitMatchPy. Supply a
preparer when a backend needs another layout. Register a distinct matcher name
when changing the preparation or inference contract for persisted parameters.

Geometry requirements belong to the backend. It can implement
`validate_geometry(named_positions, params)` to check the effective channel
positions before a new multi-input selection is inserted. The positions are
passed with input labels in chronological order, together with the validated
matcher parameters. Raise `ValueError` for unsupported geometry. UnitMatch uses
this hook to require identical probe geometry and checks the prepared geometry
again during inference. A backend without the hook does not trigger geometry
reads or comparisons during selection. It remains responsible for validating
the prepared inputs it consumes. Frozen-input identity, session-overlap checks,
and pair validation apply to every backend.

The recommended path is **plan-then-run**: pin curations by a named curation
strategy, review the plan, then run. Group-based planning uses a `SessionGroup`
of already sorted and curated members:

```python
from spyglass.spikesorting.v2.pipeline import (
    plan_v2_unit_match,
    run_v2_unit_match,
)

# curation_strategy is REQUIRED: final_curated / auto_curated / root / manual.
plan = plan_v2_unit_match("my_team", "implant_week1", curation_strategy="final_curated")
plan.as_dataframe()  # per-member pins; plan.errors lists unresolved members
summary = run_v2_unit_match(plan)  # runs UnitMatch + TrackedUnit
```

#### Matching named sorts directly (e.g. daily same-day concatenations)

`plan_v2_unit_match_from_sorts` names the already-curated sorts to match, in any
order, with no `SessionGroup`; each named sort (single-recording or same-day
concatenation) becomes one matching input:

```python
from spyglass.spikesorting.v2.pipeline import (
    plan_v2_unit_match_from_sorts,
    run_v2_unit_match,
)

plan = plan_v2_unit_match_from_sorts(
    [day1_sorting_id, day2_sorting_id],
    curation_strategy="final_curated",
)
plan.as_dataframe()  # one row per matching input; review before running
summary = run_v2_unit_match(plan)  # inputs are ordered chronologically
```

`curation_strategy` takes the same values (with
`manual_curation_choices={sorting_id: curation_id}` for `manual`). The overlap
and geometry checks run inside `UnitMatchSelection.insert_inputs` when
`run_v2_unit_match(plan)` executes. A plan that could not pin exactly one
curation per sort has `plan.ok is False`, and `run_v2_unit_match` raises listing
`plan.errors`.

Either receipt carries `summary["inputs"]`: one `UnitMatchInputSummary` per
input, in chronological order, with its identity, `source_kind` / `source_id`,
constituent `nwb_file_names` / `interval_list_names`, and whether its traces
were motion-corrected. `describe_run(summary)` renders one `input_<i>` row per
input; `UnitMatch.get_input_provenance(key, from_nwb=False)` returns the same
provenance as DataFrames.

The orchestrator wraps these table calls (the direct form calls
`UnitMatchSelection.insert_inputs(curations, matcher_params_name)` instead of
`insert_selection`). `initialize_v2_defaults()` installs the `unitmatch_default`
`MatcherParameters` row; a `SessionGroup` of days-apart sessions needs
`allow_multi_day=True`.

```python
from spyglass.spikesorting.v2.unit_matching import (
    TrackedUnit,
    UnitMatch,
    UnitMatchSelection,
)

# Pin the exact curation per member_index (no implicit "latest").
selection_key = UnitMatchSelection.insert_selection(
    "my_team",
    "implant_week1",
    "unitmatch_default",
    {0: curation_day1, 1: curation_day2},
)
UnitMatch.populate(selection_key)
pairs = UnitMatch().get_pairs(selection_key)  # UnitMatch.Pair as a DataFrame
TrackedUnit.populate(selection_key)  # one TrackedUnit per matched group
regions = TrackedUnit().get_unit_brain_regions(
    {**selection_key, "tracked_unit_id": 0}
)  # per-session sorting_id / unit_id / region_name
```

Key behaviors and caveats:

- **Explicit, reproducible curations.** `insert_selection` verifies each pinned
    curation belongs to its member, and `UnitMatch.make()` re-checks every
    input's provenance (`UnitMatchSelectionIntegrityError`), so a direct insert,
    a recreated curation, or changed source content cannot silently match the
    wrong units. A concatenation input whose member `Recording` changed or is
    gone is refused at selection and at make. `get_member_spike_times` and
    `get_unit_brain_regions` run the same check before reading live sources;
    brain regions are read live, so a corrected `Electrode` region shows up.
- **Waveforms come from the traces the sorter read** (masked, and corrected when
    a correction was selected). Each unit's sampled spikes -- only where the
    full waveform window lies in one statistics span -- are split in temporal
    order into two halves (UnitMatch's split-half templates), so a unit present
    in only part of a session is still matchable. Unlike upstream UnitMatchPy
    extraction, Spyglass draws a random spike sample, averages with the mean,
    and applies no Gaussian smoothing. A unit with fewer than two such spikes is
    left unmatched (it stays in the matchable universe).
- **Small unit counts destabilize the match calibration.** UnitMatch fits its
    threshold, prior and score distributions from the units in each run, so with
    about 20 or fewer units per session results can vary run to run and include
    bursts of false matches. UnitMatchPy 3.2.7 raised an `IndexError`
    (`get_threshold`) at about 12 units in one session in synthetic trials.
    Match only well-isolated units and treat small groups with caution
    (UnitMatch issues [#87](https://github.com/EnnyvanBeest/UnitMatch/issues/87),
    [#146](https://github.com/EnnyvanBeest/UnitMatch/issues/146),
    [#170](https://github.com/EnnyvanBeest/UnitMatch/issues/170)).
- **Tracked units are a strict partition.** `TrackedUnit` groups units that
    match *every* other input in the group (a greedy maximal-clique cover,
    largest first, ties by median edge probability), so each curated unit
    belongs to exactly one tracked unit. If A↔B and B↔C match but A↔C does not,
    A and C land in different tracked units. An unmatched unit is a singleton
    (`n_matching_inputs == 1`, `median_match_probability` NULL). A universe
    larger than `max_strict_nodes` (default 2000) raises
    `TrackedUnitBudgetExceededError`.
- **`n_matching_inputs` vs. `n_sessions_detected`.** `n_matching_inputs` counts
    distinct matching inputs (2 for a two-day match, concatenated or not).
    `n_sessions_detected` counts distinct original sessions (`nwb_file_name`)
    where a member unit actually has spikes (from
    `UnitMatch.RecordingSpikeCount`); two intervals of one session count once.
    They can differ for concatenation inputs.
- **Per-recording spike times and brain regions.**
    `TrackedUnit.get_unit_brain_regions` resolves each member unit's region per
    constituent recording (never copied from a concatenation's anchor member)
    and returns `n_spikes` / `detected`. `TrackedUnit.get_member_spike_times`
    returns spike times on each original recording's own clock (concatenation
    spikes are split by the frozen frame spans and mapped back, as
    `ConcatMemberCuration` does).

#### Matching corrected daily sorts assumes the days already line up

**Motion correction registers each day to that day's own mean position, not
across days**, and there is no cross-day registration step. In synthetic
simulations on a 32-contact single-column polymer layout (26 µm pitch), a rigid
offset of 3 / 6 / 12 µm between two corrected days recovered 22 / 12 / 1 of 24
planted neurons. Before matching corrected daily sorts, confirm by other means
(e.g. comparing the days' displacement estimates) that the days are registered
to within a few micrometers.

#### Scientific evidence: matching independently sorted daily concatenations

Evidence is **synthetic only**; no real multi-day lab recording has been
evaluated. A ground-truth benchmark
(`tests/spikesorting/v2/scripts/unitmatch_daily_concat_benchmark.py`) runs the
production matching code over simulated two- and three-day concatenations with
planted neurons on a 16-channel, 2-column probe, with pass thresholds set on
development seeds and met on held-out seeds; recall for gradually drifting and
deliberately conflicting neurons stays low. An end-to-end workflow test with
per-day motion correction, on a layout where the corrected days line up,
recovered 19 of 22 cross-day neurons (13 of 22 without correction).

### Downstream consumers

v1 and single-session v2 curations register on the same `SpikeSortingOutput`
merge table, so existing downstream code (decoding, ripple detection, etc.)
keeps working. A concat curation registers through one `ConcatMemberCuration`
row per member (see the [downstream merge gate](#chronic-same-day-recordings)).

**Every merge id identifies a registered output, not a filtered population**:
`SpikeSortingOutput().get_spike_times({"merge_id": ...})` returns every unit of
that curation, labels ignored, and automatic labels are suggestions, not
approval. The supported handoff is
`select_units_for_analysis(curation, policy=...)` on the curation you actually
reviewed (`run.auto_labeled_curation`, a `FigPackReview` result, or a
`save_manual_curation` / `commit_merges` child). It applies a named
`UnitSelectionParams` policy (`v2_accepted_single_units`,
`v2_accepted_neural_units`, `v2_unflagged_units`, or the expert `all_units`; see
the
[policy table](./SpikeSortingV2_Quickstart.md#4-select-the-analysis-population-then-analyze)),
builds the `SortedSpikesGroup` that decoding and firing-rate consumers read (one
per member session for a concat sort), and returns a receipt with the pinned
curation generation, the policy content, and every unit's verdict and reason.
The shipped rules never write `accept`, so after auto-labeling alone the
`accepted` policies select nothing.

```python
from spyglass.spikesorting.v2.pipeline import select_units_for_analysis

# Auto-labeled only: keep every unit the rules did not flag. The default
# accepted policy would select nothing here.
receipt = select_units_for_analysis(
    run.auto_labeled_curation, policy="v2_unflagged_units"
)
receipt.describe()  # unit_id -> included, labels, reason
spike_times, unit_ids = receipt.fetch_spike_data(return_unit_ids=True)
receipt.group_key  # SortedSpikesGroup key for decoding
```

For an analysis-specific population, add metric criteria from one explicit
evaluation of the same curation:

```python
evaluation = final_curation.evaluate(
    metric_params_name="minimal", auto_curation_rules_name="none"
)
selection = select_units_for_analysis(
    final_curation,
    policy="v2_accepted_single_units",
    evaluation=evaluation,
    unit_criteria={"snr": {">=": 5}},
)
spikes, identities = selection.fetch_spike_data(return_unit_ids=True)
```

Criteria use the `UnitSelectionParams` operators. Missing values fail the
predicate; a missing column or an evaluation of another curation is an error.
Membership (even empty), criteria, label policy, evaluation ID/recipes, and
selected annotation sets are frozen in `SortedSpikesGroup.UnitSelection`;
fetching and decoding read that population, and later policy edits cannot change
it. Each concatenated member gets a session-scoped group with the same unit
decisions. Across groups, identify units by `(spikesorting_merge_id, unit_id)`;
the [whole-session notebook](../notebooks/10_Spike_SortingV2_Presets.ipynb)
assembles one population from per-group final curations (failed or unreviewed
groups stay pending until reviewed or explicitly omitted).

Native splitting, per-spike deletion, unit-specific valid-time editing,
selective unmerge preserving later edits, and Phy edit re-import are
unsupported; Phy export is for inspection only.

#### What do I call next?

| Goal                                            | Call                                                                                                                                                      |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Selected spike times (the supported path)       | `select_units_for_analysis(curation).fetch_spike_data()`                                                                                                  |
| All units of one registered output              | `SpikeSortingOutput().get_spike_times({"merge_id": merge_id})`                                                                                            |
| Recording                                       | `SpikeSortingOutput().get_recording({"merge_id": merge_id})`                                                                                              |
| Original source recording (unmasked)            | `SpikeSortingOutput().get_source_recording({"merge_id": merge_id})`                                                                                       |
| Traces the sorter read (masked, corrected)      | `SpikeSortingOutput().get_sorting_input_recording({"merge_id": merge_id})`                                                                                |
| Sorting                                         | `SpikeSortingOutput().get_sorting({"merge_id": merge_id})`                                                                                                |
| Unit brain regions                              | `SpikeSortingOutput.get_unit_brain_regions({"merge_id": merge_id})`                                                                                       |
| Curation summary (the curated result)           | `CurationV2.summarize_curation(auto_summary.auto_labeled_curation.as_key())` (`auto_summary.root_curation.as_key()` inspects the uncurated root)          |
| Unit-level plots / exports of an exact curation | `ssviz.plot_waveforms(curation, unit_ids=[...])`, `ssviz.export_to_phy(curation, folder)`                                                                 |
| Analyzer/debug internals                        | `Sorting().get_analyzer({"sorting_id": run_summary["sorting_id"]})` (raw sort); `open_curation_analyzer(curation, recipe)` for a disk-backed working copy |
| v2 merge ids for a restriction                  | `get_spike_sorting_v2_merge_ids(restriction)` (`spyglass.spikesorting.v2.utils`)                                                                          |

`SpikeSortingOutput.get_restricted_merge_ids` includes v2 by default. With an
explicit `sources=` list the v2 resolver is strict: an unknown restriction key
raises `ValueError`.

**Clusterless decoding.** `UnitWaveformFeatures` extracts per-spike features for
a v2 `merge_id` under SpikeInterface 0.104: `amplitude` (used by clusterless
decoding), `full_waveform`, and `spike_location` (v2-only); other features raise
`NotImplementedError`. A zero-unit curation yields an empty-but-valid row. v2
amplitudes are in µV while v0/v1 used raw counts, so retrain decoders per
pipeline version.

### Paper export

A v2 `merge_id` exports the same way as a v1 one:

```python
from spyglass.common.common_usage import Export, ExportSelection
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

ExportSelection().start_export(paper_id="my_paper", analysis_id=1)
SpikeSortingOutput().fetch_nwb({"merge_id": merge_id})
ExportSelection().stop_export()

Export().populate_paper(paper_id="my_paper")
```

`Export.File` contains the curated units NWB, the intermediate sort NWB, and the
upstream preprocessed-recording cache, pulled in by `Export.populate_paper`'s
foreign-key cascade; you do not need to call `get_recording` / `get_sorting`
during the export (they do not log export events). Zero-unit curations export
the same way.

### Provenance in each v2 NWB

Every v2 analysis NWB embeds the lineage needed to interpret it without the
database. Read a container with `nwbfile.get_scratch(name)`, which returns a
DataFrame (a scalar header is a `key` / `value_json` table with JSON-encoded
values; a relational table has one row per member / unit / pair). Large arrays
(recording fingerprint, motion displacement, templates) are not duplicated; only
the producing params are written.

| Artifact                    | Container(s)                                                                                                                            | Carries                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Recording                   | `spyglass_v2_recording_provenance`                                                                                                      | raw source `object_id`, `recording_id`, preprocessing recipe, sort group, resolved reference mode, bad-channel handling, SpikeInterface version                                                                                                                                                                                                                                                                                                                 |
| Sorting                     | `spyglass_v2_sorting_provenance` + per-unit Units columns                                                                               | `peak_amplitude_uv` / `peak_electrode_id` / `n_spikes` / `brain_region` columns (matching `Sorting.Unit`), and a header with the recording/concat id, sorter + params, `artifact_detection_id`, display recipe, effective seed, SI + sorter versions                                                                                                                                                                                                            |
| Curated units               | `spyglass_v2_curation_provenance` + `spyglass_v2_curation_merge_lineage`                                                                | curation header (sorting/curation id, immutable `curation_uuid`, parent, source, `merges_applied`, description) and the kept→contributor merge lineage mirroring `CurationV2.MergeGroup` (raw contributors; proposed-vs-applied is the header's `merges_applied`)                                                                                                                                                                                               |
| Concat member curated units | `spyglass_v2_curation_provenance` + `spyglass_v2_curation_merge_lineage`                                                                | the chosen concat curation provenance plus `member_index` / member `nwb_file_name`; wall-clock times and local sample frames, with curated unit IDs preserved across members                                                                                                                                                                                                                                                                                    |
| UnitMatch                   | `spyglass_v2_unitmatch_provenance` + `spyglass_v2_unitmatch_inputs` + `spyglass_v2_unitmatch_input_recordings`                          | run/matcher header (matcher backend + versions) and the per-matching-input `(sorting_id, curation_id, curation_uuid, source_kind, source_id, input_start_time, waveform_traces, motion_corrected_recording_id)` table plus the per-constituent-recording `(nwb_file_name, interval_list_name, recording_id, session_start_time, start_sample, end_sample)` table                                                                                                |
| CurationEvaluation          | `spyglass_v2_curation_evaluation_provenance`                                                                                            | metric set + recipe names, auto-merge preset/rules, evaluated curation, the `source_analyzer_hashes` manifest, SI version, upstream recording/concat `content_hash`                                                                                                                                                                                                                                                                                             |
| ConcatenatedRecording       | `spyglass_v2_concat_provenance` + `spyglass_v2_concat_members`                                                                          | member artifact detection IDs, excluded frame ranges, valid observation intervals, and the ordered member frame boundaries used for session mapping                                                                                                                                                                                                                                                                                                             |
| MotionCorrectedRecording    | `spyglass_v2_motion_correction_provenance` + `spyglass_v2_motion_continuity_spans` (+ `spyglass_v2_concat_members` for a concat source) | estimation + interpolation recipe names, the resolved interpolation config, SpikeInterface version, the estimate and corrected-recording ids, the application algorithm version, the source (kind, key, `content_hash`), any `remove_channels`-dropped channel ids, the statistics spans, each continuity span's frames with its first/last source timestamp and its start on the estimation clock, and for a concat source the concatenation's member back-map |

```python
import pynwb
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.v2.sorting import Sorting

abs_path = AnalysisNwbfile.get_abs_path(
    (Sorting & {"sorting_id": sorting_id}).fetch1("analysis_file_name")
)
with pynwb.NWBHDF5IO(abs_path, "r", load_namespaces=True) as io:
    prov = io.read().get_scratch("spyglass_v2_sorting_provenance")
```

### Streaming writes and parallel populate

- **Streaming `Recording` writes.** `Recording.make` streams the preprocessed
    `ElectricalSeries` to NWB in chunks (a channel-count-scaled buffer of ≈30 s,
    capped at 5 GB), never holding the full trace array in RAM. This bounds the
    write, not the whole workflow: sorting, analyzers and browser payloads have
    their own costs, and hour-long lab recordings have not been measured. See
    [Release workload measurement](./SpikeSortingV2StorageManagement.md#release-workload-measurement).
- **Parallel populate.** `Recording`, the artifact-detection result tables, and
    `Sorting` compute outside DataJoint's framework transaction, so a long sort
    does not hold row locks that block other users. Set
    `dj.config["custom"]["spikesorting_v2_job_kwargs"] = {"n_jobs": N}` to use N
    workers in every compute stage.

### Environment

The v2 pipeline requires SpikeInterface 0.104+ and, for MountainSort, the
`spikesorting-v2` extra. Browser review needs `spikesorting-v2-curation`;
cross-session matching needs `spikesorting-v2-matching`:

```bash
pip install "spyglass-neuro[spikesorting-v2]"
```

Producing new v0/v1 output (v0/v1 artifact detection, `Waveforms`,
`QualityMetrics`, `MetricCuration`, `BurstPair`, and clusterless `UnitMarks` /
`UnitWaveformFeatures` for v0/v1 sorts) still requires the SI 0.99 environment.
See
[Two environments, one database](./SpikeSortingV2_Migration.md#two-environments-one-database).

### Observed-time metrics and downstream analysis

The standard `franklab_hippocampus_2026_09_17` review profile uses
`observed_duration_s`, `observed_firing_rate_hz`, and `observed_presence_ratio`.
These use stored observation intervals, normalized through recording sample
boundaries so the last sample contributes its full duration. Time is never
compressed across exclusions or recording gaps.

Observed firing rate counts spikes in usable spans and divides by their total
duration. Observed presence divides the observed duration of occupied bins by
total observed duration. Bins are fixed on the original timeline, anchored at
the recording's first timestamp; at least one observed spike makes a bin
occupied. Partial bins contribute only their usable duration; entirely excluded
bins contribute nothing.
`QualityMetricParameters.observed_presence_bin_duration_s` defaults to 60
seconds. Zero exposure produces unavailable rate/presence values; observed
silence produces zero. This presence definition is distinct from SI's.

| Metric family                                                      | Time/exclusion semantics                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `observed_*` columns                                               | Sample-exact usable-time duration and exposure-weighted presence.                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Raw SI `firing_rate`, `presence_ratio`, `firing_range`             | Retain SI's full-timeline definitions; use observed columns for artifact-adjusted decisions.                                                                                                                                                                                                                                                                                                                                                                                                      |
| `isi_violation`                                                    | Violating-interval fraction over retained spikes, without a duration denominator; shipped rules use this, not SI's contamination ratio. Original spike timing is preserved.                                                                                                                                                                                                                                                                                                                       |
| SI `isi_violations_ratio`, refractory-period contamination metrics | Duration-dependent SI estimates; optional expert diagnostics, not default artifact-adjusted decisions.                                                                                                                                                                                                                                                                                                                                                                                            |
| SNR, amplitude/noise overlap, waveform/template metrics            | Retain SI's metric definitions and time semantics (no `observed_*` duration accounting). The noise/whitening estimate behind SNR's denominator, `sd_ratio`'s noise standard deviation, and `nn_noise_overlap`'s noise cluster samples only the sort's artifact-free statistics spans, not SI's raw whole-recording sampling recipe -- masking is still not a claim of universal metric correction (template/waveform amplitudes are unchanged). Sparse/insufficient evidence remains unavailable. |

Evaluations record the observation-definition version, interval fingerprint, and
presence-bin width. The shipped rules use noise overlap and `isi_violation`;
neither silently substitutes a duration-based contamination estimate.

`select_units_for_analysis` freezes each included unit's observed intervals in
the same snapshot as membership. `selection.observation` (one session) or
`selection.groups[i].observation` exposes `intervals`, `duration_s`, and
`unknown_sources`. The common intervals are the **intersection** of selected
units' availability; empty/unselected groups impose no restriction. Concatenated
member snapshots are mapped to original session seconds before analysis.

`SortedSpikesGroup.get_spike_indicator(key, time, return_validity=True)` returns
counts and a validity mask on the caller's time axis. Bins crossing exclusions
contain NaN; valid bins without spikes contain zero. Firing-rate smoothing runs
separately within observed spans. Sorted-spikes decoding restricts encoding and
decoding intervals, ANDs training masks with availability, and ORs missing masks
with unavailable time. Ordinary prediction preserves gaps; parameter estimation
labels missing time `-1`. Saved results include effective intervals. An empty
training or decoding interval raises a clear error.

Legacy/imported populations without snapshots contribute no observation
restriction and are listed in `unknown_sources`. Mixed populations apply known
restrictions without inventing missing metadata. Other custom downstream
analyses must use the exposed intervals explicitly; observed-time support does
not redefine every external SI metric or implement clusterless masking.

### Manual recording exclusions

`run_v2_pipeline(..., manual_excluded_times=[[start, stop], ...])` adds manual
artifact exclusions to automatic detection. Intervals are half-open
`[start, stop)` in the **original session's seconds**, not sample indices. They
are normalized, stored on the artifact selection, and included in its identity,
so changing them creates a different artifact result and downstream sort. A
preset with automatic detection disabled still applies them. The artifact
recipe's `min_length_s` applies to the remaining valid spans. The session runner
applies the supplied intervals to each requested sort group.

For concatenated sorting, pass
`manual_excluded_times={member_index: [[start, stop], ...]}`, each in that
member's original session timestamps. Automatic and manual masks are composed
before concatenation and survive reconstruction and member export; an optional
motion stage excludes the same masked samples and re-applies the mask after
correction. Manual exclusions mask time ranges; they do not edit individual
spikes.

Development databases created before this release must follow the
[preproduction upgrade/recreation sequence](SpikeSortingV2_Migration.md#upgrading-a-preproduction-v2-database)
before initializing defaults or creating selections. No production migration
runs automatically.
