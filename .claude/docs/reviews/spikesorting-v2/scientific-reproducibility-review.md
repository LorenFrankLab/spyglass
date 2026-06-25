# Spike Sorting V2 Scientific Correctness and Reproducibility Review

Date: 2026-06-25

Scope: scientific meaning of the v2 recording, sorting, analyzer, curation,
concat, and UnitMatch pipeline. This is a different lens from the prior
DataJoint/concurrency review: the question here is whether a row's scientific
inputs and outputs mean what the database lineage says they mean, especially
under rebuild, recompute, package upgrades, and downstream reuse.

Method: local code inspection plus two independent explorer agents. This review
is read-only except for this document.

## Executive Summary

Spike sorting v2 has many strong correctness guards: deterministic selection
ids, tri-part compute/fetch separation, explicit artifact masking, sample-count
checks for concat boundaries, reference-mode validation, curation label
validation, and geometry guards in UnitMatch. The main remaining risks are not
ordinary style issues. They are scientific identity and provenance gaps.

The two highest-priority concrete bugs are in recording construction:

1. V2 reads the first acquisition `ElectricalSeries` instead of the
   `Raw.raw_object_id` selected during common-table ingest.
2. V2 maps `electrode_id` to raw `channel_name` by using the electrode id as an
   NWB electrodes-table row index.

Either can silently sort a different signal than the one implied by the
`RecordingSelection` row. Those should be fixed before broader recompute work.

The broader design issue is that selection ids are content-addressed over row
names and keys, while the compute path dereferences mutable upstream state:
interval arrays, sort-group membership/reference state, raw-series selection,
bad-channel flags, preprocessing implementation semantics, concat membership,
software versions, and runtime job kwargs. The recording-content fingerprint
plan is the right direction, but it should include the concrete source-series
and channel-mapping fixes below.

## Findings

### 1. High: Raw source series is not pinned to `Raw.raw_object_id`

`Raw` stores the exact NWB object id for the ingested raw `ElectricalSeries`
(`src/spyglass/common/common_ephys.py:286-292`) and `Raw.nwb_object` resolves
that object id explicitly (`src/spyglass/common/common_ephys.py:377-389`).
V2 does not use it. Instead, `raw_eseries_path_and_timestamp_mode` iterates
`/acquisition` and returns the first object whose `neurodata_type` is
`ElectricalSeries` (`src/spyglass/spikesorting/v2/_recording_nwb.py:32-53`).
`Recording._compute_recording_artifact` then passes that path to
`read_raw_nwb_recording` (`src/spyglass/spikesorting/v2/recording.py:1697-1704`).

Impact: an NWB with multiple acquisition `ElectricalSeries` objects can produce
v2 traces from a different source than the common `Raw` row, while intervals,
sampling rate, and database lineage still point at the ingested raw object.
Repacked or copied NWBs can also change acquisition iteration order.

Fix direction:

- Resolve the raw `ElectricalSeries` path from `Raw.raw_object_id`, not by
  taking the first acquisition series.
- Thread that path through timestamp-mode detection and
  `read_raw_nwb_recording`.
- Persist the resolved source path or object id in the v2 recording provenance.
- Add a regression NWB with two acquisition `ElectricalSeries` objects where
  only the second is the `Raw.raw_object_id`.

### 2. High: `channel_name` lookup treats `electrode_id` as a table row index

`spikeinterface_channel_ids` reads the NWB electrodes table and, when
`channel_name` exists, returns `channel_names[int(c)]` for each Spyglass
electrode id (`src/spyglass/spikesorting/v2/_recording_geometry.py:34-70`).
`restrict_recording` then slices the raw recording by those SI channel ids and
renames them back to the requested electrode ids
(`src/spyglass/spikesorting/v2/_recording_restriction.py:573-578`).

The write path already knows this distinction: `electrode_table_region` maps
electrode ids to NWB table row indices via `get_electrode_indices`
(`src/spyglass/spikesorting/v2/_nwb_metadata_helpers.py:69-114`).

Impact: for NWBs whose electrode ids are non-contiguous, non-zero-based, or
not equal to row positions, v2 can slice raw channel B and rename it as
electrode A. Downstream units, peak channels, regions, and curation provenance
then describe one electrode while the sorter saw another channel's signal.

Fix direction:

- Use `get_electrode_indices` before indexing the `channel_name` column.
- Validate that every requested electrode id is present and maps to exactly one
  channel name.
- Add regression coverage with shuffled or non-zero electrode ids and a
  `channel_name` column.

### 3. High: recording and concat ids do not fingerprint the full construction recipe

`RecordingSelection` identity contains only `nwb_file_name`, `sort_group_id`,
`interval_list_name`, `preprocessing_params_name`, and `team_name`
(`src/spyglass/spikesorting/v2/_selection_identity.py:40-46`,
`src/spyglass/spikesorting/v2/_selection_identity.py:158-206`). The actual
recording build later dereferences sort-group electrodes/reference state,
interval contents, raw valid times, preprocessing params/job kwargs,
bad-channel flags, probe metadata, and raw NWB content
(`src/spyglass/spikesorting/v2/recording.py:1055-1150`).

Concat selection identity is similarly row-name based:
`SessionGroup`, `PreprocessingParameters`, and `MotionCorrectionParameters`
(`src/spyglass/spikesorting/v2/session_group.py:330-358`). The compute path
then resolves the ordered member list and member recording ids at populate time
(`src/spyglass/spikesorting/v2/session_group.py:747-779`) and writes member
boundaries only after loading the member recordings
(`src/spyglass/spikesorting/v2/session_group.py:885-951`).

Impact: if upstream row contents change under the same names, the same
`recording_id` or `concat_recording_id` can be rebuilt into a different signal,
different channel surface, or different concat boundary plan. This is the core
recompute risk.

Fix direction:

- Add a stable recording construction fingerprint that covers raw source object
  id/path, selected electrodes and channel-name mapping, sort intervals, raw
  coverage, reference settings, preprocessing params content and schema,
  bad-channel/interpolation inputs, geometry, and relevant runtime provenance.
- For concat, include ordered member recording ids plus the member sample counts
  used for boundary mapping, and the resolved motion preset.
- Either include these fingerprints in identity or validate them before
  populate/rebuild and fail closed on mismatch.

### 4. High: stochastic and runtime provenance is only partly represented in rows

The code deliberately pins some stochastic paths, which is good. External
whitening reads `random_seed` from resolved job kwargs
(`src/spyglass/spikesorting/v2/_sorting_dispatch.py:504-515`), analyzer
`random_spikes` does the same
(`src/spyglass/spikesorting/v2/_sorting_analyzer.py:629-650`), and the
clusterless MAD path pins random slices
(`src/spyglass/spikesorting/v2/_sorting_dispatch.py:319-341`).

The gap is where the seed and runtime can come from. `_resolved_job_kwargs`
merges SpikeInterface global defaults, `dj.config['custom']
['spikesorting_v2_job_kwargs']`, and per-row blobs
(`src/spyglass/spikesorting/v2/utils.py:608-633`). Only the per-row blobs are
database provenance. A process-level or config-level `random_seed` can change
whitening, analyzer subsampling, UnitMatch bundle extraction, and clusterless
detection without changing a parameter row.

Local sorter execution also cannot pin package versions in
`SorterExecutionParamsSchema`; container versions can be pinned, but
`backend="local"` rejects `spikeinterface_version` and related fields
(`src/spyglass/spikesorting/v2/_params/sorter.py:331-437`). The MS4 schema
itself notes the sorter is nondeterministic
(`src/spyglass/spikesorting/v2/_params/sorter.py:47-55`).

Impact: rerunning the same selection after a local package upgrade, a changed
global seed, or a sorter runtime change can produce different units, metrics,
or curations with the same v2 ids.

Fix direction:

- Treat science-affecting knobs such as `random_seed` as per-row parameters,
  not global job defaults. Reject or warn on `random_seed` from global/DJ config.
- Persist the effective scientific job kwargs used by each computed artifact.
- Record local runtime provenance on computed artifacts: Spyglass version or git
  SHA, SpikeInterface version, sorter package/runtime version where available,
  UnitMatchPy version, and container digest/image where applicable.
- For reproducible sorter rows, prefer pinned containers or an explicit local
  runtime provenance row.

### 5. High: UnitMatch identity omits effective matcher and waveform-bundle inputs

`MatcherParameters` stores the matcher name, a small params blob, schema
version, and job kwargs (`src/spyglass/spikesorting/v2/unit_matching.py:99-110`).
The UnitMatch params schema intentionally exposes only v2-owned controls:
thresholds and graph cap (`src/spyglass/spikesorting/v2/_params/matcher.py:7-41`).
The backend resolves UnitMatchPy defaults dynamically and only overrides
`match_threshold` (`src/spyglass/spikesorting/v2/_unitmatch_backend.py:232-238`).

The dense waveform bundle has scientific parameters that are not in
`MatcherParameters`: `ms_before`, `ms_after`, `max_spikes_per_unit`, and `seed`
(`src/spyglass/spikesorting/v2/_unitmatch_backend.py:107-157`). The table calls
`extract_unitmatch_bundle` without passing or persisting those values
(`src/spyglass/spikesorting/v2/unit_matching.py:794-799`).

Impact: the same `unitmatch_id` can produce different pair probabilities,
pairs, and tracked units after UnitMatchPy default changes, Spyglass bundle
parameter changes, or runtime seed changes.

Fix direction:

- Promote bundle extraction params into `MatcherParameters`.
- Store the full effective UnitMatchPy parameter snapshot, or at least a stable
  hash plus UnitMatchPy version.
- Include these effective inputs in `unitmatch_id` or validate them before
  populate/rebuild.

### 6. Medium-High: curation lineage and tracked-unit universes are not frozen enough

`CurationV2.parent_curation_id` is documented as validation-only lineage
(`src/spyglass/spikesorting/v2/curation.py:64-72`). `insert_curation` always
starts from raw `Sorting.Unit` rows (`src/spyglass/spikesorting/v2/curation.py:311-354`),
validates that the parent exists (`src/spyglass/spikesorting/v2/curation.py:550-638`),
and writes curated units from the original `Sorting` NWB
(`src/spyglass/spikesorting/v2/_units_nwb.py:707-724`). Parent labels, parent
exclusions, and parent applied merges are not inherited unless the caller
re-supplies the complete desired state.

Separately, `UnitMatch.make_fetch` freezes `matchable_unit_ids` transiently in
the `member_plan` (`src/spyglass/spikesorting/v2/unit_matching.py:580-631`),
but does not persist that universe. `TrackedUnit.make` later re-derives the
node universe from current curation labels
(`src/spyglass/spikesorting/v2/unit_matching.py:864-930`). `CurationV2.UnitLabel`
part rows are directly insertable with validation
(`src/spyglass/spikesorting/v2/curation.py:108-152`).

Impact: a child curation can look like a continuation of a parent while actually
being a fresh full-state curation over raw sorter units. Label changes after a
UnitMatch run can also change singleton tracked units or force loud failures
under the same `unitmatch_id`.

Fix direction:

- Decide whether child curations are deltas from parent state or complete
  snapshots. If they are snapshots, rename/document the API more strongly and
  require callers to provide complete state.
- If they are deltas, inherit parent units, labels, exclusions, merge state, and
  spike trains before applying overrides.
- Persist the per-member matchable unit set, or at least its hash, on UnitMatch
  output and derive `TrackedUnit` from that frozen universe.

### 7. Medium: runtime aliases and semantic changes need explicit versions

Preprocessing runtime order changed to phase-shift, bandpass, interpolate bad
channels, then reference (`src/spyglass/spikesorting/v2/_recording_preprocessing.py:31-239`).
The schema docs explicitly note that the order changed at schema version 3 and
that the params blob shape was unchanged, so the runtime interpretation moved
without another bump (`src/spyglass/spikesorting/v2/_params/preprocessing.py:106-126`).
For `global_median`, filter-before-reference and reference-before-filter are
not equivalent.

Motion correction has a similar smaller alias risk: `preset="auto"` is stored
in the params row (`src/spyglass/spikesorting/v2/_params/motion_correction.py:74-106`),
then resolves through the current `AUTO_SAME_DAY_PRESET = "rigid_fast"`
(`src/spyglass/spikesorting/v2/_concat_recording.py:19-23`,
`src/spyglass/spikesorting/v2/_concat_recording.py:93-146`). The written
description records the resolved label, but selection identity still uses the
params row name (`src/spyglass/spikesorting/v2/session_group.py:872-883`).

Impact: the same parameter row can mean different traces after a code change to
runtime order or alias resolution.

Fix direction:

- Add explicit runtime-version fields for preprocessing semantics, or bump
  schema/row names when order changes affect signal values.
- Resolve `auto` aliases at insert time, or store the resolved preset in the
  selection/computed row and include it in the construction fingerprint.

### 8. Medium: analyzer and auto-curation provenance has a few remaining soft spots

The analyzer base extensions include `noise_levels`
(`src/spyglass/spikesorting/v2/_sorting_analyzer.py:617-628`). The code comments
call it unseeded in recompute context, and explicit params are pinned only for
`random_spikes` and `waveforms`
(`src/spyglass/spikesorting/v2/_sorting_analyzer.py:629-655`). Quality metrics
and auto-curation consume analyzer extensions
(`src/spyglass/spikesorting/v2/metric_curation.py:1073-1115`).

Materialized analyzer curations also lose the exact `AnalyzerCuration` source.
`AnalyzerCurationSelection` has a content-addressed identity over curation,
metric params, rules, and metric waveform recipe
(`src/spyglass/spikesorting/v2/metric_curation.py:645-724`), but
`materialize_curation` inserts a `CurationV2` row with only
`curation_source="analyzer_curation"` and no FK/hash back to the producing
`AnalyzerCuration` row (`src/spyglass/spikesorting/v2/metric_curation.py:1383-1417`).

Impact: analyzer rebuilds or extension default changes can alter quality
metrics. Also, two analyzer-curation selections that yield the same labels and
merge groups become hard to distinguish once materialized into `CurationV2`.

Fix direction:

- Pin `noise_levels` params/seed if SpikeInterface supports it, or replace it
  with deterministic full-recording estimation.
- Persist effective analyzer extension params/hashes.
- Add a source part/FK from `CurationV2` materializations back to
  `AnalyzerCurationSelection` or `AnalyzerCuration`.

## Positive Observations To Preserve

- Deterministic selection ids are a strong foundation for idempotent insertion.
- Recording writes correctly map output electrode ids to NWB table rows via
  `electrode_table_region`; the input channel slicing should reuse that pattern.
- Artifact shared groups validate timestamp fingerprints before shared masking.
- Concat validates channel/geometry compatibility and guards sample-count
  preservation before inserting `MemberBoundary` rows.
- UnitMatch validates same-probe geometry and detects UnitMatchPy silently
  dropping a session bundle.
- Curation label values are validated even on direct part-table inserts.

## Recommended Fix Order

1. Fix raw `ElectricalSeries` resolution from `Raw.raw_object_id`.
2. Fix `channel_name` resolution by mapping electrode ids to electrodes-table
   row indices.
3. Fold both fixes into the recording-content fingerprint plan.
4. Reject or persist global science-affecting job kwargs, especially
   `random_seed`.
5. Add runtime/software provenance capture for local sorting, analyzer rebuilds,
   and UnitMatch.
6. Freeze UnitMatch bundle/effective parameters and matchable unit universes.
7. Clarify curation child semantics and analyzer-curation materialization
   provenance.

## Suggested Tests

- NWB with two acquisition `ElectricalSeries` objects, where `Raw.raw_object_id`
  points to the second.
- NWB with non-contiguous/shuffled electrode ids and `channel_name`.
- Rebuild a recording after changing only upstream interval contents under the
  same interval name; expect fingerprint mismatch/fail-closed behavior.
- Same parameter rows with `random_seed` supplied globally vs per-row; expect
  either rejection or persisted effective seed provenance.
- UnitMatch run with changed bundle params or UnitMatchPy default snapshot;
  expect a different matcher-parameter fingerprint.
- Label mutation after UnitMatch but before TrackedUnit; expect frozen-universe
  behavior or an explicit, early mismatch error.
