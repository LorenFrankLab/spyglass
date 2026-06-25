# Spike Sorting V2 NWB Portability Review

Date: 2026-06-25

Scope: whether spike sorting v2 analysis NWB artifacts are self-describing and
interoperable when opened outside the originating Spyglass/DataJoint database.
This is a different lens from internal correctness: the question is what a
portable consumer can reconstruct from the files alone.

Method: local code inspection plus two independent explorer agents. This review
is read-only except for this document.

## Executive Summary

Spike sorting v2 is much stronger as a DataJoint-backed cache than as a
standalone NWB exchange format. The result payloads are often good: processed
recording traces have timestamps, conversion/offset, electrode regions, and
filtering strings; sorting NWBs store absolute spike times plus
`spike_sample_index`; analyzer curation and UnitMatch write result tables that
round-trip through DB-free helpers.

The portability gap is provenance and reconstruction context. Many facts needed
to interpret or re-use an artifact live only in DataJoint rows: raw source
series, recording construction inputs, concat member boundaries and time
transforms, sorter/runtime parameters, per-unit peak electrode metadata, curation
merge lineage, metric/rule parameters, UnitMatch session context, and tracked
unit membership. A lab with the original database can recover those; a lab with
only the NWB cannot.

The most useful fix is not to duplicate every table in ad hoc strings. Add a
small structured `spyglass_spikesorting_v2` processing module, or an equivalent
set of scratch/processing tables, and make generated NWBs pass a "no DataJoint"
readback test: open the file with PyNWB only and recover the source identity,
construction recipe, unit metadata, and session mapping needed to interpret the
artifact.

## Findings

### 1. High: recording NWBs do not carry raw-source and construction provenance

`Recording.make_fetch` resolves the scientific construction inputs from
DataJoint: selected NWB, sort group, interval, channel ids, reference mode,
preprocessing params/job kwargs, raw valid times, and bad-channel interpolation
inputs (`src/spyglass/spikesorting/v2/recording.py:1050-1163`). The NWB writer
then writes one processed `ElectricalSeries` with data, timestamps, electrodes,
conversion, offset, a human-readable `filtering` string, and a generic
description (`src/spyglass/spikesorting/v2/_recording_nwb.py:181-209`).

The file does not encode the raw source `ElectricalSeries` path/object id, raw
NWB identifier/hash, interval rows, sort group/electrode selection, reference
mode/electrode, preprocessing params content/schema, job kwargs, or bad-channel
handling. The table row stores `analysis_file_name`, `electrical_series_path`,
`object_id`, and `cache_hash` (`src/spyglass/spikesorting/v2/recording.py:1011-1021`),
but those are database-side handles, not standalone provenance.

Impact: an exported analysis NWB contains the processed signal, but a portable
reader cannot tell exactly what raw series and recipe produced it or audit that
two files represent the same recording construction.

Fix direction:

- Add structured recording provenance next to the processed `ElectricalSeries`:
  source NWB name/identifier/content hash, raw `ElectricalSeries` path/object id,
  interval list name and valid times, sort group id, ordered electrode ids and
  channel names, reference settings, preprocessing params content/schema/job
  kwargs, bad-channel handling, and software versions.
- Add a PyNWB-only test that opens a generated recording NWB and reconstructs
  this provenance without querying DataJoint.

### 2. High: concat recording NWBs cannot be split or mapped back without DataJoint

Concat intentionally drops member wall-clock gaps:
`build_concatenated_recording` uses a synthetic continuous timeline with
`ignore_times=True` (`src/spyglass/spikesorting/v2/_concat_recording.py:299-303`,
`src/spyglass/spikesorting/v2/_concat_recording.py:342`). The per-member sample
boundaries are stored only in the `ConcatenatedRecording.MemberBoundary` part
(`src/spyglass/spikesorting/v2/session_group.py:592-600`,
`src/spyglass/spikesorting/v2/session_group.py:926-951`). The NWB artifact gets
one `ElectricalSeries` and a text `filtering` value saying how many members and
which preset were used (`src/spyglass/spikesorting/v2/session_group.py:872-884`).

`split_sorting_by_session` proves what is missing from the file: it re-reads the
ordered `SessionGroup.Member` rows and `MemberBoundary` rows from DataJoint to
map concat-frame spikes back to member-local frames
(`src/spyglass/spikesorting/v2/session_group.py:994-1057`).

Impact: a standalone concat NWB cannot explain which sessions contributed,
where each member begins/ends, how concat frame indices map to local frame
indices, or what source intervals/wall-clock ranges were compressed away.

Fix direction:

- Write a concat member table to the NWB with `member_index`, source NWB
  identity, source recording id if available, source `ElectricalSeries` path and
  object id, source interval list/valid times, source sample count,
  `concat_start_sample`, `concat_end_sample`, and the time transform.
- Add a no-DataJoint test that opens only the concat NWB and reconstructs the
  same split boundaries as `split_sorting_by_session`.

### 3. High: motion-corrected concat artifacts have only lossy motion provenance

Motion correction is deliberately collapsed into corrected traces. The concat
builder documents that motion trajectories/info are not persisted and calls
`correct_motion(..., output_motion=False, output_motion_info=False)`
(`src/spyglass/spikesorting/v2/_concat_recording.py:310-313`,
`src/spyglass/spikesorting/v2/_concat_recording.py:376-380`). The NWB records a
text preset label, but not the resolved `"auto"` choice, `preset_kwargs`, job
kwargs, SpikeInterface version, displacement fields, or an explicit "motion
field intentionally not stored" declaration.

Impact: when motion correction is enabled, the traces are portable but the
motion provenance is not. A downstream reader cannot distinguish a fully
auditable corrected trace from a corrected trace whose displacement model was
discarded.

Fix direction:

- Prefer persisting the motion info/trajectory if feasible.
- If not feasible, write structured provenance that says corrected traces are
  stored but motion fields are not, including resolved preset, all kwargs,
  software versions, and the sample-count invariant used for member boundaries.

### 4. High: sorting Units NWBs lack source, sorter, artifact, and runtime context

`SortingSelection` carries the source kind plus `SorterParameters` and optional
`ArtifactDetectionSource` (`src/spyglass/spikesorting/v2/sorting.py:672-710`).
`Sorting.make_fetch` resolves source recording/concat state, sorter params,
artifact intervals, waveform params, and execution params
(`src/spyglass/spikesorting/v2/sorting.py:1208-1310`). The NWB writer receives
only the in-memory sorting, recording, parent NWB name, and `obs_intervals`
(`src/spyglass/spikesorting/v2/sorting.py:2540-2557`,
`src/spyglass/spikesorting/v2/_units_nwb.py:519-671`).

The file stores absolute spike times, sample indices, `obs_intervals`, and an
`uncurated` label (`src/spyglass/spikesorting/v2/_units_nwb.py:627-655`). It
does not store the source `recording_id`/`concat_recording_id`, sorter name and
params, job kwargs, execution backend/container, artifact detection id, artifact
interval source, display waveform recipe, sorter runtime, or software versions.

Impact: a portable consumer can read spikes but cannot explain how those units
were produced, verify that the right input recording was sorted, or recreate
the sorter call.

Fix direction:

- Write a sorting provenance table or processing object with source kind/id,
  recording/concat artifact identity, sorter params, job/runtime params,
  execution backend/container digest, artifact interval provenance, waveform
  recipe, runtime, and software versions.
- Add a PyNWB-only test that reads a sorting NWB and recovers this provenance.

### 5. High: unit-level peak/electrode/region metadata is database-only

`Sorting.Unit` and `CurationV2.Unit` store the `Electrode` FK,
`peak_amplitude_uv`, and `n_spikes`
(`src/spyglass/spikesorting/v2/sorting.py:1153-1169`,
`src/spyglass/spikesorting/v2/curation.py:90-106`). Sorting computes those rows
from the analyzer during DB insertion
(`src/spyglass/spikesorting/v2/sorting.py:2560-2572`). The Units NWB writers do
not add corresponding columns; sorting writes curation label and
`spike_sample_index`, while curated writes labels and sample indices
(`src/spyglass/spikesorting/v2/_units_nwb.py:627-655`,
`src/spyglass/spikesorting/v2/_units_nwb.py:880-922`).

Impact: outside DataJoint, a reader can see spike trains but cannot recover each
unit's peak electrode/channel, peak amplitude, spike count as Spyglass reports
it, or resolved brain region.

Fix direction:

- Add Units columns for `peak_electrode_id`, peak electrode table row or region
  link, `peak_amplitude_uv`, `n_spikes`, and optionally resolved region fields.
- Add integration coverage that opens the generated Units NWB with PyNWB only
  and compares these values to the DB rows.

### 6. High: curated Units NWBs do not export merge lineage

`CurationV2.MergeGroup` is the structured merge provenance table
(`src/spyglass/spikesorting/v2/curation.py:154-190`). The curated NWB writer
uses `kept_unit_to_contributors` to build stored spike trains, including
deduplicated merged units, but it does not serialize that mapping
(`src/spyglass/spikesorting/v2/_units_nwb.py:674-945`). The docstring even notes
that preview merge structure lives in `CurationV2.MergeGroup`
(`src/spyglass/spikesorting/v2/_units_nwb.py:698-702`).

Impact: for `apply_merge=True`, absorbed contributor units disappear from the
standalone NWB. For `apply_merge=False`, proposed merge groups are invisible in
the standalone NWB. In both cases, the curated file does not carry enough
lineage to audit curation decisions outside the database.

Fix direction:

- Write curation provenance with `sorting_id`, `curation_id`,
  `parent_curation_id`, `curation_source`, `merges_applied`, `kept_unit_id`,
  `contributor_unit_id`, and merge dedup window.
- Add no-DataJoint tests for both preview and applied merges.

### 7. Medium: AnalyzerCuration NWBs are result-only

`AnalyzerCuration.make_fetch` resolves metric params, waveform recipe,
auto-merge preset/kwargs, ordered rule rows, and job kwargs from DataJoint
(`src/spyglass/spikesorting/v2/metric_curation.py:851-888`). The NWB writer
stores only `quality_metrics`, `merge_suggestions`, and `proposed_labels`
scratch tables (`src/spyglass/spikesorting/v2/_metric_curation_nwb.py:166-196`).

Impact: the labels, metrics, and merge suggestions can travel, but the reason
they exist cannot. A reader cannot tell which metric set, metric kwargs,
template metric columns, whitened metric recipe, auto-merge preset, or rule
thresholds produced the result.

Fix direction:

- Add an analyzer-curation provenance table or JSON payload with expanded
  metric params, template columns, waveform recipe, auto-merge params, ordered
  rules, job kwargs, and software versions.
- Extend `test_metric_curation_nwb.py` beyond value round-trips to assert that
  rule/metric provenance is present in the file.

### 8. Medium: UnitMatch NWB pairs are not self-describing enough for exchange

The UnitMatch pairs table stores side A/side B `(sorting_id, curation_id,
unit_id)` plus match probability, drift, and FDR
(`src/spyglass/spikesorting/v2/_unitmatch_nwb.py:29-92`). The context that makes
those identifiers meaningful lives in DataJoint: session group membership,
member indices, source NWB names, recording dates, pinned curation choices,
matchable unit ids, matcher params, and `curation_set_hash`
(`src/spyglass/spikesorting/v2/unit_matching.py:209-225`,
`src/spyglass/spikesorting/v2/unit_matching.py:488-631`).

`TrackedUnit` is also database-only: it derives biological-unit groups from
`UnitMatch.Pair` and inserts master/member rows, but there is no NWB writer for
the tracked-unit partition (`src/spyglass/spikesorting/v2/unit_matching.py:833-960`).

Impact: a standalone UnitMatch NWB can list pairs, but not the session map,
matching inputs, or final tracked-unit identities expected by downstream
cross-session analyses.

Fix direction:

- Add `unit_match_sessions`, `unit_match_parameters`, and `unit_matchable_units`
  tables to the artifact.
- Include `member_a`/`member_b` or equivalent member indices in the pairs table.
- Add a `tracked_units` table or explicit tracked-unit export artifact.

### 9. Medium: cache hash integrity is stored outside the artifact and not verified on normal reads

The recording writer hashes the persisted file after write
(`src/spyglass/spikesorting/v2/_recording_nwb.py:211-215`) and the row stores
`cache_hash` (`src/spyglass/spikesorting/v2/recording.py:1011-1021`). Normal
readback rebuilds only when the file is missing; if the file exists, it is read
without comparing the stored hash (`src/spyglass/spikesorting/v2/recording.py:1502-1524`).
Concat readback similarly trusts the existing file
(`src/spyglass/spikesorting/v2/session_group.py:963-992`).

Impact: inside the originating database, the hash is available but not enforced
by default. Outside the database, the hash is absent unless a separate manifest
travels with the NWB.

Fix direction:

- Add an explicit `verify_hash` path for readback and export checks.
- Store a content fingerprint in the NWB provenance module or write an export
  manifest that travels with the artifact bundle.
- Test by mutating an analysis NWB and asserting verification catches it for
  both `Recording` and `ConcatenatedRecording`.

### 10. Low-Medium: processed recording artifacts are written under acquisition

`write_nwb_artifact` writes a preprocessed, analysis-stage `ElectricalSeries`
via `nwbfile.add_acquisition(series)`
(`src/spyglass/spikesorting/v2/_recording_nwb.py:194-209`). This may be a
reasonable Spyglass convention for analysis files, and it keeps SpikeInterface
readback simple, but it is semantically surprising for external NWB consumers:
the series is not raw acquisition data.

Impact: external tools may treat this as original acquired ephys unless they
inspect `filtering` and Spyglass-specific provenance.

Fix direction:

- Keep the current path if needed for compatibility, but make the processed
  status explicit in structured provenance.
- Consider a processing-module placement for future artifacts if external NWB
  consumers become a priority.

## Already Solid

- Recording artifacts preserve explicit timestamps and avoid affine assumptions
  for irregular/disjoint time vectors.
- `conversion` and `offset` are written so scaled volts can round-trip.
- Electrode table regions are mapped by row index rather than assuming
  `electrode_id == row_index`.
- Sorting Units NWBs store absolute `spike_times`, `obs_intervals`, and
  `spike_sample_index`, which is a strong internal readback contract.
- AnalyzerCuration and UnitMatch use DB-free NWB writer/readers for their result
  tables, which is a good base to extend with provenance tables.

## Recommended First Step

Define one portable artifact contract and test it before adding many bespoke
fields:

1. Create a `spyglass_spikesorting_v2` processing module or equivalent scratch
   namespace.
2. Add versioned tables/payloads for `recording_provenance`,
   `concat_members`, `sorting_provenance`, `unit_metadata`,
   `curation_lineage`, `analyzer_curation_provenance`,
   `unit_match_sessions`, `unit_match_parameters`, and `tracked_units`.
3. For every generated artifact class, add at least one PyNWB-only test that
   opens the NWB and answers: "What raw/scientific input produced this, what
   parameters were used, and how do I map the output back to sessions/electrodes
   without DataJoint?"

This can be incremental. The highest-value first slice is recording provenance
plus concat member boundaries, because every later artifact depends on those
being interpretable.

## Other Dimensions To Review Next

- API ergonomics and notebook workflow: whether users can discover and run the
  v2 path without relying on internal table knowledge.
- Operational observability: logging, progress reporting, failure diagnostics,
  and cleanup/audit commands for large populates.
- Performance and storage lifecycle: analyzer folder growth, cache eviction,
  zarr/NWB duplication, and multi-day concat costs.
- Migration and coexistence: how v1 outputs, imported sortings, and v2 outputs
  appear through shared merge/reporting APIs.
- Security and environment portability: container pinning, local sorter binary
  provenance, path assumptions, and cluster/HPC execution behavior.
