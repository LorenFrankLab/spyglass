# Spike Sorting V2 Test Coverage Review

Date: 2026-06-25

Scope: test architecture and missing regression coverage for Spike Sorting V2,
with emphasis on the scientific/reproducibility findings in
`scientific-reproducibility-review.md`.

Method: local test/source inspection plus two independent explorer agents. No
test suite was run for this review.

## Executive Summary

The v2 test suite is unusually broad. It has DB-free service tests, integration
tests over MEArec/minirec fixtures, parity tests, recompute tests, concat
round-trips, UnitMatch graph tests, analyzer lifecycle tests, and validation
tests for the parameter schemas. That structure is strong and should be
preserved.

The main weakness is fixture shape and provenance assertions. Several tests
exercise the right named path but use fixtures where the bad implementation and
the correct implementation are indistinguishable: one raw `ElectricalSeries`,
electrode ids equal to row indices, zero `starting_time`, and implicit runtime
defaults. The suite also tests internal routing of stochastic inputs better
than it tests whether those inputs become persisted scientific provenance.

## Findings

### 1. High: raw `ElectricalSeries` selection is not tested against `Raw.raw_object_id`

Current coverage:

- `test_raw_eseries_timestamp_mode_detects_rate_vs_explicit` builds one
  synthetic acquisition `ElectricalSeries` and checks rate-vs-explicit timestamp
  detection (`tests/spikesorting/v2/test_recording_services.py:156-190`).
- `test_recording_get_recording_honors_electrical_series_path` checks that
  `Recording.get_recording` reads the stored analysis-file
  `electrical_series_path` (`tests/spikesorting/v2/single_session/test_recording.py:1405-1439`).

Gap: neither test exercises raw-source selection during `Recording.make`.
The vulnerable code still selects the first acquisition `ElectricalSeries`
(`src/spyglass/spikesorting/v2/_recording_nwb.py:32-53`) and passes that path
to the raw reader (`src/spyglass/spikesorting/v2/recording.py:1697-1704`),
while common `Raw` stores an exact `raw_object_id`
(`src/spyglass/common/common_ephys.py:286-292`).

Recommended tests:

- Minimal NWB with two acquisition `ElectricalSeries` objects, distinguishable
  traces/rates, and a `Raw.raw_object_id` pointing to the second.
- Populate or directly drive `_compute_recording_artifact`; assert the raw read
  path and persisted traces come from the `raw_object_id` series.
- Unit-level helper test for "object id -> NWB path + timestamp mode" resolution
  once the resolver exists.

### 2. High: `channel_name` coverage uses only the identity electrode-id case

Current coverage:

- `test_channel_name_resolution_path_real_nwb` exercises presence/absence of an
  NWB `channel_name` column (`tests/spikesorting/v2/test_recording_provenance.py:189-253`).
- But it uses `spyglass_ids = [0, 1, 2, 3]`
  (`tests/spikesorting/v2/test_recording_provenance.py:238`), so
  `channel_names[int(electrode_id)]` and a correct id-to-row mapping produce the
  same answer.
- Writer-side electrode-table row mapping is well covered by
  `test_electrode_table_region.py`, including non-contiguous ids
  (`tests/spikesorting/v2/test_electrode_table_region.py:40-75`).

Gap: the input slicing path remains unprotected. The source code indexes
`channel_names[int(c)]` (`src/spyglass/spikesorting/v2/_recording_geometry.py:69-70`).

Recommended tests:

- Build an NWB electrodes table with ids like `[10, 11, 12, 13]` or shuffled ids
  plus `channel_name`.
- Assert `Recording._spikeinterface_channel_ids(..., [12, 10])` returns the
  channel names for electrode ids 12 and 10, not row positions 12 and 10.
- Add a small integration test where raw channels have distinguishable constant
  traces; after `restrict_recording`, verify the renamed electrode ids carry the
  correct original traces.

### 3. High: recording construction fingerprint behavior is not encoded as a test contract

Current coverage:

- Selection identity tests intentionally lock `recording_id` to
  `(nwb_file_name, sort_group_id, interval_list_name,
  preprocessing_params_name, team_name)`.
- `Recording.make_fetch` dereferences mutable state at populate time:
  sort-group electrodes/reference, interval arrays, preprocessing params,
  bad-channel flags, probe metadata, and raw coverage
  (`src/spyglass/spikesorting/v2/recording.py:1050-1150`).
- One existing test normalizes this pattern: the global-median reference test
  mutates `SortGroupV2.reference_mode`, deletes `Recording`, and repopulates
  the same recording primary key.

Gap: the future expected behavior is not tested. After the content-fingerprint
plan lands, upstream scientific-state drift under the same row names should not
silently rebuild the same `recording_id`.

Recommended tests:

- Populate a `Recording`, mutate only `IntervalList.valid_times` under the same
  interval name, and force rebuild/recompute. Expect a construction-fingerprint
  mismatch or a new identity.
- Do the same for sort-group membership/reference state and bad-channel
  interpolation inputs.
- Update existing tests that mutate upstream rows and repopulate the same
  `rec_pk` so they assert the chosen migration contract explicitly.

### 4. High: global seed/runtime provenance is not tested as a scientific contract

Current coverage:

- Tests prove per-row `random_seed` reaches analyzer `random_spikes` and is
  stripped from SpikeInterface compute kwargs
  (`tests/spikesorting/v2/single_session/test_sorting.py:766-899`,
  `tests/spikesorting/v2/single_session/test_sorting.py:1004-1069`).
- `_resolved_job_kwargs` merge behavior is explicitly tested for SI globals,
  DJ config, and row blobs
  (`tests/spikesorting/v2/test_preprocessing_schema_and_db_guards.py:76-100`).

Gap: the tests currently enshrine the merge behavior, but do not distinguish
runtime-only job settings from science-affecting inputs. A global/DJ-config
`random_seed` can influence whitening, analyzer subsampling, UnitMatch bundle
extraction, and clusterless MAD estimation without being row provenance.

Recommended tests:

- Set `dj.config["custom"]["spikesorting_v2_job_kwargs"] =
  {"random_seed": 7}` with row `job_kwargs=None`.
- Expect either a validation error for global science-affecting keys, or a
  persisted effective-seed/runtime-provenance row on Sorting/analyzer/UnitMatch
  outputs.
- Add local-runtime provenance tests that monkeypatch SpikeInterface/sorter
  version strings and expect recorded provenance or explicit mismatch.

### 5. High: UnitMatch effective inputs are not identity/provenance tested

Current coverage:

- `test_matcher_params.py` validates the exposed UnitMatch schema fields.
- `test_unitmatch_backend.py` checks registration, geometry guards, session-drop
  guards, probability orientation, and an optional end-to-end UnitMatchPy gate.
- `test_unitmatch.py` thoroughly covers graph determinism, pair canonicalization,
  selection idempotency, and table wiring.

Gap: the tests do not assert that effective UnitMatch inputs are represented in
`MatcherParameters` or `unitmatch_id`. Bundle parameters are function defaults
(`ms_before`, `ms_after`, `max_spikes_per_unit`, `seed`) in
`extract_unitmatch_bundle` (`src/spyglass/spikesorting/v2/_unitmatch_backend.py:107-157`),
and UnitMatchPy defaults are resolved dynamically
(`src/spyglass/spikesorting/v2/_unitmatch_backend.py:232-238`).

Recommended tests:

- Once matcher params include bundle params, assert `_extract_and_match` passes
  them into `extract_unitmatch_bundle`.
- Snapshot or hash the effective UnitMatchPy defaults and assert changing them
  changes matcher-parameter identity or raises a provenance mismatch.
- Add a regression that changing only bundle `seed` or waveform window cannot
  silently reuse the same `unitmatch_id`.

### 6. Medium-High: `TrackedUnit` matchable-universe drift is not tested at the DB boundary

Current coverage:

- Pure graph tests cover partitioning, tie-breaks, singleton behavior, and
  budget errors (`tests/spikesorting/v2/test_unitmatch.py:52-294`).
- `UnitMatch.make_compute` purity is tested: it uses `make_fetch`'s transient
  matchable set instead of re-reading current labels
  (`tests/spikesorting/v2/test_unitmatch.py:1269-1290`).

Gap: after `UnitMatch` is populated, `TrackedUnit.make` re-derives the node
universe from current curation labels. There is no test for label mutation
between `UnitMatch.populate` and `TrackedUnit.populate`.

Recommended tests:

- Populate `UnitMatch`, then insert a valid `CurationV2.UnitLabel` such as
  `noise` for a previously matchable but unmatched unit before
  `TrackedUnit.populate`.
- Assert the chosen contract: frozen universe from UnitMatch output, or an
  explicit early mismatch requiring UnitMatch repopulation.

### 7. Medium: concat tests cover boundary counts but not frozen member provenance or portable NWB provenance

Current coverage:

- Pure concat tests cover motion alias resolution, split frame mapping, member
  key disambiguation, channel/geometry compatibility, and motion job-kwarg
  validation (`tests/spikesorting/v2/test_concat_recording.py`).
- Integration tests cover missing-recording preconditions, materialized
  `MemberBoundary` rows, motion sample-count drift, end-to-end concat sorting,
  split back-mapping, anchor-member brain regions, and memory measurement
  (`tests/spikesorting/v2/test_session_group_concat.py`).

Gap: current tests do not assert that the ordered member identities and source
timeline transform are frozen in the concat artifact/provenance. `split_sorting_by_session`
fetches current `SessionGroup.Member` rows while applying stored boundaries.
The analysis NWB portability question is also untested: outside DataJoint, the
concat file does not appear to be asserted to contain ordered source provenance.

Recommended tests:

- Materialize concat, mutate or forge `SessionGroup.Member` rows under the same
  group, and assert split/populate fails on construction-fingerprint mismatch or
  uses frozen member provenance.
- Assert concat analysis NWB contains ordered member provenance: source NWB,
  recording id, source intervals/timestamps, concat frame start/end, and time
  transform.
- Add concat analogs of recording cache-hash readback tests.

### 8. Medium: nonzero rate-based `starting_time` is only partly covered

Current coverage:

- `test_lazy_regular_path_matches_eager_on_nonzero_start_recording` validates
  lazy timestamp math on a synthetic `NumpyRecording` with nonzero `t_start`
  (`tests/spikesorting/v2/test_recording_services.py:193-249`).

Gap: there is no full raw-NWB path test for a rate-based
`ElectricalSeries(starting_time != 0)`. The raw timestamp-mode helper test uses
`starting_time=0.0` (`tests/spikesorting/v2/test_recording_services.py:166-190`).

Recommended tests:

- Minimal rate-based NWB with `starting_time != 0`.
- Populate or drive `_compute_recording_artifact`.
- Assert persisted timestamps begin at the expected wall-clock time for both
  single and disjoint intervals.

### 9. Medium: analyzer `noise_levels` and extension-provenance tests are incomplete

Current coverage:

- Tests assert analyzer base extensions include `noise_levels`, and that
  explicit params are passed for `random_spikes` and `waveforms`.
- `_recompute.py` intentionally excludes `noise_levels` from analyzer recompute
  hashing because it is stochastic
  (`src/spyglass/spikesorting/v2/_recompute.py:1-24`).

Gap: the suite does not assert how the stochastic `noise_levels` extension is
represented in provenance. If it remains unseeded and excluded from recompute,
there should be an explicit recorded marker saying that it is intentionally
unverified, or the extension should get deterministic params.

Recommended tests:

- Analyzer provenance manifest includes effective extension params for
  `random_spikes`, `waveforms`, `templates`, and either deterministic
  `noise_levels` params or an explicit unverifiable/stochastic marker.
- Recompute tests assert that excluded extensions are reported, not silently
  forgotten.

### 10. Medium: curation parent semantics and analyzer-curation source identity need contract tests

Current coverage:

- `test_curation_v2_parent_validation_and_nonincreasing_ids` checks only parent
  existence and id allocation
  (`tests/spikesorting/v2/single_session/test_curation_insert.py:131-172`).
- Analyzer-curation materialization tests check parent id, `curation_source`,
  labels, and idempotent reuse
  (`tests/spikesorting/v2/test_analyzer_curation.py:308-350`,
  `tests/spikesorting/v2/test_analyzer_curation.py:990-1050`).

Gap: tests do not encode whether child curations are deltas from parent state or
complete snapshots. They also would not catch loss of exact
`AnalyzerCurationSelection` provenance when materialized curations collapse to
the same labels/merge groups.

Recommended tests:

- Parent with labels and merges, then child with `parent_curation_id` and no
  repeated state. Assert the chosen contract: inherited state or explicit
  complete-snapshot behavior.
- Materialize two distinct `AnalyzerCurationSelection` rows that produce
  identical labels/merges. Assert the resulting curation retains the producing
  `analyzer_curation_id` through a source part/FK, or that dedup semantics
  preserve both sources.

## What The Suite Does Well

- Clear separation between DB-free service tests and integration tests.
- Strong validation coverage for parameter schemas and duplicate-content guards.
- Good concat pure-logic coverage: split boundaries, member keys, geometry, and
  motion alias rejection.
- Good UnitMatch graph coverage: orientation, duplicate rejection, singleton
  tracked units, strict partitioning, and tie-break determinism.
- Good analyzer routing coverage for per-row `random_seed`, display vs metric
  waveform recipes, and extension mutation locks.
- Recompute has good plumbing tests for matched, unmatched, mismatch rows, stale
  env gates, and missing analyzer folders.

## Recommended Test Priority

1. Add the two recording-source regression tests: raw `Raw.raw_object_id`
   selection and non-contiguous `electrode_id -> channel_name` mapping.
2. Add construction-fingerprint contract tests for recording and concat before
   or alongside the fingerprint implementation.
3. Add tests that reject or persist global science-affecting job kwargs.
4. Add UnitMatch effective-parameter and matchable-universe drift tests.
5. Add analyzer extension-provenance tests for `noise_levels`.
6. Add curation parent/source-provenance contract tests.

