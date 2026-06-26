# Spike Sorting V2 curated NWB export/provenance review

## Scope

This review covers the post-curation artifact boundary in Spike Sorting V2:

- `CurationV2.insert_curation()` and the curated Units NWB writer
- curation labels, merge groups, sample-frame sidecars, and observation windows
- `SpikeSortingOutput.fetch_nwb()` / downstream consumer behavior
- paper-export tests and docs
- concat-backed curated sortings where they touch NWB/export provenance

It does not review analyzer recomputation or general sorting correctness except
where those concerns cross the curated NWB/export boundary.

## Method

Two independent passes reviewed this dimension:

1. A source-focused pass over `_units_nwb.py`, `curation.py`,
   `metric_curation.py`, concat source resolution, and merge dispatch.
2. A tests/docs/API pass over curation tests, export tests, downstream consumer
   tests, notebooks, and the V2 feature docs.

The findings below are the synthesized issues that survived local verification
and the two independent passes.

## Executive summary

The curated curation path has a lot of good machinery: absolute spike times are
preserved, sample-frame sidecars are written for fast and gap-correct readback,
merge provenance is queryable in DataJoint, labels are validated on direct and
helper inserts, staged files are cleaned up on transaction failure, and the
single-recording paper-export path is covered.

The biggest gap is that pre-curation sorting NWBs preserve per-unit
`obs_intervals`, but curated Units NWBs do not. That drops the observation
window precisely at the artifact users are most likely to export or hand to
downstream tools.

The second class of issues is self-description. The database knows the curation
lineage, merge contributors, source sorting, curation source, metric-curation
proposals, and concat membership. The curated NWB itself carries only the final
Units rows plus optional labels and sample indices. That can be acceptable if the
supported contract is "NWB plus Spyglass DB/export context," but the boundary
should be explicit, and a few high-risk fields such as `obs_intervals` should
travel with the Units table itself.

## What looks solid

- Pre-curation `Sorting` NWBs write absolute `spike_times`,
  `spike_sample_index`, per-unit `obs_intervals`, and an `"uncurated"` label
  placeholder. See `src/spyglass/spikesorting/v2/_units_nwb.py:520` and
  `tests/spikesorting/v2/single_session/test_sorting.py:1290`.
- Curated NWB readback prefers stored `spike_sample_index` and only falls back
  to absolute-time remapping for legacy/manual files. See
  `src/spyglass/spikesorting/v2/curation.py:1356` and
  `src/spyglass/spikesorting/v2/_units_nwb.py:73`.
- `CurationV2.MergeGroup` stores queryable kept-unit/contributor provenance and
  validates contributors against source `Sorting.Unit`. See
  `src/spyglass/spikesorting/v2/curation.py:154`.
- `CurationV2.insert_curation()` stages the heavy NWB write outside the DB
  transaction, then registers the analysis file, master row, part rows, and merge
  table row atomically. See `src/spyglass/spikesorting/v2/curation.py:401` and
  `src/spyglass/spikesorting/v2/curation.py:730`.
- `n_spikes` is reconciled after the staged curated NWB write so the DB row
  matches the stored, post-dedup spike train. See
  `src/spyglass/spikesorting/v2/curation.py:687`.
- `SpikeSortingOutput.fetch_nwb()` dispatch for V2 is directly tested. See
  `tests/spikesorting/v2/single_session/test_merge_dispatch.py:177`.
- Single-recording V2 paper export is covered for ordinary and zero-unit
  curations. See `tests/spikesorting/v2/test_export_safety.py:147`.
- Analyzer-curation metrics, merge suggestions, and proposed labels have
  DB-free NWB round-trip tests. See
  `tests/spikesorting/v2/test_metric_curation_nwb.py:46`.

## Findings

### 1. High: curated Units NWBs drop `obs_intervals`

The pre-curation sorting writer explicitly computes an observation interval
array and writes it onto every unit:

- fallback/interval resolution:
  `src/spyglass/spikesorting/v2/_units_nwb.py:592`
- per-unit write:
  `src/spyglass/spikesorting/v2/_units_nwb.py:648`

This is tested for both the full-recording fallback and artifact-backed
IntervalList path:

- `tests/spikesorting/v2/test_recording_provenance.py:95`
- `tests/spikesorting/v2/single_session/test_sorting.py:1290`

The curated writer reads the source Units spike times/sample indices and writes
the curated rows, but the per-unit kwargs contain only `spike_times`, `id`, and
optional `spike_sample_index`:

- source read:
  `src/spyglass/spikesorting/v2/_units_nwb.py:715`
- curated unit write:
  `src/spyglass/spikesorting/v2/_units_nwb.py:891`

No curated/export test currently asserts that the curated Units NWB carries
`obs_intervals`.

Impact:

- Artifact-masked sortings lose their artifact-removed observation window in
  the curated NWB.
- Disjoint recordings lose their chunked observed windows in the exported
  curation artifact.
- NWB-only consumers that compute firing rates, presence ratios, or duration
  denominators from the Units table can silently assume the wrong observation
  window.

Recommended fix:

- Read `obs_intervals` from the source sorting Units NWB alongside
  `spike_times` and `spike_sample_index`.
- Write `obs_intervals` to every curated unit.
- For merged units, validate that all contributing units share the same
  observation interval set, then write that set. If contributor intervals can
  legitimately differ in the future, define the merge rule explicitly.
- Add a curated NWB readback test mirroring the existing pre-curation
  `obs_intervals` tests, including an artifact-backed sort.

### 2. High: all-unlabeled curated NWBs can bypass include-label filtering

The curated writer intentionally omits the `curation_label` column when all
label lists are empty:

- `src/spyglass/spikesorting/v2/_units_nwb.py:904`

The behavior is explicitly tested:

- `tests/spikesorting/v2/single_session/test_curation_insert.py:1138`

That may be necessary to avoid PyNWB empty ragged-column dtype inference
failures, but a downstream consumer treats a missing label column as "do not
filter":

- `src/spyglass/spikesorting/analysis/v1/group.py:218`
- `src/spyglass/spikesorting/analysis/v1/group.py:237`

For an include-only selection, that is wrong. A group asking for
`include_labels=["accept"]` over an unlabeled V2 curation should return no units,
not all units. The current behavior would return all units whenever the
curated NWB omits `curation_label`.

This is not just theoretical API shape drift: sorted-spikes/decoding notebooks
advertise label filtering generally, while some examples remain V1-centric:

- `notebooks/py_scripts/11_Spike_Sorting_Analysis.py:29`
- `notebooks/py_scripts/42_Decoding_SortedSpikes.py:72`

Impact:

- Include-label downstream selections can silently include units that do not
  carry the requested label.
- The behavior depends on a storage optimization in the curated NWB rather than
  the semantic label state.

Recommended fix:

- At the consumer boundary, synthesize empty label lists when `curation_label`
  is missing and include/exclude filters are requested.
- Add a V2 `SortedSpikesGroup` or decoding-boundary test for an all-unlabeled
  curation with `include_labels=["accept"]`.
- Longer term, consider writing an explicit metadata flag or typed empty label
  column so "curated but all unlabeled" is distinguishable from legacy/no-label
  files.

### 3. Medium: curated NWBs are thin without the Spyglass DB context

The database has rich curation provenance:

- `CurationV2` master stores `sorting_id`, `curation_id`,
  `parent_curation_id`, `merges_applied`, `curation_source`, and description:
  `src/spyglass/spikesorting/v2/curation.py:65`
- `CurationV2.MergeGroup` stores kept-unit/contributor provenance:
  `src/spyglass/spikesorting/v2/curation.py:154`
- merge rows are inserted into `SpikeSortingOutput.CurationV2`:
  `src/spyglass/spikesorting/v2/curation.py:781`

The curated NWB writer itself resolves the anchor parent and source sorting
Units file, then writes final Units rows:

- anchor/source resolution:
  `src/spyglass/spikesorting/v2/_units_nwb.py:707`
- Units write body:
  `src/spyglass/spikesorting/v2/_units_nwb.py:774`

The curated NWB does not carry a compact provenance table with the source
`sorting_id`, `curation_id`, parent curation, source analysis file/object id,
merge contributors, `merges_applied`, `curation_source`, or merge-dedup
parameters.

The paper-export docs currently say the final export contains the curated Units
NWB, upstream preprocessed-recording cache, and intermediate sort NWB:

- `docs/src/Features/SpikeSortingV2.md:1015`

That may make the full Spyglass export reproducible, but the individual curated
NWB remains difficult to interpret or recompute outside Spyglass.

Impact:

- A detached curated NWB cannot explain fresh merged unit ids or map them back
  to contributor unit ids.
- A DB-free consumer cannot distinguish manual, analyzer-derived, figpack, root,
  child, preview, and applied-merge curations from the file alone.
- The contract is ambiguous: full Spyglass exports may be reproducible, while a
  single curated Units NWB is not self-describing.

Recommended fix:

- Add a small processing/scratch provenance table to curated NWBs with source
  sorting identity, curation identity, parent curation, source analysis
  file/object id, merge contributors, `merges_applied`, `curation_source`, and
  merge dedup delta.
- If the team does not want curated NWBs to be standalone, document that
  explicitly and point DB-free consumers to the complete paper-export bundle
  rather than the curated Units file alone.

### 4. Medium: concat-backed curated export/provenance is not covered

Concat-backed sortings use a synthetic concat timeline and anchor their analysis
NWB to the first member:

- `src/spyglass/spikesorting/v2/sorting.py:1391`

Member boundaries live in the concat table:

- `src/spyglass/spikesorting/v2/session_group.py:580`

Split-back depends on those DB rows:

- `src/spyglass/spikesorting/v2/session_group.py:994`

The curated NWB can carry `spike_sample_index`, but for concat-backed sortings
those indices are in concat-frame space. Without concat provenance in the file,
an external consumer cannot map them back to member-local sessions.

Tests cover concat split-back and anchor-member behavior:

- `tests/spikesorting/v2/test_session_group_concat.py:710`

Export safety tests, however, cover only single-recording pipeline summaries and
look up the recording cache by `recording_id`:

- `tests/spikesorting/v2/test_export_safety.py:83`

Impact:

- Concat-backed `CurationV2.fetch_nwb()` or paper export could regress without
  a test checking that the curated Units NWB and concat cache are captured.
- A detached concat curated NWB can look like it is anchored to the first member
  while its sample indices are actually concat-frame indices.

Recommended fix:

- Add a concat-backed export test that asserts final `Export.File` includes the
  curated Units NWB and the `ConcatenatedRecording` cache.
- Add concat provenance to the curated NWB provenance table proposed above:
  `concat_recording_id`, session group identity, ordered members, member start
  and end samples, and source member recording ids.
- Document concat export semantics separately from ordinary session-local V2
  curations.

### 5. Medium-low: materialized analyzer curations do not carry metric linkage in the curated NWB

Analyzer-curation has its own NWB artifact for metrics, merge suggestions, and
proposed labels:

- `src/spyglass/spikesorting/v2/_metric_curation_nwb.py:81`
- `tests/spikesorting/v2/test_metric_curation_nwb.py:46`

When an analyzer curation is materialized, `materialize_curation()` commits only
labels and merge groups into `CurationV2`:

- `src/spyglass/spikesorting/v2/metric_curation.py:1389`

The resulting curated Units NWB does not contain the metric table/object id,
metric parameter row names, auto-curation rule names, or a link back to the
`AnalyzerCuration` artifact that produced the labels.

Impact:

- A DB-free consumer can see final labels but not why metric-derived labels were
  assigned.
- The relationship between an analyzer-curation proposal and a materialized
  `CurationV2` row is recoverable through Spyglass tables, but not from the
  curated Units file.

Recommended fix:

- If analyzer-derived curated NWBs are expected to stand alone, write the
  analyzer-curation object id/selection identity into the curated provenance
  table.
- Otherwise, document that metric provenance is stored in the analyzer-curation
  artifact and DataJoint rows, not in the materialized curated Units NWB.

### 6. Low: export completeness docs are stronger than the test assertion

The V2 docs say paper export captures the curated Units NWB, upstream
preprocessed-recording cache, and intermediate sort NWB:

- `docs/src/Features/SpikeSortingV2.md:1015`

The export-safety helper returns only the curated Units NWB and the recording
cache:

- `tests/spikesorting/v2/test_export_safety.py:83`

The main test asserts those two files are present:

- `tests/spikesorting/v2/test_export_safety.py:147`

It does not assert the intermediate sort NWB, even though the docs promise it.
The cascade may already include it through `CurationV2 -> Sorting`, but the
behavior is not pinned.

Recommended fix:

- Extend the export-safety test to fetch the upstream `Sorting.analysis_file_name`
  and assert it is present in final `Export.File`.

### 7. Low: downstream notebooks still steer users toward V1-only joins

The V2 notebook correctly carries `final_merge_id` through
`SpikeSortingOutput`:

- `notebooks/py_scripts/10_Spike_SortingV2.py:490`

Some downstream sorted-spikes/decoding examples still restrict through
`SpikeSortingOutput.CurationV1`:

- `notebooks/py_scripts/42_Decoding_SortedSpikes.py:72`
- `notebooks/py_scripts/11_Spike_Sorting_Analysis.py:81`

Impact:

- Users copying downstream examples may miss the V2 merge-id path or accidentally
  write V1-only restrictions while working with V2 curations.

Recommended fix:

- Add a short V2 branch/example using `final_merge_id` directly or
  `SpikeSortingOutput().get_restricted_merge_ids(..., sources=["v2"])`.

## Follow-on review leads

- Analyzer lifecycle/storage contracts: stale analyzer folders, missing analyzer
  recovery, extension persistence, and recompute behavior.
- Delete/recompute behavior across `Sorting`, `CurationV2`, analyzer artifacts,
  external NWB files, and merge-table rows.
- Downstream consumer boundary tests for V2-specific label, concat, and
  preview-merge semantics.

