# Spike Sorting V2 Downstream Consumer Contracts Review

Date: 2026-06-25

Scope: contracts between v2 sorting/curation/unit-matching outputs and
downstream consumers, including `SpikeSortingOutput`, sorted-spikes groups,
decoding feature extraction, unit annotations, export capture, UnitMatch /
TrackedUnit public accessors, and user-facing API documentation. This is a
different lens from scientific reproducibility, DataJoint/concurrency,
dependency/runtime safety, destructive/admin operations, and general docs
integrity.

Method: local code/docs inspection plus two independent explorer-agent reviews.
This review is read-only except for this document. I did not run tests.

## Executive Summary

V2 has strong foundations for downstream compatibility. `CurationV2` is
registered into the existing `SpikeSortingOutput` merge table, v2 import failures
are lazy so v0/v1 environments can still import the merge table, decoding entry
points now reject preview merge curations, zero-unit curations have explicit
handling, and v2 unit rows carry electrode metadata needed for brain-region
queries.

The main remaining risk is that the strongest invariants are enforced at a few
consumer-specific edges rather than at the common output contract. A preview
curation is registered into `SpikeSortingOutput` exactly like a final curation,
but several generic accessors still read the raw NWB units and bypass the v2
preview warning. Cross-session UnitMatch also accepts preview curations and
matches the unmerged unit set. A few public accessors return unstable or lossy
schemas in edge cases, which is the sort of thing that breaks notebooks and
analysis code only after a cohort changes shape.

## What Looks Solid

- `SpikeSortingOutput` imports v2 lazily and preserves v0/v1 importability when
  v2 dependencies are unavailable (`src/spyglass/spikesorting/spikesorting_merge.py:24-95`).
- `SpikeSortingOutput.CurationV2` is a first-class merge part, and curation
  insertion registers v2 rows atomically with the analysis NWB and part rows
  (`src/spyglass/spikesorting/spikesorting_merge.py:127-134`,
  `src/spyglass/spikesorting/v2/curation.py:815-842`).
- V2 restriction resolution is centralized in `CurationV2.resolve_restriction`,
  including recording-vs-concat topology and lenient/strict source behavior
  (`src/spyglass/spikesorting/v2/curation.py:1416-1668`).
- Decoding-facing group creation calls
  `SpikeSortingOutput.assert_decoding_merge_ids_ok`, which rejects preview v2
  curations and multiple curation outputs for the same sorting in a decode
  group (`src/spyglass/spikesorting/spikesorting_merge.py:346-399`,
  `src/spyglass/spikesorting/analysis/v1/group.py:76-112`,
  `src/spyglass/decoding/v1/waveform_features.py:148-280`).
- `CurationV2.get_sorting()` and `Sorting.get_sorting()` both support zero-unit
  rows instead of raising on empty NWB Units tables
  (`src/spyglass/spikesorting/v2/curation.py:1335-1379`,
  `src/spyglass/spikesorting/v2/sorting.py:1773-1848`).
- Per-unit electrode and peak-amplitude metadata are stored on `Sorting.Unit`
  and carried through `CurationV2.Unit`, giving downstream code a real
  brain-region join target (`src/spyglass/spikesorting/v2/sorting.py:1153-1169`,
  `src/spyglass/spikesorting/v2/curation.py:90-106`).

## Findings

### 1. High: preview v2 curations can still be consumed as final outputs through generic accessors

`CurationV2.insert_curation()` registers every curation in `SpikeSortingOutput`,
including curations created with `apply_merge=False`
(`src/spyglass/spikesorting/v2/curation.py:833-842`). `CurationV2.get_sorting()`
does warn when proposed merges are present but unapplied
(`src/spyglass/spikesorting/v2/curation.py:1316-1333`).

Generic merge-table accessors bypass that warning. `SpikeSortingOutput.get_spike_times()`
fetches the NWB object directly and returns the raw Units spike times
(`src/spyglass/spikesorting/spikesorting_merge.py:430-445`). `get_spike_indicator()`
and `get_firing_rate()` build on that same path
(`src/spyglass/spikesorting/spikesorting_merge.py:448-570`). Those paths can
therefore analyze the unmerged preview units even though the curation has
recorded proposed merges.

The decode-specific guard is good, but it only protects consumers that call
`assert_decoding_merge_ids_ok()` before use. Direct users of `SpikeSortingOutput`
and other non-decoding consumers can still treat a preview curation as final.

Impact: direct downstream analyses can silently operate on oversplit preview
units. The difference is scientifically meaningful because a preview curation
contains an explicit statement that some units are proposed to be merged.

Recommended fix: enforce the preview/final distinction at the shared
`SpikeSortingOutput` access layer, not only at decode entry points. Either do
not register preview curations as ordinary merge outputs, or add a generic guard
to `get_spike_times()`, `get_spike_indicator()`, `get_firing_rate()`, and
fetch helpers. If preview access is useful, require an explicit
`allow_preview=True` or route callers to `CurationV2.get_merged_sorting()` /
`CurationV2.get_sorting()` deliberately. Add tests that preview curations cannot
reach generic output consumers silently.

### 2. High: UnitMatch accepts preview curations and matches unmerged units

`UnitMatchSelection.MemberCuration` pins exact `CurationV2` rows
(`src/spyglass/spikesorting/v2/unit_matching.py:197-207`), and
`UnitMatch.make_fetch()` freezes the current matchable unit IDs from those rows
(`src/spyglass/spikesorting/v2/unit_matching.py:580-625`). During compute,
`_extract_and_match()` builds each session input with `CurationV2.get_sorting()`
and then selects the frozen matchable units
(`src/spyglass/spikesorting/v2/unit_matching.py:780-799`).

That means a curation with unapplied proposed merges is matched as its unmerged
preview unit set. `CurationV2.has_unapplied_proposed_merges()` exists
(`src/spyglass/spikesorting/v2/curation.py:1381-1413`), but UnitMatch does not
use it at selection or populate time.

Impact: cross-session biological-unit identities can be built over units that
the same v2 curation already says should be merged. This is worse than a single
session downstream read because the resulting `Pair` and `TrackedUnit` rows can
become durable cross-session identities over a non-final unit universe.

Recommended fix: reject preview curations in `UnitMatchSelection.insert_selection()`
or `UnitMatch.make_fetch()` unless the caller explicitly opts into preview
matching. If preview matching is intentionally supported, persist that choice in
the selection/output identity and document that `TrackedUnit` rows are preview
identities. Add a UnitMatch test with `apply_merge=False` plus a real
multi-contributor merge group.

### 3. Medium-high: all-unlabeled v2 curations break include-label filtering semantics

The v2 NWB writer skips the `curation_label` column when every unit has an empty
label list because PyNWB cannot infer a dtype for an all-empty list-of-lists
(`src/spyglass/spikesorting/v2/_units_nwb.py:890-910`). `CurationV2.get_sorting(as_dataframe=True)`
repairs this at the table accessor level by joining `UnitLabel` and returning a
`curation_label` column with empty lists
(`src/spyglass/spikesorting/v2/curation.py:1361-1379`).

`SortedSpikesGroup.fetch_spike_data()` does not use that accessor. It reads the
NWB Units object through `SpikeSortingOutput.fetch_nwb()` and filters labels
only if either `label` or `curation_label` is physically present
(`src/spyglass/spikesorting/analysis/v1/group.py:199-236`). If the column is
missing, it skips label filtering entirely.

Impact: `include_labels=["some_label"]` against an all-unlabeled v2 curation can
return every unit instead of zero units. That is a contract inversion: the
absence of labels is interpreted as "no filter" rather than "no units match the
requested label."

Recommended fix: synthesize empty label lists in `SortedSpikesGroup.fetch_spike_data()`
when the source is a v2 Units table without a `curation_label` column, or make
the writer always emit a stable empty label column using an explicit dtype. Add
an integration test that builds an all-unlabeled v2 curation and verifies
include-label filtering returns zero matching units.

### 4. Medium: `UnitMatch.get_pairs()` loses its schema for zero-pair runs

The NWB writer defines canonical pair columns even for empty match tables
(`src/spyglass/spikesorting/v2/_unitmatch_nwb.py:29-92`). The reader returns a
list of row dicts (`src/spyglass/spikesorting/v2/_unitmatch_nwb.py:110-134`),
and `UnitMatch.get_pairs()` wraps that with a bare `pd.DataFrame(...)`
(`src/spyglass/spikesorting/v2/unit_matching.py:821-829`).

For a valid zero-pair run, `read_pairs()` returns `[]`, so `get_pairs()` returns
an empty DataFrame with no columns. Downstream code that selects
`session_a_sorting_id`, `unit_a_id`, `match_probability`, or similar columns
will fail only for single-session, thresholded, or otherwise no-match cohorts.

Impact: the public UnitMatch accessor has a different schema depending on
cardinality, even though the NWB artifact itself has a stable schema.

Recommended fix: expose a canonical `PAIR_COLUMNS` tuple from `_unitmatch_nwb.py`
or define it locally, and construct `pd.DataFrame(rows, columns=PAIR_COLUMNS)`.
Add a zero-pair schema assertion to the UnitMatch tests.

### 5. Medium: `TrackedUnit.get_unit_brain_regions()` drops curation and region disambiguators

The standard v2 helper returns `unit_id`, `electrode_id`, `region_name`,
`subregion_name`, `subsubregion_name`, and `region_resolution`, preserving the
full schema even for empty results (`src/spyglass/spikesorting/v2/utils.py:436-482`).
`TrackedUnit.Member` stores the actual member identity as
`sorting_id`, `curation_id`, and `unit_id`
(`src/spyglass/spikesorting/v2/unit_matching.py:856-862`).

`TrackedUnit.get_unit_brain_regions()` narrows each member result to only
`sorting_id`, `unit_id`, and `region_name`
(`src/spyglass/spikesorting/v2/unit_matching.py:961-1001`). It drops
`tracked_unit_id`, `curation_id`, `electrode_id`, subregion fields, and
`region_resolution`.

Impact: cross-session consumers lose the fields needed to round-trip back to
the pinned curation row or distinguish brain-region resolution. `sorting_id +
unit_id` is not the same contract as `sorting_id + curation_id + unit_id`,
especially when multiple curations of the same sort coexist.

Recommended fix: preserve existing columns for compatibility but add
`tracked_unit_id`, `curation_id`, `electrode_id`, `subregion_name`,
`subsubregion_name`, and `region_resolution`. Pin the expanded nonempty and
empty schemas in a lightweight `TrackedUnit` test.

### 6. Medium: repeated paper exports can retain stale `Export.File` rows

V2 docs rely on `Export.File` as the final exported artifact list for curated
units, upstream recording caches, and intermediate sort NWBs
(`docs/src/Features/SpikeSortingV2.md:1018-1040`). When a newer export ID
supersedes older IDs, `Export.make()` computes processed old IDs and deletes
`self.Table` twice:

`(self.Table & id_dict).delete_quick()` appears on both cleanup lines, and
`self.File` is never deleted (`src/spyglass/common/common_usage.py:540-550`).

Impact: repeated paper exports can leave stale file rows for older export IDs.
That makes the operator-facing export artifact list unreliable and can confuse
storage accounting or any downstream script that inspects files across exports
for a paper.

Recommended fix: change the second cleanup to `(self.File & id_dict).delete_quick()`
and add a two-export regression test that confirms superseded `Export.File`
rows are removed while the latest export remains intact.

### 7. Medium-low: UnitMatch NWB artifacts are thin outside DataJoint

`UnitMatch` stores pair output in an analysis NWB scratch `DynamicTable`
(`src/spyglass/spikesorting/v2/unit_matching.py:648-681`,
`src/spyglass/spikesorting/v2/_unitmatch_nwb.py:95-107`). The table columns are
the two sides' sorting/curation/unit IDs plus pair scores
(`src/spyglass/spikesorting/v2/_unitmatch_nwb.py:29-38`).

Those columns are enough when the DataJoint rows travel with the file, but the
standalone NWB object lacks `unitmatch_id`, session-group identity,
matcher-parameter identity, member indices/dates, and source NWB provenance.

Impact: exported or shared UnitMatch NWBs are harder to interpret without a live
DataJoint database. That weakens cross-session reproducibility for downstream
users who receive only files or who inspect export bundles outside Spyglass.

Recommended fix: add stable provenance columns or table metadata for
`unitmatch_id`, `team_name`, `session_group_name`, `matcher_params_name`,
member index/date mapping, and source NWB file names. Add a fetch/export
round-trip test that reads the UnitMatch NWB outside the table layer and checks
the metadata needed to identify the run.

### 8. Low: public docs misstate the v2 DataFrame shape

The main v2 docs say `Sorting.get_sorting(..., as_dataframe=True)` and
`CurationV2.get_sorting(..., as_dataframe=True)` return a DataFrame with
`unit_id` plus `spike_times` columns (`docs/src/Features/SpikeSortingV2.md:1109-1113`).
The implementation returns `unit_id` as the DataFrame index
(`src/spyglass/spikesorting/v2/_units_nwb.py:388-398`,
`src/spyglass/spikesorting/v2/curation.py:1370-1379`).

Impact: downstream notebooks following the docs may write `df["unit_id"]` and
fail, or may export the frame and accidentally drop unit IDs if the index is not
preserved.

Recommended fix: update the docs to say the DataFrame is indexed by `unit_id`
and callers should use `reset_index()` if they need a column. A docs-snippet
smoke test would keep this contract from drifting again.

