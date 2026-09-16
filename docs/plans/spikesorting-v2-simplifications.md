# Spike sorting v2 simplification candidates

Audited at `78271380`. These are maintenance opportunities, not additional
correctness findings from the preceding revision review. No pipeline source or
tests were changed during this audit.

## Recommended next patch: three bounded improvements

### 1. Keep one name for each auto-labeled output ID

Evidence: `_pipeline_run.py:955` writes `auto_curation_id` and `auto_merge_id`,
then writes those same values again as `auto_labeled_curation_id` and
`auto_labeled_merge_id`. `_pipeline_types.py:127–158` documents the mirroring.
`RunResult.auto_labeled_curation` already uses the latter names. Tests and the
tutorial still refer to both pairs.

Use `auto_labeled_curation_id` and `auto_labeled_merge_id` throughout v2. Keep
`curation_evaluation_id`, the generation UUID, and the auto-curation stage's
status: those represent different information. Keep the existing `None` values
on root-only runs and the existing concat timeline rules.

This removes a user-facing distinction that has no behavioral meaning and
eliminates duplicated assignments and equality assertions. Because this API is
pre-production, update its callers, typed summaries, notebook/script pair, and
documentation together; a deprecation adapter is unnecessary for draft v2.
Do not change production v0/v1 APIs.

Verification: existing root-only, auto-label, concat, run-reporting, and notebook
checks should pass using the canonical fields. Assert the actual source
curation/output and reuse behavior rather than equality between aliases.

### 2. Reuse curation rows within one operation

Evidence: `_curation_analyzer.py:600` resolves a curation row, then calls
`classify_curation_analyzer_namespace(key)` at line 605. The classifier fetches
the same row again at line 146. Repository search found only this internal
caller of the classifier.

Make the internal classifier consume the already-resolved row. Its job is to
classify the curation, not repeat key resolution. A private helper accepting
that row is enough; no new context class or public API is needed.

A related optional improvement is `CurationRef.children` in `curation_api.py`:
it fetches child keys and then calls `from_key()` once per child to fetch the
generation UUID. Fetch the IDs and UUIDs together and construct references
from those rows if touching this method.

Retain generation checks at public operations and validate fresh state at
mutation boundaries. Do not cache a live curation row on `CurationRef` across
calls, remove stale-reference errors, or globally replace checked keys with
unchecked keys.

Verification: existing raw/merged/preview/empty classification and stale-
generation tests must retain their behavior. Check that classification does
not refetch a row supplied by its caller. If changing child enumeration, retain
the exact generation UUIDs from the fetched rows.

### 3. Let preflight use the existing selection builders

Evidence: `_pipeline_preflight.py:1070–1113` reconstructs recording and sorting
ID assembly. It already shares the low-level payload/hash functions with the
writers, but `_selection_plan.py` contains the complete DB-free
`build_recording_selection_plan()` and `build_sorting_selection_plan()` helpers
used by insertion.

After resolving the recording input hash, call those existing builders and
take their IDs. Keep the existing artifact identity helper. Keep preflight's
read-only existence checks and its ability to report several missing inputs.
Do not call insertion functions from preflight or introduce a new saved plan.

This leaves one implementation of the field assembly and makes a future
identity-field addition less likely to change insertion without changing the
preview. The current IDs are not claimed to be wrong.

Verification: expected IDs must stay identical for existing inputs, including
artifact-disabled inputs and changed membership/reference/bad-channel inputs.
Missing inputs must still produce useful preflight diagnostics, not an early
builder exception. This cleanup must not change hashes, database keys, or
schema versions.

## Useful follow-ups, in small separate changes

### 4. Make known parameter-table metadata explicit

Evidence: `_pipeline_reporting.py:284` tries `_DEFAULT_CONTENTS`,
`_default_rows`, and `_default_payloads` using reflection. It catches arbitrary
exceptions from the builders and converts them into unknown shipped status.
The report has a fixed list of eight known parameter tables, and its callers
already know which table they are describing.

Pass the appropriate shipped-name set into the existing record-building
helper. An explicit argument removes method-name guessing and avoids hiding a
broken default builder as merely unknown metadata. If useful, extract the
common output-record fields into one small helper; keep sorter-specific and
part-table fingerprint content explicit.

Do not build a plugin registry, rewrite all default catalogs, or change
fingerprints during this refactor. In particular preserve sorter execution
parameters and ordered auto-curation Rule rows in content comparisons.

Verification: the existing eight-table coverage, dynamic-default shipped
status, backend identity, and duplicate-content reporting checks should retain
their results. No broad new test matrix is needed.

### 5. Remove an unreachable duplicate-error branch

Evidence: `utils.py:117–149` accepts a DataJoint `DuplicateError`, but also
treats a DataJoint `IntegrityError` with numeric argument `1062` as a duplicate.
The installed DataJoint `translate_query_error()` maps MySQL 1062 to
`DuplicateError`; foreign-key errors 1451/1452 map to `IntegrityError`, with the
numeric code removed. This was checked locally with real PyMySQL error objects
passed through DataJoint's translator, without a database connection.

The supposedly untranslated connector case is not handled by that branch
either: an untranslated PyMySQL exception is not a DataJoint `IntegrityError`.
The positive numeric-1062 DataJoint IntegrityError test manufactures a state
outside this query path.

Keep the shared predicate if its name helps callers, but reduce it to the
DataJoint exception-type check. Retain real duplicate-race recovery and ensure
foreign-key/unrelated errors propagate. Do not rewrite every surrounding
cleanup/retry block or remove concurrency handling.

Verification: duplicate errors remain recoverable; actual translated FK errors
remain failures. Exercise the exception translation boundary rather than
retaining a fabricated positive exception shape.

### 6. Remove narrowly identified draft/runtime compatibility leftovers

- `_analyzer_cache.py:440` retains `analyzer_curation_lock` as an alias of
  `analyzer_cache_lock`. Update the one runtime consumer in
  `metric_curation.py`, the documentation references, and alias-only test, then
  keep the canonical name. Leave lock implementation and contention tests alone.
- `_params/metric_curation.py:95–162` tries old SI quality-metric import paths
  when the modern path is absent. The v2 environment pins SI 0.104.3, and the
  template-metric code in the same module already requires the modern layout.
  Use that layout consistently while preserving lazy imports. This does not
  remove active v1 support from its separate legacy environment.

Limit this change to demonstrated leftovers. Do not infer that every use of
the words "legacy", "fallback", or "compatibility" is removable. NWB input
variation, optional dependencies, and third-party runtime workarounds have
different contracts.

### 7. Replace stale patch history with current invariants

Two concrete examples:

- `Sorting.make_insert` says its failure cleanup removes the analyzer folder
  (`sorting.py:1968`), while the implementation deliberately preserves that
  shared folder and only removes attempt-owned NWB output.
- The auto-curation comment near `_pipeline_run.py:892` describes keys as added
  only on opt-in, while the preceding code initializes the canonical
  auto-labeled keys to `None` on every run.

Correct these when touching the relevant code. Keep concise explanations of
artifact ownership, generation identity, time coordinates, and transaction
requirements. Move historical implementation narratives to git history or
design notes when they obscure the current contract. Avoid a repository-wide
comment-only cleanup before launch.

## Work to defer

`run_v2_pipeline()` combines mode validation, source preparation, sorting,
curation, and optional review in a long function. Extracting a few private
workflow helpers is a reasonable later improvement, but its stage timing,
partial failure receipts, and concat behavior make a broad refactor a larger
validation commitment. Reuse the existing stage runner when that work becomes
necessary; do not introduce a pipeline engine.

Likewise defer wholesale source-kind renaming, cache/recompute redesign,
default-catalog unification, and removal of old NWB read paths. Those changes
need a wider contract audit than the small improvements above.

For this release, keep checks that protect external artifacts, stale browser
generations, concurrent inserts, and owned-file cleanup. The aim is fewer
duplicated decisions and unreachable branches, with the existing supported
behavior preserved.
