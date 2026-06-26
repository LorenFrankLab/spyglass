# Spike Sorting V2 analyzer lifecycle/storage contracts review

## Scope

This review covers the regeneratable `SortingAnalyzer` cache and analyzer-based
auto-curation path in Spike Sorting V2:

- analyzer cache pathing, rebuild, extension mutation, and deletion
- display-vs-metric analyzer recipes
- `AnalyzerCurationSelection` / `AnalyzerCuration`
- analyzer recompute/version tables
- concat-backed analyzer coverage where it crosses this lifecycle
- tests and user-facing docs for analyzer curation and storage management

It does not re-review the raw sorting algorithm, recording content fingerprints,
or curated NWB export except where they interact with analyzer state.

## Method

Two independent passes reviewed this dimension:

1. A source-focused pass over `_sorting_analyzer.py`, `_analyzer_cache.py`,
   `sorting.py`, `metric_curation.py`, and `recompute.py`.
2. A tests/docs/API pass over analyzer lifecycle tests, analyzer recompute tests,
   concat analyzer tests, the V2 feature docs, notebooks, and storage-management
   docs.

The findings below are the synthesized issues that survived local verification
and the two independent passes.

## Executive summary

The analyzer cache has a much better shape than the earlier per-row path model:
paths are centrally derived from `(sorting_id, waveform_params_name)`, display
and metric recipes do not collide, missing folders can be rebuilt, zero-unit
sorts get a typed error, and analyzer folders are treated as regeneratable cache
rather than canonical database state.

The main correctness problem is semantic: `AnalyzerCurationSelection` is keyed to
a selected `CurationV2` row, and the docs/notebook describe iterative
auto-curation over post-merge templates, but `AnalyzerCuration.make_compute()`
loads analyzers for the base `sorting_id` and computes metrics/labels/merge
suggestions over the raw sorting namespace. A second analyzer pass after a merge
therefore appears curation-scoped but is not.

The main storage problem is that analyzer mutation is not consistently locked or
published atomically. `AnalyzerCuration` uses a per-sort file lock, but
`get_analyzer()` rebuild, invalid-folder cleanup, `Sorting.add_extensions()`, and
recompute deletion can all mutate or remove the same canonical `*.zarr` folders
outside that lock.

## What looks solid

- Analyzer cache location is centralized and derived from configuration at call
  time. See `src/spyglass/spikesorting/v2/_analyzer_cache.py:35`.
- Analyzer folders are keyed by both `sorting_id` and recipe name, so the display
  and whitened metric analyzers do not overwrite each other. See
  `src/spyglass/spikesorting/v2/_analyzer_cache.py:57`.
- `AnalyzerWaveformParameters` rejects unsafe recipe names on normal insert. See
  `src/spyglass/spikesorting/v2/sorting.py:561`.
- `Sorting.get_analyzer(rebuild=False)` gives recompute/version code a
  no-self-heal path, so missing folders can be inventoried as missing instead of
  silently rebuilt. See `src/spyglass/spikesorting/v2/sorting.py:1869`.
- Zero-unit sorts raise `ZeroUnitAnalyzerError` instead of returning an invalid
  analyzer object. See `src/spyglass/spikesorting/v2/_sorting_analyzer.py:190`.
- Analyzer rebuild reconstructs the canonical stored sorting and source
  recording, rather than re-running a sorter. See
  `src/spyglass/spikesorting/v2/_sorting_analyzer.py:362`.
- AnalyzerCuration serializes its own analyzer mutation region with a per-sort
  lock. See `src/spyglass/spikesorting/v2/metric_curation.py:930`.
- Analyzer versioning inventories both the display recipe and PC-requesting
  metric recipes. See `src/spyglass/spikesorting/v2/recompute.py:747`.

## Findings

### 1. High: AnalyzerCuration is curation-scoped in identity/docs but sorting-scoped in compute

`AnalyzerCurationSelection` explicitly references `CurationV2`:

- schema FK: `src/spyglass/spikesorting/v2/metric_curation.py:657`
- deterministic identity includes `curation_id`:
  `src/spyglass/spikesorting/v2/metric_curation.py:718`

The docstring and warning also describe chained auto-curation as computing over
post-merge templates:

- `src/spyglass/spikesorting/v2/metric_curation.py:689`
- `src/spyglass/spikesorting/v2/metric_curation.py:744`

But `AnalyzerCuration.make_compute()` reduces the selection to
`{"sorting_id": sorting_id}` and loads analyzers from `Sorting().get_analyzer()`
without applying the selected curation's unit set, labels, or proposed/applied
merges:

- sorting-only key: `src/spyglass/spikesorting/v2/metric_curation.py:916`
- display analyzer load: `src/spyglass/spikesorting/v2/metric_curation.py:933`
- metric analyzer load: `src/spyglass/spikesorting/v2/metric_curation.py:952`
- metrics and merge suggestions computed from those analyzers:
  `src/spyglass/spikesorting/v2/metric_curation.py:957`

Materialization then writes a child curation under the selected parent, but using
labels and merge groups derived from the raw sorting analyzer:

- `src/spyglass/spikesorting/v2/metric_curation.py:1399`
- `src/spyglass/spikesorting/v2/metric_curation.py:1407`

The user-facing docs and notebook repeat the post-merge promise:

- `docs/src/Features/SpikeSortingV2.md:557`
- `notebooks/py_scripts/10_Spike_SortingV2.py:344`
- `notebooks/py_scripts/10_Spike_SortingV2.py:368`

Impact:

- The documented auto -> merge -> auto loop can score the original units while
  presenting the result as a pass over the merged curation.
- Metrics, labels, and merge suggestions can include units rejected or merged by
  the parent curation.
- A child analyzer curation can look scientifically downstream of a parent
  curation while its numeric inputs came from a different unit universe.

Recommended fix:

- Decide the intended contract explicitly.
- If analyzer curation is meant to be curation-scoped, build the display/metric
  analyzers over the selected `CurationV2` sorting object, or add a
  curation-specific analyzer layer whose unit ids and templates match the parent
  curation.
- If analyzer curation is meant to be sorting-scoped only, remove or de-emphasize
  the `curation_id` semantic in the API/docs and reject chained analyzer
  curations that imply post-merge recomputation.
- Add a test with a parent curation whose unit set differs from raw
  `Sorting.Unit` through a merge or exclusion, then assert the analyzer-curation
  metrics/index are computed on the intended unit ids.

### 2. Medium-high: analyzer folder mutation is not covered by one shared lock or atomic publish

The per-sort analyzer lock exists and is documented as protecting shared zarr
mutation:

- `src/spyglass/spikesorting/v2/_analyzer_cache.py:92`

`AnalyzerCuration` holds it around analyzer load, extension computation, and
quality-metric writes:

- `src/spyglass/spikesorting/v2/metric_curation.py:930`

Other canonical-folder mutations do not use that lock:

- invalid-folder cleanup in `load_or_rebuild_analyzer()`:
  `src/spyglass/spikesorting/v2/_sorting_analyzer.py:211`
- rebuild into the canonical folder:
  `src/spyglass/spikesorting/v2/_sorting_analyzer.py:374`
- public extension mutation:
  `src/spyglass/spikesorting/v2/sorting.py:1977`
- recompute deletion:
  `src/spyglass/spikesorting/v2/recompute.py:1317`

The rebuild path writes directly to the final `*.zarr` folder, and cleanup removes
that same folder on failure:

- build target:
  `src/spyglass/spikesorting/v2/_sorting_analyzer.py:379`
- failure cleanup:
  `src/spyglass/spikesorting/v2/_sorting_analyzer.py:391`

The lock itself is also documented as local-machine only, not a cross-host
coordination primitive for shared analyzer storage:

- `src/spyglass/spikesorting/v2/_analyzer_cache.py:110`

Impact:

- A rebuild, extension add, recompute delete, or analyzer curation can race with
  another process reading or writing the same zarr store.
- A cleanup path can remove a folder another path just rebuilt or is currently
  using.
- A failed direct write can leave a partial canonical analyzer folder that later
  has to be detected and rebuilt.

Recommended fix:

- Route every mutation/removal of a sort's analyzer folders through the same
  per-sort lock: rebuild, invalid-folder cleanup, `add_extensions()`, recompute
  deletion, orphan sweep deletion if it targets live-looking folders, and
  AnalyzerCuration.
- Rebuild to a temporary sibling folder, validate loadability, then atomically
  publish into the canonical path.
- Treat cross-host shared analyzer storage as a separate policy decision:
  document single-host support clearly or use a DB-backed/portable lock for
  shared cluster writes.
- Add tests that prove rebuild/delete/add-extension paths acquire the shared lock
  or at least cannot interleave with AnalyzerCuration.

### 3. Medium-high: explicit analyzer recipe names hit the filesystem before normal-row validation

Recipe names are path-sensitive because they are embedded in folder names:

- `src/spyglass/spikesorting/v2/_analyzer_cache.py:57`

Normal `AnalyzerWaveformParameters` insertion rejects names outside
`^[A-Za-z0-9_]+$`:

- `src/spyglass/spikesorting/v2/sorting.py:561`
- `src/spyglass/spikesorting/v2/sorting.py:628`

However, `Sorting.get_analyzer()` accepts an explicit `waveform_params_name` and
forwards it directly:

- `src/spyglass/spikesorting/v2/sorting.py:1869`
- `src/spyglass/spikesorting/v2/sorting.py:1932`

`load_or_rebuild_analyzer()` resolves the folder path before fetching waveform
params:

- path resolution: `src/spyglass/spikesorting/v2/_sorting_analyzer.py:210`
- load/rmtree on existing folder:
  `src/spyglass/spikesorting/v2/_sorting_analyzer.py:211`
- params fetched later during rebuild:
  `src/spyglass/spikesorting/v2/_sorting_analyzer.py:369`

This is not arbitrary filesystem access outside the configured analyzer root in
ordinary usage, but it does mean a path-like explicit recipe can influence the
direct-child namespace before the public loader proves the recipe row is valid.

Impact:

- A bad explicit recipe can address unexpected nested/cache-root-relative paths
  before failing as an unknown parameter row.
- Error behavior depends on filesystem state rather than the database contract.
- Future callers may treat `analyzer_path()` as validation because its docstring
  says names are already insert-guarded.

Recommended fix:

- Centralize recipe-name validation in the analyzer path/resolution boundary, not
  only in the table insert path.
- In `load_or_rebuild_analyzer()`, fetch and validate the
  `AnalyzerWaveformParameters` row before any filesystem load or delete for an
  explicit recipe.
- Add tests that `waveform_params_name="../bad"` or `"bad/name"` raises before
  any path is touched, both for `rebuild=True` and `rebuild=False`.

### 4. Medium: analyzer recompute authorizes deletion from a narrow deterministic subset but deletes whole folders

Analyzer recompute hashes only the deterministic sort-time extensions:

- `src/spyglass/spikesorting/v2/_recompute.py:22`
- `src/spyglass/spikesorting/v2/_recompute.py:29`

The fresh audit rebuild also computes only that subset:

- `src/spyglass/spikesorting/v2/recompute.py:1123`

This is reasonable for proving the base analyzer can be regenerated. But the
stored analyzer folder can also contain curation and visualization extensions
such as amplitudes, correlograms, template similarity, unit locations, template
metrics, principal components, and quality metrics. `ensure_extensions()` skips
already-present extensions without checking their parameter provenance:

- `src/spyglass/spikesorting/v2/_sorting_analyzer.py:70`

Recompute deletion then removes the entire analyzer folder once the base subset
matches:

- `src/spyglass/spikesorting/v2/recompute.py:1317`

Impact:

- A folder can be authorized for deletion even though non-hashed extensions are
  missing, stale, corrupt, or parameter-incompatible.
- Conversely, curation-relevant extension drift is not detected by the recompute
  mismatch path.
- After deletion, expensive extensions are silently treated as cache and must be
  rebuilt on demand, but that policy is not explicit in the analyzer-curation
  provenance.

Recommended fix:

- Make the policy explicit: either non-base analyzer extensions are ephemeral
  cache and may be discarded after base reproducibility is proven, or they are
  scientific artifacts whose params/content need tracking.
- If they are scientific artifacts, store an extension manifest with extension
  names, params, SpikeInterface version, and deterministic content hashes where
  possible.
- At minimum, record the extension inventory and parameter manifest on
  `AnalyzerCuration` so curation outputs can be tied to the analyzer state that
  produced them.

### 5. Medium: AnalyzerCuration output is not bound to analyzer/source content provenance

`AnalyzerCuration` persists object ids for its metrics, merge groups, and labels,
but the selection identity is limited to object ids and parameter names:

- selection identity:
  `src/spyglass/spikesorting/v2/metric_curation.py:718`
- computed output fields:
  `src/spyglass/spikesorting/v2/metric_curation.py:890`

No source analyzer hash, source sorting content hash, source recording content
hash, SpikeInterface dependency manifest, or extension manifest is stored with
the analyzer-curation result. The compute path loads mutable cache folders at
runtime:

- `src/spyglass/spikesorting/v2/metric_curation.py:933`
- `src/spyglass/spikesorting/v2/metric_curation.py:952`

Impact:

- A stored analyzer-curation result cannot explain exactly which analyzer
  extension state produced its metrics.
- If an analyzer folder is rebuilt under changed dependency behavior, the old
  analyzer-curation row remains indistinguishable from a row produced under the
  rebuilt cache.
- Recompute can verify base analyzer reproducibility, but it does not make
  analyzer-curation outputs stale or current.

Recommended fix:

- Store a compact provenance payload on `AnalyzerCuration`: source analyzer
  recipe(s), source analyzer hash/manifest, source sorting/recording content
  identifiers, relevant dependency versions, and metric extension params.
- Add a stale-detection helper that compares stored provenance against current
  analyzer/version rows.
- Add a test that rebuilds or alters an analyzer extension manifest and verifies
  analyzer-curation provenance changes or the row is flagged stale.

### 6. Medium: real analyzer recompute deletion lacks a full round-trip test

Analyzer recompute tests cover versioning, matching, missing-folder inventory,
and dry-run non-deletion behavior. The real delete branch is the path that
removes a folder and marks matched recompute rows deleted:

- `src/spyglass/spikesorting/v2/recompute.py:1317`
- `src/spyglass/spikesorting/v2/recompute.py:1327`

Recording recompute has stronger round-trip coverage for non-dry-run delete and
rebuild. The analogous analyzer path should prove:

- matched analyzer recompute rows authorize deletion
- `delete_files(dry_run=False, days_since_creation=0)` removes the folder
- matching recompute rows are marked `deleted=1`
- a later `Sorting().get_analyzer()` rebuilds the cache
- stale/missing version state after rebuild is coherent

Impact:

- The most destructive analyzer recompute branch can regress while dry-run tests
  keep passing.
- The interaction between deletion bookkeeping and self-heal rebuild is not
  exercised.

Recommended fix:

- Add a slow analyzer recompute integration test mirroring the recording
  deletion round trip.
- Include both display and PC-requesting metric analyzer recipes if runtime
  allows; otherwise make the metric recipe a separate slow test.

### 7. Medium: concat-backed AnalyzerCuration is documented but not covered end to end

Concat-backed sorting and analyzer smoke tests exist, and `AnalyzerCuration`
has broad single-recording coverage. The concat-specific analyzer-curation test
surface is still thin: at least one test patches source preprocessing resolution
because true concat setup was not wired for that path.

Relevant code paths are concat-aware at the source selection layer, but
AnalyzerCuration also has to agree on:

- display recipe resolution for concat-backed sortings
- metric recipe resolution for concat-backed sortings
- analyzer reconstruction through `ConcatenatedRecording`
- curated/analyzer NWB anchoring to the first `SessionGroup.Member`
- metric/label/merge unit ids on concat timelines

Impact:

- A concat-backed sort can pass analyzer smoke tests while failing in the
  analyzer-curation path users will actually run.
- Anchor-file and recipe-resolution bugs can hide until chronic workflows use
  metric curation.

Recommended fix:

- Add an end-to-end concat analyzer-curation test:
  `SessionGroup` -> `ConcatenatedRecording` -> `Sorting` -> root `CurationV2` ->
  `AnalyzerCurationSelection` -> `AnalyzerCuration.populate()` ->
  `materialize_curation()`.
- Assert metric rows, labels, merge groups, metric waveform params, and analysis
  NWB anchor identity.

### 8. Medium-low: storage-management docs are hard to discover and recording-heavy

`docs/src/Features/SpikeSortingV2StorageManagement.md` exists, but it is not
prominently discoverable from the feature nav/index, and the concrete workflow is
recording-centric. The analyzer tables are mentioned as mirrors of the recording
storage-management tables, but there is no comparable analyzer-focused example.

Impact:

- Operators can learn the recording reclamation workflow and miss the analyzer
  cache/recompute lifecycle.
- Users may not understand that analyzer folders are regeneratable cache, while
  `AnalyzerCuration` outputs are persisted NWB/database artifacts.

Recommended fix:

- Link the storage-management page from the Spike Sorting V2 feature docs and
  feature index/nav.
- Add an analyzer-specific example covering:
  `SortingAnalyzerVersions.populate()`,
  `SortingAnalyzerRecompute.populate()`,
  `delete_files(dry_run=True)`, `delete_files(dry_run=False)`, and
  `Sorting().get_analyzer()` rebuild.
- State the policy difference between analyzer cache extensions and persisted
  analyzer-curation results.

## Suggested fix order

1. Fix or explicitly reject curation-scoped AnalyzerCuration semantics. This is
   the only issue in this review that can silently produce wrong scientific
   metrics under a documented workflow.
2. Bring analyzer cache mutation under one lock and publish rebuilds atomically,
   mirroring the recording cache hardening.
3. Validate explicit analyzer recipe names before any filesystem access.
4. Decide and document whether non-base analyzer extensions are ephemeral cache
   or tracked scientific artifacts; add the minimal provenance needed for that
   decision.
5. Add the destructive-delete round-trip and concat analyzer-curation tests.
6. Fold the storage-management docs into the Phase 5 docs/nav pass.
