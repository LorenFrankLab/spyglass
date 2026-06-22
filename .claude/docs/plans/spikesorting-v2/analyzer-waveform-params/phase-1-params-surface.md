# Phase 1 — Tracked waveform-parameter surface + lab defaults + cache key

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared-contracts](shared-contracts.md)

Restore DB-tracked waveform parameters with **region-specific** (display,
unwhitened) recipes — hippocampus 0.5/0.5, cortex 1.0/2.0 — resolved from the
sort's preprocessing recipe. After this phase the analyzer's window/subsample are
no longer hardcoded: they come from a named, queryable row, the cache path
records which row produced each analyzer, and the sort stores which display row
it resolved. The whitened metric recipe is Phase 2.

**Inputs to read first:**

- [_sorting_analyzer.py:340-411](../../../../../src/spyglass/spikesorting/v2/_sorting_analyzer.py#L340-L411) — `build_analyzer`; the hardcoded `extension_params` to replace (`:375`, `:397`).
- [_analyzer_cache.py:35-72](../../../../../src/spyglass/spikesorting/v2/_analyzer_cache.py#L35-L72) — `analyzer_path` / `analyzer_cache_root`; the sorting_id-only key.
- [sorting.py:794-831](../../../../../src/spyglass/spikesorting/v2/sorting.py#L794-L831) — `SortingSelection.resolve_source`; reuse this helper for source detection instead of re-reading source part tables.
- [sorting.py:874-970](../../../../../src/spyglass/spikesorting/v2/sorting.py#L874-L970) — `Sorting` definition (`:876`, where the secondary `display_waveform_params_name` field is added) + `Sorting.Unit` (peak_amplitude_uv at `:904`) and the sort-time analyzer build.
- [v1/metric_curation.py:67-110](../../../../../src/spyglass/spikesorting/v1/metric_curation.py#L67-L110) — v1 `WaveformParameters`, the table being mirrored.
- [_params/sorter.py:40-90](../../../../../src/spyglass/spikesorting/v2/_params/sorter.py#L40-L90) — the pydantic-`@schema`-params pattern to copy for the new schema + lookup.
- [_recipe_catalog.py:38-47,302-305](../../../../../src/spyglass/spikesorting/v2/_recipe_catalog.py#L302-L305) — the `_REGION_PREPROC` region→preproc map; add a parallel `preprocessing-recipe → (display, metric) waveform-params` map here.
- [_selection_identity.py:44,158](../../../../../src/spyglass/spikesorting/v2/_selection_identity.py#L44) — `preprocessing_params_name` is part of `recording_identity_payload` — the determinism anchor for the region-resolved window.
- [session_group.py:209-215](../../../../../src/spyglass/spikesorting/v2/session_group.py#L209-L215) / parent Phase 3 design — `ConcatenatedRecordingSelection` FKs `PreprocessingParameters`; its `preprocessing_params_name` resolver input comes from that FK's PK, not a literal field in the class definition.

**Contracts referenced:**

- [`AnalyzerWaveformParameters` table](shared-contracts.md#analyzerwaveformparameters-table) — defined here; do not bury in `QualityMetricParameters`.
- [Region resolution](shared-contracts.md#region-resolution) — the sort source's preprocessing recipe names the display row; multi-region falls back to cortex.
- [Analyzer cache identity](shared-contracts.md#analyzer-cache-identity) — the new `analyzer_path(sorting_id, waveform_params_name)` signature.

## Tasks

- **Add `AnalyzerWaveformParamsSchema`** (pydantic, `extra="forbid"`) in a new
  `_params/analyzer_waveform.py`, with fields per
  [the contract](shared-contracts.md#analyzerwaveformparameters-table)
  (`ms_before=1.0`, `ms_after=2.0`, `max_spikes_per_unit=20000`, `whiten=False`,
  `purpose="display"`). Mirror `_params/sorter.py`'s validate-and-dump style.
- **Add the `AnalyzerWaveformParameters` `dj.Lookup`** in `sorting.py` (analyzer
  is a `Sorting`-level concern; `metric_curation.py` already imports from
  `sorting.py`, so no import cycle). Definition + `insert_default` inserting the
  four region rows from the contract table (hippocampus/cortex × display/metric).
  Follow the duplicate-content guard pattern used by other v2 default inserts
  (`memory/spikesorting-v2-param-content-guard`).
- **Extend the cache key + multi-recipe cleanup.** Change
  `analyzer_path(sorting_id)` → `analyzer_path(sorting_id, waveform_params_name)`
  returning `root/f"{sorting_id}__{waveform_params_name}.zarr"`
  ([_analyzer_cache.py:57-72](../../../../../src/spyglass/spikesorting/v2/_analyzer_cache.py#L57-L72)). A sort now has MULTIPLE folders (display now;
  metric in Phase 2), so the single-folder cleanup helpers must fan out:
    - `remove_analyzer_cache(sorting_id)` (`:75`) removes **all**
      `{sorting_id}__*.zarr` folders (glob) — deleting the sort orphans every
      recipe. `Sorting.delete` (`sorting.py:1517`) already snapshots sorting_ids
      and calls it, so it inherits the fan-out unchanged.
    - `find_orphaned_analyzer_folders` enumerates `{sorting_id}__{name}.zarr` and
      treats a folder orphaned when no `Sorting` row has that `sorting_id`, OR
      `name` is not that sort's `Sorting.display_waveform_params_name` (Phase 2
      extends this to also accept any `AnalyzerCurationSelection` metric
      `waveform_params_name`).
- **Thread a resolved params *blob* (not a name) into the build, honoring the
  tri-part contract.** The make-time path must NOT do DB I/O inside
  `make_compute` (`_sorting_analyzer.py:5-9` documents the
  `make_fetch`/`make_compute`/`make_insert` split forbidding it). So:
  `Sorting.make_fetch` fetches + validates the `AnalyzerWaveformParameters` row
  and puts the resolved dict + the `waveform_params_name` in the fetched
  `NamedTuple`. `build_analyzer` / `_build_analyzer` take TWO things: the cache
  **folder** (computed by the caller via
  `analyzer_path(sorting_id, waveform_params_name)`, which carries the name /
  identity) and the **resolved params dict** (which feeds `extension_params`,
  **replacing** the hardcoded `max_spikes_per_unit=500`
  ([:375](../../../../../src/spyglass/spikesorting/v2/_sorting_analyzer.py#L375)) and `{ms_before:1.0, ms_after:2.0}` ([:397](../../../../../src/spyglass/spikesorting/v2/_sorting_analyzer.py#L397)); the `random_spikes` seed pin stays). `build_analyzer`
  itself NEVER resolves a bare name → params (no DB I/O). Only the lazy
  cache-miss rebuild path (`load_or_rebuild_analyzer`, outside `make_compute`)
  fetches the row to rebuild the dict.
- **The display recipe is resolved from the sort's REGION, not a free per-sort
  field.** The sort-time analyzer + `peak_amplitude_uv` use the region's display
  row — `franklab_hippocampus_actual_waveforms` (0.5/0.5) or
  `franklab_cortex_actual_waveforms` (1.0/2.0) — resolved in `make_fetch` from
  the sort source's preprocessing recipe (see
  [Region resolution](shared-contracts.md#region-resolution)); multi-region or
  unknown sorts fall back to the wider cortex row. The resolved name is **not** a
  user-tunable per-sort attribute and **not** added to `SortingSelection`
  identity (`_selection_plan.py:177`, which is `(recording_id, sorter,
  sorter_params_name, artifact_detection_id)`). Rationale: `peak_amplitude_uv` is
  a content-addressed sort output; because the window is determined by region (a
  property of the data/recipe), it stays deterministic for a `sorting_id` — but a
  *user-tunable* per-sort window not in `sorting_id` would violate
  content-addressing, so that is intentionally not offered. The metric recipe is
  carried on `AnalyzerCurationSelection` (Phase 2).
- **Wire source preprocessing recipe → display row, deterministically**
  ([Region resolution](shared-contracts.md#region-resolution)). Add a
  `preprocessing-recipe → (display, metric) waveform-params` map to
  `_recipe_catalog` (parallel to `_REGION_PREPROC`, `:302-305`), keyed by
  `HIPPOCAMPUS_PREPROC` / `CORTEX_PREPROC`; any other recipe → the cortex pair.
  Add a secondary `display_waveform_params_name` field to the `Sorting`
  definition (`sorting.py:876`, after `time_of_sort`). `Sorting.make_fetch`
  calls `SortingSelection.resolve_source(key)` first, then resolves the display
  name from that source's `preprocessing_params_name` through the map:
    - `RecordingSource` → `RecordingSelection.preprocessing_params_name`.
    - `ConcatenatedRecordingSource` → the
      `ConcatenatedRecordingSelection -> PreprocessingParameters` FK's
      `preprocessing_params_name` PK. Do not query `RecordingSelection` with a
      concat-only row.
  Do not re-detect source type by hand; `resolve_source` is the single
  source-part integrity check and stays in sync with the parent concat work.
  `make_compute` builds the display analyzer with it (for `peak_amplitude_uv`);
  `make_insert` stores it. Every cache-miss rebuild
  (`load_or_rebuild_analyzer`) reads the **stored**
  `Sorting.display_waveform_params_name`, never re-resolves. Do NOT add fields to
  `_PipelinePreset` — it is `extra="forbid"`; the map lives in `_recipe_catalog`.
  Determinism follows because the source preprocessing recipe is part of the
  source identity. For today's single-recording sorts, that source identity is
  `recording_id`, which already flows into `sorting_id`. For future concat-backed
  sorts, parent Phase 3 must extend `sorting_identity_payload` so
  `concat_recording_id` flows into `sorting_id` before concat populate is enabled
  (see the contract's proof).
- **`get_analyzer` defaults to the sort's display recipe.** `get_analyzer` /
  `load_or_rebuild_analyzer` resolve `waveform_params_name=None` to the stored
  `Sorting.display_waveform_params_name` (the deterministic default, see the
  [cache-identity contract](shared-contracts.md#analyzer-cache-identity)). So
  `Sorting.add_extensions` (`sorting.py:1487`, which calls `get_analyzer(key)`
  with no name) stays **display-only** with no signature change, and its existing
  tests (`test_sorting_add_extensions.py`) keep passing — they now exercise the
  sort's display analyzer. A caller wanting the whitened metric analyzer passes
  the name explicitly (Phase 2).
- **Update the burst/plot helpers' call sites** to pass the resolved
  `waveform_params_name` when they load the analyzer
  ([_metric_curation_plots.py:243+](../../../../../src/spyglass/spikesorting/v2/_metric_curation_plots.py#L243), [metric_curation.py get_peak_amps/_analyzer_for](../../../../../src/spyglass/spikesorting/v2/metric_curation.py)). In this phase they all
  resolve the sort's region display recipe; Phase 2 splits display vs metric.
- **Docs:** note in `v1-v2-divergences.md` that the analyzer waveform window /
  subsample are now tracked, region-specific `AnalyzerWaveformParameters` rows
  (hippocampus 0.5/0.5, cortex 1.0/2.0; both 20000) — and update the existing
  "Analyzer waveform window widened 0.5/0.5→1.0/2.0; subsample reduced 10×
  (5000→500)" bullet: cortex keeps 1.0/2.0 (still wider than v1's 0.5/0.5) while
  hippocampus returns to 0.5/0.5; the subsample is now 20000 (4× v1's 5000, no
  longer reduced); and all are tracked rows rather than hardcoded.
- **Smoke-test the 20000-spike build** (long-running-computation idiom). Phase 1
  ships 20000 (was 500) as the default subsample, so before treating it as the
  default: build one analyzer on a real-data slice (the larger cortex 1.0/2.0
  window is the heavier case) and record build wall-clock + peak memory in the
  PR; extrapolate to a full session. The whitened build's ADDITIONAL cost is
  measured in Phase 2.

## Deliberately not in this phase

- **Whitening / the metric analyzer / two-analyzer routing** → Phase 2. This
  phase ships the region DISPLAY recipes (unwhitened, hippo + cortex); metric
  curation reads the sort's region display recipe too, temporarily, until the
  whitened metric recipe lands in Phase 2.
- **Pipeline preset default + MS4 recommendation, auto-curation rule changes** →
  Phase 3.
- **Recompute changes** beyond passing the new cache key — the whitened-vs-
  unwhitened recompute coverage lands in Phase 2 with the second recipe.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_analyzer_waveform_params_default_rows` | the four region rows exist — hippocampus display/metric at 0.5/0.5, cortex display/metric at 1.0/2.0, all 20000; `whiten` False(display)/True(metric) (`db_unit`) |
| `test_analyzer_waveform_params_schema_rejects_extra` | unknown key raises at validation (`extra="forbid"`) |
| `test_analyzer_path_includes_params_name` | `analyzer_path(sid, "franklab_hippocampus_actual_waveforms") != analyzer_path(sid, "franklab_cortex_actual_waveforms")`; both under the configured root |
| `test_remove_analyzer_cache_removes_all_recipes` | with two `{sid}__*.zarr` folders on disk, `remove_analyzer_cache(sid)` deletes BOTH (glob), and a different sort's folder is untouched (`db_unit`, tmp cache root) |
| `test_orphan_detection_retains_display_recipe` | a `{sid}__{display_name}.zarr` matching `Sorting.display_waveform_params_name` is NOT orphaned; a `{sid}__stray.zarr` IS (Phase 2 extends this to retain `AnalyzerCurationSelection`-referenced metric folders) |
| `test_build_analyzer_uses_param_row_window` | `build_analyzer(..., waveform_params={ms_before:0.5, ms_after:0.5, max_spikes:20000, ...})` (the hippocampus row's RESOLVED dict, no name lookup) → `waveforms` ext `ms_before==0.5, ms_after==0.5`; the cortex dict → `1.0/2.0`; both cap 20000 (DB-free; reuse the `synthetic_analyzer` fixture pattern) |
| `test_display_analyzer_resolved_from_region` | a hippocampus-preset sort builds its display analyzer with the hippocampus row, a cortex-preset sort with the cortex row, and a multi-region sort falls back to cortex; `peak_amplitude_uv` is deterministic for a `sorting_id` (`slow`, `integration`) |
| `test_concat_display_analyzer_resolved_from_concat_preprocessing` | a concat-backed sort resolves `display_waveform_params_name` from the `ConcatenatedRecordingSelection -> PreprocessingParameters` FK'd `preprocessing_params_name` (hippocampus/cortex/custom fallback), not from member `RecordingSelection` rows or a single-recording-only query. Parent-Phase-3-dependent for end-to-end populate; if this subplan lands first, test the resolver with direct selection/source-part rows or a monkeypatched `resolve_source`, not `ConcatenatedRecording.populate()` (`slow`, `integration` for the end-to-end version) |
| `test_sorting_records_display_waveform_params` | `Sorting.make_insert` stores the resolved display row, and cache-miss rebuilds read the stored `Sorting.display_waveform_params_name` instead of re-resolving from the source (`slow`, `integration`) |
| `test_sorting_selection_identity_unchanged` | `sorting_id` does NOT depend on any waveform-params input (idempotency preserved) |
| `test_make_compute_does_no_param_db_io` | `Sorting.make_compute` / `build_analyzer` take a resolved params dict; the DB fetch happens in `make_fetch` (assert no query inside the compute path) |
| `test_peak_amplitudes_aligned_to_waveform_subset` (existing) | still passes — `nbefore` adapts to the selected row's window (symmetric for hippocampus, asymmetric for cortex) |

## Fixtures

- Reuse the DB-free synthetic in-memory analyzer pattern from
  `tests/spikesorting/v2/test_metric_curation_plots.py` (`synthetic_analyzer`,
  `capped_analyzer`) for the build/window tests.
- The `slow` Sorting test reuses the existing `populated_sorting_with_curation`
  integration fixture.
- Concat resolver tests are parent-Phase-3-dependent for a true populated
  `ConcatenatedRecording`. Before that parent phase lands, cover the branch with
  DB-unit fixture rows or a monkeypatched `SortingSelection.resolve_source`; do
  not require `ConcatenatedRecording.populate()`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff.
Confirm:
- Every task is implemented as specified.
- "Deliberately not in this phase" is honored — no whitening, no metric/display
  split, no preset/rule changes leak in.
- Validation tests pass; slow/integration tests are marked.
- Tests exercise behavior (the actual window/cap on a built analyzer), not the
  mock; shared setup is in fixtures.
- No docstring/test/module name references this plan or its phase numbers.
- The hardcoded `_sorting_analyzer.py:375,397` values are removed (no orphan
  constants), and the stale divergence-doc bullet is corrected.
