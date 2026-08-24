# Phase 1 — Curation-scoped analyzer

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [design D1](designs.md#d1)

Closes the one genuine v1→v2 capability gap: a curator cannot currently visualize a merged unit, because the analyzer-backed plots deliberately raise once a curation leaves the raw unit namespace. This phase introduces one shared resolver so evaluation, plotting, and (later) FigPack all read the correct analyzer for *any* curation, and routes the plot/read helpers through it.

**Inputs to read first:**

- [_analyzer_cache.py:148](../../../../src/spyglass/spikesorting/v2/_analyzer_cache.py) — `analyzer_cache_lock` (reentrant per-`sorting_id` lock); also `_publish_sibling:219`, `analyzer_path:115`, `is_canonical_analyzer_folder_name:64`, `assert_path_safe_waveform_params_name:49`. The cache primitives to extend, not fork.
- [metric_curation.py:1389-1445](../../../../src/spyglass/spikesorting/v2/metric_curation.py) — `CurationEvaluation.make_compute` merged `else:` branch: builds display (+ metric) analyzers via the shared `build_analyzer` (`:1406`/`:1419`), DB-free via `_sorting_from_units_nwb` (`:1394`). `make_compute` starts at `:1191`, is `_parallel_make = True` / DB-free (`:999-1005`), routing at `:1128`. This is the code the resolver SHARES a builder with — not reroutes.
- [metric_curation.py:191-216](../../../../src/spyglass/spikesorting/v2/metric_curation.py) — `_assert_curation_in_raw_namespace`: the raise the plot helpers must stop hitting. The guard is centralized in `_display_analyzer_key` (`:2501`), shared by `_analyzer_for` and `_locked_display_analyzer` — reroute at that chokepoint, not 11 independent sites.
- [metric_curation.py:2477-2732](../../../../src/spyglass/spikesorting/v2/metric_curation.py) — the analyzer-backed read/plot helpers to reroute (enumerated in [overview integration points](overview.md#current-codebase-integration-points)); mutating ones (`plot_units_qc`, `plot_burst_pair_metrics`) run under `_locked_display_analyzer` (`:2537-2553`).
- [visualization.py:158-164](../../../../src/spyglass/spikesorting/v2/visualization.py) — the `ssviz` facade's OWN parallel `_assert_curation_in_raw_namespace` guard + `{"sorting_id"}` resolver, which must be rerouted too or merged plots stay inconsistently rejected there.
- [curation.py:2492](../../../../src/spyglass/spikesorting/v2/curation.py) — `get_merged_sorting` (merged spike-train reconstruction the build consumes); `matches_raw_namespace:2076`.
- The Phase-4a cache memory (SI zarr relative-path publish footgun) — see [design D1](designs.md#d1).

**Contracts referenced:**

- [get_curation_analyzer](shared-contracts.md#get_curation_analyzer) — the resolver signature + routing contract; **do not weaken** the single-low-level-builder invariant (make_compute stays DB-free) or the read-only-published-analyzer design.
- [Analyzer cache manifest](shared-contracts.md#analyzer-cache-manifest) — regeneratable-cache validation fields.

**Designs referenced:** [D1](designs.md#d1).

## Tasks

- **GATING SPIKE FIRST (F6).** Before building, run the SI `merge_units` parity spike ([D1](designs.md#d1)): compare path A (CurationV2 committed sorting → `build_analyzer`) vs path B (raw analyzer → `SortingAnalyzer.merge_units(merge_unit_groups=…, new_unit_ids=…, censor_ms=…, merging_mode="hard", format="zarr")` — `merge_unit_groups` is the required first arg) on disjoint intervals, same-sample duplicates, sub-0.4 ms cross-contributor duplicates, exact unit IDs/spike trains, templates, correlograms, quality metrics, reload-after-publish. Adopt SI **only** on exact equivalence; otherwise retain the custom build (strong prior: SI's `censor_ms` is sample-based, Spyglass's dedup is absolute-time/gap-correct at `curation.py:2492`). Record the decision + evidence.
- Create `src/spyglass/spikesorting/v2/_curation_analyzer.py` implementing the **internal** resolver per [D1](designs.md#d1): namespace classification (raw / merged-committed / preview / zero-unit), **per-curation** cache path keyed by **`curation_uuid`** (Phase 0) — `(sorting_id, curation_uuid, role, waveform_recipe_hash, spikeinterface_version)`, NOT `curation_id` (reusable, `curation.py:1062-1072`) — manifest write/validate, atomic publish reusing `_analyzer_cache` primitives. **Read-only by ownership (B4):** SI has no read-only analyzer type, so the resolver is internal, the published cache is never mutated by any supported path, plots go through controlled helpers, direct access returns a `save_as(format="memory")` detached copy, and an unusual extra-extension plot uses a **context-managed, temporary** derivative (cleaned up on exit — not a second persistent cache). Preview → "commit first" error; zero-unit → raise a typed `ZeroUnitAnalyzerError` (plots render an empty-state view).
- Add `_curation_analyzer.build_merged_analyzer` (the DB-connected wrapper) that reconstructs the merged sorting (may use `get_merged_sorting`, `curation.py:2492`) and calls the **shared** low-level `build_analyzer` — the same builder `make_compute` uses at `metric_curation.py:1406`/`:1419`, preserving the zarr relative-path hidden-sibling publish. **Do NOT reroute `make_compute` through the resolver and do NOT delete its temp build** — it is a DB-free `_parallel_make` worker (`:999-1005`) and must stay DB-free. The invariant is "one low-level builder" (`build_analyzer`, already shared), not "one caller." See [B1 in shared-contracts](shared-contracts.md#get_curation_analyzer).
- Route the read/plot helpers through `get_curation_analyzer(role="display")` at the centralized `_display_analyzer_key`/`_locked_display_analyzer` chokepoint (`metric_curation.py:2501`, `:2537-2553`) — not 11 independent sites: `get_waveforms`, `get_correlograms`, `plot_correlograms`, `plot_units_qc`, `get_peak_amps`, `plot_peak_over_time`, `plot_burst_pair_metrics`, `investigate_pair_xcorrel`, `investigate_pair_peaks`, `plot_si_quality_metrics`, `plot_si_template_metrics`. A helper that needs an extension outside the standard display set passes `extra_extensions=` (→ staged derivative), never mutating the published cache (F3). Remove the `_assert_curation_in_raw_namespace` raise from these helpers; keep it only where a caller truly needs the raw analyzer (document which, if any).
- **Also reroute `visualization.py`'s own guard** (`visualization.py:158-164`) through the resolver, or merged-curation plots stay inconsistently rejected in the `ssviz` facade (M2).
- Cache cleanup — **one typed cache subsystem** (H2, resolved — not a fork): `raw` and `curation` kinds behind **one reference collector** (unions `Sorting` display + PC-requesting metric recipes AND live-`CurationV2` curation folders keyed by `curation_uuid`) and **one reclaim entry point**. `find_orphaned_analyzer_folders`/`classify_orphaned_analyzer_folders` (`sorting.py:2600-2654`) consume that collector so a live curation's folder is never an orphan; the reclaim is per-curation reference-count under the per-sort lock. The manifest guards cache *load*; the merged path's NULL `source_analyzer_hashes` row-level gap is an accepted known limitation ([M3 in D1](designs.md#d1)).
- Docs: this phase changes no user-facing entry point names yet (the facade is Phase 2), so no notebook rewrite here — but add/adjust the docstrings on the rerouted helpers to state they now accept merged curations, and remove any "raises on a merged curation" wording that is no longer true.

## Deliberately not in this phase

- The `EvaluationResult.plots` accessor and any `run.analysis_curation` sugar — that is Phase 2's facade. Phase 1 only makes the underlying helpers merge-capable.
- FigPack routing through the resolver — Phase 3 (depends on this resolver existing).
- Strict metric errors — Phase 2.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_merged_unit_waveform_renders` | After merging two units and re-evaluating, `plot_units_qc`/`get_waveforms` on the merged curation returns without raising and yields the merged unit's template (the acceptance test). |
| `test_merged_unit_correlogram_renders` | `get_correlograms`/`plot_correlograms` render for the merged curation in its own namespace. |
| `test_single_low_level_builder` | Both `make_compute`'s merged branch and the resolver's `build_merged_analyzer` call the shared `build_analyzer`; `make_compute` does NOT import or call `get_curation_analyzer` (stays DB-free — the tri-part contract / `test_tripart_dispatch_active` is intact). |
| `test_orphan_sweep_preserves_live_cache` | Running the sort-analyzer orphan sweep does NOT delete a live curation's cached analyzer folder (guards H2). |
| `test_published_cache_byte_identical` | After every supported plot/read helper (including one needing an extra extension via a context-managed derivative), the published cache folder is **byte-identical** — no supported path mutates it (B4). |
| `test_direct_access_is_memory_copy` | A caller requesting direct analyzer access gets a detached `save_as(format="memory")` copy, not the cache-backed object. |
| `test_ssviz_merged_plot` | The `visualization.py` facade renders a merged curation (its own guard is rerouted, not raising). |
| `test_cache_keyed_by_uuid` | Two distinct curations resolve to distinct folders; a **reused `curation_id`** (highest deleted, new one inserted) resolves to a **different** folder (keyed by `curation_uuid`, B1) — never the deleted generation's cache. |
| `test_crash_safety_partial_extension` *(slow)* | A folder with a valid manifest but an incomplete/half-written extension (simulated killed write) is treated as corrupt and rebuilt, never surfaced as valid (F3). |
| `test_cache_rebuilds_on_recipe_change` | Changing the waveform recipe invalidates the manifest and triggers a rebuild, not a stale load. |
| `test_cache_rebuilds_on_corrupt_folder` | A missing manifest / truncated zarr forces a rebuild. |
| `test_preview_curation_rejected` | A preview/uncommitted-merge curation raises the actionable "commit first" error, not a namespace-mixing render. |
| `test_zero_unit_raises_typed` | A zero-unit curation raises `ZeroUnitAnalyzerError`; a plot helper translates it to an empty-state view (concrete, not interpretation). |
| `test_si_merge_parity_spike_recorded` | The F6 spike produced a recorded decision (adopt-SI vs retain-custom) with parity evidence; the chosen build path is the one wired in. |
| `test_concurrent_resolve` *(slow, `pytest.mark.slow`)* | Two processes resolving the same curation analyzer serialize on the lock; one builds, the other reuses; no half-published folder is observed. |
| `test_merged_spike_train_identity` | The merged analyzer's spike trains equal the union of contributor spike trains (exact conservation) — the science the render depends on. |
| `test_orphan_cache_reclaim` | Deleting a curation makes its analyzer folder an orphan the reclaim removes; a live curation's folder is retained. |

## Fixtures

Reuse the existing MEArec-derived v2 test fixture (drift-free recording → deterministic sort; see the v2 fixtures README). Synthesize a two-unit merge in-test off that sort so the merged namespace is deterministic. Concurrency test uses two subprocesses against one `sorting_id` (mirror the existing analyzer-cache concurrency test harness).

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Every task implemented as specified; `make_compute`'s temp build is **retained** and `make_compute` is **not** rerouted through the resolver (it stays DB-free) — both paths share the one low-level `build_analyzer`.
- The published analyzer is **read-only**; extra extensions branch to a staged derivative (no in-place mutation of the shared cache); the cache is keyed **per-curation** (no cross-curation dedup); the F6 parity spike decision is recorded and the chosen build path is wired in.
- "Deliberately not in this phase" honored — no facade/FigPack/strict-error creep.
- Validation slice passes; slow/concurrency tests marked.
- Tests exercise real behavior (actual merged templates/spike trains), not tautologies; shared setup in fixtures.
- No plan/phase references in module names, docstrings, or test names.
- The zarr relative-path publish footgun is preserved (a recording-dependent extension works after a publish-move).
