# Phase 4a — Analyzer-cache concurrency, concat compatibility, bypass revalidation, temp routing

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Operational correctness (R5, R30, R31, R12). Extend the per-recording lock +
atomic-publish pattern (already shipped on the recording side) to the analyzer
cache; make concat compatibility check what it claims; re-validate bypass-inserted
rows at compute time; route scratch to the configured temp dir.

**Inputs to read first:**

- [src/spyglass/spikesorting/v2/_analyzer_cache.py:92-135](../../../../src/spyglass/spikesorting/v2/_analyzer_cache.py#L92-L135) — `analyzer_curation_lock` (held only at `metric_curation.py:926-933`).
- [src/spyglass/spikesorting/v2/_recording_fingerprint.py:236-274](../../../../src/spyglass/spikesorting/v2/_recording_fingerprint.py#L236-L274) (`recording_artifact_lock`) and [recording.py:1663-1688](../../../../src/spyglass/spikesorting/v2/recording.py#L1663-L1688) (atomic publish via `os.replace` + reconcile) — the pattern to mirror.
- Unguarded analyzer-folder mutations: [_sorting_analyzer.py:233-234](../../../../src/spyglass/spikesorting/v2/_sorting_analyzer.py#L233-L234), [:391-393](../../../../src/spyglass/spikesorting/v2/_sorting_analyzer.py#L391-L393), [:600-601](../../../../src/spyglass/spikesorting/v2/_sorting_analyzer.py#L600-L601), [:668-670](../../../../src/spyglass/spikesorting/v2/_sorting_analyzer.py#L668-L670); [sorting.py:2151-2153](../../../../src/spyglass/spikesorting/v2/sorting.py#L2151-L2153); [recompute.py:1317](../../../../src/spyglass/spikesorting/v2/recompute.py#L1317).
- [src/spyglass/spikesorting/v2/_concat_recording.py:217-287](../../../../src/spyglass/spikesorting/v2/_concat_recording.py#L217-L287) (`assert_concat_compatible`); [_unitmatch_backend.py:184](../../../../src/spyglass/spikesorting/v2/_unitmatch_backend.py#L184) (raw `get_channel_locations`); [_sorting_analyzer.py:555-574](../../../../src/spyglass/spikesorting/v2/_sorting_analyzer.py#L555-L574) (`to_2d` pattern).
- [src/spyglass/spikesorting/v2/sorting.py:1268-1272,1502-1511](../../../../src/spyglass/spikesorting/v2/sorting.py#L1502-L1511); [artifact.py:940-946](../../../../src/spyglass/spikesorting/v2/artifact.py#L940-L946); [_selection_plan.py:181-194](../../../../src/spyglass/spikesorting/v2/_selection_plan.py#L181-L194) (the insert-time check to re-run).
- [src/spyglass/spikesorting/v2/unit_matching.py:780](../../../../src/spyglass/spikesorting/v2/unit_matching.py#L780), [recompute.py:1115](../../../../src/spyglass/spikesorting/v2/recompute.py#L1115), [_sorting_dispatch.py:480-483](../../../../src/spyglass/spikesorting/v2/_sorting_dispatch.py#L480-L483) (the correct `dir=` example).

**Contracts referenced:** none.

## Tasks

1. **Acquire `analyzer_curation_lock` on every canonical-folder mutation.** The lock (`_analyzer_cache.py:92-135`) is held only by `AnalyzerCuration` (`metric_curation.py:926-933`). Wrap each mutating path in `with analyzer_curation_lock(sorting_id):` — `get_analyzer`'s invalid-folder `rmtree` (`_sorting_analyzer.py:233-234`), `_rebuild_analyzer_folder` (`:391-393`), `build_analyzer` (`:600-601,668-670`), `Sorting.delete`'s `remove_analyzer_cache` (`sorting.py:2151-2153`), and `recompute._delete_analyzer_folders` (`recompute.py:1317`). Each has `sorting_id` in scope or one fetch away.

2. **Atomic publish for analyzer builds.** Mirror the recording atomic-publish (`recording.py:1663-1688`): `build_analyzer` / `_rebuild_analyzer_folder` build into a private temp folder under the analyzer root, then `os.replace(temp_dir, canonical_dir)` while holding the lock, instead of `create_sorting_analyzer(..., overwrite=True)` into the canonical path (`_sorting_analyzer.py:600`). `os.replace` on a directory is atomic on the same filesystem (the analyzer root); a concurrent reader sees either the old or the new folder, never a half-built one. On failure, `rmtree` only the private temp.

3. **`assert_concat_compatible` — add the missing checks.** It compares channel ids + coordinates only (`_concat_recording.py:264-287`). Add a **sampling-frequency** equality check inside it (it has the recording objects). Add an **electrode-identity + brain-region** check at the DB-level caller (where `SessionGroup.Member` rows are known — `session_group.py` concat build): assert each member maps to the same electrode keys/regions in the same channel order, raising an actionable error on mismatch (so two recordings in different physical electrode spaces can't be silently concatenated and read in the anchor frame).

4. **UnitMatch geometry: project to 2D + guard shape.** At `_unitmatch_backend.py:184`, before `np.save(... recording.get_channel_locations())`, project to 2D using the same path as the analyzer (`_sorting_analyzer.py:555-574`: `probe = recording.get_probe(); if probe.ndim == 3: recording = recording.set_probe(probe.to_2d())`) and assert the saved positions are `(n_channels, 2)`. A 3D geometry must not reach the 2D matcher contract silently.

5. **Re-validate bypass-inserted rows at compute time (R31).** A direct insert can create a concat source carrying an `artifact_detection_id` that the normal insert path rejects (`_selection_plan.py:181-194`); at compute the concat branch sets `obs_intervals=None` (`sorting.py:1268-1272`) and masking is skipped (`:1502-1511`), so the sort runs unmasked while claiming artifact metadata. In `make_fetch`/`make_compute`, re-run the insert-time consistency assertion (concat source ⇒ no `artifact_detection_id`) and raise `SchemaBypassError` if violated. Similarly, `ArtifactDetection.make_compute` for a shared group (`artifact.py:940-946`) must re-assert the same-session/fs/n_samples invariant rather than trusting only the insert-time check.

6. **Route scratch to the configured temp dir.** Add `dir=spyglass_temp_dir` to `unit_matching.py:780` (`tempfile.TemporaryDirectory(prefix="unitmatch_")`) and `recompute.py:1115` (`tempfile.mkdtemp(prefix="v2_analyzer_recompute_")`), matching `_sorting_dispatch.py:480-483`. Import `from spyglass.settings import temp_dir as spyglass_temp_dir`.

7. **Docs.** CHANGELOG: analyzer cache is now lock-guarded + atomic-published (mirrors the recording artifact); concat compatibility now checks electrode identity/region/fs; UnitMatch geometry is 2D-guarded; scratch honors the configured temp dir.

## Deliberately not in this phase

- **A general analyzer-cache RAM/disk budget estimator** (PERF perf items) — NICE, not here.
- **Sorter-output-freed** (R4) — phase-1.
- **Concat identity/aliasing redesign** (R6) — phase-2 handles alias versioning; here only the *compatibility check*.
- **Removing the `overwrite=True` API entirely** — keep it for the temp-folder build; only the *canonical* publish becomes atomic.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_analyzer_lifecycle.py::test_concurrent_build_does_not_corrupt_cache` (new) | with the per-sort lock, a build publishes atomically (a reader during a rebuild sees a complete folder); assert `os.replace` is used and the temp folder is gone after success. (Concurrency simulated by asserting the temp-then-replace sequence, not a true race.) |
| `test_analyzer_lifecycle.py::test_delete_and_recompute_hold_lock` (new) | `Sorting.delete` and `recompute._delete_analyzer_folders` acquire `analyzer_curation_lock` (patch the lock to record acquisition). |
| `test_concat_recording.py::test_assert_concat_compatible_rejects_mismatched_fs` (new) | members with different sampling frequencies raise. |
| `test_session_group_concat.py::test_concat_rejects_mismatched_electrode_space` (new) | two members with same channel ids/coords but different electrode keys/regions raise at concat build. |
| `test_unitmatch_backend.py::test_bundle_geometry_is_2d` (new) | a 3D-probe recording yields a saved `(n,2)` `channel_positions.npy`; a shape guard rejects non-2D. |
| `test_sorting.py::test_concat_with_artifact_id_revalidated_at_compute` (new) | a bypass-inserted concat row carrying an `artifact_detection_id` raises `SchemaBypassError` at populate instead of sorting unmasked. |
| `test_temp_routing.py::test_unitmatch_and_recompute_use_configured_temp` (new) | with `temp_dir` patched, the UnitMatch + recompute temp dirs are created under it (not system `/tmp`). |
| (regression) `test_analyzer_lifecycle.py`, `test_session_group_concat.py`, `test_unitmatch.py`, `test_recompute.py` | normal build/delete/concat/match/recompute paths pass. |

## Fixtures

`populated_sorting_with_curation` (`conftest.py:312`) for analyzer-cache tests;
`chronic_2_session_minirec` (`conftest.py:340`) for concat compatibility;
`two_session_curated_group` (`test_unitmatch.py:508`) for UnitMatch geometry; the
temp-routing and compatibility-function tests are largely DB-free (call the
functions with patched `temp_dir` / synthetic recordings).

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Every analyzer-folder mutation acquires the lock; builds publish via temp-then-`os.replace`, never `overwrite=True` into the canonical path.
- The recording-side pattern is genuinely mirrored (same lock semantics, same atomic-publish + rollback shape).
- `assert_concat_compatible` rejects mismatched fs; the DB-level caller rejects mismatched electrode identity/region.
- UnitMatch saves 2D positions with a shape guard; the bypass-revalidation raises at compute.
- Both temp dirs pass `dir=spyglass_temp_dir`.
- Tests exercise real branches (not tautologies); no plan/phase references in code or tests.
