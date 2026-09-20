# Phase 3c — Optional motion correction independent of concatenation

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [design](designs-motion-and-matching.md#independent-motion-stage)

**Status:** Planned owner-requested feature addition (2026-09-19), not implemented. Depends on phases 3a and 3b. Ships separately from phase 4a. Original review-fix merge gates remain as recorded in PLAN.md; adding this feature does not silently redefine their scheduling.

**Inputs to read first:**

- `recording.py` — `RecordingSelection`, `Recording`, `DriftEstimate`, and motion serialization helpers (all under `src/spyglass/spikesorting/v2/`). The existing estimate is QC-only and does not pin a configurable artifact mask.
- `session_group.py` — `MotionCorrectionParameters`, `ConcatenatedRecordingSelection.MemberSnapshot`, and `ConcatenatedRecording.make_compute` / cache rebuild / `MemberBoundary`.
- `_concat_recording.py` — `resolve_motion_correction`, `mask_member_recordings`, `build_concatenated_recording`; `AUTO_SAME_DAY_PRESET` currently resolves to `rigid_fast`.
- `sorting.py` — source/artifact selection parts, `resolve_source`, `make_fetch` / `make_compute`, analyzer rebuild and statistics-span readers; `_selection_identity.py::sorting_identity_payload`.
- `_pipeline_run.py`, `_pipeline_presets.py`, `_pipeline_preflight.py`, `_pipeline_types.py`, `_recipe_catalog.py` — motion is currently coupled to concat mode.
- `curation.py::get_recording`, `_curation_analyzer.py`, `_sorting_analyzer.py`, `recompute.py`, `unit_matching.py` bundle extraction, and `concat_member_curation.py` — all consumers must agree on the effective recording while preserving original lineage.
- `_sorting_artifact_mask.py`, `_recording_geometry.py`, `_recording_nwb.py`, `_nwb_provenance.py` — phase 3a/3b's persisted coordinate and validity contracts.
- Pinned SI 0.104.3 `preprocessing/motion.py`, `sortingcomponents/motion/` and the upstream links in the shared design. Check actual signatures/defaults; do not implement against an unpinned tutorial.

**Designs referenced:** [independent motion stage](designs-motion-and-matching.md#independent-motion-stage), [valid-sample statistics](designs.md#valid-sample-statistics), [geometry normalization](designs.md#geometry-normalization).

## Tasks

- **Baseline and scientific validation manifest.** Capture existing off/concat/QC behavior and source provenance. Build seeded no-motion, rigid/nonrigid drift, gap/jump, and artifact fixtures. Separate development and held-out acceptance populations; record numeric metric gates before acceptance, as required by the shared design. Benchmark `rigid_fast`, `dredge`, and `dredge_fast` without assuming any preset is validated for polymer probes.
- **Reusable source resolution.** Add one v2 resolver/service for original lineage plus effective unwhitened traces, masks, statistics/continuity spans, and channel mapping. Move motion's computational helpers out of concat-only code. Keep original-session identity independent of whether a correction reference exists.
- **Parameterized persisted estimate.** Add selection/computed tables with exact source ownership, frozen masks/spans, a named immutable estimation recipe, saved `Motion`, diagnostics, resolved defaults, algorithm/schema versions, and content identity. Reuse the existing serialization where appropriate; preserve `DriftEstimate`'s QC semantics. Capture enough diagnostic evidence to inspect failures; do not store arbitrarily large peak arrays in relational blobs.
- **Separate application artifact.** Add a selection/computed corrected recording keyed by saved estimate plus interpolation recipe. Persist an unwhitened NWB artifact, channel/electrode map, original clocks, observation spans and concat back-map, motion reference, and content hash. Applying a different interpolation recipe reuses the estimate. Rebuild from saved motion; verify hashes before publication.
- **Masks and discontinuities.** Exclude invalid samples from estimation statistics and peak support; preserve/reapply masks after interpolation. Implement and test the shared design's acquisition-gap/member-boundary policy in the pinned SI adapter, including a common spatial reference across spans. Fail clearly for unsupported/insufficient input rather than silently joining gaps, inventing motion, or dropping channels/samples. Concat support is incomplete until known-jump and gap tests pass.
- **Selection and sort identity.** Add an optional correction-reference part (or an equivalent explicit FK contract) and enforce exact agreement with base source and masks at insert and compute time. Fold applied correction into sorting identity; estimate-only and off reuse the same uncorrected sort identity. Update pruning, deletion protection, rebuilds, NWB export, and direct-insert guards.
- **Pipeline modes.** Add `off`, `estimate`, `apply` consistently to planning/preflight/run/receipt surfaces, using a named recipe for the latter two. Remove the rule that a motion recipe implies concat. Validate geometry/support and external/internal sorter-correction compatibility. An estimation/application error cannot silently run an uncorrected sorter.
- **One concat implementation.** Make new concat artifacts own assembly/masks/boundaries and route correction through the same new stage. Remove superseded in-concat correction after the documented preproduction recreation. Do not apply the new stage to an old already-corrected concat cache. Version named recipes explicitly rather than changing the meaning of stored `auto_default` or silently disabling old correction choices.
- **Consumer consistency.** Update sorter input, initial and curated analyzers, metric/review/analyzer rebuilds, curation/merge accessors, UnitMatch extraction, concat member exports, and recomputation to resolve selected processing and original lineage intentionally. Add a direct regression against silently using original traces after sorting corrected traces. Effective geometry checks run after correction too.
- **Documentation with the feature.** Update `docs/src/Features/SpikeSortingV2.md`, quickstart, storage management, migration/recreation instructions, and CHANGELOG. Show all three modes, how to inspect a saved estimate, actual resolved preset, correction ownership relative to the sorter, and experimental status until probe-specific validation passes. Explain that `rigid_fast` contains a rigid DREDge estimator; do not repeat the earlier categorical distinction.

## Deliberately not in this phase

- Changing UnitMatch's temporal halves, probability threshold, or tracking policy (phase 4a retains its own scope).
- Accepting concat-backed matching inputs (phase 4c).
- Automatic drift-threshold selection of a correction algorithm, universal probe support, or a new sorter.
- Rewriting v0/v1 storage or introducing a generic processing-DAG framework.

## Validation slice

Names below are proposed regression targets in `tests/spikesorting/v2/test_motion_correction.py` and the existing pipeline/source/curation tests; group fixtures in shared helpers.

| Test/experiment | Required assertion |
| --- | --- |
| `test_off_and_estimate_preserve_sort_input` | Exact uncorrected traces, source identity, clocks, channel IDs and selected masks; estimate-only creates QC but does not alter the scientific sort identity |
| `test_correction_identity_pins_source_masks_and_recipe` | New mask/estimation/interpolation choices cannot reuse an incompatible result; estimate reuse survives interpolation-only changes |
| `test_estimate_and_corrected_recording_round_trip` | Saved/reloaded displacement coordinates, resolved params, calibrated traces, channel mapping, observation spans and original timestamps agree |
| `test_invalid_support_and_geometry_fail_before_sorting` | Nonfinite/insufficient estimates, unsupported geometry, zero surviving channels and incompatible sorter correction fail clearly with no successful corrected/sort row |
| `test_motion_respects_masks_and_acquisition_boundaries` | Real SI estimation/application cannot use planted masked artifacts or support windows crossing joins; no reset-of-reference discontinuity is introduced |
| `test_motion_preserves_concat_member_mapping` | Unequal-length members with clock gaps keep exact sample boundaries and export timestamps after correction/reload |
| `test_all_consumers_resolve_selected_correction` | Sorter, analyzer, curated/rebuilt analyzer and UnitMatch bundle construction see deliberately distinguishable corrected signal and the same channel mapping |
| `test_motion_failure_cleanup_and_cache_rebuild` | Failure leaves no registered partial artifact; retry is sound; rebuild reapplies saved motion and cannot replace a valid artifact on hash mismatch |
| Known-answer motion + sorting benchmark (`slow`) | Estimate error is measured in a common reference frame; corrected traces and real-sorter precision/recall, merges/splits satisfy the preregistered held-out gates; no-motion controls included |
| Paired representative polymer run (scheduled/manual) | Recorded quality, runtime/memory and border effects for the same data under off and candidate recipes; missing fixtures are visible, not reported as validation |

## Fixtures and execution

- Small synthetic fixtures with planted units and explicit displacement, artifacts and clock gaps; test actual SI estimation/interpolation. Use mocks only to inject failures or observe which source a consumer received.
- Long-duration and representative polymer fixtures in the scheduled/manual v2 lane. Keep the required small regression slice collectable per PR. Phase 5 wires execution and records fixture availability; phase 3c owns the scientific assertions.
- Recreate v2 rows/caches under the repository's preproduction policy when schemas or source identities change. Preserve v0/v1 test lanes.

## Review

Before opening the implementation PR, obtain an independent review of source/coordinate ownership, pinned SI behavior, failure/rebuild handling, and validation evidence. Verify modes work for both input shapes, estimation alone does not apply correction, no consumer silently falls back to base traces, and recipe promotion is backed by the declared benchmark. Feature docs and recreation instructions ship with the code.
