# Spike Sorting v2 follow-ups (moved from TODO.md on 2026-10-01)

TODO.md was a branch planning file and is removed before #1609 merges. Its
open items live here.

## Fixture hosting (blocks the two-session CI gate)

### F — CI AUC ship-gate (H3) — ⛔ BLOCKED on external fixture hosting (CI wiring already complete)
The SPYGLASS_V2_REQUIRE_FIXTURES honest-green gate + scheduled fetch step + the
exact enablement instructions already ship (conftest.py:94/163, test-conda.yml:300/346-350).
The only remaining work is EXTERNAL and cannot be done here without breaking CI:
- [ ] (external) Host/upload the two `mearec_polymer_128ch_2sessions_s{1,2}` fixtures + set real URLs in `_fetch.py` (currently `None`; generating in-CI is too heavy — 120s/128ch MEArec)
- [ ] (after hosting) Add the polymer pair to the CI lane's `SPYGLASS_V2_REQUIRE_FIXTURES` + drop the `|| true` on the fetch — then a green run proves AUC>0.85

## Pre-merge checklist (from TODO.md)

1. Make sure we are using all the spyglass machinery
2. no phase/plan etc in comments, function names, test names.
3. Ensure all integration tests cover the common cases (particuarly the curation scenarios)
4. Make sure we are using `delete` and not `quick_delete`, etc unless we have a good reason. Don't want to bypass cautious delete machinery.
5. Make sure all the docs are up to date and plan docs are gone.
6. if there's a chance to refactor logic where there is complicated code trying to get around a problem, but there's a simpler, cleaner, clearer, more maintainable and/or more efficient way.
7. Evaluate the names of everything, and make sure they are clear, consistent, and follow the naming conventions.
8. Make sure we have documented what is the same as v1 and what is different
9. Look for hardening against impossible cases or overly compplicated solutions.
10. Are things implemented in the same way? Are we being consistent? Are we using the same patterns and approaches for similar problems?

## U8 — god-module decomposition plan (2026-07-01, written only; no code)

Four modules 2.4-2.9K lines. Goal: readability/maintainability before Phase-6+ piles
on. This is MOVE/EXTRACT ONLY — zero behavior change; any output diff is a bug.

### Principles
- Prefer the established pattern: DB-free logic -> `_*` service modules (unit-testable
  without a DB), e.g. `_curation_transforms`, `_signal_math`, `_sorting_dispatch`,
  `_recording_*`, `_metric_curation_plots`.
- Splitting `@schema` table CLASSES across modules is allowed but has DataJoint
  import-order/FK subtleties (importing a `@schema` module opens a DB connection; FK
  children must import after parents). Reserve table-splitting for clean multi-table
  cases; never split a heavy Computed table from its Selection (tight `make()` coupling).
- Verify per module: (1) collection (no import breaks), (2) that table's full suite green
  before+after, (3) one end-to-end (notebook or pipeline run) to catch `@schema`
  registration / FK regressions.
- Sequencing: lowest-risk first to prove the split mechanics, then highest-value.
  recording (1) -> curation (2) -> metric_curation (3) -> sorting (4). Reassess after (1).
- Each module = its own reviewed PR. Do NOT bundle with feature work.

### 1. recording.py (2393; tables SortGroupV2, PreprocessingParameters, RecordingSelection, Recording, DriftEstimate) — PRIORITY 1, lowest risk
- Extract `DriftEstimate` + its motion helpers (`_motion_to/from_storage_dict`,
  `_motion_max_abs_displacement_um`, `_motion_n_temporal_bins`) -> `drift_estimate.py`.
  Self-contained drift-QC, torch-only, nothing else depends on it at import.
- Extract `SortGroupV2` + `DeletionPreview` + `_validate_reference_fields` + the
  inspect-before-destroy grouping -> `sort_group.py`.
- KEEP `RecordingSelection` + `Recording` + `PreprocessingParameters` in recording.py
  (Recording's FK chain + make()). Ensure new modules are imported so tables register.
- Effort medium. Suites: recording + concat.

### 2. curation.py (2674; ONE class CurationV2, 38 methods) — PRIORITY 2, highest value
Can't table-split. Two complementary moves:
- Body-extraction into DB-free modules (extend `_curation_transforms`/`_curation_plan`/
  `_curation_routing`): the roadmap-flagged `resolve_restriction` resolution, payload
  normalization, merge-lineage computation.
- Concern mixins composed into `@schema class CurationV2(...)`. Method clusters:
  - insert/identity: insert_curation, save_manual_curation, _next_curation_id,
    create_curation, create_merged_curation
  - resolution/restriction: resolve_restriction, resolve_effective_*, key-building
  - accessors: summarize_curation, label_options, get_merged_sorting, get_merge_groups
  - merge-lineage: ParentMergeGroup, namespace-aware merge groups
  Keep the `@schema` class, `definition`, and part tables (MergeGroup/ParentMergeGroup/
  Unit/UnitLabel) in curation.py; mixins are plain bases in `_curation_*_mixin.py`.
- Effort high. Suites: curation + composition (the phase-1c coverage).

### 3. metric_curation.py (2684; QualityMetricParameters, AutoCurationRules, CurationEvaluationSelection, CurationEvaluation) — PRIORITY 3
- Extract CurationEvaluation's plotting/diagnostic cluster (plot_correlograms,
  plot_units_qc, plot_by_sort_group_ids, investigate_pair_xcorrel/peaks,
  plot_peak_over_time, get_peak_amps) -> a plots mixin (some logic already in
  `_metric_curation_plots`).
- Move the Lookups (QualityMetricParameters, AutoCurationRules) toward the `_params/`
  cluster (validators already live in `_params/metric_curation.py`).
- KEEP CurationEvaluationSelection + CurationEvaluation core (make/metrics).
- Effort medium-high. Suites: metric-curation + auto-curation.

### 4. sorting.py (2862; SorterParameters, AnalyzerWaveformParameters, SortingSelection, Sorting) — PRIORITY 4
- Move the two Lookups toward `_params/`.
- Continue extracting Sorting.make helpers into existing `_sorting_dispatch`/
  `_sorting_units`/`_sorting_analyzer`/`_sorting_artifact_mask` (mostly done); target the
  remaining large make_* bodies + get_analyzer/get_sorting accessors.
- Effort medium. Suites: single-session + heavy sort.

### Do-not
- Don't import a `@schema` module at a test's top level (opens a DB conn at collection).
- Keep any parallel-worker kernels in DB-free modules.
- Reference no phase/plan vocabulary in the new module/function/test names.
