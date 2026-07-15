# spikesorting v2 TODO before merge
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

---

## Pre-merge remediation checklist (from 2026-06-30 PR review)

Findings reconciled from an 8-agent review + an independent owner review; severities and
file refs code-verified. Execute as ordered, revertible commits; TDD (failing test first)
on every code change. No Critical/merge-blocker found.

**Status (2026-07-01):** A, B, C, D, E, G DONE + validated. F remains blocked on external two-session fixture hosting. H remains optional polish.

### A — Repo hygiene (M3, L2) — closes TODO #2 + #5 — ✅ DONE
- [x] Remove `.claude/docs/plans/`, `.claude/docs/reviews/`, `.claude/audits/`, `.claude/docs/lessons-learned-new-pipeline.md`
- [x] Revert the `.gitignore` audit allow-list (lines ~182-185)
- [x] Confirm `git diff --check master...HEAD` passes (was failing on `.claude/audits/COMBINED_SYNTHESIS.md` whitespace)
- [x] Strip 10 review-ticket tags from comments (`CNEP-1`, `AVTM-2/3`, `R3`, `R4`, `R27`; 8 files) — keep the self-explanatory body
- [x] Fix dangling helper name `_assert_curation_not_merged` → `_assert_curation_in_raw_namespace` (curation.py:1959)
- [x] Reword `# Phase 1/2` → `# Step 1/2` (_pipeline_presets.py:914/946); fix CHANGELOG.md:141 "when Phase 5 begins" → behavioral trigger

### B — Docs correctness (M2, L1) — TODO #5/#8 — ✅ DONE
- [x] Fix `merge_id` root-vs-curated contradiction in `SpikeSortingV2.md:244,1132` (decoding-on-uncurated risk); align with Migration.md:40 + notebook 10_*.py:285; add explicit "root = uncurated; use final/auto-curated id downstream" callout
- [x] Fix `as_dataframe=True` wording: `unit_id` is the DataFrame **index**, not a column (`SpikeSortingV2.md:1308`)

### C — High correctness (H1, H2, H4) — TDD, focused tests FIRST — ✅ DONE (code-review clean)
- [x] **H1** add `electrode_group_name` to `_member_electrode_signature` via DB-free `electrode_signature_from_rows` (session_group.py / _concat_recording.py). Reused-ids-across-groups now diverge. Validated: 44 concat DB tests.
- [x] **H2** reject same-NWB members in `UnitMatchSelection` (`assert_distinct_member_sessions` → `SameSessionMatchError`) AND count `n_sessions_observed` by nwb (`derive_tracked_units(session_by_sorting=...)`). Validated: table-path DB test.
- [x] **H4** `_units_nwb._affine_recording_start_time` logs a WARNING on the no-origin fallback (keeps 0.0 for legitimate relative timing; other errors propagate).
- [x] (bonus) Fixed a pre-existing teardown bug: `test_full_unitmatch_workflow_*` super_delete'd a merge-registered CurationV2 before its master.

### D — Medium integrity (M1, M4, M6, M7) — TDD — ✅ DONE + DB-validated
- [x] **M1** retry `insert_curation` on a concurrent curation_id collision (`_next_curation_id` + restamp unit_rows + rebuild merge/NWB; `_CURATION_ID_RACE_RETRIES`). Collision integration test (TDD caught a stale-unit_rows bug).
- [x] **M4** `reject_duplicate_quality_metric_content` guard on `QualityMetricParameters.insert`; `insert_default` opts out (intentional franklab/neuropixels dup). Pure + DB tests.
- [x] **M6** both `remove_matched` use cautious `delete(safemode=False)` (materialize keys to dodge MySQL 1093); corrected comments. Cross-env matched=0 integration test.
- [x] **M7** `logger.warning` in `_metric_curation._is_finite_metric_value` `except TypeError` (NaN stays silent). Pure tests.

### E — FigPack exclusion (M5) — DECISION: document, don't split
- [x] Keep `FigPackCuration.make` monolithic (upload=True network publish can't roll back); add it to the tri-part exclusion list **explicitly** (like `TrackedUnit`) with a rationale, in BOTH the test gate (test_integrity.py / merge-dispatch) and a code/doc note, so it doesn't read as accidental inconsistency

### F — CI AUC ship-gate (H3) — ⛔ BLOCKED on external fixture hosting (CI wiring already complete)
The SPYGLASS_V2_REQUIRE_FIXTURES honest-green gate + scheduled fetch step + the
exact enablement instructions already ship (conftest.py:94/163, test-conda.yml:300/346-350).
The only remaining work is EXTERNAL and cannot be done here without breaking CI:
- [ ] (external) Host/upload the two `mearec_polymer_128ch_2sessions_s{1,2}` fixtures + set real URLs in `_fetch.py` (currently `None`; generating in-CI is too heavy — 120s/128ch MEArec)
- [ ] (after hosting) Add the polymer pair to the CI lane's `SPYGLASS_V2_REQUIRE_FIXTURES` + drop the `|| true` on the fetch — then a green run proves AUC>0.85

### G — Test coverage (L4) — TODO #3
- [x] Auto-curation **rule-derived label content** asserted e2e (priority) — strengthened the existing auto-curate test with an INDEPENDENT check vs the real nn_noise_overlap metrics
- [x] Label-based downstream filter on a genuinely-labeled curated export (drop noise / keep accept via `fetch_spike_data`)
- [x] FigPack `fetch → save_manual_curation → DB` as one chain

### H — Polish (L5) + L3 naming waiver
- [x] L3 DECISION: keep `_curation_plan`/`_selection_plan` as **domain terms**; waiver documented in both module docstrings
- [x] `_curation_plan.py` / `_curation_routing.py` use canonical `from spyglass.utils import logger`
- [x] `_artifact_intervals.py:783` comment corrected (names the cautious `.delete(safemode=False)`)
- [x] (follow-up, riskier) `publish_analyzer_atomically` self-acquire the reentrant lock (_analyzer_cache.py:266) — latent contract, no live bug
- [x] (follow-up, riskier) Rename `CurationEvaluation.get_merge_groups` → `get_suggested_merge_groups` (ripples through callers)
- [ ] (follow-up, low) `_zero_center` 15-sample window vs configurable `ms_before`; stale docstring line-numbers (curation.py:1671/2513)

**Env:** conda `spyglass_spikesorting_v2`, Colima + `source ~/spyglass_v2_env.sh`, `pytest -p no:xvfb`.

---

## UX / readiness audit (2026-07-01)

Independent 4-agent read-only audit (user journey/API, docs accuracy, error+naming UX,
notebooks) + owner checks. Env re-verified: 1759 tests collect clean (1.76s), DB-free
tests pass, container lifecycle OK. **Verdict: ready for PR review now; needs a focused
onboarding-polish pass before wide user testing. Zero code-correctness blockers.** These
findings are ADDITIONS not already covered by A–H above; severities/file-refs code-verified.

**RESOLVED 2026-07-01:** user testing runs against the lab (non-local) DB. The import-time
localhost-only guard (`_assert_v2_db_safe` + the `SPYGLASS_SPIKESORTING_V2_ALLOW_NONLOCAL_DB`
override + its 8 call sites + tests) was REMOVED (commit 504b8d61) so v2 registers its
schemas against whatever host `dj.config` points at. To keep name churn from stranding
orphan tables on the persistent DB, the DataJoint schema layer (schema/table/PK names) is
now FROZEN — Python API-name churn is still fine. Two independent reviews confirmed the
removal is clean (no stragglers, suite collects).

**IMPLEMENTED 2026-07-01 (TDD on code, black+ruff+codespell clean, targeted tests green):**
- U1 ✅ migration snippet rewritten via `RecordingSelection → IntervalList`.
- U3 ✅ doc + `DeletionPreview` docstring corrected (raises, not returns; use
  `preview_existing_entries`).
- U4 ✅ package-root lazy re-exports of the main entrypoints + `__dir__`/`__all__` (test:
  `test_package_root_reexports_*`); bare import stays dependency-light (verified).
- U5 ✅ partial single/concat calls now name the missing field (tests added; existing
  mode-error test updated to the new behavior).
- U6 ✅ `register_preset`→`register_pipeline_preset`, `clone_preset`→`clone_pipeline_preset`,
  dropped the `describe_preset` alias — across src/tests/notebooks/docs.
- U7 ✅ curation notebook: runnable `save_manual_curation(labels=...)` (§3f) + downstream
  `get_spike_times(curated_merge_id)` (§3g).
- Polish ✅ notebook heading renumber (Curation/Presets now 1-2-3), CrossSession duplicate
  import removed, artifact.py:281 + metric_curation.py:2287 messages, `MissingDisplayExtension
  Error` centralized in `exceptions.py`, migration "interim/stub" wording trimmed.
- BONUS ✅ fixed a PRE-EXISTING red test (`test_notebook_uses_visualization_facade` pointed at
  the main notebook; the `ssviz` demo moved to Curation in the Jun-30 split — repointed).
- U2 ✅ RESOLVED — localhost-only DB guard removed (commit 504b8d61); v2 runs on the lab DB
  (see RESOLVED note above). `describe_unit_match_choices` ✅ now returns a DataFrame.
- HELD: U8 (god-module decomp — plan written, code deferred), `dev_walkthrough.ipynb`
  placement, the 2 `FigPackCuration.delete_quick()` (rebuild-justified).
- All U1-U7 + the API-naming sweep + the guard removal are COMMITTED on spikesorting-v2.

### Fix before user testing (a tester hits these in the first hour)
- [x] **U1 (docs, verified)** Migration reconstruction snippet raises `KeyError` — `Recording`
      has no `saved_start`/`saved_end` column (only `duration_s`, `content_hash`, etc.;
      recording.py:1067-1077); the "range stored on the Recording row" prose is false. Rewrite
      via the real `IntervalList`/analysis-NWB accessor. SpikeSortingV2_Migration.md:124-131
- [x] **U2 (DB guard)** RESOLVED by removing the localhost-only guard (commit 504b8d61): v2
      no longer refuses a non-localhost DB, so the first-import-cell failure is gone. (Minor
      leftover: notebooks still don't explicitly name the `spyglass_spikesorting_v2` conda env
      as a prerequisite — a small nicety, not a blocker.)

### Fix before / at PR review
- [x] **U3 (docs, verified)** `set_group_by_*(confirm=False)` is documented to RETURN a
      `DeletionPreview` but RAISES `ValueError` (recording.py:327-333); the `DeletionPreview`
      docstring repeats the wrong claim (recording.py:97-99). Point users to
      `preview_existing_entries()`. SpikeSortingV2.md:52-55
- [x] **U4 (API)** Import-surface split: `initialize_v2_defaults` is on the package root but
      `run_v2_pipeline`/`describe_*` need `.pipeline`; natural
      `from spyglass.spikesorting.v2 import run_v2_pipeline` fails. Add lazy `__getattr__`
      re-exports for the main entrypoints. __init__.py:85-102
- [x] **U5 (API)** Omitting `sort_group_id` → generic "requires exactly one input mode" error
      that misreads as mode-mixing. Detect partially-set single-session fields and name the
      missing one. _pipeline_run.py:354-368
- [x] **U6 (naming, cheap pre-prod rename)** `register_preset`/`clone_preset` mismatch the
      `list_pipeline_presets`/`describe_pipeline_presets` family → rename to
      `register_pipeline_preset`/`clone_pipeline_preset`; drop the redundant `describe_preset`
      alias (3rd near-identical name). _pipeline_presets.py:534,624,697
- [x] **U7 (notebooks; overlaps G)** Curation notebook computes the hand-curated `merge_id`
      but never fetches downstream, and manual per-unit labeling is only shown commented-out
      (_Curation.py:226-234,346). Add a `get_spike_times(merge_id)` closing cell + a runnable
      `save_manual_curation(labels=...)` example.
- [ ] **U8 (maintainability; overlaps post-godmodule roadmap)** 4 god-modules — sorting.py
      2862 / metric_curation.py 2683 / curation.py 2674 / recording.py 2392 — heavy for
      reviewers. Decompose curation/sorting/recording before more piles on.

### Polish (Low)
- [x] Notebook section-number/heading debris (non-sequential ## in _Curation/_Presets) +
      duplicate `import importlib.util` in _CrossSession.py:272
- [x] `dev_walkthrough.ipynb` moved to `notebooks/dev/` and removed from the
      docs notebook symlink set
- [x] 2 cosmetic error-message inconsistencies: artifact.py:281 (str-concat vs `!r`),
      metric_curation.py:2287 (omits the offending `metric_names`)
- [x] `MissingDisplayExtensionError` lives in _visualization.py, not the central exceptions.py
- [x] `describe_unit_match_choices` returns a DataFrame, preserving the
      "describe_* → DataFrame" pattern
- [x] 2 `delete_quick()` on `FigPackCuration` replaced with dependency-aware
      `delete(safemode=False)` in the stale-offline-bundle rebuild paths
- [x] Migration doc "interim"/"stub" internal-history wording (Migration.md:60,214,226)

### Confirmed strong (don't touch)
run_v2_pipeline single-entrypoint (~4 touchpoints vs v1's ~15-20); root_/analysis_ merge-id
naming; preflight (read-only, complete, actionable); typed error UX (40+ exceptions, no
scaffolding leaks); docs ~150 names resolve to code + strong v1-parity section; describe_run;
initialize_v2_defaults stale-catalog audit.

---

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
