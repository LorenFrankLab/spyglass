# Phase 1b - Curation evaluation and final merged-unit metrics

[back to PLAN.md](PLAN.md) . [overview](overview.md)

**MUST for scientific completeness after phase-1.** Phase-1 correctly guards the
old `AnalyzerCuration` path from scoring raw units and attaching labels to a
merged unit namespace. That guard prevents wrong science, but it leaves the user
without the thing they need at the end of the workflow: quality metrics computed
on the final, committed curated unit set.

This phase adds the minimal independently-shippable fix: evaluate an existing
committed `CurationV2` row in its own unit namespace. It does **not** redesign
child-curation editing or acceptance helpers; those are split into
[phase-1c-curation-composition-acceptance.md](phase-1c-curation-composition-acceptance.md).

## Design contract

1. **`CurationEvaluation` evaluates one `CurationV2` row.** Its metrics,
   proposed labels, and proposed merge groups are keyed by exactly
   `(sorting_id, curation_id, unit_id)` from that curation. Merged units get
   recomputed metrics over their merged spike trains/templates; they do not
   inherit metrics from the highest-amplitude contributor.
2. **Evaluation is for committed curation states.** A row with unapplied proposed
   merges (`apply_merge=False` plus a real multi-unit merge group) is a
   draft/proposal, not a final downstream curation. This phase rejects those rows
   at the evaluation boundary.
3. **Evaluation outputs are proposals until accepted.** Merge suggestions and
   proposed labels are stored as evaluation results. Turning them into a new
   committed child curation is phase-1c.
4. **Do not weaken the phase-1 `AnalyzerCuration` guard.** `AnalyzerCuration`
   remains the raw-sort compatibility path. `CurationEvaluation` is the new path
   for post-merge/final metrics.

## Inputs to read first

- [src/spyglass/spikesorting/v2/metric_curation.py](../../../../src/spyglass/spikesorting/v2/metric_curation.py) - existing `AnalyzerCuration` metric compute, parameter tables, NWB write/read helpers, and the phase-1 R27 guard.
- [src/spyglass/spikesorting/v2/curation.py](../../../../src/spyglass/spikesorting/v2/curation.py) - `CurationV2.get_sorting`, `get_merged_sorting`, `has_unapplied_proposed_merges`, and `Unit` rows.
- [src/spyglass/spikesorting/v2/_units_nwb.py](../../../../src/spyglass/spikesorting/v2/_units_nwb.py) - curated units read/write, merge dedup, stored sample frames, and `obs_intervals` propagation.
- [src/spyglass/spikesorting/v2/_sorting_analyzer.py](../../../../src/spyglass/spikesorting/v2/_sorting_analyzer.py) - canonical recording reconstruction and analyzer build helpers.
- [src/spyglass/spikesorting/v1/metric_curation.py](../../../../src/spyglass/spikesorting/v1/metric_curation.py) - v1 behavior: metrics are computed from `CurationV1.get_sorting(sort_key)`, so a merged curation is scored as merged.
- [docs/src/Features/SpikeSortingV2.md](../../../../docs/src/Features/SpikeSortingV2.md) - current docs explicitly say post-merge metrics are not available.

## Tasks

1. **Add a committed-curation predicate and use it at evaluation boundaries.**
   Add `CurationV2.is_committed_curation(key)` and
   `CurationV2.assert_committed_curation(key, *, context)`:
   - return/allow `True` for root, label-only, and already-applied merge rows;
   - return/raise for preview rows where
     `has_unapplied_proposed_merges(key) is True`.

   Keep the phase-1 generic-accessor warning for existing preview rows, but make
   new `CurationEvaluationSelection.insert_selection` reject previews. Re-assert
   the same predicate in `CurationEvaluation.make_fetch` so a row planted via
   `allow_direct_insert` cannot compute.

2. **Introduce `CurationEvaluationSelection` and `CurationEvaluation`.** Put the
   canonical tables in `metric_curation.py` for now, reusing the existing
   parameter tables:

   ```python
   @schema
   class CurationEvaluationSelection(SelectionMasterInsertGuard, SpyglassMixin, dj.Manual):
       definition = """
       curation_evaluation_id: uuid
       ---
       -> CurationV2
       -> QualityMetricParameters
       -> AutoCurationRules
       -> AnalyzerWaveformParameters.proj(metric_waveform_params_name="waveform_params_name")
       """

   @schema
   class CurationEvaluation(SpyglassMixin, dj.Computed):
       definition = """
       -> CurationEvaluationSelection
       ---
       -> AnalysisNwbfile
       metrics_object_id: varchar(72)
       merge_suggestions_object_id: varchar(72)
       proposed_labels_object_id: varchar(72)
       """
   ```

   The deterministic identity payload is
   `(sorting_id, curation_id, metric_params_name, auto_curation_rules_name,
   metric_waveform_params_name)`, parallel to `AnalyzerCurationSelection`. The
   default `metric_waveform_params_name` is resolved exactly as today from the
   sort's source preprocessing recipe. Reuse `QualityMetricParameters`,
   `AutoCurationRules`, `apply_snr_peak_sign`, `apply_label_rules`,
   `AnalyzerCuration._compute_metrics`,
   `AnalyzerCuration._compute_merge_groups`, and the existing analyzer-curation
   NWB serialization functions where possible.

3. **Evaluate the curated sorting, with a raw-sort analyzer fast path.** Factor a
   recording-only resolver out of
   `_sorting_analyzer.reconstruct_recording_and_sorting`, for example
   `reconstruct_recording_for_sorting(Sorting(), {"sorting_id": ...})`. Then
   `CurationEvaluation.make_compute` should:
   - fetch the artifact-masked or concat recording for `sorting_id`;
   - fetch `curated_sorting = CurationV2.get_sorting(curation_key)`;
   - compute metrics/labels/merge suggestions over the evaluated unit set;
   - write the same three scratch tables as `AnalyzerCuration`.

   For committed root/label-only curations whose unit set is unchanged from the
   raw sort (`merges_applied=False` and no unapplied proposed merges), use the
   existing cached raw-sort analyzers via `Sorting().get_analyzer(...)` for the
   display and optional metric recipes. This is a cheap fast path and avoids a
   needless temporary analyzer in the common case.

   For committed merged curations, build temporary DISPLAY and optional whitened
   METRIC analyzers over `(recording, curated_sorting)`. Do **not** publish these
   analyzers into the canonical `analyzer_path(sorting_id, waveform_params_name)`
   cache. Their identity is curation-scoped, not sorting-scoped, and phase-4a's
   cache lifecycle is already busy. Use `TemporaryDirectory` and clean it on
   success/failure. A persistent curation-analyzer cache can be a later
   performance optimization once the scientific contract is correct.

4. **Make the unit namespace invariant explicit.** After computing metrics,
   assert:

   ```python
   set(metrics_df.index.astype(int)) == set(
       (CurationV2.Unit & curation_key).fetch("unit_id")
   )
   ```

   before labels, merge suggestions, or NWB writes. The same check belongs on
   the merge-suggestion output: every suggested member must be a unit in the
   evaluated curation. This catches stale temp analyzers, accidental raw-sort
   analyzer reuse, or preview rows that slipped past selection.

5. **Docs and changelog.** Update:
   - `docs/src/Features/SpikeSortingV2.md`: replace "post-merge metrics are not
     yet available" with the committed-curation evaluation workflow;
   - migration docs/notebooks: the pipeline returns or clearly points users to
     the final committed `merge_id` and final `CurationEvaluation` key;
   - `CHANGELOG.md`: note that final metrics on merged units are now recomputed
     from the merged unit set.

## Deliberately not in this phase

- **Parent/label composition and acceptance helpers.** This phase evaluates
  existing committed curations. Creating new committed child curations from
  evaluation outputs is [phase-1c](phase-1c-curation-composition-acceptance.md).
- **Persistent curation-scoped analyzer cache.** Temporary analyzers are slower
  but correct and easy to clean. A durable cache needs identity, recompute, and
  orphan-cleanup policy; that belongs after this scientific contract lands.
- **Same-process v1 numeric parity test.** v1 `MetricCuration` requires the
  legacy SI runtime via `_require_legacy_si_environment`, so a direct v1-v2
  numeric parity test is likely infeasible in the normal v2 test environment.
  This phase instead pins the v1 semantic contract and tests the v2 merged-unit
  metric values directly.
- **Removing `AnalyzerCuration` entirely.** It can be deprecated or hidden from
  docs later, but deleting/renaming the table is not necessary to ship correct
  final metrics.
- **FigPack/manual UI design.** The evaluation contract should inform Phase 5,
  but this phase only adds the backend evaluation state model and tests.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_curation_evaluation.py::test_final_metrics_recomputed_for_merged_unit` (new) | Create a two-unit curation, commit a merge with `apply_merge=True`, populate `CurationEvaluation` with `metric_names=["num_spikes", "firing_rate"]`; `get_metrics` contains the fresh merged unit id, excludes absorbed contributor ids, and `num_spikes` equals `CurationV2.Unit.n_spikes` / the merged spike train length. This fails on the old raw-sort analyzer path. |
| `test_curation_evaluation.py::test_curation_evaluation_rejects_preview_curation` (new) | A curation with `apply_merge=False` and a real merge group is rejected at `CurationEvaluationSelection.insert_selection` and at `make_fetch` if a row was planted directly. |
| `test_curation_evaluation.py::test_metric_namespace_matches_curation_units` (new) | Root, label-only child, and merged child evaluations all produce metric indexes exactly equal to `CurationV2.Unit` for that same key. |
| `test_curation_evaluation.py::test_root_curation_uses_cached_raw_analyzer_fast_path` (new) | A committed root/label-only curation with unchanged unit set evaluates through `Sorting.get_analyzer` rather than building a temp curation analyzer; a merged curation does not use the fast path. |
| `test_curation_evaluation.py::test_final_snr_peak_sign_uses_sorter_polarity` (new or adapted) | `CurationEvaluation` applies the same sorter-polarity SNR fix as `AnalyzerCuration`; positive-going/bidirectional sorts do not fall back to `"neg"`. |
| `test_curation_evaluation.py::test_zero_unit_curation_evaluation_writes_empty_tables` (new) | A zero-unit committed curation writes empty metrics/labels/merge-suggestion tables and `get_metrics` returns an empty DataFrame. |
| (regression) existing `test_analyzer_curation_over_merged_parent_rejected` | Still passes. `AnalyzerCuration` remains guarded; the new merged-unit metric path is `CurationEvaluation`, not raw-sort `AnalyzerCuration`. |
| (regression) existing curation merge/dedup/disjoint-interval tests | Stored frames, cross-gap dedup, `obs_intervals`, and lazy/applied merge parity remain correct. |

## Fixtures

- Final metric tests can reuse `planted_two_unit_sort` or `populated_sorting`
  plus `CurationV2.create_merged_curation`.
- Namespace and preview-rejection tests should use small DB-backed fixtures but
  avoid real sorter reruns when possible.
- SNR-polarity coverage can reuse the planted positive/negative templates from
  the phase-1 SIG-2 tests.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Final merged-unit metrics are computed from `CurationV2.get_sorting(curation_key)`,
  not `Sorting().get_analyzer({"sorting_id": ...})`.
- Root/label-only curations with unchanged unit sets use the cached raw-sort
  analyzer fast path, while merged curations use curation-scoped temp analyzers.
- The metric index invariant is enforced before NWB write and before applying
  label rules.
- Preview curations are rejected for evaluation; committed root, label-only, and
  applied-merge curations are allowed.
- Temporary curation analyzers are cleaned on success and failure and never
  collide with the canonical sorting analyzer cache.
- `AnalyzerCuration` remains guarded; no code path reintroduces the raw/merged
  namespace bug phase-1 fixed.
- Docs teach committed curation states plus `CurationEvaluation` as the normal
  final-metric workflow; preview curations are described only as legacy/draft
  behavior.
