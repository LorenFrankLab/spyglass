# Phase 4b — Auto-curation: computation failure vs expected missingness

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#metric-missingness)

**Inputs to read first:**

- `src/spyglass/spikesorting/v2/_metric_curation.py:35` (`_is_finite_metric_value`), `:108-215` (`apply_label_rules`; policy validation 165-171, all-missing warning 189-202, per-unit loop 203-215), `:239-292` (`isi_violation_fraction`, `rules_payloads_match` with its float32 `math.isclose` note).
- `src/spyglass/spikesorting/v2/metric_curation.py:269-430` (`QualityMetricParameters`; `observed_presence_bin_duration_s=60: float` at 286; `insert_default` 386-390; the shipped `metric_names` / `metric_kwargs` defaults at 311-339 — note which non-nn metrics ship: `presence_ratio`, `amplitude_cutoff`, `isi_violation`, `snr`, ...), `:432-530` (`AutoCurationRules` + `Rule` part; find the `threshold` column type), `:594-654` (shipped rule sets), `:1179` (`make_compute`; the `apply_label_rules` call — grep it), `:2142-2330` (`_compute_metrics`; voltage compute and PC compute at 2286-2300).
- SpikeInterface 0.104.3 `metrics/quality/pca_metrics.py:130-200,265-275` — nn metric params (`min_spikes` default 10, `min_fr`; `fr` computed from `sorting_analyzer.get_total_duration()`) and the bare `except` that returns NaN; `metrics/quality/misc_metrics.py` — `compute_presence_ratios` (NaN when the recording is shorter than `bin_duration_s`) and `compute_amplitude_cutoffs` (NaN + warning when the histogram cannot yield a cutoff; confirmed with a 100-spike unit); `core/analyzer_extension_core.py:1281-1286` — the calculator's "Error computing metric" warning.
- `src/spyglass/spikesorting/v2/_lookup_validation.py:340-360` — QMP fingerprint that includes `observed_presence_bin_duration_s`.
- `src/spyglass/spikesorting/v2/_params/metric_curation.py` — pydantic schemas and `params_schema_version` handling (grep `schema_version`).
- `docs/src/Features/SpikeSortingV2_Migration.md` — the preproduction upgrade/recreation sequence to extend for the column-type changes.

**Designs referenced:** [designs.md#metric-missingness](designs.md#metric-missingness).

## Tasks

- **Register eligibility rules for the shipped-rule metrics and the cheap common ones only** (design, narrowed): nn metrics (`min_spikes`, `min_fr` against SI's total duration, `pca_metrics.py:150-200`); `presence_ratio` (duration < `bin_duration_s`, `misc_metrics.py:95-110`); `amplitude_cutoff` (`n_spikes / num_histogram_bins < amplitudes_bins_min_ratio`, `misc_metrics.py:~1670`, plus the warned "no cutoff" path); `isi_violation` is Spyglass's own `isi_violation_fraction` (`_metric_curation.py:239-292`, `count / (n_spikes - 1)`) and is legitimately NaN when `n_spikes <= 1` — register that floor (a one-spike unit under the shipped `isi_reject` rule must follow `missing_policy`, not fail); `snr`, `firing_rate`, `num_spikes` as never-missing after confirming in source. Do NOT register template metrics or the remaining SI metrics; they are never inspected unless a rule references them.
- **Classifier helpers** `expected_missing_units(rule_columns, *, n_spikes_by_unit, total_duration_s, metric_kwargs, si_warned_units)` and `assert_rule_metrics_computed(metrics_df, rule_columns, expected_missing)` in `_metric_curation.py` per the design, using SI's duration semantics (`sorting_analyzer.get_total_duration()`), not the observed duration. Unregistered rule-referenced columns fail closed on NaN with the register-a-rule message.
- **Wire into `make_compute`**: after `_compute_metrics` returns `metrics_df` (and the captured SI warnings), compute `n_spikes_by_unit` from the analyzer sorting and `total_duration_s = analyzer.get_total_duration()`, derive `rule_columns` from the evaluated `AutoCurationRules.Rule` rows, call `assert_rule_metrics_computed`, and pass `expected` into `apply_label_rules(metrics_df, rule_rows, expected_missing=expected)`.
- **`apply_label_rules` semantics**: with `expected_missing` provided, a non-finite value for a unit outside the metric's expected set raises `ValueError` (computation failure) regardless of `missing_policy`; the policy governs only expected-missing units. Emit the "rule was inert" warning for `"fail"` as well as `"pass"` when every unit is missing. Update the docstring (also delete the `"ignore"` policy mention at 123-125 — the code rejects it).
- **SI calculator warnings, scoped to rule-referenced metrics**: in `_compute_metrics`, record warnings around both `compute_quality_metrics` calls and the template-metric compute. Parse the metric name out of each "Error computing metric <name>" warning; re-raise as `ValueError` ONLY when that metric produces a rule-referenced column (the mapping from SI metric name to output columns is the one `_compute_metrics` already uses to assemble `metrics_df`). Warnings for unreferenced metrics are logged at WARNING and otherwise ignored, so an unrelated custom metric that raises inside SI cannot abort an evaluation — this keeps the design's promise that unreferenced columns never cause failure. Amplitude-cutoff's own NaN warning is routed to the classifier, not escalated.
- **Threshold finiteness** (owner's probe: a validated rule with `threshold=NaN` labels nobody under `<` and everybody under `!=`): `AutoCurationRuleSchema.threshold: float = Field(allow_inf_nan=False)` (`_params/metric_curation.py:331`), and a guard in `apply_label_rules` that raises on a non-finite threshold so a row inserted via any path that bypasses pydantic still fails loudly.
- **Schema**: `observed_presence_bin_duration_s=60: double` (`metric_curation.py:286`) and `Rule.threshold` → `double`; remove the float32 tolerance workaround in `rules_payloads_match` only if the column change makes it dead (keep `math.isclose` if any other single-precision source remains); bump the affected `params_schema_version` following the existing bump convention; extend the migration doc's recreation sequence with the two `alter()` calls.
- **`QualityMetricParameters.insert_default` stale-default check** (`:386-390`): for an existing same-name row, compare its fingerprint to the incoming default and raise `DuplicateParameterContentError`-style with a message naming the row (mirror `AutoCurationRules.insert_rules` at 512-527). Also reject `insert(replace=True)` in `ImmutableParamsLookup` the way `unit_annotation.py:123-130` does, so all params tables behave alike.
- **CHANGELOG** (`[Unreleased]` → Spike sorting v2 → curation): evaluations now fail loudly when a requested metric could not be computed for a unit that meets that metric's own preconditions; `missing_policy` applies only to units SpikeInterface legitimately leaves unassessed; two params columns are now double precision (recreate per the migration sequence); shipped defaults can no longer silently diverge from stored rows.

## Deliberately not in this phase

- Whitening/noise inputs to the metrics (phase 3b).
- Cache-loader exception narrowing in `_sorting_analyzer.py:199-227` / `_curation_analyzer.py:477-481` (appendix Important).
- Preset waveform-window fallback in `clone_pipeline_preset` (appendix Important).
- Review-browser JS fixes (appendix Important).

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/spikesorting/v2/test_metric_curation_transforms.py::test_expected_missing_nn_below_floor` | for `min_spikes=10`, a 5-spike unit is expected-missing for `nn_*` columns only; a 50-spike unit for none |
| `...::test_expected_missing_presence_ratio_short_recording` | with `total_duration_s < bin_duration_s`, every unit is expected-missing for `presence_ratio` and for no other column |
| `...::test_expected_missing_amplitude_cutoff_spike_floor_and_warning` | a unit with `n_spikes < num_histogram_bins × amplitudes_bins_min_ratio` is expected-missing; a unit SI warned about is expected-missing; a unit above the floor with NaN and no warning is a failure |
| `...::test_unreferenced_columns_never_inspected` | a custom column and a template column that are all-NaN cause no failure when no rule references them |
| `...::test_unregistered_rule_column_with_nan_fails_closed` | a rule referencing `trough_half_width` (no registered rule) with a NaN raises with the register-a-rule message; the same rule with all-finite values passes |
| `tests/spikesorting/v2/test_metrics.py::test_classifier_agrees_with_si_nan_pattern` (integration, real SI) | on synthetic sortings built to hit each registered condition (2 s recording for presence_ratio; a 100-spike unit for amplitude_cutoff with default bins/ratio; a 5-spike unit for nn), `assert_rule_metrics_computed` does NOT raise on SI's real output for rules over those columns, and it DOES raise when one finite value is replaced by NaN |
| `...::test_assert_metrics_computed_raises_on_unexpected_nan` | NaN `isi_violation` for a 50-spike unit raises `ValueError` naming the metric and unit |
| `...::test_isi_violation_one_spike_unit_is_expected_missing` | on real `isi_violation_fraction` output for spike counts `[1, 2, 50]` (`[NaN, 0, 0]`), the one-spike unit is expected-missing; under `"pass"` it is unlabelled, under `"fail"` it is labelled, under `"error"` it raises; the 2- and 50-spike units are never expected-missing |
| `tests/spikesorting/v2/test_curation_evaluation.py::test_unreferenced_metric_error_does_not_abort_evaluation` (integration) | monkeypatch an unreferenced SI metric (e.g. `sd_ratio`, requested but in no rule) to raise inside the calculator; `CurationEvaluation.populate` succeeds, the column is NaN, and a WARNING names the metric |
| `...::test_nan_threshold_rejected` | `AutoCurationRuleSchema(threshold=float("nan"))` raises `ValidationError`; `apply_label_rules` with a hand-built NaN-threshold row raises instead of labelling nobody/everybody |
| `...::test_apply_label_rules_fail_policy_all_missing_warns` | `"fail"` with an all-NaN metric logs the inert warning and labels every unit (existing behavior) |
| `...::test_apply_label_rules_unexpected_nan_raises_regardless_of_policy` | with `expected_missing` given and a NaN outside it, `"pass"` and `"fail"` both raise |
| `tests/spikesorting/v2/test_curation_evaluation.py::test_evaluation_fails_when_si_swallows_metric_error` (integration) | monkeypatch SI's `isi_violations` to raise inside the calculator; `CurationEvaluation.populate` raises instead of inserting `labels_by_unit == {}` |
| `...::test_shipped_rules_still_label_on_smoke_fixture` (`pytest.mark.slow`) | the existing franklab rule set produces labels on the smoke fixture with the shipped metric set (no false failures from the classifier) |
| `tests/spikesorting/v2/test_lookup_validation.py::test_qmp_duplicate_content_detected_after_double_column` | inserting the same content under a second name with `observed_presence_bin_duration_s=0.1` raises `DuplicateParameterContentError` (currently silent) |
| `...::test_insert_default_raises_on_stale_same_name_row` | a stored `franklab_default` with different content raises with the row name |
| `...::test_params_lookup_rejects_replace` | `QualityMetricParameters().insert(rows, replace=True)` raises `DataJointError` |

## Fixtures

- Metrics DataFrames built in-test (`pandas`), no DB, for the classifier unit tests.
- Real-SI condition fixtures: `si.generate_ground_truth_recording` variants (2 s duration; a unit subsampled to 100 spikes; a unit subsampled to 5 spikes) with `create_sorting_analyzer` + `compute_quality_metrics` in the test.
- Integration tests reuse `tests/spikesorting/v2/test_curation_evaluation.py`'s fixtures (smoke fixture with planted units via the `Sorting._run_sorter` monkeypatch in `conftest.py:439,570`).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (float32 tolerance workaround if made dead; the `"ignore"` docstring mention).
- User-facing documentation listed as tasks is updated, not deferred.
