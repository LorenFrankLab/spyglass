# Phase 2 — Analyzer-driven curation + recompute verification

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#analyzercuration-replaces-v1-metriccuration--burstpair)

Replaces v1's `MetricCuration` + `BurstPair` with a single `AnalyzerCuration` table that walks SortingAnalyzer extensions to compute metrics, suggest merges, and produce auto-curation labels. Materializing the suggested labels/merges into a new `CurationV3` row is an explicit user action. This phase also ports v1's recompute verification pattern to v3 `Recording` and `Sorting` artifacts so large caches can be deleted only after a successful round-trip check.

**Inputs to read first:**

- [src/spyglass/spikesorting/v1/metric_curation.py](src/spyglass/spikesorting/v1/metric_curation.py) — entire file; v3 consolidates this.
- [src/spyglass/spikesorting/v1/burst_curation.py](src/spyglass/spikesorting/v1/burst_curation.py) — entire file; burst-pair logic moves into auto-merge presets.
- [src/spyglass/spikesorting/v1/metric_utils.py](src/spyglass/spikesorting/v1/metric_utils.py) — metric helper functions; mostly obsoleted by SI 0.104's `compute_quality_metrics`.
- [src/spyglass/spikesorting/v3/sorting.py](src/spyglass/spikesorting/v3/sorting.py) (from Phase 1) — `Sorting.get_analyzer(key)` is the upstream input.
- [.claude/docs/plans/spikesorting-v3/appendix.md § Quality metric renames in 0.104](appendix.md#quality-metric-renames-in-0104) — what metric names changed.
- [.claude/docs/plans/spikesorting-v3/appendix.md § SortingAnalyzer extension dependencies](appendix.md#sortinganalyzer-extension-dependencies) — what to compute before metrics.

**Contracts referenced:**

- [SortingAnalyzer Storage Layout](shared-contracts.md#sortinganalyzer-storage-layout) — adds extensions to an existing analyzer in place (the on-disk folder grows; this is supported by SI binary_folder format).
- [Pydantic Parameter Schema Convention](shared-contracts.md#pydantic-parameter-schema-convention) — `QualityMetricParameters`, `AutoCurationRules` get Pydantic models.
- [Job-Kwargs Resolution](shared-contracts.md#job-kwargs-resolution) — extension compute uses resolved kwargs.

**Designs referenced:** [AnalyzerCuration (replaces v1 MetricCuration + BurstPair)](designs.md#analyzercuration-replaces-v1-metriccuration--burstpair).

## Tasks

- **Implement `_params/metric_curation.py`** with Pydantic models:
  - `QualityMetricParamsSchema` — `metric_names: list[str]`, `metric_kwargs: dict[str, dict]`, `skip_pc_metrics: bool = True`. Validates each metric name against SI 0.104's exported list (import `from spikeinterface.qualitymetrics import quality_metric_list` if available; otherwise hardcode the v3-supported set).
  - `AutoCurationRulesSchema` — `label_rules: dict[str, dict]` where each value is `{"operator": ">", "threshold": 0.5, "label": "noise"}`; `auto_merge_preset: Literal["similarity_correlograms", "temporal_splits", "x_contaminations", "feature_neighbors", "none"]`; `auto_merge_kwargs: dict`.

- **Implement `metric_curation.py`** per [designs.md § AnalyzerCuration](designs.md#analyzercuration-replaces-v1-metriccuration--burstpair). Specific:
  - `QualityMetricParameters` Lookup with three default rows: `("franklab_default", ...)`, `("neuropixels_default", ...)`, `("minimal", {"metric_names": ["snr", "isi_violation", "firing_rate"], ...})`.
  - `AutoCurationRules` Lookup with default rows including a `("franklab_default_thresholds", ...)` that mirrors v1's auto-label conventions (snr < 2 → noise, isi_violation > 0.05 → mua, etc.; pull thresholds from `MetricCurationParameters` defaults at [src/spyglass/spikesorting/v1/metric_curation.py:161-191](src/spyglass/spikesorting/v1/metric_curation.py#L161-L191)).
  - `AnalyzerCurationSelection` Manual.
  - `AnalyzerCuration` Computed with `make()` that computes additional extensions, runs `compute_quality_metrics`, applies label rules, runs `compute_merge_unit_groups`, writes three NWB tables (`quality_metrics`, `merge_suggestions`, `proposed_labels`) via `AnalysisNwbfile().build()`.
  - `materialize_curation(key, description="auto-curation") -> dict` — creates a child `CurationV3` row with the proposed labels + merge groups; returns the new curation_key. Auto-registers via the Phase 1 `CurationV3.insert_curation` flow.

- **Port `BurstPair` visualization helpers** into `AnalyzerCuration` methods:
  - `plot_correlograms_by_sort_group(key)` — adapts `BurstPair.plot_by_sort_group_ids` at [src/spyglass/spikesorting/v1/burst_curation.py](src/spyglass/spikesorting/v1/burst_curation.py).
  - `investigate_pair_xcorrel(key, unit_a, unit_b)`.
  - `investigate_pair_peaks(key, unit_a, unit_b)`.
  - `plot_peak_over_time(key, unit_a, unit_b, overlap=True)`.
  - These read directly from the SortingAnalyzer's `correlograms` extension; no separate `BurstPair`-equivalent table.

- **Three-bug-class invariant on `_apply_label_rules`** addressing [#1513](https://github.com/LorenFrankLab/spyglass/issues/1513). The v0 `AutomaticCuration.get_labels` had three distinct bugs (CBroz1's analysis); v3's `_apply_label_rules` must avoid all three patterns AND test each one:
  1. **Loop completion (Bug A in #1513)**: every label rule must be processed; `return` must be outside the rule loop. Bug A was introduced by PR #1281's refactor (2025-04-22) when `return parent_labels` was indented inside the `for metric in label_params:` loop, silently dropping all rules after the first match. Test: `test_label_rules_loop_completes_all_rules` — synthesize 3 rules where rule 2 and rule 3 would add labels to a unit; assert all three apply.
  2. **List-reference isolation (Bug B in #1513)**: per-unit label lists MUST be independent objects. Bug B (since 2022-03-25) shared the same list object across all units matching the same rule, so mutating one unit's labels propagated to all of them — and back into the rule definition itself. Test: `test_label_list_isolation` — two units match the same rule; append to unit A's labels; assert unit B's labels are unchanged.
  3. **Element-level membership check (Bug C in #1513)**: when appending labels to a unit that already has labels, the check must be per-element (`for elem in new_labels: if elem not in existing:`), not list-in-list (`if new_labels not in existing:`). Bug C used the list-in-list form, which always evaluates `False` for a flat string list, so duplicate labels accumulated (e.g., `["noise", "reject", "noise", "reject"]`). Test: `test_label_dedupe_per_element` — apply two rules that both yield `["noise"]`; assert final labels = `["noise"]`, not `["noise", "noise"]`.

  Implementation pattern (from sytseng's fix referenced in #1513):
  ```python
  for unit_id in flagged_units:
      if unit_id not in labels:
          labels[unit_id] = label_list.copy()  # Bug B fix: independent list
      else:
          for element in label_list:           # Bug C fix: per-element check
              if element not in labels[unit_id]:
                  labels[unit_id].append(element)
  # `return labels` at function scope, outside the rule loop (Bug A fix)
  return labels
  ```

- **NaN sanitization for metric serialization** (addresses [#1556](https://github.com/LorenFrankLab/spyglass/issues/1556)): SI's `compute_quality_metrics` legitimately returns `nan` for low-spike units. `AnalyzerCuration.make()` writes metrics via three paths (DataJoint blob, NWB unit column, FigPack URI in Phase 5); ALL three must see NaN coerced to `None` BEFORE serialization. Add `_sanitize_for_json(df) -> df` helper that copies the DataFrame and replaces all non-finite values with `None`. The in-memory `metrics_df` retains NaN for downstream consumers that want to filter on it; only the JSON-bound path gets sanitized. Per the [Empty / NaN / Boundary Invariants contract](shared-contracts.md#empty--nan--boundary-invariants).

- **Implement `_apply_label_rules(metrics_df, label_rules)` helper** in `metric_curation.py`:
  ```python
  def _apply_label_rules(metrics_df, label_rules):
      """Apply threshold rules to a metrics DataFrame.
      Returns dict[unit_id, list[CurationLabel]].

      Rule order matters — earlier rules win on ties (e.g., 'noise' label
      from low SNR overrides 'mua' label from high ISI).
      """
      labels = defaultdict(list)
      for metric_name, rule in label_rules.items():
          op = OPS[rule["operator"]]  # {"<": operator.lt, ">": operator.gt, ...}
          flagged = metrics_df[op(metrics_df[metric_name], rule["threshold"])]
          for unit_id in flagged.index:
              if rule["label"] not in labels[unit_id]:
                  labels[unit_id].append(rule["label"])
      return dict(labels)
  ```

- **Add a `Sorting`-side method** (in [src/spyglass/spikesorting/v3/sorting.py](src/spyglass/spikesorting/v3/sorting.py) from Phase 1, extended here): `Sorting.add_extensions(key, extensions: list[str], **kwargs)`. This is a convenience for callers (including `AnalyzerCuration.make()`) to add extensions to an already-built analyzer in place. Internally: `analyzer = self.get_analyzer(key); analyzer.compute(extensions, **_resolved_job_kwargs(key) | kwargs)`. Documented as idempotent — SI's `compute()` skips already-computed extensions unless `overwrite=True`.

- **Parity test against v1 MetricCuration**. New test `test_v3_analyzer_curation_vs_v1` (slow, integration): on the Phase 0 baseline-captured sort, compute v1 `MetricCuration` (via the existing v1 code path) and v3 `AnalyzerCuration`. Compare the per-unit `snr`, `isi_violation`, `firing_rate`, `num_spikes` columns. Tolerances:
  - `snr`: ±30% (v1 uses mean, v3 uses median — they differ systematically).
  - `isi_violation`: exact (refractory-period violation count is deterministic given sorted spike times).
  - `firing_rate`: exact within float-rounding (spike count / duration).
  - `num_spikes`: exact integer match.
  Any metric outside tolerance fails the test with a per-unit diff report.

- **Port v1's `RecordingRecompute` pattern to v3-specific recompute tables** in `recompute.py`. v1's `recompute.py` ([src/spyglass/spikesorting/v1/recompute.py:1-15, 55-103, 189-224, 527-548](src/spyglass/spikesorting/v1/recompute.py#L1-L548)) is active production infrastructure. v3 keeps recompute in Phase 2, not a future phase, because chronic recordings make deliberate storage reclamation part of the MVP reliability story. Phase 2 adds final-shape recompute tables declared in the v3 draft schema:
  - **`RecordingArtifactVersions`** (Computed, FK `Recording`) — inventories PyNWB/namespace dependencies and records the current `Recording.cache_hash`.
  - **`RecordingArtifactRecomputeSelection`** (Manual, FK `RecordingArtifactVersions` + `UserEnvironment`) — plans a recompute attempt under a labeled `env_id` with a `rounding` precision and optional `xfail_reason`.
  - **`RecordingArtifactRecompute`** (Computed, FK `RecordingArtifactRecomputeSelection`) — reruns the recording reconstruction, compares hashes/object names, and records `matched`, `err_msg`, `created_at`, and `deleted`. `delete_files()` is gated on `matched=1` rows only — no deletion of files that did not round-trip.
  - **`SortingAnalyzerVersions` / `SortingAnalyzerRecomputeSelection` / `SortingAnalyzerRecompute`** — parallel trio for the SortingAnalyzer folder. The comparison inventories analyzer extension metadata and content hashes rather than NWB ElectricalSeries objects.
  - **Storage-reclamation workflow** documented in `docs/src/Pipelines/SpikeSorting/v3-storage-management.md`: recompute-verify (`matched=1`) → `delete_files()` reclaims disk → later `Recording.get_recording()` or `Sorting.get_analyzer()` rebuilds from the stored selection/parameter lineage.
  - This is Phase 2 schema, not a later migration. The zero-migration contract lists these tables as Phase 2 pure additions.

- **Documentation update**:
  - Update [docs/src/Pipelines/SpikeSorting/v3.md](docs/src/Pipelines/SpikeSorting/v3.md) (created Phase 1) with a Quality Metrics section.
  - Add `docs/src/Pipelines/SpikeSorting/v3-storage-management.md` documenting recompute verification and safe deletion.
  - CHANGELOG.md "Unreleased": "v3 `AnalyzerCuration` consolidates v1's `MetricCuration` + `BurstPair`. Auto-curation rules driven by Pydantic-validated thresholds; auto-merge via SI 0.104 presets. v3 recompute tables support safe storage reclamation for Recording and SortingAnalyzer artifacts."

## Deliberately not in this phase

- **No port of `BurstPair` schema** — its work folds into `AnalyzerCuration` auto-merge presets. The v1 `BurstPair` table stays in v1 for legacy data.
- **No removal of v1 `MetricCuration` / `BurstPair`**. Both stay in v1 indefinitely (overview Open Question #3).
- **No multi-session metrics.** Per-session only. Phase 4 handles cross-session.
- **No web-based curation UI.** Phase 5 (FigPack).
- **No automatic materialization.** `AnalyzerCuration.populate()` produces suggestions; user must call `materialize_curation()` to create the next CurationV3 row.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_quality_metric_params_validation` | `QualityMetricParameters.insert1({"params": {"metric_names": ["bogus_metric"]}})` raises validation error; valid metric list inserts. |
| `test_auto_curation_rules_validation` | `AutoCurationRules.insert1({...auto_merge_preset: "bogus_preset"...})` raises. Valid preset inserts. |
| `test_apply_label_rules_basic` | Synthetic metrics DataFrame with two rules ("snr<2 → noise", "isi_violation>0.05 → mua"); asserts correct labels per unit. |
| `test_apply_label_rules_order_matters` | Rules in order produce expected labels when a unit matches multiple thresholds; reordering rules changes outcomes for ties. |
| `test_label_rules_loop_completes_all_rules` | Three rules where rules 2 and 3 both apply to a unit; asserts all matching rules are processed and `return` is outside the rule loop. |
| `test_label_list_isolation` | Two units match the same multi-label rule; mutating unit A's labels after helper return does not change unit B's labels or the rule definition. |
| `test_label_dedupe_per_element` | Two rules emit the same label for one unit; final label list contains one copy, not duplicates from list-in-list membership checks. |
| `test_analyzer_curation_make_writes_three_tables` (slow) | After populate, `AnalyzerCuration & key` row has all three object_ids populated; `fetch_nwb()` returns three pandas DataFrames. |
| `test_analyzer_curation_metrics_match_si_compute` (slow) | Independently call `compute_quality_metrics(analyzer, ...)` and compare to the fetched `quality_metrics` DataFrame — exact match. |
| `test_metric_nan_round_trip` | Low-spike unit produces non-finite metrics; DataJoint blob and NWB serialized outputs contain `None` while the in-memory metrics DataFrame preserves NaN semantics. |
| `test_analyzer_curation_zero_unit_sorting` | Zero-unit `Sorting` row populates `AnalyzerCuration` with empty metric/merge/label tables and no missing-column errors. |
| `test_analyzer_curation_merge_suggestions_non_empty_on_obvious_split` (slow) | Synthetic sort with two units that are obvious duplicates (same spike times shifted by 1 ms); auto-merge with preset='similarity_correlograms' suggests their merge. |
| `test_analyzer_curation_burstpair_visualization_helpers` (slow) | `plot_correlograms_by_sort_group`, `investigate_pair_xcorrel`, `investigate_pair_peaks`, and `plot_peak_over_time` all render against a SortingAnalyzer-backed sort without requiring the v1 `BurstPair` table. |
| `test_materialize_curation_creates_child` | `AnalyzerCuration.materialize_curation(key)` creates a new `CurationV3` row with `parent_curation_id` pointing to the input curation; auto-registers in `SpikeSortingOutput.CurationV3`. |
| `test_add_extensions_is_idempotent` (slow) | Call `Sorting.add_extensions(key, ["correlograms"])` twice; second call is a no-op (no recompute). |
| `test_v3_analyzer_curation_vs_v1` (slow, integration) | Parity vs v1 `MetricCuration` per tolerances above. Reports per-unit diffs on failure. |
| `test_recompute_versions_inventories_correctly` | `RecordingArtifactVersions` and `SortingAnalyzerVersions` populate for existing v3 `Recording` / `Sorting` rows and record dependency/hash manifests. |
| `test_recompute_matches_under_same_env` (slow) | Recompute under the current `UserEnvironment`; `RecordingArtifactRecompute.matched == 1` and `SortingAnalyzerRecompute.matched == 1`. |
| `test_recompute_detects_mismatch_under_different_rounding` (slow) | Force a rounding/hash mismatch; recompute rows record `matched == 0` plus `Name` or `Hash` part rows describing the difference. |
| `test_delete_files_only_for_matched` | `delete_files()` refuses to delete artifacts for `matched=0` and deletes only after a successful round-trip (`matched=1`). |

## Fixtures

- **`baseline_metric_curation`** — pickle of the v1 `MetricCuration` output (fetched from `AnalysisNwbfile` columns), captured by extending `tests/spikesorting/v3/baseline_capture.py` from Phase 0. This phase's baseline-capture extension runs v1 `MetricCuration.populate` on top of the Phase 0 sort and pickles the resulting metrics DataFrame.
- **`synthetic_duplicate_units_sort`** (new in conftest) — a synthetic SI sort with two units that have ~95% overlapping spike times (offset by 1 ms). Used by the auto-merge suggestion test.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — the parity test loads the v1 baseline pickle and asserts per-metric tolerances rather than wrapping `compute_quality_metrics` and asserting it equals itself.
- Docstrings, test names, and module names don't reference this plan, phase numbers, or files inside `.claude/docs/plans/`.
- `Sorting.add_extensions()` works idempotently (SI's overwrite=False semantics respected).
- `BurstPair` visualization helpers are ported into `AnalyzerCuration` (and verified to render against a SortingAnalyzer correlograms extension).
- v1 `MetricCuration` and `BurstPair` are NOT touched (sanity check via `git diff src/spyglass/spikesorting/v1/`).
- `code_graph.py describe` returns clean output for every new table; `path --up`/`path --down` chains match the design DAG; JSON warnings are empty or explicitly accounted for in `precondition-check.md`.
- CHANGELOG.md updated.
