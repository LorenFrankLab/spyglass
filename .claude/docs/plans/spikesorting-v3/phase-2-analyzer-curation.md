# Phase 2 — Analyzer-driven curation (replaces v1 MetricCuration + BurstPair)

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#analyzercuration-replaces-v1-metriccuration--burstpair)

Replaces v1's `MetricCuration` + `BurstPair` with a single `AnalyzerCuration` table that walks SortingAnalyzer extensions to compute metrics, suggest merges, and produce auto-curation labels. Materializing the suggested labels/merges into a new `CurationV3` row is an explicit user action.

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
  - These read directly from the SortingAnalyzer's `correlograms` extension; no separate `BurstPair`-equivalent table.

- **Label-list isolation invariant** (addresses the v0 auto-curation bug class [#1513](https://github.com/LorenFrankLab/spyglass/issues/1513)): when `_apply_label_rules` builds per-unit label lists, each unit MUST receive an independent list object — NOT a shared reference. v0's bug was three independent uses of a shared `[]` default that aliased label lists across units, silently corrupting `accepted_units` filters. v3's implementation creates a fresh list per unit and explicitly copies any input default. Test: `test_label_list_isolation` synthesizes two units that fail a rule; mutates one unit's label list; asserts the other unit's label list is unchanged.

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

- **Documentation update**:
  - Update [docs/src/Pipelines/SpikeSorting/v3.md](docs/src/Pipelines/SpikeSorting/v3.md) (created Phase 1) with a Quality Metrics section.
  - CHANGELOG.md "Unreleased": "v3 `AnalyzerCuration` consolidates v1's `MetricCuration` + `BurstPair`. Auto-curation rules driven by Pydantic-validated thresholds; auto-merge via SI 0.104 presets."

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
| `test_analyzer_curation_make_writes_three_tables` (slow) | After populate, `AnalyzerCuration & key` row has all three object_ids populated; `fetch_nwb()` returns three pandas DataFrames. |
| `test_analyzer_curation_metrics_match_si_compute` (slow) | Independently call `compute_quality_metrics(analyzer, ...)` and compare to the fetched `quality_metrics` DataFrame — exact match. |
| `test_analyzer_curation_merge_suggestions_non_empty_on_obvious_split` (slow) | Synthetic sort with two units that are obvious duplicates (same spike times shifted by 1 ms); auto-merge with preset='similarity_correlograms' suggests their merge. |
| `test_materialize_curation_creates_child` | `AnalyzerCuration.materialize_curation(key)` creates a new `CurationV3` row with `parent_curation_id` pointing to the input curation; auto-registers in `SpikeSortingOutput.CurationV3`. |
| `test_add_extensions_is_idempotent` (slow) | Call `Sorting.add_extensions(key, ["correlograms"])` twice; second call is a no-op (no recompute). |
| `test_v3_analyzer_curation_vs_v1` (slow, integration) | Parity vs v1 `MetricCuration` per tolerances above. Reports per-unit diffs on failure. |

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
- CHANGELOG.md updated.
