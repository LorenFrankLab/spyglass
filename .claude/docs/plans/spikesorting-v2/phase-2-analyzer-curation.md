# Phase 2 — Analyzer-driven curation + recompute verification

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#analyzercuration-replaces-v1-metriccuration--burstpair)

Replaces v1's `MetricCuration` + `BurstPair` with a single `AnalyzerCuration` table that walks SortingAnalyzer extensions to compute metrics, suggest merges, and produce auto-curation labels. Materializing the suggested labels/merges into a new `CurationV2` row is an explicit user action. This phase also ports v1's recompute verification pattern to v2 `Recording` and `Sorting` artifacts so large caches can be deleted only after a successful round-trip check.

## Executor Checklist

- Implement metric/auto-curation parameter schemas and default rows.
- Implement `AnalyzerCurationSelection`, `AnalyzerCuration`, `materialize_curation()`, and the v1 notebook-facing fetch/promote helpers.
- Port the BurstPair visualization workflow into `AnalyzerCuration` methods without adding a separate BurstPair table.
- Implement the v2 clusterless `spike_location` decoding mark via the `spike_locations` SortingAnalyzer extension + waveform-feature adapter, and retire the dead legacy `_get_spike_locations` helper.
- Implement recording/analyzer recompute verification tables and safe deletion gates in the isolated integration database; do not test delete paths against production storage.
- Preserve NaN sanitization, empty-unit, recursive-auto-curation, and label-rule invariants from `shared-contracts.md`.
- Run the Phase 2 validation goals plus `code_graph.py describe/path` for new tables.

**Prerequisite — review-fixes baseline (do this first):**

- The review-fixes checkpoint Phases 1 and 2 MUST be merged before starting AnalyzerCuration work. This phase builds on the corrected schema baseline: optional `SortingSelection.ArtifactDetectionSource` (not nullable `artifact_detection_id`), `SortGroupV2.reference_mode`, `PreprocessingParamsSchema` v3, `ClusterlessThresholderSchema` v4, `Recording.repair()`, and loud-but-valid zero-unit behavior (`require_units` is caller opt-in). Starting AnalyzerCuration before review-fixes lands would build on the to-be-corrected nullable-FK / sentinel schema and force rework.

**Prerequisites for parity validation:**

- Before running the v1↔v2 metric parity test, produce `baseline_metric_curation` in the legacy SI 0.99 environment using `tests/spikesorting/v2/baseline_capture.py` against the Phase 0 real-data baseline sort. This capture runs v1 `MetricCuration.populate` and pickles the resulting metrics DataFrame before the SI 0.104 runtime boundary is applied. The Phase 2 SI 0.104 test loads that pickle; it does **not** run active v1 MetricCuration unless Phase 0c explicitly ported that surface. The capture uses the isolated integration database; `SPYGLASS_ALLOW_PRODUCTION_SMOKE=1` permits read-only production metadata lookup only. If `SPIKESORTING_V2_REAL_NWB_PATH` is unavailable, mark only the parity test skipped with an explicit message; the rest of Phase 2 still runs.

**Inputs to read first:**

- [src/spyglass/spikesorting/v1/metric_curation.py](../../../../src/spyglass/spikesorting/v1/metric_curation.py) — entire file; v2 consolidates this.
- [src/spyglass/spikesorting/v1/burst_curation.py](../../../../src/spyglass/spikesorting/v1/burst_curation.py) — entire file; burst-pair logic moves into auto-merge presets.
- [src/spyglass/spikesorting/v1/metric_utils.py](../../../../src/spyglass/spikesorting/v1/metric_utils.py) — metric helper functions; mostly obsoleted by SI 0.104's `compute_quality_metrics`.
- [src/spyglass/spikesorting/v2/sorting.py](../../../../src/spyglass/spikesorting/v2/sorting.py) (from Phase 1) — `Sorting.get_analyzer(key)` is the upstream input. **Zero-unit contract (review-fix C5):** on a zero-unit sort, `get_analyzer` raises a clear zero-unit-analyzer error OR returns the documented sentinel (whichever the final C5 implementation chose) — it never returns a path to a non-existent analyzer folder. `AnalyzerCuration.make()` and `add_extensions()` must handle that signal explicitly (skip / clear `EmptySortingError`), not assume a loadable analyzer.
- [.claude/docs/plans/spikesorting-v2/appendix.md § Quality metric renames in 0.104](appendix.md#quality-metric-renames-in-0104) — what metric names changed.
- [.claude/docs/plans/spikesorting-v2/appendix.md § SortingAnalyzer extension dependencies](appendix.md#sortinganalyzer-extension-dependencies) — what to compute before metrics.

**Global invariants apply:** [Environment And Database Safety](shared-contracts.md#environment-and-database-safety) and [Code Artifact Naming](shared-contracts.md#code-artifact-naming).

**Phase-specific contracts referenced:**

- [SortingAnalyzer Storage Layout](shared-contracts.md#sortinganalyzer-storage-layout) — adds extensions to an existing analyzer in place (the on-disk folder grows; this is supported by SI binary_folder format).
- [Pydantic Parameter Schema Convention](shared-contracts.md#pydantic-parameter-schema-convention) — `QualityMetricParameters` rows and `AutoCurationRules.insert_rules(row, rule_rows)` get Pydantic-backed validation; direct `AutoCurationRules.Rule` inserts are unsupported.
- [Job-Kwargs Resolution](shared-contracts.md#job-kwargs-resolution) — extension compute uses resolved kwargs.
- [Custom Exception Classes](shared-contracts.md#custom-exception-classes) — recompute and empty-sort errors use the shared exception module.

**Designs referenced:** [AnalyzerCuration (replaces v1 MetricCuration + BurstPair)](designs.md#analyzercuration-replaces-v1-metriccuration--burstpair).

## Tasks

- **Add `_get_v2_deleted_files()` helper to `common.common_file_tracking`** — companion to the existing `_get_v1_deleted_files` at [common_file_tracking.py:71-94](../../../../src/spyglass/common/common_file_tracking.py#L71-L94). Phase 2's `RecordingArtifactRecompute*` and `SortingAnalyzerRecompute*` tables (introduced below) will eventually mark v2 analysis files as `deleted=1` after a successful recompute-and-delete cycle. Without a v2 companion in `common_file_tracking`, the file-tracking infrastructure will flag these intentionally-deleted v2 files as orphans. The Phase 1b plan ([phase-1b-runtime-regressions.md § Deliberately not in this phase](phase-1b-runtime-regressions.md)) explicitly deferred this to Phase 2; it MUST land in the same PR that introduces v2's recompute machinery so the lifecycle is closed in one step. Implementation: mirror `_get_v1_deleted_files`'s try/except shape, querying v2's `RecordingArtifactRecompute().with_names & "deleted=1"` (or whichever v2 recompute table holds the `deleted` flag) for `analysis_file_name`. Document the test as `test_file_tracking_excludes_v2_deleted_files`.

- **Implement `_params/metric_curation.py`** with Pydantic models:
  - `QualityMetricParamsSchema` — `metric_names: list[str]`, `metric_kwargs: dict[str, dict]`, `skip_pc_metrics: bool = True`. The stored `metric_kwargs` dict is passed to SpikeInterface as the `metric_params=` argument. Validate each metric name against SI 0.104's exported list using `from spikeinterface.metrics.quality import get_quality_metric_list`; fall back to `spikeinterface.qualitymetrics` only if resolver testing proves that namespace is still needed for the pinned 0.104 patch range.
  - `AutoCurationRulesSchema` — validates `auto_merge_preset: Literal["similarity_correlograms", "temporal_splits", "x_contaminations", "feature_neighbors", "slay", "none"]`, `auto_merge_kwargs: dict`, and ordered rule rows shaped like `{"rule_index": 0, "rule_name": "snr_noise", "metric_name": "snr", "operator": "<", "threshold": 2.0, "label": "noise"}`. `slay` is included because SI 0.104.3 exposes it in `compute_merge_unit_groups` docs; re-verify if the pinned patch release changes.

- **Implement `metric_curation.py`** per [designs.md § AnalyzerCuration](designs.md#analyzercuration-replaces-v1-metriccuration--burstpair). Specific:
  - `QualityMetricParameters` Lookup with three default rows: `("franklab_default", ...)`, `("neuropixels_default", ...)`, `("minimal", {"metric_names": ["snr", "isi_violation", "firing_rate"], ...})`.
  - `QualityMetricParameters.show_available_metrics()` or a documented `available_quality_metrics()` helper that lists the supported SortingAnalyzer/SpikeInterface metric names. This preserves the v1 notebook-discovery workflow from `MetricParameters.show_available_metrics()` without requiring v1's exact custom metric set.
  - `AutoCurationRules` Lookup with an ordered `Rule` part table and an `insert_rules(row, rule_rows)` helper that validates the full master-plus-rules payload before inserting either table. Direct `AutoCurationRules.insert1()` and `AutoCurationRules.Rule.insert(...)` are not public APIs; the validation suite should flag missing or malformed rule rows inserted outside the helper. Default rows include `v1_default_nn_noise` with two `Rule` rows: `nn_noise_overlap > 0.1 -> noise` and `nn_noise_overlap > 0.1 -> reject`, matching the actual v1 `MetricCurationParameters.contents` default at [src/spyglass/spikesorting/v1/metric_curation.py:183-185](../../../../src/spyglass/spikesorting/v1/metric_curation.py#L183-L185). Any richer Frank-lab SNR/ISI/firing-rate thresholds are new v2 convention rows; include them only if their source is documented during implementation, and do not describe them as v1 parity.
  - `AnalyzerCurationSelection` Manual. `insert_selection()` emits a `logger.warning` (not a raise) when upstream `CurationV2.curation_source == 'analyzer_curation'` — running auto-curation on a materialized child computes metrics over post-merge templates, which is usually not what the user wants but is occasionally intentional. Lineage depth is recoverable from the `parent_curation_id` chain. See [designs.md § AnalyzerCuration](designs.md#analyzercuration-replaces-v1-metriccuration--burstpair).
  - `AnalyzerCuration` Computed with `make()` that computes additional extensions, runs `compute_quality_metrics(..., metric_params=...)`, applies label rules, runs `compute_merge_unit_groups`, writes three NWB tables (`quality_metrics`, `merge_suggestions`, `proposed_labels`) via `AnalysisNwbfile().build()`.
  - Auto-merge extension dependency is explicit: before calling `compute_merge_unit_groups`, ensure `correlograms`, `template_similarity`, `unit_locations`, `template_metrics`, `spike_amplitudes`, and `principal_components` (unless `skip_pc_metrics=True`) are computed as specified in `appendix.md § SortingAnalyzer extension dependencies`. Pass `compute_needed_extensions=False` so the Spyglass-computed extension set is the audited one rather than an implicit SI default. This kwarg exists in SI 0.104.3; re-verify in Phase 0c if the pinned patch release changes.

- **SI 0.104 metric-name + `isi_violation` corrections** (verified against the installed SI 0.104.3 source — see [appendix.md § SpikeInterface 0.104.3 source-verified findings](appendix.md#spikeinterface-01043--source-verified-findings-read-first)):
  - **`nn_isolation` / `nn_noise_overlap` are no longer separate metric names** — they are the two output *columns* of one metric, **`nn_advanced`**. `QualityMetricParamsSchema.metric_names` must request `nn_advanced` (requesting the old names **raises `ValueError`** in 0.104), while an `AutoCurationRules.Rule` still thresholds the `nn_noise_overlap` *column* of the result (so the `v1_default_nn_noise` rule's `metric_name` stays `nn_noise_overlap`). `nn_advanced` is a **PCA metric**: any `QualityMetricParameters` row feeding an nn-based rule must set `skip_pc_metrics=False` (the schema default `True` would skip it, leaving the rule no column to read) and compute `principal_components` first. Pass the Set-A nn params explicitly (`n_components`, `n_neighbors`, `max_spikes`, `radius_um`, `min_spikes`, `seed`) — SI's defaults differ.
  - **Spyglass `isi_violation` is a custom fraction `count / (n_spikes − 1)`** (`v1/metric_utils.py:16-38`), **not** SI's `isi_violations_ratio` (the Hill/UMS2000 contamination estimate, unbounded / can exceed 1). v2 must replicate the count-based fraction; computing it via SI's `isi_violation` column would yield the ratio (a different number) and break v1 parity. Guard the 0-spike `(-1)/(-1)=1.0` artifact.
  - `materialize_curation(key, description="auto-curation") -> dict` — creates a child `CurationV2` row with the proposed labels + merge groups; returns the new curation_key. Auto-registers via the Phase 1 `CurationV2.insert_curation` flow.
  - **Fetch/promote parity with v1 MetricCuration** — implement the v1 notebook-facing helpers on `AnalyzerCuration`:
    - `get_waveforms(key, fetch_all=False)` — returns the SortingAnalyzer waveforms extension or a narrow compatibility object with SI's `get_waveforms_one_unit(unit_id)` plus v1-style `get_waveforms(unit_id)` behavior needed by downstream helpers. This replaces v1 `MetricCuration.get_waveforms`.
    - `get_metrics(key) -> dict | pd.DataFrame` — fetches the serialized `quality_metrics` NWB object and returns metrics keyed by unit, matching v1's practical notebook usage.
    - `get_labels(key) -> dict[int, list[str]]` — fetches proposed labels from the `proposed_labels` object.
    - `get_merge_groups(key) -> list[list[int]]` — fetches proposed merge groups from the `merge_suggestions` object.
    - `materialize_curation(...)` remains the explicit v2 analog of `CurationV1.insert_metric_curation`; document this name in user docs and examples so users do not have to discover it in `designs.md`.

- **Port the v1 `BurstPair` notebook helpers** into `AnalyzerCuration` methods: selection insertion by curation, correlogram plots, pair cross-correlation, pair peak inspection, and peak-over-time plots. These read directly from the SortingAnalyzer's `correlograms` extension; no separate `BurstPair`-equivalent table.

- **Three-bug-class invariant on `_apply_label_rules`** addressing [#1513](https://github.com/LorenFrankLab/spyglass/issues/1513). v2's `_apply_label_rules` must avoid all three patterns AND test each one. Each `AutoCurationRules.Rule` row emits a *scalar* label; the per-unit accumulator is `dict[unit_id, list[str]]`:
  1. **Loop completion (Bug A in #1513)**: every label rule must be processed; `return` must be at function scope, outside the rule loop. A return inside the rule loop would silently drop later rules. Validation should synthesize 3 rules where rules 2 and 3 would each add a distinct label to the same unit; assert the unit ends with all three labels.
  2. **Per-unit list isolation (Bug B in #1513)**: each unit's label list MUST be an independent object. Shared list objects would let one unit's later labels contaminate another unit and the rule definition. v2 uses `defaultdict(list)`, which constructs a fresh list per new key — `.append(rule["label"])` mutates only that unit's list. Validation should use two units flagged by rule 1 (label `"noise"`) and rule 2 flagging only unit A (label `"mua"`); assert `labels[A] == ["noise", "mua"]` AND `labels[B] == ["noise"]` (B is not contaminated by A's later append).
  3. **Per-rule membership check (Bug C in #1513)**: before appending a rule's label to a unit's list, the check must be `if rule["label"] not in labels[unit_id]` — element-against-list. A list-against-list membership check would allow duplicate labels. Validation should use two rules that both yield `"noise"` for the same unit; assert final labels `== ["noise"]`, not `["noise", "noise"]`.

- **NaN sanitization for metric serialization** (addresses [#1556](https://github.com/LorenFrankLab/spyglass/issues/1556)): SI's `compute_quality_metrics` legitimately returns `nan` for low-spike units. `AnalyzerCuration.make()` writes metrics to AnalysisNWB table objects (`quality_metrics`, `merge_suggestions`, `proposed_labels`); Phase 5 later serializes the same metric table into the FigPack URI. Every serialized path must see non-finite values coerced to `None` BEFORE serialization. Add `_sanitize_for_json(df) -> df` helper that copies the DataFrame and replaces all non-finite values with `None`. The in-memory `metrics_df` retains NaN for downstream consumers that want to filter on it; only the serialized copies get sanitized. Per the [Empty / NaN / Boundary Invariants contract](shared-contracts.md#empty--nan--boundary-invariants).

- **Implement `_apply_label_rules(metrics_df, rule_rows)` helper** in `metric_curation.py`. Binding behavior: rows are processed by ascending `rule_index`; all matching labels are retained; duplicate labels for the same unit are suppressed; missing metric names raise a clear validation error before serialization. Keep the method body simple and covered by the three invariant tests above.

- **Add a `Sorting`-side method** (in [src/spyglass/spikesorting/v2/sorting.py](../../../../src/spyglass/spikesorting/v2/sorting.py) from Phase 1, extended here): `Sorting.add_extensions(key, extensions: list[str], **kwargs)`. This is a convenience for callers (including `AnalyzerCuration.make()`) to add extensions to an already-built analyzer in place. Internally: `analyzer = self.get_analyzer(key)`, resolve job kwargs from this sort's `SorterParameters` row per [shared-contracts.md § Job-Kwargs Resolution](shared-contracts.md#job-kwargs-resolution) (`sorter_job_kwargs = (SorterParameters & (SortingSelection & key)).fetch1("job_kwargs")`), then `analyzer.compute(extensions, **(_resolved_job_kwargs(sorter_job_kwargs) | kwargs))` so explicit caller `kwargs` still win. Documented as idempotent — SI's `compute()` skips already-computed extensions unless `overwrite=True`.

- **Implement the v2 clusterless `spike_location` decoding mark** (closes review item **F3** in [REVIEW-REPORT.md](REVIEW-REPORT.md) and the dead-code gap in `decoding/v1/waveform_features.py`). The v2 `SortingAnalyzer` dispatch added in Phase 1b serves only `amplitude` and `full_waveform` for v2 (`CurationV2`) sources; `spike_location` is currently rejected with an explicit `NotImplementedError` ([src/spyglass/decoding/v1/waveform_features.py](../../../../src/spyglass/decoding/v1/waveform_features.py)). `spike_location` is an optional clusterless mark (the default clusterless mark is `amplitude`), so this is a feature-completeness/parity item, not a decoding blocker. Implement it on the SortingAnalyzer path:
  - Compute the `spike_locations` extension on the v2 analyzer (depends on `random_spikes`, already computed at sort time per [appendix.md § SortingAnalyzer extension dependencies](appendix.md#sortinganalyzer-extension-dependencies)) and surface per-unit spike locations through the existing `_AnalyzerWaveformAccessor` adapter, so `spike_location` resolves for v2 sources the same way `amplitude` / `full_waveform` do. Use SI 0.104's analyzer/extension API (`SortingAnalyzer.compute("spike_locations", ...)`), NOT the removed `WaveformExtractor` form.
  - Retire the now-dead legacy module-level `_get_spike_locations(waveform_extractor, ...)` helper — delete it or replace its body with an explicit `NotImplementedError`, and route the legacy `spike_location` feature through the new analyzer-based helper. It is broken under SI 0.104 regardless of v2: `import spikeinterface as si` does not bind `si.postprocessing` (so `si.postprocessing.compute_spike_locations` raises `AttributeError` unless an unrelated import side-effect happens to load the submodule), and 0.104's `compute_spike_locations(sorting_analyzer, ...)` takes a `SortingAnalyzer`, not the `WaveformExtractor` the helper passes. It is reachable only on the legacy SI<0.101 guarded path.
  - Test `test_v2_clusterless_spike_location_feature` (slow, integration): populate `UnitWaveformFeatures` with a `waveform_features_params` set that includes `spike_location` against a v2 `CurationV2` source; assert the written feature is `(n_spikes, 2 or 3)` per unit and that no `NotImplementedError` is raised.

- **Parity test against v1 MetricCuration baseline**. New test `test_v2_analyzer_curation_vs_v1` (slow, integration): on the Phase 0 baseline-captured sort, load the legacy-environment v1 `MetricCuration` pickle and compute v2 `AnalyzerCuration` under SI 0.104. Do not invoke active v1 MetricCuration in the SI 0.104 test unless Phase 0c explicitly ported that surface. Compare the per-unit `snr`, `isi_violation`, `firing_rate`, `num_spikes` columns. Tolerances:
  - `snr`: compare as a distributional/diagnostic parity check, not a strict per-unit equivalence. v1 and SI 0.104 use different SNR definitions (mean-based vs median-based per the appendix), so the test reports per-unit ratios and asserts an aggregate bound after Phase 0/2 baseline calibration (initial guardrail: median ratio within 30% and no unexplained order-of-magnitude outliers). Tighten or relax only with recorded baseline evidence; do not hide a systematic definition change behind a silent per-unit tolerance.
  - `isi_violation`: exact — but **only if v2 computes Spyglass's `count/(n_spikes−1)` fraction** (see the SI metric-name correction above), not SI's `isi_violations_ratio`. The violation count is deterministic given sorted spike times; the ratio is a different (model-estimator) number and would not match v1.
  - `firing_rate`: exact within float-rounding (spike count / duration).
  - `num_spikes`: exact integer match.
  Any metric outside tolerance fails the test with a per-unit diff report.

- **Port v1's `RecordingRecompute` pattern to v2-specific recompute tables** in `recompute.py`. v1's `recompute.py` ([src/spyglass/spikesorting/v1/recompute.py:1-15, 55-103, 189-224, 527-548](../../../../src/spyglass/spikesorting/v1/recompute.py#L1-L548)) is active production infrastructure. v2 keeps recompute in Phase 2, not a future phase, because chronic recordings make deliberate storage reclamation part of the MVP reliability story. Phase 2 adds final-shape recompute tables declared in the v2 draft schema:
  - **`RecordingArtifactVersions`** (Computed, FK `Recording`) — inventories PyNWB/namespace dependencies and records the current `Recording.cache_hash`.
  - **`RecordingArtifactRecomputeSelection`** (Manual, FK `RecordingArtifactVersions` + `UserEnvironment`) — plans a recompute attempt under a labeled `env_id` with a `rounding` precision and optional `xfail_reason`.
  - **`RecordingArtifactRecompute`** (Computed, FK `RecordingArtifactRecomputeSelection`) — reruns the recording reconstruction, compares hashes/object names, and records `matched`, `err_msg`, `created_at`, and `deleted`. `delete_files()` is gated on `matched=1` rows **whose `env_id` matches the CURRENT `UserEnvironment`** — stale-env matched rows (e.g. a SI-0.103-era recompute) do NOT authorize deletion. The default raises `StaleEnvMatchedError` naming the stale envs; pass `force_stale_env=True` to override (audit-logged). See [designs.md § RecordingArtifactRecompute + SortingAnalyzerRecompute](designs.md#recordingartifactrecompute--sortinganalyzerrecompute). Same rule applies to `SortingAnalyzerRecompute.delete_files()`.
  - **`SortingAnalyzerVersions` / `SortingAnalyzerRecomputeSelection` / `SortingAnalyzerRecompute`** — parallel trio for the SortingAnalyzer folder. The comparison inventories analyzer extension metadata and content hashes rather than NWB ElectricalSeries objects.
  - **Storage-reclamation workflow** documented in `docs/src/Features/SpikeSortingV2StorageManagement.md`: recompute-verify (`matched=1`) → `delete_files()` reclaims disk → later `Recording.get_recording()` or `Sorting.get_analyzer()` rebuilds from the stored selection/parameter lineage. **Interaction with `Recording.repair()` (review-fix C2):** rebuild-on-access is now fail-closed on cache-hash drift — a drifted artifact is NOT silently rebuilt; `get_recording` raises `RecordingCacheDriftError`. `Recording.repair()` is the sanctioned path that recomputes AND updates `cache_hash`. The recompute tables compare against the **stored** `cache_hash`, so a recompute attempt against a recording whose hash legitimately changed (e.g. an SI version bump) must either run after `Recording.repair()` updated the row, or record `matched=0`. Document that `repair()` supersedes the stored hash and a post-repair recompute is expected to match; do not treat a `repair()`-induced hash change as a recompute failure.
  - This is Phase 2 schema, not a later migration. The zero-migration contract lists these tables as Phase 2 pure additions.

- **Preserve the v1 recompute admin surface unless explicitly rejected in code review.** The Phase 2 implementation must name the v2 equivalents for the following v1 operations:
  - `RecordingRecomputeSelection.attempt_all(...)` → bulk-create recompute attempts for eligible rows.
  - `RecordingRecomputeSelection.remove_matched(...)` → remove or mark matched attempts after verified cleanup.
  - `RecordingRecompute.with_names` → relation joining comparison rows to missing/different object names for review.
  - `RecordingRecompute.get_parent_key(...)` → recover the upstream `Recording` / `Sorting` key from a recompute row.
  - `RecordingRecompute.recheck(...)` → rerun comparison for a specific row after environment or file changes.
  - `RecordingRecompute.get_disk_space(...)` → report reclaimable disk usage before deletion.
  - `RecordingRecompute.update_secondary(...)` → backfill secondary summary fields when comparison details change.
  If any method is intentionally dropped, add it to `feature-parity.md` explicit non-parity with the reason and update the storage-management docs.

- **Documentation update**:
  - Update [docs/src/Features/SpikeSortingV2.md](../../../../docs/src/Features/SpikeSortingV2.md) (created Phase 1) with a Quality Metrics section.
  - Add `docs/src/Features/SpikeSortingV2StorageManagement.md` documenting recompute verification and safe deletion.
  - Add CHANGELOG entry noting `AnalyzerCuration` consolidation and the v2 recompute tables.

## Deliberately not in this phase

- **No port of `BurstPair` schema** — its work folds into `AnalyzerCuration` auto-merge presets. The v1 `BurstPair` table stays in v1 for legacy data.
- **No removal of v1 `MetricCuration` / `BurstPair`**. Both stay in v1 indefinitely (overview Open Question #3).
- **No multi-session metrics.** Per-session only. Phase 4 handles cross-session.
- **No web-based curation UI.** Phase 5 (FigPack).
- **No automatic materialization.** `AnalyzerCuration.populate()` produces suggestions; user must call `materialize_curation()` to create the next CurationV2 row.

## Validation goals

Behaviors the Phase 2 validation goals must cover. Each goal must have at least one assertion exercising it; the implementer chooses test names and splits.

1. **Pydantic params validation**: `QualityMetricParameters` and `AutoCurationRules.insert_rules(row, rule_rows)` reject bogus metric names / preset names / operators at insert; rule order is stored as part rows and is queryable by `metric_name`, `label`, and `rule_index`. Direct rule-part inserts are unsupported and the integrity check flags malformed or missing rule rows.
2. **Label-rule correctness — three bug-class invariants (#1513)**: every rule is processed (loop-completion); per-unit label lists are independent objects (no shared-list aliasing); duplicate labels within a unit are suppressed via element-in-list dedupe. One test per invariant.
3. **AnalyzerCuration end-to-end** (slow): `make()` writes the three NWB tables; `get_waveforms`, `get_metrics`, `get_labels`, `get_merge_groups` round-trip the written content; populate on the Phase 0 polymer fixture completes within a documented loose smoke budget for the target machine class.
4. **NaN sanitization**: a low-spike unit produces non-finite metrics; the serialized AnalysisNWB metric table shows `None`; the in-memory metrics DataFrame keeps NaN.
5. **Zero-unit Sorting**: `AnalyzerCuration` populates cleanly with empty metric/merge/label tables and no missing-column errors. The path is exercised through the review-fix C5 contract: `Sorting.get_analyzer()` on the zero-unit sort raises the clear zero-unit error / returns the documented sentinel (not a phantom folder path), and `AnalyzerCuration.make()` handles that signal explicitly rather than crashing on a failed analyzer load.
6. **Auto-merge suggestion sanity** (slow): a planted duplicate-unit fixture (~1 ms offset) yields a merge suggestion under `similarity_correlograms`.
7. **`materialize_curation()` produces a child**: child `CurationV2` has `parent_curation_id` set, `curation_source == 'analyzer_curation'`, auto-registers in `SpikeSortingOutput.CurationV2`. Recursive auto-curation logs a warning but does not raise.
8. **BurstPair visualization helpers** (slow): the ported correlogram, pair-inspection, and peak-over-time helpers render against a SortingAnalyzer-backed sort.
9. **`Sorting.add_extensions` is idempotent** (slow): second call is a no-op.
10. **v1 parity** (slow, integration): metrics from `AnalyzerCuration` match v1 `MetricCuration` per documented tolerances (SNR is aggregate/distributional because the SI definition changed; isi_violation/firing_rate/num_spikes exact).
11. **v2 clusterless `spike_location` feature** (slow, integration): `UnitWaveformFeatures` populates `spike_location` against a v2 `CurationV2` source without `NotImplementedError`; the written feature is `(n_spikes, 2 or 3)` per unit. The legacy `_get_spike_locations` helper is removed or explicitly guarded (no `si.postprocessing` attribute access on the `import spikeinterface as si` alias).

**Recompute validation (separate slice, same phase)**: version inventory populates correctly; recompute under current env yields `matched=1`; rounding/hash mismatch yields `matched=0` plus `Name`/`Hash` part rows; `delete_files()` refuses `matched=0`; stale-env `matched=1` raises `StaleEnvMatchedError`; current-env `matched=1` proceeds only inside a temporary test `SPYGLASS_BASE_DIR`; `force_stale_env=True` succeeds and writes an audit log entry. Tests prefer dry-run/preview modes unless actual deletion is the behavior under test, and all destructive-path tests follow the shared [Destructive-test guardrail](shared-contracts.md#environment-and-database-safety) for issue #1573. v1 admin-surface parity (`attempt_all`, `remove_matched`, `with_names`, `get_parent_key`, `recheck`, `get_disk_space`, `update_secondary`) is preserved or explicitly listed in `feature-parity.md`.

## Commands to run

```bash
source .venv-spikesorting-v2/bin/activate
export SPYGLASS_SKILL_DIR="${SPYGLASS_SKILL_DIR:-../spyglass-skill/skills/spyglass}"
test -f "$SPYGLASS_SKILL_DIR/scripts/code_graph.py"

pytest tests/spikesorting/v2/test_analyzer_curation.py -q
pytest tests/spikesorting/v2/test_recompute.py -q

python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe AnalyzerCurationSelection --file spyglass/spikesorting/v2/metric_curation.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe AnalyzerCuration --file spyglass/spikesorting/v2/metric_curation.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe RecordingArtifactRecompute --file spyglass/spikesorting/v2/recompute.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe SortingAnalyzerRecompute --file spyglass/spikesorting/v2/recompute.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src path --up AnalyzerCuration --file spyglass/spikesorting/v2/metric_curation.py --json
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src path --down AnalyzerCuration --file spyglass/spikesorting/v2/metric_curation.py --json

git diff --check -- src/spyglass/spikesorting/v2 tests/spikesorting/v2 docs/src/Features CHANGELOG.md
git diff --exit-code -- src/spyglass/spikesorting/v1
```

## Fixtures

- **`baseline_metric_curation`** — pickle of the v1 `MetricCuration` output captured by the prerequisite baseline step above.
- **`synthetic_duplicate_units_sort`** (new in conftest) — a synthetic SI sort with two units that have ~95% overlapping spike times (offset by 1 ms). Used by the auto-merge suggestion test.

## Review

Before opening or reviewing the implementation PR that contains this checkpoint, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation goals are covered; slow / integration tests are marked.
- Tests aren't trivial — the parity test loads the v1 baseline pickle and asserts per-metric tolerances rather than wrapping `compute_quality_metrics` and asserting it equals itself.
- Docstrings, test names, and module names don't reference this plan, phase numbers, or files inside `.claude/docs/plans/`.
- `Sorting.add_extensions()` works idempotently (SI's overwrite=False semantics respected).
- `BurstPair` visualization helpers are ported into `AnalyzerCuration` (and verified to render against a SortingAnalyzer correlograms extension).
- `AnalyzerCuration` fetch helpers (`get_waveforms`, `get_metrics`, `get_labels`, `get_merge_groups`) and `materialize_curation()` are documented and tested.
- v2 recompute admin methods are present or explicitly listed as non-parity in `feature-parity.md`.
- v1 `MetricCuration` and `BurstPair` are NOT touched (sanity check via `git diff src/spyglass/spikesorting/v1/`).
- `code_graph.py describe` returns clean output for every new table; `path --up`/`path --down` chains match the design DAG; JSON warnings are empty or explicitly accounted for in `precondition-check.md`.
- CHANGELOG.md updated.
