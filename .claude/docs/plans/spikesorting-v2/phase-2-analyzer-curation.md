# Phase 2 — Analyzer-driven curation + recompute verification

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#analyzercuration-replaces-v1-metriccuration--burstpair)

Replaces v1's `MetricCuration` + `BurstPair` with a single `AnalyzerCuration` table that walks SortingAnalyzer extensions to compute metrics, suggest merges, and produce auto-curation labels. Materializing the suggested labels/merges into a new `CurationV2` row is an explicit user action. This phase also ports v1's recompute verification pattern to v2 `Recording` and `Sorting` artifacts so large caches can be deleted only after a successful round-trip check.

## Executor Checklist

- Implement metric/auto-curation parameter schemas and default rows.
- Implement `AnalyzerCurationSelection`, `AnalyzerCuration`, `materialize_curation()`, and the v1 notebook-facing fetch/promote helpers.
- Port the BurstPair visualization workflow into `AnalyzerCuration` methods without adding a separate BurstPair table.
- Implement recording/analyzer recompute verification tables and safe deletion gates.
- Preserve NaN sanitization, empty-unit, recursive-auto-curation, and label-rule invariants from `shared-contracts.md`.
- Run the Phase 2 validation slice plus `code_graph.py describe/path` for new tables.

**Inputs to read first:**

- [src/spyglass/spikesorting/v1/metric_curation.py](../../../../src/spyglass/spikesorting/v1/metric_curation.py) — entire file; v2 consolidates this.
- [src/spyglass/spikesorting/v1/burst_curation.py](../../../../src/spyglass/spikesorting/v1/burst_curation.py) — entire file; burst-pair logic moves into auto-merge presets.
- [src/spyglass/spikesorting/v1/metric_utils.py](../../../../src/spyglass/spikesorting/v1/metric_utils.py) — metric helper functions; mostly obsoleted by SI 0.104's `compute_quality_metrics`.
- [src/spyglass/spikesorting/v2/sorting.py](../../../../src/spyglass/spikesorting/v2/sorting.py) (from Phase 1) — `Sorting.get_analyzer(key)` is the upstream input.
- [.claude/docs/plans/spikesorting-v2/appendix.md § Quality metric renames in 0.104](appendix.md#quality-metric-renames-in-0104) — what metric names changed.
- [.claude/docs/plans/spikesorting-v2/appendix.md § SortingAnalyzer extension dependencies](appendix.md#sortinganalyzer-extension-dependencies) — what to compute before metrics.

**Contracts referenced:**

- [SortingAnalyzer Storage Layout](shared-contracts.md#sortinganalyzer-storage-layout) — adds extensions to an existing analyzer in place (the on-disk folder grows; this is supported by SI binary_folder format).
- [Pydantic Parameter Schema Convention](shared-contracts.md#pydantic-parameter-schema-convention) — `QualityMetricParameters`, `AutoCurationRules` get Pydantic models.
- [Job-Kwargs Resolution](shared-contracts.md#job-kwargs-resolution) — extension compute uses resolved kwargs.

**Designs referenced:** [AnalyzerCuration (replaces v1 MetricCuration + BurstPair)](designs.md#analyzercuration-replaces-v1-metriccuration--burstpair).

## Tasks

- **Implement `_params/metric_curation.py`** with Pydantic models:
  - `QualityMetricParamsSchema` — `metric_names: list[str]`, `metric_kwargs: dict[str, dict]`, `skip_pc_metrics: bool = True`. The stored `metric_kwargs` dict is passed to SpikeInterface as the `metric_params=` argument. Validate each metric name against SI 0.104's exported list using `from spikeinterface.qualitymetrics import get_quality_metric_list` when available; otherwise hardcode the v2-supported set with a comment naming the SI version used to generate it.
  - `AutoCurationRulesSchema` — `label_rules: dict[str, dict]` where each value is `{"operator": ">", "threshold": 0.5, "label": "noise"}`; `auto_merge_preset: Literal["similarity_correlograms", "temporal_splits", "x_contaminations", "feature_neighbors", "none"]`; `auto_merge_kwargs: dict`.

- **Implement `metric_curation.py`** per [designs.md § AnalyzerCuration](designs.md#analyzercuration-replaces-v1-metriccuration--burstpair). Specific:
  - `QualityMetricParameters` Lookup with three default rows: `("franklab_default", ...)`, `("neuropixels_default", ...)`, `("minimal", {"metric_names": ["snr", "isi_violation", "firing_rate"], ...})`.
  - `QualityMetricParameters.show_available_metrics()` or a documented `available_quality_metrics()` helper that lists the supported SortingAnalyzer/SpikeInterface metric names. This preserves the v1 notebook-discovery workflow from `MetricParameters.show_available_metrics()` without requiring v1's exact custom metric set.
  - `AutoCurationRules` Lookup with default rows including a `("franklab_default_thresholds", ...)` that mirrors v1's auto-label conventions (snr < 2 → noise, isi_violation > 0.05 → mua, etc.; pull thresholds from `MetricCurationParameters` defaults at [src/spyglass/spikesorting/v1/metric_curation.py:161-191](../../../../src/spyglass/spikesorting/v1/metric_curation.py#L161-L191)).
  - `AnalyzerCurationSelection` Manual. **`insert_selection()` rejects recursive auto-curation by default**: if the upstream `CurationV2.metrics_source == 'analyzer_curation'`, raise `RecursiveAutoCurationError` pointing at `allow_recursive=True`. Computing quality metrics over post-merge templates from a materialized child is rarely what the user wants; explicit opt-in keeps the failure mode visible. See [designs.md § AnalyzerCuration](designs.md#analyzercuration-replaces-v1-metriccuration--burstpair) for the helper signature.
  - `AnalyzerCuration` Computed with `make()` that computes additional extensions, runs `compute_quality_metrics(..., metric_params=...)`, applies label rules, runs `compute_merge_unit_groups`, writes three NWB tables (`quality_metrics`, `merge_suggestions`, `proposed_labels`) via `AnalysisNwbfile().build()`.
  - Auto-merge extension dependency is explicit: before calling `compute_merge_unit_groups`, ensure `template_similarity` is computed in addition to `correlograms` and the Phase 1 core extensions. Pass `compute_needed_extensions=False` when supported so the Spyglass-computed extension set is the audited one rather than an implicit SI default.
  - `materialize_curation(key, description="auto-curation") -> dict` — creates a child `CurationV2` row with the proposed labels + merge groups; returns the new curation_key. Auto-registers via the Phase 1 `CurationV2.insert_curation` flow.
  - **Fetch/promote parity with v1 MetricCuration** — implement the v1 notebook-facing helpers on `AnalyzerCuration`:
    - `get_waveforms(key, fetch_all=False)` — returns the SortingAnalyzer waveforms extension or a narrow compatibility object with `get_unit_waveforms(unit_id)` / `get_waveforms(unit_id)` behavior needed by downstream helpers. This replaces v1 `MetricCuration.get_waveforms`.
    - `get_metrics(key) -> dict | pd.DataFrame` — fetches the serialized `quality_metrics` NWB object and returns metrics keyed by unit, matching v1's practical notebook usage.
    - `get_labels(key) -> dict[int, list[str]]` — fetches proposed labels from the `proposed_labels` object.
    - `get_merge_groups(key) -> list[list[int]]` — fetches proposed merge groups from the `merge_suggestions` object.
    - `materialize_curation(...)` remains the explicit v2 analog of `CurationV1.insert_metric_curation`; document this name in user docs and examples so users do not have to discover it in `designs.md`.

- **Port `BurstPair` visualization helpers** into `AnalyzerCuration` methods:
  - `insert_by_curation_id(curation_id_or_key, auto_curation_rules_name="...", metric_params_name="...")` or a clearly named equivalent — convenience helper that inserts the `AnalyzerCurationSelection` row for an existing `CurationV2` row. This preserves the v1 `BurstPairSelection.insert_by_curation_id` workflow even though there is no separate `BurstPairSelection` table.
  - `plot_correlograms_by_sort_group(key)` — adapts `BurstPair.plot_by_sort_group_ids` at [src/spyglass/spikesorting/v1/burst_curation.py](../../../../src/spyglass/spikesorting/v1/burst_curation.py).
  - `investigate_pair_xcorrel(key, unit_a, unit_b)`.
  - `investigate_pair_peaks(key, unit_a, unit_b)`.
  - `plot_peak_over_time(key, unit_a, unit_b, overlap=True)`.
  - These read directly from the SortingAnalyzer's `correlograms` extension; no separate `BurstPair`-equivalent table.

- **Three-bug-class invariant on `_apply_label_rules`** addressing [#1513](https://github.com/LorenFrankLab/spyglass/issues/1513). v2's `_apply_label_rules` must avoid all three patterns AND test each one. Each rule emits a *scalar* label (matching `AutoCurationRulesSchema` above); the per-unit accumulator is `dict[unit_id, list[str]]`:
  1. **Loop completion (Bug A in #1513)**: every label rule must be processed; `return` must be at function scope, outside the rule loop. A return inside the rule loop would silently drop later rules. Test: `test_label_rules_loop_completes_all_rules` — synthesize 3 rules where rules 2 and 3 would each add a distinct label to the same unit; assert the unit ends with all three labels.
  2. **Per-unit list isolation (Bug B in #1513)**: each unit's label list MUST be an independent object. Shared list objects would let one unit's later labels contaminate another unit and the rule definition. v2 uses `defaultdict(list)`, which constructs a fresh list per new key — `.append(rule["label"])` mutates only that unit's list. Test: `test_label_list_isolation` — two units flagged by rule 1 (label `"noise"`); rule 2 flags only unit A (label `"mua"`). Assert `labels[A] == ["noise", "mua"]` AND `labels[B] == ["noise"]` (B is not contaminated by A's later append).
  3. **Per-rule membership check (Bug C in #1513)**: before appending a rule's label to a unit's list, the check must be `if rule["label"] not in labels[unit_id]` — element-against-list. A list-against-list membership check would allow duplicate labels. Test: `test_label_dedupe_per_element` — two rules both yield `"noise"` for the same unit; assert final labels `== ["noise"]`, not `["noise", "noise"]`.

- **NaN sanitization for metric serialization** (addresses [#1556](https://github.com/LorenFrankLab/spyglass/issues/1556)): SI's `compute_quality_metrics` legitimately returns `nan` for low-spike units. `AnalyzerCuration.make()` writes metrics to AnalysisNWB table objects (`quality_metrics`, `merge_suggestions`, `proposed_labels`); Phase 5 later serializes the same metric table into the FigPack URI. Every serialized path must see non-finite values coerced to `None` BEFORE serialization. Add `_sanitize_for_json(df) -> df` helper that copies the DataFrame and replaces all non-finite values with `None`. The in-memory `metrics_df` retains NaN for downstream consumers that want to filter on it; only the serialized copies get sanitized. Per the [Empty / NaN / Boundary Invariants contract](shared-contracts.md#empty--nan--boundary-invariants).

- **Implement `_apply_label_rules(metrics_df, label_rules)` helper** in `metric_curation.py`. Binding behavior: rule order is preserved in output label order; all matching labels are retained; duplicate labels for the same unit are suppressed; missing metric names raise a clear validation error before serialization. Keep the method body simple and covered by the three invariant tests above.

- **Add a `Sorting`-side method** (in [src/spyglass/spikesorting/v2/sorting.py](../../../../src/spyglass/spikesorting/v2/sorting.py) from Phase 1, extended here): `Sorting.add_extensions(key, extensions: list[str], **kwargs)`. This is a convenience for callers (including `AnalyzerCuration.make()`) to add extensions to an already-built analyzer in place. Internally: `analyzer = self.get_analyzer(key); analyzer.compute(extensions, **_resolved_job_kwargs(key) | kwargs)`. Documented as idempotent — SI's `compute()` skips already-computed extensions unless `overwrite=True`.

- **Parity test against v1 MetricCuration**. New test `test_v2_analyzer_curation_vs_v1` (slow, integration): on the Phase 0 baseline-captured sort, compute v1 `MetricCuration` (via the existing v1 code path) and v2 `AnalyzerCuration`. Compare the per-unit `snr`, `isi_violation`, `firing_rate`, `num_spikes` columns. Tolerances:
  - `snr`: ±30% (v1 uses mean, v2 uses median — they differ systematically).
  - `isi_violation`: exact (refractory-period violation count is deterministic given sorted spike times).
  - `firing_rate`: exact within float-rounding (spike count / duration).
  - `num_spikes`: exact integer match.
  Any metric outside tolerance fails the test with a per-unit diff report.

- **Port v1's `RecordingRecompute` pattern to v2-specific recompute tables** in `recompute.py`. v1's `recompute.py` ([src/spyglass/spikesorting/v1/recompute.py:1-15, 55-103, 189-224, 527-548](../../../../src/spyglass/spikesorting/v1/recompute.py#L1-L548)) is active production infrastructure. v2 keeps recompute in Phase 2, not a future phase, because chronic recordings make deliberate storage reclamation part of the MVP reliability story. Phase 2 adds final-shape recompute tables declared in the v2 draft schema:
  - **`RecordingArtifactVersions`** (Computed, FK `Recording`) — inventories PyNWB/namespace dependencies and records the current `Recording.cache_hash`.
  - **`RecordingArtifactRecomputeSelection`** (Manual, FK `RecordingArtifactVersions` + `UserEnvironment`) — plans a recompute attempt under a labeled `env_id` with a `rounding` precision and optional `xfail_reason`.
  - **`RecordingArtifactRecompute`** (Computed, FK `RecordingArtifactRecomputeSelection`) — reruns the recording reconstruction, compares hashes/object names, and records `matched`, `err_msg`, `created_at`, and `deleted`. `delete_files()` is gated on `matched=1` rows **whose `env_id` matches the CURRENT `UserEnvironment`** — stale-env matched rows (e.g. a SI-0.103-era recompute) do NOT authorize deletion. The default raises `StaleEnvMatchedError` naming the stale envs; pass `force_stale_env=True` to override (audit-logged). See [designs.md § RecordingArtifactRecompute + SortingAnalyzerRecompute](designs.md#recordingartifactrecompute--sortinganalyzerrecompute). Same rule applies to `SortingAnalyzerRecompute.delete_files()`.
  - **`SortingAnalyzerVersions` / `SortingAnalyzerRecomputeSelection` / `SortingAnalyzerRecompute`** — parallel trio for the SortingAnalyzer folder. The comparison inventories analyzer extension metadata and content hashes rather than NWB ElectricalSeries objects.
  - **Storage-reclamation workflow** documented in `docs/src/Features/SpikeSortingV2StorageManagement.md`: recompute-verify (`matched=1`) → `delete_files()` reclaims disk → later `Recording.get_recording()` or `Sorting.get_analyzer()` rebuilds from the stored selection/parameter lineage.
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
  - CHANGELOG.md "Unreleased": "v2 `AnalyzerCuration` consolidates v1's `MetricCuration` + `BurstPair`. Auto-curation rules driven by Pydantic-validated thresholds; auto-merge via SI 0.104 presets. v2 recompute tables support safe storage reclamation for Recording and SortingAnalyzer artifacts."

## Deliberately not in this phase

- **No port of `BurstPair` schema** — its work folds into `AnalyzerCuration` auto-merge presets. The v1 `BurstPair` table stays in v1 for legacy data.
- **No removal of v1 `MetricCuration` / `BurstPair`**. Both stay in v1 indefinitely (overview Open Question #3).
- **No multi-session metrics.** Per-session only. Phase 4 handles cross-session.
- **No web-based curation UI.** Phase 5 (FigPack).
- **No automatic materialization.** `AnalyzerCuration.populate()` produces suggestions; user must call `materialize_curation()` to create the next CurationV2 row.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_quality_metric_params_validation` | `QualityMetricParameters().insert1({"params": {"metric_names": ["bogus_metric"]}})` raises validation error; valid metric list inserts. |
| `test_quality_metric_available_metrics_helper` | Metric-discovery helper returns or logs all metric names accepted by the validator, including the default `minimal` metrics. |
| `test_auto_curation_rules_validation` | `AutoCurationRules().insert1({...auto_merge_preset: "bogus_preset"...})` raises. Valid preset inserts. |
| `test_apply_label_rules_basic` | Synthetic metrics DataFrame with two rules ("snr<2 → noise", "isi_violation>0.05 → mua"); asserts correct labels per unit. |
| `test_apply_label_rules_order_preserved` | Rules in order produce expected multi-label lists when a unit matches multiple thresholds; reordering rules changes label order only, not label membership. |
| `test_label_rules_loop_completes_all_rules` | Three rules where rules 2 and 3 both apply to a unit; asserts all matching rules are processed and `return` is outside the rule loop. |
| `test_label_list_isolation` | Two units match the same multi-label rule; mutating unit A's labels after helper return does not change unit B's labels or the rule definition. |
| `test_label_dedupe_per_element` | Two rules emit the same label for one unit; final label list contains one copy, not duplicates from list-in-list membership checks. |
| `test_analyzer_curation_make_writes_three_tables` (slow) | After populate, `AnalyzerCuration & key` row has all three object_ids populated; `fetch_nwb()` returns three pandas DataFrames. |
| `test_analyzer_curation_metrics_match_si_compute` (slow) | Independently call `compute_quality_metrics(analyzer, ...)` and compare to the fetched `quality_metrics` DataFrame — exact match. |
| `test_metric_nan_round_trip` | Low-spike unit produces non-finite metrics; the serialized AnalysisNWB metric table contains `None` while the in-memory metrics DataFrame preserves NaN semantics. |
| `test_analyzer_curation_zero_unit_sorting` | Zero-unit `Sorting` row populates `AnalyzerCuration` with empty metric/merge/label tables and no missing-column errors. |
| `test_analyzer_curation_merge_suggestions_non_empty_on_obvious_split` (slow) | Synthetic sort with two units that are obvious duplicates (same spike times shifted by 1 ms); auto-merge with preset='similarity_correlograms' suggests their merge. |
| `test_analyzer_curation_fetch_helpers` | `get_waveforms`, `get_metrics`, `get_labels`, and `get_merge_groups` return the data written by `AnalyzerCuration.make()` and match v1 notebook-facing shapes where practical. |
| `test_analyzer_curation_burstpair_visualization_helpers` (slow) | `plot_correlograms_by_sort_group`, `investigate_pair_xcorrel`, `investigate_pair_peaks`, and `plot_peak_over_time` all render against a SortingAnalyzer-backed sort without requiring the v1 `BurstPair` table. |
| `test_analyzer_curation_insert_by_curation_id_helper` | Helper inserts an `AnalyzerCurationSelection` row from an existing curation key and returns the selection key; repeated calls are idempotent. |
| `test_materialize_curation_creates_child` | `AnalyzerCuration.materialize_curation(key)` creates a new `CurationV2` row with `parent_curation_id` pointing to the input curation; auto-registers in `SpikeSortingOutput.CurationV2`. The child's `metrics_source == 'analyzer_curation'`. |
| `test_analyzer_curation_rejects_recursive_by_default` | After `materialize_curation()` produces a child CurationV2 (with `metrics_source='analyzer_curation'`), call `AnalyzerCurationSelection.insert_selection({sorting_id, curation_id: <child>})`. Raises `RecursiveAutoCurationError`; message names `allow_recursive=True`. |
| `test_analyzer_curation_allow_recursive_override` | Same fixture with `allow_recursive=True`. Insert succeeds and `AnalyzerCuration.populate()` produces a new analyzer-curation row keyed off the child. Regression guard for the explicit-opt-in path. |
| `test_add_extensions_is_idempotent` (slow) | Call `Sorting.add_extensions(key, ["correlograms"])` twice; second call is a no-op (no recompute). |
| `test_v2_analyzer_curation_vs_v1` (slow, integration) | Parity vs v1 `MetricCuration` per tolerances above. Reports per-unit diffs on failure. |
| `test_recompute_versions_inventories_correctly` | `RecordingArtifactVersions` and `SortingAnalyzerVersions` populate for existing v2 `Recording` / `Sorting` rows and record dependency/hash manifests. |
| `test_recompute_matches_under_same_env` (slow) | Recompute under the current `UserEnvironment`; `RecordingArtifactRecompute.matched == 1` and `SortingAnalyzerRecompute.matched == 1`. |
| `test_recompute_detects_mismatch_under_different_rounding` (slow) | Force a rounding/hash mismatch; recompute rows record `matched == 0` plus `Name` or `Hash` part rows describing the difference. |
| `test_delete_files_only_for_matched` | `delete_files()` refuses to delete artifacts for `matched=0` and deletes only after a successful round-trip (`matched=1`). |
| `test_delete_files_refuses_stale_env_matched` | Insert a `RecordingArtifactRecompute` row with `matched=1` under a synthetic stale `env_id` distinct from `UserEnvironment.current()`. `delete_files(keys)` raises `StaleEnvMatchedError`; message names the stale `env_id` and points at `force_stale_env=True`. The artifact on disk is NOT deleted. Same test for `SortingAnalyzerRecompute.delete_files()`. |
| `test_delete_files_accepts_current_env_matched` | Same setup but the `matched=1` row's `env_id` equals `UserEnvironment.current()`. `delete_files(keys)` succeeds and removes the artifact. Regression guard: confirms the stale-env gate is not over-eager. |
| `test_delete_files_force_stale_env_audit` | Stale-env `matched=1` row, but caller passes `force_stale_env=True`. Deletion proceeds; an audit log entry (file or table — implementer's call, but it MUST exist and include caller identity + reason) records the override. |
| `test_recompute_admin_surface_parity` | v2 recompute tables expose equivalents of `attempt_all`, `remove_matched`, `with_names`, `get_parent_key`, `recheck`, `get_disk_space`, and `update_secondary`, or the dropped method is listed in `feature-parity.md` explicit non-parity. |

## Commands to run

```bash
export SPYGLASS_SKILL_DIR="${SPYGLASS_SKILL_DIR:-../spyglass-skill/skills/spyglass}"
test -f "$SPYGLASS_SKILL_DIR/scripts/code_graph.py"

pytest tests/spikesorting/v2/test_phase2_analyzer_curation.py -q
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

- **`baseline_metric_curation`** — pickle of the v1 `MetricCuration` output (fetched from `AnalysisNwbfile` columns), captured by extending `tests/spikesorting/v2/baseline_capture.py` from Phase 0. This phase's baseline-capture extension runs v1 `MetricCuration.populate` on top of the Phase 0 sort and pickles the resulting metrics DataFrame.
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
- `AnalyzerCuration` fetch helpers (`get_waveforms`, `get_metrics`, `get_labels`, `get_merge_groups`) and `materialize_curation()` are documented and tested.
- v2 recompute admin methods are present or explicitly listed as non-parity in `feature-parity.md`.
- v1 `MetricCuration` and `BurstPair` are NOT touched (sanity check via `git diff src/spyglass/spikesorting/v1/`).
- `code_graph.py describe` returns clean output for every new table; `path --up`/`path --down` chains match the design DAG; JSON warnings are empty or explicitly accounted for in `precondition-check.md`.
- CHANGELOG.md updated.
