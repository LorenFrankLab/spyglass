# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

- `src/spyglass/spikesorting/v2/_sorting_analyzer.py:371-400` — `build_analyzer`:
  hardcoded `extension_params` (`max_spikes_per_unit=500` at `:375`,
  `waveforms {ms_before:1.0, ms_after:2.0}` at `:397`). **Replaced** by params
  resolved from an `AnalyzerWaveformParameters` row.
- `src/spyglass/spikesorting/v2/_analyzer_cache.py:57-72` —
  `analyzer_path(sorting_id)` returns `root/f"{sorting_id}.zarr"`, keyed by
  `sorting_id` **only**. **Changed** to include `waveform_params_name` so a
  whitened metric analyzer and an unwhitened display analyzer for the same sort
  never collide.
- `src/spyglass/spikesorting/v2/sorting.py:904` —
  `Sorting.Unit.peak_amplitude_uv` (and the sort-time analyzer build it reads).
  Value **shifts**: all sorts get the larger 20000-spike subsample (was 500), and
  hippocampus sorts also move to the 0.5/0.5 window (cortex keeps 1.0/2.0);
  persisted field changes, moving the recompute baseline. The display recipe is
  resolved from the sort's **region** (not a per-sort field, not in `sorting_id`),
  so `peak_amplitude_uv` stays deterministic for a `sorting_id` (content-addressing
  preserved; `_selection_plan.py:177` identity unchanged). See
  [Region resolution](shared-contracts.md#region-resolution).
- `src/spyglass/spikesorting/v2/_metric_curation_plots.py:243+` —
  `peak_amplitudes_from_analyzer` and `burst_pair_metrics_from_analyzer`:
  **routed** so ALL burst legs (`wf_similarity`, `xcorrel_asymm`,
  `unit_distance`, amplitudes) and the merge engine read the display
  (unwhitened) analyzer; only the PC/NN quality metrics read the whitened metric
  analyzer (see the
  [routing contract](shared-contracts.md#display-vs-metric-analyzer-routing)).
- `src/spyglass/spikesorting/v2/metric_curation.py:405-440` —
  `AutoCurationRules._default_payloads`: today only `none`, `v1_default_nn_noise`,
  `similarity_merge`. **Adds** `franklab_default_auto_curation_2026_06` with an
  `isi_violation > 0.02` rule.
- `src/spyglass/spikesorting/v2/_pipeline_run.py:87` — `run_v2_pipeline`
  `pipeline_preset` default **stays an MS5** preset (runs under `numpy>=2`).
  Optionally relabel to a probe-MS5 for the polymer lab. MS4
  (`franklab_probe_hippocampus_30khz_ms4_2026_06`, `_recipe_catalog.py:357`) is
  documented as the recommended-science option for `numpy<2`, NOT the default —
  an MS4 default would fail preflight on every modern install.
- `src/spyglass/spikesorting/v1/metric_curation.py:67-110` — v1
  `WaveformParameters` (the tracked table v2 regressed from). **Reference only**;
  untouched.
- `src/spyglass/decoding/v1/waveform_features.py:204-206` — decoding
  re-extracts its **own** 0.5/0.5 waveforms via `_fetch_waveform_v2`, independent
  of the metric-curation analyzer. **Untouched**; this is why the analyzer
  window change is isolated to curation.
- `src/spyglass/spikesorting/v2/metric_curation.py:776-818` — `_compute_metrics`
  (Phase 2 splits it display/whitened). Phase 4 **joins** the already-computed
  display `template_metrics` columns onto the display-side metric frame; the
  `template_metric_columns` it surfaces is threaded via `AnalyzerCurationFetched`.
- `src/spyglass/spikesorting/v2/_params/metric_curation.py:62-137` —
  `QualityMetricParamsSchema`. Phase 4 **adds** a validated `template_metric_columns`
  field (SI **output-column** names, e.g. `trough_half_width`, validated against
  `ComputeTemplateMetrics.get_metric_columns(single_channel_names)`) + an
  `_available_template_metric_columns()` helper; a matching
  `template_metric_columns: blob` column is added to the `QualityMetricParameters`
  def.
- `src/spyglass/spikesorting/v2/_metric_curation_nwb.py:53-74` —
  `build_quality_metrics_table` (column-generic write) + `read_quality_metrics` /
  `get_metrics` (column-generic read). Phase 4 **hardens** the per-cell `float()`
  cast at `:72` against a non-scalar template column; the surfaced columns flow
  through unchanged.

## Scope and dependency policy

### Goals

- Restore DB-tracked waveform parameters (Spyglass provenance) as a new
  `AnalyzerWaveformParameters` lookup, mirroring v1's `WaveformParameters`.
- Adopt **region-specific** analyzer waveform windows as tracked rows —
  hippocampus 0.5/0.5 ms, cortex 1.0/2.0 ms, both `max_spikes_per_unit=20000`
  (cortex keeps today's window; hippocampus narrows; the subsample rises from
  500 to the lab's 20000) — resolved from the sort's region preset, parallel to
  the region filter cutoffs.
- Provide two analyzer recipes per region — unwhitened (display / BurstPair
  amplitudes) and whitened (cluster metrics) — separated in the analyzer cache.
- Ship a default auto-curation rule set that thresholds `isi_violation > 0.02`
  (the lab's ~2% policy), not only `nn_noise_overlap`.
- Surface the polymer MS4 recipe as the documented recommended-science option,
  keeping a runnable MS5 preset as the default (MS4 needs `numpy<2`).
- Document the auto → manual-merge → auto curation loop.
- Expose SpikeInterface waveform-shape (template) metrics — spike width and
  related shape measures — as columns in the per-unit metric table, configurable
  per `QualityMetricParameters` row, so downstream consumers can classify cell
  types (e.g. hippocampal interneuron vs pyramidal) with region-appropriate
  thresholds of their own.

### Non-Goals

- No migration of existing v1 `WaveformParameters` / curation rows.
- No change to decoding's separate waveform extraction (`waveform_features.py`).
- Not making MS4 runnable under `numpy>=2` — preflight keeps guarding it and MS5
  stays the runnable fallback.
- Region windows cover **hippocampus and cortex only** — other/unknown regions
  and multi-region sorts fall back to the wider cortex window rather than getting
  their own tuned window (out of scope; a future tracked row away).
- **No putative cell-type classifier and no cell-type / rate×width threshold
  rules.** Phase 4 exposes the shape metrics; it does not classify. Region-specific
  cutoffs (hippocampus ≠ cortex ≠ striatum) are downstream/user-side — a shipped
  hippocampal boundary would silently mislabel every other region. No conjunctive
  (multi-metric AND) auto-curation rule type is added either.

## Metrics

- **Provenance:** every built analyzer resolves a named
  `AnalyzerWaveformParameters` row recorded in the DB; no path resolves a
  hardcoded window.
- **Parity (intentional shift):** quality metrics change — the larger subsample
  (20000 vs 500 spikes) shifts the template-derived metrics (SNR / amplitude) for
  ALL sorts; **hippocampus** sorts shift further because their window also narrows
  to 0.5/0.5 (cortex keeps 1.0/2.0); whitening additionally shifts only the PC/NN
  metrics. Spike-train metrics (firing_rate, ISI, presence_ratio) are unchanged.
  Capture a baseline before Phase 2 and record the shift in
  `v1-v2-divergences.md`; no silent change.
- **Cache correctness:** for one `sorting_id`, the whitened and unwhitened
  analyzers resolve to distinct folders and never overwrite each other.
- **Exposure (Phase 4):** `get_metrics` returns waveform-shape columns
  (`trough_half_width`, `peak_to_trough_duration`, ...) alongside the quality
  metrics, read from the display (unwhitened) analyzer; no shipped auto-curation
  rule thresholds any of them.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Cache-key change desyncs the recompute trio (it rebuilds + hashes the analyzer) | Recompute resolves the same `waveform_params_name`; Phase 2 adds recompute coverage and a cache-collision test |
| `peak_amplitude_uv` is a persisted field that shifts | Documented in divergences; recompute baseline recaptured in Phase 2 |
| MS4 needs `numpy<2`, so it cannot be the runnable default | MS5 stays the default; MS4 is documented (docstring + `describe_pipeline_presets`) as the recommended-science option to *select* on `numpy<2`; preflight guards a selected-but-unavailable MS4 |
| 20000-spike analyzer build is slower / larger | Phase 1 smoke-tests the base build's time + memory on a real-data slice (it ships the 20000 default); Phase 2 measures the whitened build's additional cost |

## Rollout Strategy

v2 is pre-release and the schema freeze is lifted (project policy:
`memory/spikesorting-v2-schema-policy`), so table definitions change in place
with **no** deprecation period and **no** `params_schema_version` bump. New
default rows are added; the existing `v1_default_nn_noise` rule set is kept as a
named, selectable row (not removed) alongside the new Frank-lab default.

## Open Questions

1. **Whiten scope for the metric analyzer.** Resolved: whitening applies ONLY to
   PC / cluster-separation metrics (`principal_components`, `nn_advanced`,
   `isolation_distance`, `l_ratio`, `d_prime`); voltage / spike-train metrics
   (`snr`, `amplitude_cutoff`, `amplitude_median`, `firing_rate`, `num_spikes`,
   `presence_ratio`, `isi_violation`) stay on the unwhitened analyzer — whitening
   normalizes per-channel variance, so SNR/amplitude on whitened traces are
   meaningless. The whiten reuses the sorter's pinned external float64
   `sip.whiten`, but is the metric analyzer's OWN whitening for cluster metrics —
   NOT a claim of "what the sort saw" (KS4 whitens internally; clusterless
   doesn't whiten). See the
   [routing contract](shared-contracts.md#display-vs-metric-analyzer-routing).
2. **Keep `v1_default_nn_noise`?** Best answer: yes — keep it as a named row;
   make `franklab_default_auto_curation_2026_06` the one the notebook drives
   (pipeline presets do not wire analyzer curation — see phase 3).

## Estimated Effort

~750 LOC across four phases. Phase 1 ~200 (table + cache key + wiring + tests),
Phase 2 ~250 (whitened build + burst routing + recompute coverage + tests),
Phase 3 ~150 (preset default + rule set + docs + tests), Phase 4 ~150 (template
metric param + column join + writer guard + notebook/docs + tests).
