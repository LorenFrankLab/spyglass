# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

- `src/spyglass/spikesorting/v2/_sorting_analyzer.py:371-400` — `build_analyzer`:
  hardcoded `extension_params` (`max_spikes_per_unit=500` at `:375`,
  `waveforms {ms_before:1.0, ms_after:2.0}` at `:397`). **Replaced** by params
  resolved from an `AnalyzerWaveformParameters` row.
- `src/spyglass/spikesorting/v2/_analyzer_cache.py:57-72` —
  `analyzer_path(sorting_id, waveform_params_name)` returns
  `root/f"{sorting_id}__{waveform_params_name}.zarr"`. **Changed** from the old
  `sorting_id`-only path so a whitened metric analyzer and an unwhitened
  display analyzer for the same sort never collide.
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
  Optionally relabel to a probe-MS5 for the polymer lab. Local MS4
  (`franklab_probe_hippocampus_30khz_ms4_2026_06`, `_recipe_catalog.py:357`)
  still requires a compatible host runtime, but Phase 3a adds a containerized
  MS4 row/preset so the recommended-science option is runnable from modern
  `numpy>=2` Spyglass environments when Docker/Singularity is available.
- `src/spyglass/spikesorting/v2/sorting.py:168-351`,
  `src/spyglass/spikesorting/v2/_sorting_dispatch.py:277-423`, and
  `src/spyglass/spikesorting/v2/_pipeline_preflight.py` — `SorterParameters`,
  `run_si_sorter`, and preflight. Phase 3a **adds** tracked
  `execution_params` to `SorterParameters` and routes SI Docker/Singularity
  kwargs through the DB-free sorter dispatch, rather than hiding container
  choices in scientific sorter params or `job_kwargs`.
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
- `src/spyglass/spikesorting/v2/recording.py:1481`,
  `src/spyglass/spikesorting/v2/sorting.py:1412-1493`, and
  `src/spyglass/spikesorting/v2/metric_curation.py:881-1085` — existing
  `get_recording`, `get_analyzer` / `add_extensions`, `get_metrics`, and
  lab-specific plotting helpers. Phase 5 **wraps** SI widgets/exporters around
  these accessors instead of reimplementing plotting.
- `<si>/src/spikeinterface/widgets/widget_list.py:127-167`,
  `<si>/doc/modules/widgets.rst:9-17`, and
  `<si>/doc/modules/exporters.rst:52-166` — SI's native widget/export surface
  (`plot_traces`, `plot_probe_map`, `plot_sorting_summary`,
  `plot_unit_summary`, `plot_unit_waveforms`, `plot_quality_metrics`,
  `plot_template_metrics`, `plot_potential_merges`, `export_report`,
  `export_to_phy`). Phase 5 exposes the useful subset through Spyglass keys
  with local `matplotlib` defaults.
- `<si>/src/spikeinterface/sorters/runsorter.py:103-206,319-690` — SI's
  `run_sorter(..., docker_image=..., singularity_image=...)` container path.
  Phase 3a exposes this as first-class sorter execution provenance.

## Scope and dependency policy

### Goals

- Restore DB-tracked waveform parameters (Spyglass provenance) as a new
  `AnalyzerWaveformParameters` lookup, mirroring v1's `WaveformParameters`.
- Adopt **region-specific** analyzer waveform windows as tracked rows —
  hippocampus intentionally uses 0.5/0.5 ms for dense/tight spikes, cortex uses
  1.0/2.0 ms for broader waveforms, both `max_spikes_per_unit=20000` (cortex
  keeps today's window; hippocampus narrows; the subsample rises from 500 to the
  lab's 20000) — resolved from the sort's source preprocessing recipe, parallel
  to the region filter cutoffs.
- Provide two analyzer recipes per region — unwhitened (display / BurstPair
  amplitudes) and whitened (cluster metrics) — separated in the analyzer cache.
- Ship a default auto-curation rule set that thresholds `isi_violation > 0.02`
  (the lab's ~2% policy), not only `nn_noise_overlap`.
- Surface the polymer MS4 recipe as the documented recommended-science option,
  keeping a runnable MS5 preset as the default. Local MS4 remains constrained by
  the host runtime, but containerized MS4 is a first-class opt-in path for
  modern `numpy>=2` Spyglass environments.
- Document the auto → manual-merge → auto curation loop.
- Expose SpikeInterface waveform-shape (template) metrics — spike width and
  related shape measures — as columns in the per-unit metric table, configurable
  per `QualityMetricParameters` row, so downstream consumers can classify cell
  types (e.g. hippocampal interneuron vs pyramidal) with region-appropriate
  thresholds of their own. The shipped default surfaces conservative width /
  duration columns (`trough_half_width`, `peak_to_trough_duration`); slope columns
  remain opt-in unless hippocampal-window validation supports them.
- Expose a thin SpikeInterface visualization/export bridge for local inspection:
  a discoverable `v2.visualization` facade, trace and probe-map widgets from
  Spyglass recordings, sorting/unit/waveform widgets from the display analyzer,
  official metric plots from `AnalyzerCuration.get_metrics`, potential-merge
  plots from persisted Spyglass merge candidates, and optional local SI report /
  Phy exports.

### Non-Goals

- No migration of existing v1 `WaveformParameters` / curation rows.
- No change to decoding's separate waveform extraction (`waveform_features.py`).
- No default switch to MS4. MS5 stays the default; local MS4 still needs a
  compatible host runtime, and containerized MS4 is opt-in and requires
  Docker/Singularity plus a pinned image.
- Region windows cover **hippocampus and cortex only** — other/unknown regions
  and multi-region sorts fall back to the wider cortex window rather than getting
  their own tuned window (out of scope; a future tracked row away).
- **No putative cell-type classifier and no cell-type / rate×width threshold
  rules.** Phase 4 exposes the shape metrics; it does not classify. Region-specific
  cutoffs (hippocampus ≠ cortex ≠ striatum) are downstream/user-side — a shipped
  hippocampal boundary would silently mislabel every other region. No conjunctive
  (multi-metric AND) auto-curation rule type is added either.
- **No FigPack / cloud curation UI in this subplan.** Phase 5 is local SI
  widget/export plumbing only. It does not add cloud publishing by default, a
  FigPack curation state round-trip, or a new manual label model.

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
  (`trough_half_width`, `peak_to_trough_duration` by default) alongside the
  quality metrics, read from the display (unwhitened) analyzer; no shipped
  auto-curation rule thresholds any of them.
- **Inspection (Phase 5):** SI visualization/export helpers preserve routing:
  recording widgets read the saved preprocessed `Recording`; waveform/template/
  location/merge widgets and exports read the display analyzer; official metric
  plots read `AnalyzerCuration.get_metrics`; no default visualization path uses
  the whitened metric analyzer.
- **Sorter execution provenance (Phase 3a):** local vs Docker vs Singularity
  execution is recorded in `SorterParameters.execution_params`; local and
  containerized rows with identical scientific params have distinct
  `sorter_params_name` values and therefore distinct `sorting_id` provenance.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Cache-key change desyncs the recompute trio (it rebuilds + hashes the analyzer) | Recompute resolves the same `waveform_params_name`; Phase 2 adds recompute coverage and a cache-collision test |
| `peak_amplitude_uv` is a persisted field that shifts | Documented in divergences; recompute baseline recaptured in Phase 2 |
| MS4 local runtime constraints make it a bad universal default | MS5 stays the default; containerized MS4 is documented (docstring + `describe_pipeline_presets`) as the first-class recommended-science option for modern hosts, local MS4 remains available for compatible environments, and preflight guards selected-but-unavailable local or container runtimes |
| Container execution could become an untracked runtime override or silently fall back to local sorting | Phase 3a stores backend/image in `SorterParameters.execution_params`, uses distinct sorter row names, rejects container kwargs from scientific params/job kwargs, and preflight fails clearly if Docker/Singularity is unavailable |
| Container rows look pinned by image but float the installed SpikeInterface version | Shipped/recommended container rows must use `installation_mode="no-install"` with a baked SI runtime or pin `spikeinterface_version`; unpinned custom `auto` rows are labeled exploratory/non-reproducible |
| Replacing the old MATLAB sorter name-based Singularity fallback breaks custom Kilosort/IronClust rows silently | Phase 3a makes this an explicit behavior change: MATLAB-backed legacy sorters require tracked Docker/Singularity execution or fail preflight/dispatch clearly; no silent local fallback |
| 20000-spike analyzer build is slower / larger | Phase 1 smoke-tests the base build's time + memory on a real-data slice (it ships the 20000 default); Phase 2 measures the whitened build's additional cost |
| Hippocampus 0.5/0.5 display windows are shorter than SI's broad waveform defaults | This is intentional for hippocampal dense/tight waveforms. Phase 4 validates that default surfaced template-shape metrics do not boundary-clip on representative hippocampal fixtures; cortex and unknown/multi-region sorts keep the wider 1.0/2.0 fallback |
| SI widget/export wrappers accidentally use the whitened metric analyzer, recompute merge candidates, hide behind hard-to-discover table methods, mutate analyzer caches from plot calls, miss required extensions, or default to a web backend | Phase 5 wrapper tests assert display-analyzer routing, persisted `get_merge_groups` use, a discoverable `v2.visualization` facade, read-only-by-default extension policy with explicit compute opt-in, `matplotlib` defaults, explicit `sortingview` opt-in, and no populate-side plotting/export |

## Rollout Strategy

v2 is pre-release and the parent plan's pre-production direct-edit policy
applies, so the table definition changes in this subplan land in place with
**no** deprecation period. Per-phase tasks specify whether a schema-version
field changes or stays pinned. New default rows are added; the existing
`v1_default_nn_noise` rule set is kept as a named, selectable row (not removed)
alongside the new Frank-lab default.

## Open Questions

1. **Whiten scope for the metric analyzer.** Resolved: whitening applies ONLY to
   PC / cluster-separation metrics (the `principal_components` extension + SI
   0.104's PCA metrics `d_prime`, `mahalanobis`, `nearest_neighbor`,
   `nn_advanced`, `silhouette`); voltage / spike-train metrics
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

~1130 LOC across six slices. Phase 1 ~200 (table + cache key + wiring + tests),
Phase 2 ~250 (whitened build + burst routing + recompute coverage + tests),
Phase 3a ~180 (sorter execution params + container dispatch/preflight + tests),
Phase 3 ~150 (preset default + rule set + docs + tests), Phase 4 ~150 (template
metric param + column join + writer guard + notebook/docs + tests), Phase 5 ~200
(SI widget/export wrappers + notebook/docs + routing/backend tests).
