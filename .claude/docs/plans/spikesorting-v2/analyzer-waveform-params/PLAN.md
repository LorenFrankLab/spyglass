# Analyzer Waveform Parameters & Curation Defaults — Implementation Plan

**Status:** Phases 1, 2, 3a, 3, 4, and 5 complete (each reviewed, simplified, and
verified) — the plan is fully executed. Phase 3 shipped the
`franklab_default_auto_curation_2026_06` auto-curation rule set (ISI `reject`
policy), the probe-labeled MS5 default
(`franklab_probe_hippocampus_30khz_ms5_2026_06`, same rows as the tetrode MS5
alias), the MS4 recommendation (containerized for modern hosts / local for
numpy<2) via the human-facing preset fields, and the auto → manual-merge → auto
curation-loop docs + notebook section 7.

Phase 5 reconciliation note (the phase file predates the later phases): the
SpikeInterface visualization/export bridge shipped as
`spyglass.spikesorting.v2.visualization` (imported as `ssviz`) — a discoverable,
key-aware facade over SI `widgets.*` / `exporters.*` backed by a DB-free
`_visualization.py` service module (registry + missing-extension policy +
routed-metric histogram plot). Routing holds to the merged display-vs-metric
contract: recording widgets read the saved preprocessed `Recording`; sorting /
waveform / location / merge widgets and the local exports read the sort's stored
display recipe via `Sorting.get_analyzer(key)` (no `waveform_params_name`);
`plot_metrics` reads the routed `AnalyzerCuration.get_metrics()` (Phase-4
template columns surfaced as-is); `plot_potential_merges` reads the persisted
`get_merge_groups()` and never recomputes candidates. Plot helpers are read-only
by default (`compute_missing=True` opts in to display-safe extensions only),
matplotlib is the default backend (`sortingview` is explicit opt-in), and no
populate/export path publishes. Two review-hardening departures from the phase
file's first sketch: `export_to_phy` defaults `compute_pc_features=False` so SI
never computes the whitened-metric-only `principal_components` onto the unwhitened
display analyzer (opt in explicitly), and `export_si_report`'s force-computation
set is trimmed to the display-safe extensions SI's report actually renders. No
FigPack / cloud publishing / web curation UI was added (out of scope, as planned).
Notebook section 7 gained an `ssviz` step; `feature-parity.md` and
`v1-v2-divergences.md` record the bridge as an additive v2 improvement.

Phase 4 reconciliation note (the phase file predates the no-clip validation):
the shipped default surfaced template-shape column is `trough_half_width` ONLY,
not the `trough_half_width` + `peak_to_trough_duration` pair the phase file
proposed. The no-clip validation on a representative hippocampal (tetrode)
fixture found `peak_to_trough_duration` boundary-clips on the intentional
0.5/0.5 hippocampus display window — it measures to the post-trough
repolarization peak, which saturates at SI's edge-exclusion boundary and stops
discriminating cells (literature trough-to-peak is ~0.5-0.8 ms for pyramidal
cells, past the 0.5 ms post-window). The trough-local `trough_half_width` stays
interior and discriminates E/I, so it is the sole no-clip-safe default;
`peak_to_trough_duration` and the slope columns are discoverable/opt-in (reliable
on the wider 1.0/2.0 window). Per the user, the hippocampus window stays 0.5/0.5
(a window change is a separate Phase 1 revisit). Targeted suites green in the
project v2 test environment (schema/NWB/db_unit surfacing-routing-defaults +
slow no-clip & E/I separability).

Phase 3 reconciliation note (the phase file predates Phase 3a's
single-source simplification): the execution backend stays on
`SorterParameters.execution_params` only — the MS4 recommendation is conveyed
through the preset's `recommendation_status` / `intended_use` / `notes`, never
execution columns. The plan's `test_preflight_guards_missing_local_ms4` /
`test_preflight_guards_missing_container_ms4_runtime` validation rows are
satisfied by the existing Phase 3a preflight tests
(`test_preflight_ms4_preset_gets_runtime_check`,
`test_preflight_container_runtime_errors`,
`test_preflight_matlab_local_backend_errors`), so they were not duplicated.

Restore DB-tracked spike-sorting analyzer waveform parameters (so the window and
subsample that produced each analyzer are recorded and reproducible, the way v1
tracked them), adopt region-specific analyzer waveform windows (hippocampus
intentionally 0.5/0.5 ms for dense/tight waveforms, cortex 1.0/2.0 ms for
wider waveforms; 20000 spikes) resolved from the source preprocessing recipe,
split the
analyzer into an unwhitened "actual waveform" recipe for display and a whitened
recipe for cluster metrics, ship a default auto-curation rule set that uses the
lab's ~2% ISI policy, document the polymer MS4 recipe as the recommended-science
option with a first-class containerized execution path (keeping a runnable MS5
preset as the default), document the auto → manual-merge → auto curation loop in
the user notebook, expose SpikeInterface waveform-shape (template) metrics in
the per-unit metric table so downstream consumers can classify cell types with
region-appropriate thresholds (the pipeline ships no cell-type thresholds), and
add a thin SpikeInterface visualization/export bridge for local inspection.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file — each is
   self-contained (inputs to read, contracts, tasks, validation, fixtures, review).
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need broader scope / risks / rollout?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — goals, non-goals, integration points, risks, rollout, open questions.
- [shared-contracts.md](shared-contracts.md) — the `AnalyzerWaveformParameters` table, the analyzer cache-identity key, display-vs-metric routing, and sorter execution backend provenance.
- Phases (each ships as a separable PR):
  - [phase-1-params-surface.md](phase-1-params-surface.md) — tracked `AnalyzerWaveformParameters` table + region-specific display rows + cache key + region resolution (preprocessing-recipe → row, stored on `Sorting`).
  - [phase-2-whitened-metric-analyzer.md](phase-2-whitened-metric-analyzer.md) — whitened metric analyzer, display/metric routing (incl. BurstPair legs), recompute coverage.
  - [phase-3a-containerized-sorter-execution.md](phase-3a-containerized-sorter-execution.md) — first-class SpikeInterface Docker/Singularity sorter execution tracked on `SorterParameters`; containerized MS4 row/preset support without host `numpy<2`.
  - [phase-3-defaults-rules-docs.md](phase-3-defaults-rules-docs.md) — auto-curation rules (`franklab_default_auto_curation_2026_06`), MS5 default + containerized/local MS4 recommendation, curation-loop docs + notebook.
  - [phase-4-waveform-shape-metrics.md](phase-4-waveform-shape-metrics.md) — expose SI template (waveform-shape) metric **columns** in the per-unit metric table for downstream, region-specific cell typing; configurable `template_metric_columns` (SI output-column names, e.g. `trough_half_width`); no cell-type thresholds shipped. Depends on Phase 2 (display-analyzer routing); independent of Phase 3.
  - [phase-5-si-visualization-export.md](phase-5-si-visualization-export.md) — add a discoverable `v2.visualization` facade plus key-aware wrappers around SI widgets/exporters (`plot_traces`, `plot_probe_map`, sorting/unit summaries, waveforms, metrics, potential merges, `export_report`, `export_to_phy`) with `matplotlib` as the local default; display-analyzer and Spyglass-metric routing preserved; plotting is read-only by default, richer widgets can opt in to computing missing display-safe extensions, and no FigPack/cloud curation UI is added.

## Deliberately not in this plan

- **Whitened-vs-unwhitened metrics validation against v1 baselines** beyond the
  capture-then-compare in Phase 2 — a full v1↔v2 metric parity study is separate.
- **A dedicated standalone v2 curation notebook** (a `12_*`-style file) — Phase 3
  extends the existing v2 notebook's curate section; a standalone notebook is a
  follow-up only if that section outgrows inline form.
- **Waveform windows for regions beyond hippocampus / cortex** — those two are
  region-specific in this plan (hippo 0.5/0.5 by design, cortex 1.0/2.0);
  other or multi-region sorts fall back to the wider cortex window rather than
  getting a tuned one. Adding more regions is a tracked-row-and-preset-mapping
  away.
- **Full FigPack / web curation state round-trip** — Phase 5 only exposes local
  SI widget/export helpers through Spyglass keys. It does not implement FigPack,
  cloud publishing, edited label import, or a new manual-curation state model.
