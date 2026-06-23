# Analyzer Waveform Parameters & Curation Defaults — Implementation Plan

**Status:** Phases 1, 2, 3a, and 3 complete (each reviewed, simplified, and
verified). Phase 3 shipped the `franklab_default_auto_curation_2026_06`
auto-curation rule set (ISI `reject` policy), the probe-labeled MS5 default
(`franklab_probe_hippocampus_30khz_ms5_2026_06`, same rows as the tetrode MS5
alias), the MS4 recommendation (containerized for modern hosts / local for
numpy<2) via the human-facing preset fields, and the auto → manual-merge → auto
curation-loop docs + notebook section 7. Targeted suites green (65 passed:
preset/parity/curation/preflight, in the project v2 test environment). Phase 4
and Phase 5 remain to execute.

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
