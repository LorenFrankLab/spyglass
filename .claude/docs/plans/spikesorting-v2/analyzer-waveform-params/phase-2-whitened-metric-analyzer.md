# Phase 2 — Whitened metric analyzer + display/metric routing + recompute

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared-contracts](shared-contracts.md)

Add the second analyzer recipe — a **whitened** analyzer for cluster-separation
metrics — alongside the Phase 1 display analyzer, and route each consumer to the
right one: amplitudes AND voltage/spike-train metrics (snr, amplitude_cutoff,
isi_violation, ...) → display/unwhitened; ONLY PC/NN metrics
(principal_components, nn_advanced, ...) → metric/whitened. Extend the recompute
trio so both recipes are covered and never collide.

**Inputs to read first:**

- [_sorting_analyzer.py:213-411](../../../../../src/spyglass/spikesorting/v2/_sorting_analyzer.py#L213-L411) — `build_analyzer` (post-Phase-1: takes the cache folder, name-encoded, + a resolved params dict — never a bare name); where whitening must be applied to the recording before the metric build.
- [_sorting_dispatch.py:360-405](../../../../../src/spyglass/spikesorting/v2/_sorting_dispatch.py#L360-L405) — the sorter's external float64 `sip.whiten` path (seed-pinned); reuse it as the metric analyzer's own whitening (for PC/NN metrics only — NOT a claim it matches what the sorter saw; KS4 whitens internally, clusterless doesn't whiten).
- [metric_curation.py:495-577](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L495-L577) — `AnalyzerCurationSelection.insert_selection` (content-addressed PK); gets a `waveform_params_name` added to its identity.
- [metric_curation.py:755-825](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L755-L825) — `_compute_metrics` / `_compute_merge_groups` / `ensure_extensions`; consumers to route by analyzer type (PC/NN metrics to metric, merge/template/voltage/spike-time work to display).
- [sorting.py:794-831](../../../../../src/spyglass/spikesorting/v2/sorting.py#L794-L831) — `SortingSelection.resolve_source`; reuse this helper when deriving the default metric waveform row from a sort source.
- [_metric_curation_plots.py:243-345](../../../../../src/spyglass/spikesorting/v2/_metric_curation_plots.py#L243-L345) — `peak_amplitudes_from_analyzer` (display) and `burst_pair_metrics_from_analyzer` (display legs).
- [recompute.py](../../../../../src/spyglass/spikesorting/v2/recompute.py) + [_recompute.py:25-90](../../../../../src/spyglass/spikesorting/v2/_recompute.py#L25-L90) — `SortingAnalyzer{Versions,RecomputeSelection,Recompute}` + `ANALYZER_RECOMPUTE_EXTENSIONS`; must rebuild per `waveform_params_name`.
- [sorting.py:1940-1995](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1940-L1995) — sort-time `peak_amplitude_uv` computation; must read the display analyzer.

**Contracts referenced:**

- [Display-vs-metric analyzer routing](shared-contracts.md#display-vs-metric-analyzer-routing) — the per-consumer table; do not weaken (`plot_by_sort_group_ids` and all burst legs load the DISPLAY analyzer; only the PC/NN metrics load the whitened one).
- [Analyzer cache identity](shared-contracts.md#analyzer-cache-identity) — recompute resolves the same `{sorting_id}__{waveform_params_name}.zarr`.

## Tasks

- **Whitened build path.** In `build_analyzer`, when the resolved
  `AnalyzerWaveformParameters` row has `whiten=True`, apply SI external float64
  `sip.whiten` (with the seed pin from `job_kwargs`) to the recording **before**
  `create_sorting_analyzer`. This is the **metric analyzer's own** whitening for
  cluster-separation metrics — NOT a claim that it matches what the sorter saw:
  MS4/MS5 use the same external whiten, but **KS4 whitens internally** and the
  **clusterless thresholder doesn't whiten at all**, so do not justify it as
  "what the sort saw." Reuse `_sorting_dispatch`'s pinned whiten call (import or
  shared helper); do not write a second whitening.
- **The whitened metric analyzer must use `return_in_uV=False`.** Today
  `create_sorting_analyzer` hardcodes `return_in_uV=True`
  (`_sorting_analyzer.py:359`); the display (unwhitened) recipe keeps it (real
  µV amplitudes). But `sip.whiten` PRESERVES per-channel gains, so a
  `return_in_uV=True` readback re-applies them and partially un-normalizes the
  whitened space for non-uniform gains — defeating the point of whitening for the
  PC/NN metrics. Thread `return_in_uV` from the row's `whiten` flag
  (`whiten=True → return_in_uV=False`). Add an **unequal-gain covariance test**:
  on a synthetic recording with non-uniform channel gains, the whitened metric
  analyzer's noise covariance is ≈ identity (whitened), which it would NOT be if
  the gains were re-applied.
- **Attach the metric recipe to curation identity.** Add a FK /
  `waveform_params_name` to `AnalyzerCurationSelection.insert_selection`'s
  content-addressed identity (default = the sort's region metric row, e.g.
  `franklab_cortex_metric_waveforms`; see
  [Region resolution](shared-contracts.md#region-resolution)), so the metric
  analyzer params are tracked per curation (v1-parity: v1 attached
  `WaveformParameters` to `MetricCurationSelection`). The default resolver is
  source-aware and uses `SortingSelection.resolve_source(key)` for source
  detection: single-recording sorts resolve through
  `RecordingSelection.preprocessing_params_name`, while concat-backed sorts
  resolve through the `ConcatenatedRecordingSelection -> PreprocessingParameters`
  FK'd `preprocessing_params_name`.
  Update the `uuid5` identity tuple and `_find_existing_pk`.
- **Route metrics by TYPE, per the routing contract — NOT all metrics through
  the whitened analyzer.** Split `_compute_metrics`
  ([metric_curation.py:755-825](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L755-L825)): voltage / spike-train metrics
  (`snr`, `amplitude_cutoff`, `amplitude_median`, `firing_rate`, `num_spikes`,
  `presence_ratio`, `isi_violation`) compute on the **unwhitened display**
  analyzer; PC / cluster-separation metrics (`principal_components`,
  `nn_advanced`, `isolation_distance`, `l_ratio`, `d_prime`) compute on the
  **whitened metric** analyzer; merge the two metric frames by unit id. Whitening
  ALL metrics would make SNR/amplitude meaningless. `_compute_merge_groups` runs
  on the **unwhitened display** analyzer — its `similarity_correlograms` preset
  is template-derived (a `template_similarity` step + a `unit_locations` spatial
  gate), and whitening distorts both, miscalibrating the `max_distance_um` gate.
  So the whitened analyzer's extension set is just PC: `principal_components` +
  the PCA metrics (no template_similarity / unit_locations / correlograms).
- **Route amplitudes → display analyzer.** `peak_amplitudes_from_analyzer`
  callers (`get_peak_amps`, `plot_peak_over_time`, `investigate_pair_peaks`) load
  the unwhitened display analyzer; sort-time `peak_amplitude_uv`
  ([sorting.py:1940-1995](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1940-L1995)) uses the sort's region display recipe (resolved in Phase 1).
- **All burst legs read the display analyzer, per the routing contract.**
  `burst_pair_metrics_from_analyzer` reads `wf_similarity` (template_similarity),
  `xcorrel_asymm` (correlograms), and `unit_distance` (unit_locations) all from
  the **display** analyzer — they are template- or spike-time-derived and must
  match the (also-display) merge engine they corroborate; cross-unit
  `isi_violation` stays from spike times. So `plot_by_sort_group_ids` loads only
  the display analyzer (no per-leg analyzer split needed). Update the BurstPair
  docstrings accordingly.
- **Recompute coverage + cleanup for both recipes.** Extend the `SortingAnalyzer`
  recompute trio so it rebuilds with the row's `waveform_params_name` (whitening
  included; `return_in_uV=False` for the whitened recipe) and hashes the
  deterministic extensions per recipe, keyed by `{sorting_id}__{name}` — a
  whitened and unwhitened analyzer for the same `sorting_id` produce independent
  recompute rows that never overwrite. Extend `find_orphaned_analyzer_folders`
  (Phase 1) so a metric `{sid}__{name}.zarr` is NOT orphaned while any
  `AnalyzerCurationSelection` references that metric `waveform_params_name`.
  Recapture the recompute baseline for `peak_amplitude_uv` — it shifts from the
  larger 20000-spike subsample for ALL sorts, and additionally from the narrower
  window for **hippocampus** sorts (cortex keeps 1.0/2.0).
- **Capture-then-compare parity (intentional shift).** Before routing metrics to
  the whitened analyzer, save the current (Phase-1, unwhitened, the fixture's
  region display recipe, 20000-spike) quality metrics for the smoke fixture to a
  file; after, record
  the whitened-vs-
  unwhitened delta in `v1-v2-divergences.md`. This is a documented behavior
  change, not a regression — assert the metrics are finite and ordered sensibly
  (e.g. a planted oversplit still scores high `nn`/similarity), not bit-equal.
- **Smoke-test the whitened build's ADDITIONAL cost** on one real-data slice
  (the base 20000-spike build was measured in Phase 1): the whitening step plus a
  SECOND analyzer per sort. Record the delta in wall-clock + peak memory vs the
  Phase-1 display-only build, and note it in the PR (long-running-computation
  idiom).
- **Docs:** update `v1-v2-divergences.md` so the BurstPair entry and the
  metric/analyzer entries reflect two recipes (whitened analyzer for PC/NN
  metrics; unwhitened display for amplitudes + voltage/spike-train metrics) and
  the restored whiten distinction; update `feature-parity.md`
  row for `WaveformParameters` to say v2 restores the whitened/unwhitened split
  via `AnalyzerWaveformParameters` rows.

## Deliberately not in this phase

- **Pipeline preset default + MS4 recommendation / auto-curation rule set /
  curation-loop walkthrough** →
  Phase 3.
- **A FigPack viewer or any new curation UI** — out of scope for this plan.
- **Re-tuning the whiten algorithm** — reuse the sorter's existing pinned
  `sip.whiten`; do not invent a new whitening.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_metric_analyzer_is_whitened` | building with a metric row (e.g. `franklab_cortex_metric_waveforms`) produces whitened traces (differs from the unwhitened build) on a synthetic analyzer; the build uses `return_in_uV=False` |
| `test_metric_analyzer_whitened_under_unequal_gains` | on a synthetic recording with NON-uniform channel gains, the whitened metric analyzer's noise covariance is ≈ identity (would NOT be if `return_in_uV=True` re-applied the gains) |
| `test_display_and_metric_analyzers_distinct_cache` | for one `sorting_id` the two recipes write distinct folders; neither overwrites the other (`db_unit`) |
| `test_analyzer_curation_selection_tracks_waveform_params` | `insert_selection` identity includes `waveform_params_name`; two metric recipes give different `analyzer_curation_id` (`db_unit`) |
| `test_analyzer_curation_selection_resolves_concat_metric_waveform_params` | for a concat-backed sort, the default metric `waveform_params_name` resolves from the `ConcatenatedRecordingSelection -> PreprocessingParameters` FK'd `preprocessing_params_name`, not a single-recording source query. Parent-Phase-3-dependent for end-to-end populate; if this subplan lands first, test resolver behavior with direct selection/source-part rows or monkeypatched source resolution (`db_unit` or `slow` depending on fixture) |
| `test_metric_routing_by_type` | voltage metrics (`snr`, `amplitude_cutoff`, ...) compute on the unwhitened analyzer; PC/NN metrics (`nn_advanced`, `principal_components`) on the whitened one; `get_peak_amps` loads unwhitened (monkeypatch the loader, assert the name passed per metric) |
| `test_snr_unaffected_by_metric_whitening` | `snr` for a unit is the SAME whether or not a whitened metric analyzer exists (it always reads unwhitened) — guards against the whitening leaking into voltage metrics |
| `test_burst_and_merge_use_display_analyzer` | `burst_pair_metrics_from_analyzer` and `_compute_merge_groups` read the unwhitened display analyzer (not the whitened one); a planted oversplit still scores high `wf_similarity` and is proposed for merge |
| `test_analyzer_recompute_separates_recipes` | recompute rebuilds each recipe under its own key; a whitened row does not match an unwhitened row (`db_unit`) |
| `test_orphan_detection_retains_referenced_metric_folder` | a `{sid}__{metric_name}.zarr` referenced by an `AnalyzerCurationSelection` is NOT orphaned; an unreferenced metric folder IS (`db_unit`) |
| `test_viz_renders` (existing, extend) | `plot_by_sort_group_ids` renders reading only the display analyzer (no whitened-analyzer load) (`slow`, `integration`) |

Mark the integration/slow tests; the whitened-build smoke test is a manual
measurement task, not a CI assertion.

## Fixtures

- Synthetic in-memory analyzers (DB-free) for the whiten / routing / similarity
  tests, extending the `synthetic_analyzer` pattern with a whitened variant.
- One real-data slice (the MEArec smoke fixture or a checked-in recording) for
  the build-time/memory smoke measurement.
- Save the pre-whitening metric baseline to a fixture file for the
  capture-then-compare task.
- Concat metric-row resolver coverage is parent-Phase-3-dependent for a true
  populated concat sort. Before that lands, cover the branch with DB-unit fixture
  rows or monkeypatched `SortingSelection.resolve_source`, not
  `ConcatenatedRecording.populate()`.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Every task implemented; routing matches the
  [routing contract](shared-contracts.md#display-vs-metric-analyzer-routing)
  exactly (amplitudes never read the whitened analyzer; voltage/spike-train
  metrics read the unwhitened analyzer; only PC/NN cluster-separation metrics
  read the whitened one).
- Recompute cannot collide whitened/unwhitened folders; both recipes are covered.
- The intentional metric shift is captured + documented, not silently changed
  (baseline saved, delta in divergences).
- "Deliberately not in this phase" honored — no preset/rule/notebook changes.
- No docstring/test/module name references this plan or its phase numbers.
- The whiten call reuses the sorter's pinned `sip.whiten` (no second whitening
  implementation left behind).
