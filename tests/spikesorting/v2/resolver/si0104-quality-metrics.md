# SpikeInterface 0.104.3 quality-metric / auto-merge resolver

Source-of-truth verification for the analyzer-driven curation surface, run
against the **installed** SI 0.104.3 in the `spyglass_spikesorting_v2` conda
env (not docs/training). Each claim below was exercised live; commands are
reproducible from the repo root with the v2 env on PATH.

- SpikeInterface: 0.104.3
- `spikeinterface.qualitymetrics` is deprecated (removed in 0.105.0); use
  `spikeinterface.metrics.quality`.

## Metric list + import paths (verified)

- `from spikeinterface.metrics.quality import get_quality_metric_list,
  compute_quality_metrics` both resolve. `compute_quality_metrics` is also
  importable from `spikeinterface.metrics` and (deprecated)
  `spikeinterface.qualitymetrics`.
- `get_quality_metric_list()` returns:
  `['amplitude_cutoff', 'amplitude_cv', 'amplitude_median', 'd_prime',
  'drift', 'firing_range', 'firing_rate', 'isi_violation', 'mahalanobis',
  'nearest_neighbor', 'nn_advanced', 'noise_cutoff', 'num_spikes',
  'presence_ratio', 'rp_violation', 'sd_ratio', 'silhouette',
  'sliding_rp_violation', 'snr', 'synchrony']`.

## nn_advanced (verified)

- `nn_isolation` / `nn_noise_overlap` are **not** valid metric *names*.
  Requesting either via `compute_quality_metrics(metric_names=[...])` raises
  `ValueError("The metric '...' has been re-named or re-organized...")`.
- The metric **name** is `nn_advanced`; computing it yields **output columns**
  `nn_isolation` and `nn_noise_overlap`. It is a PCA metric — it requires
  `principal_components` to exist and `skip_pc_metrics=False`.
- Consequence: `QualityMetricParamsSchema.metric_names` requests `nn_advanced`;
  an `AutoCurationRules.Rule` still thresholds the `nn_noise_overlap` *column*.

## isi_violation (verified) — IMPORTANT divergence from the metric name

- Requesting metric name `isi_violation` produces **no column literally named
  `isi_violation`**. It produces two columns: `isi_violations_ratio` (SI's
  Hill/UMS2000 contamination estimate, unbounded) and `isi_violations_count`
  (the raw violation count).
- Spyglass's historical `isi_violation` is the bounded fraction
  `count / (n_spikes - 1)` with `isi_threshold_ms=2.0, min_isi_ms=0.0`
  (`v1/metric_utils.py:16-38`). v2 therefore computes its `isi_violation`
  column from `isi_violations_count / (num_spikes - 1)` rather than reading
  any SI `isi_violation*` column directly. The 0/1-spike cases are guarded
  (→ NaN, sanitized before serialization).

## auto-merge presets + compute_merge_unit_groups (verified)

- `from spikeinterface.curation import compute_merge_unit_groups`.
- Signature accepts `compute_needed_extensions` (verified present); pass
  `compute_needed_extensions=False` so the Spyglass-computed extension set is
  the audited one.
- Valid presets (`spikeinterface.curation.auto_merge._compute_merge_presets`):
  `['feature_neighbors', 'similarity_correlograms', 'slay', 'temporal_splits',
  'x_contaminations']`. `compute_merge_unit_groups` raises
  `ValueError("preset must be one of ...")` for anything else. `slay` IS a
  valid preset in 0.104.3. The `AutoCurationRules` schema additionally accepts
  the sentinel `'none'` (Spyglass-level "skip auto-merge", not an SI preset).
