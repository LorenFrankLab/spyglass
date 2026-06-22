# Shared contracts

[← back to PLAN.md](PLAN.md)

Cross-phase contracts. Each lives here once; phases link in by anchor and must
not weaken them.

- [`AnalyzerWaveformParameters` table](#analyzerwaveformparameters-table)
- [Region resolution](#region-resolution)
- [Analyzer cache identity](#analyzer-cache-identity)
- [Display-vs-metric analyzer routing](#display-vs-metric-analyzer-routing)

---

## `AnalyzerWaveformParameters` table

Defined in Phase 1; referenced by Phases 2 and 3. A `dj.Lookup` mirroring v1's
`WaveformParameters` (`v1/metric_curation.py:67-110`) so the waveform settings
that produced an analyzer are **tracked in the DB**, not hardcoded. Validated
with a pydantic schema (v2 style), name resolved from the blob.

```
@schema
class AnalyzerWaveformParameters(SpyglassMixin, dj.Lookup):
    definition = """
    waveform_params_name: varchar(64)
    ---
    waveform_params: blob   # validated AnalyzerWaveformParamsSchema dump
    params_schema_version=1: int
    """
```

```python
class AnalyzerWaveformParamsSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ms_before: float = Field(default=1.0, gt=0.0)
    ms_after: float = Field(default=2.0, gt=0.0)
    max_spikes_per_unit: int = Field(default=20000, ge=1)
    whiten: bool = False
    purpose: Literal["display", "metric"] = "display"
```

Default rows — region-specific windows (hippocampus's denser/tighter spikes
take a narrower window; cortex's broader waveforms take the wider one). Each
region has a display (unwhitened) and a metric (whitened) row; the sort's region
preset names which pair it uses (see
[Region resolution](#region-resolution)):

| `waveform_params_name` | ms_before | ms_after | max_spikes_per_unit | whiten | purpose |
| --- | --- | --- | --- | --- | --- |
| `franklab_hippocampus_actual_waveforms` | 0.5 | 0.5 | 20000 | False | display |
| `franklab_hippocampus_metric_waveforms` | 0.5 | 0.5 | 20000 | True | metric |
| `franklab_cortex_actual_waveforms` | 1.0 | 2.0 | 20000 | False | display |
| `franklab_cortex_metric_waveforms` | 1.0 | 2.0 | 20000 | True | metric |

The schema default (1.0 / 2.0) is the cortex/wide fallback for any custom row
and for unknown/multi-region sorts (see [Region resolution](#region-resolution)).

Invariants (do not weaken):
- `whiten` is part of analyzer **identity** — it MUST flow into the cache key
  (see [Analyzer cache identity](#analyzer-cache-identity)), never be resolved
  after the fact from `QualityMetricParameters`.
- **`return_in_uV` is derived from `whiten`, not a separate knob.** Display rows
  (`whiten=False`) build with `return_in_uV=True` (real µV amplitudes); metric
  rows (`whiten=True`) build with `return_in_uV=False` — because `sip.whiten`
  preserves channel gains, a `True` readback would re-apply them and
  un-normalize the whitened space (Phase 2). Since it changes analyzer content,
  it is part of the recipe, not user-tunable.
- Rows differ along two axes — region (window) and `whiten` / `purpose`. The
  cache is keyed by the row **name**, not by content (see
  [Analyzer cache identity](#analyzer-cache-identity)), so a custom row with the
  same field values but a different name resolves to a SEPARATE cache folder —
  acceptable redundancy, not a dedup target.
- `waveform_params_name` is validated **path-safe** (`^[A-Za-z0-9_]+$`) at insert
  time, since it is embedded in a cache folder name.

---

## Region resolution

Wired in Phases 1-2. A sort's waveform window is **region-specific**, resolved
from the recording's region-specific **preprocessing recipe** — the same signal
that already sets the region filter cutoff — not a free per-sort knob:

- `_recipe_catalog` maps the recording's `preprocessing_params_name`
  (region-specific: the `HIPPOCAMPUS_PREPROC` / `CORTEX_PREPROC` values of the
  existing `_REGION_PREPROC` map) to its `(display, metric)` waveform-params row
  pair — hippocampus → `franklab_hippocampus_*`, cortex → `franklab_cortex_*`.
  This lives in `_recipe_catalog`, **not** on `_PipelinePreset` (which is
  `extra="forbid"`), so no preset-schema change is needed.
- At sort time `make_fetch` resolves the **display** row from the recording's
  `preprocessing_params_name`, and `make_insert` stores it in a new secondary
  `Sorting.display_waveform_params_name` field (Phase 1). Every later rebuild /
  cache-miss load reads that STORED value (never re-resolves), so the analyzer
  and `peak_amplitude_uv` are stable. The **metric** row resolves the same way
  and flows to `AnalyzerCurationSelection` identity (Phase 2).
- **Any preprocessing recipe not in the region map (custom / multi-region) falls
  back to the wider cortex window** (`franklab_cortex_*`, 1.0/2.0) — never
  silently mix windows.

Determinism proof (do not weaken): the row name is a pure function of the
recording's `preprocessing_params_name`, which is part of `recording_identity`
(`_selection_identity.py:44`) and therefore of `recording_id` ⊆ `sorting_id`. So
the same `sorting_id` always yields the same window and the same
`peak_amplitude_uv`; the window is NOT user-tunable per sort and is NOT added to
`sorting_id` identity.

---

## Analyzer cache identity

Changed in Phase 1; the recompute trio (Phase 2) MUST resolve the same key.
Today `analyzer_path` is keyed by `sorting_id` only
(`_analyzer_cache.py:57-72`, returning `root/f"{sorting_id}.zarr"`). With two
recipes per sort, the path MUST include the waveform-params identity:

```python
def analyzer_path(sorting_id, waveform_params_name: str) -> Path:
    return analyzer_cache_root() / f"{sorting_id}__{waveform_params_name}.zarr"
```

Keyed by **name**, not a content hash: the name is the tracked provenance
identity, and `_analyzer_cache` already follows a name-like (`sorting_id`) key.
(A content hash would dedup identical-content rows but decouples the cache from
the provenance name; out of scope.)

Invariants (do not weaken):
- `waveform_params_name` is path-safe-validated before being embedded in the
  folder name (see the table contract).
- `analyzer_path` always takes an explicit `waveform_params_name`. `get_analyzer`
  / `load_or_rebuild_analyzer` default to the sort's stored display recipe
  (`Sorting.display_waveform_params_name`) when no name is given — a
  deterministic, well-defined default (the sort's OWN display analyzer), not a
  silent cross-recipe reuse. Callers needing the whitened metric recipe
  (metric-curation compute) pass it explicitly. (So `Sorting.add_extensions`,
  `sorting.py:1487`, stays display-only via this default — no signature change.)
- `find_orphaned_analyzer_folders` and the recompute path both enumerate by the
  `{sorting_id}__{waveform_params_name}.zarr` pattern, so a whitened folder is
  never mistaken for the unwhitened one.
- The recompute trio rebuilds with the row's params, so a recomputed analyzer is
  byte-comparable to the cached one for the SAME `waveform_params_name`.

---

## Display-vs-metric analyzer routing

Implemented in Phase 2; the curation-loop docs (Phase 3) describe it. A v2 sort
has two analyzers; each consumer reads the correct one:

| Consumer | Analyzer | Why |
| --- | --- | --- |
| `Sorting.Unit.peak_amplitude_uv` (sort time) | the sort's region display row (unwhitened) | a µV amplitude must come from real, unwhitened waveforms |
| **Voltage / spike-train quality metrics** — `snr`, `amplitude_cutoff`, `amplitude_median`, `firing_rate`, `num_spikes`, `presence_ratio`, `isi_violation` | display (unwhitened) | whitening normalizes per-channel variance, so SNR / amplitude on whitened traces are meaningless; these must be on real voltages |
| **PC / cluster-separation metrics** — `principal_components`, `nn_advanced` (nn_isolation / nn_noise_overlap), `isolation_distance`, `l_ratio`, `d_prime` | the sort's region metric row (whitened) | decorrelated (whitened) space is where PC/NN separation is meaningful — this is the lab's "whiten for cluster metrics" |
| **Template-shape metrics** — columns `trough_half_width`, `peak_half_width`, `peak_to_trough_duration`, `repolarization_slope`, `recovery_slope` (surfaced for downstream cell typing; Phase 4) | display (unwhitened) | waveform SHAPE must come from real templates — whitening normalizes per-channel variance and distorts the shape, so a width column on a whitened template is meaningless (same reason as `snr`). The `template_metrics` extension already lives on the display analyzer; Phase 4 surfaces its **output columns** (selected by column name, not metric name), exposed not thresholded |
| **SI visualization/export helpers** — discoverable `v2.visualization` facade plus optional table delegates for `plot_sorting_summary`, `plot_unit_summary`, `plot_unit_waveforms`, `plot_spikes_on_traces`, `plot_unit_locations`, `plot_quality_metrics`, `plot_template_metrics`, `plot_potential_merges`, `export_report`, `export_to_phy` (Phase 5) | display (unwhitened); official metric overview uses `AnalyzerCuration.get_metrics()`; potential-merge plots use persisted `AnalyzerCuration.get_merge_groups()` | visual inspection, report figures, and Phy exports should show real waveforms / locations / templates and Spyglass-routed metric provenance. Recording-only widgets (`plot_traces`, `plot_probe_map`) use the saved preprocessed `Recording`, not any analyzer. Plot helpers must not recompute merge candidates or use the whitened metric analyzer by default |
| `get_peak_amps`, `plot_peak_over_time`, `investigate_pair_peaks` | display (unwhitened) | amplitude inspection in real µV |
| **Merge engine** — `compute_merge_unit_groups` (`similarity_correlograms` etc.) | display (unwhitened) | its steps are template-derived: `template_similarity` (real waveform shapes) and a `unit_locations` spatial gate (`max_distance_um`). Whitening distorts both, so the spatial gate would be miscalibrated; run on real templates |
| `burst_pair_metrics_from_analyzer` `wf_similarity` (template_similarity) | display (unwhitened) | template_similarity is template-derived; use real shapes, consistent with the merge engine it corroborates |
| `burst_pair_metrics_from_analyzer` `xcorrel_asymm` (correlograms), `unit_distance` (unit_locations) | display (unwhitened) | correlograms are spike-time based; `unit_locations` IS template-derived (whitening-sensitive), so display gives the real physical positions |
| cross-unit `isi_violation` (burst) | neither — computed from spike times | independent of whitening |

The whitened metric analyzer is therefore built ONLY for the PC/NN metrics
(`principal_components` + `nn_advanced` / `isolation_distance` / `l_ratio` /
`d_prime`); everything template-shape, position, voltage, or spike-time based
uses the unwhitened display analyzer.

Invariant (do not weaken): ALL burst-pair legs (`wf_similarity`,
`xcorrel_asymm`, `unit_distance`, amplitudes) and the merge engine read the
**display** analyzer; ONLY the PC/NN quality metrics read the **whitened**
metric analyzer. Metric / non-display consumers pass an explicit
`waveform_params_name`; display-only helpers may use the stored display default
(`Sorting.display_waveform_params_name`), never a shared cross-recipe default.
Phase 5 visualization/export helpers are display-only by default and must not
quietly switch to the whitened metric analyzer.
