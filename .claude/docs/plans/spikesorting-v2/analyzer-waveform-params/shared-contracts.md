# Shared contracts

[← back to PLAN.md](PLAN.md)

Cross-phase contracts. Each lives here once; phases link in by anchor and must
not weaken them.

- [`AnalyzerWaveformParameters` table](#analyzerwaveformparameters-table)
- [Region resolution](#region-resolution)
- [Analyzer cache identity](#analyzer-cache-identity)
- [Display-vs-metric analyzer routing](#display-vs-metric-analyzer-routing)
- [Sorter execution backend](#sorter-execution-backend)

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
    params: blob   # validated AnalyzerWaveformParamsSchema dump
    params_schema_version=1: int
    """
```

> **As implemented (Phase 1):** the blob column is named `params` (not
> `waveform_params`) to match the four sibling v2 param Lookups
> (`PreprocessingParameters` / `SorterParameters` / …) and reuse the shared
> `validate_lookup_rows` / `reject_duplicate_parameter_content` guards verbatim;
> `AnalyzerWaveformParamsSchema` carries a `schema_version` field like its
> siblings. The PK `waveform_params_name` (the cross-phase identifier) is
> unchanged. The resolved DISPLAY recipe is stored on `Sorting` as a renamed
> **secondary FK** `display_waveform_params_name -> AnalyzerWaveformParameters`
> (DB-enforced provenance), and `fetch_waveform_params` resolves the row
> **strictly** (a missing row raises; no catalog fallback).

```python
class AnalyzerWaveformParamsSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ms_before: float = Field(default=1.0, gt=0.0)
    ms_after: float = Field(default=2.0, gt=0.0)
    max_spikes_per_unit: int = Field(default=20000, ge=1)
    whiten: bool = False
    purpose: Literal["display", "metric"] = "display"

    @model_validator(mode="after")
    def _purpose_matches_whiten(self):
        if self.purpose == "display" and self.whiten:
            raise ValueError("display waveform rows must be unwhitened")
        if self.purpose == "metric" and not self.whiten:
            raise ValueError("metric waveform rows must be whitened")
        return self
```

Default rows — region-specific windows (hippocampus's denser/tighter spikes
intentionally take a narrower 0.5/0.5 ms window; cortex's broader waveforms take
the wider 1.0/2.0 ms window). Each region has a display (unwhitened) and a
metric (whitened) row; the sort's source preprocessing recipe names which pair
it uses (see
[Region resolution](#region-resolution)):

| `waveform_params_name` | ms_before | ms_after | max_spikes_per_unit | whiten | purpose |
| --- | --- | --- | --- | --- | --- |
| `franklab_hippocampus_actual_waveforms` | 0.5 | 0.5 | 20000 | False | display |
| `franklab_hippocampus_metric_waveforms` | 0.5 | 0.5 | 20000 | True | metric |
| `franklab_cortex_actual_waveforms` | 1.0 | 2.0 | 20000 | False | display |
| `franklab_cortex_metric_waveforms` | 1.0 | 2.0 | 20000 | True | metric |

The schema default (1.0 / 2.0) is the cortex/wide fallback for any custom row
and for unknown/multi-region sorts (see [Region resolution](#region-resolution)).
Because the hippocampus row is deliberately shorter than SI's broad waveform
defaults, the surfaced template-shape default is validated against this window:
only the trough-local `trough_half_width` ships as a default (its half-amplitude
crossings stay interior to 0.5/0.5). `peak_to_trough_duration` and the slope
columns measure to the post-trough repolarization peak, which saturates at the
0.5 ms post-window's edge-exclusion boundary, so they are discoverable/opt-in,
not shipped defaults — reliable only on the wider 1.0/2.0 window.

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
- `purpose` and `whiten` are validated as a pair: display rows are always
  unwhitened, metric rows are always whitened. A custom row cannot claim
  `purpose="display"` while setting `whiten=True`, or vice versa.
- `waveform_params_name` is validated **path-safe** (`^[A-Za-z0-9_]+$`) at insert
  time, since it is embedded in a cache folder name.

---

## Region resolution

Wired in Phases 1-2. A sort's waveform window is **region-specific**, resolved
from the sort source's region-specific **preprocessing recipe** — the same
signal that already sets the region filter cutoff — not a free per-sort knob.
The resolver is source-aware and must reuse the existing
`SortingSelection.resolve_source(key)` helper for source detection; do not
re-inspect `RecordingSource` / `ConcatenatedRecordingSource` part tables in a
second bespoke branch:

- For `SortingSelection.RecordingSource`, read the upstream
  `RecordingSelection.preprocessing_params_name`.
- For `SortingSelection.ConcatenatedRecordingSource`, read the upstream
  `ConcatenatedRecordingSelection` row and use its FK'd
  `PreprocessingParameters` primary key (`preprocessing_params_name`). This is
  not a literal column in the class definition; it comes from the
  `-> PreprocessingParameters` FK. Do not query `RecordingSelection` with a
  concat-only row; concat member recordings are provenance inputs, but the
  concatenated source row is the sort input.
- `_recipe_catalog` maps that source `preprocessing_params_name`
  (region-specific: the `HIPPOCAMPUS_PREPROC` / `CORTEX_PREPROC` values of the
  existing `_REGION_PREPROC` map) to its `(display, metric)` waveform-params row
  pair — hippocampus → `franklab_hippocampus_*`, cortex → `franklab_cortex_*`.
  This lives in `_recipe_catalog`, **not** on `_PipelinePreset` (which is
  `extra="forbid"`), so no preset-schema change is needed.
- At sort time `make_fetch` resolves the **display** row from the source
  `preprocessing_params_name`, and `make_insert` stores it in a new secondary
  `Sorting.display_waveform_params_name` field (Phase 1). Every later rebuild /
  cache-miss load reads that STORED value (never re-resolves), so the analyzer
  and `peak_amplitude_uv` are stable. The **metric** row resolves the same way
  and flows to `AnalyzerCurationSelection` identity (Phase 2).
- **Any preprocessing recipe not in the region map (custom / multi-region) falls
  back to the wider cortex window** (`franklab_cortex_*`, 1.0/2.0) — never
  silently mix windows.

Determinism proof (do not weaken): the row name is a pure function of the sort
source's `preprocessing_params_name`. For single-recording sorts, that field is
part of `recording_identity` (`_selection_identity.py:44`) and therefore of
`recording_id` ⊆ today's `sorting_id` (`_selection_plan.py:177`). For
concat-backed sorts (parent Phase 3), `concat_recording_id` is the source
identity; parent Phase 3 must fold `concat_recording_id` into
`sorting_identity_payload` exactly as `recording_id` is folded in today, so the
same concat-backed `sorting_id` yields the same window and the same
`peak_amplitude_uv`. Until then, concat sorts cannot be populated and this
resolver branch is forward-compatible only. The window is NOT user-tunable per
sort and is NOT added to `sorting_id` identity.

---

## Analyzer cache identity

Changed in Phase 1; the recompute trio (Phase 2) MUST resolve the same key.
Before Phase 1, `analyzer_path` was keyed by `sorting_id` only
(`root/f"{sorting_id}.zarr"`). With two recipes per sort, the path MUST include
the waveform-params identity:

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
| **PC / cluster-separation metrics** — SI 0.104's PCA-metric set `get_quality_pca_metric_list()`: `d_prime`, `mahalanobis`, `nearest_neighbor`, `nn_advanced`, `silhouette` (plus the `principal_components` extension they need; `isolation_distance` / `l_ratio` are `mahalanobis` output columns in 0.104, not requestable metric names) | the sort's region metric row (whitened) | decorrelated (whitened) space is where PC/NN separation is meaningful — this is the lab's "whiten for cluster metrics" |
| **Template-shape metrics** — default column `trough_half_width` only; `peak_to_trough_duration`, slopes, and the other single-channel template columns are opt-in (surfaced for downstream cell typing; Phase 4) | display (unwhitened) | waveform SHAPE must come from real templates — whitening normalizes per-channel variance and distorts the shape, so a width column on a whitened template is meaningless (same reason as `snr`). The `template_metrics` extension already lives on the display analyzer; Phase 4 surfaces selected **output columns** (selected by column name, not metric name), exposed not thresholded. Only the trough-local `trough_half_width` ships as a default — it stays interior to the narrow 0.5/0.5 hippocampus window; `peak_to_trough_duration` and slopes measure to the post-trough repolarization peak, which clips on that window, so they are opt-in (reliable on 1.0/2.0) |
| **SI visualization/export helpers** — discoverable `v2.visualization` facade plus optional table delegates for `plot_sorting_summary`, `plot_unit_summary`, `plot_unit_waveforms`, `plot_spikes_on_traces`, `plot_unit_locations`, `plot_quality_metrics`, `plot_template_metrics`, `plot_potential_merges`, `export_report`, `export_to_phy` (Phase 5) | display (unwhitened); official metric overview uses `AnalyzerCuration.get_metrics()`; potential-merge plots use persisted `AnalyzerCuration.get_merge_groups()` | visual inspection, report figures, and Phy exports should show real waveforms / locations / templates and Spyglass-routed metric provenance. Recording-only widgets (`plot_traces`, `plot_probe_map`) use the saved preprocessed `Recording`, not any analyzer. Plot helpers must not recompute merge candidates or use the whitened metric analyzer by default |
| `get_peak_amps`, `plot_peak_over_time`, `investigate_pair_peaks` | display (unwhitened) | amplitude inspection in real µV |
| **Merge engine** — `compute_merge_unit_groups` (`similarity_correlograms` etc.) | display (unwhitened) | its steps are template-derived: `template_similarity` (real waveform shapes) and a `unit_locations` spatial gate (`max_distance_um`). Whitening distorts both, so the spatial gate would be miscalibrated; run on real templates |
| `burst_pair_metrics_from_analyzer` `wf_similarity` (template_similarity) | display (unwhitened) | template_similarity is template-derived; use real shapes, consistent with the merge engine it corroborates |
| `burst_pair_metrics_from_analyzer` `xcorrel_asymm` (correlograms), `unit_distance` (unit_locations) | display (unwhitened) | correlograms are spike-time based; `unit_locations` IS template-derived (whitening-sensitive), so display gives the real physical positions |
| cross-unit `isi_violation` (burst) | neither — computed from spike times | independent of whitening |

The whitened metric analyzer is therefore built ONLY for the PC/NN metrics
(the `principal_components` extension + SI 0.104's PCA metrics `d_prime` /
`mahalanobis` / `nearest_neighbor` / `nn_advanced` / `silhouette`); everything
template-shape, position, voltage, or spike-time based uses the unwhitened
display analyzer.

Invariant (do not weaken): ALL burst-pair legs (`wf_similarity`,
`xcorrel_asymm`, `unit_distance`, amplitudes) and the merge engine read the
**display** analyzer; ONLY the PC/NN quality metrics read the **whitened**
metric analyzer. Metric / non-display consumers pass an explicit
`waveform_params_name`; display-only helpers may use the stored display default
(`Sorting.display_waveform_params_name`), never a shared cross-recipe default.
Phase 5 visualization/export helpers are display-only by default and must not
quietly switch to the whitened metric analyzer.

---

## Sorter execution backend

Defined in Phase 3a; referenced by Phase 3's preset/docs work. SpikeInterface
can run sorters locally or inside Docker/Singularity via `run_sorter` execution
kwargs (`docker_image`, `singularity_image`, `delete_container_files`, and
container-install options). In v2 these are tracked as **sorter execution
provenance**, not ad hoc params:

```python
class SorterExecutionParamsSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    backend: Literal["local", "docker", "singularity"] = "local"
    container_image: str | None = None
    delete_container_files: bool = True
    installation_mode: Literal[
        "auto", "pypi", "github", "no-install"
    ] = "auto"
    spikeinterface_version: str | None = None
    extra_requirements: list[str] = Field(default_factory=list)
```

SI also supports `folder` / `dev` container install modes with a
`spikeinterface_folder_source`; this plan intentionally does not expose them in
the first pass because they require mounting a source tree as part of execution
provenance.

Reproducibility contract:
- The schema default is local execution, so `installation_mode="auto"` and
  `spikeinterface_version=None` are inert for local rows. For any shipped or
  recommended container row, however, runtime install provenance must be pinned:
  either use `installation_mode="no-install"` with a container image that already
  contains the intended SpikeInterface + sorter runtime, or use
  `installation_mode in {"pypi", "github"}` with an explicit
  `spikeinterface_version`. Do not ship a recommended row whose container
  installation floats via `installation_mode="auto"` and
  `spikeinterface_version=None`.
- Custom exploratory container rows may use SI's `auto` behavior, but
  `describe_pipeline_presets` / preflight must make clear that such a row is not
  reproducible by row content alone because the container-side SI install can
  drift with the host environment.

Storage contract:
- `SorterParameters` carries `execution_params: blob` plus
  `execution_params_schema_version=1: int`. The field lives on
  `SorterParameters`, not `QualityMetricParameters`, because execution backend
  can affect sorter output and the existing `sorter_params_name` already flows
  into `sorting_id`.
- The execution blob carries its own `schema_version`; the outer
  `execution_params_schema_version` follows the existing sorter-params pattern
  of backfilling from the blob when omitted/defaulted and cross-checking any
  explicitly supplied non-default value.
- `SorterParameters` duplicate-content fingerprints include `execution_params`
  and `execution_params_schema_version` in addition to scientific `params`,
  `params_schema_version`, `job_kwargs`, and sorter name. Because only
  `sorter_params_name` flows into `sorting_id`, changing execution backend for
  an existing logical row requires inserting a new `sorter_params_name`, not
  mutating the old row in place.
- A local MS4 row and a containerized MS4 row MUST use different
  `sorter_params_name` values. The container row is first-class provenance; it
  is not a temporary runtime override.
- Container flags are NOT sorter scientific kwargs and NOT SI job kwargs.
  `docker_image` / `singularity_image` must not be stored inside the sorter
  `params` blob, and must not be stored in `job_kwargs`. This is a global
  reserved-key rule across strict and permissive sorter schemas:
  `docker_image`, `singularity_image`, `delete_container_files`,
  `installation_mode`, `spikeinterface_version`, `spikeinterface_folder_source`,
  and `extra_requirements` are rejected from every scientific `params` blob,
  including generic / `extra="allow"` sorter rows.
- For `backend="local"`, `container_image` must be `None`; for
  `backend in {"docker", "singularity"}`, `container_image` must be an explicit
  non-empty image string or local `.sif` path. Avoid `True` as provenance even
  though SI accepts it.
- `execution_params` are resolved in `make_fetch` and passed to the DB-free
  sorter dispatch alongside `params` and resolved `job_kwargs`.
- Preflight checks the selected backend: local rows check local sorter runtime
  availability; Docker rows check Docker + the Python `docker` package;
  Singularity rows check Singularity + `spython`. Missing container runtime is a
  clear selected-preset error, not a silent fallback to local.
- MATLAB-backed legacy sorters (`kilosort2_5`, `kilosort3`, `ironclust`) must not
  silently run with `backend="local"`. A row for one of these sorters must either
  carry an explicit container backend, or preflight / dispatch raises a clear
  error explaining that Phase 3a removed the old name-based
  `singularity_image=True` fallback in favor of tracked execution provenance.

Dispatch contract:
- `run_si_sorter` builds `run_kwargs` for `sis.run_sorter`. For local rows it
  passes no container kwargs. For Docker rows it passes
  `docker_image=container_image`; for Singularity rows it passes
  `singularity_image=container_image`; for either container mode it passes
  `delete_container_files`, `installation_mode`, `spikeinterface_version`, and
  `extra_requirements` only when set/meaningful. SI's public `run_sorter`
  signature names only `delete_container_files`; the install controls reach
  `run_sorter_container` through `**sorter_params` after the container branch is
  selected. In Spyglass they still originate ONLY from `execution_params`, never
  from the scientific sorter `params` blob.
- The existing external float64 whitening path still runs before SI's
  `run_sorter` call. The resulting SpikeInterface recording must remain
  serializable for SI's container runner.
- The current MATLAB-sorter carve-out (`kilosort2_5`, `kilosort3`, `ironclust`
  defaulting to Singularity and stripping non-container-safe kwargs) becomes an
  explicit execution-policy check: no automatic name-based Singularity fallback,
  but local execution for those sorters errors clearly unless the row's
  `execution_params` selects Docker/Singularity. Any kwarg-strip behavior that SI
  still requires remains conditional on the selected container backend.
