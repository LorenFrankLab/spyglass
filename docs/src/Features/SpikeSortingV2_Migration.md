# Spike Sorting v1 → v2 Migration Guide

A short, task-oriented guide for notebook users porting a v1 spike-sorting
workflow to `spyglass.spikesorting.v2`. For the exhaustive, source-linked
list of every breaking change, see the
[CHANGELOG](../CHANGELOG.md) "Spike Sorting v2 — v1→v2 migration
reference" subsection; this page covers the deltas you actually touch in a
notebook. For the pipeline overview, see
[Spike Sorting v2](./SpikeSortingV2.md).

## 1. What you call differently

- **Parameter rows are named differently — no back-compat aliases.** The
  June 2026 catalog correction renames every Frank-lab row to a dated,
  content-stable name and drops the brief v1-name alias shim, so old strings
  no longer resolve — update them:
    - `PreprocessingParameters`: v1's single `default` (and v2's interim
      `default_franklab`) → `default`; the production region recipes are
      `franklab_hippocampus_2026_06` (600 Hz high-pass) and
      `franklab_cortex_2026_06` (300 Hz).
    - `SorterParameters` (MountainSort4): the region-encoded
      `franklab_tetrode_hippocampus_30kHz_ms4` / `franklab_probe_ctx_30kHz_ms4`
      rows — and v1's bare `franklab_tetrode_hippocampus_30KHz` /
      `franklab_probe_ctx_30KHz` — → the rate-keyed
      `franklab_30khz_ms4_2026_06` / `franklab_20khz_ms4_2026_06`. MS4 runs
      `filter=False`, so the row is region-agnostic; the high-pass band lives
      on the preproc row, not the sorter row.
    - `SorterParameters` (other): `franklab_tetrode_hippocampus_30kHz_ms5` →
      `franklab_30khz_ms5_2026_06`; the `clusterless_thresholder` row
      `default_clusterless` → `default`; the `kilosort4` row `default` →
      `franklab_neuropixels_default`.
    - `ArtifactDetectionParameters`: the production rows are
      `franklab_100uv_p07_2026_06` / `franklab_50uv_p07_2026_06`; the 500 µV
      schema default keeps the name `default`.

  Rows are keyed by the `(sorter, sorter_params_name)` pair, so a bare
  `"default"` is unambiguous per sorter.
- **`apply_merge` is singular.** `CurationV2.insert_curation(...,
  apply_merge=True)` (v1 used `apply_merges`).
- **Filter fields are `freq_min` / `freq_max`** (v1:
  `frequency_min` / `frequency_max`), nested under `bandpass_filter` in
  the preprocessing schema.
- **Sort-group referencing inherits the configured reference by default**
  (matching v1). `SortGroupV2.set_group_by_shank` and
  `set_group_by_electrode_table_column` now read each group's
  `Electrode.original_reference_electrode` and map it per group to a
  `reference_mode` — `-1` / `None` → `"none"`, `-2` → `"global_median"`,
  a non-negative id → `"specific"` (that electrode). This replaces the
  earlier v2 default of no reference (`"none"`). Override knobs:
  `set_group_by_shank` takes v1's per-group `references={electrode_group:
  ref_id}` dict back, and both helpers accept a call-wide
  `reference_mode=` (with `reference_electrode_id=` for `"specific"`) that
  forces one mode on every group; the two are mutually exclusive.
  **Three things now fail loud at group creation that v1 tolerated:**
  electrodes in one group with *mixed* configured references raise instead of
  silently mis-referencing (v1 built a `ValueError` but never raised it); a
  `"specific"` reference that is itself a member of the sort group raises (it
  would be subtracted then dropped, silently shrinking the group) — use
  `omit_ref_electrode_group=True` or a cross-group reference; and a
  `"specific"` reference electrode that does not exist in the session, or
  whose owning electrode group is ambiguous (the same `electrode_id` under
  two electrode groups), raises here instead of failing later inside
  `Recording.populate`.
- **Porting a non-curated sorter by name?** Call the opt-in helper once:

  ```python
  from spyglass.spikesorting.v2.sorting import SorterParameters

  SorterParameters.insert_default()                  # curated v2 rows
  SorterParameters.insert_default_legacy_si_sorters()  # v1 back-compat rows
  ```

  This inserts `('<sorter>', 'default')` rows for installed SI sorters
  outside v2's curated set (e.g. `kilosort2_5`), replicating v1's
  auto-insert. It is **opt-in** — `initialize_v2_defaults()` does not call
  it, so users who do not need v1 sorter names pay nothing.

## 2. What you query differently

- **No `recording_id`-keyed `IntervalList` row.** v2 stores the
  valid-times range on the `Recording` row. Reconstruct the old shape
  with:

  ```python
  row = (Recording & {"recording_id": rid}).fetch1()
  valid_times = np.asarray([[row["saved_start"], row["saved_end"]]])
  ```

- **Artifact `IntervalList` names are prefixed `artifact_detection_{uuid}`**
  (v1 used a bare UUID string). Use the helper instead of string-munging:

  ```python
  from spyglass.spikesorting.v2.utils import (
      artifact_detection_interval_list_name,
      parse_artifact_detection_interval_list_name,
  )
  ```

- **`Sorting.time_of_sort` is a `datetime`**, not a Unix-epoch int.
  Comparisons against `int(time.time())` must cast.

## 3. What's faster, safer, or more reproducible

- **Chunked artifact detection.** v2 runs a memory-bounded
  `ChunkRecordingExecutor` pass (controlled by
  `ArtifactDetectionParameters.job_kwargs`, default `chunk_duration='1s'`,
  `n_jobs=1`) instead of loading the full trace array into RAM. Output is
  frame-identical to the old path.
- **Hash-verifiable Recording rebuild.** The preprocessed `Recording`
  cache carries an `NwbfileHasher` `cache_hash`.
- **Pinned SpikeInterface (`==0.104.3`) + KS4/MS5 snapshot tests.** A SI
  version bump that would change a sorter's `extra="allow"` defaults
  surfaces as a deliberate, audited test failure.
- **Analyzer-folder disk-leak audit.**
  `Sorting.find_orphaned_analyzer_folders(dry_run=True)` surfaces 5–50 GB
  on-disk leaks from delete-override bypass.

## 4. What's not there yet

Use the v1 chain in the interim; the parent-plan phase that delivers each
v2 port is noted. How the gap surfaces depends on the feature:

- **Stubbed v2 modules** — `metric_curation`, `figpack_curation`,
  `unit_matching`, and `matcher_protocol` exist but raise an informative
  `ImportError` on any public-name import, naming the v1 fallback where
  one exists.
- **No v2 module yet** — `BurstPair` and `RecordingRecompute` have no
  `spikesorting.v2` counterpart; import them from `spyglass.spikesorting.v1`
  directly (importing a `spikesorting.v2.*` path for them is a plain
  `ModuleNotFoundError`).
- **Gated methods** — `ConcatenatedRecording` / `SessionGroup` exist in
  v2 but their unimplemented methods raise `NotImplementedError`.

| Feature | v1 fallback | v2 delivery |
| --- | --- | --- |
| Metric / auto-merge curation | `from spyglass.spikesorting.v1 import MetricCuration, MetricCurationParameters, WaveformParameters, MetricParameters` | AnalyzerCuration stage (roadmap) |
| FigURL curation views | `from spyglass.spikesorting.v1 import FigURLCuration, FigURLCurationSelection` | FigPack curation views (roadmap) |
| Burst-pair curation | `from spyglass.spikesorting.v1 import BurstPair, BurstPairParams, BurstPairSelection` | AnalyzerCuration stage (roadmap) |
| Recording recompute | `from spyglass.spikesorting.v1.recompute import RecordingRecompute, RecordingRecomputeSelection` | AnalyzerCuration stage (roadmap) |
| Concatenated recording / session group | (no v1 equivalent) | session-group concatenation (roadmap) |
| Cross-session unit matching (`UnitMatch`) | (no v1 equivalent) | cross-session unit matching (roadmap) |

## 5. What v1↔v2 comparisons WILL show

If you compare v1 and v2 outputs on the same input, expect these
**intentional, correct** differences (each is documented in the
[CHANGELOG](../CHANGELOG.md) v2 breaking-changes subsection):

- **v2 bandpass-filters BEFORE referencing** (v1 referenced first). The
  *reasoning*: the spatial common reference should be estimated from the
  band-limited spike signal, so out-of-band drift and DC offset are filtered
  out first and do not leak into the reference subtracted from every channel —
  the signal-processing-preferred order, an intentional divergence from v1.
  The two orders are *not* commutative only on the **`global_median` common
  reference** (the per-sample median is non-linear), so the preprocessed —
  and therefore sorted — output differs from v1 **only** for global-median
  sort groups. `specific`-electrode and `none` references (and a global
  *average* reference — `global_median` with `operator="average"`, where the
  mean is linear) commute with the filter, so they are numerically identical
  to v1.
- **Small spike-count delta near artifact-mask edges.** v2 fixes v1's
  off-by-one interval consolidation. A few spikes per disjoint interval
  boundary differ; v2 is correct.
- **Real differences on multi-channel clusterless sorts.** v1's
  `noise_levels=[1.0]` silently misread channels; v2 broadcasts to
  `n_channels`. v2 is the right answer.
- **Merged units may have slightly fewer spikes.** Merging contributors
  removes cross-unit double-detections within 0.4 ms (one physical spike
  detected in two units; safe because a neuron's refractory period
  forbids genuine sub-0.4 ms firing). v2 applies this on both the stored
  (`apply_merge=True`) and previewed (`get_merged_sorting`) trains; v1
  only deduped its lazy preview, so its *stored* merged trains kept the
  duplicates. v2's lower count is correct.
- **Disjoint (multi-interval) sorts: no obs/valid interval spans a gap.**
  v2 splits artifact-removed valid_times and no-artifact obs_intervals at
  the recorded-chunk boundaries, so observation durations and firing-rate
  windows exclude the inter-interval wall-clock gaps (v1/early-v2 could
  report a single gap-spanning envelope).
- **KS4 may differ after a SpikeInterface version bump** — caught by the
  pinned-version snapshot test rather than appearing silently.
- **Seed pinning improves preprocessing reproducibility, but MS4/MS5/KS4
  are not exact oracles.** v2 pins SI's whitening and noise-level seeds,
  which removes the run-to-run drift those steps introduced. The sorters
  themselves (MS4's `isosplit`, MS5, KS4) remain non-deterministic and
  must **not** be used as an exact rerun or v1↔v2 parity oracle — bound
  any comparison by qualitative metrics (unit-count order, firing-rate
  distribution shape), not spike-by-spike equality. The deterministic
  `clusterless_thresholder` path is the tight parity reference.
