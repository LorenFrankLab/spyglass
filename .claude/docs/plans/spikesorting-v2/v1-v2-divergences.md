# v1 → v2 behavioral & implementation divergences

[← back to PLAN.md](PLAN.md) · [feature-parity.md](feature-parity.md) · [parity-extensions.md](parity-extensions.md)

A catalog of the places where the v2 spike-sorting pipeline deliberately
behaves or is implemented differently from v1, with the v1 file:line anchors
that motivated each choice. These notes were extracted from v2 source
comments/docstrings during the docstring-cleanup pass (the code now states each
behavior on its own terms, without the v1 archaeology); this doc preserves the
v1→v2 forensic detail in one discoverable place rather than scattered across
inline comments a fresh reviewer can't follow.

Scope note: this is distinct from
[divergence-investigation.md](divergence-investigation.md), which is a focused
investigation of one specific numerical discrepancy (the clusterless
`detect_peaks` spike-time divergence). This file is the broad catalog.

---

## Cross-cutting (recur across multiple modules)

- **Absolute-time → frame inverse uses `searchsorted`, not an affine map.**
  Spike times are persisted in the recording's absolute wall-clock timeline.
  The inverse map back to frame indices uses `np.searchsorted` against the
  actual (gap-preserving) timestamp vector — **not** an affine
  `round((t - t_start) * fs)`. On disjoint sort intervals the timeline is
  non-uniform; the affine form (SI's `NwbSortingExtractor` default) shifts
  every post-gap frame by the accumulated gap and can push frames past the
  gap-excluded sample count. v1: `spike_times_to_valid_samples`
  (`v1/sorting.py:29`) + `v1/curation.py` `get_sorting`. v2: `_signal_math.py`
  `_spike_times_to_frames`, `_units_nwb.numpysorting_from_abs_times`.
  Out-of-bounds handling also differs: where a frame lands one past the end,
  v1 **dropped** the spike (`v1/sorting.py:60-79`) while v2 **snaps** it back to
  the last sample within a ~2-sample tolerance and **raises** beyond that
  (count-preserving, so per-spike features stay aligned). See the
  "boundary-spike handling" non-parity note in
  [feature-parity.md](feature-parity.md).

- **Clusterless threshold is a true microvolt threshold.** v1 carried a
  `threshold_unit="uv"` label (`v1/sorting.py:177`) but never enforced it — it
  thresholded in raw ADC counts. v2 scales the recording to µV (via the stored
  NWB gain/offset, `noise_levels=[1.0]`) before `detect_peaks`. No-op for
  unity-gain Frank-lab data (1 µV/count); corrective for non-unity-gain rigs
  (e.g. Intan ~0.195 µV/count, where a "100" threshold would otherwise mean
  ~19.5 µV) — a silent numerical divergence for any non-Frank-lab v1 user.

- **Preprocessing order: bandpass-filter THEN reference.** v1 referenced first,
  then filtered (`v1/recording.py:597-671`). v2 filters first. Because the
  per-sample `global_median` common reference is non-linear, v2 and v1 differ
  **numerically** on the `global_median` branch; on the `specific` / `none`
  (linear) paths the two orders commute and output is identical. v2 chose
  bandpass-before-reference as the signal-processing-preferred order.

---

## Recording stage

- **Heterogeneous gain/offset is rejected, not silently coerced.** v1 picked
  `gains[0]` and dropped channel offset entirely when channels had
  heterogeneous gain/offset, corrupting the µV↔V scaling. v2's resolver rejects
  heterogeneous gain/offset and non-positive gain. (`write_nwb_artifact`,
  `resolve_conversion_and_offset`.)

- **ElectricalSeries offset is honored on readback.** Dropping the offset
  (inherited from v1) silently biased every channel by `offset` µV for
  recordings with a non-zero DC offset (Intan / Open Ephys).
  (`_nwb_metadata_helpers.resolve_conversion_and_offset`.)

- **Rerun does not silently destroy downstream rows.** v1's
  `set_group_by_shank` silently dropped and re-created sort groups on rerun
  (and silently padded `sort_group_id`s), cascading deletes through every
  downstream Sorting and Curation row without warning. v2 inspects before
  destroying: it refuses by default and requires explicit non-overlapping ids
  or `delete_existing_entries=True, confirm=True`. (`SortGroupV2`,
  `_handle_existing`.)

- **Common-reference `operator` is a user knob.** v1 hardcoded
  `operator="median"` (`v1/recording.py:619`). v2 exposes `operator`
  (`"average"` is v2-only) and drops v1's `reference` params field; default
  `"median"` preserves v1 behavior. (`CommonReferenceParams`.)

- **A reference electrode that is also a sort-group member fails loud.** v1
  silently dropped it via `setdiff1d`; v2 raises.
  (`_reference_resolution.assert_reference_not_member`.)

- **Mixed `original_reference_electrode` across members fails loud.** On the
  auto path v1 silently fell through when members carried mixed reference
  values; v2 raises. Sentinel encoding (`None` / `-1` / `-2` / `>=0`) stays
  v1-compatible. (`_reference_resolution.resolve_group_reference`.)

- **Electrode-table-region is built from row indices, not channel ids.** v1
  passed `region=[i for i in recording.get_channel_ids()]`
  (`v1/recording.py:863`) to `create_electrode_table_region`, which interprets
  `region` as ROW INDICES into the electrodes table — so on a non-contiguous
  electrodes table v1 silently selected the wrong rows (wrong channel locations
  and brain-region attribution). v2 maps electrode ids → row indices and raises
  on an unknown id. (`_nwb_metadata_helpers.electrode_table_region`.)

- **Curated bad channels can be interpolated, not only removed.** v2 adds a
  `bad_channel_handling` knob (`"remove"` | `"interpolate"`, default `"remove"`,
  which is byte-identical to v1) controlling how `Electrode.bad_channel` flags
  are applied at materialization; `"interpolate"` re-includes the group's
  pitch-adjacent interior channels via SI kriging instead of dropping them. v1
  had no interpolation path. (`_params/preprocessing.PreprocessingParamsSchema`,
  `_recording_preprocessing.py`.)

- **Automated bad-channel detection is available (suggest-then-confirm).** v2
  adds `detect_bad_channels` (a thin wrapper over SI's `detect_bad_channels`,
  pinning only `method="coherence+psd"` and running per shank) and
  `suggest_bad_channels`, which can persist labels onto `Electrode.bad_channel`
  via `Electrode.update1`. No v1 equivalent; detection ships SI's defaults
  (`seed=None`, so the suggest step is non-deterministic). (`bad_channels.py`.)

- **Optional ADC `phase_shift` step.** v2 can apply SI `phase_shift` when the
  recording carries `inter_sample_shift` (multiplexed-ADC rigs / Neuropixels),
  warning-and-skipping otherwise; off by default. No v1 equivalent.
  (`_recording_preprocessing.py`.)

- **`ElectricalSeries.filtering` provenance reflects the steps actually run.**
  v1 hardcoded `filtering="Bandpass filtered for spike band"`
  (`v1/recording.py:877`) regardless of what ran; v2 builds the string from the
  preprocessing steps actually applied, so referencing / phase-shift /
  interpolation are distinguishable in provenance (and the string is hashed
  into the recompute content hash).
  (`_recording_preprocessing.filtering_description`.)

---

## Artifact-detection stage

- **`base_intervals` derived from the timestamp vector, split at gaps.** v1
  subtracted artifacts from the explicit `sort_interval_valid_times`
  (`v1/artifact.py:327`). v2 derives base intervals from the recording
  timestamps split at wall-clock discontinuities, so an artifact complement
  built per chunk never spans a gap. (`detect_artifacts`.)

- **Span join and removal window are frame-based, not time-based.** v1's join
  and removal window operated in timestamp space (`timestamps[end] + half_win`
  seconds), so a large disjoint gap was never bridged. v2 joins in frame space
  with an explicit gap-aware guard, and caps the removal window at the chunk
  frame bounds, to reproduce that behavior across gaps. (`detect_artifacts`.)

- **Detectors OR-combine; `min_length=1`s.** Amplitude and z-score detectors
  are OR-combined (AND would make dual-threshold strictly less sensitive than
  either single mode); v1 did the same (`spikesorting/utils.py:198`). The
  1-second minimum artifact length matches v1's hardcoded value
  (`v1/artifact.py:327-328`).

- **Defaults:** `amplitude_threshold_uv=500.0` is a v2 bug-fix default (a v1
  unit-conversion bug is documented in the CHANGELOG; 500 µV matches v1's
  effective Intan-probe behavior within ~15%). `proportion_above_threshold=1.0`
  matches v1's principled "all channels" default (v1 also had an undocumented
  0.5 variant with no justification). Migration: translate a raw v1 count via
  `raw_count_value * probe_gain_uV_per_count`. (`_params/artifact_detection.py`.)

- **Worker kernels live in a DB-free module.** The chunked-scan worker
  functions are a self-contained, DB-free copy (numpy + spikeinterface only) so
  multiprocessing spawn-workers need no DB connection. v1's equivalents
  (`spikesorting/utils.py` `_init_artifact_worker` / `_compute_artifact_chunk`)
  do not themselves touch DataJoint, but live in a module that imports DataJoint
  at module scope, so a spawn-worker that imports them drags in a DB dependency.
  `si.load` is the SI 0.104+ rename of v1's `load_extractor`.
  (`_artifact_compute.py`.)

- **Per-channel z-score is epsilon-guarded and computed in µV.** v1 applied a
  bare `scipy.stats.zscore` (`spikesorting/utils.py:185`, `:193`), so a flat
  (zero-variance) frame produced `nan`/`inf`; v2 computes the z-score manually
  on the µV-scaled traces with `std(...) + 1e-12` (`_artifact_compute.py:133`),
  so a near-flat frame yields ≈0 and is not flagged. The flagged-frame set
  differs on saturated/constant frames. (z-score is scale-invariant, so the µV
  scaling alone does not change it.)

- **`min_length` boundary is inclusive (`>=`), not strict (`>`).** v1's
  interval-length filter kept `lengths > min_length`
  (`common/common_interval.py:540`); v2 keeps `(end - start) >= min_length_s`
  (`_artifact_intervals.py:377`). An interval of exactly `min_length` (e.g.
  1.0 s) is dropped by v1 but kept by v2.

- **Default `chunk_duration` / `n_jobs` lowered.** v1 defaulted
  `chunk_duration="10s", n_jobs=4` (`v1/artifact.py:73-74`); v2 defaults a
  1-second chunk and `n_jobs=1`, resolving concurrency from job_kwargs instead.
  Performance/footprint only. (`_artifact_intervals.py`.)

---

## Sorting stage

- **`obs_intervals` = artifact-removed valid times.** v1 wrote the
  artifact-removed window as `obs_intervals` on each `add_unit`
  (`v1/sorting.py:597`); without it, firing-rate computations treat units as
  observed across the full session even where the artifact mask blanked the
  signal. v2 keeps this, and reconstructs `obs_intervals` from the recorded
  timestamp envelope (split at discontinuities) when the optional artifact pass
  is absent. (`Sorting.make_fetch`, `_units_nwb.write_sorting_units_nwb`.)

- **External float64 whitening; sorter internal whitening disabled.** v1 ran
  external whitening (`v1/sorting.py:440-444`) and disabled the sorter's
  internal whitening to avoid double-whitening. v2 keeps this and runs it after
  the artifact mask so masked frames don't bias the covariance estimate. v2 also
  pins the whitening/MAD `random_seed` (default 0) so the whitening matrix is
  reproducible by default, where v1's `sip.whiten` (`v1/sorting.py:443`) was
  unseeded — a v2 run differs numerically from an unseeded v1 run.
  (`run_si_sorter`, `_sorting_dispatch.py:217-218`, `:380-383`.)

- **Default-row install gate uses `installed_sorters()`.** v1's gate
  (`v1/sorting.py:184-189`) and the legacy-SI analog could auto-insert default
  rows for wrapper-only sorters whose binary is absent (kilosort2_5, ironclust),
  which then fail at populate with "sorter not installed". v2 gates on
  `installed_sorters()`. (`SorterParameters.insert_default`,
  `insert_default_legacy_si_sorters`.)

- **`peak_sign` threaded at sort time.** v1 made detection polarity
  configurable via `peak_channel` metric params; v2 threads `peak_sign`
  (clusterless `peak_sign` / MountainSort `detect_sign`) at sort time instead of
  relying on SI's `"neg"` default. (`Sorting._populate_unit_part`.)

- **`time_of_sort` is a native DataJoint `datetime`.** v1 stored a Unix-epoch
  int (`v1/sorting.py:239`) as a DataJoint-type workaround no longer needed.

- **Sorter params:** MountainSort4/5 drop v1's runtime `tempdir` field-mutation
  hack; MS4 `freq_min/freq_max` schema defaults adopt v1's tetrode preset
  (`v1/sorting.py:158-159`), not the bare `mountain_default` block
  (`v1/sorting.py:145-153`) that omitted those keys. Kilosort4 / generic-SI
  schemas keep `extra="allow"` to preserve v1's "try any installed SI sorter"
  escape hatch. (`_params/sorter.py`.)

- **Artifact-mask boundary frames differ by one sample.** v1 zeroed
  inter-interval and final-interval frames with `np.arange(end+1, next_start)`
  and `np.arange(last_end, len(timestamps)-1)` (`v1/sorting.py:382-388`),
  leaving the boundary sample at each artifact edge un-zeroed; v2 derives the
  mask from `searchsorted` bounds (`end=len(timestamps)`,
  `_sorting_artifact_mask.py`), so the exact set of zeroed frames at each
  artifact boundary differs. Spikes landing on a boundary sample can be masked
  in one and not the other.

- **Per-unit peak channel and `peak_amplitude_uv` are computed and stored at
  sort time.** v2 records each unit's extremum channel and template amplitude in
  µV on the `Sorting.Unit` part (`sorting.py:904`, `:1952-1992`) via
  `template_tools.get_template_extremum_{channel,amplitude}`. v1 had no
  equivalent per-unit field at sort time.

- **Global / `dj.config` job-kwargs flow into the sorter and analyzer.** v2
  resolves `n_jobs` / `chunk_duration` from SI global ← `dj.config` ← per-row
  blob and threads them into the sorter run and analyzer extension compute
  (`utils._resolved_job_kwargs`, `_sorting_dispatch.py`); v1 honored only the
  row's worker count.

---

## Curation stage

- **Merge groups stored as queryable part rows.** v1 stored merge groups as a
  single list-of-lists `merge_groups` blob column on the units NWB; that lost
  bulk-audit/provenance queryability. v2 stores them as `MergeGroup` part rows
  (one kept-unit/contributor pair per row). (`CurationV2.MergeGroup`.)

- **Merged-unit id = `max + 1` in ascending min-contributor order.** v1
  assigned fresh merged ids in user-iteration order
  (`v1/curation.py:359`/`:361`). v2 assigns the fresh `max(source ids) + 1` in
  ascending min-contributor order, so the `apply_merge=True` path and the lazy
  `get_merged_sorting` preview (which reads `MergeGroup` ordered by `unit_id`)
  assign the SAME fresh id to the SAME content group even when groups are listed
  out of min order. The integer id is an arbitrary label, so only
  id-assignment-for-reordered-input differs. (`build_curated_unit_rows`.)

- **`curation_label` column shape: scalar pre-curation, ragged post-curation.**
  Pre-curation NWB writes `curation_label` as a scalar `"uncurated"`
  (v1 shape at `v1/sorting.py:583-598`); `insert_curation` rewrites it to the
  indexed ragged-list shape post-curation (v1 shape at
  `v1/curation.py:398-403`). The write pattern (`add_unit(...)` first, then
  `add_unit_column(..., index=True)`) is required because the external reader
  (`v1/figurl_curation.py:83-101`) expects a per-unit list; a comma-joined
  string would be split per character, and pre-declaring the column makes pynwb
  fail dtype inference on the all-empty case. (`write_sorting_units_nwb`,
  `write_curated_units_nwb`.)

- **Strict-mode restriction rejects unknown keys.** v1 silently dropped unknown
  restriction keys, quietly returning wrong-but-non-empty result sets (a
  restriction that looked applied but wasn't). v2 raises `ValueError` in strict
  mode so a typo is caught. (`resolve_restriction`.)

- **Brain-region attribution covers every electrode.** v1's `fetch(limit=1)`
  under-reported regions for a multi-region probe (it returned one electrode's
  region). v2 returns a relation over every electrode in the sort group so a
  multi-region probe surfaces every represented region. (`get_sort_group_info`.)

- **Merged-unit spike trains are de-duplicated at 0.4 ms.** v1 formed a merged
  unit by a bare `np.concatenate` of contributor trains (`v1/curation.py:362`);
  v2 applies a membership-aware 0.4 ms cross-unit dedup (`_units_nwb.py:452`,
  `_signal_math._dedup_merged_spike_times`) so a spike double-detected on two
  contributors is counted once. A v2 merged unit therefore has fewer spikes than
  the contributor sum, changing its `num_spikes` / firing-rate / ISI. (Distinct
  from the clusterless `detect_peaks` divergence in
  [divergence-investigation.md](divergence-investigation.md).)

- **Merge groups are validated, raising on malformed input.** v2 rejects empty,
  singleton, duplicate-member, unknown-id, and overlapping merge groups
  (`_curation_transforms.py`); v1 silently concatenated/unioned whatever was
  passed (`v1/curation.py:359-455`).

- **Re-inserting a root curation with conflicting params fails loud.** v1
  silently returned the existing key, ignoring the new labels/merges
  (`v1/curation.py:88-93`); v2 raises `ValueError` when a root curation already
  exists for the sorting unless an explicit override is set
  (`curation.insert_curation`).

---

## Metric / analyzer-curation stage

v2 consolidates v1's `MetricCuration` + `BurstPair` + `metric_utils` into a
single `AnalyzerCuration` table built on the SI 0.104 `SortingAnalyzer` API
(CHANGELOG; v1 `MetricCuration` `v1/metric_curation.py:253`, `BurstPair`
`v1/burst_curation.py:109`, v2 `AnalyzerCuration` `v2/metric_curation.py:598`).

- **`isi_violation` is Spyglass's bounded fraction, recomputed from SI's raw
  count.** v2 reproduces v1's `count / (n_spikes - 1)` fraction
  (`v1/metric_utils.py:32-38`), NOT SI 0.104's unbounded `isi_violations_ratio`
  (Hill/UMS2000 estimate): it pulls SI's raw `isi_violations_count` column and
  recomputes the fraction, hardening the edge cases v1 left (≤1-spike → NaN,
  avoiding the `0/0` 1-spike case and the spurious `(-1)/(0-1)=1.0` 0-spike
  artifact). The default `isi_threshold_ms` also moved 1.5 → 2.0.
  (`_metric_curation.isi_violation_fraction`, `metric_curation._compute_metrics`;
  v1 default `v1/metric_curation.py:142`.)

- **The SI 0.99→0.104 nn-metric rename is guarded, not silently resolved.**
  Requesting `nn_isolation` / `nn_noise_overlap` as metric *names* raises with a
  targeted hint, because in SI 0.104 they are the two output *columns* of the
  single `nn_advanced` PCA metric. v1 resolved the old names lazily via
  `getattr(sq, name, None)` under the legacy SI-0.99 env
  (`v1/metric_curation.py:51-52`). The v2 default rows request `nn_advanced`
  (`skip_pc_metrics=False`) and the auto-curation rule thresholds the resulting
  `nn_noise_overlap` column. (`_params/metric_curation._check_metric_names`.)

- **Params are pydantic-validated at insert, not free-form blobs.** v1's
  `MetricParameters` / `MetricCurationParameters` stored unvalidated dicts
  (`v1/metric_curation.py:119`, `:184`). v2's `QualityMetricParameters` validates
  `metric_names` against the installed SI's `get_quality_metric_list()` and
  rejects orphan `metric_kwargs` keys at insert; concurrency lives in a per-row
  `job_kwargs` blob resolved at populate, not on the schema.
  (`_params/metric_curation.QualityMetricParamsSchema`.)

- **The quality-metric table also carries waveform-shape (template) metrics —
  exposed, not thresholded.** `get_metrics` surfaces SI template (waveform-shape)
  output *columns* next to the quality metrics so downstream code can classify
  cell types (firing rate × spike width: narrow/fast putative interneuron vs
  wide/slow pyramidal) with its own region-specific cutoffs. The surfaced set is
  a per-row `template_metric_columns` (SI output COLUMN names, not metric names,
  so `half_width` is selected as its `trough_half_width` column with no
  name→column ambiguity); the shipped default is the single trough-local
  `trough_half_width`. `peak_to_trough_duration` and the slope columns are
  opt-in, not defaults: they measure to the post-trough repolarization peak,
  which clips at the deliberately narrow hippocampus display window
  (`ms_after=0.5`) and is reliable only on the wider 1.0/2.0 window. The pipeline
  ships NO cell-type thresholds and NO classifier (the legacy `cellinfo.type`
  step was always manual/external); the columns are read from the unwhitened
  DISPLAY analyzer (whitening distorts shape, like `snr`). (`metric_curation.
  QualityMetricParameters.template_metric_columns`, `_compute_metrics._surface_
  template_columns`.)

- **`AutoCurationRules` is an ordered, queryable threshold-rule engine.** v1
  packed labeling into a `label_params` blob keyed by metric, one
  `[op, threshold, labels]` triple each, applied in a loop with bugs
  (`v1/metric_curation.py:523-576`). v2 splits an `auto_merge_preset` master from
  ordered `Rule` part rows (`rule_index`, `metric_name`, `operator`, `threshold`,
  `label`), adds the `!=` operator, applies rules in index order, de-dupes
  labels, gives each unit an independent list, and **raises** on a rule that
  references an absent metric column (v1 issued a no-op `Warning(...)` call —
  `v1/metric_curation.py:561-563`). Fixes the three #1513 label-engine bug
  classes. (`metric_curation.AutoCurationRules`,
  `_metric_curation.apply_label_rules`.)

- **Merge suggestion is SI `compute_merge_unit_groups` presets.** v1's real
  merge engine was `BurstPair` (per-unit-pair waveform Pearson correlation,
  cross-unit ISI, cross-correlogram asymmetry — `v1/burst_curation.py:261-305`);
  `MetricCuration._compute_merge_groups` (`v1/metric_curation.py:578-631`) was
  dormant-by-default and buggy. v2 delegates entirely to SI's preset set
  (`similarity_correlograms`, `temporal_splits`, `x_contaminations`,
  `feature_neighbors`, `slay`, + the `"none"` skip sentinel), validated at insert.
  v2 precomputes the extra `spike_locations` extension required by SI's
  `feature_neighbors` / `knn` path and passes job kwargs flat to
  `compute_merge_unit_groups`, matching the SI 0.104 API rather than nesting
  them under a `job_kwargs` parameter.
  (`metric_curation._compute_merge_groups`, `_params/metric_curation.AutoMergePreset`.)

- **Three NWB scratch tables, not one units table.** v1 wrote a single `units`
  table carrying waveforms, per-metric columns, a ragged `curation_label`, and a
  `merge_groups` column (`v1/metric_curation.py:634-724`). v2 writes
  `quality_metrics` (wide, NaN preserved on disk → `None` on read),
  `merge_suggestions` (long `(merge_group_index, unit_id)`), and `proposed_labels`
  (wide; the ragged `curation_label` column is added ONLY when ≥1 unit is labeled,
  sidestepping the #1625 empty-list-of-lists crash). Waveforms stay in the
  analyzer (not re-stored). A zero-unit sort writes three empty tables instead of
  crashing. (`_metric_curation_nwb.write_analyzer_curation_tables`.)

- **Auto-curation never silently writes a curation.** v1's
  `CurationV1.insert_metric_curation` was the natural follow-on insert
  (`v1/curation.py:131-161`). v2 requires an explicit
  `AnalyzerCuration.materialize_curation`, which forks a *child* `CurationV2`
  (`parent_curation_id` set, `curation_source="analyzer_curation"`, merge groups
  filtered to size ≥2) and warns if run on a row already produced by
  auto-curation (metrics over post-merge templates).
  (`metric_curation.materialize_curation`.)

- **The default auto-curation rule set encodes the lab's ~2% ISI policy.** The
  shipped `franklab_default_auto_curation_2026_06` rule set
  (`AutoCurationRules._default_payloads`) thresholds two metrics in order:
  `nn_noise_overlap > 0.1 -> noise` and `isi_violation > 0.02 -> reject`, so a
  unit with >2% refractory violations is removed from matchable-unit outputs
  (`CurationV2.get_matchable_unit_ids` excludes `reject`/`noise`/`artifact` by
  default). The ISI rule deliberately labels `reject`, not `mua`: `mua` is kept
  by the default matchable policy, `reject` is not. The older nn-only
  `v1_default_nn_noise`, the `similarity_merge` auto-merge preset, and the inert
  `none` row are preserved as named rows. The set runs no auto-merge
  (`auto_merge_preset="none"`); merging stays a manual step in the loop below.

- **Curation is an auto → manual-merge → auto loop, and the second pass is not
  redundant.** (1) Run automatic curation (`AnalyzerCuration` →
  `materialize_curation`). (2) Manually curate/merge oversplit clusters
  (`CurationV2.insert_curation` with `merge_groups`, using
  `plot_by_sort_group_ids` / `investigate_pair_*` to spot burst pairs). (3) Run
  analyzer curation *again* on the merged curation for the **final** quality
  metrics and labels. Merging changes a unit's template (and therefore its SNR,
  ISI-violation fraction, and PC/NN separation), so metrics computed over the
  post-merge templates are the numbers of record — the pre-merge first pass
  could not see them. The default rule set is the one the user notebook drives;
  it is NOT wired into a `_PipelinePreset` (which carries no curation fields), so
  `run_v2_pipeline` stops at the root curation and auto-curation is an explicit,
  separately-driven step.

- **Content-addressed selection PK with raw-insert/integrity guards.** v1's
  `MetricCurationSelection` PK was a random `uuid4` per insert
  (`v1/metric_curation.py:247`). v2's `AnalyzerCurationSelection` PK is a
  deterministic `uuid5` of `(sorting_id, curation_id, metric_params_name,
  auto_curation_rules_name)`; `insert_selection` is the only sanctioned path, a
  raw insert is blocked, and a non-deterministic id for the same identity raises
  `DuplicateSelectionError`. (`metric_curation.AnalyzerCurationSelection`,
  `check_rule_integrity`.)

- **SNR is compared distributionally in the parity comparator.** v1 (SI 0.99,
  mean-based) and SI 0.104 (median-based) use different SNR definitions, so
  `compare_to_v1_baseline` checks the median v2/v1 ratio within tolerance with no
  order-of-magnitude per-unit outlier, rather than per-unit equality;
  `isi_violation`/`firing_rate`/`num_spikes` are exact-within-tolerance.
  (`_metric_parity.compare_to_v1_baseline`.)

- **Analyzer waveform window + subsample are now tracked, region-specific
  rows (no longer hardcoded).** v1's `WaveformParameters` default was
  `ms_before=0.5, ms_after=0.5, max_spikes_per_unit=5000`
  (`v1/metric_curation.py:101-103`). v2 restores DB-tracked waveform parameters
  as the `AnalyzerWaveformParameters` Lookup (mirroring v1's `WaveformParameters`)
  and resolves a **region-specific** display recipe from the sort's source
  preprocessing recipe: hippocampus returns to `ms_before=0.5, ms_after=0.5`
  (dense/tight spikes), cortex keeps the wider `ms_before=1.0, ms_after=2.0`
  (broader waveforms); the subsample is `max_spikes_per_unit=20000` for both
  (4× v1's 5000 — no longer reduced). The resolved recipe name is persisted on
  `Sorting.display_waveform_params_name` (a secondary FK to
  `AnalyzerWaveformParameters`, so the recipe is DB-enforced provenance — a sort
  cannot be populated against an untracked recipe, and a referenced recipe row
  cannot be deleted) and the analyzer cache folder is keyed by it
  (`{sorting_id}__{waveform_params_name}.zarr`). This still changes every
  template-derived metric (SNR, amplitude, peak channel) relative to v1 — and
  the larger 20000 subsample shifts those metrics for ALL sorts relative to the
  earlier hardcoded 500 — independent of, and compounding with, the documented
  SNR mean→median change. The window is determined by region (a property of the
  data/recipe), not a free per-sort knob, so `peak_amplitude_uv` stays
  deterministic for a `sorting_id`.

- **The analyzer is persisted as zarr, keyed by sort + recipe.** v2 creates the
  `SortingAnalyzer` with `format="zarr"` (`_sorting_analyzer.py`) and caches it
  at `{sorting_id}__{waveform_params_name}.zarr` (`_analyzer_cache.py`) — keyed
  by both the sort and the waveform recipe so a sort's display and
  whitened-metric analyzers never collide; v1 had no `SortingAnalyzer` (it used
  on-demand SI `WaveformExtractor` folders). The analyzer recompute trio
  (`SortingAnalyzerVersions`, keyed `(sorting_id, waveform_params_name)`) hashes
  selected extension arrays from this zarr store and rebuilds each recipe under
  its own key (display, or a whitened metric recipe a curation references), so
  the two analyzers for one sort verify independently.

- **A sort has TWO analyzers; quality metrics are routed by type (restored
  whitened/unwhitened split).** v1 split waveform extraction (`WaveformParameters`)
  from metrics; v2 builds an unwhitened DISPLAY analyzer at sort time and, for
  curation, a second WHITENED METRIC analyzer (built on demand). Each consumer
  reads the correct one: voltage / spike-train quality metrics (`snr`,
  `amplitude_cutoff`, `amplitude_median`, `firing_rate`, `num_spikes`,
  `presence_ratio`, `isi_violation`), amplitudes, all four BurstPair legs, and
  the merge engine read the **unwhitened display** analyzer (whitening
  normalizes per-channel variance, so SNR / amplitude / template-shape on
  whitened traces are meaningless and the `unit_locations` spatial gate would be
  miscalibrated); ONLY the PC / cluster-separation metrics (SI's
  `get_quality_pca_metric_list()`: `d_prime`, `mahalanobis`, `nearest_neighbor`,
  `nn_advanced`, `silhouette`) read the **whitened** metric analyzer, where
  decorrelated separation is meaningful. The whitened analyzer uses
  `return_in_uV=False` (so `sip.whiten`'s preserved per-channel gains are not
  re-applied, which would un-normalize the whitened space). The metric recipe is
  tracked on `AnalyzerCurationSelection.metric_waveform_params_name` (v1-parity:
  v1 attached `WaveformParameters` to `MetricCurationSelection`). **Expect the
  PC/NN metrics to shift** relative to the earlier single-analyzer v2 path: they
  now compute in the whitened space (their intended domain), a deliberate,
  content-addressed change, not a regression — re-curate against the new values
  rather than comparing absolute PC/NN scores across the boundary. Voltage /
  spike-train metrics are unchanged by this split (they always read the display
  analyzer).

- **BurstPair merge diagnostics are ported, computed on the fly (not stored),
  and each leg is reimplemented deliberately — not literally.** v1's `BurstPair`
  computed three per-pair scalars into a queryable `BurstPairUnit` table
  (`v1/burst_curation.py:114-123`, `:261-305`). v2 recomputes them on demand
  from already-stored analyzer extensions
  (`_metric_curation_plots.burst_pair_metrics_from_analyzer`, one dict per
  ordered pair) and exposes them through `AnalyzerCuration` (`get_peak_amps`,
  `plot_by_sort_group_ids`, `investigate_pair_xcorrel`,
  `investigate_pair_peaks`, `plot_peak_over_time`) plus
  `AnalyzerCurationSelection.insert_by_curation_id`. Nothing is stored in a new
  table: the heavy inputs (`templates`, `correlograms`, `unit_locations`)
  already live in the recompute-covered analyzer zarr, so the reductions are
  cheap deterministic views; a stored table would only add cross-session
  queryability, which nothing downstream consumes. Per-leg choices:
    - **`wf_similarity`** uses SI's **cosine** `template_similarity` extension,
      NOT v1's flat `pearsonr` over concatenated mean waveforms
      (`v1/burst_curation.py:292`). v1's flatten-and-correlate assumes both
      units share channels in the same order — wrong under v2's **sparse**
      analyzers (per-unit channel supports differ); SI's extension aligns
      sparsity and is the same metric the `similarity_correlograms` merge
      engine uses, so the diagnostic matches the suggestions.
    - **peak amplitudes** are sampled at the waveform **peak** (`nbefore`), not
      the array center: with the asymmetric (1.0 / 2.0 ms) window the center
      sits ~0.5 ms into the repolarization tail, not the trough. Per-channel
      `(n_spikes, n_channels)` shape is kept (the burst plots histogram per
      channel and pick each unit's max channel), so the 1-D `spike_amplitudes`
      extension is deliberately not used here. Amplitudes and their paired spike
      times both follow the `waveforms` extension's `random_spikes` subset (the
      times come from the `random_spikes` extension, not the full train), so the
      arrays stay aligned for units with more spikes than the subsample cap.
      (`_metric_curation_plots.peak_amplitudes_from_analyzer`.)
    - **cross-unit `isi_violation`** keeps v1's "violations / merged-train
      spikes" logic, but the threshold arg is renamed `isi_threshold_s` →
      `isi_threshold_ms` (v1 named it `_s` while applying it as ms via
      `* 1e-3`); the old `isi_threshold_s` name stays as a deprecated
      keyword-only alias so external callers don't break. v2 passes **2.0 ms**
      to align with the single-unit `isi_violation` default; the shared
      `utils_burst.calculate_isi_violation` default stays 1.5 ms so v1's
      positional callers are unchanged.
    - **`xcorrel_asymm`** is unchanged (`utils_burst.calculate_ca` over the
      `correlograms` extension; directional, so computed per ordered pair).
    - **`unit_distance`** is a NEW fourth leg — euclidean distance over the
      `unit_locations` extension — the spatial check v1 lacked (two cells can
      share a waveform shape at different depths; SI's own merge preset uses
      unit-location proximity for the same reason).

### Not yet ported from v1 (metric/analyzer curation)

- **BurstPair's per-pair metrics are computed on the fly, not stored.** The
  diagnostics themselves ARE ported (see "BurstPair merge diagnostics" above);
  what is intentionally not carried over is v1's queryable `BurstPairUnit` table
  — there is no stored `wf_similarity` / `xcorrel_asymm` / cross-unit
  `isi_violation` table, by design (the only thing it would add is cross-session
  queryability, which nothing downstream needs). Automated merge *suggestion*
  remains preset-only. (v1 `BurstPairUnit` `v1/burst_curation.py:114-123`,
  `:286-305`.)

- **`peak_offset` / `peak_channel` metrics dropped from the default set.** v2
  relies on SI analyzer extensions (`unit_locations`, `template_metrics`) instead.
  (`v1/metric_curation.py:53-54`, `v1/metric_utils.py:41-74`.)

- **On-demand waveform re-extraction (`fetch_all` / `overwrite`) is gone.** v1
  re-extracted via SI `extract_waveforms` with caching (`v1/metric_curation.py:355-418`);
  v2 exposes only the sort-time waveform subsample and warns that full
  re-extraction is unsupported. (`metric_curation.get_waveforms`.)

- **Whitened-vs-non-whitened waveform recipe is no longer a curation knob.** v1's
  `WaveformParameters.whiten` (`v1/metric_curation.py:68`, `:107-110`) is subsumed
  by the analyzer's already-computed base extensions plus `skip_pc_metrics`.

---

## Recompute stage

- **Content hashing, not whole-file NWB hashing.** v1 hashed the entire
  regenerated analysis file with `NwbfileHasher` (`v1/recompute.py:619-632`,
  `:771-786`). v2's stored `cache_hash` is a volatile whole-file digest (per-object
  `object_id` attrs, creation timestamps) that is NOT reproducible across
  regenerations, so v2 hashes only reproducible *content* — the preprocessed
  `ElectricalSeries` traces for recordings, and selected analyzer extension arrays
  for analyzers — each rounded to the selection's `rounding` decimals.
  (`_recompute.hash_recording_traces`, `_recompute` module docstring.)

- **Two recompute trios: recording + analyzer.** v1 covered only the recording
  artifact (`RecordingRecompute*`, `v1/recompute.py:54`, `:199`, `:537`). v2 adds
  a whole second trio for the `SortingAnalyzer` folder, hashing only the
  deterministic extensions (`random_spikes`, `templates`, `waveforms`) and
  **deliberately excluding the unseeded/stochastic `noise_levels`**.
  (`recompute.SortingAnalyzer{Versions,RecomputeSelection,Recompute}`,
  `_recompute.ANALYZER_RECOMPUTE_EXTENSIONS`.)

- **Current-environment delete gate (`StaleEnvMatchedError`).** v2 authorizes
  deletion at the artifact level only when a `matched=1, deleted=0` row exists in
  the *current* `UserEnvironment`; an artifact matched only under a stale env
  raises `StaleEnvMatchedError` unless `force_stale_env=True` (which audit-logs).
  A match under an older SpikeInterface pin is not evidence the current env can
  regenerate. v1 deleted any `matched=1, deleted=0` row regardless of env
  (`v1/recompute.py:893-948`). (`recompute._authorize_artifacts_for_deletion`,
  `exceptions.StaleEnvMatchedError`.)

- **Age gate on authorizing rows only; NULL `created_at` = refuse; disk dedup.**
  v2 applies the `days_since_creation` floor to only the authorizing current-env
  rows (so a stale/recent sibling can't block a valid artifact), treats a NULL
  `created_at` as too-recent, and dedupes reclaimable disk by file/sort so a
  multi-env artifact is counted once. v1 filtered on a single SQL date comparison
  and summed per row (`v1/recompute.py:873-914`). (`recompute._too_recent_or_unknown`,
  `_reclaimable_disk`.)

- **Regenerates to a fresh/temp path; records any failure as retryable.** v1
  rewrote the registered analysis file in place and unlinked the temp copy on
  match (`v1/recompute.py:725-786`). v2 always regenerates to a fresh,
  unregistered path (or a temp analyzer folder) and deletes it after hashing —
  it never compares the stored file to itself — and on any regeneration exception
  records a retryable `matched=0` with the traceback logged.
  (`recompute._compute_recording_artifact`, `RecordingArtifactRecompute.make`.)

- **`recheck` cautious-deletes + repopulates.** v1 re-hashed in place
  (`v1/recompute.py:788-803`); v2 does a cautious `delete(safemode=False)` (so the
  `Name`/`Hash` diff part rows cascade and the team-permission guard applies) then
  re-`populate`s. (`recompute.*Recompute.recheck`.)

- **Rounded content is hashed as raw float64 bytes, not decimal strings.** v1's
  `NwbfileHasher` rounded only `ProcessedElectricalSeries` and hashed the
  decimal-string representation (`utils/nwb_hash.py`); v2 rounds every hashed
  float array (recording traces and the selected analyzer extension arrays) and
  hashes `np.ascontiguousarray(array).tobytes()` (`_recompute.py:43-48`). A
  consequence is that `-0.0` and `0.0` (string-equal in v1) can hash differently
  in v2, so the `rounding` knob no longer guarantees the same match tolerance.

- **Rollup digest is sha256 while sub-hashes are md5.** v2's per-array/segment
  sub-hashes use md5 (`_recompute.py:43`, `:63`) but the combined digest uses
  sha256 (`_recompute.py:82`); v1 was md5 throughout. Storage detail only.

### Not yet ported from v1 (recompute)

- **v2 recompute is fully opt-in.** v1 auto-registered a recompute selection on
  every recording creation (`SpikeSortingRecording.make` →
  `RecordingRecomputeSelection().insert(..., at_creation=True)`,
  `v1/recording.py:262-268`). v2's recording/sorting `make` paths register
  nothing, the recompute tables are **not even exported from `v2/__init__.py`**,
  and selections are created only by an explicit `attempt_all`.
  (`recompute.*RecomputeSelection.attempt_all`.)

- **No pynwb-version pre-gating** of which selections to insert. v1 gated on a
  pynwb namespace-version match (`_has_matching_env` / `_required_matches` over
  core / hdmf-common / hdmf-experimental / ndx-franklab-novela,
  `v1/recompute.py:65-140`); v2 inserts a selection for every eligible
  artifact in the current env.

- **No xfail classification.** v1 pre-marked `xfail_reason` for known-failure
  patterns (missing probe info, pynwb dtype-kwarg API breaks, NWB spec
  mismatches — `_check_xfail`, `v1/recompute.py:364-448`). v2 keeps an
  `xfail_reason` column but nothing populates it; it instead catches any
  regeneration error at `make` time as a retryable `matched=0`.

- **No multi-rounding precision short-circuit.** v1 skipped lower-precision
  attempts once a higher-precision match existed
  (`_other_roundings` / `_is_lower_rounding`, `v1/recompute.py:684-712`,
  `:838-844`). v2 carries a single `rounding` per selection with no cross-rounding
  logic.

- **No `H5pyComparator` object-level diff** for inspecting a mismatch (v1
  `Hash.compare`, `v1/recompute.py:569-576`).

---

## Interactive curation viewer (figurl → figpack)

- **The FigPack web-curation viewer is not yet implemented.** v1's FigURL /
  kachery viewer is fully working: `FigURLCuration` builds a
  `SpikeSortingView` + `MountainLayout` (summary, units, raster, amplitudes,
  correlograms, average waveforms, electrode geometry, and the interactive
  `SortingCuration2` control) and round-trips `{labelsByUnit, mergeGroups}`
  through kachery back into
  `CurationV1.insert_curation` (`v1/figurl_curation.py:60-322`). v2's
  `figpack_curation.py` is an `ImportError` stub directing callers to the v1
  FigURL chain; `figpack` exists in v2 only as a placeholder
  `CurationV2.curation_source` enum value (`v2/curation.py:82`). v2's current
  interactive-curation substitute is the programmatic, rule-driven
  `AnalyzerCuration` path, analogous to v1's `MetricCuration`, not to the
  browser-based viewer. (`v2/figpack_curation.py`.)

- **Local SpikeInterface visualization/export bridge is new in v2.** Where v1
  exposed inspection mainly through the FigURL web viewer and `MetricCuration`'s
  bespoke plots, v2 adds a discoverable, key-aware facade
  (`spyglass.spikesorting.v2.visualization`, imported as `ssviz`) that wraps
  SpikeInterface's own `widgets.*` / `exporters.*` behind Spyglass DataJoint
  keys: `available_visualizations()` (the catalog), recording widgets
  (`plot_recording_traces` / `plot_recording_probe_map`), display-analyzer
  sorting widgets (`plot_sorting_summary` / `plot_unit_summary` /
  `plot_waveforms` / `plot_spikes_on_traces` / `plot_unit_locations`), metric /
  merge views (`plot_metrics` over the routed `AnalyzerCuration.get_metrics()`;
  the raw SI `plot_si_quality_metrics` / `plot_si_template_metrics`;
  `plot_potential_merges` over the persisted `get_merge_groups()`), and local
  exports (`export_si_report` / `export_to_phy`). This is an additive v2
  improvement, not a scientific change: SI owns the plotting, Spyglass only
  resolves the key and routes to the correct source. Routing is load-bearing —
  recording widgets read the saved preprocessed `Recording`; everything
  waveform/template/location/merge/export reads the sort's unwhitened DISPLAY
  analyzer (never the whitened metric analyzer); `plot_metrics` reads the routed
  metric table; `plot_potential_merges` reads the persisted suggestion row and
  never recomputes merge candidates at plot time. Plot helpers are read-only by
  default (a missing display-safe extension raises a clear error naming
  `Sorting.add_extensions(...)`; `compute_missing=True` computes only display-safe
  extensions), the default backend is local `matplotlib` (`sortingview` is an
  explicit opt-in), and no populate/export path opens a GUI, uploads, or
  publishes. The FigPack web-curation viewer above remains separate and
  unimplemented. (`v2/visualization.py`, `v2/_visualization.py`.)

---

## Cross-cutting infrastructure

- **NWB iterators.** The two iterators port v1's
  `SpikeInterfaceRecordingDataChunkIterator` /
  `TimestampsDataChunkIterator`. The trace iterator's only change is the SI
  0.104 kwarg rename `return_scaled` → `return_in_uV`. The timestamps iterator
  narrows the call surface: v1 made the caller wrap a 1D vector in a
  `BaseRecording` subclass; v2's constructor takes `timestamps` +
  `sampling_frequency` directly. (`_nwb_iterators.py`.)

- **Artifact IntervalList name is prefixed.** v2 prefixes the name with
  `artifact_detection_`; v1 wrote the bare `str(artifact_detection_id)`
  (`v1/artifact.py:200`). A ported v1 query looking up by the bare UUID returns
  empty. (`utils.artifact_detection_interval_list_name`.)

- **`CurationLabel` enum.** Members match v1's convention list
  (`v1/curation.py`); v2 promotes it from a docstring list to a validated set.
  (`_enums.CurationLabel`.)

- **Schema-version drift guard.** `_assert_schema_version_matches` raises on a
  version mismatch so downstream code branching on the outer column cannot
  silently route v2 rows to v1 behavior (or vice versa).
  (`_lookup_validation.py`.)

- **`get_spiking_sorting_v2_merge_ids`** parallels the public
  `get_spiking_sorting_v1_merge_ids` (`v1/utils.py:37-109`) with an `as_dict`
  enhancement (v1 always returned a plain list of UUIDs). (`utils.py`.)
