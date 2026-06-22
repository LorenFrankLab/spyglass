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
  `operator="median"` (`v1/recording.py:611`). v2 exposes `operator`
  (`"average"` is v2-only) and drops v1's `reference` params field; default
  `"median"` preserves v1 behavior. (`CommonReferenceParams`.)

- **A reference electrode that is also a sort-group member fails loud.** v1
  silently dropped it via `setdiff1d`; v2 raises.
  (`_reference_resolution.assert_reference_not_member`.)

- **Mixed `original_reference_electrode` across members fails loud.** On the
  auto path v1 silently fell through when members carried mixed reference
  values; v2 raises. Sentinel encoding (`None` / `-1` / `-2` / `>=0`) stays
  v1-compatible. (`_reference_resolution.resolve_group_reference`.)

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
  multiprocessing spawn-workers need no DB connection; v1's equivalents
  (`spikesorting/utils.py:_init_artifact_worker` / `_compute_artifact_chunk`)
  pull DataJoint. `si.load` is the SI 0.104+ rename of v1's `load_extractor`.
  (`_artifact_compute.py`.)

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
  external whitening (`v1/sorting.py:428-430`) and disabled the sorter's
  internal whitening to avoid double-whitening. v2 keeps this and runs it after
  the artifact mask so masked frames don't bias the covariance estimate.
  (`run_si_sorter`.)

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

### Not yet ported from v1 (metric/analyzer curation)

- **BurstPair per-pair quantitative metrics are not stored.** v2 ports the burst
  *visualization* helpers (peak amplitudes, correlogram / xcorrel / peak-over-time
  plots) but stores no `wf_similarity` / `xcorrel_asymm` / cross-unit
  `isi_violation` table; merge suggestion is preset-only. (v1 `BurstPairUnit`
  `v1/burst_curation.py:114-123`, `:286-305`; v2 plots only in
  `_metric_curation_plots.py`.)

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
