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
