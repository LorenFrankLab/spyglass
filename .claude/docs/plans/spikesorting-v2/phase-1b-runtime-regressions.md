# Phase 1b — Runtime regressions + v1 parity fixes

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Phase 1 shipped a set of unplanned regressions vs. v1 — most discovered via a systematic v1-vs-v2 audit after Phase 1 landed. Phase 1b is a focused fix-up phase between Phase 1 and Phase 2 that addresses all of them. The guiding principle is **don't change v1 behavior without a documented reason**; every v2 divergence either matches v1 (default) or is preserved with an explicit rationale.

Scope summary:

- **Two runtime regressions** in `make()` shape (memory + transaction-blocking) — the original Phase 1b motivation.
- **Forty-seven behavior regressions found across five audit passes**: pass 1 (16 items), pass 2 (5 net new), the eight-agent deep sweep (30 net new), and a four-agent followup sweep (5 substantive new + 9 test-coverage additions + 1 Phase 5 documentation deferral). The R-tag items (R1–R18) preserve audit ordering; the N-tag items (N19–N54) come from passes 3 and 4. Combined, the plan now restores or matches v1 behavior in 47 distinct places, plus three default reverts (B1–B3), one improvement over v1 (B5), and one v2-behavior-kept (B7). Two confirmed v1 bugs (the `amplitude_thresh_uV` unit-conversion bug at [v1/utils.py:171-179](../../../../src/spyglass/spikesorting/utils.py#L171-L179) and the heterogeneous-channel-gains silent `gains[0]` pick at [v1/recording.py:858](../../../../src/spyglass/spikesorting/v1/recording.py#L858)) are fixed in v2 AND filed as upstream issues on LorenFrankLab/spyglass. One additional in-flight v1 fix (non-monotonic timestamp repair on the upstream `copilot/fix-populating-artifact-detection` branch) is ported defensively into v2 (N50) since the bug exists in both v1 master and v2.

  **R-tag enumeration gap**: R10, R12, and R14 are deliberately absent. Pass 2 assigned those tags before deduplication, but R10 turned out to duplicate the original Phase 1b memory regression (in-memory `Recording.make`), R12 duplicated the original tri-part / `_parallel_make` regression, and R14 duplicated R4 (the float64 external whitening finding). They were dropped during synthesis; the R-tag sequence skips them on purpose. Do not search elsewhere in this plan for "R10 task" etc.

  **From pass 1 + pass 2** (already in plan before the deep sweep):
  - R1: `CurationV2.get_recording` missing (merge dispatch)
  - R2: `CurationV2.get_sort_group_info` not `@classmethod` (merge dispatch)
  - R3: `Sorting._run_sorter` tempdir leak
  - R4: lost v1 external float64 whitening
  - R5: collapsed disjoint sort intervals (`_consolidate_intervals` + `concatenate_recordings` gone)
  - R6: `recording.annotate(is_filtered=True)` missing
  - R7: `min_segment_length` field dropped
  - R8: spike-out-of-bounds clip on read (test-gated)
  - R9: Singularity carve-out for KS2.5/KS3/IronClust gone
  - R11: `Sorting` has no `delete()` override → `analyzer_folder` leaks on row delete
  - R13: artifact subtraction missing v1's `min_length=1` slivers filter
  - R15: `insert_curation` rejects `labels=None` (v1 accepted)
  - R17: `cache_hash` violates its own `shared-contracts.md` spec — hashes `data.tobytes()` instead of using `NwbfileHasher` via the `_hash_nwb_recording` helper that exists but is unused
  - R18: dead `common_reference.reference` Pydantic field — runtime hardcodes the reference mode based on `ref_channel_id`, schema field never read
- **Three default / API parity reverts** (B1, B2, B3) where v2 silently changed defaults or kwarg names: B1 fixes a confirmed v1 unit-conversion bug (v2's 500 µV default matches v1's *effective* Intan-probe behavior; reverts `proportion_above_thresh` to 1.0); B2 reverts `apply_merge`→`apply_merges` rename; B3 adds the `CurationV2.MergeGroup` part table to restore queryable merge-group provenance.
- **One improvement over v1** (B5) — handle multi-segment NWBs correctly. Both v1 and v2 currently mishandle this; the user asked for v2 to fix it.
- **One v2 behavior to keep** (B7) — v2's stricter `ValueError` on heterogeneous channel gains is mathematically correct; v1's silent `gains[0]` pick is the latent bug. Document the rationale in a code comment.
- **One forward concern flagged for Phase 2** (R16) — `common.common_file_tracking._get_v1_deleted_files` queries v1's `RecordingRecompute` table to mark intentionally-deleted analysis files; v2 has no equivalent table, so when Phase 2 ships `RecordingArtifactRecompute*` and that workflow starts deleting v2 files, the file-tracking infrastructure will flag them as orphans. No Phase 1b code change; flagged in Phase 1b's "Deliberately not in this phase" with a pointer to Phase 2.

Phase 1b is **mostly schema-neutral**, with one deliberate schema exception: **B3 adds a new `CurationV2.MergeGroup` part table** to restore queryable merge-group provenance. This is a deliberate choice with the user; the exception is documented in [overview.md § Resolved Design Decisions](overview.md#resolved-design-decisions) so a future reviewer doesn't misread it as a zero-migration-policy violation. All other v2 table definitions are unchanged.

The two original runtime regressions:

1. **`Recording.make` materializes the full preprocessed array in RAM** (currently calls `recording.get_traces(return_in_uV=False)` then hands the dense ndarray to `pynwb.ecephys.ElectricalSeries`). v1 streams via `SpikeInterfaceRecordingDataChunkIterator(buffer_gb=5)` + `TimestampsDataChunkIterator(buffer_gb=5)` — see [src/spyglass/spikesorting/v1/recording.py:844-849](../../../../src/spyglass/spikesorting/v1/recording.py#L844-L849) and the iterator classes at [src/spyglass/spikesorting/v1/recording.py:883-1092](../../../../src/spyglass/spikesorting/v1/recording.py#L883-L1092). At 30 kHz × 128 ch × 1 h ≈ 110 GB float64, the current v2 path OOMs on any lab workstation; at the 24 h target named in [overview.md § Goals](overview.md#scope-and-dependency-policy) it cannot run at all.

2. **Every v2 Computed table uses monolithic `make()`**. v1 and v0 use the tri-part `make_fetch` / `make_compute` / `make_insert` split plus `_parallel_make = True` (`SpikeSortingRecording` at [v1/recording.py:159,197-261](../../../../src/spyglass/spikesorting/v1/recording.py#L159), `SpikeSorting` at [v1/sorting.py:242-322](../../../../src/spyglass/spikesorting/v1/sorting.py#L242)). **The dominant motivation is transaction blocking, not parallelism.** DataJoint's `AutoPopulate.populate` wraps the *entire* `make()` body in one transaction by default ([Spyglass #1030](https://github.com/LorenFrankLab/spyglass/issues/1030); [DataJoint #1170](https://github.com/datajoint/datajoint-python/issues/1170)). A monolithic 20-minute `Sorting.make` therefore holds the connection — and any row locks DataJoint takes — for 20 minutes, blocking every other user attempting to declare or modify tables on the same database. The tri-part dispatch (`make_fetch` re-runs inside the framework transaction; `make_compute` runs OUTSIDE; `make_insert` runs inside) lives in DataJoint's `autopopulate.py` and fires automatically when a Computed table defines those three methods AND its `make` is the inherited generator wrapper (gated by `inspect.isgeneratorfunction(make)` — see "Delete the monolithic `make` body" task below). The migration from the older `_use_transaction = False` workaround to tri-part is documented in [Spyglass #1422](https://github.com/LorenFrankLab/spyglass/issues/1422) and applied across v0/v1 in [#1066](https://github.com/LorenFrankLab/spyglass/issues/1066), [#1280](https://github.com/LorenFrankLab/spyglass/pull/1280), [#1288](https://github.com/LorenFrankLab/spyglass/pull/1288), [#1338](https://github.com/LorenFrankLab/spyglass/pull/1338). Parallel populate via `_parallel_make = True` is a secondary benefit unlocked by the same refactor. The Phase 1 inner `transaction_or_noop` blocks at [sorting.py:456](../../../../src/spyglass/spikesorting/v2/sorting.py#L456), [recording.py:781](../../../../src/spyglass/spikesorting/v2/recording.py#L781), and [artifact.py:539](../../../../src/spyglass/spikesorting/v2/artifact.py#L539) keep the *inner* commit short, but they do NOT shorten the *outer* framework transaction — only tri-part does.

Phase 1b is **mostly behavior-preserving for deterministic paths**: `clusterless_thresholder` outputs must remain bit-equivalent to Phase 1, and the bandpass+reference outputs of `Recording.make` must remain bit-equivalent within float64 round-trip tolerance. **Stochastic-sorter outputs (MS4, MS5, KS4) will change** because R4 restores v1's float64 external whitening, which deliberately differs from MS4's internal float32 whitening. The validation slice handles this asymmetry: bit-equivalence for deterministic outputs, baseline re-establishment for MS4.

## Executor Checklist

- Use the same isolated `uv` environment and isolated DataJoint integration database as Phase 1; the v2 DB-host safety guard still applies.
- Recommended landing order: (1) baseline capture on unmodified Phase 1 code for `clusterless_thresholder` and Recording output, (2) chunked-iterator port + multi-segment handling, (3) tri-part `make` refactor of `Recording` / `ArtifactDetection` / `Sorting`, (4) v1 parity restorations (whitening, disjoint intervals, `is_filtered`, `min_segment_length`, tempdir, Singularity, merge-dispatch methods, kwarg revert, artifact defaults revert), (5) `CurationV2.MergeGroup` part table + `insert_curation` integration, (6) R8 boundary-spike round-trip test → fold in clip if it fails, (7) validation, (8) documentation.
- Do not touch any other v2 `definition` string. The ONE schema change is adding the new `CurationV2.MergeGroup` part table. Do not change any Lookup contents besides the documented B1 (artifact defaults). Do not change source-part contracts.
- Do not extend the chunked iterators to `Sorting`'s units-NWB write or `CurationV2`'s curated-units NWB write — those tables write per-unit `add_unit(...)` calls with spike-time arrays whose total size is bounded by `n_units × n_spikes_per_unit`, not by `n_samples × n_channels`. Chunking is justified only where the data shape is `(n_samples, n_channels)`.
- **Treat v1 source as the spec** for every parity restoration. When in doubt about exact behavior, read the v1 file/line listed in the relevant task and match it. Exceptions (B7, B5) are called out explicitly; everywhere else, "match v1" is the rule.

**Inputs to read first:**

- [src/spyglass/spikesorting/v1/recording.py:843-1092](../../../../src/spyglass/spikesorting/v1/recording.py#L843-L1092) — `SpikeInterfaceRecordingDataChunkIterator`, `TimestampsExtractor`, `TimestampsSegment`, `TimestampsDataChunkIterator`. Port the data iterator (`SpikeInterfaceRecordingDataChunkIterator`) verbatim with only the SI 0.104 kwarg rename `return_scaled` → `return_in_uV`. The timestamps path is **redesigned, not ported verbatim**: v1's `TimestampsExtractor` wraps a 1D timestamps vector in a `BaseRecording` subclass purely to satisfy v1's `TimestampsDataChunkIterator` constructor; v2 inlines the same data via a smaller `(timestamps: np.ndarray, sampling_frequency: float)` constructor signature that drops the `_TimestampsExtractor` indirection. Same chunked output, simpler call surface.
- [src/spyglass/spikesorting/v1/recording.py:197-261](../../../../src/spyglass/spikesorting/v1/recording.py#L197-L261) — v1 `SpikeSortingRecording` tri-part `make_fetch` / `make_compute` / `make_insert` shape.
- [src/spyglass/spikesorting/v1/sorting.py:242-322](../../../../src/spyglass/spikesorting/v1/sorting.py#L242-L322) — v1 `SpikeSorting` tri-part shape including `_parallel_make = True`.
- [src/spyglass/spikesorting/v2/recording.py:665-1136](../../../../src/spyglass/spikesorting/v2/recording.py#L665-L1136) — current Phase 1 `Recording.make` (665-808), `_rebuild_nwb_artifact` (827-896), and `_write_nwb_artifact` (1056-1136). All three get refactored. Inner `transaction_or_noop` block at line 781; truncation cleanup at 753-774; rollback cleanup at 797-807.
- [src/spyglass/spikesorting/v2/sorting.py:380-485](../../../../src/spyglass/spikesorting/v2/sorting.py#L380-L485) — current Phase 1 `Sorting.make` (refactor target). Inner `transaction_or_noop` at line 456.
- [src/spyglass/spikesorting/v2/artifact.py:479-548](../../../../src/spyglass/spikesorting/v2/artifact.py#L479-L548) — current Phase 1 `ArtifactDetection.make` (refactor target). **Important: `make` (479-516) is only a dispatcher; the actual fetch/compute/insert work lives in `_make_single_recording` (518-548)**. The tri-part refactor must fold `_make_single_recording` into the three new methods, not preserve it as a fourth helper. Inner `transaction_or_noop` at line 539.
- [datajoint.autopopulate](https://github.com/datajoint/datajoint-python/blob/master/datajoint/autopopulate.py) — DataJoint's `AutoPopulate.populate` and `_populate1` dispatch. The tri-part call sequence (`make_fetch` → `make_compute` → tx-open → `make_fetch` again for hash → `make_insert` → tx-close) lives here, not in Spyglass. Fires only when `inspect.isgeneratorfunction(self.make)` is True, which means the inherited generator-based `make` from `AutoPopulate` is being used. If a subclass overrides `make` with a regular function, DataJoint falls back to monolithic and tri-part methods become dead code.
- [src/spyglass/utils/mixins/populate.py](../../../../src/spyglass/utils/mixins/populate.py) — Spyglass's `PopulateMixin.populate` only adds `_parallel_make`/`_use_transaction` flags and a non-daemon process pool; it does NOT implement the tri-part dispatch (that's DataJoint's job, not Spyglass's). Reference for confirming `_parallel_make = True` is the supported flag.
- [docs/src/Features/Mixin.md § Disable Transaction Protection](../../../../docs/src/Features/Mixin.md#L279-L294) — documents the older `_use_transaction = False` escape hatch only. The tri-part migration story is in the GitHub issues, not in this doc.
- [Spyglass issue #1030](https://github.com/LorenFrankLab/spyglass/issues/1030) and [DataJoint issue #1170](https://github.com/datajoint/datajoint-python/issues/1170) — the upstream issues that documented the long-transaction-blocks-other-users behavior.
- [Spyglass PR #1422](https://github.com/LorenFrankLab/spyglass/issues/1422), [#1066](https://github.com/LorenFrankLab/spyglass/issues/1066), [#1280](https://github.com/LorenFrankLab/spyglass/pull/1280), [#1288](https://github.com/LorenFrankLab/spyglass/pull/1288), [#1338](https://github.com/LorenFrankLab/spyglass/pull/1338) — historical record of how the spike-sorting subsystem migrated from `_use_transaction = False` to tri-part `make`; Phase 1b is bringing v2 in line with that policy.
- [research-notes.md § Chronic Recording / Large Data](research-notes.md#6-chronic-recording--large-data) — design rationale for streaming + lazy SI.

**Global invariants apply:** [Environment And Database Safety](shared-contracts.md#environment-and-database-safety) and [Code Artifact Naming](shared-contracts.md#code-artifact-naming).

**Phase-specific contracts referenced:** [Recording Cache Format](shared-contracts.md#recording-cache-format) (the NWB-resident write path is unchanged in shape; only how bytes flow into it changes), [Source Part Pattern](shared-contracts.md#source-part-pattern) (Layer-2 source re-checks must still happen at the start of `make_fetch` for `Sorting` and `ArtifactDetection`).

## Tasks

- **Baseline capture before any code change.** Add `tests/spikesorting/v2/conftest.py` fixture `phase1_baseline_artifacts` (session-scoped, cached on disk in `tests/spikesorting/v2/_fixtures/phase1_baseline/`) that runs the Phase 1 `run_v2_pipeline` against the 60 s MEArec polymer fixture with `preset="franklab_tetrode_clusterless_thresholder"` (deterministic) and saves:
  - The Recording's `cache_hash` and the `ElectricalSeries` traces + timestamps re-read from the AnalysisNwbfile (saved as numpy `.npz`).
  - The Sorting's `object_id` and the units-NWB spike-times-per-unit dict.
  - The CurationV2's `object_id` and the curated units' spike-times-per-unit dict.
  - Peak RSS during populate (measured via `resource.getrusage(RUSAGE_SELF).ru_maxrss`) and wall-clock per stage (recording, artifact, sort).
  The fixture skips with a clear message if `mearec_polymer_128ch_60s.nwb` is unavailable; do not auto-fall-back to `minirec` (`clusterless_thresholder` on a sub-minute synthetic fixture is too short for the memory check to be informative). The fixture must be regeneratable: deleting `_fixtures/phase1_baseline/` and re-running on the unmodified Phase 1 code reproduces the same hashes. Document this regeneration step in the fixture docstring.

- **Port the chunked iterators into v2** as `src/spyglass/spikesorting/v2/_nwb_iterators.py`. Single new module, no behavior change beyond the SI rename and a docstring update:

  ```python
  # src/spyglass/spikesorting/v2/_nwb_iterators.py
  """Chunked HDF5 writers for the preprocessed Recording artifact.

  Used by ``Recording._write_nwb_artifact`` to stream the
  ``(n_samples, n_channels)`` trace array and the ``(n_samples,)``
  timestamps vector into the AnalysisNwbfile via HDMF's
  ``GenericDataChunkIterator``. Without this, 30 kHz × 128 ch × 1 h
  recordings (~110 GB float64) would have to materialize in RAM
  before the NWB write — which OOMs on any lab workstation.

  Ports v1's ``SpikeInterfaceRecordingDataChunkIterator`` and
  ``TimestampsDataChunkIterator`` from
  ``spyglass.spikesorting.v1.recording``. The only API change is the
  SpikeInterface 0.104 rename ``return_scaled`` → ``return_in_uV``,
  propagated through ``_get_data``.
  """
  from __future__ import annotations

  from typing import Iterable, Optional, Tuple

  import numpy as np
  import spikeinterface as si
  from hdmf.data_utils import GenericDataChunkIterator


  class SpikeInterfaceRecordingDataChunkIterator(GenericDataChunkIterator):
      def __init__(
          self,
          recording: si.BaseRecording,
          segment_index: int = 0,
          return_in_uV: bool = False,
          buffer_gb: Optional[float] = None,
          buffer_shape: Optional[tuple] = None,
          chunk_mb: Optional[float] = None,
          chunk_shape: Optional[tuple] = None,
          display_progress: bool = False,
          progress_bar_options: Optional[dict] = None,
      ):
          self.recording = recording
          self.segment_index = segment_index
          self.return_in_uV = return_in_uV
          self.channel_ids = recording.get_channel_ids()
          super().__init__(
              buffer_gb=buffer_gb,
              buffer_shape=buffer_shape,
              chunk_mb=chunk_mb,
              chunk_shape=chunk_shape,
              display_progress=display_progress,
              progress_bar_options=progress_bar_options,
          )

      def _get_data(self, selection: Tuple[slice]) -> Iterable:
          return self.recording.get_traces(
              segment_index=self.segment_index,
              channel_ids=self.channel_ids[selection[1]],
              start_frame=selection[0].start,
              end_frame=selection[0].stop,
              return_in_uV=self.return_in_uV,
          )

      def _get_dtype(self):
          return self.recording.get_dtype()

      def _get_maxshape(self):
          return (
              self.recording.get_num_samples(segment_index=self.segment_index),
              self.recording.get_num_channels(),
          )


  class _TimestampsSegment(si.BaseRecordingSegment):
      def __init__(self, timestamps, sampling_frequency, t_start, dtype):
          si.BaseRecordingSegment.__init__(
              self,
              sampling_frequency=sampling_frequency,
              t_start=t_start,
          )
          self._timeseries = np.asarray(timestamps, dtype=dtype)

      def get_num_samples(self) -> int:
          return self._timeseries.shape[0]

      def get_traces(
          self,
          start_frame=None,
          end_frame=None,
          channel_indices=None,
      ) -> np.ndarray:
          return np.squeeze(self._timeseries[start_frame:end_frame])


  class _TimestampsExtractor(si.BaseRecording):
      def __init__(self, timestamps, sampling_frequency=30e3):
          si.BaseRecording.__init__(
              self, sampling_frequency, channel_ids=[0], dtype=np.float64
          )
          self.add_recording_segment(
              _TimestampsSegment(
                  timestamps=timestamps,
                  sampling_frequency=sampling_frequency,
                  t_start=None,
                  dtype=np.float64,
              )
          )


  class TimestampsDataChunkIterator(GenericDataChunkIterator):
      def __init__(
          self,
          timestamps: np.ndarray,
          sampling_frequency: float,
          buffer_gb: Optional[float] = None,
          buffer_shape: Optional[tuple] = None,
          chunk_mb: Optional[float] = None,
          chunk_shape: Optional[tuple] = None,
          display_progress: bool = False,
          progress_bar_options: Optional[dict] = None,
      ):
          self._extractor = _TimestampsExtractor(
              timestamps=timestamps,
              sampling_frequency=sampling_frequency,
          )
          super().__init__(
              buffer_gb=buffer_gb,
              buffer_shape=buffer_shape,
              chunk_mb=chunk_mb,
              chunk_shape=chunk_shape,
              display_progress=display_progress,
              progress_bar_options=progress_bar_options,
          )

      def _get_data(self, selection: Tuple[slice]) -> Iterable:
          return self._extractor.get_traces(
              segment_index=0,
              channel_ids=[0],
              start_frame=selection[0].start,
              end_frame=selection[0].stop,
              return_in_uV=False,
          )

      def _get_dtype(self):
          return self._extractor.get_dtype()

      def _get_maxshape(self):
          return (self._extractor.get_num_samples(segment_index=0),)
  ```

  Notes for the executor:
  - The `_Timestamps*` helpers stay private (leading underscore) — they exist only to feed the timestamps iterator and have no API surface beyond it.
  - Default `buffer_gb` for both iterators is `1` (HDMF default); `Recording._write_nwb_artifact` passes `buffer_gb=5` to match v1's choice.
  - Keep these classes out of `utils.py`; `utils.py` is already heavy and `_nwb_iterators.py` is the right home for chunked-HDF5 plumbing.

- **Rewrite `Recording._write_nwb_artifact`** at [src/spyglass/spikesorting/v2/recording.py:1056-1136](../../../../src/spyglass/spikesorting/v2/recording.py#L1056-L1136) to use the chunked iterators. Replace with:
  - Build a `SpikeInterfaceRecordingDataChunkIterator(recording, return_in_uV=False, buffer_gb=5)` and pass it as the `data` argument to `pynwb.ecephys.ElectricalSeries(...)` — HDMF/HDF5 will stream-write from it.
  - Build a `TimestampsDataChunkIterator(timestamps_array, sampling_frequency=fs, buffer_gb=5)` and pass it as `timestamps`. Acquire `timestamps_array` lazily via `recording.get_times()` only AFTER the iterator is constructed; if a future SI release makes `get_times()` itself lazy, the iterator picks that up.
  - **cache_hash is computed in R17 post-write via `_hash_nwb_recording(analysis_file_name)`** — do NOT compute it inline here. Drop the existing `hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()` line entirely (R17 replaces the hash strategy). The chunked-iterator task is purely about streaming the NWB write; the hash work moves to R17's task surface.
  - Keep the existing `gains = np.unique(recording.get_channel_gains())` check and `conversion` computation; those are O(n_channels), not O(n_samples), and don't need streaming.
  - Keep the `pynwb.NWBHDF5IO(... mode="a" ...)` write block; only the `data=` and `timestamps=` arguments change.

- **Update `Recording._rebuild_nwb_artifact`** at [src/spyglass/spikesorting/v2/recording.py:827-896](../../../../src/spyglass/spikesorting/v2/recording.py#L827-L896) — same streaming logic; it shares the helper path with the original write.

- **Delete the existing monolithic `make()` body** on `Recording`, `ArtifactDetection`, and `Sorting` as the first step of each tri-part refactor. DataJoint's tri-part dispatch only fires when `inspect.isgeneratorfunction(self.make)` returns True — i.e., when the table is using the inherited generator-based `make` from `AutoPopulate`, not an explicit subclass override. If `def make(self, key): ...` remains in the subclass, DataJoint falls back to monolithic, the new `make_fetch` / `make_compute` / `make_insert` methods become dead code, and the regression silently persists. After deletion, the executor confirms the tri-part dispatch is active by running the `test_tripart_dispatch_active` smoke test (see Validation slice) which asserts `inspect.isgeneratorfunction(Recording.make) is True` for each refactored table. For `ArtifactDetection`, also remove the `_make_single_recording` helper at [artifact.py:518-548](../../../../src/spyglass/spikesorting/v2/artifact.py#L518-L548) — its body folds directly into `make_compute` / `make_insert`. Do not keep it as a fourth indirection layer.

- **Refactor `Recording.make` into tri-part**:
  - `make_fetch(self, key)` — DB reads only: fetch the `RecordingSelection` row, the `SortGroupV2.SortGroupElectrode` rows, the reference electrode id, and the `IntervalList` row for the requested sort interval. Returns a tuple/list of those values. No SI calls, no NWB I/O.
  - `make_compute(self, key, sel_row, sg_electrodes, ref_channel_id, sort_interval_valid_times, raw_request_window)` — pure compute: open the raw NWB via `se.read_nwb_recording(...)`, slice channels and frames via `_restrict_recording`, apply `_apply_pre_motion_preprocessing`, then write the AnalysisNwbfile to disk via the refactored streaming `_write_nwb_artifact`. Returns the new analysis file name, `object_id`, `cache_hash`, and the saved-times start/end for the truncation check. Does NOT register the AnalysisNwbfile row and does NOT call `self.insert1`.
  - `make_insert(self, key, *make_compute_outputs, nwb_file_name=..., sampling_frequency=...)` — re-acquires the DB lock, runs the existing truncation check against the requested interval (raising `RecordingTruncatedError` with the staged-file unlink already in place), wraps `AnalysisNwbfile().add(...)` + `self.insert1(...)` in `transaction_or_noop` per the existing Phase 1 contract.
  - Add `_parallel_make = True` as a class-level attribute on `Recording`.
  - The except-path file unlink that Phase 1 added at [recording.py:752-770](../../../../src/spyglass/spikesorting/v2/recording.py#L752-L770) (truncation cleanup) and [recording.py:793-803](../../../../src/spyglass/spikesorting/v2/recording.py#L793-L803) (rollback cleanup) must thread through `make_insert` because that is where the registration transaction lives. The `make_compute` path's only on-failure side effect is the staged NWB on disk; if `make_compute` itself raises (e.g., during preprocessing), the staged file is removed in a `try/except` inside `make_compute` before the exception propagates — Spyglass's tri-part dispatcher will not call `make_insert` in that case.

- **Refactor `ArtifactDetection.make` into tri-part**. Current structure has TWO levels of indirection: `make` at [artifact.py:479-516](../../../../src/spyglass/spikesorting/v2/artifact.py#L479-L516) is a thin dispatcher that calls `_make_single_recording` at [artifact.py:518-548](../../../../src/spyglass/spikesorting/v2/artifact.py#L518-L548), which holds the actual `IntervalList.insert1` + `self.insert1` + `transaction_or_noop` block. The refactor REMOVES both `make` and `_make_single_recording` (per the "Delete the existing monolithic `make()` body" task above) and folds their work into:
  - `make_fetch(self, key)` — Layer-2 source re-check via `ArtifactSelection.resolve_source(key)`, fetch `ArtifactDetectionParameters.params`, validate via Pydantic. For the `recording` source, also fetch the upstream `RecordingSelection.nwb_file_name`. Raise `NotImplementedError` early for the `shared_artifact_group` source (existing Phase 1 gate; preserved). Must be deterministic and DeepHash-stable because DataJoint calls `make_fetch` twice (once before `make_compute` and once again inside the transaction for upstream-integrity hashing — see invariants note below).
  - `make_compute(self, key, source, validated, nwb_file_name)` — load the cached preprocessed recording via `Recording().get_recording(source.key)`, run the existing `_detect_artifacts` scan, return the artifact-removed `valid_times` array. No DB writes.
  - `make_insert(self, key, nwb_file_name, valid_times)` — perform `IntervalList.insert1(...)` + `self.insert1(key)`. Since `make_insert` already runs inside DataJoint's framework transaction (per `autopopulate.py`), the inner `transaction_or_noop` from the old `_make_single_recording` becomes a no-op (`transaction_or_noop` yields without opening a new transaction when one is already active — see [utils.py:29-30](../../../../src/spyglass/spikesorting/v2/utils.py#L29-L30)). The wrap can be kept defensively or dropped; if kept, add a one-line comment noting the redundancy so a future cleanup doesn't break atomicity.
  - Add `_parallel_make = True`.

- **Tri-part invariants that apply to every refactored table** (`Recording`, `ArtifactDetection`, `Sorting`):
  - `make_fetch` MUST be deterministic and DeepHash-stable. DataJoint's `autopopulate._populate1` calls `make_fetch` twice (see DataJoint `autopopulate.py`): once before `make_compute` to pull the upstream values that the compute step consumes, and a second time after acquiring the transaction to re-check upstream integrity via DeepHash. A `make_fetch` that returns a `datetime.now()`, an order-dependent dict, or a NumPy array with non-stable byte layout will spuriously raise `dj.DataJointError("DataJoint integrity error")` on every populate. Restrict returned values to fetched DB columns and IntervalList `valid_times` arrays; do not synthesize timestamps or generate UUIDs in `make_fetch`.
  - The inner `transaction_or_noop(self.connection)` blocks from the Phase 1 monolithic `make()` become no-ops once `make_insert` runs inside the framework transaction (the helper at [utils.py:29-30](../../../../src/spyglass/spikesorting/v2/utils.py#L29-L30) yields without opening a new transaction when one is already active). **Decision: keep them as defensive scaffolding** with an explanatory one-line comment (`# no-op when framework transaction is active; preserved for defense if make_insert is ever called outside populate`). The `test_curation_v2_nwb_write_outside_transaction` AST guard and the v1-parity discipline both depend on these blocks remaining in the source. Do NOT remove them. Do NOT add NEW transaction wraps inside `make_compute`; the framework provides no transaction around `make_compute` by design (that's the whole point), so any new wrap there would re-introduce the long-transaction problem.
  - The except-path file-unlink cleanup from Phase 1 (e.g., `Recording.make` at [recording.py:797-807](../../../../src/spyglass/spikesorting/v2/recording.py#L797-L807); `Sorting.make` at [sorting.py:475-485](../../../../src/spyglass/spikesorting/v2/sorting.py#L475-L485)) must move into `make_insert`. If `make_compute` itself fails (e.g., during preprocessing or sort), the staged NWB on disk gets unlinked inside `make_compute` via its own `try/except` before the exception propagates — DataJoint will not call `make_insert` in that case, so cleanup must happen wherever the file was created.

- **Refactor `Sorting.make` into tri-part** at [src/spyglass/spikesorting/v2/sorting.py:380-485](../../../../src/spyglass/spikesorting/v2/sorting.py#L380-L485):
  - `make_fetch(self, key)` — Layer-2 source re-check; for the `recording` source, fetch the `SortingSelection` row, the `SorterParameters` row (`sorter`, `params`, `job_kwargs`), and the upstream `RecordingSelection.nwb_file_name`. Concat path still raises `NotImplementedError` here.
  - `make_compute(self, key, source, sel_row, sorter_row, nwb_file_name)` — load the recording, apply artifact mask if `artifact_id`, run the sorter via `_run_sorter`, `_remove_excess_spikes`, `_build_analyzer`, `_write_units_nwb`. Returns the units-NWB file name, the units `object_id`, the analyzer folder path, `n_units`, and the sorted-object reference needed by `_populate_unit_part`. Stage the units NWB to disk; do NOT register the AnalysisNwbfile row or call `self.insert1` or `Sorting.Unit.insert`.
  - `make_insert(self, key, *make_compute_outputs, sorting_obj, recording_id, nwb_file_name)` — wrap the existing `AnalysisNwbfile().add(...)` + `self.insert1(...)` + `_populate_unit_part(...)` block in `transaction_or_noop` exactly as today. Preserve the staged-file unlink in the except path.
  - Add `_parallel_make = True`.
  - **Do not split `_populate_unit_part` across stages.** It currently runs inside the transaction at [sorting.py:468-474](../../../../src/spyglass/spikesorting/v2/sorting.py#L468-L474). Keep it in `make_insert` so the unit rows commit atomically with the master row. Re-loading the analyzer inside `_populate_unit_part` is cheap (a folder lookup) and avoids threading a large `SortingAnalyzer` object through the tri-part boundary.

- **R1 — Add `CurationV2.get_recording(cls, key)` as a `@classmethod`.** Mirrors [v1/curation.py:163-179](../../../../src/spyglass/spikesorting/v1/curation.py#L163-L179) exactly. Without it, `SpikeSortingOutput.get_recording(merge_key)` raises `AttributeError` on every v2 `merge_id` because the merge dispatcher at [spikesorting_merge.py:317](../../../../src/spyglass/spikesorting/spikesorting_merge.py#L317) does `query.get_recording(query.fetch("KEY"))`.

  **Combine with R2 + N53 in one code edit pass.** R1 adds `get_recording`; R2 converts `get_sort_group_info` to `@classmethod`; N53 converts `get_sorting` and `get_merged_sorting` (plus the `_upstream_recording_row` helper) to `@classmethod`. All four `CurationV2` accessor methods need the same `@classmethod` decorator + `self → cls` rebinding for v1 surface symmetry. Doing them together avoids partial-conversion drift and is one logical refactor; N21's code sample is written assuming N53 has landed.

  Implementation:

  ```python
  # In src/spyglass/spikesorting/v2/curation.py, on class CurationV2
  @classmethod
  def get_recording(cls, key: dict):
      """Get the recording associated with a CurationV2 row.

      Mirrors CurationV1.get_recording (v1/curation.py:163-179). Resolves
      the upstream Recording via SortingSelection.resolve_source and
      delegates to Recording().get_recording.
      """
      import spikeinterface as si
      from spyglass.spikesorting.v2.recording import Recording
      source = SortingSelection.resolve_source({"sorting_id": key["sorting_id"]})
      if source.kind != "recording":
          raise NotImplementedError(
              "CurationV2.get_recording: concat-source sorts are not yet "
              "supported."
          )
      recording = Recording().get_recording(source.key)
      recording.annotate(is_filtered=True)  # R6
      return recording
  ```

  This task also subsumes R6 (the `annotate(is_filtered=True)` line at the bottom) — the annotation lives on the only public path that hands a recording back to a downstream consumer.

- **R2 — Convert `CurationV2.get_sort_group_info` to a `@classmethod`.** Currently at [v2/curation.py:672](../../../../src/spyglass/spikesorting/v2/curation.py#L672) it is a plain instance method `def get_sort_group_info(self, key)`. The merge dispatcher at [spikesorting_merge.py:346](../../../../src/spyglass/spikesorting/spikesorting_merge.py#L346) calls `source_table.get_sort_group_info(query.fetch("KEY"))` — `source_table` is the class object from `source_class_dict`, not an instance — so the call resolves to `get_sort_group_info(query.fetch("KEY"), <missing>)` and raises `TypeError`. Change the decorator and rebind:

  ```python
  @classmethod
  def get_sort_group_info(cls, key: dict):
      # … existing body, but reference cls instead of self where applicable.
      # The body's existing logic operates on the restriction (cls & key),
      # which works the same way for classmethod-bound cls as for instance-
      # bound self.
  ```

  No body changes beyond the `self`→`cls` rename and the decorator. The existing logic at lines 672-709 already operates on `self & key`-style restrictions that work identically when `cls` is bound.

- **R3 — Fix tempdir leak in `Sorting._run_sorter`.** Cleans up the **sorter scratch tempdir** (where SI writes per-sort intermediate files) when the sort completes successfully — distinct from R11 (`Sorting.delete()` cleaning the `analyzer_folder` on row delete) and from N51 (cleaning the `analyzer_folder` on populate failure). The three cleanup paths cover three lifecycle events on two distinct folders (sorter tempdir + analyzer folder). At [v2/sorting.py:725](../../../../src/spyglass/spikesorting/v2/sorting.py#L725) replace `tempfile.mkdtemp(prefix=f"sort_{sorting_id}_")` with v1's pattern from [v1/sorting.py:419](../../../../src/spyglass/spikesorting/v1/sorting.py#L419):

  ```python
  import tempfile
  from spyglass.settings import temp_dir as spyglass_temp_dir

  # Use a context-managed TemporaryDirectory anchored to Spyglass's
  # configured temp dir; the directory is removed when the context exits.
  # Matches v1/sorting.py:419 exactly. Without this, KS-class sorters
  # leak 5-50 GB into the system /tmp on every populate.
  sorter_temp_dir = tempfile.TemporaryDirectory(dir=spyglass_temp_dir)
  os.chmod(sorter_temp_dir.name, 0o777)
  # ... pass sorter_temp_dir.name as folder=, then explicitly close at the end:
  try:
      sorting_obj = sis.run_sorter(...)
  finally:
      sorter_temp_dir.cleanup()
  ```

  Note: `tempfile.TemporaryDirectory` auto-cleans on garbage collection, but the explicit `.cleanup()` in a `finally` block makes the cleanup point obvious. Add a smoke test (`test_sorting_tempdir_cleaned_up`) that verifies the temp dir doesn't exist after `Sorting.populate` returns.

- **R4 — Restore v1's external float64 whitening.** In `Sorting._run_sorter` (currently around [v2/sorting.py:687-732](../../../../src/spyglass/spikesorting/v2/sorting.py#L687-L732)), add v1's whitening branch from [v1/sorting.py:428-430](../../../../src/spyglass/spikesorting/v1/sorting.py#L428-L430). Placement: AFTER `clusterless_thresholder` branch, AFTER `_apply_artifact_mask` (which runs in `Sorting.make` upstream — so the recording passed into `_run_sorter` is already artifact-masked), BEFORE the call to `sis.run_sorter`. This ordering matches v1 and ensures artifact-masked frames don't bias whitening's covariance estimate.

  ```python
  # Match v1's external whitening behavior. The upstream
  # Recording._apply_pre_motion_preprocessing runs bandpass + reference
  # at float64; pre-whitening here keeps the whitening step at the same
  # precision. MS4's internal whitening operates at float32 (see
  # Mountainsort4Sorter wrapper, ~L91-95), which v1 deliberately
  # bypassed for precision parity. If the schema's Pydantic default
  # says whiten=True, run whitening externally at float64 and turn the
  # sorter's internal whitening off.
  if sorter_params.get("whiten", False):
      import numpy as _np
      import spikeinterface.preprocessing as sip
      recording = sip.whiten(recording, dtype=_np.float64)
      sorter_params = {**sorter_params, "whiten": False}
  ```

  This DELIBERATELY breaks bit-equivalence with the Phase 1 MS4 baseline (Phase 1 lets MS4 whiten at float32 internally; post-fix, whitening happens at float64 externally). The validation slice handles this by re-baselining MS4 AFTER the fix, not by gating on Phase 1's pre-fix output. The `clusterless_thresholder` baseline is unaffected (no whitening) and remains the bit-equivalence gate.

- **R5 — Restore disjoint-interval concatenation.** v1's `_get_preprocessed_recording` at [v1/recording.py:556-583](../../../../src/spyglass/spikesorting/v1/recording.py#L556-L583) calls `_consolidate_intervals` to convert the requested `valid_sort_times` array into per-interval frame ranges, then frame-slices each interval and concatenates via `si.concatenate_recordings(recordings_list)`. v2 at [v2/recording.py:920-925](../../../../src/spyglass/spikesorting/v2/recording.py#L920-L925) takes only `(times[0][0], times[-1][-1])` — the outer envelope, which silently includes inter-interval gaps.

  Restore the v1 pattern inside `Recording._restrict_recording`:

  ```python
  # Match v1/recording.py:556-583. When the requested sort interval
  # is more than one disjoint chunk (e.g., run+sleep+run epochs),
  # frame-slice each chunk separately and concatenate; otherwise a
  # single frame_slice on the outer envelope silently sorts the inter-
  # chunk gaps too.
  from spyglass.spikesorting.v2.utils import (
      _consolidate_intervals,
      _get_recording_timestamps,  # B5: handles multi-segment + propagates N50 correction
  )
  from spikeinterface import concatenate_recordings

  times = _get_recording_timestamps(recording, override=all_timestamps)  # all_timestamps is N50-corrected
  intersection = sort_interval.intersect(raw_valid)
  valid_times = intersection.times  # ndarray of (start_s, end_s) tuples
  intervals_in_frames = _consolidate_intervals(valid_times, times)
  if len(intervals_in_frames) > 1:
      sliced = [
          recording.frame_slice(start_frame=s, end_frame=e)
          for s, e in intervals_in_frames
      ]
      recording = concatenate_recordings(sliced)
  else:
      s, e = intervals_in_frames[0]
      recording = recording.frame_slice(start_frame=s, end_frame=e)
  ```

  **Important**: `_consolidate_intervals` is defined ONLY at [v1/recording.py:715](../../../../src/spyglass/spikesorting/v1/recording.py#L715) — NOT in `src/spyglass/spikesorting/utils.py` (despite what an earlier draft of this plan suggested). The executor must **port v1's definition into `src/spyglass/spikesorting/v2/utils.py`** as a new private helper rather than import from v1 (v2 should not couple to v1's module). The helper is small (~20 lines): the body at v1/recording.py:715 converts seconds-to-frames via timestamp binary search and consolidates adjacent intervals.

  **Off-by-one warning** (from the 4-agent verification sweep): v1's `_consolidate_intervals` at [v1/recording.py:741-744](../../../../src/spyglass/spikesorting/v1/recording.py#L741-L744) computes `stop_indices = searchsorted(side="right") - 1`, then passes that to `frame_slice(end_frame=stop)`. But SI's `frame_slice` end is **exclusive**, so v1 silently drops the last sample (~33 µs at 30 kHz) of every disjoint interval. **Do NOT copy v1's `- 1` verbatim**: port the helper but use `side="right"` semantics WITHOUT the off-by-one subtraction. v2's existing `_get_sort_interval_window` at [v2/recording.py:920-925](../../../../src/spyglass/spikesorting/v2/recording.py#L920-L925) already uses the correct semantics; preserve that. Document the v1 divergence in the helper's docstring as "v1-bug-fixed, like B7" so future readers see why the port deliberately differs from v1's exact line shape. Add a unit test that exercises the end-of-interval sample boundary on a synthetic timestamps array.

- **R6 — `recording.annotate(is_filtered=True)` annotation.** Covered by R1's task (the annotation goes inside the new `CurationV2.get_recording` classmethod). Verify the same call exists on `Recording.get_recording` at [v2/recording.py:807-821](../../../../src/spyglass/spikesorting/v2/recording.py#L807-L821) before returning; if not, add it there too. Matches [v1/curation.py:178](../../../../src/spyglass/spikesorting/v1/curation.py#L178). Without it, downstream SI consumers may re-apply bandpass to an already-filtered recording.

- **R7 — Restore `min_segment_length` parameter on `PreprocessingParamsSchema`.** v1 default at [v1/recording.py:135](../../../../src/spyglass/spikesorting/v1/recording.py#L135) is `1` (second). v1 uses it at [v1/recording.py:472](../../../../src/spyglass/spikesorting/v1/recording.py#L472) as `sort_interval.intersect(valid_interval_times, min_length=params["min_segment_length"])` to drop intervals shorter than the threshold. v2's `PreprocessingParamsSchema` at [v2/_params/preprocessing.py](../../../../src/spyglass/spikesorting/v2/_params/preprocessing.py) doesn't have this field; sub-second slivers reach the sorter and crash downstream waveform extraction.

  Add to `PreprocessingParamsSchema`:

  ```python
  min_segment_length: float = Field(default=1.0, ge=0.0)
  ```

  And in `Recording._restrict_recording` (after the R5 changes above), pass it through:

  ```python
  intersection = sort_interval.intersect(raw_valid, min_length=validated.min_segment_length)
  ```

  Bump `params_schema_version` to `2` per the [Pydantic Parameter Schema Convention § Schema versioning](shared-contracts.md#pydantic-parameter-schema-convention). Phase 1's contents rows insert with version 1; add migration logic to read v1 rows transparently (the field has a default so the migration is just "fill in min_segment_length=1.0 when reading older rows"). Update the three default contents at [v2/recording.py:503-534](../../../../src/spyglass/spikesorting/v2/recording.py#L503-L534) to include `min_segment_length` explicitly.

- **R8 — Boundary-spike round-trip test (decision-gated).** Add `test_boundary_spike_round_trip_does_not_raise` as the FIRST step. Synthesize a `Sorting` whose units include a spike at exactly the recording's last sample, populate v2 through `run_v2_pipeline`, then call `Sorting().get_sorting(key)` and `CurationV2().get_sorting(curation_key)`. If either raises SI's "spikes exceeding the recording duration" `ValueError`, fold in v1's `spike_times_to_valid_samples` clip on read from [v1/sorting.py:29-79](../../../../src/spyglass/spikesorting/v1/sorting.py#L29-L79) and re-test. If neither raises, document the result and skip the clip. The decision is contingent on the test result, not on prediction.

- **R9 — Restore Singularity carve-out for MATLAB-based sorters.** v1's pattern at [v1/sorting.py:439-450](../../../../src/spyglass/spikesorting/v1/sorting.py#L439-L450). Add to `Sorting._run_sorter` mirroring the v1 structure:

  ```python
  # Match v1/sorting.py:439-450. KS2.5/KS3/IronClust are MATLAB-based
  # sorters that SI ships as containerized images; users without a
  # native install need singularity_image=True. The carve-out also
  # strips kwargs that don't survive containerization.
  if sorter.lower() in ("kilosort2_5", "kilosort3", "ironclust"):
      sorter_params = {
          k: v
          for k, v in sorter_params.items()
          if k not in ("tempdir", "mp_context", "max_threads_per_process")
      }
      sorting_obj = sis.run_sorter(
          sorter_name=sorter,
          recording=recording,
          folder=tmpdir,
          remove_existing_folder=True,
          singularity_image=True,
          **sorter_params,
      )
  else:
      sorting_obj = sis.run_sorter(
          sorter_name=sorter,
          recording=recording,
          folder=tmpdir,
          remove_existing_folder=True,
          **sorter_params,
      )
  ```

  None of Phase 1's default `SorterParameters` rows use these sorters, so no new schema row is needed. Users who want them insert a custom row via `SorterParameters.insert1` (validated by `GenericSorterParamsSchema` which has `extra="allow"`).

- **B1 — Revised: fix v1's `amplitude_thresh_uV` unit-conversion bug; revert `proportion_above_thresh` to v1's default.** This task encodes two related corrections discovered by the multi-agent audit.

  **The v1 bug** ([v1/utils.py:171-179](../../../../src/spyglass/spikesorting/utils.py#L171-L179)): v1's `_compute_artifact_chunk` calls `recording.get_traces(...)` with `return_scaled=False` (the default) — returns raw int16 counts from the NWB file. The comparison `np.abs(traces) > amplitude_thresh` then compares raw counts to a field named `amplitude_thresh_uV`. For Frank-lab Intan amplifiers (RHD2132/2164: 16-bit signed at ±6.4 mV → 0.195 µV/count), v1's nominal `amplitude_thresh_uV=3000` was effectively ~585 µV. v1's field name lied about its unit. The bug becomes worse on non-Intan probes — Neuropixels at gain=500 is 2.34 µV/count → v1's `3000` was effectively ~7020 µV → almost no artifacts ever flagged on Neuropixels in v1 production.

  **The v2 fix is already in place** at [v2/artifact.py:571-572](../../../../src/spyglass/spikesorting/v2/artifact.py#L571-L572): `gains = recording.get_channel_gains(); traces_uv = traces.astype(_np.float32) * gains[None, :]`. v2 honors the field name. **Do NOT revert this; it is a bug fix, not a regression.**

  **Default value**: keep v2's current `amplitude_thresh_uV = 500.0`. Justifications:
  - Matches v1's effective production behavior on Frank-lab Intan probes within ~15% (500 µV vs ~585 µV).
  - Reverting to `3000` with v2's correct uV semantics would silently make detection 5× more permissive than v1 actually was on Intan probes — a real regression in artifact masking.
  - 500 µV is a scientifically reasonable artifact threshold (true extracellular spikes are typically 50–500 µV; mechanical/biological artifacts are typically >> 1 mV).

  **`proportion_above_thresh`**: revert to v1's `1.0` (no associated bug; v1's "all channels must exceed" is the principled belt-and-suspenders default; v2's 0.5 was a silent change without justification).

  Final schema field defaults:

  ```python
  # v2/_params/artifact_detection.py
  amplitude_thresh_uV: float | None = Field(default=500.0, ge=0.0)  # uV, real units (v2 fixes v1 bug)
  proportion_above_thresh: float = Field(default=1.0, gt=0.0, le=1.0)  # v1 parity
  ```

  Update the `"default"` row in `ArtifactDetectionParameters._DEFAULT_CONTENTS` to match. The `"none"` row (`detect=False`) is unaffected. Add a docstring note to `ArtifactDetectionParamsSchema` explaining the v1 bug and the v2 fix, citing v1 file:line.

  **Pre-landing audit step (REQUIRED before merging this task):** the `proportion_above_thresh` default flip from 0.5 → 1.0 changes the behavioral outcome of every integration test that uses the `"default"` artifact preset without explicitly overriding `proportion_above_thresh`. Before landing the schema change, the executor MUST:

  1. `grep -nE "artifact_params_name.*default|insert_default\(\).*ArtifactDetection" tests/spikesorting/v2/` and review every callsite.
  2. For each integration test that passes `artifact_params_name="default"` without an explicit `proportion_above_thresh` override, run the test against the new default and confirm the test's expected outcomes still hold. Specifically: a test on a low-amplitude fixture (smoke + 60s polymer) may now flag FEWER frames as artifacts (proportion=1.0 requires ALL channels above threshold; proportion=0.5 only needed half). Tests that asserted "at least one valid interval gap" with the default preset are most at risk.
  3. Where a test's expected outcome materially changes, either tighten the test's parameter override to keep the 0.5 semantics explicit OR update the assertion to reflect the new default.
  4. Tests that pass `proportion_above_thresh=0.5` explicitly (e.g., `test_detect_artifacts_*` in the synthetic-recording bundle) are unaffected and need no change.

  **CHANGELOG entry** (must ship in this PR):

  ```
  Spike sorting v2 fixes an artifact-detection unit-conversion bug present in
  v1 since the field was introduced. v1's `_compute_artifact_chunk` compared
  raw int16 NWB counts against the `amplitude_thresh_uV` field, ignoring the
  probe's gain. For Frank-lab Intan probes (0.195 µV/count) this meant
  v1's documented default of 3000 was effectively ~585 µV; for Neuropixels at
  gain=500 (2.34 µV/count) v1's 3000 was effectively ~7020 µV. v2 correctly
  scales traces by channel gain before comparison. The v2 default of 500 µV
  matches v1's effective Intan-probe behavior within ~15%. v1 users with
  custom thresholds should translate v1_value × probe_gain × 1e-6 to get
  the v2-equivalent uV value.
  ```

  See also the "File v1 issues" task below for upstream tracking of the bug.

- **File v1 issues for confirmed bugs (audit followup)**. Two upstream GitHub issues on LorenFrankLab/spyglass, filed in the same PR cycle as Phase 1b so they don't get forgotten. The executor opens each issue with the audit evidence; resolution happens in the v1 codebase, not v2.

  **Issue 1**: "`amplitude_thresh_uV` field on `SpikeSortingArtifactDetectionParameters` compares raw int16 counts, not microvolts." Body cites [v1/utils.py:171-179](../../../../src/spyglass/spikesorting/utils.py#L171-L179), explains the Frank-lab Intan ~5×-permissive effect and Neuropixels ~14×-permissive effect, references v2's fix at [v2/artifact.py:571-572](../../../../src/spyglass/spikesorting/v2/artifact.py#L571-L572), and recommends v1 either deprecate the field with a warning or backport the gain-scaling fix.

  **Issue 2**: "`SpikeSortingRecording._write_recording_to_nwb` silently scales all channels by gains[0] when channels have heterogeneous gains." Body cites [v1/recording.py:858](../../../../src/spyglass/spikesorting/v1/recording.py#L858), explains the silent-wrong-conversion scenario, references v2's stricter raise at [v2/recording.py:1100-1109](../../../../src/spyglass/spikesorting/v2/recording.py#L1100-L1109), and recommends v1 add the same raise (or warn loudly).

  Both issue links must be recorded in the PR description so reviewers can confirm they were filed. The executor does NOT need to fix v1; just file the issues.

- **B2 — Revert `apply_merges` kwarg to v1's `apply_merge`.** At [v2/curation.py:108](../../../../src/spyglass/spikesorting/v2/curation.py#L108) in `CurationV2.insert_curation`, rename `apply_merges` to `apply_merge` and update all internal references. v1's signature at [v1/curation.py:50](../../../../src/spyglass/spikesorting/v1/curation.py#L50) is `apply_merge=False`. v2 silently renamed without a shim; existing user code calling `apply_merge=True` would silently get the default behavior because Python doesn't error on extra unknown kwargs unless `**kwargs` is absent. Actually — verify: does `insert_curation` accept `**kwargs`? Read the signature; if it does NOT accept `**kwargs`, a v1 caller passing `apply_merge=True` raises `TypeError`. If it DOES, the kwarg is silently dropped, which is worse. Either way, revert to `apply_merge`.

  Also update [overview.md § Resolved Design Decisions](overview.md#resolved-design-decisions) if `apply_merges` is documented there.

- **B3 — Add `CurationV2.MergeGroup` part table for queryable merge-group provenance.** This is the **one deliberate schema addition** in Phase 1b, chosen over the v1 NWB-column pattern for queryability and FK validation. The user-decided rationale is documented in [overview.md § Resolved Design Decisions](overview.md#resolved-design-decisions); the executor must update that section as part of this task. The new part table:

  ```python
  # In src/spyglass/spikesorting/v2/curation.py, on class CurationV2
  class MergeGroup(SpyglassMixinPart):
      """Per-merge-group provenance: kept unit ← contributor units.

      Restores v1's merge-group recoverability lost when CurationV2
      replaced v1's `merge_groups` NWB column with a per-unit kept-set.
      Choice of part table over NWB column documented in overview.md
      § Resolved Design Decisions: queryability + FK validation outweigh
      the schema-neutral promise for Phase 1b.

      One row per (kept_unit_id, contributor_unit_id). The kept unit
      appears as its own contributor for 1-unit groups (no merge); this
      makes "list every unit's merge provenance" a single restriction.
      """

      definition = """
      -> CurationV2.Unit
      contributor_unit_id: int  # source unit before the merge
      ---
      """
  ```

  Populate in `CurationV2.insert_curation` from the `kept_unit_to_contributors` dict already built by `_build_curated_unit_rows` ([v2/curation.py:407-425](../../../../src/spyglass/spikesorting/v2/curation.py#L407-L425)):

  ```python
  # Inside the existing transaction_or_noop block, after Unit/UnitLabel:
  merge_group_rows = [
      {
          "sorting_id": sorting_id,
          "curation_id": curation_id,
          "unit_id": kept_uid,
          "contributor_unit_id": contributor_uid,
      }
      for kept_uid, contributors in kept_unit_to_contributors.items()
      for contributor_uid in contributors
  ]
  cls.MergeGroup.insert(merge_group_rows)
  ```

  Add accessor `CurationV2.get_merge_groups(key) -> dict[int, list[int]]` returning `{kept_unit_id: [contributors]}` so callers don't have to do the join manually. Update `get_merged_sorting` to actually use these merge-group rows: currently at [v2/curation.py:579-588](../../../../src/spyglass/spikesorting/v2/curation.py#L579-L588) it's an alias for `get_sorting`; restore v1's semantic where it returns the post-merge SI sorting via `sc.MergeUnitsSorting` constructed from the part-table rows. v1's pattern at [v1/curation.py:228-266](../../../../src/spyglass/spikesorting/v1/curation.py#L228-L266) is the reference.

  Note on the zero-migration policy: this is technically a part-table addition to an existing v2 master table, which the policy's letter would forbid (every v2 table designed in its final shape in the phase that introduces it). The user has explicitly authorized this exception. Document it in `overview.md` so a future reviewer can find the rationale.

- **B5 — Handle multi-segment NWBs (improvement over v1).** Both v1 and v2 currently silently process only segment 0 when a multi-segment NWB is ingested — v1's `_get_recording_timestamps` helper exists at [utils.py:260-279](../../../../src/spyglass/spikesorting/utils.py#L260-L279) but is NOT called from v1's `_get_preprocessed_recording`. The user has authorized fixing this in v2.

  Replace direct `recording.get_times()` calls with `_get_recording_timestamps(recording)` at the two call sites in `Recording`:
  - [v2/recording.py:955](../../../../src/spyglass/spikesorting/v2/recording.py#L955) inside `_restrict_recording` — `times = recording.get_times()`.
  - [v2/recording.py:1089](../../../../src/spyglass/spikesorting/v2/recording.py#L1089) inside `_write_nwb_artifact` — `times = recording.get_times()`.

  ```python
  from spyglass.spikesorting.utils import _get_recording_timestamps
  times = _get_recording_timestamps(recording)
  ```

  Add a test `test_multi_segment_nwb_handled` that synthesizes a 2-segment SI recording, populates `Recording`, and asserts the written `ElectricalSeries` `timestamps` array covers BOTH segments contiguously (not just segment 0). The `synthetic_5min_32ch_30khz.nwb` fixture used by the RSS test can be extended to add a multi-segment variant.

- **B7 — Document why v2 keeps the stricter heterogeneous-gain raise.** v2's behavior at [v2/recording.py:1100-1105](../../../../src/spyglass/spikesorting/v2/recording.py#L1100-L1105) raises `ValueError` when channels in a sort group have different gains, while v1 at [v1/recording.py:858](../../../../src/spyglass/spikesorting/v1/recording.py#L858) silently picks `gains[0]`. v1's silent pick is mathematically wrong — the chosen gain is applied as the universal `conversion` factor on the ElectricalSeries, so signals on channels with different gains are scaled incorrectly. v2 is the correct behavior; we just need to document why we kept it instead of matching v1.

  Add a code comment above the v2 check:

  ```python
  # v2 raises instead of silently picking gains[0] (v1's behavior at
  # v1/recording.py:858). The "pick first" approach is a latent
  # correctness bug: the chosen gain becomes the universal conversion
  # factor on the ElectricalSeries, so signals on channels with
  # heterogeneous gains are scaled by the wrong factor. Catching
  # the input invariant at populate time is preferred to silently
  # producing incorrectly-scaled data.
  gains = _np.unique(recording.get_channel_gains())
  if len(gains) != 1:
      raise ValueError(...)
  ```

  No code logic change — this task is comment-only.

- **R11 — Add `Sorting.delete()` override to clean up the `analyzer_folder` on disk when the row is deleted.** Cleans up the **analyzer folder** specifically when `(Sorting & key).delete()` is called — distinct from R3 (sorter tempdir on sort completion) and from N51 (analyzer folder on populate failure). v2 introduced `analyzer_folder` as a path column on `Sorting` ([sorting.py:357](../../../../src/spyglass/spikesorting/v2/sorting.py#L357)) holding waveforms, templates, and metric extensions for the sort (potentially 5–50 GB for chronic-scale recordings). When the DataJoint row is deleted, the folder stays. v1 had no equivalent folder, so this is a v2-introduced leak — but v2 already knows the cleanup pattern: `ArtifactDetection.delete()` at [v2/artifact.py:694-744](../../../../src/spyglass/spikesorting/v2/artifact.py#L694-L744) explicitly removes the linked `IntervalList` rows because DataJoint doesn't cascade through `interval_list_name`-keyed dependencies. Apply the same pattern to `Sorting`:

  ```python
  # On class Sorting, after the existing accessor methods:
  def delete(self, *args, safemode=None, **kwargs):
      """Override that also removes the analyzer scratch folder on disk.

      The folder at ``_analyzer_path(key)`` is not tracked by DataJoint
      and would otherwise leak (5-50 GB per chronic sort). Mirrors
      ArtifactDetection.delete's IntervalList cleanup pattern at
      v2/artifact.py:694-744. Collects folder paths BEFORE the cascade
      delete because the row needed to compute the path is gone after.
      """
      from spyglass.spikesorting.v2.utils import _analyzer_path
      import shutil as _shutil

      folders_to_remove = [
          _analyzer_path({"sorting_id": row["sorting_id"]})
          for row in self.fetch("KEY", as_dict=True)
      ]
      if safemode is None:
          super().delete(*args, **kwargs)
      else:
          super().delete(*args, safemode=safemode, **kwargs)
      for folder in folders_to_remove:
          if folder.exists():
              _shutil.rmtree(folder, ignore_errors=False)
  ```

  Collect the path list BEFORE `super().delete()` because the row is gone afterward. Use `_shutil.rmtree(..., ignore_errors=False)` so a permission error surfaces loudly; cleanup_exc-style swallowing is reserved for the explicit `pragma: no cover` defensive paths.

- **R13 — Restore v1's `min_length=1` filter on artifact-removed valid times.** v1 at [v1/artifact.py:327-328](../../../../src/spyglass/spikesorting/v1/artifact.py#L327-L328) does `sort_interval_valid_times.subtract(artifact_intervals_s, min_length=1)` — drops valid-interval slivers shorter than 1 second. v2's hand-written subtraction at [v2/artifact.py:644-655](../../../../src/spyglass/spikesorting/v2/artifact.py#L644-L655) has no such filter. A noisy recording with frequent artifacts leaves millisecond-scale slivers that downstream `Sorting._apply_artifact_mask` iterates one by one, and SI's sorter may crash on micro-intervals.

  Add the filter to `ArtifactDetection._detect_artifacts` after the kept-list construction (around [v2/artifact.py:647-655](../../../../src/spyglass/spikesorting/v2/artifact.py#L647-L655)):

  ```python
  # Match v1/artifact.py:327-328 — drop valid-interval slivers shorter
  # than min_length seconds. Default 1.0 matches v1; expose as a
  # validated parameter on ArtifactDetectionParamsSchema so users can
  # widen it for tetrode recordings with frequent chewing artifacts.
  kept = [
      [start, end]
      for start, end in kept
      if (end - start) >= validated.min_length_s
  ]
  ```

  Add `min_length_s: float = Field(default=1.0, gt=0.0)` to `ArtifactDetectionParamsSchema` ([v2/_params/artifact_detection.py](../../../../src/spyglass/spikesorting/v2/_params/artifact_detection.py)). Bump `params_schema_version` if R7 didn't already bump it, otherwise reuse the bump. Update the `"default"` content row in `ArtifactDetectionParameters._DEFAULT_CONTENTS` to set the new field.

- **R15 — Accept `labels=None` in `CurationV2.insert_curation` matching v1.** v1 signature at [v1/curation.py:49](../../../../src/spyglass/spikesorting/v1/curation.py#L49) is `labels: Union[None, Dict[str, List[str]]] = None`; passing `None` is valid and means "no labels." v2 at [v2/curation.py:169-173](../../../../src/spyglass/spikesorting/v2/curation.py#L169-L173) raises `ValueError` when `labels is None`, forcing callers to pass `{}`. v1's behavior is the more permissive choice.

  Change the explicit-None rejection to a normalization:

  ```python
  # In CurationV2.insert_curation (v2/curation.py:169-173):
  if labels is None:
      labels = {}   # match v1: None is equivalent to "no labels"
  # ... rest of existing logic ...
  ```

  Remove the `ValueError("labels=None is invalid")` branch entirely. The downstream `_validate_labels` and label-row construction already handle empty dicts correctly. Update the docstring to match v1's intent.

- **R17 — Use `_hash_nwb_recording` per the documented contract; current `cache_hash` violates `shared-contracts.md`.** [shared-contracts.md § Recording Cache Format](shared-contracts.md#recording-cache-format) line 164 binds `cache_hash` to "content digest of the recording's `AnalysisNwbfile`, computed by Spyglass's `NwbfileHasher` (the helper `_hash_nwb_recording(analysis_file_name)` is a thin wrapper, so v2 verification uses the same hashing path as the v1 recompute machinery rather than a parallel implementation)." The runtime at [v2/recording.py:1097](../../../../src/spyglass/spikesorting/v2/recording.py#L1097) hashes only `data.tobytes()` via `hashlib.sha256` — a divergent strategy that misses timestamps, electrode tables, conversion factors, and other NWB metadata. The `_hash_nwb_recording` helper exists at [v2/utils.py:316-337](../../../../src/spyglass/spikesorting/v2/utils.py#L316-L337) but is never called.

  Replace the data-only hash with the helper:

  ```python
  # In Recording._write_nwb_artifact at v2/recording.py:1097:
  # Compute cache_hash AFTER the NWB write so the digest reflects what
  # was actually persisted (including timestamps, electrodes, conversion).
  # Matches the shared-contracts.md § Recording Cache Format spec.
  from spyglass.spikesorting.v2.utils import _hash_nwb_recording
  cache_hash = _hash_nwb_recording(analysis_file_name)
  ```

  Important reordering: the current implementation computes `cache_hash` BEFORE `io.write(nwbfile)`. After this change, the hash is computed AFTER the write, on the persisted file. Drop the no-longer-needed `import hashlib` and `np.ascontiguousarray(data).tobytes()` work from `_write_nwb_artifact`. The streaming-hash loop introduced by Phase 1b's chunked-iterator task (above) also goes away — streaming the hash only made sense when hashing trace bytes; `NwbfileHasher` handles its own I/O.

  Update `_rebuild_nwb_artifact` at [v2/recording.py:827-896](../../../../src/spyglass/spikesorting/v2/recording.py#L827-L896) the same way: post-write `_hash_nwb_recording(analysis_file_name)` instead of the streaming-hash pattern from Phase 1b's chunked-iterator task. The mismatch warning at [recording.py:888-896](../../../../src/spyglass/spikesorting/v2/recording.py#L888-L896) is unchanged in shape — only the hash function differs.

  **Baseline implication**: this changes the `cache_hash` value for every Phase 1 row. The bit-equivalence baseline test must NOT compare `cache_hash` against Phase 1's stored values (they were data-only hashes; post-R17 they are `NwbfileHasher` digests). The test compares re-read trace + timestamps arrays directly against the baseline pickle — that comparison is unaffected. See the updated bit-equivalence test description in the Validation slice for the corresponding adjustment.

- **R18 — Remove dead `common_reference.reference` field from `PreprocessingParamsSchema`.** Field defined at [v2/_params/preprocessing.py § CommonReferenceParams](../../../../src/spyglass/spikesorting/v2/_params/preprocessing.py) as `reference: Literal["global", "single", "local"] = "global"`. Runtime at [v2/recording.py:1019-1041](../../../../src/spyglass/spikesorting/v2/recording.py#L1019-L1041) hardcodes the choice based on `ref_channel_id`:
  - `ref_channel_id >= 0` → `reference="single"`
  - `ref_channel_id == -2` → `reference="global"`
  - `ref_channel_id == -1` → no referencing applied
  - Other values → raise

  The schema field is never read. v1 also hardcoded the same dispatch based on `ref_channel_id` (see [v1/recording.py:597-619](../../../../src/spyglass/spikesorting/v1/recording.py#L597-L619)). Removing the field matches v1 exactly and eliminates a false promise to users.

  Edits:
  - Remove `reference: Literal["global", "single", "local"] = "global"` from `CommonReferenceParams`.
  - Keep `operator: Literal["median", "average"] = "median"` (it IS used on the global branch).
  - The schema's `extra="forbid"` setting will surface any old parameter rows that explicitly set `reference=...` as a Pydantic validation error on next insert/load; that is the intended behavior because the rows would be lying about runtime behavior. Existing Phase 1 default contents do NOT set `reference` explicitly (they accept the default), so the removal does not break the shipped presets.
  - Bump `params_schema_version` to mark the schema-incompatible change. Reuse the version bump if R7 or R13 already bumped it.
  - Document in the schema's docstring that "v2 hardcodes the reference mode based on `ref_channel_id` (single if a specific electrode is named, global if -2 is configured for global median, none if -1). The `reference` field of v1's preprocessing params is intentionally not exposed in v2 because no production v1 workflow used it." Cite v1 line refs in the docstring so a future reviewer doesn't think this was a v2 oversight.

  Note on the `whiten` Pydantic field: also dead in Phase 1 (whitening is deferred to the sorter), but is forward-compat scaffolding for Phase 3's `ConcatenatedRecording.make` motion-correction → post-motion whitening flow. Do **not** remove the `whiten` field. Add a one-line docstring note: *"whiten is dead in Phase 1 (Phase 1 defers whitening to the sorter via Sorting._run_sorter's external whitening path); activates in Phase 3 alongside motion correction. See PreprocessingParamsSchema.to_post_motion_dict."*

- **Audit-discovered v1 parity items (N19-N48).** A subsequent multi-agent v1↔v2 sweep surfaced 30 additional regressions / behavior divergences not caught by the earlier audits. They fall into eight thematic groups; the full per-item registry is in the PR description, but the task list below covers each one. Severity in brackets.

### Artifact-detection algorithm

- **N19 [HIGH] — Restore `noise_levels=[1.0]` to `clusterless_thresholder` dispatch.** v2 at [v2/sorting.py:711](../../../../src/spyglass/spikesorting/v2/sorting.py#L711) strips `noise_levels` along with `outputs`/`random_chunk_kwargs`. When `noise_levels` is absent, SI's `detect_peaks` computes per-channel MAD and treats `detect_threshold` as a MAD multiplier; v1 deliberately kept `noise_levels=[1.0]` so `detect_threshold` stays in microvolts. A user's `detect_threshold=100.0` was 100 µV in v1; in v2 it becomes ~100×MAD ≈ 500 µV. Fix: drop `noise_levels` from the strip list, OR explicitly forward `noise_levels=[1.0]` to `detect_peaks`. v1 evidence at [v1/sorting.py:177,402-404](../../../../src/spyglass/spikesorting/v1/sorting.py#L177).

- **N20 [HIGH] — Restore v1's per-frame-across-channels z-score in artifact detection.** v1 computes `stats.zscore(traces, axis=1)` at [utils.py:185,193](../../../../src/spyglass/spikesorting/utils.py#L185), z-scoring ACROSS channels per frame (large common-mode pop detection — chewing, head movement). v2 at [v2/artifact.py:585-587](../../../../src/spyglass/spikesorting/v2/artifact.py#L585-L587) z-scores per channel over time, detecting per-channel sustained deviation instead. Same `zscore_thresh` value selects entirely different frames. Restore v1's semantics:

  ```python
  # Per-frame across-channels z-score (matches v1 utils.py:185,193).
  # Detects common-mode artifact events (chewing, movement), not
  # per-channel baseline drift.
  ch_mean = traces_uv.mean(axis=1, keepdims=True)
  ch_std = traces_uv.std(axis=1, keepdims=True) + 1e-12
  zscores = _np.abs((traces_uv - ch_mean) / ch_std)
  above_z = zscores > validated.zscore_thresh
  ```

- **N23 [MEDIUM] — Flip artifact combine logic from AND back to OR.** v2 at [v2/artifact.py:594-600](../../../../src/spyglass/spikesorting/v2/artifact.py#L594-L600) does `channel_hit = above_amp & above_z` when both thresholds set. v1 at [utils.py:198](../../../../src/spyglass/spikesorting/utils.py#L198) does `np.logical_or(above_z, above_a)`. v1 is the belt-and-suspenders default — flag a frame if EITHER detector trips. Fix: restore OR semantics. If user-configurability is desired, add a `combine: Literal["and","or"] = "or"` field to `ArtifactDetectionParamsSchema`, default "or" matching v1.

  **Update the existing AND-encoding test** at [tests/spikesorting/v2/test_single_session_pipeline.py:2948](../../../../tests/spikesorting/v2/test_single_session_pipeline.py#L2948) (`test_detect_artifacts_amplitude_and_zscore_combined`). The current docstring explicitly names "AND mode at line 596" and the test was built to pin AND semantics. When N23 lands and the combine flips to OR, that test's assertion will fail because the 80 µV uniform baseline (above amplitude threshold everywhere; z-score ~0 because the baseline is uniform) will be flagged in its entirety. **Do not delete the test** — keep its synthetic-data setup but update the assertion to verify OR semantics: with both thresholds set under default-OR, the baseline frames trip on amplitude alone AND the 200 µV step deviation trips on both. The valid-times array should now reflect the OR coverage. If a `combine` field is added with default `"or"`, the test should also parameterize over `combine=("or", "and")` to verify both code paths.

- **N37, N46 [LOW] — Restore artifact-detection verbose logging.** v2's `_detect_artifacts` is silent; v1 emitted `logger.info` for threshold config and `logger.warning("No artifacts detected.")` when the detection found nothing ([v1/artifact.py:258-261,318](../../../../src/spyglass/spikesorting/v1/artifact.py#L258-L261)). Add equivalents at the top of `_detect_artifacts` (log chosen thresholds + detect-on/off) and at the empty-frames branch ([v2/artifact.py:603](../../../../src/spyglass/spikesorting/v2/artifact.py#L603)).

- **N47 [LOW] — Surface the `resolve_source` swallow in `ArtifactDetection.delete()`** at [v2/artifact.py:712-713](../../../../src/spyglass/spikesorting/v2/artifact.py#L712-L713). Currently `except Exception: continue` silently skips IntervalList cleanup. Add `logger.error("ArtifactDetection.delete: resolve_source failed for {row}; leaving IntervalList rows in place.")` before the `continue`.

### Sorter dispatch + tempdir

- **N38 [LOW] — Restore KS2.5/KS3/IronClust kwarg-strip carve-out.** Already partially in R9 (Singularity flag). Add the matching kwarg strip: remove `tempdir`, `mp_context`, `max_threads_per_process` from `sorter_params` before calling `sis.run_sorter` for these three sorters. Reference [v1/sorting.py:439-450](../../../../src/spyglass/spikesorting/v1/sorting.py#L439-L450). Treat as part of R9's task surface.

- **N39 [LOW] — Restore `os.chmod(tempdir, 0o777)` after `mkdtemp`.** v1 sets world-writable mode at [v1/sorting.py:421](../../../../src/spyglass/spikesorting/v1/sorting.py#L421) so SI sorter subprocesses with different uid (Docker/Singularity rootless, slurm scenarios) can write into the dir. Single-line restore inside `_run_sorter` (within the R3 tempdir-cleanup task, since both touch the same code).

- **N25 [MEDIUM] — Restore `obs_intervals` on every `add_unit` call in the sort NWB.** v1 writes `obs_intervals=obs_interval` (artifact-removed sort interval, shape `(n_intervals, 2)`) on every unit at [v1/sorting.py:597](../../../../src/spyglass/spikesorting/v1/sorting.py#L597). v2 at [v2/sorting.py:817](../../../../src/spyglass/spikesorting/v2/sorting.py#L817) writes only `spike_times` and `id`. `obs_intervals` is a standard pynwb Units column (not v1-specific); without it, downstream firing-rate computations can't get the observation window from the units NWB. Compute the observation interval from the upstream `Recording` row's IntervalList and pass it to `add_unit`.

- **N22 + N30 + L2.contract-N8 (combined) [HIGH] — Apply `_resolved_job_kwargs` at every compute stage.** The shared-contracts.md `Job-Kwargs Resolution` invariant at [shared-contracts.md:443-470](shared-contracts.md#job-kwargs-resolution) binds:
  - `Recording.make() → PreprocessingParameters.job_kwargs`
  - `ArtifactDetection.make() → ArtifactDetectionParameters.job_kwargs`
  - `Sorting.make() → SorterParameters.job_kwargs`

  Phase 1 only honors the third (and only in `_build_analyzer`, not in `sis.run_sorter`). Fixes:
  - `Recording.make_compute` (after the R5/R6 task lands): fetch `PreprocessingParameters.job_kwargs`, call `_resolved_job_kwargs(...)`. Even if the NWB write path doesn't take SI-style `job_kwargs`, the resolver must be called so the override channels exist (per the contract's "tests rely on being able to override").
  - `ArtifactDetection.make_compute`: same pattern with `ArtifactDetectionParameters.job_kwargs`.
  - `Sorting._run_sorter`: compute `job_kwargs = _resolved_job_kwargs(sorter_row["job_kwargs"])` once and pass `**job_kwargs` to BOTH `sis.run_sorter(...)` and `analyzer.compute(...)`. Currently only the analyzer call receives it (at [v2/sorting.py:783](../../../../src/spyglass/spikesorting/v2/sorting.py#L783)).

### Curation

- **N21 [HIGH] — Restore `CurationV2.get_merged_sorting` to actually apply merges at fetch.** v1's [v1/curation.py:228-266](../../../../src/spyglass/spikesorting/v1/curation.py#L228-L266) always reads `merge_groups` from the NWB and applies `sc.MergeUnitsSorting`, regardless of `apply_merge=False` at insert. v2 at [v2/curation.py:579-588](../../../../src/spyglass/spikesorting/v2/curation.py#L579-L588) is just `return self.get_sorting(key)`. Code that built a curation with `apply_merges=False` (to preview) and called `get_merged_sorting` expecting merged trains gets un-merged silently.

  **Hard prerequisite: land B3 first.** N21's implementation calls `cls.get_merge_groups(key)`, which is the accessor introduced by B3's `CurationV2.MergeGroup` part table. If the executor lands N21 before B3, this method has no data source and fails. Sequence: B3 (add MergeGroup part + `get_merge_groups` accessor) → N53 (convert all four `CurationV2` accessors to `@classmethod`) → N21 (rewrite `get_merged_sorting` body). With those in place, the body becomes:

  ```python
  @classmethod
  def get_merged_sorting(cls, key):
      """Return the curated SI BaseSorting with all merge groups applied,
      regardless of merges_applied state on the row. Matches v1 semantics:
      merges are queried from CurationV2.MergeGroup and applied via
      sc.MergeUnitsSorting."""
      import spikeinterface.curation as sc
      base = cls.get_sorting(key)
      merge_groups = cls.get_merge_groups(key)  # accessor from B3
      if not any(len(c) > 1 for c in merge_groups.values()):
          return base  # nothing to merge
      units_to_merge = [contribs for kept, contribs in merge_groups.items() if len(contribs) > 1]
      return sc.MergeUnitsSorting(parent_sorting=base, units_to_merge=units_to_merge)
  ```

  This replaces the current alias body. The test `test_get_merged_sorting_applies_merges_at_fetch` populates a curation with `apply_merges=False, merge_groups=[[1,2],[4,5]]`, then asserts `get_merged_sorting(key).unit_ids` collapses [1,2] → 1 and [4,5] → 4.

- **N24 [MEDIUM] — Restore root-curation idempotency.** v1 at [v1/curation.py:88-93](../../../../src/spyglass/spikesorting/v1/curation.py#L88-L93) detects a pre-existing root row (`parent_curation_id == -1`) for the same `sorting_id` and returns its key without inserting a duplicate. v2 at [v2/curation.py:207-210](../../../../src/spyglass/spikesorting/v2/curation.py#L207-L210) always auto-increments `curation_id`, producing a new row + new analysis NWB + new merge-table entry every call. Repeat-runs of any orchestration script grow rows. Fix: short-circuit when `parent_curation_id == -1` AND a root row exists for `sorting_id` — emit `logger.warning("CurationV2: root curation already exists for sorting_id={...}; returning existing key.")` and return the existing PK without staging a new NWB.

- **N26 [MEDIUM] — Fix `curation_label` NWB column format AND the contradictory docstring.** v2's docstring at [v2/curation.py:88-93](../../../../src/spyglass/spikesorting/v2/curation.py#L88-L93) claims the column is an indexed (list-valued) column, matching v1. The actual code at [v2/curation.py:498-507](../../../../src/spyglass/spikesorting/v2/curation.py#L498-L507) writes a comma-separated string. v1 at [v1/curation.py:398-403](../../../../src/spyglass/spikesorting/v1/curation.py#L398-L403) writes `index=True` with list-of-lists values. External NWB readers that expect v1's shape (e.g., `v1/figurl_curation.py:83-101` which does `list(nwb_sorting.get("curation_label", []))`) will misparse multi-labeled units. Fix: write `curation_label` as `index=True` with the labels list per unit, matching v1 exactly. Update or remove the contradictory docstring.

- **N41 [LOW] — Restore `merges_applied` to user intent rather than effective state.** v2 at [v2/curation.py:244](../../../../src/spyglass/spikesorting/v2/curation.py#L244) stores `bool(apply_merge and merge_groups)`. v1 at [v1/curation.py:123](../../../../src/spyglass/spikesorting/v1/curation.py#L123) stores `apply_merge` verbatim. A user passing `apply_merge=True, merge_groups=None` gets `True` from v1, `False` from v2. Restore verbatim-bool. (v2's semantics is arguably more accurate, but the principle is "match v1 unless documented" — no documented rationale exists.)

- **N42 [LOW] — Restore label/metric columns to `get_sorting(as_dataframe=True)`.** v1 at [v1/curation.py:197-209](../../../../src/spyglass/spikesorting/v1/curation.py#L197-L209) returns `nwbf.units.to_dataframe()` directly, including `curation_label`, `merge_groups`, and any metrics columns. v2 at [v2/curation.py:562-577](../../../../src/spyglass/spikesorting/v2/curation.py#L562-L577) constructs a fresh DataFrame with only `unit_id` + `spike_times`. Fix: join in the `curation_label` column from `CurationV2.UnitLabel` so user-facing `as_dataframe=True` matches v1 shape. (Metrics columns are Phase 2 territory per B4; skip those.)

### Recording

- **N28 [MEDIUM] — Honor the stored `electrical_series_path` in `Recording.get_recording`.** Spec at [shared-contracts.md:162-163](shared-contracts.md) says the column drives the reader path. Runtime at [v2/recording.py:825](../../../../src/spyglass/spikesorting/v2/recording.py#L825) ignores it and uses SI auto-detect, which is ambiguous if a future writer puts more than one `ElectricalSeries` in the analysis NWB. Fix: `se.read_nwb_recording(abs_path, electrical_series_path=row["electrical_series_path"], load_time_vector=True)`.

  **Two production call sites need the same fix, not one**:
  1. `Recording.get_recording` at [v2/recording.py:811-825](../../../../src/spyglass/spikesorting/v2/recording.py#L811-L825) — the user-facing accessor.
  2. `Recording._rebuild_nwb_artifact` at [v2/recording.py:827-896](../../../../src/spyglass/spikesorting/v2/recording.py#L827-L896) — the recompute path. The rebuild reads the raw NWB and re-emits the preprocessed artifact; if it reads via SI auto-detect, a future raw NWB with multiple ElectricalSeries (e.g., LFP + raw) silently picks the wrong source.

  Also apply to `_recording_t_start` ([v2/sorting.py:523-543](../../../../src/spyglass/spikesorting/v2/sorting.py#L523-L543)) which opens the AnalysisNwbfile directly and reads the series by name. Verify the existing helper already honors `electrical_series_path` from the recording row — if not, fix as the third callsite.

- **N31 [MEDIUM] — Restore v1's `channel_name` electrode-column lookup.** v1 at [v1/recording.py:683-712](../../../../src/spyglass/spikesorting/v1/recording.py#L683-L712) reads the raw NWB's electrodes table and, if a `channel_name` string column exists, maps integer `electrode_id` → NWB string channel-name (which SI 0.104's `read_nwb_recording` uses as the channel ID). v2 at [v2/recording.py:984-994](../../../../src/spyglass/spikesorting/v2/recording.py#L984-L994) unconditionally returns ints. The Frank-lab MEArec fixture lacks `channel_name`, so this hides in tests — but production NWBs with `channel_name` would fail to match the SI channel IDs. Port v1's branch verbatim.

- **N32 [MEDIUM] — Restore `tetrode_12.5` probe-geometry patch.** v1 at [v1/recording.py:630-643](../../../../src/spyglass/spikesorting/v1/recording.py#L630-L643) explicitly constructs a `pi.Probe(ndim=2)` with positions `[[0,0],[0,12.5],[12.5,0],[12.5,12.5]]` and contact radius 6.25 µm for sort groups whose probe type is `tetrode_12.5`, AND has exactly 4 channels, AND lives on a single electrode group — covering legacy Frank-lab NWBs where contact positions weren't written into the electrode table. v2 has no equivalent. Port verbatim into a helper called after channel_slice. Geometry-aware sorters (Kilosort, MountainSort5) on legacy tetrode NWBs are affected; clusterless_thresholder + MS4 are not.

### Merge dispatch

- **N29 [MEDIUM] — Raise on unknown restriction keys in `_get_restricted_merge_ids_v2`.** Spec at [shared-contracts.md:385](shared-contracts.md) says "Unknown restriction fields should fail clearly rather than silently returning unrelated merge IDs." Current code at [spikesorting_merge.py:152-191](../../../../src/spyglass/spikesorting/spikesorting_merge.py#L152-L191) silently drops unknown keys via whitelist filtering. Fix: after collecting allowed-key restrictions, compute `unknown = set(key) - allowed` and raise `ValueError(f"Unknown restriction keys for v2 spike-sorting merge: {sorted(unknown)}. Allowed keys: ...")` if non-empty.

- **N40 [LOW] — Plumb `restrict_by_artifact` through `_get_restricted_merge_ids_v2`.** Spec promises API compatibility; current v2 branch ignores the parameter ([spikesorting_merge.py:296-306](../../../../src/spyglass/spikesorting/spikesorting_merge.py#L296-L306)). If `restrict_by_artifact=True` AND `interval_list_name=f"artifact_{uuid}"` is supplied, look up the artifact `IntervalList` row by the `(nwb_file_name, interval_list_name=f"artifact_{artifact_id}")` convention rather than the bare-UUID v1 convention.

### Schema cleanup

- **N34 [MEDIUM] — `Kilosort4Schema` `extra="forbid"` blocks v1's escape hatch.** v1 expanded sorter contents via `sis.get_default_sorter_params(sorter)` ([v1/sorting.py:184-189](../../../../src/spyglass/spikesorting/v1/sorting.py#L184-L189)); v2 at [v2/_params/sorter.py:84-103](../../../../src/spyglass/spikesorting/v2/_params/sorter.py#L84-L103) types only 5 KS4 kwargs with `extra="forbid"`, blocking users who rely on other documented KS4 params (e.g., `batch_size`, `nearest_chans`). Fix: switch `Kilosort4Schema.model_config` to `extra="allow"`. Documented KS4 fields stay as typed knobs; extras pass through to SI. Same treatment for `SpykingCircus2Schema` and `Tridesclous2Schema` if they currently use `extra="forbid"`.

- **N35 [MEDIUM] — Resolve the `WhitenParams` schema/runtime mismatch.** `default_franklab` preset at [v2/recording.py:507-512](../../../../src/spyglass/spikesorting/v2/recording.py#L507-L512) ships `WhitenParams(dtype="float32")` but `to_post_motion_dict()` at [v2/_params/preprocessing.py:73-77](../../../../src/spyglass/spikesorting/v2/_params/preprocessing.py#L73-L77) is never called in Phase 1. Saved blob says "whiten on"; no whitening happens. Fix: flip `default_franklab.whiten = None` to match `default_neuropixels` and `no_filter` presets. Add a docstring note explaining that `WhitenParams` is forward-compat scaffolding activated in Phase 3 motion-correction; Phase 1 defers whitening to the sorter via `Sorting._run_sorter`'s external-whitening path (R4).

- **N48 [LOW] — `ClusterlessThresholderSchema` dead fields, post-N19.** Schema at [v2/_params/sorter.py:118-125](../../../../src/spyglass/spikesorting/v2/_params/sorter.py#L118-L125) declares `noise_levels`, `random_chunk_kwargs`, `outputs`. **After N19 lands**, `noise_levels` is NO LONGER dead — N19 forwards it to `detect_peaks` to preserve uV threshold semantics; keep it on the schema with its `[1.0]` default. The two remaining fields ARE still dead post-N19: `random_chunk_kwargs` and `outputs` are stripped at runtime by `_run_sorter`. Drop those two from the schema, OR annotate them with `# kept for v1 row-shape parity; stripped at runtime`. Prefer the drop unless there's a v1 row migration concern — `extra="forbid"` will already reject any future row that supplies them. Cross-reference N19 in the schema docstring so a future reader understands why `noise_levels` is intentional and the other two are not.

### Code naming + UX

- **N27 [MEDIUM] — Eliminate Phase-label leakage from runtime code, tests, and user-facing exceptions.** Spec at [shared-contracts.md:56-70](shared-contracts.md) is "do not weaken" on this. Verify by `grep -rn "Phase 1\|Phase 1b" src/spyglass/spikesorting/v2/ tests/spikesorting/v2/`. Known leak points: [v2/artifact.py:175](../../../../src/spyglass/spikesorting/v2/artifact.py#L175) exception message ends "for Phase 1."; multiple test-file references in `tests/spikesorting/v2/test_single_session_pipeline.py` and `test_legacy_runtime_boundary.py`. Rename to behavior-named language ("until the cross-recording branch ships", "until concat support lands", etc).

- **N44 [LOW] — Document `test_mode` semantic gap.** v1's `SortGroup.set_group_by_shank` short-circuits when `test_mode=True` and existing rows exist (v1/recording.py:82-83). v2 doesn't honor `test_mode` anywhere. Test fixtures that previously relied on this need to opt into `delete_existing_entries=True, confirm=True` or supply explicit `sort_group_ids`. Add a note to `docs/src/Features/SpikeSortingV2.md` describing the fixture-rerun convention.

- **N45 [LOW] — Add `get_spiking_sorting_v2_merge_ids` wrapper.** v1 ships `get_spiking_sorting_v1_merge_ids` at [v1/utils.py:37-109](../../../../src/spyglass/spikesorting/v1/utils.py#L37-L109) as a notebook-discoverable helper. v2 has the equivalent functionality on `SpikeSortingOutput._get_restricted_merge_ids_v2` but as a private method, not a parallel user-facing helper. Add a thin wrapper in `src/spyglass/spikesorting/v2/utils.py`:

  ```python
  def get_spiking_sorting_v2_merge_ids(restriction: dict, as_dict: bool = False) -> list:
      """v2-side parallel of get_spiking_sorting_v1_merge_ids."""
      from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
      return SpikeSortingOutput()._get_restricted_merge_ids_v2(
          restriction, as_dict=as_dict
      )
  ```

### Downstream consumers (sparse unit_id indexing)

- **N36 [MEDIUM] — Fix v1-bug-#1273-flavored consumers that assume contiguous unit IDs.** v2 by design creates sparse unit_id sets when merges are applied (the kept-unit retains the head ID, contributors disappear from the keep set). Several downstream paths assume contiguous IDs and either crash or silently misindex on v2 merge_ids. The same v1 bug existed but was less visible because v1's merge applied at fetch time only — the stored row had contiguous IDs. Files to fix:

  - **`src/spyglass/decoding/v1/waveform_features.py:155-169`** — clusterless decoding waveform extraction. Add a v2-aware branch that indexes by NWB `.id` (true unit_id) rather than positional index.

  - **`src/spyglass/spikesorting/analysis/v1/group.py:204-207`** — `SortedSpikesGroup` unit-id resolution. Use NWB `.id` for indexing.

  - **`src/spyglass/spikesorting/analysis/v1/unit_annotation.py:74-78, 127`** — `UnitAnnotation` row-keying. Same fix.

  These files live OUTSIDE `src/spyglass/spikesorting/v2/`, so the Phase 1b PR boundary technically crosses out of v2. Include them anyway — leaving these broken means decoding and unit annotation will silently fail on v2 merge_ids. Each fix is ~5-10 LOC plus a test that builds a sorting with a merge applied and exercises the consumer.

### NWB layer

- **N43 [LOW] — Restore `curation_label="uncurated"` placeholder column on Sorting NWB.** v1 at [v1/sorting.py:583-598](../../../../src/spyglass/spikesorting/v1/sorting.py#L583-L598) always writes a `curation_label` column with value `"uncurated"` on every unit at the sort stage (i.e., BEFORE curation). v2 at [v2/sorting.py:807-828](../../../../src/spyglass/spikesorting/v2/sorting.py#L807-L828) writes only `spike_times` and `id`. External readers that grep for the `curation_label` column on a pre-curation NWB fail. One-line restore.

### 4-agent audit findings (N49-N54)

A 4-agent sweep (test-suite parity, git-log mining, notebook-walkthrough parity, fresh-eyes cross-boundary) surfaced 5 substantive new findings + 9 test-coverage additions. Folded here:

- **N49 [HIGH] — Change `SpikeSortingOutput.get_restricted_merge_ids` default `sources` from `["v0","v1"]` to `["v0","v1","v2"]`.** The current default at [spikesorting_merge.py:244](../../../../src/spyglass/spikesorting/spikesorting_merge.py#L244) silently excludes v2 from the canonical v1-notebook-pattern lookup `SpikeSortingOutput.get_restricted_merge_ids({"nwb_file_name": "X"})`. A v2 user copying the v1 notebook gets zero merge_ids back and incorrectly concludes their v2 sort didn't register. One-line fix:

  ```python
  # spikesorting_merge.py:241-247
  def get_restricted_merge_ids(
      self,
      key: dict,
      sources: list = ["v0", "v1", "v2"],  # was ["v0", "v1"]
      restrict_by_artifact: bool = True,
      as_dict: bool = False,
  ) -> Union[None, list, dict]:
  ```

  Add `test_get_restricted_merge_ids_default_includes_v2` that calls the helper without an explicit `sources` arg on a v2-populated merge_id and asserts the result is non-empty.

  **Note on v0 inclusion**: keep `"v0"` in the default for backward compatibility, even though v0 is legacy. The v2 helper at [spikesorting_merge.py:_get_restricted_merge_ids_v2](../../../../src/spyglass/spikesorting/spikesorting_merge.py) raises `RuntimeError` if v2 isn't importable; that path is already routed through `_raise_v2_unavailable` so v0/v1-only environments get a clear error instead of silent miss.

- **N50 [HIGH] — Add non-monotonic-timestamp repair to `Recording.make`.** The bug exists in BOTH v1 master AND v2: raw NWBs with floating-point precision artifacts or epoch-stitching can have non-monotonic timestamps. Downstream `np.searchsorted` in `_consolidate_intervals` (R5) returns wrong frame indices, silently shifting sort-interval windows; `ArtifactDetection.add_removal_window` can crash with `start > stop`. A fix is in-flight on the upstream `copilot/fix-populating-artifact-detection` branch (commit d5079a8e) for v1; v2 should include the same defensive check. The fix:

  ```python
  # In Recording._restrict_recording or directly in Recording.make, after
  # `recording = se.read_nwb_recording(raw_path, load_time_vector=True)`:
  all_timestamps = recording.get_times()
  diffs = _np.diff(all_timestamps)
  if _np.any(diffs <= 0):
      n_issues = int(_np.sum(diffs <= 0))
      logger.warning(
          f"Source recording {raw_path!r} has {n_issues} non-monotonic "
          "timestamp(s) — likely floating-point precision or epoch-stitching "
          "artifacts at boundaries. Adjusting to strictly increasing; "
          "consider validating the source recording."
      )
      # Vectorized correction (matches optimized form from commit 963aa915):
      # for any i where ts[i] <= ts[i-1], set ts[i] = ts[i-1] + 1/fs.
      # Done via cumulative max to ensure monotonicity.
      sample_period = 1.0 / recording.get_sampling_frequency()
      adjusted = _np.maximum.accumulate(all_timestamps)
      mask = adjusted == all_timestamps  # where the original was already monotonic
      # increment any duplicates by sample_period
      adjusted[~mask] = adjusted[~mask] + sample_period * _np.arange(1, (~mask).sum() + 1)
      all_timestamps = adjusted  # use the corrected array downstream
  ```

  **Propagation of the corrected timestamps**: SI's `BaseRecording` exposes timestamps as a derived property, not a writable attribute, so we cannot reseat the segment's `time_vector` in-place reliably across SI versions. Instead, propagate `all_timestamps` (corrected) as the **canonical timestamps array** used by:
  - **B5's `_get_recording_timestamps` helper** — replace the in-helper `recording.get_times()` call with a path that prefers a caller-supplied corrected array when one exists. Concretely, refactor `_get_recording_timestamps` to `(recording, override=None) -> array`; `Recording.make` computes `all_timestamps` once (N50), runs the monotonicity correction, then threads the corrected array through subsequent helper calls as `override=`.
  - **R5's `_consolidate_intervals(valid_times, times)` call** — `times` is now `all_timestamps` (corrected), not a fresh `recording.get_times()` call inside the helper.
  - **`_write_nwb_artifact`** — the corrected array is what gets serialized into the AnalysisNwbfile's `ElectricalSeries.timestamps` via `TimestampsDataChunkIterator(all_timestamps, ...)`.

  Document this propagation in the docstring of `_get_recording_timestamps` (B5) so a future reader understands the `override=` parameter exists specifically for N50's correction path.

  Test `test_recording_repairs_non_monotonic_timestamps` synthesizes a recording with deliberate non-monotonic boundary samples, populates `Recording`, captures `logger.warning`, and asserts the written `ElectricalSeries.timestamps` is strictly increasing AND that downstream operations (artifact detection, `_consolidate_intervals`) use the corrected array — not a stale `recording.get_times()` re-pull. CHANGELOG entry references the in-flight v1 PR; this is a defensive port, not a v1→v2 regression. Cross-reference R5 and B5 explicitly: N50 corrects, B5 propagates, R5 consumes.

- **N51 [MEDIUM] — Clean up the analyzer folder on populate failure (post-tri-part-aware).** `_build_analyzer` writes a folder to disk (1-10 GB each) BEFORE the rollback-able portion of `Sorting.make`. The Phase 1 monolithic `make` swallows the units-NWB cleanup but not the analyzer folder; after the tri-part refactor, the situation splits into TWO failure modes that BOTH need cleanup:

  - **Failure mode A — exception inside `make_compute`** (after `_build_analyzer` runs, before `make_insert` is called). DataJoint's framework will NOT call `make_insert`, so the rollback work must live inside `make_compute`'s own try/except. Wrap the work in `make_compute` that runs after `_build_analyzer` returns:

    ```python
    # Inside Sorting.make_compute, after _build_analyzer call:
    analyzer_folder = self._build_analyzer(sorting=sorting_obj, recording=recording, key=key)
    try:
        analysis_file_name, units_object_id = self._write_units_nwb(...)
    except Exception:
        # _build_analyzer succeeded but _write_units_nwb failed — clean up
        # the analyzer folder before propagating the exception.
        import shutil as _shutil
        if analyzer_folder.exists():
            try:
                _shutil.rmtree(analyzer_folder, ignore_errors=False)
            except Exception as cleanup_exc:
                logger.error(
                    "Sorting.make_compute rollback: failed to remove analyzer "
                    f"folder {analyzer_folder!r}: {cleanup_exc!r}"
                )
        raise
    return analysis_file_name, units_object_id, analyzer_folder, ...
    ```

  - **Failure mode B — exception inside `make_insert`** (after `make_compute` succeeded). The existing rollback path in `make_insert` already unlinks the staged units NWB; extend it to also clean the analyzer folder:

    ```python
    # Inside Sorting.make_insert's existing except block:
    except Exception:
        # Existing units-NWB unlink (preserved):
        try:
            abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
            if _pathlib.Path(abs_path).exists():
                _pathlib.Path(abs_path).unlink()
        except Exception as cleanup_exc:
            logger.error(...)
        # NEW: also remove the analyzer folder
        try:
            import shutil as _shutil
            if analyzer_folder.exists():
                _shutil.rmtree(analyzer_folder, ignore_errors=False)
        except Exception as cleanup_exc:
            logger.error(
                "Sorting.make_insert rollback: failed to remove analyzer "
                f"folder {analyzer_folder!r}: {cleanup_exc!r}"
            )
        raise
    ```

  - Apply the same TWO-site pattern to `_rebuild_analyzer_folder` ([v2/sorting.py:560](../../../../src/spyglass/spikesorting/v2/sorting.py#L560)) — it's monolithic (called from `get_analyzer`, not via populate), so the rollback is a single try/except wrapping the build.

  **Scope distinction**: this is distinct from **R3** (sorter `tempfile.TemporaryDirectory` auto-cleanup on success) and from **R11** (`Sorting.delete()` override removing analyzer folder when the DataJoint row is deleted). The three cleanup paths cover three distinct lifecycle events: R3 = sort completes, R11 = row deleted, N51 = populate fails.

  Tests:
  - `test_analyzer_folder_cleaned_on_make_compute_failure` — monkey-patch `_write_units_nwb` to raise; assert analyzer folder removed.
  - `test_analyzer_folder_cleaned_on_make_insert_failure` — monkey-patch `AnalysisNwbfile.add` to raise after `make_compute` returns; assert both units NWB AND analyzer folder removed.
  - `test_rebuild_analyzer_folder_cleaned_on_failure` — same pattern on `_rebuild_analyzer_folder`.

- **N52 [MEDIUM] — Add `Sorting.fetch_nwb` (or equivalent helper) for pre-curation spike-time inspection.** v1's user-facing pattern at notebook cell `3d41d3ab` does `sorting_nwb = (sgs.SpikeSorting & key).fetch_nwb()` to inspect spike times in seconds before deciding on curation. v2 has `Sorting.get_sorting(key)` which returns an SI `NwbSortingExtractor` (samples, not seconds) and `CurationV2.get_sorting(key, as_dataframe=True)` which only works post-curation. No clean v2 path for "I just sorted, let me peek at the spike times."

  Add a fetch-like helper on `Sorting` that returns spike times in seconds. Two options — pick the cleaner one:

  Option A (mirror v1 API): inherit `fetch_nwb` from `FetchMixin` and verify it works on `Sorting`. v2's `Sorting` already inherits `FetchMixin` via `SpyglassMixin` at [v2/sorting.py:341](../../../../src/spyglass/spikesorting/v2/sorting.py#L341); confirm `fetch_nwb` returns a list-of-dicts where each dict has `object_id` and the NWB's units table is accessible. If yes, this is just a documentation task.

  Option B: extend `Sorting.get_sorting` with an `as_dataframe=True` mode that mirrors v1's behavior — read the units NWB directly, return a pandas DataFrame with `unit_id` + `spike_times` (in seconds) columns. Matches `CurationV2.get_sorting`'s signature for consistency. ~10 LOC.

  Verify which works in current v2; recommendation defaults to Option A if `fetch_nwb` already works, else Option B.

- **N53 [MEDIUM] — Convert `CurationV2.get_sorting` and `get_merged_sorting` to `@classmethod` for v1 surface symmetry.** R1 and R2 already convert `get_recording` (newly added) and `get_sort_group_info` to `@classmethod`. The other two accessor methods are currently instance methods at [v2/curation.py:538](../../../../src/spyglass/spikesorting/v2/curation.py#L538) and [v2/curation.py:579](../../../../src/spyglass/spikesorting/v2/curation.py#L579). They happen to work via the merge dispatcher because DataJoint's `&` returns an instance-like restriction object, but calling `CurationV2.get_sorting(key)` directly from the class (which v1 users expect) fails with missing `self`. Fix: add `@classmethod` decorators and rebind `self` → `cls` throughout the bodies. Update `_upstream_recording_row` to `@classmethod` too since it's invoked from `get_sorting`. Trivial code change; restores v1 API symmetry across all four accessor methods.

- **N54 [LOW] — Test parity additions for v1 behaviors not currently asserted on v2.** Bundle of 9 small test additions found by the v1-test-suite-parity sweep:
  - `test_artifact_interval_list_pipeline_value` — assert the `IntervalList.pipeline` column written by `ArtifactDetection.make` is exactly `"spikesorting_artifact_v2"` (matches v1's `"spikesorting_artifact_v1"` shape; downstream consumers may grep this).
  - `test_artifact_none_preset_produces_full_recording_window` — assert `detect=False` preset writes `valid_times == [[timestamps[0], timestamps[-1]]]` end-to-end (v1 round-trip equivalence).
  - `test_get_sort_group_info_returns_expected_columns` — assert the v2 `get_sort_group_info` relation has the v1 column set (`bad_channel`, `electrode_group_name`, `filtering`, `impedance`, `probe_id`, `probe_shank`, `region_id`, `x`/`y`/`z`, etc.). v1's `test_curation_sort_info` ([v1 tests/test_curation.py:43-75](../../../../tests/spikesorting/v1/test_curation.py#L43-L75)) enumerates 22 expected columns; replicate the assertion against v2.
  - `test_zero_spike_times_column_handled` — synthesize an NWB whose `units` table lacks the `spike_times` column entirely; run through `CurationV2.insert_curation`; assert no `KeyError` and the curated NWB is well-formed. Issue #1532 edge case beyond the labeled-noise-only test already in plan.
  - `test_get_restricted_merge_ids_with_restrict_by_artifact_false` — parametrize the existing v2 merge-ids test to also exercise `restrict_by_artifact=False`. Cross-references N40.
  - `test_merge_dispatch_get_recording_on_v2_merge_id` — explicit dispatch test on `SpikeSortingOutput.get_recording(merge_key)` for v2. Cross-references R1.
  - `test_merge_dispatch_get_sorting_on_v2_merge_id` — same for `SpikeSortingOutput.get_sorting`. Cross-references R2/N53.
  - `test_merge_dispatch_get_sort_group_info_on_v2_merge_id` — same for `get_sort_group_info`. Cross-references R2.
  - `test_sorted_spikes_group_time_slice_param_consistency` — v1 asserts `fetch_spike_data` returns equivalent results across `time_slice` parameter forms (`tuple`, `slice`, `list`); replicate on a v2-sourced merge_id.
  - `test_sort_group_ids_monotonically_increasing` — assert `SortGroupV2.set_group_by_shank` produces sort_group_ids starting at 0 with no gaps (v1 convention; tested in v1 but only cardinality is asserted in v2).
  - `test_merge_dispatch_get_spike_indicator_on_v2_merge_id` — explicit dispatch test on `SpikeSortingOutput.get_spike_indicator(merge_key, time_array)` for a v2 merge_id. `get_spike_indicator` iterates over `get_spike_times` output ([spikesorting_merge.py:359-389](../../../../src/spyglass/spikesorting/spikesorting_merge.py#L359-L389)) and is the primary consumer-facing API for clusterless decoding. Without a v2-specific test, any future regression in v2's `get_spike_times` dispatch silently breaks this downstream path. Assert the returned shape is `(len(time_array), n_units)` and the dtype is float / bool.
  - `test_merge_dispatch_get_firing_rate_on_v2_merge_id` — same for `SpikeSortingOutput.get_firing_rate(merge_key, time_array)` ([spikesorting_merge.py:437-470](../../../../src/spyglass/spikesorting/spikesorting_merge.py#L437-L470)). Assert the returned array has the expected shape, is non-negative everywhere, and the unit_id axis matches `CurationV2.Unit`.

  These are bundled as one task because each test is ≤10 LOC and they all assert v1-shape contracts. Add them to `tests/spikesorting/v2/test_single_session_pipeline.py` or a new `test_v1_parity.py` module — implementer's choice.

- **Do not refactor `CurationV2.insert_curation` into tri-part — but DO add a guard that locks in the transaction discipline it already follows.** `CurationV2` is `dj.Manual`, so DataJoint's framework does NOT wrap user calls in a transaction the way `populate()` does for Computed tables. The transaction-blocking problem that drives tri-part for `Recording` / `Sorting` / `ArtifactDetection` does not apply here: `insert_curation` is user-called and opens its own short `transaction_or_noop` block manually. The Phase 1 implementation ([curation.py:223-235](../../../../src/spyglass/spikesorting/v2/curation.py#L223-L235) then [curation.py:277-295](../../../../src/spyglass/spikesorting/v2/curation.py#L277-L295)) already does the equivalent work: stage the heavy curated-units NWB write FIRST (outside any transaction), then open a short transaction that only wraps the `AnalysisNwbfile.add` + `cls.insert1(master)` + `cls.Unit.insert` + `cls.UnitLabel.insert` + `SpikeSortingOutput._merge_insert` calls. Total time inside the transaction is O(rows), not O(samples × channels).

  The risk to guard against is a future refactor moving the NWB write INSIDE the transaction block in pursuit of "better atomicity" — which would re-introduce the exact long-transaction problem tri-part solves on the Computed side. Phase 1b adds a static `test_curation_v2_nwb_write_outside_transaction` test (described in the Validation slice) that walks the AST of `insert_curation` and asserts the `_stage_curated_units_nwb` call appears before the `transaction_or_noop(...)` `with` block. No code change to `curation.py`.

- **Documentation updates** (shipped with the code change, not deferred):
  - Append a "Phase 1b runtime regressions" section to `docs/src/Features/SpikeSortingV2.md` describing the streaming-write behavior and the parallel-populate flag. One paragraph; this is implementation detail, not user-visible API.
  - Add a CHANGELOG entry: "Spike sorting v2: stream the preprocessed `Recording` ElectricalSeries to NWB via `GenericDataChunkIterator` and enable parallel `populate()` on `Recording`, `ArtifactDetection`, and `Sorting`. Fixes the in-memory materialization that prevented Phase 1 from running on production-scale recordings."
  - Update the `phase-1-modern-single-session.md § Deliberately not in this phase` list (NOT this file) to remove any implication that streaming/parallelism is permanently out of scope — Phase 1 intentionally shipped without them; Phase 1b restores both. **One-line edit, not a rewrite.**

## Deliberately not in this phase

- **No tri-part refactor of `CurationV2.insert_curation`.** Manual-table helpers don't gain anything from the tri-part split; Spyglass's parallel-populate scaffolding only applies to `dj.Computed.populate()`. The existing transaction wrap is sufficient.
- **No tri-part refactor of `ConcatenatedRecording.make()`.** Phase 1 declared it with `NotImplementedError`; Phase 3 fills in the body and can choose tri-part vs monolithic at that time.
- **No chunked-iterator extension to the units / curated-units NWB writes** ([sorting.py:778-825](../../../../src/spyglass/spikesorting/v2/sorting.py#L778-L825) and [curation.py:427-534](../../../../src/spyglass/spikesorting/v2/curation.py#L427-L534)). Spike-time arrays scale with `n_units × n_spikes_per_unit`, not `n_samples × n_channels`; streaming would add complexity without addressing a real memory pressure point.
- **No `recording.save(format="binary", chunk_duration="2s")` binary sidecar materialization.** Research notes mention this as the SI-recommended pre-sort workflow, but adopting it would change the [Recording Cache Format](shared-contracts.md#recording-cache-format) contract (add a `binary_cache_path` column, change the lifecycle). That is a schema change and is out of scope here; the streaming HDF5 write fixes the immediate OOM without touching the contract.
- **No tuning of `buffer_gb` / `chunk_mb` from v1's defaults.** v1 has shipped `buffer_gb=5` in production for years; keep the same defaults to preserve behavior. Future tuning is follow-up.
- **No revisit of the `_assert_v2_db_safe()` module-import-time guard.** That is a separate v2-graduation question and does not affect runtime regressions.
- **No backport of these changes to v1.** v1 already uses chunked iterators and tri-part make; no work required.
- **No restoration of B4** (the `metrics=...` kwarg on `insert_curation` that v1 used to write per-metric NWB columns). Phase 2 introduces `AnalyzerCuration` for metrics; the v1→v2 metric-handling gap is a deliberate Phase 2 deferral per the existing plan, not a regression to fix here. A v1 user migrating to v2 between Phase 1b and Phase 2 will lose metric columns on `insert_curation` — document in CHANGELOG but do not patch in Phase 1b.
- **No restoration of v1's `update_ids` migration helper** (B6). v2's schema is final-shape from Phase 1; there is nothing to migrate. Confirmed not-a-regression.
- **No new tables or part tables beyond `CurationV2.MergeGroup`** (B3). The MergeGroup part is the SINGLE deliberate schema addition in Phase 1b, authorized by the user as the queryability + FK-validation choice over v1's NWB-column pattern. No other new tables or part tables in this phase.
- **No changes to source-part contracts.** `SortingSelection.RecordingSource` / `ConcatenatedRecordingSource` and `ArtifactSelection.RecordingSource` / `SharedArtifactGroupSource` keep their Phase 1 shape. Layer-1 transaction-wrapped inserts and Layer-2 `resolve_source` re-checks are unchanged.
- **No v2 companion to `common_file_tracking._get_v1_deleted_files` (R16 deferral).** [common_file_tracking.py:71-94](../../../../src/spyglass/common/common_file_tracking.py#L71-L94) queries v1's `RecordingRecompute` table to identify analysis files that were intentionally deleted via the v1 recompute workflow — these are excluded from the "orphan files" report. v2 has no equivalent recompute table in Phase 1 or Phase 1b; the equivalent (`RecordingArtifactRecompute*`) lands in Phase 2. Today, no v2 sort has been intentionally deleted (no v2 recompute machinery exists), so the practical impact is zero. The forward concern is documented in [phase-2-analyzer-curation.md](phase-2-analyzer-curation.md): when Phase 2 ships the recompute workflow, it must also add a `_get_v2_deleted_files()` companion to `common_file_tracking` so file-tracking infrastructure sees v2 deletes too. Phase 1b adds NO code or schema here — flagged here so a Phase 1b reviewer doesn't mistake the gap for a regression.

## Test tiers

The validation slice below mixes tests that run in milliseconds with tests that run in minutes. Mark each test with one of the five tiers below (registered as pytest markers in `pyproject.toml`'s `[tool.pytest.ini_options]` `markers` list) so CI can run the right subset at the right cadence. Replace the blanket `@pytest.mark.slow` usage with the specific tier marker; `@pytest.mark.integration` becomes redundant for T3-T5 since the tier implies it.

| Tier | Marker | Setup cost | Definition | Per-test runtime |
| --- | --- | --- | --- | --- |
| **T1** | `@pytest.mark.unit` | none | Pure-Python: Pydantic validation, AST static checks, helper-function unit tests | <1 s |
| **T2** | `@pytest.mark.db_unit` | Docker MySQL only | DB-tier but no populate: idempotency, error paths, schema validation through `dj_conn`. v2 schema activation requires Docker at import time even when the test body doesn't query | <5 s after Docker is up |
| **T3** | `@pytest.mark.stage` | Docker + populate one Computed table | One-stage integration: `Recording`, `ArtifactDetection`, or `Sorting` populate plus its accessor round-trip | 10-60 s |
| **T4** | `@pytest.mark.pipeline` | Docker + full chain | Recording → Artifact → Sort → Curation populate; uses smoke fixture | 60-120 s |
| **T5** | `@pytest.mark.regression_gate` | Docker + heavy / 60s fixture / multiproc | Correctness gates (60 s MEArec ground truth), memory budgets, parallel populate, baseline comparison | >2 min |

Default CI runs T1 + T2 + T3. PR-target branch and nightly runs add T4 + T5. The `slow` marker is retired (replaced by `T3`-`T5`). The `integration` marker is retired (implied by `T3`-`T5`). The `memory` and `parallel` marks become `T5` sub-categorizations within the `regression_gate` tier.

Tier assignment for each validation-slice test appears as the **first column** of the table below. When a test naturally fits multiple tiers (e.g., an AST check that ALSO exercises a runtime path), pick the lowest tier whose setup is sufficient — a T1 test that incidentally needs Docker for an import side effect is a T2.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_recording_artifact_bit_equivalent_to_phase1_baseline` (slow, integration) | After the refactor, `run_v2_pipeline(..., preset="franklab_tetrode_clusterless_thresholder")` produces a `Recording` row whose **streamed traces and timestamps**, when re-read from the AnalysisNwbfile, match the Phase 1 baseline byte-for-byte. Trace array equality is exact within `0.0` (same input, same preprocessing, same dtype). Timestamps array equality is exact within `1e-12` seconds (float64 round-trip tolerance). **Do NOT compare `cache_hash` to the baseline**: R17 changes the hash strategy from `sha256(data.tobytes())` to `NwbfileHasher(analysis_file_name).hash` (per the documented contract), so Phase 1's stored hashes are not directly comparable. The trace + timestamps comparison is the authoritative byte-equivalence check; `cache_hash` is now a downstream property of those bytes plus the NWB structure. **This test exercises the Recording stage only; the MS4 sorter path is NOT bit-equivalent post-fix because R4 changes whitening from MS4-internal float32 to external float64.** R5 (disjoint-interval concat) is exercised by `test_disjoint_intervals_concatenated` below, not by this baseline — the deterministic baseline uses a single-interval sort window. |
| `test_sorting_outputs_match_phase1_baseline_for_clusterless_thresholder_only` (slow, integration) | `clusterless_thresholder` produces deterministic peak-sample indices AND does not use whitening, so it is the only sorter for which Phase 1b output should match Phase 1 byte-for-byte. Assert per-unit spike-sample equality against the Phase 1 baseline. **Stochastic sorters (MS5/MS4/KS4) are NOT in this comparison — they are not exact-rerun gates per the `overview.md` Risks table, AND R4 deliberately changes MS4 output by restoring v1's float64 external whitening.** A separate `test_ms4_outputs_match_v1_real_data_when_baseline_present` (env-var gated by `SPIKESORTING_V2_REAL_NWB_PATH`) verifies the MS4 path matches v1's behavior on real data within the documented `±50% unit count + ±30% median FR` tolerance from `overview.md`. |
| `test_curation_object_id_matches_baseline_units` (slow, integration) | The CurationV2 row's curated units re-read from the AnalysisNwbfile have spike times matching the Phase 1 baseline within `1.5 / sampling_frequency` seconds. Only exercised under the `clusterless_thresholder` preset (deterministic). MS4-baselined curation comparisons are gated by the real-data env var. |
| `test_recording_peak_rss_under_threshold` (slow, integration, marked `@pytest.mark.memory`, **Linux-only via `@pytest.mark.skipif(sys.platform != "linux", ...)`** — `ru_maxrss` units differ across OS: KB on Linux, bytes on macOS, so cross-platform tests would need a conditional unit-conversion that adds bug surface for a smoke test) | Populates `Recording` on a synthetic 5-minute 32-channel 30 kHz fixture (≈18 GB at float64; preprocessing layer at [recording.py:1025,1038,1051](../../../../src/spyglass/spikesorting/v2/recording.py#L1025) explicitly casts to `float64`). Peak RSS measured via `psutil.Process().memory_info().rss` sampled inside a `make_compute`-finished callback OR via `resource.getrusage(RUSAGE_SELF).ru_maxrss * 1024` on Linux must stay under `2 × buffer_gb_in_bytes × 1.5` headroom ≈ 15 GB. Without this fix, the same test would need ~18 GB and OOM on most workstations. The threshold is a smoke test, not a precise budget — the goal is to detect a future regression that drops the streaming path. Prefer `psutil` over `resource` if `psutil` is already a project dep; check `pyproject.toml` before adding it as a new dep just for this test. |
| `test_recording_rebuild_hash_matches` (slow, integration) | Populate `Recording`, delete the AnalysisNwbfile on disk, call `get_recording(key)` to trigger `_rebuild_nwb_artifact`, assert the recomputed `cache_hash` matches the row's stored `cache_hash`. Exercises the `_hash_nwb_recording` recompute path end-to-end (per R17 — `NwbfileHasher` over the rebuilt file). |
| `test_recording_parallel_populate_runs` (slow, integration, marked `@pytest.mark.parallel`) | Insert N=4 `RecordingSelection` rows pointing at distinct sort groups of the same MEArec fixture; call `Recording.populate(reserve_jobs=True)` with `processes=4`. Assert all four rows populate successfully. **Verify parallelism via worker-PID inspection, NOT wall-clock**: monkey-patch a small probe into `Recording.make_compute` (or use `multiprocessing.current_process()` inside the test fixture) that appends `os.getpid()` to a multiprocessing-safe `Manager.list()`. After populate completes, assert `len(set(pids)) >= 3` (give one worker slack for scheduling). Wall-clock timing is informational only — measure and log it, but do NOT gate the test on it; thermal throttling, hot caches, and CI runner variance make ratios unreliable. |
| `test_sorting_parallel_populate_does_not_deadlock` (slow, integration) | Same setup as above but for `Sorting.populate(processes=2)` with `clusterless_thresholder`. Assert no DataJoint job-table deadlock, no transaction-aborted errors, and that each sort produces a non-empty `Sorting.Unit` part-table row set. |
| `test_artifact_detection_parallel_populate_runs` (slow, integration) | `ArtifactDetection.populate(processes=2)` over two distinct recording IDs completes; both `IntervalList` rows committed; no orphan artifact rows. |
| `test_tripart_atomicity_unchanged` | Forcibly raise inside `make_insert` (monkey-patch `AnalysisNwbfile.add` to raise after the staged file exists): assert the DataJoint row is NOT inserted, the AnalysisNwbfile DB row is NOT registered, and the staged NWB on disk IS unlinked. Matches the Phase 1 atomicity test (which exercised the monolithic `make` path); the refactor must preserve identical behavior. Run for `Recording`, `ArtifactDetection`, and `Sorting`. |
| `test_tripart_dispatch_active` (fast smoke; runs at the top of the suite) | For each refactored Computed table (`Recording`, `ArtifactDetection`, `Sorting`): `assert inspect.isgeneratorfunction(Cls.make) is True` and `Cls.make_fetch is not None` and `Cls.make_compute is not None` and `Cls.make_insert is not None`. Catches in seconds the failure mode where the executor adds the tri-part methods but forgets to delete the explicit `def make(self, key): ...` — without this check, the tri-part methods would be dead code and the regression would silently persist. |
| `test_make_compute_is_pure` | Static check: `make_compute` in `Recording`, `ArtifactDetection`, and `Sorting` does not call `self.insert1`, `AnalysisNwbfile().add`, `IntervalList.insert1`, or any method on `self.connection`. Uses `ast` to walk the module source — guards against future drift back into monolithic patterns. |
| `test_curation_v2_nwb_write_outside_transaction` | Static AST check on `CurationV2.insert_curation` ([curation.py](../../../../src/spyglass/spikesorting/v2/curation.py)): the call to `_stage_curated_units_nwb` appears in the source before the `with transaction_or_noop(...)` block, and no `pynwb.NWBHDF5IO`, `AnalysisNwbfile().create`, or `_stage_curated_units_nwb` calls appear *inside* the `with` block. Rationale: `CurationV2` is `dj.Manual`, so the framework does not wrap it in a transaction; the implementation must keep doing this work itself. A future maintainer "consolidating" the NWB write into the transaction would re-introduce exactly the long-transaction problem ([Spyglass #1030](https://github.com/LorenFrankLab/spyglass/issues/1030), [DataJoint #1170](https://github.com/datajoint/datajoint-python/issues/1170)) that tri-part fixes for Computed tables. |
| `test_merge_dispatch_get_recording_works` (slow, integration) | Populate v2 through `run_v2_pipeline`, fetch the resulting `merge_id`, call `SpikeSortingOutput().get_recording({"merge_id": merge_id})`. Must return a SI `BaseRecording` (not raise `AttributeError`). Covers R1. Bonus assertion: returned recording carries `recording.get_annotation("is_filtered") is True` (covers R6). |
| `test_merge_dispatch_get_sort_group_info_works` (slow, integration) | Same setup; call `SpikeSortingOutput().get_sort_group_info({"merge_id": merge_id})`. Must return a DataJoint relation with rows for EVERY electrode in the sort group (not just one — the v1 multi-region bug remains fixed). Must not raise `TypeError` on the bound classmethod call from `source_table.get_sort_group_info(...)`. Covers R2. |
| `test_sorting_tempdir_cleaned_up` (slow, integration) | Run `Sorting.populate` for one key; capture the value of `spyglass.settings.temp_dir` before and after; assert no `sort_*` subdirectory remains in `temp_dir` post-populate. Negative test: monkey-patch `tempfile.TemporaryDirectory` to fail `.cleanup()` and assert the test framework catches the leftover for triage. Covers R3. |
| `test_external_whitening_at_float64` (slow, integration) | Run `Sorting.populate` for an MS4 sort. Capture the recording handed into `sis.run_sorter` (monkey-patch the call to capture the kwargs). Assert `recording.get_dtype() == np.float64` (post-external-whitening dtype), AND assert the captured `sorter_params["whiten"] is False` (so MS4 does not re-whiten internally). Covers R4. |
| `test_disjoint_intervals_concatenated` (slow, integration) | Create an `IntervalList` row whose `valid_times` is `[[t0, t1], [t2, t3]]` with a deliberate gap `(t1, t2)`. Populate `Recording`, then read the written `ElectricalSeries` and assert (a) `timestamps` array has NO samples in `(t1, t2)`, (b) `len(timestamps) ≈ (t1-t0+t3-t2) × fs`, (c) the gap is excluded, not silently sorted. Covers R5. Failure mode without this fix: timestamps array would be monotonic across `[t0, t3]` including the gap. |
| `test_min_segment_length_drops_slivers` | Synthetic recording where the requested sort interval intersected with `raw data valid times` produces one 5-second chunk and one 0.5-second sliver. Default `min_segment_length=1.0` must drop the sliver; explicit `min_segment_length=0.1` must keep it. Covers R7. |
| `test_boundary_spike_round_trip_does_not_raise` (slow, integration) | Construct a `Sorting` whose units include a spike at exactly `recording.get_num_samples() - 1`. Populate v2 through `run_v2_pipeline`. Call `Sorting().get_sorting(key)` AND `CurationV2().get_sorting(curation_key)`. **If either raises `ValueError("...spikes exceeding the recording duration...")` on SI 0.104:** the executor must fold in v1's `spike_times_to_valid_samples` clip from [v1/sorting.py:29-79](../../../../src/spyglass/spikesorting/v1/sorting.py#L29-L79) on the read path in both `Sorting.get_sorting` and `CurationV2.get_sorting`, then re-test until both pass. **If neither raises:** mark the test as a regression guard with a comment explaining what it would catch. Covers R8. |
| `test_singularity_carve_out_for_matlab_sorters` | Static check (no actual KS2.5 run — no MATLAB image in CI): monkey-patch `sis.run_sorter` to capture kwargs; insert a `SorterParameters` row for `sorter="kilosort2_5"` via `GenericSorterParamsSchema`; call `Sorting._run_sorter(sorter="kilosort2_5", ...)`. Assert the captured kwargs include `singularity_image=True` and do NOT include `tempdir`, `mp_context`, or `max_threads_per_process`. Covers R9. |
| `test_artifact_defaults_match_b1_revised` | Assert `ArtifactDetectionParamsSchema().model_dump()` returns `amplitude_thresh_uV == 500.0` (NOT 3000.0 — see B1-revised explaining the v1 unit bug fix) and `proportion_above_thresh == 1.0` (v1-parity revert; no v1 bug here). Assert the `"default"` row in `ArtifactDetectionParameters._DEFAULT_CONTENTS` carries those values. Covers B1-revised. Failure surfaces a future regression that re-tightens proportion or accidentally reverts amplitude to 3000. |
| `test_insert_curation_accepts_apply_merge_kwarg` | Call `CurationV2.insert_curation(sorting_key, labels={}, apply_merge=True, merge_groups=[[1,2]])`. Assert no `TypeError` and `merges_applied=True` on the resulting row. Covers B2 (v1 kwarg parity). |
| `test_merge_groups_recoverable_via_part_table` (slow, integration) | Populate a `CurationV2` with `merge_groups=[[1, 2, 3], [4, 5]]` and `apply_merge=True`. Verify `CurationV2.MergeGroup & {"sorting_id": sid, "curation_id": cid}` returns 5 rows (3 contributors for the 1-2-3 group + 2 for the 4-5 group). Verify `CurationV2.get_merge_groups(key) == {1: [1,2,3], 4: [4,5]}`. Verify `CurationV2.get_merged_sorting(key)` returns an SI `MergeUnitsSorting` whose `unit_ids` collapse the contributors into kept-units. Covers B3. |
| `test_multi_segment_nwb_handled` (slow, integration) | Synthesize a 2-segment NWB recording via `pynwb` (two separate `ElectricalSeries` calls or one with discontinuous timestamps), ingest, populate `Recording`. Assert the written `ElectricalSeries.timestamps` array covers BOTH segments contiguously (not just segment 0). Assert `Recording` row's `duration_s` equals the sum of per-segment durations, not just segment 0's duration. Covers B5. |
| `test_heterogeneous_gain_raises_with_documented_reason` | Synthesize a sort group where two electrodes have different `channel_gain` values in the NWB. Populate `Recording`. Must raise `ValueError`. Inspect the source of `_write_nwb_artifact` via `inspect.getsource` and assert the v1 latent-bug rationale comment is present (the test will keep the comment from being deleted by a future "cleanup"). Covers B7. |
| `test_sorting_delete_removes_analyzer_folder` (slow, integration) | Populate `Sorting` for one key; resolve the analyzer folder path via `_analyzer_path({"sorting_id": ...})`; assert the folder exists. Call `(Sorting & key).delete()` (with `safemode=False` for test deterministically). Assert the folder no longer exists on disk after the delete returns. Negative test: monkey-patch `shutil.rmtree` to raise `PermissionError`; assert the override propagates the exception (does NOT silently swallow disk-cleanup errors). Covers R11. |
| `test_artifact_subtraction_filters_subsecond_slivers` (slow, integration) | Synthesize a recording with closely-spaced artifact intervals that produce ~50 ms valid-interval slivers between them. With default `min_length_s=1.0`, assert the resulting IntervalList row's `valid_times` array contains no intervals shorter than 1.0 s. With explicit `min_length_s=0.01`, assert the slivers are kept. Covers R13. Failure mode without this filter: every artifact-detection run on noisy chewing-period recordings produces hundreds of millisecond-scale slivers that downstream `_apply_artifact_mask` iterates one by one. |
| `test_insert_curation_accepts_labels_none` | Call `CurationV2.insert_curation(sorting_key, labels=None, apply_merge=False)`. Must not raise. Resulting `CurationV2.UnitLabel` part has zero rows (semantically equivalent to `labels={}`). Covers R15. v1 parity test: same call signature as `CurationV1.insert_curation(sorting_id, labels=None)` at [v1/curation.py:49](../../../../src/spyglass/spikesorting/v1/curation.py#L49). |
| `test_cache_hash_uses_nwbfile_hasher` (slow, integration) | Populate `Recording` for one key. Read the stored `cache_hash` from the DataJoint row. Compute the expected hash independently: `expected = _hash_nwb_recording(row["analysis_file_name"])` (the same helper `Recording.make` and `_rebuild_nwb_artifact` use). Assert `row["cache_hash"] == expected` — this is the authoritative check that the runtime path matches the documented contract. Do NOT assert a specific digest length (32 vs 64): the length depends on `NwbfileHasher`'s implementation, which is not pinned by Phase 1b. Length headroom is provided by `cache_hash: char(64)` per [shared-contracts.md § Recording Cache Format](shared-contracts.md#recording-cache-format). Covers R17 and locks in the contract by behavior, not by hash format. |
| `test_recording_rebuild_hash_uses_nwbfile_hasher` (slow, integration) | Extension of `test_recording_rebuild_hash_matches` above: after deleting the AnalysisNwbfile on disk and calling `get_recording(key)` to trigger `_rebuild_nwb_artifact`, assert the rebuild's mismatch-warning branch uses `_hash_nwb_recording(analysis_file_name)` to recompute (not the data-only sha256). Inspect the rebuilt hash and the stored hash; both must be `NwbfileHasher` digests. Covers R17 on the rebuild path. |
| `test_common_reference_field_removed_from_schema` | `from spyglass.spikesorting.v2._params.preprocessing import CommonReferenceParams`; assert `"reference" not in CommonReferenceParams.model_fields`; assert `"operator" in CommonReferenceParams.model_fields`. Negative test: `CommonReferenceParams(reference="local")` raises `ValidationError` because `extra="forbid"` rejects the now-unknown field. Covers R18. |
| `test_clusterless_thresholder_noise_levels_preserved` (slow, integration) | Monkey-patch `spikeinterface.sortingcomponents.peak_detection.detect_peaks` to capture its kwargs. Call `_run_sorter(sorter="clusterless_thresholder", ...)`. Assert the captured kwargs include `noise_levels=[1.0]` (or equivalent — `_run_sorter` either keeps the user-supplied noise_levels OR explicitly passes `[1.0]`). Asserts `detect_threshold` is treated in microvolts, not MAD multiples. Covers N19. |
| `test_artifact_zscore_across_channels` | Synthetic 8-channel recording where one frame has a 50× common-mode spike on all 8 channels but moderate baseline noise. Run `_detect_artifacts` with `zscore_thresh=5.0, amplitude_thresh_uV=None`. Assert the spike frame is flagged (across-channels z-score sees it). Compare to a per-channel-baseline z-score implementation which would NOT flag it. Covers N20. |
| `test_artifact_combine_or_not_and` | Synthetic recording where frame A is above amplitude threshold on 1 channel but below z-score; frame B is below amplitude but above z-score on 1 channel. With both thresholds set and v2's OR semantics, both frames should be flagged. Asserts the OR semantics restored from v1 (would fail with v2's AND). Covers N23. |
| `test_artifact_detection_logs_thresholds_and_empty_result` | Capture `logger.info` and `logger.warning` output across a populate where (a) detection runs with default thresholds, (b) detection runs but finds zero artifacts, (c) detection is skipped (`detect=False`). Each case emits the expected message string. Covers N37, N46. |
| `test_artifact_delete_logs_resolve_source_failure` | Monkey-patch `ArtifactSelection.resolve_source` to raise `RuntimeError("simulated failure")`. Call `(ArtifactDetection & key).delete(safemode=False)`. Capture `logger.error` output; assert the row's PK appears in the error message. The delete still proceeds (does not propagate); the IntervalList row is left in place. Covers N47. |
| `test_matlab_sorter_carve_out_strips_kwargs_and_adds_singularity` | Monkey-patch `sis.run_sorter` to capture kwargs. Insert a `SorterParameters` row for `sorter="kilosort2_5"` with `{"tempdir": "/tmp/x", "mp_context": "fork", "max_threads_per_process": 4, "Th": 9.0}`. Call `Sorting._run_sorter(...)`. Assert the captured kwargs include `singularity_image=True`, include `Th=9.0`, and do NOT include `tempdir`, `mp_context`, or `max_threads_per_process`. Covers N38 + R9 combined. |
| `test_tempdir_world_writable` | After `Sorting._run_sorter` creates a tempdir via `tempfile.TemporaryDirectory(dir=temp_dir)`, the directory's `stat().st_mode & 0o777` includes the world-write bit (0o002). Covers N39. |
| `test_sorting_nwb_writes_obs_intervals` (slow, integration) | After `Sorting.populate(key)`, open the units NWB and verify every unit's `obs_intervals` field is non-empty and matches the requested-interval window from the upstream IntervalList. Covers N25. |
| `test_job_kwargs_resolved_in_every_compute_stage` (slow, integration) | Set `dj.config['custom']['spikesorting_v2_job_kwargs'] = {"n_jobs": 7}`. Populate `Recording`, `ArtifactDetection`, `Sorting` for one key each, monkey-patching `_resolved_job_kwargs` to capture invocations. Assert it was called at least once during each of the three populates (currently only Sorting calls it). For `Sorting`, additionally monkey-patch `sis.run_sorter` to capture its kwargs and assert `n_jobs=7` was passed to BOTH the sorter and `analyzer.compute`. Covers N22, N30, L2.contract-N8 combined. |
| `test_get_merged_sorting_applies_merges_at_fetch` (slow, integration) | Populate a `CurationV2` with `apply_merge=False, merge_groups=[[1,2,3],[4,5]]` (so the source NWB still has all individual units). Call `CurationV2().get_merged_sorting(curation_key)`. Assert the returned `BaseSorting.unit_ids` contains only the kept heads `[1, 4]` (plus any unmerged unit_ids). Assert the merged unit's spike train is the union of contributors' spike trains. Covers N21 — closely tied to B3's MergeGroup part-table availability. |
| `test_root_curation_idempotent` (slow, integration) | Call `CurationV2.insert_curation(sorting_key, labels={}, parent_curation_id=-1)` twice with the same args. Assert the second call returns the same `{sorting_id, curation_id}` PK without creating a new row, new analysis NWB, or new merge-table entry. Capture `logger.warning` and assert a "root curation already exists" message was emitted. Covers N24. |
| `test_curation_label_nwb_column_is_indexed_list` (slow, integration) | After `CurationV2.insert_curation(..., labels={1: ['mua', 'artifact']})`, open the curated-units NWB and inspect the `curation_label` column. Assert it is an `index=True` column AND the value for `unit_id=1` is a list `["mua", "artifact"]`, NOT the string `"mua,artifact"`. Covers N26. |
| `test_merges_applied_records_user_intent` | Call `insert_curation(..., apply_merge=True, merge_groups=None)`. Assert the resulting row has `merges_applied=True` (v1 parity), NOT `False`. Covers N41. |
| `test_get_sorting_dataframe_includes_curation_label` (slow, integration) | After `insert_curation` with labels, call `CurationV2().get_sorting(curation_key, as_dataframe=True)`. Assert the returned DataFrame has columns `["unit_id", "spike_times", "curation_label"]`. Covers N42. |
| `test_recording_get_recording_honors_electrical_series_path` (slow, integration) | Populate `Recording`. Monkey-patch `se.read_nwb_recording` to capture kwargs. Call `Recording().get_recording(key)`. Assert the captured kwargs include `electrical_series_path=row["electrical_series_path"]`. Covers N28. |
| `test_channel_name_lookup_branch` | Synthetic NWB whose `/general/extracellular_ephys/electrodes` table includes a `channel_name` string column. Populate `Recording`. Assert `_spikeinterface_channel_ids` mapped Spyglass int IDs to the corresponding string channel_names from the NWB. Covers N31. (Without the fix, this test is the canary for production NWBs with `channel_name`.) |
| `test_tetrode_12_5_probe_geometry_patch` (slow, integration) | Synthetic NWB with `Probe.probe_id="tetrode_12.5"`, 4 channels per sort group, single ElectrodeGroup, AND missing contact-position metadata in the NWB electrode table. Populate `Recording`. Inspect the returned recording's probe via SI's probe API; assert contact positions match `[[0,0],[0,12.5],[12.5,0],[12.5,12.5]]` µm. Covers N32. |
| `test_merge_dispatch_raises_on_unknown_restriction_keys` | Call `SpikeSortingOutput()._get_restricted_merge_ids_v2({"nwb_file_name": "x", "bogus_field": "y"})`. Assert it raises `ValueError` mentioning `bogus_field`. Covers N29. |
| `test_merge_dispatch_restrict_by_artifact_honored_in_v2` (slow, integration) | Populate `ArtifactDetection` and `Sorting`. Call `SpikeSortingOutput.get_restricted_merge_ids({"interval_list_name": f"artifact_{artifact_id}"}, restrict_by_artifact=True, sources=["v2"])`. Assert it returns the correct merge_id (i.e., the artifact-named IntervalList lookup actually fires for v2). Covers N40. |
| `test_kilosort4_schema_accepts_extra_kwargs` | `Kilosort4Schema.model_validate({"Th_universal": 9.0, "batch_size": 60000, "nearest_chans": 10})` should succeed (`extra="allow"`). Asserts the model now accepts non-typed knobs. Covers N34. |
| `test_default_franklab_whiten_none` | `PreprocessingParameters().fetch1(preproc_params_name="default_franklab")["params"]["whiten"]` is `None`. (Match the other two presets; reflect Phase 1's deferred-to-sorter whitening reality.) Covers N35. |
| `test_clusterless_schema_documents_dead_fields_or_drops_them` | Either: (a) `ClusterlessThresholderSchema.model_fields` does NOT contain `noise_levels`, `random_chunk_kwargs`, `outputs` (if dropped); OR (b) those field definitions have an `description=` containing "stripped at runtime" annotation. Either path acceptable; the test ensures the dead-field problem is addressed one way or another. Covers N48. |
| `test_no_phase_label_leakage_in_runtime_code` | `grep -rn "Phase 1\b\|Phase 1b\b\|Phase 1c\b" src/spyglass/spikesorting/v2/ tests/spikesorting/v2/` returns zero hits in non-comment lines (excluding plan-cross-references in shared-contracts.md). Covers N27. The test must allow the exception message at [artifact.py:175](../../../../src/spyglass/spikesorting/v2/artifact.py#L175) to be rewritten to behavior-named language. |
| `test_v2_merge_ids_helper_exists` | `from spyglass.spikesorting.v2.utils import get_spiking_sorting_v2_merge_ids` succeeds. The function accepts a restriction dict + `as_dict` kwarg matching v1's signature. Covers N45. |
| `test_decoding_waveform_features_handles_sparse_unit_ids` (slow, integration) | Populate `CurationV2` with `merge_groups=[[0,1,2]]` so the resulting curated units have a sparse-id NWB. Trigger `UnitWaveformFeatures` populate from `decoding/v1/waveform_features.py`. Assert no `IndexError` and the resulting features match the expected sparse-id set. Covers part of N36. |
| `test_sorted_spikes_group_handles_sparse_unit_ids` (slow, integration) | Same setup; build a `SortedSpikesGroup` for the v2 merge_id. Call `fetch_spike_data`; assert returned arrays are keyed by the actual sparse unit_ids, not positional indices. Covers part of N36. |
| `test_unit_annotation_handles_sparse_unit_ids` (slow, integration) | Same setup; insert `UnitAnnotation` rows for sparse unit_ids and `fetch1` them back. Assert the round-trip is correct. Covers part of N36. |
| `test_sorting_nwb_writes_curation_label_uncurated_placeholder` (slow, integration) | After `Sorting.populate`, open the units NWB. Assert the `curation_label` column exists and every unit has value `"uncurated"`. Covers N43. |
| `test_artifact_amplitude_thresh_in_uV_with_v1_bug_documented` | Assert the schema field is documented as truly-uV (not raw counts) AND the default is `500.0`. Assert v1-bug context is present in the schema docstring or a sibling comment. Covers B1-revised. The CHANGELOG entry must also be present in `CHANGELOG.md`; the test verifies a substring match against the file. |

The first three tests are the **correctness gates**: if any fails, the refactor changed behavior and must not merge. The memory and parallel-populate tests are **regression gates**: they make sure Phase 1b's stated benefits are actually delivered.

Mark each test with the tier marker from the **Test tiers** section above (`@pytest.mark.unit`, `@pytest.mark.db_unit`, `@pytest.mark.stage`, `@pytest.mark.pipeline`, or `@pytest.mark.regression_gate`). Register all five in `pyproject.toml`'s `[tool.pytest.ini_options]` `markers` list, including descriptions, so `pytest --markers` self-documents the convention. Retire `@pytest.mark.slow` and `@pytest.mark.integration` as redundant once the tier markers are in place. The validation slice does NOT enumerate the tier per row above — the executor assigns tiers as part of writing the tests, following the rubric: AST/Pydantic-only → T1; needs `dj_conn` but no populate → T2; populates one Computed → T3; full pipeline → T4; >2 min or multiprocessing → T5.

## Fixtures

- **Phase 1 baseline pickle/npz bundle** (`tests/spikesorting/v2/_fixtures/phase1_baseline/`) — generated by the `phase1_baseline_artifacts` fixture on the unmodified Phase 1 code. Re-generation workflow, stated explicitly so the executor can follow it without guessing:
  1. `git stash` any in-progress refactor edits.
  2. `git rev-parse HEAD` and record the SHA — this is the "Phase 1 tip" baseline commit. The expected SHA at plan time is the tip of `spikesorting-v2` before the Phase 1b refactor lands; the actual SHA is whatever the executor's branch points to before they start editing.
  3. `rm -rf tests/spikesorting/v2/_fixtures/phase1_baseline/` to force regeneration.
  4. Run `pytest tests/spikesorting/v2/test_phase1_baseline_regen.py -q` (a single-test module added by this phase whose only job is to run the fixture once and exit).
  5. `git diff --stat tests/spikesorting/v2/_fixtures/phase1_baseline/` to confirm the bundle was written.
  6. `git stash pop` to restore in-progress edits.
  Record the baseline-source SHA in `tests/spikesorting/v2/_fixtures/phase1_baseline/MANIFEST.json` along with `spikeinterface.__version__`, `numpy.__version__`, and `pynwb.__version__` from the baseline environment. The validation tests verify the MANIFEST matches the current environment before comparing baselines, so an SI version drift fails loudly instead of producing a misleading false negative.
- **`mearec_polymer_128ch_60s.nwb`** — existing Phase 0b fixture; reused.
- **`synthetic_5min_32ch_30khz.nwb`** (new) — generated in `tests/spikesorting/v2/conftest.py` via `neuroconv[mearec]` or direct `pynwb`. Used only by `test_recording_peak_rss_under_threshold`. Pre-fill with float32 random noise; no real spikes needed because the test only measures the writer's memory footprint.
- **`synthetic_multi_segment.nwb`** (new) — 2-segment synthetic NWB generated in `conftest.py`. Each segment ~30s of float64 random noise at 30 kHz on 8 channels with explicit gap in timestamps between segments. Used by `test_multi_segment_nwb_handled` (B5).
- **`synthetic_disjoint_interval.nwb`** (new) — single-segment NWB but with an `IntervalList` row whose `valid_times` has two disjoint intervals with a deliberate gap. Used by `test_disjoint_intervals_concatenated` (R5).
- **`synthetic_boundary_spike.nwb`** (new) — minimal NWB + pre-constructed `Sorting.Unit` rows where one unit has a spike at exactly `n_samples - 1`. Used by `test_boundary_spike_round_trip_does_not_raise` (R8).
- **`synthetic_heterogeneous_gain.nwb`** (new) — NWB with two electrodes in the same sort group having different `channel_gain` values. Used by `test_heterogeneous_gain_raises_with_documented_reason` (B7).

## Review

Before opening or reviewing the implementation PR that contains this checkpoint, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases (especially: no schema changes, no Manual-table tri-part refactor, no binary sidecar).
- Validation slice tests pass; slow / integration / memory / parallel tests are marked.
- Baseline byte-equivalence tests genuinely re-load and compare the AnalysisNwbfile contents — not just compare `cache_hash` (a single SHA-256 collision-tolerant check is necessary but not sufficient evidence of byte equivalence on the underlying scientific data). Compare the re-read trace + timestamps arrays directly.
- Tests aren't trivial — `test_make_compute_is_pure`, `test_curation_v2_nwb_write_outside_transaction`, and `test_tripart_dispatch_active` actually walk the AST or use `inspect.isgeneratorfunction`; the parallel-populate tests actually inspect worker counts. None of them `assert True` or simply re-state what the mock returned.
- `CurationV2.insert_curation` IS modified in this PR (B2 reverts the kwarg name to `apply_merge`; B3 inserts `MergeGroup` part rows). The diff for `curation.py` MUST NOT touch the staging vs. transaction structure that the `test_curation_v2_nwb_write_outside_transaction` AST guard locks in.
- **R1, R2 verification**: `SpikeSortingOutput.get_recording(merge_key)` and `SpikeSortingOutput.get_sort_group_info(merge_key)` actually return values on v2 `merge_id`s — not just "doesn't raise." Confirm the returned recording has `is_filtered=True` annotated (R6).
- **R3 verification**: after the PR's slow tests, manually `ls $SPYGLASS_TEMP_DIR/sort_*` and confirm no leaked dirs. The smoke test catches the in-test case; the manual check catches "the test fixture creates and cleans up but the production path leaks."
- **R4 verification**: the `test_external_whitening_at_float64` test captures the kwargs handed to `sis.run_sorter`. If `sorter_params["whiten"]` is True at that point, R4 is broken regardless of whether the test passes its dtype assertion — call out this assertion's importance to a reviewer.
- **R8 outcome documentation**: whether the test passed or required folding in the clip, the PR description must explicitly state the result and the reasoning. A future reviewer needs to know whether SI 0.104's `NwbSortingExtractor` round-trip is or is not safe at the recording boundary.
- **R11 verification**: `Sorting.delete()` exists; `test_sorting_delete_removes_analyzer_folder` exercises both the happy path and the `PermissionError` propagation. Confirm `_analyzer_path` is called BEFORE `super().delete()` so the row is still available to resolve the path.
- **R13 verification**: `min_length_s` is on `ArtifactDetectionParamsSchema` with default 1.0; the `"default"` content row reflects the new field; `test_artifact_subtraction_filters_subsecond_slivers` passes with both default and overridden values.
- **R15 verification**: `CurationV2.insert_curation(labels=None)` returns successfully; no `ValueError("labels=None is invalid")` branch remains in `curation.py`.
- **R17 verification**: `cache_hash` equals `_hash_nwb_recording(analysis_file_name)` (independently recomputed) for any populated row. `_hash_nwb_recording` is called from BOTH `_write_nwb_artifact` and `_rebuild_nwb_artifact`. The `hashlib.sha256(data.tobytes())` pattern from Phase 1 is completely removed from `recording.py`. Bit-equivalence test does NOT compare `cache_hash` against the Phase 1 baseline. Do NOT pin a specific digest-length assertion (`NwbfileHasher`'s output length isn't pinned by this plan).
- **R18 verification**: `CommonReferenceParams.model_fields` does NOT contain `"reference"`. `CommonReferenceParams(reference="local")` raises a Pydantic `ValidationError`. The schema docstring cites v1 line refs for the hardcoded-dispatch rationale. The `whiten` field on `PreprocessingParamsSchema` is preserved (forward-compat for Phase 3) with the new docstring note.
- **R16 surface check**: `phase-1b-runtime-regressions.md` "Deliberately not in this phase" mentions the file-tracking deferral; `phase-2-analyzer-curation.md` Tasks section adds the `_get_v2_deleted_files` companion. Both notes link to each other.
- **v1 issues filed**: PR description includes links to the two upstream issues (`amplitude_thresh_uV` unit-conversion bug + heterogeneous-channel-gains silent `gains[0]` pick). Reviewer can click through to confirm they were filed before the PR ships.
- **HIGH severity N-items verified to actually fix the regression**: each of N19, N20, N21, N22 has a corresponding test that captures the buggy v2 behavior, then asserts the post-fix behavior matches v1's intent (or v2's documented improvement, for B1-revised). Specifically:
  - **N19**: `test_clusterless_thresholder_noise_levels_preserved` — `detect_peaks` receives `noise_levels=[1.0]`.
  - **N20**: `test_artifact_zscore_across_channels` — common-mode spike on all channels is detected.
  - **N21**: `test_get_merged_sorting_applies_merges_at_fetch` — kept-unit set after `get_merged_sorting` collapses contributors.
  - **N22**: `test_job_kwargs_resolved_in_every_compute_stage` — `_resolved_job_kwargs` is called in all three computed stages and `n_jobs` flows through to `sis.run_sorter`.
- **N36 cross-pipeline scope**: PR diff touches `decoding/v1/waveform_features.py`, `analysis/v1/group.py`, `analysis/v1/unit_annotation.py`. The reviewer must confirm these changes are scoped to the sparse-unit_id fix and don't sneak in unrelated decoding/group/annotation refactors.
- **B1 documentation**: the CHANGELOG entry explaining the v1 unit bug is in `CHANGELOG.md`, NOT only in a code comment. A future v1 user reading release notes must encounter the bug explanation before encountering broken artifact thresholds.
- **N49 verification**: `SpikeSortingOutput.get_restricted_merge_ids` default `sources` argument is `["v0", "v1", "v2"]` (NOT `["v0", "v1"]`). Calling the helper without `sources=` on a v2 merge_id returns a non-empty result. `test_get_restricted_merge_ids_default_includes_v2` is in the test suite. Reviewer must run the test against a v2-populated DB.
- **N50 verification**: a synthetic NWB with deliberately non-monotonic timestamps populates `Recording.make` cleanly, emits the expected `logger.warning`, and the written `ElectricalSeries.timestamps` is strictly increasing. Cross-reference confirmed: the in-flight v1 PR is mentioned in the CHANGELOG entry but the v2 fix is independent (don't block on v1 PR merge).
- **N51 verification**: a forced failure in `Sorting.make_insert` (monkey-patch `AnalysisNwbfile.add` to raise after `_build_analyzer`) leaves no analyzer folder on disk. Reviewer confirms the rollback block in `make` AND `_rebuild_analyzer_folder` both remove the folder.
- **N52 verification**: either `Sorting.fetch_nwb` returns a list-of-dicts with spike-times accessible (Option A), OR `Sorting.get_sorting(key, as_dataframe=True)` returns a pandas DataFrame with `unit_id` + `spike_times` (in seconds) columns (Option B). The executor's choice is documented in the PR description.
- **N53 verification**: all four `CurationV2` accessor methods are `@classmethod`: `get_recording`, `get_sorting`, `get_sort_group_info`, `get_merged_sorting`. Calling each directly on the class with a key dict succeeds. `_upstream_recording_row` is also `@classmethod` since it's invoked from `get_sorting`.
- **N54 test bundle**: all 10 test functions exist (either in `test_single_session_pipeline.py` or a new `test_v1_parity.py`) and pass. Reviewer spot-checks 2-3 to confirm they assert real v1-shape contracts, not tautologies.
- **R5 off-by-one annotation**: the ported `_consolidate_intervals` in `v2/utils.py` uses `searchsorted(side="right")` WITHOUT the `- 1` subtraction (v1's off-by-one bug). The helper's docstring documents the v1 divergence with the "v1-bug-fixed, like B7" pattern. A unit test exercises the end-of-interval boundary.
- **B3 schema-policy exception**: `overview.md § Resolved Design Decisions` must be updated in the SAME PR to document the `CurationV2.MergeGroup` addition as an authorized exception to the zero-migration policy. If `overview.md` isn't touched, the reviewer should flag the PR as incomplete.
- **B7 comment durability**: confirm the rationale comment above the heterogeneous-gain check is present and not just paraphrased — `test_heterogeneous_gain_raises_with_documented_reason` inspects the source string and will fail on a "cleanup" pass that removes it.
- Slow / integration / memory / parallel tests are marked.
- Docstrings, test names, and module names don't reference this plan, phase numbers, or files inside `.claude/docs/plans/`. In particular, `_nwb_iterators.py` describes the data shape it serves, not "Phase 1b regressions."
- `_parallel_make = True` is set on `Recording`, `ArtifactDetection`, and `Sorting`. It is NOT set on `ConcatenatedRecording` (Phase 1 placeholder, still raises `NotImplementedError`) and NOT set on any Manual table.
- `Recording._write_nwb_artifact` and `_rebuild_nwb_artifact` no longer call `recording.get_traces(return_in_uV=False)` at the top level (the only acceptable `get_traces` callers are inside the iterator classes themselves; R17 removed the streaming hasher loop entirely in favor of `_hash_nwb_recording`).
- The CHANGELOG entry and `SpikeSortingV2.md` section land in the same PR as the code change — not deferred.
- The Phase 1 `Deliberately not in this phase` list got a one-line update; nothing else in Phase 1's plan file was edited.
- The Phase 1 `Deliberately not in this phase` list got a one-line update; nothing else in Phase 1's plan file was edited.
