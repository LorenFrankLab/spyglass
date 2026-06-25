# Spike Sorting V2 Performance and Memory Scaling Review

Date: 2026-06-25

Scope: performance, RAM pressure, scratch-disk pressure, and scaling behavior of
spikesorting v2 under long recordings, high channel counts, many units, many
sessions, and repeated recompute or cache-management jobs. This is a different
lens from the prior scientific reproducibility, DataJoint/concurrency, NWB
portability, test coverage, and operational recovery reviews.

Method: local code inspection plus two independent explorer-agent reviews. This
review is read-only except for this document. I did not run benchmarks or the
test suite for this pass.

## Executive Summary

The v2 code has already eliminated several of the worst timestamp and artifact
scaling hazards. Trace writes use HDMF chunk iterators, write buffers are bounded
by an approximate duration rather than a fixed 5 GB target, artifact detection is
chunked, and the new timestamp helpers avoid `get_times()` in several hot paths.
Those are real improvements.

The remaining scaling risks are concentrated at boundary layers where v2 crosses
into NWB or SpikeInterface abstractions that still materialize large arrays, and
around large scratch artifacts whose cost is not surfaced before work starts.
The highest-priority issue is timestamp persistence/readback: v2 writes explicit
`timestamps` into every processed recording artifact, and the local
SpikeInterface NWB reader reads timestamp datasets eagerly to infer sampling
rate. That can reintroduce O(n_samples) RAM and I/O even when internal v2 helper
code is lazy.

The next tier is job-lifetime and scratch-space behavior. Sorter output lives in
a temporary directory that is cleaned before downstream analyzer and units-NWB
staging consume the returned sorting object. Recompute hashing loads full
analyzer extension arrays, including waveforms, into memory and writes temporary
analyzers under the system temp directory. Analyzer curation, UnitMatch, and
concatenated-recording paths can legitimately create large duplicate artifacts,
but they lack preflight estimators and budget checks.

## Findings

### 1. High: processed NWB timestamp storage/readback can force O(n_samples) arrays

`write_nwb_artifact()` always resolves a timestamp vector and writes it as
`ElectricalSeries(..., timestamps=timestamps_iterator, ...)`
(`src/spyglass/spikesorting/v2/_recording_nwb.py:162-199`). For normal
single-segment rate-based recordings, `_get_recording_timestamps()` falls back
to `recording.get_times()` when no override is supplied
(`src/spyglass/spikesorting/v2/_signal_math.py:183-185`). `ConcatenatedRecording`
calls `write_nwb_artifact()` without a `timestamps_override`
(`src/spyglass/spikesorting/v2/session_group.py:876-884`), so the synthetic
continuous concatenation pays for a full timestamp vector before the chunked
writer starts.

The read side has the same problem in the local SpikeInterface version.
`Recording.get_recording()` opens cached artifacts with
`load_time_vector=True` (`src/spyglass/spikesorting/v2/recording.py:1513-1517`).
In local SI 0.104.3, `NwbRecordingExtractor` reads
`electrical_series["timestamps"][:]` to infer `t_start` and sampling frequency
before installing the lazy time vector
(`.venv/lib/python3.11/site-packages/spikeinterface/extractors/nwbextractors.py:735-749`).
`ConcatenatedRecording._load_member_recordings()` repeats that read for every
member (`src/spyglass/spikesorting/v2/session_group.py:706-714`).

Impact: a 30 kHz recording needs about 824 MB per hour just for a float64
timestamp vector. Twelve hours is about 9.7 GB before trace buffers, analyzer
buffers, or sorter memory. Narrow sort groups also pay a large disk tax because
timestamps can be comparable to or larger than the trace payload.

Fix direction:

- Write `starting_time` plus `rate` instead of explicit `timestamps` whenever the
  persisted artifact is affine/regular.
- Keep explicit `timestamps` only for genuinely irregular source timestamps or
  multi-interval selections that cannot be represented regularly.
- For concatenated regular recordings, pass a lazy affine timestamp override or
  write `starting_time/rate` directly.
- Add a regression test where a large regular recording's `get_times()` raises,
  but writing and reading the processed artifact still succeeds.
- Add a readback test with a fake large timestamp dataset and assert the reader
  does not perform a full `[:]` read when a regular representation is available.

### 2. High: explicit/irregular raw timestamp restriction still takes an eager full-vector path

The restriction path now has a lazy fast path for single-segment rate-based raw
NWB recordings, but explicit or irregular timestamps still call
`_get_recording_timestamps(recording)` and then slice or concatenate arrays
(`src/spyglass/spikesorting/v2/_recording_restriction.py:489-545`). That
preserves wall-clock correctness, but it means long raw NWBs with explicit
timestamps can allocate the whole source timestamp vector plus the selected
persisted timestamps during `Recording.populate()`.

Impact: this is not the common path for regular rate-based acquisitions, but it
is exactly the path users need for dropped samples, irregular clocks, or
multi-segment recordings. Those are also cases where the timestamp vector may be
large and scientifically important.

Fix direction:

- Use bounded helpers such as `frames_for_times()` and `_segment_times_at()` for
  interval mapping, rather than requiring a full source vector.
- Introduce a lazy timestamp-slice override that can represent one or more
  explicit timestamp slices without `np.concatenate()`.
- Add tests with a fake explicit-time recording whose `get_times()` raises while
  bounded `sample_index_to_time()` calls succeed.

### 3. High: sorter output can be deleted before the returned sorting is consumed

`_run_si_sorter()` creates a `TemporaryDirectory` under
`spyglass.settings.temp_dir`, runs `sis.run_sorter()`, returns the sorting object,
and then cleans the temp directory in the outer `finally`
(`src/spyglass/spikesorting/v2/_sorting_dispatch.py:480-615`). The caller then
uses the returned sorting object for `remove_excess_spikes()`, analyzer build,
and units-NWB staging (`src/spyglass/spikesorting/v2/sorting.py:1531-1569`).

This is safe only if every supported sorter returns an object whose spike trains
are fully materialized at return time. Generic SpikeInterface sorter support
does not make that assumption obvious; file-backed extractors can defer reads
from the sorter output folder.

Impact: larger sorter outputs are the ones most likely to be file-backed. A
sorter may appear to succeed, then fail later during analyzer or NWB staging
because the backing folder has already been removed.

Fix direction:

- Keep the sorter temp directory alive through downstream analyzer and units-NWB
  staging, then clean it after the returned sorting has been consumed.
- Or explicitly materialize the sorting before cleanup, for example through a
  `NumpySorting.from_sorting(..., copy_spike_vector=True)` style conversion if
  supported by the pinned SI version.
- Add a fake file-backed sorter regression test that fails if spike trains are
  read after the temp folder has been deleted.

### 4. Medium-high: recording preprocessing/write resolves job kwargs but does not use them

The recording stage fetches preprocessing `job_kwargs` and explicitly documents
that the current HDMF streaming writer does not consume SI-style job kwargs
(`src/spyglass/spikesorting/v2/recording.py:1122-1133`). `make_compute()` calls
`_resolved_job_kwargs(preprocessing_job_kwargs)` only as an informational/tested
resolution path, then discards the result
(`src/spyglass/spikesorting/v2/recording.py:1229-1238`).

Impact: users can set `dj.config["custom"]["spikesorting_v2_job_kwargs"]` or a
per-row `job_kwargs` blob and reasonably expect preprocessing/writing to speed
up. The recording stage is often a dominant cost because it performs lazy
preprocessing, writes the full artifact, and then hashes the persisted file.
Today that work remains HDMF-iterator driven and effectively serial.

Fix direction:

- Make the docs and operator-facing summaries explicit that recording writes do
  not honor `n_jobs` yet.
- Add timing output that shows the effective `job_kwargs` and whether the stage
  consumed them.
- Consider a parallel materialization design: save the preprocessed recording to
  a managed binary/zarr scratch artifact using SI job kwargs, then write or link
  the final NWB representation from that bounded scratch path.
- Benchmark `Recording.populate()` at `n_jobs=1/4/8` before and after any change.

### 5. Medium-high: recompute hashing can materialize multi-GB analyzer arrays and extra copies

`hash_extension_data()` calls `get_data()`, `np.asarray()`, `np.round()`, and
`np.ascontiguousarray(...).tobytes()` over each full extension array
(`src/spyglass/spikesorting/v2/_recompute.py:27-49`). The hashed extension set
includes `waveforms`, which can be very large
(`src/spyglass/spikesorting/v2/_recompute.py:20-24`). `hash_recording_traces()`
is chunked over frames, but the fixed `300_000` frame chunk is cast to float64
and rounded (`src/spyglass/spikesorting/v2/_recompute.py:53-75`), so wide probes
can still allocate large chunks and copies.

Impact: recompute is intended for confidence before reclaiming storage. On large
sorts, the verification step itself can become the largest memory consumer,
especially if multiple recompute populates run in parallel.

Fix direction:

- Stream analyzer hashes by zarr or memmap chunks instead of using
  extension-level `get_data()` for large arrays.
- Scale recording hash chunks by target bytes, not a fixed frame count.
- Avoid unnecessary float64 promotion when the tolerance model permits hashing
  rounded float32 or integer views.
- Add memory-bounded tests or benchmarks with large fake waveform arrays and
  high-channel recordings.

### 6. Medium: analyzer recompute writes large temporary analyzers to system temp

`_recompute_analyzer_hashes()` uses `tempfile.mkdtemp()` without a `dir`, builds
a full zarr analyzer inside it, and then removes it in a `finally`
(`src/spyglass/spikesorting/v2/recompute.py:903-923`). Analyzer folders are
explicitly described elsewhere as regeneratable 5-50 GB scratch artifacts.

Impact: a large recompute campaign can fill `/tmp` or a node-local system temp
filesystem instead of the configured Spyglass temp/cache storage. A killed job
can also leave a large orphan outside the usual analyzer-cache audit paths.

Fix direction:

- Put recompute temporary analyzers under `spyglass.settings.temp_dir` or a
  dedicated subdirectory of the analyzer cache root.
- Use `TemporaryDirectory(dir=...)` and log the location before building.
- Add stale-temp cleanup/audit coverage for interrupted recompute jobs.

### 7. Medium: analyzer curation can double scratch usage and holds the per-sort lock across heavy compute

Metric curation always opens the display analyzer and conditionally opens or
builds a second whitened metric analyzer when PC/NN metrics are requested
(`src/spyglass/spikesorting/v2/metric_curation.py:930-965`). The lock is held
across analyzer loading/building, metric computation, and merge-group computation
(`src/spyglass/spikesorting/v2/metric_curation.py:930-972`). Default analyzer
builds compute random spikes, noise levels, templates, and waveforms
(`src/spyglass/spikesorting/v2/_sorting_analyzer.py:624-666`), with the shipped
recipe cap at 20,000 spikes per unit.

There are also two missed job-kwargs propagation points. `create_sorting_analyzer`
is called without job kwargs while estimating sparsity
(`src/spyglass/spikesorting/v2/_sorting_analyzer.py:587-602`), and
`compute_quality_metrics()` is called without job kwargs in both voltage and
PC-metric branches (`src/spyglass/spikesorting/v2/metric_curation.py:1102-1115`,
`src/spyglass/spikesorting/v2/metric_curation.py:1153-1164`).

Impact: PC/NN metric rows can create and retain two analyzer folders for a sort,
then perform heavy extension and metric work under a per-sort lock. That is
usually correct for cache integrity, but it serializes repeated curations and
can surprise operators on disk usage.

Fix direction:

- Add a preflight disk/RAM estimator for display and metric analyzer folders
  based on units, channels, waveform window, and `max_spikes_per_unit`.
- Consider lower metric-analyzer defaults, or make PC/NN metrics an explicitly
  "large scratch" option in preflight output.
- Pass filtered job kwargs to analyzer creation and to `compute_quality_metrics`
  where SI supports them, stripping Spyglass-only keys such as `random_seed`.
- Revisit lock scope once extension mutation points are isolated from pure
  DataFrame and merge-rule computation.

### 8. Medium: UnitMatch dense bundle extraction has no scratch/RAM budget and uses system temp

`extract_unitmatch_bundle()` intentionally builds dense, all-channel analyzers
for two recording halves (`sparse=False`), computes random spikes, waveforms, and
templates, then stacks both template arrays into
`(n_units, spike_width, n_channels, 2)`
(`src/spyglass/spikesorting/v2/_unitmatch_backend.py:107-186`). The caller uses
`tempfile.TemporaryDirectory(prefix="unitmatch_")` without a configured `dir`
(`src/spyglass/spikesorting/v2/unit_matching.py:754-799`).

Impact: dense all-channel waveforms are appropriate for UnitMatch, but the memory
and temp-disk footprint scales quickly with high channel counts, many units, and
many sessions. The temporary files also bypass the configured Spyglass scratch
location.

Fix direction:

- Put UnitMatch temporary directories under `spyglass.settings.temp_dir`.
- Add an estimator before extraction:
  `n_units * spike_width * n_channels * 2 * dtype_size`, plus waveforms and
  analyzer overhead.
- Stream or delete each half/session bundle as soon as the matcher no longer
  needs it.
- Add a smoke benchmark around high-channel synthetic sessions.

### 9. Medium: concat motion correction and split back-mapping can scale poorly

The `auto_default` motion row resolves to `preset="auto"`, which becomes
`rigid_fast` for same-day groups (`src/spyglass/spikesorting/v2/session_group.py:269-285`).
In the local SI source, the `rigid_fast` preset has empty `select_kwargs`
(`.venv/lib/python3.11/site-packages/spikeinterface/preprocessing/motion.py:116-126`),
and SI uses `gather_mode="memory"` when no peak selection is configured
(`.venv/lib/python3.11/site-packages/spikeinterface/preprocessing/motion.py:371-375`).
V2 then calls `correct_motion(..., output_motion=False,
output_motion_info=False, **motion_kwargs)`
(`src/spyglass/spikesorting/v2/_concat_recording.py:371-382`).

On the split-back side, `split_unit_spike_trains()` loops over every member and
every unit, allocating a boolean mask for each pair
(`src/spyglass/spikesorting/v2/_concat_recording.py:176-214`).

Impact: long same-day concatenations can hit memory pressure during motion
estimation before the chunked NWB write starts. Large concat sorts with many
members, units, or spikes can also pay O(members * spikes) work during
back-mapping.

Fix direction:

- Make the default same-day motion preset include bounded peak selection or a
  documented memory budget.
- Consider a disk-backed motion workspace under configured scratch if SI exposes
  one that preserves v2's side-artifact contract.
- Split each sorted spike train once using `np.searchsorted(boundaries)` and
  build member outputs incrementally.
- Add equivalence tests for many-member/dense-spike split back-mapping.

## Already Solid

- Trace writing is chunked through `SpikeInterfaceRecordingDataChunkIterator`
  and does not materialize full trace arrays
  (`src/spyglass/spikesorting/v2/_nwb_iterators.py:30-106`).
- Timestamp writes use a dedicated iterator and can accept lazy timestamp
  overrides (`src/spyglass/spikesorting/v2/_nwb_iterators.py:167-243`).
- Write buffer sizing is bounded by approximate duration and capped, with tests
  for narrow groups (`src/spyglass/spikesorting/v2/utils.py:393-434`,
  `tests/spikesorting/v2/test_write_buffer_gb.py`).
- Artifact detection defaults to chunked scanning and stores run intervals rather
  than per-frame boolean flags
  (`src/spyglass/spikesorting/v2/_artifact_intervals.py:42-160`).
- Artifact masking maps intervals to frame ranges and uses SI `silence_periods`
  instead of building a full trigger vector
  (`src/spyglass/spikesorting/v2/_sorting_artifact_mask.py:124-262`).
- Shared-artifact matching uses `timestamp_fingerprint()` instead of full
  timestamp equality checks
  (`src/spyglass/spikesorting/v2/artifact.py:340-386`).
- Sorter temp directories are correctly rooted under `spyglass.settings.temp_dir`
  and SI global job kwargs are restored after sorter execution
  (`src/spyglass/spikesorting/v2/_sorting_dispatch.py:480-615`).

## Suggested Fix Order

1. Fix regular processed-artifact timestamp representation and concat timestamp
   overrides. This has the largest RAM/disk payoff and makes the prior lazy
   timestamp work durable across NWB boundaries.
2. Fix sorter temp lifetime or explicit materialization. This is a correctness
   and reliability issue that becomes more likely with large file-backed sorter
   outputs.
3. Move recompute and UnitMatch temporary directories under configured Spyglass
   scratch, then add stale-temp cleanup/audit coverage.
4. Stream recompute hashes for large analyzer extensions and byte-bound recording
   trace chunks.
5. Add preflight estimators for analyzer, UnitMatch, and concat materialization.
6. Improve job-kwargs propagation for analyzer creation, quality metrics, and
   operator-facing summaries.
7. Optimize concat split back-mapping after the larger memory risks are handled.
