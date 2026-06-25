# Spike Sorting V2 Timebase, Sample-Index, and Interval Alignment Review

Date: 2026-06-25

Scope: frame/time conversion, timestamp persistence, interval slicing,
artifact valid-times, sort-time masking, unit spike-time readback, concat member
boundaries, sampling-frequency assumptions, and cross-session alignment. This
review focuses on sample-level correctness rather than general performance or
schema design.

Method: local static source/test/docs inspection, targeted arithmetic probes,
and two independent explorer-agent reviews. This review is read-only except for
this document. I did not run the test suite.

## Executive Summary

This is one of the stronger areas of v2. The explicit-timestamp path has
received real care: disjoint intervals preserve wall-clock gaps, spike readback
prefers stored `spike_sample_index`, artifact intervals are gap-aware, concat
member boundaries use half-open frame ranges, and tests cover several nasty
off-by-one cases.

The main remaining risk is the newer fast path for rate-based recordings. It
replaces timestamp search with affine arithmetic, but uses raw floating-point
`ceil` / `floor` on frame coordinates. Exact sample-boundary times at common
30 kHz rates can evaluate slightly above an integer, so a requested interval can
lose its first sample. The other findings are narrower: concat should explicitly
reject mixed sampling rates, bypassed concat+artifact rows can claim artifact
metadata while sorting unmasked, and some public/helper paths need stronger
boundary tests.

## What Looks Solid

- `_consolidate_intervals()` uses `searchsorted` with half-open frame ranges and
  validates monotonic timestamps before mapping intervals
  (`src/spyglass/spikesorting/v2/_signal_math.py:201-262`).
- `_spike_times_to_frames()` maps absolute spike times against the actual
  timestamp vector, handles tiny roundoff by nearest-sample snapping, and raises
  on spikes in disjoint gaps (`src/spyglass/spikesorting/v2/_signal_math.py:265-381`).
- `restrict_recording()` carries a `timestamps_override` through
  `frame_slice()` / `concatenate_recordings()`, so persisted recordings preserve
  raw wall-clock timestamps even when SpikeInterface drops them from the sliced
  object (`src/spyglass/spikesorting/v2/_recording_restriction.py:368-579`).
- Units NWBs store absolute `spike_times` plus a Spyglass
  `spike_sample_index` sidecar, and both `Sorting.get_sorting()` and
  `CurationV2.get_sorting()` prefer the sidecar when present
  (`src/spyglass/spikesorting/v2/_units_nwb.py:35-98`,
  `src/spyglass/spikesorting/v2/sorting.py:1773-1853`,
  `src/spyglass/spikesorting/v2/curation.py:1263-1359`).
- Artifact detection builds valid intervals per recorded chunk rather than over
  one gap-spanning envelope (`src/spyglass/spikesorting/v2/_artifact_intervals.py:267-354`,
  `src/spyglass/spikesorting/v2/_artifact_intervals.py:450-514`).
- Concat split math is pure and half-open: a spike at a boundary belongs to the
  next member (`src/spyglass/spikesorting/v2/_concat_recording.py:149-214`).
- Tests cover disjoint readback, final-sample spike readback, artifact gap
  handling, concat boundary split math, and nonzero-start timestamp services
  (`tests/spikesorting/v2/test_disjoint_readback.py:26`,
  `tests/spikesorting/v2/single_session/test_disjoint_intervals.py:15`,
  `tests/spikesorting/v2/test_disjoint_artifact.py:214`,
  `tests/spikesorting/v2/test_concat_recording.py:85`,
  `tests/spikesorting/v2/test_recording_services.py:81`).

## Findings

### 1. High: rate-based interval restriction can lose boundary samples

For single-segment rate-based recordings, `restrict_recording()` uses the lazy
regular-grid path instead of materializing timestamps
(`src/spyglass/spikesorting/v2/_recording_restriction.py:489-508`).
That path converts interval seconds to frame bounds with:

- `rel_start = (interval_start - t_start) * sampling_frequency`
- `start_indices = ceil(rel_start)`
- `stop_indices = floor(rel_stop) + 1`

(`src/spyglass/spikesorting/v2/_recording_restriction.py:274-280`).

The problem is floating-point representation. A true sample-boundary start can
evaluate just above an integer, for example `119.00000000000001`, so `ceil()`
advances the interval start by one frame. A targeted arithmetic probe showed
this at ordinary 30 kHz rates and nonzero starts; the existing parity test uses
`fs=2.0` / `t_start=10.0` and does not hit the failure
(`tests/spikesorting/v2/test_recording_services.py:81-100`).

Impact: rate-based raw recordings can silently drop or shift edge samples during
recording materialization and recompute. The persisted recording and content
hash then become self-consistent around the wrong slice, making the drift hard
to diagnose later.

Fix direction:

- Snap near-integer relative frame coordinates before `ceil` / `floor`, using a
  tolerance tied to floating-point precision and sampling frequency.
- Or replace the arithmetic conversion with a searchsorted-equivalent helper
  that uses the affine formula but preserves the same boundary semantics as
  `_consolidate_intervals()`.
- Add equivalence tests against `_consolidate_intervals()` for `fs=30000`,
  nonzero and large `t_start`, exact sample-boundary starts/stops, and multiple
  intervals.

### 2. Medium: concat sampling-frequency compatibility is implicit

`assert_concat_compatible()` front-loads channel-id and geometry checks before
calling `spikeinterface.concatenate_recordings()`
(`src/spyglass/spikesorting/v2/_concat_recording.py:217-289`). It does not
explicitly check `get_sampling_frequency()`, and tests cover channel/geometry
compatibility but not a same-channel mixed-rate case.

`ConcatenatedRecording.make_compute()` later stores one concat
`sampling_frequency`, computes `total_duration_s = corrected_n_samples / fs`,
and persists member boundaries as raw sample counts
(`src/spyglass/spikesorting/v2/session_group.py:829-897`). The split helper
then returns per-member `NumpySorting` objects using the sampling frequency from
the caller-provided concat sorting object
(`src/spyglass/spikesorting/v2/session_group.py:1030-1058`).

Impact: if mixed-rate members get past SpikeInterface's stitch checks, every
duration and local-frame-to-seconds conversion downstream is wrong. Even if SI
currently raises, the error is deferred to compute time and is less explicit
than the existing channel/geometry diagnostics.

Fix direction:

- Add sampling-frequency equality to `assert_concat_compatible()` with a clear
  error naming the first mismatching member.
- Add a pure compatibility test with identical channel IDs/geometry but 30 kHz
  vs 20 kHz recordings.
- Add docs text that concat performs no resampling and requires identical
  sampling frequency across members.
- In `split_sorting_by_session()`, optionally compare
  `sorting.get_sampling_frequency()` with the stored concat row and raise on
  mismatch before returning per-member sortings.

### 3. Medium: bypassed concat+artifact selections claim artifact metadata while sorting unmasked

The normal selection builder rejects `concat_recording_id` combined with
`artifact_detection_id` (`src/spyglass/spikesorting/v2/_selection_plan.py:181-194`).
That is the right user-facing contract.

The computed path does not revalidate the invariant. For concat sources,
`Sorting.make_fetch()` unconditionally sets `obs_intervals = None`
(`src/spyglass/spikesorting/v2/sorting.py:1268-1272`). `make_compute()` applies
the artifact mask only when both `artifact_detection_id` and `obs_intervals`
are present (`src/spyglass/spikesorting/v2/sorting.py:1502-1511`).

Impact: a raw/bypassed `SortingSelection.ArtifactDetectionSource` row on a
concat-backed sorting can make the relational metadata look artifact-backed,
while the sort actually runs on the full unmasked concat recording and writes
units with full-window observation intervals.

Fix direction:

- Revalidate in `Sorting.make_fetch()` or `make_compute()` that concat sources
  have no `ArtifactDetectionSource` row.
- Add a bypass-style test that manually inserts the inconsistent parts and
  asserts populate fails before sorter execution.

### 4. Medium-low: shared artifact detection trusts insert-time time-axis checks

`SharedArtifactGroup.insert_group()` performs strong normal-path checks: same
session, same sampling frequency, exact timestamp fingerprint, matching sample
count, and matching dtype (`src/spyglass/spikesorting/v2/artifact.py:295-410`).
But `ArtifactDetection.make_compute()` later reloads member recordings and
aggregates channels by frame index without rechecking those guarantees
(`src/spyglass/spikesorting/v2/artifact.py:927-956`).

Impact: a raw-inserted or corrupted shared group can compute valid times on one
member's timeline and write them for all member NWB files. This is a bypass
state, but the consequence is directly timebase-related: artifact intervals are
frame-aligned to the wrong recording.

Fix direction:

- Persist a member time-axis fingerprint on the group, or rerun a bounded
  compatibility check in `make_fetch()` before compute.
- Add a bypass/recompute test with equal sampling frequency and sample count but
  shifted timestamps, expecting a loud failure.

### 5. Low: concat duration and endpoint-based observation intervals use different conventions

`ConcatenatedRecording.total_duration_s` is stored as `n_samples / fs`
(`src/spyglass/spikesorting/v2/session_group.py:868-895`). Persisted timestamps
are sample-centered, so a nonempty concat recording's endpoint span is
`(n_samples - 1) / fs`. Units NWB observation intervals are derived from
timestamp endpoints when no artifact mask is provided
(`src/spyglass/spikesorting/v2/_units_nwb.py:592-606`), while reporting can use
stored duration fields.

Impact: denominators can differ by one sample depending on whether a workflow
uses stored `total_duration_s` or sums `obs_intervals`. This is tiny for long
recordings, but it is still a convention mismatch in a codebase that otherwise
tries hard to make sample/interval semantics explicit.

Fix direction:

- Choose and document one duration convention: sample count duration (`n / fs`)
  or endpoint span (`(n - 1) / fs`).
- Make concat row duration, Units `obs_intervals`, and reporting use that
  convention consistently.
- Add a test comparing concat `total_duration_s`, persisted timestamp endpoints,
  and `obs_intervals`.

### 6. Low: manual/legacy `valid_times` can still hide a one-frame artifact at a disjoint seam

The normal artifact detection path widens detections with a positive
`removal_window_ms`, so the single-frame seam edge is mostly unreachable from
validated detector parameters (`src/spyglass/spikesorting/v2/_params/artifact_detection.py:72-75`).
But `apply_artifact_mask()` accepts `valid_times` directly, and it explicitly
drops width-one complement ranges when they look like pure inter-chunk gaps
(`src/spyglass/spikesorting/v2/_sorting_artifact_mask.py:203-229`).
The comment notes the remaining edge: a genuine one-frame artifact exactly on a
chunk's final sample is treated like a boundary gap.

Impact: a hand-built, legacy, or corrupted artifact `IntervalList` can leave a
one-frame boundary artifact unmasked. Normal v2 detection is much less exposed,
but this is still an integrity edge at the consumer boundary.

Fix direction:

- Prefer frame-native artifact masks internally, or carry explicit metadata that
  distinguishes "pure inter-chunk gap" from "valid_times intentionally excludes
  this boundary sample."
- Add a direct `apply_artifact_mask()` test for a manually supplied
  one-frame seam artifact and decide whether to reject ambiguous input or mask it.

## Coverage and Documentation Follow-ups

- Add an end-to-end concat split test that plants spikes at `0`, `n0 - 1`,
  `n0`, and `n0 + n1 - 1`, then asserts exact local frames through
  `ConcatenatedRecording.split_sorting_by_session()`. The pure helper already
  covers this; the table-backed glue only checks in-range frames
  (`tests/spikesorting/v2/test_concat_recording.py:85-100`,
  `tests/spikesorting/v2/test_session_group_concat.py:763-795`).
- Add UnitMatch backend coverage for nonzero-start recordings and spikes around
  the cross-validation half boundary. Existing backend tests use zero-origin
  sortings.
- Fix the docs status contradiction: the UnitMatch workflow is documented as
  available/recommended for cross-session matching, while the status section
  still says cross-session unit matching is not yet available
  (`docs/src/Features/SpikeSortingV2.md:880-905`,
  `docs/src/Features/SpikeSortingV2.md:1055`).
- Label downstream `get_spike_times()` docs as seconds at the primary table and
  snippet, and distinguish that from SpikeInterface sample-frame trains
  (`docs/src/Features/SpikeSortingV2.md:985-998`).

## Suggested Fix Order

1. Fix the rate-based `_consolidate_regular_intervals()` boundary math and add
   30 kHz/nonzero-start parity tests.
2. Add explicit concat sampling-frequency validation and docs.
3. Revalidate concat+artifact absence in `Sorting.make_fetch()` / `make_compute()`.
4. Decide the one-sample duration convention for concat rows vs observation
   intervals.
5. Add table-backed seam and UnitMatch half-boundary tests.
