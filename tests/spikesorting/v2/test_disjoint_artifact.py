"""Disjoint-recording artifact-detection regression tests.

Artifact detection ran its complement over a single envelope
``timestamps[0]..timestamps[-1]``; for DISJOINT sort intervals that
envelope includes the wall-clock gaps ``Recording.make`` excluded, so
the artifact-removed ``valid_times`` could span a gap. That inflates the
obs_intervals duration/firing-rate metadata and can let a sub-
``min_length_s`` sliver survive by borrowing gap time. The fix splits the
persisted timestamp vector at discontinuities into base chunks and
subtracts artifacts per chunk (v1 subtracts from the explicit
``sort_interval_valid_times``).

The hermetic tests here pin the discontinuity-split helper; the
DB-gated tests exercise ``_detect_artifacts`` / ``_apply_artifact_mask``
on a synthetic disjoint recording. A full-pipeline integration test
lives in ``test_single_session_pipeline.py``.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_base_intervals_from_timestamps_contiguous():
    """A contiguous (uniform) timeline yields exactly one base interval."""
    from spyglass.spikesorting.v2.utils import _base_intervals_from_timestamps

    fs = 30000.0
    t0 = 12.5
    ts = t0 + np.arange(300) / fs
    intervals = _base_intervals_from_timestamps(ts, fs)
    assert len(intervals) == 1
    assert intervals[0][0] == pytest.approx(ts[0])
    assert intervals[0][1] == pytest.approx(ts[-1])


def test_base_intervals_from_timestamps_splits_at_gap():
    """A wall-clock gap (diff > 1.5/fs) splits the vector into two chunks.

    The chunk bounds are the first/last sample of each chunk -- the
    inter-chunk gap is NOT part of any base interval, so an artifact
    complement built per-chunk can never span it.
    """
    from spyglass.spikesorting.v2.utils import _base_intervals_from_timestamps

    fs = 30000.0
    dt = 1.0 / fs
    chunk = np.arange(100) * dt
    t0 = 10.0
    gap = 0.5
    chunk2_start = t0 + chunk[-1] + gap + dt
    ts = np.concatenate([t0 + chunk, chunk2_start + chunk])

    intervals = _base_intervals_from_timestamps(ts, fs)
    assert len(intervals) == 2
    assert intervals[0][0] == pytest.approx(ts[0])
    assert intervals[0][1] == pytest.approx(ts[99])
    assert intervals[1][0] == pytest.approx(ts[100])
    assert intervals[1][1] == pytest.approx(ts[-1])
    # No base interval spans the gap.
    for start, end in intervals:
        assert not (
            start < ts[99] + gap / 2 < end
        ), "a base interval must not span the inter-chunk gap"


def test_base_intervals_from_timestamps_empty():
    """An empty timestamp vector yields no base intervals."""
    from spyglass.spikesorting.v2.utils import _base_intervals_from_timestamps

    assert _base_intervals_from_timestamps(np.asarray([]), 30000.0) == []


def test_base_intervals_from_timestamps_three_chunks_two_gaps():
    """Three chunks separated by TWO gaps yield three correct intervals.

    Every existing disjoint fixture is a single gap (two chunks); this
    pins the ``zip(starts, ends)`` indexing on the multi-gap path. With
    chunk lengths 80 / 120 / 100 the per-chunk bounds must come from the
    correct sample offsets, not a hard-coded 2-chunk assumption. (Closes
    review "not-checked" item #6.)
    """
    from spyglass.spikesorting.v2.utils import _base_intervals_from_timestamps

    fs = 30000.0
    dt = 1.0 / fs
    n1, n2, n3 = 80, 120, 100
    t0 = 7.0
    c1 = t0 + np.arange(n1) * dt
    c2_start = c1[-1] + 0.5 + dt  # 0.5 s gap after chunk 1
    c2 = c2_start + np.arange(n2) * dt
    c3_start = c2[-1] + 0.3 + dt  # 0.3 s gap after chunk 2
    c3 = c3_start + np.arange(n3) * dt
    ts = np.concatenate([c1, c2, c3])

    intervals = _base_intervals_from_timestamps(ts, fs)
    assert len(intervals) == 3
    # Per-chunk inclusive [first, last] sample times, by absolute index.
    expected = [
        (ts[0], ts[n1 - 1]),
        (ts[n1], ts[n1 + n2 - 1]),
        (ts[n1 + n2], ts[-1]),
    ]
    for (got_s, got_e), (exp_s, exp_e) in zip(intervals, expected):
        assert got_s == pytest.approx(exp_s)
        assert got_e == pytest.approx(exp_e)
    # Neither gap is spanned by any base interval.
    gap1_mid = 0.5 * (ts[n1 - 1] + ts[n1])
    gap2_mid = 0.5 * (ts[n1 + n2 - 1] + ts[n1 + n2])
    for start, end in intervals:
        assert not (start < gap1_mid < end)
        assert not (start < gap2_mid < end)


def test_base_intervals_gap_threshold_robust_to_subsample_jitter():
    """The ``1.5 / fs`` split threshold straddles the decision boundary.

    Within-chunk timestamp jitter (a diff of ``1.4 / fs`` -- below the
    1.5-sample threshold) must NOT spuriously split a chunk, while a
    genuine missing-sample gap (a diff of ``1.6 / fs`` -- just above the
    threshold) MUST split. Pins both sides of the boundary so a future
    threshold change (e.g. to ``1.0 / fs``) is caught.
    """
    from spyglass.spikesorting.v2.utils import _base_intervals_from_timestamps

    fs = 30000.0
    dt = 1.0 / fs

    # Jitter just BELOW the boundary: one within-chunk diff is 1.4/fs.
    # Build cumulative times so only that one diff is enlarged.
    diffs_no_split = np.full(50, dt)
    diffs_no_split[25] = 1.4 * dt
    ts_no_split = 3.0 + np.concatenate([[0.0], np.cumsum(diffs_no_split)])
    assert (
        len(_base_intervals_from_timestamps(ts_no_split, fs)) == 1
    ), "a 1.4/fs sub-threshold jitter must not split a chunk"

    # Gap just ABOVE the boundary: one diff is 1.6/fs -> split into two.
    diffs_split = np.full(50, dt)
    diffs_split[25] = 1.6 * dt
    ts_split = 3.0 + np.concatenate([[0.0], np.cumsum(diffs_split)])
    intervals = _base_intervals_from_timestamps(ts_split, fs)
    assert (
        len(intervals) == 2
    ), "a 1.6/fs supra-threshold gap must split the chunk"
    # The split lands exactly at the enlarged diff (between sample 25 and
    # 26 of the cumulative vector).
    assert intervals[0][1] == pytest.approx(ts_split[25])
    assert intervals[1][0] == pytest.approx(ts_split[26])


def _disjoint_recording(chunk_len=200, n_ch=4, fs=30000.0, gap_s=0.5):
    """Build a synthetic 2-chunk disjoint SI recording (gap-preserving times)."""
    import spikeinterface as si

    dt = 1.0 / fs
    rng = np.random.default_rng(0)
    traces = (rng.standard_normal((2 * chunk_len, n_ch)) * 2.0).astype(
        np.float32
    )
    rec = si.NumpyRecording([traces], sampling_frequency=fs)
    rec.set_channel_gains([1.0] * n_ch)  # 1 uV/count
    rec.set_channel_offsets([0.0] * n_ch)
    t1 = np.arange(chunk_len) * dt
    t2 = (chunk_len * dt + gap_s) + np.arange(chunk_len) * dt
    rec.set_times(np.concatenate([t1, t2]))
    return rec, traces, fs, chunk_len, gap_s


@pytest.mark.usefixtures("dj_conn")
def test_detect_artifacts_valid_times_never_cross_gap():
    """``_detect_artifacts`` keeps valid_times within recorded chunks.

    A transient in chunk 1 plus the (artifact-free) chunk 2 must yield
    valid intervals that each lie within a single chunk -- never one
    interval spanning the inter-chunk wall-clock gap. Subtracting from a
    single envelope (the old behavior) would merge the post-transient
    chunk-1 tail with chunk 2 across the gap.
    """
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    rec, traces, fs, chunk_len, gap_s = _disjoint_recording()
    # Plant a supra-threshold transient at frame 50 (chunk 1).
    traces[50, :] = 5000.0
    times = rec.get_times()
    gap_mid = 0.5 * (times[chunk_len - 1] + times[chunk_len])

    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_thresh_uV=500.0,
        zscore_thresh=None,
        proportion_above_thresh=1.0,
        removal_window_ms=1.0,
        join_window_ms=1.0,
        min_length_s=0.001,
    )
    valid_times = ArtifactDetection._detect_artifacts(rec, validated)

    assert len(valid_times) >= 1
    for start, end in valid_times:
        assert not (start < gap_mid < end), (
            f"valid interval [{start}, {end}] spans the inter-chunk gap "
            f"(gap_mid={gap_mid}); per-chunk subtraction should prevent it."
        )
    # Chunk 2 (artifact-free) survives as its own interval past the gap.
    assert any(
        start >= times[chunk_len] for start, end in valid_times
    ), "expected a valid interval inside chunk 2 (after the gap)"


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_preserves_chunk_boundary_frame():
    """``_apply_artifact_mask`` zeros artifacts but not the gap boundary.

    With gap-respecting valid_times, the complement between chunk 1's last
    frame and chunk 2's first frame is a pure wall-clock gap (zero real
    frames). The mask must zero the in-chunk artifact frames yet leave the
    chunk-boundary sample untouched -- masking it would zero a good sample
    per gap.
    """
    from spyglass.spikesorting.v2.sorting import Sorting

    rec, traces, fs, chunk_len, gap_s = _disjoint_recording()
    times = rec.get_times()
    k = chunk_len - 1  # chunk 1's last frame
    # Distinctive, clearly-nonzero boundary samples to detect over-masking.
    traces[k, :] = 7.0
    traces[k + 1, :] = 8.0
    traces[50:60, :] = 9.0  # the artifact span we will remove

    valid_times = np.array(
        [
            [times[0], times[50]],  # chunk 1 up to the artifact
            [times[60], times[k]],  # chunk 1 after artifact .. last frame
            [times[k + 1], times[-1]],  # chunk 2
        ]
    )
    masked = Sorting._apply_artifact_mask(
        recording=rec, valid_times=valid_times
    )
    mt = masked.get_traces(return_in_uV=False)

    assert np.all(mt[50:60, :] == 0), "artifact frames must be zeroed"
    assert np.all(
        mt[k, :] == 7.0
    ), "chunk-boundary frame must NOT be masked (pure-gap over-mask)"
    assert np.all(mt[k + 1, :] == 8.0), "chunk 2's first frame must survive"


@pytest.mark.usefixtures("dj_conn")
def test_detect_artifacts_removal_window_does_not_spill_across_gap():
    """Chunk 1's removal window must not reach across the gap into chunk 2.

    The removal window expands the detected artifact; expanding it in
    FRAME space (``end_f + half_window_frames``) lets an artifact near
    chunk 1's end land in chunk 2's frames across a large wall-clock gap,
    so chunk 2's first samples get removed. v1's window is time-based
    (``timestamps[end] + half_win`` seconds), which cannot cross a gap
    orders of magnitude larger than the window. Chunk 2 must stay fully
    valid: a valid interval starts at ``times[chunk_len]`` (chunk 2's
    first sample), NOT ``times[chunk_len + half_window_frames]``.
    """
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    rec, traces, fs, chunk_len, gap_s = _disjoint_recording()
    # Transient on the 3rd-to-last frame of chunk 1; the 1 ms removal
    # window (half = ceil(0.5 ms * fs) = 15 frames at 30 kHz) would, in
    # frame space, reach ~12 frames into chunk 2.
    traces[chunk_len - 3, :] = 5000.0
    times = rec.get_times()
    chunk2_start = times[chunk_len]

    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_thresh_uV=500.0,
        zscore_thresh=None,
        proportion_above_thresh=1.0,
        removal_window_ms=1.0,
        join_window_ms=0.0,
        min_length_s=0.001,
    )
    valid_times = ArtifactDetection._detect_artifacts(rec, validated)

    starts = [s for s, _ in valid_times]
    assert any(abs(s - chunk2_start) <= 1.0 / fs for s in starts), (
        "chunk 2's first sample was removed: chunk-1's artifact removal "
        f"window spilled across the gap. valid_times={valid_times.tolist()}, "
        f"expected a chunk-2 interval starting at {chunk2_start}."
    )


@pytest.mark.usefixtures("dj_conn")
def test_detect_artifacts_join_does_not_bridge_gap():
    """Artifacts in different chunks must not join across the gap.

    Frame indices are contiguous across the inter-chunk gap, so a
    frame-space join (``f - cur_end <= join_window_frames``) can merge an
    artifact near chunk 1's end with one near chunk 2's start even though
    they are 0.5 s apart in wall-clock. The merged span then straddles
    both chunks and over-masks chunk 1's valid tail. The join must not
    bridge a timestamp gap.
    """
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    rec, traces, fs, chunk_len, gap_s = _disjoint_recording()
    # Artifact 20 frames before chunk 1's end, and one at chunk 2's first
    # frame: 20 frames apart, within the 30-frame (1 ms) join window, but
    # 0.5 s apart in wall-clock.
    traces[chunk_len - 20, :] = 5000.0
    traces[chunk_len, :] = 5000.0
    times = rec.get_times()

    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_thresh_uV=500.0,
        zscore_thresh=None,
        proportion_above_thresh=1.0,
        removal_window_ms=1.0,  # half = 15 frames -> tail beyond 195 free
        join_window_ms=1.0,  # 30 frames -> would bridge the 20-frame gap
        min_length_s=1e-5,
    )
    valid_times = ArtifactDetection._detect_artifacts(rec, validated)

    # The chunk-1 tail just before the gap (beyond artifact's own 15-frame
    # window) must remain valid -- it must not be removed by a span that
    # bridged to the chunk-2 artifact.
    tail_t = times[chunk_len - 4]
    covered = any(s <= tail_t < e for s, e in valid_times)
    assert covered, (
        f"chunk-1 valid tail near {tail_t} was removed: the frame-space "
        f"join bridged the gap. valid_times={valid_times.tolist()}"
    )
