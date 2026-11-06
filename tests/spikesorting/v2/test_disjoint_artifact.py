"""Artifact detection on disjoint recordings.

Artifacts are subtracted per recorded chunk, so ``valid_times`` never span a
wall-clock gap. Full-pipeline coverage is in
``single_session/test_disjoint_intervals.py``.
"""

from __future__ import annotations

import numpy as np
import pytest


def _base_intervals(timestamps, fs):
    """``base_intervals_and_gaps`` chunks for an explicit timestamp vector."""
    import warnings

    import spikeinterface as si

    from spyglass.spikesorting.v2._core.signal_math import (
        base_intervals_and_gaps,
    )

    timestamps = np.asarray(timestamps, dtype=np.float64)
    rec = si.NumpyRecording(
        traces_list=[np.zeros((timestamps.size, 1), dtype="float32")],
        sampling_frequency=fs,
    )
    with warnings.catch_warnings():  # set_times warns "not recommended"
        warnings.simplefilter("ignore")
        rec.set_times(timestamps, segment_index=0)
    return base_intervals_and_gaps(rec, fs).base_intervals


def test_base_intervals_and_gaps_contiguous():
    """A contiguous (uniform) timeline yields exactly one base interval."""
    fs = 30000.0
    t0 = 12.5
    ts = t0 + np.arange(300) / fs
    intervals = _base_intervals(ts, fs)
    assert len(intervals) == 1
    assert intervals[0][0] == pytest.approx(ts[0])
    assert intervals[0][1] == pytest.approx(ts[-1])


def test_base_intervals_and_gaps_splits_at_gap():
    """A wall-clock gap (diff > 1.5/fs) splits the vector into two chunks.

    Bounds are each chunk's first/last sample; no interval spans the gap.
    """
    fs = 30000.0
    dt = 1.0 / fs
    chunk = np.arange(100) * dt
    t0 = 10.0
    gap = 0.5
    chunk2_start = t0 + chunk[-1] + gap + dt
    ts = np.concatenate([t0 + chunk, chunk2_start + chunk])

    intervals = _base_intervals(ts, fs)
    assert len(intervals) == 2
    assert intervals[0][0] == pytest.approx(ts[0])
    assert intervals[0][1] == pytest.approx(ts[99])
    assert intervals[1][0] == pytest.approx(ts[100])
    assert intervals[1][1] == pytest.approx(ts[-1])
    for start, end in intervals:
        assert not (
            start < ts[99] + gap / 2 < end
        ), "a base interval must not span the inter-chunk gap"


def test_base_intervals_and_gaps_empty():
    """An empty timestamp vector yields no base intervals."""
    assert _base_intervals(np.asarray([]), 30000.0) == []


def test_base_intervals_and_gaps_three_chunks_two_gaps():
    """Three chunks separated by TWO gaps yield three correct intervals.

    Uses unequal chunk lengths (80 / 120 / 100) so offsets are checked.
    """
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

    intervals = _base_intervals(ts, fs)
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
    gap1_mid = 0.5 * (ts[n1 - 1] + ts[n1])
    gap2_mid = 0.5 * (ts[n1 + n2 - 1] + ts[n1 + n2])
    for start, end in intervals:
        assert not (start < gap1_mid < end)
        assert not (start < gap2_mid < end)


def test_base_intervals_gap_threshold_robust_to_subsample_jitter():
    """The ``1.5 / fs`` split threshold straddles the decision boundary.

    A ``1.4 / fs`` diff does not split; a ``1.6 / fs`` diff does.
    """
    fs = 30000.0
    dt = 1.0 / fs

    # Jitter just BELOW the boundary: one within-chunk diff is 1.4/fs.
    # Build cumulative times so only that one diff is enlarged.
    diffs_no_split = np.full(50, dt)
    diffs_no_split[25] = 1.4 * dt
    ts_no_split = 3.0 + np.concatenate([[0.0], np.cumsum(diffs_no_split)])
    assert (
        len(_base_intervals(ts_no_split, fs)) == 1
    ), "a 1.4/fs sub-threshold jitter must not split a chunk"

    # Gap just ABOVE the boundary: one diff is 1.6/fs -> split into two.
    diffs_split = np.full(50, dt)
    diffs_split[25] = 1.6 * dt
    ts_split = 3.0 + np.concatenate([[0.0], np.cumsum(diffs_split)])
    intervals = _base_intervals(ts_split, fs)
    assert (
        len(intervals) == 2
    ), "a 1.6/fs supra-threshold gap must split the chunk"
    # Split between samples 25 and 26.
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
    """``detect_artifacts`` keeps valid_times within recorded chunks.

    Each valid interval lies within a single chunk.
    """
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts

    rec, traces, fs, chunk_len, gap_s = _disjoint_recording()
    # Plant a supra-threshold transient at frame 50 (chunk 1).
    traces[50, :] = 5000.0
    times = rec.get_times()
    gap_mid = 0.5 * (times[chunk_len - 1] + times[chunk_len])

    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=500.0,
        zscore_threshold=None,
        proportion_above_threshold=1.0,
        removal_window_ms=1.0,
        join_window_ms=1.0,
        min_length_s=0.001,
    )
    valid_times = detect_artifacts(rec, validated)

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
    """``apply_artifact_mask`` zeros artifacts but not the gap boundary.

    The complement across a gap holds no real frames, so the samples on
    either side stay untouched.
    """
    from spyglass.spikesorting.v2._sorting.artifact_mask import (
        apply_artifact_mask,
    )

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
    masked = apply_artifact_mask(recording=rec, valid_times=valid_times)
    mt = masked.get_traces(return_in_uV=False)

    assert np.all(mt[50:60, :] == 0), "artifact frames must be zeroed"
    assert np.all(
        mt[k, :] == 7.0
    ), "chunk-boundary frame must NOT be masked (pure-gap over-mask)"
    assert np.all(mt[k + 1, :] == 8.0), "chunk 2's first frame must survive"


@pytest.mark.usefixtures("dj_conn")
def test_detect_artifacts_removal_window_does_not_spill_across_gap():
    """Chunk 1's removal window must not reach across the gap into chunk 2.

    The window is applied in time, not frames, so a valid interval starts
    at chunk 2's first sample ``times[chunk_len]``.
    """
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts

    rec, traces, fs, chunk_len, gap_s = _disjoint_recording()
    # 3rd-to-last frame of chunk 1; a 15-frame half-window in frame space
    # would reach ~12 frames into chunk 2.
    traces[chunk_len - 3, :] = 5000.0
    times = rec.get_times()
    chunk2_start = times[chunk_len]

    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=500.0,
        zscore_threshold=None,
        proportion_above_threshold=1.0,
        removal_window_ms=1.0,
        join_window_ms=0.0,
        min_length_s=0.001,
    )
    valid_times = detect_artifacts(rec, validated)

    assert any(
        start == chunk2_start and end == times[-1] for start, end in valid_times
    ), (
        "chunk 2's first sample was removed: chunk-1's artifact removal "
        f"window spilled across the gap. valid_times={valid_times.tolist()}, "
        f"expected a chunk-2 interval starting at {chunk2_start}."
    )


@pytest.mark.usefixtures("dj_conn")
def test_detect_artifacts_join_does_not_bridge_gap():
    """Artifacts in different chunks must not join across the gap.

    Frame indices are contiguous across the gap, so a frame-space join
    would merge them and over-mask chunk 1's tail.
    """
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts

    rec, traces, fs, chunk_len, gap_s = _disjoint_recording()
    # 20 frames apart (inside the 30-frame join window) but 0.5 s apart in
    # wall-clock.
    traces[chunk_len - 20, :] = 5000.0
    traces[chunk_len, :] = 5000.0
    times = rec.get_times()

    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=500.0,
        zscore_threshold=None,
        proportion_above_threshold=1.0,
        removal_window_ms=1.0,  # half = 15 frames -> tail beyond 195 free
        join_window_ms=1.0,  # 30 frames -> would bridge the 20-frame gap
        min_length_s=1e-5,
    )
    valid_times = detect_artifacts(rec, validated)

    # Chunk 1's tail beyond the artifact's own window stays valid.
    tail_t = times[chunk_len - 4]
    covered = any(s <= tail_t < e for s, e in valid_times)
    assert covered, (
        f"chunk-1 valid tail near {tail_t} was removed: the frame-space "
        f"join bridged the gap. valid_times={valid_times.tolist()}"
    )
