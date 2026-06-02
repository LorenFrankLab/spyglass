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
        assert not (start < ts[99] + gap / 2 < end), (
            "a base interval must not span the inter-chunk gap"
        )


def test_base_intervals_from_timestamps_empty():
    """An empty timestamp vector yields no base intervals."""
    from spyglass.spikesorting.v2.utils import _base_intervals_from_timestamps

    assert _base_intervals_from_timestamps(np.asarray([]), 30000.0) == []


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
    assert any(start >= times[chunk_len] for start, end in valid_times), (
        "expected a valid interval inside chunk 2 (after the gap)"
    )


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
    assert np.all(mt[k, :] == 7.0), (
        "chunk-boundary frame must NOT be masked (pure-gap over-mask)"
    )
    assert np.all(mt[k + 1, :] == 8.0), "chunk 2's first frame must survive"
