"""Full-timestamp-vector reference implementations for the interval tests.

Production maps intervals to frames and finds wall-clock gaps without
materializing the recording's timestamp vector
(``_recording_restriction._consolidate_regular_intervals``,
``_signal_math.base_intervals_and_gaps``). These references compute the same
quantities directly from a whole ``(n_samples,)`` timestamp array with
``searchsorted`` / ``np.diff``, so tests can check the bounded-memory paths
against them. DB-free.
"""

from __future__ import annotations

import numpy as np

from spyglass.spikesorting.v2._core.signal_math import (
    assert_monotonic_timestamps,
    assert_positive_sampling_frequency,
)


def consolidate_intervals(intervals, timestamps):
    """Convert ``(start_time, stop_time)`` second intervals to frame indices.

    Sorts and merges overlapping/adjacent intervals, then maps each onto a
    half-open ``[start_frame, end_frame_exclusive)`` pair. The end uses
    ``searchsorted(side="right")`` -- the count of timestamps ``<= stop_time``
    -- which is exactly the exclusive end ``frame_slice`` expects, so the
    final sample of each interval is retained.

    Parameters
    ----------
    intervals : array-like
        Iterable of ``(start_seconds, stop_seconds)`` tuples.
    timestamps : numpy.ndarray
        Monotonically non-decreasing wall-clock timestamps, ``(n_samples,)``.

    Returns
    -------
    numpy.ndarray
        ``(n_consolidated, 2)`` int64 ``(start_frame, end_frame_exclusive)``.
    """
    intervals = np.asarray(intervals)
    if intervals.ndim == 1:
        intervals = intervals.reshape(-1, 2)
    if intervals.shape[1] != 2:
        raise ValueError("Input array must have shape (N_Intervals, 2).")

    # Sort defensively; stable ordering by start.
    if not np.all(intervals[:-1] <= intervals[1:]):
        intervals = intervals[np.argsort(intervals[:, 0])]

    assert_monotonic_timestamps(timestamps, context="consolidate_intervals: ")
    start_indices = np.searchsorted(timestamps, intervals[:, 0], side="left")
    # Exclusive end: ``side="right"`` returns the count of timestamps <= value,
    # which is exactly the half-open end ``frame_slice`` expects.
    stop_indices = np.searchsorted(timestamps, intervals[:, 1], side="right")

    consolidated = []
    start, stop = int(start_indices[0]), int(stop_indices[0])
    for next_start, next_stop in zip(start_indices, stop_indices):
        next_start = int(next_start)
        next_stop = int(next_stop)
        # Overlap / adjacency in exclusive-end form: next_start <= stop
        # (== means strictly adjacent).
        if next_start <= stop:
            stop = max(stop, next_stop)
        else:
            consolidated.append((start, stop))
            start, stop = next_start, next_stop

    consolidated.append((start, stop))
    return np.asarray(consolidated, dtype=np.int64)


def base_intervals_from_timestamps(timestamps, fs):
    """Split a (possibly gap-preserving) timestamp vector into recorded chunks.

    Consecutive samples within a chunk differ by ~1 sample period; a diff
    greater than 1.5 sample periods (a missing sample) is a wall-clock gap.
    Returns one ``[start, end]`` (inclusive first/last sample times, seconds)
    per chunk; a contiguous vector yields a single ``[t0, t_end]``.

    Parameters
    ----------
    timestamps : array-like, shape (n_samples,)
        Recording timestamps in seconds, monotonically increasing.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    list[list[float]]
        ``[[start, end], ...]`` inclusive per-chunk bounds; ``[]`` for an
        empty input.
    """
    ts = np.asarray(timestamps, dtype=float)
    if ts.size == 0:
        return []
    assert_monotonic_timestamps(ts, context="base_intervals_from_timestamps: ")
    fs = assert_positive_sampling_frequency(
        fs, context="base_intervals_from_timestamps: "
    )
    sample_period = 1.0 / float(fs)
    gap_after = np.flatnonzero(np.diff(ts) > 1.5 * sample_period)
    starts = [0, *(int(i) + 1 for i in gap_after)]
    ends = [*(int(i) for i in gap_after), ts.size - 1]
    return [[float(ts[s]), float(ts[e])] for s, e in zip(starts, ends)]
