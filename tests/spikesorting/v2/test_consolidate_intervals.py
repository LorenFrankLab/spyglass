"""Exact frame pairs from ``_consolidate_regular_intervals``.

Intervals in seconds map to half-open ``[start, end)`` frames for
``frame_slice``. No database; a 1 Hz grid makes frame equal time.
"""

from __future__ import annotations

import numpy as np
import pytest


def _consolidate(intervals):
    """Frame bounds on a 10-sample grid where frame == integer time (1 Hz)."""
    from spyglass.spikesorting.v2._recording.restriction import (
        _consolidate_regular_intervals,
    )

    return _consolidate_regular_intervals(
        np.asarray(intervals), n_samples=10, sampling_frequency=1.0, t_start=0.0
    )


def test_consolidate_exclusive_end_includes_final_sample():
    """``[2 s, 5 s]`` -> ``[2, 6)``; the exclusive end keeps the last sample."""
    out = _consolidate([[2.0, 5.0]])
    np.testing.assert_array_equal(out, np.array([[2, 6]]))
    assert out.dtype == np.int64


def test_consolidate_merges_frame_adjacent_intervals():
    """Frame-adjacent intervals merge: ``[2,4] & [5,7] s`` -> ``[2, 8)``.

    ``[2, 5)`` and ``[5, 8)`` merge because ``next_start <= stop``.
    """
    out = _consolidate([[2.0, 4.0], [5.0, 7.0]])
    np.testing.assert_array_equal(out, np.array([[2, 8]]))


def test_consolidate_keeps_disjoint_intervals_separate():
    """A genuine gap stays split: ``[2,3] & [6,8] s`` -> ``[2,4)``,``[6,9)``."""
    out = _consolidate([[2.0, 3.0], [6.0, 8.0]])
    np.testing.assert_array_equal(out, np.array([[2, 4], [6, 9]]))


def test_consolidate_reorders_unsorted_input():
    """Unsorted input is sorted by start before consolidation.

    ``[[6,8],[2,3]]`` yields ``[2,4)`` then ``[6,9)``.
    """
    out = _consolidate([[6.0, 8.0], [2.0, 3.0]])
    np.testing.assert_array_equal(out, np.array([[2, 4], [6, 9]]))


def test_consolidate_single_sample_interval():
    """A zero-width ``[5 s, 5 s]`` interval covers one frame, ``[5, 6)``."""
    out = _consolidate([[5.0, 5.0]])
    np.testing.assert_array_equal(out, np.array([[5, 6]]))


def test_consolidate_combined_reorder_merge_and_split():
    """One call exercising reorder + adjacency-merge + a preserved gap.

    ``[5,7]``, ``[2,4]``, ``[8,8]`` s chain into a single ``[2, 9)``.
    """
    out = _consolidate([[5.0, 7.0], [2.0, 4.0], [8.0, 8.0]])
    np.testing.assert_array_equal(out, np.array([[2, 9]]))


@pytest.mark.parametrize("t_start", [0.0, 3600.0, 1.7e9, 1.7e9 + 0.123456])
def test_regular_restriction_matches_timestamp_search(t_start):
    """On-sample bounds map like ``searchsorted`` on ``t_start + k / fs``.

    At epoch-scale start times, ``(t - t_start) * fs`` is off an integer by
    up to ~4e-3 frames, so a tolerance snap misses the sample line.
    """
    from spyglass.spikesorting.v2._recording.restriction import (
        _consolidate_regular_intervals,
    )

    fs = 30000.0
    n_samples = 2_000_000
    rng = np.random.default_rng(0)
    frames = np.sort(rng.choice(n_samples - 2, size=400, replace=False)) + 1
    timestamps = t_start + np.arange(n_samples, dtype=np.float64) / fs
    # Separate intervals so no two merge; each bound is a sample's timestamp.
    starts, stops = timestamps[frames[0::2]], timestamps[frames[1::2]]
    got = np.concatenate(
        [
            _consolidate_regular_intervals(
                np.array([[a, b]]),
                n_samples=n_samples,
                sampling_frequency=fs,
                t_start=t_start,
            )
            for a, b in zip(starts, stops)
        ]
    )
    expected = np.column_stack(
        (
            np.searchsorted(timestamps, starts, side="left"),
            np.searchsorted(timestamps, stops, side="right"),
        )
    )
    np.testing.assert_array_equal(got, expected)
