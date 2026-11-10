"""Exact frame pairs from ``_consolidate_regular_intervals``.

Intervals in seconds map to half-open ``[start, end)`` frames for
``frame_slice``. No database; a 1 Hz grid makes frame equal time.
"""

from __future__ import annotations

import numpy as np


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
