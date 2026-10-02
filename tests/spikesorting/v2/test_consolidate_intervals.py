"""Hermetic exact-frame tests for ``_consolidate_regular_intervals``.

``_recording_restriction._consolidate_regular_intervals`` maps
``(start_seconds, stop_seconds)`` intervals onto half-open
``[start_frame, end_frame_exclusive)`` index pairs suitable for
``recording.frame_slice``. v1's helper computed an INCLUSIVE end and fed it to
``frame_slice(end_frame=...)`` whose ``end_frame`` is EXCLUSIVE, silently
dropping the last sample of every interval; v2's end is the true exclusive
bound. These tests pin the exact integer pairs (exclusive end,
adjacency-merge, defensive reorder, single-sample interval), which the
integration-level ±5%/±1500-sample check in
``single_session/test_disjoint_intervals.py::test_disjoint_sort_intervals_concatenated``
cannot resolve.

Pure / synthetic -- no database. A 10-sample grid at 1 Hz starting at 0 s is
used so the frame index equals the integer time, making every expected frame
hand-verifiable.
"""

from __future__ import annotations

import numpy as np


def _consolidate(intervals):
    """Frame bounds on a 10-sample grid where frame == integer time (1 Hz)."""
    from spyglass.spikesorting.v2._recording_restriction import (
        _consolidate_regular_intervals,
    )

    return _consolidate_regular_intervals(
        np.asarray(intervals), n_samples=10, sampling_frequency=1.0, t_start=0.0
    )


def test_consolidate_exclusive_end_includes_final_sample():
    """``[2 s, 5 s]`` -> ``[2, 6)`` -- the final sample is kept (v1 dropped it).

    The end frame is ``floor(5) + 1 == 6``, the true half-open end. v1
    subtracted 1 here and lost the sample at t = 5 s (frame 5). The exclusive
    end MUST be 6, not 5.
    """
    out = _consolidate([[2.0, 5.0]])
    np.testing.assert_array_equal(out, np.array([[2, 6]]))
    assert out.dtype == np.int64


def test_consolidate_merges_frame_adjacent_intervals():
    """Frame-adjacent intervals merge: ``[2,4] & [5,7] s`` -> ``[2, 8)``.

    First interval -> ``[2, 5)``; second -> ``[5, 8)``. The exclusive-end
    adjacency test ``next_start <= stop`` (``5 <= 5``) merges them into a
    single ``[2, 8)`` span. (v1's inclusive-end form used
    ``stop >= next_start - 1`` for the same effect.)
    """
    out = _consolidate([[2.0, 4.0], [5.0, 7.0]])
    np.testing.assert_array_equal(out, np.array([[2, 8]]))


def test_consolidate_keeps_disjoint_intervals_separate():
    """A genuine gap stays split: ``[2,3] & [6,8] s`` -> ``[2,4)``,``[6,9)``.

    First -> ``[2, 4)``; second start frame 6 is past the first's exclusive
    end 4 (``6 <= 4`` is False), so they remain two rows.
    """
    out = _consolidate([[2.0, 3.0], [6.0, 8.0]])
    np.testing.assert_array_equal(out, np.array([[2, 4], [6, 9]]))


def test_consolidate_reorders_unsorted_input():
    """Unsorted input is sorted by start before consolidation.

    ``[[6,8],[2,3]]`` must yield the same result as the sorted
    ``[[2,3],[6,8]]`` -- ``[2,4)`` then ``[6,9)`` -- not a spurious merge or
    a reversed-order output.
    """
    out = _consolidate([[6.0, 8.0], [2.0, 3.0]])
    np.testing.assert_array_equal(out, np.array([[2, 4], [6, 9]]))


def test_consolidate_single_sample_interval():
    """A zero-width ``[5 s, 5 s]`` interval covers exactly one frame ``[5, 6)``.

    ``start = ceil(5) = 5``; ``stop = floor(5) + 1 = 6``. One sample
    (frame 5), not zero and not two.
    """
    out = _consolidate([[5.0, 5.0]])
    np.testing.assert_array_equal(out, np.array([[5, 6]]))


def test_consolidate_combined_reorder_merge_and_split():
    """One call exercising reorder + adjacency-merge + a preserved gap.

    Input (unsorted): ``[5,7]``, ``[2,4]``, ``[8,8]`` s on the 10-sample
    grid. Sorted: ``[2,4]`` -> ``[2,5)``, ``[5,7]`` -> ``[5,8)`` (merges
    with the first into ``[2, 8)``), ``[8,8]`` -> ``[8, 9)`` (frame-adjacent
    to 8, so it merges too) giving a single ``[2, 9)``. Pins that adjacency
    chains correctly rather than leaving fragments.
    """
    out = _consolidate([[5.0, 7.0], [2.0, 4.0], [8.0, 8.0]])
    np.testing.assert_array_equal(out, np.array([[2, 9]]))
