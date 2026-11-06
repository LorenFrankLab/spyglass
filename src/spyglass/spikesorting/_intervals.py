"""Interval-set algebra shared by the v1 and v2 pipelines.

Intervals are ``(start, stop)`` rows in seconds or frames. This module depends
only on NumPy: it declares no ``dj.schema`` and imports no schema module, so
database-free code can import it.
"""

from __future__ import annotations

import numpy as np


def merge_sorted_intervals(intervals) -> list[list]:
    """Merge start-sorted ``(start, stop)`` rows that overlap or touch.

    A row whose start is at or before the running stop extends it, so
    touching rows (``start == previous stop``) merge too. Rows are taken as
    given -- nothing is sorted, validated, or dropped, so a caller that must
    discard empty rows filters them first -- and the values are not
    converted, so integer frames stay ints and Python floats stay floats.

    Parameters
    ----------
    intervals : iterable of (start, stop)
        Rows sorted by start, in seconds or frames.

    Returns
    -------
    list[list]
        ``[[start, stop], ...]`` merged rows; ``[]`` for no input rows.
    """
    merged: list[list] = []
    for start, stop in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return merged


def _normalize(intervals, *, merge=True):
    """Sort by start, merge overlapping/adjacent/duplicate rows, drop
    zero-length rows.

    Guards :func:`intersect_intervals` (and every intersection built on
    it) against a caller-supplied interval set
    that is unsorted or carries overlapping/duplicate rows -- notably the
    identical-input fast path below, which would otherwise return duplicate
    rows unchanged when both operands are identically duplicated. With
    ``merge=False`` the rows are only filtered and sorted.
    """
    intervals = np.asarray(intervals, dtype=float).reshape(-1, 2)
    intervals = intervals[intervals[:, 1] > intervals[:, 0]]
    if len(intervals) == 0:
        return intervals
    intervals = intervals[np.argsort(intervals[:, 0], kind="stable")]
    if not merge:
        return intervals
    return np.asarray(
        merge_sorted_intervals(intervals.tolist()), dtype=float
    ).reshape(-1, 2)


def intersect_intervals(left, right, *, merge=True):
    """Intersect two interval sets, omitting zero-length overlaps.

    Each operand is sorted and stripped of zero-length rows first. By
    default its overlapping or touching rows are also merged, so the result
    has no touching rows. With ``merge=False`` touching rows stay separate
    and every shared edge survives in the result (e.g. ``[(0, 10)]`` with
    ``[(0, 5), (5, 10)]`` gives both halves); each operand must then
    already be disjoint.
    """
    left = _normalize(left, merge=merge)
    right = _normalize(right, merge=merge)
    if np.array_equal(left, right):
        return left
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i, 0], right[j, 0])
        stop = min(left[i, 1], right[j, 1])
        if start < stop:
            result.append((start, stop))
        if left[i, 1] < right[j, 1]:
            i += 1
        else:
            j += 1
    return np.asarray(result, dtype=float).reshape(-1, 2)
