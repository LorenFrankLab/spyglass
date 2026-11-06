"""Interval-set algebra shared by the v1 and v2 pipelines.

Intervals are ``(start, stop)`` rows in seconds or frames. NumPy only, no
schema imports.
"""

from __future__ import annotations

import numpy as np


def merge_sorted_intervals(intervals) -> list[list]:
    """Merge start-sorted ``(start, stop)`` rows that overlap or touch.

    Rows are not sorted, validated, filtered, or type-converted.

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
    """Drop zero-length rows, sort by start, and merge overlaps.

    With ``merge=False`` the rows are only filtered and sorted.
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

    Each operand is normalized first (see :func:`_normalize`). With
    ``merge=False`` touching rows stay separate (``[(0, 10)]`` with
    ``[(0, 5), (5, 10)]`` gives both halves); operands must then be disjoint.
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
