import itertools
from functools import reduce
from typing import List, Union

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spyglass.common.common_session import Session  # noqa: F401


class Interval:
    """Class to handle interval lists"""

    def __init__(
        self, interval_list: Union[List[int], dict, np.ndarray], from_inds=False
    ):
        """Initialize the Intervals class with a list of intervals.

        Parameters
        ----------
        interval_list : dict or np.ndarray
            A key of IntervalList table, a numpy array of intervals, or a list
            of indices (integers).
        """
        self.key = None
        self.times = self._extract(interval_list, from_inds=from_inds)

    def __repr__(self):
        joined = "\n\t".join(self.times.astype(str))
        return f"Interval({joined})"

    def __len__(self):
        return len(self.times)

    def __getitem__(self, item):
        """Get item from the interval list."""
        if isinstance(item, int):
            return Interval(self.times[item])
        elif isinstance(item, slice):
            return Interval(self.times[item])
        else:
            raise ValueError(
                f"Unrecognized item type: {type(item)}. Must be int or slice."
            )

    def __iter__(self):
        """Iterate over the intervals in the interval list."""
        for interval in self.times:
            yield Interval(interval)

    def __eq__(self, other):
        """Check if two interval lists are equal."""
        return np.array_equal(self.times, self._extract(other))

    def __lt__(self, other):
        """Check if this interval list is less than another."""
        return np.all(self.times < self._extract(other))

    def __gt__(self, other):
        """Check if this interval list is greater than another."""
        return np.all(self.times > self._extract(other))

    def _extract(self, interval_list, from_inds=False):
        """Extract interval_list from a given object."""
        if from_inds:
            return self.from_inds(interval_list)
        elif interval := getattr(interval_list, "times", None):
            return interval
        elif isinstance(interval_list, dict):
            return self._import_from_table(interval_list)
        elif isinstance(interval_list, (list, np.ndarray)):
            return interval_list
        else:
            raise ValueError(
                f"Unrecognized interval_list type: {type(interval_list)}"
            )

    @staticmethod
    def from_inds(list_frames):
        """Converts a list of indices to a list of intervals.

        e.g. [2,3,4,6,7,8,9,10] -> [[2,4],[6,10]]

        Parameters
        ----------
        list_frames : array_like of int
        """
        list_frames = np.unique(list_frames)
        interval_list = []
        for key, group in itertools.groupby(
            enumerate(list_frames), lambda t: t[1] - t[0]
        ):
            group = list(group)
            interval_list.append([group[0][1], group[-1][1]])
        return np.asarray(interval_list)

    @staticmethod
    def _by_length(x, min_length=0.0, max_length=1e10):
        """Select intervals of certain lengths from an interval list."""
        lengths = np.ravel(np.diff(x))
        return x[np.logical_and(lengths > min_length, lengths < max_length)]

    def by_length(self, min_length=0.0, max_length=1e10):
        """Select intervals of certain lengths from an interval list.

        Parameters
        ----------
        min_length : float, optional
            Minimum interval length in seconds. Defaults to 0.0.
        max_length : float, optional
            Maximum interval length in seconds. Defaults to 1e10.
        """
        return Interval(self._by_length(self.times, min_length, max_length))

    def contains(self, timestamps, as_indices=False, padding: int = None):
        """Find timestamps that are contained in an interval list.

        Parameters
        ----------
        timestamps : array_like
        as_indices : bool, optional
            If True, return indices of timestamps in the interval list.
            If False, return the timestamps themselves. Defaults to False.
        padding : int, optional
            If True, pad the first and last timestamps by val. Defaults to None.
        """
        ind = []
        for interval in self.times:
            ind += np.ravel(
                np.argwhere(
                    np.logical_and(
                        timestamps >= interval[0], timestamps <= interval[1]
                    )
                )
            ).tolist()
        ret = np.asarray(ind) if as_indices else timestamps[ind]

        if padding:
            if ret[0] > 0:
                ret[0] -= padding
            if ret[-1] != len(timestamps) - padding:
                ret[-1] += padding

        return Interval(ret)

    def excludes(self, timestamps, as_indices=False):
        """Find timestamps that are not contained in an interval list.

        Parameters
        ----------
        timestamps : array_like
        """
        contained_times = self.contains(timestamps, as_indices=as_indices).times
        if as_indices:
            timestamps = np.arange(len(timestamps))
        return Interval(np.setdiff1d(timestamps, contained_times))

    @staticmethod
    def _expand_1d(interval_list):
        """Expand a 1D interval list to 2D."""
        if interval_list.ndim == 1:
            return np.expand_dims(interval_list, 0)
        return interval_list

    def _union_consolidate(self, interval_list):
        return Interval(reduce(self._union_concat, interval_list))

    def union_consolidate(self):
        return Interval(self._union_consolidate(self.times))

    def _consolidate(self, interval_list):
        """Consolidate overlapping intervals in an interval list."""
        if interval_list.ndim == 1:
            return self._expand_1d(interval_list)

        interval_list = interval_list[np.argsort(interval_list[:, 0])]
        interval_list = self._union_consolidate(interval_list).times

        # reduce may convert to 1D, so check and expand
        return self._expand_1d(interval_list)

    def consolidate(self):
        return Interval(self._consolidate(self.times))

    @staticmethod
    def _set_intersect(interval1, interval2):
        """Takes the (set-theoretic) intersection of two intervals"""
        start = max(interval1[0], interval2[0])
        end = min(interval1[1], interval2[1])
        return np.array([start, end]) if end > start else None

    def _intersect(self, interval1, interval2, min_length=0):
        """Finds the intersection of two interval lists."""

        # Consolidate interval lists to disjoint by sorting & applying union
        interval_list1 = self._consolidate(interval1)
        interval_list2 = self._consolidate(interval2)

        # then do pairwise comparison and collect intersections
        intersection = [
            self._set_intersect(i2, i1)
            for i2 in interval_list2
            for i1 in interval_list1
            if self._set_intersect(i1, i2) is not None
        ]

        # if no intersection, then return an empty list
        if not intersection:
            return []

        intersection = np.asarray(intersection)
        intersection = intersection[np.argsort(intersection[:, 0])]

        return self._by_length(intersection, min_length=min_length)

    def intersect(self, other, min_length=0):
        """Finds the intersections between this and another interval list.

        Each interval is (start time, stop time)

        Parameters
        ----------
        interval_list1 : np.array, (N,2) where N = number of intervals
        interval_list2 : np.array, (N,2) where N = number of intervals
        min_length : float, optional.
            Minimum length of intervals to include, default 0

        Returns
        -------
        interval_list: np.array, (N,2)
        """
        return Interval(
            self._intersect(self.times, self._extract(other), min_length)
        )

    def _union_concat(self, interval1, interval2):
        """Compare last interval of interval list to given interval.

        If overlapping, take union. If not, concatenate interval to list.
        Recursively called with `reduce`.
        """

        interval1 = self._expand_1d(interval1)
        interval2 = self._expand_1d(interval2)

        end1, start1 = interval1[-1], interval2[0]
        intersect = self._set_intersect(end1, start1)
        set_union = self._expand_1d(
            np.array([end1, start1])
            if intersect is None
            else np.array([min(end1[0], start1[0]), max(end1[1], start1[1])])
        )

        return np.concatenate((interval1[:-1], set_union), axis=0)

    def union_adjacent_index(self, other):
        """Union index-adjacent intervals. If not adjacent, just concatenate.

        e.g. [a,b] and [b+1, c] is converted to [a,c]

        Parameters
        ----------
        interval1 : np.array
        interval2 : np.array
        """
        interval1 = np.atleast_2d(self.times)
        interval2 = np.atleast_2d(self._extract(other))

        if not (
            interval1[-1][1] + 1 == interval2[0][0]
            or interval2[0][1] + 1 == interval1[-1][0]
        ):
            return Interval(np.concatenate((interval1, interval2), axis=0))

        x = np.array(
            [
                [
                    np.min([interval1[-1][0], interval2[0][0]]),
                    np.max([interval1[-1][1], interval2[0][1]]),
                ]
            ]
        )
        return Interval(np.concatenate((interval1[:-1], x), axis=0))

    def union(
        self,
        other: np.ndarray,
        min_length: float = 0.0,
        max_length: float = 1e10,
    ) -> np.ndarray:
        """Finds the union (all times in one or both) for two interval lists

        Parameters
        ----------
        interval_list1 : np.ndarray
            The first interval list [start, stop]
        interval_list2 : np.ndarray
            The second interval list [start, stop]
        min_length : float, optional
            Minimum length of interval for inclusion in output, default 0.0
        max_length : float, optional
            Maximum length of interval for inclusion in output, default 1e10

        Returns
        -------
        np.ndarray
            Array of intervals [start, stop]
        """
        interval_list1 = self.times
        interval_list2 = self._extract(other)

        def _parallel_union(interval_list):
            """Create a parallel list where 1 is start and -1 the end"""
            interval_list = np.ravel(interval_list)
            interval_list_start_end = np.ones(interval_list.shape)
            interval_list_start_end[1::2] = -1
            return interval_list, interval_list_start_end

        il1, il1_start_end = _parallel_union(interval_list1)
        il2, il2_start_end = _parallel_union(interval_list2)

        # Concatenate the two lists so we can resort the intervals and apply the
        # same sorting to the start-end arrays
        combined_intervals = np.concatenate((il1, il2))
        ss = np.concatenate((il1_start_end, il2_start_end))
        sort_ind = np.argsort(combined_intervals)
        combined_intervals = combined_intervals[sort_ind]

        # a cumulative sum of 1 indicates the beginning of a joint interval; a
        # cumulative sum of 0 indicates the end
        union_starts = np.ravel(
            np.array(np.where(np.cumsum(ss[sort_ind]) == 1))
        )
        union_stops = np.ravel(np.array(np.where(np.cumsum(ss[sort_ind]) == 0)))
        union = [
            [combined_intervals[start], combined_intervals[stop]]
            for start, stop in zip(union_starts, union_stops)
        ]

        return Interval(np.asarray(union))

    def censor(self, timestamps):
        """Returns new interval list that starts/ends at first/last timestamp

        Parameters
        ----------
        interval_list : numpy array of intervals [start, stop]
            interval list from IntervalList valid times
        timestamps : numpy array or list

        Returns
        -------
        interval_list (numpy array of intervals [start, stop])
        """
        interval_list = self.times
        # check that all timestamps are in the interval list
        if len(self.contains(timestamps, as_indices=True)) != len(timestamps):
            raise ValueError("Interval_list must contain all timestamps")

        timestamps_interval = np.asarray([[timestamps[0], timestamps[-1]]])
        return Interval(self._intersect(interval_list, timestamps_interval))

    def set_difference_inds(self, other, reverse=False):
        """Find the indices of self not in other

        e.g.
        intervals1 = [(0, 5), (8, 10)]
        intervals2 = [(1, 2), (3, 4), (6, 9)]

        result = [(0, 1), (4, 5), (9, 10)]

        Parameters
        ----------
        other : Union[Intervals, np.ndarray]
        reverse : bool, optional
            If True, reverse the order of the intervals. Defaults to False.

        Returns
        -------
        List[Tuple[int, int]]
        """
        intervals1 = self.times if not reverse else self._extract(other)
        intervals2 = self._extract(other) if not reverse else self.times

        result = []
        i = j = 0
        while i < len(intervals1) and j < len(intervals2):
            if intervals1[i][1] <= intervals2[j][0]:
                result.append(intervals1[i])
                i += 1
            elif intervals2[j][1] <= intervals1[i][0]:
                j += 1
            else:
                if intervals1[i][0] < intervals2[j][0]:
                    result.append((intervals1[i][0], intervals2[j][0]))
                if intervals1[i][1] > intervals2[j][1]:
                    intervals1[i] = (intervals2[j][1], intervals1[i][1])
                    j += 1
                else:
                    i += 1
        result += intervals1[i:]
        return Interval(result)

    def complement(self, other, min_length=0.0):
        """
        Finds intervals in intervals1 that are not in intervals2

        Parameters
        ----------
        min_length : float, optional
            Minimum interval length in seconds. Defaults to 0.0.
        """
        intervals1 = self.times
        intervals2 = self._extract(other)

        result = []

        for start1, end1 in intervals1:
            subtracted = [(start1, end1)]

            for start2, end2 in intervals2:
                new_subtracted = []

                for s, e in subtracted:
                    if start2 <= s and e <= end2:
                        continue
                    if e <= start2 or end2 <= s:
                        new_subtracted.append((s, e))
                        continue
                    if start2 > s:
                        new_subtracted.append((s, start2))
                    if end2 < e:
                        new_subtracted.append((end2, e))

                subtracted = new_subtracted

            result.extend(subtracted)

        return Interval(
            self._by_length(
                np.asarray(result), min_length=min_length, max_length=1e100
            )
        )

    def add_removal_window(self, removal_window_ms, timestamps):
        """Add half of a removal window to start/end of each interval"""
        half_win = removal_window_ms / 1000 * 0.5
        new_interval = np.zeros((len(self.times), 2), dtype=np.float64)
        # for each interval, add half removal window to start and end
        for idx, interval in enumerate(self.times):
            start, end = interval[0], interval[1]
            new_interval[idx] = [
                timestamps[start] - half_win,
                np.minimum(timestamps[end] + half_win, timestamps[-1]),
            ]
        # NOTE: np.min and if len cond were in
        # spikesorting_artifact, but not lfp_artifact
        return (
            self._union_consolidate(new_interval)
            if len(new_interval) > 1
            else Interval(new_interval)
        )

    def to_indices(self, timestamps):
        """Convert intervals to indices in the given timestamps.

        Parameters
        ----------
        timestamps : array_like
            The timestamps to convert the intervals to indices in.

        Returns
        -------
        np.ndarray
            The indices of the intervals in the given timestamps.
        """
        starts = np.searchsorted(timestamps, self.times[:, 0])
        ends = np.searchsorted(timestamps, self.times[:, 1])
        return np.column_stack((starts, ends))
        # # Prev approach:
        # return [
        #     np.searchsorted(timestamps, interval) for interval in self.times
        # ]

    def to_seconds(self, timestamps):
        """Convert intervals to seconds in the given timestamps.

        Parameters
        ----------
        timestamps : array_like
            The timestamps to convert the intervals to seconds in.

        Returns
        -------
        np.ndarray
            The seconds of the intervals in the given timestamps.
        """
        return [(timestamps[i[0]], timestamps[i[1]]) for i in self.times]

    # ---------------------------- Requiring Table ----------------------------

    def _import_from_table(self, interval_key, return_table=False):
        from spyglass.common.common_interval import IntervalList

        query = IntervalList & interval_key
        if return_table:
            return query
        times = query.fetch("valid_times")
        if len(times) == 0:
            raise ValueError(f"No valid times found for {interval_key}")
        if len(times) == 1:
            return times[0]

        # Should be union?
        raise ValueError(
            f"Multiple valid times found for {interval_key}: {times}"
        )

    def _get_table(self):
        if not self.key:
            raise ValueError(
                "Must initialize with a key run table-based operations"
            )
        return self._import_from_table(self.key, return_table=True)
