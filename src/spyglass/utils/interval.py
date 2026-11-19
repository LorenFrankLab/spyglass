"""Interval lists: arrays of [start, stop] rows and the set operations on them.

`Interval` needs no database. Only a key of `IntervalList` passed as input
reads the table, which is imported at that point.
"""

import itertools
from functools import reduce
from typing import (
    TYPE_CHECKING,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

from spyglass.settings import test_mode
from spyglass.utils.logging import logger

if TYPE_CHECKING:
    import datajoint as dj

T = TypeVar("T", bound="Interval")
IntervalLike = Union[T, np.ndarray, List[int], dict]


class Interval:
    """Class to handle interval lists

    NOTE: methods with _-prefix are for internal use comparing two interval
    lists to allow for inverting non-communicative methods (i.e., AxB!=BxA).
    External equivalents (without _-prefix) runs these methods on self vs other.
    """

    # TODO: Class currently isn't aware of whether intervals are indices or
    # timestamps. This should be fixed in the future.

    def __init__(
        self,
        interval_list: IntervalLike = None,
        name=None,
        from_inds=False,
        no_overlap=False,
        no_duplicates=True,
        warn=not test_mode,  # warn by default, unless running pytests
        **kwargs,
    ) -> None:
        """Initialize the Intervals class with a list of intervals.

        Parameters
        ----------
        interval_list : dict or np.ndarray
            A key of IntervalList table, a numpy array of intervals, or a list
            of indices (integers).
        name : str, optional
            Name of the interval list. If not provided, can be taken from
            kwargs["interval_list_name"].
        from_inds : bool, optional
            If True, treat the interval_list as a list of indices and
            convert to intervals. Defaults to False.
        no_overlap : bool, optional
            If True, consolidate overlapping intervals. Defaults to False.
        no_duplicates : bool, optional
            If True, remove duplicate intervals. Defaults to True.
        kwargs : dict, optional
            Additional keyword arguments to pass to the class, including
            "valid_times" and "interval_list_name" for times and name.
        """

        self.kwargs = dict(  # Returned objects will set this behavior
            kwargs,
            no_overlap=no_overlap,
            no_duplicates=no_duplicates,
            warn=False,  # child instances do not warn
        )

        self.key = dict()  # let _extract decide if this is a key
        self.name = name or kwargs.get("interval_list_name")
        self.pipeline = kwargs.get("pipeline")
        self.nwb_file_name = kwargs.get("nwb_file_name")
        self.times = kwargs.get("valid_times") or interval_list
        self.times = self._extract(self.times, from_inds=from_inds)

        if warn or no_duplicates:
            dupes_removed = np.unique(self.times, axis=0)
        if no_duplicates:
            self.times = dupes_removed
        if warn and not np.array_equal(self.times, dupes_removed):
            logger.warning(
                "Found duplicate(s). Use no_duplicates flag to remove or not: "
                f"{self.to_str(self.times)} vs {self.to_str(dupes_removed)}"
            )

        if warn or no_overlap:  # otherwise recursive _consolidate call
            overlap_removed = self._consolidate(self.times)
        if no_overlap:
            self.times = overlap_removed
        if warn and not np.array_equal(self.times, overlap_removed):
            logger.warning(
                "Found overlap(s). Use no_overlap flag to consolidate or not: "
                f"{self.to_str(self.times)} vs {self.to_str(overlap_removed)}"
            )

    def to_str(self, array: np.ndarray, remove_breaks=True) -> str:
        """Convert the interval list to a string."""
        kwarg = dict(separator=", ") if remove_breaks else dict()
        link = " " if remove_breaks else "\n\t"
        if isinstance(array, Iterable):
            return link.join(
                [np.array2string(i, **kwarg) for i in np.array(array)]
            )
        return str(array)

    def __repr__(self) -> str:
        return f"Interval({self.to_str(self.times, remove_breaks=False)})"

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, item: Union[int, slice, tuple]) -> np.ndarray:
        """Get item from the interval list."""
        if isinstance(item, (slice, int, tuple)):
            return self.times[item]
        else:
            raise ValueError(
                f"Unrecognized item type: {type(item)}. Must be int, slice, or tuple."
            )

    def __iter__(self) -> iter:
        """Iterate over the intervals in the interval list."""
        for interval in self.times:
            yield interval

    def __hash__(self) -> int:  # required for dict dunder
        """Hash the interval list for use in sets or dictionaries."""
        times = self.times
        if hasattr(times, "tolist"):
            times = times.tolist()
        if not isinstance(times, Iterable):
            times = [times]
        return hash(f"{self.name}{tuple(times)}")

    def set_key(self, **kwargs: dict) -> None:
        """Set the key for the interval list."""
        self.key = kwargs
        self.nwb_file_name = kwargs.get("nwb_file_name") or kwargs.get("nwb")
        self.name = kwargs.get("interval_list_name") or kwargs.get("name")
        self.pipeline = kwargs.get("pipeline")

    @property
    def as_dict(self) -> dict:
        """Convert the interval list to a dictionary."""
        ret = {
            "nwb_file_name": self.nwb_file_name,
            "interval_list_name": self.name,
            "valid_times": self.times,
            "pipeline": self.pipeline,
        }
        return {k: v for k, v in ret.items() if v is not None}

    @property
    def primary_key(self):
        if not (self.name and self.nwb_file_name):
            raise ValueError("Both name and file name required for primary key")
        return {
            "interval_list_name": self.name,
            "nwb_file_name": self.nwb_file_name,
        }

    def __eq__(self, other: IntervalLike) -> bool:
        """Check if two interval lists are equal."""
        return np.array_equal(self.times, self._extract(other))

    def _extract(
        self, interval_list: IntervalLike, from_inds: bool = False
    ) -> np.ndarray:
        times = None

        # extract times from interval_list based on type
        if from_inds:
            times = self.from_inds(interval_list)
        elif hasattr(interval_list, "times"):
            times = interval_list.times
        elif isinstance(interval_list, dict):
            times = self._import_from_table(interval_list)
        elif isinstance(
            interval_list, (np.generic, np.ndarray, list, int, float, tuple)
        ):
            times = interval_list
        elif interval_list is None:
            return np.array([])

        # validate times format
        if times is None:
            raise TypeError(
                f"Unrecognized interval_list type: {type(interval_list)}"
            )

        times = self._expand_1d(np.asarray(times))
        if len(times) and not np.all(np.diff(times, axis=1) >= 0):
            raise ValueError(
                "All intervals must be in the form [start, stop] with start <= stop."
            )
        if len(times) and times.shape[1] != 2:
            raise ValueError(
                f"Intervals must have shape (N, 2). Got shape {times.shape}."
            )

        return np.asarray(times)

    @staticmethod
    def from_inds(list_frames) -> List[List[int]]:
        """Converts a list of indices to a list of intervals.

        To use, initialize the Interval class with from_inds=True.

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
    def _by_length(
        x, min_length: Optional[float] = 0.0, max_length: Optional[float] = 1e10
    ) -> List[float]:
        """Select intervals of certain lengths from an interval list."""
        min_length = 0 if min_length is None else min_length
        max_length = 1e10 if max_length is None else max_length
        lengths = np.ravel(np.diff(x))
        return x[np.logical_and(lengths > min_length, lengths < max_length)]

    def by_length(
        self,
        min_length: Optional[float] = 0.0,
        max_length: Optional[float] = 1e10,
    ) -> T:
        """Select intervals of certain lengths from an interval list.

        Parameters
        ----------
        min_length : float, optional
            Minimum interval length in seconds. Defaults to 0.0.
        max_length : float, optional
            Maximum interval length in seconds. Defaults to 1e10.
        """
        return Interval(
            self._by_length(self.times, min_length, max_length), **self.kwargs
        )

    def contains(
        self,
        timestamps: np.ndarray,
        as_indices: Optional[bool] = False,
        padding: Optional[int] = None,
        *,
        half_open_mask: bool = False,
    ) -> T:
        """Find timestamps that are contained in an interval list.

        Parameters
        ----------
        timestamps : array_like
        as_indices : bool, optional
            If True, return indices of timestamps in the interval list.
            If False, return the timestamps themselves. Defaults to False.
        padding : int, optional
            If True, pad the first and last timestamps by val. Defaults to None.
        half_open_mask : bool, optional
            If True, return a boolean mask shaped like `timestamps`, True where
            a timestamp is in some half-open [start, stop) interval;
            `as_indices` and `padding` are ignored. Defaults to False.
        """
        if half_open_mask:
            timestamps = np.asarray(timestamps)
            rows = self._consolidate(self.times, merge_touching=True)
            if not len(rows):
                return np.zeros(timestamps.shape, dtype=bool)
            index = np.searchsorted(rows[:, 0], timestamps, side="right") - 1
            return (index >= 0) & (timestamps < rows[np.maximum(index, 0), 1])

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

        return ret

    def excludes(
        self, timestamps: np.ndarray, as_indices: Optional[bool] = False
    ) -> T:
        """Find timestamps that are not contained in an interval list.

        Parameters
        ----------
        timestamps : array_like
        """
        contained_times = self.contains(timestamps, as_indices=as_indices)
        if as_indices:
            timestamps = np.arange(len(timestamps))
        return np.setdiff1d(timestamps, contained_times)

    @staticmethod
    def _expand_1d(interval_list: np.ndarray) -> np.ndarray:
        """Expand a 1D interval list to 2D."""
        if interval_list.ndim == 1 and interval_list.size > 0:
            return np.expand_dims(interval_list, 0)
        return interval_list

    def _union_consolidate(self, interval_list: IntervalLike) -> T:
        kwargs = dict(self.kwargs, no_overlap=False)
        return Interval(reduce(self._union_concat, interval_list), **kwargs)

    def union_consolidate(self) -> T:
        return Interval(self._union_consolidate(self.times), **self.kwargs)

    def _consolidate(
        self, interval_list: IntervalLike, merge_touching: bool = False
    ) -> T:
        """Consolidate overlapping intervals in an interval list."""
        if not isinstance(interval_list, np.ndarray):
            interval_list = np.asarray(interval_list)
        if interval_list.ndim == 1:
            return self._expand_1d(interval_list)
        if merge_touching:
            return self._merge_touching(interval_list)

        interval_list = interval_list[np.argsort(interval_list[:, 0])]
        interval_list = self._union_consolidate(interval_list).times

        # reduce may convert to 1D, so check and expand
        return self._expand_1d(interval_list)

    @staticmethod
    def _merge_touching(interval_list: np.ndarray) -> np.ndarray:
        """Sort (n, 2) rows by start and merge rows that overlap or touch."""
        if not len(interval_list):
            return interval_list
        rows = interval_list[np.argsort(interval_list[:, 0], kind="stable")]
        # A row opens a new interval only if it starts after every earlier stop
        reach = np.maximum.accumulate(rows[:, 1])
        first = np.flatnonzero(np.r_[True, rows[1:, 0] > reach[:-1]])
        last = np.r_[first[1:] - 1, len(rows) - 1]
        return np.column_stack((rows[first, 0], reach[last]))

    def consolidate(self, *, merge_touching: bool = False) -> T:
        """Merge overlapping intervals.

        Parameters
        ----------
        merge_touching : bool, optional
            If True, also merge intervals that share an endpoint, so no two
            output intervals touch. Defaults to False.
        """
        return Interval(
            self._consolidate(self.times, merge_touching), **self.kwargs
        )

    @staticmethod
    def _set_intersect(
        interval1: IntervalLike, interval2: IntervalLike
    ) -> Union[np.ndarray, None]:
        """Takes the (set-theoretic) intersection of two intervals"""
        start = max(interval1[0], interval2[0])
        end = min(interval1[1], interval2[1])
        return np.array([start, end]) if end > start else None

    def _intersect(
        self,
        interval1: IntervalLike,
        interval2: IntervalLike,
        min_length: Optional[float] = 0,
        merge_touching: bool = False,
    ) -> T:
        """Finds the intersection of two interval lists."""

        # Consolidate interval lists to disjoint by sorting & applying union
        interval_list1 = self._consolidate(interval1, merge_touching)
        interval_list2 = self._consolidate(interval2, merge_touching)

        if self._is_disjoint(interval_list1) and self._is_disjoint(
            interval_list2
        ):
            intersection = self._sweep_intersect(interval_list1, interval_list2)
        else:  # pairwise comparison to collect intersections
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

    @staticmethod
    def _is_disjoint(interval_list: np.ndarray) -> bool:
        """Whether each row starts at or after the previous row's stop."""
        return len(interval_list) < 2 or bool(
            np.all(interval_list[1:, 0] >= interval_list[:-1, 1])
        )

    @staticmethod
    def _sweep_intersect(
        interval_list1: np.ndarray, interval_list2: np.ndarray
    ) -> List[List[float]]:
        """Positive-length overlaps of two sorted, disjoint interval lists.

        Gives the same overlaps as comparing every pair, in one pass.
        """
        intersection, i, j = [], 0, 0
        while i < len(interval_list1) and j < len(interval_list2):
            start = max(interval_list1[i][0], interval_list2[j][0])
            stop = min(interval_list1[i][1], interval_list2[j][1])
            if start < stop:
                intersection.append([start, stop])
            if interval_list1[i][1] < interval_list2[j][1]:
                i += 1
            else:
                j += 1
        return intersection

    def intersect(
        self,
        other: IntervalLike,
        min_length: Optional[float] = 0,
        *,
        merge_touching: bool = False,
    ) -> T:
        """Finds the intersections between this and another interval list.

        Each interval is (start time, stop time)

        Parameters
        ----------
        other : Union[Interval, np.array, list, dict]
            Interval list to intersect with self.
        min_length : float, optional.
            Minimum length of intervals to include, default 0
        merge_touching : bool, optional
            If True, first merge the intervals of each list that overlap or
            touch, so touching intervals come out as one. Defaults to False.

        Returns
        -------
        interval_list: np.array, (N,2)
        """
        return Interval(
            self._intersect(
                self.times, self._extract(other), min_length, merge_touching
            ),
            **self.kwargs,
        )

    def _union_concat(
        self, interval1: IntervalLike, interval2: IntervalLike
    ) -> np.ndarray:
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

    def union_adjacent_index(self, other: IntervalLike) -> T:
        """Union index-adjacent intervals. If not adjacent, just concatenate.

        e.g. [a,b] and [b+1, c] is converted to [a,c]

        Parameters
        ----------
        other : Union[Interval, np.ndarray]
            Interval list to union with self.
        """
        interval1 = np.atleast_2d(self.times)
        interval2 = np.atleast_2d(self._extract(other))

        if not (
            interval1[-1][1] + 1 == interval2[0][0]
            or interval2[0][1] + 1 == interval1[-1][0]
        ):
            return Interval(
                np.concatenate((interval1, interval2), axis=0), **self.kwargs
            )

        x = np.array(
            [
                [
                    np.min([interval1[-1][0], interval2[0][0]]),
                    np.max([interval1[-1][1], interval2[0][1]]),
                ]
            ]
        )
        return Interval(
            np.concatenate((interval1[:-1], x), axis=0), **self.kwargs
        )

    def union_adjacent_consolidate(self) -> T:
        times = [self.times] if len(self.times) > 1 else self.times
        return Interval(
            self._expand_1d(reduce(self.union_adjacent_index, times)),
            **self.kwargs,
        )

    def union(
        self,
        other: IntervalLike,
        min_length: Optional[float] = 0.0,
        max_length: Optional[float] = 1e10,
    ) -> T:
        """Finds the union (all times in one or both) for two interval lists

        Parameters
        ----------
        other : Union[Interval, np.ndarray]
            Interval list to union with self.
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
        if len(combined_intervals) == 0:
            return Interval(np.array([]), **self.kwargs)
        ss = np.concatenate((il1_start_end, il2_start_end))
        sort_ind = np.lexsort((-1 * ss, combined_intervals))
        combined_intervals = combined_intervals[sort_ind]
        ss_cumsum = np.cumsum(ss[sort_ind])
        if np.any(ss_cumsum < 0):
            raise ValueError(
                "Negative cumulative sum found in union. "
                + "This indicates an error in the interval lists, "
                + "such as an end time before a start time. "
                + "Please check the input interval lists for validity."
            )

        # a switch of cumulative sum from 0 to 1 indicates the beginning of a
        # joint interval; a cumulative sum of 0 indicates the end
        cumsum_flip = np.logical_and(
            ss_cumsum[1:] == 1,
            ss_cumsum[:-1] == 0,
        )
        union_starts = np.ravel(np.array(np.where(cumsum_flip)[0] + 1))
        union_starts = (
            np.insert(union_starts, 0, 0)
            if ss[sort_ind][0] == 1
            else union_starts
        )
        union_stops = np.ravel(np.array(np.where(ss_cumsum == 0)))
        if union_starts.size != union_stops.size:
            raise ValueError(
                "Mismatched number of union starts and stops. "
                + "This indicates an error in the interval lists, "
                + "such as an end time before a start time. "
                + "Please check the input interval lists for validity."
            )
        union = [
            [combined_intervals[start], combined_intervals[stop]]
            for start, stop in zip(union_starts, union_stops)
        ]

        return Interval(np.asarray(union), **self.kwargs)

    def censor(self, timestamps: Union[np.ndarray, List[int]]) -> T:
        """Returns new interval list that starts/ends at first/last timestamp

        Parameters
        ----------
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
        return Interval(
            self._intersect(interval_list, timestamps_interval), **self.kwargs
        )

    def subtract(
        self,
        other: IntervalLike,
        reverse: Optional[bool] = False,
        min_length: Optional[float] = None,
        max_length: Optional[float] = None,
    ) -> T:
        """Finds intervals in self that are not in other.

        Works on indices:
            intervals1 = [(0, 5), (8, 10)]
            intervals2 = [(1, 2), (3, 4), (6, 9)]
            result = [(0, 1), (2, 3), (4, 5), (9, 10)]

        Parameters
        ----------
        other : Union[Interval, np.ndarray]
        reverse : bool, optional
            If True, reverse the order of the intervals. Defaults to False.
        min_length : float, optional
            Minimum interval length in seconds. If None, no filtering applied.
        max_length : float, optional
            Maximum interval length in seconds. If None, no filtering applied.

        Returns
        -------
        Interval
            The resulting intervals after the difference operation.
            If indices, List[Tuple[int, int]].
        """
        intervals1 = self.times if not reverse else self._extract(other)
        intervals2 = self._extract(other) if not reverse else self.times

        result = []
        i, j = 0, 0
        while i < len(intervals1):
            start1, end1 = intervals1[i]
            while j < len(intervals2) and intervals2[j][1] <= start1:
                j += 1  # Skip intervals2 that end before start1
            current_start = start1
            # Scan with k, leaving j in place: an interval in intervals2 can
            # also overlap the next interval in intervals1
            k = j
            while k < len(intervals2) and intervals2[k][0] < end1:
                start2, end2 = intervals2[k]
                # Add the non-overlap before the current interval in intervals2
                if current_start < start2:
                    result.append((current_start, min(start2, end1)))
                # Update the current start to exclude the overlapping part
                current_start = max(current_start, end2)
                k += 1  # next interval in intervals2
            if current_start < end1:  # if non-overlap btwn last part and end1
                result.append((current_start, end1))
            i += 1  # next interval in intervals1

        if min_length is not None or max_length is not None:
            result = self._by_length(
                np.asarray(result), min_length=min_length, max_length=max_length
            )

        return Interval(result, **self.kwargs)

    def add_removal_window(
        self, removal_window_ms: int, timestamps: np.ndarray
    ) -> T:
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
            else Interval(new_interval, **self.kwargs)
        )

    def to_indices(
        self, timestamps: np.ndarray, as_interval: bool = False
    ) -> Union[List[List[int]], T]:
        """Convert intervals to indices in the given timestamps.

        Parameters
        ----------
        timestamps : array_like
            The timestamps to convert the intervals to indices in.
        as_interval : bool, optional
            If True, return as an Interval object. Defaults to False.

        Returns
        -------
        np.ndarray
            The indices of the intervals in the given timestamps.
        """
        ret = np.searchsorted(timestamps, self.times.ravel()).reshape(-1, 2)
        if as_interval:
            if len(ret) == 0:
                return Interval([], **self.kwargs)
            if not hasattr(ret[0], "__len__"):
                # handle list of ints case from v0.spikesorting_artifact
                ret = [ret]
            ret = Interval(ret, **self.kwargs)
        return ret

    def to_seconds(self, timestamps: np.ndarray) -> List[Tuple[float]]:
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

    def _import_from_table(
        self, interval_key: dict, return_table: Optional[bool] = False
    ) -> Union["dj.expression.QueryExpression", np.ndarray]:
        # Imported here: loading the table module connects to the database.
        from spyglass.common.common_interval import IntervalList

        self.key = interval_key
        query = IntervalList & {
            k: v
            for k, v in interval_key.items()
            if k in IntervalList.primary_key
        }

        if return_table:
            return query
        count = len(query)
        if count != 1:  # Should be union of existing intervals?
            raise ValueError(
                f"Found {count} interval entries found for {interval_key}"
            )
        self.name, self.pipeline, self.nwb_file_name, times = query.fetch1(
            "interval_list_name", "pipeline", "nwb_file_name", "valid_times"
        )
        return times
