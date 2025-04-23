import itertools
from functools import reduce
from typing import List, Union

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pynwb import NWBFile

from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.dj_helper_fn import get_child_tables

schema = dj.schema("common_interval")

# TODO: ADD export to NWB function to save relevant intervals in an NWB file


@schema
class IntervalList(SpyglassMixin, dj.Manual):
    definition = """
    # Time intervals used for analysis
    -> Session
    interval_list_name: varchar(170)  # descriptive name of this interval list
    ---
    valid_times: longblob  # numpy array with start/end times for each interval
    pipeline = "": varchar(64)  # type of interval list
    """

    # See #630, #664. Excessive key length.

    @classmethod
    def insert_from_nwbfile(cls, nwbf: NWBFile, *, nwb_file_name: str):
        """Add each entry in the NWB file epochs table to the IntervalList.

        The interval list name for each epoch is set to the first tag for the
        epoch. If the epoch has no tags, then 'interval_x' will be used as the
        interval list name, where x is the index (0-indexed) of the epoch in the
        epochs table. The start time and stop time of the epoch are stored in
        the valid_times field as a numpy array of [start time, stop time] for
        each epoch.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.
        nwb_file_name : str
            The file name of the NWB file, used as a primary key to the Session
            table.
        """
        if nwbf.epochs is None:
            logger.info("No epochs found in NWB file.")
            return

        epochs = nwbf.epochs.to_dataframe()

        # Create a list of dictionaries to insert
        epoch_inserts = epochs.apply(
            lambda epoch_data: {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": (
                    epoch_data.tags[0]
                    if epoch_data.tags
                    else f"interval_{epoch_data.name}"
                ),
                "valid_times": np.asarray(
                    [[epoch_data.start_time, epoch_data.stop_time]]
                ),
            },
            axis=1,
        ).tolist()

        cls.insert(epoch_inserts, skip_duplicates=True)

    def fetch_interval(self):
        """Fetch interval list object for a given key."""
        if not len(self) == 1:
            raise ValueError(f"Expected one row, got {len(self)}")
        return Interval(self.fetch1("valid_times"))

    def plot_intervals(self, figsize=(20, 5), return_fig=False):
        """Plot the intervals in the interval list."""
        interval_list = pd.DataFrame(self)
        fig, ax = plt.subplots(figsize=figsize)
        interval_count = 0
        for row in interval_list.itertuples(index=False):
            for interval in row.valid_times:
                ax.plot(interval, [interval_count, interval_count])
                ax.scatter(
                    interval,
                    [interval_count, interval_count],
                    alpha=0.8,
                    zorder=2,
                )
            interval_count += 1
        ax.set_yticks(np.arange(interval_list.shape[0]))
        ax.set_yticklabels(interval_list.interval_list_name)
        ax.set_xlabel("Time [s]")
        ax.grid(True)
        if return_fig:
            return fig

    def plot_epoch_pos_raw_intervals(self, figsize=(20, 5), return_fig=False):
        """Plot an epoch's position, raw data, and valid times intervals."""
        interval_list = pd.DataFrame(self)
        fig, ax = plt.subplots(figsize=(30, 3))

        raw_data_valid_times = interval_list.loc[
            interval_list.interval_list_name == "raw data valid times"
        ].valid_times
        interval_y = 1

        for interval in np.asarray(raw_data_valid_times)[0]:
            ax.plot(interval, [interval_y, interval_y])
            ax.scatter(interval, [interval_y, interval_y], alpha=0.8, zorder=2)

        epoch_valid_times = (
            interval_list.set_index("interval_list_name")
            .filter(regex=r"^[0-9]", axis=0)
            .valid_times
        )
        interval_y = 2
        for epoch, valid_times in zip(
            epoch_valid_times.index, epoch_valid_times
        ):
            for interval in valid_times:
                ax.plot(interval, [interval_y, interval_y])
                ax.scatter(
                    interval, [interval_y, interval_y], alpha=0.8, zorder=2
                )
                ax.text(
                    interval[0] + np.diff(interval)[0] / 2,
                    interval_y,
                    epoch,
                    ha="center",
                    va="bottom",
                )

        pos_valid_times = (
            interval_list.set_index("interval_list_name")
            .filter(regex=r"^pos \d+ valid times$", axis=0)
            .valid_times
        ).sort_index(key=lambda index: [int(name.split()[1]) for name in index])
        interval_y = 0
        for epoch, valid_times in zip(pos_valid_times.index, pos_valid_times):
            for interval in valid_times:
                ax.plot(interval, [interval_y, interval_y])
                ax.scatter(
                    interval, [interval_y, interval_y], alpha=0.8, zorder=2
                )
                ax.text(
                    interval[0] + np.diff(interval)[0] / 2,
                    interval_y,
                    epoch.replace(" valid times", ""),
                    ha="center",
                    va="bottom",
                )

        ax.set_ylim((-0.25, 2.25))
        ax.set_yticks(np.arange(3))
        ax.set_yticklabels(["pos valid times", "raw data valid times", "epoch"])
        ax.set_xlabel("Time [s]")
        ax.grid(True)
        if return_fig:
            return fig

    def cautious_insert(self, inserts, **kwargs):
        """On existing primary key, check secondary key and update if needed.

        `replace=True` will attempt to delete/replace the existing entry. When
        the row has a foreign key constraint, this will fail. This method will
        check if an update is needed.

        Parameters
        ----------
        inserts : list of dict
            List of dictionaries to insert.
        **kwargs : dict
            Additional keyword arguments to pass to `insert`.
        """
        if not isinstance(inserts, list):
            inserts = [inserts]
        if not isinstance(inserts[0], dict):
            raise ValueError("Input must be a list of dictionaries.")

        pk = self.heading.primary_key

        def pk_match(row):
            match = self & {k: v for k, v in row.items() if k in pk}
            return match.fetch(as_dict=True)[0] if match else None

        def sk_match(new, old):
            return (
                np.array_equal(new["valid_times"], old["valid_times"])
                and new["pipeline"] == old["pipeline"]
            )

        basic_inserts, need_update = [], []
        for row in inserts:
            existing = pk_match(row)
            if not existing:  # if no existing entry, insert
                basic_inserts.append(row)
            elif existing and not sk_match(row, existing):  # diff sk, update
                need_update.append(row)

        self.insert(basic_inserts, **kwargs)
        for row in need_update:
            self.update1(row)

    def cleanup(self, dry_run=True):
        """Clean up orphaned IntervalList entries."""
        orphans = self - get_child_tables(self)
        if dry_run:
            return orphans
        orphans.super_delete(warn=False)


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


def intervals_by_length(interval_list, min_length=0.0, max_length=1e10):
    """Select intervals of certain lengths from an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    min_length : float, optional
        Minimum interval length in seconds. Defaults to 0.0.
    max_length : float, optional
        Maximum interval length in seconds. Defaults to 1e10.
    """
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log("intervals_by_length")

    return Interval(interval_list).by_length(min_length, max_length).times


def interval_list_contains_ind(interval_list, timestamps):
    """Find indices of list of timestamps contained in an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    timestamps : array_like
    """
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log("interval_list_contains_ind")

    return Interval(interval_list).contains(timestamps, as_indices=True).times


def interval_list_contains(interval_list, timestamps):
    """Find timestamps that are contained in an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    timestamps : array_like
    """
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log("interval_list_contains")
    return Interval(interval_list).contains(timestamps).times


def interval_list_excludes_ind(interval_list, timestamps):
    """Find indices of timestamps that are not contained in an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    timestamps : array_like
    """
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log("interval_list_excludes_ind")
    return Interval(interval_list).excludes(timestamps, as_indices=True).times


def interval_list_excludes(interval_list, timestamps):
    """Find timestamps that are not contained in an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    timestamps : array_like
    """
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log("interval_list_excludes")
    return Interval(interval_list).excludes(timestamps).times


def consolidate_intervals(interval_list):
    """Consolidate overlapping intervals in an interval list."""
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log("interval_list_excludes")
    return Interval(interval_list).consolidate().times


def interval_list_intersect(interval_list1, interval_list2, min_length=0):
    """Finds the intersections between two interval lists

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
    return Interval(interval_list1).intersect(interval_list2, min_length).times


def union_adjacent_index(interval1, interval2):
    """Union index-adjacent intervals. If not adjacent, just concatenate.

    e.g. [a,b] and [b+1, c] is converted to [a,c]

    Parameters
    ----------
    interval1 : np.array
    interval2 : np.array
    """
    return Interval(interval1).union_adjacent_index(interval2).times


def interval_list_union(
    interval_list1: np.ndarray,
    interval_list2: np.ndarray,
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
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log("interval_list_union")

    return (
        Interval(interval_list1)
        .union(interval_list2, min_length, max_length)
        .times
    )


def interval_list_censor(interval_list, timestamps):
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
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log("interval_list_censor")
    return Interval(interval_list).censor(timestamps).times


def interval_from_inds(list_frames):
    """Converts a list of indices to a list of intervals.

    e.g. [2,3,4,6,7,8,9,10] -> [[2,4],[6,10]]

    Parameters
    ----------
    list_frames : array_like of int
    """
    return Interval(list_frames, from_inds=True).times


def interval_set_difference_inds(intervals1, intervals2):
    """
    e.g.
    intervals1 = [(0, 5), (8, 10)]
    intervals2 = [(1, 2), (3, 4), (6, 9)]

    result = [(0, 1), (4, 5), (9, 10)]

    Parameters
    ----------
    intervals1 : _type_
        _description_
    intervals2 : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log("interval_set_difference_inds")
    return Interval(intervals1).set_difference_inds(intervals2).times


def interval_list_complement(intervals1, intervals2, min_length=0.0):
    """
    Finds intervals in intervals1 that are not in intervals2

    Parameters
    ----------
    min_length : float, optional
        Minimum interval length in seconds. Defaults to 0.0.
    """
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log("interval_list_complement")
    return Interval(intervals1).complement(intervals2, min_length).times
