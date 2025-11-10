import itertools
import warnings
from functools import reduce
from typing import Iterable, List, Optional, Tuple, TypeVar, Union

import datajoint as dj
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb

from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils import SpyglassIngestion, logger
from spyglass.utils.dj_helper_fn import get_child_tables

schema = dj.schema("common_interval")

# TODO: ADD export to NWB function to save relevant intervals in an NWB file


@schema
class IntervalList(SpyglassIngestion, dj.Manual):
    definition = """
    # Time intervals used for analysis
    -> Session
    interval_list_name: varchar(170)  # descriptive name of this interval list
    ---
    valid_times: longblob  # numpy array with start/end times for each interval
    pipeline = "": varchar(64)  # type of interval list
    """

    # See #630, #664. Excessive key length.

    @property
    def _source_nwb_object_type(self):
        return pynwb.epoch.TimeIntervals

    @property
    def table_key_to_obj_attr(self):
        return {
            "self": {
                "interval_list_name": self.interval_name_from_tags,
                "valid_times": self.interval_from_start_stop_time,
            }
        }

    @staticmethod
    def interval_name_from_tags(epoch_row):
        """Extract interval name from tags attribute.

        This function handles both:
        1. NWB objects (pynwb.epoch.TimeIntervals rows) with .tags attribute
        2. Pandas namedtuples from DataFrame.itertuples()

        SpyglassIngestion converts table-like NWB objects to DataFrames and
        iterates using .itertuples(), which produces namedtuples.
        """
        tags = getattr(epoch_row, "tags", None)

        # For namedtuples from itertuples(), the index is stored as 'Index'
        if hasattr(epoch_row, "Index"):
            name = epoch_row.Index
        else:
            name = getattr(epoch_row, "name", None)

        # Handle formats: list, tuple, numpy array, single value, or None
        if isinstance(tags, (list, tuple, np.ndarray)):
            return tags[0] if len(tags) > 0 else f"interval_{name}"
        elif tags:  # Single value (string or other scalar)
            return tags
        else:
            return f"interval_{name}"

    @staticmethod
    def interval_from_start_stop_time(epoch_row):
        """Extract start and stop times from epoch row.

        This function handles both:
        1. NWB objects with .start_time and .stop_time attributes
        2. Pandas namedtuples from DataFrame.itertuples() (new ingestion pattern)

        Returns
        -------
        np.ndarray
            Array of shape (1, 2) containing [start_time, stop_time]
        """
        start_time = getattr(epoch_row, "start_time", None)
        stop_time = getattr(epoch_row, "stop_time", None)
        return np.asarray([[start_time, stop_time]])

    def fetch_interval(self):
        """Fetch interval list object for a given key."""
        if not len(self) == 1:
            raise ValueError(f"Expected one row, got {len(self)}")
        return Interval(self.fetch1())

    def plot_intervals(
        self, start_time: float = 0, return_fig: bool = False
    ) -> Optional[plt.Figure]:
        """
        Plot all intervals in the given IntervalList table.

        Parameters
        ----------
        start_time : float, optional
            The reference time (in seconds) for the interval comparison plot.
            For example, the first timepoint of a session. Defaults to 0.
        return_fig : bool, optional
            If True, return the matplotlib Figure object. Defaults to False.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            The matplotlib Figure object if `return_fig` is True, otherwise None.

        Raises
        ------
        ValueError
            If more than one unique `nwb_file_name` is found in the IntervalList.
            The intended use is to compare intervals within a single NWB file.
        UserWarning
            If more than 100 intervals are being plotted.
        """
        interval_lists_df = pd.DataFrame(self)

        if len(interval_lists_df["nwb_file_name"].unique()) > 1:
            raise ValueError(
                ">1 nwb_file_name found in IntervalList. the intended use of plot_intervals is to compare intervals within a single nwb_file_name."
            )

        interval_list_names = interval_lists_df["interval_list_name"].values

        n_compare = len(interval_list_names)

        if n_compare > 100:
            warnings.warn(
                f"plot_intervals is plotting {n_compare} intervals. if this is unintended, please pass in a smaller IntervalList.",
                UserWarning,
            )

        # plot broken bar horizontals
        fig, ax = plt.subplots(figsize=(20, 2 / 3 * n_compare))

        # get n colors
        cmap = plt.get_cmap("turbo", n_compare)
        custom_palette = [mpl.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

        def convert_intervals_to_range(intervals, start_time):
            return [
                ((i[0] - start_time) / 60, (i[1] - i[0]) / 60)
                for i in intervals
            ]  # return time in minutes

        all_intervals = interval_lists_df["valid_times"].values
        for i, (intervals, color) in enumerate(
            zip(all_intervals, custom_palette)
        ):
            int_range = convert_intervals_to_range(intervals, start_time)
            ax.broken_barh(
                int_range, (10 * (i + 1), 6), facecolors=color, alpha=0.7
            )

        ax.set_ylim(5, 10 * (n_compare + 1) + 5)
        ax.set_xlabel("time from start (min)", fontsize=16)
        ax.set_yticks(
            np.arange(n_compare) * 10 + 15,
            labels=interval_list_names,
            fontsize=16,
        )
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels())
        ax.tick_params(axis="x", labelsize=16)

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

    def insert(self, *args, **kwargs):
        """Insert with cautious insert by default."""
        self.cautious_insert(*args, **kwargs)

    def super_insert(self, *args, **kwargs):
        """Insert without cautious insert."""
        super().insert(*args, **kwargs)

    def cautious_insert(self, inserts, update=False, **kwargs):
        """On existing primary key, check secondary key and update if needed.

        `replace=True` will attempt to delete/replace the existing entry. When
        the row has a foreign key constraint, this will fail. This method will
        check if an update is needed.

        Parameters
        ----------
        inserts : list of dict
            List of dictionaries to insert.
        update : bool, optional
            If True, update the existing entry. Defaults to False.
        **kwargs : dict
            Additional keyword arguments to pass to `insert`.
        """
        if not isinstance(inserts, (list, tuple)):  # Table.insert1 makes tuple
            inserts = [inserts]
        if not inserts:  # No data to insert
            return
        if not isinstance(inserts[0], dict):
            self.super_insert(inserts, **kwargs)  # fallback
            return

        pk = self.heading.primary_key

        def pk_match(row):
            match = self & {k: str(v) for k, v in row.items() if k in pk}
            return match.fetch(as_dict=True)[0] if match else None

        def sk_match(new, old):
            return np.array_equal(
                new["valid_times"], old["valid_times"]
            ) and new.get("pipeline", "") == old.get("pipeline", "")

        basic_inserts, need_update = [], []
        for row in inserts:
            existing = pk_match(row)
            if not existing:  # if no existing entry, insert
                basic_inserts.append(row)
            elif existing and not sk_match(row, existing):  # diff sk, update
                need_update.append(row)

        self.super_insert(basic_inserts, **kwargs)

        if update:
            for row in need_update:
                self.update1(row)
        elif need_update:
            raise ValueError(
                f"Found {len(need_update)} rows with existing names, but "
                + f"different times:{need_update}"
            )

    def cleanup(self, dry_run=True):
        """Clean up orphaned IntervalList entries."""
        orphans = self - get_child_tables(self)
        if dry_run:
            return orphans
        orphans.super_delete(warn=False)


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
        warn=True,
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

    def __getitem__(self, item) -> T:
        """Get item from the interval list."""
        if isinstance(item, (slice, int)):
            return Interval(self.times[item], **self.kwargs)
        # elif isinstance(item, slice):
        #     return Interval(self.times[item], **self.kwargs)
        else:
            raise ValueError(
                f"Unrecognized item type: {type(item)}. Must be int or slice."
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
        """Extract interval_list from a given object."""
        if from_inds:
            return self.from_inds(interval_list)
        elif hasattr(interval_list, "times"):
            return interval_list.times
        elif isinstance(interval_list, dict):
            return self._import_from_table(interval_list)
        elif isinstance(
            interval_list, (np.generic, np.ndarray, list, int, float, tuple)
        ):
            return interval_list
        elif interval_list is None:
            return np.array([])
        else:
            raise TypeError(
                f"Unrecognized interval_list type: {type(interval_list)}"
            )

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
        if interval_list.ndim == 1:
            return np.expand_dims(interval_list, 0)
        return interval_list

    def _union_consolidate(self, interval_list: IntervalLike) -> T:
        kwargs = dict(self.kwargs, no_overlap=False)
        return Interval(reduce(self._union_concat, interval_list), **kwargs)

    def union_consolidate(self) -> T:
        return Interval(self._union_consolidate(self.times), **self.kwargs)

    def _consolidate(self, interval_list: IntervalLike) -> T:
        """Consolidate overlapping intervals in an interval list."""
        if not isinstance(interval_list, np.ndarray):
            interval_list = np.asarray(interval_list)
        if interval_list.ndim == 1:
            return self._expand_1d(interval_list)

        interval_list = interval_list[np.argsort(interval_list[:, 0])]
        interval_list = self._union_consolidate(interval_list).times

        # reduce may convert to 1D, so check and expand
        return self._expand_1d(interval_list)

    def consolidate(self) -> T:
        return Interval(self._consolidate(self.times), **self.kwargs)

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
    ) -> T:
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

    def intersect(
        self, other: IntervalLike, min_length: Optional[float] = 0
    ) -> T:
        """Finds the intersections between this and another interval list.

        Each interval is (start time, stop time)

        Parameters
        ----------
        other : Union[Interval, np.array, list, dict]
            Interval list to intersect with self.
        min_length : float, optional.
            Minimum length of intervals to include, default 0

        Returns
        -------
        interval_list: np.array, (N,2)
        """
        return Interval(
            self._intersect(self.times, self._extract(other), min_length),
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
            result = [(0, 1), (4, 5), (9, 10)]

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
            while j < len(intervals2) and intervals2[j][0] < end1:
                start2, end2 = intervals2[j]
                # Add the non-overlap before the current interval in intervals2
                if current_start < start2:
                    result.append((current_start, min(start2, end1)))
                # Update the current start to exclude the overlapping part
                current_start = max(current_start, end2)
                j += 1  # next interval in intervals2
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
    ) -> Union[dj.expression.QueryExpression, np.ndarray]:
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

    ActivityLog().deprecate_log("intervals_by_length", alt="Interval.by_length")

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

    ActivityLog().deprecate_log(
        "interval_list_contains_ind", alt="Interval.contains"
    )

    return Interval(interval_list).contains(timestamps, as_indices=True)


def interval_list_contains(interval_list, timestamps):
    """Find timestamps that are contained in an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    timestamps : array_like
    """
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log(
        "interval_list_contains", alt="Interval.contains"
    )
    return Interval(interval_list).contains(timestamps)


def interval_list_excludes_ind(interval_list, timestamps):
    """Find indices of timestamps that are not contained in an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    timestamps : array_like
    """
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log(
        "interval_list_excludes_ind",
        alt="Interval.excludes(timestamps, as_indices=True)",
    )
    return Interval(interval_list).excludes(timestamps, as_indices=True)


def interval_list_excludes(interval_list, timestamps):
    """Find timestamps that are not contained in an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    timestamps : array_like
    """
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log(
        "interval_list_excludes", alt="Interval.excludes"
    )
    return Interval(interval_list).excludes(timestamps)


def consolidate_intervals(interval_list):
    """Consolidate overlapping intervals in an interval list."""
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log(
        "consolidate_intervals", alt="Interval.consolidate"
    )
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
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log(
        "interval_list_intersect", alt="Interval.intersect"
    )
    return Interval(interval_list1).intersect(interval_list2, min_length).times


def union_adjacent_index(interval1, interval2):
    """Union index-adjacent intervals. If not adjacent, just concatenate.

    e.g. [a,b] and [b+1, c] is converted to [a,c]

    Parameters
    ----------
    interval1 : np.array
    interval2 : np.array
    """
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log(
        "union_adjacent_index", alt="Interval.union_adjacent_index"
    )
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

    ActivityLog().deprecate_log("interval_list_union", alt="Interval.union")

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

    ActivityLog().deprecate_log("interval_list_censor", alt="Interval.censor")
    return Interval(interval_list).censor(timestamps).times


def interval_from_inds(list_frames):
    """Converts a list of indices to a list of intervals.

    e.g. [2,3,4,6,7,8,9,10] -> [[2,4],[6,10]]

    Parameters
    ----------
    list_frames : array_like of int
    """
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log(
        "interval_from_inds", alt="Interval(list_frames, from_inds=True)"
    )
    return Interval(list_frames, from_inds=True).times


def interval_set_difference_inds(intervals1, intervals2):
    """
    e.g.
    intervals1 = [(0, 5), (8, 10)]
    intervals2 = [(1, 2), (3, 4), (6, 9)]

    result = [(0, 1), (4, 5), (9, 10)]

    Parameters
    ----------
    intervals1 : IntervalLike
    intervals2 : IntervalLike

    Returns
    -------
    np.ndarray
    """
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log(
        "interval_set_difference_inds", alt="Interval.subtract"
    )
    return Interval(intervals1).subtract(intervals2).times


def interval_list_complement(intervals1, intervals2, min_length=0.0):
    """
    Finds intervals in intervals1 that are not in intervals2

    Parameters
    ----------
    min_length : float, optional
        Minimum interval length in seconds. Defaults to 0.0.
    """
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log(
        "interval_list_complement",
        alt="Interval(one).subtract(two, min_length=min_length)",
    )
    return (
        Interval(intervals1).subtract(intervals2, min_length=min_length).times
    )
