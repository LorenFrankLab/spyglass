import itertools
from functools import reduce

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.dj_helper_fn import get_child_tables

from .common_session import Session  # noqa: F401

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
    pipeline = "": varchar(64)  # type of interval list (e.g. 'position', 'spikesorting_recording_v1')
    """

    # See #630, #664. Excessive key length.

    @classmethod
    def insert_from_nwbfile(cls, nwbf, *, nwb_file_name):
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

        for _, epoch_data in epochs.iterrows():
            epoch_dict = {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": (
                    epoch_data.tags[0]
                    if epoch_data.tags
                    else f"interval_{epoch_data[0]}"
                ),
                "valid_times": np.asarray(
                    [[epoch_data.start_time, epoch_data.stop_time]]
                ),
            }

            cls.insert1(epoch_dict, skip_duplicates=True)

    def plot_intervals(self, figsize=(20, 5), return_fig=False):
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

    def nightly_cleanup(self, dry_run=True):
        orphans = self - get_child_tables(self)
        if dry_run:
            return orphans
        orphans.super_delete(warn=False)


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
    lengths = np.ravel(np.diff(interval_list))
    return interval_list[
        np.logical_and(lengths > min_length, lengths < max_length)
    ]


def interval_list_contains_ind(interval_list, timestamps):
    """Find indices of list of timestamps contained in an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    timestamps : array_like
    """
    ind = []
    for interval in interval_list:
        ind += np.ravel(
            np.argwhere(
                np.logical_and(
                    timestamps >= interval[0], timestamps <= interval[1]
                )
            )
        ).tolist()
    return np.asarray(ind)


def interval_list_contains(interval_list, timestamps):
    """Find timestamps that are contained in an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    timestamps : array_like
    """
    ind = []
    for interval in interval_list:
        ind += np.ravel(
            np.argwhere(
                np.logical_and(
                    timestamps >= interval[0], timestamps <= interval[1]
                )
            )
        ).tolist()
    return timestamps[ind]


def interval_list_excludes_ind(interval_list, timestamps):
    """Find indices of timestamps that are not contained in an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    timestamps : array_like
    """

    contained_inds = interval_list_contains_ind(interval_list, timestamps)
    return np.setdiff1d(np.arange(len(timestamps)), contained_inds)


def interval_list_excludes(interval_list, timestamps):
    """Find timestamps that are not contained in an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    timestamps : array_like
    """
    contained_times = interval_list_contains(interval_list, timestamps)
    return np.setdiff1d(timestamps, contained_times)


def consolidate_intervals(interval_list):
    if interval_list.ndim == 1:
        interval_list = np.expand_dims(interval_list, 0)
    else:
        interval_list = interval_list[np.argsort(interval_list[:, 0])]
        interval_list = reduce(_union_concat, interval_list)
        # the following check is needed in the case where the interval list is a
        # single element (behavior of reduce)
        if interval_list.ndim == 1:
            interval_list = np.expand_dims(interval_list, 0)
    return interval_list


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

    # Consolidate interval lists to disjoint int'ls by sorting & applying union
    interval_list1 = consolidate_intervals(interval_list1)
    interval_list2 = consolidate_intervals(interval_list2)

    # then do pairwise comparison and collect intersections
    intersecting_intervals = [
        _intersection(interval2, interval1)
        for interval2 in interval_list2
        for interval1 in interval_list1
        if _intersection(interval2, interval1) is not None
    ]

    # if no intersection, then return an empty list
    if not intersecting_intervals:
        return []

    intersecting_intervals = np.asarray(intersecting_intervals)
    intersecting_intervals = intersecting_intervals[
        np.argsort(intersecting_intervals[:, 0])
    ]

    return intervals_by_length(intersecting_intervals, min_length=min_length)


def _intersection(interval1, interval2):
    """Takes the (set-theoretic) intersection of two intervals"""
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    intersection = np.array([start, end]) if end > start else None
    return intersection


def _union(interval1, interval2):
    """Takes the (set-theoretic) union of two intervals"""
    if _intersection(interval1, interval2) is None:
        return np.array([interval1, interval2])
    return np.array(
        [min(interval1[0], interval2[0]), max(interval1[1], interval2[1])]
    )


def _union_concat(interval_list, interval):
    """Compare last interval of interval list to given interval.

    If overlapping, take union. If not, concatenate interval to interval list.

    Recursively called with `reduce`.
    """
    if interval_list.ndim == 1:
        interval_list = np.expand_dims(interval_list, 0)
    if interval.ndim == 1:
        interval = np.expand_dims(interval, 0)

    x = _union(interval_list[-1], interval[0])
    x = np.expand_dims(x, 0) if x.ndim == 1 else x

    return np.concatenate((interval_list[:-1], x), axis=0)


def union_adjacent_index(interval1, interval2):
    """Union index-adjacent intervals. If not adjacent, just concatenate.

    e.g. [a,b] and [b+1, c] is converted to [a,c]

    Parameters
    ----------
    interval1 : np.array
    interval2 : np.array
    """
    interval1 = np.atleast_2d(interval1)
    interval2 = np.atleast_2d(interval2)

    if (
        interval1[-1][1] + 1 == interval2[0][0]
        or interval2[0][1] + 1 == interval1[-1][0]
    ):
        x = np.array(
            [
                [
                    np.min([interval1[-1][0], interval2[0][0]]),
                    np.max([interval1[-1][1], interval2[0][1]]),
                ]
            ]
        )
        return np.concatenate((interval1[:-1], x), axis=0)
    else:
        return np.concatenate((interval1, interval2), axis=0)


# TODO: test interval_list_union code


def _parallel_union(interval_list):
    """Create a parallel list where 1 is start and -1 the end"""
    interval_list = np.ravel(interval_list)
    interval_list_start_end = np.ones(interval_list.shape)
    interval_list_start_end[1::2] = -1
    return interval_list, interval_list_start_end


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
    union_starts = np.ravel(np.array(np.where(np.cumsum(ss[sort_ind]) == 1)))
    union_stops = np.ravel(np.array(np.where(np.cumsum(ss[sort_ind]) == 0)))
    union = [
        [combined_intervals[start], combined_intervals[stop]]
        for start, stop in zip(union_starts, union_stops)
    ]

    return np.asarray(union)


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
    # check that all timestamps are in the interval list
    if len(interval_list_contains_ind(interval_list, timestamps)) != len(
        timestamps
    ):
        raise ValueError("Interval_list must contain all timestamps")

    timestamps_interval = np.asarray([[timestamps[0], timestamps[-1]]])
    return interval_list_intersect(interval_list, timestamps_interval)


def interval_from_inds(list_frames):
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
    return result


def interval_list_complement(intervals1, intervals2, min_length=0.0):
    """
    Finds intervals in intervals1 that are not in intervals2

    Parameters
    ----------
    min_length : float, optional
        Minimum interval length in seconds. Defaults to 0.0.
    """

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

    return intervals_by_length(
        np.asarray(result), min_length=min_length, max_length=1e100
    )
