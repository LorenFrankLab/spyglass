import itertools

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pynwb import NWBFile

from spyglass.common.common_interval_obj import Interval
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
    return Interval(interval_list).by_length(min_length, max_length).times


def interval_list_contains_ind(interval_list, timestamps):
    """Find indices of list of timestamps contained in an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    timestamps : array_like
    """
    return Interval(interval_list).contains(timestamps, as_indices=True).times


def interval_list_contains(interval_list, timestamps):
    """Find timestamps that are contained in an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    timestamps : array_like
    """
    return Interval(interval_list).contains(timestamps).times


def interval_list_excludes_ind(interval_list, timestamps):
    """Find indices of timestamps that are not contained in an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    timestamps : array_like
    """
    return Interval(interval_list).excludes(timestamps, as_indices=True).times


def interval_list_excludes(interval_list, timestamps):
    """Find timestamps that are not contained in an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    timestamps : array_like
    """
    return Interval(interval_list).excludes(timestamps).times


def consolidate_intervals(interval_list):
    """Consolidate overlapping intervals in an interval list."""
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


def _union_concat(interval_list, interval):
    """Compare last interval of interval list to given interval.

    If overlapping, take union. If not, concatenate interval to interval list.

    Recursively called with `reduce`.
    """

    # _union_concat GETS CALLED BY reduce - will be trickier to replace

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
    return Interval(intervals1).set_difference_inds(intervals2).times


def interval_list_complement(intervals1, intervals2, min_length=0.0):
    """
    Finds intervals in intervals1 that are not in intervals2

    Parameters
    ----------
    min_length : float, optional
        Minimum interval length in seconds. Defaults to 0.0.
    """
    return Interval(intervals1).complement(intervals2, min_length).times
