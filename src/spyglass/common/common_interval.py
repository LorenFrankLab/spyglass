import warnings
from typing import Optional

import datajoint as dj
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb

from spyglass.common.common_session import Session  # noqa: F401
from spyglass.settings import test_mode
from spyglass.utils import SpyglassIngestion, logger
from spyglass.utils.dj_helper_fn import get_child_tables

schema = dj.schema("common_interval")

_INTERVAL_DOC = (
    "https://lorenfranklab.github.io/spyglass/latest/Features/Intervals/"
)

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
    _source_nwb_object_name = "epochs"

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
                ">1 nwb_file_name found in IntervalList. "
                + "the intended use of plot_intervals is to compare intervals "
                + "within a single nwb_file_name."
            )

        interval_list_names = interval_lists_df["interval_list_name"].values

        n_compare = len(interval_list_names)

        if n_compare > 100:
            warnings.warn(
                f"plot_intervals is plotting {n_compare} intervals. "
                + "if this is unintended, please pass in a smaller IntervalList.",
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
            if getattr(intervals, "shape", None) == (2,):
                intervals = [intervals]
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


# Re-exported so existing imports from this module keep working.
from spyglass.utils.interval import Interval, IntervalLike, T  # noqa: F401
