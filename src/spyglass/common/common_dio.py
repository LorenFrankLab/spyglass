import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb

from spyglass.common.common_ephys import Raw
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.nwb_helper_fn import get_data_interface, get_nwb_file

schema = dj.schema("common_dio")


@schema
class DIOEvents(SpyglassMixin, dj.Imported):
    definition = """
    -> Session
    dio_event_name: varchar(80)   # the name assigned to this DIO event
    ---
    dio_object_id: varchar(40)    # the object id of the data in the NWB file
    -> IntervalList               # the list of intervals for this object
    """

    _nwb_table = Nwbfile

    def make(self, key):
        """Make without transaction

        Allows populate_all_common to work within a single transaction."""
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)

        behav_events = get_data_interface(
            nwbf, "behavioral_events", pynwb.behavior.BehavioralEvents
        )
        if behav_events is None:
            logger.warning(
                "No conforming behavioral events data interface found in "
                + f"{nwb_file_name}\n"
            )
            return  # See #849

        # Times for these events correspond to the valid times for the raw data
        # If no raw data found, create a default interval list named
        # "dio data valid times"
        if raw_query := (Raw() & {"nwb_file_name": nwb_file_name}):
            key["interval_list_name"] = (raw_query).fetch1("interval_list_name")
        else:
            key["interval_list_name"] = "dio data valid times"

        dio_inserts = []
        time_range_list = []
        for event_series in behav_events.time_series.values():
            timestamps = event_series.get_timestamps()
            if len(timestamps) == 0:  # Can be either np array or HDMF5 dataset
                logger.warning(
                    f"No timestamps found for DIO event {event_series.name} "
                    + f"in {nwb_file_name}. Skipping."
                )
                continue
            key["dio_event_name"] = event_series.name
            key["dio_object_id"] = event_series.object_id
            dio_inserts.append(key.copy())
            time_range_list.extend([timestamps[0], timestamps[-1]])

        if key["interval_list_name"] == "dio data valid times":
            # insert a default interval list for DIO events if no raw data
            interval_key = {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": key["interval_list_name"],
                "valid_times": np.array(
                    [[np.min(time_range_list), np.max(time_range_list)]]
                ),
            }
            IntervalList.insert1(interval_key)

        self.insert(
            dio_inserts,
            skip_duplicates=True,
            allow_direct_insert=True,
        )

    def plot_all_dio_events(self, return_fig=False):
        """Plot all DIO events in the session.

        Examples
        --------
        > restr1 = {'nwb_file_name': 'arthur20220314_.nwb'}
        > restr2 = {'nwb_file_name': 'arthur20220316_.nwb'}
        > (DIOEvents & restr1).plot_all_dio_events()
        > (DIOEvents & [restr1, restr2]).plot_all_dio_events()

        """
        behavioral_events = self.fetch_nwb()
        nwb_file_names = np.unique(
            [event["nwb_file_name"] for event in behavioral_events]
        )
        epoch_valid_times = (
            pd.DataFrame(
                IntervalList()
                & [
                    {"nwb_file_name": nwb_file_name}
                    for nwb_file_name in nwb_file_names
                ]
            )
            .set_index("interval_list_name")
            .filter(regex=r"^[0-9]", axis=0)
            .valid_times
        )

        n_events = len(behavioral_events)

        _, axes = plt.subplots(
            n_events,
            1,
            figsize=(15, n_events * 0.3),
            dpi=100,
            sharex=True,
            constrained_layout=True,
        )

        for ind, (ax, event) in enumerate(zip(axes.flat, behavioral_events)):
            for epoch_name, epoch in epoch_valid_times.items():
                start_time, stop_time = epoch.squeeze()
                ax.axvspan(start_time, stop_time, alpha=0.5)
                if ind == 0:
                    ax.text(
                        start_time + (stop_time - start_time) / 2,
                        1.001,
                        epoch_name,
                        ha="center",
                        va="bottom",
                    )
            ax.step(
                np.asarray(event["dio"].timestamps),
                np.asarray(event["dio"].data),
                where="post",
                color="black",
            )
            ax.set_ylabel(
                event["dio_event_name"], rotation=0, ha="right", va="center"
            )
            ax.set_yticks([])
        ax.set_xlabel("Time")

        if len(nwb_file_names) == 1:
            plt.suptitle(f"DIO events in {nwb_file_names[0]}")
        else:
            plt.suptitle(f"DIO events in {', '.join(nwb_file_names)}")

        if return_fig:
            return plt.gcf()
