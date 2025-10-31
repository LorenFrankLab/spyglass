from typing import Dict, List

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb

from spyglass.common.common_ephys import Raw
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils import SpyglassIngestion, logger

schema = dj.schema("common_dio")


@schema
class DIOEvents(SpyglassIngestion, dj.Imported):
    definition = """
    -> Session
    dio_event_name: varchar(80)   # the name assigned to this DIO event
    ---
    dio_object_id: varchar(40)    # the object id of the data in the NWB file
    -> IntervalList               # the list of intervals for this object
    """

    _nwb_table = Nwbfile
    _source_nwb_object_name = "behavioral_events"

    @property
    def _source_nwb_object_type(self):
        """The _source_nwb_object_type property."""
        return pynwb.behavior.BehavioralEvents

    def generate_entries_from_nwb_object(
        self, nwb_obj, base_key=None
    ) -> Dict["SpyglassIngestion", List[dict]]:
        """Generate entries from nwb object.

        Parameters
        ----------
        nwb_obj : pynwb.NWBFile
            The NWB file object.
        base_key : dict, optional
            The base key to use for the entries, by default None

        Returns
        -------
        Dict[SpyglassIngestion, List[dict]]
            A dictionary with the table class as the key and a list of entries
            as the value.
        """
        base_key = base_key or dict()
        nwb_file_name = base_key.get("nwb_file_name", None)
        if not nwb_file_name:
            raise ValueError("nwb_file_name must be provided in base_key")

        # Times for these events correspond to the valid times for the raw data
        # If no raw data found, create a default interval list named
        # "dio data valid times"
        interval_list_name = "dio data valid times"
        if raw_query := (Raw() & {"nwb_file_name": nwb_file_name}):
            interval_list_name = (raw_query).fetch1("interval_list_name")

        dio_inserts = []
        time_range_list = []
        for event_series in nwb_obj.time_series.values():
            timestamps = event_series.get_timestamps()
            if len(timestamps) == 0:  # Can be either np array or HDMF5 dataset
                logger.warning(
                    f"No timestamps found for DIO event {event_series.name}"
                )
                continue
            dio_inserts.append(
                dict(
                    base_key,
                    interval_list_name=interval_list_name,
                    dio_event_name=event_series.name,
                    dio_object_id=event_series.object_id,
                )
            )
            time_range_list.extend([timestamps[0], timestamps[-1]])

        interval_key = []
        if not raw_query:
            interval_key.append(
                dict(
                    nwb_file_name=nwb_file_name,
                    interval_list_name=interval_list_name,
                    valid_times=np.array(
                        [[np.min(time_range_list), np.max(time_range_list)]]
                    ),
                )
            )

        return {
            IntervalList: interval_key,
            DIOEvents: dio_inserts,
        }

    def make(self, key):
        """Make function to populate the table. For backward compatibility"""
        from spyglass.common.common_usage import ActivityLog

        ActivityLog().deprecate_log(
            self, "DIOEvents.make", alt="insert_from_nwbfile"
        )

        # Call the new SpyglassIngestion method
        self.insert_from_nwbfile(key["nwb_file_name"])

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
