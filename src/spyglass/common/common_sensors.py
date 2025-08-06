"""Schema for headstage or other environmental sensors."""

from typing import Optional

import datajoint as dj
import numpy as np
import pandas as pd
import pynwb

from spyglass.common.common_ephys import Raw
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.nwb_helper_fn import get_data_interface, get_nwb_file

schema = dj.schema("common_sensors")


@schema
class SensorData(SpyglassMixin, dj.Imported):
    definition = """
    -> Session
    ---
    sensor_data_object_id: varchar(40)  # object id of the data in the NWB file
    -> IntervalList                     # the list of intervals for this object
    """

    _nwb_table = Nwbfile

    def make(self, key: dict) -> None:
        """Populate SensorData using the analog BehavioralEvents from the NWB.

        Parameters
        ----------
        key: dict
            The key for the session to populate the SensorData for.
            This should include the nwb_file_name and session_id.

        Raises
        -------
        ValueError
            If the number of columns in the description does not match
            the number of columns in the data.
        """

        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile().get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)

        sensor = get_data_interface(
            nwbf, "analog", pynwb.behavior.BehavioralEvents
        )

        # Validate the sensor data
        if sensor is None:
            logger.info(f"No conforming sensor data found in {nwb_file_name}\n")
            return

        columns = sensor.time_series["analog"].description.split()
        columns = [col for col in columns if col not in ["time", "timestamps"]]
        n_cols = sensor.time_series["analog"].data.shape[1]
        if len(columns) != n_cols:
            raise ValueError(
                f"Number of columns in description ({len(columns)}) "
                f"does not match number of columns in data ({n_cols}). "
                f"Columns: {sensor.time_series['analog'].description}. "
                "Please check the NWB file."
            )

        # Create the NWB file object
        key["sensor_data_object_id"] = sensor.time_series["analog"].object_id

        # the valid times for these data are the same as the valid times for
        # the raw ephys data
        key["interval_list_name"] = (
            Raw & {"nwb_file_name": nwb_file_name}
        ).fetch1("interval_list_name")

        self.insert1(key, allow_direct_insert=True)

    def fetch1_dataframe(
        self, interval_list_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Fetch the sensor data as a DataFrame.

        Parameters
        ----------
        interval_list_name: str, optional
            The name of the interval list to filter the data by.
            If None, no filtering is applied.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the sensor data, indexed by time.
            If no data is found, None is returned.

        Raises
        -------
        ValueError
            If more than one sensor data object is found or
            if the specified interval list is not found.

        """
        if len(self) == 0:
            return None
        _ = self.ensure_single_entry()

        nwb = self.fetch_nwb()[0]
        columns = nwb["sensor_data"].description.split()
        columns = [col for col in columns if col not in ["time", "timestamps"]]

        if interval_list_name is None:
            # corresponds to `raw data valid times` interval
            return pd.DataFrame(
                nwb["sensor_data"].data,
                index=pd.Index(nwb["sensor_data"].timestamps, name="time"),
                columns=columns,
            )

        nwb_file_name = self.fetch1("nwb_file_name")
        valid_times = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_list_name,
            }
        ).fetch1("valid_times")

        if len(valid_times) == 0:
            raise ValueError(
                f"No valid times found for {nwb_file_name} "
                f"and {interval_list_name}"
            )

        sensor_data_df = []
        for start_time, end_time in valid_times:
            start_ind, end_ind = np.searchsorted(
                nwb["sensor_data"].timestamps, [start_time, end_time]
            )
            sensor_data_df.append(
                pd.DataFrame(
                    nwb["sensor_data"].data[start_ind:end_ind, :],
                    index=pd.Index(
                        nwb["sensor_data"].timestamps[start_ind:end_ind],
                        name="time",
                    ),
                    columns=columns,
                )
            )

        return pd.concat(sensor_data_df, axis=0).sort_index()
