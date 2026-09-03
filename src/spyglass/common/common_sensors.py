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
from spyglass.utils import SpyglassIngestion

schema = dj.schema("common_sensors")


@schema
class SensorData(SpyglassIngestion, dj.Imported):
    definition = """
    -> Session
    ---
    sensor_data_object_id: varchar(40)  # object id of the data in the NWB file
    -> IntervalList                     # the list of intervals for this object
    """

    _nwb_table = Nwbfile
    # The enclosing ProcessingModule and the inner TimeSeries share this name,
    # so the type below is what distinguishes the three.
    _source_nwb_object_name = "analog"
    _only_ingest_first = True  # first match wins, as get_data_interface did

    _source_nwb_object_type = pynwb.behavior.BehavioralEvents

    @property
    def table_key_to_obj_attr(self):
        return {"self": {"sensor_data_object_id": self._analog_object_id}}

    @staticmethod
    def _analog_series(nwb_obj):
        """Return the analog TimeSeries held by a BehavioralEvents object.

        Parameters
        ----------
        nwb_obj : pynwb.behavior.BehavioralEvents
            The sensor data container.

        Returns
        -------
        pynwb.base.TimeSeries
            The series named 'analog'.
        """
        return nwb_obj.time_series["analog"]

    def _analog_object_id(self, nwb_obj) -> str:
        """Validate the analog series and return its object id.

        The id stored is the series', not the enclosing container's.

        Parameters
        ----------
        nwb_obj : pynwb.behavior.BehavioralEvents
            The sensor data container.

        Returns
        -------
        str
            Object id of the analog series.

        Raises
        ------
        ValueError
            If the description does not name one column per column of data.
        """
        series = self._analog_series(nwb_obj)

        columns = [
            col
            for col in series.description.split()
            if col not in ["time", "timestamps"]
        ]
        n_cols = series.data.shape[1]

        if len(columns) != n_cols:
            raise ValueError(
                f"Number of columns in description ({len(columns)}) "
                f"does not match number of columns in data ({n_cols}). "
                f"Columns: {series.description}. "
                "Please check the NWB file."
            )

        return series.object_id

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Attach the raw ephys interval, which these data share."""
        super_ins = super().generate_entries_from_nwb_object(nwb_obj, base_key)
        self_key = super_ins[self][0]

        # the valid times for these data are the same as the valid times for
        # the raw ephys data
        self_key["interval_list_name"] = (
            Raw & {"nwb_file_name": self_key["nwb_file_name"]}
        ).fetch1("interval_list_name")

        return super_ins

    def make(self, key: dict) -> None:
        """Deprecated in favor of insert_from_nwbfile."""
        raise NotImplementedError(
            "SensorData.make is deprecated. Use insert_from_nwbfile."
        )

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
