"""Schema for headstage or other environmental sensors."""

import datajoint as dj
import pynwb

from spyglass.common.common_ephys import Raw
from spyglass.common.common_interval import IntervalList  # noqa: F401
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

    def make(self, key):
        """Populate SensorData using the analog BehavioralEvents from the NWB."""

        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile().get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)

        sensor = get_data_interface(
            nwbf, "analog", pynwb.behavior.BehavioralEvents
        )
        if sensor is None:
            logger.info(f"No conforming sensor data found in {nwb_file_name}\n")
            return

        key["sensor_data_object_id"] = sensor.time_series["analog"].object_id

        # the valid times for these data are the same as the valid times for
        # the raw ephys data

        key["interval_list_name"] = (
            Raw & {"nwb_file_name": nwb_file_name}
        ).fetch1("interval_list_name")
        self.insert1(key)
