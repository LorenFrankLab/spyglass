import datajoint as dj
import pynwb

from .common_ephys import Raw
from .common_interval import IntervalList  # noqa: F401
from .common_nwbfile import Nwbfile
from .common_session import Session  # noqa: F401
from .nwb_helper_fn import get_data_interface

schema = dj.schema('common_dio')


@schema
class DIOEvents(dj.Imported):
    definition = """
    -> Session
    dio_event_name: varchar(80)  # the name assigned to this DIO event
    ---
    nwb_object_id: varchar(80)   # the object id of the data in the NWB file
    -> IntervalList              # the list of intervals for this object
    """

    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        with pynwb.NWBHDF5IO(path=nwb_file_abspath, mode='r') as io:
            nwbf = io.read()
            # Get the data interface for 'behavioral_events"
            behav_events = get_data_interface(nwbf, 'behavioral_events').time_series
            # the times for these events correspond to the valid times for the raw data
            key['interval_list_name'] = (Raw() & {'nwb_file_name': nwb_file_name}).fetch1('interval_list_name')
            for event_series in behav_events:
                key['dio_event_name'] = event_series
                key['nwb_object_id'] = behav_events[event_series].object_id
                self.insert1(key, skip_duplicates='True')
