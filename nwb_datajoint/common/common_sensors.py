# this is the schema for headstage or other environmental sensors

import pynwb
import datajoint as dj
from .common_session import Session
from .common_nwbfile import Nwbfile
from .common_interval import IntervalList
from .common_ephys import Raw
from .nwb_helper_fn import get_data_interface

used = [Session, IntervalList]

schema = dj.schema('common_sensors')


@schema
class SensorData(dj.Imported):
    definition = """                                                                             
    -> Session
    ---
    nwb_object_id: varchar(40)  # the object id of the data in the NWB file
    -> IntervalList       # the list of intervals for this object                                                                        
    """

    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile().get_abs_path(nwb_file_name)
        with pynwb.NWBHDF5IO(path=nwb_file_abspath, mode='r') as io:
            nwbf = io.read()
            sensor = get_data_interface(nwbf, 'analog')
            if sensor is not None:
                key['nwb_object_id'] = sensor.time_series['analog'].object_id
                key['interval_list_name'] = (Raw() & {'nwb_file_name' : nwb_file_name}).fetch1('interval_list_name')
                self.insert1(key)
