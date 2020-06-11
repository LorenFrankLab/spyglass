# this is the schema for headstage or other environmental sensors


import datajoint as dj
from .common_session import Session
from .common_interval import IntervalList
import pynwb

schema = dj.schema('common_sensors')


@schema
class Sensor(dj.Imported):
    definition = """                                                                             
    -> Session
    sensor_type: enum('Acclerometer', 'Gyro', 'Magnetometer')
    ---
    nwb_object_id: int  # the object id of the data in the NWB file
    -> IntervalList       # the list of intervals for this object                                                                        
    """