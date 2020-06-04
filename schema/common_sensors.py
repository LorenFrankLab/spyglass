# this is the schema for headstage or other environmental sensors


import datajoint as dj
import common_session
import common_interval
import pynwb

[common_session, common_interval]

schema = dj.schema('common_sensors')


@schema
class Sensor(dj.Imported):
    definition = """                                                                             
    -> common_session.Session
    sensor_type: enum('Acclerometer', 'Gyro', 'Magnetometer')
    ---
    nwb_object_id: int  # the object id of the data in the NWB file
    -> common_interval.IntervalList       # the list of intervals for this object                                                                        
    """