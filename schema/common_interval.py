import datajoint as dj
import common_session

schema = dj.schema('common_interval')

#define the schema for intervals
@schema
class IntervalList(dj.Manual):
    definition = """
    -> common_session.Session
    interval_name: varchar(200) #descriptive name of this interval list
    ---
    valid_times: longblob # 2D numpy array with start and end times for each interval
    """