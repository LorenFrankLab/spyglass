from .common_session import Session
import datajoint as dj
import numpy as np

used = [Session]

schema = dj.schema('common_interval')

# TODO: ADD export to NWB function to save relevant intervals in an NWB file

# define the schema for intervals
@schema
class IntervalList(dj.Manual):
    definition = """
    -> Session
    interval_list_name: varchar(200) #descriptive name of this interval list
    ---
    valid_times: longblob # 2D numpy array with start and end times for each interval
    """

    def insert_from_nwbfile(self, nwbf, *, nwb_file_name):
        '''
        :param nwbf:
        :param nwb_file_name:
        :return: None
        Adds each of the entries in the nwb epochs table to the Interval list
        '''
        epochs = nwbf.epochs.to_dataframe()
        epoch_dict = dict()
        epoch_dict['nwb_file_name'] = nwb_file_name
        for e in epochs.iterrows():
            epoch_dict['interval_list_name'] = e[1].tags[0]
            epoch_dict['valid_times'] = np.asarray([[e[1].start_time, e[1].stop_time]])
            self.insert1(epoch_dict, skip_duplicates=True)

@schema
class SortIntervalList(dj.Manual):
    definition = """
    -> Session
    sort_interval_list_name: varchar(200) #descriptive name of this interval list
    ---
    sort_intervals: longblob # 2D numpy array with start and end times for each interval to be used for spike sorting
    """

    

