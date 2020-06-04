import common_session
import datajoint as dj
import pynwb
import numpy as np

schema = dj.schema('common_interval')

# define the schema for intervals
@schema
class IntervalList(dj.Manual):
    definition = """
    -> common_session.Session
    interval_name: varchar(200) #descriptive name of this interval list
    ---
    valid_times: longblob # 2D numpy array with start and end times for each interval
    """

    def insert_from_nwb(self, nwb_file_name):
        '''
        :param nwb_file_name:
        :return: None
        Adds each of the entries in the nwb epochs table to the Interval list
        '''
        try:
            io = pynwb.NWBHDF5IO(nwb_file_name, mode='r')
            nwbf = io.read()
        except:
            print('Error: nwbfile {} cannot be opened for reading\n'.format(nwb_raw_file_name))
            io.close()
            return None

        epochs = nwbf.epochs.to_dataframe()
        epoch_dict = dict()
        epoch_dict['nwb_file_name'] = nwb_file_name
        for e in epochs.iterrows():
            epoch_dict['interval_name'] = e[1].tags[0]
            epoch_dict['valid_times'] = np.asarray([[e[1].start_time, e[1].stop_time]])
            self.insert1(epoch_dict, skip_duplicates="True")


    # FIX: ADD export to NWB function to save relevant intervals in an NWB file
