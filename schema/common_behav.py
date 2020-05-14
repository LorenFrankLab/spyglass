
import pynwb
import numpy as np

import common_interval
import common_session
import datajoint as dj
import nwb_helper_fn as nh

[common_session, common_interval]

schema = dj.schema('common_behav')


@schema
class RawPosition(dj.Imported):
    definition = """
    -> common_session.Session
    ---
    nwb_object_id: varchar(80)            # the object id of the data in the NWB file
    -> common_interval.IntervalList       # the list of intervals for this object
    """

    def make(self, key):
        try:
            io = pynwb.NWBHDF5IO(key['nwb_file_name'], mode='r')
            nwbf = io.read()
        except:
            print('Error in Position: nwbfile {} cannot be opened for reading\n'.format(
                key['nwb_file_name']))
            print(io.read())
            return

        # Get the position data. FIX: change Position to position when name changed or fix helper function to allow
        # upper or lower case
        position = nh.get_data_interface(nwbf, 'position')
        if position is None:
            position = nh.get_data_interface(nwbf, 'Position')

        if position is not None:
            pos_interval_name = 'pos valid times'
            # get the valid intervals for the position data
            p = position.get_spatial_series()
            timestamps = np.asarray(p.timestamps)
            # estimate the sampling rate
            sampling_rate = nh.estimate_sampling_rate(timestamps, 1.75)
            print("Processing raw position data. Estimated sampling rate: {} Hz".format(sampling_rate))
            # add the valid intervals to the Interval list
            interval_dict = dict()
            interval_dict['nwb_file_name'] = key['nwb_file_name']
            interval_dict['interval_name'] = pos_interval_name
            interval_dict['valid_times'] = nh.get_valid_intervals(timestamps, sampling_rate, 1.75, 0)
            common_interval.IntervalList.insert1(interval_dict, skip_duplicates="True")

            key['nwb_object_id'] = position.object_id
            # this is created when we populate the Task schema
            key['interval_name'] = pos_interval_name
            self.insert1(key)

        else:
            print('No position data interface found in  {}\n'.format(key['nwb_file_name']))
            return




@schema
class HeadDir(dj.Imported):
    definition = """
    -> common_session.Session
    ---
    nwb_object_id: int  # the object id of the data in the NWB file
    -> common_interval.IntervalList       # the list of intervals for this object
    """

    def make(self, key):
        try:
            io = pynwb.NWBHDF5IO(key['nwb_file_name'], mode='r')
            nwbf = io.read()
        except:
            print('Error in HeadDir: nwbfile {} cannot be opened for reading\n'.format(
                key['nwb_file_name']))
            print(io.read())
            return
        # position data is stored in the Behavior module
        try:
            behav_mod = nwbf.get_processing_module("Behavior")
            headdir = behav_mod.data_interfaces['Head Direction']

        except:
            print('Error in HeadDir: no Behavior module found in {}\n'.format(
                  key['nwb_file_name']))
            return
        key['nwb_object_id'] = -1
        # this is created when we populate the Task schema
        key['interval_name'] = 'task epochs'
        self.insert1(key)


@schema
class Speed(dj.Imported):
    definition = """
    -> common_session.Session
    ---
    nwb_object_id: int  # the object id of the data in the NWB file
    -> common_interval.IntervalList       # the list of intervals for this object
    """

    def make(self, key):
        try:
            io = pynwb.NWBHDF5IO(key['nwb_file_name'], mode='r')
            nwbf = io.read()
        except:
            print('Error in Speed: nwbfile {} cannot be opened for reading\n'.format(
                key['nwb_file_name']))
            print(io.read())
            return
        # position data is stored in the Behavior module
        try:
            behav_mod = nwbf.get_processing_module("Behavior")
            speed = behav_mod.data_interfaces['Speed']

        except:
            print('Error in Speed: no Behavior module found in {}\n'.format(
                  key['nwb_file_name']))
            return
        key['nwb_object_id'] = -1
        # this is created when we populate the Task schema
        key['interval_name'] = 'task epochs'
        self.insert1(key)


@schema
class LinPos(dj.Imported):
    definition = """
    -> common_session.Session
    ---
    nwb_object_id: int  # the object id of the data in the NWB file
    -> common_interval.IntervalList       # the list of intervals for this object
    """

    def make(self, key):
        try:
            io = pynwb.NWBHDF5IO(key['nwb_file_name'], mode='r')
            nwbf = io.read()
        except:
            print('Error in LinPos: nwbfile {} cannot be opened for reading\n'.format(
                key['nwb_file_name']))
            print(io.read())
            return
        # position data is stored in the Behavior module
        try:
            behav_mod = nwbf.get_processing_module("Behavior")
            linpos = behav_mod.data_interfaces['Linearized Position']

        except:
            print('Error in LinPos: no Behavior module found in {}\n'.format(
                  key['nwb_file_name']))
            return
        key['nwb_object_id'] = -1
        # this is created when we populate the Task schema
        key['interval_name'] = 'task epochs'
        self.insert1(key)
