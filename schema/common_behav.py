
import pynwb

import common_interval
import common_session
import datajoint as dj

[common_session, common_interval]

schema = dj.schema('common_behav')


@schema
class Position(dj.Imported):
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
            print('Error in Position: nwbfile {} cannot be opened for reading\n'.format(
                key['nwb_file_name']))
            print(io.read())
            return
        # position data is stored in the Behavior module
        try:
            behav_mod = nwbf.get_processing_module("Behavior")
            pos = behav_mod.data_interfaces['Position']

        except:
            print('Error in Position: no Behavior module found in {}\n'.format(
                nwb_file_name))
            return
        key['nwb_object_id'] = -1
        # this is created when we populate the Task schema
        key['interval_name'] = 'task epochs'
        self.insert1(key)


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
                nwb_file_name))
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
                nwb_file_name))
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
                nwb_file_name))
            return
        key['nwb_object_id'] = -1
        # this is created when we populate the Task schema
        key['interval_name'] = 'task epochs'
        self.insert1(key)
