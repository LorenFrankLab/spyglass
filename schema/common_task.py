# Test of automatic datajoint schema generation from NWB file
import numpy as np
import pynwb

import common_interval
import common_session
import datajoint as dj
import nwb_helper_fn as nh
import franklab_nwb_extensions.fl_extension as fl_extension

schema = dj.schema("common_task", locals())
[common_session, common_interval]


@schema
class Apparatus(dj.Manual):
# NOTE: this needs to be updated once we figure out how we're going to define the apparatus
    definition = """
     apparatus_name: varchar(80)
     """

    def insert_from_nwb(self, nwb_file_name):
        try:
            io = pynwb.NWBHDF5IO(nwb_file_name, mode='r')
            nwbf = io.read()
        except:
            print('Error: nwbfile {} cannot be opened for reading\n'.format(
                nwb_file_name))
            print(io.read())
            io.close()
            return
        # If we're going to use we will need to add specific apparatus information (cad files?)
        apparatus_dict = dict()
        apparatus_mod = []
        try:
            apparatus_mod = nwbf.get_processing_module("Apparatus")
        except:
            print('No Apparatus module found in {}\n'.format(nwb_file_name))
        if apparatus_mod != []:
            for d in apparatus_mod.data_interfaces:
                if type(apparatus_mod[d]) == franklabnwb.fl_extension.Apparatus:
                    # see this Apparaus if is already in the database
                    if {'apparatus_name': d} not in common_task.ApparatusInfo():
                        apparatus_dict['apparatus_name'] = d
                        common_task.ApparatusInfo.insert1(apparatus_dict)
                    else:
                        print('Skipping apparatus {}; already in schema\n'.format(d))
        io.close()

@schema
class Task(dj.Manual):
    definition = """
     task_name: varchar(80)
     ---
     task_description='':   varchar(255)  # description of this task
     task_type='': varchar(80)  # type of task
     task_subtype='': varchar(80) # subtype of task
     nwb_object_id='': varchar(255)  #the NWB object identifier for a class that describes this task
     """

    def insert_from_nwb(self, nwb_file_name):
        try:
            io = pynwb.NWBHDF5IO(nwb_file_name, mode='r')
            nwbf = io.read()
        except:
            print('Error: nwbfile {} cannot be opened for reading\n'.format(
                nwb_file_name))
            print(io.read())
            io.close()
            return

        task_dict = dict()
        #search through he processing modules to see if we can find a behavior module. This should probably be a
        # general function
        task_interface = nh.get_data_interface(nwbf, 'task')
        if task_interface is not None:
            # check the datatype
            if task_interface.data_type == 'DynamicTable':
                taskdf = task_interface.to_dataframe()
                # go through the rows of the dataframe and add the task if it doesn't exist
                for task_entry in taskdf.iterrows():
                    task_dict['task_name'] = task_entry[1].task_name
                    task_dict['task_description'] = task_entry[1].task_description
                    # FIX if types are defined and NWB object exist
                    task_dict['task_type'] = ''
                    task_dict['task_subtype'] = ''
                    task_dict['nwb_object_id'] = ''
                    self.insert1(task_dict, skip_duplicates='True')

        io.close()



@schema
class TaskEpoch(dj.Manual):
    # Tasks, apparatus, session and time intervals
    definition = """
     ->common_session.Session
     epoch: int  #the session epoch for this task and apparatus(0 based)
     ---
     -> Task
     -> Apparatus
     -> common_interval.IntervalList
     exposure: int # the number of this exposure to the apparatus and task
     """

    def insert_from_nwb(self, nwb_file_name):
        '''
        Inserts Task epochs from the nwb file's epochs object. Currently assumes fields named epoch, exposure,
        task_name, and apparatus_name. Not currently working with raw NWB files
        :param nwb_file_name:
        :return: None
        '''
        # load up the NWB file and insert information for each of the task epochs
        try:
            io = pynwb.NWBHDF5IO(key['nwb_file_name'], mode='r')
            nwbf = io.read()
        except:
            print('Error in Task: nwbfile {} cannot be opened for reading\n'.format(
                key['nwb_file_name']))
            return
        # start by inserting the Apparati from the nwb file. The ApparatusInfo schema will already have been populated
        try:
            apparatus_mod = nwbf.get_processing_module("Apparatus")
        except:
            print('No Apparatus module found in {}\n'.format(
                key['nwb_file_name']))
            return

        epochs = nwbf.epochs.to_dataframe()
        interval_dict = dict()
        all_epoch_intervals = np.zeros((len(epochs), 2), dtype=np.float64)
        for enum in range(len(epochs)):
            key['epoch'] = enum
            key['exposure'] = epochs['exposure'][enum]
            # Right now to get the task and apparatus information we need to get the container name
            key['task_name'] = epochs['task'][enum]._AbstractContainer__name
            #key['task_object_id'] = epochs['task'][enum].obj_id
            # replace with line above when nwb object IDs are ready
            key['task_object_id'] = -1
            key['apparatus_name'] = epochs['apparatus'][enum]._AbstractContainer__name
            # replace when nwb object IDs are ready
            key['apparatus_object_id'] = -1

            # create an interval structure for this epoch
            interval_dict['nwb_file_name'] = key['nwb_file_name']
            interval_dict['interval_name'] = 'task epoch {}'.format(enum)
            all_epoch_intervals[enum] = np.asarray(
                [epochs['start_time'][enum], epochs['stop_time'][enum]])
            interval_dict['valid_times'] = all_epoch_intervals[enum]
            common_interval.IntervalList.insert1(
                interval_dict, skip_duplicates=True)
            # add this interval
            key['interval_name'] = interval_dict['interval_name']
            self.insert1(key)
        # now create a new interval with all the epochs
        interval_dict['interval_name'] = 'task epochs'
        interval_dict['valid_times'] = all_epoch_intervals
        common_interval.IntervalList.insert1(
            interval_dict, skip_duplicates=True)
