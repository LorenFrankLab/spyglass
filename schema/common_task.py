# Test of automatic datajoint schema generation from NWB file
import numpy as np
import pynwb

import common_interval
import common_session
import datajoint as dj
import franklab_nwb_extensions.fl_extension as fl_extension

schema = dj.schema("common_task", locals())
[common_session, common_interval]


@schema
class Apparatus(dj.Manual):
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
     task_type='': varchar(80)
     task_subtype='': varchar(80)
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
        task_mod = []
        try:
            task_mod = nwbf.get_processing_module("Task")
        except:
            print('No Task module found in {}\n'.format(nwb_file_name))
        if task_mod != []:
            for d in task_mod.data_interfaces:
                if type(task_mod[d]) == fl_extension.Task:
                    # see this task if is already in the database
                    if {'task_name': d} not in self:
                        # FIX task type and subtype would need to be in the NWB file
                        task_dict['task_name'] = d
                        task_dict['task_type'] = ''
                        task_dict['task_subtype'] = ''
                        common_task.TaskInfo.insert1(task_dict)
                    else:
                        print('Skipping task {}; already in schema\n'.format(d))
        io.close()



@schema
class TaskEpoch(dj.Imported):
    # Tasks, apparatus, session and time intervals
    definition = """
     ->common_session.Session
     epoch: int  #the session epoch for this task and apparatus(0 based)
     ---
     -> TaskInfo
     -> ApparatusInfo
     -> common_interval.IntervalList
     task_object_id: int # TO BE converted an NWB datatype when available
     apparatus_object_id: int # TO BE converted an NWB datatype when available
     exposure: int # the number of this exposure to the apparatus and task
     """

    def make(self, key):
        # load up the NWB file and insert information for each of the task epochs
        try:
            io = pynwb.NWBHDF5IO(key['nwb_file_name'], mode='r')
            nwbf = io.read()
        except:
            print('Error in Task: nwbfile {} cannot be opened for reading\n'.format(
                key['nwb_file_name']))
            print(io.read())
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
