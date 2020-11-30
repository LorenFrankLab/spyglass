# Test of automatic datajoint schema generation from NWB file
import numpy as np
import pynwb

from .common_session import Session
from .common_nwbfile import Nwbfile
from .common_interval import IntervalList
from .common_device import CameraDevice
from .nwb_helper_fn import get_data_interface

used = [Session, IntervalList, CameraDevice]

import datajoint as dj
# import franklabnwb.fl_extension as fl_extension


schema = dj.schema("common_task", locals())


@schema
class Apparatus(dj.Manual):
    definition = """
     apparatus_name: varchar(80)
     ---
     #-> AnalysisNWBFile
     nwb_object_id='': varchar(255)  #the NWB object identifier for a class that describes this apparatus
     """

    # def insert_from_nwbfile(self, nwbf):
    #     # If we're going to use we will need to add specific apparatus information (cad files?)
    #     apparatus_dict = dict()
    #     apparatus_mod = []
    #     try:
    #         apparatus_mod = nwbf.get_abs_path("Apparatus")
    #     except:
    #         print('No Apparatus module found in NWB file')
    #         return
    #     if apparatus_mod != []:
    #         for d in apparatus_mod.data_interfaces:
    #             pass
    #             # TODO: restore this functionality
    #             if type(apparatus_mod[d]) == franklabnwb.fl_extension.Apparatus:
    #                 # see this Apparaus if is already in the database
    #                 if {'apparatus_name': d} not in common_task.ApparatusInfo():
    #                     apparatus_dict['apparatus_name'] = d
    #                     common_task.ApparatusInfo.insert1(apparatus_dict)
    #                 else:
    #                     print('Skipping apparatus {}; already in schema\n'.format(d))

@schema
class Task(dj.Manual):
    definition = """
     task_name: varchar(80)
     ---
     task_description='':   varchar(255)  # description of this task
     task_type='': varchar(80)  # type of task
     task_subtype='': varchar(80) # subtype of task
     
     """

    # def insert_from_nwbfile(self, nwbf):
    #     task_dict = dict()
    #     #search through he processing modules to see if we can find a behavior module. This should probably be a
    #     # general function
    #     task_interface = get_data_interface(nwbf, 'task')
    #     if task_interface is not None:
    #         # check the datatype
    #         if task_interface.data_type == 'DynamicTable':
    #             taskdf = task_interface.to_dataframe()
    #             # go through the rows of the dataframe and add the task if it doesn't exist
    #             for task_entry in taskdf.iterrows():
    #                 task_dict['task_name'] = task_entry[1].task_name
    #                 task_dict['task_description'] = task_entry[1].task_description
    #                 # FIX if types are defined and NWB object exist
    #                 task_dict['task_type'] = ''
    #                 task_dict['task_subtype'] = ''
    #                 task_dict['nwb_object_id'] = ''
    #                 self.insert1(task_dict, skip_duplicates='True')



@schema
class TaskEpoch(dj.Imported):
    # Tasks, session and time intervals
    definition = """
     -> Session
     epoch: int  #the session epoch for this task and apparatus(1 based)
     ---
     -> Task
     -> CameraDevice
     -> IntervalList
     """

    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile().get_abs_path(nwb_file_name)
        with pynwb.NWBHDF5IO(path=nwb_file_abspath, mode='r') as io:
            nwbf = io.read()
            camera_name = dict()
            # the tasks refer to the camera_id which is unique for the NWB file but not for CameraDevice schema, so we need to look up the right camer
            for d in nwbf.devices:
                if 'camera_device' in d:
                    device = nwbf.devices[d]
                    # get the camera ID
                    c = str.split(d)
                    camera_name[int(c[1])] = device.camera_name

            # find the task modules and for each one, add the task to the Task schema if it isn't there 
            # and then add an entry for each epoch
            # TODO: change to new task structure
            for task_num in range(0,10):
                task_str = 'task_' + str(task_num)
                task = get_data_interface(nwbf, task_str)
                if task is not None:
                    # check if the task is in the Task table and if not, add it
                    if len(((Task() & {'task_name': task.task_name[0]}).fetch())) == 0:
                        Task().insert1({'task_name': task.task_name[0],
                                        'task_description': task.task_description[0]})
                    key['task_name'] = task.task_name[0]
                    # get the camera used for this task. Use 'none' if there was no camera
                    try:
                        camera_id = int(task.camera_id[0])
                        key['camera_name'] = camera_name[camera_id]
                    except:
                        key['camera_name'] = 'none'
                    # make sure the 'none' camera is in the CameraDevice table and add if not
                    if len((CameraDevice() & {'camera_name' : 'none'}).fetch()) == 0:
                        CameraDevice().insert1({'camera_name' : 'none'})
                    # get the interval list for this task, which corresponds to the matching epoch for the raw data.
                    # Users should define more restrictive intervals as required for analyses
                    session_intervals = (IntervalList() & {'nwb_file_name' : nwb_file_name}).fetch('interval_list_name')
                    for epoch in task.task_epochs[0]:
                        key['epoch'] = epoch
                        target_interval = str(epoch).zfill(2)
                        for interval in session_intervals:
                            if target_interval in interval:
                                break
                        key['interval_list_name'] = interval
                        self.insert1(key)