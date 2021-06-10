import datajoint as dj
import pynwb

from .common_session import Session  # noqa: F401
from .common_nwbfile import Nwbfile
from .common_interval import IntervalList
from .common_device import CameraDevice
from .nwb_helper_fn import get_nwb_file

schema = dj.schema("common_task")


@schema
class Apparatus(dj.Manual):
    definition = """
     apparatus_name: varchar(80)
     ---
     #-> AnalysisNWBFile
     nwb_object_id='': varchar(255)  # the NWB object identifier for a class that describes this apparatus
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
     task_description = NULL: varchar(255)  # description of this task
     task_type = NULL: varchar(80)          # type of task
     task_subtype = NULL: varchar(80)       # subtype of task
     """

    def insert_from_nwbfile(self, nwbf):
        """Insert tasks from an NWB file.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.
        """
        tasks_mod = nwbf.processing.get('tasks')
        if tasks_mod is None:
            print(f'No tasks processing module found in {nwbf}\n')
            return
        for task in tasks_mod.data_interfaces.values():
            if isinstance(task, pynwb.core.DynamicTable):
                self.insert_from_task_table(task)

    def insert_from_task_table(self, task_table):
        """Insert tasks from a pynwb DynamicTable containing task metadata.

        Duplicate tasks will not be added.

        Parameters
        ----------
        task_table : pynwb.core.DynamicTable
            The table representing task metadata.
        """
        taskdf = task_table.to_dataframe()
        for task_entry in taskdf.iterrows():
            task_dict = dict()
            task_dict['task_name'] = task_entry[1].task_name
            task_dict['task_description'] = task_entry[1].task_description
            self.insert1(task_dict, skip_duplicates=True)

    @classmethod
    def check_task_table(cls, task_table):
        """Check whether the pynwb DynamicTable containing task metadata conforms to the expected format.

        The table should be an instance of pynwb.core.DynamicTable and contain the columns 'task_name' and
        'task_description'.

        Parameters
        ----------
        task_table : pynwb.core.DynamicTable
            The table representing task metadata.

        Returns
        -------
        bool
            Whether the DynamicTable conforms to the expected format for loading data into the Task table.
        """
        return (isinstance(task_table, pynwb.core.DynamicTable) and hasattr(task_table, 'task_name') and
                hasattr(task_table, 'task_description'))


@schema
class TaskEpoch(dj.Imported):
    # Tasks, session and time intervals
    definition = """
     -> Session
     epoch: int  # the session epoch for this task and apparatus (1-based)
     ---
     -> Task
     -> CameraDevice
     -> IntervalList
     """

    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile().get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        camera_name = dict()
        # the tasks refer to the camera_id which is unique for the NWB file but not for CameraDevice schema, so we
        # need to look up the right camera
        # map camera ID (in camera name) to camera_name
        for d in nwbf.devices:
            if 'camera_device' in d:
                device = nwbf.devices[d]
                # get the camera ID
                c = str.split(d)
                camera_name[int(c[1])] = device.camera_name

        # find the task modules and for each one, add the task to the Task schema if it isn't there
        # and then add an entry for each epoch
        tasks_mod = nwbf.processing.get('tasks')
        if tasks_mod is None:
            print(f'No tasks processing module found in {nwbf}\n')
            return
        for task in tasks_mod.data_interfaces.values():
            if self.check_task_table(task):
                # check if the task is in the Task table and if not, add it
                Task.insert_from_task_table(task)
                key['task_name'] = task.task_name[0]

                # get the camera used for this task. Use 'none' if there was no camera
                # TODO is there ever no camera?
                if hasattr(task, 'camera_id'):
                    camera_id = int(task.camera_id[0])
                    key['camera_name'] = camera_name[camera_id]
                else:
                    key['camera_name'] = CameraDevice.UNKNOWN

                # get the interval list for this task, which corresponds to the matching epoch for the raw data.
                # users should define more restrictive intervals as required for analyses
                session_intervals = (IntervalList() & {'nwb_file_name': nwb_file_name}).fetch('interval_list_name')
                for epoch in task.task_epochs[0]:
                    # TODO in beans file, task_epochs[0] is 1x2 dset of ints, so epoch would be an int
                    key['epoch'] = epoch
                    target_interval = str(epoch).zfill(2)
                    for interval in session_intervals:
                        if target_interval in interval:  # TODO this is not true for the beans file
                            break
                    # TODO case when interval is not found is not handled
                    key['interval_list_name'] = interval
                    self.insert1(key)

    @classmethod
    def check_task_table(cls, task_table):
        """Check whether the pynwb DynamicTable containing task metadata conforms to the expected format.

        The table should be an instance of pynwb.core.DynamicTable and contain the columns 'task_name',
        'task_description', 'camera_id', 'and 'task_epochs'.

        Parameters
        ----------
        task_table : pynwb.core.DynamicTable
            The table representing task metadata.

        Returns
        -------
        bool
            Whether the DynamicTable conforms to the expected format for loading data into the TaskEpoch table.
        """

        # TODO this could be more strict and check data types, but really it should be schematized
        return (Task.check_task_table(task_table) and hasattr(task_table, 'camera_id') and
                hasattr(task_table, 'task_epochs'))
