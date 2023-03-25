import datajoint as dj
import ndx_franklab_novela
import pynwb

from .common_device import CameraDevice
from .common_interval import IntervalList
from .common_nwbfile import Nwbfile
from .common_session import Session  # noqa: F401
from ..utils.nwb_helper_fn import get_nwb_file

schema = dj.schema("common_task")


@schema
class Task(dj.Manual):
    definition = """
     task_name: varchar(80)
     ---
     task_description = NULL: varchar(2000)    # description of this task
     task_type = NULL: varchar(2000)           # type of task
     task_subtype = NULL: varchar(2000)        # subtype of task
     """

    @classmethod
    def insert_from_nwbfile(cls, nwbf):
        """Insert tasks from an NWB file.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.
        """
        tasks_mod = nwbf.processing.get("tasks")
        if tasks_mod is None:
            print(f"No tasks processing module found in {nwbf}\n")
            return
        for task in tasks_mod.data_interfaces.values():
            if cls.check_task_table(task):
                cls.insert_from_task_table(task)

    @classmethod
    def insert_from_task_table(cls, task_table):
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
            task_dict["task_name"] = task_entry[1].task_name
            task_dict["task_description"] = task_entry[1].task_description
            cls.insert1(task_dict, skip_duplicates=True)

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
        return (
            isinstance(task_table, pynwb.core.DynamicTable)
            and hasattr(task_table, "task_name")
            and hasattr(task_table, "task_description")
        )


@schema
class TaskEpoch(dj.Imported):
    # Tasks, session and time intervals
    definition = """
     -> Session
     epoch: int  # the session epoch for this task and apparatus(1 based)
     ---
     -> Task
     -> [nullable] CameraDevice
     -> IntervalList
     task_environment = NULL: varchar(200)  # the environment the animal was in
     """

    def make(self, key):
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile().get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        camera_names = dict()
        # the tasks refer to the camera_id which is unique for the NWB file but not for CameraDevice schema, so we
        # need to look up the right camera
        # map camera ID (in camera name) to camera_name
        for device in nwbf.devices.values():
            if isinstance(device, ndx_franklab_novela.CameraDevice):
                # get the camera ID
                camera_id = int(str.split(device.name)[1])
                camera_names[camera_id] = device.camera_name

        # find the task modules and for each one, add the task to the Task schema if it isn't there
        # and then add an entry for each epoch
        tasks_mod = nwbf.processing.get("tasks")
        if tasks_mod is None:
            print(f"No tasks processing module found in {nwbf}\n")
            return

        for task in tasks_mod.data_interfaces.values():
            if self.check_task_table(task):
                # check if the task is in the Task table and if not, add it
                Task.insert_from_task_table(task)
                key["task_name"] = task.task_name[0]

                # get the CameraDevice used for this task (primary key is camera name so we need
                # to map from ID to name)
                camera_id = int(task.camera_id[0])
                if camera_id in camera_names:
                    key["camera_name"] = camera_names[camera_id]
                else:
                    print(
                        f"No camera device found with ID {camera_id} in NWB file {nwbf}\n"
                    )

                # Add task environment
                if hasattr(task, "task_environment"):
                    key["task_environment"] = task.task_environment[0]

                # get the interval list for this task, which corresponds to the matching epoch for the raw data.
                # Users should define more restrictive intervals as required for analyses
                session_intervals = (
                    IntervalList() & {"nwb_file_name": nwb_file_name}
                ).fetch("interval_list_name")
                for epoch in task.task_epochs[0]:
                    # TODO in beans file, task_epochs[0] is 1x2 dset of ints, so epoch would be an int
                    key["epoch"] = epoch
                    target_interval = str(epoch).zfill(2)
                    for interval in session_intervals:
                        if (
                            target_interval in interval
                        ):  # TODO this is not true for the beans file
                            break
                    # TODO case when interval is not found is not handled
                    key["interval_list_name"] = interval
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
        return (
            Task.check_task_table(task_table)
            and hasattr(task_table, "camera_id")
            and hasattr(task_table, "task_epochs")
        )
