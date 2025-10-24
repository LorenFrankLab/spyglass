import datajoint as dj
import ndx_franklab_novela
import pynwb

from spyglass.common.common_device import CameraDevice  # noqa: F401
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.dj_helper_fn import accept_divergence
from spyglass.utils.nwb_helper_fn import get_config, get_nwb_file

schema = dj.schema("common_task")


@schema
class Task(SpyglassMixin, dj.Manual):
    definition = """
     task_name: varchar(80)
     ---
     task_description = NULL: varchar(2000)    # description of this task
     task_type = NULL: varchar(2000)           # type of task
     task_subtype = NULL: varchar(2000)        # subtype of task
     """

    def insert_from_nwbfile(self, nwbf: pynwb.NWBFile):
        """Insert tasks from an NWB file.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.
        """
        tasks_mod = nwbf.processing.get("tasks")
        if tasks_mod is None:
            logger.warning(f"No tasks processing module found in {nwbf}\n")
            return
        for task in tasks_mod.data_interfaces.values():
            if self.is_nwb_task_table(task):
                self.insert_from_task_table(task)

    def insert_from_task_table(self, task_table: pynwb.core.DynamicTable):
        """Insert tasks from a pynwb DynamicTable containing task metadata.

        Duplicate tasks will check for matching secondary keys and not be added.

        Parameters
        ----------
        task_table : pynwb.core.DynamicTable
            The table representing task metadata.
        """
        taskdf = task_table.to_dataframe()

        task_dicts = taskdf.apply(
            lambda row: dict(
                task_name=row.task_name,
                task_description=row.task_description,
            ),
            axis=1,
        ).tolist()

        # Check if the task is already in the table
        # if so check that the secondary keys all match
        def unequal_vals(key, a, b):
            a, b = a.get(key) or "", b.get(key, "") or ""
            return a != b  # prevent false positive on None != ""

        inserts = []
        for task_dict in task_dicts:
            query = self & {"task_name": task_dict["task_name"]}
            if not query:
                inserts.append(task_dict)  # only append novel tasks
                continue
            existing = query.fetch1()
            for key in set(task_dict).union(existing):
                if not unequal_vals(key, task_dict, existing):
                    continue  # skip if values are equal
                if not accept_divergence(
                    key,
                    task_dict.get(key),
                    existing.get(key),
                    self._test_mode,
                    self.camel_name,
                ):
                    # If the user does not accept the divergence,
                    # raise an error to prevent data inconsistency
                    raise ValueError(
                        f"Task {task_dict['task_name']} already exists "
                        + f"with different values for {key}: "
                        + f"{task_dict.get(key)} != {existing.get(key)}"
                    )
        # Insert the tasks into the table
        self.insert(inserts)

    @classmethod
    def is_nwb_task_table(cls, task_table: pynwb.core.DynamicTable) -> bool:
        """Check format of pynwb DynamicTable containing task metadata.

        The table should be an instance of pynwb.core.DynamicTable and contain
        the columns 'task_name' and 'task_description'.

        Parameters
        ----------
        task_table : pynwb.core.DynamicTable
            The table representing task metadata.

        Returns
        -------
        bool
            Whether the DynamicTable conforms to the expected format for loading
            data into the Task table.
        """
        return (
            isinstance(task_table, pynwb.core.DynamicTable)
            and hasattr(task_table, "task_name")
            and hasattr(task_table, "task_description")
        )


@schema
class TaskEpoch(SpyglassMixin, dj.Imported):
    # Tasks, session and time intervals
    definition = """
     -> Session
     epoch: int  # the session epoch for this task and apparatus(1 based)
     ---
     -> Task
     -> [nullable] CameraDevice
     -> IntervalList
     task_environment = NULL: varchar(200)  # the environment the animal was in
     camera_names : blob # list of keys corresponding to entry in CameraDevice
     """

    def make(self, key):
        """Populate TaskEpoch from the processing module in the NWB file."""
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile().get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        config = get_config(nwb_file_abspath, calling_table=self.camel_name)
        camera_names = dict()

        # the tasks refer to the camera_id which is unique for the NWB file but
        # not for CameraDevice schema, so we need to look up the right camera
        # map camera ID (in camera name) to camera_name

        for device in nwbf.devices.values():
            if isinstance(device, ndx_franklab_novela.CameraDevice):
                # get the camera ID
                camera_id = int(str.split(device.name)[1])
                camera_names[camera_id] = device.camera_name
        if device_list := config.get("CameraDevice"):
            for device in device_list:
                camera_names.update(
                    {
                        name: id
                        for name, id in zip(
                            device.get("camera_name"),
                            device.get("camera_id", -1),
                        )
                    }
                )

        # find the task modules and for each one, add the task to the Task
        # schema if it isn't there and then add an entry for each epoch

        tasks_mod = nwbf.processing.get("tasks")
        config_tasks = config.get("Tasks", [])
        if tasks_mod is None and (not config_tasks):
            logger.warning(
                f"No tasks processing module found in {nwbf} or config\n"
            )
            return

        task_inserts = []  # inserts for Task table
        task_epoch_inserts = []  # inserts for TaskEpoch table
        for task_table in tasks_mod.data_interfaces.values():
            if not self.is_nwb_task_epoch(task_table):
                continue
            task_inserts.append(task_table)
            task_df = task_table.to_dataframe()
            for task in task_df.itertuples(index=False):
                key["task_name"] = task.task_name

                # get the CameraDevice used for this task (primary key is
                # camera name so we need to map from ID to name)

                camera_ids = task.camera_id
                valid_camera_ids = [
                    camera_id
                    for camera_id in camera_ids
                    if camera_id in camera_names.keys()
                ]
                if valid_camera_ids:
                    key["camera_names"] = [
                        {"camera_name": camera_names[camera_id]}
                        for camera_id in valid_camera_ids
                    ]
                else:
                    logger.warning(
                        f"No camera device found with ID {camera_ids} in NWB "
                        + f"file {nwbf}\n"
                    )
                # Add task environment
                if hasattr(task, "task_environment"):
                    key["task_environment"] = task.task_environment

                # get the interval list for this task, which corresponds to the
                # matching epoch for the raw data. Users should define more
                # restrictive intervals as required for analyses

                session_intervals = (
                    IntervalList() & {"nwb_file_name": nwb_file_name}
                ).fetch("interval_list_name")
                for epoch in task.task_epochs:
                    key["epoch"] = epoch
                    target_interval = self.get_epoch_interval_name(
                        epoch, session_intervals
                    )
                    if target_interval is None:
                        logger.warning("Skipping epoch.")
                        continue
                    key["interval_list_name"] = target_interval
                    task_epoch_inserts.append(key.copy())

        # Add tasks from config
        for task in config_tasks:
            new_key = {
                **key,
                "task_name": task.get("task_name"),
                "task_environment": task.get("task_environment", None),
            }
            # add cameras
            camera_ids = task.get("camera_id", [])
            valid_camera_ids = [
                camera_id
                for camera_id in camera_ids
                if camera_id in camera_names.keys()
            ]
            if valid_camera_ids:
                new_key["camera_names"] = [
                    {"camera_name": camera_names[camera_id]}
                    for camera_id in valid_camera_ids
                ]
            session_intervals = (
                IntervalList() & {"nwb_file_name": nwb_file_name}
            ).fetch("interval_list_name")
            for epoch in task.get("task_epochs", []):
                new_key["epoch"] = epoch
                target_interval = self.get_epoch_interval_name(
                    epoch, session_intervals
                )
                if target_interval is None:
                    logger.warning("Skipping epoch.")
                    continue
                new_key["interval_list_name"] = target_interval
                task_epoch_inserts.append(key.copy())

        # check if the task entries are in the Task table and if not, add it
        [
            Task().insert_from_task_table(task_table)
            for task_table in task_inserts
        ]
        self.insert(task_epoch_inserts, allow_direct_insert=True)

    @classmethod
    def get_epoch_interval_name(cls, epoch, session_intervals):
        """Get the interval name for a given epoch based on matching number"""
        target_interval = str(epoch).zfill(2)
        possible_targets = [
            interval
            for interval in session_intervals
            if target_interval in interval
        ]
        if not possible_targets:
            logger.warning(f"Interval not found for epoch {epoch}.")
        elif len(possible_targets) > 1:
            logger.warning(
                f"Multiple intervals found for epoch {epoch}. "
                + f"matches are {possible_targets}."
            )
        else:
            return possible_targets[0]

    @classmethod
    def update_entries(cls, restrict=True):
        """Update entries in the TaskEpoch table based on a restriction."""
        existing_entries = (cls & restrict).fetch("KEY")
        for row in existing_entries:
            if (cls & row).fetch1("camera_names"):
                continue
            row["camera_names"] = [
                {"camera_name": (cls & row).fetch1("camera_name")}
            ]
            cls.update1(row=row)

    @classmethod
    def is_nwb_task_epoch(cls, task_table: pynwb.core.DynamicTable) -> bool:
        """Check format of pynwb DynamicTable containing task metadata.

        The table should be an instance of pynwb.core.DynamicTable and contain
        the columns 'task_name', 'task_description', 'camera_id', 'and
        'task_epochs'.

        Parameters
        ----------
        task_table : pynwb.core.DynamicTable
            The table representing task metadata.

        Returns
        -------
        bool
            Whether the DynamicTable conforms to the expected format for
            loading data into the TaskEpoch table.
        """

        return (
            Task.is_nwb_task_table(task_table)
            and hasattr(task_table, "camera_id")
            and hasattr(task_table, "task_epochs")
        )
