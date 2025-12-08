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

    @classmethod
    def _get_valid_camera_names(cls, camera_ids, camera_names, context=""):
        """Get valid camera names for given camera IDs.

        Parameters
        ----------
        camera_ids : list
            List of camera IDs to validate
        camera_names : dict
            Mapping of camera ID to camera name
        context : str, optional
            Context string for warning message

        Returns
        -------
        list or None
            List of camera name dicts, or None if no valid cameras found
        """
        valid_camera_ids = [
            camera_id
            for camera_id in camera_ids
            if camera_id in camera_names.keys()
        ]
        if valid_camera_ids:
            return [
                {"camera_name": camera_names[camera_id]}
                for camera_id in valid_camera_ids
            ]
        if camera_ids:  # Only warn if camera_ids were specified
            logger.warning(
                f"No camera device found with ID {camera_ids}{context}\n"
            )
        return None

    @classmethod
    def _process_task_epochs(
        cls, base_key, task_epochs, nwb_file_name, session_intervals
    ):
        """Process task epochs and create TaskEpoch insert entries.

        Parameters
        ----------
        base_key : dict
            Base key dict with task_name, camera_names, etc.
        task_epochs : list
            List of epoch numbers/identifiers
        nwb_file_name : str
            Name of the NWB file
        session_intervals : list
            Available interval names from IntervalList

        Returns
        -------
        list
            List of dicts ready for TaskEpoch insertion
        """
        inserts = []
        for epoch in task_epochs:
            epoch_key = base_key.copy()
            epoch_key["epoch"] = epoch
            target_interval = cls.get_epoch_interval_name(
                epoch, session_intervals
            )
            if target_interval is None:
                continue
            epoch_key["interval_list_name"] = target_interval
            inserts.append(epoch_key)
        return inserts

    def make(self, key):
        """Populate TaskEpoch from the processing module in the NWB file."""
        nwb_file_name = key["nwb_file_name"]
        nwb_dict = dict(nwb_file_name=nwb_file_name)
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

        sess_intervals = (IntervalList & nwb_dict).fetch("interval_list_name")

        task_inserts = []  # inserts for Task table
        task_epoch_inserts = []  # inserts for TaskEpoch table
        for task_table in tasks_mod.data_interfaces.values():
            if not self.is_nwb_task_epoch(task_table):
                continue
            task_inserts.append(task_table)
            task_df = task_table.to_dataframe()
            for task in task_df.itertuples(index=False):
                key["task_name"] = task.task_name

                # Get valid camera names for this task
                camera_names_list = self._get_valid_camera_names(
                    task.camera_id,
                    camera_names,
                    context=f" in NWB file {nwbf}",
                )
                if camera_names_list:
                    key["camera_names"] = camera_names_list

                # Add task environment if present
                if hasattr(task, "task_environment"):
                    key["task_environment"] = task.task_environment

                # Process all epochs for this task
                task_epoch_inserts.extend(
                    self._process_task_epochs(
                        key, task.task_epochs, nwb_file_name, sess_intervals
                    )
                )

        # Add tasks from config
        for task in config_tasks:
            task_key = {
                **key,
                "task_name": task.get("task_name"),
                "task_environment": task.get("task_environment", None),
            }

            # Add cameras if specified
            camera_names_list = self._get_valid_camera_names(
                task.get("camera_id", []), camera_names
            )
            if camera_names_list:
                task_key["camera_names"] = camera_names_list

            # Process all epochs for this task
            task_epoch_inserts.extend(
                self._process_task_epochs(
                    task_key,
                    task.get("task_epochs", []),
                    nwb_file_name,
                    sess_intervals,
                )
            )

        # check if the task entries are in the Task table and if not, add it
        [
            Task().insert_from_task_table(task_table)
            for task_table in task_inserts
        ]
        self.insert(task_epoch_inserts, allow_direct_insert=True)

    @classmethod
    def get_epoch_interval_name(cls, epoch, session_intervals):
        """Get the interval name for a given epoch based on matching number.

        This method implements flexible matching to handle various epoch tag
        formats. It tries multiple formats to find a match:
        1. Exact match (e.g., "1")
        2. Two-digit zero-padded (e.g., "01")
        3. Three-digit zero-padded (e.g., "001")
        If multiple matches are found, the two-digit only match is prioritized if
        present. If no unique match is found, a warning is logged.

        Parameters
        ----------
        epoch : int or str
            The epoch number to search for
        session_intervals : list of str
            List of interval names from IntervalList

        Returns
        -------
        str or None
            The matching interval name, or None if no unique match is found

        Examples
        --------
        >>> session_intervals = ["1", "02", "003"]
        >>> TaskEpoch.get_epoch_interval_name(1, session_intervals)
        '1'
        >>> TaskEpoch.get_epoch_interval_name(2, session_intervals)
        '02'
        >>> TaskEpoch.get_epoch_interval_name(3, session_intervals)
        '003'
        """
        if epoch in session_intervals:
            return epoch

        two_digit_matches = [
            interval
            for interval in session_intervals
            if str(epoch).zfill(2) in interval
        ]
        if len(set(two_digit_matches)) == 1:
            return two_digit_matches[0]

        # Try multiple formats:
        possible_formats = [
            str(epoch),  # Try exact match first (e.g., "1")
            str(epoch).zfill(2),  # Try 2-digit zero-pad (e.g., "01")
            str(epoch).zfill(3),  # Try 3-digit zero-pad (e.g., "001")
        ]
        unique_formats = list(dict.fromkeys(possible_formats))

        # Find matches for any format, remove duplicates preserving order
        possible_targets = [
            interval
            for interval in session_intervals
            for target in unique_formats
            if target in interval
        ]

        if len(set(possible_targets)) == 1:
            return possible_targets[0]

        warn = "Multiple" if len(possible_targets) > 1 else "No"

        logger.warning(
            f"{warn} interval(s) found for epoch {epoch}. "
            f"Available intervals: {session_intervals}"
        )
        return None

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
