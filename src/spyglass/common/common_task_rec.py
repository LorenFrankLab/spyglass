"""Schema to ingest ndx_structured_behavior data into Spyglass.

Tables:
- TaskRecordingTypes: Stores metadata about the types of task recordings
    (actions, events, states, arguments).
- TaskRecording: Stores metadata about specific task recordings, including
    references to the actual data stored in the NWB file.

Example use:
```python
if __name__ == "__main__":
    from pathlib import Path

    from spyglass.settings import raw_dir

    nwb_file_name = "beadl_light_chasing_task.nwb"
    nwb_dict = dict(nwb_file_name=nwb_file_name)

    data_path = Path(raw_dir) / nwb_file_name
    if not data_path.exists():
        raise FileNotFoundError(
            f"Example NWB file not found at {data_path}. "
            + "Please run ndx-structured-behavior/src/pynwb/tests/example.py"
            + " to generate, and move it to the raw_dir."
        )

    # Example usage
    nwbf = get_nwb_file(nwb_file_name)
    if not Nwbfile() & nwb_dict:
        _ = Nwbfile().insert_from_relative_file_name(nwb_file_name)

    rec_types = TaskRecordingTypes()
    if not rec_types & nwb_dict:
        rec_types.insert_from_nwbfile(nwbf)

    task_rec = TaskRecording()
    if not task_rec & nwb_dict:
        task_rec.insert_from_nwbfile(nwbf)

    # Fetch actions DataFrame
    actions_df = task_rec.fetch1_dataframe("actions")
    print(actions_df.head())
```
"""

from pathlib import Path

import datajoint as dj
import pynwb

from spyglass.common import IntervalList, Nwbfile  # noqa: F401
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger
from spyglass.utils.nwb_helper_fn import get_nwb_file

schema = dj.schema("common_task_rec")


@schema
class TaskRecordingTypes(SpyglassMixin, dj.Manual):
    """Table to store task recording types."""

    definition = """
    # Task recording types
    -> Nwbfile
    ---
    action_description=NULL : varchar(255)  # Description of action types
    event_description=NULL  : varchar(255)  # Description of event types
    state_description=NULL  : varchar(255)  # Description of state types
    """

    # TODO: Convert to SpyglassIngestion pending #1377

    class ActionTypes(SpyglassMixinPart):
        """Table to store action types for task recording."""

        definition = """
        -> TaskRecordingTypes
        id: int unsigned  # Unique identifier for the action type
        ---
        action_name : varchar(32)  # Action type name
        """

    class EventTypes(SpyglassMixinPart):
        """Table to store event types for task recording."""

        definition = """
        -> TaskRecordingTypes
        id : int unsigned  # Unique identifier for the event type
        ---
        event_name : varchar(32)  # Event type name
        """

    class StateTypes(SpyglassMixinPart):
        """Table to store state types for task recording."""

        definition = """
        -> TaskRecordingTypes
        id : int unsigned  # Unique identifier for the state type
        ---
        state_name : varchar(32) # State type name
        """

    class Arguments(SpyglassMixinPart):
        """Table to store arguments for task recording."""

        definition = """
        -> TaskRecordingTypes
        argument_name             : varchar(255)  # Argument name
        ---
        argument_description=NULL : varchar(255)
        expression=NULL           : varchar(127)
        expression_type=NULL      : varchar(32)
        output_type               : varchar(32)
        """

    def _extract_types(
        self,
        master_key: dict,
        sub_table: pynwb.core.DynamicTable,
        reset_index: bool = True,
    ):
        """Extract columns from a DynamicTable."""
        df = sub_table.to_dataframe()
        if reset_index:
            df = df.reset_index()

        return [{**master_key, **row} for row in df.to_dict("records")]

    def insert_from_nwbfile(self, nwbf: pynwb.NWBFile):
        """Insert task recording types from an NWB file.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.
        """
        task_info = nwbf.fields.get("lab_meta_data", dict()).get("task")
        if not task_info:
            logger.warning(
                "No task information found in NWB file lab_meta_data. "
                f"Skipping: {nwb_file_name}"
            )
            return

        nwb_file_name = Path(nwbf.get_read_io().source).name
        master_key = dict(nwb_file_name=nwb_file_name)
        self_insert = master_key.copy()

        action_inserts = []
        if action_types := task_info.fields.get("action_types"):
            self_insert["action_description"] = action_types.description
            action_inserts = self._extract_types(master_key, action_types)

        event_inserts = []
        if event_types := task_info.fields.get("event_types"):
            self_insert["event_description"] = event_types.description
            event_inserts = self._extract_types(master_key, event_types)

        state_inserts = []
        if state_types := task_info.fields.get("state_types"):
            self_insert["state_description"] = state_types.description
            state_inserts = self._extract_types(master_key, state_types)

        argument_inserts = []
        if arg_types := task_info.fields.get("task_arguments"):
            argument_inserts = self._extract_types(
                master_key, arg_types, reset_index=False
            )

        self.insert1(self_insert)
        self.ActionTypes.insert(action_inserts)
        self.EventTypes.insert(event_inserts)
        self.StateTypes.insert(state_inserts)
        self.Arguments.insert(argument_inserts)


@schema
class TaskRecording(SpyglassMixin, dj.Manual):
    """Table to store task recording metadata."""

    definition = """
    -> TaskRecordingTypes
    ---
    actions_object_id=NULL : varchar(40)
    events_object_id=NULL  : varchar(40)
    states_object_id=NULL  : varchar(40)
    trials_object_id=NULL  : varchar(40)
    """

    _nwb_table = Nwbfile

    # TODO: Convert to SpyglassIngestion pending #1377

    def insert_from_nwbfile(self, nwbf: pynwb.NWBFile):
        """Insert task recording from an NWB file.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.
        """
        nwb_file_name = Path(nwbf.get_read_io().source).name
        nwb_dict = dict(nwb_file_name=nwb_file_name)

        # Check if TaskRecordingTypes entry exists. Attempt insert or return.
        types_tbl = TaskRecordingTypes()
        if not types_tbl & nwb_dict:
            types_tbl.insert_from_nwbfile(nwb_file_name, nwbf)
        if not types_tbl & nwb_dict:
            logger.warning(
                f"TaskRecordingTypes not found for {nwb_file_name}. "
                "Skipping TaskRecording insertion."
            )
            return

        self_insert = nwb_dict.copy()
        rec_fields = nwbf.acquisition["task_recording"].fields
        for table_name in ["actions", "events", "states"]:
            table_obj = rec_fields.get(table_name)
            if not table_obj:
                continue
            self_insert[f"{table_name}_object_id"] = table_obj.object_id

        if trials := nwbf.fields.get("trials"):
            self_insert["trials_object_id"] = trials.object_id

        self.insert1(self_insert)

    def fetch1_dataframe(self, table_name: str):
        """Fetch a DataFrame for a specific table name."""
        if table_name not in ["actions", "events", "states", "trials"]:
            raise ValueError(f"Invalid table name: {table_name}")

        _ = self.ensure_single_entry()
        return self.fetch_nwb()[0][table_name]
