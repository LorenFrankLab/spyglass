"""Ingest `ndx_structured_behavior` task recordings into Spyglass.

Tables
------
- `TaskRecordingTypes`: the vocabulary a structured-behavior task declares --
    its action, event and state types, and its arguments. Read from the
    `Task` object in `nwbf.lab_meta_data`.
- `TaskRecording`: object ids for the recorded actions, events and states,
    plus the file's trials table. Read from the `TaskRecording` object in
    `nwbf.acquisition`.

The extension is not a Spyglass dependency: it has no PyPI release yet, so it
cannot be pinned in `pyproject.toml` and is imported here inside a `try`. A
file that does not use it simply has none of these objects. Each table also
declares `_extension_requirements`, so a file written against an older version
of the schema is skipped with a warning rather than raising.

Example use
-----------
```python
from spyglass.common import Nwbfile
from spyglass.common.common_task_rec import TaskRecording, TaskRecordingTypes

nwb_file_name = "beadl_light_chasing_task.nwb"

TaskRecordingTypes().insert_from_nwbfile(nwb_file_name)
for part in (
    TaskRecordingTypes.ActionTypes,
    TaskRecordingTypes.EventTypes,
    TaskRecordingTypes.StateTypes,
    TaskRecordingTypes.Arguments,
):
    part().insert_from_nwbfile(nwb_file_name)
TaskRecording().insert_from_nwbfile(nwb_file_name)

actions_df = (TaskRecording & {"nwb_file_name": nwb_file_name}).fetch1_dataframe(
    "actions"
)
```

`populate_all_common` runs the same sequence, so ingesting a session through
`insert_sessions` fills these tables without the calls above.
"""

import datajoint as dj

from spyglass.common.common_nwbfile import Nwbfile
from spyglass.utils import SpyglassIngestion

try:  # ndx_structured_behavior has no PyPI release, so it cannot be pinned
    from ndx_structured_behavior import (
        ActionTypesTable,
        EventTypesTable,
        StateTypesTable,
        Task,
        TaskArgumentsTable,
    )
    from ndx_structured_behavior import TaskRecording as NwbTaskRecording
except ImportError:
    # Fall back to matching on class name. .
    ActionTypesTable = "ActionTypesTable"
    EventTypesTable = "EventTypesTable"
    StateTypesTable = "StateTypesTable"
    Task = "Task"
    TaskArgumentsTable = "TaskArgumentsTable"
    NwbTaskRecording = "TaskRecording"

schema = dj.schema("common_task_rec")

# Minimum schema version. The 0.2.0 spec is the first to name the columns
# these tables map, so an older file cannot be read by this mapping.
_EXTENSION = {"ndx-structured-behavior": "0.2.0"}

# Sub-tables of the lab_meta_data `Task` object, and of the `TaskRecording`
# object, are optional. A file may declare states but no actions.


def _description_of(group_name: str):
    """Build a mapping callable reading a sub-table's description.

    The generic nested-object form of `table_key_to_obj_attr` raises when the
    named object is absent, and every one of these groups is optional, so the
    lookup is done in a callable that tolerates a missing group.

    Parameters
    ----------
    group_name : str
        Attribute of the `Task` object holding the sub-table.

    Returns
    -------
    callable
        Takes the `Task` object, returns the group's description or None.
    """

    def get_description(task_obj):
        return getattr(getattr(task_obj, group_name, None), "description", None)

    get_description.__name__ = f"{group_name}.description"
    return get_description


def _object_id_of(group_name: str):
    """Build a mapping callable reading a sub-object's `object_id`.

    Parameters
    ----------
    group_name : str
        Attribute of the `TaskRecording` object holding the sub-table.

    Returns
    -------
    callable
        Takes the `TaskRecording` object, returns its id or None.
    """

    def get_object_id(recording_obj):
        return getattr(
            getattr(recording_obj, group_name, None), "object_id", None
        )

    get_object_id.__name__ = f"{group_name}.object_id"
    return get_object_id


@schema
class TaskRecordingTypes(SpyglassIngestion, dj.Manual):
    definition = """
    # Types declared by an ndx_structured_behavior task
    -> Nwbfile
    ---
    action_description=NULL : varchar(255)  # Description of action types
    event_description=NULL  : varchar(255)  # Description of event types
    state_description=NULL  : varchar(255)  # Description of state types
    """

    _source_nwb_object_type = Task
    _extension_requirements = _EXTENSION

    table_key_to_obj_attr = {
        "self": {
            "action_description": _description_of("action_types"),
            "event_description": _description_of("event_types"),
            "state_description": _description_of("state_types"),
        }
    }

    class ActionTypes(SpyglassIngestion, dj.Part):
        definition = """
        -> TaskRecordingTypes
        id: int unsigned  # Unique identifier for the action type
        ---
        action_name : varchar(32)  # Action type name
        """

        _source_nwb_object_type = ActionTypesTable
        _extension_requirements = _EXTENSION

        # `Index` is the row id: SpyglassIngestion expands a DynamicTable with
        # `to_dataframe().itertuples()`, which names the index `Index`.
        table_key_to_obj_attr = {
            "self": {"id": "Index", "action_name": "action_name"}
        }

    class EventTypes(SpyglassIngestion, dj.Part):
        definition = """
        -> TaskRecordingTypes
        id : int unsigned  # Unique identifier for the event type
        ---
        event_name : varchar(32)  # Event type name
        """

        _source_nwb_object_type = EventTypesTable
        _extension_requirements = _EXTENSION

        table_key_to_obj_attr = {
            "self": {"id": "Index", "event_name": "event_name"}
        }

    class StateTypes(SpyglassIngestion, dj.Part):
        definition = """
        -> TaskRecordingTypes
        id : int unsigned  # Unique identifier for the state type
        ---
        state_name : varchar(32) # State type name
        """

        _source_nwb_object_type = StateTypesTable
        _extension_requirements = _EXTENSION

        table_key_to_obj_attr = {
            "self": {"id": "Index", "state_name": "state_name"}
        }

    class Arguments(SpyglassIngestion, dj.Part):
        definition = """
        -> TaskRecordingTypes
        argument_name             : varchar(255)  # Argument name
        ---
        argument_description=NULL : varchar(255)
        expression=NULL           : varchar(127)
        expression_type=NULL      : varchar(32)
        output_type               : varchar(32)
        """

        _source_nwb_object_type = TaskArgumentsTable
        _extension_requirements = _EXTENSION

        # Keyed by argument name, so the row id is not stored.
        table_key_to_obj_attr = {
            "self": {
                "argument_name": "argument_name",
                "argument_description": "argument_description",
                "expression": "expression",
                "expression_type": "expression_type",
                "output_type": "output_type",
            }
        }


@schema
class TaskRecording(SpyglassIngestion, dj.Manual):
    definition = """
    # Object ids for one ndx_structured_behavior task recording
    -> TaskRecordingTypes
    ---
    actions_object_id=NULL : varchar(40)
    events_object_id=NULL  : varchar(40)
    states_object_id=NULL  : varchar(40)
    trials_object_id=NULL  : varchar(40)
    """

    _nwb_table = Nwbfile
    _source_nwb_object_type = NwbTaskRecording
    _extension_requirements = _EXTENSION

    @property
    def table_key_to_obj_attr(self):
        """Map the recording's sub-tables, plus the file's trials table.

        `trials` hangs off the NWB file rather than off the recording, so it
        is read from the id cached by `get_nwb_objects` rather than from the
        object being mapped. Declared here, and not set in an override, so
        that the ingestion-mapping doc generator sees all four columns.
        """
        return {
            "self": {
                "actions_object_id": _object_id_of("actions"),
                "events_object_id": _object_id_of("events"),
                "states_object_id": _object_id_of("states"),
                "trials_object_id": self._trials_object_id,
            }
        }

    _trials_id = None  # set per file by get_nwb_objects

    def get_nwb_objects(self, nwb_file, nwb_file_name=None):
        """Return the file's recording objects, noting its trials table.

        Parameters
        ----------
        nwb_file : pynwb.NWBFile
            The source file.
        nwb_file_name : str, optional
            Name of the source file. Unused; kept for the mixin's signature.

        Returns
        -------
        list
            The file's `TaskRecording` objects.
        """
        self._trials_id = getattr(
            getattr(nwb_file, "trials", None), "object_id", None
        )
        return super().get_nwb_objects(nwb_file, nwb_file_name)

    def _trials_object_id(self, recording_obj):
        """Return the trials id cached for the file being ingested.

        Parameters
        ----------
        recording_obj : object
            The `TaskRecording` object being mapped. Unused: trials is a
            sibling of the recording, not a member of it.

        Returns
        -------
        str or None
            Object id of the file's trials table, if it has one.
        """
        return self._trials_id

    def fetch1_dataframe(self, table_name: str):
        """Fetch one of the recorded tables as a DataFrame.

        Parameters
        ----------
        table_name : str
            One of 'actions', 'events', 'states' or 'trials'.

        Returns
        -------
        pandas.DataFrame
            The named table.

        Raises
        ------
        ValueError
            If `table_name` is not one this table stores, or if this entry did
            not record it.
        """
        valid = ("actions", "events", "states", "trials")
        if table_name not in valid:
            raise ValueError(
                f"Invalid table name: {table_name}. Expected one of {valid}."
            )

        _ = self.ensure_single_entry()
        # fetch_nwb resolves each `<name>_object_id` attribute to the table it
        # points at, already as a DataFrame. A null id yields no key at all.
        table = self.fetch_nwb()[0].get(table_name)
        if table is None:
            raise ValueError(
                f"No {table_name} recorded for {self.fetch1('KEY')}."
            )

        return table
