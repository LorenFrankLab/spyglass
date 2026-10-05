"""Ingest `ndx_structured_behavior` task recordings into Spyglass.

Tables
------
- `TaskRecordingTypes`: the vocabulary a structured-behavior task declares --
    its action, event and state types, and its arguments. Read from the
    `Task` object in `nwbf.lab_meta_data`.
- `TaskRecording`: object ids for the recorded actions, events and states,
    plus the file's trials table. Read from the `TaskRecording` object in
    `nwbf.acquisition`.

The extension has no PyPI release, so it cannot be pinned in `pyproject.toml`
and may be missing or unusable. It is imported inside a `try`, and each file is
gated on its own cached spec version at ingestion.

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

from importlib.util import find_spec

import datajoint as dj
from packaging.version import Version

from spyglass.common.common_nwbfile import Nwbfile
from spyglass.utils import SpyglassIngestion, logger

_EXTENSION_NAME = "ndx-structured-behavior"
# Exact, not a floor: the mappings below name the columns of one spec, and this
# extension is pre-1.0 -- 0.1.0 named different ones.
_EXTENSION_VERSION = "0.2.0"

try:
    from ndx_structured_behavior import (
        ActionTypesTable,
        EventTypesTable,
        StateTypesTable,
        Task,
        TaskArgumentsTable,
    )
    from ndx_structured_behavior import TaskRecording as NwbTaskRecording
except ImportError as err:
    # Fall back to matching on class name: pynwb rebuilds these classes from
    # the namespace cached in each file, so ingestion works without the
    # package. Not being installed is the ordinary case and stays quiet; a
    # module that is present but still fails to import is a real problem worth
    # naming -- 0.2.0 reads `pynwb.event`, which needs pynwb >= 4.1.
    if find_spec("ndx_structured_behavior"):
        logger.warning(f"Matching {_EXTENSION_NAME} objects by name: {err}")
    ActionTypesTable = "ActionTypesTable"
    EventTypesTable = "EventTypesTable"
    StateTypesTable = "StateTypesTable"
    Task = "Task"
    TaskArgumentsTable = "TaskArgumentsTable"
    NwbTaskRecording = "TaskRecording"

schema = dj.schema("common_task_rec")


class _StructuredBehaviorIngestion(SpyglassIngestion):
    """Shared by every table here, which all gate on the same spec version."""

    def check_extension_requirements(self, nwb_file_name: str) -> bool:
        """Whether a file's cached spec is the one these mappings target.

        Replaces the mixin's check because it sets an exact match for only
        tables in this schema.

        Parameters
        ----------
        nwb_file_name : str
            The file about to be ingested.

        Returns
        -------
        bool
            True if the file caches exactly `_EXTENSION_VERSION`. A file with
            no such namespace returns False silently -- having no structured
            behavior is ordinary, not a problem to report.
        """
        from spyglass.utils.nwb_hash import get_file_namespaces

        found = get_file_namespaces(Nwbfile().get_abs_path(nwb_file_name)).get(
            _EXTENSION_NAME
        )

        if found and Version(found) == Version(_EXTENSION_VERSION):
            return True

        if found:
            self._warn_msg(
                f"{nwb_file_name} declares {_EXTENSION_NAME} {found}, not "
                + f"{_EXTENSION_VERSION}. Skipping {self.camel_name}."
            )

        return False


def _attr_of(group_name: str, attr: str):
    """Mapping callable reading `<group_name>.<attr>` off a parent object.

    Every sub-table of `Task` and of `TaskRecording` is optional -- a file may
    declare states but no actions -- and the nested-object form of
    `table_key_to_obj_attr` raises when the named object is absent.

    Parameters
    ----------
    group_name : str
        Attribute of the parent holding the sub-table.
    attr : str
        Attribute to read off that sub-table.

    Returns
    -------
    callable
        Takes the parent object, returns the value or None. Named for the
        pair, so the ingestion-mapping docs read `action_types.description`.
    """

    def get_attr(parent_obj):
        return getattr(getattr(parent_obj, group_name, None), attr, None)

    get_attr.__name__ = f"{group_name}.{attr}"
    return get_attr


@schema
class TaskRecordingTypes(_StructuredBehaviorIngestion, dj.Manual):
    definition = """
    # Types declared by an ndx_structured_behavior task
    -> Nwbfile
    ---
    action_description=NULL : varchar(255)  # Description of action types
    event_description=NULL  : varchar(255)  # Description of event types
    state_description=NULL  : varchar(255)  # Description of state types
    """

    _source_nwb_object_type = Task

    table_key_to_obj_attr = {
        "self": {
            "action_description": _attr_of("action_types", "description"),
            "event_description": _attr_of("event_types", "description"),
            "state_description": _attr_of("state_types", "description"),
        }
    }

    class ActionTypes(_StructuredBehaviorIngestion, dj.Part):
        definition = """
        -> TaskRecordingTypes
        id: int unsigned  # Unique identifier for the action type
        ---
        action_name : varchar(32)  # Action type name
        """

        _source_nwb_object_type = ActionTypesTable

        # `Index` is the row id: SpyglassIngestion expands a DynamicTable with
        # `to_dataframe().itertuples()`, which names the index `Index`.
        table_key_to_obj_attr = {
            "self": {"id": "Index", "action_name": "action_name"}
        }

    class EventTypes(_StructuredBehaviorIngestion, dj.Part):
        definition = """
        -> TaskRecordingTypes
        id : int unsigned  # Unique identifier for the event type
        ---
        event_name : varchar(32)  # Event type name
        """

        _source_nwb_object_type = EventTypesTable

        table_key_to_obj_attr = {
            "self": {"id": "Index", "event_name": "event_name"}
        }

    class StateTypes(_StructuredBehaviorIngestion, dj.Part):
        definition = """
        -> TaskRecordingTypes
        id : int unsigned  # Unique identifier for the state type
        ---
        state_name : varchar(32) # State type name
        """

        _source_nwb_object_type = StateTypesTable

        table_key_to_obj_attr = {
            "self": {"id": "Index", "state_name": "state_name"}
        }

    class Arguments(_StructuredBehaviorIngestion, dj.Part):
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
class TaskRecording(_StructuredBehaviorIngestion, dj.Manual):
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
                "actions_object_id": _attr_of("actions", "object_id"),
                "events_object_id": _attr_of("events", "object_id"),
                "states_object_id": _attr_of("states", "object_id"),
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

        # Every column here is nullable, and fetch_nwb resolves each
        # `<name>_object_id` it is handed -- guarding only against `""`, so a
        # NULL id reaches `nwbf.objects[None]` and raises TypeError. Ask for
        # the one id wanted, so a recording that stored no trials table can
        # still return its actions.
        id_attr = f"{table_name}_object_id"
        if not self.fetch1(id_attr):
            raise ValueError(
                f"No {table_name} recorded for {self.fetch1('KEY')}."
            )

        # fetch_nwb resolves the id to the table it points at, already as a
        # DataFrame, keyed by the attribute name without its `_object_id`. The
        # primary key rides along because fetch_nwb resolves the file path
        # from it; only the `_object_id` attrs are turned into objects.
        return self.fetch_nwb(*self.primary_key, id_attr)[0][table_name]
