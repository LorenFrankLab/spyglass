"""Ingestion of `ndx_structured_behavior` task recordings. See #1349.

The extension has no PyPI release, so it is not a Spyglass dependency and may
not be installed. These tests skip in that case; `common_task_rec` itself
falls back to matching NWB objects by class name, so ingestion of a file that
uses the extension does not require it to be importable.
"""

from pathlib import Path

import pytest

try:
    from ndx_structured_behavior import (
        ActionsTable,
        ActionTypesTable,
        EventTypesTable,
        StatesTable,
        StateTypesTable,
        Task,
        TaskArgumentsTable,
        TaskRecording,
        TrialsTable,
        add_event,
        create_events_table,
    )

    HAS_NDX_SB = True
except ImportError:
    HAS_NDX_SB = False

pytestmark = pytest.mark.skipif(
    not HAS_NDX_SB, reason="ndx_structured_behavior not installed"
)

ACTION_DESC = "the actions this task can take"
EVENT_DESC = "the events this task can see"
STATE_DESC = "the states this task can be in"

ACTION_NAMES = ["open_valve", "flash_led"]
EVENT_NAMES = ["tone_on", "port_poke"]
STATE_NAMES = ["iti", "cue", "response"]

ARGUMENT = dict(
    argument_name="reward_volume_ul",
    argument_description="volume delivered per correct trial",
    expression="5",
    expression_type="const",
    output_type="integer",
)


@pytest.fixture(scope="module")
def task_rec(common):
    """The `common_task_rec` module, imported once the database is up."""
    from spyglass.common import common_task_rec

    return common_task_rec


def _ingested(file_stem, with_trials, verbose_context):
    """Write and ingest a minimal structured-behavior file, then clean up.

    Parameters
    ----------
    file_stem : str
        Basename for the raw file, without extension.
    with_trials : bool
        Whether the file gets a trials table. A file without one leaves
        `TaskRecording.trials_object_id` null, which is the case that must not
        break fetching the tables it does have.
    verbose_context : context manager
        The suite's teardown-logging context.

    Yields
    ------
    dict
        Key of the ingested copy.
    """
    from pynwb import NWBHDF5IO
    from pynwb.testing.mock.file import mock_NWBFile, mock_Subject

    from spyglass.common import Nwbfile
    from spyglass.data_import import insert_sessions
    from spyglass.settings import raw_dir

    nwbfile = mock_NWBFile(
        identifier=f"structured_behavior_{file_stem}",
        session_description="Mock NWB file demonstrating TaskRecording import",
    )
    mock_Subject(nwbfile=nwbfile)

    # --- the task's vocabulary, read by TaskRecordingTypes and its parts
    action_types = ActionTypesTable(description=ACTION_DESC)
    for action_name in ACTION_NAMES:
        action_types.add_row(action_name=action_name)

    event_types = EventTypesTable(description=EVENT_DESC)
    for event_name in EVENT_NAMES:
        event_types.add_row(event_name=event_name)

    state_types = StateTypesTable(description=STATE_DESC)
    for state_name in STATE_NAMES:
        state_types.add_row(state_name=state_name)

    task_arguments = TaskArgumentsTable()
    task_arguments.add_row(**ARGUMENT)

    nwbfile.add_lab_meta_data(
        Task(
            event_types=event_types,
            state_types=state_types,
            action_types=action_types,
            task_arguments=task_arguments,
        )
    )

    # --- what was actually recorded, read by TaskRecording
    actions = ActionsTable(
        description="recorded actions", action_types_table=action_types
    )
    actions.add_action(action_type=0, timestamp=0.4, duration=0.1, value="open")
    actions.add_action(action_type=1, timestamp=0.5, duration=0.1, value="on")

    events = create_events_table(
        event_types_table=event_types, description="recorded events"
    )
    add_event(events, event_type=0, timestamp=0.4, duration=0.1, value="on")

    states = StatesTable(
        description="recorded states", state_types_table=state_types
    )
    states.add_state(state_type=0, start_time=0.0, stop_time=0.1)
    states.add_state(state_type=1, start_time=0.1, stop_time=0.3)

    if with_trials:
        trials = TrialsTable(
            description="recorded trials",
            states_table=states,
            events_table=events,
            actions_table=actions,
        )
        trials.add_trial(
            start_time=0.0,
            stop_time=0.8,
            states=[0, 1],
            events=[0],
            actions=[0, 1],
        )
        nwbfile.trials = trials

    nwbfile.add_acquisition(
        TaskRecording(actions=actions, states=states, events=events)
    )

    file_path = Path(raw_dir) / f"{file_stem}.nwb"
    nwb_dict = dict(nwb_file_name=f"{file_stem}_.nwb")
    file_path.unlink(missing_ok=True)

    with NWBHDF5IO(file_path, mode="w") as io:
        io.write(nwbfile)

    insert_sessions([str(file_path)], raise_err=True)

    yield nwb_dict

    with verbose_context:
        file_path.unlink(missing_ok=True)
        (Nwbfile & nwb_dict).delete(safemode=False)


@pytest.fixture(scope="module")
def import_task_recording_nwb(verbose_context):
    """A file with every table, trials included."""
    yield from _ingested("test_task_recording", True, verbose_context)


@pytest.fixture(scope="module")
def import_no_trials_nwb(verbose_context):
    """A file whose recording has no trials table, leaving that id null."""
    yield from _ingested("test_task_rec_no_trials", False, verbose_context)


def test_recording_types_descriptions(task_rec, import_task_recording_nwb):
    """The master row carries each type table's description."""
    entry = (task_rec.TaskRecordingTypes & import_task_recording_nwb).fetch1()

    assert entry["action_description"] == ACTION_DESC
    assert entry["event_description"] == EVENT_DESC
    assert entry["state_description"] == STATE_DESC


@pytest.mark.parametrize(
    "part_name, column, expected",
    [
        ("ActionTypes", "action_name", ACTION_NAMES),
        ("EventTypes", "event_name", EVENT_NAMES),
        ("StateTypes", "state_name", STATE_NAMES),
    ],
)
def test_type_parts(
    task_rec, import_task_recording_nwb, part_name, column, expected
):
    """Each type part holds one row per type, keyed by the source row id."""
    part = getattr(task_rec.TaskRecordingTypes, part_name)
    query = part & import_task_recording_nwb

    ids, names = query.fetch("id", column, order_by="id")
    assert list(ids) == list(range(len(expected)))
    assert list(names) == expected


def test_arguments_part(task_rec, import_task_recording_nwb):
    """The arguments part carries every declared column."""
    entry = (
        task_rec.TaskRecordingTypes.Arguments & import_task_recording_nwb
    ).fetch1()

    for column, value in ARGUMENT.items():
        assert entry[column] == value, f"{column} did not round-trip"


def test_recording_object_ids(task_rec, import_task_recording_nwb):
    """Every recorded table, and the file's trials, is referenced."""
    entry = (task_rec.TaskRecording & import_task_recording_nwb).fetch1()

    for column in (
        "actions_object_id",
        "events_object_id",
        "states_object_id",
        "trials_object_id",
    ):
        assert entry[column], f"{column} was not ingested"


@pytest.mark.parametrize(
    "table_name, n_rows",
    [("actions", 2), ("events", 1), ("states", 2), ("trials", 1)],
)
def test_fetch1_dataframe(
    task_rec, import_task_recording_nwb, table_name, n_rows
):
    """Each stored id resolves to the table it was read from."""
    query = task_rec.TaskRecording & import_task_recording_nwb
    df = query.fetch1_dataframe(table_name)

    assert len(df) == n_rows, f"{table_name} has {len(df)} rows, want {n_rows}"


def test_fetch1_dataframe_rejects_unknown(task_rec, import_task_recording_nwb):
    """A name this table does not store is refused, not looked up."""
    query = task_rec.TaskRecording & import_task_recording_nwb

    with pytest.raises(ValueError, match="Invalid table name"):
        query.fetch1_dataframe("trials_object_id")


def test_no_trials_is_null(task_rec, import_no_trials_nwb):
    """A file with no trials table stores a null id, not a failed ingest."""
    entry = (task_rec.TaskRecording & import_no_trials_nwb).fetch1()

    assert entry["trials_object_id"] is None
    assert entry["actions_object_id"], "actions should still be ingested"


@pytest.mark.parametrize("table_name", ["actions", "events", "states"])
def test_fetch1_dataframe_ignores_null_siblings(
    task_rec, import_no_trials_nwb, table_name
):
    """A null id must not break fetching the tables that were recorded.

    fetch_nwb resolves every `*_object_id` it is handed and guards only
    against `""`, so passing all four columns would look up
    `nwbf.objects[None]` and raise TypeError for a file with no trials.
    """
    query = task_rec.TaskRecording & import_no_trials_nwb

    assert len(query.fetch1_dataframe(table_name)) > 0


def test_fetch1_dataframe_missing_table(task_rec, import_no_trials_nwb):
    """Asking for a table this entry did not record raises, not KeyError."""
    query = task_rec.TaskRecording & import_no_trials_nwb

    with pytest.raises(ValueError, match="No trials recorded"):
        query.fetch1_dataframe("trials")
