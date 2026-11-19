"""Ingestion of sleep scoring and seizure events stored in an NWB file.

Chronic EEG/EMG recordings are usually scored offline into fixed-length epochs
(for example Wake, NREM and REM every few seconds), and seizures are detected
as start/stop intervals. In NWB both are ``TimeIntervals`` tables, typically in
a processing module:

- a sleep scoring table has one row per scored epoch and a ``sleep_state``
  column with the state label;
- a seizure table is a ``TimeIntervals`` named ``seizure_events``, one row per
  seizure.

``SleepStateScoring`` and ``SeizureEvents`` keep one entry per table and one
part row per epoch or seizure, so states and seizures can be restricted in
queries. Other columns of the tables (e.g. a numeric score, or the window a row
belongs to) stay in the NWB file and come back with ``fetch1_dataframe()``.
Files without such tables are a no-op.
"""

import datajoint as dj
import pandas as pd

from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils.dj_mixin import SpyglassIngestion
from spyglass.utils.nwb_helper_fn import is_nwb_obj_type

schema = dj.schema("common_sleep")

SLEEP_STATE_COLUMN = "sleep_state"
SEIZURE_TABLE_NAME = "seizure_events"


def _fetch1_dataframe(table) -> pd.DataFrame:
    """The rows of one ingested TimeIntervals, read from the NWB file."""
    key = table.fetch1("KEY")  # enforce exactly one entry
    restricted = table & key
    try:
        # fetch_nwb() already returns a DynamicTable as a DataFrame
        return restricted.fetch_nwb()[0][table._nwb_object_attr]
    finally:
        restricted.close_nwb()


@schema
class SleepStateScoring(SpyglassIngestion, dj.Manual):
    definition = """
    # A sleep scoring table: a TimeIntervals with a sleep_state column
    -> Session
    scoring_name: varchar(80)          # name of the TimeIntervals in the file
    ---
    scoring_object_id: varchar(40)     # NWB object id, for fetch_nwb()
    description = "": varchar(2000)
    n_epochs: int unsigned             # number of scored epochs (rows)
    """

    class Epoch(SpyglassIngestion, dj.Part):
        definition = """
        # One scored epoch
        -> master
        epoch_index: int unsigned          # row of the TimeIntervals
        ---
        start_time: double                 # seconds
        stop_time: double                  # seconds
        sleep_state: varchar(80)           # state label, e.g. NREM
        """

        table_key_to_obj_attr = {}  # entries come from the master

    # FKs -> Session, not -> Nwbfile, so point fetch_nwb() at the file table.
    _nwb_table = Nwbfile
    _nwb_object_attr = "scoring"  # fetch_nwb() key, from scoring_object_id
    _source_nwb_object_type = "TimeIntervals"
    # Entries come from generate_entries_from_nwb_object
    table_key_to_obj_attr = {}

    def get_nwb_objects(self, nwb_file, nwb_file_name=None):
        """The TimeIntervals of the file that have a sleep_state column."""
        return [
            obj
            for obj in nwb_file.objects.values()
            if is_nwb_obj_type(obj, self._source_nwb_object_type)
            and SLEEP_STATE_COLUMN in obj.colnames
        ]

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Master and ``.Epoch`` entries for one sleep scoring table."""
        base_key = (base_key or {}).copy()
        master_key = dict(base_key, scoring_name=nwb_obj.name)
        starts = nwb_obj["start_time"].data[:]
        stops = nwb_obj["stop_time"].data[:]
        states = nwb_obj[SLEEP_STATE_COLUMN].data[:]
        return {
            self: [
                dict(
                    master_key,
                    scoring_object_id=nwb_obj.object_id,
                    description=nwb_obj.description or "",
                    n_epochs=len(starts),
                )
            ],
            self.Epoch: [
                dict(
                    master_key,
                    epoch_index=index,
                    start_time=float(start),
                    stop_time=float(stop),
                    sleep_state=str(state),
                )
                for index, (start, stop, state) in enumerate(
                    zip(starts, stops, states)
                )
            ],
        }

    def fetch1_dataframe(self) -> pd.DataFrame:
        """All columns of one scoring table, one row per epoch."""
        return _fetch1_dataframe(self)

    def fetch_epochs_dataframe(self) -> pd.DataFrame:
        """The scored epochs of the restricted entries, from the database."""
        return (
            (self * self.Epoch)
            .proj("start_time", "stop_time", "sleep_state")
            .fetch(format="frame")
            .reset_index()
        )


@schema
class SeizureEvents(SpyglassIngestion, dj.Manual):
    definition = """
    # A seizure table: a TimeIntervals named seizure_events
    -> Session
    seizure_table_name: varchar(80)    # name of the TimeIntervals in the file
    ---
    seizure_object_id: varchar(40)     # NWB object id, for fetch_nwb()
    description = "": varchar(2000)
    n_events: int unsigned             # number of seizures (rows)
    """

    class Event(SpyglassIngestion, dj.Part):
        definition = """
        # One seizure
        -> master
        event_index: int unsigned          # row of the TimeIntervals
        ---
        start_time: double                 # seconds
        stop_time: double                  # seconds
        """

        table_key_to_obj_attr = {}  # entries come from the master

    # FKs -> Session, not -> Nwbfile, so point fetch_nwb() at the file table.
    _nwb_table = Nwbfile
    _nwb_object_attr = "seizure"  # fetch_nwb() key, from seizure_object_id
    _source_nwb_object_type = "TimeIntervals"
    _source_nwb_object_name = SEIZURE_TABLE_NAME
    # Entries come from generate_entries_from_nwb_object
    table_key_to_obj_attr = {}

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Master and ``.Event`` entries for one seizure table."""
        base_key = (base_key or {}).copy()
        master_key = dict(base_key, seizure_table_name=nwb_obj.name)
        starts = nwb_obj["start_time"].data[:]
        stops = nwb_obj["stop_time"].data[:]
        return {
            self: [
                dict(
                    master_key,
                    seizure_object_id=nwb_obj.object_id,
                    description=nwb_obj.description or "",
                    n_events=len(starts),
                )
            ],
            self.Event: [
                dict(
                    master_key,
                    event_index=index,
                    start_time=float(start),
                    stop_time=float(stop),
                )
                for index, (start, stop) in enumerate(zip(starts, stops))
            ],
        }

    def fetch1_dataframe(self) -> pd.DataFrame:
        """All columns of one seizure table, one row per seizure."""
        return _fetch1_dataframe(self)

    def fetch_events_dataframe(self) -> pd.DataFrame:
        """The seizures of the restricted entries, from the database."""
        return (
            (self * self.Event)
            .proj("start_time", "stop_time")
            .fetch(format="frame")
            .reset_index()
        )
