"""Test ingestion of sleep scoring and seizure ``TimeIntervals`` tables.

A small file holds, in a ``behavior`` processing module:

- ``sleep_states``: three scored epochs with a ``sleep_state`` column and a
  ``sleep_score`` column that stays in the file;
- ``seizure_events``: two seizures with a ``duration`` column;
- ``other_intervals``: a TimeIntervals that is neither, so is not ingested.

It verifies the entries and part rows of ``SleepStateScoring`` and
``SeizureEvents``, that the other table and the epochs table are skipped, and
that ``fetch1_dataframe()`` returns every column from the file.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject

EPOCHS = [(0.0, 5.0, "Wake", 0), (5.0, 10.0, "NREM", 1), (10.0, 15.0, "REM", 2)]
SEIZURES = [(2.5, 4.0), (11.0, 13.5)]


@pytest.fixture(scope="module")
def sleep_nwb_key(verbose_context):
    from spyglass.common import Nwbfile
    from spyglass.data_import import insert_sessions
    from spyglass.settings import raw_dir
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    nwbfile = NWBFile(
        session_description="test sleep ingestion",
        identifier="test_sleep",
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    nwbfile.subject = Subject(
        subject_id="test_sleep_subject", species="Rattus norvegicus"
    )
    nwbfile.add_epoch(start_time=0.0, stop_time=15.0, tags=["recording"])

    sleep_states = TimeIntervals(
        name="sleep_states", description="test sleep scoring"
    )
    sleep_states.add_column(name="sleep_state", description="state label")
    sleep_states.add_column(name="sleep_score", description="state code")
    for start, stop, state, score in EPOCHS:
        sleep_states.add_row(
            start_time=start,
            stop_time=stop,
            sleep_state=state,
            sleep_score=score,
        )

    seizures = TimeIntervals(name="seizure_events", description="test seizures")
    seizures.add_column(name="duration", description="seconds")
    for start, stop in SEIZURES:
        seizures.add_row(
            start_time=start, stop_time=stop, duration=stop - start
        )

    other = TimeIntervals(name="other_intervals", description="not ingested")
    other.add_row(start_time=1.0, stop_time=2.0)

    behavior = nwbfile.create_processing_module("behavior", "behavior")
    for table in (sleep_states, seizures, other):
        behavior.add(table)

    raw_file_name = "test_sleep.nwb"
    nwb_path = Path(raw_dir) / raw_file_name
    key = dict(nwb_file_name=get_nwb_copy_filename(raw_file_name))
    if (Nwbfile & key) or nwb_path.exists():
        (Nwbfile & key).delete(safemode=False)
        nwb_path.unlink(missing_ok=True)
    with NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)

    insert_sessions(raw_file_name, raise_err=True)

    yield key

    with verbose_context:
        (Nwbfile & key).delete(safemode=False)
        nwb_path.unlink(missing_ok=True)


def test_sleep_state_scoring(common, sleep_nwb_key):
    scoring = common.SleepStateScoring & sleep_nwb_key
    entry = scoring.fetch1()
    assert entry["scoring_name"] == "sleep_states"
    assert entry["description"] == "test sleep scoring"
    assert entry["n_epochs"] == len(EPOCHS)

    epochs = scoring.fetch_epochs_dataframe().sort_values("epoch_index")
    np.testing.assert_allclose(epochs["start_time"], [e[0] for e in EPOCHS])
    np.testing.assert_allclose(epochs["stop_time"], [e[1] for e in EPOCHS])
    assert list(epochs["sleep_state"]) == [e[2] for e in EPOCHS]

    nrem = common.SleepStateScoring.Epoch & sleep_nwb_key & 'sleep_state="NREM"'
    assert len(nrem) == 1


def test_sleep_state_scoring_dataframe(common, sleep_nwb_key):
    df = (common.SleepStateScoring & sleep_nwb_key).fetch1_dataframe()
    assert list(df["sleep_score"]) == [e[3] for e in EPOCHS]


def test_seizure_events(common, sleep_nwb_key):
    seizures = common.SeizureEvents & sleep_nwb_key
    entry = seizures.fetch1()
    assert entry["seizure_table_name"] == "seizure_events"
    assert entry["n_events"] == len(SEIZURES)

    events = seizures.fetch_events_dataframe().sort_values("event_index")
    np.testing.assert_allclose(events["start_time"], [s[0] for s in SEIZURES])
    np.testing.assert_allclose(events["stop_time"], [s[1] for s in SEIZURES])

    df = seizures.fetch1_dataframe()
    np.testing.assert_allclose(df["duration"], [b - a for a, b in SEIZURES])


def test_other_intervals_not_ingested(common, sleep_nwb_key):
    names = (common.SleepStateScoring & sleep_nwb_key).fetch("scoring_name")
    assert list(names) == ["sleep_states"]
    names = (common.SeizureEvents & sleep_nwb_key).fetch("seizure_table_name")
    assert list(names) == ["seizure_events"]
