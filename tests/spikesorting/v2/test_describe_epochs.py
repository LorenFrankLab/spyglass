"""Read-only epoch discovery preserves original bounds and cache readiness."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from spyglass.spikesorting.v2._orchestration.epochs import (
    epoch_rows_to_dataframe,
)

pytestmark = pytest.mark.unit


def test_epoch_chooser_preserves_disjoint_original_bounds_and_exact_keys():
    frame = epoch_rows_to_dataframe(
        "session.nwb",
        [{"epoch": 2, "task_name": "track", "interval_list_name": "run"}],
        [
            {
                "interval_list_name": "run",
                "valid_times": np.array([[100, 110], [140, 155]]),
            },
            {
                "interval_list_name": "raw data valid times",
                "valid_times": [[90, 180]],
            },
        ],
        [],
    )
    run = frame.set_index("interval_list_name").loc["run"]
    assert run["epoch_label"] == "epoch 2: track"
    assert run["task_epoch_key"] == {"nwb_file_name": "session.nwb", "epoch": 2}
    assert run["interval_key"] == {
        "nwb_file_name": "session.nwb",
        "interval_list_name": "run",
    }
    assert run["interval_bounds"] == ((100.0, 110.0), (140.0, 155.0))
    assert run["start_time"] == 100 and run["stop_time"] == 155
    assert run["duration_s"] == 25  # the 30-second gap is not recording time
    assert run["recording_readiness"] == "not_selected"
    assert str(frame["epoch"].dtype) == "Int64"
    assert (
        frame.set_index("interval_list_name").loc[
            "raw data valid times", "kind"
        ]
        == "interval_list"
    )


def test_epoch_chooser_shows_missing_cache_and_available_recording_keys():
    cached, missing, selected = [uuid.UUID(int=index) for index in (1, 2, 3)]
    records = [
        {
            "recording_id": cached,
            "sort_group_id": 0,
            "interval_list_name": "run",
            "recording_computed": True,
            "cache_present": True,
        },
        {
            "recording_id": missing,
            "sort_group_id": 0,
            "interval_list_name": "run",
            "recording_computed": True,
            "cache_present": False,
            "cache_error": "Rebuild the recording explicitly.",
        },
        {
            "recording_id": selected,
            "sort_group_id": 1,
            "interval_list_name": "sleep",
            "recording_computed": False,
            "cache_present": False,
        },
    ]
    frame = epoch_rows_to_dataframe(
        "session.nwb",
        [],
        [
            {"interval_list_name": name, "valid_times": [[i * 10, i * 10 + 5]]}
            for i, name in enumerate(("run", "sleep"))
        ],
        records,
    ).set_index("interval_list_name")
    assert frame.loc["run", "recording_keys"] == (
        {"recording_id": cached},
        {"recording_id": missing},
    )
    assert frame.loc["run", "ready_recording_keys"] == (
        {"recording_id": cached},
    )
    assert frame.loc["run", "recording_readiness"] == "cache_present"
    assert frame.loc["sleep", "recording_readiness"] == "not_computed"
    assert not frame.loc["sleep", "ready_recording_keys"]
    # A consumer editing the table does not mutate the fetched source rows.
    frame.loc["run", "recordings"][1]["cache_error"] = "changed"
    assert records[1]["cache_error"] == "Rebuild the recording explicitly."


def test_epoch_chooser_reports_empty_malformed_and_orphaned_intervals():
    frame = epoch_rows_to_dataframe(
        "session.nwb",
        [{"epoch": 3, "task_name": "track", "interval_list_name": "missing"}],
        [
            {"interval_list_name": "empty", "valid_times": []},
            {"interval_list_name": "bad", "valid_times": [[10, 5]]},
        ],
        [],
    ).set_index("interval_list_name")
    assert frame.loc["empty", "interval_status"] == "empty"
    assert frame.loc["empty", "duration_s"] == 0
    assert frame.loc["bad", "interval_status"] == "invalid"
    assert "start < stop" in frame.loc["bad", "interval_error"]
    assert frame.loc["missing", "interval_status"] == "missing"
    assert frame.loc["missing", "task_epoch_key"]["epoch"] == 3


def test_epoch_chooser_lists_only_intervals_a_user_would_sort():
    """Pipeline output intervals are hidden unless a task or recording uses one."""
    times = [[0.0, 1.0]]

    def interval(name, pipeline=""):
        return {
            "interval_list_name": name,
            "valid_times": times,
            "pipeline": pipeline,
        }

    frame = epoch_rows_to_dataframe(
        "session.nwb",
        [{"epoch": 1, "task_name": "run", "interval_list_name": "01_r1"}],
        [
            interval("01_r1"),
            interval("raw data valid times"),
            interval("my interval"),
            interval("pos 0 valid times", "position"),
            interval("recording_abc", "spikesorting_recording_v2"),
            interval(
                "artifact_detection_abc", "spikesorting_artifact_detection_v2"
            ),
            interval("lfp valid times", "lfp_v1"),
            interval("sorted pipeline output", "lfp_v1"),
        ],
        [
            {
                "interval_list_name": "sorted pipeline output",
                "sort_group_id": 0,
                "recording_id": "r0",
            }
        ],
    )

    assert list(frame["interval_list_name"]) == [
        "01_r1",
        "my interval",
        "raw data valid times",
        "sorted pipeline output",
    ]


def test_epoch_chooser_has_stable_empty_columns():
    frame = epoch_rows_to_dataframe("empty.nwb", [], [], [])
    assert frame.empty
    assert "interval_key" in frame and "ready_recording_keys" in frame
    assert str(frame["epoch"].dtype) == "Int64"


@pytest.mark.parametrize("bad_sort_group", [True, 1.5])
def test_describe_epochs_rejects_lossy_sort_group_before_database(
    bad_sort_group,
):
    from spyglass.spikesorting.v2.pipeline import describe_epochs

    with pytest.raises(ValueError, match="sort_group_id"):
        describe_epochs("session.nwb", bad_sort_group)


@pytest.mark.db_unit
def test_describe_epochs_rejects_a_session_that_is_not_ingested(dj_conn):
    """An unknown session is named with the trailing-underscore hint instead
    of returning an empty table."""
    from spyglass.spikesorting.v2.pipeline import describe_epochs

    with pytest.raises(ValueError, match="no_such_session.nwb.*not ingested"):
        describe_epochs("no_such_session.nwb")
