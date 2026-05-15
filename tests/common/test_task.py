"""Tests for common_task.py - Task and TaskEpoch tables."""

from unittest.mock import Mock

import pynwb
import pytest
from pynwb.core import DynamicTable


def _make_task_table(
    with_camera=True,
    with_epochs=True,
    with_name=True,
    with_desc=True,
):
    """Create a pynwb DynamicTable mimicking the task format."""
    table = DynamicTable(name="tasks", description="test tasks")
    if with_name:
        table.add_column(name="task_name", description="task name")
    if with_desc:
        table.add_column(name="task_description", description="task desc")
    if with_camera:
        table.add_column(name="camera_id", description="camera id")
    if with_epochs:
        table.add_column(name="task_epochs", description="task epochs")
    return table


# ------------------------------------------------------------------ #
# Task.is_nwb_task_table
# ------------------------------------------------------------------ #


def test_is_nwb_task_table_valid(common):
    """DynamicTable with task_name and task_description is valid."""
    table = _make_task_table()
    assert common.Task.is_nwb_task_table(table)


def test_is_nwb_task_table_not_dynamic_table(common):
    """Non-DynamicTable objects fail is_nwb_task_table."""
    assert not common.Task.is_nwb_task_table("not a table")
    assert not common.Task.is_nwb_task_table(None)
    assert not common.Task.is_nwb_task_table(42)


def test_is_nwb_task_table_missing_task_name(common):
    """DynamicTable without task_name fails is_nwb_task_table."""
    table = _make_task_table(with_name=False)
    assert not common.Task.is_nwb_task_table(table)


def test_is_nwb_task_table_missing_task_description(common):
    """DynamicTable without task_description fails is_nwb_task_table."""
    table = _make_task_table(with_desc=False)
    assert not common.Task.is_nwb_task_table(table)


def test_is_nwb_task_table_empty_table(common):
    """Empty DynamicTable (no columns) fails is_nwb_task_table."""
    table = DynamicTable(name="tasks", description="test")
    assert not common.Task.is_nwb_task_table(table)


# ------------------------------------------------------------------ #
# TaskEpoch.is_nwb_task_epoch
# ------------------------------------------------------------------ #


def test_is_nwb_task_epoch_valid(common):
    """Full task table with camera_id and task_epochs is valid."""
    table = _make_task_table(
        with_camera=True,
        with_epochs=True,
    )
    assert common.TaskEpoch.is_nwb_task_epoch(table)


def test_is_nwb_task_epoch_missing_camera(common):
    """Table without camera_id fails is_nwb_task_epoch."""
    table = _make_task_table(with_camera=False, with_epochs=True)
    assert not common.TaskEpoch.is_nwb_task_epoch(table)


def test_is_nwb_task_epoch_missing_epochs(common):
    """Table without task_epochs fails is_nwb_task_epoch."""
    table = _make_task_table(with_camera=True, with_epochs=False)
    assert not common.TaskEpoch.is_nwb_task_epoch(table)


def test_is_nwb_task_epoch_missing_both(common):
    """Table without camera_id or task_epochs fails."""
    table = _make_task_table(with_camera=False, with_epochs=False)
    assert not common.TaskEpoch.is_nwb_task_epoch(table)


def test_is_nwb_task_epoch_not_dynamic_table(common):
    """Non-DynamicTable objects fail is_nwb_task_epoch."""
    assert not common.TaskEpoch.is_nwb_task_epoch("not a table")
    assert not common.TaskEpoch.is_nwb_task_epoch(None)


# ------------------------------------------------------------------ #
# TaskEpoch._get_valid_camera_names
# ------------------------------------------------------------------ #


def test_get_valid_camera_names_all_valid(common):
    """All camera IDs found returns a list of camera_name dicts."""
    camera_names = {1: "cam1", 2: "cam2"}
    result = common.TaskEpoch._get_valid_camera_names([1, 2], camera_names)
    assert result == [
        {"camera_name": "cam1"},
        {"camera_name": "cam2"},
    ]


def test_get_valid_camera_names_partial(common):
    """Only valid IDs are included in the result."""
    camera_names = {1: "cam1"}
    result = common.TaskEpoch._get_valid_camera_names([1, 99], camera_names)
    assert result == [{"camera_name": "cam1"}]


def test_get_valid_camera_names_none_valid(common):
    """No matching IDs returns None."""
    camera_names = {1: "cam1"}
    result = common.TaskEpoch._get_valid_camera_names([99, 100], camera_names)
    assert result is None


def test_get_valid_camera_names_empty_ids(common):
    """Empty camera_ids list returns None without warning."""
    camera_names = {1: "cam1"}
    result = common.TaskEpoch._get_valid_camera_names([], camera_names)
    assert result is None


def test_get_valid_camera_names_empty_both(common):
    """Empty IDs and empty names map both return None."""
    result = common.TaskEpoch._get_valid_camera_names([], {})
    assert result is None


def test_get_valid_camera_names_single(common):
    """Single matching camera_id returns a one-element list."""
    camera_names = {5: "front_cam"}
    result = common.TaskEpoch._get_valid_camera_names([5], camera_names)
    assert result == [{"camera_name": "front_cam"}]


def test_get_valid_camera_names_context_no_effect(common):
    """Context string does not affect returned value."""
    camera_names = {1: "cam1"}
    with_context = common.TaskEpoch._get_valid_camera_names(
        [1], camera_names, context=" in test NWB"
    )
    without_context = common.TaskEpoch._get_valid_camera_names(
        [1], camera_names
    )
    assert with_context == without_context


# ------------------------------------------------------------------ #
# TaskEpoch._check_videos_without_task
# ------------------------------------------------------------------ #


def test_check_videos_without_task_no_videos(common):
    """NWB file with no ImageSeries returns without warning."""
    mock_nwbf = Mock()
    mock_nwbf.objects.values.return_value = []
    # Should not raise
    common.TaskEpoch._check_videos_without_task(mock_nwbf, "test.nwb")


def test_check_videos_without_task_non_image_series(common):
    """Non-ImageSeries objects in NWB are ignored."""
    mock_other = Mock()  # plain Mock, not ImageSeries spec
    mock_nwbf = Mock()
    mock_nwbf.objects.values.return_value = [mock_other]
    # Should return early without warning
    common.TaskEpoch._check_videos_without_task(mock_nwbf, "test.nwb")


def test_check_videos_without_task_with_image_series(common):
    """NWB file with ImageSeries logs a warning and returns."""
    mock_video = Mock(spec=pynwb.image.ImageSeries)
    mock_video.name = "test_video"
    mock_nwbf = Mock()
    mock_nwbf.objects.values.return_value = [mock_video]
    # Should not raise even when ImageSeries present
    common.TaskEpoch._check_videos_without_task(mock_nwbf, "test.nwb")


# ------------------------------------------------------------------ #
# TaskEpoch._process_task_epochs
# ------------------------------------------------------------------ #


def test_process_task_epochs_basic(common):
    """_process_task_epochs maps epoch numbers to interval names."""
    base_key = {
        "task_name": "test_task",
        "nwb_file_name": "test.nwb",
        "camera_names": [],
    }
    session_intervals = ["01_s1", "02_r1"]
    result = common.TaskEpoch._process_task_epochs(
        base_key, [1], "test.nwb", session_intervals
    )
    assert len(result) == 1
    assert result[0]["epoch"] == 1
    assert result[0]["interval_list_name"] == "01_s1"


def test_process_task_epochs_no_match(common):
    """Epochs with no matching interval are skipped."""
    base_key = {
        "task_name": "test_task",
        "nwb_file_name": "test.nwb",
        "camera_names": [],
    }
    session_intervals = ["01_s1", "02_r1"]
    result = common.TaskEpoch._process_task_epochs(
        base_key, [99], "test.nwb", session_intervals
    )
    assert len(result) == 0


def test_process_task_epochs_multiple(common):
    """Multiple matching epochs all get processed."""
    base_key = {
        "task_name": "test_task",
        "nwb_file_name": "test.nwb",
        "camera_names": [],
    }
    session_intervals = ["01_s1", "02_r1", "03_s2"]
    result = common.TaskEpoch._process_task_epochs(
        base_key, [1, 2, 3], "test.nwb", session_intervals
    )
    assert len(result) == 3


def test_process_task_epochs_empty(common):
    """Empty task_epochs list returns empty list."""
    base_key = {
        "task_name": "test_task",
        "nwb_file_name": "test.nwb",
        "camera_names": [],
    }
    result = common.TaskEpoch._process_task_epochs(
        base_key, [], "test.nwb", ["01_s1"]
    )
    assert result == []
