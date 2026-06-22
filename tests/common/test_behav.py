# pylint: disable=protected-access

import pytest
from pandas import DataFrame

from ..conftest import TEARDOWN


@pytest.mark.slow
def test_invalid_interval(pos_src):
    """Test invalid interval"""
    with pytest.raises(ValueError):
        pos_src.get_pos_interval_name("invalid_interval")


def test_invalid_epoch_num(common):
    """Test invalid epoch num"""
    with pytest.raises(ValueError):
        common.PositionSource.get_epoch_num("invalid_epoch_num")


def test_valid_epoch_num(common):
    """Test valid epoch num"""
    epoch_num = common.PositionSource.get_epoch_num("pos 1 valid times")
    assert epoch_num == 1, "PositionSource get_epoch_num failed"


@pytest.mark.slow
def test_pos_source_make(common):
    """Test custom populate"""
    common.PositionSource().make(common.Session())


def test_pos_source_make_invalid(common):
    """Test invalid populate"""
    with pytest.raises(ValueError):
        common.PositionSource().make(dict())


def test_raw_position_fetch_nwb(common, mini_pos, mini_pos_interval_dict):
    """Test RawPosition fetch nwb"""
    fetched = DataFrame(
        (common.RawPosition & mini_pos_interval_dict)
        .fetch_nwb()[0]["raw_position"]
        .data
    )
    raw = DataFrame(mini_pos["led_0_series_0"].data)
    # compare with mini_pos
    assert fetched.equals(raw), "RawPosition fetch_nwb failed"


def test_raw_position_fetch1_df(common, mini_pos, mini_pos_interval_dict):
    """Test RawPosition fetch1 dataframe"""
    fetched = (common.RawPosition & mini_pos_interval_dict).fetch1_dataframe()
    fetched.reset_index(drop=True, inplace=True)
    fetched.columns = range(fetched.shape[1])
    fetched = fetched.iloc[:, :2]

    raw = DataFrame(mini_pos["led_0_series_0"].data)
    assert fetched.equals(raw), "RawPosition fetch1_dataframe failed"


def test_raw_position_fetch_multi_df(common):
    """Test RawPosition fetch1 dataframe"""
    shape = common.RawPosition().fetch1_dataframe().shape
    assert shape == (542, 8), "RawPosition.PosObj fetch1_dataframe failed"


@pytest.fixture(scope="session", name="pop_state_script")
def pop_state_script_fixture(common):
    """Populate state script"""
    keys = common.StateScriptFile.key_source
    common.StateScriptFile.populate()
    yield keys


def test_populate_state_script(common, pop_state_script):
    """Test populate state script

    See #849. Expect no result for this table."""
    assert len(common.StateScriptFile.key_source) == len(
        pop_state_script
    ), "StateScript populate unexpected effect"


def test_videofile_update_entries(common):
    """Test update entries"""
    key = common.VideoFile().fetch(as_dict=True)[0]
    common.VideoFile().update_entries(key)


def test_fetch_key_from_path_no_side_effect(common, tmp_path):
    """fetch_key_from_path must not call update_entries by default."""
    from unittest.mock import patch

    fake = tmp_path / "no_such_video.mp4"
    fake.touch()

    with patch.object(common.VideoFile, "update_entries") as mock_update:
        with pytest.raises(ValueError):
            common.VideoFile().fetch_key_from_path(str(fake))
        mock_update.assert_not_called()


def test_fetch_key_from_path_update_on_miss(common, tmp_path):
    """update_on_miss=True triggers update_entries then retries."""
    from unittest.mock import patch

    fake = tmp_path / "no_such_video.mp4"
    fake.touch()

    with patch.object(common.VideoFile, "update_entries") as mock_update:
        with pytest.raises(ValueError):
            common.VideoFile().fetch_key_from_path(
                str(fake), update_on_miss=True
            )
        mock_update.assert_called_once()


def test_videofile_getabspath(common):
    """Test get absolute path"""
    key = common.VideoFile().fetch(as_dict=True)[0]
    path = common.VideoFile().get_abs_path(key)
    file_part = key["nwb_file_name"].split("2")[0] + "_0" + str(key["epoch"])
    assert file_part in path, "VideoFile get_abs_path failed"


@pytest.mark.skipif(not TEARDOWN, reason="No teardown: expect no change.")
def test_pos_interval_no_transaction(verbose_context, common, mini_restr):
    """Test no transaction"""
    before = common.PositionIntervalMap().fetch()
    with verbose_context:
        common.PositionIntervalMap().make(mini_restr)
    after = common.PositionIntervalMap().fetch()
    expected_insertions = 4
    assert len(after) - len(before) == expected_insertions, (
        f"PositionIntervalMap failed to insert the expected number of entries. "
        f"Expected {expected_insertions}, but got {len(after) - len(before)}."
    )
    assert (
        "" in after["position_interval_name"]
    ), "PositionIntervalMap null insert failed"


def test_get_pos_interval_name(pos_interval_01):
    """Test get pos interval name"""
    names = [f"pos {x} valid times" for x in range(1)]
    assert pos_interval_01 == names, "get_pos_interval_name failed"


def test_convert_epoch(common, mini_dict, pos_interval_01):
    this_key = (
        common.IntervalList & mini_dict & {"interval_list_name": "01_s1"}
    ).fetch1()
    ret = common.common_behav.convert_epoch_interval_name_to_position_interval_name(
        this_key
    )
    assert (
        ret == pos_interval_01[0]
    ), "convert_epoch_interval_name_to_position_interval_name failed"


def test_prepare_video_entry_with_external_file(common):
    """Test _prepare_video_entry with external_file attribute."""
    from pathlib import Path
    from unittest.mock import MagicMock, Mock

    # Create mock video object with external_file
    mock_device = MagicMock()
    mock_device.name = "camera_device 1"
    mock_device.camera_name = "test_camera"

    mock_video = Mock()
    mock_video.device = mock_device
    mock_video.object_id = "test_object_id"
    mock_video.name = "generic_video_name"
    mock_video.external_file = ["file1.mp4", "file2.mp4", "file3.mp4"]

    # Mock CameraDevice table to have the camera
    key = {"test": "key"}

    # Test with file_idx=None (should use index 0)
    video_file = common.VideoFile()
    with pytest.importorskip("unittest.mock").patch.object(
        common.common_behav, "CameraDevice"
    ) as mock_camera_device:
        mock_camera_device.__and__.return_value = True  # Camera exists

        result = video_file._prepare_video_entry(key, mock_video)

        expected_filename = Path("file1.mp4").name
        assert expected_filename in result["path"]


def test_prepare_video_entry_with_file_idx(common):
    """Test _prepare_video_entry with specific file_idx."""
    from pathlib import Path
    from unittest.mock import MagicMock, Mock

    # Create mock video object
    mock_device = MagicMock()
    mock_device.name = "camera_device 2"
    mock_device.camera_name = "test_camera"

    mock_video = Mock()
    mock_video.device = mock_device
    mock_video.object_id = "test_object_id"
    mock_video.name = "generic_video_name"
    mock_video.external_file = ["file1.mp4", "file2.mp4", "file3.mp4"]

    key = {"test": "key"}

    # Test with file_idx=1 (should use second file)
    video_file = common.VideoFile()
    with pytest.importorskip("unittest.mock").patch.object(
        common.common_behav, "CameraDevice"
    ) as mock_camera_device:
        mock_camera_device.__and__.return_value = True

        result = video_file._prepare_video_entry(key, mock_video, file_idx=1)

        expected_filename = Path("file2.mp4").name
        assert expected_filename in result["path"]


def test_prepare_video_entry_with_empty_external_file(common):
    """Test _prepare_video_entry with empty external_file list."""
    from unittest.mock import MagicMock, Mock

    # Create mock video object with empty external_file
    mock_device = MagicMock()
    mock_device.name = "camera_device 3"
    mock_device.camera_name = "test_camera"

    mock_video = Mock()
    mock_video.device = mock_device
    mock_video.object_id = "test_object_id"
    mock_video.name = "fallback_video_name"
    mock_video.external_file = []  # Empty list

    key = {"test": "key"}

    video_file = common.VideoFile()
    with pytest.importorskip("unittest.mock").patch.object(
        common.common_behav, "CameraDevice"
    ) as mock_camera_device:
        mock_camera_device.__and__.return_value = True

        result = video_file._prepare_video_entry(key, mock_video)

        # Should use video object name as fallback
        assert "fallback_video_name" in result["path"]


# Additional comprehensive tests for common_behav.py coverage improvement


def test_prepare_video_entry_missing_camera(common):
    """Test _prepare_video_entry with missing camera in CameraDevice table."""
    from unittest.mock import MagicMock, Mock, patch

    mock_device = MagicMock()
    mock_device.name = "camera_device 4"
    mock_device.camera_name = "nonexistent_camera"

    mock_video = Mock()
    mock_video.device = mock_device
    mock_video.object_id = "test_object_id"
    mock_video.name = "test_video"
    mock_video.external_file = ["test.mp4"]

    key = {"test": "key"}

    video_file = common.VideoFile()
    with patch.object(
        common.common_behav, "CameraDevice"
    ) as mock_camera_device:
        # Mock camera not found
        mock_camera_device.__and__.return_value = False

        with pytest.raises(KeyError, match="No camera with camera_name"):
            video_file._prepare_video_entry(key, mock_video)


def test_prepare_video_entry_invalid_device_regex(common):
    """Test _prepare_video_entry with invalid device name regex pattern."""
    from unittest.mock import MagicMock, Mock, patch

    mock_device = MagicMock()
    mock_device.name = "invalid_camera_name"  # Doesn't match regex
    mock_device.camera_name = "test_camera"

    mock_video = Mock()
    mock_video.device = mock_device
    mock_video.object_id = "test_object_id"
    mock_video.name = "test_video"

    key = {"test": "key"}

    video_file = common.VideoFile()
    with patch.object(
        common.common_behav, "CameraDevice"
    ) as mock_camera_device:
        mock_camera_device.__and__.return_value = True

        with pytest.raises(ValueError, match="does not match expected pattern"):
            video_file._prepare_video_entry(key, mock_video)


def test_prepare_video_entry_file_idx_out_of_range(common):
    """Test _prepare_video_entry with file_idx beyond external_file range."""
    from unittest.mock import MagicMock, Mock, patch

    mock_device = MagicMock()
    mock_device.name = "camera_device 5"
    mock_device.camera_name = "test_camera"

    mock_video = Mock()
    mock_video.device = mock_device
    mock_video.object_id = "test_object_id"
    mock_video.name = "fallback_name"
    mock_video.external_file = ["file1.mp4"]  # Only 1 file

    key = {"test": "key"}

    video_file = common.VideoFile()
    with patch.object(
        common.common_behav, "CameraDevice"
    ) as mock_camera_device:
        mock_camera_device.__and__.return_value = True

        # Request file_idx=2 when only 1 file exists
        result = video_file._prepare_video_entry(key, mock_video, file_idx=2)

        # Should fallback to video object name
        assert "fallback_name" in result["path"]


def test_validate_video_timestamps_single_file(common):
    """Test _validate_video_timestamps for single-file ImageSeries."""
    from unittest.mock import Mock, patch

    import numpy as np

    from spyglass.common.common_interval import Interval

    # Create mock video object
    mock_video = Mock()
    mock_video.timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mock_video.starting_frame = None  # Single file

    # Mock valid times that overlap well (>90%)
    valid_times = Mock(spec=Interval)
    valid_times.contains.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    valid_times.times = [[0.5, 5.5]]  # Covers whole video

    video_file = common.VideoFile()

    with patch.object(video_file, "_prepare_video_entry") as mock_prepare:
        mock_prepare.return_value = {"test": "entry"}

        entries, failure, overlap_pct = video_file._validate_video_timestamps(
            mock_video, valid_times, {"key": "value"}
        )

        assert len(entries) == 1
        assert failure is None
        assert overlap_pct == 1.0
        mock_prepare.assert_called_once()


def test_validate_video_timestamps_single_file_poor_overlap(common):
    """Test _validate_video_timestamps with poor timestamp overlap."""
    from unittest.mock import Mock

    import numpy as np

    mock_video = Mock()
    mock_video.timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mock_video.starting_frame = None

    # Mock valid times with poor overlap
    valid_times = Mock()
    valid_times.contains.return_value = np.array([1.0])  # Only 1/5 = 20%
    valid_times.times = [[6.0, 7.0]]  # Doesn't overlap

    video_file = common.VideoFile()

    entries, failure, overlap_pct = video_file._validate_video_timestamps(
        mock_video, valid_times, {"key": "value"}
    )

    assert len(entries) == 0
    assert "Only 20.0%" in failure
    assert overlap_pct == pytest.approx(0.2)


def test_validate_multifile_timestamps(common):
    """Test _validate_multifile_timestamps validation logic."""
    from unittest.mock import Mock, patch

    import numpy as np

    mock_video = Mock()
    mock_video.object_id = "test_id"

    timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    starting_frame = np.array([0, 3])  # Two files: [0-3) and [3-6)

    # Mock valid times
    valid_times = Mock()
    # File 1 (timestamps 1-3): good overlap, File 2 (timestamps 4-6): poor overlap
    valid_times.contains.side_effect = [
        np.array([1.0, 2.0, 3.0]),  # File 1: 3/3 timestamps = 100% > 90%
        np.array([4.0]),  # File 2: 1/3 timestamps = 33% < 90%
    ]

    video_file = common.VideoFile()

    with patch.object(video_file, "_prepare_video_entry") as mock_prepare:
        mock_prepare.return_value = {"test": "entry"}

        entries, max_overlap = video_file._validate_multifile_timestamps(
            mock_video,
            timestamps,
            starting_frame,
            valid_times,
            {"key": "value"},
        )

        # Should only return entry for file 1; max overlap is 100% from file 1
        assert len(entries) == 1
        assert max_overlap == pytest.approx(1.0)
        mock_prepare.assert_called_once_with(
            {"key": "value"}, mock_video, file_idx=0
        )


def test_validate_multifile_timestamps_no_valid_segments(common):
    """Test _validate_multifile_timestamps when no segments meet threshold."""
    from unittest.mock import Mock

    import numpy as np

    mock_video = Mock()
    timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    starting_frame = np.array([0, 3])

    # Mock valid times with poor overlap for all segments
    valid_times = Mock()
    valid_times.contains.side_effect = [
        np.array([1.0]),  # File 1: 1/3 = 33% < 90%
        np.array([4.0]),  # File 2: 1/3 = 33% < 90%
    ]

    video_file = common.VideoFile()

    entries, max_overlap = video_file._validate_multifile_timestamps(
        mock_video, timestamps, starting_frame, valid_times, {"key": "value"}
    )

    assert len(entries) == 0
    assert max_overlap == pytest.approx(1 / 3)


def test_videofile_make_no_videos(common):
    """Test VideoFile.make when no ImageSeries found in NWB file."""
    from unittest.mock import Mock, patch

    # Mock NWB file with no ImageSeries objects
    mock_nwbf = Mock()
    mock_nwbf.objects.values.return_value = []  # No objects

    video_file = common.VideoFile()

    with (
        patch("spyglass.common.common_behav.get_nwb_file") as mock_get_nwb,
        patch.object(common.Nwbfile, "get_abs_path") as mock_get_path,
    ):

        mock_get_nwb.return_value = mock_nwbf
        mock_get_path.return_value = "/fake/path.nwb"

        # Should not raise, just return early with warning
        video_file.make({"nwb_file_name": "test.nwb"})


def test_videofile_make_camera_device_error(common):
    """Test VideoFile.make with camera device KeyError during processing."""
    from unittest.mock import Mock, patch

    import pynwb

    # Mock NWB file with ImageSeries
    mock_video = Mock(spec=pynwb.image.ImageSeries)
    mock_video.name = "test_video"
    mock_video.device.camera_name = "missing_camera"

    mock_nwbf = Mock()
    mock_nwbf.objects.values.return_value = [mock_video]

    # Mock TaskEpoch and IntervalList
    mock_interval = Mock()

    video_file = common.VideoFile()

    with (
        patch("spyglass.common.common_behav.get_nwb_file") as mock_get_nwb,
        patch.object(common.Nwbfile, "get_abs_path") as mock_get_path,
        patch.object(common.TaskEpoch, "__and__") as mock_task_epoch,
        patch.object(common.IntervalList, "__and__") as mock_interval_list,
        patch.object(video_file, "_validate_video_timestamps") as mock_validate,
    ):

        mock_get_nwb.return_value = mock_nwbf
        mock_get_path.return_value = "/fake/path.nwb"
        mock_task_epoch.return_value.fetch1.return_value = "epoch01"
        mock_interval_list.return_value.fetch_interval.return_value = (
            mock_interval
        )

        # Simulate KeyError from _validate_video_timestamps
        mock_validate.side_effect = KeyError("Camera not found")

        # Should handle error gracefully
        video_file.make({"nwb_file_name": "test.nwb"})


def test_videofile_make_unexpected_error(common):
    """Test VideoFile.make with unexpected error during processing."""
    from unittest.mock import Mock, patch

    import pynwb

    mock_video = Mock(spec=pynwb.image.ImageSeries)
    mock_video.name = "test_video"

    mock_nwbf = Mock()
    mock_nwbf.objects.values.return_value = [mock_video]

    mock_interval = Mock()

    video_file = common.VideoFile()

    with (
        patch("spyglass.common.common_behav.get_nwb_file") as mock_get_nwb,
        patch.object(common.Nwbfile, "get_abs_path") as mock_get_path,
        patch.object(common.TaskEpoch, "__and__") as mock_task_epoch,
        patch.object(common.IntervalList, "__and__") as mock_interval_list,
        patch.object(video_file, "_validate_video_timestamps") as mock_validate,
    ):

        mock_get_nwb.return_value = mock_nwbf
        mock_get_path.return_value = "/fake/path.nwb"
        mock_task_epoch.return_value.fetch1.return_value = "epoch01"
        mock_interval_list.return_value.fetch_interval.return_value = (
            mock_interval
        )

        # Simulate unexpected error
        mock_validate.side_effect = RuntimeError("Unexpected error")

        # Should handle error gracefully
        video_file.make({"nwb_file_name": "test.nwb"})


def test_videofile_report_partial_import(common, caplog):
    """Test VideoFile._report_partial_import logging functionality."""
    from collections import defaultdict

    failed_videos = defaultdict(list)
    failed_videos["timestamp_mismatch"].append(
        {"name": "video1", "reason": "Poor overlap", "overlap_percent": 20.0}
    )
    failed_videos["missing_camera"].append(
        {"name": "video2", "camera": "cam1", "error": "Not found"}
    )
    failed_videos["other"].append(
        {"name": "video3", "error": "RuntimeError: Something broke"}
    )

    common.VideoFile._report_partial_import("test.nwb", failed_videos, 5, 2)

    # Check that logging contains expected information for diagnostics
    assert "VideoFile Partial Import" in caplog.text
    assert "Imported 2/5 ImageSeries" in caplog.text
    assert "Timestamp mismatches" in caplog.text
    assert "Missing camera devices" in caplog.text
    assert "Other errors" in caplog.text


def test_videofile_get_abs_path_with_stored_path(common):
    """Test VideoFile.get_abs_path when path is already stored in database."""
    from pathlib import Path
    from unittest.mock import patch

    # Mock stored path that exists
    stored_path = "/stored/video/path.mp4"
    video_info = {"path": stored_path}

    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(common.VideoFile, "__and__") as mock_query,
    ):

        mock_query.return_value.fetch1.return_value = video_info

        result = common.VideoFile.get_abs_path({"nwb_file_name": "test.nwb"})

        assert result == stored_path


def test_videofile_get_abs_path_fallback_external_file(common):
    """Test VideoFile.get_abs_path fallback to external_file when object name fails."""
    from pathlib import Path
    from unittest.mock import MagicMock, Mock, patch

    import pynwb

    # Mock scenario where stored path doesn't exist, object name doesn't exist,
    # but external_file path exists
    video_info = {"path": None, "video_file_object_id": "obj_123"}

    # Mock ImageSeries with external_file
    mock_video = Mock(spec=pynwb.image.ImageSeries)
    mock_video.name = "nonexistent_video.mp4"
    mock_video.external_file = ["/real/external/path.mp4"]

    mock_nwbf = Mock()
    mock_nwbf.objects = MagicMock()
    mock_nwbf.objects.__getitem__.return_value = mock_video

    with (
        patch.object(common.VideoFile, "__and__") as mock_query,
        patch.object(common.Nwbfile, "get_abs_path") as mock_get_abs_path,
        patch("spyglass.common.common_behav.get_nwb_file") as mock_get_nwb,
        patch.object(Path, "exists") as mock_exists,
    ):

        mock_query.return_value.fetch1.return_value = video_info
        mock_get_abs_path.return_value = "/nwb/path.nwb"
        mock_get_nwb.return_value = mock_nwbf

        # Object-name path misses, then absolute external path resolves.
        mock_exists.side_effect = [False, True]

        result = common.VideoFile.get_abs_path({"nwb_file_name": "test.nwb"})

        assert result == "/real/external/path.mp4"


def test_videofile_get_abs_path_non_image_series_fallback(common):
    """Test VideoFile.get_abs_path when object_id isn't ImageSeries."""
    from pathlib import Path
    from unittest.mock import MagicMock, Mock, patch

    import pynwb

    video_info = {"path": None, "video_file_object_id": "stale_obj_123"}

    # Mock stale object that's not an ImageSeries
    mock_stale_obj = Mock()  # Not an ImageSeries

    # Mock real ImageSeries to fallback to
    mock_video = Mock(spec=pynwb.image.ImageSeries)
    mock_video.name = "real_video.mp4"
    mock_video.external_file = ["real_video.mp4"]

    mock_nwbf = Mock()
    mock_nwbf.objects = MagicMock()
    mock_nwbf.objects.__getitem__.return_value = mock_stale_obj
    mock_nwbf.objects.values.return_value = [mock_stale_obj, mock_video]

    with (
        patch.object(common.VideoFile, "__and__") as mock_query,
        patch.object(common.Nwbfile, "get_abs_path") as mock_get_abs_path,
        patch("spyglass.common.common_behav.get_nwb_file") as mock_get_nwb,
        patch.object(Path, "exists") as mock_exists,
    ):

        mock_query.return_value.fetch1.return_value = video_info
        mock_get_abs_path.return_value = "/nwb/path.nwb"
        mock_get_nwb.return_value = mock_nwbf

        # Fallback object-name path misses, then the external-file lookup under
        # video_dir resolves.
        mock_exists.side_effect = [False, True]

        result = common.VideoFile.get_abs_path({"nwb_file_name": "test.nwb"})

        # Should use the real ImageSeries object, not the stale reference
        assert "real_video.mp4" in result


def test_videofile_get_abs_path_no_image_series_error(common):
    """Test VideoFile.get_abs_path when no ImageSeries found in NWB."""
    from unittest.mock import MagicMock, Mock, patch

    video_info = {"path": None, "video_file_object_id": "obj_123"}

    # Mock non-ImageSeries object
    mock_other_obj = Mock()

    mock_nwbf = Mock()
    mock_nwbf.objects = MagicMock()
    mock_nwbf.objects.__getitem__.return_value = mock_other_obj
    mock_nwbf.objects.values.return_value = [mock_other_obj]

    with (
        patch.object(common.VideoFile, "__and__") as mock_query,
        patch.object(common.Nwbfile, "get_abs_path") as mock_get_abs_path,
        patch("spyglass.common.common_behav.get_nwb_file") as mock_get_nwb,
    ):

        mock_query.return_value.fetch1.return_value = video_info
        mock_get_abs_path.return_value = "/nwb/path.nwb"
        mock_get_nwb.return_value = mock_nwbf

        with pytest.raises(FileNotFoundError, match="No ImageSeries found"):
            common.VideoFile.get_abs_path({"nwb_file_name": "test.nwb"})


def test_videofile_update_entries_with_null_values(common):
    """Test VideoFile.update_entries with NULL camera_name or path values."""
    from unittest.mock import Mock, patch

    # Mock entries with NULL values
    entries_with_nulls = [
        {
            "nwb_file_name": "test1.nwb",
            "epoch": 1,
            "camera_name": None,
            "path": "/path",
        },
        {
            "nwb_file_name": "test2.nwb",
            "epoch": 2,
            "camera_name": "cam1",
            "path": None,
        },
    ]

    row_relation = Mock()
    row_relation.fetch_nwb.return_value = [
        {"video_file": Mock(device=Mock(camera_name="new_camera"))}
    ]

    restrict_relation = Mock()
    filtered_relation = Mock()
    filtered_relation.fetch.return_value = entries_with_nulls
    restrict_relation.__and__ = Mock(return_value=filtered_relation)

    def and_side_effect(arg):
        if arg is True:
            return restrict_relation
        if isinstance(arg, dict):
            return row_relation
        return Mock()

    with (
        patch.object(common.VideoFile, "__and__") as mock_query,
        patch.object(common.VideoFile, "get_abs_path") as mock_get_abs_path,
        patch.object(common.VideoFile, "update1") as mock_update1,
    ):
        mock_query.side_effect = and_side_effect
        mock_get_abs_path.return_value = "/new/abs/path.mp4"

        common.VideoFile.update_entries()

        # Should call update for both entries
        assert mock_update1.call_count == 2


def test_position_source_invalid_interval_name(pos_src):
    """Test PositionSource with invalid interval name formats."""
    with pytest.raises(ValueError, match="Invalid interval name"):
        pos_src.get_epoch_num("not_an_epoch_name")


def test_position_source_insert_from_nwbfile(common):
    """Test PositionSource.insert_from_nwbfile with no spatial series."""
    from unittest.mock import Mock, patch

    # Mock NWB file with no spatial series
    mock_nwbf = Mock()

    with (
        patch("spyglass.common.common_behav.get_nwb_file") as mock_get_nwb,
        patch(
            "spyglass.common.common_behav.get_all_spatial_series"
        ) as mock_get_spatial,
    ):

        mock_get_nwb.return_value = mock_nwbf
        mock_get_spatial.return_value = None  # No spatial series found

        # Should not raise, just log info and return
        common.PositionSource.insert_from_nwbfile("test.nwb")

    with (
        patch("spyglass.common.common_behav.get_nwb_file") as mock_get_nwb,
        patch(
            "spyglass.common.common_behav.get_all_spatial_series"
        ) as mock_get_spatial,
    ):

        mock_get_nwb.return_value = mock_nwbf
        mock_get_spatial.return_value = None  # No spatial series found

        # Should not raise, just log info and return
        common.PositionSource.insert_from_nwbfile("test.nwb")

    with (
        patch("spyglass.common.common_behav.get_nwb_file") as mock_get_nwb,
        patch(
            "spyglass.common.common_behav.get_all_spatial_series"
        ) as mock_get_spatial,
    ):

        mock_get_nwb.return_value = mock_nwbf
        mock_get_spatial.return_value = None  # No spatial series found

        # Should not raise, just log info and return
        common.PositionSource.insert_from_nwbfile("test.nwb")
