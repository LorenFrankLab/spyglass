from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import pytest


def test_valid_option_error(sgp):
    validate = sgp.v1.dlc_utils.validate_option

    with pytest.raises(ValueError):
        validate(option=None, permit_none=False)
    with pytest.raises(KeyError):
        validate(option="invalid_option", options=["a", "b"])
    with pytest.raises(TypeError):
        validate(option="invalid_string", types=(int, float))
    with pytest.raises(ValueError):
        validate(option=10, val_range=(0, 5))


def test_valid_list_error(sgp):
    validate = sgp.v1.dlc_utils.validate_list

    ret = validate(option_list=None, permit_none=True, required_items=["a"])
    assert ret is None, "Expected None when permit_none is True"

    with pytest.raises(ValueError):
        validate(option_list=None, permit_none=False, required_items=["a"])
    with pytest.raises(KeyError):
        validate(option_list=["a", "b"], required_items=["a", "c"])


def test_log_decorator_no_path(sgp):
    from spyglass.position.v1.dlc_utils import file_log
    from spyglass.utils import logger

    class TestClass:
        @file_log(logger, console=False)
        def test_function(self):
            logger.info("TEST")
            return "test"

    _ = TestClass().test_function()
    with open("temp_TestClass.log", "r") as f:
        log_content = f.read()

    assert "TEST" in log_content, "Log content should contain 'TEST'"


def test_infer_dir_error(sgp):
    with pytest.raises(ValueError):
        sgp.v1.dlc_utils.infer_output_dir(dict(a=1))


def test_fix_windows_path(sgp):
    path_func = sgp.v1.dlc_utils._to_Path
    path = str(path_func("C:\\Users\\test\\file.txt"))
    assert path == "C:/Users/test/file.txt", "Path should be converted to Unix"


def test_find_mp4_error(sgp):
    with pytest.raises(FileNotFoundError):
        sgp.v1.dlc_utils.find_mp4(video_path="/invalid_path/")
    with pytest.raises(FileNotFoundError):
        sgp.v1.dlc_utils.find_mp4(video_path=".")
    with open("test.mp4", "w") as f:
        f.write("test content")
    path = sgp.v1.dlc_utils.find_mp4(
        video_path=".", video_filename="test.mp4", video_filetype="mp4"
    )
    assert str(path) == "test.mp4", "Path should be the same as the test file"


def test_convert_mp4_error(sgp):
    with pytest.raises(NotImplementedError):
        sgp.v1.dlc_utils._convert_mp4(
            filename="any", video_path="any", dest_path="any", videotype="mp5"
        )


def test_videofile_frames(common, sgp, video_keys):
    key = video_keys[0]
    path = common.VideoFile().get_abs_path(key)
    frames = sgp.v1.dlc_utils._check_packets(Path(path), count_frames=True)
    assert frames == 270, "Frame count should be 270"


# Define a fixture for a sample dataframe
@pytest.fixture(scope="module")
def dlc_df():
    # Create a DataFrame with time as index and x, y, orientation columns
    data = {
        "x": [0, 1, np.nan, 3],
        "y": [0, 2, np.nan, 4],
        "orientation": [0, 90, np.nan, 180],
    }
    return pd.DataFrame(data, index=[0, 1, 2, 3])


@pytest.fixture
def spans_to_interp():
    # Define spans to interpolate
    return [(1, 2)]


@pytest.mark.parametrize(
    "test_name, spans_to_interp, max_cm_to_interp, expected_result",
    [
        ("valid_interpolation", [(1, 2)], float("inf"), False),
        ("no_interpolation_due_to_change", [(1, 2)], 50, False),
        ("missing_start_end", [(1, 2)], float("inf"), True),
    ],
)
def test_interp_pos(
    sgp, dlc_df, spans_to_interp, max_cm_to_interp, expected_result, test_name
):
    """Test the interp_pos function"""
    if test_name == "missing_start_end":
        dlc_df.loc[0, "x"] = np.nan
        dlc_df.loc[3, "y"] = np.nan

    result = sgp.v1.dlc_utils.interp_pos(
        dlc_df.copy(), spans_to_interp, max_cm_to_interp=max_cm_to_interp
    )
    assert result["x"].isna().iloc[0:1].all() == expected_result


# Parameterize tests for interp_orientation
@pytest.mark.parametrize(
    "test_name, spans_to_interp, max_cm_to_interp, expected_result",
    [
        ("valid_interpolation", [(1, 2)], float("inf"), False),
        ("no_interpolation_due_to_change", [(1, 2)], 50, False),
        ("missing_start_end", [(1, 2)], float("inf"), True),
    ],
)
def test_interp_orientation(
    sgp, dlc_df, spans_to_interp, max_cm_to_interp, expected_result, test_name
):
    """Test the interp_orientation function"""
    if test_name == "missing_start_end":
        dlc_df.loc[0, "orientation"] = np.nan
        dlc_df.loc[3, "orientation"] = np.nan

    result = sgp.v1.dlc_utils.interp_orientation(
        dlc_df.copy(), spans_to_interp, max_cm_to_interp=max_cm_to_interp
    )

    # Check if the result has NaN where expected based on the test
    assert result["orientation"].isna().iloc[1:3].all() == expected_result


def test_vid_maker_processor_error(sgp):
    with pytest.raises(ValueError):
        sgp.v1.dlc_utils_makevid.VideoMaker(
            video_filename="any",
            position_mean="any",
            orientation_mean="any",
            centroids="any",
            position_time="any",
            processor="any",
        )


def test_vid_maker_filenotfound(sgp):
    with pytest.raises(FileNotFoundError):
        sgp.v1.dlc_utils_makevid.VideoMaker(
            video_filename="any",
            position_mean="any",
            orientation_mean="any",
            centroids="any",
            position_time="any",
        )


def test_recent_files(sgp):
    with open("temp1.txt", "w") as f:
        f.write("test1")
    sleep(1)  # Ensure a time difference
    with open("temp2.txt", "w") as f:
        f.write("test2")
    recent = sgp.utils.get_most_recent_file(".", "txt")
    assert str(recent) == "temp2.txt", f"Expected temp2.txt, got {recent}"
