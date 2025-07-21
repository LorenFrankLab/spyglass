from pathlib import Path

import datajoint as dj
import pytest

from tests.conftest import skip_if_no_dlc


@pytest.fixture(scope="session")
def pos_est_sel(sgp):
    yield sgp.v1.position_dlc_pose_estimation.DLCPoseEstimationSelection()


@pytest.fixture(scope="session")
def vid_crop_tools(sgp, video_keys, pos_est_sel):
    vid_path, vid_name, _, _ = sgp.v1.dlc_utils.get_video_info(video_keys[0])
    get_video_crop = pos_est_sel.get_video_crop

    yield Path(vid_path) / vid_name, get_video_crop


@skip_if_no_dlc
def test_rename_non_default_columns(vid_crop_tools):
    vid_path, get_video_crop = vid_crop_tools

    input = "0, 10, 0, 1000"
    output = get_video_crop(vid_path, input)
    expected = [0, 10, 0, 1000]

    assert output == expected, "get_video_crop did not parse string"


@skip_if_no_dlc
def test_vid_crop_none(vid_crop_tools):
    vid_path, get_video_crop = vid_crop_tools
    output = get_video_crop(vid_path, "none")
    assert output is None, "get_video_crop did not detect 'none'"


@skip_if_no_dlc
def test_vid_crop_error(vid_crop_tools):
    vid_path, get_video_crop = vid_crop_tools
    with pytest.raises(ValueError):
        _ = get_video_crop(vid_path, "0, 10, 0, -10")


def test_invalid_video(pos_est_sel, pose_estimation_key):
    _ = pose_estimation_key  # Ensure populated
    example_key = pos_est_sel.fetch("KEY", as_dict=True)[0]
    example_key["nwb_file_name"] = "invalid.nwb"
    with pytest.raises(FileNotFoundError):
        pos_est_sel.insert_estimation_task(example_key)


def test_pose_est_dataframe(populate_pose_estimation):
    pose_cols = populate_pose_estimation.fetch_dataframe().columns

    for bp in ["tailBase", "tailMid", "tailTip"]:
        for val in ["video_frame_ind", "x", "y"]:
            col = (bp, val)
            assert col in pose_cols, f"PoseEstimation df missing column {col}."


@skip_if_no_dlc
def test_fetch_video_path(sgp):
    pose_tbl = sgp.v1.position_dlc_pose_estimation.DLCPoseEstimation()

    path = (pose_tbl & dj.Top(limit=1)).fetch_video_path()
    assert Path(path).exists(), f"Video path {path} does not exist."
