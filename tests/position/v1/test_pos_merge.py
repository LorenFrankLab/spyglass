from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def merge_df(sgp, pos_merge, dlc_key, populate_dlc):
    merge_key = (pos_merge.DLCPosV1 & dlc_key).fetch1("KEY")
    yield (pos_merge & merge_key).fetch1_dataframe()


def test_merge_dlc_fetch1_dataframe(merge_df):
    df_cols = merge_df.columns
    exp_cols = [
        "video_frame_ind",
        "position_x",
        "position_y",
        "orientation",
        "velocity_x",
        "velocity_y",
        "speed",
    ]

    assert all(
        e in df_cols for e in exp_cols
    ), f"Unexpected cols in position merge dataframe: {df_cols}"


def test_merge_fetch_pose_dataframe(pos_merge, dlc_key, populate_dlc):
    _ = populate_dlc
    merge_key = (pos_merge.DLCPosV1 & dlc_key).fetch1("KEY")
    df = (pos_merge & merge_key).fetch_pose_dataframe()
    assert not df.empty, "Pose dataframe is empty"

    level0 = df.columns.get_level_values(0).unique().tolist()
    level1 = df.columns.get_level_values(1).unique().tolist()
    expected_level0 = ["tailBase", "tailMid", "tailTip", "whiteLED"]
    expected_level1 = ["video_frame_ind", "x", "y", "likelihood"]
    assert level0 == expected_level0, f"Unexpected level 0 columns: {level0}"
    assert level1 == expected_level1, f"Unexpected level 1 columns: {level1}"


def test_merge_fetch_video_path(pos_merge, dlc_key, populate_dlc):
    _ = populate_dlc
    merge_key = (pos_merge.DLCPosV1 & dlc_key).fetch1("KEY")
    path = (pos_merge & merge_key).fetch_video_path()
    assert Path(path).exists(), f"Video path does not exist: {path}"
