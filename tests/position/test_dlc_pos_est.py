import pytest


@pytest.fixture(scope="session")
def pos_est_sel(sgp):
    yield sgp.v1.position_dlc_pose_estimation.DLCPoseEstimationSelection()


@pytest.mark.usefixtures("skipif_no_dlc")
def test_rename_non_default_columns(sgp, common, pos_est_sel, video_keys):
    vid_path, vid_name, _, _ = sgp.v1.dlc_utils.get_video_info(video_keys[0])

    input = "0, 10, 0, 1000"
    output = pos_est_sel.get_video_crop(vid_path + vid_name, input)
    expected = [0, 10, 0, 1000]

    assert (
        output == expected
    ), f"{pos_est_sel.table_name}.get_video_crop did not return expected output"


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
