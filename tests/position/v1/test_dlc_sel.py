import datajoint as dj
import pytest


def test_dlc_video_default(sgp):
    expected_default = {
        "dlc_pos_video_params_name": "default",
        "params": {
            "incl_likelihood": True,
            "percent_frames": 1,
            "video_params": {"arrow_radius": 20, "circle_radius": 6},
        },
    }

    # run twice to trigger fetch existing
    assert sgp.v1.DLCPosVideoParams.get_default() == expected_default
    assert sgp.v1.DLCPosVideoParams.get_default() == expected_default


def test_dlc_video_populate(populate_dlc_video):
    assert hasattr(populate_dlc_video, "fps"), "Make failed to return object"


def test_null_position(sgp, mini_dict):
    pos_tbl = sgp.v1.position_dlc_selection.DLCPosV1()
    key = pos_tbl.make_null_position_nwb(mini_dict)
    df = (pos_tbl & key).fetch1_dataframe()
    assert df.shape == (1, 0), "Null position dataframe is not empty"


def test_multi_likelihood_error(sgp):
    """Test that an error is raised when multiple likelihoods are selected"""
    assert False


@pytest.fixture(scope="module")
def dlc_pose_tables(sgp):
    """Fixture to create and return the DLCPosV1 and DLCPoseEstimation tables"""
    pos_tbl = sgp.v1.position_dlc_selection.DLCPosV1()
    estim_tbl = sgp.v1.position_dlc_selection.DLCPoseEstimation()
    key = (pos_tbl & dj.Top(limit=1)).fetch1("KEY")

    return pos_tbl, estim_tbl, key


def test_pose_dataframe(dlc_pose_tables):
    pos_tbl, estim_tbl, key = dlc_pose_tables
    df1 = (pos_tbl & key).fetch_pose_dataframe()
    df2 = (estim_tbl & key).fetch_dataframe()
    assert (
        df1.shape == df2.shape and df1.columns == df2.columns
    ), "Pose dataframes do not match"


def test_pose_video_path(dlc_pose_tables):
    pos_tbl, estim_tbl, key = dlc_pose_tables
    path1 = (pos_tbl & key).fetch_video_path()
    path2 = (estim_tbl & key).fetch1("video_path")
    assert path1 == path2, "Pose video paths do not match"
