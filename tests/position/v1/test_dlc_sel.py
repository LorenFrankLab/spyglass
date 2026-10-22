import datajoint as dj
import numpy as np
import pynwb
import pytest

from tests.conftest import skip_if_no_pose


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
    assert populate_dlc_video, "DLCPosVideo table not populated correctly"


@pytest.mark.slow
def test_null_position(common, sgp, mini_dict):

    pos_tbl = sgp.v1.position_dlc_selection.DLCPosV1()
    fname = pos_tbl.make_null_position_nwb(mini_dict)["analysis_file_name"]
    fpath = common.AnalysisNwbfile().get_abs_path(fname)

    with pynwb.NWBHDF5IO(fpath, "r") as io:
        df = io.read().fields["scratch"]["pandas_table"].to_dataframe()

    assert df.shape == (0, 0), "Null position dataframe is not empty"


@pytest.fixture(scope="module")
def dlc_pose_tables(sgp):
    """Fixture to create and return the DLCPosV1 and DLCPoseEstimation tables"""
    pos_tbl = sgp.v1.position_dlc_selection.DLCPosV1()
    estim_tbl = sgp.v1.position_dlc_selection.DLCPoseEstimation()
    key = (pos_tbl & dj.Top(limit=1)).fetch1("KEY")

    return pos_tbl, estim_tbl, key


@skip_if_no_pose
def test_pose_dataframe(dlc_pose_tables):
    pos_tbl, estim_tbl, key = dlc_pose_tables
    df1 = (pos_tbl & key).fetch_pose_dataframe()
    df2 = (estim_tbl & key).fetch_dataframe()
    assert df1.shape == df2.shape and np.all(
        df1.columns == df2.columns
    ), "Pose dataframes do not match"


@skip_if_no_pose
def test_pose_video_path(sgp, dlc_pose_tables):
    pos_tbl, _, key = dlc_pose_tables
    path1 = (pos_tbl & key).fetch_video_path()
    estim_sel_tbl = sgp.v1.position_dlc_selection.DLCPoseEstimationSelection()
    path2 = (estim_sel_tbl & key).fetch1("video_path")
    assert path1 == path2, "Pose video paths do not match"


class TestV1VideoKeyword:
    """V1 calls `make_video(centroids=...)`; the shared class takes
    `bodypart_centroids`.

    `VideoMaker.__init__` collects unknown keywords in `**kwargs`, so
    repointing V1 at the shared module without the synonym would drop
    `centroids` silently and raise TypeError for a missing
    `bodypart_centroids`.
    """

    KWARGS = dict(
        video_filename="any",
        position_mean="any",
        orientation_mean="any",
        position_time="any",
    )

    def test_centroids_reaches_bodypart_centroids(self):
        from spyglass.position.utils.make_video import make_video

        # processor="any" fails validation, which happens after binding -- so
        # ValueError, not TypeError, proves the keyword was accepted.
        with pytest.raises(ValueError, match="processor"):
            make_video(**self.KWARGS, centroids="any", processor="any")

    def test_missing_centroids_still_raises(self):
        from spyglass.position.utils.make_video import make_video

        with pytest.raises(TypeError, match="bodypart_centroids"):
            make_video(**self.KWARGS, processor="any")
