import pytest


@pytest.fixture(scope="session")
def si_params_tbl(sgp):
    yield sgp.v1.DLCSmoothInterpParams()


def test_si_params_default(si_params_tbl):
    assert si_params_tbl.get_default() == {
        "dlc_si_params_name": "default",
        "params": {
            "interp_params": {"max_cm_to_interp": 15},
            "interpolate": True,
            "likelihood_thresh": 0.95,
            "max_cm_between_pts": 20,
            "num_inds_to_span": 20,
            "smooth": True,
            "smoothing_params": {
                "smooth_method": "moving_avg",
                "smoothing_duration": 0.05,
            },
        },
    }
    assert si_params_tbl.get_nan_params() == {
        "dlc_si_params_name": "just_nan",
        "params": {
            "interpolate": False,
            "likelihood_thresh": 0.95,
            "max_cm_between_pts": 20,
            "num_inds_to_span": 20,
            "smooth": False,
        },
    }
    assert list(si_params_tbl.get_available_methods()) == [
        "moving_avg"
    ], f"{si_params_tbl.table_name}: unexpected available methods"


def test_invalid_params_insert(si_params_tbl):
    with pytest.raises(KeyError):
        si_params_tbl.insert1({"params": "invalid"})


@pytest.fixture(scope="session")
def si_df(sgp, si_key, populate_si, bodyparts):
    yield (
        sgp.v1.DLCSmoothInterp() & {**si_key, "bodypart": bodyparts[0]}
    ).fetch1_dataframe()


def test_cohort_fetch1_dataframe(si_df):
    df_cols = si_df.columns
    exp_cols = ["video_frame_ind", "x", "y"]
    assert all(
        e in df_cols for e in exp_cols
    ), f"Unexpected cols in DLCSmoothInterp dataframe: {df_cols}"


def test_all_nans(populate_pose_estimation, sgp):
    pose_est_tbl = populate_pose_estimation
    df = pose_est_tbl.BodyPart().fetch1_dataframe()
    with pytest.raises(ValueError):
        sgp.v1.position_dlc_position.nan_inds(df, 10, 0.99, 10)
