import pytest
from numpy import isclose as np_isclose


def test_si_params_default(sgp):
    assert sgp.v1.DLCSmoothInterpParams.get_default() == {
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
    assert sgp.v1.DLCSmoothInterpParams.get_nan_params() == {
        "dlc_si_params_name": "just_nan",
        "params": {
            "interpolate": False,
            "likelihood_thresh": 0.95,
            "max_cm_between_pts": 20,
            "num_inds_to_span": 20,
            "smooth": False,
        },
    }


@pytest.fixture(scope="session")
def si_df(sgp, si_key, populate_si, bodyparts):
    yield (
        sgp.v1.DLCSmoothInterp() & {**si_key, "bodypart": bodyparts[0]}
    ).fetch1_dataframe()


@pytest.mark.parametrize(
    "column, exp_sum",
    [
        ("video_frame_ind", 36312),
        ("x", 17987),
        ("y", 2983),
    ],
)
def test_centroid_fetch1_dataframe(si_df, column, exp_sum):
    tolerance = abs(si_df[column].iloc[0] * 0.1)
    assert np_isclose(
        si_df[column].sum(), exp_sum, atol=tolerance
    ), f"Sum of {column} in SmoothInterp dataframe is not as expected"
