import pandas as pd


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


def test_si_fetch1_dataframe(sgp, si_key, populate_si, bodyparts):
    fetched_df = (
        sgp.v1.DLCSmoothInterp() & {**si_key, "bodypart": bodyparts[0]}
    ).fetch1_dataframe()
    assert isinstance(fetched_df, pd.DataFrame)
    raise NotImplementedError
