import datajoint as dj
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def si_params_tbl(sgp):
    yield sgp.v1.DLCSmoothInterpParams()


def test_si_params_default(si_params_tbl):
    default1 = {
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
    assert si_params_tbl.get_default() == default1, "default params error"
    # rerun to cover fetch1 case
    assert si_params_tbl.get_default() == default1, "default params reran"
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


def test_si_params_insert(si_params_tbl):
    params_name = "test_params"
    si_params_tbl.insert_params(
        params_name=params_name,
        params={
            "max_cm_between_pts": 20,
            "likelihood_thresh": 0.2,
            "smooth": False,
            "smoothing_params": {
                "smooth_method": "moving_avg",
                "smoothing_duration": 0.05,
            },
        },
        skip_duplicates=True,
    )
    assert si_params_tbl & {
        "dlc_si_params_name": params_name
    }, "Failed to insert params"


def test_si_interpolate(sgp, si_params_tbl, si_key, pose_estimation_key):
    """Tests interpolation and smoothing"""
    _ = si_key
    params = si_params_tbl.get_nan_params()
    _ = params.pop("dlc_si_params_name")
    params["params"].update(
        dict(
            interpolate=dict(max_cm_to_interp=15),
            smooth=True,
            likelihood_thresh=0.5,
            smoothing_params=dict(
                smooth_method="moving_avg", smoothing_duration=0.05
            ),
        )
    )
    params_key = dict(dlc_si_params_name="test_interpolate")
    si_params_tbl.insert1(dict(params_key, **params), skip_duplicates=True)

    sel_tbl = sgp.v1.DLCSmoothInterpSelection()
    si_tbl = sgp.v1.DLCSmoothInterp()
    sel_key = sel_tbl.fetch("KEY", as_dict=True)[0]
    sel_key.update(params_key)
    sel_tbl.insert1(sel_key, skip_duplicates=True)
    si_tbl.populate(sel_key)

    cols = (si_tbl & sel_key).fetch1_dataframe().columns.tolist()
    assert cols == ["video_frame_ind", "x", "y"], f"Unexpected cols: {cols}"


@pytest.fixture(scope="session")
def si_df(sgp, si_key, populate_si, bodyparts):
    _ = si_key, populate_si, bodyparts
    yield (sgp.v1.DLCSmoothInterp() & dj.Top()).fetch1_dataframe()


def test_cohort_fetch1_dataframe(si_df):
    df_cols = si_df.columns
    exp_cols = ["video_frame_ind", "x", "y"]
    assert all(
        e in df_cols for e in exp_cols
    ), f"Unexpected cols in DLCSmoothInterp dataframe: {df_cols}"


def test_all_nans(populate_pose_estimation, sgp):
    pose_est_tbl = populate_pose_estimation
    restr_tbl = pose_est_tbl.BodyPart & dj.Top()
    df = restr_tbl.fetch1_dataframe()
    with pytest.raises(ValueError):
        sgp.v1.position_dlc_position.nan_inds(df, 10, 0.99, 10)


# Parameterized test cases using pytest.mark.parametrize
@pytest.mark.parametrize(
    "bad_inds_mask, inds_to_span, expected_good, expected_modified",
    [
        # Test 1: No good spans (all bad)
        (np.array([True, True, True, True]), 50, None, []),
        # Test 2: One good span (no modification needed)
        (np.array([True, False, False, True]), 50, [(1, 2)], [(1, 2)]),
        # Test 3: Two good spans, and they merge
        (
            np.array([True, False, False, True, False, False, True]),
            50,
            [(1, 2), (4, 5)],
            [(1, 5)],
        ),
        (  # Test 4: Two good spans that do not merge (gap > threshold)
            np.array([True, False, True, False, True, False]),
            1,
            [(1, 1), (3, 3), (5, 5)],
            [(1, 1), (3, 3), (5, 5)],
        ),
        (  # Test 5: Multiple good spans with an exact gap for merging
            np.array([True, False, True, False, True, False]),
            2,
            [(1, 1), (3, 3), (5, 5)],
            [(1, 5)],
        ),
        # Test 6: Continuous good indices (no gaps)
        (np.array([True, False, False, False, True]), 50, [(1, 3)], [(1, 3)]),
    ],
)
def test_get_good_spans(
    bad_inds_mask, inds_to_span, expected_good, expected_modified, sgp
):
    get_good_spans = sgp.v1.position_dlc_position.get_good_spans
    good, modified_spans = get_good_spans(bad_inds_mask, inds_to_span)

    # Assert the results match the expected values
    assert good == expected_good
    assert modified_spans == expected_modified


def test_subthresh_percent(sgp):
    np.random.seed(0)
    set_percent, set_len, low, cutoff, high = 50, 10, 0.1, 0.5, 0.9
    set_count = int(set_len * set_percent / 100)
    xvals, yvals = np.random.rand(set_len), np.random.rand(set_len)
    likelihood = np.array([low] * set_count + [high] * (set_len - set_count))
    df = pd.DataFrame({"x": xvals, "y": yvals, "likelihood": likelihood})

    _, sub_percent = sgp.v1.position_dlc_position.get_subthresh_inds(
        dlc_df=df, likelihood_thresh=cutoff, ret_sub_thresh=True
    )
    assert sub_percent == set_percent, f"Unexpected subthresh %: {sub_percent}"


@pytest.fixture
def dlc_df_with_nans():
    data = {
        "x": [0, 1, np.nan, 3, 4, 5],  # NaN at position 2 inside the span (1,4)
        "y": [0, 2, 2, 4, 5, 6],
        "likelihood": [0.9, 0.95, 0.1, 0.85, 0.9, 0.9],
    }
    df = pd.DataFrame(data, index=[0, 1, 2, 3, 4, 5])
    return df


def test_nan_inds_covers_isnan_branch(sgp, dlc_df_with_nans):
    dlc_out, bad_mask = sgp.v1.position_dlc_position.nan_inds(
        dlc_df_with_nans.copy(), 10, 0.2, 1
    )

    # Check that NaN remains or gets set in positions as expected
    # Position 2 has low likelihood so should be NaN
    assert np.isnan(dlc_out.iloc[2]["x"])
    assert np.isnan(dlc_out.iloc[2]["y"])
    assert bad_mask[2]
    assert not np.isnan(dlc_out.iloc[1]["x"])
    assert not np.isnan(dlc_out.iloc[1]["y"])
