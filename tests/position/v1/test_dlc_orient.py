import numpy as np
import pandas as pd
import pytest

from .conftest import generate_led_df


def test_insert_params(sgp):
    params_name = "test_params"
    params_key = {"dlc_orientation_params_name": params_name}
    params_tbl = sgp.v1.DLCOrientationParams()
    params_tbl.insert_params(
        params_name=params_name, params={}, skip_duplicates=True
    )
    assert params_tbl & params_key, "Failed to insert params"

    defaults = params_tbl.get_default()
    assert (
        defaults.get("params", {}).get("bodypart1") == "greenLED"
    ), "Failed to insert default params"

    # get after insert
    assert defaults == params_tbl.get_default(), "Failed to get default params"


def test_orient_fetch1_dataframe(sgp, orient_key, populate_orient):
    """Fetches dataframe, but example data has one led, no orientation"""
    fetched_df = (sgp.v1.DLCOrientation & orient_key).fetch1_dataframe()
    assert isinstance(fetched_df, pd.DataFrame)


@pytest.mark.parametrize(
    "key, points, exp_sum",
    [
        ("none", ["none"], 0.0),
        ("red_green_orientation", ["bodypart1", "bodypart2"], -2.356),
        ("red_led_bisector", ["led1", "led2", "led3"], -1.571),
    ],
)
def test_orient_calcs(sgp, key, points, exp_sum):
    func = sgp.v1.position_dlc_orient._key_to_func_dict[key]

    df = generate_led_df(points, inc_vals=True)
    df_sum = np.nansum(func(df, **{p: p for p in points}))

    assert np.isclose(
        df_sum, exp_sum, atol=0.001
    ), f"Failed to calculate orient via {key}"
