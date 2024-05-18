import numpy as np
import pytest
from numpy import isclose as np_isclose

from .conftest import generate_led_df


@pytest.fixture(scope="session")
def centroid_df(sgp, centroid_key, populate_centroid):
    yield (sgp.v1.DLCCentroid & centroid_key).fetch1_dataframe()


@pytest.mark.parametrize(
    "column, exp_sum",
    [
        ("video_frame_ind", 36312),
        ("position_x", 17987),
        ("position_y", 2983),
        ("velocity_x", -1.489),
        ("velocity_y", 4.160),
        ("speed", 12957),
    ],
)
def test_centroid_fetch1_dataframe(centroid_df, column, exp_sum):
    tolerance = abs(centroid_df[column].iloc[0] * 0.1)
    assert np_isclose(
        centroid_df[column].sum(), exp_sum, atol=tolerance
    ), f"Sum of {column} in Centroid dataframe is not as expected"


@pytest.fixture(scope="session")
def params_tbl(sgp):
    yield sgp.v1.DLCCentroidParams()


def test_insert_default_params(params_tbl):
    ret = params_tbl.get_default()
    assert "default" in params_tbl.fetch(
        "dlc_centroid_params_name"
    ), "Default params not inserted"
    assert (
        ret["dlc_centroid_params_name"] == "default"
    ), "Default params not inserted"


def test_validate_params(params_tbl):
    params = params_tbl.get_default()
    params["dlc_centroid_params_name"] = "test"
    params_tbl.insert1(params)


@pytest.mark.parametrize(
    "key", ["four_led_centroid", "two_pt_centroid", "one_pt_centroid"]
)
def test_centroid_calcs(key, sgp):
    points = sgp.v1.position_dlc_centroid._key_to_points[key]
    func = sgp.v1.position_dlc_centroid._key_to_func_dict[key]

    df = generate_led_df(points)
    ret = func(df, max_LED_separation=100, points={p: p for p in points})

    assert np.all(ret[:-1] == 1), f"Centroid calculation failed for {key}"
    assert np.all(np.isnan(ret[-1])), f"Centroid calculation failed for {key}"

    with pytest.raises(KeyError):
        func(df)  # Missing led separation/point names
