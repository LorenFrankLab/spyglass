import numpy as np
import pytest

from .conftest import generate_led_df


@pytest.fixture(scope="session")
def centroid_df(sgp, centroid_key, populate_centroid):
    yield (sgp.v1.DLCCentroid & centroid_key).fetch1_dataframe()


def test_centroid_fetch1_dataframe(centroid_df):
    df_cols = centroid_df.columns
    exp_cols = [
        "video_frame_ind",
        "position_x",
        "position_y",
        "velocity_x",
        "velocity_y",
        "speed",
    ]

    assert all(
        e in df_cols for e in exp_cols
    ), f"Unexpected cols in position merge dataframe: {df_cols}"


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
    params["dlc_centroid_params_name"] = "other test"
    params_tbl.insert1(params, skip_duplicates=True)


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
