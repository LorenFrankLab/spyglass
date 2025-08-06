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
    "key", ["one_pt_centroid", "two_pt_centroid", "four_led_centroid"]
)
def test_centroid_calcs(key, sgp):
    from spyglass.position.v1.dlc_utils import Centroid

    points = sgp.v1.position_dlc_centroid._key_to_points[key]

    df = generate_led_df(points)
    ret = Centroid(
        df, max_LED_separation=100, points={p: p for p in points}
    ).centroid

    fail_msg = f"Centroid calculation failed for {key}"
    assert np.all(ret[:-1] == 1), fail_msg
    assert np.all(np.isnan(ret[-1])), fail_msg


def test_centroid_error(sgp):
    from spyglass.position.v1.dlc_utils import Centroid

    one_pt = {"point1": "point1"}
    df = generate_led_df(one_pt)
    Centroid(df, points=one_pt)  # no sep ok on one point

    two_pt = {"point1": "point1", "point2": "point2"}
    with pytest.raises(ValueError):
        Centroid(df, points=two_pt)  # Missing led separation for valid points

    three_pt = {"point1": "point1", "point2": "point2", "point3": "point3"}
    three_pt_df = generate_led_df(three_pt)
    with pytest.raises(ValueError):  # invalid point number
        Centroid(three_pt_df, points=three_pt, max_LED_separation=100)


@pytest.fixture(scope="session")
def centroid_tbls(sgp):
    from spyglass.position.v1.position_dlc_centroid import (
        DLCCentroid,
        DLCCentroidParams,
        DLCCentroidSelection,
    )

    yield DLCCentroidParams, DLCCentroidSelection, DLCCentroid


@pytest.fixture(scope="session")
def centroid_selection_key(sgp, cohort_key, centroid_tbls):
    _, _, DLCCentroid = centroid_tbls

    yield {k: v for k, v in cohort_key.items() if k in DLCCentroid.primary_key}


def test_centroid_null(centroid_tbls, centroid_selection_key):
    DLCCentroidParams, DLCCentroidSelection, DLCCentroid = centroid_tbls

    params_pk = dict(dlc_centroid_params_name="null")
    upstream = dict(**params_pk, **centroid_selection_key)

    DLCCentroidParams.insert1(
        dict(
            params_pk,
            params=dict(
                centroid_method="null", points=dict(), max_LED_separation=0
            ),
        ),
        skip_duplicates=True,
    )
    DLCCentroidSelection().insert1(upstream, skip_duplicates=True)
    DLCCentroid().populate(upstream)
    df = (DLCCentroid() & upstream).fetch1_dataframe()

    assert df.empty, "Centroid with no points should be empty"


def test_centoid_point_err(centroid_tbls, params_tbl, centroid_selection_key):
    DLCCentroidParams, DLCCentroidSelection, DLCCentroid = centroid_tbls

    params_pk = dict(dlc_centroid_params_name="point_err")
    upstream = dict(**params_pk, **centroid_selection_key)

    query = DLCCentroidParams & params_pk
    if query:
        query.delete(safemode=False)
    DLCCentroidParams.insert1(
        dict(
            params_pk,
            params=dict(
                points=dict(point1="bad"),
                centroid_method="one_pt_centroid",
                max_LED_separation=0,
            ),
        ),
        skip_duplicates=True,
    )
    DLCCentroidSelection().insert1(upstream, skip_duplicates=True)

    with pytest.raises(ValueError):
        DLCCentroid().populate(upstream)


def test_centoid_no_process(
    centroid_tbls, centroid_selection_key, centroid_key
):
    """Test no interpolate, no smooth"""
    DLCCentroidParams, DLCCentroidSelection, DLCCentroid = centroid_tbls

    params_pk = dict(dlc_centroid_params_name="no_interpolate")
    upstream = dict(**params_pk, **centroid_selection_key)
    params_key = dict(
        params_pk,
        params=dict(
            centroid_method="one_pt_centroid",
            points=dict(point1="whiteLED"),
            interpolate=False,
            smooth=False,
            max_LED_separation=0,
            speed_smoothing_std_dev=0.1,
        ),
    )

    DLCCentroidParams.insert1(params_key, skip_duplicates=True)
    DLCCentroidSelection().insert1(upstream, skip_duplicates=True)
    DLCCentroid().populate(upstream)
    df_no_interp = (DLCCentroid() & upstream).fetch1_dataframe()
    df_interp = (DLCCentroid() & centroid_key).fetch1_dataframe()

    avg_interp_speed = df_interp["speed"].mean()
    avg_no_interp_speed = df_no_interp["speed"].mean()
    assert avg_interp_speed != avg_no_interp_speed, "Speed should be different"
