import pytest
from numpy import isclose as np_isclose


@pytest.fixture(scope="session")
def centroid_df(sgp, centroid_key):
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
