import pytest
from numpy import isclose as np_isclose


@pytest.fixture(scope="session")
def merge_df(sgp, pos_merge, dlc_key, populate_dlc):
    merge_key = (pos_merge.DLCPosV1 & dlc_key).fetch1("KEY")
    yield (pos_merge & merge_key).fetch1_dataframe()


@pytest.mark.parametrize(
    "column, exp_sum",
    [  # NOTE: same as test_centroid_fetch1_dataframe
        ("video_frame_ind", 36312),
        ("position_x", 17987),
        ("position_y", 2983),
        ("velocity_x", -1.489),
        ("velocity_y", 4.160),
        ("speed", 12957),
    ],
)
def test_merge_dlc_fetch1_dataframe(merge_df, column, exp_sum):
    tolerance = abs(merge_df[column].iloc[0] * 0.1)
    assert np_isclose(
        merge_df[column].sum(), exp_sum, atol=tolerance
    ), f"Sum of {column} in Merge.DLCPosV1 dataframe is not as expected"
