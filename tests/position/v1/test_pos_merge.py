import pytest


@pytest.fixture(scope="session")
def merge_df(sgp, pos_merge, dlc_key, populate_dlc):
    merge_key = (pos_merge.DLCPosV1 & dlc_key).fetch1("KEY")
    yield (pos_merge & merge_key).fetch1_dataframe()


def test_merge_dlc_fetch1_dataframe(merge_df):
    df_cols = merge_df.columns
    exp_cols = [
        "video_frame_ind",
        "position_x",
        "position_y",
        "orientation",
        "velocity_x",
        "velocity_y",
        "speed",
    ]

    assert all(
        e in df_cols for e in exp_cols
    ), f"Unexpected cols in position merge dataframe: {df_cols}"
