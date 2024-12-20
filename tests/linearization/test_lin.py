import pytest


def test_fetch1_dataframe(lin_v1, lin_merge, lin_merge_key):
    df = (lin_merge & lin_merge_key).fetch1_dataframe().round(3).sum().to_dict()
    exp = {
        "linear_position": 3249449.258,
        "projected_x_position": 472245.797,
        "projected_y_position": 317857.473,
        "track_segment_id": 31158.0,
    }

    for k in exp:
        assert (
            pytest.approx(df[k], rel=1e-3) == exp[k]
        ), f"Value differs from expected: {k}"


# TODO: Add more tests of this pipeline, not just the fetch1_dataframe method
