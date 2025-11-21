import pytest


def test_decode_param_fetch(decode_v1, decode_clusterless_params_insert):
    from non_local_detector.environment import Environment

    key = decode_clusterless_params_insert
    ret = (decode_v1.core.DecodingParameters & key).fetch1()["decoding_params"]
    env = ret["environments"][0]
    assert isinstance(env, Environment), "fetch failed to restore class"


def test_null_pos_group(caplog, decode_v1, pop_pos_group):
    file, group = pop_pos_group.fetch1("nwb_file_name", "position_group_name")
    pop_pos_group.create_group(file, group, ["dummy_pos"])
    assert "already exists" in caplog.text


def test_upsampled_pos_group(pop_pos_group_upsampled):
    ret = pop_pos_group_upsampled.fetch_position_info()[0]
    sample_freq = ret.index.to_series().diff().mode().iloc[0]
    pytest.approx(sample_freq, 0.001) == 1 / 250, "Upsampled data not at 250 Hz"


def test_position_group_non_chronological_order(pop_pos_group):
    """Test that fetch_position_info handles non-chronological merge_id order.

    This test verifies that when position data from multiple merge_ids is
    concatenated, the result is properly sorted by time index before slicing.
    This prevents returning an empty dataframe when merge_ids are not in
    chronological order.
    """
    # Fetch position info - internally may not be in chronological order
    position_info, position_variables = pop_pos_group.fetch_position_info()

    # Verify the dataframe is not empty
    assert len(position_info) > 0, "Position info should not be empty"

    # Verify the index is sorted (monotonically increasing)
    assert (
        position_info.index.is_monotonic_increasing
    ), "Position info index should be sorted in chronological order"

    # Verify position variables are present
    assert position_variables is not None
    assert len(position_variables) > 0
