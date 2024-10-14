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
