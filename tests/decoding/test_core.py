def test_decode_param_fetch(decode_v1, decode_clusterless_params_insert):
    _ = decode_clusterless_params_insert
    ret = decode_v1.DecodingParameters.fetch()
    assert ret is not None
    __import__("pdb").set_trace()


def test_null_pos_group(caplog, decode_v1, pop_pos_group):
    file, group = pop_pos_group.fetch("nwb_file_name", "position_group_name")[0]
    pop_pos_group.create_group(file, group, ["dummy_pos"])
    assert "already exists" in caplog.text


def test_upsampled_pos_group(pop_pos_group_upsampled):
    ret = pop_pos_group_upsampled.fetch_posision_info()
    assert ret is not None
    __import__("pdb").set_trace()
