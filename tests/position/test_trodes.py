import pytest
from datajoint.hash import key_hash


@pytest.fixture(scope="session")
def params_table(trodes_params_table):
    return trodes_params_table


def test_add_params(params_table, trodes_params):
    tbl = params_table
    assert tbl & tbl.default_params, "Failed to add default params"
    assert tbl & trodes_params, "Failed to add custom params"


def test_param_keys(params_table):
    exp = set(params_table.default_params.keys())
    act = set(params_table.get_accepted_params())
    assert exp == act, "Accepted params do not match default params"


@pytest.fixture(scope="session")
def sel_table(teardown, params_table, trodes_sel_table, pos_interval_key):
    new_name = "led_back"
    restr_dict = {"trodes_pos_params_name": new_name}
    trodes_sel_table.insert_with_default(
        pos_interval_key,
        skip_duplicates=True,
        edit_defaults={"led1_is_front": 0},
        edit_name=new_name,
    )
    yield trodes_sel_table & restr_dict


def test_sel_default(sel_table):
    assert sel_table, "Add with default func failed"


def test_sel_insert_error(trodes_sel_table, pos_interval_key):
    bad_key = {"nwb_file_name": "BAD NAME"}
    with pytest.raises(ValueError):
        trodes_sel_table.insert_with_default(key=bad_key)
    with pytest.raises(ValueError):
        trodes_sel_table.insert_with_default(
            key=pos_interval_key, edit_defaults=bad_key
        )


def test_fetch_df(trodes_pos_v1, trodes_params):
    upsampled = {"trodes_pos_params_name": "single_led_upsampled"}
    df = (
        (trodes_pos_v1 & upsampled)
        .fetch1_dataframe(add_frame_ind=True)
        .round(3)
        .sum()
        .to_dict()
    )
    exp = {
        "position_x": 230389.335,
        "position_y": 295368.260,
        "orientation": 4716.906,
        "velocity_x": 1726.304,
        "velocity_y": -1675.276,
        "speed": 6257.273,
    }
    for k in exp:
        assert (
            pytest.approx(df[k], rel=1e-3) == exp[k]
        ), f"Value differs from expected: {k}"


def test_trodes_video(sgp, trodes_pos_v1):
    _ = trodes_pos_v1  # ensure table is populated
    vid_tbl = sgp.v1.TrodesPosVideo()
    _ = vid_tbl.populate()
    assert len(vid_tbl) == 2, "Failed to populate TrodesPosVideo"
