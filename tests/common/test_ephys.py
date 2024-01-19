import pytest
from numpy import array_equal


def test_create_from_config(mini_insert, common_ephys, mini_path):
    before = common_ephys.Electrode().fetch()
    common_ephys.Electrode.create_from_config(mini_path.stem)
    after = common_ephys.Electrode().fetch()
    # Because already inserted, expect no change
    assert array_equal(
        before, after
    ), "Electrode.create_from_config had unexpected effect"


def test_raw_object(mini_insert, common_ephys, mini_dict, mini_content):
    obj_fetch = common_ephys.Raw().nwb_object(mini_dict).object_id
    obj_raw = mini_content.get_acquisition().object_id
    assert obj_fetch == obj_raw, "Raw.nwb_object did not return expected object"


def test_set_lfp_electrodes(mini_insert, common_ephys, mini_copy_name):
    before = common_ephys.LFPSelection().fetch()
    common_ephys.LFPSelection().set_lfp_electrodes(mini_copy_name, [0])
    after = common_ephys.LFPSelection().fetch()
    # Because already inserted, expect no change
    assert (
        len(after) == len(before) + 1
    ), "Set LFP electrodes had unexpected effect"


@pytest.mark.skip(reason="Not testing V0: common lfp")
def test_lfp():
    pass
