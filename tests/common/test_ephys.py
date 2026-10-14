import pytest
from numpy import array_equal

from ..conftest import TEARDOWN


def test_create_from_config(mini_insert, common_ephys, mini_copy_name):
    before = common_ephys.Electrode().fetch()
    common_ephys.Electrode.create_from_config(mini_copy_name)
    after = common_ephys.Electrode().fetch()
    # Because already inserted, expect no change
    assert array_equal(
        before, after
    ), "Electrode.create_from_config had unexpected effect"


def test_raw_object(mini_insert, common_ephys, mini_dict, mini_content):
    obj_fetch = common_ephys.Raw().nwb_object(mini_dict).object_id
    obj_raw = mini_content.get_acquisition().object_id
    assert obj_fetch == obj_raw, "Raw.nwb_object did not return expected object"


def test_electrode_populate(common_ephys):
    common_ephys.Electrode.populate()
    assert len(common_ephys.Electrode()) == 128, "Electrode.populate failed"


def test_elec_group_populate(pop_common_electrode_group):
    assert (
        len(pop_common_electrode_group) == 32
    ), "ElectrodeGroup.populate failed"


def test_raw_populate(common_ephys):
    common_ephys.Raw.populate()
    assert len(common_ephys.Raw()) == 1, "Raw.populate failed"


def test_sample_count_populate(common_ephys):
    common_ephys.SampleCount.populate()
    assert len(common_ephys.SampleCount()) == 1, "SampleCount.populate failed"


@pytest.mark.skipif(not TEARDOWN, reason="No teardown: expect no change.")
def test_set_lfp_electrodes(mini_insert, common_ephys, mini_copy_name):
    before = common_ephys.LFPSelection().fetch()
    common_ephys.LFPSelection().set_lfp_electrodes(mini_copy_name, [0])
    after = common_ephys.LFPSelection().fetch()
    assert (
        len(after) == len(before) + 1
    ), "Set LFP electrodes had unexpected effect"


@pytest.mark.skip(reason="Not testing V0: common lfp")
def test_lfp():
    pass


def test_one_rate_estimate_serves_the_whole_mapping(
    common_ephys, mini_copy_name, mini_path, monkeypatch
):
    """`sampling_rate` and `valid_times` share one rate estimate.

    `table_key_to_obj_attr` maps `sampling_rate` to `_rate_fallback` and
    `valid_times` to `_valid_times_from_raw`, which needs the rate too. Each
    estimate reads a million timestamps off disk, so without the memo the
    mapping paid for two.
    """
    import pynwb

    import spyglass.common.common_ephys as ce

    calls = []
    original = ce.estimate_sampling_rate

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ce, "estimate_sampling_rate", counted)

    copy = mini_path.parent / mini_copy_name
    with pynwb.NWBHDF5IO(str(copy), mode="r", load_namespaces=True) as io:
        eseries = io.read().acquisition["e-series"]
        entries = common_ephys.Raw()._entries_from_mapping(
            eseries, {"nwb_file_name": mini_copy_name}
        )

    key = next(iter(entries.values()))[0]

    assert len(calls) == 1, (
        f"One mapping should estimate the rate once; it ran {len(calls)} "
        "times -- the memo is not firing"
    )
    assert key["sampling_rate"], "The rate still reaches the key"
    assert len(key["valid_times"]), "And valid_times still computes"
