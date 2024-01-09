import pytest
from numpy import array_equal


def test_invalid_device(common, populate_exception):
    device_dict = common.DataAcquisitionDevice.fetch(as_dict=True)[0]
    device_dict["other"] = "invalid"
    with pytest.raises(populate_exception):
        common.DataAcquisitionDevice._add_device(device_dict)


def test_spikegadets_system_alias(mini_insert, common):
    assert (
        common.DataAcquisitionDevice()._add_system("MCU") == "SpikeGadgets"
    ), "SpikeGadgets MCU alias not found"


def test_invalid_probe(common, populate_exception):
    probe_dict = common.ProbeType.fetch(as_dict=True)[0]
    probe_dict["other"] = "invalid"
    with pytest.raises(populate_exception):
        common.Probe._add_probe_type(probe_dict)


def test_create_probe(common, mini_devices, mini_path, mini_copy_name):
    probe_id = common.Probe.fetch("KEY", as_dict=True)[0]
    probe_type = common.ProbeType.fetch("KEY", as_dict=True)[0]
    before = common.Probe.fetch()
    common.Probe.create_from_nwbfile(
        nwb_file_name=mini_copy_name,
        nwb_device_name="probe 0",
        contact_side_numbering=False,
        **probe_id,
        **probe_type,
    )
    after = common.Probe.fetch()
    # Because already inserted, expect no change
    assert array_equal(
        before, after
    ), "Probe create_from_nwbfile had unexpected effect"
