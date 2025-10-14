import pytest
from numpy import array_equal


def test_invalid_device(common, populate_exception, mini_insert):
    device_dict = common.DataAcquisitionDevice.fetch(as_dict=True)[0]
    device_dict["data_acquisition_device_system"] = "invalid"
    with pytest.raises(populate_exception):
        common.DataAcquisitionDevice._add_device(device_dict, test_mode=True)


def test_get_device(common, mini_content):
    dev = common.DataAcquisitionDevice.get_all_device_names(
        nwbf=mini_content, config=[]
    )
    assert len(dev) == 3, "Unexpected number of devices found"


def test_spike_gadgets_system_alias(mini_insert, common):
    assert (
        common.DataAcquisitionDevice()._add_system("MCU") == "SpikeGadgets"
    ), "SpikeGadgets MCU alias not found"


def test_replace_nan_with_default(utils):
    """Test that NaN values in probe geometry fields are properly replaced with -1.0."""
    # Test with NaN values (similar to the issue case)
    test_data = {
        "probe_id": "nTrode32_probe description",
        "probe_shank": 0,
        "contact_size": float("nan"),
        "probe_electrode": 194,
        "rel_x": float("nan"),
        "rel_y": float("nan"),
        "rel_z": float("nan"),
    }

    result = utils.dj_helper_fn._replace_nan_with_default(test_data)

    # Check that NaN values were replaced with -1.0
    assert result["contact_size"] == -1.0
    assert result["rel_x"] == -1.0
    assert result["rel_y"] == -1.0
    assert result["rel_z"] == -1.0

    # Check that non-NaN values were preserved
    assert result["probe_id"] == "nTrode32_probe description"
    assert result["probe_shank"] == 0
    assert result["probe_electrode"] == 194
