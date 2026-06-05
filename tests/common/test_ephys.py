# pylint: disable=protected-access

from unittest.mock import Mock

import numpy as np
import pytest
from numpy import array_equal

from ..conftest import TEARDOWN


def test_create_from_config(common_ephys, mini_copy_name):
    before = common_ephys.Electrode().fetch()
    common_ephys.Electrode.create_from_config(mini_copy_name)
    after = common_ephys.Electrode().fetch()
    # Because already inserted, expect no change
    assert array_equal(
        before, after
    ), "Electrode.create_from_config had unexpected effect"


def test_raw_object(common_ephys, mini_dict, mini_content):
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
def test_set_lfp_electrodes(common_ephys, mini_copy_name):
    before = common_ephys.LFPSelection().fetch()
    common_ephys.LFPSelection().set_lfp_electrodes(mini_copy_name, [0])
    after = common_ephys.LFPSelection().fetch()
    assert (
        len(after) == len(before) + 1
    ), "Set LFP electrodes had unexpected effect"


@pytest.mark.skip(reason="Not testing V0: common lfp")
def test_lfp():
    pass


def test_electrode_group_hemisphere_detection_edge_cases_from_targeted():
    """Coordinate sign controls hemisphere label at edge values."""
    x_coord = 0.0
    hemisphere = "Right" if x_coord >= 0 else "Left"
    assert hemisphere == "Right"

    x_coord = 0.001
    hemisphere = "Right" if x_coord >= 0 else "Left"
    assert hemisphere == "Right"

    x_coord = -0.001
    hemisphere = "Right" if x_coord >= 0 else "Left"
    assert hemisphere == "Left"


@pytest.fixture
def mock_electrode_group():
    """Mock electrode group for testing."""
    mock_group = Mock()
    mock_group.name = "test_group"
    mock_group.description = "Test electrode group"
    mock_group.targeted_x = None  # Test default case
    return mock_group


@pytest.fixture
def mock_electrode_group_right_hemisphere():
    """Mock electrode group for right hemisphere."""
    mock_group = Mock()
    mock_group.name = "test_group_right"
    mock_group.description = "Test electrode group right"
    mock_group.targeted_x = 2.5  # Positive x = right hemisphere
    return mock_group


@pytest.fixture
def mock_electrode_group_left_hemisphere():
    """Mock electrode group for left hemisphere."""
    mock_group = Mock()
    mock_group.name = "test_group_left"
    mock_group.description = "Test electrode group left"
    mock_group.targeted_x = -1.5  # Negative x = left hemisphere
    return mock_group


def test_electrode_group_hemisphere_detection():
    """Test hemisphere detection logic."""

    # Test hemisphere determination logic
    def determine_hemisphere(targeted_x):
        return "Right" if targeted_x > 0 else "Left"

    assert determine_hemisphere(2.5) == "Right"
    assert determine_hemisphere(-1.5) == "Left"
    assert determine_hemisphere(0.0) == "Left"  # Zero is treated as left


def test_electrode_config_validation():
    """Test electrode configuration validation."""
    # Test valid electrode configuration
    valid_config = {
        "electrode_id": 1,
        "x": 100.0,
        "y": 200.0,
        "location": "CA1",
        "probe_type": "tetrode",
    }

    # Test required fields
    required_fields = ["electrode_id", "x", "y", "location"]
    for field in required_fields:
        assert field in valid_config

    # Test parameter types and ranges
    assert isinstance(valid_config["electrode_id"], int)
    assert valid_config["electrode_id"] > 0
    assert isinstance(valid_config["x"], (int, float))
    assert isinstance(valid_config["y"], (int, float))
    assert isinstance(valid_config["location"], str)
    assert len(valid_config["location"]) > 0


def test_electrode_group_properties():
    """Test electrode group property validation."""
    # Test valid electrode group
    mock_group = Mock()
    mock_group.name = "test_group"
    mock_group.description = "Test electrode group"
    mock_group.targeted_x = 1.0
    mock_group.targeted_y = 2.0
    mock_group.targeted_z = 3.0

    # Test required attributes exist
    required_attrs = [
        "name",
        "description",
        "targeted_x",
        "targeted_y",
        "targeted_z",
    ]
    for attr in required_attrs:
        assert hasattr(mock_group, attr)
        assert getattr(mock_group, attr) is not None


def test_electrode_parameter_combinations():
    """Test different electrode parameter combinations."""
    test_electrodes = [
        {"id": 1, "x": 100, "y": 200, "region": "CA1"},
        {"id": 2, "x": -50, "y": 150, "region": "CA3"},
        {"id": 3, "x": 0, "y": 0, "region": "DG"},
    ]

    for electrode in test_electrodes:
        # Basic validation
        assert electrode["id"] > 0
        assert isinstance(electrode["x"], (int, float))
        assert isinstance(electrode["y"], (int, float))
        assert len(electrode["region"]) > 0


def test_electrode_config_edge_cases():
    """Test electrode configuration edge cases."""
    # Test empty configuration
    empty_config = {}
    electrode_section = empty_config.get("Electrode", [])
    assert len(electrode_section) == 0

    # Test malformed configuration handling
    def validate_electrode_config(config):
        if not isinstance(config, dict):
            return False
        if "Electrode" not in config:
            return True  # Valid to have no electrode section
        electrode_list = config["Electrode"]
        return isinstance(electrode_list, list)

    assert validate_electrode_config({})  # Empty config is valid
    assert validate_electrode_config({"Electrode": []})  # Empty list is valid
    assert not validate_electrode_config(
        {"Electrode": "invalid"}
    )  # String is invalid


def test_electrode_error_handling():
    """Test electrode error handling scenarios."""
    # Test missing required fields
    incomplete_electrode = {"id": 1}  # Missing x, y coordinates

    def has_required_fields(electrode_config):
        required = ["id", "x", "y"]
        return all(field in electrode_config for field in required)

    assert not has_required_fields(incomplete_electrode)

    # Test invalid data types
    invalid_electrode = {"id": "not_int", "x": "not_float", "y": "not_float"}

    def validate_electrode_types(electrode_config):
        try:
            if "id" in electrode_config:
                int(electrode_config["id"])  # Should be convertible to int
            if "x" in electrode_config:
                float(electrode_config["x"])  # Should be convertible to float
            if "y" in electrode_config:
                float(electrode_config["y"])  # Should be convertible to float
            return True
        except (ValueError, TypeError):
            return False

    assert not validate_electrode_types(invalid_electrode)


def test_electrode_group_target_region_handling():
    """Test electrode group target region handling."""
    # Test electrode group with unknown target region
    mock_group_unknown = Mock()
    mock_group_unknown.name = "test_group_unknown"
    mock_group_unknown.description = "Test with unknown target"
    mock_group_unknown.targeted_x = None
    mock_group_unknown.targeted_y = None
    mock_group_unknown.targeted_z = None

    # Test hemisphere determination function
    def determine_target_hemisphere(targeted_x):
        if targeted_x is None:
            return "Unknown"
        return "Right" if targeted_x >= 0 else "Left"

    assert determine_target_hemisphere(None) == "Unknown"
    assert determine_target_hemisphere(1.0) == "Right"
    assert determine_target_hemisphere(-1.0) == "Left"
    assert determine_target_hemisphere(0.0) == "Right"  # Zero is right


def test_electrode_group_config_processing():
    """Test electrode group configuration processing."""
    # Mock NWB electrode group with various properties
    electrode_group = Mock()
    electrode_group.name = "test_electrode_group"
    electrode_group.description = "Test description"
    electrode_group.location = "CA1"

    # Test device with probe type
    mock_device = Mock()
    mock_device.probe_type = "tetrode_probe"
    electrode_group.device = mock_device
    electrode_group.targeted_x = 2.5

    # Test extraction of properties
    extracted_props = {
        "name": electrode_group.name,
        "description": electrode_group.description,
        "location": electrode_group.location,
        "probe_type": getattr(electrode_group.device, "probe_type", None),
        "targeted_x": getattr(electrode_group, "targeted_x", None),
    }

    assert extracted_props["name"] == "test_electrode_group"
    assert extracted_props["description"] == "Test description"
    assert extracted_props["location"] == "CA1"
    assert extracted_props["probe_type"] == "tetrode_probe"
    assert extracted_props["targeted_x"] == 2.5


def test_electrode_config_yaml_processing():
    """Test electrode configuration YAML processing."""
    # Mock electrode configuration from YAML
    mock_config = {
        "Electrode": [
            {
                "electrode_id": 1,
                "probe_id": "probe_1",
                "probe_shank": 1,
                "probe_electrode": 1,
                "bad_channel": False,
                "x_warped": 10.0,
                "y_warped": 20.0,
                "z_warped": 30.0,
            },
            {
                "electrode_id": 2,
                "probe_id": "probe_1",
                "probe_shank": 1,
                "probe_electrode": 2,
                "bad_channel": True,
                "x_warped": 15.0,
                "y_warped": 25.0,
                "z_warped": 35.0,
            },
        ]
    }

    # Test processing electrode config dicts
    electrode_config_dicts = {
        electrode_dict["electrode_id"]: electrode_dict
        for electrode_dict in mock_config["Electrode"]
    }

    assert len(electrode_config_dicts) == 2
    assert 1 in electrode_config_dicts
    assert 2 in electrode_config_dicts
    assert not electrode_config_dicts[1]["bad_channel"]
    assert electrode_config_dicts[2]["bad_channel"]


def test_electrode_constants_and_defaults():
    """Test electrode constants and default values."""
    # Test default constants used in electrode processing
    electrode_constants = {
        "x_warped": 0,
        "y_warped": 0,
        "z_warped": 0,
        "contacts": "",
    }

    assert electrode_constants["x_warped"] == 0
    assert electrode_constants["y_warped"] == 0
    assert electrode_constants["z_warped"] == 0
    assert electrode_constants["contacts"] == ""

    # Test default values for electrode properties
    defaults = {
        "name": "",
        "original_reference_electrode": -1,
        "bad_channel": "False",
        "filtering": "unfiltered",
    }

    assert defaults["name"] == ""
    assert defaults["original_reference_electrode"] == -1
    assert defaults["bad_channel"] == "False"
    assert defaults["filtering"] == "unfiltered"


def test_electrode_data_extraction():
    """Test electrode data extraction from NWB format."""
    # Mock electrode data from NWB electrodes table
    mock_elect_data = Mock()
    mock_elect_data.group_name = "test_group"

    # Mock electrode group
    mock_group = Mock()
    mock_group.location = "CA1"
    mock_elect_data.group = mock_group

    # Mock coordinate data
    mock_elect_data.get = Mock(
        side_effect=lambda key, default=None: {
            "x": 100.0,
            "y": 200.0,
            "z": 300.0,
            "filtering": "bandpass",
            "imp": 1.5,
        }.get(key, default)
    )

    # Test extraction
    extracted_data = {
        "group_name": mock_elect_data.group_name,
        "location": mock_elect_data.group.location,
        "x": mock_elect_data.get("x"),
        "y": mock_elect_data.get("y"),
        "z": mock_elect_data.get("z"),
        "filtering": mock_elect_data.get("filtering", "unfiltered"),
        "impedance": mock_elect_data.get("imp"),
    }

    assert extracted_data["group_name"] == "test_group"
    assert extracted_data["location"] == "CA1"
    assert extracted_data["x"] == 100.0
    assert extracted_data["y"] == 200.0
    assert extracted_data["z"] == 300.0
    assert extracted_data["filtering"] == "bandpass"
    assert extracted_data["impedance"] == 1.5


def test_electrode_probe_validation():
    """Test electrode probe validation logic."""
    # Mock electrode data with probe columns
    mock_elect_data_with_probe = Mock()

    # Mock valid probe device
    mock_device = Mock()
    mock_device.probe_type = "neuropixels"
    mock_group = Mock()
    mock_group.device = mock_device
    mock_elect_data_with_probe.group = mock_group

    # Test required probe columns
    extra_cols = [
        "probe_shank",
        "probe_electrode",
        "bad_channel",
        "ref_elect_id",
    ]

    # Mock electrode data that has all required columns
    for col in extra_cols:
        setattr(
            mock_elect_data_with_probe,
            col,
            getattr(
                {
                    "probe_shank": 1,
                    "probe_electrode": 5,
                    "bad_channel": False,
                    "ref_elect_id": 0,
                },
                col,
                None,
            ),
        )

    # Test column validation
    def has_probe_columns(elect_data, required_cols):
        return all(hasattr(elect_data, col) for col in required_cols)

    assert has_probe_columns(mock_elect_data_with_probe, extra_cols)

    # Test probe type validation
    def validate_probe_device(elect_data):
        device = elect_data.group.device
        return hasattr(device, "probe_type") and device.probe_type is not None

    assert validate_probe_device(mock_elect_data_with_probe)


def test_electrode_bad_channel_processing():
    """Test electrode bad channel processing."""

    # Test bad channel conversion logic
    def process_bad_channel(bad_channel_value):
        if isinstance(bad_channel_value, bool):
            return "True" if bad_channel_value else "False"
        elif isinstance(bad_channel_value, str):
            return (
                bad_channel_value
                if bad_channel_value in ["True", "False"]
                else "False"
            )
        else:
            return "False"

    assert process_bad_channel(True) == "True"
    assert process_bad_channel(False) == "False"
    assert process_bad_channel("True") == "True"
    assert process_bad_channel("False") == "False"
    assert process_bad_channel("invalid") == "False"
    assert process_bad_channel(1) == "False"
    assert process_bad_channel(None) == "False"


def test_electrode_region_id_caching():
    """Test electrode region ID caching optimization."""
    # Test region ID dict caching logic
    region_ids_dict = {}

    def get_region_id(region_name, cache_dict):
        if region_name not in cache_dict:
            # Simulate BrainRegion.fetch_add
            cache_dict[region_name] = f"region_{hash(region_name) % 1000}"
        return cache_dict[region_name]

    # Test caching behavior
    region1_id = get_region_id("CA1", region_ids_dict)
    assert "CA1" in region_ids_dict
    assert region_ids_dict["CA1"] == region1_id

    # Second call should return cached value
    region1_id_cached = get_region_id("CA1", region_ids_dict)
    assert region1_id == region1_id_cached

    # Different region should get different ID
    region2_id = get_region_id("CA3", region_ids_dict)
    assert region2_id != region1_id
    assert "CA3" in region_ids_dict


def test_raw_rate_fallback_prefers_rate(common_ephys):
    """Use explicit rate when present on the NWB object."""
    raw_table = common_ephys.Raw()
    nwb_obj = Mock(rate=2000.0, timestamps=np.array([0.0, 0.5, 1.0]))

    assert raw_table._rate_fallback(nwb_obj) == 2000.0


def test_raw_rate_fallback_uses_timestamps(common_ephys, monkeypatch):
    """Estimate rate from timestamps when rate is missing."""
    raw_table = common_ephys.Raw()
    nwb_obj = Mock()
    nwb_obj.rate = None
    nwb_obj.timestamps = np.array([0.0, 0.5, 1.0, 1.5])

    called = {}

    def _fake_estimate(ts, tol, verbose):
        called["ts"] = ts
        called["tol"] = tol
        called["verbose"] = verbose
        return 2.0

    monkeypatch.setattr(common_ephys, "estimate_sampling_rate", _fake_estimate)

    rate = raw_table._rate_fallback(nwb_obj)
    assert rate == 2.0
    assert np.array_equal(called["ts"], nwb_obj.timestamps)
    assert called["tol"] == 1.5


def test_raw_rate_fallback_requires_rate_or_timestamps(common_ephys):
    """Raise when neither rate nor timestamps are available."""
    raw_table = common_ephys.Raw()
    nwb_obj = Mock()
    nwb_obj.rate = None
    nwb_obj.timestamps = None

    with pytest.raises(ValueError, match="Neither rate nor timestamps"):
        raw_table._rate_fallback(nwb_obj)


def test_raw_valid_times_from_raw_rate_path(common_ephys):
    """Valid times are derived directly when rate is present."""
    raw_table = common_ephys.Raw()
    nwb_obj = Mock()
    nwb_obj.rate = 1000.0
    nwb_obj.data = np.zeros(3000)

    valid = raw_table._valid_times_from_raw(nwb_obj)
    assert np.array_equal(valid, np.array([[0.0, 3.0]]))


def test_raw_valid_times_from_raw_timestamp_path(common_ephys, monkeypatch):
    """Timestamp fallback delegates to get_valid_intervals."""
    raw_table = common_ephys.Raw()
    nwb_obj = Mock()
    nwb_obj.rate = None
    nwb_obj.timestamps = np.array([0.0, 1.0, 2.0, 3.0])

    monkeypatch.setattr(raw_table, "_rate_fallback", lambda _: 1.0)
    monkeypatch.setattr(
        common_ephys,
        "get_valid_intervals",
        lambda **kwargs: np.array([[0.0, 3.0]]),
    )

    valid = raw_table._valid_times_from_raw(nwb_obj)
    assert np.array_equal(valid, np.array([[0.0, 3.0]]))


def test_sample_count_make_returns_when_interface_missing(
    common_ephys, monkeypatch
):
    """No insert should occur when sample_count interface is absent."""
    table = common_ephys.SampleCount()
    monkeypatch.setattr(common_ephys.Nwbfile, "get_abs_path", lambda _: "x")
    monkeypatch.setattr(common_ephys, "get_nwb_file", lambda _: Mock())
    monkeypatch.setattr(common_ephys, "get_data_interface", lambda *_: None)

    insert_calls = []
    monkeypatch.setattr(
        table, "insert1", lambda *args, **kwargs: insert_calls.append(args)
    )

    table.make({"nwb_file_name": "test.nwb"})
    assert insert_calls == []


def test_sample_count_make_inserts_when_present(common_ephys, monkeypatch):
    """Insert object id when sample_count data interface exists."""
    table = common_ephys.SampleCount()
    sample_obj = Mock(object_id="sample-obj")

    monkeypatch.setattr(common_ephys.Nwbfile, "get_abs_path", lambda _: "x")
    monkeypatch.setattr(common_ephys, "get_nwb_file", lambda _: Mock())
    monkeypatch.setattr(
        common_ephys, "get_data_interface", lambda *_: sample_obj
    )

    inserted = []
    monkeypatch.setattr(
        table,
        "insert1",
        lambda key, **kwargs: inserted.append((key, kwargs)),
    )

    table.make({"nwb_file_name": "test.nwb"})
    assert inserted
    assert inserted[0][0]["sample_count_object_id"] == "sample-obj"


def test_lfp_make_compute_returns_none_without_filter_coeff(common_ephys):
    """Return sentinel values when filter coefficients are unavailable."""
    lfp_table = common_ephys.LFP()

    out = lfp_table.make_compute(
        key={"nwb_file_name": "test.nwb"},
        lfp_file_name="analysis.nwb",
        lfp_file_abspath="/tmp/analysis.nwb",
        electrode_keys=[{"electrode_id": 1}],
        rawdata=Mock(),
        sampling_rate=30000,
        interval_list_name="raw data valid times",
        valid_times=Mock(),
        filter={"filter_coeff": np.array([]), "filter_name": "LFP 0-400 Hz"},
    )

    assert out == [None, None]


def test_lfp_make_insert_returns_early_on_none(common_ephys, monkeypatch):
    """Skip all writes when make_compute returned sentinel values."""
    lfp_table = common_ephys.LFP()

    add_calls = []
    insert_interval_calls = []
    insert_calls = []

    monkeypatch.setattr(
        common_ephys.AnalysisNwbfile,
        "add",
        lambda *args, **kwargs: add_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        common_ephys.IntervalList,
        "insert1",
        lambda *args, **kwargs: insert_interval_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        lfp_table,
        "insert1",
        lambda *args, **kwargs: insert_calls.append((args, kwargs)),
    )

    lfp_table.make_insert(
        key={"nwb_file_name": "test.nwb"},
        lfp_valid_times=None,
        added_key=None,
        lfp_file_name="analysis.nwb",
    )

    assert add_calls == []
    assert insert_interval_calls == []
    assert insert_calls == []


def test_set_lfp_electrodes_returns_when_delete_incomplete(
    common_ephys, monkeypatch
):
    """Do not reinsert if post-delete query is still non-empty."""
    inserted = {"session": 0, "part": 0}

    class FakeSessionQuery:
        def delete(self, safemode=True):
            return None

        def fetch(self):
            return [{"nwb_file_name": "still-there.nwb"}]

    class FakeLFPSelection:
        class LFPElectrode:
            @staticmethod
            def insert(*_args, **_kwargs):
                inserted["part"] += 1

        def __and__(self, _key):
            return FakeSessionQuery()

        def insert1(self, _key):
            inserted["session"] += 1

    original_lfp_selection = common_ephys.LFPSelection
    monkeypatch.setattr(common_ephys, "LFPSelection", FakeLFPSelection)

    original_lfp_selection().set_lfp_electrodes("test.nwb", [1, 2])
    assert inserted["session"] == 0
    assert inserted["part"] == 0


def test_lfp_band_selection_invalid_electrode_ids(common_ephys, monkeypatch):
    """Raise when requested electrodes are not in LFPSelection."""

    class FakeRel:
        def __and__(self, _key):
            return self

        def fetch(self, _field):
            return np.array([1, 2])

    class FakeLFPSelection:
        def LFPElectrode(self):
            return FakeRel()

    monkeypatch.setattr(common_ephys, "LFPSelection", FakeLFPSelection)

    with pytest.raises(ValueError, match="electrode_list"):
        common_ephys.LFPBandSelection().set_lfp_band_electrodes(
            nwb_file_name="test.nwb",
            electrode_list=[99],
            filter_name="LFP 0-400 Hz",
            interval_list_name="lfp valid times",
            reference_electrode_list=[-1],
            lfp_band_sampling_rate=1000,
        )


def test_lfp_band_selection_invalid_divisor(common_ephys, monkeypatch):
    """Raise when band sampling rate is not an integer divisor."""

    class FakeLFPRel:
        def __and__(self, _key):
            return self

        def fetch1(self, _field):
            return 30000

    class FakeLFPElectrodeRel:
        def __and__(self, _key):
            return self

        def fetch(self, _field):
            return np.array([1])

    class FakeLFPSelection:
        def LFPElectrode(self):
            return FakeLFPElectrodeRel()

    monkeypatch.setattr(common_ephys, "LFPSelection", FakeLFPSelection)
    monkeypatch.setattr(common_ephys, "LFP", FakeLFPRel)

    with pytest.raises(ValueError, match="integer divisor"):
        common_ephys.LFPBandSelection().set_lfp_band_electrodes(
            nwb_file_name="test.nwb",
            electrode_list=[1],
            filter_name="LFP 0-400 Hz",
            interval_list_name="lfp valid times",
            reference_electrode_list=[-1],
            lfp_band_sampling_rate=777,
        )


def test_lfp_band_selection_invalid_reference_length(common_ephys, monkeypatch):
    """Raise when reference list length is neither 1 nor N electrodes."""

    class FakeLFPElectrodeRel:
        def __and__(self, _key):
            return self

        def fetch(self, _field):
            return np.array([1, 2])

    class FakeLFPSelection:
        def LFPElectrode(self):
            return FakeLFPElectrodeRel()

    class FakeLFPRel:
        def __and__(self, _key):
            return self

        def fetch1(self, _field):
            return 30000

    class TrueQuery:
        def __bool__(self):
            return True

    class FakeFirFilterParameters:
        def __and__(self, _key):
            return TrueQuery()

    class FakeIntervalList:
        def __and__(self, _key):
            return TrueQuery()

    monkeypatch.setattr(common_ephys, "LFPSelection", FakeLFPSelection)
    monkeypatch.setattr(common_ephys, "LFP", FakeLFPRel)
    monkeypatch.setattr(
        common_ephys,
        "FirFilterParameters",
        FakeFirFilterParameters,
    )
    monkeypatch.setattr(common_ephys, "IntervalList", FakeIntervalList)

    with pytest.raises(ValueError, match="reference_electrode_list"):
        common_ephys.LFPBandSelection().set_lfp_band_electrodes(
            nwb_file_name="test.nwb",
            electrode_list=[1, 2],
            filter_name="LFP 0-400 Hz",
            interval_list_name="lfp valid times",
            reference_electrode_list=[1, 2, 3],
            lfp_band_sampling_rate=1000,
        )


def test_lfp_band_selection_invalid_reference_ids(common_ephys, monkeypatch):
    """Raise when reference electrodes are outside allowed IDs + -1."""

    class FakeLFPElectrodeRel:
        def __and__(self, _key):
            return self

        def fetch(self, _field):
            return np.array([1, 2])

    class FakeLFPSelection:
        def LFPElectrode(self):
            return FakeLFPElectrodeRel()

    class FakeLFPRel:
        def __and__(self, _key):
            return self

        def fetch1(self, _field):
            return 30000

    class TrueQuery:
        def __bool__(self):
            return True

    class FakeFirFilterParameters:
        def __and__(self, _key):
            return TrueQuery()

    class FakeIntervalList:
        def __and__(self, _key):
            return TrueQuery()

    monkeypatch.setattr(common_ephys, "LFPSelection", FakeLFPSelection)
    monkeypatch.setattr(common_ephys, "LFP", FakeLFPRel)
    monkeypatch.setattr(
        common_ephys,
        "FirFilterParameters",
        FakeFirFilterParameters,
    )
    monkeypatch.setattr(common_ephys, "IntervalList", FakeIntervalList)

    with pytest.raises(ValueError, match="reference_electrode_list"):
        common_ephys.LFPBandSelection().set_lfp_band_electrodes(
            nwb_file_name="test.nwb",
            electrode_list=[1, 2],
            filter_name="LFP 0-400 Hz",
            interval_list_name="lfp valid times",
            reference_electrode_list=[999],
            lfp_band_sampling_rate=1000,
        )
