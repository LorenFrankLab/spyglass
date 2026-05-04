import pytest
from ndx_ophys_devices import ViralVectorInjection


@pytest.mark.slow
def test_virus_injection(
    opto_only_nwb,
    common,
    virus_dict,
):
    # data_import.insert_sessions(opto_only_nwb, raise_err=True)
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    key = {"nwb_file_name": get_nwb_copy_filename(opto_only_nwb)}

    # Test Virus injection
    injection_query = common.VirusInjection() & key
    assert (
        len(injection_query) == 1
    ), "Expected exactly one VirusInjection for the test file."
    assert isinstance(
        (injection_query).fetch_nwb()[0]["injection"],
        ViralVectorInjection,
    ), "VirusInjection did not fetch ViralVectorInjection object as expected."
    assert (common.Virus & (injection_query).proj("virus_name")).fetch1(
        "construct_name"
    ) == virus_dict[
        "construct_name"
    ], "VirusInjection did not fetch the expected virus construct name."


@pytest.mark.slow
def test_optical_fiber(
    opto_only_nwb,
    common,
    fiber_model_dict,
):
    # data_import.insert_sessions(opto_only_nwb, raise_err=True)
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    key = {"nwb_file_name": get_nwb_copy_filename(opto_only_nwb)}
    # Fiber implant checks
    # common.OpticalFiberImplant().make(key)
    implant_query = common.OpticalFiberImplant & key
    assert (
        len(implant_query) == 1
    ), "Expected exactly one OpticalFiberImplant for the given key."
    assert (common.OpticalFiberDevice() & implant_query).fetch1(
        "model"
    ) == fiber_model_dict[
        "model_number"
    ], "OpticalFiberDevice did not fetch the expected fiber model."


@pytest.mark.slow
def test_optogenetic_protocol(
    opto_only_nwb,
    common,
    opto_epoch_dict,
    data_import,
):
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    key = {"nwb_file_name": get_nwb_copy_filename(opto_only_nwb)}
    # Protocol checks
    protocol_query = common.OptogeneticProtocol() & key
    assert (
        len(protocol_query) == 1
    ), "Expected exactly one OptogeneticProtocol for the given key."
    assert (common.TaskEpoch & protocol_query).fetch1(
        "interval_list_name"
    ) == opto_epoch_dict[
        "epoch_name"
    ], "TaskEpoch did not fetch the expected epoch name."

    assert (
        protocol_query.fetch_nwb()[0]["stimulus"].name == "stimulus_channel"
    ), "OptogeneticProtocol did not fetch the expected stimulus channel name."
    assert (
        protocol_query.fetch1("pulse_length")
        == opto_epoch_dict["pulse_length_in_ms"]
    ), "OptogeneticProtocol did not import the expected pulse length."

    assert (
        len(protocol_query.SpatialConditional() & key) == 1
    ), "SpatialConditional did not fetch the expected number of entries."
    assert (protocol_query.SpatialConditional() & key).fetch1(
        "nodes"
    ).shape == opto_epoch_dict[
        "spatial_filter_region_node_coordinates_in_pixels"
    ].shape, "SpatialConditional did not properly import nodes."
    assert (
        len(protocol_query.SpeedConditional() & key) == 1
    ), "SpeedConditional did not fetch the expected number of entries."
    assert (protocol_query.SpeedConditional() & key).fetch1(
        "speed_threshold"
    ) == opto_epoch_dict[
        "speed_filter_threshold_in_cm_per_s"
    ], "SpeedConditional did not import the expected speed threshold."
    assert (
        len(protocol_query.RippleTrigger() & key) == 1
    ), "RippleTrigger did not fetch the expected number of entries."
    assert (protocol_query.RippleTrigger() & key).fetch1(
        "threshold_sd"
    ) == opto_epoch_dict[
        "ripple_filter_threshold_sd"
    ], "RippleTrigger did not import the expected threshold SD."
    assert (
        len(protocol_query.ThetaTrigger() & key) == 1
    ), "ThetaTrigger did not fetch the expected number of entries."
    assert (protocol_query.ThetaTrigger() & key).fetch1(
        "filter_phase"
    ) == opto_epoch_dict[
        "theta_filter_phase_in_deg"
    ], "ThetaTrigger did not import the expected filter phase."


def _make_opto_row(**kwargs):
    """Return a Mock with the NWB row attributes used by make_*_entry methods."""
    from unittest.mock import Mock

    defaults = dict(
        epoch_number=1,
        convenience_code="test_stim",
        pulse_length_in_ms=10.0,
        number_pulses_per_pulse_train=5,
        period_in_ms=100.0,
        intertrain_interval_in_ms=500.0,
        power_in_mW=2.5,
        ripple_filter_threshold_sd=3.0,
        ripple_filter_num_above_threshold=2,
        ripple_filter_lockout_period_in_samples=30,
        theta_filter_phase_in_deg=180.0,
        theta_filter_reference_ntrode=1,
        theta_filter_lockout_period_in_samples=20,
        speed_filter_threshold_in_cm_per_s=5.0,
        speed_filter_on_above_threshold=True,
    )
    defaults.update(kwargs)
    row = Mock(**defaults)
    row.stimulus_signal.object_id = "abc123"
    return row


def test_make_epoch_entry():
    """make_epoch_entry maps NWB row attributes to table column names."""
    from spyglass.common.common_optogenetics import OptogeneticProtocol

    row = _make_opto_row()
    result = OptogeneticProtocol.make_epoch_entry("test.nwb", row)

    assert result["nwb_file_name"] == "test.nwb"
    assert result["epoch"] == 1
    assert result["description"] == "test_stim"
    assert result["pulse_length"] == 10.0
    assert result["pulses_per_train"] == 5
    assert result["period"] == 100.0
    assert result["intertrain_interval"] == 500.0
    assert result["stimulus_power"] == 2.5
    assert result["stimulus_object_id"] == "abc123"


def test_make_ripple_trigger_entry():
    """make_ripple_trigger_entry maps ripple filter attributes correctly."""
    from spyglass.common.common_optogenetics import OptogeneticProtocol

    row = _make_opto_row()
    result = OptogeneticProtocol.make_ripple_trigger_entry("test.nwb", row)

    assert result["nwb_file_name"] == "test.nwb"
    assert result["epoch"] == 1
    assert result["threshold_sd"] == 3.0
    assert result["n_above_threshold"] == 2
    assert result["ripple_lockout_period"] == 30


def test_make_theta_trigger_entry():
    """make_theta_trigger_entry maps theta filter attributes correctly."""
    from spyglass.common.common_optogenetics import OptogeneticProtocol

    row = _make_opto_row()
    result = OptogeneticProtocol.make_theta_trigger_entry("test.nwb", row)

    assert result["nwb_file_name"] == "test.nwb"
    assert result["epoch"] == 1
    assert result["filter_phase"] == 180.0
    assert result["reference_ntrode"] == 1
    assert result["theta_lockout_period"] == 20


def test_make_speed_filter_entry():
    """make_speed_filter_entry maps speed filter attributes correctly."""
    from spyglass.common.common_optogenetics import OptogeneticProtocol

    row = _make_opto_row()
    result = OptogeneticProtocol.make_speed_filter_entry("test.nwb", row)

    assert result["nwb_file_name"] == "test.nwb"
    assert result["epoch"] == 1
    assert result["speed_threshold"] == 5.0
    assert result["active_above_threshold"] is True


def test_get_stimulus_on_intervals_nominal():
    """get_stimulus_on_intervals pairs on/off transitions correctly."""
    import numpy as np
    from unittest.mock import Mock, patch

    from spyglass.common.common_optogenetics import OptogeneticProtocol

    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    data = np.array([1, 0, 1, 0, 1, 0])  # three clean on/off pairs

    mock_stim = Mock()
    mock_stim.get_timestamps.return_value = times
    mock_stim.data = data

    mock_interval = Mock()
    mock_interval.contains.return_value = np.ones(len(times), dtype=bool)

    protocol = OptogeneticProtocol()
    with (
        patch.object(protocol, "ensure_single_entry"),
        patch.object(
            type(protocol),
            "fetch_nwb",
            return_value=[{"stimulus": mock_stim}],
        ),
        patch("spyglass.common.common_optogenetics.IntervalList") as mock_il,
        patch("spyglass.common.common_optogenetics.TaskEpoch"),
    ):
        mock_il.__and__ = Mock(return_value=Mock())
        mock_il.__and__.return_value.fetch_interval = Mock(
            return_value=mock_interval
        )
        result = protocol.get_stimulus_on_intervals({})

    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result[:, 0], [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(result[:, 1], [1.0, 3.0, 5.0])


def test_get_stimulus_on_intervals_leading_off():
    """Leading off-sample before first on is discarded."""
    import numpy as np
    from unittest.mock import Mock, patch

    from spyglass.common.common_optogenetics import OptogeneticProtocol

    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    data = np.array([0, 1, 0, 1, 0])  # off comes before first on

    mock_stim = Mock()
    mock_stim.get_timestamps.return_value = times
    mock_stim.data = data

    mock_interval = Mock()
    mock_interval.contains.return_value = np.ones(len(times), dtype=bool)

    protocol = OptogeneticProtocol()
    with (
        patch.object(protocol, "ensure_single_entry"),
        patch.object(
            type(protocol),
            "fetch_nwb",
            return_value=[{"stimulus": mock_stim}],
        ),
        patch("spyglass.common.common_optogenetics.IntervalList") as mock_il,
        patch("spyglass.common.common_optogenetics.TaskEpoch"),
    ):
        mock_il.__and__ = Mock(return_value=Mock())
        mock_il.__and__.return_value.fetch_interval = Mock(
            return_value=mock_interval
        )
        result = protocol.get_stimulus_on_intervals({})

    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result[:, 0], [1.0, 3.0])
    np.testing.assert_array_equal(result[:, 1], [2.0, 4.0])
