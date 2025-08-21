from ndx_optogenetics import OptogeneticVirusInjection


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
        OptogeneticVirusInjection,
    ), "VirusInjection did not fetch OptogeneticVirusInjection object as expected."
    assert (common.Virus & (injection_query).proj("virus_name")).fetch1(
        "construct_name"
    ) == virus_dict[
        "construct_name"
    ], "VirusInjection did not fetch the expected virus construct name."


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
        "fiber_model"
    ], "OpticalFiberDevice did not fetch the expected fiber model."


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
