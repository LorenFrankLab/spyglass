from pytest import approx


def test_load_file(minirec_content):
    assert minirec_content is not None


def test_insert_session(minirec_insert, minirec_content, minirec_restr, common):
    subj_raw = minirec_content.subject
    meta_raw = minirec_content

    sess_data = (common.Session & minirec_restr).fetch1()
    assert (
        sess_data["subject_id"] == subj_raw.subject_id
    ), "Subjuect ID not match"

    attrs = [
        ("institution_name", "institution"),
        ("lab_name", "lab"),
        ("session_id", "session_id"),
        ("session_description", "session_description"),
        ("experiment_description", "experiment_description"),
    ]

    for sess_attr, meta_attr in attrs:
        assert sess_data[sess_attr] == getattr(
            meta_raw, meta_attr
        ), f"Session table {sess_attr} not match raw data {meta_attr}"

    time_attrs = [
        ("session_start_time", "session_start_time"),
        ("timestamps_reference_time", "timestamps_reference_time"),
    ]
    for sess_attr, meta_attr in time_attrs:
        # a. strip timezone info from meta_raw
        # b. convert to timestamp
        # c. compare precision to 1 second
        assert sess_data[sess_attr].timestamp() == approx(
            getattr(meta_raw, meta_attr).replace(tzinfo=None).timestamp(), abs=1
        ), f"Session table {sess_attr} not match raw data {meta_attr}"


def test_insert_electrode_group(minirec_insert, minirec_content, common):
    group_name = "0"
    egroup_data = (
        common.ElectrodeGroup & {"electrode_group_name": group_name}
    ).fetch1()
    egroup_raw = minirec_content.electrode_groups.get(group_name)

    assert (
        egroup_data["description"] == egroup_raw.description
    ), "ElectrodeGroup description not match"

    assert egroup_data["region_id"] == (
        common.BrainRegion & {"region_name": egroup_raw.location}
    ).fetch1(
        "region_id"
    ), "Region ID does not match across raw data and BrainRegion table"


def test_insert_electrode(
    minirec_insert, minirec_content, minirec_restr, common
):
    electrode_id = "0"
    e_data = (common.Electrode & {"electrode_id": electrode_id}).fetch1()
    e_raw = minirec_content.electrodes.get(int(electrode_id)).to_dict().copy()

    attrs = [
        ("x", "x"),
        ("y", "y"),
        ("z", "z"),
        ("impedance", "imp"),
        ("filtering", "filtering"),
        ("original_reference_electrode", "ref_elect_id"),
    ]

    for e_attr, meta_attr in attrs:
        assert (  # KeyError: 0 here  ↓
            e_data[e_attr] == e_raw[int(electrode_id)][meta_attr]
        ), f"Electrode table {e_attr} not match raw data {meta_attr}"


def test_insert_raw(minirec_insert, minirec_content, minirec_restr, common):
    raw_data = (common.Raw & minirec_restr).fetch1()
    raw_raw = minirec_content.get_acquisition()

    attrs = [
        ("comments", "comments"),
        ("description", "description"),
    ]
    for raw_attr, meta_attr in attrs:
        assert raw_data[raw_attr] == getattr(
            raw_raw, meta_attr
        ), f"Raw table {raw_attr} not match raw data {meta_attr}"


def test_insert_sample_count(minirec_insert, minirec_content, common):
    # commont.SampleCount
    assert False, "TODO"


def test_insert_dio(minirec_insert, minirec_content, common):
    # commont.DIOEvents
    assert False, "TODO"


def test_insert_pos(minirec_insert, minirec_content, common):
    # commont.PositionSource * common.RawPosition
    assert False, "TODO"


def test_insert_device(minirec_insert, minirec_devices, common):
    this_device = "dataacq_device0"
    device_raw = minirec_devices.get(this_device)
    device_data = (
        common.DataAcquisitionDevice
        & {"data_acquisition_device_name": this_device}
    ).fetch1()

    attrs = [
        ("data_acquisition_device_name", "name"),
        ("data_acquisition_device_system", "system"),
        ("data_acquisition_device_amplifier", "amplifier"),
        ("adc_circuit", "adc_circuit"),
    ]

    for device_attr, meta_attr in attrs:
        assert device_data[device_attr] == getattr(
            device_raw, meta_attr
        ), f"Device table {device_attr} not match raw data {meta_attr}"


def test_insert_camera(minirec_insert, minirec_devices, common):
    camera_raw = minirec_devices.get("camera_device 0")
    camera_data = (
        common.CameraDevice & {"camera_name": camera_raw.camera_name}
    ).fetch1()

    attrs = [
        ("camera_name", "camera_name"),
        ("manufacturer", "manufacturer"),
        ("model", "model"),
        ("lens", "lens"),
        ("meters_per_pixel", "meters_per_pixel"),
    ]
    for camera_attr, meta_attr in attrs:
        assert camera_data[camera_attr] == getattr(
            camera_raw, meta_attr
        ), f"Camera table {camera_attr} not match raw data {meta_attr}"


def test_insert_probe(minirec_insert, minirec_devices, common):
    this_probe = "probe 0"
    probe_raw = minirec_devices.get(this_probe)
    probe_id = probe_raw.probe_type

    probe_data = (
        common.Probe * common.ProbeType & {"probe_id": probe_id}
    ).fetch1()

    attrs = [
        ("probe_type", "probe_type"),
        ("probe_description", "probe_description"),
        ("contact_side_numbering", "contact_side_numbering"),
    ]

    for probe_attr, meta_attr in attrs:
        assert probe_data[probe_attr] == str(
            getattr(probe_raw, meta_attr)
        ), f"Probe table {probe_attr} not match raw data {meta_attr}"

    assert probe_data["num_shanks"] == len(
        probe_raw.shanks
    ), "Number of shanks in ProbeType number not raw data"
