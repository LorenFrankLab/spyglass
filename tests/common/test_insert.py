from datajoint.hash import key_hash
from pandas import DataFrame, Index
from pytest import approx


def test_insert_session(mini_insert, mini_content, mini_restr, common):
    subj_raw = mini_content.subject
    meta_raw = mini_content

    session_data = (common.Session & mini_restr).fetch1()
    assert (
        session_data["subject_id"] == subj_raw.subject_id
    ), "Subject ID not match"

    attrs = [
        ("institution_name", "institution"),
        ("lab_name", "lab"),
        ("session_id", "session_id"),
        ("session_description", "session_description"),
        ("experiment_description", "experiment_description"),
    ]

    for session_attr, meta_attr in attrs:
        assert session_data[session_attr] == getattr(
            meta_raw, meta_attr
        ), f"Session table {session_attr} not match raw data {meta_attr}"

    time_attrs = [
        ("session_start_time", "session_start_time"),
        ("timestamps_reference_time", "timestamps_reference_time"),
    ]
    for session_attr, meta_attr in time_attrs:
        # a. strip timezone info from meta_raw
        # b. convert to timestamp
        # c. compare precision to 1 second
        assert session_data[session_attr].timestamp() == approx(
            getattr(meta_raw, meta_attr).replace(tzinfo=None).timestamp(), abs=1
        ), f"Session table {session_attr} not match raw data {meta_attr}"


def test_insert_electrode_group(mini_insert, mini_content, common):
    group_name = "0"
    elec_group_data = (
        common.ElectrodeGroup & {"electrode_group_name": group_name}
    ).fetch1()
    elec_group_raw = mini_content.electrode_groups.get(group_name)

    assert (
        elec_group_data["description"] == elec_group_raw.description
    ), "ElectrodeGroup description not match"

    assert elec_group_data["region_id"] == (
        common.BrainRegion & {"region_name": elec_group_raw.location}
    ).fetch1(
        "region_id"
    ), "Region ID does not match across raw data and BrainRegion table"


def test_insert_electrode(mini_insert, mini_content, mini_restr, common):
    electrode_id = "0"
    e_data = (common.Electrode & {"electrode_id": electrode_id}).fetch1()
    e_raw = mini_content.electrodes.get(int(electrode_id)).to_dict().copy()

    attrs = [
        ("x", "x"),
        ("y", "y"),
        ("z", "z"),
        ("impedance", "imp"),
        ("filtering", "filtering"),
        ("original_reference_electrode", "ref_elect_id"),
    ]

    for e_attr, meta_attr in attrs:
        assert (
            e_data[e_attr] == e_raw[meta_attr][int(electrode_id)]
        ), f"Electrode table {e_attr} not match raw data {meta_attr}"


def test_insert_raw(mini_insert, mini_content, mini_restr, common):
    raw_data = (common.Raw & mini_restr).fetch1()
    raw_raw = mini_content.get_acquisition()

    attrs = [
        ("comments", "comments"),
        ("description", "description"),
    ]
    for raw_attr, meta_attr in attrs:
        assert raw_data[raw_attr] == getattr(
            raw_raw, meta_attr
        ), f"Raw table {raw_attr} not match raw data {meta_attr}"


def test_insert_sample_count(mini_insert, mini_content, mini_restr, common):
    sample_data = (common.SampleCount & mini_restr).fetch1()
    sample_full = mini_content.processing.get("sample_count")
    if not sample_full:
        assert False, "No sample count data in raw data"
    sample_raw = sample_full.data_interfaces.get("sample_count")
    assert (
        sample_data["sample_count_object_id"] == sample_raw.object_id
    ), "SampleCount insertion error"


def test_insert_dio(mini_insert, mini_behavior, mini_restr, common):
    events_data = (common.DIOEvents & mini_restr).fetch(as_dict=True)
    events_raw = mini_behavior.get_data_interface(
        "behavioral_events"
    ).time_series

    assert len(events_data) == len(events_raw), "Number of events not match"

    event = [p for p in events_raw.keys() if "Poke" in p][0]
    event_raw = events_raw.get(event)
    # event_data = (common.DIOEvents & {"dio_event_name": event}).fetch(as_dict=True)[0]
    event_data = (common.DIOEvents & {"dio_event_name": event}).fetch1()

    assert (
        event_data["dio_object_id"] == event_raw.object_id
    ), "DIO Event insertion error"


def test_insert_pos(
    mini_insert,
    common,
    mini_behavior,
    mini_restr,
    mini_pos_series,
    mini_pos_tbl,
):
    pos_data = (common.PositionSource.SpatialSeries & mini_restr).fetch()
    pos_raw = mini_behavior.get_data_interface("position").spatial_series

    assert len(pos_data) == len(pos_raw), "Number of spatial series not match"

    raw_obj_id = pos_raw[mini_pos_series].object_id
    data_obj_id = mini_pos_tbl.fetch1("raw_position_object_id")

    assert data_obj_id == raw_obj_id, "PosObject insertion error"


def test_fetch_pos_obj(
    mini_insert, common, mini_pos, mini_pos_series, mini_pos_tbl
):
    pos_key = (
        common.PositionSource.SpatialSeries & mini_pos_tbl.fetch("KEY")
    ).fetch(as_dict=True)[0]
    pos_df = (common.RawPosition & pos_key).fetch1_dataframe().iloc[:, 0:2]

    series = mini_pos[mini_pos_series]
    raw_df = DataFrame(
        data=series.data,
        index=Index(series.timestamps, name="time"),
        columns=[col + "1" for col in series.description.split(", ")],
    )
    assert key_hash(pos_df) == key_hash(raw_df), "Spatial series fetch error"


def test_insert_device(mini_insert, mini_devices, common):
    this_device = "dataacq_device0"
    device_raw = mini_devices.get(this_device)
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


def test_insert_camera(mini_insert, mini_devices, common):
    camera_raw = mini_devices.get("camera_device 0")
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


def test_insert_probe(mini_insert, mini_devices, common):
    this_probe = "probe 0"
    probe_raw = mini_devices.get(this_probe)
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


def test_dio_only_insert(dio_only_nwb, common, data_import):
    """Test that DIOEvents can be inserted from a NWB file with only DIO data."""
    data_import.insert_sessions(dio_only_nwb)

    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    key = {"nwb_file_name": get_nwb_copy_filename(dio_only_nwb)}
    interval = (common.IntervalList() & key).fetch1()
    assert interval["interval_list_name"] == "dio data valid times"
    assert (
        interval["valid_times"][0][0] == 0.0
        and interval["valid_times"][0][1] == 19.0
    ), "Interval does not match dio"
