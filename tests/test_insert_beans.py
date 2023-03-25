from datetime import datetime
import kachery_cloud as kcl
import os
import pathlib
import pynwb
import pytest


@pytest.mark.skip(reason="test_path needs to be updated")
def test_insert_sessions():
    print(
        "In test_insert_sessions, os.environ['SPYGLASS_BASE_DIR'] is",
        os.environ["SPYGLASS_BASE_DIR"],
    )
    raw_dir = pathlib.Path(os.environ["SPYGLASS_BASE_DIR"]) / "raw"
    nwbfile_path = raw_dir / "test.nwb"

    from spyglass.common import Session, DataAcquisitionDevice, CameraDevice, Probe
    from spyglass.data_import import insert_sessions

    test_path = "ipfs://bafybeie4svt3paz5vr7cw7mkgibutbtbzyab4s24hqn5pzim3sgg56m3n4"
    try:
        local_test_path = kcl.load_file(test_path)
    except Exception as e:
        if os.environ.get("KACHERY_CLOUD_EPHEMERAL", None) != "TRUE":
            print(
                "Cannot load test file in non-ephemeral mode. Kachery cloud client may need to be registered."
            )
        raise e

    # move the file to spyglass raw dir
    os.rename(local_test_path, nwbfile_path)

    # test that the file can be read. this is not used otherwise
    with pynwb.NWBHDF5IO(path=str(nwbfile_path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        assert nwbfile is not None

    insert_sessions(nwbfile_path.name)

    x = (Session() & {"nwb_file_name": "test_.nwb"}).fetch1()
    assert x["nwb_file_name"] == "test_.nwb"
    assert x["subject_id"] == "Beans"
    assert x["institution_name"] == "University of California, San Francisco"
    assert x["lab_name"] == "Loren Frank"
    assert x["session_id"] == "beans_01"
    assert x["session_description"] == "Reinforcement leaarning"
    assert x["session_start_time"] == datetime(2019, 7, 18, 15, 29, 47)
    assert x["timestamps_reference_time"] == datetime(1970, 1, 1, 0, 0)
    assert x["experiment_description"] == "Reinforcement learning"

    x = DataAcquisitionDevice().fetch()
    assert len(x) == 1
    assert x[0]["device_name"] == "dataacq_device0"
    assert x[0]["system"] == "SpikeGadgets"
    assert x[0]["amplifier"] == "Intan"
    assert x[0]["adc_circuit"] == "Intan"

    x = CameraDevice().fetch()
    assert len(x) == 2
    # NOTE order of insertion is not consistent so cannot use x[0]
    expected1 = dict(
        camera_name="beans sleep camera",
        # meters_per_pixel=0.00055,  # cannot check floating point values this way
        manufacturer="",
        model="unknown",
        lens="unknown",
        camera_id=0,
    )
    assert CameraDevice() & expected1
    assert (CameraDevice() & expected1).fetch("meters_per_pixel") == 0.00055
    expected2 = dict(
        camera_name="beans run camera",
        # meters_per_pixel=0.002,
        manufacturer="",
        model="unknown2",
        lens="unknown2",
        camera_id=1,
    )
    assert CameraDevice() & expected2
    assert (CameraDevice() & expected2).fetch("meters_per_pixel") == 0.002

    x = Probe().fetch()
    assert len(x) == 1
    assert x[0]["probe_type"] == "128c-4s8mm6cm-20um-40um-sl"
    assert x[0]["probe_description"] == "128 channel polyimide probe"
    assert x[0]["num_shanks"] == 4
    assert x[0]["contact_side_numbering"] == "True"
