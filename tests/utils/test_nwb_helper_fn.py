import datetime

import pynwb
import pytest


@pytest.fixture(scope="module")
def get_electrode_indices(common):
    from spyglass.common import get_electrode_indices  # noqa: E402

    return get_electrode_indices


@pytest.fixture(scope="module")
def custom_nwbfile(common):
    nwbfile = pynwb.NWBFile(
        session_description="session_description",
        identifier="identifier",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )
    dev = nwbfile.create_device(name="device")
    elec_group = nwbfile.create_electrode_group(
        name="electrodes",
        description="description",
        location="location",
        device=dev,
    )
    for i in range(10):
        nwbfile.add_electrode(
            id=100 + i,
            x=0.0,
            y=0.0,
            z=0.0,
            imp=-1.0,
            location="location",
            filtering="filtering",
            group=elec_group,
        )
    electrode_region = nwbfile.electrodes.create_region(
        name="electrodes",
        region=[2, 3, 4, 5],
        description="description",  # indices
    )
    nwbfile.add_acquisition(
        pynwb.ecephys.ElectricalSeries(
            name="eseries",
            data=[0, 1, 2],
            timestamps=[0.0, 1.0, 2.0],
            electrodes=electrode_region,
        )
    )
    yield nwbfile


def test_electrode_nwbfile(get_electrode_indices, custom_nwbfile):
    ret = get_electrode_indices(custom_nwbfile, [102, 105])
    assert ret == [2, 5]


def test_electrical_series(get_electrode_indices, custom_nwbfile):
    eseries = custom_nwbfile.acquisition["eseries"]
    ret = get_electrode_indices(eseries, [102, 105])
    assert ret == [0, 3]
