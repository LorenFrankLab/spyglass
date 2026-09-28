import datetime

import numpy as np
import pynwb
import pytest

from spyglass.utils.nwb_helper_fn import _get_epoch_groups, _get_pos_dict


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


def test_get_epoch_groups_with_timestamps():
    """_get_epoch_groups works when SpatialSeries has explicit timestamps."""
    spatial_series = pynwb.behavior.SpatialSeries(
        name="series_0",
        data=np.zeros((100, 2)),
        timestamps=np.linspace(0.0, 99.0 / 30.0, 100),
        reference_frame="unknown",
    )
    position = pynwb.behavior.Position(spatial_series=spatial_series)
    epoch_groups = _get_epoch_groups(position)
    assert len(epoch_groups) == 1
    assert list(epoch_groups.keys())[0] == pytest.approx(0.0)
    assert 0 in epoch_groups[list(epoch_groups.keys())[0]]


def test_get_epoch_groups_with_rate():
    """_get_epoch_groups works when SpatialSeries uses starting_time + rate."""
    spatial_series = pynwb.behavior.SpatialSeries(
        name="series_0",
        data=np.zeros((100, 2)),
        starting_time=5.0,
        rate=30.0,
        reference_frame="unknown",
    )
    position = pynwb.behavior.Position(spatial_series=spatial_series)
    assert spatial_series.timestamps is None
    epoch_groups = _get_epoch_groups(position)
    assert len(epoch_groups) == 1
    assert list(epoch_groups.keys())[0] == pytest.approx(5.0)
    assert 0 in epoch_groups[list(epoch_groups.keys())[0]]


def test_get_pos_dict_with_rate():
    """_get_pos_dict handles SpatialSeries that omit explicit timestamps."""
    spatial_series = pynwb.behavior.SpatialSeries(
        name="series_0",
        data=np.zeros((100, 2)),
        starting_time=0.0,
        rate=30.0,
        reference_frame="unknown",
    )
    position = pynwb.behavior.Position(spatial_series=spatial_series)
    epoch_groups = _get_epoch_groups(position)

    pos_dict = _get_pos_dict(position.spatial_series, epoch_groups)

    assert list(pos_dict.keys()) == [0]
    assert len(pos_dict[0]) == 1
    assert pos_dict[0][0]["raw_position_object_id"] == spatial_series.object_id
    assert pos_dict[0][0]["name"] == "series_0"
    np.testing.assert_allclose(
        pos_dict[0][0]["valid_times"],
        np.array([[-1e-7, 3.3 + 1e-7]]),
        atol=1e-9,
    )


def test_get_pos_dict_with_timestamps():
    """_get_pos_dict handles SpatialSeries that provide explicit timestamps."""
    spatial_series = pynwb.behavior.SpatialSeries(
        name="series_0",
        data=np.zeros((100, 2)),
        timestamps=np.linspace(0.0, 99.0 / 30.0, 100),
        reference_frame="unknown",
    )
    position = pynwb.behavior.Position(spatial_series=spatial_series)
    epoch_groups = _get_epoch_groups(position)

    pos_dict = _get_pos_dict(position.spatial_series, epoch_groups)

    assert list(pos_dict.keys()) == [0]
    assert len(pos_dict[0]) == 1
    assert pos_dict[0][0]["raw_position_object_id"] == spatial_series.object_id
    assert pos_dict[0][0]["name"] == "series_0"
    np.testing.assert_allclose(
        pos_dict[0][0]["valid_times"],
        np.array([[-1e-7, 3.3 + 1e-7]]),
        atol=1e-9,
    )
