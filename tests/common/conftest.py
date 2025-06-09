from pathlib import Path

import numpy as np
import pytest
from pynwb import NWBHDF5IO
from pynwb.behavior import BehavioralEvents
from pynwb.testing.mock.behavior import mock_TimeSeries
from pynwb.testing.mock.file import mock_NWBFile


@pytest.fixture(scope="session")
def interval_list(common):
    yield common.IntervalList()


@pytest.fixture(scope="session")
def mini_devices(mini_content):
    yield mini_content.devices


@pytest.fixture(scope="session")
def mini_behavior(mini_content):
    yield mini_content.processing.get("behavior")


@pytest.fixture(scope="session")
def mini_pos(mini_behavior):
    yield mini_behavior.get_data_interface("position").spatial_series


@pytest.fixture(scope="session")
def mini_pos_series(mini_pos):
    yield next(iter(mini_pos))


@pytest.fixture(scope="session")
def mini_beh_events(mini_behavior):
    yield mini_behavior.get_data_interface("behavioral_events")


@pytest.fixture(scope="session")
def mini_pos_interval_dict(mini_insert, common):
    yield {"interval_list_name": common.PositionSource.get_pos_interval_name(0)}


@pytest.fixture(scope="session")
def mini_pos_tbl(common, mini_pos_series):
    yield common.PositionSource.SpatialSeries * common.RawPosition.PosObject & {
        "name": mini_pos_series
    }


@pytest.fixture(scope="session")
def pos_src(common):
    yield common.PositionSource()


@pytest.fixture(scope="session")
def pos_interval_01(pos_src):
    yield [pos_src.get_pos_interval_name(x) for x in range(1)]


@pytest.fixture(scope="session")
def common_ephys(common):
    yield common.common_ephys


@pytest.fixture(scope="session")
def pop_common_electrode_group(common_ephys):
    common_ephys.ElectrodeGroup.populate()
    yield common_ephys.ElectrodeGroup()


@pytest.fixture(scope="session")
def dio_only_nwb(raw_dir):
    nwbfile = mock_NWBFile(
        identifier="my_identifier", session_description="my_session_description"
    )
    time_series = mock_TimeSeries(
        name="my_time_series", timestamps=np.arange(20), data=np.ones((20, 1))
    )
    behavioral_events = BehavioralEvents(
        name="behavioral_events", time_series=time_series
    )
    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="behavior module"
    )
    behavior_module.add(behavioral_events)

    from spyglass.settings import raw_dir

    file_name = "mock_behavior.nwb"
    nwbfile_path = Path(raw_dir) / file_name
    with NWBHDF5IO(nwbfile_path, "w") as io:
        io.write(nwbfile)

    yield file_name
