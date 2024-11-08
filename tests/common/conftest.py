import pytest


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
