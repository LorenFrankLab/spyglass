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
def mini_pos_tbl(common, mini_pos_series):
    yield common.PositionSource.SpatialSeries * common.RawPosition.PosObject & {
        "name": mini_pos_series
    }
