import pytest


@pytest.fixture
def sensor_data(common, mini_insert):
    tbl = common.common_sensors.SensorData()
    tbl.populate()
    yield tbl


def test_sensor_data_insert(sensor_data, mini_insert, mini_restr, mini_content):
    obj_fetch = (sensor_data & mini_restr).fetch1("sensor_data_object_id")
    obj_raw = (
        mini_content.processing["analog"]
        .data_interfaces["analog"]
        .time_series["analog"]
        .object_id
    )
    assert (
        obj_fetch == obj_raw
    ), "SensorData object_id does not match raw object_id."
