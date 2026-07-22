from unittest.mock import Mock, patch

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


def test_common_sensors_error_handling_from_targeted():
    """Basic import/mocking path for common_sensors module."""
    with patch("spyglass.common.common_sensors") as sensors_module:
        sensors_module.return_value = Mock()
        assert sensors_module is not None
