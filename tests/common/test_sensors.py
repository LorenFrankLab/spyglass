import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sensor_data(common, mini_insert):
    """SensorData as ingested from the mini file.

    Previously called `populate()`. SensorData now ingests via
    `insert_from_nwbfile`, which `mini_insert` already drives, and its `make`
    is a deprecation shim -- calling populate() would try to run it for any
    other session in the database.
    """
    yield common.common_sensors.SensorData()


@pytest.fixture
def analog_ts(mini_content):
    """The raw analog BehavioralEvents time series backing SensorData."""
    yield (
        mini_content.processing["analog"]
        .data_interfaces["analog"]
        .time_series["analog"]
    )


def _expected_columns(analog_ts):
    return [
        c
        for c in analog_ts.description.split()
        if c not in ("time", "timestamps")
    ]


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


def test_sensor_data_fetch1_dataframe(sensor_data, mini_restr, analog_ts):
    """fetch1_dataframe reconstructs the raw analog data exactly."""
    df = (sensor_data & mini_restr).fetch1_dataframe()

    assert list(df.columns) == _expected_columns(analog_ts)
    assert df.index.name == "time"
    assert np.array_equal(df.to_numpy(), analog_ts.data[:])
    assert np.array_equal(df.index.to_numpy(), analog_ts.timestamps[:])


def test_sensor_data_fetch1_dataframe_empty(common):
    """An empty query returns None rather than raising."""
    empty = common.common_sensors.SensorData() & "nwb_file_name = 'none_.nwb'"
    assert empty.fetch1_dataframe() is None


def test_sensor_data_fetch1_dataframe_interval(
    sensor_data, mini_restr, analog_ts
):
    """Restricting by interval yields a sorted subset with the same columns."""
    entry = sensor_data & mini_restr
    interval_name = entry.fetch1("interval_list_name")

    df = entry.fetch1_dataframe(interval_list_name=interval_name)

    assert isinstance(df, pd.DataFrame) and not df.empty
    assert list(df.columns) == _expected_columns(analog_ts)
    assert df.index.name == "time"
    assert df.index.is_monotonic_increasing
    assert len(df) <= analog_ts.data.shape[0]


def test_sensor_data_fetch1_dataframe_empty_interval(
    common, sensor_data, mini_restr
):
    """An interval registered with no valid times raises ValueError."""
    entry = sensor_data & mini_restr
    interval_key = {
        "nwb_file_name": entry.fetch1("nwb_file_name"),
        "interval_list_name": "test_empty_sensor_interval",
        "valid_times": np.empty((0, 2)),
    }
    common.common_interval.IntervalList.insert1(
        interval_key, skip_duplicates=True
    )
    pk = {
        "nwb_file_name": interval_key["nwb_file_name"],
        "interval_list_name": interval_key["interval_list_name"],
    }
    try:
        with pytest.raises(ValueError, match="No valid times"):
            entry.fetch1_dataframe(
                interval_list_name="test_empty_sensor_interval"
            )
    finally:
        (common.common_interval.IntervalList & pk).delete(safemode=False)
