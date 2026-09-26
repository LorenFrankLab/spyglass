import numpy as np
import pytest


@pytest.fixture(scope="module")
def mua(common):
    from spyglass.mua import v1

    return v1


@pytest.fixture(scope="module")
def mua_keys(mua, pop_spikes_group, pos_merge_key, pos_interval):
    params = mua.MuaEventsParameters()
    params.insert_default()
    default = (params & {"mua_param_name": "default"}).fetch1("mua_param_dict")
    params.insert1(
        {
            "mua_param_name": "legacy_speed_zscore",
            "mua_param_dict": {
                **default,
                "use_speed_threshold_for_zscore": True,
            },
        },
        skip_duplicates=True,
    )
    keys = [
        {
            **pop_spikes_group,
            "mua_param_name": name,
            "pos_merge_id": pos_merge_key["merge_id"],
            "detection_interval": pos_interval,
        }
        for name in ("default", "legacy_speed_zscore")
    ]
    mua.MuaEventsV1().populate(keys)
    yield keys


def test_mua_times_columns(mua, mua_keys):
    mua_times = (mua.MuaEventsV1 & mua_keys[0]).fetch1_dataframe()
    assert len(mua_times) > 0, "No MUA events detected"
    for column in (
        "n_samples",
        "max_sustained_zscore",
        "peak_time",
        "clipped_start",
        "clipped_end",
        "n_active_units",
    ):
        assert column in mua_times.columns
    assert "max_thresh" not in mua_times.columns
    # Per-unit counts reach the detector, so it counts units, not one column
    assert mua_times.n_active_units.max() > 1


def test_legacy_speed_zscore_row_populates(mua, mua_keys, pos_interval):
    """A stored use_speed_threshold_for_zscore=True is applied as 1.x did."""
    from ripple_detection import multiunit_HSE_detector

    from spyglass.common import IntervalList
    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

    legacy_key = mua_keys[1]
    stored = (mua.MuaEventsV1 & legacy_key).fetch1_dataframe()

    speed = mua.MuaEventsV1.get_speed(legacy_key)
    time = speed.index.to_numpy()
    speed = speed.to_numpy()
    spike_indicator = SortedSpikesGroup.get_spike_indicator(legacy_key, time)
    valid_times = (
        IntervalList
        & {
            "nwb_file_name": legacy_key["nwb_file_name"],
            "interval_list_name": pos_interval,
        }
    ).fetch1("valid_times")
    mask = np.zeros_like(time, dtype=bool)
    for start, end in valid_times:
        mask |= (time >= start) & (time <= end)
    time, speed, spike_indicator = (
        time[mask],
        speed[mask],
        spike_indicator[mask],
    )

    expected = multiunit_HSE_detector(
        time,
        spike_indicator,
        speed,
        1 / np.median(np.diff(time)),
        minimum_duration=0.015,
        zscore_threshold=2.0,
        close_event_threshold=0.0,
        speed_threshold=4.0,
        normalization_mask=speed < 4.0,
    )
    assert len(stored) == len(expected)
    np.testing.assert_allclose(
        stored[["start_time", "end_time"]].to_numpy(),
        expected[["start_time", "end_time"]].to_numpy(),
    )
