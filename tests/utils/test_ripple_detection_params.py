import numpy as np
import pytest

from spyglass.utils.ripple_detection_params import (
    detection_kwargs_from_params,
)

FS = 1000.0


@pytest.fixture(scope="module")
def signals():
    rng = np.random.default_rng(0)
    time = np.arange(20_000) / FS
    speed = np.where((time > 5) & (time < 10), 20.0, 0.0)
    speed[100] = 4.0  # exactly at the threshold: 1.x's flag left it out
    multiunit = rng.poisson(0.02, size=(time.size, 8))
    return time, speed, multiunit


def test_current_keys_pass_through(signals):
    time, speed, _ = signals
    params = {"speed_threshold": 4.0, "minimum_duration": 0.015}
    assert detection_kwargs_from_params(params, time, speed) == params


def test_time_range_becomes_inclusive_mask(signals):
    time, speed, _ = signals
    params = {"normalization_time_range": (1.0, 2.0)}
    kwargs = detection_kwargs_from_params(params, time, speed)
    assert "normalization_time_range" not in kwargs
    np.testing.assert_array_equal(
        kwargs["normalization_mask"], (time >= 1.0) & (time <= 2.0)
    )
    assert params == {"normalization_time_range": (1.0, 2.0)}  # unmodified


def test_speed_flag_becomes_strict_speed_mask(signals):
    time, speed, _ = signals
    kwargs = detection_kwargs_from_params(
        {"use_speed_threshold_for_zscore": True, "speed_threshold": 4.0},
        time,
        speed,
    )
    assert "use_speed_threshold_for_zscore" not in kwargs
    np.testing.assert_array_equal(kwargs["normalization_mask"], speed < 4.0)
    assert not kwargs["normalization_mask"][100]


def test_speed_flag_uses_detector_default_threshold(signals):
    time, speed, _ = signals
    kwargs = detection_kwargs_from_params(
        {"use_speed_threshold_for_zscore": True}, time, speed
    )
    np.testing.assert_array_equal(kwargs["normalization_mask"], speed < 4.0)


def test_false_speed_flag_is_dropped(signals):
    time, speed, _ = signals
    kwargs = detection_kwargs_from_params(
        {"use_speed_threshold_for_zscore": False}, time, speed
    )
    assert kwargs == {}


def test_time_range_takes_precedence_over_speed_flag(signals):
    time, speed, _ = signals
    kwargs = detection_kwargs_from_params(
        {
            "use_speed_threshold_for_zscore": True,
            "normalization_time_range": (1.0, 2.0),
        },
        time,
        speed,
    )
    np.testing.assert_array_equal(
        kwargs["normalization_mask"], (time >= 1.0) & (time <= 2.0)
    )


def test_mask_and_time_range_raise(signals):
    time, speed, _ = signals
    with pytest.raises(ValueError, match="normalization_time_range"):
        detection_kwargs_from_params(
            {
                "normalization_mask": np.ones_like(time, dtype=bool),
                "normalization_time_range": (1.0, 2.0),
            },
            time,
            speed,
        )


def test_stored_legacy_row_runs_on_ripple_detection(signals):
    """A 1.x MuaEventsParameters row fails in ripple_detection 2.0 as is."""
    from ripple_detection import multiunit_HSE_detector

    time, speed, multiunit = signals
    params = {
        "minimum_duration": 0.015,
        "zscore_threshold": 2.0,
        "close_event_threshold": 0.0,
        "speed_threshold": 4.0,
        "use_speed_threshold_for_zscore": True,
    }
    with pytest.raises(TypeError, match="use_speed_threshold_for_zscore"):
        multiunit_HSE_detector(time, multiunit, speed, FS, **params)

    translated = multiunit_HSE_detector(
        time,
        multiunit,
        speed,
        FS,
        **detection_kwargs_from_params(params, time, speed),
    )
    explicit = multiunit_HSE_detector(
        time,
        multiunit,
        speed,
        FS,
        minimum_duration=0.015,
        zscore_threshold=2.0,
        close_event_threshold=0.0,
        speed_threshold=4.0,
        normalization_mask=speed < 4.0,
    )
    assert len(translated) > 0
    assert translated.equals(explicit)
