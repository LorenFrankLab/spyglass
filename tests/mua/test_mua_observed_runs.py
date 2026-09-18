"""Multiunit event detection never joins samples across unobserved time."""

import numpy as np
import pandas as pd
import pytest
from ripple_detection import multiunit_HSE_detector

from spyglass.utils.spikesorting import contiguous_observed_runs

SAMPLING_FREQUENCY = 1000.0
DEPRECATION_MATCH = "use_speed_threshold_for_zscore"


def test_runs_split_on_unobserved_samples_and_timestamp_gaps():
    time = np.arange(10) / SAMPLING_FREQUENCY
    time[6:] += 1.0  # a one-second jump between sample 5 and sample 6
    mask = np.ones(10, dtype=bool)
    mask[2:4] = False

    runs = contiguous_observed_runs(time, mask, SAMPLING_FREQUENCY)

    assert [run.tolist() for run in runs] == [[0, 1], [4, 5], [6, 7, 8, 9]]
    assert all(run.dtype.kind == "i" for run in runs)


def test_runs_are_empty_when_nothing_is_observed():
    time = np.arange(10) / SAMPLING_FREQUENCY

    assert (
        contiguous_observed_runs(
            time, np.zeros(10, dtype=bool), SAMPLING_FREQUENCY
        )
        == []
    )
    assert (
        contiguous_observed_runs(
            np.array([]), np.array([], dtype=bool), SAMPLING_FREQUENCY
        )
        == []
    )


def test_one_sample_gap_tolerance_keeps_a_jittered_run_intact():
    """A sample-to-sample jitter below 1.5 dt is not a recording gap."""
    time = np.arange(6) / SAMPLING_FREQUENCY
    time[3:] += 0.4 / SAMPLING_FREQUENCY  # 1.4 dt step, still contiguous
    mask = np.ones(6, dtype=bool)

    runs = contiguous_observed_runs(time, mask, SAMPLING_FREQUENCY)

    assert [run.tolist() for run in runs] == [[0, 1, 2, 3, 4, 5]]


@pytest.fixture(scope="module")
def detection():
    """The DB-free detection module.

    Imported inside a fixture: ``spyglass.mua.v1.__init__`` pulls in the
    schema module, which needs a live database connection, and collection
    runs before the session fixtures bring one up.
    """
    from spyglass.mua.v1 import _detection

    return _detection


def _planted_indicator(time, burst_windows, rng, background_rate=10.0):
    """Poisson background with high-rate bursts in the given time windows."""
    indicator = rng.poisson(
        background_rate / SAMPLING_FREQUENCY, size=(time.size, 2)
    ).astype(float)
    for start, stop in burst_windows:
        in_burst = (time >= start) & (time < stop)
        indicator[in_burst] = rng.poisson(
            0.3, size=(int(in_burst.sum()), 2)
        ).astype(float)
    return indicator


@pytest.fixture(scope="module")
def synthetic_mua():
    """Four seconds of two-unit spiking with two planted bursts.

    The animal is immobile except for one stretch, so the deprecated
    speed-threshold normalization and an explicit mask select different
    samples than the full population does.
    """
    rng = np.random.default_rng(0)
    time = np.arange(0, 4, 1 / SAMPLING_FREQUENCY)
    speed = np.zeros(time.size)
    speed[(time >= 2.5) & (time < 3.0)] = 10.0
    indicator = _planted_indicator(time, [(0.5, 0.56), (1.5, 1.56)], rng=rng)
    return time, indicator, speed


@pytest.fixture(scope="module")
def gapped_mua():
    """Two bursts straddling a 380 ms unobserved interval."""
    rng = np.random.default_rng(2)
    time = np.arange(0, 2, 1 / SAMPLING_FREQUENCY)
    speed = np.zeros(time.size)
    indicator = _planted_indicator(time, [(0.5, 0.56), (0.94, 1.0)], rng=rng)
    gap_start, gap_end = 0.56, 0.94
    mask = ~((time >= gap_start) & (time < gap_end))
    indicator[~mask] = np.nan  # unobserved bins carry no spike evidence
    return time, indicator, speed, mask, (gap_start, gap_end)


@pytest.mark.parametrize(
    "make_kwargs",
    [
        pytest.param(lambda speed: {}, id="defaults"),
        pytest.param(
            lambda speed: {"use_speed_threshold_for_zscore": True},
            id="deprecated_speed_threshold",
        ),
        pytest.param(
            lambda speed: {"normalization_method": "median_mad"},
            id="median_mad",
        ),
        pytest.param(
            lambda speed: {"normalization_time_range": (0.5, 1.5)},
            id="normalization_time_range",
        ),
        pytest.param(
            lambda speed: {"normalization_mask": speed < 4.0},
            id="explicit_normalization_mask",
        ),
    ],
)
def test_single_run_equals_current_detector(
    synthetic_mua, detection, make_kwargs
):
    """One fully observed run reproduces the detector step for step."""
    time, indicator, speed = synthetic_mua
    kwargs = make_kwargs(speed)
    mask = np.ones(time.size, dtype=bool)

    if kwargs.get("use_speed_threshold_for_zscore"):
        with pytest.warns(DeprecationWarning, match=DEPRECATION_MATCH):
            expected = multiunit_HSE_detector(
                time, indicator, speed, SAMPLING_FREQUENCY, **kwargs
            )
        with pytest.warns(DeprecationWarning, match=DEPRECATION_MATCH):
            actual = detection.detect_multiunit_events_in_observed_runs(
                time, indicator, speed, SAMPLING_FREQUENCY, mask, **kwargs
            )
    else:
        expected = multiunit_HSE_detector(
            time, indicator, speed, SAMPLING_FREQUENCY, **kwargs
        )
        actual = detection.detect_multiunit_events_in_observed_runs(
            time, indicator, speed, SAMPLING_FREQUENCY, mask, **kwargs
        )

    assert len(expected) > 0  # the comparison would be vacuous otherwise
    pd.testing.assert_frame_equal(actual, expected)


def test_no_events_returns_the_detector_empty_frame(synthetic_mua, detection):
    time, _, speed = synthetic_mua
    silent = np.zeros((time.size, 2))
    mask = np.ones(time.size, dtype=bool)

    expected = multiunit_HSE_detector(time, silent, speed, SAMPLING_FREQUENCY)
    actual = detection.detect_multiunit_events_in_observed_runs(
        time, silent, speed, SAMPLING_FREQUENCY, mask
    )

    assert expected.empty
    pd.testing.assert_frame_equal(actual, expected)


def test_nan_inside_an_observed_run_is_refused(synthetic_mua, detection):
    """The observed mask and the indicator must agree before detection."""
    time, indicator, speed = synthetic_mua
    indicator = indicator.copy()
    indicator[100] = np.nan

    with pytest.raises(ValueError, match="non-finite firing rate"):
        detection.detect_multiunit_events_in_observed_runs(
            time,
            indicator,
            speed,
            SAMPLING_FREQUENCY,
            np.ones(time.size, dtype=bool),
        )


def test_mask_and_time_range_together_raise(synthetic_mua, detection):
    time, indicator, speed = synthetic_mua
    mask = np.ones(time.size, dtype=bool)

    with pytest.raises(ValueError, match="Cannot specify both"):
        detection.detect_multiunit_events_in_observed_runs(
            time,
            indicator,
            speed,
            SAMPLING_FREQUENCY,
            mask,
            normalization_mask=speed < 4.0,
            normalization_time_range=(0.5, 1.5),
        )


def test_short_run_in_normalization_but_not_events(detection):
    """A run too short to hold an event still sets the normalization."""
    rng = np.random.default_rng(1)
    time = np.arange(0, 2, 1 / SAMPLING_FREQUENCY)
    speed = np.zeros(time.size)
    indicator = _planted_indicator(time, [(1.0, 1.06)], rng=rng)
    short_run = (time >= 0.1) & (time < 0.11)  # 10 ms: under minimum_duration
    indicator[short_run] = 1.0
    long_run = (time >= 0.5) & (time < 2.0)

    with_short = short_run | long_run
    args = (time, indicator, speed, SAMPLING_FREQUENCY)

    events = detection.detect_multiunit_events_in_observed_runs(
        *args, with_short
    )
    assert len(events) > 0
    assert (events.start_time >= 0.5).all()  # nothing from the short run

    included, runs = detection.normalize_observed_rate(*args, with_short)
    excluded, _ = detection.normalize_observed_rate(*args, long_run)

    assert [run.size for run in runs] == [10, 1500]
    assert events.max_zscore.max() == pytest.approx(
        np.nanmax(included[long_run])
    )
    assert np.nanmax(included[long_run]) != pytest.approx(
        np.nanmax(excluded[long_run])
    )


def test_helper_runs_end_to_end_on_a_single_column_indicator(
    synthetic_mua, detection
):
    time, indicator, speed = synthetic_mua
    multiunit = indicator.sum(axis=1, keepdims=True)
    mask = np.ones(time.size, dtype=bool)

    events = detection.detect_multiunit_events_in_observed_runs(
        time, multiunit, speed, SAMPLING_FREQUENCY, mask
    )

    assert list(events.columns) == list(
        multiunit_HSE_detector(
            time, multiunit, speed, SAMPLING_FREQUENCY
        ).columns
    )
    assert len(events) > 0


def test_events_cannot_bridge_unobserved_gap(
    gapped_mua, detection, monkeypatch
):
    time, indicator, speed, mask, (gap_start, gap_end) = gapped_mua
    normalized_sizes = []
    real_normalize_signal = detection.normalize_signal

    def assert_finite_then_normalize(data, **kwargs):
        """No NaN may reach the detector's numerical routines."""
        assert np.isfinite(np.asarray(data, dtype=float)).all()
        normalized_sizes.append(np.asarray(data).size)
        return real_normalize_signal(data, **kwargs)

    monkeypatch.setattr(
        detection, "normalize_signal", assert_finite_then_normalize
    )

    events = detection.detect_multiunit_events_in_observed_runs(
        time, indicator, speed, SAMPLING_FREQUENCY, mask
    )

    assert normalized_sizes == [int(mask.sum())]
    assert len(events) == 2
    spans_gap = (events.start_time < gap_end) & (events.end_time > gap_start)
    assert not spans_gap.any()


def test_event_adjacent_to_gap_is_detected_once(detection):
    rng = np.random.default_rng(3)
    time = np.arange(0, 2, 1 / SAMPLING_FREQUENCY)
    speed = np.zeros(time.size)
    indicator = _planted_indicator(time, [(0.5, 0.56)], rng=rng)
    mask = np.ones(time.size, dtype=bool)
    mask[(time >= 0.56) & (time < 0.58)] = False  # 20 unobserved bins
    indicator[~mask] = np.nan

    events = detection.detect_multiunit_events_in_observed_runs(
        time, indicator, speed, SAMPLING_FREQUENCY, mask
    )

    assert len(events) == 1


def test_event_numbers_unique_after_concat(gapped_mua, detection):
    time, indicator, speed, mask, _ = gapped_mua

    events = detection.detect_multiunit_events_in_observed_runs(
        time, indicator, speed, SAMPLING_FREQUENCY, mask
    )

    assert events.index.name == "event_number"
    assert events.index.tolist() == [1, 2]
    assert events.start_time.is_monotonic_increasing


@pytest.mark.skip(
    reason="The pop_spikes_group fixture cannot be built under "
    "SpikeInterface >= 0.101: its pop_rec dependency populates the legacy "
    "SpikeSortingRecording, whose _get_filtered_recording calls "
    "recording.channel_slice (spikesorting/v1/recording.py:644), removed in "
    "SI 0.101 -> AttributeError: 'FrameSliceRecording' object has no "
    "attribute 'channel_slice'. Needs a SortedSpikesGroup fixture that does "
    "not run the legacy v1 recording path."
)
@pytest.mark.slow
@pytest.mark.integration
def test_event_numbers_unique_after_concat_nwb_round_trip(
    pop_spikes_group, pos_merge_key, monkeypatch
):
    """Two runs' events survive the NWB write with distinct event numbers."""
    from spyglass.mua.v1.mua import MuaEventsParameters, MuaEventsV1
    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

    param_name = "observed_runs_two_bursts"
    MuaEventsParameters().insert1(
        {
            "mua_param_name": param_name,
            "mua_param_dict": {
                "minimum_duration": 0.015,
                "zscore_threshold": 2.0,
                "close_event_threshold": 0.0,
                "speed_threshold": 1e6,  # keep every event: speed is not
                # what this test is about
            },
        },
        skip_duplicates=True,
    )

    def planted_indicator(key, time, return_unit_ids=False, **kwargs):
        """One burst on either side of an unobserved stretch."""
        n_time = np.asarray(time).size
        indicator = np.zeros((n_time, 1))
        observed = np.ones(n_time, dtype=bool)
        observed[int(0.4 * n_time) : int(0.6 * n_time)] = False
        indicator[int(0.15 * n_time) : int(0.20 * n_time)] = 5.0
        indicator[int(0.80 * n_time) : int(0.85 * n_time)] = 5.0
        indicator[~observed] = np.nan
        if kwargs.get("return_validity"):
            return indicator, observed
        return indicator

    monkeypatch.setattr(
        SortedSpikesGroup, "get_spike_indicator", planted_indicator
    )

    key = {
        "mua_param_name": param_name,
        **pop_spikes_group,
        "pos_merge_id": pos_merge_key["merge_id"],
        "detection_interval": "01_s1",
    }
    MuaEventsV1().populate(key)

    events = (MuaEventsV1 & key).fetch1_dataframe()

    assert events.index.to_list() == [1, 2]
    assert events.start_time.is_monotonic_increasing
