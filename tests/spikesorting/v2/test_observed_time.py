"""Observation exposure is sample-exact and distinct from neural silence."""

import numpy as np
import pytest

from spyglass.spikesorting.v2._observed_time import (
    ObservationAvailability,
    contains_times,
    observed_intervals,
    observed_metrics,
    population_availability,
)


def test_observed_intervals_end_from_timestamps():
    """The interval end comes from the recording's own timestamps, not the
    nominal sampling rate -- a clock that actually runs slower than its
    declared rate must not drop real samples off the end of an interval."""
    import spikeinterface.core as si

    # A recording declared at 1000 Hz whose clock actually runs at 990 Hz.
    # Under the old ``t + n/fs`` (declared-fs) arithmetic the computed end
    # falls short of the recording's own last timestamp, excluding samples
    # that were genuinely recorded.
    n = 1000
    actual_fs = 990.0
    recording = si.NumpyRecording(np.zeros((n, 1)), 1000.0)
    times = np.arange(n) / actual_fs
    recording.set_times(times)
    # ``detect_artifacts``'s no-detection fallback: valid_times is the
    # recording's own actual envelope.
    valid_times = [[float(times[0]), float(times[-1])]]

    intervals = observed_intervals(recording, valid_times)
    assert intervals.shape == (1, 2)
    assert np.all(intervals[1:, 0] >= intervals[:-1, 1])
    contained = contains_times(intervals, times)
    assert contained.all(), f"{int((~contained).sum())} of {n} samples excluded"

    # Regular-clock case: the new data-derived end equals the old
    # nominal-rate ``t + n/fs`` exactly when the clock is regular.
    fs = 1000.0
    recording_regular = si.NumpyRecording(np.zeros((n, 1)), fs)
    regular_times = np.arange(n) / fs
    recording_regular.set_times(regular_times)
    regular_valid = [[float(regular_times[0]), float(regular_times[-1])]]
    regular_intervals = observed_intervals(recording_regular, regular_valid)
    np.testing.assert_allclose(regular_intervals, [[0.0, n / fs]])

    # Multi-interval case: a timestamp gap splits the recording into two
    # recorded chunks; every sample of each chunk must be contained.
    gapped_times = np.r_[np.arange(500) / fs, 100 + np.arange(500) / fs]
    recording_gapped = si.NumpyRecording(np.zeros((1000, 1)), fs)
    recording_gapped.set_times(gapped_times)
    gapped_valid = [
        [float(gapped_times[0]), float(gapped_times[499])],
        [float(gapped_times[500]), float(gapped_times[-1])],
    ]
    gapped_intervals = observed_intervals(recording_gapped, gapped_valid)
    assert gapped_intervals.shape == (2, 2)
    assert np.all(gapped_intervals[1:, 0] >= gapped_intervals[:-1, 1])
    assert contains_times(gapped_intervals, gapped_times).all()


def test_observed_intervals_rejects_overlapping_output():
    """A fast actual clock under a slower declared rate can make the
    data-derived end of one kept interval overtake the next interval's
    start; ``observed_intervals`` must raise instead of returning
    overlapping output."""
    import spikeinterface.core as si

    n = 20
    declared_fs = 1000.0
    actual_fs = 4000.0
    recording = si.NumpyRecording(np.zeros((n, 1)), declared_fs)
    times = np.arange(n) / actual_fs
    recording.set_times(times)
    # Excludes exactly frame 10 (a single missing sample); the actual gap
    # (2 sample periods at 4000 Hz) is smaller than the 1/1000s pad the
    # nominal-rate arithmetic would add to the previous interval's end.
    valid_times = [
        [float(times[0]), float(times[10])],
        [float(times[11]), float(times[-1])],
    ]
    with pytest.raises(ValueError, match="not sorted|overlap|disjoint"):
        observed_intervals(recording, valid_times)


def test_observed_duration_counts_final_samples_and_preserves_recording_gaps():
    import spikeinterface.core as si

    recording = si.NumpyRecording(np.zeros((2000, 1)), 1000)
    recording.set_times(
        np.r_[np.arange(1000) / 1000, 10 + np.arange(1000) / 1000]
    )
    full = observed_intervals(recording, [[0, 0.999], [10, 10.999]])
    np.testing.assert_allclose(full, [[0, 1], [10, 11]])
    partial = observed_intervals(
        recording, [[0, 0.4], [0.6, 0.999], [10, 10.999]]
    )
    np.testing.assert_allclose(partial, [[0, 0.4], [0.6, 1], [10, 11]])
    assert np.diff(partial, axis=1).sum() == pytest.approx(1.8)
    assert contains_times(
        partial, [0.399, 0.4, 0.599, 0.6, 1, 10, 11]
    ).tolist() == [True, False, False, True, False, True, False]


def test_concat_exclusive_end_accepts_rounding_between_equivalent_clocks():
    import spikeinterface.core as si

    n_samples = 300_000
    sampling_frequency = 30_000.00000000018
    recording = si.NumpyRecording(np.zeros((n_samples, 1)), sampling_frequency)
    exclusive_stop = n_samples / sampling_frequency
    last_sample_plus_period = (n_samples - 1) / sampling_frequency + (
        1 / sampling_frequency
    )
    assert exclusive_stop > last_sample_plus_period
    actual = observed_intervals(recording, [[0, exclusive_stop]])
    np.testing.assert_allclose(actual, [[0, exclusive_stop]])


def test_observed_metrics_weight_partial_bins_and_ignore_excluded_spikes():
    result = observed_metrics([1, 2.5, 4.5], [[0, 3], [5, 6]], bin_duration_s=2)
    assert result == {
        "observed_duration_s": 4,
        "observed_firing_rate_hz": 0.5,
        "observed_presence_ratio": 0.75,
    }
    empty = observed_metrics([], [], bin_duration_s=2)
    assert empty["observed_duration_s"] == 0
    assert np.isnan(empty["observed_firing_rate_hz"])
    silent = observed_metrics([], [[0, 6]], bin_duration_s=2)
    assert (
        silent["observed_firing_rate_hz"]
        == silent["observed_presence_ratio"]
        == 0
    )


def test_population_intersects_selected_sources_and_reports_unknown_coverage():
    availability = population_availability(
        [
            (
                "first",
                [1],
                {"observed_intervals_by_unit": {"1": [[0, 2], [3, 10]]}},
            ),
            ("second", [2], {"observed_intervals_by_unit": {"2": [[1, 6]]}}),
            ("empty", [], {"observed_intervals_by_unit": {}}),
            ("legacy", [3], {}),
        ]
    )
    np.testing.assert_array_equal(availability.intervals, [[1, 2], [3, 6]])
    assert availability.duration_s == 4
    assert availability.unknown_sources == ("legacy",)
    assert availability.valid_bins([1, 1.5, 2, 2.5, 3, 3.5, 4]).tolist() == [
        True,
        True,
        False,
        False,
        True,
        True,
        True,
    ]
    np.testing.assert_array_equal(
        availability.restrict([[0, 4]]), [[1, 2], [3, 4]]
    )
    assert ObservationAvailability(None).contains([1, 100]).all()
    assert (
        not ObservationAvailability(np.empty((0, 2))).contains([1, 100]).any()
    )


def test_unmasked_and_short_recordings_have_defined_presence():
    full = observed_metrics([0.25, 1.25], [[0, 2]], bin_duration_s=1)
    assert full["observed_firing_rate_hz"] == 1
    assert full["observed_presence_ratio"] == 1
    short = observed_metrics([0.1], [[0, 0.5]], bin_duration_s=60)
    assert short["observed_presence_ratio"] == 1
    assert short["observed_firing_rate_hz"] == 2


def test_binned_counts_and_smoothing_do_not_bridge_exclusions(
    common, monkeypatch
):
    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

    time = np.arange(10, dtype=float)
    availability = ObservationAvailability(np.array([[0, 3], [6, 10]]))
    monkeypatch.setattr(
        SortedSpikesGroup,
        "get_observation_intervals",
        lambda key, **kwargs: availability,
    )
    monkeypatch.setattr(
        SortedSpikesGroup,
        "fetch_spike_data",
        lambda key, return_unit_ids: (
            [np.array([1.0, 2.0, 4.0])],
            [{"unit_id": 1}],
        ),
    )
    counts, valid = SortedSpikesGroup.get_spike_indicator(
        {}, time, return_validity=True
    )
    assert np.isnan(counts[3:6]).all()
    assert np.all(counts[6:] == 0)
    assert not valid[3:6].any()
    rate = SortedSpikesGroup.get_firing_rate({}, time, smoothing_sigma=2)
    assert np.isnan(rate[3:6]).all()
    assert np.all(rate[6:] == 0)
    assert rate[:3].sum() > 0

    # The legacy digitize convention counts the time axis's final endpoint in
    # the preceding bin. An excluded spike exactly there must still be dropped.
    availability = ObservationAvailability(np.array([[0, 3]]))
    monkeypatch.setattr(
        SortedSpikesGroup,
        "fetch_spike_data",
        lambda key, return_unit_ids: (
            [np.array([2.0, 3.0])],
            [{"unit_id": 1}],
        ),
    )
    counts = SortedSpikesGroup.get_spike_indicator({}, time[:4])
    assert counts[2, 0] == 1
    assert np.isnan(counts[3, 0])


def test_group_smoothing_stops_at_a_timestamp_jump(common, monkeypatch):
    """A gap in the time axis ends a run even where the units were observed.

    The samples either side of the jump are both observed, so an availability
    mask alone would treat them as one run and smooth a spike across seven
    seconds of missing samples.
    """
    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

    time = np.r_[np.arange(4.0), 10.0 + np.arange(6.0)]
    monkeypatch.setattr(
        SortedSpikesGroup,
        "get_observation_intervals",
        lambda key, **kwargs: ObservationAvailability(np.array([[0.0, 14.0]])),
    )
    monkeypatch.setattr(
        SortedSpikesGroup,
        "fetch_spike_data",
        lambda key, return_unit_ids: ([np.array([1.0])], [{"unit_id": 1}]),
    )

    counts, valid = SortedSpikesGroup.get_spike_indicator(
        {}, time, return_validity=True
    )
    assert counts[1, 0] == 1
    assert valid.tolist() == [True] * 8 + [False, False]

    rate = SortedSpikesGroup.get_firing_rate({}, time, smoothing_sigma=2)
    assert rate[:4, 0].sum() > 0
    assert rate[4:8, 0].tolist() == [0.0] * 4
    assert np.isnan(rate[8:, 0]).all()


def test_group_smoothing_splits_a_jump_with_every_bin_observed(
    common, monkeypatch
):
    """A fully observed population still splits at a timestamp jump.

    Every bin here is inside the observed span, so a validity mask alone
    offers no reason to split; only the clock jump does. Smoothing the whole
    axis would carry the single spike across six seconds of missing samples.
    """
    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

    time = np.r_[np.arange(4.0), 10.0 + np.arange(4.0)]
    monkeypatch.setattr(
        SortedSpikesGroup,
        "get_observation_intervals",
        lambda key, **kwargs: ObservationAvailability(np.array([[0.0, 14.0]])),
    )
    monkeypatch.setattr(
        SortedSpikesGroup,
        "fetch_spike_data",
        lambda key, return_unit_ids: ([np.array([1.0])], [{"unit_id": 1}]),
    )

    counts, valid = SortedSpikesGroup.get_spike_indicator(
        {}, time, return_validity=True
    )
    assert counts[1, 0] == 1
    assert valid.all()

    rate = SortedSpikesGroup.get_firing_rate({}, time, smoothing_sigma=2)
    assert rate[:4, 0].sum() > 0
    assert rate[4:, 0].tolist() == [0.0] * 4


def test_shared_masks_are_stored_once_and_keep_original_fingerprint(
    monkeypatch,
):
    import hashlib
    import json

    from spyglass.spikesorting.v2 import _observation_io as io
    from spyglass.spikesorting.v2 import _observed_time as math

    calls = []

    def convert(recording, intervals):
        calls.append(intervals)
        return np.asarray(intervals)

    monkeypatch.setattr(io, "observed_intervals", convert)
    intervals = [[0.0, 2.0], [3.0, 10.0]]
    by_unit = {str(i): intervals for i in range(500)}
    by_unit["501"] = [[1.0, 4.0]]
    compact = io.canonical_unit_intervals(None, by_unit)
    assert len(compact["masks"]) == len(calls) == 2
    assert (
        io.observation_fingerprint(compact)
        == hashlib.sha256(
            json.dumps(by_unit, sort_keys=True).encode()
        ).hexdigest()
    )
    assert len(json.dumps(compact)) < len(json.dumps(by_unit)) / 2
    intersections = []
    original = math.intersect_intervals

    def intersect(a, b):
        intersections.append((a, b))
        return original(a, b)

    monkeypatch.setattr(math, "intersect_intervals", intersect)
    all_shared = math.population_availability(
        [("source", list(range(500)), {"observation_intervals": compact})]
    )
    np.testing.assert_array_equal(all_shared.intervals, intervals)
    assert not intersections
    combined = math.population_availability(
        [("source", [1, 501], {"observation_intervals": compact})]
    )
    np.testing.assert_array_equal(combined.intervals, [[1, 2], [3, 4]])
    assert len(intersections) == 1


def test_review_timeline_cache_is_pinned_to_curation_generation(
    tmp_path, monkeypatch
):
    from spyglass.spikesorting.v2 import _observation_io as io

    calls = []

    def build(key):
        calls.append(key)
        return {
            "excluded": np.array([[1, 2]]),
            "mappings": [],
            "concatenated": False,
        }

    monkeypatch.setattr(io, "review_timeline", build)
    path = tmp_path / "timeline.json"
    for _ in range(2):
        result = io.cached_review_timeline(
            {"curation_id": 1}, cache_path=path, curation_uuid="first"
        )
        np.testing.assert_array_equal(result["excluded"], [[1, 2]])
    assert len(calls) == 1
    io.cached_review_timeline(
        {"curation_id": 1}, cache_path=path, curation_uuid="replacement"
    )
    assert len(calls) == 2
