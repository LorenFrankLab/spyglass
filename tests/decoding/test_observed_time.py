"""Observed time constrains actual decoder calls, including explicit masks."""

import json
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytestmark = [pytest.mark.database, pytest.mark.integration]


class RecordingDetector:
    """Record the external model boundary while Spyglass runs its real decoder."""

    instances: ClassVar[list] = []

    def __init__(self, **kwargs):
        self.initial_conditions_ = np.array([1.0])
        self.discrete_state_transitions_ = np.array([[1.0]])
        self.calls = []
        self.instances.append(self)

    def _record(self, method, time, position, spike_times, **masks):
        self.calls.append(
            {
                "method": method,
                "time": np.asarray(time).copy(),
                "position": np.asarray(position).copy(),
                "spike_times": [
                    np.asarray(train).copy() for train in spike_times
                ],
                **{
                    name: None if mask is None else np.asarray(mask).copy()
                    for name, mask in masks.items()
                },
            }
        )

    @staticmethod
    def _results(time):
        return xr.Dataset(
            {
                "acausal_posterior": (
                    ("time", "state_bins"),
                    np.ones((len(time), 1)),
                )
            },
            coords={"time": time, "states": ["local"], "state_bins": [0]},
        )

    def fit(self, *, position_time, position, spike_times, is_training=None):
        self._record(
            "fit", position_time, position, spike_times, is_training=is_training
        )

    def predict(
        self, *, position_time, position, spike_times, time, is_missing=None
    ):
        np.testing.assert_array_equal(position_time, time)
        self._record(
            "predict", time, position, spike_times, is_missing=is_missing
        )
        return self._results(time)

    def estimate_parameters(
        self,
        *,
        position_time,
        position,
        spike_times,
        time,
        is_training=None,
        is_missing=None,
    ):
        np.testing.assert_array_equal(position_time, time)
        self._record(
            "estimate",
            time,
            position,
            spike_times,
            is_training=is_training,
            is_missing=is_missing,
        )
        return self._results(time)


@pytest.fixture
def recording_detector(monkeypatch):
    from non_local_detector.models import base

    instances = []
    monkeypatch.setattr(RecordingDetector, "instances", instances)
    monkeypatch.setattr(base, "SortedSpikesDetector", RecordingDetector)
    return instances


@pytest.mark.parametrize("estimate", [False, True])
def test_decoder_branches_preserve_missing_time(
    decode_v1, recording_detector, estimate
):
    time = np.arange(10, dtype=float)
    training = (time < 3) | (time >= 7)
    explicit_missing = time == 8
    _, result = decode_v1.sorted_spikes.SortedSpikesDecodingV1()._run_decoder(
        key={"estimate_decoding_params": estimate},
        decoding_params={},
        decoding_kwargs={
            "is_training": training,
            "is_missing": explicit_missing,
        },
        position_info=pd.DataFrame({"x": time}, index=time),
        position_variable_names=["x"],
        spike_times=[np.array([1.0, 7.0])],
        decoding_interval=np.array([[0.0, 2.0], [7.0, 9.0]]),
    )
    calls = recording_detector[0].calls
    np.testing.assert_array_equal(calls[0]["is_training"], training)
    if estimate:
        missing = ~training | explicit_missing
        assert [call["method"] for call in calls] == ["estimate"]
        np.testing.assert_array_equal(calls[0]["is_missing"], missing)
        np.testing.assert_array_equal(
            result.interval_labels, [0, 0, 0, -1, -1, -1, -1, 1, -1, 2]
        )
    else:
        assert [call["method"] for call in calls] == [
            "fit",
            "predict",
            "predict",
        ]
        np.testing.assert_array_equal(result.time, [0, 1, 2, 7, 8, 9])
        np.testing.assert_array_equal(calls[1]["time"], [0, 1, 2])
        np.testing.assert_array_equal(calls[2]["time"], [7, 8, 9])
        np.testing.assert_array_equal(
            calls[1]["is_missing"], [False, False, False]
        )
        np.testing.assert_array_equal(
            calls[2]["is_missing"], [False, True, False]
        )
        np.testing.assert_array_equal(
            result.is_missing, [False, False, False, False, True, False]
        )


@pytest.fixture
def observed_population(
    decode_v1,
    monkeypatch,
    mock_results_storage,
    decode_sel_key,
    decode_spike_params_insert,
    pop_pos_group,
    pop_spikes_group,
):
    """Use real population reads with independently indexed availability oracles."""
    from spyglass.spikesorting.v2._observed_time import ObservationAvailability

    mod = decode_v1.sorted_spikes
    key = {
        **decode_sel_key,
        **decode_spike_params_insert,
        **pop_spikes_group,
        "estimate_decoding_params": False,
    }
    table = mod.SortedSpikesDecodingV1
    position, variables = table.fetch_position_info(key)
    time = position.index.to_numpy()
    assert len(time) >= 30
    first_stop, second_start = len(time) // 3, 2 * len(time) // 3
    encoding_start, encoding_stop = len(time) // 10, 9 * len(time) // 10
    stop = np.nextafter(time[-1], np.inf)
    observation_intervals = np.array(
        [[time[0], time[first_stop]], [time[second_start], stop]]
    )
    observation = ObservationAvailability(observation_intervals)
    observed = np.zeros(len(time), dtype=bool)
    observed[:first_stop] = True
    observed[second_start:] = True
    encoding = np.zeros(len(time), dtype=bool)
    encoding[encoding_start : encoding_stop + 1] = True
    invalid = [encoding_start + 1, second_start + 1]
    finite_position = position[variables].notna().all(axis=1).to_numpy()
    finite_position[invalid] = False
    expected_training = observed & encoding & finite_position
    assert expected_training.sum() >= 2

    # Fetch the actual PositionGroup data, adding two invalid positions inside
    # otherwise eligible encoding spans. No make/decoder method is bypassed.
    fetch_position = table.fetch_position_info

    def position_with_invalid_rows(cls, key):
        frame, columns = fetch_position(key)
        np.testing.assert_array_equal(frame.index, time)
        frame = frame.copy()
        frame.loc[time[invalid], columns[0]] = np.nan
        return frame, columns

    monkeypatch.setattr(
        table, "fetch_position_info", classmethod(position_with_invalid_rows)
    )
    availability = {"value": observation}
    monkeypatch.setattr(
        mod.SortedSpikesGroup,
        "get_observation_intervals",
        lambda key, **kwargs: availability["value"],
    )
    encoding_intervals = np.array([[time[encoding_start], time[encoding_stop]]])
    decoding_intervals = np.array([[time[0], stop]])
    spike_times = table.fetch_spike_data(key, filter_by_interval=False)
    # Compare directly to fixture endpoints rather than calling the same
    # availability.contains() helper the production code uses.
    expected_spikes = [
        train[
            ((train >= time[0]) & (train < time[first_stop]))
            | ((train >= time[second_start]) & (train < stop))
        ]
        for train in spike_times
    ]
    assert sum(map(len, expected_spikes)) < sum(map(len, spike_times))
    assert sum(map(len, expected_spikes)) > 0
    relation = mod.DecodingParameters & decode_spike_params_insert
    original = {
        **decode_spike_params_insert,
        "decoding_kwargs": relation.fetch1("decoding_kwargs"),
    }
    interval_keys = [
        {"nwb_file_name": key["nwb_file_name"], "interval_list_name": name}
        for name in (
            "observed_time_test_encoding",
            "observed_time_test_decoding",
        )
    ]
    mod.IntervalList.insert(
        [
            {**interval_key, "valid_times": intervals}
            for interval_key, intervals in zip(
                interval_keys,
                [encoding_intervals, decoding_intervals],
                strict=True,
            )
        ]
    )
    key.update(
        encoding_interval=interval_keys[0]["interval_list_name"],
        decoding_interval=interval_keys[1]["interval_list_name"],
    )
    try:
        yield {
            "mod": mod,
            "key": key,
            "time": time,
            "availability": availability,
            "observed": observed,
            "encoding": encoding,
            "invalid": invalid,
            "training": expected_training,
            "first_stop": first_stop,
            "second_start": second_start,
            "spike_times": spike_times,
            "expected_spikes": expected_spikes,
            "observation_intervals": observation_intervals,
            "encoding_intervals": np.array(
                [
                    [time[encoding_start], time[first_stop]],
                    [time[second_start], time[encoding_stop]],
                ]
            ),
            "requested_encoding_intervals": encoding_intervals,
            "params_key": decode_spike_params_insert,
            "results_storage": mock_results_storage,
        }
    finally:
        cleanup_key = {
            k: v for k, v in key.items() if k != "estimate_decoding_params"
        }
        try:
            (mod.SortedSpikesDecodingSelection & cleanup_key).super_delete(
                warn=False, safemode=False, force_masters=True
            )
        finally:
            try:
                (mod.IntervalList & interval_keys).delete_quick()
            finally:
                mod.DecodingParameters.update1(original)


def _populate_population(case, kwargs, estimate):
    mod = case["mod"]
    key = {**case["key"], "estimate_decoding_params": estimate}
    mod.DecodingParameters.update1(
        {**case["params_key"], "decoding_kwargs": kwargs}
    )
    mod.SortedSpikesDecodingSelection.insert1(key)
    mod.SortedSpikesDecodingV1.populate(key)
    result_path = (mod.SortedSpikesDecodingV1 & key).fetch1("results_path")
    return case["results_storage"]["results"][str(result_path)]


@pytest.mark.parametrize("estimate", [False, True])
@pytest.mark.parametrize("training_mode", ["omitted", "none", "array"])
def test_make_applies_exact_observation_and_encoding_masks(
    observed_population, recording_detector, estimate, training_mode
):
    case = observed_population
    time = case["time"]
    explicit_missing = np.zeros(len(time), dtype=bool)
    missing_index = case["second_start"] + 2
    explicit_missing[missing_index] = True
    kwargs = {"is_missing": explicit_missing}
    expected_training = case["training"].copy()
    if training_mode == "none":
        kwargs["is_training"] = None
    elif training_mode == "array":
        caller_training = np.ones(len(time), dtype=bool)
        caller_training[np.flatnonzero(expected_training)[0]] = False
        kwargs["is_training"] = caller_training
        expected_training &= caller_training

    result = _populate_population(case, kwargs, estimate)
    assert len(recording_detector) == 1
    calls = recording_detector[0].calls
    np.testing.assert_array_equal(calls[0]["is_training"], expected_training)
    assert not calls[0]["is_training"][case["invalid"]].any()
    assert np.isnan(calls[0]["position"][case["invalid"], 0]).all()
    assert (case["observed"] & ~case["encoding"]).any()
    np.testing.assert_array_equal(calls[0]["time"], time)
    for call in calls:
        for actual, expected in zip(
            call["spike_times"], case["expected_spikes"], strict=True
        ):
            np.testing.assert_array_equal(actual, expected)

    expected_missing = ~case["observed"] | explicit_missing
    if estimate:
        assert [call["method"] for call in calls] == ["estimate"]
        np.testing.assert_array_equal(calls[0]["is_missing"], expected_missing)
        np.testing.assert_array_equal(result.time, time)
        expected_labels = np.full(len(time), -1, dtype=int)
        expected_labels[: case["first_stop"]] = 0
        expected_labels[case["second_start"] : missing_index] = 1
        expected_labels[missing_index + 1 :] = 2
        np.testing.assert_array_equal(result.interval_labels, expected_labels)
    else:
        assert [call["method"] for call in calls] == [
            "fit",
            "predict",
            "predict",
        ]
        first_stop, second_start = case["first_stop"], case["second_start"]
        np.testing.assert_array_equal(calls[1]["time"], time[:first_stop])
        np.testing.assert_array_equal(calls[2]["time"], time[second_start:])
        np.testing.assert_array_equal(
            calls[1]["is_missing"], explicit_missing[:first_stop]
        )
        np.testing.assert_array_equal(
            calls[2]["is_missing"], explicit_missing[second_start:]
        )
        np.testing.assert_array_equal(result.time, time[case["observed"]])
        np.testing.assert_array_equal(
            result.is_missing, explicit_missing[case["observed"]]
        )
        np.testing.assert_array_equal(
            result.interval_labels,
            np.r_[
                np.zeros(first_stop, dtype=int),
                np.ones(len(time) - second_start, dtype=int),
            ],
        )

    for attr, expected in (
        ("spyglass_observation_intervals", case["observation_intervals"]),
        ("spyglass_encoding_intervals", case["encoding_intervals"]),
        ("spyglass_decoding_intervals", case["observation_intervals"]),
    ):
        np.testing.assert_array_equal(json.loads(result.attrs[attr]), expected)
    assert (
        json.loads(result.attrs["spyglass_observation_unknown_sources"]) == []
    )
    stored_kwargs = (
        case["mod"].DecodingParameters & case["params_key"]
    ).fetch1("decoding_kwargs")
    np.testing.assert_array_equal(stored_kwargs["is_missing"], explicit_missing)
    if training_mode == "omitted":
        assert "is_training" not in stored_kwargs
    elif training_mode == "none":
        assert stored_kwargs["is_training"] is None
    else:
        np.testing.assert_array_equal(
            stored_kwargs["is_training"], caller_training
        )


@pytest.mark.parametrize(
    "reason", ["no_observation", "no_encoding_overlap", "all_false"]
)
def test_empty_training_rejects_populate_before_decoder_or_results(
    observed_population, recording_detector, reason
):
    from spyglass.spikesorting.v2._observed_time import ObservationAvailability

    case = observed_population
    mod = case["mod"]
    kwargs = {}
    if reason == "no_observation":
        case["availability"]["value"] = ObservationAvailability(
            np.empty((0, 2))
        )
    elif reason == "no_encoding_overlap":
        mod.IntervalList.update1(
            {
                "nwb_file_name": case["key"]["nwb_file_name"],
                "interval_list_name": case["key"]["encoding_interval"],
                "valid_times": np.array(
                    [
                        [
                            case["time"][case["first_stop"]],
                            case["time"][case["second_start"] - 1],
                        ]
                    ]
                ),
            }
        )
    else:
        kwargs["is_training"] = np.zeros(len(case["time"]), dtype=bool)
    saved = set(case["results_storage"]["results"])
    with pytest.raises(ValueError, match="No observed training time"):
        _populate_population(case, kwargs, False)
    assert recording_detector == []
    assert not (mod.SortedSpikesDecodingV1 & case["key"])
    assert set(case["results_storage"]["results"]) == saved


@pytest.mark.parametrize("estimate", [False, True])
def test_unknown_coverage_preserves_explicit_none_training(
    observed_population, recording_detector, estimate
):
    from spyglass.spikesorting.v2._observed_time import ObservationAvailability

    case = observed_population
    case["availability"]["value"] = ObservationAvailability(
        None, ("legacy-source",)
    )
    result = _populate_population(case, {"is_training": None}, estimate)
    calls = recording_detector[0].calls
    assert calls[0]["is_training"] is None
    assert [call["method"] for call in calls] == (
        ["estimate"] if estimate else ["fit", "predict"]
    )
    if estimate:
        np.testing.assert_array_equal(
            calls[0]["is_missing"], np.zeros(len(case["time"]), dtype=bool)
        )
        np.testing.assert_array_equal(
            result.interval_labels, np.zeros(len(case["time"]), dtype=int)
        )
    else:
        assert calls[1]["is_missing"] is None
    for call in calls:
        for actual, expected in zip(
            call["spike_times"], case["spike_times"], strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(result.time, case["time"])
    assert json.loads(result.attrs["spyglass_observation_intervals"]) is None
    assert json.loads(result.attrs["spyglass_observation_unknown_sources"]) == [
        "legacy-source"
    ]
    np.testing.assert_array_equal(
        json.loads(result.attrs["spyglass_encoding_intervals"]),
        case["requested_encoding_intervals"],
    )
