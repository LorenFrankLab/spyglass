"""Observed time constrains actual decoder calls, including explicit masks."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.mark.parametrize("estimate", [False, True])
def test_decoder_branches_preserve_missing_time(
    decode_v1, monkeypatch, estimate
):
    from non_local_detector.models import base

    calls = []

    class Detector:
        def __init__(self, **kwargs):
            self.initial_conditions_ = np.array([1.0])
            self.discrete_state_transitions_ = np.array([[1.0]])

        def fit(self, *, position_time, position, spike_times, is_training):
            calls.append(("fit", position_time, is_training))

        def predict(
            self,
            *,
            position_time,
            position,
            spike_times,
            time,
            is_missing=None,
        ):
            calls.append(("predict", time, is_missing))
            return xr.Dataset(
                coords={"time": time, "states": ["local"], "state_bins": [0]}
            )

        def estimate_parameters(
            self,
            *,
            position_time,
            position,
            spike_times,
            time,
            is_training,
            is_missing,
        ):
            calls.append(("fit", position_time, is_training))
            return self.predict(
                position_time=position_time,
                position=position,
                spike_times=spike_times,
                time=time,
                is_missing=is_missing,
            )

    monkeypatch.setattr(base, "SortedSpikesDetector", Detector)
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
    np.testing.assert_array_equal(calls[0][2], training)
    if estimate:
        missing = ~training | explicit_missing
        np.testing.assert_array_equal(calls[1][2], missing)
        np.testing.assert_array_equal(
            result.interval_labels.values == -1, missing
        )
    else:
        np.testing.assert_array_equal(result.time, [0, 1, 2, 7, 8, 9])
        np.testing.assert_array_equal(calls[1][2], [False, False, False])
        np.testing.assert_array_equal(calls[2][2], [False, True, False])
        np.testing.assert_array_equal(
            result.is_missing, [False, False, False, False, True, False]
        )


@pytest.mark.parametrize("estimate", [False, True])
@pytest.mark.parametrize("no_overlap", [False, True])
def test_make_applies_observation_to_explicit_masks(
    decode_v1,
    monkeypatch,
    mock_sorted_spikes_decoder,
    mock_decoder_save,
    decode_sel_key,
    group_name,
    decode_spike_params_insert,
    pop_pos_group,
    pop_spikes_group,
    estimate,
    no_overlap,
):
    from spyglass.spikesorting.v2._observed_time import ObservationAvailability

    mod = decode_v1.sorted_spikes
    key = {
        **decode_sel_key,
        **decode_spike_params_insert,
        **pop_spikes_group,
        "estimate_decoding_params": estimate,
    }
    table = mod.SortedSpikesDecodingV1
    position, _ = table.fetch_position_info(key)
    time = position.index.to_numpy()
    first, last = float(time[len(time) // 3]), float(time[2 * len(time) // 3])
    observation = ObservationAvailability(
        np.empty((0, 2))
        if no_overlap
        else np.array(
            [[time[0], first], [last, np.nextafter(time[-1], np.inf)]]
        )
    )
    monkeypatch.setattr(
        mod.SortedSpikesGroup,
        "get_observation_intervals",
        lambda key, **kwargs: observation,
    )
    # Stored caller masks must never make excluded time available.
    relation = mod.DecodingParameters & decode_spike_params_insert
    original = {
        **decode_spike_params_insert,
        "decoding_kwargs": relation.fetch1("decoding_kwargs"),
    }
    updated = {
        **original,
        "decoding_kwargs": {
            **(original["decoding_kwargs"] or {}),
            "is_training": np.ones(len(time), dtype=bool),
            "is_missing": np.zeros(len(time), dtype=bool),
        },
    }
    mod.DecodingParameters.update1(updated)
    captured = {}

    def run(self, **kwargs):
        captured.update(kwargs)
        return mock_sorted_spikes_decoder(self, **kwargs)

    def save(self, classifier, results, key):
        assert "spyglass_observation_intervals" in results.attrs
        return mock_decoder_save(self, classifier, results, key)

    monkeypatch.setattr(table, "_run_decoder", run)
    monkeypatch.setattr(table, "_save_decoder_results", save)
    try:
        mod.SortedSpikesDecodingSelection.insert1(key, skip_duplicates=True)
        (table & key).super_delete(
            warn=False, safemode=False, force_masters=True
        )
        if no_overlap:
            with pytest.raises(ValueError, match="No observed training time"):
                table.populate(key)
            assert not captured
            return
        table.populate(key)
        valid = observation.contains(time)
        assert all(
            observation.contains(train).all()
            for train in captured["spike_times"]
        )
        assert not captured["decoding_kwargs"]["is_training"][~valid].any()
        assert captured["decoding_kwargs"]["is_missing"][~valid].all()
        for start, stop in captured["decoding_interval"]:
            assert not (start < last and stop >= first)
    finally:
        (table & key).super_delete(
            warn=False, safemode=False, force_masters=True
        )
        mod.DecodingParameters.update1(original)
