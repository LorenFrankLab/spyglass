"""Regression tests for decodes whose state coordinate became scalar."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.mark.parametrize(
    ("module_name", "table_name", "detector_name"),
    [
        ("sorted_spikes", "SortedSpikesDecodingV1", "SortedSpikesDetector"),
        ("clusterless", "ClusterlessDecodingV1", "ClusterlessDetector"),
    ],
)
def test_single_state_transition_coordinates(
    decode_v1, monkeypatch, module_name, table_name, detector_name
):
    """Both decoders must attach a 1x1 transition to a scalar state coord."""
    from non_local_detector.models import base

    class FakeDetector:
        initial_conditions_ = np.array([0.5, 0.5])
        discrete_state_transitions_ = np.array([[1.0]])

        def __init__(self, **kwargs):
            pass

        def estimate_parameters(self, **kwargs):
            return xr.Dataset(
                {
                    "acausal_posterior": (
                        ("time", "state_bins"),
                        np.full((2, 2), 0.5),
                    )
                },
                coords={
                    "time": [0.0, 1.0],
                    "state_bins": [0, 1],
                    "states": "Continuous",
                },
            )

    monkeypatch.setattr(base, detector_name, FakeDetector)
    table = getattr(getattr(decode_v1, module_name), table_name)
    kwargs = {
        "key": {"estimate_decoding_params": True},
        "decoding_params": {},
        "decoding_kwargs": {},
        "position_info": pd.DataFrame(
            {"position": [0.0, 1.0]}, index=[0.0, 1.0]
        ),
        "position_variable_names": ["position"],
        "spike_times": [],
        "decoding_interval": np.array([[0.0, 1.0]]),
    }
    if module_name == "clusterless":
        kwargs["spike_waveform_features"] = []

    _, results = table._run_decoder(None, **kwargs)
    transitions = results["discrete_state_transitions"]
    assert transitions.dims == ("states_from", "states_to")
    np.testing.assert_array_equal(transitions.values, [[1.0]])
    assert transitions.coords["states_from"].values.tolist() == ["Continuous"]
    assert transitions.coords["states_to"].values.tolist() == ["Continuous"]
