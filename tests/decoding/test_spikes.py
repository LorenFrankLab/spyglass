import numpy as np
import pandas as pd
import pytest


def test_spikes_decoding(spikes_decoding, result_coordinates):
    results = spikes_decoding.fetch_results()
    assert result_coordinates.issubset(
        results.coords._names
    ), "Incorrect coordinates in results"


@pytest.mark.skip(reason="JAX issues")
def test_fetch_model(spikes_decoding):
    from non_local_detector.models.base import SortedSpikesDetector

    assert isinstance(
        spikes_decoding.fetch_model(), SortedSpikesDetector
    ), "Model is not ClusterlessDetector"


@pytest.mark.skip(reason="JAX issues")
def test_fetch_environments(spikes_decoding, spikes_decoding_key):
    from non_local_detector.environment import Environment

    env = spikes_decoding.fetch_environments(spikes_decoding_key)[0]
    assert isinstance(env, Environment), "Fetched obj not Environment type"


@pytest.mark.skip(reason="Need track graph")
def test_fetch_linearized_position(spikes_decoding, spikes_decoding_key):
    lin_pos = spikes_decoding.fetch_linear_position_info(spikes_decoding_key)
    assert lin_pos is not None, "Linearized position is None"


@pytest.mark.skip(reason="JAX issues")
def test_fetch_spike_by_interval(
    decode_v1, spikes_decoding, spikes_decoding_key
):
    begin, end = decode_v1.clusterless._get_interval_range(spikes_decoding_key)
    spikes = spikes_decoding.fetch_spike_data(
        spikes_decoding_key, filter_by_interval=True
    )[0][0]
    assert np.all((spikes >= begin) & (spikes <= end)), "Spikes not in interval"


@pytest.mark.skip(reason="JAX issues")
def test_spikes_by_place(spikes_decoding, spikes_decoding_key):
    spikes = spikes_decoding.spike_times_sorted_by_place_field_peak()
    eg = next(iter(spikes.values()))[0]  # get first value
    assert eg.shape[0] > 0, "Spikes by place failed"


@pytest.mark.skip(reason="JAX issues")
def test_get_orientation_col(spikes_decoding):
    df = pd.DataFrame(columns=["orientation"])
    ret = spikes_decoding.get_orientation_col(df)
    assert ret == "orientation", "Orientation column not found"


@pytest.mark.skip(reason="JAX issues")
def test_spikes_decoding_estimated(
    spikes_decoding_estimated, result_coordinates
):
    assert result_coordinates.issubset(
        spikes_decoding_estimated.fetch_results().coords._names
    ), "Incorrect coordinates in estimated"
