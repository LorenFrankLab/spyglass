import numpy as np
import pandas as pd
import pytest


def test_fetch_results(clusterless_pop, result_coordinates):
    results = clusterless_pop.fetch_results()
    assert result_coordinates.issubset(
        results.coords._names
    ), "Incorrect coordinates in results"


@pytest.mark.skip(reason="JAX issues")
def test_fetch_model(clusterless_pop):
    from non_local_detector.models.base import ClusterlessDetector

    assert isinstance(
        clusterless_pop.fetch_model(), ClusterlessDetector
    ), "Model is not ClusterlessDetector"


def test_fetch_environments(clusterless_pop, clusterless_key):
    from non_local_detector.environment import Environment

    env = clusterless_pop.fetch_environments(clusterless_key)[0]
    assert isinstance(env, Environment), "Fetched obj not Environment type"


@pytest.mark.skip(reason="Need track graph")
def test_fetch_linearized_position(clusterless_pop, clusterless_key):
    lin_pos = clusterless_pop.fetch_linear_position_info(clusterless_key)
    assert lin_pos is not None, "Linearized position is None"


# NOTE: Impacts spikesorting merge tests
def test_fetch_spike_by_interval(decode_v1, clusterless_pop, clusterless_key):
    begin, end = decode_v1.clusterless._get_interval_range(clusterless_key)
    spikes = clusterless_pop.fetch_spike_data(
        clusterless_key, filter_by_interval=True
    )[0][0]
    assert np.all((spikes >= begin) & (spikes <= end)), "Spikes not in interval"


def test_get_orientation_col(clusterless_pop):
    df = pd.DataFrame(columns=["orientation"])
    ret = clusterless_pop.get_orientation_col(df)
    assert ret == "orientation", "Orientation column not found"


def test_get_firing_rate(
    common, decode_interval, clusterless_pop, clusterless_key
):
    interval = (
        common.IntervalList & {"interval_list_name": decode_interval}
    ).fetch1("valid_times")[0]
    rate = clusterless_pop.get_firing_rate(key=clusterless_key, time=interval)
    assert rate.shape == (2, 1), "Incorrect firing rate shape"


def test_clusterless_estimated(clusterless_pop_estimated, result_coordinates):
    results = clusterless_pop_estimated.fetch_results()
    assert result_coordinates.issubset(
        results.coords._names
    ), "Incorrect coordinates in estimated"


@pytest.mark.skip(reason="Need track graph")
def test_get_ahead(clusterless_pop):
    dist = clusterless_pop.get_ahead_behind_distance()
    assert dist is not None, "Distance is None"


def test_insert_existing_group(caplog, group_unitwave):
    file, group = group_unitwave.fetch1(
        "nwb_file_name", "waveform_features_group_name"
    )
    group_unitwave.create_group(file, group, ["dummy_data"])
    assert "already exists" in caplog.text, "No warning issued."
