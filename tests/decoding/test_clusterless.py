import pandas as pd


def test_fetch_results(clusterless_pop, result_coordinates):
    results = clusterless_pop.fetch_results()
    assert (
        results.coords._names == result_coordinates
    ), "Incorrect coordinates in results"


def test_fetch_model(clusterless_pop):
    model = clusterless_pop.fetch_model()
    assert model is not None, "Model is None"
    __import__("pdb").set_trace()


def test_fetch_environments(clusterless_pop):
    envs = clusterless_pop.fetch_environments()
    assert envs is not None, "Environments is None"
    __import__("pdb").set_trace()


def test_fetch_linearized_position(clusterless_pop, clusterless_key):
    lin_pos = clusterless_pop.fetch_linear_position_info(clusterless_key)
    assert lin_pos is not None, "Linearized position is None"
    __import__("pdb").set_trace()
    assert False


def test_fetch_spike_by_interval(clusterless_pop, clusterless_key):
    spike_intervals = clusterless_pop.fetch_spike_intervals(clusterless_key)
    assert spike_intervals is not None, "Spike intervals is None"
    __import__("pdb").set_trace()
    assert False


def test_get_orientation_col(clusterless_pop):
    df = pd.DataFrame(columns=["orientation"])
    ret = clusterless_pop.get_orientation_col(df)
    assert ret == "orientation", "Orientation column not found"


def test_get_firing_rate(
    common, decode_interval, clusterless_pop, clusterless_key
):
    interval = (
        common.IntervalList & {"interval_list_name": decode_interval}
    ).fetch1("valid_times")
    rate = clusterless_pop.get_firing_rate(key=clusterless_key, time=interval)
    assert rate is not None, "Firing rate is None"
    __import__("pdb").set_trace()


def test_clusterless_estimated(clusterless_pop_estimated, result_coordinates):
    results = clusterless_pop_estimated.fetch_results()
    assert (
        results.coords._names == result_coordinates
    ), "Incorrect coordinates in estimated"


def test_get_ahead(clusterless_pop):
    dist = clusterless_pop.get_ahead_behind_distance()
    assert dist is not None, "Distance is None"
    __import__("pdb").set_trace()


def test_insert_existing_group(caplog, group_unitwave):
    file, group = group_unitwave.fetch(
        "nwb_file_name", "waveform_features_group_name"
    )[0]
    group_unitwave.insert_existing_group(file, group, ["dummy_data"])
    assert "already exists" in caplog.text, "No warning issued."
