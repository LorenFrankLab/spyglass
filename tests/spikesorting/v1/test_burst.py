def test_burst_params(spike_v1, burst_params_key):
    params = (spike_v1.BurstPairParams() & burst_params_key).fetch1("params")
    assert (
        params["correl_window_ms"] == 100.0
        and params["correl_method"] == "numba"
    ), "Incorrect default burst parameters"


def test_burst_pairs(spike_v1, pop_burst):
    assert len(pop_burst) > 0
