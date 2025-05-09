def test_burst_params(spike_v0, burst_params_key):
    params = (spike_v0.BurstPairParams() & burst_params_key).fetch1()
    assert (
        params["burst_params_name"] == "default"
        and params["params"]["sorter"] == "mountainsort4"
    ), "Default burst parameters not inserted correctly"


def test_burst_pairs(spike_v0, pop_burst):
    assert len(pop_burst), "Burst pairs not populated correctly"
