import pytest


def test_burst_params(spike_v0, burst_params_key):
    params = (spike_v0.BurstPairParams() & burst_params_key).fetch1()
    assert (
        params["burst_params_name"] == "default"
        and params["params"]["sorter"] == "mountainsort4"
    ), "Default burst parameters not inserted correctly"


def test_burst_pairs(spike_v0, pop_burst):
    assert len(pop_burst), "Burst pairs not populated correctly"


@pytest.fixture(scope="session")
def burst_key(pop_burst):
    key = pop_burst.fetch("KEY", as_dict=True)[0]
    u1, u2 = pop_burst.BurstPairUnit().fetch("unit1", "unit2")
    return pop_burst, key, list(zip(u1, u2))


def axis_label_match(
    fig, xlabel="waveform similarity", ylabel="cross-correlogram asymmetry"
):
    x, y = fig.get_xlabel(), fig.get_ylabel()
    return x == xlabel and y == ylabel, "Figure axes changed"


def test_burst_plot_by_sort_group(burst_key):
    tbl, key, _ = burst_key
    fig = tbl.plot_by_sort_group_ids(key, return_fig=True)[0].get_axes()[0]
    assert axis_label_match(fig)


def test_xcorrel(burst_key):
    tbl, key, pairs = burst_key
    fig = tbl.investigate_pair_xcorrel(key, pairs, return_fig=True).get_axes()
    assert axis_label_match(fig[0], "ms", "")


def test_peaks(burst_key):
    tbl, key, pairs = burst_key
    fig = tbl.investigate_pair_peaks(key, pairs, return_fig=True).get_axes()[0]
    assert axis_label_match(fig, "uV", "percent")
    assert fig.get_title().startswith("channel")


def test_peak_over_time(burst_key):
    tbl, key, pairs = burst_key
    fig = tbl.plot_peak_over_time(key, pairs, return_fig=True).get_axes()[0]
    assert axis_label_match(fig)
