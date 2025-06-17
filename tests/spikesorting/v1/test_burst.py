import numpy as np
import pytest


def test_burst_params(spike_v1, burst_params_key):
    params1 = (spike_v1.BurstPairParams() & burst_params_key).fetch1("params")
    assert (
        params1["correl_window_ms"] == 100.0
        and params1["correl_method"] == "numba"
    ), "Incorrect default burst parameters"

    params2 = spike_v1.BurstPairParams().get_params(burst_params_key)
    assert params1 == params2, "get_params() method not working correctly"


def test_burst_pairs(spike_v1, pop_burst):
    assert len(pop_burst) > 0


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
    fig = iter(tbl.plot_by_sort_group_ids(key, return_fig=True).values())
    assert axis_label_match(next(fig).get_axes()[0])


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


@pytest.fixture(scope="session")
def validate_pairs(spike_v1):
    from spyglass.spikesorting.utils_burst import validate_pairs
    from spyglass.spikesorting.v1.burst_curation import BurstPair

    yield validate_pairs, BurstPair().BurstPairUnit()


def test_validate_pairs_error(validate_pairs):
    validate_pairs, unit_tbl = validate_pairs
    with pytest.raises(ValueError):  # Invalid pair IDs
        validate_pairs(unit_tbl, pairs=(98, 99))  # also tests tuple type


@pytest.fixture(scope="session")
def calc_ca():
    from spyglass.spikesorting.utils_burst import calculate_ca

    yield calculate_ca


def test_calc_ca_error(calc_ca):
    with pytest.raises(ValueError):  # mismatch in array lengths
        calc_ca(np.array([1, 2]), np.array([3, 4, 5]))


@pytest.mark.parametrize(
    "bins, correl, expected_result",
    [
        (
            np.array([-10, -5, 0, 5, 10]),
            np.array([0.1, 0.3, 0.0, 0.4, 0.2]),
            0.2,
        ),
        # Test for zero case (both left and right sums are 0)
        (np.array([-10, -5, 0, 5, 10]), np.array([0.0, 0.0, 0.0, 0.0, 0.0]), 0),
        # Test for right side only (bins > 0)
        (np.array([1, 2, 3, 4]), np.array([0.1, 0.2, 0.3, 0.4]), 1.0),
        # Test for left side only (bins < 0)
        (np.array([-1, -2, -3, -4]), np.array([0.1, 0.2, 0.3, 0.4]), -1.0),
        # Test for empty arrays (edge case)
        (np.array([]), np.array([]), 0),
    ],
)
def test_calculate_ca(calc_ca, bins, correl, expected_result):
    # Calculate the CA using the function
    result = calc_ca(bins, correl)
    assert np.isclose(
        result, expected_result
    ), f"Expected {expected_result}, got {result}"


@pytest.fixture(scope="session")
def calc_isi():
    from spyglass.spikesorting.utils_burst import calculate_isi_violation

    yield calculate_isi_violation


@pytest.mark.parametrize(
    "peak1, peak2, isi_threshold_s, expected_result",
    [  # Normal Case - Some ISI violations. All ISIs are >= 1.5ms
        (np.array([1, 3, 5]), np.array([2, 4, 6]), 1.5e3, 5 / 6),
        # Case with Empty first
        (np.array([]), np.array([2, 4, 6]), 1.5e3, 0.0),
        # Case with Empty second
        (np.array([1, 2, 3]), np.array([]), 1.5e3, 4 / 6),
        # Case with Different Thresholds
        (np.array([1, 3, 5]), np.array([2, 4, 6]), 1.0e3, 0.0),
        # Spikes too close, below threshold
        (np.array([1, 3, 5]), np.array([2, 4, 6]), 0.5e3, 0.0),
    ],
)
def test_calculate_isi_violation(
    calc_isi, peak1, peak2, isi_threshold_s, expected_result
):
    result = calc_isi(peak1, peak2, isi_threshold_s)
    assert np.isclose(
        result, expected_result
    ), f"Expected {expected_result}, got {result}"
