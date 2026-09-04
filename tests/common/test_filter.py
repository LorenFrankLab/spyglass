"""Tests for `FirFilterParameters` (`spyglass.common.common_filter`).

These are DB-backed and use the session fixtures in tests/conftest.py.

Coverage gap worth knowing about: `filter_data_nwb` -- the method the LFP
pipeline actually calls -- has no direct test here (`test_filter_data` is
skipped), only indirect coverage via `LFPV1().populate()` in tests/lfp/. It
carries its own copies of the reversed-interval and all-empty-interval
guards tested below against `filter_data`; duplicated guards can drift.
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def filter_parameters(common):
    yield common.FirFilterParameters()


@pytest.fixture(scope="session")
def filter_dict(filter_parameters):
    yield {"filter_name": "test", "fs": 10}


@pytest.fixture(scope="session")
def add_filter(filter_parameters, filter_dict):
    filter_parameters.add_filter(
        **filter_dict, filter_type="lowpass", band_edges=[1, 2]
    )


@pytest.fixture(scope="session")
def filter_coeff(filter_parameters, filter_dict):
    yield filter_parameters._filter_restrict(**filter_dict)["filter_coeff"]


def test_add_filter(filter_parameters, add_filter, filter_dict):
    """Test add filter"""
    assert filter_parameters & filter_dict, "add_filter failed"


def test_filter_restrict(
    filter_parameters, add_filter, filter_dict, filter_coeff
):
    assert sum(filter_coeff) == pytest.approx(
        0.999134, abs=1e-6
    ), "filter_restrict failed"


def test_plot_magitude(filter_parameters, add_filter, filter_dict):
    fig = filter_parameters.plot_magnitude(**filter_dict, return_fig=True)
    assert sum(fig.get_axes()[0].lines[0].get_xdata()) == pytest.approx(
        163837.5, abs=1
    ), "plot_magnitude failed"


def test_plot_fir_filter(
    filter_parameters, add_filter, filter_dict, filter_coeff
):
    fig = filter_parameters.plot_fir_filter(**filter_dict, return_fig=True)
    assert sum(fig.get_axes()[0].lines[0].get_ydata()) == sum(
        filter_coeff
    ), "Plot filter failed"


def test_filter_delay(filter_parameters, add_filter, filter_dict):
    delay = filter_parameters.filter_delay(**filter_dict)
    assert delay == 27, "filter_delay failed"


def test_time_bound_warning(filter_parameters, add_filter, filter_dict):
    with pytest.warns(UserWarning):
        filter_parameters._time_bound_check(1, 3, [2, 5], 4)


@pytest.mark.skip(reason="Not testing V0: filter_data")
def test_filter_data(filter_parameters, mini_content):
    pass


def test_filter_data_rejects_all_empty_intervals(
    filter_parameters, add_filter, filter_coeff
):
    """Degenerate valid_times must fail loudly, not build an empty output.

    Every interval here clips to zero samples. Without an explicit check the
    empty interval list only surfaces later as an opaque unpack error.

    This covers `filter_data` only. `filter_data_nwb` carries its own copy of
    the same guard, where the consequence is worse (a zero-length
    ElectricalSeries is written to the analysis file before the failure), but
    exercising it needs an NWB fixture -- see the module docstring.
    """
    timestamps = np.arange(100, dtype=float)
    data = np.zeros((100, 2))
    degenerate = np.array([[10.0, 10.0], [50.0, 50.0]])
    with pytest.raises(ValueError, match="No samples to filter"):
        filter_parameters.filter_data(
            timestamps, data, filter_coeff, degenerate, [0, 1], 1
        )


def test_filter_data_rejects_reversed_interval(
    filter_parameters, add_filter, filter_coeff
):
    """A reversed interval is malformed input, not an empty one.

    Skipping zero-sample intervals must not also swallow `[stop, start]`, which
    would silently drop data instead of telling the caller their valid_times are
    wrong. As above, `filter_data_nwb` has a duplicate of this guard that is not
    covered here.
    """
    timestamps = np.arange(100, dtype=float)
    data = np.zeros((100, 2))
    reversed_and_valid = np.array([[10.0, 5.0], [20.0, 25.0]])
    with pytest.raises(ValueError, match="Reversed interval"):
        filter_parameters.filter_data(
            timestamps, data, filter_coeff, reversed_and_valid, [0, 1], 1
        )


def test_filter_data_timestamps_align_with_data(
    filter_parameters, add_filter, filter_coeff
):
    """Decimated timestamps must index the same samples as the decimated data.

    `filter_data` slices `timestamps[start:stop:decimation]` while the filter
    decimates its OUTPUT starting at `first_ind = filter_delay`. Those two have
    to land on the same input samples. The FIR suite cannot catch a drift here:
    its reference decimates with the same expression as the implementation, so
    phase is only ever checked inside the engine, never against the timestamp
    vector. A regression would shift LFP in time by up to `decimation` raw
    samples, silently.

    An impulse is the sharpest probe: the symmetric filter's peak response must
    come back at the impulse's own timestamp.
    """
    n_time, decimation, impulse_at = 400, 5, 200
    timestamps = np.arange(n_time, dtype=float) / 100.0
    data = np.zeros((n_time, 1))
    data[impulse_at, 0] = 1.0

    filtered, new_timestamps = filter_data_helper(
        filter_parameters, timestamps, data, filter_coeff, decimation
    )

    assert len(new_timestamps) == filtered.shape[0]
    peak_time = new_timestamps[np.argmax(np.abs(filtered[:, 0]))]
    # `impulse_at` is a multiple of `decimation`, so the impulse sample survives
    # decimation and the peak must land on it EXACTLY. A tolerance of one
    # decimated step would accept the very off-by-one-bin drift this test
    # exists to catch.
    assert impulse_at % decimation == 0
    assert peak_time == pytest.approx(timestamps[impulse_at])


def filter_data_helper(filter_parameters, timestamps, data, coeff, decimation):
    """Filter one all-valid interval, returning (filtered_data, timestamps)."""
    valid_times = np.array([[timestamps[0], timestamps[-1]]])
    return filter_parameters.filter_data(
        timestamps, data, coeff, valid_times, [0], decimation
    )


def test_calc_filter_delay(filter_parameters, filter_coeff):
    delay = filter_parameters.calc_filter_delay(filter_coeff)
    assert delay == 27, "filter_delay failed"


def test_create_standard_filters(filter_parameters):
    filter_parameters.create_standard_filters()
    assert filter_parameters & {
        "filter_name": "LFP 0-400 Hz"
    }, "create_standard_filters failed"
