import pytest
from numpy import array_equal


@pytest.fixture(scope="session")
def interval_list(common):
    yield common.IntervalList()


def test_plot_intervals(mini_insert, interval_list):
    fig = interval_list.plot_intervals(return_fig=True)
    interval_list_name = fig.get_axes()[0].get_yticklabels()[0].get_text()
    times_fetch = (
        interval_list & {"interval_list_name": interval_list_name}
    ).fetch1("valid_times")[0]
    times_plot = fig.get_axes()[0].lines[0].get_xdata()

    assert array_equal(times_fetch, times_plot), "plot_intervals failed"


def test_plot_epoch(mini_insert, interval_list):
    fig = interval_list.plot_epoch_pos_raw_intervals(return_fig=True)
    epoch_label = fig.get_axes()[0].get_yticklabels()[-1].get_text()
    assert epoch_label == "epoch", "plot_epoch failed"

    epoch_interval = fig.get_axes()[0].lines[0].get_ydata()
    assert array_equal(epoch_interval, [1, 1]), "plot_epoch failed"
