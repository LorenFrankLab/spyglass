import numpy as np
import pytest
from numpy import array_equal


@pytest.fixture(scope="session")
def interval_list(common):
    yield common.IntervalList()


def test_plot_intervals(interval_list, mini_dict, start_time=0):
    interval_query = interval_list & mini_dict
    fig = (interval_query).plot_intervals(
        return_fig=True, start_time=start_time
    )

    ax = fig.axes[0]

    # extract interval list names from the figure
    fig_interval_list_names = np.array(
        [label.get_text() for label in ax.get_yticklabels()]
    )

    # extract interval list names from the IntervalList table
    fetch_interval_list_names = np.array(
        interval_query.fetch("interval_list_name")
    )

    # check that the interval list names match between the two methods
    assert array_equal(
        fig_interval_list_names, fetch_interval_list_names
    ), "plot_intervals failed: plotted interval list names do not match"

    # extract the interval times from the figure
    intervals_fig = []
    for collection in ax.collections:
        collection_data = []
        for patch in collection.get_paths():
            # Extract patch vertices to get x data
            vertices = patch.vertices
            x_start = vertices[0, 0]
            x_end = vertices[2, 0]
            collection_data.append(
                [x_start * 60 + start_time, x_end * 60 + start_time]
            )
        intervals_fig.append(np.array(collection_data))
    intervals_fig = np.array(intervals_fig, dtype="object")

    # extract interval times from the IntervalList table
    intervals_fetch = interval_list.fetch("valid_times")

    all_equal = True
    for i_fig, i_fetch in zip(intervals_fig, intervals_fetch):
        # permit rounding errors up to the 4th decimal place, as a result of inaccuracies during unit conversions
        i_fig = np.round(i_fig.astype("float"), 4)
        i_fetch = np.round(i_fetch.astype("float"), 4)
        if not array_equal(i_fig, i_fetch):
            all_equal = False

    # check that the interval list times match between the two methods
    assert (
        all_equal
    ), "plot_intervals failed: plotted interval times do not match"


def test_plot_epoch(mini_insert, interval_list):
    restr_interval = interval_list & "interval_list_name like 'raw%'"
    fig = restr_interval.plot_epoch_pos_raw_intervals(return_fig=True)
    epoch_label = fig.get_axes()[0].get_yticklabels()[-1].get_text()
    assert epoch_label == "epoch", "plot_epoch failed"

    epoch_interval = fig.get_axes()[0].lines[0].get_ydata()
    assert array_equal(epoch_interval, [1, 1]), "plot_epoch failed"
