import pytest
from numpy import allclose, array


@pytest.fixture(scope="session")
def dio_events(common):
    yield common.common_dio.DIOEvents


def test_pop_dioevents(dio_events, mini_beh_events):
    """Populate DIOEvents table."""
    dio_events.populate()
    events_raw = [e for e in mini_beh_events.fields["time_series"].keys()]
    events_fetch = [e for e in dio_events.fetch("dio_event_name")]
    assert set(events_fetch) == set(events_raw), "Mismatch in events populated."


@pytest.fixture(scope="session")
def dio_fig(mini_insert, dio_events, mini_restr):
    yield (dio_events & mini_restr).plot_all_dio_events(return_fig=True)


def test_plot_dio_axes(dio_fig, dio_events):
    """Check that all events are plotted."""
    events_fig = set(x.yaxis.get_label().get_text() for x in dio_fig.get_axes())
    events_fetch = set(dio_events.fetch("dio_event_name"))
    assert events_fig == events_fetch, "Mismatch in events plotted."


def test_plot_dio_data(common, dio_fig):
    """Hash summary of figure object."""
    data_fig = dio_fig.get_axes()[0].lines[0].get_xdata()
    data_block = (
        common.IntervalList & 'interval_list_name LIKE "raw%"'
    ).fetch1("valid_times")
    data_fetch = array((data_block[0][0], data_block[-1][1]))
    assert allclose(
        data_fig, data_fetch, atol=1e-8
    ), "Mismatch in data plotted."
