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


def test_calc_filter_delay(filter_parameters, filter_coeff):
    delay = filter_parameters.calc_filter_delay(filter_coeff)
    assert delay == 27, "filter_delay failed"


def test_create_standard_filters(filter_parameters):
    filter_parameters.create_standard_filters()
    assert filter_parameters & {
        "filter_name": "LFP 0-400 Hz"
    }, "create_standard_filters failed"
