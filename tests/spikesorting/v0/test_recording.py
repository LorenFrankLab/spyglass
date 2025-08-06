import numpy as np


def test_sort_group(pop_sort_group, mini_dict):
    fetched = sum(
        (pop_sort_group.SortGroupElectrode & mini_dict).fetch("electrode_id")
    )
    expected = sum(range(1, 128))  # 1 to 127 inclusive
    assert fetched == expected, "Failed to insert into v0.SortGroupElectrode"


def test_sort_interval(pop_sort_interval, mini_dict):
    tbl = pop_sort_interval & mini_dict
    assert tbl, "Failed to insert into v0.SortInterval"


def test_pop_rec_params(pop_rec_params):
    tbl, name, params = pop_rec_params
    fetched = (tbl & dict(preproc_params_name=name)).fetch1("preproc_params")
    assert fetched == params, "Failed to insert preproc_params"


def test_pop_rec(pop_rec_v0):
    tbl = pop_rec_v0
    assert tbl, "Failed to insert into v0.Recording"
