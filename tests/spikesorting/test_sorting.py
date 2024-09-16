def test_sorting(spike_v1, pop_sort):
    n_sorts = len(spike_v1.SpikeSorting & pop_sort)
    assert n_sorts >= 1, "SpikeSorting population failed"
