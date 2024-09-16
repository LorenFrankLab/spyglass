def test_sort_group(spike_v1, pop_rec):
    max_id = max(spike_v1.SortGroup.fetch("sort_group_id"))
    assert (
        max_id == 31
    ), "SortGroup.insert_sort_group failed to insert all records"


def test_spike_sorting(spike_v1, pop_rec):
    n_records = len(spike_v1.SpikeSortingRecording())
    assert n_records == 1, "SpikeSortingRecording failed to insert a record"
