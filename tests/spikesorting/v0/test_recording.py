def test_sort_group(pop_sort_group, mini_dict):
    fetched = sum(
        (pop_sort_group.SortGroupElectrode & mini_dict).fetch("electrode_id")
    )
    expected = sum(range(1, 128))  # 1 to 127 inclusive
    assert fetched == expected, "Failed to insert into v0.SortGroupElectrode"


def test_sort_interval(pop_sort_interval, mini_dict):
    tbl = pop_sort_interval & mini_dict
    assert len(tbl) == 1, "Failed to insert into v0.SortInterval"


def test_pop_rec_params(pop_rec_params):
    tbl, name, params = pop_rec_params
    fetched = (tbl & dict(preproc_params_name=name)).fetch1("preproc_params")
    assert fetched == params, "Failed to insert preproc_params"


def test_pop_rec(pop_rec):
    tbl = pop_rec
    assert tbl
    assert False


def test_recompute_env(spike_v0, pop_rec):
    """Test recompute to temp_dir"""
    from spyglass.spikesorting.v0 import spikesorting_recompute as recompute

    key = pop_rec.fetch("KEY", as_dict=True)[0]

    recompute.RecordingRecompute().populate(key)

    ret = (recompute.RecordingRecompute() & key).fetch1("matched")
    assert ret, "Recompute failed"
