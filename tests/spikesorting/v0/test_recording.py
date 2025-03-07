import shutil
import numpy as np
from pathlib import Path


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


def test_pop_rec(pop_rec):
    tbl = pop_rec
    assert tbl, "Failed to insert into v0.Recording"


def test_recompute(pop_rec):
    key = pop_rec.fetch(as_dict=True)[0]
    path = key["recording_path"]
    pre_rec = pop_rec.load_recording(key)

    # delete the file to force recompute
    shutil.rmtree(path, ignore_errors=True)

    post_rec = pop_rec.load_recording(key)

    assert Path(path).exists(), "Recompute failed"

    pre_loc = pre_rec.get_channel_locations()
    post_loc = post_rec.get_channel_locations()
    assert np.array_equal(
        pre_loc, post_loc
    ), "Recompute failed to preserve channel locations"


def test_recompute_env(spike_v0, pop_rec):
    """Test recompute to temp_dir"""
    from spyglass.spikesorting.v0 import spikesorting_recompute as recompute

    select = recompute.RecordingRecomputeSelection()

    key = select.fetch("KEY")[0]
    key["logged_at_creation"] = False  # Prevent skip of recompute
    select.update1(key)

    recompute.RecordingRecompute().populate(key)

    ret = (recompute.RecordingRecompute() & key).fetch1("matched")
    assert ret, "Recompute failed"
