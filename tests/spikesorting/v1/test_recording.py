from pathlib import Path


def test_sort_group(spike_v1, pop_rec):
    max_id = max(spike_v1.SortGroup.fetch("sort_group_id"))
    assert (
        max_id == 31
    ), "SortGroup.insert_sort_group failed to insert all records"


def test_spike_sorting(spike_v1, pop_rec):
    n_records = len(spike_v1.SpikeSortingRecording())
    assert n_records == 1, "SpikeSortingRecording failed to insert a record"


def test_recompute(spike_v1, pop_rec, common):
    key = spike_v1.SpikeSortingRecording().fetch(
        "analysis_file_name", as_dict=True
    )[0]
    restr_tbl = spike_v1.SpikeSortingRecording() & key
    pre = restr_tbl.fetch_nwb()[0]["object_id"]

    file_path = common.AnalysisNwbfile.get_abs_path(key["analysis_file_name"])
    Path(file_path).unlink()  # delete the file to force recompute

    post = restr_tbl.fetch_nwb()[0]["object_id"]  # trigger recompute

    assert (
        pre.object_id == post.object_id
        and pre.electrodes.object_id == post.electrodes.object_id
    ), "Recompute failed to preserve object_ids"


def test_recompute_env(spike_v1, pop_rec):
    """Test recompute to temp_dir"""
    from spyglass.spikesorting.v1 import recompute

    select = recompute.RecordingRecomputeSelection()

    key = select.fetch("KEY")[0]
    key["logged_at_creation"] = False  # Prevent skip of recompute
    select.update1(key)

    recompute.RecordingRecompute().populate()

    ret = (recompute.RecordingRecompute() & key).fetch("matched")[0]
    assert ret, "Recompute failed"
