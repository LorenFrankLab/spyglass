from pathlib import Path

import datajoint as dj
import pytest

from tests.conftest import VERBOSE


@pytest.mark.slow
def test_recompute(spike_v1, pop_rec, common, utils):
    key = spike_v1.SpikeSortingRecording().fetch(
        "analysis_file_name", as_dict=True
    )[0]
    restr_tbl = spike_v1.SpikeSortingRecording() & key
    pre = restr_tbl.fetch_nwb()[0]["object_id"]

    file_path = common.AnalysisNwbfile.get_abs_path(key["analysis_file_name"])
    Path(file_path).unlink()  # delete the file to force recompute
    utils.nwb_helper_fn.close_nwb_files()  # clear the io cache to trigger re-compute

    post = restr_tbl.fetch_nwb()[0]["object_id"]  # trigger recompute

    assert (
        pre.object_id == post.object_id
        and pre.electrodes.object_id == post.electrodes.object_id
    ), "Recompute failed to preserve object_ids"


@pytest.fixture(scope="module")
def recomp_module():
    """Fixture to ensure recompute module is loaded."""
    from spyglass.spikesorting.v1 import recompute

    return recompute


@pytest.fixture(scope="module")
def recomp_selection(recomp_module):
    """Fixture to ensure recompute selection is loaded."""
    yield recomp_module.RecordingRecomputeSelection()


@pytest.fixture(scope="module")
def recomp_tbl(recomp_module, recomp_selection, spike_v1, pop_rec):
    """Fixture to ensure recompute table is loaded."""
    _ = spike_v1, pop_rec  # Ensure pop_rec is used to load the recording

    key = recomp_selection.fetch("KEY")[0]
    key["logged_at_creation"] = False  # Prevent skip of recompute
    recomp_selection.update1(key)

    recomp_tbl = recomp_module.RecordingRecompute()
    recomp_tbl.populate()

    yield recomp_tbl


@pytest.mark.slow
def test_recompute_env(recomp_tbl):
    """Test recompute to temp_dir"""

    ret = (recomp_tbl & dj.Top()).fetch("matched")[0]
    assert ret, "Recompute failed"


def test_selection_attempt(caplog, recomp_selection):
    """Test that the selection reattempt does not add new entries."""
    _ = recomp_selection.attempt_all()
    prev_len = len(recomp_selection)
    ret = recomp_selection.attempt_all()
    post_len = len(recomp_selection)
    assert ret is None, "Selection attempt failed"
    assert prev_len == post_len, "Selection attempt should not add new entries"


def test_delete_dry_run(recomp_tbl):
    """Test dry run delete."""
    prev_len = len(recomp_tbl)
    _ = recomp_tbl.delete_files(dry_run=True)
    post_len = len(recomp_tbl)
    assert prev_len == post_len, "Dry run delete should not remove entries"


def test_recompute_disk_check(recomp_tbl):
    """Test that the disk check works."""
    from spyglass.utils.dj_helper_fn import bytes_to_human_readable

    key = recomp_tbl.fetch("KEY")[0]
    path, _ = recomp_tbl._get_paths(key)
    size = Path(path).stat().st_size if Path(path).exists() else 0
    expected = bytes_to_human_readable(size)
    result = recomp_tbl.get_disk_space(which="old", restr=key)
    assert expected in result, "Disk check failed"


@pytest.mark.slow
def test_recheck(recomp_tbl):
    """Test that recheck works."""
    key = recomp_tbl.fetch("KEY")[0]
    result = recomp_tbl.recheck(key)
    assert result, "Recheck failed"
