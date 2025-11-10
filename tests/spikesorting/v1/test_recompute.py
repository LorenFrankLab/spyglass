from pathlib import Path

import datajoint as dj
import pytest

from tests.conftest import VERBOSE


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


def test_recompute_env(recomp_tbl):
    """Test recompute to temp_dir"""

    ret = (recomp_tbl & dj.Top()).fetch("matched")[0]
    assert ret, "Recompute failed"


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy")
def test_selection_attempt(caplog, recomp_selection):
    """Test that the selection attempt works."""
    _ = recomp_selection.attempt_all()
    assert "No rows" in caplog.text, "Selection attempt failed null log"


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy")
def test_delete_dry_run(caplog, recomp_tbl):
    """Test dry run delete."""
    _ = recomp_tbl.delete_files(dry_run=True)
    assert "DRY" in caplog.text, "Dry run delete failed to log"


def test_recheck(recomp_tbl):
    """Test that recheck works."""
    key = recomp_tbl.fetch("KEY")[0]
    result = recomp_tbl.recheck(key)
    assert result, "Recheck failed"
