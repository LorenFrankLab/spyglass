import shutil
from pathlib import Path

import datajoint as dj
import numpy as np
import pytest

from tests.conftest import VERBOSE


@pytest.fixture(scope="module")
def recomp_module(pop_rec_v0):
    """Fixture to ensure recompute module is loaded."""
    from spyglass.spikesorting.v0 import spikesorting_recompute as recompute

    _ = pop_rec_v0

    return recompute


def test_recompute(pop_rec_v0):
    key = pop_rec_v0.fetch(as_dict=True)[0]
    path = key["recording_path"]
    pre_rec = pop_rec_v0.load_recording(key)

    # delete the file to force recompute
    shutil.rmtree(path, ignore_errors=True)

    post_rec = pop_rec_v0.load_recording(key)

    assert Path(path).exists(), "Recompute failed"

    pre_loc = pre_rec.get_channel_locations()
    post_loc = post_rec.get_channel_locations()
    assert np.array_equal(
        pre_loc, post_loc
    ), "Recompute failed to preserve channel locations"


@pytest.fixture(scope="module")
def recomp_selection(recomp_module):
    """Fixture to ensure recompute selection is loaded."""

    yield recomp_module.RecordingRecomputeSelection()


@pytest.fixture(scope="module")
def recomp_tbl(recomp_module):
    """Fixture to ensure recompute table is loaded."""
    yield recomp_module.RecordingRecompute()


@pytest.fixture(scope="module")
def recomp_repop(pop_rec_v0, recomp_selection, recomp_tbl):
    """Fixture to ensure recompute repopulation is loaded."""
    _ = pop_rec_v0  # Ensure pop_rec is used to load the recording

    _ = recomp_selection.attempt_all()
    key = recomp_selection.fetch("KEY")[0]
    key["logged_at_creation"] = False  # Prevent skip of recompute
    recomp_selection.update1(key)

    recomp_tbl.populate(key)
    yield recomp_tbl


def test_recompute_env(recomp_repop):
    """Test recompute match"""

    ret = (recomp_repop & dj.Top()).fetch1("matched")
    assert ret, "Recompute failed"


def test_selection_restr(recomp_repop, user_env_tbl, recomp_selection):
    """Test that the selection env restriction works."""
    _ = recomp_repop  # Ensure recompute repop is used to load the recording
    env_dict = user_env_tbl.this_env
    manual_restr = recomp_selection & env_dict
    assert len(recomp_selection.this_env) == len(
        manual_restr
    ), "Recompute selection table property does not match env restriction"


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy")
def test_recompute_compare(caplog, recomp_repop, recomp_tbl):
    """Test recompute compare."""
    _ = recomp_repop
    _ = recomp_tbl.Hash().compare()
    assert caplog.text == "", "No output for matched recompute compare"


def test_get_disk_space(recomp_tbl):
    """Test get_disk_space."""
    space = recomp_tbl.get_disk_space(restr=True)
    assert "Total:" in space, "Disk space retrieval failed"


def test_recheck(recomp_tbl, recomp_repop):
    """Test recheck method."""
    _ = recomp_repop  # Ensure recompute populated
    key = recomp_tbl.fetch("KEY")[0]
    result = recomp_tbl.recheck(key)
    assert result, "Recheck failed"
