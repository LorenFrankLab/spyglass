import shutil
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import VERBOSE


@pytest.fixture(scope="module")
def recomp_module():
    """Fixture to ensure recompute module is loaded."""
    from spyglass.spikesorting.v0 import spikesorting_recompute as recompute

    return recompute


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


@pytest.fixture(scope="module")
def recomp_selection(recomp_module):
    """Fixture to ensure recompute selection is loaded."""

    yield recomp_module.RecordingRecomputeSelection()


@pytest.fixture(scope="module")
def recomp_tbl(recomp_module):
    """Fixture to ensure recompute table is loaded."""
    yield recomp_module.RecordingRecompute()


def test_selection_restr(user_env_tbl, recomp_selection):
    """Test that the selection env restriction works."""
    env_dict = user_env_tbl.this_env
    manual_restr = recomp_selection & env_dict
    assert (
        recomp_selection.this_env == manual_restr
    ), "Recompute selection table property does not match env restriction"


def test_recompute_env(pop_rec, recomp_selection, recomp_tbl):
    """Test recompute to temp_dir"""
    _ = pop_rec  # Ensure pop_rec is used to load the recording

    key = recomp_selection.fetch("KEY")[0]
    key["logged_at_creation"] = False  # Prevent skip of recompute
    recomp_selection.update1(key)

    recomp_tbl.populate(key)

    ret = (recomp_module.RecordingRecompute() & key).fetch1("matched")
    assert ret, "Recompute failed"


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy")
def test_recompute_compare(caplog, recomp_tbl):
    """Test recompute compare."""
    _ = recomp_tbl.Hash().compare()
    assert "???" in caplog.text, "Intentional fail"


def test_get_disk_space(recomp_tbl):
    """Test get_disk_space."""
    space = recomp_tbl.get_disk_space(restr=True)
    assert "Total:" in space, "Disk space retrieval failed"
