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
    _ = spike_v1  # Ensure the recording is populated

    # Ensure a selection entry exists for this recording, regardless of what
    # prior runs left behind. attempt_all() restricts to `this_env`, so it
    # silently inserts nothing whenever the current environment does not
    # match the one that made the recording -- including when the env cannot
    # be resolved at all. force_attempt lifts that restriction.
    if not len(recomp_selection & pop_rec):
        recomp_selection.attempt_all(restr=pop_rec, force_attempt=True)

    keys = (recomp_selection & pop_rec).fetch("KEY")
    assert keys, f"No recompute selection entries for {pop_rec}"
    key = keys[0]
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


@pytest.fixture
def spare_attempt(recomp_module, recomp_selection, pop_rec):
    """Yield a disposable recompute entry, its table and attempt file.

    Delete tests run on this extra 'rounding' variant of the recording, so
    that they neither depend on nor remove the entry other tests use.
    """
    recomp_module.RecordingRecomputeVersions().populate(pop_rec)

    restr = dict(pop_rec, rounding=99)
    recomp_selection.insert1(
        restr, force_attempt=True, skip_xfail=False, skip_duplicates=True
    )
    assert recomp_selection & restr, f"Could not add selection entry {restr}"

    tbl = recomp_module.RecordingRecompute()
    key = (recomp_selection & restr).fetch1("KEY")
    tbl.insert1(
        dict(key, matched=False, err_msg="spare attempt"),
        allow_direct_insert=True,
    )

    path = tbl._get_paths(key)[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

    yield tbl, key, path

    path.unlink(missing_ok=True)
    (tbl & key).delete_quick()
    (recomp_selection & key).delete_quick()


def test_delete_no_prompt(spare_attempt, monkeypatch):
    """Test that safemode=False removes rows and files without prompting."""
    import datajoint.table

    tbl, key, path = spare_attempt

    def no_prompt(*args, **kwargs):  # pragma: no cover
        pytest.fail("safemode=False prompted for user input")

    # recompute calls dj.utils.user_choice, Table.delete its own import
    monkeypatch.setattr("datajoint.utils.user_choice", no_prompt)
    monkeypatch.setattr(datajoint.table, "user_choice", no_prompt)
    monkeypatch.setitem(dj.config, "safemode", True)

    (tbl & key).delete(safemode=False)

    assert not (tbl & key), "Delete did not remove the entry"
    assert not path.exists(), "Delete did not remove the attempt file"


def test_delete_cancelled(spare_attempt, monkeypatch):
    """Test that answering 'no' keeps both the attempt file and the row."""
    tbl, key, path = spare_attempt

    monkeypatch.setattr("datajoint.utils.user_choice", lambda *a, **kw: "no")
    monkeypatch.setitem(dj.config, "safemode", True)

    (tbl & key).delete(safemode=True)

    assert tbl & key, "Cancelled delete removed the entry"
    assert path.exists(), "Cancelled delete removed the attempt file"
