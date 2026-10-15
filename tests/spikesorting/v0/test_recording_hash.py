"""Tests for `SpikeSortingRecording` hash storage and recompute verification.

The stored hash is the baseline a recompute is checked against, so these cover
all four states it can be in: written on insert, matching, mismatched, and the
legacy null left by rows inserted before the hash was persisted.
"""

import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="function")
def rec_tbl(spike_v0):
    """The `SpikeSortingRecording` table."""
    return spike_v0.SpikeSortingRecording()


@pytest.fixture(scope="function")
def rec_key(pop_rec_v0, rec_tbl):
    """One populated recording entry, as a full dict, with files on disk.

    `test_recompute` removes the recording directory while its memory maps are
    still open. On NFS that silly-renames each file to `.nfsXXXX` and leaves
    the directory in place, and `_make_file` treats any existing directory as
    complete, so the state never self-heals. Move it aside and rebuild rather
    than depend on the order these tests happen to run in.
    """
    key = pop_rec_v0.fetch(as_dict=True)[0]
    path = Path(key["recording_path"])

    def is_real(f):  # rglob matches the .nfsXXXX leftovers; they are not data
        return f.is_file() and not f.name.startswith(".")

    if path.exists() and any(is_real(f) for f in path.rglob("*")):
        return key

    stale = path.with_name(f"{path.name}.stale")
    shutil.rmtree(stale, ignore_errors=True)

    if path.exists():  # rename succeeds where unlink does not
        path.rename(stale)

    # Clear the stored hash first. The rebuild need not be byte-identical, and
    # a mismatch would delete the directory this fixture exists to restore.
    # The rebuild becomes the new baseline.
    pk = {k: key[k] for k in rec_tbl.primary_key}
    rec_tbl.update1({**pk, "hash": None})
    rec_info = rec_tbl._make_file(pk)
    rec_tbl.update1({**pk, "hash": rec_info["hash"]})

    shutil.rmtree(stale, ignore_errors=True)

    return pop_rec_v0.fetch(as_dict=True)[0]


@pytest.fixture(scope="function")
def rec_pk(rec_key, rec_tbl):
    """The primary key alone, as `_hash_check` is given it in production.

    Restricting on a full row would carry a `hash` value, and a stale one
    would match no row at all.
    """
    return {k: rec_key[k] for k in rec_tbl.primary_key}


@pytest.fixture(scope="function")
def rec_copy(rec_key, tmp_path):
    """A throwaway copy of the recording directory.

    `_hash_check` deletes the directory it is given on mismatch, so tests that
    exercise that branch must not hand it the real one.
    """
    dest = tmp_path / "rec_copy"
    shutil.copytree(rec_key["recording_path"], dest)
    return dest


def test_hash_stored_on_insert(rec_key):
    """The hash must be persisted, not merely computed and discarded.

    Without it there is no baseline, and every recompute takes the null branch.
    """
    assert rec_key["hash"] is not None, "No hash stored on insert"


def test_hash_check_accepts_match(rec_tbl, rec_key, rec_pk, rec_copy):
    """A recompute that reproduces the stored directory is accepted.

    The stored hash is set from the copy rather than trusted to match what is
    on disk. A recompute here need not be byte-identical, and this asserts the
    accept branch, not that spikeinterface rebuilds deterministically. The
    copy also keeps the real directory out of reach of the delete on mismatch.
    """
    rec_tbl.update1({**rec_pk, "hash": rec_tbl._dir_hash(rec_copy)})

    try:
        assert rec_tbl._hash_check(rec_pk, rec_copy), "Matching hash rejected"
    finally:
        rec_tbl.update1({**rec_pk, "hash": rec_key["hash"]})

    assert rec_copy.exists(), "Matching hash check deleted the recording"


def test_hash_check_rejects_mismatch(rec_tbl, rec_pk, rec_copy):
    """A recompute that differs raises, and takes the bad directory with it."""
    (rec_copy / "tampered.bin").write_bytes(b"not in the stored hash")

    with pytest.raises(ValueError, match="Hash mismatch"):
        rec_tbl._hash_check(rec_pk, rec_copy)

    assert not rec_copy.exists(), "Mismatched recording was left on disk"


def test_hash_check_accepts_legacy_null(
    caplog, rec_tbl, rec_key, rec_pk, rec_copy
):
    """A row predating hash storage has no baseline, so accept and warn.

    Deleting the recompute would destroy a good directory over a comparison
    that was never possible.
    """
    rec_tbl.update1({**rec_pk, "hash": None})

    try:
        assert rec_tbl._hash_check(rec_pk, rec_copy), "Legacy null rejected"
    finally:
        rec_tbl.update1({**rec_pk, "hash": rec_key["hash"]})

    assert rec_copy.exists(), "Legacy null hash check deleted the recording"
    assert "update_ids" in caplog.text, "No backfill hint logged"


def test_make_file_verifies_default_dir(monkeypatch, rec_tbl, rec_pk, tmp_path):
    """A rebuild of the canonical directory must reach `_hash_check`.

    The guard compares `base_dir` against `settings.recording_dir`, which is a
    `str`, while `_make_file` builds `base_dir` as a `Path`. The two are never
    equal, so every mismatched recompute was silently accepted.

    `recording_dir` is redirected here rather than the real one relocated: the
    session fixture holds memory-mapped traces open under it, and moving that
    directory out from under them breaks every later test.
    """
    monkeypatch.setattr(
        "spyglass.spikesorting.v0.spikesorting_recording.recording_dir",
        str(tmp_path),  # a str, as settings supplies it
    )

    checked = []

    class FakeRecording:
        """Stand-in for the filtered recording, so no rebuild is needed."""

        def save(self, folder, **kwargs):
            Path(folder).mkdir(parents=True)
            (Path(folder) / "fake.json").write_text("{}")

    monkeypatch.setattr(
        rec_tbl,
        "_hash_check",
        lambda key, rec_path: checked.append(Path(rec_path)) or True,
    )
    monkeypatch.setattr(
        rec_tbl, "_get_filtered_recording", lambda key: FakeRecording()
    )

    rec_info = rec_tbl._make_file(rec_pk)

    assert checked == [
        tmp_path / rec_tbl._get_recording_name(rec_pk)
    ], "Recompute of the default directory skipped hash verification"
    assert rec_info["hash"], "Rebuild returned no hash to store"
