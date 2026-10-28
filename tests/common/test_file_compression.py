"""Tests for retroactive NWB file compression."""

import os
import pathlib
import stat
from shutil import which

import h5py
import numpy as np
import pytest

no_h5repack = pytest.mark.skipif(
    which("h5repack") is None, reason="h5repack not on PATH"
)


@pytest.fixture
def repack_src(tmp_path):
    """A file standing in for a raw NWB file, group-readable."""
    src = tmp_path / "session_.nwb"
    src.write_bytes(b"original")
    src.chmod(0o664)
    return src


@pytest.fixture
def repack_tmp(tmp_path):
    """A repack output as ``tempfile.mkstemp`` leaves it: 0600."""
    tmp = tmp_path / "session_.nwb.tmp"
    tmp.write_bytes(b"repacked")
    tmp.chmod(0o600)
    return tmp


@pytest.fixture
def uncompressed_h5(tmp_path):
    """An HDF5 file with two uncompressed datasets."""
    path = tmp_path / "uncompressed_.nwb"
    with h5py.File(path, "w") as f:
        f.create_dataset("a", data=np.arange(2048, dtype="int16"))
        f["a"].attrs["unit"] = "mV"
        f.create_group("grp").create_dataset("b", data=np.zeros(2048))
    return path


@pytest.fixture
def gzip_copy(tmp_path):
    """Build a gzip-compressed copy of a file, as h5repack would."""

    def _copy(src):
        dst = tmp_path / f"repacked_{src.name}"

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out = dst_f.create_dataset(
                    name, data=obj[()], compression="gzip", compression_opts=6
                )
            else:
                out = dst_f.require_group(name)
            out.attrs.update(obj.attrs)

        with h5py.File(src, "r") as src_f, h5py.File(dst, "w") as dst_f:
            dst_f.attrs.update(src_f.attrs)
            src_f.visititems(visit)
        return dst

    return _copy


@pytest.fixture
def compressed_h5(tmp_path):
    """An HDF5 file whose every dataset already has a compression filter."""
    path = tmp_path / "compressed_.nwb"
    with h5py.File(path, "w") as f:
        f.create_dataset("a", data=np.arange(2048), compression="gzip")
        f.create_group("grp").create_dataset(
            "b", data=np.zeros(2048), compression="gzip"
        )
    return path


# ----------------------------- replace in place -----------------------------


def test_replace_preserves_mode(repack_src, repack_tmp):
    """Replacing a source file keeps the source's permissions."""
    from spyglass.common.common_file_compression import (
        _replace_preserving_mode,
    )

    before = stat.S_IMODE(repack_src.stat().st_mode)
    _replace_preserving_mode(tmp=repack_tmp, src=repack_src)

    assert repack_src.read_bytes() == b"repacked", "Repacked data not moved"
    assert (
        stat.S_IMODE(repack_src.stat().st_mode) == before
    ), "Repack changed the file's permissions"
    assert not repack_tmp.exists(), "Temp file left behind"


def test_replace_preserves_group_read(repack_src, repack_tmp):
    """A group-readable source stays readable by its group."""
    from spyglass.common.common_file_compression import (
        _replace_preserving_mode,
    )

    _replace_preserving_mode(tmp=repack_tmp, src=repack_src)

    assert (
        repack_src.stat().st_mode & stat.S_IRGRP
    ), "Repacked file is no longer group-readable"


def test_replace_missing_src(tmp_path, repack_tmp):
    """Replacing a file that does not exist raises."""
    from spyglass.common.common_file_compression import (
        _replace_preserving_mode,
    )

    with pytest.raises(FileNotFoundError):
        _replace_preserving_mode(tmp=repack_tmp, src=tmp_path / "absent.nwb")

    assert repack_tmp.exists(), "Temp file removed on a failed replace"


def test_replace_preserves_mode_0640(tmp_path, repack_tmp):
    """Mode is copied from the source, not forced to a fixed value."""
    from spyglass.common.common_file_compression import (
        _replace_preserving_mode,
    )

    src = tmp_path / "other_.nwb"
    src.write_bytes(b"original")
    os.chmod(src, 0o640)

    _replace_preserving_mode(tmp=repack_tmp, src=src)

    assert stat.S_IMODE(src.stat().st_mode) == 0o640


# -------------------------------- h5repack ---------------------------------


def test_h5repack_missing(monkeypatch, tmp_path):
    """A missing h5repack names the tool and how to install it."""
    from spyglass.common.common_file_compression import _h5repack

    monkeypatch.setattr(
        "spyglass.common.common_file_compression.which", lambda _: None
    )

    with pytest.raises(RuntimeError, match="h5repack not found"):
        _h5repack(src=tmp_path / "a.nwb", dst=tmp_path / "b.nwb", level=6)


@no_h5repack
def test_h5repack_compresses(uncompressed_h5, tmp_path):
    """h5repack compresses every dataset without changing the contents."""
    from spyglass.common.common_file_compression import (
        _h5repack,
        _has_uncompressed_dataset,
        _verify,
    )

    dst = tmp_path / "out.nwb"
    _h5repack(src=uncompressed_h5, dst=dst, level=6)

    assert not _has_uncompressed_dataset(dst)
    _verify(src=uncompressed_h5, dst=dst)  # contents survived the repack


# -------------------------------- counting ---------------------------------


def test_detects_uncompressed(uncompressed_h5):
    """A file with uncompressed datasets is flagged."""
    from spyglass.common.common_file_compression import (
        _has_uncompressed_dataset,
    )

    assert _has_uncompressed_dataset(uncompressed_h5)


def test_detects_fully_compressed(compressed_h5):
    """A file whose every dataset is compressed is not flagged."""
    from spyglass.common.common_file_compression import (
        _has_uncompressed_dataset,
    )

    assert not _has_uncompressed_dataset(compressed_h5)


def test_detects_partially_compressed(tmp_path):
    """One uncompressed dataset among compressed ones is enough."""
    from spyglass.common.common_file_compression import (
        _has_uncompressed_dataset,
    )

    path = tmp_path / "mixed_.nwb"
    with h5py.File(path, "w") as f:
        f.create_dataset("a", data=np.arange(2048), compression="gzip")
        f.create_dataset("b", data=np.arange(2048))

    assert _has_uncompressed_dataset(path)


def test_no_datasets_is_not_uncompressed(tmp_path):
    """A file with no datasets is left alone, as before."""
    from spyglass.common.common_file_compression import (
        _has_uncompressed_dataset,
    )

    path = tmp_path / "empty_.nwb"
    with h5py.File(path, "w") as f:
        f.create_group("grp")

    assert not _has_uncompressed_dataset(path)


# ------------------------------- verification -------------------------------


def test_verify_accepts_compressed_copy(uncompressed_h5, gzip_copy):
    """A lossless repack hashes the same as its source."""
    from spyglass.common.common_file_compression import _verify

    _verify(src=uncompressed_h5, dst=gzip_copy(uncompressed_h5))


def test_verify_rejects_changed_value(uncompressed_h5, gzip_copy):
    """A single altered value fails verification."""
    from spyglass.common.common_file_compression import _verify

    dst = gzip_copy(uncompressed_h5)
    with h5py.File(dst, "r+") as f:
        f["a"][0] += 1

    with pytest.raises(ValueError, match="does not match its source"):
        _verify(src=uncompressed_h5, dst=dst)


def test_verify_rejects_dropped_dataset(uncompressed_h5, gzip_copy):
    """A dropped dataset fails verification."""
    from spyglass.common.common_file_compression import _verify

    dst = gzip_copy(uncompressed_h5)
    with h5py.File(dst, "r+") as f:
        del f["grp/b"]

    with pytest.raises(ValueError, match="does not match its source"):
        _verify(src=uncompressed_h5, dst=dst)


def test_verify_rejects_dropped_attr(uncompressed_h5, gzip_copy):
    """A dropped attribute fails verification."""
    from spyglass.common.common_file_compression import _verify

    dst = gzip_copy(uncompressed_h5)
    with h5py.File(dst, "r+") as f:
        del f["a"].attrs["unit"]

    with pytest.raises(ValueError, match="does not match its source"):
        _verify(src=uncompressed_h5, dst=dst)


# ------------------------------ stage / commit ------------------------------


def test_stage_skips_compressed(compressed_h5):
    """An already-compressed file stages nothing and records no level."""
    from spyglass.common.common_file_compression import _StagedRepack

    before = compressed_h5.read_bytes()
    item = _StagedRepack.stage(compressed_h5, gzip_level=6)
    tmp, stats = item.tmp, item.stats

    assert tmp is None, "Staged a repack for an already-compressed file"
    assert stats["gzip_level"] is None, "Recorded as repacked"
    assert stats["original_size_bytes"] == stats["repacked_size_bytes"]
    assert compressed_h5.read_bytes() == before, "File was rewritten"


def test_stage_missing_file(tmp_path):
    """Staging a file that does not exist raises before any work."""
    from spyglass.common.common_file_compression import _StagedRepack

    with pytest.raises(FileNotFoundError):
        _StagedRepack.stage(tmp_path / "absent_.nwb", gzip_level=6)


def test_stage_leaves_source_untouched(monkeypatch, uncompressed_h5, gzip_copy):
    """A successful stage writes a sibling and does not touch the source."""
    from spyglass.common import common_file_compression as cfc

    before = uncompressed_h5.read_bytes()
    monkeypatch.setattr(
        cfc,
        "_h5repack",
        lambda src, dst, level: pathlib.Path(dst).write_bytes(
            gzip_copy(src).read_bytes()
        ),
    )

    item = cfc._StagedRepack.stage(uncompressed_h5, gzip_level=6)
    tmp, stats = item.tmp, item.stats

    assert tmp is not None and tmp.exists(), "Nothing staged"
    assert uncompressed_h5.read_bytes() == before, "Source was modified"
    assert stats["gzip_level"] == 6
    assert stats["repacked_size_bytes"] == tmp.stat().st_size


def test_stage_verify_failure_cleans_up(monkeypatch, uncompressed_h5):
    """A failed verification leaves the source and removes the temp file."""
    from spyglass.common import common_file_compression as cfc

    before = uncompressed_h5.read_bytes()

    def fake_repack(src, dst, level):
        with h5py.File(dst, "w") as f:  # a dataset short of the source
            f.create_dataset("a", data=np.arange(2048), compression="gzip")

    monkeypatch.setattr(cfc, "_h5repack", fake_repack)

    with pytest.raises(ValueError, match="does not match its source"):
        cfc._StagedRepack.stage(uncompressed_h5, gzip_level=6)

    assert uncompressed_h5.read_bytes() == before, "Source was modified"
    leftovers = list(uncompressed_h5.parent.glob("*.nwb.tmp"))
    assert not leftovers, f"Temp files left behind: {leftovers}"


def test_stage_h5repack_failure_cleans_up(monkeypatch, uncompressed_h5):
    """An h5repack failure leaves the source and removes the temp file."""
    from spyglass.common import common_file_compression as cfc

    before = uncompressed_h5.read_bytes()

    def boom(src, dst, level):
        raise RuntimeError("h5repack exploded")

    monkeypatch.setattr(cfc, "_h5repack", boom)

    with pytest.raises(RuntimeError, match="exploded"):
        cfc._StagedRepack.stage(uncompressed_h5, gzip_level=6)

    assert uncompressed_h5.read_bytes() == before, "Source was modified"
    leftovers = list(uncompressed_h5.parent.glob("*.nwb.tmp"))
    assert not leftovers, f"Temp files left behind: {leftovers}"


def test_populate_needs_restriction():
    """An unrestricted populate is refused: repacking is in place."""
    from spyglass.common.common_file_compression import CompressedNwbfile

    with pytest.raises(ValueError, match="requires a restriction"):
        CompressedNwbfile().populate()


# --------------------------------- preflight --------------------------------


def test_space_check_passes(uncompressed_h5):
    """A directory with room for a second copy passes."""
    from spyglass.common.common_file_compression import _assert_space_to_stage

    _assert_space_to_stage(uncompressed_h5)


def test_space_check_refuses(monkeypatch, uncompressed_h5):
    """Too little free space refuses before any temp file is made."""
    from collections import namedtuple

    from spyglass.common import common_file_compression as cfc

    usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(cfc, "disk_usage", lambda _: usage(1, 1, 0))

    with pytest.raises(RuntimeError, match="Not enough space"):
        cfc._StagedRepack.stage(uncompressed_h5, gzip_level=6)

    leftovers = list(uncompressed_h5.parent.glob("*.nwb.tmp"))
    assert not leftovers, f"Temp files left behind: {leftovers}"


# ------------------------------- gzip level ---------------------------------


def test_compressed_data_uses_shared_level():
    """Inline writes take the level from the one shared constant."""
    from spyglass.utils.h5py_helper_fn import GZIP_LEVEL, compressed_data

    wrapped = compressed_data(np.arange(128))

    assert wrapped.io_settings["compression"] == "gzip"
    assert wrapped.io_settings["compression_opts"] == GZIP_LEVEL


def test_compressed_data_uses_shared_shuffle():
    """Inline writes take the shuffle setting from the same place."""
    from spyglass.utils.h5py_helper_fn import SHUFFLE, compressed_data

    assert compressed_data(np.arange(128)).io_settings["shuffle"] is SHUFFLE


def test_repack_orders_shuf_before_gzip(monkeypatch, tmp_path):
    """h5repack applies filters in order, so SHUF must come first."""
    from spyglass.common import common_file_compression as cfc

    seen = {}

    class Result:
        returncode = 0
        stdout = stderr = ""
        args = []

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return Result()

    monkeypatch.setattr(cfc.subprocess, "run", fake_run)
    monkeypatch.setattr(cfc, "SHUFFLE", True)
    monkeypatch.setattr(cfc, "which", lambda _: "/usr/bin/h5repack")

    cfc._h5repack(src=tmp_path / "a.nwb", dst=tmp_path / "b.nwb", level=6)

    cmd = seen["cmd"]
    assert cmd[:5] == ["h5repack", "-f", "SHUF", "-f", "GZIP=6"]
    assert "-m" in cmd and cmd[cmd.index("-m") + 1] == "1"


def test_repack_omits_shuf_when_off(monkeypatch, tmp_path):
    """With shuffle off, only the gzip filter is passed."""
    from spyglass.common import common_file_compression as cfc

    seen = {}

    class Result:
        returncode = 0
        stdout = stderr = ""
        args = []

    monkeypatch.setattr(
        cfc.subprocess,
        "run",
        lambda cmd, **kw: (seen.update(cmd=cmd), Result())[1],
    )
    monkeypatch.setattr(cfc, "SHUFFLE", False)
    monkeypatch.setattr(cfc, "which", lambda _: "/usr/bin/h5repack")

    cfc._h5repack(src=tmp_path / "a.nwb", dst=tmp_path / "b.nwb", level=6)

    assert "SHUF" not in seen["cmd"]
    assert seen["cmd"][:3] == ["h5repack", "-f", "GZIP=6"]


def test_compressed_data_level_override():
    """An explicit level wins over the shared constant."""
    from spyglass.utils.h5py_helper_fn import compressed_data

    wrapped = compressed_data(np.arange(128), compression_opts=1)

    assert wrapped.io_settings["compression_opts"] == 1


def test_repack_uses_shared_level(monkeypatch, uncompressed_h5):
    """The retroactive repack passes the same constant to h5repack."""
    from spyglass.common import common_file_compression as cfc
    from spyglass.utils.h5py_helper_fn import GZIP_LEVEL

    seen = {}

    def fake_repack(src, dst, level):
        seen["level"] = level
        raise RuntimeError("stop here")

    monkeypatch.setattr(cfc, "_h5repack", fake_repack)
    monkeypatch.setattr(
        cfc.Nwbfile, "get_abs_path", staticmethod(lambda name, **kw: name)
    )

    with pytest.raises(RuntimeError, match="stop here"):
        cfc._StagedRepack.stage(uncompressed_h5, gzip_level=GZIP_LEVEL)

    assert seen["level"] == GZIP_LEVEL


# ----------------------------- eligible datasets ----------------------------


def test_scalar_only_uncompressed_needs_no_repack(tmp_path):
    """A scalar cannot carry a filter, so it does not force a repack."""
    from spyglass.common.common_file_compression import (
        _has_uncompressed_dataset,
    )

    path = tmp_path / "scalars_.nwb"
    with h5py.File(path, "w") as f:
        f.create_dataset("a", data=np.arange(4096), compression="gzip")
        f.create_dataset("identifier", data="abc")  # scalar, no filter
        f.create_dataset("empty", shape=(0,), dtype="f8")  # zero-length

    assert not _has_uncompressed_dataset(path)


def test_eligible_uncompressed_still_detected(tmp_path):
    """An uncompressed chunkable dataset is still reported."""
    from spyglass.common.common_file_compression import (
        _has_uncompressed_dataset,
    )

    path = tmp_path / "mixed_.nwb"
    with h5py.File(path, "w") as f:
        f.create_dataset("a", data=np.arange(4096), compression="gzip")
        f.create_dataset("identifier", data="abc")
        f.create_dataset("b", data=np.arange(4096))  # eligible, uncompressed

    assert _has_uncompressed_dataset(path)


# -------------------------------- make_fetch --------------------------------


def test_make_fetch_resolves_unregistered_raw(monkeypatch, mini_copy_name):
    """The raw file is not in Nwbfile, so its path must not be looked up."""
    from spyglass.common import LabMember
    from spyglass.common.common_file_compression import CompressedNwbfile

    # make_fetch imports LabMember lazily, so patch the class itself
    monkeypatch.setattr(
        LabMember, "check_admin_privilege", lambda self, *a, **k: None
    )

    (paths,) = CompressedNwbfile().make_fetch({"nwb_file_name": mini_copy_name})

    assert paths["copy"].endswith(mini_copy_name)
    assert paths["raw"].endswith(mini_copy_name.replace("_.nwb", ".nwb"))
    assert not paths["raw"].endswith("_.nwb")


# -------------------------------- ownership ---------------------------------


def test_replace_preserves_ownership(repack_src, repack_tmp):
    """An owned file keeps its uid and gid across the replace."""
    import os

    from spyglass.common.common_file_compression import (
        _replace_preserving_mode,
    )

    before = repack_src.stat()
    _replace_preserving_mode(tmp=repack_tmp, src=repack_src)
    after = repack_src.stat()

    assert (after.st_uid, after.st_gid) == (before.st_uid, before.st_gid)
    assert os.stat(repack_src).st_uid == os.geteuid()


def test_replace_refuses_foreign_owner(monkeypatch, repack_src, repack_tmp):
    """A file owned by another user is left alone, not reassigned."""
    import os

    from spyglass.common import common_file_compression as cfc

    before = repack_src.read_bytes()

    # Look like a different user, so the chown branch is taken, and refuse
    # it the way the kernel refuses giving a file away. Read the real uid
    # first: cfc.os is the os module the lambda would otherwise re-enter.
    foreign = os.geteuid() + 1
    monkeypatch.setattr(cfc.os, "geteuid", lambda: foreign)

    def refuse(*a, **k):
        raise PermissionError("not permitted")

    monkeypatch.setattr(cfc.os, "chown", refuse)

    with pytest.raises(PermissionError, match="Cannot preserve ownership") as e:
        cfc._replace_preserving_mode(tmp=repack_tmp, src=repack_src)

    msg = str(e.value)
    assert "sudo chown" in msg and "sudo mv" in msg, "No recovery command"
    assert str(repack_tmp) in msg, "Message does not name the staged file"

    assert repack_src.read_bytes() == before, "Source was replaced"
    assert repack_tmp.exists(), "Verified repack discarded on a fixable error"


def test_stage_overwrites_leftover_tmp(monkeypatch, uncompressed_h5):
    """A staged file from a previous attempt is replaced, not appended to."""
    from spyglass.common import common_file_compression as cfc

    stale = uncompressed_h5.with_name(uncompressed_h5.name + ".repack.tmp")
    stale.write_bytes(b"stale from an earlier run")

    monkeypatch.setattr(
        cfc,
        "_h5repack",
        lambda src, dst, level: pathlib.Path(dst).write_bytes(b"fresh"),
    )
    monkeypatch.setattr(cfc, "_verify", lambda src, dst: None)

    tmp = cfc._StagedRepack.stage(uncompressed_h5, gzip_level=6).tmp

    assert tmp == stale, "Staging path is not deterministic"
    assert tmp.read_bytes() == b"fresh", "Leftover was not overwritten"


# ------------------------------- staleness ---------------------------------


def test_commit_rejects_changed_source(monkeypatch, uncompressed_h5):
    """A source written to during the repack is not overwritten."""
    from spyglass.common import common_file_compression as cfc

    monkeypatch.setattr(
        cfc,
        "_h5repack",
        lambda src, dst, level: pathlib.Path(dst).write_bytes(b"packed"),
    )
    monkeypatch.setattr(cfc, "_verify", lambda src, dst: None)

    item = cfc._StagedRepack.stage(uncompressed_h5, gzip_level=6)

    # a metadata writer touches the source while the repack ran
    with h5py.File(item.src, "a") as f:
        f.attrs["object_id"] = "rewritten"
    after = item.src.read_bytes()

    with pytest.raises(RuntimeError, match="changed while it was being"):
        item.commit()

    assert item.src.read_bytes() == after, "The newer write was discarded"
    assert item.tmp.exists(), "Stale staging removed instead of left"


def test_commit_proceeds_on_unchanged_source(monkeypatch, uncompressed_h5):
    """An untouched source is replaced as normal."""
    from spyglass.common import common_file_compression as cfc
    from spyglass.utils import dj_helper_fn

    monkeypatch.setattr(
        cfc,
        "_h5repack",
        lambda src, dst, level: pathlib.Path(dst).write_bytes(b"packed"),
    )
    monkeypatch.setattr(cfc, "_verify", lambda src, dst: None)
    # commit() imports it lazily, so patch it where it lives
    monkeypatch.setattr(
        dj_helper_fn, "_resolve_external_table", lambda **kw: None
    )

    item = cfc._StagedRepack.stage(uncompressed_h5, gzip_level=6)
    item.commit()

    assert item.src.read_bytes() == b"packed"
    assert not item.tmp.exists()
    assert item.kept.exists(), "Original not set aside for release"


# -------------------------------- rollback ----------------------------------


def _staged_pair(cfc, monkeypatch, a, b):
    """Stage two files with h5repack and verification stubbed out."""
    monkeypatch.setattr(
        cfc,
        "_h5repack",
        lambda src, dst, level: pathlib.Path(dst).write_bytes(b"packed"),
    )
    monkeypatch.setattr(cfc, "_verify", lambda src, dst: None)
    return {
        "copy": cfc._StagedRepack.stage(a, gzip_level=6),
        "raw": cfc._StagedRepack.stage(b, gzip_level=6),
    }


@pytest.fixture
def two_sources(tmp_path):
    """Two uncompressed files standing in for a session's pair."""
    made = []
    for name in ("sess_.nwb", "sess.nwb"):
        path = tmp_path / name
        with h5py.File(path, "w") as f:
            f.create_dataset("a", data=np.arange(2048))
        made.append(path)
    return made


def test_insert_failure_restores_every_original(
    monkeypatch, two_sources, mini_dict
):
    """A failing insert puts both sources back as they were."""
    from spyglass.common import common_file_compression as cfc
    from spyglass.utils import dj_helper_fn

    a, b = two_sources
    before = (a.read_bytes(), b.read_bytes())
    monkeypatch.setattr(
        dj_helper_fn, "_resolve_external_table", lambda **kw: None
    )
    staged = _staged_pair(cfc, monkeypatch, a, b)

    tbl = cfc.CompressedNwbfile()
    monkeypatch.setattr(
        type(tbl),
        "insert",
        lambda *ar, **kw: (_ for _ in ()).throw(RuntimeError("insert failed")),
    )

    with pytest.raises(RuntimeError, match="insert failed"):
        tbl.make_insert(mini_dict, staged)

    assert (a.read_bytes(), b.read_bytes()) == before, "Sources not restored"
    assert not list(a.parent.glob("*.orig")), "Set-aside originals left"


def test_second_commit_failure_restores_the_first(
    monkeypatch, two_sources, mini_dict
):
    """A failure on the second file undoes the first file's replace."""
    from spyglass.common import common_file_compression as cfc
    from spyglass.utils import dj_helper_fn

    a, b = two_sources
    before = (a.read_bytes(), b.read_bytes())
    monkeypatch.setattr(
        dj_helper_fn, "_resolve_external_table", lambda **kw: None
    )
    staged = _staged_pair(cfc, monkeypatch, a, b)

    calls = {"n": 0}
    real_replace = cfc._replace_preserving_mode

    def fail_on_second(tmp, src, stat_src=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise PermissionError("second file refused")
        real_replace(tmp=tmp, src=src, stat_src=stat_src)

    monkeypatch.setattr(cfc, "_replace_preserving_mode", fail_on_second)

    with pytest.raises(PermissionError, match="second file refused"):
        cfc.CompressedNwbfile().make_insert(mini_dict, staged)

    assert (a.read_bytes(), b.read_bytes()) == before, "First not rolled back"
    assert not list(a.parent.glob("*.orig")), "Set-aside originals left"
