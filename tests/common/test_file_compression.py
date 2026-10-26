"""Tests for retroactive NWB file compression."""

import os
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
        f.create_group("grp").create_dataset("b", data=np.zeros(2048))
    return path


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
    """h5repack writes every dataset with a compression filter."""
    from spyglass.common.common_file_compression import (
        _count_compressed_datasets,
        _count_datasets,
        _h5repack,
    )

    dst = tmp_path / "out.nwb"
    _h5repack(src=uncompressed_h5, dst=dst, level=6)

    assert _count_datasets(dst) == _count_datasets(uncompressed_h5)
    assert _count_compressed_datasets(dst) == _count_datasets(dst)


# -------------------------------- counting ---------------------------------


def test_counts_uncompressed(uncompressed_h5):
    """Datasets are counted; none register as compressed."""
    from spyglass.common.common_file_compression import (
        _count_compressed_datasets,
        _count_datasets,
    )

    assert _count_datasets(uncompressed_h5) == 2
    assert _count_compressed_datasets(uncompressed_h5) == 0


def test_counts_compressed(compressed_h5):
    """A fully compressed file reports every dataset as compressed."""
    from spyglass.common.common_file_compression import (
        _count_compressed_datasets,
        _count_datasets,
    )

    assert _count_datasets(compressed_h5) == 2
    assert _count_compressed_datasets(compressed_h5) == 2


# ------------------------------- repack file --------------------------------


def test_repack_skips_compressed(compressed_h5):
    """An already-compressed file is recorded without being repacked."""
    from spyglass.common.common_file_compression import _repack_file

    before = compressed_h5.read_bytes()
    stats = _repack_file(str(compressed_h5), gzip_level=6)

    assert stats["gzip_level"] == 0, "Recorded as repacked"
    assert stats["datasets_compressed"] == stats["datasets_total"]
    assert compressed_h5.read_bytes() == before, "File was rewritten"


def test_repack_missing_file(tmp_path):
    """Repacking a file that does not exist raises before any work."""
    from spyglass.common.common_file_compression import _repack_file

    with pytest.raises(FileNotFoundError):
        _repack_file(str(tmp_path / "absent_.nwb"), gzip_level=6)


def test_repack_verify_failure_leaves_source(monkeypatch, uncompressed_h5):
    """A failed verification leaves the source and removes the temp file."""
    from spyglass.common import common_file_compression as cfc

    before = uncompressed_h5.read_bytes()

    def fake_repack(src, dst, level):
        with h5py.File(dst, "w") as f:  # a dataset short of the source
            f.create_dataset("a", data=np.arange(2048), compression="gzip")

    monkeypatch.setattr(cfc, "_h5repack", fake_repack)

    with pytest.raises(ValueError, match="Dataset count mismatch"):
        cfc._repack_file(str(uncompressed_h5), gzip_level=6)

    assert uncompressed_h5.read_bytes() == before, "Source was modified"
    leftovers = list(uncompressed_h5.parent.glob("*.nwb.tmp"))
    assert not leftovers, f"Temp files left behind: {leftovers}"


def test_repack_h5repack_failure_leaves_source(monkeypatch, uncompressed_h5):
    """An h5repack failure leaves the source and removes the temp file."""
    from spyglass.common import common_file_compression as cfc

    before = uncompressed_h5.read_bytes()

    def boom(src, dst, level):
        raise RuntimeError("h5repack exploded")

    monkeypatch.setattr(cfc, "_h5repack", boom)

    with pytest.raises(RuntimeError, match="exploded"):
        cfc._repack_file(str(uncompressed_h5), gzip_level=6)

    assert uncompressed_h5.read_bytes() == before, "Source was modified"
    leftovers = list(uncompressed_h5.parent.glob("*.nwb.tmp"))
    assert not leftovers, f"Temp files left behind: {leftovers}"
