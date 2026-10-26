"""Tests for retroactive NWB file compression."""

import os
import stat

import pytest


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
