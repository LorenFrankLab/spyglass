"""Tests for `DandiPath.download_file_from_dandi` staging and arguments.

No network: the asset lookup and the HTTP filesystem are both replaced, so
what is under test is the destination handling and the staging path.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="function")
def dandi_tbl(common):
    """The `DandiPath` table, with its key and URL lookups stubbed out."""
    from spyglass.common.common_dandi import DandiPath

    _ = common  # connection must be live before the schema is declared
    tbl = DandiPath()

    with patch.object(tbl, "_resolve_key", side_effect=lambda k, p: k or {}):
        with patch.object(tbl, "_asset_url", return_value="http://x/a.nwb"):
            yield tbl


@pytest.fixture(scope="function")
def fake_http(monkeypatch):
    """Replace the http filesystem with one that records and writes staged."""
    staged = []

    def get_file(url, path):
        staged.append(Path(path))
        Path(path).write_bytes(b"content")

    fs = MagicMock()
    fs.get_file.side_effect = get_file

    monkeypatch.setattr(
        "spyglass.common.common_dandi.fsspec.filesystem", lambda _: fs
    )

    return staged


def test_download_requires_destination(dandi_tbl):
    """A key names the file on the archive, not where it lands locally.

    Without this the key-only form fell through to `Path(None)`.
    """
    with pytest.raises(ValueError, match="Must provide dest"):
        dandi_tbl.download_file_from_dandi(key={"nwb_file_name": "a.nwb"})


def test_download_stages_uniquely(dandi_tbl, fake_http, tmp_path):
    """Two downloads of one file must not share a staging path.

    A fixed `<dest>.part` lets concurrent workers read, rename, or unlink each
    other's partial copy. The name must differ per call.
    """
    dest = tmp_path / "sub" / "file.nwb"

    assert dandi_tbl.download_file_from_dandi(key={}, dest=str(dest))
    assert dandi_tbl.download_file_from_dandi(key={}, dest=str(dest))

    first, second = fake_http

    assert first != second, "Concurrent downloads share a staging path"
    assert first.parent == dest.parent, "Staged outside the destination dir"
    assert first.name.endswith(".part"), "Staging path not marked partial"
    assert dest.read_bytes() == b"content", "Download did not reach dest"


def test_download_cleans_up_on_failure(dandi_tbl, monkeypatch, tmp_path):
    """A failed transfer leaves no partial file where a local copy would be.

    `LocalBackend` would otherwise hand the fragment to the next caller.
    """
    dest = tmp_path / "file.nwb"

    def get_file(url, path):  # drop mid-transfer, as a reset connection does
        Path(path).write_bytes(b"half a")
        raise OSError("connection reset")

    fs = MagicMock()
    fs.get_file.side_effect = get_file

    monkeypatch.setattr(
        "spyglass.common.common_dandi.fsspec.filesystem", lambda _: fs
    )

    with pytest.raises(OSError, match="connection reset"):
        dandi_tbl.download_file_from_dandi(key={}, dest=str(dest))

    assert not dest.exists(), "Failed download left a file at dest"
    assert not list(tmp_path.glob("*.part")), "Failed download left a fragment"
