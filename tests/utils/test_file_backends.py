"""Tests for the pluggable file-backend registry used by get_nwb_file."""

from unittest.mock import MagicMock, patch

import pytest

import spyglass.utils.nwb_helper_fn as helper
import spyglass.utils.file_backends as backends_mod
from spyglass.utils.file_backends import (
    DandiBackend,
    FileBackend,
    KacheryBackend,
    LocalBackend,
    get_backends,
    register_backend,
)


@pytest.fixture(autouse=True)
def restore_backend_chain():
    """Undo any registration, so the chain does not leak between tests."""
    original = list(backends_mod._BACKENDS)
    yield
    backends_mod._BACKENDS[:] = original


@pytest.fixture(autouse=True)
def clear_nwb_cache():
    """Keep fake IO objects out of the module-level open-file cache.

    `get_nwb_file` caches `(io, nwbfile)` pairs, and the session teardown calls
    `close()` on every cached io. Fakes used here are not real IO handles.
    """
    yield
    helper.__open_nwb_files.clear()


class _FakeDownloadBackend(FileBackend):
    """Download-only backend, exercising the inherited `open`."""

    name = "fake_dl"
    supports_streaming = False

    def __init__(self, has=True, download_ok=True):
        self._has = has
        self._download_ok = download_ok
        self.calls = []

    def has(self, nwb_file_path):
        self.calls.append("has")
        return self._has

    def download(self, nwb_file_path, dest=None):
        self.calls.append("download")
        return self._download_ok


class _FakeStreamBackend(FileBackend):
    """Streaming backend, overriding `open`."""

    name = "fake_stream"
    supports_streaming = True

    def __init__(self, has=True):
        self._has = has
        self.calls = []

    def has(self, nwb_file_path):
        self.calls.append("has")
        return self._has

    def open(self, nwb_file_path):
        self.calls.append("open")
        return ("io", "nwbfile")


def test_default_chain_order():
    """Local disk is tried first, then remote sources in fallback order."""
    assert [b.name for b in get_backends()] == ["local", "kachery", "Dandi"]


def test_get_backends_returns_a_copy():
    """Callers cannot mutate the chain in place."""
    got = get_backends()
    got.clear()
    assert [b.name for b in get_backends()] == ["local", "kachery", "Dandi"]


def test_register_backend_appends_by_default():
    """A registered backend lands at the end of the chain."""
    register_backend(_FakeStreamBackend())
    assert [b.name for b in get_backends()][-1] == "fake_stream"


def test_register_backend_honors_index():
    """A backend can be placed ahead of local disk."""
    register_backend(_FakeStreamBackend(), index=0)
    assert [b.name for b in get_backends()][0] == "fake_stream"


def test_local_backend_has_and_open(tmp_path):
    """LocalBackend reports on-disk presence and reads directly."""
    missing = str(tmp_path / "missing.nwb")
    present = tmp_path / "present.nwb"
    present.write_text("not really nwb")

    backend = LocalBackend()
    assert backend.has(missing) is False
    assert backend.has(str(present)) is True
    assert backend.supports_streaming is False

    with patch(
        "spyglass.utils.file_backends.open_local_nwb",
        return_value=("io", "nwbfile"),
    ) as mock_open:
        assert backend.open(str(present)) == ("io", "nwbfile")
    mock_open.assert_called_once_with(str(present))


def test_download_only_backend_uses_default_open(tmp_path):
    """A backend without `open` downloads, then reads the local copy."""
    backend = _FakeDownloadBackend(download_ok=True)
    target = str(tmp_path / "file.nwb")

    with patch(
        "spyglass.utils.file_backends.open_local_nwb",
        return_value=("io", "nwbfile"),
    ) as mock_open:
        result = backend.open(target)

    assert result == ("io", "nwbfile")
    assert backend.calls == ["download"]
    mock_open.assert_called_once_with(target)


def test_failed_download_raises_file_not_found(tmp_path):
    """A download that reports failure surfaces as FileNotFoundError."""
    backend = _FakeDownloadBackend(download_ok=False)

    with pytest.raises(FileNotFoundError, match="fake_dl"):
        backend.open(str(tmp_path / "file.nwb"))


def test_base_download_not_implemented(tmp_path):
    """A streaming-only backend reports that download is unavailable."""
    with pytest.raises(NotImplementedError, match="fake_stream"):
        _FakeStreamBackend().download(str(tmp_path / "file.nwb"))


def test_get_nwb_file_tries_backends_in_order(tmp_path):
    """Backends are consulted in order; the first that has the file wins."""

    first = _FakeStreamBackend(has=False)
    second = _FakeStreamBackend(has=True)
    target = str(tmp_path / "missing.nwb")

    with (
        patch.object(helper, "get_backends", return_value=[first, second]),
        patch("os.path.exists", return_value=False),
    ):
        result = helper.get_nwb_file(target)

    assert result == "nwbfile"
    assert first.calls == ["has"]  # asked, declined, not opened
    assert second.calls == ["has", "open"]


def test_get_nwb_file_falls_through_when_backend_cannot_supply(tmp_path):
    """A backend that claims the file but fails does not halt the chain."""

    broken = _FakeDownloadBackend(has=True, download_ok=False)
    working = _FakeStreamBackend(has=True)
    target = str(tmp_path / "missing.nwb")

    with (
        patch.object(helper, "get_backends", return_value=[broken, working]),
        patch("os.path.exists", return_value=False),
    ):
        result = helper.get_nwb_file(target)

    assert result == "nwbfile"
    assert broken.calls == ["has", "download"]
    assert working.calls == ["has", "open"]


def test_get_nwb_file_error_names_the_backends_tried(tmp_path):
    """The not-found message lists the backends that were consulted."""

    target = str(tmp_path / "missing.nwb")
    backends = [KacheryBackend(), DandiBackend()]

    with (
        patch.object(helper, "get_backends", return_value=backends),
        patch("os.path.exists", return_value=False),
        patch.object(KacheryBackend, "has", return_value=False),
        patch.object(DandiBackend, "has", return_value=False),
    ):
        with pytest.raises(
            FileNotFoundError, match="not found in kachery or Dandi"
        ):
            helper.get_nwb_file(target)


def test_kachery_has_is_false_when_unavailable(tmp_path):
    """KacheryBackend.has short-circuits before touching the database."""
    import spyglass.sharing.sharing_kachery as skm

    with patch.object(skm, "_kachery_available", False):
        # A database query here would raise; returning False proves we
        # short-circuited on availability first.
        assert KacheryBackend().has(str(tmp_path / "x.nwb")) is False


def test_file_from_dandi_is_deprecated_alias():
    """The old name still works and is recorded in the deprecation log."""

    with (
        patch.object(helper, "file_is_remote", return_value=True) as mock_fn,
        patch("spyglass.common.common_usage.ActivityLog") as mock_log,
    ):
        assert helper.file_from_dandi("/some/path.nwb") is True

    mock_log.return_value.deprecate_log.assert_called_once_with(
        "file_from_dandi", alt="file_is_remote"
    )
    mock_fn.assert_called_once_with("/some/path.nwb")


def test_backends_declare_streaming_support():
    """Streaming capability is declared, not inferred."""
    assert KacheryBackend.supports_streaming is False
    assert DandiBackend.supports_streaming is True


def test_recompute_still_runs_after_backends_fail(tmp_path):
    """The recompute fallback is unchanged by the registry refactor."""

    target = str(tmp_path / "missing.nwb")
    query = MagicMock()
    query._make_file.return_value = "recomputed"

    with (
        patch.object(helper, "get_backends", return_value=[]),
        patch("os.path.exists", return_value=False),
    ):
        assert helper.get_nwb_file(target, query_expression=query) == (
            "recomputed"
        )

    query._make_file.assert_called_once_with(recompute_file_name="missing.nwb")
