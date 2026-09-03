"""Tests for the pluggable file-backend registry used by get_nwb_file."""

from unittest.mock import MagicMock, patch

import pytest

import spyglass.utils.nwb_helper_fn as helper
from spyglass.utils.file_backends import (
    BackendUnavailable,
    DandiBackend,
    FileBackend,
    KacheryBackend,
    LocalBackend,
    get_backends,
)


@pytest.fixture(autouse=True)
def clear_nwb_cache():
    """Keep fake IO objects out of the module-level open-file cache.

    `get_nwb_file` caches `Opened` records, and the session teardown calls
    `close()` on every cached io. Fakes used here are not real IO handles.

    Removes only what a test added. The cache is process-wide, so clearing it
    outright would drop real `NWBHDF5IO` handles opened by session setup or an
    earlier test, leaving them unclosed and unreachable at teardown.
    """
    cache = helper.__open_nwb_files
    before = set(cache)

    yield

    for path in set(cache) - before:
        del cache[path]


class _FakeDownloadBackend(FileBackend):
    """Download-only backend, exercising the inherited `open`."""

    name = "fake_dl"
    supports_streaming = False
    supports_download = True

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
    """Streaming-only backend, exercising the inherited `open`."""

    name = "fake_stream"
    supports_streaming = True
    supports_download = False

    def __init__(self, has=True):
        self._has = has
        self.calls = []

    def has(self, nwb_file_path):
        self.calls.append("has")
        return self._has

    def stream(self, nwb_file_path):
        self.calls.append("stream")
        return ("io", "nwbfile")


class _FakeDualBackend(_FakeStreamBackend):
    """Backend that can both stream and download."""

    name = "fake_dual"
    supports_streaming = True
    supports_download = True

    def download(self, nwb_file_path, dest=None):
        self.calls.append("download")
        return True


@pytest.fixture
def prefer_download():
    """Set the download preference for one test, then restore it."""
    from spyglass.settings import sg_config

    prior = sg_config.prefer_download
    sg_config.prefer_download = True
    yield
    sg_config.prefer_download = prior


def test_default_chain_order():
    """Local disk is tried first, then remote sources in fallback order."""
    assert [b.name for b in get_backends()] == ["local", "kachery", "Dandi"]


def test_get_backends_returns_a_copy():
    """Callers cannot mutate the chain in place."""
    got = get_backends()
    got.clear()
    assert [b.name for b in get_backends()] == ["local", "kachery", "Dandi"]


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
        "spyglass.utils.file_backends._open_local_nwb",
        return_value=("io", "nwbfile"),
    ) as mock_open:
        assert backend.open(str(present)) == ("io", "nwbfile", False)
    mock_open.assert_called_once_with(str(present))


def test_download_only_backend_uses_default_open(tmp_path):
    """A backend without `open` downloads, then reads the local copy."""
    backend = _FakeDownloadBackend(download_ok=True)
    target = str(tmp_path / "file.nwb")

    with patch(
        "spyglass.utils.file_backends._open_local_nwb",
        return_value=("io", "nwbfile"),
    ) as mock_open:
        result = backend.open(target)

    assert result == ("io", "nwbfile", False)
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


def test_base_stream_not_implemented(tmp_path):
    """A download-only backend reports that streaming is unavailable."""
    with pytest.raises(NotImplementedError, match="fake_dl"):
        _FakeDownloadBackend().stream(str(tmp_path / "file.nwb"))


def test_streaming_backend_streams_by_default(tmp_path):
    """Absent a preference, a stream-capable backend streams."""
    backend = _FakeDualBackend()
    assert backend.open(str(tmp_path / "file.nwb")) == ("io", "nwbfile", True)
    assert backend.calls == ["stream"]


def test_prefer_download_downloads_instead(tmp_path, prefer_download):
    """With prefer_download set, a dual backend transfers the whole file."""
    backend = _FakeDualBackend()
    target = str(tmp_path / "file.nwb")

    with patch(
        "spyglass.utils.file_backends._open_local_nwb",
        return_value=("io", "nwbfile"),
    ) as mock_open:
        assert backend.open(target) == ("io", "nwbfile", False)

    assert backend.calls == ["download"]
    mock_open.assert_called_once_with(target)


def test_prefer_download_ignored_when_backend_cannot(tmp_path, prefer_download):
    """A preference never breaks a backend that can only stream."""
    backend = _FakeStreamBackend()
    assert backend.open(str(tmp_path / "file.nwb")) == ("io", "nwbfile", True)
    assert backend.calls == ["stream"]


def test_prefer_download_does_not_affect_local(tmp_path, prefer_download):
    """LocalBackend reads from disk regardless of the preference."""
    present = tmp_path / "present.nwb"
    present.write_text("not really nwb")

    with patch(
        "spyglass.utils.file_backends._open_local_nwb",
        return_value=("io", "nwbfile"),
    ) as mock_open:
        assert LocalBackend().open(str(present)) == ("io", "nwbfile", False)
    mock_open.assert_called_once_with(str(present))


def test_dandi_has_delegates_to_resolve(tmp_path):
    """`has` is `_resolve` asked as a predicate, so the two cannot disagree."""
    backend = DandiBackend()
    target = str(tmp_path / "x.nwb")

    with patch.object(DandiBackend, "_resolve", return_value=None):
        assert backend.has(target) is False

    with patch.object(DandiBackend, "_resolve", return_value="x.nwb"):
        assert backend.has(target) is True


def test_dandi_download_reports_absence_without_raising(tmp_path):
    """The bool already carries the answer, so `download` returns it."""
    with patch.object(DandiBackend, "_resolve", return_value=None):
        assert DandiBackend().download(str(tmp_path / "x.nwb")) is False


def test_dandi_stream_raises_when_dandi_lacks_the_file(tmp_path):
    """`stream` has no bool channel, so absence is the one raise left."""
    with patch.object(DandiBackend, "_resolve", return_value=None):
        with pytest.raises(BackendUnavailable, match="not found in Dandi"):
            DandiBackend().stream(str(tmp_path / "x.nwb"))


@pytest.mark.parametrize("backend", get_backends(), ids=lambda b: b.name)
def test_capability_flags_match_implementations(backend):
    """A declared flag that nothing implements would silently misroute `open`.

    The flags are declarations, so nothing but a test keeps them honest.
    """
    cls = type(backend)

    assert backend.supports_streaming == (cls.stream is not FileBackend.stream)
    assert backend.supports_download == (
        cls.download is not FileBackend.download
    )


def test_prefer_download_round_trips_through_settings():
    """The setting accepts the same string forms as other boolean settings."""
    from spyglass.settings import sg_config

    prior = sg_config.prefer_download
    try:
        sg_config.prefer_download = "true"
        assert sg_config.prefer_download is True
        sg_config.prefer_download = "false"
        assert sg_config.prefer_download is False
    finally:
        sg_config.prefer_download = prior


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
    assert second.calls == ["has", "stream"]


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
    assert working.calls == ["has", "stream"]


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


def test_backends_declare_capabilities():
    """Capability is declared, not inferred.

    Dandi declares both, so `prefer_download` is honored there; kachery has no
    streaming path.
    """
    assert KacheryBackend.supports_streaming is False
    assert KacheryBackend.supports_download is True
    assert DandiBackend.supports_streaming is True
    assert DandiBackend.supports_download is True


def test_backend_unavailable_is_a_file_not_found():
    """The miss signal stays catchable as FileNotFoundError for old callers."""
    assert issubclass(BackendUnavailable, FileNotFoundError)


def test_genuine_open_error_propagates(tmp_path):
    """A backend that holds the file but fails to read it does not fall
    through: the real error surfaces instead of a silent recompute."""

    class _BrokenBackend(_FakeStreamBackend):
        name = "broken"

        def stream(self, nwb_file_path):
            raise OSError("truncated file")

    broken = _BrokenBackend(has=True)
    never = _FakeStreamBackend(has=True)
    query = MagicMock()
    target = str(tmp_path / "missing.nwb")

    with (
        patch.object(helper, "get_backends", return_value=[broken, never]),
        patch("os.path.exists", return_value=False),
    ):
        with pytest.raises(OSError, match="truncated file"):
            helper.get_nwb_file(target, query_expression=query)

    assert never.calls == []  # chain halted
    query._make_file.assert_not_called()  # and nothing was recomputed


def test_file_is_remote_false_when_not_open(tmp_path):
    """A path that was never opened is not remote."""
    assert helper.file_is_remote(str(tmp_path / "never_opened.nwb")) is False


def test_file_is_remote_false_for_local_read(tmp_path):
    """A file read from disk is not remote."""
    present = tmp_path / "present.nwb"
    present.write_text("not really nwb")

    with patch(
        "spyglass.utils.file_backends._open_local_nwb",
        return_value=(MagicMock(), "nwbfile"),
    ):
        helper._open_nwb_file(str(present), source=LocalBackend())

    assert helper.file_is_remote(str(present)) is False


def test_file_is_remote_true_for_non_http_streaming(tmp_path):
    """Any streaming backend counts, not only those reading over HTTP.

    The fake streams from something with no HTTPFileSystem anywhere in its
    internals, which the previous heuristic would have misread as local.
    """
    target = str(tmp_path / "streamed.nwb")

    io = MagicMock()
    io._HDF5IO__built = {"root/S3FileSystem-ish": object()}

    class _S3Backend(_FakeStreamBackend):
        name = "fake_s3"

        def stream(self, nwb_file_path):
            return (io, "nwbfile")

    helper._open_nwb_file(target, source=_S3Backend())

    assert helper.file_is_remote(target) is True


def test_file_is_remote_false_when_stream_backend_downloads(
    tmp_path, prefer_download
):
    """A stream-capable backend that downloaded has a local copy to check."""
    target = str(tmp_path / "downloaded.nwb")

    with patch(
        "spyglass.utils.file_backends._open_local_nwb",
        return_value=(MagicMock(), "nwbfile"),
    ):
        helper._open_nwb_file(target, source=_FakeDualBackend())

    assert helper.file_is_remote(target) is False


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
