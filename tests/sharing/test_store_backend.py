"""Tests for the shared-store file backend.

The backend's contract with `get_nwb_file` is narrow: say whether it holds a
readable file, and supply it. What these tests cover is the boundary where
that contract is easy to get wrong — an unconfigured instance, a refusal that
must not stop the chain, and a partial download that must not be mistaken for
a local copy.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from spyglass.utils.file_backends import (
    BackendUnavailable,
    StoreBackend,
    get_backends,
)

RECORD = {"file_id": "f1", "sha256": "ab" * 32, "size_bytes": 7}


def _client(**kwargs):
    """A broker client stub with sensible defaults."""
    defaults = dict(
        configured=True,
        logged_in=True,
        find=lambda name=None, sha256=None: RECORD,
        content_url=lambda file_id: f"https://s.org/api/v1/file/{file_id}/content",
        auth_headers=lambda: {"Authorization": "Bearer tok"},
    )
    return SimpleNamespace(**{**defaults, **kwargs})


@pytest.fixture
def store_module(common):
    """The sharing schema, declared so `_known_hash` has tables to read."""
    from spyglass.sharing import sharing_store

    return sharing_store


@pytest.fixture
def backend():
    """A fresh backend, so the per-process resolve memo starts empty."""
    return StoreBackend()


def _with_client(client):
    return patch(
        "spyglass.sharing.store_client.get_client", return_value=client
    )


# -------------------------------- chain ---------------------------------


def test_store_sits_after_local(backend):
    """A local copy still wins, and DANDI stays the fallback."""
    names = [b.name for b in get_backends()]

    assert names.index("store") == names.index("local") + 1
    assert names.index("store") < names.index("Dandi")


def test_store_declares_both_capabilities(backend):
    """Streaming by default, with prefer_download honored as for DANDI."""
    assert backend.supports_streaming and backend.supports_download


# ------------------------------ resolution ------------------------------


def test_unconfigured_instance_holds_nothing(backend, tmp_path):
    """An instance attached to no broker is ordinary, not an error."""
    with _with_client(_client(configured=False)):
        assert backend.has(str(tmp_path / "a.nwb")) is False


def test_not_logged_in_holds_nothing(backend, tmp_path):
    """A configured broker the user has not logged into is skipped quietly."""
    with _with_client(_client(logged_in=False)):
        assert backend.has(str(tmp_path / "a.nwb")) is False


def test_has_is_true_when_the_broker_resolves(backend, tmp_path):
    with _with_client(_client()):
        assert backend.has(str(tmp_path / "a.nwb")) is True


def test_a_refusal_falls_through(backend, tmp_path):
    """403 and 404 are indistinguishable here; both move to the next backend."""
    with _with_client(_client(find=lambda **kw: None)):
        assert backend.has(str(tmp_path / "a.nwb")) is False


def test_resolution_is_memoized(backend, tmp_path):
    """`has` and the `open` that follows it do not each pay a round trip."""
    calls = []

    def _find(name=None, sha256=None):
        calls.append(name)
        return RECORD

    with _with_client(_client(find=_find)):
        target = str(tmp_path / "a.nwb")
        backend.has(target)
        backend.has(target)

    assert calls == ["a.nwb"]


def test_the_broker_is_asked_by_file_name(backend, tmp_path):
    """The broker knows Spyglass names, not local absolute paths."""
    seen = {}

    def _find(name=None, sha256=None):
        seen["name"] = name
        return RECORD

    with _with_client(_client(find=_find)):
        backend.has(str(tmp_path / "sub" / "minirec20230622_.nwb"))

    assert seen["name"] == "minirec20230622_.nwb"


# ------------------------------- transfer -------------------------------


def test_stream_without_a_readable_file_raises(backend, tmp_path):
    """`stream` owes an (io, nwbfile) pair and has no value for 'no'."""
    with _with_client(_client(find=lambda **kw: None)):
        with pytest.raises(BackendUnavailable, match="not in the shared store"):
            backend.stream(str(tmp_path / "a.nwb"))


def test_download_returns_false_when_unresolvable(backend, tmp_path):
    """A miss is a bool, not an exception; `open` turns it into the error."""
    with _with_client(_client(find=lambda **kw: None)):
        assert backend.download(str(tmp_path / "a.nwb")) is False


def test_download_writes_the_file(backend, tmp_path, monkeypatch):
    """A completed transfer lands at the path Spyglass expects."""
    import requests

    target = tmp_path / "a.nwb"

    monkeypatch.setattr(
        requests, "get", lambda url, **kw: _FakeStream([b"pay", b"load"])
    )

    with _with_client(_client()):
        assert backend.download(str(target)) is True

    assert target.read_bytes() == b"payload"


def test_an_interrupted_download_raises_and_leaves_no_partial_file(
    backend, tmp_path, monkeypatch
):
    """A partial file would look local to the next call and never be retried.

    The broker said it held this file, so a dropped transfer is a failed read.
    Returning False would make it a miss, and `get_nwb_file` would recompute.
    """
    import requests

    target = tmp_path / "a.nwb"

    def _boom(url, **kwargs):
        raise requests.ConnectionError("dropped")

    monkeypatch.setattr(requests, "get", _boom)

    with _with_client(_client()):
        with pytest.raises(requests.ConnectionError, match="dropped"):
            backend.download(str(target))

    assert not target.exists()
    assert list(tmp_path.glob("*.part")) == []


class _FakeStream:
    """Minimal streaming `requests.Response` stand-in."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=None):
        return iter(self._chunks)


def test_a_stale_memo_does_not_outlive_the_login(backend, tmp_path):
    """Logging out mid-session declines the transfer rather than crashing."""
    target = str(tmp_path / "a.nwb")

    with _with_client(_client()):
        assert backend.has(target) is True  # memoize the record

    with _with_client(_client(logged_in=False)):  # user logged out
        assert backend.download(target) is False
        with pytest.raises(BackendUnavailable):
            backend.stream(target)


def test_a_login_mid_session_is_not_blocked_by_a_cached_no(backend, tmp_path):
    """The chain is built once at import; the notebook logs in afterward.

    Caching "not in the store" from a probe made before `store_url` was set
    would make every file touched before login permanently unavailable.
    """
    target = str(tmp_path / "a.nwb")

    with _with_client(_client(configured=False)):
        assert backend.has(target) is False

    with _with_client(_client()):  # user configures a broker and logs in
        assert backend.has(target) is True


# ------------------------- unambiguous resolution -------------------------


def test_resolution_prefers_a_recorded_hash(backend, tmp_path, monkeypatch):
    """A name the broker indexes per-owner is settled by the local digest.

    Two owners can register the same `spyglass_name`, and the broker returns
    whichever row it finds first with no owner field to tell them apart. The
    Spyglass database keys the upload on the file name, so the digest it
    recorded names the bytes rather than anyone's registration of them.
    """
    asked = {}

    def _find(name=None, sha256=None):
        asked.update(name=name, sha256=sha256)
        return RECORD

    monkeypatch.setattr(StoreBackend, "_known_hash", lambda self, n: "cd" * 32)

    with _with_client(_client(find=_find)):
        backend.has(str(tmp_path / "a.nwb"))

    assert asked == {"name": None, "sha256": "cd" * 32}


def test_resolution_falls_back_to_the_name(backend, tmp_path, monkeypatch):
    """A file shared from another Spyglass instance has no local digest."""
    asked = {}

    def _find(name=None, sha256=None):
        asked.update(name=name, sha256=sha256)
        return RECORD

    monkeypatch.setattr(StoreBackend, "_known_hash", lambda self, n: None)

    with _with_client(_client(find=_find)):
        backend.has(str(tmp_path / "a.nwb"))

    assert asked == {"name": "a.nwb", "sha256": None}


def test_an_unreachable_sharing_schema_falls_back(backend, monkeypatch):
    """No grants, no schema, no connection: resolve by name rather than fail.

    `_known_hash` is an improvement on name resolution, not a requirement for
    it. An instance that cannot read the sharing tables must still be able to
    fetch a file.
    """
    import builtins

    real_import = builtins.__import__

    def _refuse(name, *args, **kwargs):
        if name == "spyglass.sharing.sharing_store":
            raise RuntimeError("command denied to user")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _refuse)

    assert backend._known_hash("a.nwb") is None


def test_an_unshared_file_has_no_recorded_hash(backend, store_module):
    """A name this instance never uploaded resolves by name, as before."""
    assert backend._known_hash("never-shared-by-anyone.nwb") is None


def test_a_miss_is_not_cached(backend, tmp_path):
    """A refusal, a miss and an outage are all None; caching one strands it.

    The chain holds one instance for the process, so a file probed before it
    is shared must resolve once it is.
    """
    target = str(tmp_path / "later.nwb")
    client = _client()
    client.find = lambda **kw: None

    with _with_client(client):
        assert backend.has(target) is False

    client.find = lambda **kw: {"file_id": "f1"}

    with _with_client(client):
        assert backend.has(target) is True, "Cached the earlier miss"


def test_a_local_query_failure_falls_back_to_the_name(backend, tmp_path):
    """`_known_hash` is an optimization; a denied SELECT must not break `has`.

    Without this a connection loss or a missing grant escapes the chain,
    stopping the fallback to DANDI.
    """
    import datajoint as dj

    asked = []
    client = _client()
    client.find = lambda **kw: asked.append(kw) or {"file_id": "f1"}

    denied = dj.errors.AccessError("Insufficient privileges.", "", "")
    target = tmp_path / "a.nwb"

    with patch(
        "spyglass.sharing.sharing_store.SharedFile.fetch", side_effect=denied
    ):
        with _with_client(client):
            assert backend.has(str(target)) is True

    assert asked == [{"name": target.name}], "Did not fall back to the name"
