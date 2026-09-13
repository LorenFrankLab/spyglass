"""Tests for the shared-storage broker client.

Everything here runs against a fake transport rather than a live broker. The
client's job is to turn the broker's vocabulary of status codes into a typed
answer, so what is worth testing is that translation, not HTTP.
"""

import json
import stat
from types import SimpleNamespace

import pytest

from spyglass.sharing import store_client as sc

BROKER = "https://store.example.org"
TOKEN = {"access_token": "tok", "tier": "verified", "github_login": "someone"}


class _FakeResponse:
    """Minimal stand-in for `requests.Response`."""

    def __init__(self, status_code=200, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text if text else json.dumps(payload or {})

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.fixture
def token_file(tmp_path, monkeypatch):
    """Point the token cache at a temporary file."""
    path = tmp_path / "store_token.json"
    monkeypatch.setenv("SPYGLASS_STORE_TOKEN", str(path))
    return path


@pytest.fixture
def client(token_file):
    """A logged-in client for a fake broker."""
    return sc.StoreClient(base_url=BROKER, token="tok")


@pytest.fixture
def transport(monkeypatch):
    """Replace `requests.request` with a scripted responder.

    Returns a list to append responses to, and records each call so a test can
    assert on the method, URL, and headers the client chose.
    """
    import requests

    scripted, calls = [], []

    def _fake(method, url, **kwargs):
        calls.append(SimpleNamespace(method=method, url=url, **kwargs))
        return scripted.pop(0)

    monkeypatch.setattr(requests, "request", _fake)

    return SimpleNamespace(scripted=scripted, calls=calls)


# --------------------------------- urls ---------------------------------


def test_url_is_versioned(client):
    """Every route sits below the versioned prefix."""
    assert client.url("/file/resolve") == f"{BROKER}/api/v1/file/resolve"


def test_trailing_slash_does_not_double(token_file):
    """A configured URL written with a slash still builds one clean path."""
    assert sc.StoreClient(base_url=BROKER + "/").url("/file") == (
        f"{BROKER}/api/v1/file"
    )


def test_unconfigured_client_refuses_to_build_a_url(token_file):
    """An instance with no broker says so rather than calling nothing."""
    unset = sc.StoreClient(base_url="")

    assert unset.configured is False
    with pytest.raises(sc.StoreNotConfigured):
        unset.url("/file/resolve")


def test_content_url_is_the_stable_one(client):
    """The client holds the broker URL, not a signature that can expire."""
    assert client.content_url("f1") == f"{BROKER}/api/v1/file/f1/content"


# --------------------------------- token --------------------------------


def test_token_file_is_owner_only(token_file):
    """A broker credential is written at 0600, never world-readable."""
    sc._write_token(BROKER, TOKEN)

    mode = stat.S_IMODE(token_file.stat().st_mode)
    assert mode == stat.S_IRUSR | stat.S_IWUSR


def test_tokens_are_kept_per_broker(token_file):
    """A second broker does not overwrite the first one's token."""
    sc._write_token(BROKER, TOKEN)
    sc._write_token("https://other.org", {"access_token": "other"})

    assert sc.StoreClient(base_url=BROKER).token == "tok"
    assert sc.StoreClient(base_url="https://other.org").token == "other"


def test_unreadable_token_cache_is_ignored(token_file):
    """A corrupted cache sends the user back to login, it does not raise."""
    token_file.write_text("{ not json")

    assert sc.StoreClient(base_url=BROKER).logged_in is False


def test_logout_forgets_only_this_broker(token_file):
    """Logging out of one broker leaves the other credential in place."""
    sc._write_token(BROKER, TOKEN)
    sc._write_token("https://other.org", {"access_token": "other"})

    sc.StoreClient(base_url=BROKER).logout()

    assert sc.StoreClient(base_url=BROKER).logged_in is False
    assert sc.StoreClient(base_url="https://other.org").token == "other"


def test_call_without_a_token_names_the_fix(token_file):
    """The error tells the user to log in rather than reporting a 401."""
    with pytest.raises(sc.StoreAuthError, match="spyglass-store login"):
        sc.StoreClient(base_url=BROKER).auth_headers()


# -------------------------------- statuses ------------------------------


@pytest.mark.parametrize(
    "status, expected",
    [
        (401, sc.StoreAuthError),
        (403, sc.StoreForbidden),
        (404, sc.StoreNotFound),
        (428, sc.StorePending),
        (429, sc.StoreQuotaExceeded),
        (500, sc.StoreError),
    ],
)
def test_status_becomes_a_typed_error(client, transport, status, expected):
    """Each status the broker uses maps to one exception type."""
    transport.scripted.append(
        _FakeResponse(status, {"detail": "nope"}, {"Retry-After": "7"})
    )

    with pytest.raises(expected):
        client.resolve(name="a.nwb")


def test_quota_error_carries_the_brokers_own_retry_estimate(client, transport):
    """Retry-After beats a fixed interval that stampedes every client."""
    transport.scripted.append(
        _FakeResponse(429, {"detail": "exhausted"}, {"Retry-After": "42"})
    )

    with pytest.raises(sc.StoreQuotaExceeded) as err:
        client.resolve(name="a.nwb")

    assert err.value.retry_after == 42


def test_transport_failure_is_a_store_error(client, monkeypatch):
    """A broker that cannot be reached is not a missing file."""
    import requests

    def _boom(*args, **kwargs):
        raise requests.ConnectionError("no route")

    monkeypatch.setattr(requests, "request", _boom)

    with pytest.raises(sc.StoreError, match="Could not reach the broker"):
        client.resolve(name="a.nwb")


# --------------------------------- files --------------------------------


def test_resolve_prefers_hash_over_name(client, transport):
    """A hash is unambiguous; a Spyglass name is not unique across owners."""
    transport.scripted.append(_FakeResponse(200, {"file_id": "f1"}))

    client.resolve(name="a.nwb", sha256="ab" * 32)

    assert transport.calls[0].params == {"sha256": "ab" * 32}


def test_resolve_needs_one_of_name_or_hash(client):
    """Neither argument is a caller bug, not a broker round trip."""
    with pytest.raises(ValueError, match="name or sha256"):
        client.resolve()


def test_bearer_token_is_sent(client, transport):
    """Every authenticated route carries the broker token."""
    transport.scripted.append(_FakeResponse(200, {"file_id": "f1"}))

    client.resolve(name="a.nwb")

    assert transport.calls[0].headers["Authorization"] == "Bearer tok"


@pytest.mark.parametrize(
    "status", [404, 403, 401]
)  # missing, refused, logged out
def test_find_collapses_miss_and_refusal(client, transport, status):
    """The resolution chain moves on from all three the same way."""
    transport.scripted.append(_FakeResponse(status, {"detail": "no"}))

    assert client.find(name="a.nwb") is None


def test_find_returns_none_when_the_broker_is_down(client, transport):
    """An unreachable broker must not stop the chain from trying DANDI."""
    transport.scripted.append(_FakeResponse(503, {"detail": "down"}))

    assert client.find(name="a.nwb") is None


def test_register_declares_visibility(client, transport):
    """Scope and teams travel with the registration, not afterward."""
    transport.scripted.append(_FakeResponse(201, {"file_id": "f1"}))

    client.register(
        sha256="ab" * 32,
        size_bytes=10,
        spyglass_name="a.nwb",
        file_class="raw",
        scope="group",
        teams=["My Team"],
    )

    body = transport.calls[0].json
    assert body["visibility"] == {"scope": "group", "teams": ["My Team"]}
    assert body["file_class"] == "raw"


def test_upload_skips_transfer_when_deduplicated(client, transport, tmp_path):
    """Identical bytes already in the store need a registration only."""
    path = tmp_path / "a.nwb"
    path.write_bytes(b"payload")

    transport.scripted.append(
        _FakeResponse(201, {"file_id": "f1", "deduplicated": True})
    )

    result = client.upload(str(path))

    assert result["deduplicated"] is True
    assert len(transport.calls) == 1  # register only, no PUT


def test_upload_sends_the_signed_checksum_headers(
    client, transport, tmp_path, monkeypatch
):
    """Upload headers are covered by the signature and cannot be dropped."""
    import requests

    path = tmp_path / "a.nwb"
    path.write_bytes(b"payload")

    transport.scripted.append(
        _FakeResponse(
            201,
            {
                "file_id": "f1",
                "deduplicated": False,
                "upload_url": "https://obj.example.org/put",
                "upload_headers": {"x-amz-checksum-sha256": "abc="},
            },
        )
    )

    puts = []

    def _fake_put(url, **kwargs):
        puts.append(SimpleNamespace(url=url, **kwargs))
        return _FakeResponse(200, {})

    monkeypatch.setattr(requests, "put", _fake_put)

    client.upload(str(path))

    assert puts[0].headers == {"x-amz-checksum-sha256": "abc="}


def test_upload_reports_a_refused_object(
    client, transport, tmp_path, monkeypatch
):
    """A checksum mismatch surfaces as an error, not a silent success."""
    import requests

    path = tmp_path / "a.nwb"
    path.write_bytes(b"payload")

    transport.scripted.append(
        _FakeResponse(
            201,
            {
                "file_id": "f1",
                "deduplicated": False,
                "upload_url": "https://obj.example.org/put",
                "upload_headers": {},
            },
        )
    )
    monkeypatch.setattr(
        requests, "put", lambda url, **kw: _FakeResponse(400, text="bad digest")
    )

    with pytest.raises(sc.StoreError, match="refused"):
        client.upload(str(path))


def test_upload_of_a_missing_file_says_so(client, tmp_path):
    """The client checks before hashing a file that is not there."""
    with pytest.raises(FileNotFoundError):
        client.upload(str(tmp_path / "absent.nwb"))


def test_set_visibility_sends_scope_and_teams(client, transport):
    """Widening access is one PATCH against the file the broker knows."""
    transport.scripted.append(_FakeResponse(200, {"file_id": "f1"}))

    client.set_visibility("f1", scope="public")

    assert transport.calls[0].method == "PATCH"
    assert transport.calls[0].json == {"scope": "public", "teams": []}


# --------------------------------- login --------------------------------


def test_login_polls_until_approved(client, transport, token_file, monkeypatch):
    """428 is 'still waiting', and is the only status the loop continues on."""
    monkeypatch.setattr(sc.time, "sleep", lambda _: None)

    transport.scripted.extend(
        [
            _FakeResponse(
                200,
                {
                    "device_code": "dc",
                    "user_code": "WXYZ-1234",
                    "verification_uri": "https://github.com/login/device",
                    "interval": 1,
                    "expires_in": 900,
                },
            ),
            _FakeResponse(428, {"detail": "pending"}, {"Retry-After": "1"}),
            _FakeResponse(200, TOKEN),
        ]
    )

    record = client.login()

    assert record["access_token"] == "tok"
    assert json.loads(token_file.read_text())[BROKER] == TOKEN


def test_login_stops_on_a_real_refusal(client, transport, monkeypatch):
    """A too-young GitHub account fails now, not after fifteen minutes."""
    monkeypatch.setattr(sc.time, "sleep", lambda _: None)

    transport.scripted.extend(
        [
            _FakeResponse(
                200,
                {
                    "device_code": "dc",
                    "user_code": "WXYZ-1234",
                    "verification_uri": "https://github.com/login/device",
                    "interval": 1,
                    "expires_in": 900,
                },
            ),
            _FakeResponse(403, {"detail": "account is 2 days old"}),
        ]
    )

    with pytest.raises(sc.StoreForbidden, match="2 days old"):
        client.login()


def test_login_does_not_send_a_token(client, transport, monkeypatch):
    """The two auth routes are what produce a token; they cannot need one."""
    monkeypatch.setattr(sc.time, "sleep", lambda _: None)

    transport.scripted.extend(
        [
            _FakeResponse(
                200,
                {
                    "device_code": "dc",
                    "user_code": "WXYZ-1234",
                    "verification_uri": "https://github.com/login/device",
                    "interval": 1,
                    "expires_in": 900,
                },
            ),
            _FakeResponse(200, TOKEN),
        ]
    )

    sc.StoreClient(base_url=BROKER).login()  # no token at all

    assert "Authorization" not in transport.calls[0].headers


def test_cli_status_reports_not_logged_in(token_file, capsys):
    """`spyglass-store status` exits non-zero before a first login."""
    assert sc.login_cli(["status", "--url", BROKER]) == 1


# -------------------------------- config --------------------------------


def test_store_url_round_trips_through_settings():
    """The broker URL is a session setting like any other custom key."""
    from spyglass.settings import sg_config

    prior = sg_config.store_url
    try:
        sg_config.store_url = "https://store.example.org/"
        assert sg_config.store_url == "https://store.example.org"
        assert sg_config.config["store_url"] == "https://store.example.org"

        sg_config.store_url = None  # detaching is not an error
        assert sg_config.store_url == ""
    finally:
        sg_config.store_url = prior


def test_an_unset_store_url_is_the_default(token_file):
    """Most instances are attached to no broker; that is not a failure."""
    from spyglass.settings import sg_config

    prior = sg_config.store_url
    try:
        sg_config.store_url = ""
        assert sc.StoreClient().configured is False
    finally:
        sg_config.store_url = prior


def test_find_lets_a_quota_refusal_through(client, transport):
    """Throttled is not missing.

    A 429 means the file exists and is readable, only not right now.
    Collapsing it to None would send `get_nwb_file` off to recompute
    something it could have waited for, and would make the `except
    StoreQuotaExceeded` the Data Sync notebook documents unreachable.
    """
    transport.scripted.append(
        _FakeResponse(429, {"detail": "exhausted"}, {"Retry-After": "30"})
    )

    with pytest.raises(sc.StoreQuotaExceeded) as err:
        client.find(name="a.nwb")

    assert err.value.retry_after == 30
