"""Client for the shared-storage broker.

The broker is a small self-hosted service that decides who may read a file and
hands back a short-lived signed URL for it. Spyglass never holds an
object-store credential: everything here goes through the broker's versioned
HTTP API, and the only secret this module keeps is a broker token.

Six endpoints make up the contract:

===============================  =============================================
``POST /auth/device``            Begin a login, returning a code to type.
``POST /auth/token``             Poll until the user has approved it.
``GET  /file/resolve``           Name or hash to a file record.
``POST /file``                   Register an upload, get somewhere to put it.
``GET  /file/{id}/content``      Redirect to a freshly signed URL.
``PATCH /file/{id}/visibility``  Change who may read a file. Owner only.
===============================  =============================================

Login is GitHub's device flow, run by the broker on our behalf. There is no
browser callback, so it works unchanged over SSH and inside a container: the
user types a code at ``github.com/login/device`` and this module polls until
the broker stops answering 428. What comes back is a *broker* token — it can
read nothing on GitHub, and the GitHub token it was exchanged for never
reaches this process.

Notes
-----
The broker returns no expiry and offers no refresh endpoint, so a token is
used until it is rejected. `StoreAuthError` on a call the token used to be
allowed to make means "run `spyglass-store login` again", and is worded that
way.
"""

import json
import os
import stat
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

from spyglass.utils.logging import logger

#: Versioned API prefix. The path carries the version so that a broker upgrade
#: cannot break a pinned client — which matters more than usual here, because
#: we do not control when a user upgrades Spyglass.
API_PREFIX = "/api/v1"

#: Where the broker token is cached, unless `SPYGLASS_STORE_TOKEN` overrides
#: it. One file holds every broker the user has logged into, keyed by base URL,
#: so pointing at a second instance does not silently present the first one's
#: token.
DEFAULT_TOKEN_PATH = Path.home() / ".spyglass" / "store_token.json"

#: Seconds to wait for an ordinary API call. Generous, because the broker
#: reaches a database and an object store before it answers.
DEFAULT_TIMEOUT = 30


class StoreError(Exception):
    """The broker could not do what was asked.

    Base of every error this module raises, so a caller that does not care
    which way a broker call failed can catch one type.
    """


class StoreNotConfigured(StoreError):
    """No broker is configured for this Spyglass instance.

    Not a misconfiguration. Most instances are attached to no broker, and the
    shared-store backend reads this as "I hold nothing" and steps aside.
    """


class StoreAuthError(StoreError):
    """The broker did not accept this identity.

    Raised for a missing token and for one the broker has stopped honoring.
    Since there is no refresh endpoint, recovery is always a fresh login.
    """


class StoreForbidden(StoreError):
    """The broker refused this identity the thing it asked for.

    Distinct from `StoreNotFound` here even though the file backend treats the
    two alike, because a caller who declared a share wants to be told that the
    refusal was about permission and not about a missing file.
    """


class StoreNotFound(StoreError):
    """The broker holds no such file."""


class StorePending(StoreError):
    """The user has not yet approved the login at GitHub.

    The broker answers 428 for this, which is not a failure — it is the
    expected state for as long as the user is still typing the code. Its own
    type so that the polling loop can continue on this and only this, and let
    every real refusal out immediately.

    Attributes
    ----------
    interval : int
        Seconds the broker asks the client to wait before polling again.
    """

    def __init__(self, message: str, interval: int = 5):
        super().__init__(message)
        self.interval = interval


class StoreQuotaExceeded(StoreError):
    """This read would exceed the tier's allowance.

    Expected rather than exceptional on an unverified account. `retry_after`
    carries the broker's own estimate of when capacity frees up, which beats a
    fixed interval that would have every client retry at the same moment.

    Attributes
    ----------
    retry_after : int
        Seconds until the oldest counted read ages out.
    """

    def __init__(self, message: str, retry_after: int = 0):
        super().__init__(message)
        self.retry_after = retry_after


def token_path() -> Path:
    """Return the file the broker token is cached in.

    `SPYGLASS_STORE_TOKEN` takes precedence, so a container or a shared
    account can place the credential somewhere other than a home directory
    that may not persist.

    Returns
    -------
    pathlib.Path
        Path to the token file. May not exist yet.
    """
    override = os.environ.get("SPYGLASS_STORE_TOKEN")
    return Path(override).expanduser() if override else DEFAULT_TOKEN_PATH


def _read_tokens() -> dict:
    """Return every cached broker credential, keyed by base URL.

    A file that is missing, empty, or unreadable yields an empty mapping: a
    corrupted cache should send the user back through login, not raise out of
    an unrelated call.

    Returns
    -------
    dict
        Mapping of base URL to credential record.
    """
    path = token_path()

    if not path.exists():
        return {}

    try:
        stored = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning(f"Ignoring unreadable store token cache: {path}")
        return {}

    return stored if isinstance(stored, dict) else {}


def _write_token(base_url: str, record: dict) -> None:
    """Cache one broker's credential at 0600.

    The mode is set on the descriptor before anything is written, rather than
    chmod'ed afterward, so the token is never briefly world-readable.

    Parameters
    ----------
    base_url : str
        Broker this credential belongs to.
    record : dict
        The broker's token response.
    """
    path = token_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    tokens = _read_tokens()
    tokens[base_url] = record

    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR
    )
    with os.fdopen(fd, "w") as f:
        json.dump(tokens, f, indent=2)

    # An existing file keeps its old mode through O_CREAT, so narrow it too.
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


class StoreClient:
    """A connection to one shared-storage broker.

    Holds a base URL and a token; every method is one HTTP call against the
    versioned API. Nothing here decides what a user may read — the broker does
    that, and this class turns its answer into an exception or a value.

    Parameters
    ----------
    base_url : str, optional
        Broker origin, e.g. ``https://store.example.org``. Defaults to
        `sg_config.store_url`.
    token : str, optional
        Broker token. Defaults to the cached credential for `base_url`.
    timeout : int, optional
        Per-request timeout in seconds.

    Attributes
    ----------
    base_url : str
        Broker origin, without a trailing slash. "" if none is configured.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        from spyglass.settings import _clean_store_url

        if base_url is None:  # only then is a loaded config needed
            from spyglass.settings import sg_config

            base_url = sg_config.store_url

        self.base_url = _clean_store_url(base_url)
        self.timeout = timeout
        self._token = token
        self._record = None
        self._info = None

    # ------------------------------- state -------------------------------

    @property
    def configured(self) -> bool:
        """True if a broker URL is set.

        Returns
        -------
        bool
            False when this instance is attached to no broker.
        """
        return bool(self.base_url)

    @property
    def record(self) -> dict:
        """The cached credential for this broker, or an empty dict.

        Returns
        -------
        dict
            Keys `access_token`, `tier`, and `github_login` when logged in.
        """
        if self._record is None:
            self._record = _read_tokens().get(self.base_url, {})
        return self._record

    @property
    def token(self) -> Optional[str]:
        """The broker token, from the constructor or the cache.

        Returns
        -------
        str or None
            None when the user has never logged in to this broker.
        """
        return self._token or self.record.get("access_token") or None

    @property
    def tier(self) -> str:
        """The tier this token was issued at, as of the last login.

        Advisory only. The broker re-reads the tier on every call, so an admin
        promotion takes effect without a new login and this value can be stale.

        Returns
        -------
        str
            Tier name, or "" when not logged in.
        """
        return self.record.get("tier", "")

    @property
    def github_login(self) -> str:
        """The GitHub account this token was issued to.

        Returns
        -------
        str
            Login name, or "" when not logged in.
        """
        return self.record.get("github_login", "")

    @property
    def logged_in(self) -> bool:
        """True if a token is available for this broker.

        Says nothing about whether the broker still honors it. Only a call can
        answer that.

        Returns
        -------
        bool
        """
        return bool(self.token)

    # ------------------------------ plumbing -----------------------------

    def url(self, path: str) -> str:
        """Return the absolute URL for an API path.

        Parameters
        ----------
        path : str
            Path below the version prefix, e.g. ``/file/resolve``.

        Returns
        -------
        str
            Absolute URL.

        Examples
        --------
        >>> StoreClient(base_url="https://s.org/").url("/file/resolve")
        'https://s.org/api/v1/file/resolve'
        """
        if not self.configured:
            raise StoreNotConfigured(
                "No shared-storage broker is configured. Set `store_url` in "
                + "dj_local_conf.json, or `sg_config.store_url = ...`."
            )
        return urljoin(self.base_url + "/", (API_PREFIX + path).lstrip("/"))

    def auth_headers(self) -> dict:
        """Return the Authorization header for this broker.

        Returns
        -------
        dict
            Bearer header.

        Raises
        ------
        StoreAuthError
            If no token is available.
        """
        if not self.token:
            raise StoreAuthError(
                "Not logged in to the shared store. Run `spyglass-store "
                + "login`, or `StoreClient().login()`."
            )
        return {"Authorization": f"Bearer {self.token}"}

    def _request(self, method: str, path: str, auth: bool = True, **kwargs):
        """Make one API call and translate the broker's status into an error.

        The mapping is the whole point of this method: every route answers
        with the same vocabulary of statuses, and turning them into exceptions
        once means no caller has to inspect a status code.

        Parameters
        ----------
        method : str
            HTTP method.
        path : str
            Path below the version prefix.
        auth : bool, optional
            Send the bearer token. False only for the two login routes, which
            are what produce it.
        **kwargs
            Passed through to `requests.request`.

        Returns
        -------
        requests.Response
            A response with a 2xx or 3xx status.

        Raises
        ------
        StorePending, StoreAuthError, StoreForbidden, StoreNotFound,
        StoreQuotaExceeded
            As the status warrants.
        StoreError
            For anything else the broker refused, and for a transport failure.
        """
        import requests

        headers = {**(kwargs.pop("headers", None) or {})}
        if auth:
            headers.update(self.auth_headers())

        try:
            response = requests.request(
                method,
                self.url(path),
                headers=headers,
                timeout=self.timeout,
                **kwargs,
            )
        except requests.RequestException as err:
            raise StoreError(f"Could not reach the broker: {err}") from err

        if response.ok:
            return response

        detail = self._detail(response)

        if response.status_code == 428:
            raise StorePending(
                detail, interval=int(response.headers.get("Retry-After", 5))
            )
        if response.status_code == 401:
            raise StoreAuthError(
                f"The broker did not accept this token ({detail}). Run "
                + "`spyglass-store login` again; broker tokens cannot be "
                + "refreshed."
            )
        if response.status_code == 403:
            raise StoreForbidden(detail)
        if response.status_code == 404:
            raise StoreNotFound(detail)
        if response.status_code == 429:
            raise StoreQuotaExceeded(
                detail, retry_after=int(response.headers.get("Retry-After", 0))
            )

        raise StoreError(f"Broker returned {response.status_code}: {detail}")

    @staticmethod
    def _detail(response) -> str:
        """Return the broker's explanation for a refusal.

        Every broker error carries a JSON `detail`. A proxy standing in front
        of it may not, so the raw body is the fallback.

        Parameters
        ----------
        response : requests.Response
            A non-2xx response.

        Returns
        -------
        str
            Human-readable reason.
        """
        try:
            return str(response.json().get("detail", response.text))
        except ValueError:
            return response.text.strip()[:200]

    # -------------------------------- auth -------------------------------

    def login(self, poll_interval: Optional[int] = None) -> dict:
        """Run GitHub device flow against the broker and cache the token.

        Prints the code and the URL, then blocks until the user approves or
        the code expires. There is no browser callback, so this works
        unchanged over SSH and inside a container.

        Parameters
        ----------
        poll_interval : int, optional
            Seconds between polls. Defaults to the interval the broker asks
            for, which is GitHub's — polling faster earns a slow-down, not a
            faster answer.

        Returns
        -------
        dict
            The token record: `access_token`, `tier`, `github_login`.

        Raises
        ------
        StoreError
            If the code expires before the user approves it, or the broker
            refuses the resulting account.
        """
        start = self._request("POST", "/auth/device", auth=False).json()

        logger.info(
            f"\n  To finish logging in, open {start['verification_uri']}"
            + f"\n  and enter the code: {start['user_code']}\n"
        )

        interval = poll_interval or start.get("interval", 5)
        deadline = time.monotonic() + start.get("expires_in", 900)

        while time.monotonic() < deadline:
            time.sleep(interval)
            try:
                record = self._request(
                    "POST",
                    "/auth/token",
                    auth=False,
                    json={"device_code": start["device_code"]},
                ).json()
            except StorePending as pending:
                # The only outcome this loop continues on. Every other refusal
                # — an expired code, a GitHub account below the minimum age —
                # surfaces now rather than after another quarter hour of
                # polling. The broker may also ask us to slow down.
                interval = max(interval, pending.interval)
                continue

            _write_token(self.base_url, record)
            self._record = record
            self._token = record["access_token"]

            logger.info(
                f"Logged in to {self.base_url} as {record.get('github_login')}"
                + f" (tier: {record.get('tier')})."
            )
            return record

        raise StoreError(
            "The login code expired before it was approved. Run login again."
        )

    def logout(self) -> None:
        """Forget the cached credential for this broker.

        The broker is not told. Its token remains valid until an admin revokes
        it; this only removes the local copy.
        """
        tokens = _read_tokens()
        if tokens.pop(self.base_url, None) is None:
            return

        path = token_path()
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        with os.fdopen(fd, "w") as f:
            json.dump(tokens, f, indent=2)

        self._record, self._token = {}, None

    # -------------------------------- files ------------------------------

    def resolve(
        self, name: Optional[str] = None, sha256: Optional[str] = None
    ) -> dict:
        """Look up a file by Spyglass name or by content hash.

        Prefer `sha256` where it is known. Nothing enforces uniqueness of a
        Spyglass name across owners, so a name resolves to whichever matching
        registration the broker returns first, and the response carries no
        owner to disambiguate with.

        Parameters
        ----------
        name : str, optional
            Spyglass file name, e.g. ``minirec20230622_.nwb``.
        sha256 : str, optional
            Hex digest of the file's bytes. Takes precedence over `name`.

        Returns
        -------
        dict
            Keys `file_id`, `sha256`, `size_bytes`, `spyglass_name`,
            `file_class`.

        Raises
        ------
        ValueError
            If neither argument is given.
        StoreNotFound
            If the broker holds no such file.
        StoreForbidden
            If this identity may not read it.
        """
        if not name and not sha256:
            raise ValueError("Pass one of name or sha256.")

        params = {"sha256": sha256} if sha256 else {"name": name}

        return self._request("GET", "/file/resolve", params=params).json()

    def find(
        self, name: Optional[str] = None, sha256: Optional[str] = None
    ) -> Optional[dict]:
        """Resolve a file, or return None if it is unavailable.

        A refusal and a miss are deliberately collapsed here, because the
        resolution chain does the same thing with both: move on to the next
        backend. Note what that costs — a file the user could read after
        linking their GitHub account is indistinguishable from one that does
        not exist, so `resolve` remains available for callers that need to
        tell the two apart.

        Being throttled is **not** collapsed. A quota refusal means the file
        exists and is readable, only not right now, and answering "not found"
        would send the caller off to recompute something it could have waited
        for. `StoreQuotaExceeded` propagates.

        Parameters
        ----------
        name : str, optional
            Spyglass file name.
        sha256 : str, optional
            Hex digest of the file's bytes.

        Returns
        -------
        dict or None
            The file record, or None if missing, refused, or unreachable.

        Raises
        ------
        StoreQuotaExceeded
            If this read would exceed the tier's allowance.
        """
        try:
            return self.resolve(name=name, sha256=sha256)
        except StoreQuotaExceeded:
            raise
        except StoreAuthError as err:
            # Not debug. The user has a token the broker has stopped
            # honoring, and every later call will do the same thing; saying
            # so once beats a silent fall-through to recompute.
            logger.warning(
                f"Shared store rejected this login ({err}). "
                + "Run `spyglass-store login` again."
            )
            return None
        except (StoreNotFound, StoreForbidden) as err:
            logger.debug(f"Store has no readable {name or sha256}: {err}")
            return None
        except StoreError as err:  # broker down, DNS, proxy
            logger.warning(f"Shared store unavailable: {err}")
            return None

    def content_url(self, file_id: str) -> str:
        """Return the stable URL for a file's bytes.

        Stable, not signed: each request to it is answered with a *fresh*
        redirect. Holding this rather than the signed URL it redirects to is
        what lets a multi-gigabyte read outlive a single signature.

        Parameters
        ----------
        file_id : str
            Broker file id, from `resolve`.

        Returns
        -------
        str
            Absolute URL. Requires the bearer header to fetch.
        """
        return self.url(f"/file/{file_id}/content")

    def info(self) -> dict:
        """Return the broker's description of itself, fetched once.

        Memoized per instance, never on disk: a restart must pick up a
        backend change, since a stale answer means uploads that look verified
        and are not.

        Returns
        -------
        dict
            At least `upload_digests`. A broker without this route, or one
            unreachable, yields both digests.
        """
        if self._info is None:
            try:
                self._info = self._request("GET", "/info").json()
            except StoreError as err:  # includes 404 on an older broker
                from spyglass.utils.nwb_hash import UPLOAD_DIGESTS

                logger.debug(f"Broker /info unavailable ({err}); both.")
                self._info = {"upload_digests": list(UPLOAD_DIGESTS)}

        return self._info

    def upload_digests(self) -> List[str]:
        """Return the digest names to compute and send on registration.

        Names this client does not implement are dropped, so a newer broker
        naming an unknown digest still works.

        Returns
        -------
        list of str
            Always includes "sha256", the object's address.
        """
        from spyglass.utils.nwb_hash import DIGESTS

        declared = self.info().get("upload_digests") or []
        known = [name for name in declared if name in DIGESTS]

        return sorted(set(known) | {"sha256"})

    def register(
        self,
        sha256: str,
        size_bytes: int,
        spyglass_name: str,
        file_class: str = "analysis",
        scope: str = "private",
        teams: Optional[List[str]] = None,
        content_md5: Optional[str] = None,
    ) -> dict:
        """Declare an upload and ask where to put the bytes.

        Registration is per owner: two people registering the same bytes get
        separate rows over one shared object.

        Parameters
        ----------
        sha256 : str
            Hex digest of the file's bytes, from `sha256_file`.
        size_bytes : int
            File size.
        spyglass_name : str
            The name Spyglass knows the file by.
        file_class : str, optional
            "raw" or "analysis".
        scope : str, optional
            "private", "group", or "public".
        teams : list of str, optional
            `LabTeam` names, required when `scope` is "group".
        content_md5 : str, optional
            Hex MD5 of the same bytes, signed into the presigned URL as
            `Content-MD5`. Omitted from the body when unknown, not sent null.

        Returns
        -------
        dict
            Keys `file_id`, `deduplicated`, and — when the object is not
            already stored — `upload_url` and `upload_headers`.

        Raises
        ------
        StoreForbidden
            If this tier may not upload. An unverified account cannot.
        """
        body = {
            "sha256": sha256,
            "size_bytes": size_bytes,
            "spyglass_name": spyglass_name,
            "file_class": file_class,
            "visibility": {"scope": scope, "teams": list(teams or [])},
        }

        if content_md5:  # optional on the wire; a null is worse than absent
            body["content_md5"] = content_md5

        return self._request("POST", "/file", json=body).json()

    def upload(
        self,
        file_path: str,
        spyglass_name: Optional[str] = None,
        file_class: str = "analysis",
        scope: str = "private",
        teams: Optional[List[str]] = None,
        sha256: Optional[str] = None,
        content_md5: Optional[str] = None,
    ) -> dict:
        """Register a local file and transfer its bytes if they are new.

        Parameters
        ----------
        file_path : str
            Path to the file to upload.
        spyglass_name : str, optional
            Name to register it under. Defaults to the file's own name.
        file_class : str, optional
            "raw" or "analysis".
        scope : str, optional
            "private", "group", or "public".
        teams : list of str, optional
            `LabTeam` names, required when `scope` is "group".
        sha256 : str, optional
            Precomputed digest, to avoid re-reading a large file.
        content_md5 : str, optional
            Precomputed MD5, from a caller that already hashed the file.

        Returns
        -------
        dict
            The registration, with `file_id` and `deduplicated`.

        Raises
        ------
        FileNotFoundError
            If `file_path` does not exist.
        StoreError
            If the object store refuses the bytes.
        """
        from spyglass.utils.nwb_hash import digest_file

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot upload missing file: {path}")

        have = {"sha256": sha256, "md5": content_md5}
        missing = [n for n in self.upload_digests() if not have.get(n)]

        if missing:  # one read, however many digests it feeds
            have.update(digest_file(path, algorithms=missing))

        sha256, content_md5 = have["sha256"], have.get("md5")

        target = self.register(
            sha256=sha256,
            size_bytes=path.stat().st_size,
            spyglass_name=spyglass_name or path.name,
            file_class=file_class,
            scope=scope,
            teams=teams,
            content_md5=content_md5,
        )

        if target.get("deduplicated"):
            # Someone already uploaded these exact bytes. Content addressing
            # means the registration is all that was missing.
            logger.info(f"{path.name} already stored; registered only.")
            return target

        self._put_bytes(path, target)

        return target

    def _put_bytes(self, path: Path, target: dict) -> None:
        """Write a file to the signed upload URL.

        `upload_headers` are sent verbatim and are not optional: they carry
        the checksum the store verifies against, and they are covered by the
        URL's signature. Dropping them fails the upload rather than skipping
        the check.

        Parameters
        ----------
        path : pathlib.Path
            File to send.
        target : dict
            The registration response, carrying `upload_url` and
            `upload_headers`.

        Raises
        ------
        StoreError
            If the store rejects the bytes, typically on a checksum mismatch.
        """
        import requests

        url = target.get("upload_url")
        if not url:
            raise StoreError(
                "The broker registered the file but returned no upload URL."
            )

        logger.info(f"Uploading {path.name} to the shared store.")

        with path.open("rb") as f:
            try:
                response = requests.put(
                    url,
                    data=f,
                    headers=target.get("upload_headers") or {},
                    timeout=None,  # a whole-file transfer, not an API call
                )
            except requests.RequestException as err:
                raise StoreError(
                    f"Upload of {path.name} failed: {err}"
                ) from err

        if not response.ok:
            raise StoreError(
                f"The object store refused {path.name} "
                + f"({response.status_code}): {response.text.strip()[:200]}"
            )

    def set_visibility(
        self, file_id: str, scope: str, teams: Optional[List[str]] = None
    ) -> dict:
        """Change who may read a file.

        Owner only. A teammate who can read a file cannot widen access to it.

        Parameters
        ----------
        file_id : str
            Broker file id.
        scope : str
            "private", "group", or "public".
        teams : list of str, optional
            `LabTeam` names. Required when `scope` is "group": a group scope
            naming no team is rejected rather than quietly meaning private.

        Returns
        -------
        dict
            The visibility as the broker now records it.

        Raises
        ------
        StoreForbidden
            If this identity does not own the file.
        """
        body = {"scope": scope, "teams": list(teams or [])}

        return self._request(
            "PATCH", f"/file/{file_id}/visibility", json=body
        ).json()


def get_client(base_url: Optional[str] = None) -> StoreClient:
    """Return a client for the configured broker.

    Parameters
    ----------
    base_url : str, optional
        Override the configured broker.

    Returns
    -------
    StoreClient
        A client. Check `configured` before using it; an instance attached to
        no broker is the ordinary case.
    """
    return StoreClient(base_url=base_url)


def login_cli(argv: Optional[List[str]] = None) -> int:
    """Console entry point for `spyglass-store`.

    Login has to be runnable before anything else works, so it lives outside
    the DataJoint session: the user may not yet have a working config, and
    should not have to open Python to get a token.

    Parameters
    ----------
    argv : list of str, optional
        Arguments, for testing. Defaults to `sys.argv[1:]`.

    Returns
    -------
    int
        Process exit status.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="spyglass-store",
        description="Log in to a Spyglass shared-storage broker.",
    )
    parser.add_argument(
        "command",
        choices=["login", "logout", "status"],
        help="login: run device flow. logout: forget the local token. "
        "status: show who the broker thinks you are.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Broker base URL. Defaults to `store_url` in the config.",
    )
    args = parser.parse_args(argv)

    try:
        client = StoreClient(base_url=args.url)

        if args.command == "login":
            client.login()
        elif args.command == "logout":
            client.logout()
            logger.info(f"Forgot the local token for {client.base_url}.")
        else:
            if not client.logged_in:
                logger.info(f"Not logged in to {client.base_url}.")
                return 1
            logger.info(
                f"{client.base_url}: {client.github_login} "
                + f"(tier: {client.tier}), token in {token_path()}."
            )
    except StoreError as err:
        logger.error(str(err))
        return 1

    return 0
