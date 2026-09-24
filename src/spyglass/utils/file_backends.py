"""Pluggable backends for fetching NWB files that are not present locally.

`get_nwb_file` resolves a missing file by walking an ordered list of remote
backends. Each backend reports whether it holds a file and either streams it or
downloads it to the expected local path.

Local disk is itself a backend, so resolution is a single uniform loop. The
recompute fallback stays inline in `get_nwb_file`, since it needs a query
expression that is not part of this protocol.

The chain is fixed in code rather than user-configurable: local must be tried
first, and letting a config file put a network source ahead of disk would only
ever be a mistake.

What a backend *can* do is declared by `supports_streaming` and
`supports_download`. What it *does* is those capabilities narrowed by user
preference, and `open` reports the outcome in `Opened.streamed` so no caller
has to infer it.

Notes
-----
Backends import their supporting modules inside method bodies rather than at
module scope. This keeps optional dependencies optional, avoids import cycles,
and lets tests patch module attributes such as
`spyglass.sharing.sharing_kachery._kachery_available`. `spyglass.settings` is
imported this way for the cycle reason: it imports `dj_helper_fn`, which
imports `nwb_helper_fn`, which imports this module.
"""

import os
from abc import abstractmethod
from pathlib import Path
from typing import (
    List,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import pynwb

from spyglass.utils.logging import logger


class Opened(NamedTuple):
    """The result of opening a file, and how it was read.

    `streamed` is reported by the backend that did the work rather than
    inferred afterward, which would mean guessing at the private internals of
    whatever filesystem implementation the backend chose. It is what
    `file_is_remote` answers with.

    Attributes
    ----------
    io : pynwb.NWBHDF5IO
        Open IO handle.
    nwbfile : pynwb.NWBFile
        The file it read.
    streamed : bool
        True if the bytes crossed the network rather than coming from disk.
    """

    io: pynwb.NWBHDF5IO
    nwbfile: pynwb.NWBFile
    streamed: bool


class BackendUnavailable(FileNotFoundError):
    """A backend cannot supply the file it was asked for.

    Raised only for the expected miss: the backend does not hold the file, or
    its transfer did not produce one. `get_nwb_file` catches this and moves on
    to the next backend.

    Genuine failures — a corrupt file, a network error, a DataJoint error — are
    not this, and must propagate rather than be mistaken for a miss. Otherwise
    the real error is swallowed and the file may be silently recomputed.

    Subclasses `FileNotFoundError` so existing callers that catch that keep
    working; resolution catches only this narrower type.
    """


@runtime_checkable
class FileBackend(Protocol):
    """A remote source of NWB files.

    Doubles as the structural type and the base class. Inheriting from it
    supplies the default `open` and enforces `has` at instantiation, exactly as
    an ABC would; a third-party backend that implements the same members
    without inheriting still satisfies `isinstance` checks.

    Subclasses must implement `has`, plus at least one of `stream` and
    `download`, declaring which by setting the matching flag. `open` is
    concrete and picks between them, so a backend that can do both needs no
    dispatch logic of its own.

    Attributes
    ----------
    name : str
        Short identifier, used in configuration and log messages.
    supports_streaming : bool
        True if the backend implements `stream`.
    supports_download : bool
        True if the backend implements `download`.

    Notes
    -----
    The flags are capability declarations, not promises about any given call:
    a backend that can do both defers to `sg_config.prefer_download`. `open`
    reports what it actually did in `Opened.streamed`.
    """

    name: str = "base"
    supports_streaming: bool = False
    supports_download: bool = False

    @abstractmethod
    def has(self, nwb_file_path: str) -> bool:
        """Return True if this backend can supply the given file.

        Parameters
        ----------
        nwb_file_path : str
            Absolute path of the file as Spyglass expects it locally.

        Returns
        -------
        bool
            True if a subsequent `open` is worth attempting.
        """
        raise NotImplementedError

    def download(self, nwb_file_path: str, dest: Optional[str] = None) -> bool:
        """Fetch the file to local disk.

        Parameters
        ----------
        nwb_file_path : str
            Absolute path of the file as Spyglass expects it locally.
        dest : str, optional
            Destination path. Defaults to `nwb_file_path`. Backends that resolve
            their own destination may ignore this.

        Returns
        -------
        bool
            True if the file is present locally after the call.
        """
        raise NotImplementedError(
            f"Backend '{self.name}' does not implement download."
        )

    def stream(
        self, nwb_file_path: str
    ) -> Tuple[pynwb.NWBHDF5IO, pynwb.NWBFile]:
        """Read the file over the network without writing a local copy.

        Implemented by backends that set `supports_streaming`.

        Parameters
        ----------
        nwb_file_path : str
            Absolute path of the file as Spyglass expects it locally.

        Returns
        -------
        tuple of (pynwb.NWBHDF5IO, pynwb.NWBFile)
            Open IO handle and the file it read.
        """
        raise NotImplementedError(
            f"Backend '{self.name}' does not implement stream."
        )

    def will_stream(self, nwb_file_path: str) -> bool:
        """Return True if `open` would stream this file rather than download.

        Streams when the backend can and the user has not opted out. A user who
        prefers download but whose backend cannot download is served by
        streaming anyway: the setting is a preference, never a failure mode.

        The preference is read per backend: `custom.backends.<name>` first,
        then the instance-wide `custom.prefer_download`.

        Override to give per-file answers if the backend streams some files and
        downloads others.

        Parameters
        ----------
        nwb_file_path : str
            Absolute path of the file as Spyglass expects it locally.

        Returns
        -------
        bool
            True if the next `open` call will read over the network.
        """
        from spyglass.settings import sg_config

        if not self.supports_streaming:
            return False

        prefers_download = sg_config.backend_prefers_download(self.name)

        return not (self.supports_download and prefers_download)

    def open(self, nwb_file_path: str) -> Opened:
        """Open the file and report how it was read.

        Streams if the backend supports it and the user has not set
        `prefer_download`; otherwise downloads the file and reads the local
        copy. Backends rarely need to override this.

        Parameters
        ----------
        nwb_file_path : str
            Absolute path of the file as Spyglass expects it locally.

        Returns
        -------
        Opened
            Open IO handle, the file it read, and whether it was streamed.

        Raises
        ------
        BackendUnavailable
            If the download did not produce a local file. Errors raised while
            reading a file that was transferred propagate untouched.
        """
        if self.will_stream(nwb_file_path):
            return Opened(*self.stream(nwb_file_path), streamed=True)

        if not self.download(nwb_file_path):
            raise BackendUnavailable(
                f"Backend '{self.name}' could not download "
                + f"{Path(nwb_file_path).name}"
            )
        return Opened(*_open_local_nwb(nwb_file_path), streamed=False)


def _open_local_nwb(
    nwb_file_path: str,
) -> Tuple[pynwb.NWBHDF5IO, pynwb.NWBFile]:
    """Open an NWB file from local disk without caching it.

    Parameters
    ----------
    nwb_file_path : str
        Absolute path to a local NWB file.

    Returns
    -------
    tuple of (pynwb.NWBHDF5IO, pynwb.NWBFile)
        Open IO handle and the file it read.
    """
    io = pynwb.NWBHDF5IO(path=nwb_file_path, mode="r", load_namespaces=True)
    return io, io.read()


class LocalBackend(FileBackend):
    """Read a file already present on local disk.

    First in the resolution chain. Neither downloads nor streams, so it
    overrides `open` to read directly.
    """

    name = "local"
    supports_streaming = False
    supports_download = False

    def has(self, nwb_file_path: str) -> bool:
        """Return True if the file exists on disk."""
        return os.path.exists(nwb_file_path)

    def open(self, nwb_file_path: str) -> Opened:
        """Open the file from disk."""
        return Opened(*_open_local_nwb(nwb_file_path), streamed=False)


class KacheryBackend(FileBackend):
    """Fetch analysis files shared through kachery-cloud.

    Download only. kachery has no streaming path, so this backend relies on the
    inherited `open`.
    """

    name = "kachery"
    supports_streaming = False
    supports_download = True

    def has(self, nwb_file_path: str) -> bool:
        """Return True if kachery is installed and knows this file."""
        from spyglass.sharing import sharing_kachery

        if not sharing_kachery._kachery_available:
            logger.debug(
                "kachery unavailable; skipping kachery check for %s",
                nwb_file_path,
            )
            return False

        return bool(
            sharing_kachery.AnalysisNwbfileKachery
            & {"analysis_file_name": Path(nwb_file_path).name}
        )

    def download(self, nwb_file_path: str, dest: Optional[str] = None) -> bool:
        """Download via kachery, which resolves its own destination path."""
        from spyglass.sharing import sharing_kachery

        return bool(
            sharing_kachery.AnalysisNwbfileKachery.download_file(
                Path(nwb_file_path).name, permit_fail=True
            )
        )


class DandiBackend(FileBackend):
    """Fetch files published to a DANDI archive.

    Streams by default. Also implements `download`, so a user on a slow
    connection can set `prefer_download` and get one sequential transfer
    instead of many range requests.
    """

    name = "Dandi"
    supports_streaming = True
    supports_download = True

    def _resolve(self, nwb_file_path: str) -> Optional[str]:
        """Return the name DANDI knows this file by, or None if it has neither.

        Raw files are published without the trailing underscore Spyglass uses
        locally, so the two naming schemes are tried in turn.

        The single lookup for this backend: `has` is this question asked as a
        predicate, so the two cannot disagree and the archive is queried once
        per question rather than once per caller.

        Parameters
        ----------
        nwb_file_path : str
            Absolute path of the file as Spyglass expects it locally.

        Returns
        -------
        str or None
            Path or file name to hand to `DandiPath`, or None if DANDI holds
            the file under neither scheme.
        """
        from spyglass.common.common_dandi import DandiPath

        dandi_path = DandiPath()

        if dandi_path.has_file_path(nwb_file_path):
            return nwb_file_path
        if dandi_path.has_raw_path(nwb_file_path):
            return dandi_path.raw_from_path(nwb_file_path)["filename"]
        return None

    def has(self, nwb_file_path: str) -> bool:
        """Return True if DANDI holds this file under either naming scheme."""
        return self._resolve(nwb_file_path) is not None

    def will_stream(self, nwb_file_path: str) -> bool:
        """Stream if DANDI holds the file only under its raw name.

        A raw session published as `X.nwb` is not the `X_.nwb` link copy
        Spyglass tracks locally. Writing the DANDI bytes to the tracked path
        leaves a file that fails the DataJoint filepath checksum on every
        later fetch, so `prefer_download` does not apply to these.

        Parameters
        ----------
        nwb_file_path : str
            Absolute path of the file as Spyglass expects it locally.

        Returns
        -------
        bool
            True if the next `open` call will read over the network.
        """
        if self._resolve(nwb_file_path) != nwb_file_path:
            return True
        return super().will_stream(nwb_file_path)

    def stream(
        self, nwb_file_path: str
    ) -> Tuple[pynwb.NWBHDF5IO, pynwb.NWBFile]:
        """Stream the file from DANDI over HTTP range requests.

        Raises
        ------
        BackendUnavailable
            If DANDI holds the file under neither naming scheme. `stream` owes
            its caller an `(io, nwbfile)` pair and has no value with which to
            say no.
        """
        from spyglass.common.common_dandi import DandiPath

        path_to_load = self._resolve(nwb_file_path)
        if path_to_load is None:
            raise BackendUnavailable(
                f"File not found in Dandi: {Path(nwb_file_path).name}"
            )

        return DandiPath().fetch_file_from_dandi(nwb_file_path=path_to_load)

    def download(self, nwb_file_path: str, dest: Optional[str] = None) -> bool:
        """Download the whole file from DANDI to the local path.

        The destination is the local path Spyglass expects, not the DANDI
        name, so the file resolves locally on the next call.

        Returns False rather than raising when DANDI has no such file: the
        bool already carries that answer, and `open` turns it into the one
        `BackendUnavailable` the resolver looks for.
        """
        from spyglass.common.common_dandi import DandiPath

        path_to_load = self._resolve(nwb_file_path)
        if path_to_load is None:
            return False

        return DandiPath().download_file_from_dandi(
            nwb_file_path=path_to_load,
            dest=dest or nwb_file_path,
        )


class StoreBackend(FileBackend):
    """Fetch files from a self-hosted shared-storage broker.

    Sits directly after `LocalBackend`, so a copy already on disk still wins
    and DANDI stays available as the fallback for published data.

    Streams by default over HTTP range requests against the broker's *stable*
    content URL. That URL is not itself signed: each request to it is answered
    with a fresh redirect to a short-lived signed URL, which is what lets a
    read of a multi-gigabyte file outlive any single signature.

    Also implements `download`, so `prefer_download` behaves here as it does
    for DANDI.
    """

    name = "store"
    supports_streaming = True
    supports_download = True

    def __init__(self):
        # Memo for one process, so `has` and the `open` that follows it do not
        # each pay a round trip. Safe against a revoked share: a file id is
        # not a capability, and the broker re-authorizes every content fetch.
        self._resolved = {}

    def _client(self):
        """Return a broker client, or None if this instance has no broker.

        Returns
        -------
        StoreClient or None
            None when `store_url` is unset or the user has never logged in.
            Both are ordinary states, not errors.
        """
        from spyglass.sharing.store_client import get_client

        client = get_client()

        if not client.configured:
            return None

        if not client.logged_in:
            logger.debug(
                "Shared store configured but not logged in; run "
                + "`spyglass-store login` to read from it."
            )
            return None

        return client

    def _known_hash(self, name: str) -> Optional[str]:
        """Return the digest this instance recorded for a file name, if any.

        Resolving by name is ambiguous at the broker: registration is per
        owner, nothing enforces that a `spyglass_name` is unique across them,
        and the resolve endpoint returns the first matching row with no owner
        field to disambiguate by. Two people who share a
        `minirec20230622_.nwb` therefore produce a nondeterministic winner,
        and a reader can be handed someone else's private row and refused a
        file they could in fact read.

        The Spyglass database settles it. `SharedFileSelection` is keyed on
        the file name, so within one instance a name maps to exactly one
        upload and one digest — and content addressing means that digest names
        the bytes rather than anyone's registration of them. Where the row
        exists, this is the authority the broker's name index is not.

        Absent for a file someone else shared from a different Spyglass
        instance, which is the case that still falls back to the name.

        Parameters
        ----------
        name : str
            Spyglass file name.

        Returns
        -------
        str or None
            Hex digest, or None if this instance has no record of the upload.
        """
        try:
            from spyglass.sharing.sharing_store import (
                SharedAnalysisFile,
                SharedFile,
            )
        except Exception as err:  # no such schema, no grants, no connection
            logger.debug(f"No local sharing record available: {err}")
            return None

        for table, attr in (
            (SharedFile, "nwb_file_name"),
            (SharedAnalysisFile, "analysis_file_name"),
        ):
            digests = (table & {attr: name}).fetch("sha256")
            if len(digests):
                return digests[0]

        return None

    def _resolve(self, nwb_file_path: str) -> Optional[dict]:
        """Return the broker's record for this file, or None.

        The single lookup for this backend, so `has` and `open` cannot
        disagree about what the broker holds.

        Resolves by content hash where this instance recorded one, and by name
        otherwise. See `_known_hash` for why that distinction matters.

        A refusal and a miss both return None. That is what the resolution
        chain needs — try the next backend either way — but it does mean a
        file the user could read after linking their GitHub account looks
        exactly like one that does not exist. `StoreClient.resolve` raises the
        distinction for callers that need it.

        Parameters
        ----------
        nwb_file_path : str
            Absolute path of the file as Spyglass expects it locally.

        Returns
        -------
        dict or None
            Broker file record, or None if unavailable.

        Raises
        ------
        StoreQuotaExceeded
            If this read would exceed the tier's allowance. The one case this
            backend does not turn into a `False`, and a deliberate exception
            to the rule that `has` never raises: being throttled means the
            file *is* there and *is* readable, just not yet. Answering "no"
            would send `get_nwb_file` on to recompute an analysis the user
            could have had by waiting, which is far more expensive than the
            error.
        """
        name = Path(nwb_file_path).name

        if name in self._resolved:
            return self._resolved[name]

        client = self._client()

        if client is None:
            # Nothing is cached here. The chain instance is built once at
            # import, and the notebook's own instructions have the user set
            # `store_url` or log in *mid-session* — caching "no" would make
            # every file probed before that permanently unavailable.
            return None

        digest = self._known_hash(name)

        self._resolved[name] = (
            client.find(sha256=digest) if digest else client.find(name=name)
        )

        return self._resolved[name]

    def _resolved_with_client(self, nwb_file_path: str):
        """Return the file record and a live client, or (None, None).

        The record is memoized but the client is not, so a user who logged out
        mid-session would otherwise reach a transfer holding a cached record
        and no way to authorize it. Asking for both together means a transfer
        either has everything it needs or declines.

        Parameters
        ----------
        nwb_file_path : str
            Absolute path of the file as Spyglass expects it locally.

        Returns
        -------
        tuple of (dict or None, StoreClient or None)
            The broker record and the client to fetch it with.
        """
        record = self._resolve(nwb_file_path)
        client = self._client() if record is not None else None

        return (record, client) if client is not None else (None, None)

    def has(self, nwb_file_path: str) -> bool:
        """Return True if the broker holds a file this user may read."""
        return self._resolve(nwb_file_path) is not None

    def stream(
        self, nwb_file_path: str
    ) -> Tuple[pynwb.NWBHDF5IO, pynwb.NWBFile]:
        """Read the file over range requests, caching blocks locally.

        Raises
        ------
        BackendUnavailable
            If the broker holds no readable file under this name. `stream`
            owes its caller an `(io, nwbfile)` pair and has no value with
            which to say no.
        """
        import fsspec
        import h5py
        from fsspec.implementations.cached import CachingFileSystem

        from spyglass.settings import temp_dir

        record, client = self._resolved_with_client(nwb_file_path)
        if record is None:
            raise BackendUnavailable(
                f"File not in the shared store: {Path(nwb_file_path).name}"
            )

        # The bearer header rides every range request, because every one of
        # them is re-authorized and re-signed by the broker. It is dropped
        # when the redirect crosses to the object store, which is required:
        # an S3 endpoint that receives an Authorization header alongside a
        # presigned URL leaves presigned mode and refuses the request.
        fs = fsspec.filesystem(
            "http", client_kwargs={"headers": client.auth_headers()}
        )
        cached = CachingFileSystem(
            fs=fs, cache_storage=f"{temp_dir}/store-cache"
        )

        fs_file = cached.open(client.content_url(record["file_id"]), "rb")
        io = pynwb.NWBHDF5IO(file=h5py.File(fs_file))

        return io, io.read()

    def download(self, nwb_file_path: str, dest: Optional[str] = None) -> bool:
        """Fetch the whole file to local disk in one transfer.

        Writes to a temporary sibling and renames on success, so an
        interrupted transfer never leaves a partial file that the local
        backend would then happily open. The staging name carries a random
        token so two workers fetching the same missing file cannot overwrite
        or unlink each other's partial copy.

        Returns
        -------
        bool
            True if the file is present locally after the call. False — not an
            exception — when the broker holds nothing readable, since `open`
            is what turns that into the one error the resolver looks for.
        """
        from uuid import uuid4

        import requests

        record, client = self._resolved_with_client(nwb_file_path)
        if record is None:
            return False

        target = Path(dest or nwb_file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(f"{target.suffix}.{uuid4().hex[:8]}.part")

        logger.info(f"Downloading {target.name} from the shared store.")

        try:
            with requests.get(
                client.content_url(record["file_id"]),
                headers=client.auth_headers(),
                stream=True,
                timeout=None,  # a whole-file transfer, not an API call
            ) as response:
                response.raise_for_status()
                with temp.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024**2):
                        f.write(chunk)
            temp.replace(target)
        except requests.RequestException as err:
            logger.warning(f"Shared-store download failed: {err}")
            return False
        finally:
            temp.unlink(missing_ok=True)

        return target.exists()


# The resolution chain, in order. Local disk first, then remote sources.
_BACKENDS: List[FileBackend] = [
    LocalBackend(),
    StoreBackend(),
    KacheryBackend(),
    DandiBackend(),
]


def get_backends() -> List[FileBackend]:
    """Return the file backends in resolution order.

    Returns
    -------
    list of FileBackend
        A copy of the chain, so callers cannot mutate it in place.
    """
    return list(_BACKENDS)
