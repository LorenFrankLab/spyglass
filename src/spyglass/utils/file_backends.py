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

What a backend *can* do is declared by `supports_streaming`. What it *does* is
that capability narrowed by user preference: `sg_config.prefer_download` makes
a stream-capable backend transfer the whole file instead, for users whose
connection makes many small range requests more expensive than one sequential
download.

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
from typing import List, Optional, Protocol, Tuple, runtime_checkable

import pynwb

from spyglass.utils.logging import logger


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
    `download`. `open` is concrete and picks between them, so a backend that
    can do both needs no dispatch logic of its own.

    Attributes
    ----------
    name : str
        Short identifier, used in configuration and log messages.
    supports_streaming : bool
        True if the backend implements `stream`. A capability declaration, not
        a promise about what `open` will do on any given call: the user may
        prefer download (see `sg_config.prefer_download`).
    """

    name: str = "base"
    supports_streaming: bool = False

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

        The single decision point: `open` consults it, and so does the resolver,
        which records the answer so `file_is_remote` can report how the file was
        actually read instead of inferring it afterward.

        Streams when the backend can and the user has not opted out. A user who
        prefers download but whose backend cannot download is served by
        streaming anyway: the setting is a preference, never a failure mode.

        A backend that overrides `open` must override this to match.

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
        if not sg_config.prefer_download:
            return True

        # No download to fall back on, so serve the file rather than the
        # preference.
        if type(self).download is FileBackend.download:
            logger.debug(
                "%s cannot download; streaming despite prefer_download",
                self.name,
            )
            return True
        return False

    def open(self, nwb_file_path: str) -> Tuple[pynwb.NWBHDF5IO, pynwb.NWBFile]:
        """Return an open `(io, nwbfile)` pair.

        Streams if the backend supports it and the user has not set
        `prefer_download`; otherwise downloads the file and reads the local
        copy. Backends rarely need to override this.

        Parameters
        ----------
        nwb_file_path : str
            Absolute path of the file as Spyglass expects it locally.

        Returns
        -------
        tuple of (pynwb.NWBHDF5IO, pynwb.NWBFile)
            Open IO handle and the file it read.

        Raises
        ------
        BackendUnavailable
            If the download did not produce a local file. Errors raised while
            reading a file that was transferred propagate untouched.
        """
        if self.will_stream(nwb_file_path):
            return self.stream(nwb_file_path)

        if not self.download(nwb_file_path):
            raise BackendUnavailable(
                f"Backend '{self.name}' could not download "
                + f"{Path(nwb_file_path).name}"
            )
        return _open_local_nwb(nwb_file_path)


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

    def has(self, nwb_file_path: str) -> bool:
        """Return True if the file exists on disk."""
        return os.path.exists(nwb_file_path)

    def open(self, nwb_file_path: str) -> Tuple[pynwb.NWBHDF5IO, pynwb.NWBFile]:
        """Open the file from disk."""
        return _open_local_nwb(nwb_file_path)


class KacheryBackend(FileBackend):
    """Fetch analysis files shared through kachery-cloud.

    Download only. kachery has no streaming path, so this backend relies on the
    inherited `open`.
    """

    name = "kachery"
    supports_streaming = False

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

    def stream(
        self, nwb_file_path: str
    ) -> Tuple[pynwb.NWBHDF5IO, pynwb.NWBFile]:
        """Stream the file from DANDI over HTTP range requests.

        Raises
        ------
        BackendUnavailable
            If DANDI holds the file under neither naming scheme. Unreachable
            via `get_nwb_file`, which gates on `has` first; this covers a
            direct call, where there is no pair to return instead.
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


# The resolution chain, in order. Local disk first, then remote sources.
_BACKENDS: List[FileBackend] = [
    LocalBackend(),
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
