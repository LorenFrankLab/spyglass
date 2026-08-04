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

Notes
-----
Backends import their supporting modules inside method bodies rather than at
module scope. This keeps optional dependencies optional, avoids import cycles,
and lets tests patch module attributes such as
`spyglass.sharing.sharing_kachery._kachery_available`.
"""

import os
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Protocol, Tuple, runtime_checkable

import pynwb

from spyglass.utils.logging import logger


@runtime_checkable
class FileBackend(Protocol):
    """A remote source of NWB files.

    Doubles as the structural type and the base class. Inheriting from it
    supplies the default `open` and enforces `has` at instantiation, exactly as
    an ABC would; a third-party backend that implements the same members
    without inheriting still satisfies `isinstance` checks.

    Subclasses must implement `has`. A backend that can stream overrides
    `open`. A backend that can only transfer whole files implements `download`
    and inherits the default `open`, which downloads and then reads locally.

    Attributes
    ----------
    name : str
        Short identifier, used in configuration and log messages.
    supports_streaming : bool
        True if `open` reads over the network without writing a local copy.
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

    def open(self, nwb_file_path: str) -> Tuple[pynwb.NWBHDF5IO, pynwb.NWBFile]:
        """Return an open `(io, nwbfile)` pair.

        Default implementation downloads the file and opens the local copy.
        Streaming backends override this.

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
        FileNotFoundError
            If the download did not produce a local file.
        """
        if not self.download(nwb_file_path):
            raise FileNotFoundError(
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
    """Stream files published to a DANDI archive.

    Streaming only. `download` is not implemented, so `open` is overridden.
    """

    name = "Dandi"
    supports_streaming = True

    def has(self, nwb_file_path: str) -> bool:
        """Return True if DANDI holds this file under either naming scheme."""
        from spyglass.common.common_dandi import DandiPath

        return bool(
            DandiPath().has_file_path(file_path=nwb_file_path)
            or DandiPath().has_raw_path(file_path=nwb_file_path)
        )

    def open(self, nwb_file_path: str) -> Tuple[pynwb.NWBHDF5IO, pynwb.NWBFile]:
        """Stream the file from DANDI over HTTP range requests."""
        from spyglass.common.common_dandi import DandiPath

        if DandiPath().has_file_path(nwb_file_path):
            path_to_load = nwb_file_path
        elif DandiPath().has_raw_path(nwb_file_path):
            path_to_load = DandiPath().raw_from_path(nwb_file_path)["filename"]
        else:
            raise FileNotFoundError(
                f"File not found in Dandi: {Path(nwb_file_path).name}"
            )

        return DandiPath().fetch_file_from_dandi(nwb_file_path=path_to_load)


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
