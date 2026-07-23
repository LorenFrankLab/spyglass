"""Utilities and protocols for dependency injection in position v2."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Protocol, Union


def default_pk_name(
    prefix: str,
    params: dict = None,
    limit: int = 32,
    include_hash: bool = True,
) -> str:
    """Generate a default primary-key name from prefix and params.

    Format: ``PREFIX-YYYYMMDD-HASH`` where YYYYMMDD is today (UTC) and HASH
    is an 8-character hex digest of *params* (or empty dict when omitted).

    Parameters
    ----------
    prefix : str
        Short label, e.g. ``'mdl'`` or ``'mp'``.
    params : dict, optional
        Data to hash. Defaults to ``{}``.
    limit : int, optional
        Maximum length of the returned string, by default 32.
    include_hash : bool, optional
        Append the hash component when True (default).
    """
    when = datetime.now(timezone.utc)
    h = ""
    if include_hash:
        raw = json.dumps(params or {}, sort_keys=True, default=str)
        h = "-" + hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{prefix}-{when:%Y%m%d}{h}"[:limit]


class FileSystemProtocol(Protocol):
    """Protocol for file system operations.

    Enables dependency injection of file I/O operations into strategy classes.
    """

    def glob(self, pattern: str) -> List[str]:
        """Find files matching a glob pattern.

        Parameters
        ----------
        pattern : str
            Glob pattern to match

        Returns
        -------
        List[str]
            List of file paths matching the pattern
        """
        ...

    def read_yaml(self, path: Union[str, Path]) -> Dict:
        """Read a YAML file and return its contents.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the YAML file

        Returns
        -------
        Dict
            The YAML file contents as a dictionary
        """
        ...

    def exists(self, path: Union[str, Path]) -> bool:
        """Check if a path exists.

        Parameters
        ----------
        path : Union[str, Path]
            Path to check

        Returns
        -------
        bool
            True if the path exists, False otherwise
        """
        ...

    def getmtime(self, path: Union[str, Path]) -> float:
        """Get the modification time of a file.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the file

        Returns
        -------
        float
            Modification time as timestamp
        """
        ...


class RealFileSystem:
    """Real filesystem implementation of FileSystemProtocol.

    This is the default implementation that uses actual file system operations.
    Tests can inject a stub implementation to avoid file system dependencies.
    """

    def glob(self, pattern: str) -> List[str]:
        """Find files matching a glob pattern."""
        import glob

        return glob.glob(pattern)

    def read_yaml(self, path: Union[str, Path]) -> Dict:
        """Read a YAML file and return its contents."""
        # Import here to avoid circular dependency
        from .yaml_io import load_yaml

        return load_yaml(path)

    def exists(self, path: Union[str, Path]) -> bool:
        """Check if a path exists."""
        return Path(path).exists()

    def getmtime(self, path: Union[str, Path]) -> float:
        """Get the modification time of a file."""
        return Path(path).stat().st_mtime
