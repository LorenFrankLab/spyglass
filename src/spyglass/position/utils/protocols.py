"""Utilities for position v2.

``RealFileSystem`` is the default for ``PoseToolStrategy(filesystem=...)``;
tests pass a dict-backed stub there instead of building a project on disk.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Union


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


class RealFileSystem:
    """File operations a PoseToolStrategy needs to locate a model.

    Tests inject a stub with these four methods instead.
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
