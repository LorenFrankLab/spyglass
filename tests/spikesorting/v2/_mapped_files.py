"""Check that no file in a temporary directory is memory-mapped at cleanup.

On NFS, removing a file that a process still maps leaves a ``.nfsXXXX``
placeholder in its directory, so ``TemporaryDirectory`` cleanup fails with
``OSError: [Errno 39] Directory not empty``. Local file systems remove such
files without error, so the failure does not show on a laptop or in CI.
These helpers make the condition visible on any Linux host: they read
``/proc/self/maps`` when the directory is cleaned up.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

_MAPS = Path("/proc/self/maps")


def mapped_paths_under(root) -> list[str]:
    """Files under ``root`` that this process maps into memory now."""
    root = str(Path(root).resolve())
    paths = set()
    for line in _MAPS.read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith(root):
            paths.add(parts[5])
    return sorted(paths)


def record_mapped_files_at_cleanup(monkeypatch) -> list[list[str]]:
    """Record, at each ``TemporaryDirectory`` cleanup, the files still mapped.

    Returns the list that receives one entry per cleaned directory. Skips
    the test where ``/proc/self/maps`` is not available.
    """
    if not _MAPS.exists():
        pytest.skip("needs /proc/self/maps (Linux)")
    seen = []
    real_cleanup = tempfile.TemporaryDirectory.cleanup

    def cleanup(self):
        seen.append(mapped_paths_under(self.name))
        return real_cleanup(self)

    monkeypatch.setattr(tempfile.TemporaryDirectory, "cleanup", cleanup)
    return seen
