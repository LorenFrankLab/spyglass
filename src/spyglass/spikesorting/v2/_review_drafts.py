"""Review-scoped draft storage and atomic revision checks, independent of HTTP."""

import hashlib
import json
from pathlib import Path

from spyglass.spikesorting.v2._json_io import write_json


class DraftError(ValueError):
    """A draft conflict or access refusal with its HTTP status."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _revision(data: bytes) -> str:
    return '"' + hashlib.sha256(data).hexdigest() + '"'


class ReviewDraftStore:
    """Own one bundle's annotations file and its cross-process save lock."""

    def __init__(self, root):
        self.root = Path(root).resolve()

    @property
    def path(self):
        path = (self.root / "annotations.json").resolve()
        if not path.is_relative_to(self.root):
            raise DraftError(403, "Forbidden: path outside review bundle")
        return path

    def read(self):
        """Return payload and revision from the same bytes; absent files read empty."""
        try:
            data = self.path.read_bytes()
        except FileNotFoundError:
            data = b""
        return json.loads(data) if data else {}, _revision(data), bool(data)

    def save(self, value: dict, expected_revision: str) -> str:
        """Compare and replace atomically; refuse a stale draft revision."""
        from filelock import FileLock

        path = self.path
        with FileLock(str(path) + ".lock"):
            data = path.read_bytes() if path.exists() else b""
            if expected_revision != _revision(data):
                raise DraftError(409, "Another tab changed this draft.")
            write_json(path, value)
            return _revision(path.read_bytes())
