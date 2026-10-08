"""Atomic JSON files shared by local review services and their workers."""

import json
import tempfile
from pathlib import Path


def write_json(path, value):
    """Replace a JSON document without sharing temporary names between writers."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}
