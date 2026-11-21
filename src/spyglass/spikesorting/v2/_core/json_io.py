"""Atomic JSON files shared by local review services and their workers."""

import json
import uuid
from pathlib import Path


def write_json(path, value):
    """Replace a JSON document without sharing temporary names between writers.

    The file keeps its current permissions, and a new file gets the umask's,
    so other reviewers of a shared bundle can still read it.
    """
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x") as stream:
            json.dump(value, stream, indent=2)
        if path.exists():
            temporary.chmod(path.stat().st_mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}
