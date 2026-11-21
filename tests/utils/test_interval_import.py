"""Importing `spyglass.utils.interval` needs no database."""

import os
import subprocess
import sys

CHILD_SCRIPT = """
import sys

import datajoint.connection


def refuse_connection(*args, **kwargs):
    raise SystemExit("database connection attempted")


datajoint.connection.Connection.__init__ = refuse_connection

from spyglass.utils.interval import Interval

assert Interval([[0, 2]]).subtract([[1, 3]]) == [[0, 1]]
common = sorted(m for m in sys.modules if m.startswith("spyglass.common"))
assert not common, f"imported table modules: {common}"
print("IMPORT OK")
"""


def test_interval_import_needs_no_database():
    """Import and use `Interval` with every connection attempt refused.

    Runs in a fresh interpreter: this one has already imported the tables.
    """
    proc = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT],
        env=dict(os.environ, DJ_HOST="no-db.invalid"),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert "IMPORT OK" in proc.stdout, proc.stderr
