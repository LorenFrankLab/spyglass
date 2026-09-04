"""Tests that spyglass.decoding tolerates a broken non_local_detector.

``non_local_detector`` imports ``jax`` at package init, which can raise when
``numpy``/``jax`` versions are incompatible. Importing ``spyglass.decoding``
(and declaring the merge tables) must survive that, so ingestion is not taken
down by a decoding-only dependency. See #1619, TODO(#1609).
"""

import os
import subprocess
import sys

import pytest

# Stands in for the real package on the child's path. Mirrors the failure
# seen in the wild: jax raises during non_local_detector's package init.
BROKEN_NON_LOCAL_DETECTOR = (
    "raise AttributeError(\n"
    "    \"module 'numpy.dtypes' has no attribute 'StringDType'\"\n"
    ")\n"
)

CHILD_SCRIPT = '''
"""Run in a fresh interpreter with a broken non_local_detector shadowed in."""

import sys

import datajoint as dj

dj.config.load(sys.argv[1])

try:
    import non_local_detector  # noqa: F401
except AttributeError:
    pass  # expected: the stand-in package raises on import
else:
    raise SystemExit("stand-in did not shadow the installed non_local_detector")

import spyglass.decoding  # noqa: E402,F401
from spyglass.decoding.v1.core import DecodingParameters  # noqa: E402
from spyglass.utils.dj_helper_fn import declare_all_merge_tables  # noqa: E402

declare_all_merge_tables()

assert DecodingParameters().contents == [], "contents forced a broken import"

print("COLD IMPORT OK")
'''


@pytest.mark.slow
@pytest.mark.database
def test_cold_import_with_broken_non_local_detector(tmp_path, dj_config):
    """A cold import of spyglass.decoding survives a broken dependency.

    Runs in a subprocess because the failure only exists at import time: once
    this process has imported ``spyglass.decoding`` successfully, no amount of
    monkeypatching reproduces it.
    """
    nld_dir = tmp_path / "non_local_detector"
    nld_dir.mkdir()
    (nld_dir / "__init__.py").write_text(BROKEN_NON_LOCAL_DETECTOR)

    script = tmp_path / "cold_import.py"
    script.write_text(CHILD_SCRIPT)

    # tmp_path is sys.path[0] for the child (it holds the script), so the
    # stand-in package shadows the installed one. PYTHONPATH makes that
    # explicit rather than incidental.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    proc = subprocess.run(
        [sys.executable, str(script), dj_config],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert proc.returncode == 0, (
        "importing spyglass.decoding failed with a broken "
        f"non_local_detector:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "COLD IMPORT OK" in proc.stdout


def test_insert_does_not_mutate_caller_rows(decode_v1):
    """DecodingParameters.insert converts a copy, not the caller's rows.

    Regression test: the override used to write the dict-converted params back
    into the passed row, so a second insert of the same row re-converted an
    already-converted dict and raised AttributeError.
    """
    from non_local_detector import ContFragClusterlessClassifier

    table = decode_v1.core.DecodingParameters()
    rows = [
        {
            "decoding_param_name": "test_insert_no_mutate",
            "decoding_params": ContFragClusterlessClassifier(),
            "decoding_kwargs": dict(),
        }
    ]

    table.insert(rows, skip_duplicates=True)

    assert isinstance(
        rows[0]["decoding_params"], ContFragClusterlessClassifier
    ), "insert mutated the caller's row"

    table.insert(rows, skip_duplicates=True)  # must not raise
