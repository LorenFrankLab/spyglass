"""The shared position helpers must import without a database.

``utils/__init__`` imports ``general``, which imported ``common_behav`` at
module scope. That gave every pure-math helper in the package a live
DataJoint connection, so the math could not be tested or reused without one.
"""

import subprocess
import sys


def test_centroid_imports_without_datajoint_connection():
    """Importing a math helper must not open a connection."""
    code = (
        "import datajoint as dj\n"
        "def _boom(*a, **k):\n"
        "    raise AssertionError('connection attempted on import')\n"
        "dj.conn = _boom\n"
        "from spyglass.position.utils.centroid import calculate_centroid\n"
        "assert callable(calculate_centroid)\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert "ok" in result.stdout, result.stderr[-2000:]


def test_general_does_not_import_video_file_at_module_scope():
    """VideoFile belongs inside get_video_info, not at module scope."""
    from pathlib import Path

    import spyglass.position.utils.general as general

    src = Path(general.__file__).read_text()
    header = src.split("def ", 1)[0]
    assert "common_behav import VideoFile" not in header
