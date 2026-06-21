"""Analyzer-cache path policy.

The SortingAnalyzer folder is regeneratable scratch whose location is a pure
function of ``sorting_id`` + one configured root. These tests pin that
policy: the ``spikesorting_v2_analyzer_dir`` config override, the
``temp_dir`` fallback (unchanged default), the ``sorting_id``-keyed path, and
the ``remove_analyzer_cache`` semantics. They do not touch the database.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spyglass.spikesorting.v2._analyzer_cache import (
    analyzer_cache_root,
    analyzer_path,
    remove_analyzer_cache,
)


def test_analyzer_cache_root_honors_config(restore_custom_config):
    import datajoint as dj

    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = "/tmp/v2_custom_an"
    assert analyzer_cache_root() == Path("/tmp/v2_custom_an")
    assert analyzer_path("abc") == Path("/tmp/v2_custom_an/abc.analyzer")


def test_analyzer_cache_root_falls_back_to_temp_dir(restore_custom_config):
    import datajoint as dj

    from spyglass.settings import temp_dir

    dj.config["custom"].pop("spikesorting_v2_analyzer_dir", None)
    expected = Path(temp_dir) / "spikesorting_v2" / "analyzers"
    assert analyzer_cache_root() == expected
    assert analyzer_path("s1") == expected / "s1.analyzer"


def test_empty_config_value_falls_back(restore_custom_config):
    """An empty-string override is falsy -> fall back to temp_dir."""
    import datajoint as dj

    from spyglass.settings import temp_dir

    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = ""
    assert (
        analyzer_cache_root()
        == Path(temp_dir) / "spikesorting_v2" / "analyzers"
    )


def test_remove_analyzer_cache(tmp_path, restore_custom_config):
    import datajoint as dj

    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
    sid = "deadbeef"

    # Absent folder: no-op with missing_ok (default), raises otherwise.
    assert remove_analyzer_cache(sid) is False
    with pytest.raises(FileNotFoundError):
        remove_analyzer_cache(sid, missing_ok=False)

    # Present folder (with contents): removed, returns True.
    folder = analyzer_path(sid)
    folder.mkdir(parents=True)
    (folder / "waveforms.bin").write_text("scratch")
    assert remove_analyzer_cache(sid) is True
    assert not folder.exists()


def test_analyzer_cache_import_pulls_no_db_layer_modules():
    """A cold import of ``_analyzer_cache`` pulls in neither
    ``spyglass.common`` nor any v2 *schema* module (it reads ``dj.config`` /
    ``temp_dir`` only at call time), so importing it opens no DB connection.
    """
    import subprocess
    import sys
    import textwrap

    probe = textwrap.dedent("""
        import sys
        import spyglass.spikesorting.v2._analyzer_cache as c
        assert hasattr(c, "analyzer_path")
        assert hasattr(c, "analyzer_cache_root")
        assert hasattr(c, "remove_analyzer_cache")
        leaked = sorted(
            m
            for m in sys.modules
            if m.startswith("spyglass.common")
            or m in (
                "spyglass.spikesorting.v2.recording",
                "spyglass.spikesorting.v2.artifact",
                "spyglass.spikesorting.v2.sorting",
            )
        )
        assert not leaked, "cold import pulled in DB-layer modules: " + repr(leaked)
        """)
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        "analyzer-cache helper must import without pulling in the DB-layer "
        f"(spyglass.common) or v2 schema modules\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
