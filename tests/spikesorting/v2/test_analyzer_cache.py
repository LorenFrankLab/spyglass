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
import spikeinterface.full as si

from spyglass.spikesorting.v2._analyzer_cache import (
    analyzer_cache_root,
    analyzer_path,
    remove_analyzer_cache,
)
from spyglass.spikesorting.v2._sorting_analyzer import build_analyzer


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


# --------------------------------------------------------------------------- #
# ``build_analyzer`` projects a 3D probe to 2D before building the analyzer.
#
# Spyglass stores electrode geometry in 3D (z is typically 0), but the
# SortingAnalyzer ``unit_locations`` extension and the spikeinterface-gui probe
# view assume 2D contact positions and raise
# ``could not broadcast input array from shape (3,) into shape (2,)`` on a 3D
# probe. ``build_analyzer`` projects the probe to 2D so both work; this is the
# regression guard. It is a fast, DB-free unit test: it drives ``build_analyzer``
# directly with a synthetic recording carrying a 3D probe, passing ``sorter_row``
# / ``job_kwargs`` / ``analyzer_folder`` so no database read happens.
# --------------------------------------------------------------------------- #


@pytest.fixture
def recording_3d_and_sorting():
    """A synthetic recording (forced to a 3D probe) and its sorting."""
    recording, sorting = si.generate_ground_truth_recording(
        durations=[10.0], num_channels=8, num_units=4, seed=0
    )
    recording = recording.set_probe(recording.get_probe().to_3d(axes="xy"))
    assert recording.get_probe().ndim == 3
    return recording, sorting


class TestBuildAnalyzerProbeProjection:
    def test_analyzer_probe_is_2d(self, recording_3d_and_sorting, tmp_path):
        recording, sorting = recording_3d_and_sorting
        folder = tmp_path / "sort.analyzer"

        build_analyzer(
            sorting,
            recording,
            key={"sorting_id": "test-3d-probe"},
            sorter_row={"job_kwargs": {}},
            job_kwargs={},
            analyzer_folder=folder,
        )

        analyzer = si.load_sorting_analyzer(folder)
        assert analyzer.get_probe().ndim == 2

    def test_unit_locations_computes_after_projection(
        self, recording_3d_and_sorting, tmp_path
    ):
        # Without the 2D projection this raises ValueError:
        # "could not broadcast input array from shape (3,) into shape (2,)".
        recording, sorting = recording_3d_and_sorting
        folder = tmp_path / "sort.analyzer"

        build_analyzer(
            sorting,
            recording,
            key={"sorting_id": "test-3d-probe"},
            sorter_row={"job_kwargs": {}},
            job_kwargs={},
            analyzer_folder=folder,
        )

        analyzer = si.load_sorting_analyzer(folder)
        analyzer.compute("unit_locations")
        unit_locations = analyzer.get_extension("unit_locations").get_data()
        assert unit_locations.shape[0] == sorting.get_num_units()
