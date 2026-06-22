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

# A display (unwhitened) recipe blob -- build_analyzer requires the resolved
# waveform params (it never picks a default), so the probe-projection tests
# that exercise the build path pass this explicitly.
_DISPLAY_PARAMS = {
    "ms_before": 1.0,
    "ms_after": 2.0,
    "max_spikes_per_unit": 20000,
    "whiten": False,
    "purpose": "display",
    "schema_version": 1,
}


def test_analyzer_cache_root_honors_config(restore_custom_config):
    import datajoint as dj

    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = "/tmp/v2_custom_an"
    assert analyzer_cache_root() == Path("/tmp/v2_custom_an")
    assert analyzer_path("abc", "rec") == Path(
        "/tmp/v2_custom_an/abc__rec.zarr"
    )


def test_analyzer_cache_root_falls_back_to_temp_dir(restore_custom_config):
    import datajoint as dj

    from spyglass.settings import temp_dir

    dj.config["custom"].pop("spikesorting_v2_analyzer_dir", None)
    expected = Path(temp_dir) / "spikesorting_v2" / "analyzers"
    assert analyzer_cache_root() == expected
    assert analyzer_path("s1", "rec") == expected / "s1__rec.zarr"


def test_analyzer_path_includes_params_name(restore_custom_config):
    """The cache folder is keyed by ``(sorting_id, waveform_params_name)``.

    Two recipes for one sort resolve to distinct folders (so a whitened metric
    analyzer never overwrites the unwhitened display one), both under the
    configured root.
    """
    import datajoint as dj

    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = "/tmp/v2_custom_an"
    root = Path("/tmp/v2_custom_an")
    display = analyzer_path("sid1", "franklab_hippocampus_actual_waveforms")
    metric = analyzer_path("sid1", "franklab_hippocampus_metric_waveforms")
    assert display != metric
    assert display.parent == root and metric.parent == root
    assert display == root / "sid1__franklab_hippocampus_actual_waveforms.zarr"


def test_empty_config_value_falls_back(restore_custom_config):
    """An empty-string override is falsy -> fall back to temp_dir."""
    import datajoint as dj

    from spyglass.settings import temp_dir

    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = ""
    assert (
        analyzer_cache_root()
        == Path(temp_dir) / "spikesorting_v2" / "analyzers"
    )


def test_remove_analyzer_cache_removes_all_recipes(
    tmp_path, restore_custom_config
):
    """``remove_analyzer_cache`` globs every ``{sid}__*.zarr`` recipe folder.

    A sort has multiple analyzer recipes on disk (display + metric); deleting
    the sort must orphan every one. A different sort's folder is untouched.
    """
    import datajoint as dj

    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
    sid = "deadbeef"

    # Absent folder: no-op with missing_ok (default), raises otherwise.
    assert remove_analyzer_cache(sid) is False
    with pytest.raises(FileNotFoundError):
        remove_analyzer_cache(sid, missing_ok=False)

    # Two recipe folders for this sort, plus one for a different sort.
    display = analyzer_path(sid, "franklab_hippocampus_actual_waveforms")
    metric = analyzer_path(sid, "franklab_hippocampus_metric_waveforms")
    other = analyzer_path("cafef00d", "franklab_cortex_actual_waveforms")
    for folder in (display, metric, other):
        folder.mkdir(parents=True)
        (folder / "waveforms.bin").write_text("scratch")

    assert remove_analyzer_cache(sid) is True
    assert not display.exists()
    assert not metric.exists()
    # A different sort's folder is left in place.
    assert other.exists()


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
        folder = tmp_path / "sort.zarr"

        build_analyzer(
            sorting,
            recording,
            key={"sorting_id": "test-3d-probe"},
            sorter_row={"job_kwargs": {}},
            job_kwargs={},
            analyzer_folder=folder,
            waveform_params=_DISPLAY_PARAMS,
        )

        analyzer = si.load_sorting_analyzer(folder)
        assert analyzer.get_probe().ndim == 2

    def test_unit_locations_computes_after_projection(
        self, recording_3d_and_sorting, tmp_path
    ):
        # Without the 2D projection this raises ValueError:
        # "could not broadcast input array from shape (3,) into shape (2,)".
        recording, sorting = recording_3d_and_sorting
        folder = tmp_path / "sort.zarr"

        build_analyzer(
            sorting,
            recording,
            key={"sorting_id": "test-3d-probe"},
            sorter_row={"job_kwargs": {}},
            job_kwargs={},
            analyzer_folder=folder,
            waveform_params=_DISPLAY_PARAMS,
        )

        analyzer = si.load_sorting_analyzer(folder)
        analyzer.compute("unit_locations")
        unit_locations = analyzer.get_extension("unit_locations").get_data()
        assert unit_locations.shape[0] == sorting.get_num_units()


class TestBuildAnalyzerWaveformParams:
    """``build_analyzer`` reads window / subsample from the resolved params.

    The window + subsample are no longer hardcoded -- they come from the
    resolved ``AnalyzerWaveformParameters`` blob the caller threads in. These
    drive ``build_analyzer`` directly with each region's resolved dict (no name
    lookup, no DB read), then read the window / cap back off the built
    analyzer's ``waveforms`` / ``random_spikes`` extensions.
    """

    @staticmethod
    def _build(recording, sorting, folder, waveform_params):
        build_analyzer(
            sorting,
            recording,
            key={"sorting_id": "test-window"},
            sorter_row={"job_kwargs": {}},
            job_kwargs={},
            analyzer_folder=folder,
            waveform_params=waveform_params,
        )
        return si.load_sorting_analyzer(folder)

    def test_hippocampus_window(self, recording_3d_and_sorting, tmp_path):
        recording, sorting = recording_3d_and_sorting
        analyzer = self._build(
            recording,
            sorting,
            tmp_path / "hippo.zarr",
            {
                "ms_before": 0.5,
                "ms_after": 0.5,
                "max_spikes_per_unit": 20000,
                "whiten": False,
                "purpose": "display",
                "schema_version": 1,
            },
        )
        wf = analyzer.get_extension("waveforms").params
        rs = analyzer.get_extension("random_spikes").params
        assert wf["ms_before"] == 0.5 and wf["ms_after"] == 0.5
        assert rs["max_spikes_per_unit"] == 20000

    def test_cortex_window(self, recording_3d_and_sorting, tmp_path):
        recording, sorting = recording_3d_and_sorting
        analyzer = self._build(
            recording,
            sorting,
            tmp_path / "cortex.zarr",
            {
                "ms_before": 1.0,
                "ms_after": 2.0,
                "max_spikes_per_unit": 20000,
                "whiten": False,
                "purpose": "display",
                "schema_version": 1,
            },
        )
        wf = analyzer.get_extension("waveforms").params
        rs = analyzer.get_extension("random_spikes").params
        assert wf["ms_before"] == 1.0 and wf["ms_after"] == 2.0
        assert rs["max_spikes_per_unit"] == 20000

    def test_analyzer_folder_required(self, recording_3d_and_sorting):
        """Omitting ``analyzer_folder`` raises (the caller must resolve it)."""
        recording, sorting = recording_3d_and_sorting
        with pytest.raises(ValueError, match="analyzer_folder is required"):
            build_analyzer(
                sorting,
                recording,
                key={"sorting_id": "test-window"},
                sorter_row={"job_kwargs": {}},
                job_kwargs={},
            )

    def test_rejects_whitened_metric_recipe(
        self, recording_3d_and_sorting, tmp_path
    ):
        """A whitened (metric) recipe is rejected -- only the unwhitened
        display analyzer is built here. Prevents a direct caller from silently
        getting an unwhitened analyzer at a metric-named folder before the
        whitened build lands."""
        recording, sorting = recording_3d_and_sorting
        with pytest.raises(NotImplementedError, match="whitened"):
            build_analyzer(
                sorting,
                recording,
                key={"sorting_id": "test-metric"},
                sorter_row={"job_kwargs": {}},
                job_kwargs={},
                analyzer_folder=tmp_path / "metric.zarr",
                waveform_params={
                    "ms_before": 0.5,
                    "ms_after": 0.5,
                    "max_spikes_per_unit": 20000,
                    "whiten": True,
                    "purpose": "metric",
                    "schema_version": 1,
                },
            )
