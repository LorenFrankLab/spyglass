"""The scheduled cleanup script's spike sorting v2 orphan report.

The script's table imports are stubbed, so no test here needs a database.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).resolve().parents[1] / "maintenance_scripts/cleanup.py"
V2_MODULES = (
    "spyglass.spikesorting.v2.figpack_curation",
    "spyglass.spikesorting.v2.sorting",
)


@pytest.fixture
def cleanup(monkeypatch):
    """Load cleanup.py with its database-backed imports stubbed."""
    stubs = {
        "spyglass.common": ("AnalysisNwbfile", "Nwbfile"),
        "spyglass.common.common_version": ("SpyglassVersions",),
        "spyglass.decoding.decoding_merge": ("DecodingOutput",),
        "spyglass.decoding.v1.clusterless": ("schema",),
        "spyglass.decoding.v1.sorted_spikes": ("schema",),
        "spyglass.spikesorting.v0.spikesorting_recording": (
            "SpikeSortingRecording",
        ),
        "spyglass.spikesorting.v0.spikesorting_sorting": ("SpikeSorting",),
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        for attr in attrs:
            setattr(module, attr, None)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("cleanup_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_orphan_report_skips_without_v2(cleanup, monkeypatch, capsys):
    """A v0/v1 environment that cannot import v2 skips the report."""
    for name in V2_MODULES:
        monkeypatch.setitem(sys.modules, name, None)  # import raises

    cleanup.report_v2_orphans()

    assert "Skipping spike sorting v2 orphan report" in capsys.readouterr().out


def test_v2_orphan_report_is_a_dry_run(cleanup, monkeypatch, tmp_path, capsys):
    """The report counts and sizes orphans and never asks to delete."""
    orphan = tmp_path / "orphan.analyzer"
    (orphan / "extensions").mkdir(parents=True)
    (orphan / "extensions" / "waveforms.npy").write_bytes(b"x" * 1000)
    staging = tmp_path / ".staging"
    staging.mkdir()
    (staging / "partial.bin").write_bytes(b"x" * 24)
    calls = []

    class Sorting:
        @staticmethod
        def find_orphaned_analyzer_folders(*, dry_run):
            calls.append(("analyzer", dry_run))
            return {"disk_side": [str(orphan)], "staging": [str(staging)]}

    class FigPackCuration:
        @staticmethod
        def find_orphaned_bundles(*, dry_run):
            calls.append(("figpack", dry_run))
            return {"disk_side": [], "staging": []}

    for name, table in zip(V2_MODULES, (FigPackCuration, Sorting)):
        module = types.ModuleType(name)
        setattr(module, table.__name__, table)
        monkeypatch.setitem(sys.modules, name, module)

    cleanup.report_v2_orphans()

    assert calls == [("analyzer", True), ("figpack", True)]
    out = capsys.readouterr().out
    assert "analyzer folders: 1 unreferenced, 1 staging (1,024 bytes)" in out
    assert "bundle folders: 0 unreferenced, 0 staging (0 bytes)" in out
