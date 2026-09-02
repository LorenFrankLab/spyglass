"""v0/v1 spike-sorting stays importable + queryable under modern SpikeInterface.

The default install pins SpikeInterface 0.104 (the v2 baseline). Active v0/v1
*compute* (``make`` / waveform extraction) is gated behind
``_require_legacy_si_environment`` (raises under SI >= 0.101), but **read
access** must keep working: a user with existing v0/v1 sorts has to be able to
import the table modules and query / fetch their rows without a legacy
SpikeInterface install. This pins that contract so a future change that adds a
top-level removed-API import (breaking import, hence table access) fails here
instead of silently locking users out of their data.

Runs in the main ``run-tests`` job (SI 0.104); the legacy job tests the
compute paths under SI 0.99. The genuinely-removed SI APIs
(``WaveformExtractor`` / ``extract_waveforms`` / ``load_waveforms``) are
lazy-imported inside the guarded compute functions, so module import does not
touch them.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from packaging.version import Version

# The SI-bearing v0/v1 table modules (those with top-level SpikeInterface
# imports and ``_require_legacy_si_environment`` guards on their compute paths).
# These are the modules whose import would break first if a removed SI API were
# imported at module scope. The figurl / sortingview modules are intentionally
# omitted: they gate on optional sharing extras, not on the SI version.
_LEGACY_TABLE_MODULES = [
    "spyglass.spikesorting.v1.artifact",
    "spyglass.spikesorting.v1.curation",
    "spyglass.spikesorting.v1.metric_curation",
    "spyglass.spikesorting.v1.burst_curation",
    "spyglass.spikesorting.v1.recording",
    "spyglass.spikesorting.v1.sorting",
    "spyglass.spikesorting.v1.recompute",
    "spyglass.spikesorting.v0.spikesorting_artifact",
    "spyglass.spikesorting.v0.spikesorting_curation",
    "spyglass.spikesorting.v0.spikesorting_burst",
    "spyglass.spikesorting.v0.spikesorting_recording",
    "spyglass.spikesorting.v0.spikesorting_sorting",
    "spyglass.spikesorting.v0.spikesorting_recompute",
]


def test_compat_loader_dispatches_to_available_si_api(monkeypatch):
    """The loader wrapper selects the API exposed by the installed SI."""
    import spikeinterface as si

    from spyglass.spikesorting import _si_compat

    source = {"serialized": "extractor"}
    loaded = object()
    calls = []

    def _load(value):
        calls.append(value)
        return loaded

    loader_name = (
        "load_extractor"
        if callable(getattr(si, "load_extractor", None))
        else "load"
    )
    monkeypatch.setattr(si, loader_name, _load)

    assert _si_compat.load_extractor(source) is loaded
    assert calls == [source]


def test_compat_numpy_sorting_constructor_preserves_unit_ids():
    """The constructor wrapper works across SI names and keeps empty units."""
    from spyglass.spikesorting._si_compat import (
        numpy_sorting_from_samples_and_labels,
    )

    samples = np.asarray([5, 10, 15], dtype=np.int64)
    labels = np.asarray([2, 2, 10], dtype=np.int64)
    sorting = numpy_sorting_from_samples_and_labels(
        samples,
        labels,
        sampling_frequency=30_000.0,
        unit_ids=[2, 10, 99],
    )

    assert list(sorting.get_unit_ids()) == [2, 10, 99]
    np.testing.assert_array_equal(
        sorting.get_unit_spike_train(unit_id=2), [5, 10]
    )
    np.testing.assert_array_equal(
        sorting.get_unit_spike_train(unit_id=10), [15]
    )
    np.testing.assert_array_equal(sorting.get_unit_spike_train(unit_id=99), [])


def test_v0_v1_read_paths_under_modern_si(dj_conn, monkeypatch):
    """Legacy recording and curation reads reach the compatibility shim."""
    import spikeinterface as si

    if Version(si.__version__) < Version("0.101"):
        pytest.skip("Modern-SI read-path regression test")

    from spyglass.spikesorting import _si_compat
    from spyglass.spikesorting.v0.spikesorting_curation import Curation
    from spyglass.spikesorting.v0.spikesorting_recording import (
        SpikeSortingRecording,
    )

    loaded = object()
    sources = []

    def _load(source):
        sources.append(source)
        return loaded

    monkeypatch.setattr(_si_compat, "load_extractor", _load)
    monkeypatch.setattr(
        SpikeSortingRecording,
        "_fetch_recording_path",
        lambda self, key: "recording-folder",
    )
    monkeypatch.setattr(
        Curation,
        "_load_sorting_info",
        lambda self, key: ("sorting-folder", []),
    )

    assert SpikeSortingRecording().load_recording({}) is loaded
    assert Curation().get_curated_sorting({}) is loaded
    assert sources == ["recording-folder", "sorting-folder"]


@pytest.mark.parametrize("module_name", _LEGACY_TABLE_MODULES)
def test_legacy_table_module_imports_under_modern_si(dj_conn, module_name):
    """Each SI-bearing v0/v1 table module imports under the modern SI pin.

    A top-level import of an API removed in SI 0.101+ would raise here -- which
    would also block read access to that table's rows. ``dj_conn`` is required
    because importing a ``@schema`` module declares its tables.
    """
    try:
        importlib.import_module(module_name)
    except (
        Exception
    ) as exc:  # noqa: BLE001 - report the offending module clearly
        import spikeinterface

        pytest.fail(
            f"{module_name} failed to import under SpikeInterface "
            f"{spikeinterface.__version__}: {type(exc).__name__}: {exc}. "
            "v0/v1 table modules must stay importable so existing rows remain "
            "queryable; only compute is gated."
        )


def test_legacy_tables_are_queryable_under_modern_si(dj_conn):
    """Existing v0/v1 rows stay readable: a table query does not trip the guard.

    Read access (``len`` / ``fetch``) must work under SI 0.104 -- the legacy
    guard fires only on compute (``make`` / waveform extraction), never on a
    query. An empty test database is fine; the assertion is that the query
    returns rather than raising ``RuntimeError`` from the legacy guard.
    """
    from spyglass.spikesorting.v0.spikesorting_curation import (
        CuratedSpikeSorting,
    )
    from spyglass.spikesorting.v1.sorting import SpikeSorting

    # Declaring + counting (an empty result is expected) exercises schema
    # declaration and a fetch without touching the guarded compute path.
    assert len(SpikeSorting()) >= 0
    assert len(CuratedSpikeSorting()) >= 0
