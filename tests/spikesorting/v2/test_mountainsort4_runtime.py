"""Native MountainSort4 runtime checks and a real sort, without a database."""

import importlib
import uuid
from types import SimpleNamespace

import numpy as np
import pytest


def _runtime_checks():
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2._orchestration.preflight import (
        _check_local_sorter_runtime,
    )

    checks = {}

    def check(name, ok, detail):
        checks[name] = (bool(ok), detail)
        return bool(ok)

    _check_local_sorter_runtime(
        SimpleNamespace(sorter="mountainsort4"), sis, set(), check
    )
    return checks


def test_ms4_preflight_does_not_require_legacy_backend(monkeypatch):
    """An absent historical backend must not reject the bundled algorithm."""
    real_import = importlib.import_module
    imported = []

    def without_legacy_backend(name, *args, **kwargs):
        imported.append(name)
        if name == "ml_ms4alg":
            raise ModuleNotFoundError("No module named 'ml_ms4alg'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", without_legacy_backend)
    checks = _runtime_checks()
    assert checks["sorter_installed"][0]
    assert checks["sorter_runtime_available"][0], checks
    assert "mountainsort4" in imported
    assert "ml_ms4alg" not in imported


@pytest.mark.parametrize("error", [ImportError, OSError])
def test_ms4_preflight_rejects_broken_runtime(monkeypatch, error):
    """Finding the package must not hide failed compiled dependency imports."""
    real_import = importlib.import_module

    def broken_runtime(name, *args, **kwargs):
        if name == "mountainsort4":
            raise error("MS4 compiled dependency unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", broken_runtime)
    checks = _runtime_checks()
    assert checks["sorter_installed"][0]
    ok, detail = checks["sorter_runtime_available"]
    assert not ok
    assert "MS4 compiled dependency unavailable" in detail


def test_native_ms4_recovers_planted_spikes(tmp_path, monkeypatch):
    """Execute MS4 on NumPy 2 through v2, including its old-API compatibility."""
    import spikeinterface as si
    from spikeinterface.comparison import compare_sorter_to_ground_truth

    import spyglass.settings as settings
    from spyglass.spikesorting.v2._sorting.dispatch import run_si_sorter

    assert int(np.__version__.split(".")[0]) >= 2
    assert not hasattr(np, "Inf")
    monkeypatch.setattr(settings, "temp_dir", str(tmp_path))
    recording, ground_truth = si.generate_ground_truth_recording(
        durations=[10.0],
        sampling_frequency=30_000,
        num_channels=4,
        num_units=2,
        seed=42,
    )
    # The MS4 worker needs a recording it can reload in a child process.
    recording = recording.save(
        folder=tmp_path / "recording", n_jobs=1, progress_bar=False
    )
    sorting = run_si_sorter(
        "mountainsort4",
        {
            "filter": False,
            "whiten": True,
            "num_workers": 1,
            "detect_threshold": 3,
            "clip_size": 50,
            "detect_interval": 10,
            "adjacency_radius": -1,
        },
        recording,
        uuid.uuid4(),
        {"n_jobs": 1, "random_seed": 0},
    )
    assert not hasattr(np, "Inf")
    assert not list(tmp_path.glob("sort_*"))
    spikes = sorting.to_spike_vector()
    assert spikes.size > 0
    assert np.all(spikes["sample_index"] >= 0)
    assert np.all(spikes["sample_index"] < recording.get_num_samples())
    # Check recovered physical events rather than merely accepting nonempty
    # output. MS4 can oversplit, so require one planted unit at >=70% accuracy.
    performance = compare_sorter_to_ground_truth(
        ground_truth, sorting
    ).get_performance()
    assert (performance["accuracy"] >= 0.7).any(), performance
