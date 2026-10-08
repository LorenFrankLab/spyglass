"""Runtime contracts between resolved computation and database adapters."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from spyglass.spikesorting.v2._sorting import analyzer as _sorting_analyzer


@pytest.mark.parametrize("missing", ["sorter_row", "job_kwargs"])
def test_analyzer_build_requires_resolved_execution_inputs(missing):
    """An omitted input fails before importing tables or reading artifacts."""
    kwargs = {"sorter_row": {}, "job_kwargs": {}}
    kwargs.pop(missing)
    with pytest.raises(TypeError, match=missing):
        _sorting_analyzer.build_analyzer(object(), object(), {}, **kwargs)


@pytest.mark.parametrize("unresolved", ["sorter_row", "job_kwargs"])
def test_analyzer_build_rejects_unresolved_execution_inputs(unresolved):
    """Explicit None cannot enable an implicit database/configuration fallback."""
    kwargs = {"sorter_row": {}, "job_kwargs": {}}
    kwargs[unresolved] = None
    with pytest.raises(ValueError, match="must be resolved before computation"):
        _sorting_analyzer.build_analyzer(object(), object(), {}, **kwargs)


def test_analyzer_build_does_not_resolve_ambient_configuration(
    tmp_path, monkeypatch
):
    """A build uses its supplied configuration even when globals are unusable."""
    import numpy as np
    import spikeinterface as si
    from spikeinterface.core import NumpyRecording, NumpySorting

    from spyglass.spikesorting.v2._core import job_config as _job_config

    def unexpected_lookup():
        raise AssertionError("computation consulted ambient job configuration")

    monkeypatch.setattr(_job_config, "_ambient_job_kwargs", unexpected_lookup)
    recording = NumpyRecording(
        [np.zeros((3000, 4), dtype="float32")], sampling_frequency=30_000
    )
    recording.set_property(
        "location", np.array([[0, 0], [0, 20], [0, 40], [0, 60]])
    )
    recording.set_channel_gains(1.0)
    recording.set_channel_offsets(0.0)
    sorting = NumpySorting.from_unit_dict(
        {1: np.array([500, 1500])}, sampling_frequency=30_000
    )
    calls = []

    class Analyzer:
        def compute(self, *args, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(si, "create_sorting_analyzer", lambda **kw: Analyzer())
    _sorting_analyzer.build_analyzer(
        sorting,
        recording,
        {"sorting_id": "resolved-inputs"},
        sorter_row={"job_kwargs": {"random_seed": 99}},
        job_kwargs={"random_seed": 7, "n_jobs": 1},
        analyzer_folder=tmp_path / "resolved.analyzer",
        waveform_params={
            "ms_before": 1.0,
            "ms_after": 2.0,
            "max_spikes_per_unit": 100,
            "whiten": False,
            "sparsity": {"method": "dense"},
        },
    )
    assert calls[0]["extension_params"]["random_spikes"]["seed"] == 7
    assert calls[0]["n_jobs"] == 1


def test_artifact_construction_has_no_database_imports():
    """Artifact kernels cannot gain a call-time table dependency unnoticed."""
    from spyglass.spikesorting.v2._artifacts import (
        intervals as _artifact_intervals,
    )

    tree = ast.parse(Path(inspect.getfile(_artifact_intervals)).read_text())
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("spyglass.common") or module in {
                "datajoint",
                "spyglass.spikesorting.v2.artifact",
                "spyglass.spikesorting.v2._artifacts.readers",
            }:
                forbidden.append(module)
    assert not forbidden
