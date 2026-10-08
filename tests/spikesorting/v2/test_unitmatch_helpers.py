"""The matcher fixture helper intercepts preparation and restores plugin state."""

import sys
from types import ModuleType

import numpy as np
import pytest
import spikeinterface as si

from spyglass.spikesorting.v2 import matcher_protocol as protocol
from tests.spikesorting.v2._unitmatch_helpers import (
    install_fixture_pairer,
    restore_matcher_registry,
)


@pytest.fixture
def fixture_lookup(monkeypatch):
    """Only the parameter-table insert needs a stand-in for these helper checks."""
    for registry in (
        "_MATCHER_REGISTRY",
        "_SCHEMA_REGISTRY",
        "_PREPARER_REGISTRY",
    ):
        monkeypatch.setattr(
            protocol, registry, dict(getattr(protocol, registry))
        )
    module = ModuleType("spyglass.spikesorting.v2.unit_matching")

    class Parameters:
        failure = None

        @classmethod
        def insert1(cls, row, **kwargs):
            if cls.failure is not None:
                raise cls.failure

    module.MatcherParameters = Parameters
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return Parameters


def _source(name):
    sorting = si.NumpySorting.from_unit_dict(
        {7: np.array([100, 200, 300]), 19: np.array([120])}, 1000
    )
    recording = si.NumpyRecording(np.ones((500, 2), dtype=np.float32), 1000)
    recording.set_channel_locations([[0, 0], [0, 20]])
    recording.set_channel_gains(1.0)
    recording.set_channel_offsets(0.0)
    return protocol.MatcherInputSource(
        {"sorting_id": name, "curation_id": 0},
        recording,
        sorting,
        "2026-01-01",
        [(0, 500)],
    )


def _registry_snapshot():
    return tuple(
        dict(registry)
        for registry in (
            protocol._MATCHER_REGISTRY,
            protocol._SCHEMA_REGISTRY,
            protocol._PREPARER_REGISTRY,
        )
    )


def test_fixture_stub_observes_units_without_real_extraction(
    fixture_lookup, monkeypatch, tmp_path
):
    from spyglass.spikesorting.v2._matching import (
        waveforms as _waveform_bundles,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Stubbed fixture ran waveform extraction")

    monkeypatch.setattr(_waveform_bundles, "extract_waveform_bundle", forbidden)
    before = _registry_snapshot()
    seen = []
    saved = install_fixture_pairer(
        monkeypatch,
        matcher_name="helper_stub",
        matcher_params_name="helper_stub",
        pairs=[],
        seen_unit_ids=seen,
    )
    try:
        prepared = protocol.get_input_preparer("helper_stub").prepare(
            _source("a"), tmp_path / "bundle", {}, {"n_jobs": 1}
        )
        assert seen == [[7, 19]]
        assert prepared.session_input.bundle_dir.is_dir()
        assert not list(prepared.session_input.bundle_dir.iterdir())
        assert prepared.excluded_unit_ids == ()
    finally:
        restore_matcher_registry(saved)
    assert _registry_snapshot() == before


def test_real_bundle_fixture_pairs_follow_retained_unit_ids(
    fixture_lookup, monkeypatch, tmp_path
):
    seen = []
    saved = install_fixture_pairer(
        monkeypatch,
        matcher_name="helper_real",
        matcher_params_name="helper_real",
        pairs=[[7, 7], [19, 19]],
        read_bundles=True,
        seen_unit_ids=seen,
    )
    try:
        preparer = protocol.get_input_preparer("helper_real")
        inputs = [
            preparer.prepare(_source(name), tmp_path / name, {}, {"n_jobs": 1})
            for name in ("a", "b")
        ]
        assert [item.excluded_unit_ids for item in inputs] == [(19,), (19,)]
        assert all(
            item.session_input.geometry_path.is_file() for item in inputs
        )
        pairs = protocol.get_matcher("helper_real").match(
            [item.session_input for item in inputs],
            {"pairs": [[7, 7], [19, 19]], "probability": 0.99},
        )
        assert seen == [[7], [7]]
        assert [(pair.unit_a_id, pair.unit_b_id) for pair in pairs] == [(7, 7)]
    finally:
        restore_matcher_registry(saved)


def test_lookup_failure_restores_every_registry(fixture_lookup, monkeypatch):
    before = _registry_snapshot()
    fixture_lookup.failure = RuntimeError("Parameter insert failed")
    with pytest.raises(RuntimeError, match="Parameter insert failed"):
        install_fixture_pairer(
            monkeypatch,
            matcher_name="helper_failed",
            matcher_params_name="helper_failed",
            pairs=[],
        )
    assert _registry_snapshot() == before
