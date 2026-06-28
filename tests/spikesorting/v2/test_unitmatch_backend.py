"""Tests for the UnitMatch matcher backend.

The backend is the concrete MatcherProtocol implementation. Registration and
the degenerate single-session case are hermetic; the end-to-end match is an
integration test that builds two bundles from the committed polymer fixture
(pseudo-session split of the same ground-truth neurons) and checks that planted
correspondences score above random pairs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "mearec_polymer_128ch_60s.nwb"
)


def test_backend_registers_unitmatch():
    import spyglass.spikesorting.v2._unitmatch_backend  # noqa: F401 (registers)
    from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema
    from spyglass.spikesorting.v2.matcher_protocol import (
        _get_matcher_schema,
        get_matcher,
    )

    assert get_matcher("unitmatch").name == "unitmatch"
    assert _get_matcher_schema("unitmatch") is UnitMatchParamsSchema


def test_match_single_session_returns_empty():
    from spyglass.spikesorting.v2._unitmatch_backend import UnitMatchBackend
    from spyglass.spikesorting.v2.matcher_protocol import SessionMatcherInput

    one = SessionMatcherInput(
        session_key={"sorting_id": "s", "curation_id": 0},
        waveform_dir=Path("/tmp/does-not-matter"),
        channel_positions_path=Path("/tmp/does-not-matter/cp.npy"),
    )
    assert UnitMatchBackend().match([one], {}) == []


def _two_one_unit_sessions():
    from spyglass.spikesorting.v2.matcher_protocol import SessionMatcherInput

    return [
        SessionMatcherInput(
            session_key={"sorting_id": "A", "curation_id": 0},
            waveform_dir=Path("/x"),
            channel_positions_path=Path("/x/cp.npy"),
        ),
        SessionMatcherInput(
            session_key={"sorting_id": "B", "curation_id": 1},
            waveform_dir=Path("/y"),
            channel_positions_path=Path("/y/cp.npy"),
        ),
    ]


def test_one_directional_match_is_rejected():
    """A pair above threshold in only one CV direction is not emitted."""
    from spyglass.spikesorting.v2._unitmatch_backend import UnitMatchBackend

    inputs = _two_one_unit_sessions()
    session_switch = np.array([0, 1, 2])
    original_ids = np.array([[10], [20]])
    one_sided = np.array([[0.0, 0.9], [0.2, 0.0]])  # M[1,0]=0.2 below threshold
    pairs = UnitMatchBackend._pairs_from_matrix(
        one_sided, session_switch, original_ids, inputs, 0.5
    )
    assert pairs == []


def test_bidirectional_match_reports_mean_probability():
    """Both CV directions above threshold -> pair with the mean probability."""
    from spyglass.spikesorting.v2._unitmatch_backend import UnitMatchBackend

    inputs = _two_one_unit_sessions()
    session_switch = np.array([0, 1, 2])
    original_ids = np.array([[10], [20]])
    both = np.array([[0.0, 0.9], [0.8, 0.0]])
    pairs = UnitMatchBackend._pairs_from_matrix(
        both, session_switch, original_ids, inputs, 0.5
    )
    assert len(pairs) == 1
    assert pairs[0].unit_a_id == 10 and pairs[0].unit_b_id == 20
    assert pairs[0].session_a_sorting_id == "A"
    assert pairs[0].match_probability == pytest.approx(0.85)


def test_match_raises_if_unitmatch_drops_a_session(tmp_path, monkeypatch):
    """A dropped bundle (fewer loaded sessions than inputs) fails loudly.

    UnitMatchPy's load_good_waveforms excludes a session whose bundle fails to
    load instead of raising; without a guard, the compact session indexes would
    misattribute one session's units to another session's Spyglass key.
    """
    from types import SimpleNamespace

    from spyglass.spikesorting.v2 import _unitmatch_backend as backend
    from spyglass.spikesorting.v2.matcher_protocol import SessionMatcherInput

    positions = tmp_path / "cp.npy"
    np.save(positions, np.zeros((4, 2)))
    inputs = [
        SessionMatcherInput(
            session_key={"sorting_id": s, "curation_id": 0},
            waveform_dir=tmp_path,
            channel_positions_path=positions,
        )
        for s in ("A", "B")
    ]

    # Fake UnitMatch namespace: load_good_waveforms returns only ONE session's
    # good_units while two inputs were provided (a silently dropped session).
    def _fake_load(wave_paths, label_paths, param, good_units_only=True):
        param["n_units"], param["n_sessions"] = 1, 1
        return (
            np.zeros((1, 10, 4, 2)),  # waveform
            np.array([0]),  # session_id
            np.array([0, 1]),  # session_switch
            np.zeros((1, 1)),  # within_session
            [np.array([[5]])],  # good_units -- length 1, not 2
            param,
        )

    fake_um = SimpleNamespace(
        default_params=SimpleNamespace(
            get_default_param=lambda: {"match_threshold": 0.5}
        ),
        utils=SimpleNamespace(
            paths_from_KS=lambda dirs: ([], [], [np.zeros((4, 3))]),
            get_probe_geometry=lambda pos, param: param,
            load_good_waveforms=_fake_load,
        ),
    )
    monkeypatch.setattr(backend, "_require_unitmatch", lambda: fake_um)

    with pytest.raises(RuntimeError, match="loaded 1 session"):
        backend.UnitMatchBackend().match(inputs, {})


def test_match_returns_empty_when_no_good_units(tmp_path, monkeypatch):
    """Two sessions that load zero good units -> no pairs, no divide-by-zero.

    The prior-probability computation divides by ``n_units ** 2``; the backend
    must return early (it is a public MatcherProtocol impl and cannot assume the
    table layer's non-empty-matchable precondition).
    """
    from types import SimpleNamespace

    from spyglass.spikesorting.v2 import _unitmatch_backend as backend
    from spyglass.spikesorting.v2.matcher_protocol import SessionMatcherInput

    positions = tmp_path / "cp.npy"
    np.save(positions, np.zeros((4, 2)))
    inputs = [
        SessionMatcherInput(
            session_key={"sorting_id": s, "curation_id": 0},
            waveform_dir=tmp_path,
            channel_positions_path=positions,
        )
        for s in ("A", "B")
    ]

    # Both sessions load (len(good_units) == len(inputs)) but with n_units == 0.
    def _fake_load(wave_paths, label_paths, param, good_units_only=True):
        param["n_units"], param["n_sessions"] = 0, 2
        return (
            np.zeros((0, 10, 4, 2)),  # waveform
            np.array([], dtype=int),  # session_id
            np.array([0, 0, 0]),  # session_switch
            np.zeros((0, 0)),  # within_session
            [np.empty((0, 1)), np.empty((0, 1))],  # good_units -- len 2, empty
            param,
        )

    fake_um = SimpleNamespace(
        default_params=SimpleNamespace(
            get_default_param=lambda: {"match_threshold": 0.5}
        ),
        utils=SimpleNamespace(
            paths_from_KS=lambda dirs: ([], [], [np.zeros((4, 3))]),
            get_probe_geometry=lambda pos, param: param,
            load_good_waveforms=_fake_load,
        ),
    )
    monkeypatch.setattr(backend, "_require_unitmatch", lambda: fake_um)

    assert backend.UnitMatchBackend().match(inputs, {}) == []


def test_match_rejects_mismatched_probe_geometry(tmp_path, monkeypatch):
    """Sessions with different channel geometry are rejected up front.

    UnitMatch assumes one probe across the group (it derives geometry from the
    first session and runs per-channel loops); cross-probe matching is out of
    scope, so a channel-position mismatch must raise a clear error before
    UnitMatch runs rather than failing deep in a shape mismatch.
    """
    from types import SimpleNamespace

    from spyglass.spikesorting.v2 import _unitmatch_backend as backend
    from spyglass.spikesorting.v2.matcher_protocol import SessionMatcherInput

    cp_a = tmp_path / "cp_a.npy"
    cp_b = tmp_path / "cp_b.npy"
    np.save(cp_a, np.zeros((4, 2)))
    np.save(
        cp_b, np.zeros((8, 2))
    )  # different channel count -> different probe
    inputs = [
        SessionMatcherInput(
            session_key={"sorting_id": "A", "curation_id": 0},
            waveform_dir=tmp_path,
            channel_positions_path=cp_a,
        ),
        SessionMatcherInput(
            session_key={"sorting_id": "B", "curation_id": 0},
            waveform_dir=tmp_path,
            channel_positions_path=cp_b,
        ),
    ]
    # Past the import guard; the geometry check raises before any UnitMatch call.
    fake_um = SimpleNamespace(
        default_params=SimpleNamespace(
            get_default_param=lambda: {"match_threshold": 0.5}
        )
    )
    monkeypatch.setattr(backend, "_require_unitmatch", lambda: fake_um)

    with pytest.raises(ValueError, match="same probe|probe geometry"):
        backend.UnitMatchBackend().match(inputs, {})


def test_bundle_geometry_is_2d(tmp_path, monkeypatch):
    """A 3D-probe recording yields a saved ``(n_channels, 2)``
    ``channel_positions.npy``.

    Spyglass stores 3D electrode geometry (z typically 0), but the UnitMatch
    matcher contract requires 2D channel positions; the bundle projects the
    probe to 2D (mirroring the analyzer path) and guards the saved shape so a 3D
    probe never reaches the matcher silently. UnitMatch is faked so the test
    does not require the optional UnitMatchPy extra.
    """
    import types

    import spikeinterface.full as si

    from spyglass.spikesorting.v2 import _unitmatch_backend as backend

    fake_um = types.SimpleNamespace(
        extract_raw_data=types.SimpleNamespace(
            save_avg_waveforms=lambda *a, **k: None
        )
    )
    monkeypatch.setattr(backend, "_require_unitmatch", lambda: fake_um)

    recording, sorting = si.generate_ground_truth_recording(
        durations=[2.0], num_channels=8, num_units=3, seed=0
    )
    recording_3d = recording.set_probe(recording.get_probe().to_3d(axes="xy"))
    assert recording_3d.get_probe().ndim == 3

    backend.extract_unitmatch_bundle(
        tmp_path / "sess", recording_3d, sorting, seed=0
    )
    positions = np.load(tmp_path / "sess" / "channel_positions.npy")
    assert positions.shape == (8, 2), positions.shape


def test_bundle_rejects_non_2d_positions(tmp_path, monkeypatch):
    """The bundle guards the 2D contract: a non-``(n, 2)`` channel-positions
    array is rejected up front (before the expensive split-half build) rather
    than handed to the matcher."""
    import types

    import spikeinterface.full as si

    from spyglass.spikesorting.v2 import _unitmatch_backend as backend

    fake_um = types.SimpleNamespace(
        extract_raw_data=types.SimpleNamespace(
            save_avg_waveforms=lambda *a, **k: None
        )
    )
    monkeypatch.setattr(backend, "_require_unitmatch", lambda: fake_um)

    recording, sorting = si.generate_ground_truth_recording(
        durations=[2.0], num_channels=8, num_units=3, seed=0
    )
    # Force a 3D positions array past the probe (the planar probe needs no
    # projection, so this exercises the shape guard directly).
    monkeypatch.setattr(
        recording, "get_channel_locations", lambda *a, **k: np.zeros((8, 3))
    )
    with pytest.raises(ValueError, match=r"2D"):
        backend.extract_unitmatch_bundle(
            tmp_path / "sess", recording, sorting, seed=0
        )


def test_get_matcher_bootstraps_default_after_clear():
    """get_matcher re-registers the built-in backend even if the registry was cleared."""
    from spyglass.spikesorting.v2 import matcher_protocol as mp

    saved_m, saved_s = dict(mp._MATCHER_REGISTRY), dict(mp._SCHEMA_REGISTRY)
    try:
        mp._MATCHER_REGISTRY.clear()
        mp._SCHEMA_REGISTRY.clear()
        assert mp.get_matcher("unitmatch").name == "unitmatch"
    finally:
        mp._MATCHER_REGISTRY.clear()
        mp._MATCHER_REGISTRY.update(saved_m)
        mp._SCHEMA_REGISTRY.clear()
        mp._SCHEMA_REGISTRY.update(saved_s)


def _read_polymer_gt():
    """Return (recording, gt_spike_trains_seconds) from the polymer fixture."""
    import spikeinterface.preprocessing as spre
    from pynwb import NWBHDF5IO
    from spikeinterface.extractors import read_nwb_recording

    rec = read_nwb_recording(
        str(FIXTURE), electrical_series_path="acquisition/e-series"
    )
    rec = spre.bandpass_filter(rec, freq_min=300.0, freq_max=6000.0)
    rec = rec.set_probe(rec.get_probe().to_2d())
    with NWBHDF5IO(str(FIXTURE), "r") as io:
        gt = io.read().processing["ground_truth"].data_interfaces["units"]
        trains = {
            int(u): np.asarray(gt["spike_times"][i])
            for i, u in enumerate(gt.id[:])
        }
    return rec, trains


@pytest.fixture(scope="module")
def two_session_inputs(tmp_path_factory):
    """Build two bundles with a known overlapping unit set (8 shared)."""
    from spikeinterface.core import NumpySorting

    from spyglass.spikesorting.v2._unitmatch_backend import (
        extract_unitmatch_bundle,
    )
    from spyglass.spikesorting.v2.matcher_protocol import SessionMatcherInput

    if not FIXTURE.exists():
        pytest.skip("polymer 60s fixture not present")
    # UnitMatchPy is the optional `spikesorting-v2-matching` extra; the default
    # v2 CI env does not install it, so skip cleanly rather than fail.
    pytest.importorskip("UnitMatchPy")

    rec, trains = _read_polymer_gt()
    fs = rec.get_sampling_frequency()
    n = rec.get_num_samples()
    mid = n // 2
    s1_ids, s2_ids = list(range(0, 16)), list(range(8, 24))
    shared = sorted(set(s1_ids) & set(s2_ids))

    def sorting_for(ids):
        labels = np.concatenate([np.full(len(trains[u]), u) for u in ids])
        times = np.concatenate([trains[u] for u in ids])
        order = np.argsort(times)
        return NumpySorting.from_times_and_labels(
            times[order], labels[order], sampling_frequency=fs
        )

    root = tmp_path_factory.mktemp("um_bundles")
    inputs = []
    for name, ids, (a, b) in [
        ("S1", s1_ids, (0, mid)),
        ("S2", s2_ids, (mid, n)),
    ]:
        sdir = root / name
        extract_unitmatch_bundle(
            sdir,
            rec.frame_slice(a, b),
            sorting_for(ids).frame_slice(a, b),
        )
        inputs.append(
            SessionMatcherInput(
                session_key={"sorting_id": name, "curation_id": 0},
                waveform_dir=sdir,
                channel_positions_path=sdir / "channel_positions.npy",
            )
        )
    return inputs, s1_ids, s2_ids, shared


@pytest.mark.slow
@pytest.mark.integration
def test_match_recovers_planted_correspondences(two_session_inputs):
    from spyglass.spikesorting.v2._unitmatch_backend import UnitMatchBackend

    inputs, _s1_ids, _s2_ids, shared = two_session_inputs
    pairs = UnitMatchBackend().match(inputs, {"match_threshold": 0.5})

    # every pair spans the two sessions and carries both side keys
    assert pairs, "expected non-empty matches on planted correspondences"
    for p in pairs:
        assert p.session_a_sorting_id != p.session_b_sorting_id
        assert 0.0 <= p.match_probability <= 1.0

    # the planted (shared GT unit) pairs should be recovered as high-prob matches
    # shared GT units keep the same unit_id in both sessions, so a planted
    # correspondence surfaces as the (u, u) pair.
    found = {
        (p.unit_a_id, p.unit_b_id) for p in pairs if p.match_probability > 0.5
    }
    recovered = sum(1 for u in shared if (u, u) in found)
    assert (
        recovered >= len(shared) - 1
    ), f"recovered only {recovered}/{len(shared)} planted correspondences"
