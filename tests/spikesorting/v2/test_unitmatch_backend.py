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
        curation_key={"sorting_id": "s", "curation_id": 0},
        waveform_dir=Path("/tmp/does-not-matter"),
        channel_positions_path=Path("/tmp/does-not-matter/cp.npy"),
    )
    assert UnitMatchBackend().match([one], {}) == []


def _two_one_unit_sessions():
    from spyglass.spikesorting.v2.matcher_protocol import SessionMatcherInput

    return [
        SessionMatcherInput(
            curation_key={"sorting_id": "A", "curation_id": 0},
            waveform_dir=Path("/x"),
            channel_positions_path=Path("/x/cp.npy"),
        ),
        SessionMatcherInput(
            curation_key={"sorting_id": "B", "curation_id": 1},
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
            curation_key={"sorting_id": s, "curation_id": 0},
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
            curation_key={"sorting_id": s, "curation_id": 0},
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
            curation_key={"sorting_id": "A", "curation_id": 0},
            waveform_dir=tmp_path,
            channel_positions_path=cp_a,
        ),
        SessionMatcherInput(
            curation_key={"sorting_id": "B", "curation_id": 0},
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


# --------------------------------------------------------------------------- #
# Bundle construction on planted units                                         #
# --------------------------------------------------------------------------- #

#: Planted-session sampling rate (Hz) and channel count. At 10 kHz the default
#: 1.5 ms window is 15 samples on each side of the spike.
_FS = 10_000.0
_N_CH = 4
_HALF_WIDTH = 15
_SPIKE_WIDTH = 2 * _HALF_WIDTH


def _planted_template(channel: int, amplitude: float = 20.0) -> np.ndarray:
    """A spike template with its trough at the window centre on ``channel``.

    Returns
    -------
    template : np.ndarray, shape (spike_width, n_channels)
        Aligned with SpikeInterface's waveform window: row ``_HALF_WIDTH`` is
        the spike sample.
    """
    t = np.arange(-_HALF_WIDTH, _HALF_WIDTH, dtype=float)
    shape = -np.exp(-0.5 * (t / 2.0) ** 2) + 0.3 * np.exp(
        -0.5 * ((t - 6.0) / 3.0) ** 2
    )
    spatial = np.exp(-0.5 * ((np.arange(_N_CH) - channel) / 0.7) ** 2)
    return amplitude * shape[:, None] * spatial[None, :]


def _planted_session(duration_s, units, *, noise_std=1.0, seed=0):
    """A recording + sorting with each unit's template added at its spikes.

    Parameters
    ----------
    duration_s : float
    units : dict
        ``{unit_id: (spike_samples, template, scales)}``, in sorting order.
        ``template`` is ``(spike_width, n_channels)`` or ``None`` (spikes with
        no signal); ``scales`` is a per-spike amplitude factor or ``None``
        (all 1).
    noise_std : float
        Standard deviation of the white background noise (0 for none).
    seed : int
        Noise seed.
    """
    import probeinterface as pi
    import spikeinterface as si

    n_samples = int(duration_s * _FS)
    rng = np.random.default_rng(seed)
    traces = noise_std * rng.standard_normal((n_samples, _N_CH))
    for samples, template, scales in units.values():
        if template is None:
            continue
        scales = np.ones(len(samples)) if scales is None else scales
        for sample, scale in zip(samples, scales):
            lo, hi = sample - _HALF_WIDTH, sample + _HALF_WIDTH
            t_lo, t_hi = max(0, -lo), _SPIKE_WIDTH - max(0, hi - n_samples)
            traces[max(lo, 0) : min(hi, n_samples)] += (
                scale * template[t_lo:t_hi]
            )
    recording = si.NumpyRecording(
        [traces.astype(np.float32)], sampling_frequency=_FS
    )
    recording.set_channel_gains([1.0] * _N_CH)
    recording.set_channel_offsets([0.0] * _N_CH)
    probe = pi.generate_linear_probe(num_elec=_N_CH, ypitch=20)
    probe.set_device_channel_indices(np.arange(_N_CH))
    recording = recording.set_probe(probe)
    sorting = si.NumpySorting.from_unit_dict(
        {
            uid: np.asarray(samples, dtype=np.int64)
            for uid, (samples, _, _) in units.items()
        },
        sampling_frequency=_FS,
    )
    return recording, sorting


def _train(start_s: float, stop_s: float, period_s: float = 0.5):
    """Regularly spaced spike samples in ``[start_s, stop_s)``."""
    return np.round(np.arange(start_s, stop_s, period_s) * _FS).astype(int)


@pytest.fixture
def saved_bundles(monkeypatch):
    """Fake UnitMatchPy whose ``save_avg_waveforms`` records what it would save.

    Returns a dict ``{session_dir: {"waveforms": (n_units, spike_width,
    n_channels, 2), "unit_ids": [int, ...]}}`` filled by each save call.
    """
    import types

    from spyglass.spikesorting.v2 import _unitmatch_backend as backend

    saved = {}

    def _save(avg_waveforms, save_dir, all_unit_ids, good_units, **kwargs):
        assert kwargs == {"extract_good_units_only": False}
        assert list(all_unit_ids) == list(good_units)
        saved[Path(save_dir)] = {
            "waveforms": np.array(avg_waveforms),
            "unit_ids": [int(u) for u in all_unit_ids],
        }

    fake_um = types.SimpleNamespace(
        extract_raw_data=types.SimpleNamespace(save_avg_waveforms=_save)
    )
    monkeypatch.setattr(backend, "_require_unitmatch", lambda: fake_um)
    return saved


def _good_unit_ids(session_dir) -> list[int]:
    """Unit ids labelled ``good`` in a bundle's ``cluster_group.tsv``."""
    rows = np.loadtxt(
        Path(session_dir) / "cluster_group.tsv", dtype=str, delimiter="\t"
    )
    assert tuple(rows[0]) == ("cluster_id", "group")
    assert set(rows[1:, 1]) == {"good"}
    return [int(u) for u in rows[1:, 0]]


#: Unit 12 fires only in the first 30 s of a 60 s session; its amplitude ramps
#: from 0.8x to 1.2x across its spikes so a temporal split is distinguishable
#: from a random one. Units 7 and 3 fire throughout. Ids are sparse and not
#: sorted so an index/id mix-up changes which template lands where.
_DRIFT_OUT_TEMPLATES = {
    7: _planted_template(0),
    3: _planted_template(3),
    12: _planted_template(1, amplitude=25.0),
}
_DRIFT_OUT_SAMPLES = {
    7: _train(0.1, 60.0),
    3: _train(0.25, 60.0),
    12: _train(0.4, 30.0),
}
_DRIFT_OUT_SCALES = np.linspace(0.8, 1.2, len(_DRIFT_OUT_SAMPLES[12]))


@pytest.fixture(scope="module")
def drift_out_session():
    """60 s session with a unit that fires only in its first half."""
    return _planted_session(
        60.0,
        {
            uid: (
                _DRIFT_OUT_SAMPLES[uid],
                _DRIFT_OUT_TEMPLATES[uid],
                _DRIFT_OUT_SCALES if uid == 12 else None,
            )
            for uid in (7, 3, 12)
        },
    )


def test_bundle_halves_are_per_unit_temporal(
    tmp_path, saved_bundles, drift_out_session
):
    """A unit firing only early in the session still gets two real halves,
    split in spike-time order.

    Unit 12 has 60 spikes, all within the 200-spike draw (2 x the default
    per-half cap of 100), so cv0 is exactly its first 30 spikes and cv1 its
    last 30. Its planted amplitude ramps 0.8 -> 1.2, so each half's projection
    onto the planted template must equal that half's mean planted scale; a
    random split would give two means near 1.0.
    """
    from spyglass.spikesorting.v2 import _unitmatch_backend as backend

    recording, sorting = drift_out_session
    session_dir = tmp_path / "sess"
    excluded = backend.extract_unitmatch_bundle(
        session_dir, recording, sorting, seed=0
    )

    assert excluded == []
    saved = saved_bundles[session_dir]
    assert saved["unit_ids"] == [7, 3, 12] == _good_unit_ids(session_dir)
    waveforms = saved["waveforms"]
    assert waveforms.shape == (3, _SPIKE_WIDTH, _N_CH, 2)

    template = _DRIFT_OUT_TEMPLATES[12]
    cv0, cv1 = waveforms[2, ..., 0], waveforms[2, ..., 1]
    assert np.any(cv0 != 0) and np.any(cv1 != 0)

    # Unit-to-noise distance: how far the planted unit's mean waveform sits from
    # the noise floor's mean (zero). Unit-to-unit: from each other planted unit.
    unit_to_noise = np.linalg.norm(template)
    unit_to_unit = min(
        np.linalg.norm(template - _DRIFT_OUT_TEMPLATES[other])
        for other in (7, 3)
    )
    half_to_half = np.linalg.norm(cv0 - cv1)
    assert half_to_half < 0.5 * min(unit_to_noise, unit_to_unit), (
        half_to_half,
        unit_to_noise,
        unit_to_unit,
    )

    n_half = len(_DRIFT_OUT_SCALES) // 2
    for cv, planted_scales in (
        (cv0, _DRIFT_OUT_SCALES[:n_half]),
        (cv1, _DRIFT_OUT_SCALES[n_half:]),
    ):
        projection = np.sum(cv * template) / np.sum(template * template)
        assert projection == pytest.approx(planted_scales.mean(), abs=0.02)
        residual = cv - planted_scales.mean() * template
        assert np.linalg.norm(residual) < 0.1 * unit_to_noise


def test_bundle_no_zero_halves_invariant(tmp_path, saved_bundles):
    """A kept unit whose half is exactly all-zero fails loudly, naming the unit.

    The recording is silent except for unit 7's planted spikes, so unit 12's
    interior spikes (full waveform support, >= 2 sampled) average to exact
    zeros on every channel in both halves.
    """
    from spyglass.spikesorting.v2 import _unitmatch_backend as backend

    recording, sorting = _planted_session(
        10.0,
        {
            7: (_train(0.1, 10.0), _planted_template(0), None),
            12: (_train(0.3, 10.0), None, None),
        },
        noise_std=0.0,
    )
    with pytest.raises(RuntimeError, match="all-zero") as excinfo:
        backend.extract_unitmatch_bundle(
            tmp_path / "sess", recording, sorting, seed=0
        )
    message = str(excinfo.value)
    assert "unit 12 half 0" in message and "unit 12 half 1" in message
    assert "unit 7 " not in message
    assert saved_bundles == {}


def test_bundle_excludes_units_with_fewer_than_two_sampled_spikes(
    tmp_path, saved_bundles
):
    """Units without two sampled full-support spikes are left out and returned.

    Unit 7 has one interior spike. Unit 12 has two spikes, both closer to a
    recording border than the waveform half-width: without a sampling margin
    they would be drawn and zero-filled (an all-zero half); with it they are
    never sampled. Units 3 and 5 are healthy, and their saved arrays must be
    their own planted templates.
    """
    from spyglass.spikesorting.v2 import _unitmatch_backend as backend

    duration_s = 10.0
    n_samples = int(duration_s * _FS)
    templates = {3: _planted_template(0), 5: _planted_template(3)}
    recording, sorting = _planted_session(
        duration_s,
        {
            7: (np.array([n_samples // 2]), _planted_template(1), None),
            3: (_train(0.1, duration_s), templates[3], None),
            12: (np.array([3, n_samples - 3]), _planted_template(2), None),
            5: (_train(0.3, duration_s), templates[5], None),
        },
    )
    session_dir = tmp_path / "sess"
    excluded = backend.extract_unitmatch_bundle(
        session_dir, recording, sorting, seed=0
    )

    assert excluded == [7, 12]
    assert _good_unit_ids(session_dir) == [3, 5]
    saved = saved_bundles[session_dir]
    assert saved["unit_ids"] == [3, 5]
    assert saved["waveforms"].shape == (2, _SPIKE_WIDTH, _N_CH, 2)
    for row, uid in enumerate(saved["unit_ids"]):
        for k in (0, 1):
            residual = saved["waveforms"][row, ..., k] - templates[uid]
            assert np.linalg.norm(residual) < 0.15 * np.linalg.norm(
                templates[uid]
            ), (uid, k)


def test_bundle_all_units_excluded_raises(tmp_path, saved_bundles):
    """A session with no matchable unit raises before writing any file."""
    from spyglass.spikesorting.v2 import _unitmatch_backend as backend

    duration_s = 10.0
    n_samples = int(duration_s * _FS)
    recording, sorting = _planted_session(
        duration_s,
        {
            7: (np.array([n_samples // 2]), _planted_template(1), None),
            12: (np.array([3, n_samples - 3]), _planted_template(2), None),
        },
    )
    session_dir = tmp_path / "sess"
    with pytest.raises(backend.NoMatchableUnitsError) as excinfo:
        backend.extract_unitmatch_bundle(
            session_dir, recording, sorting, seed=0
        )
    assert isinstance(excinfo.value, ValueError)
    assert str(session_dir) in str(excinfo.value)
    assert "fewer than two" in str(excinfo.value)
    assert not session_dir.exists()
    assert saved_bundles == {}


def test_bundle_writes_kept_units_raw_waveforms(tmp_path):
    """With real UnitMatchPy, the on-disk bundle holds exactly the kept units,
    each file carrying that unit's own two halves."""
    pytest.importorskip("UnitMatchPy")
    from spyglass.spikesorting.v2 import _unitmatch_backend as backend

    duration_s = 10.0
    n_samples = int(duration_s * _FS)
    templates = {3: _planted_template(0), 5: _planted_template(3)}
    recording, sorting = _planted_session(
        duration_s,
        {
            7: (np.array([n_samples // 2]), _planted_template(1), None),
            3: (_train(0.1, duration_s), templates[3], None),
            5: (_train(0.3, duration_s), templates[5], None),
        },
    )
    session_dir = tmp_path / "sess"
    excluded = backend.extract_unitmatch_bundle(
        session_dir, recording, sorting, seed=0
    )

    assert excluded == [7]
    assert _good_unit_ids(session_dir) == [3, 5]
    files = sorted(p.name for p in (session_dir / "RawWaveforms").iterdir())
    assert files == ["Unit3_RawSpikes.npy", "Unit5_RawSpikes.npy"]
    for uid, template in templates.items():
        wave = np.load(
            session_dir / "RawWaveforms" / f"Unit{uid}_RawSpikes.npy"
        )
        assert wave.shape == (_SPIKE_WIDTH, _N_CH, 2)
        for k in (0, 1):
            residual = wave[..., k] - template
            assert np.linalg.norm(residual) < 0.15 * np.linalg.norm(template)


def test_bundle_control_matches_previous_construction(tmp_path, saved_bundles):
    """On a stationary session, per-unit temporal halves reproduce the
    recording-half templates (Pearson r > 0.99 for every unit and half).

    The previous construction is the local copy kept by the half-split
    experiment script; both constructions run on that script's control
    session A (60 s, 16 channels, 20 units that fire throughout). The spike
    cap is at least every unit's spike count, so neither construction
    subsamples: the new one averages all of a unit's interior spikes, the old
    one all of its spikes in each recording half. For a stationary unit the
    first/second half of its spikes and the recording halves then hold nearly
    the same spikes, so the correlation measures the construction rather than
    random-subset noise.
    """
    from tests.spikesorting.v2.scripts import (
        unitmatch_half_split_experiment as experiment,
    )

    from spyglass.spikesorting.v2 import _unitmatch_backend as backend

    recording, sorting = experiment.make_dataset(0)
    (rec_a, sort_a), _ = experiment.split_sessions(recording, sorting)
    new_dir, old_dir = tmp_path / "per_unit", tmp_path / "time_half"
    all_spikes = max(
        sort_a.get_unit_spike_train(uid).size for uid in sort_a.get_unit_ids()
    )
    kwargs = dict(
        ms_before=experiment.MS_BEFORE,
        ms_after=experiment.MS_AFTER,
        max_spikes_per_unit=all_spikes,
        seed=experiment.BUNDLE_SEED,
        job_kwargs=experiment.JOB_KWARGS,
    )
    excluded = backend.extract_unitmatch_bundle(
        new_dir, rec_a, sort_a, **kwargs
    )
    experiment.extract_time_half_bundle(old_dir, rec_a, sort_a, **kwargs)

    assert excluded == []
    new, old = saved_bundles[new_dir], saved_bundles[old_dir]
    assert new["unit_ids"] == old["unit_ids"]
    assert new["waveforms"].shape == old["waveforms"].shape
    n_units = len(new["unit_ids"])
    corr = np.array(
        [
            [
                np.corrcoef(
                    new["waveforms"][i, ..., k].ravel(),
                    old["waveforms"][i, ..., k].ravel(),
                )[0, 1]
                for k in (0, 1)
            ]
            for i in range(n_units)
        ]
    )
    assert np.all(corr > 0.99), dict(zip(new["unit_ids"], corr.round(4)))


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
                curation_key={"sorting_id": name, "curation_id": 0},
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


def _gate_test_record(seed, scenario, unit_ids, drift_out_units, matched):
    """Build one hand-built ``evaluate_gates`` input record.

    ``matched`` is the set of ``(i, j)`` pairs that pass matching; every other
    ``(i, j)`` pair over ``unit_ids`` counts as a miss. No UnitMatchPy or DB
    access happens here -- ``score_pairs`` is pure counting.
    """
    from tests.spikesorting.v2.scripts.unitmatch_half_split_experiment import (
        score_pairs,
    )

    scored_s = [] if scenario == "control" else list(drift_out_units)
    passing_pairs = [(i, j, 1.0) for i, j in matched]
    return {
        "seed": seed,
        "scenario": scenario,
        "condition": "per_unit",
        "unit_ids": list(unit_ids),
        "drift_out_units": scored_s,
        "seed_drift_out_units": list(drift_out_units),
        "counts": score_pairs(passing_pairs, unit_ids, scored_s),
        "fitted": None,
        "passing_pairs": passing_pairs,
    }


def _gate(gates, name, scenario):
    return next(g for g in gates if g.name == name and g.scenario == scenario)


def test_evaluate_gates_g1_g2_are_exact_at_their_boundary():
    """G1 (>=) and G2 (<=) pass exactly on the threshold, fail one count over.

    |S| = 25 makes both the diagonal denominator (25) and the off-diagonal
    denominator (25 * 24 = 600) exact multiples of the thresholds' reduced
    denominators (5 and 200), so 0.80 and 0.015 land on an exact fraction
    instead of a repeating binary float.
    """
    from tests.spikesorting.v2.scripts.unitmatch_half_split_experiment import (
        evaluate_gates,
    )

    s_units = list(range(25))
    off_diag = [(i, j) for i in s_units for j in s_units if i != j]

    def matched(n_diag, n_off_diag):
        return {(i, i) for i in s_units[:n_diag]} | set(off_diag[:n_off_diag])

    # G1 20/25 == 0.80 exactly; G2 9/600 == 0.015 exactly.
    passing = _gate_test_record(
        0, "driftout_A", s_units, s_units, matched(20, 9)
    )
    gates = evaluate_gates([passing], "per_unit")
    g1, g2 = (
        _gate(gates, "G1 drift-out recall", "driftout_A"),
        _gate(gates, "G2 SxS false-pair rate", "driftout_A"),
    )
    assert g1.passed is True, g1
    assert g2.passed is True, g2

    # One fewer drift-out true pair (19/25 == 0.76) and one more S x S false
    # pair (10/600 == 0.01667) each move one count past the boundary.
    failing = _gate_test_record(
        0, "driftout_A", s_units, s_units, matched(19, 10)
    )
    gates = evaluate_gates([failing], "per_unit")
    g1, g2 = (
        _gate(gates, "G1 drift-out recall", "driftout_A"),
        _gate(gates, "G2 SxS false-pair rate", "driftout_A"),
    )
    assert g1.passed is False, g1
    assert g2.passed is False, g2


def test_evaluate_gates_g3a_g4_are_exact_at_their_boundary():
    """G3a and G4 (paired against control) pass exactly on the threshold.

    25 non-S units make the paired denominators 25 (recall) and 600 (false
    positives), exact multiples of the thresholds' reduced denominators (25
    and 200): 1/25 == 0.04 and 3/600 == 0.005 land on an exact fraction.
    """
    from tests.spikesorting.v2.scripts.unitmatch_half_split_experiment import (
        evaluate_gates,
    )

    s_units = [999]  # a single drift-out unit; irrelevant to G3a/G4
    non_s = list(range(1000, 1025))
    unit_ids = s_units + non_s
    off_diag = [(a, b) for a in non_s for b in non_s if a != b]

    def healthy(n_true, n_false):
        return {(u, u) for u in non_s[:n_true]} | set(off_diag[:n_false])

    control = _gate_test_record(1, "control", unit_ids, s_units, healthy(25, 0))

    # control 25/25 -> scenario 24/25: drop == 1/25 == 0.04 exactly.
    # control 0/600 -> scenario 3/600: increase == 3/600 == 0.005 exactly.
    passing = _gate_test_record(
        1, "driftout_A", unit_ids, s_units, healthy(24, 3)
    )
    gates = evaluate_gates([control, passing], "per_unit")
    g3a, g4 = (
        _gate(gates, "G3a paired healthy recall drop", "driftout_A"),
        _gate(gates, "G4 paired healthy FP increase", "driftout_A"),
    )
    assert g3a.passed is True, g3a
    assert g4.passed is True, g4

    # One fewer paired recall match (23/25) pushes the drop to 2/25 == 0.08.
    failing_g3a = _gate_test_record(
        1, "driftout_A", unit_ids, s_units, healthy(23, 3)
    )
    gates = evaluate_gates([control, failing_g3a], "per_unit")
    assert (
        _gate(gates, "G3a paired healthy recall drop", "driftout_A").passed
        is False
    ), gates

    # One more false positive (4/600) pushes the increase to 4/600 == 0.00667.
    failing_g4 = _gate_test_record(
        1, "driftout_A", unit_ids, s_units, healthy(24, 4)
    )
    gates = evaluate_gates([control, failing_g4], "per_unit")
    assert (
        _gate(gates, "G4 paired healthy FP increase", "driftout_A").passed
        is False
    ), gates
