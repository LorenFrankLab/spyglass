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
    import spyglass.spikesorting.v2._matching.unitmatch_backend  # noqa: F401 (registers)
    from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema
    from spyglass.spikesorting.v2.matcher_protocol import (
        _get_matcher_schema,
        get_matcher,
    )

    assert get_matcher("unitmatch").name == "unitmatch"
    assert _get_matcher_schema("unitmatch") is UnitMatchParamsSchema


def test_match_single_session_returns_empty():
    from spyglass.spikesorting.v2._matching.unitmatch_backend import (
        UnitMatchBackend,
    )
    from spyglass.spikesorting.v2.matcher_protocol import SessionMatcherInput

    one = SessionMatcherInput(
        curation_key={"sorting_id": "s", "curation_id": 0},
        bundle_dir=Path("/tmp/does-not-matter"),
        geometry_path=Path("/tmp/does-not-matter/cp.npy"),
    )
    assert UnitMatchBackend().match([one], {}) == []


@pytest.mark.parametrize(
    "metadata, message",
    [
        (
            {"layout": "features", "geometry_path": Path("/unused")},
            "requires 'split_half_waveforms'",
        ),
        ({"layout_version": 2, "geometry_path": Path("/unused")}, "version 2"),
        ({}, "requires geometry_path"),
    ],
)
def test_match_rejects_unsupported_bundle_metadata_before_importing_unitmatch(
    monkeypatch, metadata, message
):
    from spyglass.spikesorting.v2._matching import unitmatch_backend as backend
    from spyglass.spikesorting.v2.matcher_protocol import SessionMatcherInput

    def forbidden():
        raise AssertionError("Invalid bundle triggered UnitMatchPy import")

    monkeypatch.setattr(backend, "_require_unitmatch", forbidden)
    inputs = [
        SessionMatcherInput(
            {"sorting_id": session, "curation_id": 0},
            bundle_dir=Path("/unused"),
            **metadata,
        )
        for session in ("a", "b")
    ]
    with pytest.raises(ValueError, match=message):
        backend.UnitMatchBackend().match(inputs, {})


def _two_one_unit_sessions():
    from spyglass.spikesorting.v2.matcher_protocol import SessionMatcherInput

    return [
        SessionMatcherInput(
            curation_key={"sorting_id": "A", "curation_id": 0},
            bundle_dir=Path("/x"),
            geometry_path=Path("/x/cp.npy"),
        ),
        SessionMatcherInput(
            curation_key={"sorting_id": "B", "curation_id": 1},
            bundle_dir=Path("/y"),
            geometry_path=Path("/y/cp.npy"),
        ),
    ]


def test_one_directional_match_is_rejected():
    """A pair above threshold in only one CV direction is not emitted."""
    from spyglass.spikesorting.v2._matching.unitmatch_backend import (
        UnitMatchBackend,
    )

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
    from spyglass.spikesorting.v2._matching.unitmatch_backend import (
        UnitMatchBackend,
    )

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


@pytest.mark.parametrize("column_ids", [False, True])
@pytest.mark.parametrize("empty_middle", [False, True])
def test_pairs_from_matrix_preserves_sparse_ids_and_session_boundaries(
    column_ids, empty_middle
):
    """Uneven inputs, including an empty input, keep their keys and unit ids."""
    from spyglass.spikesorting.v2._matching.unitmatch_backend import (
        UnitMatchBackend,
    )
    from spyglass.spikesorting.v2.matcher_protocol import SessionMatcherInput

    inputs = [
        SessionMatcherInput(
            curation_key={"sorting_id": name, "curation_id": index + 4},
            bundle_dir=Path("/unused"),
            geometry_path=Path("/unused/positions.npy"),
        )
        for index, name in enumerate(("A", "B", "C"))
    ]
    ids = np.array([7, 3, 12, 7, 3] + ([] if empty_middle else [45]))
    boundaries = [0, 2, 2, 5] if empty_middle else [0, 2, 5, 6]
    probability = np.zeros((len(ids), len(ids)))
    # Same-input candidates and a candidate exactly on the strict threshold
    # must be ignored. The surviving directions have different probabilities.
    probability[0, 1] = probability[1, 0] = 0.99
    probability[0, 2], probability[2, 0] = 0.5, 0.99
    probability[1, 2], probability[2, 1] = 0.9, 0.2
    probability[0, 3], probability[3, 0] = 0.9, 0.8
    probability[1, 4], probability[4, 1] = 0.75, 0.625
    later, later_curation = ("C", 6) if empty_middle else ("B", 5)
    expected = {
        ("A", 4, 7, later, later_curation, 7): 0.85,
        ("A", 4, 3, later, later_curation, 3): 0.6875,
    }
    if not empty_middle:
        probability[2, 5], probability[5, 2] = 0.875, 0.625
        expected[("B", 5, 12, "C", 6, 45)] = 0.75

    pairs = UnitMatchBackend._pairs_from_matrix(
        probability,
        boundaries,
        ids[:, None] if column_ids else ids,
        inputs,
        0.5,
    )
    actual = {
        (
            pair.session_a_sorting_id,
            pair.session_a_curation_id,
            pair.unit_a_id,
            pair.session_b_sorting_id,
            pair.session_b_curation_id,
            pair.unit_b_id,
        ): pair.match_probability
        for pair in pairs
    }
    assert len(pairs) == len(actual) == len(expected)
    assert actual == pytest.approx(expected)


def test_match_raises_if_unitmatch_drops_a_session(tmp_path, monkeypatch):
    """A dropped bundle (fewer loaded sessions than inputs) fails loudly.

    UnitMatchPy's load_good_waveforms excludes a session whose bundle fails to
    load instead of raising; without a guard, the compact session indexes would
    misattribute one session's units to another session's Spyglass key.
    """
    from types import SimpleNamespace

    from spyglass.spikesorting.v2._matching import unitmatch_backend as backend
    from spyglass.spikesorting.v2.matcher_protocol import SessionMatcherInput

    positions = tmp_path / "cp.npy"
    np.save(positions, np.zeros((4, 2)))
    inputs = [
        SessionMatcherInput(
            curation_key={"sorting_id": s, "curation_id": 0},
            bundle_dir=tmp_path,
            geometry_path=positions,
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

    from spyglass.spikesorting.v2._matching import unitmatch_backend as backend
    from spyglass.spikesorting.v2.matcher_protocol import SessionMatcherInput

    positions = tmp_path / "cp.npy"
    np.save(positions, np.zeros((4, 2)))
    inputs = [
        SessionMatcherInput(
            curation_key={"sorting_id": s, "curation_id": 0},
            bundle_dir=tmp_path,
            geometry_path=positions,
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

    from spyglass.spikesorting.v2._matching import unitmatch_backend as backend
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
            bundle_dir=tmp_path,
            geometry_path=cp_a,
        ),
        SessionMatcherInput(
            curation_key={"sorting_id": "B", "curation_id": 0},
            bundle_dir=tmp_path,
            geometry_path=cp_b,
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

    from spyglass.spikesorting.v2._matching import unitmatch_backend as backend

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

    from spyglass.spikesorting.v2._matching import unitmatch_backend as backend

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

    from spyglass.spikesorting.v2._matching import unitmatch_backend as backend

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
    from spyglass.spikesorting.v2._matching import (
        waveforms as _waveform_bundles,
    )

    real_save = _waveform_bundles.save_waveform_arrays

    def save_arrays(waveforms, directory, unit_ids):
        real_save(waveforms, directory, unit_ids)
        saved[Path(directory)] = {
            "waveforms": np.stack(
                [
                    np.load(
                        Path(directory)
                        / "RawWaveforms"
                        / f"Unit{uid}_RawSpikes.npy"
                    )
                    for uid in unit_ids
                ]
            ),
            "unit_ids": [int(uid) for uid in unit_ids],
        }

    monkeypatch.setattr(_waveform_bundles, "save_waveform_arrays", save_arrays)
    return saved


def _good_unit_ids(session_dir) -> list[int]:
    """Unit ids labelled ``good`` in a bundle's ``cluster_group.tsv``."""
    rows = np.loadtxt(
        Path(session_dir) / "cluster_group.tsv", dtype=str, delimiter="\t"
    )
    assert tuple(rows[0]) == ("cluster_id", "group")
    assert set(rows[1:, 1]) == {"good"}
    return [int(u) for u in rows[1:, 0]]


@pytest.mark.parametrize("fail_after_extraction", [False, True])
def test_bundle_waveforms_use_mmap_and_scratch_is_reclaimed(
    tmp_path, monkeypatch, saved_bundles, fail_after_extraction
):
    """Real extraction uses disk-backed waveforms and cleans failed attempts."""
    import spikeinterface as si

    from spyglass import settings
    from spyglass.spikesorting.v2._matching import unitmatch_backend as backend

    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    monkeypatch.setattr(settings, "temp_dir", str(scratch_root))
    recording, sorting = _planted_session(
        2.0,
        {7: (_train(0.1, 2.0), _planted_template(0), None)},
        noise_std=0,
    )
    real_create = si.create_sorting_analyzer
    folders = []

    def create(*args, **kwargs):
        analyzer = real_create(*args, **kwargs)
        folders.append(Path(analyzer.folder))
        real_compute = analyzer.compute

        def compute(name, **compute_kwargs):
            result = real_compute(name, **compute_kwargs)
            if name == "waveforms":
                assert isinstance(
                    analyzer.get_extension(name).data["waveforms"], np.memmap
                )
                if fail_after_extraction:
                    raise RuntimeError("interrupted waveform extraction")
            return result

        monkeypatch.setattr(analyzer, "compute", compute)
        return analyzer

    monkeypatch.setattr(si, "create_sorting_analyzer", create)
    destination = tmp_path / "bundle"
    if fail_after_extraction:
        with pytest.raises(RuntimeError, match="interrupted waveform"):
            backend.extract_unitmatch_bundle(destination, recording, sorting)
        assert destination not in saved_bundles
        assert not destination.exists()
    else:
        assert (
            backend.extract_unitmatch_bundle(destination, recording, sorting)
            == []
        )
        halves = saved_bundles[destination]["waveforms"][0]
        np.testing.assert_allclose(
            halves[:, :, 0], _planted_template(0), atol=1e-6
        )
        np.testing.assert_allclose(
            halves[:, :, 1], _planted_template(0), atol=1e-6
        )
    assert folders and all(not folder.exists() for folder in folders)
    assert not list(scratch_root.iterdir())


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
    from spyglass.spikesorting.v2._matching import unitmatch_backend as backend

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
    from spyglass.spikesorting.v2._matching import unitmatch_backend as backend

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
    from spyglass.spikesorting.v2._matching import unitmatch_backend as backend

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
    from spyglass.spikesorting.v2._matching import unitmatch_backend as backend

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
    from spyglass.spikesorting.v2._matching import unitmatch_backend as backend

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

    The recording-half (``time_half``) construction is the local copy kept by
    the half-split experiment script; both constructions run on that script's
    control session A (60 s, 16 channels, 20 units that fire throughout). The
    spike cap is at least every unit's spike count, so neither construction
    subsamples: the per-unit one averages all of a unit's interior spikes,
    the recording-half one all of its spikes in each recording half. For a stationary unit the
    first/second half of its spikes and the recording halves then hold nearly
    the same spikes, so the correlation measures the construction rather than
    random-subset noise.
    """
    from tests.spikesorting.v2.scripts import (
        unitmatch_half_split_experiment as experiment,
    )

    from spyglass.spikesorting.v2._matching import unitmatch_backend as backend

    recording, sorting = experiment.make_dataset(0)
    (rec_a, sort_a), _ = experiment.split_sessions(recording, sorting)
    per_unit_dir, time_half_dir = tmp_path / "per_unit", tmp_path / "time_half"
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
        per_unit_dir, rec_a, sort_a, **kwargs
    )
    experiment.extract_time_half_bundle(time_half_dir, rec_a, sort_a, **kwargs)

    assert excluded == []
    per_unit, time_half = (
        saved_bundles[per_unit_dir],
        saved_bundles[time_half_dir],
    )
    assert per_unit["unit_ids"] == time_half["unit_ids"]
    assert per_unit["waveforms"].shape == time_half["waveforms"].shape
    n_units = len(per_unit["unit_ids"])
    corr = np.array(
        [
            [
                np.corrcoef(
                    per_unit["waveforms"][i, ..., k].ravel(),
                    time_half["waveforms"][i, ..., k].ravel(),
                )[0, 1]
                for k in (0, 1)
            ]
            for i in range(n_units)
        ]
    )
    assert np.all(corr > 0.99), dict(zip(per_unit["unit_ids"], corr.round(4)))


# ---- waveform windows inside one statistics span ---------------------------


@pytest.mark.parametrize(
    ("spans", "nbefore", "nafter", "samples", "expected"),
    [
        # A join at frame 100: windows [s - 10, s + 10) must not cross it.
        # 10 / 90 / 110 / 190 touch a span edge from inside; 9 / 91 / 109 /
        # 191 run one sample past it.
        (
            [(0, 100), (100, 200)],
            10,
            10,
            [9, 10, 90, 91, 100, 109, 110, 190, 191],
            [False, True, True, False, False, False, True, True, False],
        ),
        # An acquisition gap [100, 150): a window inside the gap or straddling
        # either of its edges is not kept.
        (
            [(0, 100), (150, 250)],
            10,
            10,
            [95, 120, 145, 155, 160, 240, 241],
            [False, False, False, False, True, True, False],
        ),
        # An artifact exclusion [300, 320) between two spans of one recording.
        (
            [(200, 300), (320, 400)],
            10,
            10,
            [289, 290, 295, 310, 325, 330, 331],
            [True, True, False, False, False, True, True],
        ),
        # An asymmetric window: 5 samples before, 15 after.
        (
            [(0, 100)],
            5,
            15,
            [4, 5, 85, 86],
            [False, True, True, False],
        ),
        # Unsorted spikes, and spikes before the first / after the last span.
        (
            [(50, 100)],
            10,
            10,
            [200, 60, -5, 75, 45],
            [False, True, False, True, False],
        ),
    ],
)
def test_window_in_one_span_known_answers(
    spans, nbefore, nafter, samples, expected
):
    """Hand-computed answers at span edges, joins, gaps and exclusions."""
    from spyglass.spikesorting.v2._core.signal_math import (
        frames_with_window_in_one_span,
    )

    kept = frames_with_window_in_one_span(
        samples, spans, n_before=nbefore, n_after=nafter
    )
    assert kept.dtype == bool
    assert kept.tolist() == expected


def test_window_in_one_span_rejects_malformed_spans():
    """Unsorted, overlapping or empty spans, or a negative window, raise."""
    from spyglass.spikesorting.v2._core.signal_math import (
        frames_with_window_in_one_span,
    )

    for spans in ([(100, 200), (0, 100)], [(0, 100), (90, 200)], [(5, 5)]):
        with pytest.raises(ValueError, match="spans must be sorted"):
            frames_with_window_in_one_span([50], spans, n_before=1, n_after=1)
    with pytest.raises(ValueError, match="must not be negative"):
        frames_with_window_in_one_span([50], [(0, 100)], n_before=-1, n_after=1)
    # No span holds anything.
    assert frames_with_window_in_one_span(
        [50], [], n_before=1, n_after=1
    ).tolist() == [False]


def test_window_with_no_samples_after_needs_the_frame_in_its_span():
    """With ``n_after=0`` the window excludes the frame itself; a frame on
    the edge between adjacent spans is dropped, because the frame and its
    window would sit in different spans."""
    from spyglass.spikesorting.v2._core.signal_math import (
        frames_with_window_in_one_span,
    )

    spans = [(50, 59), (59, 371)]
    kept = frames_with_window_in_one_span(
        [58, 59, 64], spans, n_before=5, n_after=0
    )
    assert kept.tolist() == [True, False, True]


@pytest.fixture
def sampled_frames(monkeypatch):
    """Record the frames SpikeInterface's ``random_spikes`` draws.

    Wraps ``random_spikes_selection`` where the ``random_spikes`` extension
    calls it and appends a structured array (``sample_index``,
    ``unit_index``, ``segment_index``) of the drawn spikes per call. The
    number of spikes the analyzer's sorting offered each call is appended to
    the list's ``offered`` attribute.
    """
    from spikeinterface.core import analyzer_extension_core

    real = analyzer_extension_core.random_spikes_selection

    class _Drawn(list):
        offered: list

    drawn = _Drawn()
    drawn.offered = []

    def _record(sorting, *args, **kwargs):
        indices = real(sorting, *args, **kwargs)
        spikes = sorting.to_spike_vector()
        drawn.append(spikes[indices])
        drawn.offered.append(spikes.size)
        return indices

    monkeypatch.setattr(
        analyzer_extension_core, "random_spikes_selection", _record
    )
    return drawn


def _halves_cut_from_traces(traces, drawn, unit_index):
    """Two halves of one unit, cut by hand from ``traces`` at ``drawn``.

    The drawn frames of the unit are put in time order; half 0 is the mean
    window of the first ``n // 2`` and half 1 of the rest. Returns
    ``(half_0, half_1)``, each ``(spike_width, n_channels)``.
    """
    frames = np.sort(drawn["sample_index"][drawn["unit_index"] == unit_index])
    windows = np.stack(
        [traces[s - _HALF_WIDTH : s + _HALF_WIDTH] for s in frames]
    )
    n_half = len(frames) // 2
    return windows[:n_half].mean(axis=0), windows[n_half:].mean(axis=0)


def test_full_recording_span_matches_the_unfiltered_bundle(
    tmp_path, saved_bundles, sampled_frames
):
    """One span over the whole recording gives the bundle the segment-border
    margin alone gives, byte for byte.

    Each unit has spikes inside the waveform half-width of both recording
    edges (removed by the span filter and by SpikeInterface's own margin)
    and more spikes than the per-unit draw, so the random draw is exercised
    and a changed candidate set would change which spikes are drawn.
    """
    from spyglass.spikesorting.v2._matching import unitmatch_backend as backend

    duration_s = 12.0
    n_samples = int(duration_s * _FS)
    # Windows past an edge: removed by the span filter and by the margin.
    edge_frames = [3, _HALF_WIDTH - 1, n_samples - 2]
    # Windows touching an edge from inside: kept by the span filter; the
    # margin keeps the first and drops the last (it keeps frames below
    # ``n_samples - margin``).
    touching_frames = [_HALF_WIDTH, n_samples - _HALF_WIDTH]
    units = {}
    for uid, (channel, offset) in {
        4: (0, 0.1),
        9: (2, 0.13),
        2: (3, 0.2),
    }.items():
        samples = np.sort(
            np.concatenate(
                [
                    edge_frames,
                    touching_frames,
                    _train(offset, duration_s - 0.1, period_s=0.1),
                ]
            )
        )
        units[uid] = (samples, _planted_template(channel), None)
    recording, sorting = _planted_session(duration_s, units)
    kwargs = dict(max_spikes_per_unit=20, seed=3)

    unfiltered_dir, spans_dir = tmp_path / "unfiltered", tmp_path / "spans"
    assert (
        backend.extract_unitmatch_bundle(
            unfiltered_dir, recording, sorting, **kwargs
        )
        == []
    )
    assert (
        backend.extract_unitmatch_bundle(
            spans_dir,
            recording,
            sorting,
            statistics_spans=[(0, n_samples)],
            **kwargs,
        )
        == []
    )

    # The span filter removed the edge spikes before sampling, and nothing
    # else; the same spikes were then drawn.
    unfiltered_drawn, spans_drawn = sampled_frames
    n_units = len(units)
    assert sampled_frames.offered == [
        sorting.to_spike_vector().size,
        sorting.to_spike_vector().size - len(edge_frames) * n_units,
    ]
    assert len(spans_drawn) == len(unfiltered_drawn) > 0
    assert spans_drawn.tolist() == unfiltered_drawn.tolist()
    assert len(unfiltered_drawn) == 2 * kwargs["max_spikes_per_unit"] * n_units

    unfiltered, spans = saved_bundles[unfiltered_dir], saved_bundles[spans_dir]
    assert spans["unit_ids"] == unfiltered["unit_ids"] == [4, 9, 2]
    assert spans["waveforms"].dtype == unfiltered["waveforms"].dtype
    np.testing.assert_array_equal(spans["waveforms"], unfiltered["waveforms"])
    for name in ("channel_positions.npy", "cluster_group.tsv"):
        assert (spans_dir / name).read_bytes() == (
            unfiltered_dir / name
        ).read_bytes()


def test_spans_keep_windows_off_a_member_join(
    tmp_path, saved_bundles, sampled_frames
):
    """On two members concatenated into one segment, no drawn window runs
    across the join, and each half is the mean of the windows cut from the
    traces at the drawn frames.

    Unit 5 fires every 2 ms from 30 ms before to 30 ms after the join, so
    most of its spikes have a window across the join; without spans such
    windows are drawn. Unit 8 fires only in the first member, unit 6 in both.
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2._matching import unitmatch_backend as backend

    member_s = 4.0
    n_member = int(member_s * _FS)
    join = n_member
    dense = np.arange(join - 300, join + 300, 20)
    units_by_member = [
        {
            5: (dense[dense < join], _planted_template(0), None),
            8: (_train(0.05, member_s - 0.05, 0.2), _planted_template(3), None),
            6: (_train(0.1, member_s - 0.1, 0.3), _planted_template(2), None),
        },
        {
            5: (dense[dense >= join] - join, _planted_template(0), None),
            8: (np.array([], dtype=int), _planted_template(3), None),
            6: (_train(0.1, member_s - 0.1, 0.3), _planted_template(2), None),
        },
    ]
    members = [
        _planted_session(member_s, units, seed=seed)
        for seed, units in enumerate(units_by_member)
    ]
    recording = si.concatenate_recordings([rec for rec, _ in members])
    recording = recording.set_probe(members[0][0].get_probe())
    assert recording.get_num_segments() == 1
    sorting = si.NumpySorting.from_unit_dict(
        {
            uid: np.concatenate(
                [
                    np.asarray(units_by_member[0][uid][0]),
                    np.asarray(units_by_member[1][uid][0]) + join,
                ]
            ).astype(np.int64)
            for uid in (5, 8, 6)
        },
        sampling_frequency=_FS,
    )
    spans = [(0, join), (join, 2 * n_member)]
    traces = recording.get_traces(return_in_uV=True)

    session_dir = tmp_path / "joined"
    excluded = backend.extract_unitmatch_bundle(
        session_dir, recording, sorting, statistics_spans=spans, seed=0
    )
    (drawn,) = sampled_frames

    assert excluded == []
    saved = saved_bundles[session_dir]
    assert saved["unit_ids"] == [5, 8, 6] == _good_unit_ids(session_dir)
    for frame in drawn["sample_index"]:
        lo, hi = frame - _HALF_WIDTH, frame + _HALF_WIDTH
        assert any(a <= lo and hi <= b for a, b in spans), frame
    # Unit 5 keeps only the spikes at least a half-width from the join.
    unit_5_frames = drawn["sample_index"][drawn["unit_index"] == 0]
    assert set(unit_5_frames) == {
        s for s in dense if s + _HALF_WIDTH <= join or s - _HALF_WIDTH >= join
    }
    # Unit 8 fires only in member 0, and all its drawn frames are there.
    assert np.all(drawn["sample_index"][drawn["unit_index"] == 1] < join)
    for row, uid in enumerate(saved["unit_ids"]):
        half_0, half_1 = _halves_cut_from_traces(traces, drawn, row)
        assert np.any(half_0 != 0) and np.any(half_1 != 0), uid
        np.testing.assert_allclose(
            saved["waveforms"][row, ..., 0], half_0, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            saved["waveforms"][row, ..., 1], half_1, rtol=1e-6, atol=1e-6
        )

    # Without spans the segment has no join, so windows across it are drawn.
    backend.extract_unitmatch_bundle(
        tmp_path / "no_spans", recording, sorting, seed=0
    )
    unfiltered = sampled_frames[1]["sample_index"]
    assert np.any(
        (unfiltered - _HALF_WIDTH < join) & (unfiltered + _HALF_WIDTH > join)
    )


def test_spans_exclude_units_without_two_supported_spikes(
    tmp_path, saved_bundles
):
    """A unit with fewer than two spikes whose window fits one span is left
    out and returned; when that leaves no unit, nothing is written."""
    from spyglass.spikesorting.v2._matching import unitmatch_backend as backend

    duration_s = 4.0
    n_samples = int(duration_s * _FS)
    exclusion = (20_000, 20_400)
    spans = [(0, exclusion[0]), (exclusion[1], n_samples)]
    templates = {3: _planted_template(0), 7: _planted_template(2)}
    # Unit 7: one spike clear of the exclusion, the others straddle an edge.
    unit_7 = np.array([10_000, exclusion[0] - 5, exclusion[1] + 5])
    recording, sorting = _planted_session(
        duration_s,
        {
            3: (_train(0.1, duration_s - 0.1), templates[3], None),
            7: (unit_7, templates[7], None),
        },
    )
    session_dir = tmp_path / "sess"
    excluded = backend.extract_unitmatch_bundle(
        session_dir, recording, sorting, statistics_spans=spans, seed=0
    )
    assert excluded == [7]
    assert saved_bundles[session_dir]["unit_ids"] == [3]

    only_7 = sorting.select_units([7])
    lone_dir = tmp_path / "lone"
    with pytest.raises(backend.NoMatchableUnitsError, match="statistics span"):
        backend.extract_unitmatch_bundle(
            lone_dir, recording, only_7, statistics_spans=spans, seed=0
        )
    assert not lone_dir.exists()
    # Without spans, unit 7 has three full-support spikes and is kept.
    assert (
        backend.extract_unitmatch_bundle(
            tmp_path / "no_spans", recording, only_7, seed=0
        )
        == []
    )


def test_spans_must_describe_the_recording(tmp_path, saved_bundles):
    """Spans past the recording's end, or a multi-segment recording, raise
    before any analyzer is built."""
    import spikeinterface as si

    from spyglass.spikesorting.v2._matching import unitmatch_backend as backend

    recording, sorting = _planted_session(
        2.0, {3: (_train(0.1, 1.9), _planted_template(0), None)}
    )
    n_samples = recording.get_num_samples()
    with pytest.raises(ValueError, match="past the recording"):
        backend.extract_unitmatch_bundle(
            tmp_path / "a",
            recording,
            sorting,
            statistics_spans=[(0, n_samples + 1)],
        )
    two_segments = si.append_recordings([recording, recording])
    two_segment_sorting = si.NumpySorting.from_unit_dict(
        [{3: _train(0.1, 1.9)}, {3: _train(0.1, 1.9)}], sampling_frequency=_FS
    )
    with pytest.raises(ValueError, match="single-segment"):
        backend.extract_unitmatch_bundle(
            tmp_path / "b",
            two_segments,
            two_segment_sorting,
            statistics_spans=[(0, n_samples)],
        )
    assert saved_bundles == {}


def test_get_matcher_bootstraps_default_after_clear():
    """get_matcher re-registers the built-in backend even if the registry was cleared."""
    from spyglass.spikesorting.v2 import matcher_protocol as mp

    saved_m, saved_s = dict(mp._MATCHER_REGISTRY), dict(mp._SCHEMA_REGISTRY)
    saved_preparers = dict(mp._PREPARER_REGISTRY)
    try:
        mp._MATCHER_REGISTRY.clear()
        mp._SCHEMA_REGISTRY.clear()
        assert mp.get_matcher("unitmatch").name == "unitmatch"
    finally:
        mp._MATCHER_REGISTRY.clear()
        mp._MATCHER_REGISTRY.update(saved_m)
        mp._SCHEMA_REGISTRY.clear()
        mp._SCHEMA_REGISTRY.update(saved_s)
        mp._PREPARER_REGISTRY.clear()
        mp._PREPARER_REGISTRY.update(saved_preparers)


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

    from spyglass.spikesorting.v2._matching.unitmatch_backend import (
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
                bundle_dir=sdir,
                geometry_path=sdir / "channel_positions.npy",
            )
        )
    return inputs, s1_ids, s2_ids, shared


@pytest.mark.slow
@pytest.mark.integration
def test_match_recovers_planted_correspondences(two_session_inputs):
    from spyglass.spikesorting.v2._matching.unitmatch_backend import (
        UnitMatchBackend,
    )

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


def _gate_test_record(
    seed,
    scenario,
    unit_ids,
    drift_out_units,
    matched,
    true_pair_probs=None,
    condition="per_unit",
):
    """Build one hand-built ``evaluate_gates`` input record.

    ``matched`` is the set of ``(i, j)`` pairs that pass matching; every other
    ``(i, j)`` pair over ``unit_ids`` counts as a miss. ``true_pair_probs`` is
    an optional ``{unit_id: (p_ab, p_ba)}`` map, stored the way
    :func:`run_one` stores it (string keys, list values), for the healthy
    true-pair probability drop gate; omitted units are simply not paired.
    ``condition`` defaults to ``"per_unit"`` (the gated condition); pass
    ``"time_half"`` to build the baseline side of an excess-drop
    comparison. No UnitMatchPy or DB access
    happens here -- ``score_pairs`` is pure counting.
    """
    from tests.spikesorting.v2.scripts.unitmatch_half_split_experiment import (
        score_pairs,
    )

    scored_s = [] if scenario == "control" else list(drift_out_units)
    passing_pairs = [(i, j, 1.0) for i, j in matched]
    return {
        "seed": seed,
        "scenario": scenario,
        "condition": condition,
        "unit_ids": list(unit_ids),
        "drift_out_units": scored_s,
        "seed_drift_out_units": list(drift_out_units),
        "counts": score_pairs(passing_pairs, unit_ids, scored_s),
        "fitted": None,
        "passing_pairs": passing_pairs,
        "true_pair_probs": {
            str(u): list(p) for u, p in (true_pair_probs or {}).items()
        },
    }


def _gate(gates, name, scenario):
    return next(g for g in gates if g.name == name and g.scenario == scenario)


def test_evaluate_gates_drift_out_recall_is_exact_and_sxs_rate_is_diagnostic():
    """Drift-out recall (>=) passes exactly on the threshold, fails one
    count under it.

    The S x S false-pair rate is a printed diagnostic, not an acceptance
    gate -- UnitMatch's per-run match threshold is unstable when a session
    has few units, so a burst of false pairs is not attributable to the
    cross-validation-half construction under test. Its exact-fraction value
    is still checked at its reported limit.

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

    # Recall 20/25 == 0.80 exactly; S x S rate 9/600 == 0.015 exactly.
    at_boundary = _gate_test_record(
        0, "driftout_A", s_units, s_units, matched(20, 9)
    )
    gates = evaluate_gates([at_boundary], "per_unit")
    recall, sxs_rate = (
        _gate(gates, "drift-out recall", "driftout_A"),
        _gate(
            gates,
            "S x S false-pair rate (diagnostic, not gated)",
            "driftout_A",
        ),
    )
    assert recall.passed is True, recall
    assert sxs_rate.passed is None, sxs_rate
    assert sxs_rate.value == pytest.approx(0.015), sxs_rate

    # One fewer drift-out true pair (19/25 == 0.76) moves the recall past
    # its boundary; one more S x S false pair (10/600 == 0.01667) moves the
    # diagnostic S x S rate past its reported limit, but it still has no
    # pass/fail verdict.
    past_boundary = _gate_test_record(
        0, "driftout_A", s_units, s_units, matched(19, 10)
    )
    gates = evaluate_gates([past_boundary], "per_unit")
    recall, sxs_rate = (
        _gate(gates, "drift-out recall", "driftout_A"),
        _gate(
            gates,
            "S x S false-pair rate (diagnostic, not gated)",
            "driftout_A",
        ),
    )
    assert recall.passed is False, recall
    assert sxs_rate.passed is None, sxs_rate
    assert sxs_rate.value == pytest.approx(10 / 600), sxs_rate


def test_evaluate_gates_healthy_fp_increase_is_exact_at_its_boundary():
    """The healthy false-pair rate increase (paired against control) passes
    exactly on the threshold and fails one false pair over it.

    25 non-S units make the paired false-positive denominator 600, an exact
    multiple of 0.005's reduced denominator (200): 3/600 == 0.005 lands on an
    exact fraction.
    """
    from tests.spikesorting.v2.scripts.unitmatch_half_split_experiment import (
        evaluate_gates,
    )

    s_units = [999]  # a single drift-out unit; irrelevant to this gate
    non_s = list(range(1000, 1025))
    unit_ids = s_units + non_s
    off_diag = [(a, b) for a in non_s for b in non_s if a != b]

    def healthy(n_true, n_false):
        return {(u, u) for u in non_s[:n_true]} | set(off_diag[:n_false])

    control = _gate_test_record(1, "control", unit_ids, s_units, healthy(25, 0))

    # control 0/600 -> scenario 3/600: increase == 3/600 == 0.005 exactly.
    passing = _gate_test_record(
        1, "driftout_A", unit_ids, s_units, healthy(24, 3)
    )
    gates = evaluate_gates([control, passing], "per_unit")
    fp_increase = _gate(gates, "healthy false-pair rate increase", "driftout_A")
    assert fp_increase.passed is True, fp_increase

    # One more false positive (4/600) pushes the increase to 4/600 == 0.00667.
    failing_fp = _gate_test_record(
        1, "driftout_A", unit_ids, s_units, healthy(24, 4)
    )
    gates = evaluate_gates([control, failing_fp], "per_unit")
    assert (
        _gate(gates, "healthy false-pair rate increase", "driftout_A").passed
        is False
    ), gates


def test_evaluate_gates_healthy_recall_drop_is_a_diagnostic_not_a_gate():
    """The count-based healthy recall drop is printed but never gates
    acceptance.

    Same fixture as the healthy false-pair boundary test, with a recall drop
    above the reported 0.04 limit (2/25 == 0.08): the diagnostic reports
    that value but ``passed`` is always ``None``.
    """
    from tests.spikesorting.v2.scripts.unitmatch_half_split_experiment import (
        evaluate_gates,
    )

    s_units = [999]
    non_s = list(range(1000, 1025))
    unit_ids = s_units + non_s

    control = _gate_test_record(
        1, "control", unit_ids, s_units, {(u, u) for u in non_s}
    )
    scenario = _gate_test_record(
        1, "driftout_A", unit_ids, s_units, {(u, u) for u in non_s[:23]}
    )
    gates = evaluate_gates([control, scenario], "per_unit")
    recall_drop = _gate(
        gates,
        "healthy recall drop (diagnostic, not gated)",
        "driftout_A",
    )
    assert recall_drop.passed is None, recall_drop
    assert recall_drop.value == pytest.approx(2 / 25)


def _healthy_prob_drop_gate(records):
    from tests.spikesorting.v2.scripts.unitmatch_half_split_experiment import (
        evaluate_gates,
    )

    gates = evaluate_gates(records, "per_unit")
    return _gate(
        gates,
        "healthy true-pair probability drop",
        "driftout_A",
    )


def test_evaluate_gates_healthy_prob_drop_is_exact_at_its_boundary():
    """The healthy true-pair probability drop passes on dyadic drops at or
    below 0.04 and fails clearly above it.

    Every ``q`` here is a dyadic fraction (a binary fraction like ``1/32``),
    so every value AND every subtraction is bit-exact in IEEE 754 double
    precision -- no case here depends on 1-ulp rounding luck the way
    ``0.05 - 0.01`` or ``0.07 - 0.03`` would (those are not exactly
    representable in binary and only pass/fail by which way they happen to
    round). One paired non-S unit per case, control ``q = 0.5 = 1/2``:

    - scenario ``q = 0.46875 = 15/32`` -> drop ``0.03125 = 1/32``: PASS,
      comfortably under the threshold.
    - scenario ``q = 0.4609375 = 59/128`` -> drop ``0.0390625 = 5/128``:
      PASS, close to the threshold but still a clean dyadic value below it.
    - scenario ``q = 0.453125 = 29/64`` -> drop ``0.046875 = 3/64``: FAIL,
      clearly (not by a hair) above the threshold.
    """
    s_units = [99]
    unit_ids = s_units + [1]

    def gate_for(scenario_q):
        control = _gate_test_record(
            1, "control", unit_ids, s_units, set(), {1: (0.5, 0.5)}
        )
        scenario = _gate_test_record(
            1,
            "driftout_A",
            unit_ids,
            s_units,
            set(),
            {1: (scenario_q, scenario_q)},
        )
        return _healthy_prob_drop_gate([control, scenario])

    comfortably_passing = gate_for(0.46875)
    assert comfortably_passing.passed is True, comfortably_passing
    assert comfortably_passing.value == pytest.approx(0.03125, abs=0)

    near_boundary_passing = gate_for(0.4609375)
    assert near_boundary_passing.passed is True, near_boundary_passing
    assert near_boundary_passing.value == pytest.approx(0.0390625, abs=0)

    clearly_failing = gate_for(0.453125)
    assert clearly_failing.passed is False, clearly_failing
    assert clearly_failing.value == pytest.approx(0.046875, abs=0)


def test_evaluate_gates_healthy_prob_drop_uses_min_of_directed_probabilities():
    """The probability drop's ``q_u = min(p_ab, p_ba)`` -- not ``max`` or a
    mean -- is what decides the drop. Every other probability-drop fixture
    in this module uses
    ``p_ab == p_ba``, so a change from ``min`` to ``max`` or to a mean would
    pass every one of them unnoticed; this fixture uses asymmetric directed
    probabilities so the three choices disagree.

    Control ``(p_ab, p_ba) = (0.5, 0.75)`` -> ``q = min = 0.5``. Scenario
    ``(0.4375, 0.9375)`` -> ``q = min = 0.4375``. Every value here is dyadic
    (a binary fraction), so the drop is bit-exact:

    - ``min``: drop ``0.5 - 0.4375 = 0.0625`` -- clearly above the 0.04
      limit: FAIL.
    - ``max`` (wrong): drop ``0.75 - 0.9375 = -0.1875`` -- a rise, passes
      trivially.
    - mean of the two directions (wrong): drop
      ``0.625 - 0.6875 = -0.0625`` -- also passes trivially.

    Both wrong implementations pass where the correct one fails, so this
    case would catch a ``min`` -> ``max``/mean swap that every symmetric
    fixture here misses.
    """
    s_units = [99]
    unit_ids = s_units + [1]

    control = _gate_test_record(
        1, "control", unit_ids, s_units, set(), {1: (0.5, 0.75)}
    )
    scenario = _gate_test_record(
        1, "driftout_A", unit_ids, s_units, set(), {1: (0.4375, 0.9375)}
    )
    gate = _healthy_prob_drop_gate([control, scenario])
    assert gate.value == pytest.approx(0.0625, abs=0), gate
    assert gate.passed is False, gate


def test_evaluate_gates_healthy_prob_drop_pools_units_not_per_seed_means():
    """The pooled mean is ONE mean over every paired unit, not a mean of
    per-seed means -- the two differ whenever seeds contribute unequal
    paired counts.

    Seed 1 has a single paired non-S unit with ``q_control = 1.0``; seed 2
    has three paired non-S units all with ``q_control = 0.0``. A naive
    mean-of-per-seed-means would average 1.0 and 0.0 to 0.5; the correct
    pooled mean over all four units is ``1.0 / 4 = 0.25``.
    """
    seed1_units = [901, 1]  # 901 is S (irrelevant to the healthy drop)
    seed2_units = [902, 11, 12, 13]  # 902 is S

    seed1_control = _gate_test_record(
        1, "control", seed1_units, [901], set(), {1: (1.0, 1.0)}
    )
    seed1_scenario = _gate_test_record(
        1, "driftout_A", seed1_units, [901], set(), {1: (1.0, 1.0)}
    )
    seed2_control = _gate_test_record(
        2,
        "control",
        seed2_units,
        [902],
        set(),
        {u: (0.0, 0.0) for u in (11, 12, 13)},
    )
    seed2_scenario = _gate_test_record(
        2,
        "driftout_A",
        seed2_units,
        [902],
        set(),
        {u: (0.0, 0.0) for u in (11, 12, 13)},
    )

    gate = _healthy_prob_drop_gate(
        [seed1_control, seed1_scenario, seed2_control, seed2_scenario]
    )
    assert "n_paired=4" in gate.detail, gate
    # The naive (wrong) mean-of-per-seed-means would report a control mean of
    # (1.0 + 0.0) / 2 == 0.5; the correct pooled mean over all 4 units is
    # 1.0 / 4 == 0.25. ``scenario`` mirrors ``control`` here (drop == 0), so
    # this checks the mean itself, not the drop.
    assert "control mean 0.2500" in gate.detail, gate
    assert "scenario mean 0.2500" in gate.detail, gate
    assert gate.value == pytest.approx(0.0), gate


def test_paired_true_pair_probs_drops_unit_missing_on_either_side():
    """A unit missing a probability on EITHER side is dropped the same way
    and counted in ``n_unpaired``, not silently ignored.

    Unit 1 has a probability on both sides (paired); unit 2 is missing from
    the control side; unit 3 is missing from the scenario side. Both 2 and 3
    must be dropped identically (symmetric treatment) and both counted in
    ``n_unpaired`` -- regardless of which side is missing.
    """
    from tests.spikesorting.v2.scripts.unitmatch_half_split_experiment import (
        paired_true_pair_probs,
    )

    s_units = [999]
    unit_ids = s_units + [1, 2, 3]

    control = _gate_test_record(
        1, "control", unit_ids, s_units, set(), {1: (0.9, 0.9), 3: (0.8, 0.8)}
    )
    scenario = _gate_test_record(
        1,
        "driftout_A",
        unit_ids,
        s_units,
        set(),
        {1: (0.7, 0.7), 2: (0.6, 0.6)},
    )

    pairs, n_unpaired = paired_true_pair_probs(scenario, control)
    assert [p[0] for p in pairs] == [1]
    assert (
        n_unpaired == 2
    )  # unit 2 (missing control), unit 3 (missing scenario)


def test_evaluate_gates_healthy_prob_drop_reports_unpaired_and_capture_missing():
    """The probability-drop gate's detail reports n_paired, n_unpaired and
    n_capture_missing_runs, not just the paired count.

    Seed 1's non-S units are 1 (paired) and 2 (missing from the scenario
    side -> unpaired). Seed 2's scenario run has NO recorded probabilities
    at all despite having non-S units -- a stand-in for UnitMatch returning
    early or the capture wrap never firing -- so it must be flagged as a
    capture-missing run (on top of its unit also landing in n_unpaired).
    """
    seed1_units = [900, 1, 2]
    seed1_control = _gate_test_record(
        1,
        "control",
        seed1_units,
        [900],
        set(),
        {1: (0.9, 0.9), 2: (0.8, 0.8)},
    )
    seed1_scenario = _gate_test_record(
        1, "driftout_A", seed1_units, [900], set(), {1: (0.7, 0.7)}
    )
    seed2_units = [950, 21]
    seed2_control = _gate_test_record(
        2, "control", seed2_units, [950], set(), {21: (0.9, 0.9)}
    )
    seed2_scenario = _gate_test_record(
        2, "driftout_A", seed2_units, [950], set(), {}
    )

    gate = _healthy_prob_drop_gate(
        [seed1_control, seed1_scenario, seed2_control, seed2_scenario]
    )
    assert "n_paired=1" in gate.detail, gate
    assert "n_unpaired=2" in gate.detail, gate
    assert "n_capture_missing_runs=1" in gate.detail, gate


def _excess_prob_drop_gate(records):
    from tests.spikesorting.v2.scripts.unitmatch_half_split_experiment import (
        evaluate_gates,
    )

    gates = evaluate_gates(records, "per_unit")
    return _gate(
        gates,
        "excess healthy true-pair probability drop vs time_half",
        "driftout_A",
    )


def test_evaluate_gates_excess_prob_drop_pass_fail_and_not_evaluated():
    """The excess drop is the per_unit healthy true-pair probability drop
    minus the same drop measured on ``time_half`` (the baseline
    construction) for the same seed, and is
    not evaluated at all when no ``time_half`` run exists for that seed.

    - PASS: per_unit drop (``0.5 -> 0.5``, drop ``0.0``) is no worse than the
      time_half drop (``0.5 -> 0.5``, drop ``0.0``); excess
      ``0.0 - 0.0 = 0.0 <= 0.04``.
    - FAIL: per_unit drop (``0.5 -> 0.375``, drop ``0.125``) is worse than
      the time_half drop (``0.5 -> 0.5``, drop ``0.0``) by more than the 0.04
      limit; excess ``0.125 - 0.0 = 0.125 > 0.04``.
    - Not evaluated: only the ``per_unit`` records exist for the seed (no
      ``time_half`` run), so there is nothing to compare against -- unlike a
      silent pass on an empty comparison, ``passed`` is ``None`` and the
      detail says a time_half run is not available.
    """
    s_units = [99]
    unit_ids = s_units + [1]

    def per_unit_pair(scenario_q):
        control = _gate_test_record(
            1, "control", unit_ids, s_units, set(), {1: (0.5, 0.5)}
        )
        scenario = _gate_test_record(
            1,
            "driftout_A",
            unit_ids,
            s_units,
            set(),
            {1: (scenario_q, scenario_q)},
        )
        return control, scenario

    def time_half_pair(scenario_q):
        control = _gate_test_record(
            1,
            "control",
            unit_ids,
            s_units,
            set(),
            {1: (0.5, 0.5)},
            condition="time_half",
        )
        scenario = _gate_test_record(
            1,
            "driftout_A",
            unit_ids,
            s_units,
            set(),
            {1: (scenario_q, scenario_q)},
            condition="time_half",
        )
        return control, scenario

    baseline_control, baseline_scenario = time_half_pair(0.5)

    per_unit_control, per_unit_scenario = per_unit_pair(0.5)
    passing = _excess_prob_drop_gate(
        [
            per_unit_control,
            per_unit_scenario,
            baseline_control,
            baseline_scenario,
        ]
    )
    assert passing.passed is True, passing
    assert passing.value == pytest.approx(0.0, abs=0), passing

    per_unit_control, per_unit_scenario = per_unit_pair(0.375)
    failing = _excess_prob_drop_gate(
        [
            per_unit_control,
            per_unit_scenario,
            baseline_control,
            baseline_scenario,
        ]
    )
    assert failing.passed is False, failing
    assert failing.value == pytest.approx(0.125, abs=0), failing

    not_evaluated = _excess_prob_drop_gate(
        [per_unit_control, per_unit_scenario]
    )
    assert not_evaluated.passed is None, not_evaluated
    assert (
        "time_half paired probability run not available" in not_evaluated.detail
    ), not_evaluated


def _write_raw_waveform(session_dir, unit_id, half0, half1):
    """Write one unit's ``RawWaveforms/Unit{id}_RawSpikes.npy`` (shape
    ``(spike_width, n_channels, 2)``) so :func:`non_s_bit_identical_halves`
    has a file to compare."""
    (session_dir / "RawWaveforms").mkdir(parents=True, exist_ok=True)
    wave = np.stack([half0, half1], axis=-1)
    np.save(session_dir / "RawWaveforms" / f"Unit{unit_id}_RawSpikes.npy", wave)


def test_evaluate_gates_template_identity_fails_on_one_differing_half(tmp_path):
    """Healthy template bit-identity fails as soon as one non-S template
    half differs from control.

    Two non-S units (3 and 5) with identical A/B bundles between the control
    and scenario run except unit 5's session-B half 1, which differs by one
    value: 7/8 halves (2 units x 2 sessions x 2 halves) are bit-identical, so
    the gate must fail (identical != total). A companion bundle pair with
    every half identical passes.
    """
    from tests.spikesorting.v2.scripts.unitmatch_half_split_experiment import (
        evaluate_gates,
    )

    half_shape = (4, 2)  # (spike_width, n_channels)
    unit_ids = [3, 5]
    s_units = []

    def build(root, *, unit5_half1_delta=0.0):
        for label in ("A", "B"):
            d = root / label
            for uid in unit_ids:
                base0 = np.full(half_shape, float(uid))
                base1 = np.full(half_shape, float(uid) + 1.0)
                if uid == 5 and label == "B":
                    base1 = base1 + unit5_half1_delta
                _write_raw_waveform(d, uid, base0, base1)
        return {label: str(root / label) for label in ("A", "B")}

    control_dirs = build(tmp_path / "control")
    failing_dirs = build(tmp_path / "failing", unit5_half1_delta=1.0)
    passing_dirs = build(tmp_path / "passing")

    def record(scenario, bundle_dirs):
        r = _gate_test_record(1, scenario, unit_ids, s_units, set())
        r["bundle_dirs"] = bundle_dirs
        return r

    control = record("control", control_dirs)
    failing = record("driftout_A", failing_dirs)
    gates = evaluate_gates([control, failing], "per_unit")
    identity = _gate(gates, "healthy template bit-identity", "driftout_A")
    assert identity.passed is False, identity
    assert identity.detail == "7/8", identity

    passing = record("driftout_A", passing_dirs)
    gates = evaluate_gates([control, passing], "per_unit")
    identity = _gate(gates, "healthy template bit-identity", "driftout_A")
    assert identity.passed is True, identity
    assert identity.detail == "8/8", identity


def test_evaluate_gates_template_identity_counts_missing_unit_as_not_identical(
    tmp_path,
):
    """A non-S unit present in control but excluded from the scenario bundle
    (or the reverse) counts as NOT identical -- it must not be skipped.

    Unit 3 is present (and bit-identical) in both control and scenario, both
    sessions. Unit 5 is present in control but entirely missing from the
    scenario bundle (both sessions) -- as if it had fewer than two sampled
    spikes in that run only. Unit 5 contributes 4 non-identical comparisons
    (2 sessions x 2 halves) and 0 identical ones; unit 3 contributes 4
    identical comparisons: total 4/8, not 4/4 (which is what skipping unit 5
    would have reported).
    """
    from tests.spikesorting.v2.scripts.unitmatch_half_split_experiment import (
        evaluate_gates,
    )

    unit_ids = [3, 5]
    s_units = []

    for label in ("A", "B"):
        _write_raw_waveform(
            tmp_path / "control" / label,
            3,
            np.full((4, 2), 3.0),
            np.full((4, 2), 4.0),
        )
        _write_raw_waveform(
            tmp_path / "control" / label,
            5,
            np.full((4, 2), 5.0),
            np.full((4, 2), 6.0),
        )
        # Scenario bundle: unit 3 identical to control; unit 5 excluded
        # entirely (no RawWaveforms file at all).
        _write_raw_waveform(
            tmp_path / "scenario" / label,
            3,
            np.full((4, 2), 3.0),
            np.full((4, 2), 4.0),
        )

    control_dirs = {
        label: str(tmp_path / "control" / label) for label in ("A", "B")
    }
    scenario_dirs = {
        label: str(tmp_path / "scenario" / label) for label in ("A", "B")
    }

    def record(scenario, bundle_dirs):
        r = _gate_test_record(1, scenario, unit_ids, s_units, set())
        r["bundle_dirs"] = bundle_dirs
        return r

    control = record("control", control_dirs)
    scenario = record("driftout_A", scenario_dirs)
    gates = evaluate_gates([control, scenario], "per_unit")
    identity = _gate(gates, "healthy template bit-identity", "driftout_A")
    assert identity.passed is False, identity
    assert identity.detail == "4/8", identity


def test_true_pair_probability_lookup_sparse_reordered_ids():
    """The true-pair lookup resolves each unit by id, not by position.

    Session 0 (A) keeps units ``[7, 3]`` (not sorted); session 1 (B) keeps
    ``[12, 7, 3]`` -- unit 12 has no partner in A, and the shared units 7/3
    sit at different ranks in each session. Every cell of the probability
    matrix is a distinct value, so a wrong index would read the wrong cell.
    """
    from tests.spikesorting.v2.scripts.unitmatch_half_split_experiment import (
        true_pair_directed_probs,
    )

    original_ids = np.array([7, 3, 12, 7, 3])
    session_switch = np.array([0, 2, 5])  # session 0: idx 0-1; session 1: 2-4
    n = 5
    prob_matrix = np.arange(n * n, dtype=float).reshape(n, n)

    # unit 7: A index 0, B index 3 -> (prob[0, 3], prob[3, 0]).
    assert true_pair_directed_probs(
        prob_matrix, session_switch, original_ids, 7
    ) == (3.0, 15.0)
    # unit 3: A index 1, B index 4 -> (prob[1, 4], prob[4, 1]).
    assert true_pair_directed_probs(
        prob_matrix, session_switch, original_ids, 3
    ) == (9.0, 21.0)
    # unit 12 is only in B -> no true cross-session pair.
    assert (
        true_pair_directed_probs(prob_matrix, session_switch, original_ids, 12)
        is None
    )

    # A column-vector original_ids (UnitMatchPy's per-session good_units
    # shape) must resolve the same way, not broadcast into an (n, n) mask.
    column_ids = original_ids.reshape(-1, 1)
    assert true_pair_directed_probs(
        prob_matrix, session_switch, column_ids, 7
    ) == (3.0, 15.0)
