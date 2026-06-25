"""Content-fingerprint identity for the v2 preprocessed recording cache.

The fingerprint is the representation-blind scientific identity of a persisted
recording: it is read back entirely from the on-disk ElectricalSeries (traces,
timestamps, persisted geometry, scaling metadata) and must be deterministic for
a given file yet discriminate any scientifically-meaningful perturbation. These
tests are DB-free -- they synthesize tiny analysis-style NWBs and never touch
DataJoint.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.spikesorting.v2._ingest_helpers import (
    write_processed_recording_nwb,
)


def _baseline(out_path, **overrides):
    """Write a small deterministic processed-recording NWB; return its path +
    in-file ElectricalSeries path."""
    n_frames, n_channels = 50, 4
    rng = np.random.default_rng(0)
    params = dict(
        traces=rng.normal(0.0, 50.0, size=(n_frames, n_channels)).astype(
            "float32"
        ),
        timestamps=np.arange(n_frames, dtype=float) / 30_000.0,
        rel_positions=np.array(
            [[0.0, 0.0], [0.0, 20.0], [0.0, 40.0], [0.0, 60.0]]
        ),
        conversion=1e-6,
        offset=0.0,
        filtering="bandpass filter 300-6000 Hz",
        channel_ids=[0, 1, 2, 3],
    )
    params.update(overrides)
    return write_processed_recording_nwb(out_path, **params)


def _aggregate(path, es_path):
    from spyglass.spikesorting.v2._recompute import combined_hash
    from spyglass.spikesorting.v2._recording_fingerprint import (
        recording_content_fingerprint,
    )

    return combined_hash(
        recording_content_fingerprint(path, electrical_series_path=es_path)
    )


def test_recording_content_fingerprint_deterministic(tmp_path):
    """The same persisted file yields an identical component dict AND aggregate
    across repeated reads -- the fingerprint is a pure function of the bytes."""
    from spyglass.spikesorting.v2._recording_fingerprint import (
        recording_content_fingerprint,
    )

    path, es_path = _baseline(tmp_path / "rec.nwb")
    first = recording_content_fingerprint(path, electrical_series_path=es_path)
    second = recording_content_fingerprint(path, electrical_series_path=es_path)

    assert first == second
    # The component dict names each axis so a recompute diff can localize drift.
    assert set(first) == {"traces", "timestamps", "geometry", "metadata"}
    assert _aggregate(path, es_path) == _aggregate(path, es_path)


def test_recording_content_fingerprint_discriminates(tmp_path):
    """Every scientifically-meaningful perturbation changes the aggregate;
    sub-``TRACE_ROUNDING`` float noise does not."""
    from spyglass.spikesorting.v2._recording_fingerprint import TRACE_ROUNDING

    base_path, es = _baseline(tmp_path / "base.nwb")
    base = _aggregate(base_path, es)

    rng = np.random.default_rng(0)
    base_traces = rng.normal(0.0, 50.0, size=(50, 4)).astype("float32")
    base_ts = np.arange(50, dtype=float) / 30_000.0
    base_pos = np.array([[0.0, 0.0], [0.0, 20.0], [0.0, 40.0], [0.0, 60.0]])

    # Perturbed traces (well above the rounding floor).
    bumped = base_traces.copy()
    bumped[0, 0] += 10.0
    p, e = _baseline(tmp_path / "traces.nwb", traces=bumped)
    assert _aggregate(p, e) != base, "perturbed traces must change the hash"

    # Perturbed timestamps.
    p, e = _baseline(tmp_path / "ts.nwb", timestamps=base_ts + 1.0)
    assert _aggregate(p, e) != base, "perturbed timestamps must change the hash"

    # Perturbed gain (conversion).
    p, e = _baseline(tmp_path / "gain.nwb", conversion=2e-6)
    assert _aggregate(p, e) != base, "perturbed gain must change the hash"

    # Channel order (reverse channels + their geometry together).
    p, e = _baseline(
        tmp_path / "order.nwb",
        traces=base_traces[:, ::-1].copy(),
        rel_positions=base_pos[::-1].copy(),
        channel_ids=[3, 2, 1, 0],
    )
    assert _aggregate(p, e) != base, "channel reorder must change the hash"

    # Electrode geometry (move one contact).
    moved = base_pos.copy()
    moved[0, 1] += 5.0
    p, e = _baseline(tmp_path / "geom.nwb", rel_positions=moved)
    assert _aggregate(p, e) != base, "perturbed geometry must change the hash"

    # Sub-rounding noise is absorbed. Use integer-valued traces (exact in
    # float32, a clean 5e-5 from any 4-decimal rounding boundary) and noise
    # that is representable in float32 yet an order of magnitude below
    # TRACE_ROUNDING, so it perturbs the stored bytes but never flips a rounding
    # decision -- isolating that the ROUNDING (not float32 storage) absorbs it.
    clean = rng.integers(-100, 100, size=(50, 4)).astype("float32")
    noise = 2.0 * 10.0 ** (-(TRACE_ROUNDING + 1))  # 2e-5 µV, < 5e-5 boundary
    noisy = (clean.astype("float64") + noise).astype("float32")
    assert not np.array_equal(noisy, clean), (
        "precondition: the noise must perturb the stored float32 bytes"
    )
    p_clean, e_clean = _baseline(tmp_path / "clean.nwb", traces=clean)
    p_noisy, e_noisy = _baseline(tmp_path / "noisy.nwb", traces=noisy)
    assert _aggregate(p_noisy, e_noisy) == _aggregate(p_clean, e_clean), (
        "sub-TRACE_ROUNDING noise must be absorbed (same hash)"
    )


def test_fingerprint_geometry_parity(tmp_path):
    """The persisted-region geometry hash equals one recomputed from
    SpikeInterface ``get_channel_locations`` for an unperturbed file -- pinning
    the SI readback surface as parity, not as the canonical fingerprint source.
    """
    import spikeinterface.extractors as se

    from spyglass.spikesorting.v2._recording_fingerprint import (
        geometry_component_hash,
        recording_content_fingerprint,
    )

    path, es_path = _baseline(tmp_path / "rec.nwb")
    components = recording_content_fingerprint(
        path, electrical_series_path=es_path
    )

    recording = se.read_nwb_recording(
        str(path), electrical_series_path=es_path, load_time_vector=True
    )
    si_locations = np.asarray(recording.get_channel_locations(), dtype=float)

    assert components["geometry"] == geometry_component_hash(si_locations), (
        "persisted-region geometry must match SpikeInterface's "
        "get_channel_locations for an unperturbed file"
    )


def test_fingerprint_components_match_canonical_set(tmp_path):
    """The producer emits exactly the canonical ``FINGERPRINT_COMPONENTS`` keys.

    ``content_hash`` is ``combined_hash`` of this dict, so a silently dropped or
    renamed component would yield a stable-but-wrong scalar; pinning the key set
    guards that identity contract.
    """
    from spyglass.spikesorting.v2._recording_fingerprint import (
        FINGERPRINT_COMPONENTS,
        recording_content_fingerprint,
    )

    path, es_path = _baseline(tmp_path / "rec.nwb")
    components = recording_content_fingerprint(
        path, electrical_series_path=es_path
    )
    assert set(components) == set(FINGERPRINT_COMPONENTS)


def test_geometry_component_hash_rejects_empty_coords():
    """A persisted region with no coordinate columns (no probe geometry) is a
    structural impossibility -- hashing it would fold an empty array into a
    content-free constant, so it raises instead."""
    from spyglass.spikesorting.v2._recording_fingerprint import (
        geometry_component_hash,
    )

    with pytest.raises(ValueError, match="no probe geometry"):
        geometry_component_hash(np.zeros((4, 0)))


def test_geometry_component_hash_distinguishes_rel_z():
    """A 3-D layout (rel_x/rel_y/rel_z) does not collide with the 2-D layout of
    the same in-plane coordinates -- ``rel_z`` is folded into the geometry hash,
    so a depth change cannot pass as identical geometry."""
    from spyglass.spikesorting.v2._recording_fingerprint import (
        geometry_component_hash,
    )

    xy = np.array([[0.0, 0.0], [0.0, 20.0], [0.0, 40.0], [0.0, 60.0]])
    xyz = np.column_stack([xy, np.zeros(4)])
    xyz_deep = np.column_stack([xy, np.full(4, 5.0)])

    assert geometry_component_hash(xy) != geometry_component_hash(xyz), (
        "2-D and 3-D geometry of the same in-plane coords must not collide"
    )
    assert geometry_component_hash(xyz) != geometry_component_hash(xyz_deep), (
        "a rel_z (depth) change must change the geometry hash"
    )


def test_fingerprint_rejects_zero_segment_recording(tmp_path, monkeypatch):
    """A degenerate readback (zero segments) is refused, not hashed to a
    content-free constant -- two distinct broken/truncated readbacks must never
    report 'no drift' on traces that were never actually read."""
    import spikeinterface.extractors as se

    from spyglass.spikesorting.v2._recording_fingerprint import (
        recording_content_fingerprint,
    )

    path, es_path = _baseline(tmp_path / "rec.nwb")

    class _ZeroSegmentRecording:
        def get_num_segments(self):
            return 0

    monkeypatch.setattr(
        se, "read_nwb_recording", lambda *a, **k: _ZeroSegmentRecording()
    )
    with pytest.raises(ValueError, match="no trace samples"):
        recording_content_fingerprint(path, electrical_series_path=es_path)


class TestRecordingArtifactLock:
    """Per-``recording_id`` lock serializing one recording's artifact slot.

    A rebuild (``get_recording`` read-repair / ``_rebuild_nwb_artifact``) and a
    reclamation (``RecordingArtifactRecompute.delete_files``) of the SAME
    recording must never interleave; different recordings stay free to run
    concurrently. Mirrors the ``analyzer_curation_lock`` contention contract.
    DB-free: only ``analyzer_cache_root`` resolves (via the custom config dir).
    """

    def test_same_recording_is_mutually_exclusive(
        self, tmp_path, restore_custom_config
    ):
        import datajoint as dj
        from filelock import Timeout

        from spyglass.spikesorting.v2._recording_fingerprint import (
            recording_artifact_lock,
        )

        dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
        held = recording_artifact_lock("rec-A")
        contender = recording_artifact_lock("rec-A")
        with held:
            with pytest.raises(Timeout):
                contender.acquire(timeout=0)

    def test_different_recordings_do_not_block(
        self, tmp_path, restore_custom_config
    ):
        import datajoint as dj

        from spyglass.spikesorting.v2._recording_fingerprint import (
            recording_artifact_lock,
        )

        dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
        with recording_artifact_lock("rec-A"):
            with recording_artifact_lock("rec-B").acquire(timeout=0):
                pass
