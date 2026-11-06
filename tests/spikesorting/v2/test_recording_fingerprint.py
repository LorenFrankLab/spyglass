"""Content-fingerprint identity for the v2 preprocessed recording cache.

Read from the stored ElectricalSeries (traces, timestamps, geometry,
scaling); deterministic per file and sensitive to meaningful changes. No DB.
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
    from spyglass.spikesorting.v2._storage.recompute import combined_hash
    from spyglass.spikesorting.v2._recording.fingerprint import (
        recording_content_fingerprint,
    )

    return combined_hash(
        recording_content_fingerprint(path, electrical_series_path=es_path)
    )


def test_recording_content_fingerprint_deterministic(tmp_path):
    """Repeated reads of one file give identical components and aggregate."""
    from spyglass.spikesorting.v2._recording.fingerprint import (
        recording_content_fingerprint,
    )

    path, es_path = _baseline(tmp_path / "rec.nwb")
    first = recording_content_fingerprint(path, electrical_series_path=es_path)
    second = recording_content_fingerprint(path, electrical_series_path=es_path)

    assert first == second
    # The component dict names each axis so a recompute diff can localize drift.
    assert set(first) == {"traces", "timestamps", "geometry", "metadata"}
    assert _aggregate(path, es_path) == _aggregate(path, es_path)


def test_timestamp_hash_is_bounded_and_matches_whole_vector(
    tmp_path, monkeypatch
):
    """Streaming includes chunk boundaries and retains the persisted hash format."""
    import hashlib

    import h5py
    from spikeinterface.core import BaseRecording

    from spyglass.spikesorting.v2._storage.recompute import combined_hash
    from spyglass.spikesorting.v2._recording.fingerprint import (
        TIMESTAMP_ROUNDING,
        recording_content_fingerprint,
    )

    n_frames = 600_017
    times = np.arange(n_frames, dtype=float) / 30_000.0 + 1234.0
    times[300_000:] += 0.5
    times[-1] += 0.1
    path, es_path = _baseline(
        tmp_path / "chunked.nwb",
        traces=np.zeros((n_frames, 4), dtype=np.float32),
        timestamps=times,
    )
    expected = hashlib.md5(
        np.round(times, TIMESTAMP_ROUNDING).astype("<f8").tobytes()
    ).hexdigest()
    getitem = h5py.Dataset.__getitem__
    sizes = []

    def bounded_getitem(dataset, index):
        result = getitem(dataset, index)
        if dataset.name.endswith("/timestamps"):
            sizes.append(np.size(result))
            assert np.size(result) <= 300_000
        return result

    def no_get_times(*args, **kwargs):
        pytest.fail("fingerprinting must not materialize the time vector")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", bounded_getitem)
    monkeypatch.setattr(BaseRecording, "get_times", no_get_times)
    components = recording_content_fingerprint(
        path, electrical_series_path=es_path
    )
    assert sizes
    assert components["timestamps"] == combined_hash({"segment_0": expected})


def test_timestamp_hash_reads_contiguous_slices(tmp_path, monkeypatch):
    """Each timestamp chunk is read as one slice, not as a list of points.

    h5py reads a fancy index point by point (~47 s on a 20 min recording).
    """
    import h5py

    n_frames = 600_017
    path, es_path = _baseline(
        tmp_path / "sliced.nwb",
        traces=np.zeros((n_frames, 4), dtype=np.float32),
        timestamps=np.arange(n_frames, dtype=float) / 30_000.0 + 1234.0,
    )
    getitem = h5py.Dataset.__getitem__
    indices = []

    def recording_getitem(dataset, index):
        if dataset.name.endswith("/timestamps"):
            indices.append(index)
        return getitem(dataset, index)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", recording_getitem)
    _aggregate(path, es_path)
    chunk_reads = [index for index in indices if not np.isscalar(index)]
    assert chunk_reads
    assert all(isinstance(index, slice) for index in chunk_reads), [
        type(index).__name__ for index in chunk_reads
    ]


def test_recording_content_fingerprint_discriminates(tmp_path):
    """Every scientifically-meaningful perturbation changes the aggregate;
    sub-``TRACE_ROUNDING`` float noise does not."""
    from spyglass.spikesorting.v2._recording.fingerprint import TRACE_ROUNDING

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

    # Integer traces sit 5e-5 from any rounding boundary, so this float32
    # noise changes the bytes but not the rounded values.
    clean = rng.integers(-100, 100, size=(50, 4)).astype("float32")
    noise = 2.0 * 10.0 ** (-(TRACE_ROUNDING + 1))  # 2e-5 µV, < 5e-5 boundary
    noisy = (clean.astype("float64") + noise).astype("float32")
    assert not np.array_equal(
        noisy, clean
    ), "precondition: the noise must perturb the stored float32 bytes"
    p_clean, e_clean = _baseline(tmp_path / "clean.nwb", traces=clean)
    p_noisy, e_noisy = _baseline(tmp_path / "noisy.nwb", traces=noisy)
    assert _aggregate(p_noisy, e_noisy) == _aggregate(
        p_clean, e_clean
    ), "sub-TRACE_ROUNDING noise must be absorbed (same hash)"


@pytest.mark.integration
@pytest.mark.nwb
@pytest.mark.io_heavy
@pytest.mark.parametrize(
    "overrides",
    [
        {"channel_ids": [10, 20, 30, 40]},
        {"filtering": "no temporal filtering"},
    ],
    ids=["electrode_identity", "filtering_provenance"],
)
def test_fingerprint_discriminates_metadata_without_signal_changes(
    tmp_path, overrides
):
    """Metadata contributes independently of voltage, time, and probe positions."""
    from spyglass.spikesorting.v2._recording.fingerprint import (
        recording_content_fingerprint,
    )

    base_path, base_series = _baseline(tmp_path / "base.nwb")
    changed_path, changed_series = _baseline(
        tmp_path / "changed.nwb", **overrides
    )
    base = recording_content_fingerprint(
        base_path, electrical_series_path=base_series
    )
    changed = recording_content_fingerprint(
        changed_path, electrical_series_path=changed_series
    )
    for component in ("traces", "timestamps", "geometry"):
        assert base[component] == changed[component], component
    assert base["metadata"] != changed["metadata"]
    assert _aggregate(base_path, base_series) != _aggregate(
        changed_path, changed_series
    )


def test_fingerprint_geometry_parity(tmp_path):
    """The persisted geometry hash equals one from SpikeInterface
    ``get_channel_locations``.

    Holds only for 2-D electrodes; real artifacts store ``(n_channels, 3)``
    while SI reloads ``(n_channels, 2)``.
    """
    import spikeinterface.extractors as se

    from spyglass.spikesorting.v2._recording.fingerprint import (
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

    ``content_hash`` combines this dict, so a dropped key changes identity.
    """
    from spyglass.spikesorting.v2._recording.fingerprint import (
        FINGERPRINT_COMPONENTS,
        recording_content_fingerprint,
    )

    path, es_path = _baseline(tmp_path / "rec.nwb")
    components = recording_content_fingerprint(
        path, electrical_series_path=es_path
    )
    assert set(components) == set(FINGERPRINT_COMPONENTS)


def test_geometry_component_hash_rejects_empty_coords():
    """A region with no coordinate columns raises rather than hashing an
    empty array."""
    from spyglass.spikesorting.v2._recording.fingerprint import (
        geometry_component_hash,
    )

    with pytest.raises(ValueError, match="no probe geometry"):
        geometry_component_hash(np.zeros((4, 0)))


def test_geometry_component_hash_distinguishes_rel_z():
    """A 3-D layout does not collide with the 2-D layout of the same
    in-plane coordinates."""
    from spyglass.spikesorting.v2._recording.fingerprint import (
        geometry_component_hash,
    )

    xy = np.array([[0.0, 0.0], [0.0, 20.0], [0.0, 40.0], [0.0, 60.0]])
    xyz = np.column_stack([xy, np.zeros(4)])
    xyz_deep = np.column_stack([xy, np.full(4, 5.0)])

    assert geometry_component_hash(xy) != geometry_component_hash(
        xyz
    ), "2-D and 3-D geometry of the same in-plane coords must not collide"
    assert geometry_component_hash(xyz) != geometry_component_hash(
        xyz_deep
    ), "a rel_z (depth) change must change the geometry hash"


def test_fingerprint_rejects_zero_segment_recording(tmp_path, monkeypatch):
    """A zero-segment readback raises rather than hashing to a constant."""
    from spyglass.spikesorting.v2._storage import nwb as _recording_nwb
    from spyglass.spikesorting.v2._recording.fingerprint import (
        recording_content_fingerprint,
    )

    path, es_path = _baseline(tmp_path / "rec.nwb")

    class _ZeroSegmentRecording:
        def get_num_segments(self):
            return 0

    monkeypatch.setattr(
        _recording_nwb,
        "read_recording_nwb",
        lambda *a, **k: _ZeroSegmentRecording(),
    )
    with pytest.raises(ValueError, match="no trace samples"):
        recording_content_fingerprint(path, electrical_series_path=es_path)


class TestRecordingArtifactLock:
    """Per-``recording_id`` lock serializing one recording's artifact slot.

    A rebuild and a reclamation of the same recording never interleave;
    different recordings run concurrently. No DB.
    """

    def test_same_recording_is_mutually_exclusive(
        self, tmp_path, restore_custom_config
    ):
        import datajoint as dj
        from filelock import Timeout

        from spyglass.spikesorting.v2._recording.fingerprint import (
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

        from spyglass.spikesorting.v2._recording.fingerprint import (
            recording_artifact_lock,
        )

        dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
        with recording_artifact_lock("rec-A"):
            with recording_artifact_lock("rec-B").acquire(timeout=0):
                pass
