"""Direct coverage for the chunked NWB-write iterators.

``Recording._write_nwb_artifact`` streams the preprocessed traces and the
timestamps vector into the AnalysisNwbfile via these iterators. Exercised only
transitively through the full write path, a chunk/buffer-sizing or
``np.squeeze`` regression could silently corrupt persisted timestamps (the
spine of disjoint-recording readback), so they are tested directly here.
Hermetic -- no DataJoint / DB.
"""

from __future__ import annotations

import numpy as np


def test_timestamps_iterator_reconstructs_with_final_one_sample_chunk():
    """A ``TimestampsDataChunkIterator`` over a non-uniform (gapped) vector
    reconstructs the input EXACTLY, including a forced final 1-sample buffer
    (the ``n_samples % buffer == 1`` tail edge).
    """
    from spyglass.spikesorting.v2._storage.iterators import (
        TimestampsDataChunkIterator,
    )

    # [chunk1, large wall-clock gap, chunk2] -> non-uniform timestamps. Length
    # 7 with buffer length 3 forces buffers of 3, 3, 1; the final 1-sample
    # buffer is the length-1 tail edge.
    ts = np.array(
        [0.0, 1e-4, 2e-4, 5.0, 5.0001, 5.0002, 5.0003], dtype=np.float64
    )
    it = TimestampsDataChunkIterator(
        timestamps=ts,
        buffer_shape=(3,),
        chunk_shape=(1,),
    )
    # 1D maxshape matches the 1D timestamps vector.
    assert it._get_maxshape() == (ts.shape[0],)
    assert np.dtype(it._get_dtype()) == np.float64

    recon = np.full_like(ts, np.nan)
    n_buffers = 0
    for chunk in it:
        recon[chunk.selection] = chunk.data
        n_buffers += 1
    assert n_buffers >= 3, "expected a 3,3,1 buffering (final 1-sample buffer)"
    assert not np.isnan(recon).any(), "some samples were never written"
    assert np.array_equal(recon, ts), "chunked write corrupted the timestamps"


def test_timestamps_iterator_single_sample_input_round_trips():
    """A length-1 timestamps vector (entirely the 0-d squeeze edge) round-trips."""
    from spyglass.spikesorting.v2._storage.iterators import (
        TimestampsDataChunkIterator,
    )

    ts = np.array([42.0], dtype=np.float64)
    it = TimestampsDataChunkIterator(timestamps=ts)
    assert it._get_maxshape() == (1,)
    recon = np.full_like(ts, np.nan)
    for chunk in it:
        recon[chunk.selection] = chunk.data
    assert np.array_equal(recon, ts)


def test_timestamps_iterator_length_one_chunk_returns_1d():
    """A single-sample chunk must return a 1-D ``(1,)`` array, not a 0-d
    scalar. The slice of the 1-D timestamps vector is already 1-D, so no
    squeeze is needed; squeezing collapsed an ``n_samples % buffer == 1`` tail
    to 0-d, which only survived by HDMF's broadcast-assignment luck."""
    from spyglass.spikesorting.v2._storage.iterators import (
        TimestampsDataChunkIterator,
    )

    it = TimestampsDataChunkIterator(timestamps=np.arange(5, dtype=np.float64))
    tail = it._get_data((slice(4, 5),))  # length-1 final chunk
    assert tail.ndim == 1 and tail.shape == (
        1,
    ), f"length-1 chunk must stay 1-D (1,); got shape {tail.shape}"
    multi = it._get_data((slice(0, 3),))
    assert multi.shape == (3,)


def _write_through_iterator(tmp_path, iterator):
    """Stream ``iterator`` into an HDF5 dataset the way the NWB write does."""
    import h5py
    from hdmf.backends.hdf5.h5tools import HDF5IO

    path = tmp_path / "traces.h5"
    with h5py.File(path, "w") as handle:
        HDF5IO.__chunked_iter_fill__(handle, "data", iterator)
    return path


def test_recording_iterator_chunks_hold_every_channel(tmp_path):
    """Each written chunk spans every channel, so a short full-channel read
    (SpikeInterface reads ~1 s windows) touches one or two chunks.

    HDMF's default shape keeps the array's aspect ratio, which for a long
    32-channel recording is ~37 s of ONE channel per chunk: every 1 s read then
    touches 32 chunks, and through h5py's default 8 MiB chunk cache a 20 min
    pass took ~8 min instead of seconds.
    """
    import h5py
    import spikeinterface as si

    from spyglass.spikesorting.v2._storage.iterators import (
        SpikeInterfaceRecordingDataChunkIterator,
    )

    for n_channels in (4, 32):
        traces = np.random.default_rng(0).normal(size=(2_000_000, n_channels))
        recording = si.NumpyRecording([traces], sampling_frequency=30000.0)
        iterator = SpikeInterfaceRecordingDataChunkIterator(
            recording=recording, buffer_gb=0.1
        )
        path = _write_through_iterator(tmp_path, iterator)
        with h5py.File(path, "r") as handle:
            dataset = handle["data"]
            assert dataset.chunks[1] == n_channels
            chunk_bytes = np.prod(dataset.chunks) * dataset.dtype.itemsize
            assert chunk_bytes <= 10e6
            assert dataset.chunks[0] >= 30000, "chunks shorter than 1 s"
            np.testing.assert_array_equal(dataset[:], traces)


def test_recording_iterator_keeps_an_explicit_chunk_shape(tmp_path):
    """A caller's ``chunk_shape`` still wins over the all-channel default."""
    import h5py
    import spikeinterface as si

    from spyglass.spikesorting.v2._storage.iterators import (
        SpikeInterfaceRecordingDataChunkIterator,
    )

    traces = np.arange(40_000 * 8, dtype=np.float64).reshape(40_000, 8)
    recording = si.NumpyRecording([traces], sampling_frequency=30000.0)
    iterator = SpikeInterfaceRecordingDataChunkIterator(
        recording=recording, chunk_shape=(1000, 2), buffer_shape=(10_000, 8)
    )
    path = _write_through_iterator(tmp_path, iterator)
    with h5py.File(path, "r") as handle:
        assert handle["data"].chunks == (1000, 2)
        np.testing.assert_array_equal(handle["data"][:], traces)
