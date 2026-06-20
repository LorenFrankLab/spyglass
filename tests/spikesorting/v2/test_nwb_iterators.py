"""Direct coverage for the chunked NWB-write iterators.

``Recording._write_nwb_artifact`` streams the preprocessed traces and the
timestamps vector into the AnalysisNwbfile via these iterators; before this
file they were exercised only transitively through the full write path, so a
chunk/buffer-sizing or ``np.squeeze`` regression could silently corrupt
persisted timestamps (the spine of disjoint-recording readback). Audit
test-hardening #15. Hermetic -- no DataJoint / DB.
"""

from __future__ import annotations

import numpy as np


def test_timestamps_iterator_reconstructs_with_final_one_sample_chunk():
    """A ``TimestampsDataChunkIterator`` over a non-uniform (gapped) vector
    reconstructs the input EXACTLY, including a forced final 1-sample buffer
    (the ``n_samples % buffer == 1`` tail edge of ``_TimestampsSegment``).
    """
    from spyglass.spikesorting.v2._nwb_iterators import (
        TimestampsDataChunkIterator,
    )

    # [chunk1, large wall-clock gap, chunk2] -> non-uniform timestamps. Length
    # 7 with buffer length 3 forces buffers of 3, 3, 1; the final 1-sample
    # buffer is the length-1 tail edge of get_traces.
    ts = np.array(
        [0.0, 1e-4, 2e-4, 5.0, 5.0001, 5.0002, 5.0003], dtype=np.float64
    )
    it = TimestampsDataChunkIterator(
        timestamps=ts,
        sampling_frequency=30000.0,
        buffer_shape=(3,),
        chunk_shape=(1,),
    )
    # 1D maxshape: the timestamps "recording" has a single channel.
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
    from spyglass.spikesorting.v2._nwb_iterators import (
        TimestampsDataChunkIterator,
    )

    ts = np.array([42.0], dtype=np.float64)
    it = TimestampsDataChunkIterator(timestamps=ts, sampling_frequency=30000.0)
    assert it._get_maxshape() == (1,)
    recon = np.full_like(ts, np.nan)
    for chunk in it:
        recon[chunk.selection] = chunk.data
    assert np.array_equal(recon, ts)


def test_timestamps_segment_length_one_chunk_returns_1d():
    """A single-sample chunk must return a 1-D ``(1,)`` array, not a 0-d
    scalar. The slice of the 1-D timestamps vector is already 1-D, so no
    squeeze is needed; squeezing collapsed an ``n_samples % buffer == 1`` tail
    to 0-d, which only survived by HDMF's broadcast-assignment luck."""
    from spyglass.spikesorting.v2._nwb_iterators import _TimestampsSegment

    seg = _TimestampsSegment(
        timestamps=np.arange(5, dtype=np.float64),
        sampling_frequency=30000.0,
        t_start=None,
        dtype=np.float64,
    )
    tail = seg.get_traces(start_frame=4, end_frame=5)  # length-1 final chunk
    assert (
        tail.ndim == 1 and tail.shape == (1,)
    ), f"length-1 chunk must stay 1-D (1,); got shape {tail.shape}"
    multi = seg.get_traces(start_frame=0, end_frame=3)
    assert multi.shape == (3,)
