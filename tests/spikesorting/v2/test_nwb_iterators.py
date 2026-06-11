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
    that exercises the ``np.squeeze`` 0-d path in ``_TimestampsSegment``.
    """
    from spyglass.spikesorting.v2._nwb_iterators import (
        TimestampsDataChunkIterator,
    )

    # [chunk1, large wall-clock gap, chunk2] -> non-uniform timestamps. Length
    # 7 with buffer length 3 forces buffers of 3, 3, 1; the final 1-sample
    # buffer squeezes to a 0-d array in get_traces.
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
