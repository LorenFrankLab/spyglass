"""Chunked HDF5 writers for the preprocessed Recording artifact.

Used by ``Recording._write_nwb_artifact`` to stream the
``(n_samples, n_channels)`` trace array and the ``(n_samples,)``
timestamps vector into the ``AnalysisNwbfile`` via HDMF's
``GenericDataChunkIterator``. Without these iterators, 30 kHz x 128 ch
x 1 h recordings (~110 GB float64) would have to materialize in RAM
before the NWB write, which OOMs on any lab workstation.

The trace iterator reads ``(n_samples, n_channels)`` slices via the
recording's ``get_traces(...)``, using the SpikeInterface 0.104
``return_in_uV`` kwarg.

The timestamps iterator slices a 1D ``timestamps`` vector directly, so a
lazy timestamp vector materializes only the chunk HDMF requests.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import spikeinterface as si
from hdmf.data_utils import GenericDataChunkIterator


class SpikeInterfaceRecordingDataChunkIterator(GenericDataChunkIterator):
    """HDMF chunked iterator over a SpikeInterface ``BaseRecording``.

    Reads ``(n_samples, n_channels)`` slices via the recording's
    ``get_traces(...)`` so the NWB writer can stream the dataset
    without materializing the full array. A ``buffer_gb=5`` default
    balances RAM against write throughput; smaller buffers trade RAM
    for write throughput.
    """

    def __init__(
        self,
        recording: si.BaseRecording,
        segment_index: int = 0,
        return_in_uV: bool = False,
        buffer_gb: Optional[float] = None,
        buffer_shape: Optional[tuple] = None,
        chunk_mb: Optional[float] = None,
        chunk_shape: Optional[tuple] = None,
        display_progress: bool = False,
        progress_bar_options: Optional[dict] = None,
    ):
        """Build the iterator over a SpikeInterface recording.

        Parameters
        ----------
        recording : si.BaseRecording
            The recording whose traces are streamed.
        segment_index : int, optional
            Segment to read from. Default ``0``.
        return_in_uV : bool, optional
            Return scaled microvolt traces rather than raw counts.
            Default ``False`` (v2 writes unscaled traces).
        buffer_gb : float, optional
            Target buffer size in GB. ``None`` uses HDMF's default.
        buffer_shape : tuple, optional
            Explicit buffer shape, overriding ``buffer_gb``.
        chunk_mb : float, optional
            Target chunk size in MB. ``None`` uses HDMF's default.
        chunk_shape : tuple, optional
            Explicit chunk shape, overriding ``chunk_mb``.
        display_progress : bool, optional
            Show a progress bar during iteration. Default ``False``.
        progress_bar_options : dict, optional
            Keyword options forwarded to the progress bar.
        """
        self.recording = recording
        self.segment_index = segment_index
        self.return_in_uV = return_in_uV
        self.channel_ids = recording.get_channel_ids()
        super().__init__(
            buffer_gb=buffer_gb,
            buffer_shape=buffer_shape,
            chunk_mb=chunk_mb,
            chunk_shape=chunk_shape,
            display_progress=display_progress,
            progress_bar_options=progress_bar_options,
        )

    def _get_data(self, selection: Tuple[slice]) -> Iterable:
        return self.recording.get_traces(
            segment_index=self.segment_index,
            channel_ids=self.channel_ids[selection[1]],
            start_frame=selection[0].start,
            end_frame=selection[0].stop,
            return_in_uV=self.return_in_uV,
        )

    def _get_dtype(self):
        return self.recording.get_dtype()

    def _get_maxshape(self):
        return (
            self.recording.get_num_samples(segment_index=self.segment_index),
            self.recording.get_num_channels(),
        )


class TimestampsDataChunkIterator(GenericDataChunkIterator):
    """HDMF chunked iterator over a 1D ``(n_samples,)`` timestamps vector.

    Each buffer is a direct slice of ``timestamps``. A Spyglass lazy
    timestamp vector is kept as-is, so only the chunks requested by HDMF are
    materialized; any other array-like is converted once to ``float64``.
    """

    def __init__(
        self,
        timestamps,
        buffer_gb: Optional[float] = None,
        buffer_shape: Optional[tuple] = None,
        chunk_mb: Optional[float] = None,
        chunk_shape: Optional[tuple] = None,
        display_progress: bool = False,
        progress_bar_options: Optional[dict] = None,
    ):
        """Build the iterator over a 1D timestamps vector.

        Parameters
        ----------
        timestamps : array-like, shape (n_samples,)
            The wall-clock timestamps vector to stream. Lazy timestamp vectors
            are consumed chunk-by-chunk without first converting the whole
            object to a NumPy array.
        buffer_gb : float, optional
            Target buffer size in GB. ``None`` uses HDMF's default.
        buffer_shape : tuple, optional
            Explicit buffer shape, overriding ``buffer_gb``.
        chunk_mb : float, optional
            Target chunk size in MB. ``None`` uses HDMF's default.
        chunk_shape : tuple, optional
            Explicit chunk shape, overriding ``chunk_mb``.
        display_progress : bool, optional
            Show a progress bar during iteration. Default ``False``.
        progress_bar_options : dict, optional
            Keyword options forwarded to the progress bar.
        """
        self._timestamps = (
            timestamps
            if getattr(timestamps, "_spyglass_lazy_timestamps", False)
            else np.asarray(timestamps, dtype=np.float64)
        )
        super().__init__(
            buffer_gb=buffer_gb,
            buffer_shape=buffer_shape,
            chunk_mb=chunk_mb,
            chunk_shape=chunk_shape,
            display_progress=display_progress,
            progress_bar_options=progress_bar_options,
        )

    def _get_data(self, selection: Tuple[slice]) -> Iterable:
        # ``_get_maxshape`` is a 1-tuple, so HDMF passes a 1-tuple
        # ``(slice,)``. The slice of the 1-D vector is already 1-D; do not
        # ``np.squeeze`` it, which would collapse a length-1 final chunk
        # (``n_samples % buffer == 1``) to a 0-d scalar.
        return np.asarray(self._timestamps[selection[0]], dtype=np.float64)

    def _get_dtype(self):
        return np.dtype(np.float64)

    def _get_maxshape(self):
        return (self._timestamps.shape[0],)
