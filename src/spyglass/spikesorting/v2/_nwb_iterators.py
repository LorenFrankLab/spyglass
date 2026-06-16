"""Chunked HDF5 writers for the preprocessed Recording artifact.

Used by ``Recording._write_nwb_artifact`` to stream the
``(n_samples, n_channels)`` trace array and the ``(n_samples,)``
timestamps vector into the ``AnalysisNwbfile`` via HDMF's
``GenericDataChunkIterator``. Without these iterators, 30 kHz x 128 ch
x 1 h recordings (~110 GB float64) would have to materialize in RAM
before the NWB write, which OOMs on any lab workstation.

The iterators are direct ports of v1's
``SpikeInterfaceRecordingDataChunkIterator`` and
``TimestampsDataChunkIterator`` from
``spyglass.spikesorting.v1.recording``. The only intentional change to
the trace iterator is the SpikeInterface 0.104 kwarg rename
``return_scaled`` -> ``return_in_uV``, propagated through ``_get_data``.

The timestamps iterator **narrows the public call surface, but keeps the
internal shim**. v1 makes the *caller* wrap a 1D timestamps vector in a
``BaseRecording`` subclass (``TimestampsExtractor``) purely to satisfy the
``TimestampsDataChunkIterator`` constructor signature. v2's constructor
takes ``timestamps`` and ``sampling_frequency`` directly and builds the
wrapper itself: the ``_TimestampsExtractor`` / ``_TimestampsSegment``
indirection is retained as a private detail (HDMF's base iterator hooks
still expect a recording-segment-like object), it is just no longer the
caller's responsibility. The underlying chunked output shape and
semantics are unchanged; only the call surface is smaller.
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
    without materializing the full array. ``buffer_gb=5`` matches v1's
    production choice; smaller buffers trade RAM for write throughput.
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


class _TimestampsSegment(si.BaseRecordingSegment):
    """One-segment shim that lets the timestamps array masquerade as a recording.

    Private helper: the chunked iterator hands the segment's
    ``get_traces`` slice to HDMF, which expects a recording-segment-like
    interface. The shim has no API surface beyond the iterator.
    """

    def __init__(self, timestamps, sampling_frequency, t_start, dtype):
        si.BaseRecordingSegment.__init__(
            self,
            sampling_frequency=sampling_frequency,
            t_start=t_start,
        )
        # v1 assigns ``self._timeseries = timestamps`` (no copy) and
        # trusts the caller's dtype. v2 routes through
        # ``np.asarray(..., dtype=dtype)`` so a caller passing the
        # wrong dtype gets a deterministic promotion instead of a
        # mismatch surfacing later in ``_get_dtype``. ``np.asarray``
        # is a no-op (no copy) when ``timestamps`` is already
        # ``dtype=float64``, which is the production path.
        self._timeseries = np.asarray(timestamps, dtype=dtype)

    def get_num_samples(self) -> int:
        return self._timeseries.shape[0]

    def get_traces(
        self,
        start_frame=None,
        end_frame=None,
        channel_indices=None,
    ) -> np.ndarray:
        return np.squeeze(self._timeseries[start_frame:end_frame])


class _TimestampsExtractor(si.BaseRecording):
    """Private one-channel recording wrapper around a timestamps vector."""

    def __init__(self, timestamps, sampling_frequency=30e3):
        si.BaseRecording.__init__(
            self, sampling_frequency, channel_ids=[0], dtype=np.float64
        )
        self.add_recording_segment(
            _TimestampsSegment(
                timestamps=timestamps,
                sampling_frequency=sampling_frequency,
                t_start=None,
                dtype=np.float64,
            )
        )


class TimestampsDataChunkIterator(GenericDataChunkIterator):
    """HDMF chunked iterator over a 1D ``(n_samples,)`` timestamps vector.

    Unlike v1's version, the constructor takes the raw timestamps
    array + sampling frequency directly. The internal
    ``_TimestampsExtractor`` indirection is kept private because HDMF's
    base iterator hooks expect a recording-segment-like object; callers
    should not import it.
    """

    def __init__(
        self,
        timestamps: np.ndarray,
        sampling_frequency: float,
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
        timestamps : np.ndarray, shape (n_samples,)
            The wall-clock timestamps vector to stream.
        sampling_frequency : float
            Sampling frequency of the recording, in Hz.
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
        self._extractor = _TimestampsExtractor(
            timestamps=timestamps,
            sampling_frequency=sampling_frequency,
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
        # ``_get_maxshape`` returns a 1-tuple, so HDMF passes a
        # 1-tuple ``(slice,)`` here (not ``(slice, slice)`` as a
        # 2D shape would). ``selection[0]`` is the row-axis slice;
        # the timestamps "recording" has a single channel and no
        # column-axis selection exists.
        return self._extractor.get_traces(
            segment_index=0,
            channel_ids=[0],
            start_frame=selection[0].start,
            end_frame=selection[0].stop,
            return_in_uV=False,
        )

    def _get_dtype(self):
        return self._extractor.get_dtype()

    def _get_maxshape(self):
        return (self._extractor.get_num_samples(segment_index=0),)
