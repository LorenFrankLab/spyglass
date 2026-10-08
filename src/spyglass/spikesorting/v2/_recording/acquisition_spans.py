"""Lazy frame spans with their original acquisition clocks (DB-free)."""

from spikeinterface.core import BaseRecording, BaseRecordingSegment

from spyglass.spikesorting.v2._recording.restriction import (
    _LazyRecordingTimestamps,
)


class _SpanTimestamps(_LazyRecordingTimestamps):
    """Also support SI's vectorized sample_index_to_time without a full read."""

    ndim = 1

    def __getitem__(self, item):
        import numpy as np

        from spyglass.spikesorting.v2._core.signal_math import _segment_times_at

        if isinstance(item, np.ndarray) and np.issubdtype(
            item.dtype, np.integer
        ):
            indices = np.where(item < 0, item + len(self), item)
            if np.any((indices < 0) | (indices >= len(self))):
                raise IndexError(item)
            return _segment_times_at(self.recording, self.start + indices)
        return super().__getitem__(item)


class AcquisitionSpanRecording(BaseRecording):
    """Keep a continuous acquisition span without loading its time vector.

    Unlike SI's FrameSliceRecording, constructing this view does not eagerly
    slice an explicit HDF5 clock. Reconstruction in workers retains the same
    bounded-memory behavior and the same filter boundaries.
    """

    def __init__(self, parent_recording, start_frame, end_frame):
        if parent_recording.get_num_segments() != 1:
            raise ValueError("AcquisitionSpanRecording requires one segment.")
        start_frame, end_frame = int(start_frame), int(end_frame)
        if (
            not 0
            <= start_frame
            < end_frame
            <= parent_recording.get_num_samples()
        ):
            raise ValueError("Invalid acquisition span frame bounds.")
        super().__init__(
            sampling_frequency=parent_recording.get_sampling_frequency(),
            channel_ids=parent_recording.channel_ids,
            dtype=parent_recording.get_dtype(),
        )
        self.add_recording_segment(
            _AcquisitionSpanSegment(parent_recording, start_frame, end_frame)
        )
        parent_recording.copy_metadata(self)
        self._parent = parent_recording
        self._kwargs = {
            "parent_recording": parent_recording,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }


class _AcquisitionSpanSegment(BaseRecordingSegment):
    def __init__(self, parent_recording, start_frame, end_frame):
        super().__init__(
            sampling_frequency=parent_recording.get_sampling_frequency(),
            time_vector=_SpanTimestamps(
                parent_recording, start_frame, end_frame
            ),
        )
        self._source = parent_recording._recording_segments[0]
        self._start = start_frame
        self._length = end_frame - start_frame

    def get_num_samples(self):
        return self._length

    def get_traces(self, start_frame, end_frame, channel_indices):
        return self._source.get_traces(
            self._start + start_frame,
            self._start + end_frame,
            channel_indices,
        )
