"""write_buffer_gb sizes the streaming-write buffer to a fixed duration,
capped at 5 GB, so narrow groups do not buffer the whole recording.
"""

from __future__ import annotations

import pytest

from spyglass.spikesorting.v2._storage.metadata import write_buffer_gb


def _buffered_seconds(gb, n_channels, fs=30000.0, itemsize=8):
    """Recording duration (s) a buffer of ``gb`` GB spans for this geometry."""
    return gb * 1e9 / (n_channels * itemsize * fs)


def test_narrow_group_buffer_is_small():
    """A 4-channel group buffers far less than the 5 GB cap (bounded by time)."""
    gb = write_buffer_gb(n_channels=4, sampling_frequency=30000.0)
    assert 0 < gb < 0.5


def test_buffer_holds_same_duration_regardless_of_channel_count():
    """The contract is a bounded DURATION, not a fixed byte size: a narrow and
    a wide group (both below the cap) buffer the same number of seconds."""
    narrow = write_buffer_gb(n_channels=4, sampling_frequency=30000.0)
    wide = write_buffer_gb(n_channels=128, sampling_frequency=30000.0)
    assert wide > narrow
    assert _buffered_seconds(narrow, 4) == pytest.approx(
        _buffered_seconds(wide, 128)
    )


def test_cap_binds_for_huge_channel_count():
    """A pathologically wide group is capped at the 5 GB ceiling."""
    gb = write_buffer_gb(n_channels=4000, sampling_frequency=30000.0)
    assert gb == 5.0
