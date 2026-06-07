"""write_buffer_gb bounds the streaming-write buffer to ~N seconds of data.

A fixed 5 GB buffer buffers the WHOLE recording for narrow sort groups (a
4-channel tetrode = ~87 min in 5 GB), defeating streaming and spiking RAM
under parallel per-group workers. The buffer should scale with channel count
so a tetrode and a Neuropixels group both buffer ~the same bounded duration.

Hermetic -- pure function, no DB.
"""

from __future__ import annotations

from spyglass.spikesorting.v2.utils import write_buffer_gb


def test_narrow_group_buffer_is_small():
    """A 4-channel group buffers far less than the 5 GB cap (bounded by time)."""
    gb = write_buffer_gb(n_channels=4, sampling_frequency=30000.0)
    assert 0 < gb < 0.5


def test_buffer_scales_with_channel_count():
    narrow = write_buffer_gb(n_channels=4, sampling_frequency=30000.0)
    wide = write_buffer_gb(n_channels=128, sampling_frequency=30000.0)
    # 128 channels buffer ~32x more than 4 (same bounded duration).
    assert wide > narrow
    assert wide == narrow * 32  # 128/4


def test_cap_binds_for_huge_channel_count():
    """A pathologically wide group is capped at the 5 GB ceiling."""
    gb = write_buffer_gb(n_channels=4000, sampling_frequency=30000.0)
    assert gb == 5.0
