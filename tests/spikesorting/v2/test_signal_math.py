"""Tests for the ``_signal_math`` pure scientific helpers.

Covers the boundary invariant guards (positive sampling frequency,
monotonic timestamps) wired into the frame-mapping functions
(``_consolidate_intervals`` / ``_spike_times_to_frames`` /
``_base_intervals_from_timestamps``) and the absolute-time merge dedup
(``_dedup_merged_spike_times``) that the lazy units-NWB merge is built on.
All pure numpy -- no DB, no NWB IO.
"""

from __future__ import annotations

import pytest


def test_assert_positive_sampling_frequency_rejects_nonpositive_or_nonfinite():
    """A finite positive fs passes (and is returned as float); 0 / negative /
    NaN / inf raise -- a bad fs silently yields an inf/NaN sample period and
    mis-maps every frame."""
    from spyglass.spikesorting.v2._signal_math import (
        assert_positive_sampling_frequency,
    )

    assert assert_positive_sampling_frequency(30000) == 30000.0
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite positive"):
            assert_positive_sampling_frequency(bad, context="ctx: ")


def test_assert_monotonic_timestamps_rejects_empty_and_backward_steps():
    """A non-decreasing vector (duplicates allowed) and a single sample pass;
    an empty vector or a backward step raise -- searchsorted-based frame
    mapping would otherwise silently mis-slice the recording."""
    import numpy as np

    from spyglass.spikesorting.v2._signal_math import (
        assert_monotonic_timestamps,
    )

    assert_monotonic_timestamps(np.array([0.0, 1.0, 1.0, 2.0]))
    assert_monotonic_timestamps(np.array([5.0]))
    with pytest.raises(ValueError, match="empty"):
        assert_monotonic_timestamps(np.array([]), context="ctx: ")
    with pytest.raises(ValueError, match="monotonic"):
        assert_monotonic_timestamps(np.array([0.0, 2.0, 1.0]), context="ctx: ")
    # A NaN must NOT slip through: ``np.diff`` across NaN is NaN and
    # ``NaN < 0`` is False, so a NaN-masked vector would otherwise pass the
    # backward-step check yet mis-slice via searchsorted (NaN sorts as +inf).
    with pytest.raises(ValueError, match="non-finite"):
        assert_monotonic_timestamps(
            np.array([0.0, 1.0, np.nan, 0.5, 2.0]), context="ctx: "
        )


def test_consolidate_intervals_rejects_nonmonotonic_timestamps():
    """The frame mapping is ``searchsorted``-based, so a backward step in the
    timestamp vector would silently mis-slice; reject it at the boundary."""
    import numpy as np

    from spyglass.spikesorting.v2._signal_math import _consolidate_intervals

    with pytest.raises(ValueError, match="monotonic"):
        _consolidate_intervals(
            np.array([[0.0, 1.0]]), np.array([0.0, 2.0, 1.0])
        )


def test_spike_times_to_frames_rejects_nonmonotonic_recording_times():
    """``_spike_times_to_frames`` searchsorts spike times into the recording
    timeline; a backward step there mis-maps every spike frame."""
    import numpy as np

    from spyglass.spikesorting.v2._signal_math import _spike_times_to_frames

    with pytest.raises(ValueError, match="monotonic"):
        _spike_times_to_frames(
            np.array([0.0, 2.0, 1.0]), np.array([0.5]), 3, unit_id=0
        )


def test_base_intervals_from_timestamps_rejects_nonpositive_fs():
    """A zero/negative sampling frequency makes the gap threshold ``1.5/fs``
    infinite/negative, silently collapsing or exploding the chunk split."""
    import numpy as np

    from spyglass.spikesorting.v2._signal_math import (
        _base_intervals_from_timestamps,
    )

    with pytest.raises(ValueError, match="finite positive"):
        _base_intervals_from_timestamps(np.array([0.0, 1.0, 2.0]), 0.0)


def test_dedup_merged_spike_times_drops_cross_unit_coincidences():
    """Cross-contributor spikes within delta collapse; far ones are kept."""
    from spyglass.spikesorting.v2._signal_math import (
        _dedup_merged_spike_times,
    )

    # unit A=[0.0], unit B=[0.0001] -> 0.1 ms apart, different units -> dedup.
    near = _dedup_merged_spike_times([[0.0], [0.0001]], delta_s=0.4e-3)
    assert near.tolist() == [0.0]
    # unit A=[0.0], unit B=[0.001] -> 1 ms apart -> both kept.
    far = _dedup_merged_spike_times([[0.0], [0.001]], delta_s=0.4e-3)
    assert far.tolist() == [0.0, 0.001]


def test_dedup_merged_spike_times_keeps_within_unit_close_pair():
    """A close pair from the SAME contributor is a real event, kept.

    The membership guard mirrors SI's ``(diff>delta)|(diff(membership)==0)``:
    a sub-delta pair is dropped only when the two spikes came from DIFFERENT
    contributors (a cross-unit double-detection), never within one unit.
    """
    from spyglass.spikesorting.v2._signal_math import (
        _dedup_merged_spike_times,
    )

    out = _dedup_merged_spike_times([[0.0, 0.0001]], delta_s=0.4e-3)
    assert out.tolist() == [0.0, 0.0001]


def test_dedup_merged_spike_times_empty():
    from spyglass.spikesorting.v2._signal_math import (
        _dedup_merged_spike_times,
    )

    assert _dedup_merged_spike_times([], delta_s=0.4e-3).tolist() == []
