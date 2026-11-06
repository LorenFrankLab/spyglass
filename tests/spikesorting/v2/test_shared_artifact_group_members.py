"""Tests for ``_shared_artifact_group.validate_shared_artifact_group_members``.

Members must share one session (IntervalList is keyed by nwb_file_name)
and one sampling frequency (aggregate_channels requires it). DB-free.
"""

from __future__ import annotations

import pytest


def test_validate_shared_group_members_returns_single_session():
    """Members in one session at one sampling frequency validate and yield the
    shared nwb_file_name."""
    from spyglass.spikesorting.v2._artifacts.shared_group import (
        validate_shared_artifact_group_members,
    )

    nwb = validate_shared_artifact_group_members(
        [
            {"nwb_file_name": "s.nwb", "sampling_frequency": 30000.0},
            {"nwb_file_name": "s.nwb", "sampling_frequency": 30000.0},
        ]
    )
    assert nwb == "s.nwb"


def test_validate_shared_group_members_accepts_one_shot_iterable():
    """One-shot iterators work although validation makes two passes."""
    from spyglass.spikesorting.v2._artifacts.shared_group import (
        validate_shared_artifact_group_members,
    )

    rows = (
        {"nwb_file_name": "s.nwb", "sampling_frequency": fs}
        for fs in (30000.0, 30000.0)
    )

    assert validate_shared_artifact_group_members(rows) == "s.nwb"


def test_validate_shared_group_members_rejects_multiple_sessions():
    """Members spanning more than one session raise."""
    from spyglass.spikesorting.v2._artifacts.shared_group import (
        validate_shared_artifact_group_members,
    )

    with pytest.raises(ValueError, match="members span 2 sessions"):
        validate_shared_artifact_group_members(
            [
                {"nwb_file_name": "a.nwb", "sampling_frequency": 30000.0},
                {"nwb_file_name": "b.nwb", "sampling_frequency": 30000.0},
            ]
        )


def test_validate_shared_group_members_rejects_differing_sampling_frequency():
    """Members with differing sampling frequencies raise."""
    from spyglass.spikesorting.v2._artifacts.shared_group import (
        validate_shared_artifact_group_members,
    )

    with pytest.raises(ValueError, match="differing sampling frequencies"):
        validate_shared_artifact_group_members(
            [
                {"nwb_file_name": "s.nwb", "sampling_frequency": 30000.0},
                {"nwb_file_name": "s.nwb", "sampling_frequency": 20000.0},
            ]
        )


# --------------------------------------------------------------------------- #
# ``make_compute`` re-checks the aggregation invariants on loaded recordings,
# raising ``SchemaBypassError`` for members inserted around ``insert_group``.
# --------------------------------------------------------------------------- #


def _np_recording(n_samples, fs, dtype="float32", channel_ids=(1, 2)):
    import numpy as np
    import spikeinterface as si

    return si.NumpyRecording(
        [np.zeros((n_samples, len(channel_ids)), dtype=dtype)],
        sampling_frequency=fs,
        channel_ids=list(channel_ids),
    )


def _recording_with_different_interior_clock():
    import numpy as np

    recording = _np_recording(100, 30000.0)
    times = np.arange(100, dtype=np.float64) / 30000.0
    # Frame count, rate, dtype, and clock endpoints all agree with the
    # reference. Only the acquisition times of interior samples differ.
    times[40:60] += 0.25 / 30000.0
    recording.set_times(times, with_warning=False)
    return recording


def test_assert_shared_group_aggregatable_accepts_matching_members():
    from spyglass.spikesorting.v2._artifacts.shared_group import (
        assert_shared_group_recordings_aggregatable,
    )

    a = _np_recording(100, 30000.0)
    b = _np_recording(100, 30000.0)
    assert_shared_group_recordings_aggregatable(
        [a, b], ["r1", "r2"], ["s.nwb", "s.nwb"]
    )  # no raise


@pytest.mark.parametrize(
    "make_second,nwb_names,match",
    [
        (lambda: _np_recording(100, 20000.0), ["s.nwb", "s.nwb"], "sampling"),
        # Sub-Hz drift that default np.isclose accepts misaligns the time axis.
        (lambda: _np_recording(100, 30000.3), ["s.nwb", "s.nwb"], "sampling"),
        (lambda: _np_recording(50, 30000.0), ["s.nwb", "s.nwb"], "sample"),
        (
            lambda: _np_recording(100, 30000.0, dtype="int16"),
            ["s.nwb", "s.nwb"],
            "dtype",
        ),
        (lambda: _np_recording(100, 30000.0), ["s.nwb", "b.nwb"], "session"),
        (
            _recording_with_different_interior_clock,
            ["s.nwb", "s.nwb"],
            "timestamp vector",
        ),
    ],
)
def test_assert_shared_group_aggregatable_rejects_bypass(
    make_second, nwb_names, match
):
    from spyglass.spikesorting.v2._artifacts.shared_group import (
        assert_shared_group_recordings_aggregatable,
    )
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError

    a = _np_recording(100, 30000.0)
    with pytest.raises(SchemaBypassError, match=match):
        assert_shared_group_recordings_aggregatable(
            [a, make_second()], ["r1", "r2"], nwb_names
        )
