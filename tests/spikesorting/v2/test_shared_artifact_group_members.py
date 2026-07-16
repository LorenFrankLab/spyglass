"""Tests for ``_shared_artifact_group.validate_shared_artifact_group_members``.

Pure validation of shared-artifact-group member metadata: members must
all live in one session (IntervalList is keyed by nwb_file_name) and share
a sampling frequency (aggregate_channels requires identical fs). No DB --
member metadata is passed in directly.
"""

from __future__ import annotations

import pytest


def test_validate_shared_group_members_returns_single_session():
    """Members in one session at one sampling frequency validate and yield the
    shared nwb_file_name."""
    from spyglass.spikesorting.v2._shared_artifact_group import (
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
    """The helper documents an iterable input, so one-shot iterators must work
    even though validation needs two passes over the member metadata."""
    from spyglass.spikesorting.v2._shared_artifact_group import (
        validate_shared_artifact_group_members,
    )

    rows = (
        {"nwb_file_name": "s.nwb", "sampling_frequency": fs}
        for fs in (30000.0, 30000.0)
    )

    assert validate_shared_artifact_group_members(rows) == "s.nwb"


def test_validate_shared_group_members_rejects_multiple_sessions():
    """Members spanning more than one session raise -- a shared artifact pass
    is meaningless across sessions (IntervalList is keyed by nwb_file_name)."""
    from spyglass.spikesorting.v2._shared_artifact_group import (
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
    """Members with differing sampling frequencies raise -- aggregate_channels
    requires identical fs."""
    from spyglass.spikesorting.v2._shared_artifact_group import (
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
# Compute-boundary re-assertion of the channel-aggregation invariants.
#
# ``ArtifactDetection.make_compute`` aggregates the shared-group members with
# ``si.aggregate_channels``; a direct insert of a SharedGroupSource part can
# bypass ``insert_group``'s check, so the same invariants (one session, fs,
# n_samples, dtype, timestamp vector) are re-asserted over the loaded
# recordings and raise ``SchemaBypassError`` on a corrupted member set.
# --------------------------------------------------------------------------- #


def _np_recording(n_samples, fs, dtype="float32", channel_ids=(1, 2)):
    import numpy as np
    import spikeinterface as si

    return si.NumpyRecording(
        [np.zeros((n_samples, len(channel_ids)), dtype=dtype)],
        sampling_frequency=fs,
        channel_ids=list(channel_ids),
    )


def test_assert_shared_group_aggregatable_accepts_matching_members():
    from spyglass.spikesorting.v2._shared_artifact_group import (
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
        # The regression case: a sub-Hz drift NumPy's default np.isclose would
        # accept must still be rejected (it misaligns the aggregated time axis).
        (lambda: _np_recording(100, 30000.3), ["s.nwb", "s.nwb"], "sampling"),
        (lambda: _np_recording(50, 30000.0), ["s.nwb", "s.nwb"], "sample"),
        (
            lambda: _np_recording(100, 30000.0, dtype="int16"),
            ["s.nwb", "s.nwb"],
            "dtype",
        ),
        (lambda: _np_recording(100, 30000.0), ["s.nwb", "b.nwb"], "session"),
    ],
)
def test_assert_shared_group_aggregatable_rejects_bypass(
    make_second, nwb_names, match
):
    from spyglass.spikesorting.v2._shared_artifact_group import (
        assert_shared_group_recordings_aggregatable,
    )
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError

    a = _np_recording(100, 30000.0)
    with pytest.raises(SchemaBypassError, match=match):
        assert_shared_group_recordings_aggregatable(
            [a, make_second()], ["r1", "r2"], nwb_names
        )
