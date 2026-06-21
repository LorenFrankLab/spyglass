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
