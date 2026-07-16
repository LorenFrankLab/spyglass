"""SessionGroup / ConcatenatedRecording schema-shape invariants.

Behavioral / schema-shape only: ``SessionGroup.Member`` primary-key uniqueness
within a master, and the ``ConcatenatedRecording.total_duration_s`` column
(named differently from ``Recording.duration_s``).
"""

from __future__ import annotations

import pytest


@pytest.mark.usefixtures("dj_conn")
def test_session_group_member_index_unique_within_master():
    """Two members sharing ``(group, member_index)`` collide.

    ``member_index`` is part of the ``Member`` primary key, so a second row
    with the same group + index raises a duplicate-key error regardless of the
    other fields. Built via the FK-checks-off bypass (the upstream Session /
    SortGroupV2 / IntervalList FKs are irrelevant to the PK uniqueness check).
    """
    import datajoint as dj

    from spyglass.spikesorting.v2.session_group import SessionGroup

    owner, group = "a31_team", "a31_group"
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        SessionGroup.insert1(
            {
                "session_group_owner": owner,
                "session_group_name": group,
                "description": "a31 uniqueness probe",
            },
            allow_direct_insert=True,
        )
        member = {
            "session_group_owner": owner,
            "session_group_name": group,
            "member_index": 0,
            "nwb_file_name": "a31_x_.nwb",
            "sort_group_id": 0,
            "interval_list_name": "raw data valid times",
            "team_name": owner,
        }
        SessionGroup.Member.insert1(member, allow_direct_insert=True)
        # Same (group, member_index), different downstream fields -> collision.
        dup = {**member, "sort_group_id": 1}
        with pytest.raises(dj.errors.DuplicateError):
            SessionGroup.Member.insert1(dup, allow_direct_insert=True)
    finally:
        try:
            (
                SessionGroup.Member
                & {
                    "session_group_owner": owner,
                    "session_group_name": group,
                }
            ).delete_quick()
            (
                SessionGroup
                & {
                    "session_group_owner": owner,
                    "session_group_name": group,
                }
            ).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")


@pytest.mark.usefixtures("dj_conn")
def test_concatenated_recording_has_total_duration_s_column():
    """``ConcatenatedRecording`` declares ``total_duration_s``.

    Semantic check via the heading (not a source-string match). Pairs with the
    CHANGELOG entry naming the column-name divergence
    (``total_duration_s`` vs ``Recording.duration_s``).
    """
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    assert "total_duration_s" in ConcatenatedRecording.heading.attributes


@pytest.mark.usefixtures("dj_conn")
def test_concat_selection_stores_member_set_hash():
    """``ConcatenatedRecordingSelection`` records the folded member-set hash as a
    secondary attribute (the hash is also folded into ``concat_recording_id``).
    """
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )

    attrs = ConcatenatedRecordingSelection.heading.attributes
    assert "member_set_hash" in attrs
    assert "member_set_hash" not in ConcatenatedRecordingSelection.primary_key


@pytest.mark.usefixtures("dj_conn")
def test_concat_selection_member_snapshot_part_shape():
    """The frozen ``MemberSnapshot`` part captures each member's logical identity
    plus its ``Recording`` content hash, keyed by ``member_index``.

    Stored as plain columns (not FKs) so the snapshot is frozen -- a later edit
    to ``SessionGroup.Member`` / ``RecordingSelection`` cannot cascade into it.
    """
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )

    snapshot = ConcatenatedRecordingSelection.MemberSnapshot
    attrs = snapshot.heading.attributes
    for column in (
        "member_index",
        "nwb_file_name",
        "sort_group_id",
        "interval_list_name",
        "team_name",
        "recording_id",
        "recording_content_hash",
    ):
        assert column in attrs, column
    assert "member_index" in snapshot.primary_key
    # Frozen snapshot: the only foreign-key parent is its own master.
    assert snapshot.parents() == [
        ConcatenatedRecordingSelection.full_table_name
    ]
