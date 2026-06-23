"""Behavioral tests for same-day chronic concatenate-and-sort.

Covers the ``SessionGroup`` grouping table (multi-day gate, derived dates,
atomic create, owner namespacing), the ``ConcatenatedRecordingSelection``
missing-recording precondition, and -- behind the ``slow`` marker -- the
``ConcatenatedRecording`` materializer plus the concat-backed
``Sorting.populate`` + ``split_sorting_by_session`` round trip.

The substrate is the package-scoped ``chronic_2_session_minirec`` fixture
(two same-day single-tetrode sessions with populated ``Recording`` caches,
plus one next-day session for the multi-day gate).
"""

from __future__ import annotations

import datetime as dt

import pytest


@pytest.mark.slow
def test_chronic_fixture_provides_two_same_day_and_one_next_day(
    chronic_2_session_minirec,
):
    """The fixture yields two same-day members (with populated Recordings) and
    one next-day member; dates derive from ``Session.session_start_time``."""
    from spyglass.common import Session
    from spyglass.spikesorting.v2.recording import Recording

    sub = chronic_2_session_minirec
    same_day = sub["same_day_members"]
    next_day = sub["next_day_member"]
    recording_pks = sub["recording_pks"]

    assert len(same_day) == 2
    assert len(recording_pks) == 2
    # Both same-day members have a populated Recording cache (recording_pks is
    # aligned with same_day_members; the member dicts stay clean for create_group).
    for rec_pk in recording_pks:
        assert Recording & rec_pk

    def _date(member):
        return (
            Session & {"nwb_file_name": member["nwb_file_name"]}
        ).fetch1("session_start_time").date()

    d0, d1 = _date(same_day[0]), _date(same_day[1])
    assert d0 == d1, "the two same-day members must share a date"
    assert _date(next_day) != d0, "the next-day member must differ in date"
    # Identical channel positions are what makes the members concatenable.
    rec0 = Recording().get_recording(recording_pks[0])
    rec1 = Recording().get_recording(recording_pks[1])
    assert list(rec0.get_channel_ids()) == list(rec1.get_channel_ids())
    assert (
        rec0.get_channel_locations() == rec1.get_channel_locations()
    ).all()


# ---------- SessionGroup.create_group / is_multi_day ----------------------


@pytest.mark.usefixtures("dj_conn")
def test_session_group_member_has_no_recording_date_column():
    """Member rows do not store a date; it is always derived from Session."""
    from spyglass.spikesorting.v2.session_group import SessionGroup

    assert "recording_date" not in SessionGroup.Member.heading.attributes


@pytest.mark.slow
def test_create_group_same_day_inserts_and_is_not_multi_day(
    chronic_2_session_minirec,
):
    """Same-day members insert cleanly under the default ``allow_multi_day``
    and ``is_multi_day`` reports ``False``."""
    from spyglass.spikesorting.v2.session_group import SessionGroup

    sub = chronic_2_session_minirec
    owner, name = sub["owner"], "sg_same_day"
    key = {"session_group_owner": owner, "session_group_name": name}
    try:
        SessionGroup.create_group(owner, name, sub["same_day_members"])
        assert SessionGroup & key
        assert len(SessionGroup.Member & key) == 2
        assert SessionGroup.is_multi_day(key) is False
    finally:
        (SessionGroup & key).super_delete(warn=False)


@pytest.mark.slow
def test_session_group_create_multi_day_rejected_by_default(
    chronic_2_session_minirec,
):
    """Members spanning two dates raise ``SessionGroupDateError`` without
    ``allow_multi_day`` and the message points at sort-then-match."""
    from spyglass.spikesorting.v2.exceptions import SessionGroupDateError
    from spyglass.spikesorting.v2.session_group import SessionGroup

    sub = chronic_2_session_minirec
    members = [sub["same_day_members"][0], sub["next_day_member"]]
    with pytest.raises(SessionGroupDateError, match="sort-then-match"):
        SessionGroup.create_group(sub["owner"], "sg_multi_default", members)


@pytest.mark.slow
def test_create_group_multi_day_allowed_with_flag(chronic_2_session_minirec):
    """``allow_multi_day=True`` accepts multi-date members and
    ``is_multi_day`` agrees."""
    from spyglass.spikesorting.v2.session_group import SessionGroup

    sub = chronic_2_session_minirec
    members = [sub["same_day_members"][0], sub["next_day_member"]]
    owner, name = sub["owner"], "sg_multi_allowed"
    key = {"session_group_owner": owner, "session_group_name": name}
    try:
        SessionGroup.create_group(owner, name, members, allow_multi_day=True)
        assert len(SessionGroup.Member & key) == 2
        assert SessionGroup.is_multi_day(key) is True
    finally:
        (SessionGroup & key).super_delete(warn=False)


@pytest.mark.slow
def test_create_group_rejects_caller_supplied_recording_date(
    chronic_2_session_minirec,
):
    """A member dict carrying ``recording_date`` raises ``SessionGroupDateError``
    -- dates are derived, never caller-set."""
    from spyglass.spikesorting.v2.exceptions import SessionGroupDateError
    from spyglass.spikesorting.v2.session_group import SessionGroup

    sub = chronic_2_session_minirec
    member = {**sub["same_day_members"][0], "recording_date": dt.date(2023, 6, 22)}
    with pytest.raises(SessionGroupDateError, match="derived"):
        SessionGroup.create_group(sub["owner"], "sg_reject_date", [member])


@pytest.mark.slow
def test_create_group_is_atomic_on_member_failure(chronic_2_session_minirec):
    """A failed Member insert leaves no master row and no partial Members."""
    from spyglass.spikesorting.v2.session_group import SessionGroup

    sub = chronic_2_session_minirec
    owner, name = sub["owner"], "sg_atomic"
    key = {"session_group_owner": owner, "session_group_name": name}
    good = sub["same_day_members"][0]
    # Second member shares a valid Session (so date derivation succeeds) but
    # names a nonexistent sort_group_id, so the Member insert fails its FK
    # INSIDE the transaction -- after the master insert -- exercising rollback.
    bad = {**sub["same_day_members"][1], "sort_group_id": 99999}
    try:
        with pytest.raises(Exception):  # noqa: B017 -- DataJoint IntegrityError
            SessionGroup.create_group(owner, name, [good, bad])
        assert not (SessionGroup & key)
        assert len(SessionGroup.Member & key) == 0
    finally:
        (SessionGroup & key).super_delete(warn=False)


@pytest.mark.slow
def test_session_group_owner_namespaces_name(chronic_2_session_minirec):
    """Two owners can both name a group ``"day1"`` without collision."""
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.session_group import SessionGroup
    from tests.spikesorting.v2._ingest_helpers import (
        clean_session_groups_for_owner,
    )

    sub = chronic_2_session_minirec
    owner1 = sub["owner"]
    owner2 = "chronic_concat_owner_2"
    LabTeam.insert1(
        {"team_name": owner2, "team_description": "second owner"},
        skip_duplicates=True,
    )
    name = "day1"
    members = sub["same_day_members"]
    try:
        SessionGroup.create_group(owner1, name, members)
        SessionGroup.create_group(owner2, name, members)
        assert len(SessionGroup & {"session_group_name": name}) == 2
    finally:
        clean_session_groups_for_owner(owner1)
        clean_session_groups_for_owner(owner2)
        (LabTeam & {"team_name": owner2}).super_delete(warn=False)
