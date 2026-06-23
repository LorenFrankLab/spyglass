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


@pytest.fixture
def same_day_group(chronic_2_session_minirec):
    """A same-day ``SessionGroup`` over the two populated members.

    Function-scoped: creates the group, yields the substrate (group key,
    members, recording PKs, preprocessing recipe), and tears the group (plus
    any concat lineage built on it) down for the owner afterward.
    """
    from spyglass.spikesorting.v2.session_group import SessionGroup
    from tests.spikesorting.v2._ingest_helpers import (
        clean_session_groups_for_owner,
    )

    sub = chronic_2_session_minirec
    owner, name = sub["owner"], "sg_concat"
    SessionGroup.create_group(owner, name, sub["same_day_members"])
    yield {
        "group_key": {
            "session_group_owner": owner,
            "session_group_name": name,
        },
        "owner": owner,
        "same_day_members": sub["same_day_members"],
        "next_day_member": sub["next_day_member"],
        "preprocessing_params_name": sub["preprocessing_params_name"],
        "recording_pks": sub["recording_pks"],
    }
    clean_session_groups_for_owner(owner)


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


# ---------- ConcatenatedRecordingSelection.insert_selection ---------------


@pytest.mark.slow
def test_concat_selection_inserts_and_is_idempotent(same_day_group):
    """A selection over members with populated Recordings returns a PK-only
    dict and a repeat call returns the SAME concat_recording_id."""
    import uuid

    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )

    grp = same_day_group
    request = {
        **grp["group_key"],
        "preprocessing_params_name": grp["preprocessing_params_name"],
        "motion_correction_params_name": "none",
    }
    pk = ConcatenatedRecordingSelection.insert_selection(request)
    assert set(pk) == {"concat_recording_id"}
    assert isinstance(pk["concat_recording_id"], uuid.UUID)
    again = ConcatenatedRecordingSelection.insert_selection(dict(request))
    assert again["concat_recording_id"] == pk["concat_recording_id"]


@pytest.mark.slow
def test_concat_selection_missing_recording_raises(chronic_2_session_minirec):
    """A member with no populated Recording for the requested preprocessing
    recipe raises ``MissingRecordingForConcatError`` naming the missing member."""
    from spyglass.spikesorting.v2.exceptions import (
        MissingRecordingForConcatError,
    )
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
        SessionGroup,
    )
    from tests.spikesorting.v2._ingest_helpers import (
        clean_session_groups_for_owner,
    )

    sub = chronic_2_session_minirec
    owner, name = sub["owner"], "sg_missing_rec"
    # The next-day member has no populated Recording; build a multi-day group
    # so the precondition (not the date gate) is what fails on insert.
    members = [sub["same_day_members"][0], sub["next_day_member"]]
    SessionGroup.create_group(owner, name, members, allow_multi_day=True)
    try:
        with pytest.raises(MissingRecordingForConcatError, match="populate"):
            ConcatenatedRecordingSelection.insert_selection(
                {
                    "session_group_owner": owner,
                    "session_group_name": name,
                    "preprocessing_params_name": "default",
                    "motion_correction_params_name": "none",
                }
            )
    finally:
        clean_session_groups_for_owner(owner)


@pytest.mark.slow
def test_concat_selection_distinct_for_distinct_motion_params(same_day_group):
    """Changing only the motion-correction recipe yields a distinct
    concat_recording_id (independent selections)."""
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
        MotionCorrectionParameters,
    )

    MotionCorrectionParameters.insert_default()
    grp = same_day_group
    base = {
        **grp["group_key"],
        "preprocessing_params_name": grp["preprocessing_params_name"],
    }
    none_pk = ConcatenatedRecordingSelection.insert_selection(
        {**base, "motion_correction_params_name": "none"}
    )
    rigid_pk = ConcatenatedRecordingSelection.insert_selection(
        {**base, "motion_correction_params_name": "rigid_fast_default"}
    )
    assert none_pk["concat_recording_id"] != rigid_pk["concat_recording_id"]


# ---------- ConcatenatedRecording.make / get_recording -------------------


def _populate_concat_none(group_key, preprocessing_params_name):
    """Insert + populate a no-motion ConcatenatedRecording; return its PK."""
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
    )

    concat_pk = ConcatenatedRecordingSelection.insert_selection(
        {
            **group_key,
            "preprocessing_params_name": preprocessing_params_name,
            "motion_correction_params_name": "none",
        }
    )
    ConcatenatedRecording.populate(concat_pk, reserve_jobs=False)
    return concat_pk


@pytest.mark.slow
def test_concatenated_recording_make_shape(same_day_group):
    """The materialized concat row carries the NWB pointers, channel count,
    summed duration, cumulative integer boundaries, and reads back as one
    mono-segment recording the length of both members."""
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    grp = same_day_group
    concat_pk = _populate_concat_none(
        grp["group_key"], grp["preprocessing_params_name"]
    )

    row = (ConcatenatedRecording & concat_pk).fetch1()
    assert row["analysis_file_name"]
    assert row["electrical_series_path"]
    assert row["object_id"]
    assert int(row["n_channels"]) == 4

    n0 = Recording().get_recording(grp["recording_pks"][0]).get_num_samples()
    n1 = Recording().get_recording(grp["recording_pks"][1]).get_num_samples()
    fs = float(row["sampling_frequency"])
    assert row["total_duration_s"] == pytest.approx((n0 + n1) / fs)

    idxs, ends = (ConcatenatedRecording.MemberBoundary & concat_pk).fetch(
        "member_index", "end_sample", order_by="member_index"
    )
    assert list(idxs) == [0, 1]
    assert [int(e) for e in ends] == [n0, n0 + n1]

    concat_rec = ConcatenatedRecording().get_recording(concat_pk)
    assert concat_rec.get_num_segments() == 1
    assert concat_rec.get_num_samples() == n0 + n1


@pytest.mark.slow
def test_concatenated_recording_make_never_calls_recording_populate(
    same_day_group, monkeypatch
):
    """make() consumes cached Recording artifacts -- it must NOT call
    Recording.populate (a DataJoint anti-pattern)."""
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    def _boom(*args, **kwargs):
        raise AssertionError(
            "ConcatenatedRecording.make must not call Recording.populate"
        )

    monkeypatch.setattr(Recording, "populate", _boom)
    grp = same_day_group
    concat_pk = _populate_concat_none(
        grp["group_key"], grp["preprocessing_params_name"]
    )
    assert ConcatenatedRecording & concat_pk


@pytest.mark.slow
def test_concat_make_uses_selection_row_not_uuid_key(same_day_group):
    """Two concat selections over different member sets populate independently:
    each row's MemberBoundary count and duration reflect its OWN group, proving
    make() restricts by the fetched selection row, not the UUID-only key."""
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        SessionGroup,
    )

    grp = same_day_group
    owner = grp["owner"]
    # Group A is the fixture's two-member group; group B has only member 0.
    SessionGroup.create_group(
        owner, "sg_concat_b", [grp["same_day_members"][0]]
    )
    a_pk = _populate_concat_none(
        grp["group_key"], grp["preprocessing_params_name"]
    )
    b_pk = _populate_concat_none(
        {"session_group_owner": owner, "session_group_name": "sg_concat_b"},
        grp["preprocessing_params_name"],
    )
    assert len(ConcatenatedRecording.MemberBoundary & a_pk) == 2
    assert len(ConcatenatedRecording.MemberBoundary & b_pk) == 1
    a_dur = (ConcatenatedRecording & a_pk).fetch1("total_duration_s")
    b_dur = (ConcatenatedRecording & b_pk).fetch1("total_duration_s")
    assert a_dur > b_dur


@pytest.mark.slow
def test_motion_correction_preset_auto_rejects_multi_day(
    chronic_2_session_minirec,
):
    """A multi-day group with the 'auto' motion preset is rejected at
    materialization: 'auto' is single-day only, no silent DREDge dispatch."""
    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
        MotionCorrectionParameters,
        SessionGroup,
    )
    from tests.spikesorting.v2._ingest_helpers import (
        clean_session_groups_for_owner,
    )

    sub = chronic_2_session_minirec
    owner, name = sub["owner"], "sg_multi_auto"
    MotionCorrectionParameters.insert_default()

    # The 'auto' preset rejection happens at make() time, AFTER the
    # selection-time precondition that every member has a populated Recording,
    # so the next-day member's Recording must be populated first.
    next_day = sub["next_day_member"]
    next_day_rec_pk = RecordingSelection.insert_selection(
        {
            **next_day,
            "preprocessing_params_name": "default",
            "team_name": owner,
        }
    )
    if not (Recording & next_day_rec_pk):
        Recording.populate(next_day_rec_pk, reserve_jobs=False)

    members = [sub["same_day_members"][0], next_day]
    SessionGroup.create_group(owner, name, members, allow_multi_day=True)
    try:
        concat_pk = ConcatenatedRecordingSelection.insert_selection(
            {
                "session_group_owner": owner,
                "session_group_name": name,
                "preprocessing_params_name": "default",
                "motion_correction_params_name": "auto_default",
            }
        )
        # Call make() directly so the raw ValueError surfaces (populate would
        # wrap it). It raises before writing any artifact.
        with pytest.raises(ValueError, match="single-day only"):
            ConcatenatedRecording().make(concat_pk)
        assert not (ConcatenatedRecording & concat_pk)
    finally:
        clean_session_groups_for_owner(owner)
        (Recording & next_day_rec_pk).super_delete(warn=False)
        (RecordingSelection & next_day_rec_pk).super_delete(warn=False)
