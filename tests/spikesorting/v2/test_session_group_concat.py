"""Behavioral tests for same-day chronic concatenate-and-sort.

Covers ``SessionGroup``, ``ConcatenatedRecordingSelection``,
``ConcatenatedRecording`` and the concat-backed sort and split. Uses
``chronic_2_session_minirec``: two same-day sessions plus one next-day
session.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from pathlib import Path

import pytest

from tests.spikesorting.v2._concat_helpers import select_unmasked_concat
from tests.spikesorting.v2._sorter_stub import plant_sorter

logger = logging.getLogger(__name__)


class _PeakRSS:
    """Context manager: sample process RSS in a background thread; expose peak.

    ``peak_mb`` is the polled high-water RSS during the body and
    ``elapsed_s`` its wall-clock runtime.
    """

    def __init__(self, interval_s: float = 0.05):
        import psutil

        self._proc = psutil.Process()
        self._interval_s = interval_s
        self._peak_bytes = 0
        self._start_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self.elapsed_s = 0.0

    def _sample(self):
        while not self._stop.is_set():
            self._peak_bytes = max(
                self._peak_bytes, self._proc.memory_info().rss
            )
            self._stop.wait(self._interval_s)

    def __enter__(self):
        self._start_bytes = self._peak_bytes = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.elapsed_s = time.perf_counter() - self._t0
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def peak_mb(self) -> float:
        return self._peak_bytes / (1024 * 1024)

    @property
    def growth_mb(self) -> float:
        """Peak RSS above the RSS on entry: what the body itself added."""
        return (self._peak_bytes - self._start_bytes) / (1024 * 1024)


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
    # recording_pks is aligned with same_day_members.
    for rec_pk in recording_pks:
        assert Recording & rec_pk

    def _date(member):
        return (
            (Session & {"nwb_file_name": member["nwb_file_name"]})
            .fetch1("session_start_time")
            .date()
        )

    d0, d1 = _date(same_day[0]), _date(same_day[1])
    assert d0 == d1, "the two same-day members must share a date"
    assert _date(next_day) != d0, "the next-day member must differ in date"
    # Identical channel positions are what makes the members concatenable.
    rec0 = Recording().get_recording(recording_pks[0])
    rec1 = Recording().get_recording(recording_pks[1])
    assert list(rec0.get_channel_ids()) == list(rec1.get_channel_ids())
    assert (rec0.get_channel_locations() == rec1.get_channel_locations()).all()


# ---------- SessionGroup.create_group / is_multi_day ----------------------


@pytest.fixture
def same_day_group(chronic_2_session_minirec):
    """A same-day ``SessionGroup`` over the two populated members.

    Yields the group key, members, recording PKs and preprocessing recipe;
    tears down the group and its concat lineage.
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


@pytest.mark.usefixtures("dj_conn")
def test_create_group_rejects_empty_members():
    """An empty members list is rejected at create time."""
    from spyglass.spikesorting.v2.session_group import SessionGroup

    with pytest.raises(ValueError, match="empty"):
        SessionGroup.create_group("any_owner", "empty_group", [])


@pytest.mark.slow
def test_create_group_rejects_duplicate_logical_members(
    chronic_2_session_minirec,
):
    """Two members with the same (nwb, sort_group, interval, team) are
    rejected; the Member PK is member_index only."""
    from spyglass.spikesorting.v2.session_group import SessionGroup

    sub = chronic_2_session_minirec
    member = sub["same_day_members"][0]
    with pytest.raises(ValueError, match="duplicate logical member"):
        SessionGroup.create_group(
            sub["owner"], "sg_dup", [member, dict(member)]
        )


@pytest.mark.slow
def test_assert_concat_preflight_guards_members_and_auto_curate(
    same_day_group,
):
    """Concat preflight passes a fully-ingested group and fails fast on a
    missing auto-curation prerequisite.
    """
    from spyglass.spikesorting.v2._orchestration.preflight import (
        assert_concat_preflight,
    )
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.spikesorting.v2.pipeline import _PIPELINE_PRESETS

    gk = same_day_group["group_key"]
    owner = gk["session_group_owner"]
    name = gk["session_group_name"]
    bundle = _PIPELINE_PRESETS["franklab_concat_hippocampus_30khz_ms5_2026_09"]

    # A valid, fully-ingested group passes every per-member prerequisite.
    assert assert_concat_preflight(owner, name, bundle) == []

    bad = bundle.model_copy(update={"metric_params_name": "does_not_exist_xyz"})
    with pytest.raises(PreflightError, match="QualityMetricParameters"):
        assert_concat_preflight(owner, name, bad, auto_curate=True)


@pytest.mark.slow
def test_assert_concat_preflight_rejects_member_with_coincident_contacts(
    same_day_group, monkeypatch
):
    """A member whose contacts coincide fails concat preflight by name.

    The check runs before any member recording is built.
    """
    import spyglass.spikesorting.v2._orchestration.preflight_checks as checks
    from spyglass.spikesorting.v2._orchestration.preflight import (
        assert_concat_preflight,
    )
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.spikesorting.v2.pipeline import _PIPELINE_PRESETS
    from spyglass.spikesorting.v2.session_group import SessionGroup

    gk = same_day_group["group_key"]
    last = (SessionGroup.Member & gk).fetch(
        "member_index",
        "nwb_file_name",
        "sort_group_id",
        order_by="member_index DESC",
        limit=1,
        as_dict=True,
    )[0]
    problem = "contacts coincide in every plane"
    monkeypatch.setattr(
        checks,
        "sort_group_geometry_problem",
        lambda nwb_file_name, sort_group_id: (
            problem
            if (nwb_file_name, int(sort_group_id))
            == (last["nwb_file_name"], int(last["sort_group_id"]))
            else None
        ),
    )
    bundle = _PIPELINE_PRESETS["franklab_concat_hippocampus_30khz_ms5_2026_09"]

    with pytest.raises(PreflightError) as raised:
        assert_concat_preflight(
            gk["session_group_owner"], gk["session_group_name"], bundle
        )
    assert f"concat member {last['member_index']}" in str(raised.value)
    assert problem in str(raised.value)


@pytest.mark.usefixtures("dj_conn")
def test_create_group_rejects_missing_member_key():
    """A member dict missing a required key raises a typed error, not KeyError."""
    from spyglass.spikesorting.v2.exceptions import SessionGroupInputError
    from spyglass.spikesorting.v2.session_group import SessionGroup

    with pytest.raises(SessionGroupInputError, match="missing required key"):
        SessionGroup.create_group(
            "any_owner", "sg_missing_key", [{"nwb_file_name": "x.nwb"}]
        )


@pytest.mark.usefixtures("dj_conn")
def test_create_group_rejects_nonexistent_session():
    """A member referencing a non-ingested session raises a typed error, not a
    bare fetch1 failure."""
    from spyglass.spikesorting.v2.exceptions import SessionGroupInputError
    from spyglass.spikesorting.v2.session_group import SessionGroup

    with pytest.raises(SessionGroupInputError, match="not an ingested Session"):
        SessionGroup.create_group(
            "any_owner",
            "sg_no_session",
            [
                {
                    "nwb_file_name": "not_ingested_.nwb",
                    "sort_group_id": 0,
                    "interval_list_name": "raw data valid times",
                }
            ],
        )


@pytest.mark.slow
def test_create_group_rejects_missing_foreign_key(chronic_2_session_minirec):
    """A member with a valid session but a non-existent SortGroupV2 raises a
    typed error, not a raw DataJoint integrity error."""
    from spyglass.spikesorting.v2.exceptions import SessionGroupInputError
    from spyglass.spikesorting.v2.session_group import SessionGroup

    sub = chronic_2_session_minirec
    bad_member = {**sub["same_day_members"][0], "sort_group_id": 99999}
    with pytest.raises(SessionGroupInputError, match="does not exist"):
        SessionGroup.create_group(sub["owner"], "sg_bad_fk", [bad_member])


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
    """A member dict carrying ``recording_date`` raises
    ``SessionGroupDateError``; dates are derived."""
    from spyglass.spikesorting.v2.exceptions import SessionGroupDateError
    from spyglass.spikesorting.v2.session_group import SessionGroup

    sub = chronic_2_session_minirec
    member = {
        **sub["same_day_members"][0],
        "recording_date": dt.date(2023, 6, 22),
    }
    with pytest.raises(SessionGroupDateError, match="derived"):
        SessionGroup.create_group(sub["owner"], "sg_reject_date", [member])


@pytest.mark.slow
def test_create_group_is_atomic_on_member_failure(
    chronic_2_session_minirec, monkeypatch
):
    """A failure after the first Member write rolls back master and parts."""
    import datajoint as dj

    from spyglass.spikesorting.v2.exceptions import SessionGroupInputError
    from spyglass.spikesorting.v2.session_group import SessionGroup

    sub = chronic_2_session_minirec
    owner, name = sub["owner"], "sg_atomic"
    key = {"session_group_owner": owner, "session_group_name": name}
    real_insert = SessionGroup.Member.insert
    reached = []

    def fail_after_first_member(rows, **kwargs):
        assert SessionGroup.connection.in_transaction
        assert SessionGroup & key, "master must be visible before member insert"
        real_insert(rows[:1], **kwargs)
        assert len(SessionGroup.Member & key) == 1
        reached.append("first member stored")
        raise dj.errors.IntegrityError(
            "injected failure after first Member write"
        )

    monkeypatch.setattr(
        SessionGroup.Member, "insert", staticmethod(fail_after_first_member)
    )
    try:
        with pytest.raises(SessionGroupInputError) as error:
            SessionGroup.create_group(owner, name, sub["same_day_members"])
        assert isinstance(error.value.__cause__, dj.errors.IntegrityError)
        assert reached == ["first member stored"]
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

    grp = same_day_group
    request = {
        **grp["group_key"],
        "preprocessing_params_name": grp["preprocessing_params_name"],
    }
    pk = select_unmasked_concat(request)
    assert set(pk) == {"concat_recording_id"}
    assert isinstance(pk["concat_recording_id"], uuid.UUID)
    again = select_unmasked_concat(dict(request))
    assert again["concat_recording_id"] == pk["concat_recording_id"]


@pytest.mark.slow
def test_concat_id_folds_member_set(same_day_group):
    """``concat_recording_id`` is the id of (identity + member_set_hash), not
    of the identity alone, and the stored hash matches the snapshot rows."""
    from spyglass.spikesorting.v2._recording.concat import member_set_hash
    from spyglass.spikesorting.v2._core.selection_identity import (
        deterministic_id,
    )
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )

    grp = same_day_group
    pk = select_unmasked_concat(
        {
            **grp["group_key"],
            "preprocessing_params_name": grp["preprocessing_params_name"],
        }
    )
    row = (ConcatenatedRecordingSelection & pk).fetch1()
    identity = {
        f: row[f] for f in ConcatenatedRecordingSelection._IDENTITY_FIELDS
    }
    set_hash = row["member_set_hash"]

    folded = deterministic_id(
        "concat_recording", {**identity, "member_set_hash": set_hash}
    )
    unfolded = deterministic_id("concat_recording", identity)
    assert pk["concat_recording_id"] == folded
    assert pk["concat_recording_id"] != unfolded

    snap = (ConcatenatedRecordingSelection.MemberSnapshot & pk).fetch(
        as_dict=True, order_by="member_index"
    )
    assert member_set_hash(snap) == set_hash


@pytest.mark.slow
def test_concat_member_snapshot_freezes_recording_identity(same_day_group):
    """``MemberSnapshot`` freezes each member's identity, ``recording_id`` and
    ``Recording.content_hash`` for later verification."""
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )

    grp = same_day_group
    pk = select_unmasked_concat(
        {
            **grp["group_key"],
            "preprocessing_params_name": grp["preprocessing_params_name"],
        }
    )
    snap = (ConcatenatedRecordingSelection.MemberSnapshot & pk).fetch(
        as_dict=True, order_by="member_index"
    )
    assert [r["member_index"] for r in snap] == [0, 1]
    for row, rec_pk in zip(snap, grp["recording_pks"]):
        assert str(row["recording_id"]) == str(rec_pk["recording_id"])
        assert row["recording_content_hash"] == (Recording & rec_pk).fetch1(
            "content_hash"
        )
        assert row["team_name"] == grp["owner"]


@pytest.mark.slow
def test_concat_id_changes_with_member_set(same_day_group):
    """Same params over a different member set mint a different
    ``concat_recording_id`` and ``member_set_hash``."""
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
        SessionGroup,
    )

    grp = same_day_group
    both = select_unmasked_concat(
        {
            **grp["group_key"],
            "preprocessing_params_name": grp["preprocessing_params_name"],
        }
    )
    # Same owner and params, only the first member.
    owner = grp["owner"]
    SessionGroup.create_group(
        owner, "sg_concat_subset", [grp["same_day_members"][0]]
    )
    subset = select_unmasked_concat(
        {
            "session_group_owner": owner,
            "session_group_name": "sg_concat_subset",
            "preprocessing_params_name": grp["preprocessing_params_name"],
        }
    )
    assert subset["concat_recording_id"] != both["concat_recording_id"]
    assert (ConcatenatedRecordingSelection & subset).fetch1(
        "member_set_hash"
    ) != (ConcatenatedRecordingSelection & both).fetch1("member_set_hash")


@pytest.mark.slow
def test_concat_member_edit_remints_id_and_freezes_old_snapshot(
    same_day_group,
):
    """Editing the live member set re-mints the concat id on re-selection and
    leaves the original concat's MemberSnapshot unchanged."""
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
        SessionGroup,
    )

    grp = same_day_group
    request = {
        **grp["group_key"],
        "preprocessing_params_name": grp["preprocessing_params_name"],
    }
    original = select_unmasked_concat(dict(request))
    original_snapshot = (
        ConcatenatedRecordingSelection.MemberSnapshot & original
    ).fetch(as_dict=True, order_by="member_index")
    assert [r["member_index"] for r in original_snapshot] == [0, 1]

    # Drop the second member under the same group_key.
    (
        SessionGroup.Member & {**grp["group_key"], "member_index": 1}
    ).delete_quick()

    edited = select_unmasked_concat(dict(request))
    assert edited["concat_recording_id"] != original["concat_recording_id"]

    still_frozen = (
        ConcatenatedRecordingSelection.MemberSnapshot & original
    ).fetch(as_dict=True, order_by="member_index")
    assert [r["member_index"] for r in still_frozen] == [0, 1]
    assert {str(r["recording_id"]) for r in still_frozen} == {
        str(r["recording_id"]) for r in original_snapshot
    }


@pytest.mark.slow
def test_concat_selection_missing_recording_raises(chronic_2_session_minirec):
    """A member with no populated Recording for the requested preprocessing
    recipe raises ``MissingRecordingForConcatError`` naming the missing member.
    """
    from spyglass.spikesorting.v2.exceptions import (
        MissingRecordingForConcatError,
    )
    from spyglass.spikesorting.v2.session_group import (
        SessionGroup,
    )
    from tests.spikesorting.v2._ingest_helpers import (
        clean_session_groups_for_owner,
    )

    sub = chronic_2_session_minirec
    owner, name = sub["owner"], "sg_missing_rec"
    # Multi-day, so the missing recording (not the date gate) fails.
    members = [sub["same_day_members"][0], sub["next_day_member"]]
    SessionGroup.create_group(owner, name, members, allow_multi_day=True)
    try:
        with pytest.raises(MissingRecordingForConcatError, match="populate"):
            select_unmasked_concat(
                {
                    "session_group_owner": owner,
                    "session_group_name": name,
                    "preprocessing_params_name": "default",
                }
            )
    finally:
        clean_session_groups_for_owner(owner)


@pytest.mark.slow
def test_concat_selection_needs_the_member_recording_for_current_inputs(
    chronic_2_session_minirec,
):
    """A member recording built before a sort-group change is not used.

    After a reference change the member is reported missing, both before
    and after a second ``RecordingSelection`` row is added for it.
    """
    from spyglass.spikesorting.v2.exceptions import (
        MissingRecordingForConcatError,
    )
    from spyglass.spikesorting.v2.recording import (
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.session_group import SessionGroup
    from tests.spikesorting.v2._ingest_helpers import (
        clean_session_groups_for_owner,
    )

    sub = chronic_2_session_minirec
    owner, name = sub["owner"], "sg_changed_member_inputs"
    members = sub["same_day_members"]
    SessionGroup.create_group(owner, name, members)
    sort_group_key = {
        "nwb_file_name": members[0]["nwb_file_name"],
        "sort_group_id": members[0]["sort_group_id"],
    }
    original_mode = (SortGroupV2 & sort_group_key).fetch1("reference_mode")
    changed_mode = "none" if original_mode != "none" else "global_median"
    request = {
        "session_group_owner": owner,
        "session_group_name": name,
        "preprocessing_params_name": "default",
    }
    new_selection = None
    try:
        SortGroupV2().update1(
            {**sort_group_key, "reference_mode": changed_mode}
        )
        with pytest.raises(MissingRecordingForConcatError, match="populate"):
            select_unmasked_concat(request)

        new_selection = RecordingSelection.insert_selection(
            {
                **members[0],
                "preprocessing_params_name": "default",
                "team_name": owner,
            }
        )
        assert new_selection != sub["recording_pks"][0]
        with pytest.raises(MissingRecordingForConcatError, match="populate"):
            select_unmasked_concat(request)
    finally:
        clean_session_groups_for_owner(owner)
        if new_selection is not None:
            (RecordingSelection & new_selection).super_delete(warn=False)
        SortGroupV2().update1(
            {**sort_group_key, "reference_mode": original_mode}
        )


@pytest.mark.usefixtures("dj_conn")
def test_concat_selection_rejects_unknown_fields():
    """A concat selection refuses fields outside its identity, including
    ``motion_correction_params_name``, before reading any member."""
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
        SessionGroup,
    )

    group_key = {
        "session_group_owner": "no_such_owner",
        "session_group_name": "no_such_group",
    }
    assert not (SessionGroup & group_key)
    request = {**group_key, "preprocessing_params_name": "default"}
    with pytest.raises(
        ValueError,
        match=r"unknown field\(s\) \['motion_correction_params_name'\]",
    ):
        ConcatenatedRecordingSelection.insert_selection(
            {**request, "motion_correction_params_name": "auto_default"},
            artifact_detection_ids={},
        )
    with pytest.raises(ValueError, match=r"unknown field\(s\) \['typo'\]"):
        ConcatenatedRecordingSelection.insert_selection(
            {**request, "typo": 1}, artifact_detection_ids={}
        )
    # The minted id is accepted and ignored.
    with pytest.raises(ValueError, match="has no members"):
        ConcatenatedRecordingSelection.insert_selection(
            {**request, "concat_recording_id": None},
            artifact_detection_ids={},
        )


@pytest.mark.usefixtures("dj_conn")
def test_concat_rejects_mismatched_electrode_space(monkeypatch):
    """A SessionGroup whose members map to different physical electrode spaces
    (different electrode ids or brain regions) is rejected.

    The concat is read in the anchor member's electrode frame. The
    signature source is stubbed here.
    """
    from spyglass.spikesorting.v2 import session_group as sg

    members = [
        {"member_index": 0, "sort_group_id": 0, "nwb_file_name": "a.nwb"},
        {"member_index": 1, "sort_group_id": 0, "nwb_file_name": "b.nwb"},
    ]
    signatures = {
        0: ((0, "hpc"), (1, "hpc")),
        1: ((0, "cortex"), (1, "cortex")),  # different brain region
    }
    monkeypatch.setattr(
        sg,
        "_member_electrode_signature",
        lambda member: signatures[member["member_index"]],
    )
    with pytest.raises(ValueError, match="electrode"):
        sg.assert_members_share_electrode_space(members)

    signatures[1] = signatures[0]
    sg.assert_members_share_electrode_space(members)


@pytest.mark.slow
def test_concat_make_fetch_rejects_mismatched_electrode_space(
    same_day_group, monkeypatch
):
    """``ConcatenatedRecording.make_fetch`` re-checks electrode-space
    compatibility, catching drift after selection or a bypassing insert."""
    from spyglass.spikesorting.v2 import session_group as sg
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
    )

    key = select_unmasked_concat(
        {
            **same_day_group["group_key"],
            "preprocessing_params_name": same_day_group[
                "preprocessing_params_name"
            ],
        }
    )
    # Post-selection drift.
    signatures = {0: ((0, "hpc"),), 1: ((0, "cortex"),)}
    monkeypatch.setattr(
        sg,
        "_member_electrode_signature",
        lambda member: signatures[member["member_index"]],
    )
    with pytest.raises(ValueError, match="electrode space"):
        ConcatenatedRecording().make_fetch(key)


def test_concat_with_artifact_id_revalidated_at_compute(dj_conn):
    """A bypass-inserted concat sort carrying an ``ArtifactDetectionSource`` row
    raises ``SchemaBypassError`` at ``make_fetch``.

    Member masks belong to the concat source, so a sorting-stage artifact id
    cannot describe them.
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    SorterParameters.insert_default()
    sid = uuid.uuid4()
    art_id = uuid.uuid4()
    art_merge_id = uuid.uuid4()
    concat_id = uuid.uuid4()
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        from spyglass.spikesorting.v2.artifact import (
            RecordingArtifactDetection,
        )
        from spyglass.spikesorting.v2.artifact_output import (
            ArtifactDetectionOutput,
        )

        SortingSelection.insert1(
            {
                "sorting_id": sid,
                "sorter": "mountainsort5",
                "sorter_params_name": "franklab_30khz_ms5_2026_06",
            },
            allow_direct_insert=True,
        )
        SortingSelection.ConcatenatedRecordingSource.insert1(
            {"sorting_id": sid, "concat_recording_id": concat_id}
        )
        # Register a bypass recording artifact in the merge and point the
        # ArtifactDetectionSource part at it.
        RecordingArtifactDetection().insert(
            [{"artifact_detection_id": art_id}], allow_direct_insert=True
        )
        dj.Table.insert(
            ArtifactDetectionOutput(),
            [{"merge_id": art_merge_id, "source": "RecordingSource"}],
        )
        dj.Table.insert(
            ArtifactDetectionOutput.RecordingSource(),
            [{"merge_id": art_merge_id, "artifact_detection_id": art_id}],
        )
        SortingSelection.ArtifactDetectionSource.insert1(
            {"sorting_id": sid, "artifact_detection_merge_id": art_merge_id}
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        with pytest.raises(SchemaBypassError, match="concat source"):
            Sorting().make_fetch({"sorting_id": sid})
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (
                SortingSelection.ArtifactDetectionSource & {"sorting_id": sid}
            ).delete_quick()
            (
                SortingSelection.ConcatenatedRecordingSource
                & {"sorting_id": sid}
            ).delete_quick()
            (SortingSelection & {"sorting_id": sid}).delete_quick()
            (
                ArtifactDetectionOutput.RecordingSource
                & {"merge_id": art_merge_id}
            ).delete_quick()
            (
                ArtifactDetectionOutput & {"merge_id": art_merge_id}
            ).delete_quick()
            (
                RecordingArtifactDetection & {"artifact_detection_id": art_id}
            ).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")


# ---------- ConcatenatedRecording.make / get_recording -------------------


def _concat_selection(group_key, preprocessing_params_name):
    """Insert an unmasked ConcatenatedRecordingSelection; return its PK."""
    return select_unmasked_concat(
        {
            **group_key,
            "preprocessing_params_name": preprocessing_params_name,
        }
    )


def _populate_concat(group_key, preprocessing_params_name):
    """Insert + populate a ConcatenatedRecording; return its PK."""
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    concat_pk = _concat_selection(group_key, preprocessing_params_name)
    ConcatenatedRecording.populate(concat_pk, reserve_jobs=False)
    return concat_pk


def _member_sample_counts(grp):
    """Per-member ``Recording`` sample counts, aligned with the group's members."""
    from spyglass.spikesorting.v2.recording import Recording

    return [
        Recording().get_recording(rec_pk).get_num_samples()
        for rec_pk in grp["recording_pks"]
    ]


def _ensure_clusterless_sorter_params():
    """Insert the smoke clusterless sorter params row (idempotent)."""
    from spyglass.spikesorting.v2.sorting import SorterParameters
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAM_NAME,
        SMOKE_CLUSTERLESS_PARAMS,
    )

    SorterParameters.insert1(
        {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": SMOKE_CLUSTERLESS_PARAM_NAME,
            "params": SMOKE_CLUSTERLESS_PARAMS,
        },
        skip_duplicates=True,
        allow_duplicate_params=True,
    )


def _member_snapshot(grp):
    """insert_selection over a group + return its frozen MemberSnapshot rows."""
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )

    sel = select_unmasked_concat(
        {
            **grp["group_key"],
            "preprocessing_params_name": grp["preprocessing_params_name"],
        }
    )
    return (ConcatenatedRecordingSelection.MemberSnapshot & sel).fetch(
        as_dict=True, order_by="member_index"
    )


@pytest.mark.slow
def test_load_member_recordings_returns_aligned_counts_and_indices(
    same_day_group,
):
    """``_load_member_recordings`` returns recordings, sample counts and
    indices aligned in member_index order."""
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording
    from spyglass.spikesorting.v2._recording.concat_fetch import (
        resolve_snapshot_recordings,
    )

    grp = same_day_group
    member_plan = resolve_snapshot_recordings(_member_snapshot(grp))
    member_traces = tuple(
        Recording().resolve_stored_traces(plan["recording_pk"])
        for plan in member_plan
    )
    recordings, sample_counts, member_indices = (
        ConcatenatedRecording._load_member_recordings(
            member_plan, member_traces
        )
    )
    expected = _member_sample_counts(grp)
    assert member_indices == [0, 1]
    assert sample_counts == expected
    assert [int(r.get_num_samples()) for r in recordings] == expected


@pytest.mark.slow
def test_resolve_snapshot_recordings_raises_on_missing_recording(
    same_day_group,
):
    """A frozen member whose ``Recording`` is gone raises
    ``MissingRecordingForConcatError``; dropping it would shift later
    boundaries."""
    from spyglass.spikesorting.v2.exceptions import (
        MissingRecordingForConcatError,
    )
    from spyglass.spikesorting.v2._recording.concat_fetch import (
        resolve_snapshot_recordings,
    )

    snapshot = _member_snapshot(same_day_group)
    # Point the first frozen member at a recording_id with no Recording row.
    snapshot[0] = {
        **snapshot[0],
        "recording_id": "00000000-0000-0000-0000-000000000000",
    }
    with pytest.raises(MissingRecordingForConcatError, match="populate"):
        resolve_snapshot_recordings(snapshot)


@pytest.mark.slow
def test_resolve_snapshot_recordings_raises_on_member_content_drift(
    same_day_group,
):
    """A member whose live ``Recording.content_hash`` differs from the
    snapshot raises ``ConcatMemberDriftError``."""
    from spyglass.spikesorting.v2.exceptions import ConcatMemberDriftError
    from spyglass.spikesorting.v2._recording.concat_fetch import (
        resolve_snapshot_recordings,
    )

    snapshot = _member_snapshot(same_day_group)
    # Freeze a content hash that diverges from the live Recording's.
    snapshot[0] = {**snapshot[0], "recording_content_hash": "f" * 64}
    with pytest.raises(ConcatMemberDriftError, match="content"):
        resolve_snapshot_recordings(snapshot)


@pytest.mark.slow
def test_concat_offset_epochs_populate_read_and_rebuild(
    chronic_2_session_minirec,
):
    """Real selected epochs with unequal inferred rates survive the full cache path."""
    import numpy as np
    import spikeinterface as si

    from spyglass.common import AnalysisNwbfile, IntervalList
    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
        SessionGroup,
    )

    fixture = chronic_2_session_minirec
    owner = fixture["owner"]
    source = fixture["same_day_members"][0]
    nwb_file_name = source["nwb_file_name"]
    group_key = {
        "session_group_owner": owner,
        "session_group_name": "concat_timestamp_roundoff",
    }
    intervals = [
        ("concat roundoff first epoch", [0.0, 1.5], 0),
        ("concat roundoff second epoch", [2.0, 3.5], 60_000),
    ]
    recording_keys, interval_keys, members = [], [], []
    try:
        for name, times, _ in intervals:
            interval_key = {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": name,
            }
            IntervalList.insert1(
                {
                    **interval_key,
                    "valid_times": np.asarray([times]),
                    "pipeline": "concat_roundoff_test",
                }
            )
            interval_keys.append(interval_key)
            member = {
                **source,
                "interval_list_name": name,
                "team_name": owner,
            }
            recording_key = RecordingSelection.insert_selection(
                {
                    **member,
                    "preprocessing_params_name": fixture[
                        "preprocessing_params_name"
                    ],
                }
            )
            recording_keys.append(recording_key)
            Recording.populate(recording_key, reserve_jobs=False)
            members.append(member)

        recordings = [Recording().get_recording(key) for key in recording_keys]
        rates = [recording.get_sampling_frequency() for recording in recordings]
        assert (
            abs(rates[0] - rates[1]) > 1e-9
        ), "The persisted epoch clocks must exercise inferred-rate disagreement"
        # The source fixture is rate-based at exactly 30 kHz. Both inclusive
        # 1.5 s selections contain 45,001 frames, on these original clocks.
        expected_member_times = [
            np.arange(first, first + 45_001, dtype=np.float64) / 30_000
            for _, _, first in intervals
        ]
        for recording, times in zip(recordings, expected_member_times):
            assert recording.get_num_samples() == 45_001
            np.testing.assert_array_equal(recording.get_times(), times)
        expected_traces = np.concatenate(
            [recording.get_traces() for recording in recordings]
        )
        expected_uv = np.concatenate(
            [
                recording.get_traces(return_in_uV=True)
                for recording in recordings
            ]
        )

        SessionGroup.create_group(
            owner, group_key["session_group_name"], members
        )
        concat_key = select_unmasked_concat(
            {
                **group_key,
                "preprocessing_params_name": fixture[
                    "preprocessing_params_name"
                ],
            }
        )
        ConcatenatedRecording.populate(concat_key, reserve_jobs=False)
        row = (ConcatenatedRecording & concat_key).fetch1()
        assert int(row["n_samples"]) == 90_002
        np.testing.assert_array_equal(
            row["continuity_spans"], [[0, 45_001], [45_001, 90_002]]
        )
        np.testing.assert_array_equal(row["continuity_start_s"], [0, 2])
        np.testing.assert_array_equal(row["continuity_end_s"], [1.5, 3.5])
        indices, boundaries = (
            ConcatenatedRecording.MemberBoundary & concat_key
        ).fetch("member_index", "end_sample", order_by="member_index")
        np.testing.assert_array_equal(indices, [0, 1])
        np.testing.assert_array_equal(boundaries, [45_001, 90_002])
        snapshot = (
            ConcatenatedRecordingSelection.MemberSnapshot & concat_key
        ).fetch("recording_id", order_by="member_index")
        assert [str(recording_id) for recording_id in snapshot] == [
            str(key["recording_id"]) for key in recording_keys
        ]

        def assert_cache(recording):
            assert recording.get_num_segments() == 1
            assert recording.get_num_samples() == 90_002
            np.testing.assert_array_equal(
                recording.get_traces(), expected_traces
            )
            np.testing.assert_array_equal(
                recording.get_traces(return_in_uV=True), expected_uv
            )
            np.testing.assert_array_equal(
                recording.channel_ids, recordings[0].channel_ids
            )
            np.testing.assert_array_equal(
                recording.get_channel_locations(),
                recordings[0].get_channel_locations(),
            )
            np.testing.assert_allclose(
                recording.get_times(),
                np.arange(90_002) / 30_000,
                rtol=0,
                atol=0.01 / 30_000,
            )

        assert_cache(ConcatenatedRecording().get_recording(concat_key))
        # Exercise the persisted member mapping with boundary and final-frame
        # spikes. Expected member keys and local frames come from the request.
        sorting = si.NumpySorting.from_unit_dict(
            [
                {
                    7: np.asarray([0, 45_000, 45_002, 90_001]),
                    9: np.asarray([45_001]),
                }
            ],
            sampling_frequency=30_000,
        )
        split = ConcatenatedRecording().split_sorting_by_session(
            sorting, concat_key
        )
        expected_keys = [
            (
                nwb_file_name,
                member["sort_group_id"],
                member["interval_list_name"],
                owner,
            )
            for member in members
        ]
        assert set(split) == set(expected_keys)
        for key, frames, boundary_frames in zip(
            expected_keys, ([0, 45_000], [1, 45_000]), ([], [0])
        ):
            np.testing.assert_array_equal(
                split[key].get_unit_spike_train(7), frames
            )
            np.testing.assert_array_equal(
                split[key].get_unit_spike_train(9), boundary_frames
            )

        cache_path = Path(
            AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        )
        cache_path.unlink()
        assert_cache(ConcatenatedRecording().get_recording(concat_key))
        assert cache_path.exists()
        assert (ConcatenatedRecording & concat_key).fetch1(
            "content_hash"
        ) == row["content_hash"]
        for key, times in zip(recording_keys, expected_member_times):
            np.testing.assert_array_equal(
                Recording().get_recording(key).get_times(), times
            )
    finally:
        # Only this test's group, selections, and intervals are removed.
        for key in (ConcatenatedRecordingSelection & group_key).fetch(
            "KEY", as_dict=True
        ):
            (ConcatenatedRecording & key).super_delete(warn=False)
            (ConcatenatedRecordingSelection & key).super_delete(warn=False)
        (SessionGroup & group_key).super_delete(warn=False)
        for key in recording_keys:
            (Recording & key).super_delete(warn=False)
            (RecordingSelection & key).super_delete(warn=False)
        for key in interval_keys:
            (IntervalList & key).super_delete(warn=False)


@pytest.mark.slow
def test_concatenated_recording_make_shape(same_day_group):
    """The materialized concat row carries the NWB pointers, channel count,
    summed duration, cumulative integer boundaries, and reads back as one
    mono-segment recording the length of both members."""
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    grp = same_day_group
    concat_pk = _populate_concat(
        grp["group_key"], grp["preprocessing_params_name"]
    )

    row = (ConcatenatedRecording & concat_pk).fetch1()
    assert row["analysis_file_name"]
    assert row["electrical_series_path"]
    assert row["object_id"]
    assert int(row["n_channels"]) == 4

    n0, n1 = _member_sample_counts(grp)
    fs = float(row["sampling_frequency"])
    assert row["total_duration_s"] == pytest.approx((n0 + n1) / fs)
    # Exact integer count; the split-back guard relies on it.
    assert int(row["n_samples"]) == n0 + n1

    idxs, ends = (ConcatenatedRecording.MemberBoundary & concat_pk).fetch(
        "member_index", "end_sample", order_by="member_index"
    )
    assert list(idxs) == [0, 1]
    assert [int(e) for e in ends] == [n0, n0 + n1]

    concat_rec = ConcatenatedRecording().get_recording(concat_pk)
    assert concat_rec.get_num_segments() == 1
    assert concat_rec.get_num_samples() == n0 + n1


@pytest.mark.slow
def test_concat_get_recording_rebuilds_on_missing(same_day_group):
    """``get_recording`` rebuilds a missing concat cache file, reconciles the
    ``~external`` checksum and returns identical content."""
    import numpy as np

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    grp = same_day_group
    concat_pk = _populate_concat(
        grp["group_key"], grp["preprocessing_params_name"]
    )
    row = (ConcatenatedRecording & concat_pk).fetch1()
    abs_path = AnalysisNwbfile.get_abs_path(
        row["analysis_file_name"], from_schema=True
    )
    assert Path(abs_path).exists()
    before = ConcatenatedRecording().get_recording(concat_pk).get_traces()

    Path(abs_path).unlink()
    rec = ConcatenatedRecording().get_recording(concat_pk)
    assert rec.get_num_samples() == sum(_member_sample_counts(grp))
    assert Path(
        AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
    ).exists(), "the canonical concat file must be back on disk after rebuild"
    np.testing.assert_array_equal(rec.get_traces(), before)


@pytest.mark.slow
def test_concat_rebuild_refuses_on_content_drift(same_day_group, monkeypatch):
    """A concat rebuild whose fingerprint differs from ``content_hash``
    raises ``RecordingContentDriftError`` and writes nothing.
    """
    import shutil

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._recording import fingerprint as fp_mod
    from spyglass.spikesorting.v2.exceptions import (
        RecordingContentDriftError,
    )
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    grp = same_day_group
    concat_pk = _populate_concat(
        grp["group_key"], grp["preprocessing_params_name"]
    )
    row = (ConcatenatedRecording & concat_pk).fetch1()
    abs_path = AnalysisNwbfile.get_abs_path(
        row["analysis_file_name"], from_schema=True
    )
    backup = str(abs_path) + ".drift.bak"
    shutil.copy2(abs_path, backup)

    real_fp = fp_mod.recording_content_fingerprint

    def _drifted_fp(*args, **kwargs):
        components = dict(real_fp(*args, **kwargs))
        components["traces"] = "drifted-" + components["traces"]
        return components

    monkeypatch.setattr(fp_mod, "recording_content_fingerprint", _drifted_fp)
    try:
        Path(abs_path).unlink()
        with pytest.raises(RecordingContentDriftError, match="content_hash"):
            ConcatenatedRecording()._rebuild_nwb_artifact(concat_pk)
        assert not Path(abs_path).exists(), (
            "the canonical concat slot must stay missing on a drift refusal -- "
            "drifted bytes must never be installed"
        )
    finally:
        shutil.move(backup, abs_path)


@pytest.mark.slow
def test_concat_split_conserves_all_spikes(same_day_group):
    """``split_sorting_by_session`` maps every (unit, frame) to exactly one
    member and back. Boundaries not covering the frozen member set raise
    ``ConcatSplitError``."""
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2._recording.concat import member_split_key
    from spyglass.spikesorting.v2.exceptions import ConcatSplitError
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
    )

    grp = same_day_group
    concat_pk = _populate_concat(
        grp["group_key"], grp["preprocessing_params_name"]
    )
    n0, n1 = _member_sample_counts(grp)
    total = n0 + n1
    # Synthetic concat-frame sorting: spikes spanning member 0, the boundary
    # frame (n0 -> first frame of member 1), and the last valid frame.
    trains = {
        7: np.array([0, n0 - 1, n0, total - 1]),
        9: np.array([5, n0 + 3]),
    }
    sorting = si.NumpySorting.from_unit_dict(
        [trains], sampling_frequency=30_000.0
    )
    per_member = ConcatenatedRecording().split_sorting_by_session(
        sorting, concat_pk
    )
    assert len(per_member) == 2

    # Independently re-derive each member's start offset from the frozen
    # snapshot + boundaries to reconstruct the global frames per unit.
    snapshot = (
        ConcatenatedRecordingSelection.MemberSnapshot & concat_pk
    ).fetch(as_dict=True, order_by="member_index")
    indices, ends = (ConcatenatedRecording.MemberBoundary & concat_pk).fetch(
        "member_index", "end_sample"
    )
    end_by_index = {int(i): int(e) for i, e in zip(indices, ends)}
    ordered_ends = [end_by_index[int(r["member_index"])] for r in snapshot]
    starts = [0, *ordered_ends[:-1]]
    start_by_key = {
        member_split_key(row): start for row, start in zip(snapshot, starts)
    }
    for unit_id, original in trains.items():
        reconstructed = np.sort(
            np.concatenate(
                [
                    per_member[key].get_unit_spike_train(unit_id) + start
                    for key, start in start_by_key.items()
                ]
            )
        )
        np.testing.assert_array_equal(reconstructed, np.sort(original))

    (
        ConcatenatedRecording.MemberBoundary & concat_pk & {"member_index": 1}
    ).delete_quick()
    with pytest.raises(
        ConcatSplitError, match="one boundary per frozen member"
    ):
        ConcatenatedRecording().split_sorting_by_session(sorting, concat_pk)


@pytest.mark.slow
def test_concat_nwb_reconstructs_member_boundaries(same_day_group):
    """The concat NWB self-describes the split.

    It embeds member provenance and frame boundaries that reproduce
    ``split_sorting_by_session`` without the database, and no
    motion-correction fields.
    """
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._storage.provenance import (
        CONCAT_MEMBERS,
        CONCAT_PROVENANCE,
        read_long_provenance,
        read_provenance_values,
    )
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        SessionGroup,
    )

    grp = same_day_group
    concat_pk = _populate_concat(
        grp["group_key"], grp["preprocessing_params_name"]
    )
    row = (ConcatenatedRecording & concat_pk).fetch1()
    abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])

    header = read_provenance_values(abs_path, CONCAT_PROVENANCE)
    assert header["concat_recording_id"] == str(
        concat_pk["concat_recording_id"]
    )
    assert not any("motion" in field for field in header)

    members = sorted(
        read_long_provenance(abs_path, CONCAT_MEMBERS),
        key=lambda r: r["member_index"],
    )
    # Every row names its source Recording and full frame span.
    expected_members = []
    concat_start = 0
    for index, (member, recording_key) in enumerate(
        zip(grp["same_day_members"], grp["recording_pks"], strict=True)
    ):
        count = Recording().get_recording(recording_key).get_num_samples()
        expected_members.append(
            {
                "member_index": index,
                "recording_id": str(recording_key["recording_id"]),
                "nwb_file_name": member["nwb_file_name"],
                "interval_list_name": member["interval_list_name"],
                "artifact_detection_id": "none",
                "start_sample": 0,
                "end_sample": count,
                "concat_start_sample": concat_start,
                "concat_end_sample": concat_start + count,
                "provenance_schema_version": 1,
            }
        )
        concat_start += count
    assert members == expected_members
    assert concat_start == row["n_samples"]

    # Boundaries match MemberBoundary and tile the concat timeline.
    idxs, ends = (ConcatenatedRecording.MemberBoundary & concat_pk).fetch(
        "member_index", "end_sample", order_by="member_index"
    )
    boundary_by_index = {int(i): int(e) for i, e in zip(idxs, ends)}
    prev = 0
    for m in members:
        assert m["concat_start_sample"] == prev
        assert m["concat_end_sample"] == boundary_by_index[m["member_index"]]
        assert (m["end_sample"] - m["start_sample"]) == (
            m["concat_end_sample"] - m["concat_start_sample"]
        )
        assert m["recording_id"]
        prev = m["concat_end_sample"]

    # Member identity travels in the file (compared to, not sourced from, the DB).
    expected = {
        int(r["member_index"]): r
        for r in (SessionGroup.Member & grp["group_key"]).fetch(
            as_dict=True, order_by="member_index"
        )
    }
    for m in members:
        exp = expected[m["member_index"]]
        assert m["nwb_file_name"] == exp["nwb_file_name"]
        assert m["interval_list_name"] == exp["interval_list_name"]


@pytest.mark.slow
def test_concatenated_recording_make_never_calls_recording_populate(
    same_day_group, monkeypatch
):
    """make() reads cached Recording artifacts and never calls
    Recording.populate."""
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    def _boom(*args, **kwargs):
        raise AssertionError(
            "ConcatenatedRecording.make must not call Recording.populate"
        )

    monkeypatch.setattr(Recording, "populate", _boom)
    grp = same_day_group
    concat_pk = _populate_concat(
        grp["group_key"], grp["preprocessing_params_name"]
    )
    assert ConcatenatedRecording & concat_pk


@pytest.mark.slow
def test_concat_make_uses_selection_row_not_uuid_key(same_day_group):
    """Concats over different member sets populate independently: make()
    restricts by the fetched selection row, not the UUID-only key."""
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
    a_pk = _populate_concat(grp["group_key"], grp["preprocessing_params_name"])
    b_pk = _populate_concat(
        {"session_group_owner": owner, "session_group_name": "sg_concat_b"},
        grp["preprocessing_params_name"],
    )
    assert len(ConcatenatedRecording.MemberBoundary & a_pk) == 2
    assert len(ConcatenatedRecording.MemberBoundary & b_pk) == 1
    a_dur = (ConcatenatedRecording & a_pk).fetch1("total_duration_s")
    b_dur = (ConcatenatedRecording & b_pk).fetch1("total_duration_s")
    assert a_dur > b_dur


@pytest.mark.slow
def test_concatenated_recording_is_the_uncorrected_member_concatenation(
    same_day_group,
):
    """Materialization applies no motion correction: the persisted traces are
    exactly the members' stored traces in member order (an unmasked selection),
    every member channel is kept, and the MemberBoundary ends are the running
    member sample counts."""
    import numpy as np

    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    grp = same_day_group
    concat_pk = _populate_concat(
        grp["group_key"], grp["preprocessing_params_name"]
    )
    members = [
        Recording().get_recording(rec_pk) for rec_pk in grp["recording_pks"]
    ]
    n0, n1 = (member.get_num_samples() for member in members)
    row = (ConcatenatedRecording & concat_pk).fetch1()
    concat = ConcatenatedRecording().get_recording(concat_pk)

    np.testing.assert_array_equal(
        concat.get_traces(),
        np.concatenate([member.get_traces() for member in members], axis=0),
    )
    for member in members:
        assert concat.get_num_channels() == member.get_num_channels()
    assert row["n_channels"] == members[0].get_num_channels()
    assert row["n_samples"] == n0 + n1
    ends = (ConcatenatedRecording.MemberBoundary & concat_pk).fetch(
        "end_sample", order_by="member_index"
    )
    assert [int(e) for e in ends] == [n0, n0 + n1]


@pytest.mark.slow
def test_concat_make_raises_on_sample_count_drift(same_day_group, monkeypatch):
    """A stitched sample count differing from the member sum raises and
    persists nothing."""
    import numpy as np
    import spikeinterface as si

    import spyglass.spikesorting.v2._recording.concat as concat_mod
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
    )

    def _drifted(recordings):
        # Simulate a concatenation that changed the sample count.
        total = sum(int(r.get_num_samples()) for r in recordings)
        return si.NumpyRecording(
            [np.zeros((total + 1, 4), dtype=np.float32)],
            sampling_frequency=recordings[0].get_sampling_frequency(),
        )

    grp = same_day_group
    concat_pk = _concat_selection(
        grp["group_key"], grp["preprocessing_params_name"]
    )
    monkeypatch.setattr(concat_mod, "build_concatenated_recording", _drifted)
    # make_compute directly; it raises before the NWB write.
    fetched = ConcatenatedRecording().make_fetch(concat_pk)
    with pytest.raises(RuntimeError, match="sample count"):
        ConcatenatedRecording().make_compute(concat_pk, **fetched._asdict())
    assert not (ConcatenatedRecording & concat_pk)


@pytest.mark.slow
def test_make_compute_reads_fetched_members_and_only_stages_output(
    same_day_group, monkeypatch
):
    """``make_fetch`` resolves each member's cached file (hash-stable across
    DataJoint's two fetches); ``make_compute`` reads those files and queries
    the DB only to stage its output, reproducing the stored content hash."""
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2._storage.staged_outputs import (
        unlink_staged_analysis_file as _unlink_staged_analysis_file,
    )
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording
    from tests.spikesorting.v2._tripart_helpers import (
        fetch_hash,
        forbid_db_queries,
    )

    grp = same_day_group
    concat_pk = _populate_concat(
        grp["group_key"], grp["preprocessing_params_name"]
    )
    table = ConcatenatedRecording()
    fetched = table.make_fetch(concat_pk)
    assert fetched.member_traces == tuple(
        Recording().resolve_stored_traces(plan["recording_pk"])
        for plan in fetched.member_plan
    )
    assert fetch_hash(table.make_fetch(concat_pk)) == fetch_hash(fetched)
    with forbid_db_queries(
        monkeypatch, "ConcatenatedRecording.make_compute", allow_staging=True
    ):
        computed = table.make_compute(concat_pk, *fetched)
    try:
        assert computed.content_hash == (
            ConcatenatedRecording & concat_pk
        ).fetch1("content_hash")
    finally:
        _unlink_staged_analysis_file(
            computed.analysis_file_name, context="test"
        )


# ---------- concat-backed Sorting end-to-end ------------------------------


@pytest.mark.slow
def test_concat_sort_end_to_end_and_split(same_day_group, caplog):
    """A concat-backed Sorting.populate runs end-to-end: it finds units,
    anchors its analysis NWB + per-unit electrodes to the first member,
    raises on concat brain regions without the anchor opt-in (and returns the
    anchor-member frame with it), and ``split_sorting_by_session`` returns one
    per-member sorting in each member's local frame with unit ids preserved."""
    from spyglass.spikesorting.v2._storage.analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2.exceptions import (
        ConcatBrainRegionAmbiguousError,
    )
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAM_NAME,
    )

    grp = same_day_group
    concat_pk = _populate_concat(
        grp["group_key"], grp["preprocessing_params_name"]
    )

    _ensure_clusterless_sorter_params()
    sort_pk = SortingSelection.insert_selection(
        {
            "concat_recording_id": concat_pk["concat_recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": SMOKE_CLUSTERLESS_PARAM_NAME,
        }
    )
    Sorting.populate(sort_pk, reserve_jobs=False)

    row = (Sorting & sort_pk).fetch1()
    assert int(row["n_units"]) > 0, "concat sort found no units"

    # The analyzer folder exists for the resolved display recipe.
    folder = analyzer_path(
        sort_pk["sorting_id"], row["display_waveform_params_name"]
    )
    assert folder.exists()

    # Unit electrodes are anchored to the first member's NWB.
    first_nwb = grp["same_day_members"][0]["nwb_file_name"]
    unit_nwbs = set((Sorting.Unit & sort_pk).fetch("nwb_file_name"))
    assert unit_nwbs == {first_nwb}

    # Concat brain regions: ambiguous by default, anchor-member with opt-in.
    with pytest.raises(ConcatBrainRegionAmbiguousError):
        Sorting().get_unit_brain_regions(sort_pk)
    regions = Sorting().get_unit_brain_regions(
        sort_pk, allow_anchor_member=True
    )
    assert (regions["region_resolution"] == "anchor_member").all()

    sorting_obj = Sorting().get_analyzer(sort_pk).sorting
    split = ConcatenatedRecording().split_sorting_by_session(
        sorting_obj, concat_pk
    )
    members = grp["same_day_members"]
    # Keyed by the full member identity.
    assert set(split) == {
        (
            m["nwb_file_name"],
            m["sort_group_id"],
            m["interval_list_name"],
            grp["owner"],
        )
        for m in members
    }
    assert len(split) == len(members)
    member_samples = {
        m["nwb_file_name"]: n
        for m, n in zip(members, _member_sample_counts(grp))
    }
    all_unit_ids = set(sorting_obj.unit_ids)
    for (nwb_file_name, _sg, _interval, _team), member_sorting in split.items():
        assert set(member_sorting.unit_ids) == all_unit_ids
        for unit_id in member_sorting.unit_ids:
            frames = member_sorting.get_unit_spike_train(unit_id=unit_id)
            if len(frames):
                assert int(frames.min()) >= 0
                assert int(frames.max()) < member_samples[nwb_file_name]

    # Concat curations stay out of the session-scoped merge table: their spike
    # times use the synthetic concat timeline.
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import (
        CONCAT_MERGE_GATE_MESSAGE,
        CurationV2,
    )

    caplog.clear()
    curation_key = CurationV2.insert_curation(sorting_key=sort_pk)
    assert not (SpikeSortingOutput.CurationV2 & curation_key)
    assert (
        CONCAT_MERGE_GATE_MESSAGE.format(sorting_id=sort_pk["sorting_id"])
        in caplog.text
    )

    # Session-scoped provenance resolves through the anchor member.
    sorter, meta_nwb = CurationV2.get_sort_metadata(
        {"sorting_id": sort_pk["sorting_id"]}
    )
    assert sorter == "clusterless_thresholder"
    assert meta_nwb == first_nwb
    sg_info = CurationV2.get_sort_group_info(curation_key)
    sg_nwbs = set((sg_info).fetch("nwb_file_name"))
    assert sg_nwbs == {first_nwb}

    # Curated readback remains source-aware inside CurationV2 and reconstructs
    # the sorting against the concat timeline.
    curated_sorting = CurationV2.get_sorting(curation_key)
    assert set(curated_sorting.unit_ids) == all_unit_ids

    # describe_units uses the concat recording's total_duration_s.
    from spyglass.spikesorting.v2.pipeline import describe_units

    assert (
        Sorting.resolve_anchor_nwb_file_name(
            {"sorting_id": sort_pk["sorting_id"]}
        )
        == first_nwb
    )
    units_df = describe_units(sort_pk["sorting_id"])
    assert len(units_df) == int(row["n_units"])
    # This fixture explicitly selects no member detections, so valid duration
    # equals the whole concat duration rather than one member's duration.
    concat_total_duration_s = (ConcatenatedRecording & concat_pk).fetch1(
        "total_duration_s"
    )
    assert (units_df["firing_rate_hz"] > 0).all()
    assert units_df["firing_rate_hz"].to_numpy() == pytest.approx(
        units_df["n_spikes"].to_numpy() / concat_total_duration_s
    )


@pytest.mark.slow
def test_concat_curation_evaluation_acceptance_creates_committed_child(
    same_day_group, curation_evaluation_defaults
):
    """A concat-backed evaluation can be accepted into a committed child.

    The accepted child is re-evaluated.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAM_NAME,
    )

    grp = same_day_group
    concat_pk = _populate_concat(
        grp["group_key"], grp["preprocessing_params_name"]
    )
    _ensure_clusterless_sorter_params()
    sort_pk = SortingSelection.insert_selection(
        {
            "concat_recording_id": concat_pk["concat_recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": SMOKE_CLUSTERLESS_PARAM_NAME,
        }
    )
    Sorting.populate(sort_pk, reserve_jobs=False)
    assert int((Sorting & sort_pk).fetch1("n_units")) > 0

    curation = CurationV2.insert_curation(sorting_key=sort_pk)
    sel = CurationEvaluationSelection.insert_selection(
        {
            **curation,
            "metric_params_name": "minimal",
            "auto_curation_rules_name": "none",
        }
    )
    CurationEvaluation.populate(sel, reserve_jobs=False)

    # The evaluation NWB carries concat_recording_id (recording_id is None).
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._storage.provenance import (
        CURATION_EVALUATION_PROVENANCE,
        read_provenance_values,
    )

    eval_prov = read_provenance_values(
        AnalysisNwbfile.get_abs_path(
            (CurationEvaluation & sel).fetch1("analysis_file_name")
        ),
        CURATION_EVALUATION_PROVENANCE,
    )
    assert eval_prov["source_kind"] == "concatenated_recording"
    assert eval_prov["recording_id"] is None
    assert eval_prov["concat_recording_id"] == str(
        concat_pk["concat_recording_id"]
    )

    metrics = CurationEvaluation.get_metrics(sel)
    expected = {int(u) for u in (CurationV2.Unit & curation).fetch("unit_id")}
    assert set(metrics.index) == expected

    child = CurationEvaluation().use_evaluation_labels(sel)
    assert CurationV2.is_committed_curation(child)
    assert (CurationV2 & child).fetch1("curation_source") == (
        "curation_evaluation"
    )
    assert (
        set(int(u) for u in (CurationV2.Unit & child).fetch("unit_id"))
        == expected
    )

    child_sel = CurationEvaluationSelection.insert_selection(
        {
            **child,
            "metric_params_name": "minimal",
            "auto_curation_rules_name": "none",
        }
    )
    CurationEvaluation.populate(child_sel, reserve_jobs=False)
    child_metrics = CurationEvaluation.get_metrics(child_sel)
    assert set(child_metrics.index) == expected


@pytest.mark.slow
def test_preproc_restriction_excludes_concat_and_includes_recording_curations(
    same_day_group,
):
    """A preprocessing restriction returns only merge-registered curations.

    Concat-backed CurationV2 rows are deliberately gated from
    SpikeSortingOutput, while a recording-backed curation under the same recipe
    remains discoverable. Concat-shaped restrictions therefore return no ids.
    """
    import uuid

    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAM_NAME,
    )

    grp = same_day_group
    preproc = grp["preprocessing_params_name"]
    _ensure_clusterless_sorter_params()

    # A concat-backed sort + curation under the shared preprocessing recipe.
    # (The concat lineage is torn down by the ``same_day_group`` fixture.)
    concat_pk = _populate_concat(grp["group_key"], preproc)
    concat_sort = SortingSelection.insert_selection(
        {
            "concat_recording_id": concat_pk["concat_recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": SMOKE_CLUSTERLESS_PARAM_NAME,
        }
    )
    Sorting.populate(concat_sort, reserve_jobs=False)
    concat_curation = CurationV2.insert_curation(sorting_key=concat_sort)
    assert not (SpikeSortingOutput.CurationV2 & concat_curation)

    # A single-recording sort + curation on a member, SAME preproc recipe.
    rec_sort = SortingSelection.insert_selection(
        {
            "recording_id": grp["recording_pks"][0]["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": SMOKE_CLUSTERLESS_PARAM_NAME,
        }
    )
    Sorting.populate(rec_sort, reserve_jobs=False)
    try:
        rec_curation = CurationV2.insert_curation(sorting_key=rec_sort)
        rec_merge_id = (SpikeSortingOutput.CurationV2 & rec_curation).fetch1(
            "merge_id"
        )

        # The bare preprocessing-recipe restriction surfaces only the
        # merge-registered recording-backed curation.
        by_preproc = SpikeSortingOutput().get_restricted_merge_ids(
            {"preprocessing_params_name": preproc}, sources=["v2"]
        )
        assert rec_merge_id in by_preproc

        # An unmatched recipe returns neither.
        by_absent_preproc = SpikeSortingOutput().get_restricted_merge_ids(
            {"preprocessing_params_name": "no_such_preproc_recipe"},
            sources=["v2"],
        )
        assert rec_merge_id not in by_absent_preproc

        # Concat-shaped restrictions remain valid query shapes, but there is no
        # merge row to return until per-member wall-clock rows exist.
        by_concat_and_preproc = SpikeSortingOutput().get_restricted_merge_ids(
            {
                "concat_recording_id": concat_pk["concat_recording_id"],
                "preprocessing_params_name": preproc,
            },
            sources=["v2"],
        )
        assert len(by_concat_and_preproc) == 0
        by_concat_wrong_preproc = SpikeSortingOutput().get_restricted_merge_ids(
            {
                "concat_recording_id": concat_pk["concat_recording_id"],
                "preprocessing_params_name": "no_such_preproc_recipe",
            },
            sources=["v2"],
        )
        assert len(by_concat_wrong_preproc) == 0

        # An unknown artifact id excludes both; None keeps both.
        by_preproc_unknown_artifact = (
            SpikeSortingOutput().get_restricted_merge_ids(
                {
                    "preprocessing_params_name": preproc,
                    "artifact_detection_id": str(uuid.uuid4()),
                },
                sources=["v2"],
            )
        )
        assert rec_merge_id not in by_preproc_unknown_artifact
        by_preproc_no_artifact = SpikeSortingOutput().get_restricted_merge_ids(
            {
                "preprocessing_params_name": preproc,
                "artifact_detection_id": None,
            },
            sources=["v2"],
        )
        assert rec_merge_id in by_preproc_no_artifact
    finally:
        # ``same_day_group`` teardown only follows concat lineage.
        clear_curations_for(rec_sort)
        (Sorting & rec_sort).super_delete(warn=False)
        (SortingSelection & rec_sort).super_delete(warn=False)


# ---------- memory / runtime measurement ----------------------------------


@pytest.mark.slow
def test_concat_materialization_memory_runtime_is_measured(same_day_group):
    """Measures and logs peak RSS and runtime of concat materialization,
    with generous bounds for the tiny fixture."""
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    grp = same_day_group
    with _PeakRSS() as measured:
        concat_pk = _populate_concat(
            grp["group_key"], grp["preprocessing_params_name"]
        )
    logger.info(
        "concat minirec materialization: peak_rss=%.0f MB (+%.0f MB), "
        "runtime=%.1f s",
        measured.peak_mb,
        measured.growth_mb,
        measured.elapsed_s,
    )
    assert ConcatenatedRecording & concat_pk
    # Growth, not process RSS, which holds earlier tests' state.
    assert measured.growth_mb < 4096
    assert measured.elapsed_s < 300


@pytest.mark.slow
def test_concat_chronic_real_dataset_memory_runtime(request, dj_conn):
    """Opt-in real-chronic memory/runtime check (``pytest --run-chronic``).

    Uses the 1-hour 30 kHz NWB at ``SPIKESORTING_V2_CHRONIC_TEST_PATH`` and
    its lowest sort group. Asserts peak RSS < 8 GB and runtime < 10 min on 16
    cores; logs concat and sort timings separately."""
    import os

    if not request.config.getoption("--run-chronic"):
        pytest.skip(
            "pass --run-chronic to run the real-chronic memory/runtime check"
        )
    chronic_path = os.environ.get("SPIKESORTING_V2_CHRONIC_TEST_PATH")
    if not chronic_path:
        pytest.skip(
            "set SPIKESORTING_V2_CHRONIC_TEST_PATH to a real 1-hour chronic NWB"
        )

    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        SessionGroup,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from tests.spikesorting.v2._ingest_helpers import (
        clean_session_groups_for_owner,
        configure_v2_run_inputs,
        copy_and_insert_nwb,
    )
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAM_NAME,
    )

    owner, name = "chronic_real_gate_owner", "chronic_real_gate"
    nwb_file_name = copy_and_insert_nwb(chronic_path)
    run = configure_v2_run_inputs(nwb_file_name, owner)
    member = {
        "nwb_file_name": run["nwb_file_name"],
        "sort_group_id": run["sort_group_id"],
        "interval_list_name": run["interval_list_name"],
    }
    rec_pk = RecordingSelection.insert_selection(
        {**member, "preprocessing_params_name": "default", "team_name": owner}
    )
    if not (Recording & rec_pk):
        Recording.populate(rec_pk, reserve_jobs=False)

    clean_session_groups_for_owner(owner)
    SessionGroup.create_group(owner, name, [member])
    _ensure_clusterless_sorter_params()
    try:
        concat_pk = _concat_selection(
            {"session_group_owner": owner, "session_group_name": name},
            "default",
        )
        with _PeakRSS() as concat_measured:
            ConcatenatedRecording.populate(concat_pk, reserve_jobs=False)
        sort_pk = SortingSelection.insert_selection(
            {
                "concat_recording_id": concat_pk["concat_recording_id"],
                "sorter": "clusterless_thresholder",
                "sorter_params_name": SMOKE_CLUSTERLESS_PARAM_NAME,
            }
        )
        with _PeakRSS() as sort_measured:
            Sorting.populate(sort_pk, reserve_jobs=False)

        peak_mb = max(concat_measured.peak_mb, sort_measured.peak_mb)
        total_s = concat_measured.elapsed_s + sort_measured.elapsed_s
        logger.info(
            "real-chronic gate: concat materialize=%.1f s, sort=%.1f s, "
            "total=%.1f s, peak_rss=%.0f MB",
            concat_measured.elapsed_s,
            sort_measured.elapsed_s,
            total_s,
            peak_mb,
        )
        assert peak_mb < 8 * 1024, f"peak RSS {peak_mb:.0f} MB exceeds 8 GB"
        assert total_s < 600, f"runtime {total_s:.0f} s exceeds 10 min"
    finally:
        clean_session_groups_for_owner(owner)


@pytest.mark.slow
def test_concat_applied_merge_through_curation_and_evaluation(
    same_day_group, curation_evaluation_defaults
):
    """A committed APPLIED-MERGE on a CONCAT-backed sort flows through
    ``CurationV2.get_sorting`` and ``CurationEvaluation`` carrying its MERGED
    unit set, without creating an unsafe SpikeSortingOutput row.

    A planted 2-unit sort hosts the merge; the clusterless unit count is
    data-dependent.
    """
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from tests.spikesorting.v2._smoke_constants import SMOKE_CLUSTERLESS_PARAMS

    grp = same_day_group
    concat_pk = _populate_concat(
        grp["group_key"], grp["preprocessing_params_name"]
    )
    two_unit_params = "minirec_concat_two_unit_merge"
    SorterParameters.insert1(
        {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": two_unit_params,
            "params": dict(SMOKE_CLUSTERLESS_PARAMS),
        },
        skip_duplicates=True,
        allow_duplicate_params=True,
    )

    def _plant(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
        statistics_spans=None,
    ):
        samples = np.array([500, 1000, 1500, 600, 1100, 1600], dtype=np.int64)
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    sort_pk = SortingSelection.insert_selection(
        {
            "concat_recording_id": concat_pk["concat_recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": two_unit_params,
        }
    )
    sorting_key = {"sorting_id": sort_pk["sorting_id"]}

    def _drop_output_and_curations():
        keys = (CurationV2 & sorting_key).fetch("KEY", as_dict=True)
        if keys:
            for mid in (SpikeSortingOutput.CurationV2 & keys).fetch("merge_id"):
                (SpikeSortingOutput & {"merge_id": mid}).super_delete(
                    warn=False
                )
        (CurationV2 & sorting_key).super_delete(warn=False)

    mp = pytest.MonkeyPatch()
    try:
        plant_sorter(mp, _plant)
        if not (Sorting & sort_pk):
            Sorting.populate(sort_pk, reserve_jobs=False)
    finally:
        mp.undo()
    unit_ids = sorted(int(u) for u in (Sorting.Unit & sort_pk).fetch("unit_id"))
    assert len(unit_ids) >= 2, "planted concat sort must yield >=2 units"

    _drop_output_and_curations()
    try:
        merged = CurationV2.insert_curation(
            sorting_key,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            apply_merge=True,
        )
        merged_units = sorted(
            int(u) for u in (CurationV2.Unit & merged).fetch("unit_id")
        )
        assert len(merged_units) == 1, merged_units
        merged_uid = merged_units[0]
        n_spikes = int(
            (CurationV2.Unit & merged & {"unit_id": merged_uid}).fetch1(
                "n_spikes"
            )
        )

        # CurationV2.get_sorting on the concat merged curation: the merged
        # namespace (one unit), train = union of the two contributors.
        sorting = CurationV2().get_sorting(merged)
        assert sorted(int(u) for u in sorting.get_unit_ids()) == merged_units
        raw = Sorting().get_sorting(sorting_key)
        expected_n = sum(
            len(raw.get_unit_spike_train(unit_id=u)) for u in unit_ids
        )
        assert n_spikes == expected_n
        assert (
            len(sorting.get_unit_spike_train(unit_id=merged_uid)) == expected_n
        )

        assert not (SpikeSortingOutput.CurationV2 & merged)

        # Metrics index the merged unit.
        sel = CurationEvaluationSelection.insert_selection(
            {
                **merged,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(sel, reserve_jobs=False)
        assert set(CurationEvaluation.get_metrics(sel).index) == set(
            merged_units
        )
    finally:
        _drop_output_and_curations()
        (Sorting & sort_pk).super_delete(warn=False)
        (SortingSelection & sort_pk).super_delete(warn=False)
        (
            SorterParameters & {"sorter_params_name": two_unit_params}
        ).super_delete(warn=False)


@pytest.mark.slow
def test_concat_analyzer_receives_statistics_spans_after_reload(
    same_day_group, curation_evaluation_defaults
):
    """A concat sort persists span-split statistics spans and its analyzers
    rebuild noise levels from them after reload.

    Spans split at every member join; the rebuilt display analyzer carries
    the sort-time noise levels, within 2% of the members' exact MAD.
    """
    import shutil

    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        analyzer_path,
        load_analyzer_folder,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for
    from tests.spikesorting.v2._smoke_constants import SMOKE_CLUSTERLESS_PARAMS

    grp = same_day_group
    concat_pk = _populate_concat(
        grp["group_key"], grp["preprocessing_params_name"]
    )
    params_name = "minirec_concat_statistics_spans"
    SorterParameters.insert1(
        {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": params_name,
            "params": dict(SMOKE_CLUSTERLESS_PARAMS),
        },
        skip_duplicates=True,
        allow_duplicate_params=True,
    )
    received = {}

    def _plant(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
        statistics_spans=None,
    ):
        received["spans"] = statistics_spans
        n = int(recording.get_num_samples())
        samples = np.arange(500, n - 500, 2000, dtype=np.int64)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples],
            labels_list=[np.zeros(samples.size, dtype=np.int32)],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    sort_pk = SortingSelection.insert_selection(
        {
            "concat_recording_id": concat_pk["concat_recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": params_name,
        }
    )
    mp = pytest.MonkeyPatch()
    try:
        plant_sorter(mp, _plant)
        (Sorting & sort_pk).super_delete(warn=False)
        Sorting.populate(sort_pk, reserve_jobs=False)
    finally:
        mp.undo()

    try:
        spans = Sorting().get_statistics_spans(sort_pk)
        assert spans and received["spans"] == spans
        stored = (ConcatenatedRecording & concat_pk).fetch1("statistics_spans")
        assert [tuple(int(v) for v in row) for row in stored] == spans

        n_total = int((ConcatenatedRecording & concat_pk).fetch1("n_samples"))
        ends = [
            int(e)
            for e in (ConcatenatedRecording.MemberBoundary & concat_pk).fetch(
                "end_sample", order_by="member_index"
            )
        ]
        assert len(ends) == 2 and ends[-1] == n_total
        for start, end in spans:
            for join in ends[:-1]:
                assert not (start < join < end), (spans, ends)
        # Unmasked, single-interval members: exactly one span per member, so
        # the spans split at the join and nowhere else.
        assert spans == [(0, ends[0]), (ends[0], n_total)]

        display_name = (Sorting & sort_pk).fetch1(
            "display_waveform_params_name"
        )
        folder = analyzer_path(sort_pk["sorting_id"], display_name)
        sort_time_noise = np.asarray(
            load_analyzer_folder(folder)
            .get_extension("noise_levels")
            .get_data()
        )
        shutil.rmtree(folder)

        clear_curations_for(sort_pk)
        root = CurationV2.insert_curation(sorting_key=sort_pk)
        sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(sel, reserve_jobs=False)
        assert CurationEvaluation & sel
        rebuilt_noise = np.asarray(
            load_analyzer_folder(folder)
            .get_extension("noise_levels")
            .get_data()
        )
        np.testing.assert_array_equal(rebuilt_noise, sort_time_noise)

        traces = (
            ConcatenatedRecording()
            .get_recording(concat_pk)
            .get_traces(return_in_uV=True)
            .astype(np.float64)
        )
        exact_mad = (
            np.median(np.abs(traces - np.median(traces, axis=0)), axis=0)
            / 0.6744897501960817
        )
        np.testing.assert_allclose(rebuilt_noise, exact_mad, rtol=0.02)
    finally:
        clear_curations_for(sort_pk)
        (Sorting & sort_pk).super_delete(warn=False)
        (SortingSelection & sort_pk).super_delete(warn=False)
        (SorterParameters & {"sorter_params_name": params_name}).super_delete(
            warn=False
        )


@pytest.mark.slow
@pytest.mark.smoke
def test_run_v2_pipeline_concat_mode_routes_session_group(same_day_group):
    """run_v2_pipeline concat mode: member recordings -> concat -> sort -> curation.

    Member detection is disabled. Only session-aligned member outputs enter
    the merge; a rerun reuses work.
    """
    import spyglass.spikesorting.v2._orchestration.presets as presets_mod
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCurationV2,
    )
    from spyglass.spikesorting.v2.curation import CONCAT_MERGE_GATE_MESSAGE
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.pipeline import (
        register_pipeline_preset,
        run_v2_concat_pipeline,
    )
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAM_NAME,
    )

    initialize_v2_defaults()
    _ensure_clusterless_sorter_params()
    grp = same_day_group
    preset_name = "test_concat_clusterless_smoke"
    # Smoke clusterless sorter; concat mode needs no motion recipe.
    register_pipeline_preset(
        preset_name,
        {
            "preprocessing_params_name": grp["preprocessing_params_name"],
            "artifact_detection_params_name": None,
            "sorter": "clusterless_thresholder",
            "sorter_params_name": SMOKE_CLUSTERLESS_PARAM_NAME,
            "metric_params_name": "minimal",
            "auto_curation_rules_name": "none",
        },
        validate_rows=False,
    )
    try:
        summary = run_v2_concat_pipeline(
            session_group_owner=grp["group_key"]["session_group_owner"],
            session_group_name=grp["group_key"]["session_group_name"],
            pipeline_preset=preset_name,
        )
        # Concat-shaped manifest: concat source keys, no single-session/artifact.
        assert "concat_recording_id" in summary
        assert summary["concat_recording_status"] in {"computed", "reused"}
        assert len(summary["member_recording_ids"]) == 2
        assert "recording_id" not in summary
        assert "artifact_detection_id" not in summary
        assert "artifact_detection" not in summary["stage_seconds"]
        # The member-recording build and the concat are each their own stage.
        assert summary["member_recording_status"] in {"computed", "reused"}
        assert "member_recording" in summary["stage_seconds"]
        assert "concat_recording" in summary["stage_seconds"]
        assert summary["member_curation_status"] == "computed"
        assert "member_curation" in summary["stage_seconds"]
        assert set(summary["member_merge_ids"]) == {0, 1}
        assert len(summary["member_merge_ids"]) == 2
        # The synthetic concat row remains gated; one safe merge row is
        # registered for each member session instead.
        assert summary["n_units"] >= 0
        assert summary["root_merge_id"] is None
        curation_key = {
            "sorting_id": summary["sorting_id"],
            "curation_id": summary["root_curation_id"],
        }
        assert not (SpikeSortingOutput.CurationV2 & curation_key)
        assert len(ConcatMemberCurationV2 & curation_key) == 2
        assert (
            len(SpikeSortingOutput.ConcatMemberCurationV2 & curation_key) == 2
        )
        assert (
            dict(CurationRef.from_key(curation_key).member_merge_ids)
            == summary["member_merge_ids"]
        )
        assert (
            CONCAT_MERGE_GATE_MESSAGE.format(sorting_id=summary["sorting_id"])
            in summary["warnings"]
        )

        # Idempotent: a rerun reuses the concat sort + curation.
        rerun = run_v2_concat_pipeline(
            session_group_owner=grp["group_key"]["session_group_owner"],
            session_group_name=grp["group_key"]["session_group_name"],
            pipeline_preset=preset_name,
        )
        assert rerun["sorting_id"] == summary["sorting_id"]
        assert rerun["root_merge_id"] is None
        assert rerun["member_merge_ids"] == summary["member_merge_ids"]
        assert rerun["member_curation_status"] == "reused"
        assert (
            CONCAT_MERGE_GATE_MESSAGE.format(sorting_id=summary["sorting_id"])
            in rerun["warnings"]
        )
        assert rerun["concat_recording_status"] == "reused"

        # Auto-curation yields a child off the concat timeline; member ids
        # point at it, one per session, with one warning for the gate.
        auto_summary = run_v2_concat_pipeline(
            session_group_owner=grp["group_key"]["session_group_owner"],
            session_group_name=grp["group_key"]["session_group_name"],
            pipeline_preset=preset_name,
            auto_curate=True,
        )
        assert auto_summary["auto_labeled_curation_id"] is not None
        assert auto_summary["root_merge_id"] is None
        assert auto_summary["auto_labeled_merge_id"] is None
        assert not (
            SpikeSortingOutput.CurationV2
            & {
                "sorting_id": auto_summary["sorting_id"],
                "curation_id": auto_summary["auto_labeled_curation_id"],
            }
        )
        child_key = {
            "sorting_id": auto_summary["sorting_id"],
            "curation_id": auto_summary["auto_labeled_curation_id"],
        }
        assert len(ConcatMemberCurationV2 & child_key) == 2
        assert len(auto_summary["member_merge_ids"]) == 2
        assert set(
            (SpikeSortingOutput.ConcatMemberCurationV2 & child_key).fetch(
                "merge_id"
            )
        ) == set(auto_summary["member_merge_ids"].values())
        gate_message = CONCAT_MERGE_GATE_MESSAGE.format(
            sorting_id=summary["sorting_id"]
        )
        assert auto_summary["warnings"].count(gate_message) == 1

        # One SortedSpikesGroup per member on its session timeline; unit
        # verdicts are shared since the curation is one object.
        from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
        from spyglass.spikesorting.v2.analysis_selection import (
            select_units_for_analysis,
        )
        from spyglass.spikesorting.v2.concat_member_curation import (
            ConcatMemberCurationV2,
        )

        receipt = select_units_for_analysis(
            auto_summary.auto_labeled_curation, policy="all_units"
        )
        created_groups = [dict(g.group_key) for g in receipt.groups]
        try:
            assert len(receipt.groups) == 2
            assert {g.member_index for g in receipt.groups} == set(
                auto_summary["member_merge_ids"]
            )
            for group in receipt.groups:
                assert (
                    group.merge_id
                    == auto_summary["member_merge_ids"][group.member_index]
                )
                member_row = (
                    ConcatMemberCurationV2
                    & child_key
                    & {"member_index": group.member_index}
                ).fetch1()
                assert group.nwb_file_name == member_row["nwb_file_name"]
                assert (
                    group.group_key["nwb_file_name"]
                    == member_row["nwb_file_name"]
                )
                # Spike data comes from the member's session-timeline row.
                member_times = SpikeSortingOutput().get_spike_times(
                    {"merge_id": group.merge_id}
                )
                group_times = group.fetch_spike_data()
                assert len(group_times) == len(member_times)
            with pytest.raises(ValueError, match="one group per member"):
                receipt.group_key
        finally:
            for group_key in created_groups:
                (SortedSpikesGroup & group_key).super_delete(warn=False)
    finally:
        presets_mod._PIPELINE_PRESETS.pop(preset_name, None)
