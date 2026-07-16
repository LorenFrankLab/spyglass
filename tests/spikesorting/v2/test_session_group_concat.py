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
import logging
import threading
import time
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


class _PeakRSS:
    """Context manager: sample process RSS in a background thread; expose peak.

    A REAL measurement (psutil polling of the live process RSS), not a mocked
    metric: ``peak_mb`` is the high-water-mark resident-set size observed while
    the ``with`` body ran, and ``elapsed_s`` is its wall-clock runtime. Used by
    both the synthetic memory/runtime smoke and the ``--run-chronic`` gate so
    the real-data budget assertions run against a proven measurement path.
    """

    def __init__(self, interval_s: float = 0.05):
        import psutil

        self._proc = psutil.Process()
        self._interval_s = interval_s
        self._peak_bytes = 0
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
        self._peak_bytes = self._proc.memory_info().rss
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


@pytest.mark.usefixtures("dj_conn")
def test_create_group_rejects_empty_members():
    """An empty members list is rejected up front, not deferred to an obscure
    ``members[0]`` failure in ConcatenatedRecording.make."""
    from spyglass.spikesorting.v2.session_group import SessionGroup

    with pytest.raises(ValueError, match="empty"):
        SessionGroup.create_group("any_owner", "empty_group", [])


@pytest.mark.slow
def test_create_group_rejects_duplicate_logical_members(
    chronic_2_session_minirec,
):
    """Two members with the same (nwb, sort_group, interval, team) are rejected
    -- the Member PK is member_index only, so a duplicate would silently
    concatenate the same recording twice."""
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
    """Concat preflight passes a fully-ingested group but fails fast on a
    missing auto-curation prerequisite (rather than deep in the sort compute).

    The passing case is also a regression guard for the per-member Raw /
    'raw data valid times' / sort-group-electrode / sampling-rate checks: a
    VALID same-day group must not trip any of them.
    """
    from spyglass.spikesorting.v2._pipeline_preflight import (
        assert_concat_preflight,
    )
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.spikesorting.v2.pipeline import _PIPELINE_PRESETS

    gk = same_day_group["group_key"]
    owner = gk["session_group_owner"]
    name = gk["session_group_name"]
    bundle = _PIPELINE_PRESETS["franklab_concat_hippocampus_30khz_ms5_2026_06"]

    # A valid, fully-ingested group passes every per-member prerequisite.
    assert assert_concat_preflight(owner, name, bundle) == []

    # auto_curate=True with a missing metric row fails fast at preflight,
    # not after the member + concat + sort compute.
    bad = bundle.model_copy(update={"metric_params_name": "does_not_exist_xyz"})
    with pytest.raises(PreflightError, match="QualityMetricParameters"):
        assert_concat_preflight(owner, name, bad, auto_curate=True)


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
    """A member dict carrying ``recording_date`` raises ``SessionGroupDateError``
    -- dates are derived, never caller-set."""
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
def test_concat_id_folds_member_set(same_day_group):
    """``concat_recording_id`` folds the ordered frozen member set: it equals the
    deterministic id of (identity + member_set_hash) and DIFFERS from the
    pre-fold formula (identity alone), and the stored hash is the hash of the
    captured snapshot rows."""
    from spyglass.spikesorting.v2._concat_recording import member_set_hash
    from spyglass.spikesorting.v2._selection_identity import deterministic_id
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )

    grp = same_day_group
    pk = ConcatenatedRecordingSelection.insert_selection(
        {
            **grp["group_key"],
            "preprocessing_params_name": grp["preprocessing_params_name"],
            "motion_correction_params_name": "none",
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
    """The frozen ``MemberSnapshot`` captures each member's logical identity, its
    resolved ``recording_id``, and its ``Recording.content_hash`` -- the values a
    later materialize / rebuild verifies the live recordings against."""
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )

    grp = same_day_group
    pk = ConcatenatedRecordingSelection.insert_selection(
        {
            **grp["group_key"],
            "preprocessing_params_name": grp["preprocessing_params_name"],
            "motion_correction_params_name": "none",
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
    """Two selections identical in params but over a DIFFERENT frozen member set
    mint different ``concat_recording_id``s (and different ``member_set_hash``es)
    -- a different member set is a different concat, not a silent reuse."""
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
        SessionGroup,
    )

    grp = same_day_group
    both = ConcatenatedRecordingSelection.insert_selection(
        {
            **grp["group_key"],
            "preprocessing_params_name": grp["preprocessing_params_name"],
            "motion_correction_params_name": "none",
        }
    )
    # A second group over only the first member -- same owner + params, a
    # different ordered member set. (Torn down with the owner by same_day_group.)
    owner = grp["owner"]
    SessionGroup.create_group(
        owner, "sg_concat_subset", [grp["same_day_members"][0]]
    )
    subset = ConcatenatedRecordingSelection.insert_selection(
        {
            "session_group_owner": owner,
            "session_group_name": "sg_concat_subset",
            "preprocessing_params_name": grp["preprocessing_params_name"],
            "motion_correction_params_name": "none",
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
    """Editing the LIVE SessionGroup.Member set under a fixed
    group name+params re-mints a different concat_recording_id on re-selection,
    and the original concat's frozen MemberSnapshot is unchanged by the edit (old
    concat rows stay valid; the change is not silently absorbed)."""
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
        SessionGroup,
    )

    grp = same_day_group
    request = {
        **grp["group_key"],
        "preprocessing_params_name": grp["preprocessing_params_name"],
        "motion_correction_params_name": "none",
    }
    original = ConcatenatedRecordingSelection.insert_selection(dict(request))
    original_snapshot = (
        ConcatenatedRecordingSelection.MemberSnapshot & original
    ).fetch(as_dict=True, order_by="member_index")
    assert [r["member_index"] for r in original_snapshot] == [0, 1]

    # Edit the LIVE member set under the SAME group_key: drop the second member.
    (
        SessionGroup.Member & {**grp["group_key"], "member_index": 1}
    ).delete_quick()

    # Re-selection over the now-different live member set mints a DIFFERENT id.
    edited = ConcatenatedRecordingSelection.insert_selection(dict(request))
    assert edited["concat_recording_id"] != original["concat_recording_id"]

    # The original concat's frozen snapshot is untouched by the live edit -- the
    # old concat row stays valid (reads its own frozen members), not invalidated.
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


@pytest.mark.usefixtures("dj_conn")
def test_concat_rejects_mismatched_electrode_space(monkeypatch):
    """A SessionGroup whose members map to different physical electrode spaces
    (different electrode ids or brain regions) is rejected.

    The concat result is read in the ANCHOR member's electrode frame, so two
    members with the same local channel layout but different underlying
    electrodes/regions would be silently mis-attributed. The per-member
    electrode/region signature query is exercised end to end by the chronic
    concat path; here the signature source is stubbed so the comparison logic
    (raise on mismatch, accept on match) is checked directly.
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

    # Matching signatures across members pass.
    signatures[1] = signatures[0]
    sg.assert_members_share_electrode_space(members)


@pytest.mark.slow
def test_concat_make_fetch_rejects_mismatched_electrode_space(
    same_day_group, monkeypatch
):
    """``ConcatenatedRecording.make_fetch`` re-asserts electrode-space
    compatibility at the compute boundary -- so member drift after selection (or
    a raw ``allow_direct_insert`` selection) that bypassed ``insert_selection``'s
    check still fails before the concat is read in the anchor member's frame."""
    from spyglass.spikesorting.v2 import session_group as sg
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
    )

    # The members match at selection time (insert_selection's check passes).
    key = ConcatenatedRecordingSelection.insert_selection(
        {
            **same_day_group["group_key"],
            "preprocessing_params_name": same_day_group[
                "preprocessing_params_name"
            ],
            "motion_correction_params_name": "none",
        }
    )
    # Simulate post-selection drift: the per-member electrode/region signatures
    # now diverge, so the compute-boundary re-check must reject.
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
    raises ``SchemaBypassError`` at ``make_fetch`` instead of sorting unmasked.

    ``insert_selection`` rejects a concat source combined with an
    ``artifact_detection_id``, but a direct insert of the part rows bypasses it.
    A concat sort observes the full recording (``obs_intervals=None``), so a
    stray artifact id would otherwise be silently ignored while the sort claims
    artifact metadata -- the compute-boundary re-check catches it.
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
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        SortingSelection.insert1(
            {
                "sorting_id": sid,
                "sorter": "mountainsort5",
                "sorter_params_name": "franklab_30khz_ms5_2026_06",
            },
            allow_direct_insert=True,
        )
        SortingSelection.ConcatenatedRecordingSource.insert1(
            {"sorting_id": sid, "concat_recording_id": uuid.uuid4()}
        )
        # The bypass: a concat source must NOT carry an artifact pass.
        SortingSelection.ArtifactDetectionSource.insert1(
            {"sorting_id": sid, "artifact_detection_id": uuid.uuid4()}
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        with pytest.raises(SchemaBypassError, match="concat source"):
            Sorting().make_fetch({"sorting_id": sid})
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (SortingSelection & {"sorting_id": sid}).super_delete(warn=False)
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")


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


def _concat_selection(group_key, preprocessing_params_name, motion="none"):
    """Insert a ConcatenatedRecordingSelection (any motion preset); return PK."""
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
        MotionCorrectionParameters,
    )

    if motion != "none":
        MotionCorrectionParameters.insert_default()
    return ConcatenatedRecordingSelection.insert_selection(
        {
            **group_key,
            "preprocessing_params_name": preprocessing_params_name,
            "motion_correction_params_name": motion,
        }
    )


def _populate_concat(group_key, preprocessing_params_name, motion="none"):
    """Insert + populate a ConcatenatedRecording; return its PK."""
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    concat_pk = _concat_selection(group_key, preprocessing_params_name, motion)
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

    sel = ConcatenatedRecordingSelection.insert_selection(
        {
            **grp["group_key"],
            "preprocessing_params_name": grp["preprocessing_params_name"],
            "motion_correction_params_name": "none",
        }
    )
    return (ConcatenatedRecordingSelection.MemberSnapshot & sel).fetch(
        as_dict=True, order_by="member_index"
    )


@pytest.mark.slow
def test_load_member_recordings_returns_aligned_counts_and_indices(
    same_day_group,
):
    """``_load_member_recordings`` loads each member's cached Recording from the
    snapshot-resolved plan in member_index order and returns sample counts /
    indices aligned element-wise with the loaded recordings -- the core per-member
    materialization contract, exercised without driving a full populate."""
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    grp = same_day_group
    member_plan = ConcatenatedRecording._resolve_snapshot_recordings(
        _member_snapshot(grp)
    )
    recordings, sample_counts, member_indices = (
        ConcatenatedRecording._load_member_recordings(member_plan)
    )
    expected = _member_sample_counts(grp)
    assert member_indices == [0, 1]
    assert sample_counts == expected
    # The returned recordings are the same objects the counts were taken from.
    assert [int(r.get_num_samples()) for r in recordings] == expected


@pytest.mark.slow
def test_resolve_snapshot_recordings_raises_on_missing_recording(
    same_day_group,
):
    """A frozen member whose ``Recording`` is gone raises
    ``MissingRecordingForConcatError`` rather than silently dropping it (which
    would shift every later member's boundary)."""
    from spyglass.spikesorting.v2.exceptions import (
        MissingRecordingForConcatError,
    )
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    snapshot = _member_snapshot(same_day_group)
    # Point the first frozen member at a recording_id with no Recording row.
    snapshot[0] = {
        **snapshot[0],
        "recording_id": "00000000-0000-0000-0000-000000000000",
    }
    with pytest.raises(MissingRecordingForConcatError, match="populate"):
        ConcatenatedRecording._resolve_snapshot_recordings(snapshot)


@pytest.mark.slow
def test_resolve_snapshot_recordings_raises_on_member_content_drift(
    same_day_group,
):
    """A frozen member whose live ``Recording.content_hash`` no longer matches the
    snapshot raises ``ConcatMemberDriftError`` -- the concat would be built from
    different underlying data than its id was minted for."""
    from spyglass.spikesorting.v2.exceptions import ConcatMemberDriftError
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    snapshot = _member_snapshot(same_day_group)
    # Freeze a content hash that diverges from the live Recording's.
    snapshot[0] = {**snapshot[0], "recording_content_hash": "f" * 64}
    with pytest.raises(ConcatMemberDriftError, match="content"):
        ConcatenatedRecording._resolve_snapshot_recordings(snapshot)


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
    # n_samples is the EXACT integer concat sample count (the split-back guard's
    # authoritative basis), not a float duration round-trip.
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
    """A missing concat cache file is rebuilt on demand: ``get_recording``
    reconciles the ``~external`` checksum and returns a valid, content-identical
    recording -- mirroring ``Recording.get_recording``. Uses ``motion='none'`` so
    the rebuild is bit-deterministic."""
    import numpy as np

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    grp = same_day_group
    concat_pk = _populate_concat(
        grp["group_key"], grp["preprocessing_params_name"], motion="none"
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
    """A concat rebuild whose fingerprint diverges from the stored
    ``content_hash`` raises ``RecordingContentDriftError``; the canonical slot is
    NOT written (drifted bytes are never served) -- mirrors the recording side.
    """
    import shutil

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import _recording_fingerprint as fp_mod
    from spyglass.spikesorting.v2.exceptions import (
        RecordingContentDriftError,
    )
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    grp = same_day_group
    concat_pk = _populate_concat(
        grp["group_key"], grp["preprocessing_params_name"], motion="none"
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
    """``split_sorting_by_session`` back-maps a concat-frame sorting so every input
    (unit, frame) lands in exactly one member and reconstructs to its original
    frame -- per-spike membership, not just summed counts. A MemberBoundary set
    that no longer covers the frozen member set raises ``ConcatSplitError``."""
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2._concat_recording import member_split_key
    from spyglass.spikesorting.v2.exceptions import ConcatSplitError
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
    )

    grp = same_day_group
    concat_pk = _populate_concat(
        grp["group_key"], grp["preprocessing_params_name"], motion="none"
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

    # A MemberBoundary set short of the frozen member set is rejected, not
    # silently truncated.
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

    It embeds the ordered member provenance + frame boundaries that reproduce
    ``split_sorting_by_session``'s mapping without reading live
    ``SessionGroup.Member``, plus the RESOLVED motion preset + kwargs (the
    producing params, not the displacement field).
    """
    import pynwb

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._nwb_provenance import (
        CONCAT_MEMBERS,
        CONCAT_PROVENANCE,
        read_long_provenance,
        read_provenance_values,
    )
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        SessionGroup,
    )

    grp = same_day_group
    concat_pk = _populate_concat(
        grp["group_key"],
        grp["preprocessing_params_name"],
        motion="auto_default",
    )
    row = (ConcatenatedRecording & concat_pk).fetch1()
    abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])

    header = read_provenance_values(abs_path, CONCAT_PROVENANCE)
    assert header["concat_recording_id"] == str(
        concat_pk["concat_recording_id"]
    )
    # The RESOLVED preset (e.g. rigid_fast for auto same-day), and the producing
    # kwargs -- not the displacement field.
    assert header["motion_preset"] == row["motion_preset"]
    assert isinstance(header["motion_kwargs"], dict)

    members = sorted(
        read_long_provenance(abs_path, CONCAT_MEMBERS),
        key=lambda r: r["member_index"],
    )

    # Frame boundaries reproduce MemberBoundary (split_sorting_by_session's
    # arithmetic basis), contiguously partitioning the concat timeline.
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

    # The motion displacement field is NOT written (params only).
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        scratch_names = set(io.read().scratch.keys())
    assert not any("displacement" in name for name in scratch_names)


@pytest.mark.slow
def test_resolved_motion_preset_persisted(same_day_group):
    """A same-day concat built with ``preset='auto'`` stores the RESOLVED preset
    string (``'rigid_fast'``) on the row, not the unresolved ``'auto'`` alias --
    so the row records what motion correction actually ran.
    """
    from spyglass.spikesorting.v2._concat_recording import AUTO_SAME_DAY_PRESET
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    grp = same_day_group
    concat_pk = _populate_concat(
        grp["group_key"],
        grp["preprocessing_params_name"],
        motion="auto_default",
    )
    stored = (ConcatenatedRecording & concat_pk).fetch1("motion_preset")
    assert stored == AUTO_SAME_DAY_PRESET == "rigid_fast"


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
    concat_pk = _populate_concat(
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
def test_concatenated_recording_make_with_rigid_fast_motion(same_day_group):
    """Materializing with a REAL motion preset (rigid_fast) runs correct_motion
    on the concatenated segment and preserves the sample count, so the
    MemberBoundary back-mapping stays valid. Exercises the production motion
    branch (resolve_motion_correction -> real preset, job-kwargs resolution, the
    correct_motion call) and the post-motion sample-count invariant -- the
    other materialization tests use preset='none' and never run motion."""
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    grp = same_day_group
    concat_pk = _populate_concat(
        grp["group_key"],
        grp["preprocessing_params_name"],
        "rigid_fast_default",
    )

    n0, n1 = _member_sample_counts(grp)
    row = (ConcatenatedRecording & concat_pk).fetch1()
    fs = float(row["sampling_frequency"])
    # Motion correction is interpolation -> sample count preserved -> the
    # boundaries (built from PRE-motion member counts) remain valid.
    assert row["total_duration_s"] == pytest.approx((n0 + n1) / fs)
    ends = (ConcatenatedRecording.MemberBoundary & concat_pk).fetch(
        "end_sample", order_by="member_index"
    )
    assert [int(e) for e in ends] == [n0, n0 + n1]
    assert (
        ConcatenatedRecording().get_recording(concat_pk).get_num_samples()
        == n0 + n1
    )


@pytest.mark.slow
def test_concat_make_raises_on_motion_sample_count_drift(
    same_day_group, monkeypatch
):
    """If motion correction ever changed the sample count, make() raises
    (and persists nothing) rather than writing MemberBoundary rows that would
    misalign split_sorting_by_session's back-mapping."""
    import numpy as np
    import spikeinterface as si

    import spyglass.spikesorting.v2._concat_recording as concat_mod
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
    )

    def _drifted(
        recordings, *, motion_preset, preset_kwargs=None, job_kwargs=None
    ):
        # Simulate a motion step that changed the sample count.
        total = sum(int(r.get_num_samples()) for r in recordings)
        return si.NumpyRecording(
            [np.zeros((total + 1, 4), dtype=np.float32)],
            sampling_frequency=recordings[0].get_sampling_frequency(),
        )

    grp = same_day_group
    concat_pk = _concat_selection(
        grp["group_key"], grp["preprocessing_params_name"], "none"
    )
    monkeypatch.setattr(concat_mod, "build_concatenated_recording", _drifted)
    # Drive make_compute directly so the raw RuntimeError surfaces; it raises
    # before the NWB write (the guard precedes write_nwb_artifact), so nothing
    # is persisted and no analysis file orphans.
    fetched = ConcatenatedRecording().make_fetch(concat_pk)
    with pytest.raises(RuntimeError, match="sample count"):
        ConcatenatedRecording().make_compute(concat_pk, **fetched._asdict())
    assert not (ConcatenatedRecording & concat_pk)


@pytest.mark.slow
def test_motion_correction_preset_auto_rejects_multi_day(
    chronic_2_session_minirec,
):
    """A multi-day group with the 'auto' motion preset is rejected at
    materialization: 'auto' is single-day only, no silent DREDge dispatch."""
    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        SessionGroup,
    )
    from tests.spikesorting.v2._ingest_helpers import (
        clean_session_groups_for_owner,
    )

    sub = chronic_2_session_minirec
    owner, name = sub["owner"], "sg_multi_auto"

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
        concat_pk = _concat_selection(
            {"session_group_owner": owner, "session_group_name": name},
            "default",
            "auto_default",
        )
        # Drive make_compute directly so the raw ValueError surfaces (populate
        # would wrap it). resolve_motion_correction raises before any artifact
        # write; make_fetch resolves the (multi-day) status it checks against.
        fetched = ConcatenatedRecording().make_fetch(concat_pk)
        with pytest.raises(ValueError, match="single-day only"):
            ConcatenatedRecording().make_compute(concat_pk, **fetched._asdict())
        assert not (ConcatenatedRecording & concat_pk)
    finally:
        clean_session_groups_for_owner(owner)
        (Recording & next_day_rec_pk).super_delete(warn=False)
        (RecordingSelection & next_day_rec_pk).super_delete(warn=False)


# ---------- concat-backed Sorting end-to-end ------------------------------


@pytest.mark.slow
def test_concat_sort_end_to_end_and_split(same_day_group):
    """A concat-backed Sorting.populate runs end-to-end: it finds units,
    anchors its analysis NWB + per-unit electrodes to the first member,
    raises on concat brain regions without the anchor opt-in (and returns the
    anchor-member frame with it), and ``split_sorting_by_session`` returns one
    per-member sorting in each member's local frame with unit ids preserved."""
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2.exceptions import (
        ConcatBrainRegionAmbiguousError,
    )
    from spyglass.spikesorting.v2.recording import Recording
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

    # Parent anchoring: every Sorting.Unit electrode belongs to the FIRST
    # member's NWB (the deterministic anchor), not the second member's.
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

    # split_sorting_by_session: one entry per member, local frames, unit ids
    # preserved.
    sorting_obj = Sorting().get_analyzer(sort_pk).sorting
    split = ConcatenatedRecording().split_sorting_by_session(
        sorting_obj, concat_pk
    )
    members = grp["same_day_members"]
    # Keyed by the full member identity (nwb, sort_group_id, interval, team) so
    # two distinct members never collide; one entry per member.
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
        # Unit ids are preserved across every member.
        assert set(member_sorting.unit_ids) == all_unit_ids
        # Every spike falls within that member's local sample range.
        for unit_id in member_sorting.unit_ids:
            frames = member_sorting.get_unit_spike_train(unit_id=unit_id)
            if len(frames):
                assert int(frames.min()) >= 0
                assert int(frames.max()) < member_samples[nwb_file_name]

    # Merge-id resolution: a concat-backed curation resolves through
    # SortingSelection.ConcatenatedRecordingSource by concat_recording_id and
    # by the session-group fields, and combines with curation_id.
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    curation_key = CurationV2.insert_curation(sorting_key=sort_pk)
    merge_id = (SpikeSortingOutput.CurationV2 & curation_key).fetch1("merge_id")
    by_concat = SpikeSortingOutput().get_restricted_merge_ids(
        {"concat_recording_id": concat_pk["concat_recording_id"]},
        sources=["v2"],
    )
    assert merge_id in by_concat
    by_group = SpikeSortingOutput().get_restricted_merge_ids(
        {**grp["group_key"], "curation_id": curation_key["curation_id"]},
        sources=["v2"],
    )
    assert merge_id in by_group
    # A restriction with NO source key (only curation_id) must still include
    # the concat-backed curation -- a broad v2 query unions both source
    # families, it does not silently default to recording-source only.
    by_curation = SpikeSortingOutput().get_restricted_merge_ids(
        {"curation_id": curation_key["curation_id"]}, sources=["v2"]
    )
    assert merge_id in by_curation

    # Session-scoped downstream provenance resolves through the anchor member
    # for concat sorts (instead of raising): get_sort_metadata yields the first
    # member's nwb, and get_sort_group_info returns that member's electrodes.
    sorter, meta_nwb = CurationV2.get_sort_metadata(
        {"sorting_id": sort_pk["sorting_id"]}
    )
    assert sorter == "clusterless_thresholder"
    assert meta_nwb == first_nwb
    sg_info = CurationV2.get_sort_group_info(curation_key)
    sg_nwbs = set((sg_info).fetch("nwb_file_name"))
    assert sg_nwbs == {first_nwb}

    # Curated readback is source-aware: CurationV2.get_sorting reconstructs the
    # sorting against the concat timeline, and the merge dispatch's
    # get_recording returns the materialized ConcatenatedRecording cache (the
    # timeline the curated spike times live in) -- not a per-member Recording.
    curated_sorting = CurationV2.get_sorting(curation_key)
    assert set(curated_sorting.unit_ids) == all_unit_ids
    merge_recording = SpikeSortingOutput().get_recording({"merge_id": merge_id})
    concat_recording = ConcatenatedRecording().get_recording(concat_pk)
    assert (
        merge_recording.get_num_samples() == concat_recording.get_num_samples()
    )
    assert list(merge_recording.get_channel_ids()) == list(
        concat_recording.get_channel_ids()
    )

    # The concat brain-region anchor opt-in is reachable through the merge API:
    # default raises, allow_anchor_member=True returns anchor-member regions.
    with pytest.raises(ConcatBrainRegionAmbiguousError):
        SpikeSortingOutput().get_unit_brain_regions({"merge_id": merge_id})
    merge_regions = SpikeSortingOutput().get_unit_brain_regions(
        {"merge_id": merge_id}, allow_anchor_member=True
    )
    assert (merge_regions["region_resolution"] == "anchor_member").all()

    # Reporting / analyzer-curation provenance is concat-aware (no crash on the
    # empty RecordingSource join): the CurationEvaluation NWB anchor resolves to
    # the first member, and describe_units computes firing rate against the
    # concat recording's total_duration_s.
    from spyglass.spikesorting.v2.metric_curation import (
        _nwb_file_name_for_sorting,
    )
    from spyglass.spikesorting.v2.pipeline import describe_units

    assert (
        _nwb_file_name_for_sorting({"sorting_id": sort_pk["sorting_id"]})
        == first_nwb
    )
    units_df = describe_units(sort_pk["sorting_id"])
    assert len(units_df) == int(row["n_units"])
    # Firing rate is computed against the CONCAT recording's total_duration_s
    # (concat sorts have no artifact pass), not a per-member recording: assert
    # the exact denominator, not merely > 0, so a regression to any positive
    # duration would fail.
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

    Exercises the source-aware concat branch of ``make_fetch`` (it resolves the
    recording through ``ConcatenatedRecording.get_recording``, not a per-member
    ``Recording``) and the DB-free concat reconstruction in ``make_compute``;
    then accepts labels with ``use_evaluation_labels`` and re-evaluates the
    accepted child to prove the normal curation workflow works for concat-backed
    sorts.
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

    # The concat-backed evaluation NWB is self-identifying: it carries the
    # concat_recording_id (recording_id is None for a concat source) so the file
    # is interpretable standalone rather than only via the upstream join.
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._nwb_provenance import (
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
def test_unit_match_rejects_concat_backed_curation_member(same_day_group):
    """A concat-backed curation cannot be a UnitMatch member (per-member
    pinning is single-session only), and the rejection raises the member-
    validation exception type, not a bare ValueError.
    """
    from spyglass.spikesorting.v2 import unit_matching as um
    from spyglass.spikesorting.v2.exceptions import (
        UnitMatchSelectionIntegrityError,
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

    # The member-identity resolver rejects a concat sort; with the validator's
    # exc_class it raises UnitMatchSelectionIntegrityError (a RuntimeError),
    # matching the other member-validation failures rather than a bare
    # ValueError.
    with pytest.raises(UnitMatchSelectionIntegrityError, match="concat-backed"):
        um._curation_member_identity(
            sort_pk["sorting_id"],
            exc_class=UnitMatchSelectionIntegrityError,
        )
    # The default (standalone) call still raises a plain ValueError.
    with pytest.raises(ValueError, match="concat-backed"):
        um._curation_member_identity(sort_pk["sorting_id"])


@pytest.mark.slow
def test_preproc_restriction_includes_concat_and_recording_curations(
    same_day_group,
):
    """A ``{preprocessing_params_name}`` restriction matches BOTH source families.

    ``preprocessing_params_name`` lives on both ``RecordingSelection`` and
    ``ConcatenatedRecordingSelection``, so a bare preprocessing-recipe query
    must surface a recording-backed AND a concat-backed curation that share the
    recipe (the recording-only routing silently dropped the concat one), and
    ``{concat_recording_id, preprocessing_params_name}`` must be accepted as a
    valid concat restriction rather than rejected as contradictory.
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
    concat_merge_id = (SpikeSortingOutput.CurationV2 & concat_curation).fetch1(
        "merge_id"
    )

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
        assert concat_merge_id != rec_merge_id

        # The bare preprocessing-recipe restriction surfaces BOTH families.
        by_preproc = SpikeSortingOutput().get_restricted_merge_ids(
            {"preprocessing_params_name": preproc}, sources=["v2"]
        )
        assert concat_merge_id in by_preproc
        assert rec_merge_id in by_preproc

        # The cross-source branch genuinely FILTERS (rather than falling through
        # to "all sorts"): an unmatched recipe surfaces neither family.
        by_absent_preproc = SpikeSortingOutput().get_restricted_merge_ids(
            {"preprocessing_params_name": "no_such_preproc_recipe"},
            sources=["v2"],
        )
        assert concat_merge_id not in by_absent_preproc
        assert rec_merge_id not in by_absent_preproc

        # {concat_recording_id, preprocessing_params_name} is a valid pair (no
        # contradiction). The concat branch must APPLY the shared preproc filter,
        # not merely route on concat_recording_id: the matching recipe returns
        # the concat curation, a mismatched recipe excludes it, and it never
        # surfaces the recording-backed one.
        by_concat_and_preproc = SpikeSortingOutput().get_restricted_merge_ids(
            {
                "concat_recording_id": concat_pk["concat_recording_id"],
                "preprocessing_params_name": preproc,
            },
            sources=["v2"],
        )
        assert concat_merge_id in by_concat_and_preproc
        assert rec_merge_id not in by_concat_and_preproc
        by_concat_wrong_preproc = SpikeSortingOutput().get_restricted_merge_ids(
            {
                "concat_recording_id": concat_pk["concat_recording_id"],
                "preprocessing_params_name": "no_such_preproc_recipe",
            },
            sources=["v2"],
        )
        assert concat_merge_id not in by_concat_wrong_preproc

        # The post-union artifact filter still applies in the cross-source
        # branch: an artifact id neither (artifact-free) sort has excludes both,
        # while artifact_detection_id=None (no artifact pass) keeps both.
        by_preproc_unknown_artifact = (
            SpikeSortingOutput().get_restricted_merge_ids(
                {
                    "preprocessing_params_name": preproc,
                    "artifact_detection_id": str(uuid.uuid4()),
                },
                sources=["v2"],
            )
        )
        assert concat_merge_id not in by_preproc_unknown_artifact
        assert rec_merge_id not in by_preproc_unknown_artifact
        by_preproc_no_artifact = SpikeSortingOutput().get_restricted_merge_ids(
            {
                "preprocessing_params_name": preproc,
                "artifact_detection_id": None,
            },
            sources=["v2"],
        )
        assert concat_merge_id in by_preproc_no_artifact
        assert rec_merge_id in by_preproc_no_artifact
    finally:
        # ``same_day_group`` teardown only follows concat lineage, so the
        # recording-backed sort must be cleaned explicitly or its merge row
        # leaks into later broad preproc queries (order-dependent failures).
        clear_curations_for(rec_sort)
        (Sorting & rec_sort).super_delete(warn=False)
        (SortingSelection & rec_sort).super_delete(warn=False)


# ---------- memory / runtime measurement ----------------------------------


@pytest.mark.slow
def test_concat_materialization_memory_runtime_is_measured(same_day_group):
    """Real (psutil) peak-RSS + runtime measurement of the concat materialize
    path on the synthetic fixture. The bounds are generous (the minirec is
    tiny); the point is that the measurement is REAL and logged, so the
    ``--run-chronic`` gate reuses a proven measurement path rather than a
    mocked metric."""
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    grp = same_day_group
    with _PeakRSS() as measured:
        concat_pk = _populate_concat(
            grp["group_key"], grp["preprocessing_params_name"]
        )
    logger.info(
        "concat minirec materialization: peak_rss=%.0f MB, runtime=%.1f s",
        measured.peak_mb,
        measured.elapsed_s,
    )
    assert ConcatenatedRecording & concat_pk
    # Generous sanity bounds for a 2x5s 4-channel synthetic concat. The real
    # lab budget (peak < 8 GB, runtime < 10 min) lives in the --run-chronic
    # gate below.
    assert measured.peak_mb < 4096
    assert measured.elapsed_s < 300


@pytest.mark.slow
def test_concat_chronic_real_dataset_memory_runtime(request, dj_conn):
    """Opt-in real-chronic memory/runtime gate (``pytest --run-chronic``).

    Skipped by default. Given a real 1-hour, 30 kHz chronic NWB via
    ``SPIKESORTING_V2_CHRONIC_TEST_PATH``, ingests it, builds a same-day
    ``SessionGroup`` over its lowest sort group (one shank / ~32 channels for a
    polymer probe), and runs the Phase end-to-end path
    (``ConcatenatedRecording.populate`` then concat-backed ``Sorting.populate``).
    Asserts peak RSS < 8 GB and total runtime < 10 min on a 16-core machine,
    and logs the materialization vs sort timings separately so motion-correction
    vs sorter cost is visible. Reports timing + memory even on pass."""
    import os

    if not request.config.getoption("--run-chronic"):
        pytest.skip(
            "pass --run-chronic to run the real-chronic memory/runtime gate"
        )
    chronic_path = os.environ.get("SPIKESORTING_V2_CHRONIC_TEST_PATH")
    if not chronic_path:
        pytest.skip(
            "set SPIKESORTING_V2_CHRONIC_TEST_PATH to a real 1-hour chronic NWB"
        )

    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
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
            "rigid_fast_default",
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
def test_concat_applied_merge_through_downstream_and_evaluation(
    same_day_group, curation_evaluation_defaults
):
    """A committed APPLIED-MERGE on a CONCAT-backed sort flows through
    ``CurationV2.get_sorting``, ``SpikeSortingOutput.get_spike_times``, and
    ``CurationEvaluation`` carrying its MERGED unit set.

    Pins concat reconstruction + merged namespace together (the prior concern):
    the concat clusterless sort's unit count is data-dependent, so plant a
    deterministic 2-unit sort on the concat recording to host a real merge.
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
        mp.setattr(Sorting, "_run_sorter", staticmethod(_plant))
        if not (Sorting & sort_pk):
            Sorting.populate(sort_pk, reserve_jobs=False)
    finally:
        mp.undo()
    unit_ids = sorted(int(u) for u in (Sorting.Unit & sort_pk).fetch("unit_id"))
    assert len(unit_ids) >= 2, "planted concat sort must yield >=2 units"

    _drop_output_and_curations()
    try:
        merged = CurationV2.create_merged_curation(
            sorting_key, merge_groups=[[unit_ids[0], unit_ids[1]]]
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

        # SpikeSortingOutput downstream consumer dispatches on merge_id.
        merge_id = (SpikeSortingOutput.CurationV2 & merged).fetch1("merge_id")
        spike_times = SpikeSortingOutput().get_spike_times(
            {"merge_id": merge_id}
        )
        assert len(spike_times) == 1
        assert len(spike_times[0]) == n_spikes

        # CurationEvaluation over the MERGED CONCAT curation exercises the concat
        # reconstruction + merged-namespace analyzer together; metrics index the
        # merged unit.
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
def test_run_v2_pipeline_concat_mode_routes_session_group(same_day_group):
    """run_v2_pipeline concat mode: member recordings -> concat -> sort -> curation.

    With a concat preset (motion pinned -> concat-mode, no artifact stage), the
    orchestrator populates each member's Recording, concatenates them, sorts,
    and curates -- returning a concat-shaped manifest (concat_recording_id +
    member_recording_ids, no recording_id / artifact keys) registered on the
    merge table, and idempotent on rerun.
    """
    import spyglass.spikesorting.v2._pipeline_presets as presets_mod
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.pipeline import (
        register_pipeline_preset,
        run_v2_pipeline,
    )
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAM_NAME,
    )

    initialize_v2_defaults()
    _ensure_clusterless_sorter_params()
    grp = same_day_group
    preset_name = "test_concat_clusterless_smoke"
    # A concat preset (motion set -> concat-mode) using the default preproc +
    # smoke clusterless sorter the chronic minirec is known to sort. The "none"
    # motion row is a valid concat motion recipe (no correction).
    register_pipeline_preset(
        preset_name,
        {
            "preprocessing_params_name": grp["preprocessing_params_name"],
            "artifact_detection_params_name": None,
            "sorter": "clusterless_thresholder",
            "sorter_params_name": SMOKE_CLUSTERLESS_PARAM_NAME,
            "metric_params_name": "minimal",
            "auto_curation_rules_name": "none",
            "motion_correction_params_name": "none",
        },
        validate_rows=False,
    )
    try:
        summary = run_v2_pipeline(
            concat_session_group_owner=grp["group_key"]["session_group_owner"],
            concat_session_group_name=grp["group_key"]["session_group_name"],
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
        # Sorted + curated + registered on the merge table.
        assert summary["n_units"] >= 0
        assert SpikeSortingOutput.CurationV2 & {
            "merge_id": summary["root_merge_id"]
        }

        # Idempotent: a rerun reuses the concat sort + curation.
        rerun = run_v2_pipeline(
            concat_session_group_owner=grp["group_key"]["session_group_owner"],
            concat_session_group_name=grp["group_key"]["session_group_name"],
            pipeline_preset=preset_name,
        )
        assert rerun["sorting_id"] == summary["sorting_id"]
        assert rerun["root_merge_id"] == summary["root_merge_id"]
        assert rerun["concat_recording_status"] == "reused"
    finally:
        presets_mod._PIPELINE_PRESETS.pop(preset_name, None)
