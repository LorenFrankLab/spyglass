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


@pytest.mark.slow
def test_load_member_recordings_returns_aligned_counts_and_indices(
    same_day_group,
):
    """``_load_member_recordings`` loads each member's cached Recording in
    member_index order and returns sample counts / indices aligned element-wise
    with the loaded recordings -- the core per-member materialization contract,
    exercised without driving a full populate."""
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        SessionGroup,
    )

    grp = same_day_group
    members = (SessionGroup.Member & grp["group_key"]).fetch(
        as_dict=True, order_by="member_index"
    )
    recordings, sample_counts, member_indices = (
        ConcatenatedRecording._load_member_recordings(
            members, grp["preprocessing_params_name"]
        )
    )
    expected = _member_sample_counts(grp)
    assert member_indices == [0, 1]
    assert sample_counts == expected
    # The returned recordings are the same objects the counts were taken from.
    assert [int(r.get_num_samples()) for r in recordings] == expected


@pytest.mark.slow
def test_load_member_recordings_raises_on_missing_recording(same_day_group):
    """A member whose Recording cache is absent raises
    ``MissingRecordingForConcatError`` rather than silently dropping it (which
    would shift every later member's boundary)."""
    from spyglass.spikesorting.v2.exceptions import (
        MissingRecordingForConcatError,
    )
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        SessionGroup,
    )

    grp = same_day_group
    members = (SessionGroup.Member & grp["group_key"]).fetch(
        as_dict=True, order_by="member_index"
    )
    # No RecordingSelection exists under a bogus preprocessing recipe, so every
    # member is "missing" -- the defensive precondition fires.
    with pytest.raises(MissingRecordingForConcatError, match="not populated"):
        ConcatenatedRecording._load_member_recordings(
            members, "no_such_preprocessing_recipe"
        )


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
    a_pk = _populate_concat(
        grp["group_key"], grp["preprocessing_params_name"]
    )
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

    def _drifted(recordings, *, motion_preset, preset_kwargs=None, job_kwargs=None):
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
    # Call make() directly so the raw RuntimeError surfaces; it raises before
    # the NWB write, so nothing is persisted and no analysis file orphans.
    with pytest.raises(RuntimeError, match="sample count"):
        ConcatenatedRecording().make(concat_pk)
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
        # Call make() directly so the raw ValueError surfaces (populate would
        # wrap it). It raises before writing any artifact.
        with pytest.raises(ValueError, match="single-day only"):
            ConcatenatedRecording().make(concat_pk)
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
    regions = Sorting().get_unit_brain_regions(sort_pk, allow_anchor_member=True)
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
    merge_recording = SpikeSortingOutput().get_recording(
        {"merge_id": merge_id}
    )
    concat_recording = ConcatenatedRecording().get_recording(concat_pk)
    assert (
        merge_recording.get_num_samples()
        == concat_recording.get_num_samples()
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
    # empty RecordingSource join): the AnalyzerCuration NWB anchor resolves to
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
    concat_total_duration_s = (
        ConcatenatedRecording & concat_pk
    ).fetch1("total_duration_s")
    assert (units_df["firing_rate_hz"] > 0).all()
    assert units_df["firing_rate_hz"].to_numpy() == pytest.approx(
        units_df["n_spikes"].to_numpy() / concat_total_duration_s
    )


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
