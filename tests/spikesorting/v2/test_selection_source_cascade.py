"""Deleting a selection's source removes every selection made from it.

A ``SortingSelection`` or ``MotionEstimateSelection`` row names its source
recording or concatenation, its optional artifact detection and, for a sort,
its optional motion-corrected recording. A delete that reaches the selection
through any of these must remove the whole row, so no selection outlives its
source. Each test builds and deletes its own recordings on the drift session.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from tests.spikesorting.v2._motion_db_helpers import (
    MOTION_TEAM,
    masked_artifact,
    populated_corrected,
    populated_estimate,
    session_start_s,
    sorter_key,
)
from tests.spikesorting.v2._sorter_stub import plant_sorter

pytestmark = [pytest.mark.slow, pytest.mark.integration]


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
    import spikeinterface as si

    samples = np.array([500, 1500, 2500, 3500, 4500], dtype=np.int64)
    return si.NumpySorting.from_samples_and_labels(
        samples_list=[samples],
        labels_list=[np.zeros(samples.size, dtype=np.int32)],
        sampling_frequency=recording.get_sampling_frequency(),
    )


def _populate_sorts(sort_keys) -> None:
    from spyglass.spikesorting.v2.sorting import Sorting

    mp = pytest.MonkeyPatch()
    try:
        plant_sorter(mp, _plant)
        Sorting.populate(sort_keys, reserve_jobs=False)
    finally:
        mp.undo()
    assert len(Sorting & sort_keys) == len(sort_keys)


def _member_run(nwb_file_name, name, start_s, end_s) -> dict:
    """Run inputs for a recording of ``[start_s, end_s)`` after session start."""
    from spyglass.common import IntervalList
    from tests.spikesorting.v2._ingest_helpers import configure_v2_run_inputs

    t0 = session_start_s(nwb_file_name)
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": name,
            "valid_times": np.array([[t0 + start_s, t0 + end_s]]),
            "pipeline": "selection_source_cascade_test",
        },
        skip_duplicates=True,
    )
    return configure_v2_run_inputs(
        nwb_file_name, MOTION_TEAM, interval_list_name=name
    )


def _recording(nwb_file_name, name, start_s=1.0, end_s=9.0) -> dict:
    """A populated ``Recording`` of its own interval; returns its key."""
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )

    run = _member_run(nwb_file_name, name, start_s, end_s)
    key = RecordingSelection.insert_selection(
        {**run, "preprocessing_params_name": "default"}
    )
    if not (Recording & key):
        Recording.populate(key, reserve_jobs=False)
    return key


def _concat(nwb_file_name, group_name) -> dict:
    """A populated two-member ``ConcatenatedRecording``; returns its key."""
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        SessionGroup,
    )
    from tests.spikesorting.v2._concat_helpers import select_unmasked_concat

    members = [
        _member_run(nwb_file_name, f"{group_name} a", 2.0, 12.0),
        _member_run(nwb_file_name, f"{group_name} b", 14.0, 26.0),
    ]
    for run in members:
        key = RecordingSelection.insert_selection(
            {**run, "preprocessing_params_name": "default"}
        )
        if not (Recording & key):
            Recording.populate(key, reserve_jobs=False)
    SessionGroup.create_group(MOTION_TEAM, group_name, members)
    concat_key = select_unmasked_concat(
        {
            "session_group_owner": MOTION_TEAM,
            "session_group_name": group_name,
            "preprocessing_params_name": "default",
        }
    )
    ConcatenatedRecording.populate(concat_key, reserve_jobs=False)
    return concat_key


def _sort(**source) -> dict:
    from spyglass.spikesorting.v2.sorting import SortingSelection

    return SortingSelection.insert_selection({**source, **sorter_key()})


def _estimate(**source) -> dict:
    from spyglass.spikesorting.v2.motion import MotionEstimateSelection

    return MotionEstimateSelection.insert_selection(
        {"motion_estimation_params_name": "dredge_fast_v1", **source}
    )


def _outcome(call) -> str:
    """What ``call()`` does: ``"ok"`` or the exception it raises."""
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - recorded, not handled
        return f"{type(exc).__name__}: {str(exc)[:300]}"
    return "ok"


def _drop_survivors(table, keys) -> None:
    """Delete rows a damaging cascade left, so module teardown can run."""
    keys = [key for key in keys if table & key]
    if keys:
        (table & keys).super_delete(
            warn=False, safemode=False, force_masters=True
        )


@pytest.fixture(autouse=True)
def _drop_session_groups(drift_recording):
    """Delete the session groups a test made, before the session teardown.

    The session cleanup deletes ``RecordingSelection`` rows without
    ``force_masters``, which refuses to reach ``SessionGroup.Member`` first.
    """
    from spyglass.spikesorting.v2.session_group import SessionGroup

    before = set(
        (SessionGroup & {"session_group_owner": MOTION_TEAM}).fetch(
            "session_group_name"
        )
    )
    yield
    made = [
        {"session_group_owner": MOTION_TEAM, "session_group_name": name}
        for name in (SessionGroup & {"session_group_owner": MOTION_TEAM}).fetch(
            "session_group_name"
        )
        if name not in before
    ]
    _drop_survivors(SessionGroup, made)


def _remaining(table, keys) -> list:
    """The keys among ``keys`` that still have a row in ``table``."""
    return [key for key in keys if table & key]


def test_recording_delete_removes_every_sort_of_it(drift_recording):
    """A sort with an artifact detection and one without both go.

    The two sorts reach ``SortingSelection`` through different delete
    branches (the recording itself and its artifact detection).
    """
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    nwb = drift_recording["nwb_file_name"]
    recording_key = _recording(nwb, "cascade sorts")
    t0 = session_start_s(nwb)
    artifact_key = masked_artifact(recording_key, [t0 + 3.0, t0 + 3.5])
    rec = {"recording_id": recording_key["recording_id"]}
    masked_sort = _sort(**rec, **artifact_key)
    plain_sort = _sort(**rec)
    _populate_sorts([masked_sort])

    (Recording & recording_key).delete(safemode=False)

    assert not (Recording & recording_key)
    assert _remaining(SortingSelection, [masked_sort, plain_sort]) == []
    assert _remaining(Sorting, [masked_sort, plain_sort]) == []


def test_recording_delete_removes_every_motion_estimate_of_it(
    drift_recording,
):
    """Estimates with and without an artifact detection both go."""
    from spyglass.spikesorting.v2.motion import MotionEstimateSelection
    from spyglass.spikesorting.v2.recording import Recording

    nwb = drift_recording["nwb_file_name"]
    recording_key = _recording(nwb, "cascade estimates")
    t0 = session_start_s(nwb)
    artifact_key = masked_artifact(recording_key, [t0 + 3.0, t0 + 3.5])
    rec = {"recording_id": recording_key["recording_id"]}
    masked = _estimate(**rec, **artifact_key)
    plain = _estimate(**rec)

    (Recording & recording_key).delete(safemode=False)

    assert not (Recording & recording_key)
    assert _remaining(MotionEstimateSelection, [masked, plain]) == []


def test_concat_delete_removes_every_sort_and_estimate_of_it(drift_recording):
    """Sorts of the concat (one reading its corrected traces) and the estimate.

    The corrected sort is also reached through the estimate's corrected
    recording, a second delete branch into ``SortingSelection``.
    """
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecordingSelection,
        MotionEstimateSelection,
    )
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording
    from spyglass.spikesorting.v2.sorting import SortingSelection

    concat_key = _concat(drift_recording["nwb_file_name"], "cascade concat")
    concat = {"concat_recording_id": concat_key["concat_recording_id"]}
    estimate = populated_estimate(**concat)
    corrected_key = populated_corrected(estimate)
    corrected_sort = _sort(**concat, **corrected_key)
    plain_sort = _sort(**concat)

    (ConcatenatedRecording & concat_key).delete(safemode=False)

    assert not (ConcatenatedRecording & concat_key)
    assert _remaining(SortingSelection, [corrected_sort, plain_sort]) == []
    assert _remaining(MotionEstimateSelection, [estimate]) == []
    assert _remaining(MotionCorrectedRecordingSelection, [corrected_key]) == []


def test_artifact_delete_removes_only_the_selections_that_used_it(
    drift_recording,
):
    """Selections without the detection, and the recording, stay."""
    from spyglass.spikesorting.v2.artifact import RecordingArtifactDetection
    from spyglass.spikesorting.v2.motion import MotionEstimateSelection
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import SortingSelection

    nwb = drift_recording["nwb_file_name"]
    recording_key = _recording(nwb, "cascade artifact")
    t0 = session_start_s(nwb)
    artifact_key = masked_artifact(recording_key, [t0 + 3.0, t0 + 3.5])
    rec = {"recording_id": recording_key["recording_id"]}
    masked_sort, plain_sort = _sort(**rec, **artifact_key), _sort(**rec)
    masked_estimate = _estimate(**rec, **artifact_key)
    plain_estimate = _estimate(**rec)

    (RecordingArtifactDetection & artifact_key).delete(safemode=False)

    assert _remaining(SortingSelection, [masked_sort, plain_sort]) == [
        plain_sort
    ]
    assert _remaining(
        MotionEstimateSelection, [masked_estimate, plain_estimate]
    ) == [plain_estimate]
    assert Recording & recording_key
    # The kept rows still resolve to their source.
    assert SortingSelection.resolve_source(plain_sort).key == rec
    assert MotionEstimateSelection.resolve_source(plain_estimate).key == rec


# ---- delete permission -------------------------------------------------------

_OWNER = "Cascade Owner"
_OUTSIDER = "Cascade Outsider"
_OUTSIDER_USER = "cascade_outsider"


def _as_outsider(nwb_file_name):
    """Make ``_OWNER`` the session's experimenter and add a non-team member.

    Returns a callable that restores the session's experimenters.
    """
    from spyglass.common import LabMember, Session

    LabMember.insert1(
        {
            "lab_member_name": _OWNER,
            "first_name": "Cascade",
            "last_name": "Owner",
        },
        skip_duplicates=True,
    )
    LabMember.insert1(
        {
            "lab_member_name": _OUTSIDER,
            "first_name": "Cascade",
            "last_name": "Outsider",
        },
        skip_duplicates=True,
    )
    LabMember.LabMemberInfo.insert1(
        {
            "lab_member_name": _OUTSIDER,
            "google_user_name": _OUTSIDER,
            "datajoint_user_name": _OUTSIDER_USER,
            "admin": 0,
        },
        skip_duplicates=True,
    )
    experimenter = {"nwb_file_name": nwb_file_name, "lab_member_name": _OWNER}
    added = not (Session.Experimenter & experimenter)
    Session.Experimenter.insert1(experimenter, skip_duplicates=True)

    def restore():
        if added:
            (Session.Experimenter & experimenter).delete_quick()

    return restore


def _permission_outcomes(rows) -> dict:
    """``_check_delete_permission`` as the outsider, per table name.

    Returns ``"refused"`` (with the message) or ``"allowed"`` for each.
    """
    import datajoint as dj

    outcomes = {}
    user = dj.config["database.user"]
    dj.config["database.user"] = _OUTSIDER_USER
    try:
        for name, query in rows.items():
            try:
                query._check_delete_permission()
            except PermissionError as exc:
                outcomes[name] = ("refused", str(exc))
            else:
                outcomes[name] = ("allowed", None)
    finally:
        dj.config["database.user"] = user
    return outcomes


def test_non_team_member_cannot_delete_a_single_session_sort(
    drift_recording,
):
    """The sort's selection, sort and curation all trace back to the session.

    ``Session -> ... -> Recording -> SortingSelection`` is a directed path,
    so the check finds the session's experimenter and refuses a lab member
    who shares no team with them.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    nwb = drift_recording["nwb_file_name"]
    recording_key = _recording(nwb, "cascade permission")
    sort_key = _sort(recording_id=recording_key["recording_id"])
    _populate_sorts([sort_key])
    curation_key = CurationV2.insert_curation(sorting_key=sort_key)

    restore = _as_outsider(nwb)
    try:
        outcomes = _permission_outcomes(
            {
                "SortingSelection": SortingSelection & sort_key,
                "Sorting": Sorting & sort_key,
                "CurationV2": CurationV2 & curation_key,
            }
        )
    finally:
        restore()

    for name, (outcome, message) in outcomes.items():
        assert outcome == "refused", f"{name}: the outsider passed the check"
        assert "is not on a team with" in message, (name, message)


def test_concat_sort_permission_check_records_issue_1725(drift_recording):
    """A concat sort still passes the check, with a warning (issue #1725).

    No directed path reaches ``SortingSelection`` from ``Session`` through a
    concatenation: ``ConcatenatedRecordingSelection`` names a session group,
    and its members' sessions sit on part tables. The permission check
    follows a single path and finds no session, so it warns and allows the
    delete. This pins that known limitation; it is not the intended outcome.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    nwb = drift_recording["nwb_file_name"]
    concat_key = _concat(nwb, "cascade permission concat")
    sort_key = _sort(concat_recording_id=concat_key["concat_recording_id"])
    _populate_sorts([sort_key])
    curation_key = CurationV2.insert_curation(sorting_key=sort_key)

    restore = _as_outsider(nwb)
    try:
        outcomes = _permission_outcomes(
            {
                "SortingSelection": SortingSelection & sort_key,
                "Sorting": Sorting & sort_key,
                "CurationV2": CurationV2 & curation_key,
            }
        )
    finally:
        restore()

    assert outcomes == {
        "SortingSelection": ("allowed", None),
        "Sorting": ("allowed", None),
        "CurationV2": ("allowed", None),
    }


# ---- ids do not depend on how the source is stored ---------------------------

# The sorting ids match those from when the source was on part tables. The
# motion ids (and the corrected sort) also include the source's content_hash.
_PINNED_IDS = {
    "sort_recording": "eb79157c-f05c-5490-94e4-fb7502824ec7",
    "sort_recording_artifact": "05735f90-ae27-5d1c-8e7a-43a194974286",
    "sort_concat": "a614e306-d4d3-5ada-864e-f0aa259800d2",
    "sort_recording_corrected": "5f771e28-d2d2-5c3e-b215-fa8e985717f7",
    "estimate_recording": "7c651dfb-5e18-591e-b4aa-4d071b7245de",
    "estimate_concat": "16ff004d-357e-5dca-997f-ce871db76737",
}


def test_selection_ids_match_the_pinned_values(
    drift_recording, discontinuous_sources
):
    """Each selection kind gets the id it got with the source on part tables."""
    recording_key = drift_recording["recording_key"]
    rec = {"recording_id": recording_key["recording_id"]}
    concat = {
        "concat_recording_id": discontinuous_sources["concat_key"][
            "concat_recording_id"
        ]
    }
    t0 = session_start_s(drift_recording["nwb_file_name"])
    artifact_key = masked_artifact(recording_key, [t0 + 5.0, t0 + 5.5])
    estimate = populated_estimate(**rec)
    corrected_key = populated_corrected(estimate)

    ids = {
        "sort_recording": _sort(**rec)["sorting_id"],
        "sort_recording_artifact": _sort(**rec, **artifact_key)["sorting_id"],
        "sort_concat": _sort(**concat)["sorting_id"],
        "sort_recording_corrected": _sort(**rec, **corrected_key)["sorting_id"],
        "estimate_recording": estimate["motion_estimate_id"],
        "estimate_concat": _estimate(**concat)["motion_estimate_id"],
    }
    assert {name: uuid.UUID(str(v)) for name, v in ids.items()} == {
        name: uuid.UUID(v) for name, v in _PINNED_IDS.items()
    }


# ---- other masters the same cascade can damage --------------------------------
#
# A delete with ``force_masters=True`` (Spyglass's default) deletes a master
# only through the first of its parts the cascade reaches; a later branch into
# another part of the same master deletes part rows only (DataJoint
# ``Table.delete``, ``visited_masters``). These tests record where that still
# leaves a master without some of its parts.


def _concat_of(group_name, runs, artifact_ids) -> dict:
    """A populated concat of existing member recordings; returns its key."""
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
        SessionGroup,
    )

    SessionGroup.create_group(MOTION_TEAM, group_name, runs)
    concat_key = ConcatenatedRecordingSelection.insert_selection(
        {
            "session_group_owner": MOTION_TEAM,
            "session_group_name": group_name,
            "preprocessing_params_name": "default",
        },
        artifact_detection_ids=artifact_ids,
    )
    ConcatenatedRecording.populate(concat_key, reserve_jobs=False)
    return concat_key


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A force_masters cascade deletes a master only through the first of "
        "its parts it reaches (DataJoint visited_masters): the artifact "
        "branch deletes the masked concat, the Recording branch then deletes "
        "only the plain concat's MemberSnapshot row for R."
    ),
)
def test_member_recording_delete_removes_every_concat_of_it(drift_recording):
    """Two concatenations share member R, one masking it, one not.

    The masked concat's ``MemberSnapshot`` is also reached through R's
    artifact detection, a second branch into the same master.
    """
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
    )

    nwb = drift_recording["nwb_file_name"]
    member_r = _recording(nwb, "snapshot r", 2.0, 8.0)
    _recording(nwb, "snapshot y", 12.0, 18.0)
    runs = [
        _member_run(nwb, "snapshot r", 2.0, 8.0),
        _member_run(nwb, "snapshot y", 12.0, 18.0),
    ]
    t0 = session_start_s(nwb)
    artifact_key = masked_artifact(member_r, [t0 + 4.0, t0 + 4.5])
    masked = _concat_of(
        "snapshot masked",
        runs,
        {0: artifact_key["artifact_detection_id"], 1: None},
    )
    plain = _concat_of("snapshot plain", runs, {0: None, 1: None})

    (Recording & member_r).delete(safemode=False)

    surviving = _remaining(ConcatenatedRecordingSelection, [masked, plain])
    try:
        report = {
            str(key["concat_recording_id"]): {
                "member rows (of 2)": len(
                    ConcatenatedRecordingSelection.MemberSnapshot & key
                ),
                "ConcatenatedRecording row": bool(ConcatenatedRecording & key),
                "make_fetch": _outcome(
                    lambda: ConcatenatedRecording().make_fetch(key)
                ),
                "get_recording": _outcome(
                    lambda: ConcatenatedRecording().get_recording(key)
                ),
                "sort selection": _outcome(
                    lambda: _sort(
                        concat_recording_id=key["concat_recording_id"]
                    )
                ),
            }
            for key in surviving
        }
        assert surviving == [], f"concats outlived member R: {report}"
    finally:
        _drop_survivors(ConcatenatedRecordingSelection, surviving)


def _plant_shared_detection(group_name, recording_keys) -> dict:
    """Rows of a shared detection over ``recording_keys``, registered in
    ``ArtifactDetectionOutput``. No traces are read; the cascade only needs
    the rows and their foreign keys."""
    import datajoint as dj

    from spyglass.spikesorting.v2.artifact import (
        SharedArtifactDetection,
        SharedArtifactGroup,
        SharedArtifactSelection,
    )
    from spyglass.spikesorting.v2.artifact_output import (
        ArtifactDetectionOutput,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection

    nwb = (RecordingSelection & recording_keys[0]).fetch1("nwb_file_name")
    detection_id, merge_id = uuid.uuid4(), uuid.uuid4()
    dj.Table.insert(
        SharedArtifactGroup(),
        [{"shared_artifact_group_name": group_name, "nwb_file_name": nwb}],
    )
    dj.Table.insert(
        SharedArtifactGroup.Member(),
        [
            {"shared_artifact_group_name": group_name, **key}
            for key in recording_keys
        ],
    )
    dj.Table.insert(
        SharedArtifactSelection(),
        [
            {
                "artifact_detection_id": detection_id,
                "artifact_detection_params_name": "none",
                "shared_artifact_group_name": group_name,
                "member_set_hash": "0" * 64,
            }
        ],
    )
    dj.Table.insert(
        SharedArtifactDetection(),
        [{"artifact_detection_id": detection_id}],
        allow_direct_insert=True,
    )
    dj.Table.insert(
        ArtifactDetectionOutput(),
        [{"merge_id": merge_id, "source": "SharedArtifactDetection"}],
    )
    dj.Table.insert(
        ArtifactDetectionOutput.SharedArtifactDetection(),
        [{"merge_id": merge_id, "artifact_detection_id": detection_id}],
    )
    return {"merge_id": merge_id, "artifact_detection_id": detection_id}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A force_masters cascade deletes a master only through the first of "
        "its parts it reaches (DataJoint visited_masters): the second branch "
        "into ArtifactDetectionOutput deletes only a part row, leaving a "
        "merge row with no part that a sort of R2 still references."
    ),
)
def test_recording_delete_leaves_no_artifact_output_without_its_part(
    drift_recording,
):
    """R has its own detection and is in a shared group with R2; a sort of R2
    uses the shared detection. Deleting R reaches ``ArtifactDetectionOutput``
    through both of its parts."""
    from spyglass.spikesorting.v2.artifact_output import (
        ArtifactDetectionOutput,
    )
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import SortingSelection

    nwb = drift_recording["nwb_file_name"]
    member_r = _recording(nwb, "shared r", 2.0, 8.0)
    member_r2 = _recording(nwb, "shared r2", 12.0, 18.0)
    t0 = session_start_s(nwb)
    own = masked_artifact(member_r, [t0 + 4.0, t0 + 4.5])
    own_merge = {"merge_id": ArtifactDetectionOutput.get_merge_id(own)}
    shared = _plant_shared_detection("cascade shared", [member_r, member_r2])
    sort_r2 = {"sorting_id": uuid.uuid4()}
    SortingSelection.insert1(
        {
            **sort_r2,
            **sorter_key(),
            **member_r2,
            "artifact_detection_merge_id": shared["merge_id"],
        },
        allow_direct_insert=True,
    )

    (Recording & member_r).delete(safemode=False)

    merges = [own_merge, {"merge_id": shared["merge_id"]}]
    partless = [
        key
        for key in _remaining(ArtifactDetectionOutput, merges)
        if not any(
            part & key
            for part in (
                ArtifactDetectionOutput.RecordingArtifactDetection,
                ArtifactDetectionOutput.SharedArtifactDetection,
            )
        )
    ]
    try:
        report = {
            "partless merge rows": partless,
            "sort of R2 kept": bool(SortingSelection & sort_r2),
            "resolve_source of that sort": _outcome(
                lambda: SortingSelection.resolve_source(sort_r2)
            ),
        }
        assert partless == [], f"partless ArtifactDetectionOutput: {report}"
    finally:
        _drop_survivors(SortingSelection, [sort_r2])
        _drop_survivors(ArtifactDetectionOutput, partless)


def _curated_sorts_over_r_and_concat(nwb, prefix):
    """A curated sort of R and a curated sort of a concat (R, Y), with the
    concat's member curations populated."""
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCurationV2,
    )
    from spyglass.spikesorting.v2.curation import CurationV2

    member_r = _recording(nwb, f"{prefix} r", 2.0, 8.0)
    _recording(nwb, f"{prefix} y", 12.0, 18.0)
    runs = [
        _member_run(nwb, f"{prefix} r", 2.0, 8.0),
        _member_run(nwb, f"{prefix} y", 12.0, 18.0),
    ]
    concat_key = _concat_of(f"{prefix} concat", runs, {0: None, 1: None})
    sort_r = _sort(**member_r)
    sort_c = _sort(concat_recording_id=concat_key["concat_recording_id"])
    _populate_sorts([sort_r, sort_c])
    curation_r = CurationV2.insert_curation(sorting_key=sort_r)
    curation_c = CurationV2.insert_curation(sorting_key=sort_c)
    ConcatMemberCurationV2.populate(curation_c, reserve_jobs=False)
    return member_r, curation_r, curation_c


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A force_masters cascade deletes a master only through the first of "
        "its parts it reaches (DataJoint visited_masters): the second branch "
        "into SpikeSortingOutput deletes only a part row, leaving a merge "
        "row with no part."
    ),
)
def test_recording_delete_leaves_no_spike_sorting_output_without_its_part(
    drift_recording,
):
    """R is sorted on its own and as a concat member; deleting R reaches
    ``SpikeSortingOutput`` through its ``CurationV2`` and
    ``ConcatMemberCurationV2`` parts."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.recording import Recording

    member_r, curation_r, curation_c = _curated_sorts_over_r_and_concat(
        drift_recording["nwb_file_name"], "merge"
    )
    v2_parts = (
        SpikeSortingOutput.CurationV2,
        SpikeSortingOutput.ConcatMemberCurationV2,
    )
    merges = [
        {"merge_id": merge_id}
        for part in v2_parts
        for merge_id in (part & [curation_r, curation_c]).fetch("merge_id")
    ]
    # R's curation and the two concat members; a concat curation itself is
    # registered through its members only.
    assert len(merges) == 3, merges

    (Recording & member_r).delete(safemode=False)

    partless = [
        key
        for key in _remaining(SpikeSortingOutput, merges)
        if not any(
            part & key for part in SpikeSortingOutput.parts(as_objects=True)
        )
    ]
    try:
        assert partless == [], f"partless SpikeSortingOutput: {partless}"
    finally:
        _drop_survivors(SpikeSortingOutput, partless)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A force_masters cascade deletes a master only through the first of "
        "its parts it reaches (DataJoint visited_masters): the match over "
        "the concat is reached later through Input and InputRecording, "
        "which lose their rows while the UnitMatchSelection row stays."
    ),
)
def test_recording_delete_removes_every_match_of_it(drift_recording):
    """One match pins a curation of R's own sort, another a curation of a
    concat with R as a member; deleting R reaches ``UnitMatchSelection``
    through ``Input`` (by both curations) and ``InputRecording``."""
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        UnitMatchSelection,
    )

    member_r, curation_r, curation_c = _curated_sorts_over_r_and_concat(
        drift_recording["nwb_file_name"], "match"
    )
    MatcherParameters.insert_default()
    matches = [
        UnitMatchSelection.insert_inputs(
            [curation],
            "unitmatch_default",
            unit_policy="v2_unflagged_single_units",
        )
        for curation in (curation_r, curation_c)
    ]

    (Recording & member_r).delete(safemode=False)

    surviving = _remaining(UnitMatchSelection, matches)
    try:
        parts = {
            str(key["unitmatch_id"]): (
                len(UnitMatchSelection.Input & key),
                len(UnitMatchSelection.InputRecording & key),
            )
            for key in surviving
        }
        assert surviving == [], f"matches outlived R: {parts}"
    finally:
        _drop_survivors(UnitMatchSelection, surviving)
