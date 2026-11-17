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

# Computed at 9a50238fd, when the source was stored on part tables.
_PINNED_IDS = {
    "sort_recording": "eb79157c-f05c-5490-94e4-fb7502824ec7",
    "sort_recording_artifact": "05735f90-ae27-5d1c-8e7a-43a194974286",
    "sort_concat": "a614e306-d4d3-5ada-864e-f0aa259800d2",
    "sort_recording_corrected": "fe5db092-a3f8-5cc2-8022-9534c0995f76",
    "estimate_recording": "52f242cd-ca67-5070-80fd-1a1129f43f93",
    "estimate_concat": "82232a34-4144-52db-b797-5382c102d26a",
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
