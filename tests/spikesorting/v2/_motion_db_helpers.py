"""Shared DB helpers for the motion-correction test modules.

Names, row builders and leaves-first cleanup used by
``test_motion_correction.py`` and ``test_motion_consumers.py``; the
module-scoped ``drift_recording`` / ``discontinuous_sources`` fixtures that use
them live in ``conftest.py``. Schema tables are imported inside each function,
so importing this module declares no schema.
"""

from __future__ import annotations

import numpy as np

MOTION_TEAM = "motion_estimate_team"
DRIFT_NWB = "motion_drift_polymer.nwb"
DRIFT_DURATION_S = 30.0
MEMBER_A_INTERVAL = "motion member a"
MEMBER_B_INTERVAL = "motion member b"
CONCAT_GROUP = "motion_concat"


def drop_sorts_of_estimates(estimate_keys) -> None:
    """Delete every sort that reads a corrected recording of these estimates.

    Run before deleting the estimates: their cascade reaches
    ``SortingSelection.MotionCorrectionSource``, which DataJoint refuses to
    delete before its master ("part before master"). A test that fails before
    its own cleanup would otherwise turn the module teardown into that error.
    """
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    if not estimate_keys:
        return
    corrected = (MotionCorrectedRecordingSelection & estimate_keys).proj()
    drop_pipeline_sorts(
        (SortingSelection.MotionCorrectionSource & corrected).fetch(
            "sorting_id"
        )
    )


def drop_motion_selections(source_key) -> None:
    """Delete every motion-estimate selection on a source (masters first),
    after the sorts that read their corrected recordings.

    ``source_key`` is a ``{"recording_id": ...}`` or
    ``{"concat_recording_id": ...}`` restriction.
    """
    from spyglass.spikesorting.v2.motion import MotionEstimateSelection

    source_part = (
        MotionEstimateSelection.ConcatenatedRecordingSource
        if "concat_recording_id" in source_key
        else MotionEstimateSelection.RecordingSource
    )
    keys = (source_part & source_key).fetch("KEY", as_dict=True)
    if keys:
        drop_sorts_of_estimates(keys)
        (MotionEstimateSelection & keys).super_delete(
            warn=False, safemode=False
        )


def session_start_s(nwb_file_name) -> float:
    """The session's first raw valid time, in seconds."""
    from spyglass.common import IntervalList

    valid = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    return float(valid[0][0])


def populated_estimate(**source) -> dict:
    """The ``dredge_fast_v1`` estimate of a source, populated if missing."""
    from spyglass.spikesorting.v2.motion import (
        MotionEstimate,
        MotionEstimateSelection,
    )

    key = MotionEstimateSelection.insert_selection(
        {"motion_estimation_params_name": "dredge_fast_v1", **source}
    )
    if not (MotionEstimate & key):
        MotionEstimate.populate(key, reserve_jobs=False)
    return key


def select_corrected(
    estimate_key, interpolation="kriging_force_extrapolate_v1"
) -> dict:
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecordingSelection,
        MotionInterpolationParameters,
    )

    MotionInterpolationParameters.insert_default()
    return MotionCorrectedRecordingSelection.insert_selection(
        {
            "motion_estimate_id": estimate_key["motion_estimate_id"],
            "motion_interpolation_params_name": interpolation,
        }
    )


def populated_corrected(
    estimate_key, interpolation="kriging_force_extrapolate_v1"
) -> dict:
    from spyglass.spikesorting.v2.motion import MotionCorrectedRecording

    key = select_corrected(estimate_key, interpolation)
    if not (MotionCorrectedRecording & key):
        MotionCorrectedRecording.populate(key, reserve_jobs=False)
    return key


def drop_corrected(key) -> None:
    """Delete a corrected recording row with its analysis file row and file.

    Leaves no orphaned ``AnalysisNwbfile`` row or file behind, so a test can
    repopulate the same selection.
    """
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.motion import MotionCorrectedRecording

    names = (MotionCorrectedRecording & key).fetch("analysis_file_name")
    paths = [AnalysisNwbfile.get_abs_path(name) for name in names]
    (MotionCorrectedRecording & key).delete_quick()
    for name, path in zip(names, paths):
        (AnalysisNwbfile & {"analysis_file_name": name}).delete_quick()
        Path(path).unlink(missing_ok=True)


def sorter_key() -> dict:
    """The smoke clusterless row: a fast real sorter with no own correction."""
    from spyglass.spikesorting.v2.sorting import SorterParameters
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAM_NAME,
        SMOKE_CLUSTERLESS_PARAMS,
    )

    SorterParameters().insert1(
        {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": SMOKE_CLUSTERLESS_PARAM_NAME,
            "params": dict(SMOKE_CLUSTERLESS_PARAMS),
            "params_schema_version": 4,
            "job_kwargs": None,
        },
        skip_duplicates=True,
    )
    return {
        "sorter": "clusterless_thresholder",
        "sorter_params_name": SMOKE_CLUSTERLESS_PARAM_NAME,
    }


def masked_artifact(recording_key, excluded_s) -> dict:
    """A populated manual-exclusion artifact detection on a recording."""
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )

    artifact_key = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": recording_key["recording_id"],
            "artifact_detection_params_name": "none",
            "manual_excluded_times": np.array([excluded_s]),
        }
    )
    RecordingArtifactDetection.populate(artifact_key, reserve_jobs=False)
    return artifact_key


def drop_sorts(sort_keys) -> None:
    """Delete sorts (analyzer folders included) and their selections."""
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    sort_keys = [key for key in sort_keys if key]
    if not sort_keys:
        return
    if Sorting & sort_keys:
        (Sorting & sort_keys).delete(safemode=False)
    (SortingSelection & sort_keys).super_delete(warn=False, safemode=False)


def drop_pipeline_sorts(sorting_ids) -> None:
    """Delete run_v2_pipeline sorts leaves-first: match selections pinning
    their curations, member and root merges, curations, sorts, then the
    selections (so no part outlives its master)."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    keys = [{"sorting_id": sid} for sid in set(sorting_ids) if sid]
    if not keys:
        return
    pinned = (UnitMatchSelection.MemberCuration & keys).fetch("unitmatch_id")
    if len(pinned):
        (
            UnitMatchSelection & [{"unitmatch_id": u} for u in set(pinned)]
        ).super_delete(warn=False, safemode=False)
    for part in (
        SpikeSortingOutput.ConcatMemberCuration,
        SpikeSortingOutput.CurationV2,
    ):
        for merge_id in (part & keys).fetch("merge_id"):
            (SpikeSortingOutput & {"merge_id": merge_id}).super_delete(
                warn=False, safemode=False
            )
    for table in (ConcatMemberCuration, CurationV2, Sorting, SortingSelection):
        (table & keys).super_delete(warn=False, safemode=False)
