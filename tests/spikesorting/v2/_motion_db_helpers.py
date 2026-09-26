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


def drop_motion_selections(recording_key) -> None:
    """Delete every motion-estimate selection on a recording (masters first)."""
    from spyglass.spikesorting.v2.motion import MotionEstimateSelection

    keys = (MotionEstimateSelection.RecordingSource & recording_key).fetch(
        "KEY", as_dict=True
    )
    if keys:
        (MotionEstimateSelection & keys).super_delete(
            warn=False, safemode=False
        )


def drop_concat_motion_selections(concat_key) -> None:
    from spyglass.spikesorting.v2.motion import MotionEstimateSelection

    keys = (
        MotionEstimateSelection.ConcatenatedRecordingSource & concat_key
    ).fetch("KEY", as_dict=True)
    if keys:
        (MotionEstimateSelection & keys).super_delete(
            warn=False, safemode=False
        )


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


def populated_corrected(estimate_key, interpolation=None) -> dict:
    from spyglass.spikesorting.v2.motion import MotionCorrectedRecording

    key = select_corrected(
        estimate_key, *(() if interpolation is None else (interpolation,))
    )
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
    """Delete run_v2_pipeline sorts leaves-first: member and root merges,
    curations, sorts, then the selections (so no part outlives its master)."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    keys = [{"sorting_id": sid} for sid in set(sorting_ids) if sid]
    if not keys:
        return
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
