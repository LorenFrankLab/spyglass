"""Insert paths behind the motion selection tables.

``motion`` imports this module, so its tables are imported inside the
functions.
"""

from __future__ import annotations

import uuid

import datajoint as dj

from spyglass.spikesorting.v2._core.lookup_validation import (
    _ensure_lookup_row_exists,
)
from spyglass.spikesorting.v2._core.selection_identity import (
    reject_unknown_fields,
)
from spyglass.spikesorting.v2._motion.estimation import (
    motion_corrected_selection_identity,
    motion_estimate_selection_identity,
)
from spyglass.spikesorting.v2.artifact import (
    assert_artifact_detection_covers_recording,
)
from spyglass.spikesorting.v2.artifact_output import (
    insert_artifact_bound_selection,
)


def insert_estimate_selection(table_cls, key: dict) -> dict:
    """Insert (or find) the selection for a source, mask and recipe.

    The body of ``MotionEstimateSelection.insert_selection`` (see its
    docstring for the request fields and errors). ``table_cls`` is the
    ``MotionEstimateSelection`` class; ``_find_existing_pk`` is called on it.
    """
    from spyglass.spikesorting.v2.motion import (
        _SOURCE_TABLES,
        MotionEstimationParameters,
        _source_filter_problem,
    )

    caller = "MotionEstimateSelection.insert_selection"
    reject_unknown_fields(key, table_cls._INPUT_FIELDS, caller=caller)
    recording_id = key.get("recording_id")
    concat_recording_id = key.get("concat_recording_id")
    if (recording_id is None) == (concat_recording_id is None):
        raise ValueError(
            f"{caller}: pass exactly one of recording_id or "
            "concat_recording_id."
        )
    artifact_detection_id = key.get("artifact_detection_id")
    if artifact_detection_id is not None:
        artifact_detection_id = uuid.UUID(str(artifact_detection_id))
        if concat_recording_id is not None:
            raise ValueError(
                f"{caller}: a concatenated recording carries its own "
                "member artifact masks; artifact_detection_id applies to "
                "a single recording only."
            )
    params_name = key.get("motion_estimation_params_name")
    if params_name is None:
        raise ValueError(
            f"{caller}: motion_estimation_params_name is required."
        )

    if concat_recording_id is not None:
        source_kind = "concatenated_recording"
        source_key = {
            "concat_recording_id": uuid.UUID(str(concat_recording_id))
        }
        source_part = table_cls.ConcatenatedRecordingSource
    else:
        source_kind = "recording"
        source_key = {"recording_id": uuid.UUID(str(recording_id))}
        source_part = table_cls.RecordingSource
    source_table = _SOURCE_TABLES[source_kind]

    params_key = {"motion_estimation_params_name": params_name}
    _ensure_lookup_row_exists(
        MotionEstimationParameters,
        params_key,
        helper_name=caller,
        insert_default_path="MotionEstimationParameters.insert_default()",
    )
    estimation_params = (MotionEstimationParameters & params_key).fetch1(
        "params"
    )
    content_hashes = (source_table & source_key).fetch("content_hash")
    if len(content_hashes) == 0:
        raise ValueError(
            f"{caller}: {source_key} is not in {source_table.__name__}. "
            "Populate it before selecting a motion estimate on it."
        )
    problem = _source_filter_problem(source_kind, source_key)
    if problem is not None:
        raise ValueError(
            f"{caller}: {source_key} cannot be motion-estimated: {problem}"
        )
    if source_kind == "recording":
        assert_artifact_detection_covers_recording(
            recording_id=source_key["recording_id"],
            artifact_detection_id=artifact_detection_id,
            caller=caller,
        )

    motion_estimate_id, master_row = motion_estimate_selection_identity(
        source_kind=source_kind,
        source_id=next(iter(source_key.values())),
        source_content_hash=content_hashes[0],
        artifact_detection_id=artifact_detection_id,
        motion_estimation_params_name=params_name,
        estimation_params=estimation_params,
    )
    explicit = key.get("motion_estimate_id")
    if explicit is not None and uuid.UUID(str(explicit)) != (
        motion_estimate_id
    ):
        raise ValueError(
            f"{caller}: motion_estimate_id {explicit} does not match the "
            f"id derived from this selection ({motion_estimate_id})."
        )

    def _existing():
        return table_cls._find_existing_pk(
            master_row,
            source_part,
            source_key,
            artifact_detection_id,
            motion_estimate_id,
        )

    existing = _existing()
    if existing is not None:
        return existing
    return insert_estimate_selection_rows(
        table_cls,
        {"motion_estimate_id": motion_estimate_id, **master_row},
        source_part,
        {"motion_estimate_id": motion_estimate_id, **source_key},
        artifact_detection_id,
        refetch=_existing,
    )


def insert_estimate_selection_rows(
    table_cls,
    master_row,
    source_part,
    source_row,
    artifact_detection_id,
    *,
    refetch,
) -> dict:
    """Insert the master and its parts atomically, locked against deletion.

    Uses the artifact-bound insert shared with ``SortingSelection``; a
    duplicate-key race refetches the winner. ``table_cls`` is the
    ``MotionEstimateSelection`` class.

    Recovery has no savepoint, so the master insert must stay first: the
    deterministic key collides there before any part row is written.
    """
    motion_estimate_id = master_row["motion_estimate_id"]

    def insert_rows(art_merge_id) -> None:
        table_cls.insert1(master_row, allow_direct_insert=True)
        source_part.insert1(source_row)
        if art_merge_id is not None:
            table_cls.ArtifactDetectionSource.insert1(
                {
                    "motion_estimate_id": motion_estimate_id,
                    "artifact_detection_merge_id": art_merge_id,
                }
            )

    return insert_artifact_bound_selection(
        table_cls,
        {"motion_estimate_id": motion_estimate_id},
        artifact_detection_id,
        insert_rows=insert_rows,
        refetch=refetch,
        consumer="a motion estimate",
        inputs=(
            f"source {source_row} / "
            f"artifact_detection_id={artifact_detection_id}"
        ),
        input_tables=(
            "MotionEstimationParameters, Recording / ConcatenatedRecording "
            "or ArtifactDetectionOutput"
        ),
    )


def insert_corrected_selection(table_cls, key: dict) -> dict:
    """Insert (or find) the selection for a saved estimate and a recipe.

    The body of ``MotionCorrectedRecordingSelection.insert_selection`` (see
    its docstring for the request fields and errors). ``table_cls`` is the
    ``MotionCorrectedRecordingSelection`` class; ``_find_existing_pk`` is
    called on it.
    """
    from spyglass.spikesorting.v2.motion import (
        MotionEstimate,
        MotionInterpolationParameters,
        _estimate_source,
    )

    caller = "MotionCorrectedRecordingSelection.insert_selection"
    reject_unknown_fields(key, table_cls._INPUT_FIELDS, caller=caller)
    missing = [
        name
        for name in (
            "motion_estimate_id",
            "motion_interpolation_params_name",
        )
        if key.get(name) is None
    ]
    if missing:
        raise ValueError(f"{caller}: {missing} are required.")
    motion_estimate_id = uuid.UUID(str(key["motion_estimate_id"]))
    params_name = key["motion_interpolation_params_name"]
    params_key = {"motion_interpolation_params_name": params_name}
    _ensure_lookup_row_exists(
        MotionInterpolationParameters,
        params_key,
        helper_name=caller,
        insert_default_path=("MotionInterpolationParameters.insert_default()"),
    )
    if not (MotionEstimate & {"motion_estimate_id": motion_estimate_id}):
        raise ValueError(
            f"{caller}: motion estimate {motion_estimate_id} is not "
            "populated. Populate MotionEstimate before selecting a "
            "corrected recording on it."
        )
    _estimate_source(motion_estimate_id)
    corrected_id, master_row = motion_corrected_selection_identity(
        motion_estimate_id=motion_estimate_id,
        motion_interpolation_params_name=params_name,
        interpolation_params=(
            MotionInterpolationParameters & params_key
        ).fetch1("params"),
    )
    explicit = key.get("motion_corrected_recording_id")
    if explicit is not None and uuid.UUID(str(explicit)) != corrected_id:
        raise ValueError(
            f"{caller}: motion_corrected_recording_id {explicit} does not "
            f"match the id derived from this selection ({corrected_id})."
        )
    existing = table_cls._find_existing_pk(master_row, corrected_id)
    if existing is not None:
        return existing
    try:
        table_cls.insert1(
            {"motion_corrected_recording_id": corrected_id, **master_row},
            allow_direct_insert=True,
        )
    except dj.errors.DuplicateError:
        existing = table_cls._find_existing_pk(master_row, corrected_id)
        if existing is None:
            raise
        return existing
    return {"motion_corrected_recording_id": corrected_id}
