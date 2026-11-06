"""Insert path behind ``SortingSelection.insert_selection``.

DataJoint tables are imported inside the functions.
"""

from __future__ import annotations

from spyglass.spikesorting.v2._recording.source import (
    SourceLineage,
    correction_lineage_mismatch,
)


def insert_selection(table_cls, key: dict) -> dict:
    """Insert master + exactly one source part; return PK-only dict.

    The body of ``SortingSelection.insert_selection`` (see its docstring).
    ``table_cls`` is ``SortingSelection``.

    Duplicate-key recovery has no savepoint, so the master insert must stay
    the transaction's first statement.
    """
    from spyglass.spikesorting.v2._core.selection_plan import (
        build_sorting_selection_plan,
    )
    from spyglass.spikesorting.v2.artifact_output import (
        insert_artifact_bound_selection,
    )
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording
    from spyglass.spikesorting.v2.sorting import SorterParameters

    plan = build_sorting_selection_plan(key)

    if plan.source_kind == "concat":
        source_part = table_cls.ConcatenatedRecordingSource
        source_row = plan.concat_source_row
    else:
        source_part = table_cls.RecordingSource
        source_row = plan.recording_source_row

    def existing_pk() -> dict | None:
        return table_cls._find_existing_pk(
            plan.master_restriction,
            plan.source_restriction,
            plan.artifact_detection_id,
            plan.sorting_id,
            source_part,
            plan.motion_corrected_recording_id,
        )

    existing = existing_pk()
    if existing is not None:
        return existing

    # Clearer than the FK IntegrityError.
    from spyglass.spikesorting.v2._core.lookup_validation import (
        _ensure_lookup_row_exists,
    )

    _ensure_lookup_row_exists(
        SorterParameters,
        plan.master_restriction,
        helper_name="SortingSelection.insert_selection",
        insert_default_path="SorterParameters.insert_default()",
    )
    if plan.source_kind == "recording":
        from spyglass.spikesorting.v2.artifact import (
            assert_artifact_detection_covers_recording,
        )

        assert_artifact_detection_covers_recording(
            recording_id=plan.source_restriction["recording_id"],
            artifact_detection_id=plan.artifact_detection_id,
            caller="SortingSelection.insert_selection",
        )

    # Otherwise a missing source surfaces as a misleading part-FK bypass error.
    if plan.source_kind == "recording":
        if not (Recording & plan.source_restriction):
            raise ValueError(
                "SortingSelection.insert_selection: recording_id "
                f"{plan.source_restriction['recording_id']} is not in "
                "Recording. Populate the Recording (e.g. via "
                "run_v2_pipeline or Recording.populate) before selecting a "
                "sort on it."
            )
    elif not (ConcatenatedRecording & plan.source_restriction):
        raise ValueError(
            "SortingSelection.insert_selection: concat_recording_id "
            f"{plan.source_restriction['concat_recording_id']} is not in "
            "ConcatenatedRecording. Populate the ConcatenatedRecording "
            "before selecting a sort on it."
        )
    if plan.motion_corrected_recording_id is not None:
        validate_motion_correction_source(plan)

    # Holds the artifact detection's advisory lock across the transaction.
    def insert_rows(art_merge_id) -> None:
        # This helper is the validation boundary.
        table_cls.insert1(plan.master_row, allow_direct_insert=True)
        source_part.insert1(source_row)
        if art_merge_id is not None:
            table_cls.ArtifactDetectionSource.insert1(
                {
                    "sorting_id": plan.master_row["sorting_id"],
                    "artifact_detection_merge_id": art_merge_id,
                }
            )
        if plan.motion_corrected_recording_id is not None:
            table_cls.MotionCorrectionSource.insert1(
                {
                    "sorting_id": plan.master_row["sorting_id"],
                    "motion_corrected_recording_id": (
                        plan.motion_corrected_recording_id
                    ),
                }
            )

    return insert_artifact_bound_selection(
        table_cls,
        {k: plan.master_row[k] for k in table_cls.primary_key},
        plan.artifact_detection_id,
        insert_rows=insert_rows,
        refetch=existing_pk,
        consumer="a sort",
        inputs=(
            f"source {plan.source_restriction} / "
            f"artifact_detection_id={plan.artifact_detection_id} / "
            "motion_corrected_recording_id="
            f"{plan.motion_corrected_recording_id}"
        ),
        input_tables=(
            "Recording / ConcatenatedRecording, ArtifactDetectionOutput or "
            "MotionCorrectedRecording"
        ),
    )


def find_existing_pk(
    table_cls,
    master_restriction,
    source_restriction,
    artifact_detection_id,
    deterministic_id,
    source_part,
    motion_corrected_recording_id,
) -> dict | None:
    """Return the canonical master PK for this sort selection, or None.

    Matches the sorter, the source through ``source_part``, the artifact
    detection, and the motion correction, so none of those variants alias.
    A match with a non-deterministic ``sorting_id`` raises
    ``DuplicateSelectionError``. ``table_cls`` is ``SortingSelection``.
    """
    from spyglass.spikesorting.v2._core.selection_identity import (
        existing_selection_pk,
    )

    candidates = (
        (table_cls * source_part) & master_restriction & source_restriction
    ).fetch("KEY", as_dict=True)
    master_ids = {
        cand["sorting_id"]
        for cand in candidates
        if table_cls.resolve_artifact_detection(
            {"sorting_id": cand["sorting_id"]}
        )
        == artifact_detection_id
        and table_cls.resolve_motion_correction(
            {"sorting_id": cand["sorting_id"]}
        )
        == motion_corrected_recording_id
    }
    return existing_selection_pk(
        master_ids,
        deterministic_id,
        pk_field="sorting_id",
        bypass_message=lambda bypassed: (
            f"SortingSelection has {len(master_ids)} master rows for "
            f"{master_restriction | source_restriction} with "
            f"artifact_detection_id={artifact_detection_id} and "
            f"motion_corrected_recording_id={motion_corrected_recording_id} "
            "whose sorting_id is not the "
            f"deterministic id {deterministic_id}: {bypassed}. This is a "
            "non-deterministic selection row (a raw insert bypassing "
            "insert_selection); drop it and re-insert via "
            "insert_selection."
        ),
    )


def validate_motion_correction_source(plan) -> None:
    """Check a requested corrected recording against the sort request.

    Parameters
    ----------
    plan : SortingSelectionPlan
        The validated request, with ``motion_corrected_recording_id``.

    Raises
    ------
    ValueError
        If the corrected recording is not populated, its motion was
        estimated on another source or under another artifact detection,
        or the sorter params row would correct motion again.
    """
    from spyglass.spikesorting.v2._params.sorter import (
        reject_internal_motion_correction,
    )
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecording,
        MotionCorrectedRecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import SorterParameters

    caller = "SortingSelection.insert_selection"
    corrected_key = {
        "motion_corrected_recording_id": plan.motion_corrected_recording_id
    }
    if not (MotionCorrectedRecording & corrected_key):
        raise ValueError(
            f"{caller}: motion_corrected_recording_id "
            f"{plan.motion_corrected_recording_id} is not in "
            "MotionCorrectedRecording. Populate the corrected recording "
            "before selecting a sort on it."
        )
    mismatches = correction_lineage_mismatch(
        SourceLineage(
            kind=(
                "recording"
                if plan.source_kind == "recording"
                else "concatenated_recording"
            ),
            key=plan.source_restriction,
            artifact_detection_id=plan.artifact_detection_id,
        ),
        MotionCorrectedRecordingSelection.resolve_source(corrected_key),
    )
    if mismatches:
        raise ValueError(
            f"{caller}: motion-corrected recording "
            f"{plan.motion_corrected_recording_id} was not made from this "
            f"sort's source and mask ({'; '.join(mismatches)}). Select a "
            "corrected recording estimated on the same source with the "
            "same artifact detection."
        )
    reject_internal_motion_correction(
        plan.master_restriction["sorter"],
        (SorterParameters & plan.master_restriction).fetch1("params"),
        sorter_params_name=plan.master_restriction["sorter_params_name"],
    )
