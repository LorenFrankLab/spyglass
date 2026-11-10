"""Join assembly behind ``CurationV2.resolve_restriction``.

Key classification lives in ``_curation.routing``.
"""

from __future__ import annotations

from spyglass.spikesorting.v2._curation.routing import (
    NO_ARTIFACT_RESTRICTION,
    NO_MOTION_CORRECTION_RESTRICTION,
    classify_and_normalize_restriction,
)
from spyglass.spikesorting.v2.artifact_output import ArtifactDetectionOutput
from spyglass.spikesorting.v2.recording import RecordingSelection
from spyglass.spikesorting.v2.session_group import (
    ConcatenatedRecordingSelection,
)
from spyglass.spikesorting.v2.sorting import SortingSelection
from spyglass.utils import logger


def resolve_restriction(
    table_cls,
    key: dict,
    *,
    restrict_by_artifact: bool = True,
    strict: bool = True,
):
    """Resolve an interpretable restriction to the matching CurationV2 rows.

    The body of ``CurationV2.resolve_restriction`` (see its docstring for the
    accepted keys, the ``strict`` / ``restrict_by_artifact`` semantics and
    the errors). ``table_cls`` is ``CurationV2``.
    """
    plan = classify_and_normalize_restriction(
        key, restrict_by_artifact=restrict_by_artifact, strict=strict
    )
    if plan is None:
        return None
    if plan.unresolved_name_warning is not None:
        logger.warning(plan.unresolved_name_warning)

    # Route through the named source; ``shared_restriction`` filters either.
    if plan.concat_restriction:
        concat_sel = ConcatenatedRecordingSelection & {
            **plan.concat_restriction,
            **plan.shared_restriction,
        }
        sort_concat_source = (
            SortingSelection.ConcatenatedRecordingSource * concat_sel.proj()
        )
        sort_master = SortingSelection * sort_concat_source.proj()
    elif plan.rec_restriction:
        rec_table = RecordingSelection & {
            **plan.rec_restriction,
            **plan.shared_restriction,
        }
        sort_rec_source = SortingSelection.RecordingSource * rec_table.proj()
        sort_master = SortingSelection * sort_rec_source.proj()
    elif plan.shared_restriction:
        # A shared key alone must match both source families.
        rec_match = (
            SortingSelection.RecordingSource
            * (RecordingSelection & plan.shared_restriction).proj()
        )
        concat_match = (
            SortingSelection.ConcatenatedRecordingSource
            * (ConcatenatedRecordingSelection & plan.shared_restriction).proj()
        )
        sort_master = SortingSelection & [
            rec_match.proj(),
            concat_match.proj(),
        ]
    else:
        # The master already spans both source families.
        sort_master = SortingSelection
    # Artifact ids: None excludes standalone and concat-member detections;
    # a UUID matches either.
    sort_master = sort_master & plan.sort_restriction
    if plan.artifact_detection_id is not NO_ARTIFACT_RESTRICTION:
        member_artifacts = ConcatenatedRecordingSelection.MemberSnapshot
        if plan.artifact_detection_id is None:
            masked_concats = (
                SortingSelection.ConcatenatedRecordingSource
                & (
                    member_artifacts & "artifact_detection_id IS NOT NULL"
                ).proj()
            )
            sort_master = (
                sort_master
                - SortingSelection.ArtifactDetectionSource.proj()
                - masked_concats.proj()
            )
        else:
            # The part stores the merge id; restricting by the natural id
            # would be silently dropped. An unknown id (None) matches nothing.
            try:
                art_merge_id = ArtifactDetectionOutput.get_merge_id(
                    {"artifact_detection_id": plan.artifact_detection_id}
                )
            except KeyError:
                art_merge_id = None
            artifact_detection_source = (
                SortingSelection.ArtifactDetectionSource
                & {"artifact_detection_merge_id": art_merge_id}
            )
            concat_match = (
                SortingSelection.ConcatenatedRecordingSource
                & (
                    member_artifacts
                    & {"artifact_detection_id": plan.artifact_detection_id}
                ).proj()
            )
            sort_master = sort_master & [
                artifact_detection_source.proj(),
                concat_match.proj(),
            ]
    corrected_id = plan.motion_corrected_recording_id
    if corrected_id is not NO_MOTION_CORRECTION_RESTRICTION:
        correction_source = SortingSelection.MotionCorrectionSource
        if corrected_id is None:
            sort_master = sort_master - correction_source.proj()
        else:
            sort_master = (
                sort_master
                & (
                    correction_source
                    & {"motion_corrected_recording_id": corrected_id}
                ).proj()
            )

    return (
        table_cls * sort_master.proj("sorting_id")
    ) & plan.curation_restriction
