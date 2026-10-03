"""Join assembly behind ``CurationV2.resolve_restriction``.

:func:`resolve_restriction` turns an interpretable v2 restriction into the
matching ``CurationV2`` rows by joining through the selection tables of the
input source the restriction names (single recording or concatenation), then
applying the artifact-detection, motion-correction, sort and curation filters.
The DB-free key classification it starts from lives in ``_curation_routing``.

Imports without the DB layer: the DataJoint tables are imported inside the
function.
"""

from __future__ import annotations


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
    from spyglass.spikesorting.v2._curation_routing import (
        NO_ARTIFACT_RESTRICTION,
        NO_MOTION_CORRECTION_RESTRICTION,
        classify_and_normalize_restriction,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection
    from spyglass.utils import logger

    # Pure key-classification / normalization (unknown-key handling,
    # artifact-interval-name -> id mapping, uuid normalization, per-source
    # split) lives in the DB-free ``_curation_routing`` module; this function
    # owns only the DataJoint join assembly.
    plan = classify_and_normalize_restriction(
        key, restrict_by_artifact=restrict_by_artifact, strict=strict
    )
    if plan is None:
        return None
    if plan.unresolved_name_warning is not None:
        logger.warning(plan.unresolved_name_warning)

    # Route through the input source the restriction names. A concat
    # restriction joins SortingSelection.ConcatenatedRecordingSource ->
    # ConcatenatedRecordingSelection; a single-recording restriction joins
    # SortingSelection.RecordingSource -> RecordingSelection. A cross-source
    # ``shared_restriction`` (the preprocessing recipe) filters whichever
    # family routes, since it lives on both source selections.
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
        # A cross-source key with NO source-specific key (e.g. a bare
        # ``{preprocessing_params_name}`` query) must match BOTH families:
        # sorts whose RecordingSource recipe matches OR whose
        # ConcatenatedRecordingSource recipe matches. Routing through one
        # family only would silently drop the other.
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
        # No source-specific or cross-source key (an unrestricted v2 query
        # or a sort-/curation-only restriction): match BOTH source families.
        # Every SortingSelection has exactly one input source, so the master
        # itself IS the recording-source union concat-source set --
        # restricting through one source part here would silently drop the
        # other (e.g. omit concat-backed curations from a broad v2 merge
        # query).
        sort_master = SortingSelection
    # Artifact dependencies live on the standalone sort's optional part
    # or on frozen concat members. None excludes both; a UUID matches any
    # selected detection; only an absent key leaves the query unrestricted.
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
            # ``ArtifactDetectionSource`` stores the
            # ``artifact_detection_merge_id`` (the ArtifactDetectionOutput
            # merge PK), NOT the natural ``artifact_detection_id`` -- a dict
            # restriction with the natural id would be a dropped (unknown)
            # key that matches EVERY artifact-backed sort. Resolve the id to
            # its merge id and restrict on the real column. An id that no
            # sort registered yields ``None`` -> the non-nullable merge-id
            # column matches nothing -> the correct empty result.
            from spyglass.spikesorting.v2.artifact_output import (
                ArtifactDetectionOutput,
            )

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
    # A sort reads motion-corrected traces only through its optional
    # MotionCorrectionSource part. None keeps sorts of their source's own
    # traces, an id keeps sorts of that corrected recording, and an absent
    # key keeps both.
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
