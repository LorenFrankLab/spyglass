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
    # Semijoins on a source column never match a NULL, so each picks one
    # source family.
    if plan.concat_restriction:
        concat_sel = ConcatenatedRecordingSelection & {
            **plan.concat_restriction,
            **plan.shared_restriction,
        }
        sort_master = SortingSelection & concat_sel.proj()
    elif plan.rec_restriction:
        rec_table = RecordingSelection & {
            **plan.rec_restriction,
            **plan.shared_restriction,
        }
        sort_master = SortingSelection & rec_table.proj()
    elif plan.shared_restriction:
        # A shared key alone must match both source families.
        sort_master = SortingSelection & [
            (RecordingSelection & plan.shared_restriction).proj(),
            (ConcatenatedRecordingSelection & plan.shared_restriction).proj(),
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
            # Anti-join on sorting_id: NOT IN on a NULL source column would
            # drop every single-recording sort.
            masked_concats = (
                SortingSelection
                & (
                    member_artifacts & "artifact_detection_id IS NOT NULL"
                ).proj()
            )
            sort_master = (
                sort_master & "artifact_detection_merge_id IS NULL"
            ) - masked_concats.proj()
        else:
            # The row stores the merge id; restricting by the natural id
            # would be silently dropped. An unknown id matches nothing.
            try:
                art_merge_id = ArtifactDetectionOutput.get_merge_id(
                    {"artifact_detection_id": plan.artifact_detection_id}
                )
            except KeyError:
                art_merge_id = None
            single_match = (
                "FALSE"
                if art_merge_id is None
                else {"artifact_detection_merge_id": art_merge_id}
            )
            concat_match = (
                SortingSelection
                & (
                    member_artifacts
                    & {"artifact_detection_id": plan.artifact_detection_id}
                ).proj()
            )
            sort_master = sort_master & [single_match, concat_match.proj()]
    sort_master = _restrict_motion(
        sort_master, plan.motion_corrected_recording_id
    )

    return (
        table_cls * sort_master.proj("sorting_id")
    ) & plan.curation_restriction


def resolve_member_restriction(
    member_cls,
    key: dict,
    *,
    restrict_by_artifact: bool = True,
    strict: bool = True,
):
    """Resolve a restriction to the matching concat-member output rows.

    Recording keys (``nwb_file_name``, ``sort_group_id``,
    ``interval_list_name``, ``team_name``, ``recording_id``),
    ``member_index`` and ``artifact_detection_id`` match each member's own
    frozen ``MemberSnapshot``; concat, shared, sort, motion and curation
    keys match its concat sort. Mixed concat and recording keys raise, as
    in ``resolve_restriction``, which logs the unresolved-artifact warning.

    Parameters
    ----------
    member_cls : type
        ``ConcatMemberCurationV2``.
    key : dict
        Interpretable restriction.
    restrict_by_artifact : bool, optional
        Map an ``artifact_detection_{uuid}`` interval name to its id.
    strict : bool, optional
        Raise on an unknown key; ``False`` returns ``None`` instead.

    Returns
    -------
    datajoint.expression.QueryExpression or None
        The matching ``member_cls`` rows, or ``None`` when ``strict=False``
        and the key names no v2 column.
    """
    plan = classify_and_normalize_restriction(
        key,
        restrict_by_artifact=restrict_by_artifact,
        strict=strict,
        member_keys=True,
    )
    if plan is None:
        return None
    concat_sel = ConcatenatedRecordingSelection & {
        **plan.concat_restriction,
        **plan.shared_restriction,
    }
    sort_master = SortingSelection & concat_sel.proj() & plan.sort_restriction
    sort_master = _restrict_motion(
        sort_master, plan.motion_corrected_recording_id
    )
    snapshot = (
        ConcatenatedRecordingSelection.MemberSnapshot
        & plan.rec_restriction
        & plan.member_restriction
    )
    if plan.artifact_detection_id is None:
        snapshot = snapshot & "artifact_detection_id IS NULL"
    elif plan.artifact_detection_id is not NO_ARTIFACT_RESTRICTION:
        snapshot = snapshot & {
            "artifact_detection_id": plan.artifact_detection_id
        }
    member_sources = (
        SortingSelection.proj("concat_recording_id") * snapshot.proj()
    ).proj()
    return (
        member_cls & plan.curation_restriction & sort_master.proj()
    ) & member_sources


def _restrict_motion(sort_master, corrected_id):
    """Restrict sorts by motion correction: a UUID, ``None`` or wildcard."""
    if corrected_id is NO_MOTION_CORRECTION_RESTRICTION:
        return sort_master
    return sort_master & {"motion_corrected_recording_id": corrected_id}
