"""Typed public contracts for the v2 pipeline orchestration helpers.

This module is DB-free by design: it imports only stdlib typing helpers and
``uuid.UUID``. Keep it that way so humans, IDEs, and code-generation agents can
inspect the pipeline input / result shapes without importing DataJoint schema
modules or opening a database connection.

Annotations are deliberately NOT postponed (no ``from __future__ import
annotations``): with stringized annotations a ``TypedDict`` cannot see the
``NotRequired`` wrapper, so every key would be misclassified as required and
``__required_keys__`` / ``__optional_keys__`` -- the runtime metadata callers
and codegen inspect -- would be wrong.
"""

from typing import Any, Literal, NotRequired, TypeAlias, TypedDict
from uuid import UUID

StageStatus: TypeAlias = Literal["computed", "reused", "skipped"]
PipelineOutcome: TypeAlias = Literal["ok", "failed"]


class RunV2PipelineRequiredInputs(TypedDict):
    """Required ``run_v2_pipeline`` keyword arguments for a single-session run.

    A concat run instead supplies ``concat_session_group_owner`` /
    ``concat_session_group_name`` (see ``RunV2PipelineInputs``).
    """

    nwb_file_name: str
    sort_group_id: int
    interval_list_name: str
    team_name: str


class RunV2PipelineInputs(TypedDict, total=False):
    """Keyword-argument bundle accepted by ``run_v2_pipeline``.

    Useful for typed ``run_v2_pipeline(**inputs)`` call sites. Every key is
    optional because the call accepts exactly one of two input modes, validated
    at call time:

    - single-session: ``nwb_file_name`` + ``sort_group_id`` +
      ``interval_list_name`` + ``team_name``
    - concat: ``concat_session_group_owner`` + ``concat_session_group_name``

    The remaining keys mirror the function defaults.
    """

    nwb_file_name: str
    sort_group_id: int
    interval_list_name: str
    team_name: str
    concat_session_group_owner: str
    concat_session_group_name: str
    pipeline_preset: str
    description: str
    require_units: bool
    auto_curate: bool
    preflight: bool
    figpack: bool
    figpack_label_options: list[str] | None


class RunV2PipelineSessionRequiredInputs(TypedDict):
    """Required ``run_v2_pipeline_session`` keyword arguments."""

    nwb_file_name: str
    interval_list_name: str
    team_name: str
    pipeline_preset: str


class RunV2PipelineSessionInputs(
    RunV2PipelineSessionRequiredInputs, total=False
):
    """Keyword-argument bundle accepted by ``run_v2_pipeline_session``."""

    sort_group_ids: list[int]
    description: str
    require_units: bool
    auto_curate: bool
    preflight: bool
    continue_on_error: bool


class PipelineStageSeconds(TypedDict):
    """Per-stage wall-clock seconds for one ``run_v2_pipeline`` call.

    The source-stage keys depend on the input mode: single-session runs carry
    ``recording`` (+ ``artifact_detection``), concat runs carry
    ``concat_recording`` instead. ``sorting`` / ``curation`` are always present.
    """

    sorting: float
    curation: float
    # Single-session source stages.
    recording: NotRequired[float]
    artifact_detection: NotRequired[float]
    # Concat source stages (concat mode only).
    member_recording: NotRequired[float]
    concat_recording: NotRequired[float]
    # Present only when ``run_v2_pipeline(auto_curate=True)``.
    auto_curation: NotRequired[float]
    # Present only when ``run_v2_pipeline(figpack=True)``.
    figpack: NotRequired[float]


class RunV2PipelineSummary(TypedDict):
    """Return value of ``run_v2_pipeline``.

    The source keys depend on the input mode. Single-session runs carry
    ``recording_id`` / ``recording_status`` and ``artifact_detection_id`` /
    ``artifact_detection_status``; concat runs carry ``concat_recording_id`` /
    ``concat_recording_status`` and ``member_recording_ids`` instead, and omit
    the artifact keys (a concat sort has no ArtifactDetectionSource row). The
    sorting / curation / merge keys are always present.
    """

    pipeline_preset: str
    sorting_id: UUID
    curation_id: int
    merge_id: UUID
    n_units: int
    sorting_status: StageStatus
    curation_status: StageStatus
    stage_seconds: PipelineStageSeconds
    warnings: list[str]
    # Single-session source keys. ``artifact_detection_id`` is None when the
    # preset runs no artifact detection (status is then "skipped").
    recording_id: NotRequired[UUID]
    recording_status: NotRequired[StageStatus]
    artifact_detection_id: NotRequired["UUID | None"]
    artifact_detection_status: NotRequired[StageStatus]
    # Concat source keys (concat mode only).
    member_recording_status: NotRequired[StageStatus]
    member_recording_ids: NotRequired[list[UUID]]
    concat_recording_id: NotRequired[UUID]
    concat_recording_status: NotRequired[StageStatus]
    # Auto-curation keys, present only when ``run_v2_pipeline(auto_curate=True)``:
    # the CurationEvaluation suggestion selection PK, and the materialized child
    # CurationV2 (its curation_id + merge table id) whose labels are the
    # evaluation's verdict.
    curation_evaluation_id: NotRequired[UUID]
    auto_curation_id: NotRequired[int]
    auto_merge_id: NotRequired[UUID]
    auto_curation_status: NotRequired[StageStatus]
    # FigPack keys, present only when ``run_v2_pipeline(figpack=True)``: the
    # published curation-view URI (a local bundle path offline) and its stage
    # status. ``figpack_uri`` is absent and ``figpack_status`` is ``"skipped"``
    # when the sort found zero units (no analyzer to summarize).
    figpack_uri: NotRequired[str]
    figpack_status: NotRequired[StageStatus]


class RunV2PipelineSessionOk(RunV2PipelineSummary):
    """Successful entry returned by ``run_v2_pipeline_session``."""

    sort_group_id: int
    outcome: Literal["ok"]


class RunV2PipelineSessionFailed(TypedDict):
    """Failed entry returned by ``run_v2_pipeline_session``."""

    sort_group_id: int
    pipeline_preset: str
    outcome: Literal["failed"]
    error_type: str
    error: str
    partial_run_summary: dict[str, Any] | None
    # Preflight advisories for this group are carried even on failure so the
    # batch warning count / describe_run do not under-report failed groups.
    warnings: list[str]


RunV2PipelineSessionResult: TypeAlias = (
    RunV2PipelineSessionOk | RunV2PipelineSessionFailed
)


class UnitMatchCurationChoice(TypedDict):
    """One pickable curation for a SessionGroup member.

    From ``describe_unit_match_choices``: a committed ``CurationV2`` the user may
    pin for this member. ``parent_curation_id == -1`` marks a root curation.
    """

    sorting_id: UUID
    curation_id: int
    parent_curation_id: int
    description: str


class UnitMatchMemberChoices(TypedDict):
    """One SessionGroup member and the curations available to pin for it.

    From ``describe_unit_match_choices``. Assemble ``run_v2_unit_match``'s
    ``curation_choices`` as ``{member_index: {"sorting_id": ...,
    "curation_id": ...}}`` by picking one entry from each member's ``choices``.
    """

    member_index: int
    nwb_file_name: str
    sort_group_id: int
    interval_list_name: str
    team_name: str
    choices: list[UnitMatchCurationChoice]


class RunV2UnitMatchSummary(TypedDict):
    """Return value of ``run_v2_unit_match``.

    The cross-session match manifest: the ``UnitMatch`` selection PK plus the
    pairwise-match and tracked-unit results, with per-stage status / timing.
    """

    session_group_owner: str
    session_group_name: str
    matcher_params_name: str
    unitmatch_id: UUID
    unitmatch_status: StageStatus
    n_pairs: int
    tracked_unit_status: StageStatus
    n_tracked_units: int
    # Per-stage wall-clock seconds (keys ``unit_match`` / ``tracked_unit``) this
    # call -- ≈0 on an idempotent re-run, NOT cumulative compute cost.
    stage_seconds: dict[str, float]
    warnings: list[str]


__all__ = [
    "PipelineOutcome",
    "PipelineStageSeconds",
    "RunV2PipelineInputs",
    "RunV2PipelineRequiredInputs",
    "RunV2PipelineSessionFailed",
    "RunV2PipelineSessionInputs",
    "RunV2PipelineSessionOk",
    "RunV2PipelineSessionRequiredInputs",
    "RunV2PipelineSessionResult",
    "RunV2PipelineSummary",
    "RunV2UnitMatchSummary",
    "StageStatus",
    "UnitMatchCurationChoice",
    "UnitMatchMemberChoices",
]
