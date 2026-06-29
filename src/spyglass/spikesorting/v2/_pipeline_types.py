"""Typed public contracts for the v2 pipeline orchestration helpers.

This module is DB-free by design: it imports only stdlib typing helpers and
``uuid.UUID``. Keep it that way so humans, IDEs, and code-generation agents can
inspect the pipeline input / result shapes without importing DataJoint schema
modules or opening a database connection.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypeAlias, TypedDict
from uuid import UUID

StageStatus: TypeAlias = Literal["computed", "reused", "skipped"]
PipelineOutcome: TypeAlias = Literal["ok", "failed"]


class RunV2PipelineRequiredInputs(TypedDict):
    """Required ``run_v2_pipeline`` keyword arguments."""

    nwb_file_name: str
    sort_group_id: int
    interval_list_name: str
    team_name: str


class RunV2PipelineInputs(RunV2PipelineRequiredInputs, total=False):
    """Keyword-argument bundle accepted by ``run_v2_pipeline``.

    Useful for typed ``run_v2_pipeline(**inputs)`` call sites. Optional keys
    mirror the function defaults.
    """

    pipeline_preset: str
    description: str
    require_units: bool
    auto_curate: bool
    preflight: bool


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
    """Per-stage wall-clock seconds for one ``run_v2_pipeline`` call."""

    recording: float
    artifact_detection: float
    sorting: float
    curation: float
    # Present only when ``run_v2_pipeline(auto_curate=True)``.
    auto_curation: NotRequired[float]


class RunV2PipelineSummary(TypedDict):
    """Return value of ``run_v2_pipeline``."""

    pipeline_preset: str
    recording_id: UUID
    # None when the preset runs no artifact detection (artifact_detection_status
    # is then "skipped" and no ArtifactDetectionSource row is created).
    artifact_detection_id: "UUID | None"
    sorting_id: UUID
    curation_id: int
    merge_id: UUID
    n_units: int
    recording_status: StageStatus
    artifact_detection_status: StageStatus
    sorting_status: StageStatus
    curation_status: StageStatus
    stage_seconds: PipelineStageSeconds
    warnings: list[str]
    # Auto-curation keys, present only when ``run_v2_pipeline(auto_curate=True)``:
    # the CurationEvaluation suggestion selection PK, and the materialized child
    # CurationV2 (its curation_id + merge table id) whose labels are the
    # evaluation's verdict.
    curation_evaluation_id: NotRequired[UUID]
    auto_curation_id: NotRequired[int]
    auto_merge_id: NotRequired[UUID]
    auto_curation_status: NotRequired[StageStatus]


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
    "StageStatus",
]
