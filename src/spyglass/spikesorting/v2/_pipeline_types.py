"""Typed public contracts for the v2 pipeline orchestration helpers.

This module is DB-free by design: it imports only stdlib typing helpers,
``typing_extensions`` (for ``NotRequired``), and ``uuid.UUID``. Keep it that way
so humans, IDEs, and code-generation agents can inspect the pipeline input /
result shapes without importing DataJoint schema modules or opening a database
connection.

``NotRequired`` is imported from ``typing_extensions`` rather than ``typing``:
``typing.NotRequired`` exists only on Python 3.11+, and this package supports
3.10. ``typing_extensions`` (a transitive dependency of pydantic / datajoint)
back-ports the identical object, so the runtime metadata below is unchanged.

Annotations are deliberately NOT postponed (no ``from __future__ import
annotations``): with stringized annotations a ``TypedDict`` cannot see the
``NotRequired`` wrapper, so every key would be misclassified as required and
``__required_keys__`` / ``__optional_keys__`` -- the runtime metadata callers
and codegen inspect -- would be wrong.
"""

from typing import Any, Literal, TypeAlias, TypedDict
from uuid import UUID

from typing_extensions import NotRequired

StageStatus: TypeAlias = Literal["computed", "reused", "skipped"]
PipelineOutcome: TypeAlias = Literal["ok", "failed"]
SourceMode: TypeAlias = Literal["single_session", "concat"]


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


class _RunV2SummaryBase(TypedDict):
    """Keys present in every ``run_v2_pipeline`` summary (both input modes).

    Not a public type on its own -- a summary always carries a ``source_mode``
    discriminant from one of the concrete subclasses below. This base holds the
    always-present sort / curation / merge keys plus the flag-gated
    auto-curation / FigPack keys (which apply to either mode), so the
    single-session and concat summaries share one definition.
    """

    pipeline_preset: str
    sorting_id: UUID
    # The ROOT (uncurated) curation the run always creates. Named ``root_*`` --
    # not bare ``merge_id`` / ``curation_id`` -- so a root-only run has nothing
    # called simply ``merge_id`` to copy downstream by mistake (the root is
    # uncurated and not analysis-ready).
    root_curation_id: int
    root_merge_id: UUID
    # The ANALYSIS-ready (downstream-science) curation. Always present, so a
    # consumer can branch on it: ``None`` on a root-only run
    # (``auto_curate=False``) -- there is no analysis-ready id yet, curate first
    # -- and equal to ``auto_curation_id`` / ``auto_merge_id`` when
    # ``auto_curate=True``.
    analysis_curation_id: "int | None"
    analysis_merge_id: "UUID | None"
    n_units: int
    sorting_status: StageStatus
    curation_status: StageStatus
    stage_seconds: PipelineStageSeconds
    warnings: list[str]
    # Auto-curation keys, present only when ``run_v2_pipeline(auto_curate=True)``:
    # the CurationEvaluation suggestion selection PK, and the materialized child
    # CurationV2 (its curation_id + merge table id) whose labels are the
    # evaluation's verdict. ``analysis_curation_id`` / ``analysis_merge_id``
    # mirror these when auto-curation ran.
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


class RunV2SingleSessionSummary(_RunV2SummaryBase):
    """``run_v2_pipeline`` summary for a single-session run.

    ``source_mode == "single_session"``. Always carries the recording and
    artifact-detection keys; ``artifact_detection_id`` is ``None`` (and
    ``artifact_detection_status`` is ``"skipped"``) when the preset runs no
    artifact detection.
    """

    source_mode: Literal["single_session"]
    recording_id: UUID
    recording_status: StageStatus
    artifact_detection_id: "UUID | None"
    artifact_detection_status: StageStatus


class RunV2ConcatSummary(_RunV2SummaryBase):
    """``run_v2_pipeline`` summary for a concat (SessionGroup) run.

    ``source_mode == "concat"``. Carries the per-member recording PKs and the
    ConcatenatedRecording keys in place of the single-session recording keys,
    and has no artifact stage (a concat SortingSelection has no
    ArtifactDetectionSource row).
    """

    source_mode: Literal["concat"]
    member_recording_status: StageStatus
    member_recording_ids: list[UUID]
    concat_recording_id: UUID
    concat_recording_status: StageStatus


# A run_v2_pipeline summary is exactly one of the two modes; ``source_mode`` is
# the discriminant a consumer narrows on.
RunV2PipelineSummary: TypeAlias = RunV2SingleSessionSummary | RunV2ConcatSummary


class RunV2PipelineSessionOk(RunV2SingleSessionSummary):
    """Successful entry returned by ``run_v2_pipeline_session``.

    ``run_v2_pipeline_session`` sorts every sort group of one session, so each
    entry is a single-session summary plus the sort group id and outcome.
    """

    sort_group_id: int
    outcome: Literal["ok"]


class RunV2PipelineSessionFailed(TypedDict):
    """Failed entry returned by ``run_v2_pipeline_session``."""

    sort_group_id: int
    pipeline_preset: str
    outcome: Literal["failed"]
    error_type: str
    error: str
    # For a stage failure (PipelineStageError): the failing stage name and the
    # underlying error type it wrapped (e.g. "IndexError"). Both None for a
    # preflight / zero-unit failure, which is not a stage failure.
    stage: str | None
    original_error_type: str | None
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


class UnitMatchStageSeconds(TypedDict):
    """Per-stage wall-clock seconds for one ``run_v2_unit_match`` call.

    ≈0 on an idempotent re-run (NOT cumulative compute cost).
    """

    unit_match: float
    tracked_unit: float


class RunV2UnitMatchSummary(TypedDict):
    """Return value of ``run_v2_unit_match``.

    The cross-session match manifest: the ``UnitMatch`` selection PK plus the
    pairwise-match and tracked-unit results, with per-stage status / timing.
    """

    session_group_owner: str
    session_group_name: str
    matcher_params_name: str
    unitmatch_id: UUID
    unit_match_status: StageStatus
    n_pairs: int
    tracked_unit_status: StageStatus
    n_tracked_units: int
    stage_seconds: UnitMatchStageSeconds
    warnings: list[str]


__all__ = [
    "PipelineOutcome",
    "PipelineStageSeconds",
    "RunV2ConcatSummary",
    "RunV2PipelineInputs",
    "RunV2PipelineSessionFailed",
    "RunV2PipelineSessionInputs",
    "RunV2PipelineSessionOk",
    "RunV2PipelineSessionRequiredInputs",
    "RunV2PipelineSessionResult",
    "RunV2PipelineSummary",
    "RunV2SingleSessionSummary",
    "RunV2UnitMatchSummary",
    "SourceMode",
    "StageStatus",
    "UnitMatchCurationChoice",
    "UnitMatchMemberChoices",
    "UnitMatchStageSeconds",
]
