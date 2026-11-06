"""Typed public contracts for the v2 pipeline orchestration helpers.

Keep this module DB-free so the shapes can be inspected without a database.

- ``TypedDict`` comes from ``typing_extensions``: Python 3.10's stdlib version
  does not recognize ``NotRequired``.
- No ``from __future__ import annotations``: stringized annotations hide
  ``NotRequired`` and break ``__required_keys__``.
"""

from typing import (
    Any,
    Literal,
    Mapping,
    NamedTuple,
    Sequence,
    TypeAlias,
)
from uuid import UUID

from typing_extensions import NotRequired, TypedDict

StageStatus: TypeAlias = Literal["computed", "reused", "skipped"]
PipelineOutcome: TypeAlias = Literal["ok", "failed"]
SourceMode: TypeAlias = Literal["single_session", "concat"]
# ``estimate`` sorts the uncorrected source; ``apply`` sorts the corrected one.
MotionMode: TypeAlias = Literal["off", "estimate", "apply"]


class RunV2PipelineOptions(TypedDict, total=False):
    """Options shared by ``run_v2_pipeline`` and ``run_v2_concat_pipeline``.

    Every key is optional and mirrors the function default.
    """

    pipeline_preset: str
    curation_description: str
    require_units: bool
    auto_curate: bool
    preflight: bool
    manual_excluded_times: (
        Sequence[Sequence[float]]
        | Mapping[int, Sequence[Sequence[float]]]
        | None
    )
    motion_mode: MotionMode
    motion_correction_params_name: str | None
    motion_estimate_id: UUID | str | None


class RunV2PipelineInputs(RunV2PipelineOptions):
    """Keyword-argument bundle for a typed ``run_v2_pipeline(**inputs)`` call.

    The sort group's four source fields are required; the options are not.
    """

    nwb_file_name: str
    sort_group_id: int
    interval_list_name: str
    team_name: str


class RunV2ConcatPipelineInputs(RunV2PipelineOptions):
    """Keyword-argument bundle for ``run_v2_concat_pipeline(**inputs)``."""

    session_group_owner: str
    session_group_name: str


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
    curation_description: str
    require_units: bool
    auto_curate: bool
    preflight: bool
    continue_on_error: bool
    manual_excluded_times: Sequence[Sequence[float]] | None
    motion_mode: MotionMode
    motion_correction_params_name: str | None


class PipelineStageSeconds(TypedDict):
    """Per-stage wall-clock seconds for one pipeline run.

    Source-stage keys depend on the entry point; ``sorting`` and ``curation``
    are always present.
    """

    sorting: float
    curation: float
    recording: NotRequired[float]
    artifact_detection: NotRequired[float]
    member_recording: NotRequired[float]
    member_artifact_detection: NotRequired[float]
    concat_recording: NotRequired[float]
    member_curation: NotRequired[float]
    # ``motion_corrected_recording`` only for ``"apply"``.
    motion_estimate: NotRequired[float]
    motion_corrected_recording: NotRequired[float]
    # Present only when ``run_v2_pipeline(auto_curate=True)``.
    auto_curation: NotRequired[float]


class _RunV2SummaryBase(TypedDict):
    """Keys in every ``run_v2_pipeline`` / ``run_v2_concat_pipeline`` summary.

    Not public on its own; each subclass adds the ``source_mode``
    discriminant.
    """

    pipeline_preset: str
    scientific_config: dict
    sorting_id: UUID
    # ``root_*``, not bare ``merge_id``, so the uncurated root is not copied
    # downstream by mistake.
    root_curation_id: int
    # None for concat; use member_merge_ids.
    root_merge_id: "UUID | None"
    # None on a root-only run; the merge id is also None for concat. Every
    # unit is still present; select through ``select_units_for_analysis``.
    auto_labeled_curation_id: "int | None"
    auto_labeled_merge_id: "UUID | None"
    # Pin the accessors against a deleted-and-recreated numeric id.
    root_curation_uuid: UUID
    auto_labeled_curation_uuid: "UUID | None"
    # ``EffectiveSortConfig.as_dict()``; None if the sorter row is absent.
    sorter_config: "dict | None"
    n_units: int
    sorting_status: StageStatus
    curation_status: StageStatus
    stage_seconds: PipelineStageSeconds
    warnings: list[str]
    # Motion keys are None where the mode does not produce them; see
    # ``run_v2_pipeline``'s Returns.
    motion_mode: MotionMode
    motion_correction_params_name: "str | None"
    motion_estimate_id: "UUID | None"
    motion_estimate_supplied: bool
    motion_corrected_recording_id: "UUID | None"
    motion_estimation_preset: "str | None"
    motion_removed_channel_ids: "list | None"
    motion_spans_without_evidence: "list[dict] | None"
    motion_estimate_status: NotRequired[StageStatus]
    motion_corrected_recording_status: NotRequired[StageStatus]
    # Only with ``auto_curate=True``.
    curation_evaluation_id: NotRequired[UUID]
    auto_curation_status: NotRequired[StageStatus]


class RunV2SingleSessionSummary(_RunV2SummaryBase):
    """``run_v2_pipeline`` summary for a single-session run.

    ``artifact_detection_id`` is ``None`` and its status ``"skipped"`` when
    the preset runs no artifact detection.
    """

    source_mode: Literal["single_session"]
    recording_id: UUID
    recording_status: StageStatus
    artifact_detection_id: "UUID | None"
    artifact_detection_status: StageStatus


class MemberArtifactSummary(TypedDict):
    """Exact per-member detection and frame-based masked duration."""

    member_index: int
    artifact_detection_id: UUID
    status: StageStatus
    masked_duration_s: float


class RunV2ConcatSummary(_RunV2SummaryBase):
    """``run_v2_concat_pipeline`` summary (SessionGroup run).

    Carries per-member recording, artifact and merge ids plus the
    concatenation keys.
    """

    source_mode: Literal["concat"]
    member_recording_status: StageStatus
    member_recording_ids: list[UUID]
    member_artifact_detection_status: StageStatus
    member_artifacts: list[MemberArtifactSummary]
    artifact_masked_duration_s: float
    concat_recording_id: UUID
    concat_recording_status: StageStatus
    member_curation_status: StageStatus
    # Keyed by member index: one NWB may contribute several members.
    member_merge_ids: dict[int, UUID]


# Narrow on ``source_mode``.
RunV2PipelineSummary: TypeAlias = RunV2SingleSessionSummary | RunV2ConcatSummary


class MotionEstimateDiagnostics(TypedDict):
    """Peak and displacement counts of one saved ``MotionEstimate``.

    The same-named ``MotionEstimate`` columns.
    """

    n_peaks_detected: int
    n_peaks_kept: int
    max_abs_displacement_um: float
    n_temporal_bins: int


class EstimateMotionReceipt(TypedDict):
    """Return value of ``estimate_motion`` / ``estimate_concat_motion``.

    Pass ``motion_estimate_id`` back with ``motion_mode="apply"`` to sort with
    this estimate. ``source_mode`` selects which source keys are present.
    """

    pipeline_preset: str
    source_mode: SourceMode
    motion_correction_params_name: str
    motion_estimate_id: UUID
    motion_estimation_preset: str
    motion_estimate_status: StageStatus
    motion_spans_without_evidence: list[dict]
    motion_diagnostics: MotionEstimateDiagnostics
    stage_seconds: dict[str, float]
    warnings: list[str]
    recording_id: NotRequired[UUID]
    recording_status: NotRequired[StageStatus]
    artifact_detection_id: NotRequired["UUID | None"]
    artifact_detection_status: NotRequired[StageStatus]
    member_recording_ids: NotRequired[list[UUID]]
    member_recording_status: NotRequired[StageStatus]
    member_artifact_detection_status: NotRequired[StageStatus]
    member_artifacts: NotRequired[list[MemberArtifactSummary]]
    artifact_masked_duration_s: NotRequired[float]
    concat_recording_id: NotRequired[UUID]
    concat_recording_status: NotRequired[StageStatus]


class RunV2PipelineSessionOk(RunV2SingleSessionSummary):
    """Successful entry returned by ``run_v2_pipeline_session``.

    A single-session summary plus the sort group id and outcome.
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
    # None unless a stage failed.
    stage: str | None
    original_error_type: str | None
    partial_run_summary: dict[str, Any] | None
    # Preflight advisories, kept on failure too.
    warnings: list[str]


RunV2PipelineSessionResult: TypeAlias = (
    RunV2PipelineSessionOk | RunV2PipelineSessionFailed
)


class UnitMatchCurationChoice(TypedDict):
    """One discovered curation candidate for a SessionGroup member.

    Not validated for matching; ``UnitMatchSelection.insert_selection`` does
    that. ``parent_curation_id == -1`` marks a root curation.
    """

    sorting_id: UUID
    curation_id: int
    curation_uuid: UUID
    parent_curation_id: int
    curation_source: str
    description: str


class UnitMatchMemberChoices(TypedDict):
    """One SessionGroup member and the curations available to pin for it.

    Pick one entry per member's ``choices`` to build ``run_v2_unit_match``'s
    ``curation_choices``.
    """

    member_index: int
    nwb_file_name: str
    sort_group_id: int
    interval_list_name: str
    team_name: str
    choices: list[UnitMatchCurationChoice]


class UnitMatchStageSeconds(TypedDict):
    """Per-stage wall-clock seconds for one ``run_v2_unit_match`` call."""

    unit_match: float
    tracked_unit: float


class UnitMatchInputSummary(NamedTuple):
    """One matching input of a ``run_v2_unit_match`` receipt.

    Read from the frozen selection rows. ``nwb_file_names`` /
    ``interval_list_names`` are in recording order.
    ``motion_corrected_recording_id`` is ``None`` when waveforms came from
    the sort's own source; ``waveform_traces`` names the traces kind.
    """

    input_index: int
    sorting_id: UUID
    curation_id: int
    curation_uuid: UUID
    source_kind: str
    source_id: UUID
    nwb_file_names: tuple[str, ...]
    interval_list_names: tuple[str, ...]
    n_recordings: int
    motion_corrected_recording_id: "UUID | None"
    waveform_traces: str


class RunV2UnitMatchSummary(TypedDict):
    """Return value of ``run_v2_unit_match``.

    Group identity is ``None`` for a run of named sorts. ``inputs`` is in
    ``input_index`` order.
    """

    session_group_owner: "str | None"
    session_group_name: "str | None"
    matcher_params_name: str
    unit_match_id: UUID
    inputs: tuple[UnitMatchInputSummary, ...]
    unit_match_status: StageStatus
    n_pairs: int
    tracked_unit_status: StageStatus
    n_tracked_units: int
    stage_seconds: UnitMatchStageSeconds
    warnings: list[str]


__all__ = [
    "EstimateMotionReceipt",
    "MotionEstimateDiagnostics",
    "MotionMode",
    "PipelineOutcome",
    "PipelineStageSeconds",
    "RunV2ConcatSummary",
    "RunV2ConcatPipelineInputs",
    "RunV2PipelineInputs",
    "RunV2PipelineOptions",
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
    "UnitMatchInputSummary",
    "UnitMatchMemberChoices",
    "UnitMatchStageSeconds",
]
