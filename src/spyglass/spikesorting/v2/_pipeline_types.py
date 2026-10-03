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

from typing import Any, Literal, NamedTuple, TypeAlias, TypedDict
from uuid import UUID

from typing_extensions import NotRequired

StageStatus: TypeAlias = Literal["computed", "reused", "skipped"]
PipelineOutcome: TypeAlias = Literal["ok", "failed"]
SourceMode: TypeAlias = Literal["single_session", "concat"]
# ``off``: no motion stage; ``estimate``: save a motion estimate of the sort's
# source and sort the uncorrected source (same sort as ``off``); ``apply``:
# save the estimate, a motion-corrected recording, and sort the corrected one.
MotionMode: TypeAlias = Literal["off", "estimate", "apply"]


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
    curation_description: str
    require_units: bool
    auto_curate: bool
    preflight: bool
    build_figpack_view: bool
    figpack_label_options: list[str] | None
    motion_mode: MotionMode
    motion_correction_params_name: str | None
    motion_estimate_id: UUID | str | None


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
    motion_mode: MotionMode
    motion_correction_params_name: str | None


class PipelineStageSeconds(TypedDict):
    """Per-stage wall-clock seconds for one ``run_v2_pipeline`` call.

    The source-stage keys depend on the input mode: single-session runs carry
    ``recording`` (+ ``artifact_detection``), concat runs carry
    ``member_recording``, ``member_artifact_detection``, ``concat_recording``,
    and ``member_curation`` instead. ``sorting`` / ``curation`` are always present.
    """

    sorting: float
    curation: float
    # Single-session source stages.
    recording: NotRequired[float]
    artifact_detection: NotRequired[float]
    # Concat source stages (concat mode only).
    member_recording: NotRequired[float]
    member_artifact_detection: NotRequired[float]
    concat_recording: NotRequired[float]
    member_curation: NotRequired[float]
    # Motion stages: ``motion_estimate`` when ``motion_mode`` is ``"estimate"``
    # or ``"apply"``; ``motion_corrected_recording`` only for ``"apply"``.
    motion_estimate: NotRequired[float]
    motion_corrected_recording: NotRequired[float]
    # Present only when ``run_v2_pipeline(auto_curate=True)``.
    auto_curation: NotRequired[float]
    # Present only when ``run_v2_pipeline(build_figpack_view=True)``.
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
    scientific_config: dict
    sorting_id: UUID
    # The ROOT (uncurated) curation the run always creates. Named ``root_*`` --
    # not bare ``merge_id`` / ``curation_id`` -- so a root-only run has nothing
    # called simply ``merge_id`` to copy downstream by mistake (the root is
    # uncurated and not a reviewed result).
    root_curation_id: int
    # A concat curation's synthetic-timeline row stays out of
    # SpikeSortingOutput; RunV2ConcatSummary exposes wall-clock-safe member rows.
    root_merge_id: "UUID | None"
    # The AUTO-LABELED child curation. Always present, so a consumer can
    # branch on it: ``None`` on a root-only run (``auto_curate=False``), set
    # when a run uses ``auto_curate=True``. Concat carries the child curation
    # id but leaves the merge id None. Automatic labels are NOT scientific
    # approval and its merge id is NOT a filtered unit set: every unit (noise /
    # reject / artifact included) is still in the row. Downstream selection
    # goes through ``select_units_for_analysis`` (SortedSpikesGroup +
    # UnitSelectionParams), never straight through this merge id.
    auto_labeled_curation_id: "int | None"
    auto_labeled_merge_id: "UUID | None"
    # Generation UUIDs of the root / auto-labeled CurationV2 rows at run time.
    # ``RunResult.root_curation`` / ``.auto_labeled_curation`` pin these, so a
    # receipt cannot silently resolve a deleted-and-recreated numeric id.
    root_curation_uuid: UUID
    auto_labeled_curation_uuid: "UUID | None"
    # What the sort stage executes (``EffectiveSortConfig.as_dict()``): the
    # kwargs handed to SpikeInterface, whiten routing, seed, resolved job
    # kwargs, execution backend. ``None`` only when preflight was bypassed and
    # the preset's SorterParameters row is absent.
    sorter_config: "dict | None"
    n_units: int
    sorting_status: StageStatus
    curation_status: StageStatus
    stage_seconds: PipelineStageSeconds
    warnings: list[str]
    # The motion stage of this run, in both input modes. ``motion_mode`` and
    # ``motion_correction_params_name`` echo the request (the name is ``None``
    # for ``"off"``). ``motion_estimate_id`` and ``motion_estimation_preset``
    # (the SpikeInterface preset the estimation recipe resolved to) are set
    # for ``"estimate"`` and ``"apply"``; ``motion_estimate_supplied`` is True
    # when the caller passed that estimate (``motion_estimate_id=``, apply
    # only) instead of the run selecting it; ``motion_corrected_recording_id``
    # and ``motion_removed_channel_ids`` (the source channels
    # ``border_mode="remove_channels"`` dropped; empty otherwise) only for
    # ``"apply"``. ``motion_spans_without_evidence`` lists the estimate's
    # continuity spans that kept no peak
    # (``MotionEstimate.get_spans_without_evidence``; empty when every span
    # has evidence) for ``"estimate"`` and ``"apply"``. Every one is ``None``
    # where it does not apply.
    motion_mode: MotionMode
    motion_correction_params_name: "str | None"
    motion_estimate_id: "UUID | None"
    motion_estimate_supplied: bool
    motion_corrected_recording_id: "UUID | None"
    motion_estimation_preset: "str | None"
    motion_removed_channel_ids: "list | None"
    motion_spans_without_evidence: "list[dict] | None"
    # Stage statuses of the motion stages (present with the matching
    # ``stage_seconds`` key).
    motion_estimate_status: NotRequired[StageStatus]
    motion_corrected_recording_status: NotRequired[StageStatus]
    # Auto-curation keys, present only when ``run_v2_pipeline(auto_curate=True)``:
    # the CurationEvaluation suggestion selection PK and the stage status. The
    # materialized child itself is the always-present ``auto_labeled_*`` pair.
    curation_evaluation_id: NotRequired[UUID]
    auto_curation_status: NotRequired[StageStatus]
    # FigPack keys, present only when
    # ``run_v2_pipeline(build_figpack_view=True)``: the published curation-view
    # URI (a local bundle path offline) and its stage status. ``figpack_uri`` is
    # absent and ``figpack_status`` is ``"skipped"`` when the sort found zero
    # units (no analyzer to summarize).
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


class MemberArtifactSummary(TypedDict):
    """Exact per-member detection and frame-based masked duration."""

    member_index: int
    artifact_detection_id: UUID
    status: StageStatus
    masked_duration_s: float


class RunV2ConcatSummary(_RunV2SummaryBase):
    """``run_v2_pipeline`` summary for a concat (SessionGroup) run.

    ``source_mode == "concat"``. Carries the per-member recording PKs and the
    ConcatenatedRecording keys in place of the single-session recording keys,
    one wall-clock-aligned merge ID per member session, and per-member artifact
    results. Masks belong to the concat source, not SortingSelection.
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
    # Keyed by the frozen member index, not NWB filename: one NWB may
    # legitimately contribute several intervals or sort groups.
    member_merge_ids: dict[int, UUID]


# A run_v2_pipeline summary is exactly one of the two modes; ``source_mode`` is
# the discriminant a consumer narrows on.
RunV2PipelineSummary: TypeAlias = RunV2SingleSessionSummary | RunV2ConcatSummary


class MotionEstimateDiagnostics(TypedDict):
    """Peak and displacement counts of one saved ``MotionEstimate``.

    The same-named ``MotionEstimate`` columns: peaks detected on the masked
    source, peaks kept (localized inside one statistics span and detected
    clear of span joins), the largest absolute displacement in micrometers
    over all temporal bins and spatial windows, and the number of temporal
    bins.
    """

    n_peaks_detected: int
    n_peaks_kept: int
    max_abs_displacement_um: float
    n_temporal_bins: int


class EstimateMotionReceipt(TypedDict):
    """Return value of ``estimate_motion``.

    The saved estimate of the source a ``run_v2_pipeline`` call with the same
    source arguments would sort, before any sort. Pass
    ``motion_estimate_id`` to ``run_v2_pipeline(motion_mode="apply",
    motion_estimate_id=...)`` to sort the recording corrected with exactly
    this estimate. ``source_mode`` says which source keys are present:
    single-session receipts carry the recording and artifact-detection keys
    (``artifact_detection_id`` is ``None`` and its status ``"skipped"`` when
    the preset runs no artifact detection); concat receipts carry the member
    and concatenation keys, as in :class:`RunV2ConcatSummary`.
    ``stage_seconds`` holds each source stage and ``motion_estimate``.
    """

    pipeline_preset: str
    source_mode: SourceMode
    motion_correction_params_name: str
    motion_estimate_id: UUID
    # The SpikeInterface preset the recipe's estimation row resolved to.
    motion_estimation_preset: str
    motion_estimate_status: StageStatus
    # ``MotionEstimate.get_spans_without_evidence``; empty when every
    # continuity span kept a peak.
    motion_spans_without_evidence: list[dict]
    motion_diagnostics: MotionEstimateDiagnostics
    stage_seconds: dict[str, float]
    warnings: list[str]
    # Single-session source keys.
    recording_id: NotRequired[UUID]
    recording_status: NotRequired[StageStatus]
    artifact_detection_id: NotRequired["UUID | None"]
    artifact_detection_status: NotRequired[StageStatus]
    # Concat source keys.
    member_recording_ids: NotRequired[list[UUID]]
    member_recording_status: NotRequired[StageStatus]
    member_artifact_detection_status: NotRequired[StageStatus]
    member_artifacts: NotRequired[list[MemberArtifactSummary]]
    artifact_masked_duration_s: NotRequired[float]
    concat_recording_id: NotRequired[UUID]
    concat_recording_status: NotRequired[StageStatus]


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
    """One discovered curation candidate for a SessionGroup member.

    Any ``CurationV2`` row sorted from one of the member's recordings -- NOT
    pre-validated for matching (a preview or a wrong-member pin is rejected
    later by ``UnitMatchSelection.insert_selection``); one such candidate is
    one row of the ``describe_unit_match_choices`` table.
    ``parent_curation_id == -1`` marks a root curation.
    """

    sorting_id: UUID
    curation_id: int
    parent_curation_id: int
    curation_source: str
    description: str


class UnitMatchMemberChoices(TypedDict):
    """One SessionGroup member and the curations available to pin for it.

    The structured per-member form that ``describe_unit_match_choices`` tabulates
    (one row per member x choice). Assemble ``run_v2_unit_match``'s
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


class UnitMatchInputSummary(NamedTuple):
    """One matching input of a ``run_v2_unit_match`` receipt.

    Read from the run's frozen ``UnitMatchSelection.Input`` /
    ``InputRecording`` rows -- never from a live ``SessionGroup`` and
    without opening the run's NWB.
    ``nwb_file_names`` / ``interval_list_names`` list the input's
    constituent original recordings in recording order (one for a
    single-recording sort, one per member for a concatenation sort).
    ``motion_corrected_recording_id`` is the ``MotionCorrectedRecording``
    the input's bundle was read from, ``None`` when it read the sort's own
    source; ``waveform_traces`` names that traces kind
    (``"motion_corrected_recording"``, ``"recording"`` or
    ``"concatenated_recording"``).
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

    The cross-session match manifest: the ``UnitMatch`` selection PK plus the
    pairwise-match and tracked-unit results, with per-stage status / timing.
    ``session_group_owner`` / ``session_group_name`` name the ``SessionGroup``
    matched, and are ``None`` for a run of named sorts
    (``plan_v2_unit_match_from_sorts``). ``inputs`` holds one
    :class:`UnitMatchInputSummary` per matching input, in ``input_index``
    (chronological) order.
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
    "UnitMatchInputSummary",
    "UnitMatchMemberChoices",
    "UnitMatchStageSeconds",
]
