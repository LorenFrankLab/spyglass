"""Orchestration entrypoints across the spike sorting tables.

- ``run_v2_pipeline``: recording -> artifact -> sort -> curation for one
  session, with optional metric auto-curation (``auto_curate=True``). Browser
  review starts from ``RunResult.start_review``.
- ``run_v2_concat_pipeline``: the same stages on a same-day ``SessionGroup``.
- ``run_v2_unit_match``: cross-session unit matching.
- ``estimate_motion`` / ``estimate_concat_motion``: save only the motion
  estimate, for inspection before a run with
  ``motion_mode="apply", motion_estimate_id=...``.

Presets are Pydantic-validated bundles of Lookup-row names. The default is
``franklab_probe_hippocampus_30khz_ms4_2026_06``; see
``describe_pipeline_presets()`` and ``list_pipeline_presets()``.

Re-running with the same inputs inserts nothing and returns the same summary
(same ``root_merge_id``).

This module re-exports the public names from ``_orchestration`` so user code
imports from one path.
"""

from __future__ import annotations

from spyglass.spikesorting.v2._orchestration.geometry import (
    describe_sort_groups,
    plot_sort_group_geometry,
)
from spyglass.spikesorting.v2._orchestration.epochs import describe_epochs
from spyglass.spikesorting.v2._orchestration.preflight import (
    PreflightCheck,
    PreflightReport,
    PreflightSessionReport,
    preflight_v2_pipeline,
    preflight_v2_pipeline_session,
)
from spyglass.spikesorting.v2._orchestration.presets import (
    _PIPELINE_PRESETS,
    _PipelinePreset,
    clone_pipeline_preset,
    describe_pipeline_preset,
    describe_pipeline_presets,
    describe_recommendation_status,
    list_pipeline_presets,
    register_pipeline_preset,
)
from spyglass.spikesorting.v2._orchestration.reporting import (
    describe_parameter_rows,
    describe_run,
    describe_units,
)
from spyglass.spikesorting.v2._orchestration.matching import (
    describe_unit_match_choices,
    plan_v2_unit_match,
    plan_v2_unit_match_from_sorts,
    plan_v2_unit_match_from_curations,
    run_v2_unit_match,
)
from spyglass.spikesorting.v2._orchestration.motion import (
    estimate_concat_motion,
    estimate_motion,
)
from spyglass.spikesorting.v2._orchestration.run import (
    run_v2_concat_pipeline,
    run_v2_pipeline,
)
from spyglass.spikesorting.v2._orchestration.session import (
    run_v2_pipeline_session,
)
from spyglass.spikesorting.v2._matching.planning import (
    UnitMatchInputPlan,
    UnitMatchPlan,
)
from spyglass.spikesorting.v2._matching.preflight import (
    UnitMatchPreflightCheck,
    UnitMatchPreflightReport,
    preflight_v2_unit_match,
)
from spyglass.spikesorting.v2._curation.analyzer import open_curation_analyzer
from spyglass.spikesorting.v2.analysis_selection import (
    V2_UNIT_SELECTION_POLICIES,
    SelectedGroup,
    UnitSelectionReceipt,
    UnitSelectionSnapshot,
    resolve_unit_selection,
    select_units_for_analysis,
)
from spyglass.spikesorting.v2.annotation_api import (
    AnnotationDefinitionRef,
    AnnotationSetRef,
    read_unit_properties,
)
from spyglass.spikesorting.v2.curation_api import (
    CurationRef,
    EvaluationResult,
    EvaluationSpec,
    MergedCuration,
    MergeEvaluateReceipt,
    RunResult,
)
from spyglass.spikesorting.v2.review_api import (
    CurationChangeSet,
    FigPackReview,
    MergeLabelConflict,
    ReviewImportReceipt,
    ReviewProfileRef,
    ReviewStageStatus,
)
from spyglass.spikesorting.v2._orchestration.types import (
    EstimateMotionReceipt,
    MotionEstimateDiagnostics,
    MotionMode,
    PipelineOutcome,
    PipelineStageSeconds,
    RunV2ConcatSummary,
    RunV2ConcatPipelineInputs,
    RunV2PipelineInputs,
    RunV2PipelineOptions,
    RunV2PipelineSessionFailed,
    RunV2PipelineSessionInputs,
    RunV2PipelineSessionOk,
    RunV2PipelineSessionRequiredInputs,
    RunV2PipelineSessionResult,
    RunV2PipelineSummary,
    RunV2SingleSessionSummary,
    RunV2UnitMatchSummary,
    SourceMode,
    StageStatus,
    UnitMatchCurationChoice,
    UnitMatchInputSummary,
    UnitMatchMemberChoices,
    UnitMatchStageSeconds,
)

__all__ = [
    # also lazy names of the package root
    "run_v2_pipeline",
    "run_v2_concat_pipeline",
    "run_v2_pipeline_session",
    "estimate_motion",
    "estimate_concat_motion",
    "run_v2_unit_match",
    "plan_v2_unit_match",
    "plan_v2_unit_match_from_sorts",
    "plan_v2_unit_match_from_curations",
    "preflight_v2_unit_match",
    "preflight_v2_pipeline",
    "preflight_v2_pipeline_session",
    "describe_run",
    "describe_units",
    "describe_parameter_rows",
    "describe_sort_groups",
    "describe_epochs",
    "describe_pipeline_presets",
    "describe_pipeline_preset",
    "describe_recommendation_status",
    "list_pipeline_presets",
    "register_pipeline_preset",
    "clone_pipeline_preset",
    "describe_unit_match_choices",
    "plot_sort_group_geometry",
    "RunResult",
    "CurationRef",
    "EvaluationSpec",
    "EvaluationResult",
    "MergedCuration",
    "MergeEvaluateReceipt",
    "ReviewProfileRef",
    "FigPackReview",
    "CurationChangeSet",
    "MergeLabelConflict",
    "ReviewImportReceipt",
    "ReviewStageStatus",
    "AnnotationDefinitionRef",
    "AnnotationSetRef",
    "read_unit_properties",
    "open_curation_analyzer",
    "select_units_for_analysis",
    "resolve_unit_selection",
    "UnitSelectionSnapshot",
    "UnitSelectionReceipt",
    "SelectedGroup",
    "V2_UNIT_SELECTION_POLICIES",
    # facade only: preflight reports
    "PreflightCheck",
    "PreflightReport",
    "PreflightSessionReport",
    "UnitMatchPreflightCheck",
    "UnitMatchPreflightReport",
    # typed run summaries / plans
    "UnitMatchPlan",
    "UnitMatchInputPlan",
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
    "EstimateMotionReceipt",
    "MotionEstimateDiagnostics",
    # enums / carriers
    "MotionMode",
    "SourceMode",
    "StageStatus",
    "UnitMatchCurationChoice",
    "UnitMatchInputSummary",
    "UnitMatchMemberChoices",
    "UnitMatchStageSeconds",
]
