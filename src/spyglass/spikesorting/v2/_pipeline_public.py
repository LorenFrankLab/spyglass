"""Public import-surface names for the v2 pipeline facade.

Kept dependency-light so the package root can expose lazy re-exports without
importing DataJoint / SpikeInterface-backed modules at import time.
"""

# Names re-exported lazily at the package root (``spyglass.spikesorting.v2``)
# and by the facade.
PACKAGE_ROOT_REEXPORTS = (
    "run_v2_pipeline",
    "run_v2_pipeline_session",
    "run_v2_unit_match",
    "plan_v2_unit_match",
    "preflight_v2_pipeline",
    "preflight_v2_pipeline_session",
    "describe_run",
    "describe_units",
    "describe_parameter_rows",
    "describe_sort_groups",
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
    "UnitSelectionReceipt",
    "SelectedGroup",
    "V2_UNIT_SELECTION_POLICIES",
)

# Facade-only names: importable from ``spyglass.spikesorting.v2.pipeline``
# but not re-exported at the package root.
_FACADE_ONLY_EXPORTS = (
    # preflight reports
    "PreflightCheck",
    "PreflightReport",
    "PreflightSessionReport",
    # typed run summaries / plans
    "UnitMatchPlan",
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
    # enums / carriers
    "SourceMode",
    "StageStatus",
    "UnitMatchCurationChoice",
    "UnitMatchMemberChoices",
    "UnitMatchStageSeconds",
)


PIPELINE_FACADE_EXPORTS = (
    *PACKAGE_ROOT_REEXPORTS,
    *_FACADE_ONLY_EXPORTS,
)
