"""Public import-surface names for the v2 pipeline facade.

Kept dependency-light so the package root can expose lazy re-exports without
importing DataJoint / SpikeInterface-backed modules at import time.
"""

PIPELINE_FACADE_EXPORTS = (
    # presets
    "list_pipeline_presets",
    "describe_pipeline_presets",
    "describe_pipeline_preset",
    "register_pipeline_preset",
    "clone_pipeline_preset",
    # geometry
    "describe_sort_groups",
    "plot_sort_group_geometry",
    # preflight
    "PreflightCheck",
    "PreflightReport",
    "PreflightSessionReport",
    "preflight_v2_pipeline",
    "preflight_v2_pipeline_session",
    # reporting
    "describe_parameter_rows",
    "describe_run",
    "describe_units",
    # run / matching
    "run_v2_pipeline",
    "run_v2_pipeline_session",
    "run_v2_unit_match",
    "plan_v2_unit_match",
    "describe_unit_match_choices",
    # public typed summaries / plans
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
    "SourceMode",
    "StageStatus",
    "UnitMatchCurationChoice",
    "UnitMatchMemberChoices",
    "UnitMatchStageSeconds",
)


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
    "list_pipeline_presets",
    "register_pipeline_preset",
    "clone_pipeline_preset",
    "describe_unit_match_choices",
    "plot_sort_group_geometry",
)


_missing_from_facade = set(PACKAGE_ROOT_REEXPORTS) - set(
    PIPELINE_FACADE_EXPORTS
)
if _missing_from_facade:
    raise RuntimeError(
        "PACKAGE_ROOT_REEXPORTS must be a subset of "
        f"PIPELINE_FACADE_EXPORTS; missing {sorted(_missing_from_facade)}"
    )
