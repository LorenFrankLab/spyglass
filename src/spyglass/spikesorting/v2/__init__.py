"""SpikeInterface ``SortingAnalyzer``-based spike sorting pipeline for Spyglass.

Submodules load lazily, so ``import spyglass.spikesorting.v2`` does not import
SpikeInterface or the table modules. Tables (``from spyglass.spikesorting.v2 import
Recording``) and the pipeline functions load on first use, including
``initialize_v2_defaults``, which installs every default Lookup row in one call.
``CurationLabel``, the canonical curation label set, is the one eager name.
"""

from importlib import import_module

from spyglass.spikesorting.v2._core.enums import CurationLabel
from spyglass.utils.parallel import install_openblas_fork_repair

# SpikeInterface's forked workers otherwise deadlock SciPy's OpenBLAS.
install_openblas_fork_repair()


#: Tables, imported from their module on first access.
_TABLE_MODULES = {
    "ArtifactDetectionParameters": "artifact",
    "SharedArtifactGroup": "artifact",
    "RecordingArtifactSelection": "artifact",
    "SharedGroupArtifactSelection": "artifact",
    "RecordingArtifactDetection": "artifact",
    "SharedGroupArtifactDetection": "artifact",
    "ArtifactDetectionOutput": "artifact_output",
    "ConcatMemberCurationV2": "concat_member_curation",
    "CurationV2": "curation",
    "FigPackCurationSelection": "figpack_curation",
    "FigPackCuration": "figpack_curation",
    "QualityMetricParameters": "metric_curation",
    "AutoCurationRules": "metric_curation",
    "CurationEvaluationSelection": "metric_curation",
    "CurationEvaluation": "metric_curation",
    "MotionEstimationParameters": "motion",
    "MotionInterpolationParameters": "motion",
    "MotionCorrectionParameters": "motion",
    "MotionEstimateSelection": "motion",
    "MotionEstimate": "motion",
    "MotionCorrectedRecordingSelection": "motion",
    "MotionCorrectedRecording": "motion",
    "PipelinePreset": "pipeline_preset",
    "RecordingArtifactVersions": "recompute",
    "RecordingArtifactRecomputeSelection": "recompute",
    "RecordingArtifactRecompute": "recompute",
    "SortingAnalyzerVersions": "recompute",
    "SortingAnalyzerRecomputeSelection": "recompute",
    "SortingAnalyzerRecompute": "recompute",
    "SortGroupV2": "recording",
    "PreprocessingParameters": "recording",
    "RecordingSelection": "recording",
    "Recording": "recording",
    "DriftEstimate": "recording",
    "CurationReviewProfile": "review_profile",
    "SessionGroup": "session_group",
    "ConcatenatedRecordingSelection": "session_group",
    "ConcatenatedRecording": "session_group",
    "SorterParameters": "sorting",
    "AnalyzerWaveformParameters": "sorting",
    "PreprocessingWaveformParameters": "sorting",
    "SortingSelection": "sorting",
    "Sorting": "sorting",
    "MatcherParameters": "unit_matching",
    "UnitMatchSelection": "unit_matching",
    "UnitMatch": "unit_matching",
    "TrackedUnit": "unit_matching",
}

# Names after ``CurationLabel`` resolve lazily from ``.pipeline``.
__all__ = [
    "CurationLabel",
    "initialize_v2_defaults",
    "verify_v2_default_catalog",
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
    "preflight_v2_concat_pipeline",
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
    "MergeEvaluateReceipt",
    "ReviewProfileRef",
    "describe_review_profiles",
    "FigPackReview",
    "CurationChangeSet",
    "MergeLabelConflict",
    "ReviewImportReceipt",
    "ReviewStageStatus",
    "open_curation_analyzer",
    "select_units_for_analysis",
    "resolve_unit_selection",
    "UnitSelectionSnapshot",
    "UnitSelectionReceipt",
    "SelectedGroup",
    "V2_UNIT_SELECTION_POLICIES",
]


def __dir__():
    # Expose lazy re-exports to tab-completion.
    return sorted({*globals(), *__all__, *_TABLE_MODULES})


def __getattr__(name):
    if name in _TABLE_MODULES:
        module = import_module(f"{__name__}.{_TABLE_MODULES[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in __all__:
        pipeline = import_module(f"{__name__}.pipeline")
        value = getattr(pipeline, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
