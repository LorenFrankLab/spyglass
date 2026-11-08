"""SpikeInterface ``SortingAnalyzer``-based spike sorting pipeline for Spyglass.

Submodules load lazily, so ``import spyglass.spikesorting.v2`` does not import
Pydantic or SpikeInterface. Tables (``from spyglass.spikesorting.v2 import
Recording``) and the pipeline functions load on first use. Eager names:

- ``initialize_v2_defaults``: installs every default Lookup row in one call.
- ``CurationLabel``: the canonical curation label set.
"""

from importlib import import_module

from spyglass.spikesorting.v2._core.enums import CurationLabel
from spyglass.utils.parallel import install_openblas_fork_repair

# SpikeInterface's forked workers otherwise deadlock SciPy's OpenBLAS.
install_openblas_fork_repair()


def initialize_v2_defaults() -> None:
    """Install the default Lookup rows for every v2 spike-sorting stage.

    Calls ``insert_default()`` on ``PreprocessingParameters``,
    ``ArtifactDetectionParameters``, ``SorterParameters``,
    ``AnalyzerWaveformParameters``, ``QualityMetricParameters``,
    ``AutoCurationRules``, ``MatcherParameters``, ``CurationReviewProfile``,
    and ``MotionCorrectionParameters`` (with the estimation and interpolation
    recipes it names), then adds the v2 unit-selection policies. Idempotent.
    Warns when a stored default row differs from the shipped content.

    Examples
    --------
    >>> from spyglass.spikesorting.v2 import initialize_v2_defaults
    >>> initialize_v2_defaults()
    """
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2.motion import MotionCorrectionParameters
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.review_profile import CurationReviewProfile
    from spyglass.spikesorting.v2.sorting import (
        AnalyzerWaveformParameters,
        SorterParameters,
    )
    from spyglass.spikesorting.v2.unit_matching import MatcherParameters

    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()
    AnalyzerWaveformParameters.insert_default()
    QualityMetricParameters.insert_default()
    AutoCurationRules.insert_default()
    MatcherParameters.insert_default()
    CurationReviewProfile.insert_default()
    MotionCorrectionParameters.insert_default()
    # Additive rows in the shared UnitSelectionParams table.
    from spyglass.spikesorting.v2.analysis_selection import (
        ensure_v2_unit_selection_policies,
    )

    ensure_v2_unit_selection_policies()

    # insert_default skips existing keys, so a diverged stored default is
    # never reseeded; warn about it here.
    from spyglass.spikesorting.v2._orchestration.reporting import (
        verify_v2_default_catalog,
    )
    from spyglass.utils import logger

    stale = verify_v2_default_catalog(strict=False)
    if stale:
        logger.warning(
            "initialize_v2_defaults: %d stored default row(s) diverge from the "
            "shipped content and were NOT reseeded (insert_default skips "
            "existing keys): %s. Drop the stale row(s) and re-run, or call "
            "verify_v2_default_catalog(strict=True) to inspect.",
            len(stale),
            stale,
        )


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

# Names after the first three resolve lazily from ``.pipeline``.
__all__ = [
    "initialize_v2_defaults",
    "verify_v2_default_catalog",
    "CurationLabel",
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
    "MergeEvaluateReceipt",
    "ReviewProfileRef",
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
    if name == "verify_v2_default_catalog":
        from spyglass.spikesorting.v2._orchestration.reporting import (
            verify_v2_default_catalog,
        )

        globals()[name] = verify_v2_default_catalog
        return verify_v2_default_catalog
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
