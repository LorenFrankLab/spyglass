"""SpikeInterface ``SortingAnalyzer``-based spike sorting pipeline for Spyglass.

Submodules load lazily, so ``import spyglass.spikesorting.v2`` does not import
SpikeInterface or the table modules. Tables (``from spyglass.spikesorting.v2 import
Recording``) load on first use. Eager names: ``initialize_v2_defaults``
installs the default Lookup rows in one call, and ``CurationLabel`` is the
canonical curation label set.
"""

from importlib import import_module

from spyglass.spikesorting.v2._core.enums import CurationLabel
from spyglass.utils.parallel import install_openblas_fork_repair

# SpikeInterface's forked workers otherwise deadlock SciPy's OpenBLAS.
install_openblas_fork_repair()


def initialize_v2_defaults() -> None:
    """Install the default Lookup rows for the v2 spike-sorting stages.

    Calls ``insert_default()`` on ``PreprocessingParameters``,
    ``ArtifactDetectionParameters``, ``SorterParameters``,
    ``AnalyzerWaveformParameters``, ``QualityMetricParameters``,
    ``AutoCurationRules``, and ``MotionCorrectionParameters`` (with the
    estimation and interpolation recipes it names), then adds the v2
    unit-selection policies. Idempotent.

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
    from spyglass.spikesorting.v2.sorting import (
        AnalyzerWaveformParameters,
        SorterParameters,
    )

    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()
    AnalyzerWaveformParameters.insert_default()
    QualityMetricParameters.insert_default()
    AutoCurationRules.insert_default()
    MotionCorrectionParameters.insert_default()
    # Additive rows in the shared UnitSelectionParams table.
    from spyglass.spikesorting.v2.analysis_selection import (
        ensure_v2_unit_selection_policies,
    )

    ensure_v2_unit_selection_policies()


#: Tables, imported from their module on first access.
_TABLE_MODULES = {
    "ArtifactDetectionParameters": "artifact",
    "SharedArtifactGroup": "artifact",
    "RecordingArtifactSelection": "artifact",
    "SharedArtifactSelection": "artifact",
    "RecordingArtifactDetection": "artifact",
    "SharedArtifactDetection": "artifact",
    "ArtifactDetectionOutput": "artifact_output",
    "ConcatMemberCurationV2": "concat_member_curation",
    "CurationV2": "curation",
    "CurationIdCounter": "curation",
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
    "RecordingRecomputeVersions": "recompute",
    "RecordingRecomputeSelection": "recompute",
    "RecordingRecompute": "recompute",
    "SortingAnalyzerVersions": "recompute",
    "SortingAnalyzerRecomputeSelection": "recompute",
    "SortingAnalyzerRecompute": "recompute",
    "SortGroupV2": "recording",
    "PreprocessingParameters": "recording",
    "RecordingSelection": "recording",
    "Recording": "recording",
    "SessionGroup": "session_group",
    "ConcatenatedRecordingSelection": "session_group",
    "ConcatenatedRecording": "session_group",
    "SorterParameters": "sorting",
    "AnalyzerWaveformParameters": "sorting",
    "PreprocessingWaveformParameters": "sorting",
    "SortingSelection": "sorting",
    "Sorting": "sorting",
}

#: Pipeline names, imported from their module on first access.
_PIPELINE_MODULES = {
    "open_curation_analyzer": "_curation.analyzer",
    "V2_UNIT_SELECTION_POLICIES": "analysis_selection",
    "resolve_unit_selection": "analysis_selection",
    "select_units_for_analysis": "analysis_selection",
    "CurationRef": "curation_api",
    "EvaluationResult": "curation_api",
    "RunResult": "curation_api",
}

__all__ = [
    "initialize_v2_defaults",
    "CurationLabel",
    "open_curation_analyzer",
    "V2_UNIT_SELECTION_POLICIES",
    "resolve_unit_selection",
    "select_units_for_analysis",
    "CurationRef",
    "EvaluationResult",
    "RunResult",
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
    if name in _PIPELINE_MODULES:
        module = import_module(f"{__name__}.{_PIPELINE_MODULES[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
