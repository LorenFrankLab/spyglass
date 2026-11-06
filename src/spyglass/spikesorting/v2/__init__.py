"""SpikeInterface ``SortingAnalyzer``-based spike sorting pipeline for Spyglass.

Submodules load lazily, so ``import spyglass.spikesorting.v2`` does not import
Pydantic or SpikeInterface. Eager names:

- ``initialize_v2_defaults``: installs every default Lookup row in one call.
- ``CurationLabel``: the canonical curation label set.
"""

from importlib import import_module

from spyglass.spikesorting.v2._core.enums import CurationLabel
from spyglass.spikesorting.v2._core.job_config import (
    install_openblas_fork_repair,
)
from spyglass.spikesorting.v2._orchestration.exports import (
    PACKAGE_ROOT_REEXPORTS,
)

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


# Entrypoints re-exported lazily from ``.pipeline``, resolved on first access.
_PIPELINE_REEXPORTS = PACKAGE_ROOT_REEXPORTS


__all__ = [
    "initialize_v2_defaults",
    "verify_v2_default_catalog",
    "CurationLabel",
    *_PIPELINE_REEXPORTS,
]


def __dir__():
    # Expose lazy re-exports to tab-completion.
    return sorted({*globals(), *__all__})


def __getattr__(name):
    if name == "verify_v2_default_catalog":
        from spyglass.spikesorting.v2._orchestration.reporting import (
            verify_v2_default_catalog,
        )

        globals()[name] = verify_v2_default_catalog
        return verify_v2_default_catalog
    if name in _PIPELINE_REEXPORTS:
        pipeline = import_module(f"{__name__}.pipeline")
        value = getattr(pipeline, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
