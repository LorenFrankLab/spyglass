"""Modern SpikeInterface-based spike sorting pipeline for Spyglass.

This package targets the SpikeInterface ``SortingAnalyzer`` API. Table and
helper modules are imported lazily by callers rather than re-exported here, so
``import spyglass.spikesorting.v2`` stays free of optional runtime
dependencies (Pydantic, modern SpikeInterface) until a submodule is used.

The eager symbols are kept dependency-light: ``initialize_v2_defaults``
installs every default Lookup row the pipeline needs in one call (removing the
"forgot to call ``insert_default``" first-run friction), and ``CurationLabel``
is re-exported from the stdlib-only ``_enums`` module so notebook users can
discover the canonical label set without importing a table module.
"""

from spyglass.spikesorting.v2._enums import CurationLabel


def initialize_v2_defaults() -> None:
    """Install the default Lookup rows for every v2 spike-sorting stage.

    Calls ``insert_default()`` on ``PreprocessingParameters``,
    ``ArtifactDetectionParameters``, ``SorterParameters``,
    ``AnalyzerWaveformParameters``, ``MotionCorrectionParameters``,
    ``QualityMetricParameters``, ``AutoCurationRules``, and
    ``MatcherParameters`` (each accepts duplicate-row noise), so a notebook
    user can run one helper instead of remembering the per-table calls before
    the first ``run_v2_pipeline`` / cross-session match invocation. Idempotent.

    ``MotionCorrectionParameters`` presets are seeded here so a missing
    motion-preset row does not surface as an opaque FK violation on the first
    ``ConcatenatedRecording`` run (the same-day chronic concatenate-and-sort
    consumer).

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
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.session_group import (
        MotionCorrectionParameters,
    )
    from spyglass.spikesorting.v2.sorting import (
        AnalyzerWaveformParameters,
        SorterParameters,
    )
    from spyglass.spikesorting.v2.unit_matching import MatcherParameters

    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()
    AnalyzerWaveformParameters.insert_default()
    MotionCorrectionParameters.insert_default()
    QualityMetricParameters.insert_default()
    AutoCurationRules.insert_default()
    MatcherParameters.insert_default()

    # ``insert_default`` skips existing-PK rows, so a stored same-name default
    # whose content has diverged from the shipped content (e.g. a row seeded at
    # an old schema version, or a hand-edited blob) is NOT reseeded and goes
    # unnoticed. Audit for that here and WARN (non-strict) -- an admin can run
    # ``verify_v2_default_catalog(strict=True)`` to fail hard.
    from spyglass.spikesorting.v2._pipeline_reporting import (
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


__all__ = [
    "initialize_v2_defaults",
    "verify_v2_default_catalog",
    "CurationLabel",
]


def __getattr__(name):
    # Lazily re-export ``verify_v2_default_catalog`` from the reporting module so
    # ``from spyglass.spikesorting.v2 import verify_v2_default_catalog`` works
    # without importing the DataJoint reporting layer at package import.
    if name == "verify_v2_default_catalog":
        from spyglass.spikesorting.v2._pipeline_reporting import (
            verify_v2_default_catalog,
        )

        return verify_v2_default_catalog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
