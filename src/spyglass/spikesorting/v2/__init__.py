"""SpikeInterface ``SortingAnalyzer``-based spike sorting pipeline for Spyglass.

Submodules load lazily, so ``import spyglass.spikesorting.v2`` does not import
Pydantic or SpikeInterface. Tables (``from spyglass.spikesorting.v2 import
Recording``) and the pipeline functions load on first use. Eager names:

- ``initialize_v2_defaults``: installs every default Lookup row in one call.
- ``CurationLabel``: the canonical curation label set.
"""

from spyglass.spikesorting.v2._core.enums import CurationLabel
from spyglass.utils.parallel import install_openblas_fork_repair

# SpikeInterface's forked workers otherwise deadlock SciPy's OpenBLAS.
install_openblas_fork_repair()


def initialize_v2_defaults() -> None:
    """Install the default Lookup rows for the v2 spike-sorting stages.

    Calls ``insert_default()`` on ``PreprocessingParameters``,
    ``ArtifactDetectionParameters``, ``SorterParameters``,
    ``AnalyzerWaveformParameters``, ``QualityMetricParameters``,
    ``AutoCurationRules``, ``CurationReviewProfile``, ``MatcherParameters``,
    and ``MotionCorrectionParameters`` (with the estimation and interpolation
    recipes it names), then adds the v2 unit-selection policies. Idempotent.

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


__all__ = [
    "initialize_v2_defaults",
    "CurationLabel",
]
