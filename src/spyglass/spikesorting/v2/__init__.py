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
    ``ArtifactDetectionParameters``, and ``MotionCorrectionParameters`` (with
    the estimation and interpolation recipes it names). Idempotent.

    Examples
    --------
    >>> from spyglass.spikesorting.v2 import initialize_v2_defaults
    >>> initialize_v2_defaults()
    """
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.motion import MotionCorrectionParameters
    from spyglass.spikesorting.v2.recording import PreprocessingParameters

    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    MotionCorrectionParameters.insert_default()


__all__ = [
    "initialize_v2_defaults",
    "CurationLabel",
]
