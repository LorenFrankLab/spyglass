"""Modern SpikeInterface-based spike sorting pipeline for Spyglass.

This package targets the SpikeInterface ``SortingAnalyzer`` API. Table and
helper modules are imported lazily by callers rather than re-exported here, so
``import spyglass.spikesorting.v2`` stays free of optional runtime
dependencies (Pydantic, modern SpikeInterface) until a submodule is used.

The eager symbols are kept dependency-light: ``initialize_v2_defaults``
installs every default Lookup row the pipeline needs in one call (removing the
"forgot to call ``insert_default``" first-run friction), and ``CurationLabel``
is re-exported from the stdlib-only ``_core.enums`` module so notebook users can
discover the canonical label set without importing a table module.
"""

from spyglass.spikesorting.v2._core.enums import CurationLabel
from spyglass.spikesorting.v2._core.job_config import (
    install_openblas_fork_repair,
)

# SpikeInterface's forked workers otherwise deadlock SciPy's OpenBLAS.
install_openblas_fork_repair()


def initialize_v2_defaults() -> None:
    """Install the default Lookup rows for the v2 spike-sorting stages.

    Calls ``insert_default()`` on ``PreprocessingParameters``,
    ``ArtifactDetectionParameters``, and ``MotionCorrectionParameters`` (with
    the estimation and interpolation recipes it names) -- each accepts
    duplicate-row noise -- so a notebook user can run one helper instead of
    remembering the per-table calls. Idempotent.

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
