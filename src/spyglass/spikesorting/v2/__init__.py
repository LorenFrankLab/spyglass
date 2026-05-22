"""Modern SpikeInterface-based spike sorting pipeline for Spyglass.

This package targets the SpikeInterface ``SortingAnalyzer`` API. Table and
helper modules are imported lazily by callers rather than re-exported here, so
``import spyglass.spikesorting.v2`` stays free of optional runtime
dependencies (Pydantic, modern SpikeInterface) until a submodule is used.

The ``initialize_v2_defaults`` helper is the only eager symbol: it installs
every default Lookup row the pipeline needs in one call, removing the
"forgot to call ``insert_default``" first-run friction.
"""


def initialize_v2_defaults() -> None:
    """Install the default Lookup rows for every v2 spike-sorting stage.

    Calls ``insert_default()`` on ``PreprocessingParameters``,
    ``ArtifactDetectionParameters``, and ``SorterParameters`` (each
    accepts duplicate-row noise), so a notebook user can run one helper
    instead of remembering three calls before the first
    ``run_v2_pipeline`` invocation. Idempotent.

    Examples
    --------
    >>> from spyglass.spikesorting.v2 import initialize_v2_defaults
    >>> initialize_v2_defaults()
    """
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.sorting import SorterParameters

    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()


__all__ = ["initialize_v2_defaults"]
