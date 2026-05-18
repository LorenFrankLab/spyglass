"""Modern SpikeInterface-based spike sorting pipeline for Spyglass.

This package targets the SpikeInterface ``SortingAnalyzer`` API. Table and
helper modules are imported lazily by callers rather than re-exported here, so
``import spyglass.spikesorting.v2`` stays free of optional runtime
dependencies (Pydantic, modern SpikeInterface) until a submodule is used.
"""

__all__: list[str] = []
