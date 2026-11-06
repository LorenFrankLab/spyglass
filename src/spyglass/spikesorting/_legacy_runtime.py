"""Guard for v0/v1 entry points that need SpikeInterface < 0.101.

- Compute paths that call APIs removed in SI 0.101 (``extract_waveforms``,
  ``WaveformExtractor``, quality-metric renames) raise a clear error instead
  of failing inside SpikeInterface.
- Read paths are not guarded; renamed APIs go through ``_si_compat``.
"""

from __future__ import annotations

from packaging.version import Version

_LEGACY_BOUNDARY = Version("0.101")


def _legacy_runtime_message(component: str) -> str:
    """Compose the error message raised by the legacy-environment guard."""
    return (
        f"{component} requires the legacy SpikeInterface 0.99 environment. "
        "Existing v0/v1 rows, and their saved recordings, sortings, and "
        "binary-folder waveforms, remain readable under SpikeInterface "
        "0.101 and later; "
        "computing new v0/v1 sortings (other than clusterless_thresholder), "
        "waveforms, quality metrics, artifact detection, burst curation, and "
        "clusterless features is not. "
        "To continue this workflow: "
        "either run it in a separate legacy environment built from "
        "environments/environment_spikesorting_legacy.yml, or switch new "
        "processing to the modern v2 spike-sorting pipeline (see 'Two "
        "environments, one database' in the Spike Sorting v2 migration guide)."
    )


def _require_legacy_si_environment(component: str) -> None:
    """Raise ``RuntimeError`` when SpikeInterface is past the legacy boundary.

    Parameters
    ----------
    component : str
        Name of the guarded entry point for the error message, e.g.
        ``"v1 SpikeSorting.make"``.

    Raises
    ------
    RuntimeError
        When the installed SpikeInterface is 0.101 or later.
    """
    import spikeinterface

    if Version(spikeinterface.__version__) >= _LEGACY_BOUNDARY:
        raise RuntimeError(_legacy_runtime_message(component))
