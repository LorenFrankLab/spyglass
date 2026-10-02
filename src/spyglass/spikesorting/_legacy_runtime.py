"""Runtime guard for v0/v1 spike-sorting paths that require legacy SpikeInterface.

Several v0 and v1 active populate / curation / recompute paths still call
SpikeInterface APIs that were removed or renamed in SpikeInterface 0.101+
(``WaveformExtractor`` construction / ``extract_waveforms`` /
``ChunkRecordingExecutor`` signature widening / quality-metric renames). Those
entry points are gated behind an explicit legacy-environment error rather than
allowed to crash with an opaque ``AttributeError`` or ``AssertionError`` from
inside SpikeInterface. Reading a previously saved binary waveform folder is
*not* among them: 0.101+ reads those back as a ``MockWaveformExtractor``, so
that read is routed through ``_si_compat.load_waveforms`` and left ungated.

The guard is intentionally narrow:

- Read-only / query paths that do not invoke removed APIs continue to work and
  are not guarded. Renamed loading and ``NumpySorting`` APIs are routed through
  ``spyglass.spikesorting._si_compat`` so those paths work under both
  SpikeInterface 0.99 and 0.101+.
- v0/v1 schemas are unchanged; ``SpikeSortingOutput`` merge queries on existing
  rows keep functioning.
- Modern (v2) spike-sorting code is unaffected.

Callers add ``_require_legacy_si_environment()`` as the first statement of any
``make()`` / public entry point that needs the legacy SpikeInterface runtime.
The helper is a no-op when running under SpikeInterface < 0.101.
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
        "computing new v0/v1 waveforms, quality metrics, artifact detection, "
        "burst curation, and clusterless features is not. "
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
        Human-readable name of the guarded entry point, e.g.
        ``"v1 SpikeSorting.make"``. Used in the error message so the
        traceback points at the workflow the caller invoked.

    Raises
    ------
    RuntimeError
        When the installed SpikeInterface version is at or past 0.101 (the
        first release that removed the WaveformExtractor-era APIs the v0/v1
        active runtime paths still depend on).
    """
    import spikeinterface

    if Version(spikeinterface.__version__) >= _LEGACY_BOUNDARY:
        raise RuntimeError(_legacy_runtime_message(component))
