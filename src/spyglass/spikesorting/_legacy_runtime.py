"""Runtime guard for v0/v1 spike-sorting paths that require legacy SpikeInterface.

Several v0 and v1 active populate / curation / recompute paths still call
SpikeInterface APIs that were removed or renamed in SpikeInterface 0.101+
(``WaveformExtractor`` / ``extract_waveforms`` / ``load_waveforms`` /
``ChunkRecordingExecutor`` signature widening / quality-metric renames). Those
entry points are gated behind an explicit legacy-environment error rather than
allowed to crash with an opaque ``AttributeError`` or ``AssertionError`` from
inside SpikeInterface.

The guard is intentionally narrow:

- Read-only / query paths that do not invoke removed APIs continue to work and
  are not guarded.
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
    """Compose the prescribed legacy-environment error message."""
    return (
        f"{component} requires the legacy SpikeInterface 0.99 environment. "
        "Existing v0/v1 rows remain queryable under the new pin; only active "
        "populate / curation / recompute is gated. To continue this workflow: "
        "either downgrade to a legacy Spyglass install pinned to "
        "spikeinterface<0.101 in a separate conda environment, or switch new "
        "processing to the modern v2 spike-sorting pipeline (see "
        "src/spyglass/spikesorting/v2/ and the CHANGELOG entry for the "
        "SpikeInterface 0.104 boundary)."
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
