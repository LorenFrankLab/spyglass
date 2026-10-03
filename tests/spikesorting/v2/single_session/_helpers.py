"""Shared test helpers for the single-session pipeline suite.

These helpers are used by more than one of the ``test_*.py`` modules in
this subpackage (curation/dispatch tests clear curations; sorting and
artifact tests wrap synthetic traces), so they live here rather than in
any single module.
"""


def _clear_curations(sorting_key):
    """Drop a sorting's CurationV2 rows + merge masters (shared helper)."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    clear_curations_for(sorting_key)


def _build_synthetic_rec(traces, fs=30_000.0):
    """Helper: wrap a (n_samples, n_channels) array as a SI NumpyRecording
    with unit gains so trace values can be reasoned about as microvolts."""
    import spikeinterface as si

    rec = si.NumpyRecording(traces_list=[traces], sampling_frequency=fs)
    rec.set_channel_gains([1.0] * traces.shape[1])
    # Offsets too: threshold_unit="uv" scales to uV (scale_to_uV), which
    # requires both gains AND offsets to be set.
    rec.set_channel_offsets([0.0] * traces.shape[1])
    return rec
