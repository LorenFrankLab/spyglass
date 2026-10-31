"""Shared test helpers for the single-session pipeline suite.

These helpers are used by more than one of the ``test_*.py`` modules in
this subpackage (curation/dispatch tests clear curations; sorting and
artifact tests wrap synthetic traces), so they live here rather than in
any single module.
"""


def fixture_nwb_path(stem):
    """Resolve shared generated inputs from the v2 fixture package."""
    from pathlib import Path

    return Path(__file__).resolve().parents[1] / "fixtures" / f"{stem}.nwb"


def validate_v1_baseline_case(meta, *, sorter, sort_group_id):
    """A capture must belong to the matrix case its directory represents."""
    actual_sorter = meta.get("sorter")
    actual_group = meta.get("sort_group_id")
    if (
        actual_sorter != sorter
        or type(actual_group) is not int
        or actual_group != sort_group_id
    ):
        raise ValueError(
            f"v1 baseline must describe sorter={sorter!r}, sort_group_id={sort_group_id}; "
            f"got sorter={actual_sorter!r}, sort_group_id={actual_group!r}. "
            "Recapture or move the baseline into the matching case directory."
        )


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
