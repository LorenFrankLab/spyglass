"""Shared test helpers for the single-session pipeline suite."""


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
    """Drop a sorting's CurationV2 rows and merge masters."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    clear_curations_for(sorting_key)


def _build_synthetic_rec(traces, fs=30_000.0):
    """Wrap a (n_samples, n_channels) array as an SI NumpyRecording with unit
    gains, so trace values read as microvolts."""
    import spikeinterface as si

    rec = si.NumpyRecording(traces_list=[traces], sampling_frequency=fs)
    rec.set_channel_gains([1.0] * traces.shape[1])
    # threshold_unit="uv" scales to uV, which needs both gains and offsets.
    rec.set_channel_offsets([0.0] * traces.shape[1])
    return rec
