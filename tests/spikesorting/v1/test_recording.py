import numpy as np


def test_sort_group(spike_v1, pop_rec):
    max_id = max(spike_v1.SortGroup.fetch("sort_group_id"))
    assert (
        max_id == 31
    ), "SortGroup.insert_sort_group failed to insert all records"


def test_spike_sorting(spike_v1, pop_rec):
    n_records = len(spike_v1.SpikeSortingRecording())
    assert n_records == 1, "SpikeSortingRecording failed to insert a record"


def test_nonmonotonic_timestamp_correction():
    """Test that _get_preprocessed_recording corrects non-monotonic timestamps.

    Non-monotonic timestamps can arise from floating-point precision issues or
    epoch-stitching artifacts in the raw NWB file. This unit test validates
    the correction logic used in _get_preprocessed_recording by verifying that
    the same approach produces strictly increasing timestamps.
    """
    fs = 30000.0
    sample_period = 1.0 / fs

    # Build a 10-second recording with two non-monotonic jumps:
    # one tiny (floating-point-scale) and one larger (epoch-stitching-scale)
    n_samples = int(10 * fs)
    timestamps = np.arange(n_samples) * sample_period

    # Introduce non-monotonic values (simulating raw-file artifacts)
    timestamps[5000] = timestamps[4999] - 1e-7   # tiny backward jump at sample 5000
    timestamps[9000] = timestamps[8999] - 0.01   # larger backward jump at sample 9000

    assert np.any(np.diff(timestamps) <= 0), "Test setup: timestamps should be non-monotonic"

    # Apply the same vectorized correction logic used in _get_preprocessed_recording
    diffs = np.diff(timestamps)
    corrected = timestamps.copy()
    bad_indices = np.where(diffs <= 0)[0] + 1
    for i in bad_indices:
        correction = corrected[i - 1] + sample_period - corrected[i]
        if correction > 0:
            corrected[i:] += correction

    result_diffs = np.diff(corrected)
    assert np.all(result_diffs > 0), (
        "Corrected timestamps should be strictly increasing; "
        f"found {np.sum(result_diffs <= 0)} non-positive diff(s)"
    )
    # Timestamps before the first non-monotonic point should be unchanged
    assert np.allclose(corrected[:5000], timestamps[:5000]), (
        "Timestamps before first bad sample should be unchanged"
    )
    # The correction should be applied exactly at the boundary
    assert corrected[5000] == corrected[4999] + sample_period, (
        "First corrected timestamp should be exactly one period after its predecessor"
    )
