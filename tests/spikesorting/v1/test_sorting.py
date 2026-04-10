import numpy as np
import pytest


@pytest.mark.slow
def test_sorting(spike_v1, pop_sort):
    """Integration test for SpikeSorting with real spikeinterface operations.

    This test validates the full spike sorting pipeline including:
    - Real spikeinterface spike sorting (~90s)
    - Real file I/O operations (~5s)
    - Full database integration
    """
    n_sorts = len(spike_v1.SpikeSorting & pop_sort)
    assert n_sorts >= 1, "SpikeSorting population failed"


# ============================================================================
# Excess Spike Removal Tests (invalid spike_times from float precision)
# ============================================================================


def test_sorting_get_sorting_removes_excess_spikes():
    """Test that SpikeSorting.get_sorting() removes spike samples >= n_samples.

    Verifies that floating-point rounding in the seconds-to-samples round-trip
    (which causes np.searchsorted to return n_samples) is handled correctly by
    the actual method. If the filtering logic in get_sorting() is removed this
    test will fail.
    """
    from unittest.mock import MagicMock, patch

    import pandas as pd

    from spyglass.spikesorting.v1.sorting import SpikeSorting

    sampling_frequency = 30000.0
    n_samples = 300  # 10 ms recording
    recording_times = np.arange(n_samples, dtype=float) / sampling_frequency

    # 10 valid spikes, plus one just beyond the last timestamp
    spike_times = np.append(
        recording_times[:10].copy(), recording_times[-1] + 1e-9
    )
    units_df = pd.DataFrame({"spike_times": [spike_times]})

    mock_recording = MagicMock()
    mock_recording.get_sampling_frequency.return_value = sampling_frequency
    mock_recording.get_times.return_value = recording_times
    mock_recording.get_num_samples.return_value = n_samples

    mock_nwbf = MagicMock()
    mock_nwbf.units.to_dataframe.return_value = units_df
    mock_io = MagicMock()
    mock_io.__enter__ = MagicMock(return_value=mock_io)
    mock_io.__exit__ = MagicMock(return_value=False)
    mock_io.read.return_value = mock_nwbf

    # Mock SpikeSortingRecording for the join query and recording retrieval
    mock_ssr = MagicMock()
    mock_ssr.__mul__.return_value.__and__.return_value.fetch1.return_value = (
        "fake_rec_id"
    )
    mock_ssr.get_recording.return_value = mock_recording

    with (
        patch(
            "spyglass.spikesorting.v1.sorting.SpikeSortingRecording", mock_ssr
        ),
        patch(
            "spyglass.spikesorting.v1.sorting.SpikeSorting"
        ) as mock_ss_tbl,
        patch(
            "spyglass.spikesorting.v1.sorting.AnalysisNwbfile.get_abs_path",
            return_value="/fake/path.nwb",
        ),
        patch(
            "spyglass.spikesorting.v1.sorting.pynwb.NWBHDF5IO",
            return_value=mock_io,
        ),
    ):
        mock_ss_tbl.__and__.return_value.fetch1.return_value = (
            "test_analysis.nwb"
        )
        sorting = SpikeSorting.get_sorting(key={})

    spike_train = sorting.get_unit_spike_train(unit_id=0, segment_index=0)
    assert np.all(
        spike_train < n_samples
    ), "Excess spikes should have been removed by SpikeSorting.get_sorting()"
    assert len(spike_train) == 10, (
        f"Expected 10 spikes after removing 1 excess, got {len(spike_train)}"
    )
