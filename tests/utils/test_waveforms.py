"""Database-free checks for the waveform amplitude helper."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(scope="session")
def mini_insert():
    """These tests need no sample-data ingestion."""
    yield


def test_peak_amplitude_uses_nbefore_for_asymmetric_window():
    import spikeinterface as si
    from spikeinterface.core import generate_ground_truth_recording

    from spyglass.utils.waveforms import _get_peak_amplitude

    recording, sorting = generate_ground_truth_recording(
        durations=[2.0], num_channels=4, num_units=2, seed=0
    )
    waveform_extractor = si.extract_waveforms(
        recording,
        sorting,
        folder=None,
        mode="memory",
        ms_before=0.5,
        ms_after=2.0,
        sparse=False,
        max_spikes_per_unit=20,
        seed=0,
        progress_bar=False,
    )
    unit_id = sorting.unit_ids[0]
    waveforms = waveform_extractor.get_waveforms(unit_id)
    nbefore = waveform_extractor.nbefore
    assert nbefore != waveforms.shape[1] // 2  # window is asymmetric

    amplitudes = _get_peak_amplitude(waveform_extractor, unit_id)

    assert amplitudes.shape == (waveforms.shape[0], waveforms.shape[2])
    np.testing.assert_array_equal(amplitudes, waveforms[:, nbefore])
