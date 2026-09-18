"""
Shared fixtures for spikesorting tests.

This file contains common fixtures used across the spikesorting test modules,
reducing duplication and maintenance burden.
"""

from typing import NamedTuple

import pytest


@pytest.fixture(scope="session")
def burst_params_key_shared():
    """Shared burst parameters key for both v0 and v1.

    This fixture is identical in v0 and v1, so we consolidate it here.
    Both v0 and v1 conftest files can import this to avoid duplication.
    """
    yield dict(burst_params_name="default")


MS_BEFORE = 1.0
MS_AFTER = 2.0
NUM_CHANNELS = 4
NUM_UNITS = 3
DURATION_S = 5.0
MAX_SPIKES_PER_UNIT = 20
SEED = 0


class WrittenWaveforms(NamedTuple):
    """A waveform folder on disk plus the shape its waveforms must have."""

    path: str
    unit_ids: list
    n_spikes_per_unit: int
    n_samples: int
    n_channels: int


def write_waveform_folder(folder) -> WrittenWaveforms:
    """Write a waveform folder with the installed SpikeInterface generation.

    SpikeInterface 0.99 writes a ``WaveformExtractor`` folder directly. From
    0.101 the equivalent on-disk product is a binary-folder ``SortingAnalyzer``
    carrying the ``random_spikes`` and ``waveforms`` extensions, which
    ``load_waveforms`` reads back as a ``MockWaveformExtractor``.

    Parameters
    ----------
    folder : str | pathlib.Path
        Destination folder. Must not already exist.

    Returns
    -------
    WrittenWaveforms
        The folder path and the waveform shape implied by the parameters used
        to write it, so callers assert against independently known values.
    """
    import spikeinterface as si

    recording, sorting = si.generate_ground_truth_recording(
        durations=[DURATION_S],
        num_channels=NUM_CHANNELS,
        num_units=NUM_UNITS,
        seed=SEED,
    )
    folder = str(folder)

    if hasattr(si, "WaveformExtractor"):  # SpikeInterface 0.99
        si.extract_waveforms(
            recording,
            sorting,
            folder=folder,
            ms_before=MS_BEFORE,
            ms_after=MS_AFTER,
            max_spikes_per_unit=MAX_SPIKES_PER_UNIT,
            sparse=False,
        )
    else:
        analyzer = si.create_sorting_analyzer(
            sorting=sorting,
            recording=recording,
            format="binary_folder",
            folder=folder,
            sparse=False,
        )
        analyzer.compute(
            "random_spikes",
            max_spikes_per_unit=MAX_SPIKES_PER_UNIT,
            seed=SEED,
        )
        analyzer.compute("waveforms", ms_before=MS_BEFORE, ms_after=MS_AFTER)

    sampling_frequency = recording.get_sampling_frequency()
    n_samples = int(MS_BEFORE * sampling_frequency / 1000.0) + int(
        MS_AFTER * sampling_frequency / 1000.0
    )
    n_spikes_per_unit = min(
        [MAX_SPIKES_PER_UNIT]
        + [
            len(sorting.get_unit_spike_train(unit_id))
            for unit_id in sorting.unit_ids
        ]
    )
    return WrittenWaveforms(
        path=folder,
        unit_ids=list(sorting.unit_ids),
        n_spikes_per_unit=n_spikes_per_unit,
        n_samples=n_samples,
        n_channels=NUM_CHANNELS,
    )


@pytest.fixture(scope="session")
def written_waveforms(tmp_path_factory) -> WrittenWaveforms:
    """A real waveform folder, written once for the whole test session."""
    folder = tmp_path_factory.mktemp("waveform_reads") / "waveforms"
    return write_waveform_folder(folder)
