"""Compatibility helpers for SpikeInterface APIs renamed after 0.99.

The package pins SpikeInterface 0.104, but existing v0/v1 rows can contain
recordings and sortings written under 0.99. Read paths must therefore work
with either API generation. Attribute presence, rather than version parsing,
selects the available implementation.
"""

from __future__ import annotations


def load_extractor(source):
    """Load a saved SpikeInterface recording or sorting.

    Parameters
    ----------
    source : str | pathlib.Path | dict
        A saved extractor folder, metadata file, or serialized extractor
        dictionary.

    Returns
    -------
    spikeinterface.BaseRecording | spikeinterface.BaseSorting
        The deserialized extractor.
    """
    import spikeinterface as si

    loader = getattr(si, "load_extractor", None) or si.load
    return loader(source)


def load_waveforms(folder):
    """Load a saved waveform folder under either SpikeInterface generation.

    Parameters
    ----------
    folder : str | pathlib.Path
        A saved ``WaveformExtractor`` folder, or a binary-folder
        ``SortingAnalyzer`` written by the back-compatibility API.

    Returns
    -------
    WaveformExtractor | MockWaveformExtractor
        SpikeInterface 0.99 returns a ``WaveformExtractor``; 0.101 and later
        return a ``MockWaveformExtractor`` exposing the same ``get_waveforms``
        / ``nbefore`` / ``nafter`` / ``sorting`` surface.

    Raises
    ------
    RuntimeError
        When the folder holds Zarr-format legacy waveforms, which 0.101 and
        later cannot read.
    """
    import spikeinterface as si

    from spyglass.spikesorting._legacy_runtime import _legacy_runtime_message

    legacy = getattr(si, "WaveformExtractor", None)
    if legacy is not None:
        return legacy.load_from_folder(folder)
    try:
        return si.load_waveforms(folder, with_recording=False)
    except NotImplementedError as exc:  # Zarr-format legacy waveforms
        raise RuntimeError(
            _legacy_runtime_message("Zarr-format WaveformExtractor folders")
        ) from exc


def numpy_sorting_from_samples_and_labels(
    samples, labels, sampling_frequency, unit_ids=None
):
    """Build a ``NumpySorting`` from sample indices and unit labels.

    SpikeInterface 0.99 calls this constructor ``from_times_labels`` while
    0.101 and later call it ``from_samples_and_labels``. Both versions share
    the same positional argument semantics.
    """
    from spikeinterface.core import NumpySorting

    constructor = getattr(
        NumpySorting, "from_samples_and_labels", None
    ) or getattr(NumpySorting, "from_times_labels")
    return constructor(
        samples,
        labels,
        sampling_frequency,
        unit_ids=unit_ids,
    )
