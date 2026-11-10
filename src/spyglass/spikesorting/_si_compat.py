"""Compatibility helpers for SpikeInterface APIs renamed after 0.99.

Existing v0/v1 rows hold objects written under SI 0.99, so read paths must
work with either API generation. Attribute presence, not version parsing,
selects the implementation.
"""

from __future__ import annotations


def quality_metrics_module():
    """Return SpikeInterface's quality-metrics namespace for either generation.

    Returns ``metrics`` rather than ``metrics.quality``, which lacks
    ``compute_num_spikes``.

    Returns
    -------
    module
        ``spikeinterface.metrics``, or ``spikeinterface.qualitymetrics`` under
        SpikeInterface 0.99.
    """
    try:
        import spikeinterface.metrics as sq
    except ModuleNotFoundError:  # SI 0.99 (legacy v0/v1 runtime env)
        import spikeinterface.qualitymetrics as sq
    return sq


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
        ``MockWaveformExtractor`` under SI 0.101+, with the same
        ``get_waveforms``, ``nbefore``, ``nafter``, and ``sorting`` API.

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
            _legacy_runtime_message(
                "Loading a Zarr-format WaveformExtractor folder"
            )
        ) from exc


def numpy_sorting_from_samples_and_labels(
    samples, labels, sampling_frequency, unit_ids=None
):
    """Build a ``NumpySorting`` from sample indices and unit labels.

    SI 0.99 names it ``from_times_labels``; 0.101+ uses
    ``from_samples_and_labels`` with the same positional arguments.
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
