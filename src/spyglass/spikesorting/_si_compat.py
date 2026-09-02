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
