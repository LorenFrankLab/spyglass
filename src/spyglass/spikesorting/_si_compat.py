"""Compatibility helpers for SpikeInterface APIs renamed after 0.99.

Existing v0/v1 rows hold objects written under SI 0.99, so read paths must
work with either API generation. Attribute presence, not version parsing,
selects the implementation: ``_PROVIDERS`` lists where each generation
provides a capability, and the first one present is used.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import warnings

#: Releases these shims are tested on (0.99 in the legacy CI lane, 0.104 in
#: the others); other releases warn.
TESTED_SI_VERSIONS = ("0.99", "0.104")

#: Capability -> providers in preference order, each a ``"module:attribute"``
#: target and its keyword arguments. A moved API needs one more provider.
_PROVIDERS: dict[str, tuple[tuple[str, dict], ...]] = {
    # ``metrics``, not ``metrics.quality``, which lacks ``compute_num_spikes``.
    "quality_metrics_module": (
        ("spikeinterface.metrics", {}),
        ("spikeinterface.qualitymetrics", {}),
    ),
    "load_extractor": (
        ("spikeinterface:load_extractor", {}),
        ("spikeinterface:load", {}),
    ),
    # SI 0.99 also has ``load_waveforms``, so its own loader comes first.
    "load_waveforms": (
        ("spikeinterface:WaveformExtractor.load_from_folder", {}),
        ("spikeinterface:load_waveforms", {"with_recording": False}),
    ),
    "numpy_sorting_from_samples_and_labels": (
        ("spikeinterface.core:NumpySorting.from_samples_and_labels", {}),
        ("spikeinterface.core:NumpySorting.from_times_labels", {}),
    ),
}


def installed_si_release() -> str:
    """The installed SpikeInterface's ``major.minor``, e.g. ``"0.104"``."""
    import spikeinterface

    return ".".join(spikeinterface.__version__.split(".")[:2])


@functools.cache
def _warn_if_untested() -> None:
    """Warn once per process if the installed SI is not a tested release."""
    import spikeinterface

    if installed_si_release() not in TESTED_SI_VERSIONS:
        warnings.warn(
            f"SpikeInterface {spikeinterface.__version__} is installed, but "
            "the v0/v1 read-path shims (spyglass.spikesorting._si_compat) are "
            f"tested only on SpikeInterface {', '.join(TESTED_SI_VERSIONS)}. "
            "Reading saved v0/v1 recordings, sortings and waveforms may fail.",
            RuntimeWarning,
            stacklevel=4,
        )


def _lookup(target: str):
    """Return the object at ``"module:attribute.path"``, or ``None``."""
    module_name, _, attribute_path = target.partition(":")
    if importlib.util.find_spec(module_name) is None:
        return None
    found = importlib.import_module(module_name)
    for attribute in filter(None, attribute_path.split(".")):
        found = getattr(found, attribute, None)
        if found is None:
            return None
    return found


def _provider(capability: str):
    """Return the first provider of ``capability`` the installed SI has.

    Raises
    ------
    RuntimeError
        If SpikeInterface has none of the capability's providers.
    """
    _warn_if_untested()
    for target, kwargs in _PROVIDERS[capability]:
        found = _lookup(target)
        if found is not None:
            return functools.partial(found, **kwargs) if kwargs else found
    import spikeinterface

    targets = ", ".join(target for target, _ in _PROVIDERS[capability])
    raise RuntimeError(
        f"SpikeInterface {spikeinterface.__version__} has none of {targets} "
        f"for {capability!r}; add its API to "
        "spyglass.spikesorting._si_compat._PROVIDERS."
    )


def quality_metrics_module():
    """Return SpikeInterface's quality-metrics namespace for either generation.

    Returns
    -------
    module
        ``spikeinterface.metrics``, or ``spikeinterface.qualitymetrics`` under
        SpikeInterface 0.99.
    """
    return _provider("quality_metrics_module")


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
    return _provider("load_extractor")(source)


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
    from spyglass.spikesorting._legacy_runtime import _legacy_runtime_message

    try:
        return _provider("load_waveforms")(folder)
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

    Raises
    ------
    RuntimeError
        If the sorting's spike frames do not span ``samples``: the
        constructor no longer reads its first argument as sample indices.
    """
    import numpy as np

    sorting = _provider("numpy_sorting_from_samples_and_labels")(
        samples,
        labels,
        sampling_frequency,
        unit_ids=unit_ids,
    )
    # A renamed constructor is caught by the lookup; this catches a change
    # in what its first argument means.
    given = np.concatenate(
        [
            np.ravel(segment)
            for segment in (
                [samples] if isinstance(samples, np.ndarray) else samples
            )
        ]
    )
    frames = sorting.to_spike_vector()["sample_index"]
    if given.size and (
        frames.size == 0
        or (frames.min(), frames.max()) != (given.min(), given.max())
    ):
        raise RuntimeError(
            "NumpySorting did not keep sample indices "
            f"{given.min()}..{given.max()} as spike frames: this "
            "SpikeInterface reads the constructor's first argument "
            "differently. Check spyglass.spikesorting._si_compat._PROVIDERS."
        )
    return sorting
