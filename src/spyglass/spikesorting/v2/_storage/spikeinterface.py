"""SpikeInterface 0.104 storage adapters used by the v2 pipeline.

These functions own the private extractor and extension fields needed for
lazy NWB reloads and disk-backed analyzers. The dependency pin and real NWB /
analyzer round-trip tests must move together when upgrading SpikeInterface.
"""

from pathlib import Path

import numpy as np


def use_pickle_recording_serialization(recording) -> None:
    """Keep custom recording wrappers out of unsupported JSON reconstruction."""
    recording._serializability["json"] = False


def retain_pynwb_reader(recording) -> None:
    """Preserve the metadata-complete reader when an NWB extractor reloads."""
    recording._kwargs["use_pynwb"] = True


def close_nwb_recording(recording) -> None:
    """Close an owned NWB extractor's reader without waiting for collection.

    SI 0.104.3 exposes reader cleanup only through its extractor destructor.
    Apply this to newly opened extractors, not shared recordings or wrappers.
    """
    nwbfile = getattr(recording, "_nwbfile", None)
    if nwbfile is not None:
        reader = nwbfile.get_read_io()
        if reader is not None:
            reader.close()
        return
    file = getattr(recording, "_file", None)
    if file is not None:
        store = getattr(file, "store", file)
        close = getattr(store, "close", None)
        if close is not None:
            close()


def saved_extension_metadata(analyzer, name: str):
    """Load a saved extension's params and run status without its data.

    SI 0.104.3 writes ``run_info.json`` only when a compute finishes, so an
    extension without it is interrupted or still running.

    Parameters
    ----------
    analyzer : spikeinterface.SortingAnalyzer
        A ``binary_folder`` analyzer that lists ``name`` as saved.
    name : str
        The extension name.

    Returns
    -------
    extension : spikeinterface.core.sortinganalyzer.AnalyzerExtension
        A detached instance with ``params`` and ``run_info`` loaded and empty
        ``data``; it is not registered on ``analyzer``.
    completed : bool
        Whether the saved run finished.
    """
    from spikeinterface.core.sortinganalyzer import get_extension_class

    extension = get_extension_class(name)(analyzer)
    extension.load_params()
    extension.load_run_info()
    completed = bool((extension.run_info or {}).get("run_completed", False))
    return extension, completed


def saved_extension_data_file(analyzer, name: str, data_name: str) -> Path:
    """Return the ``.npy`` file SI's binary format saves ``data_name`` in."""
    from spikeinterface.core.sortinganalyzer import get_extension_class

    extension = get_extension_class(name)(analyzer)
    return extension._get_binary_extension_folder() / f"{data_name}.npy"


def attach_memmapped_waveforms(analyzer) -> None:
    """Attach a complete saved waveform extension without loading it into RAM."""
    if "waveforms" not in analyzer.get_saved_extension_names():
        return
    extension, completed = saved_extension_metadata(analyzer, "waveforms")
    data_file = saved_extension_data_file(analyzer, "waveforms", "waveforms")
    if not completed or not data_file.is_file():
        return
    extension.data["waveforms"] = np.load(data_file, mmap_mode="r")
    analyzer.extensions["waveforms"] = extension
