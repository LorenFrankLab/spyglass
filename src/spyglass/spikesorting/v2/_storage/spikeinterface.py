"""SpikeInterface 0.104 storage adapters used by the v2 pipeline.

These functions own the private extractor and extension fields needed for
lazy NWB reloads and disk-backed analyzers. The dependency pin and real NWB /
analyzer round-trip tests must move together when upgrading SpikeInterface.
"""

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


def attach_memmapped_waveforms(analyzer) -> None:
    """Attach a complete saved waveform extension without loading it into RAM."""
    from spikeinterface.core.sortinganalyzer import get_extension_class

    if "waveforms" not in analyzer.get_saved_extension_names():
        return
    extension = get_extension_class("waveforms")(analyzer)
    extension.load_params()
    extension.load_run_info()
    run_info = extension.run_info
    data_file = extension._get_binary_extension_folder() / "waveforms.npy"
    if (
        run_info is not None and not run_info.get("run_completed", False)
    ) or not data_file.is_file():
        return
    extension.data["waveforms"] = np.load(data_file, mmap_mode="r")
    analyzer.extensions["waveforms"] = extension
