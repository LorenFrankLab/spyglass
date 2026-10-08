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
