"""Builder for a synthetic chronic-EEG NWB fixture (core pynwb, no ndx extension).

Mirrors the structure of the Gonzalez-Sulser chronic-EEG recordings in
``DANDI:001888``: a raw multi-channel ``ElectricalSeries`` in
``nwbfile.acquisition`` whose channels are probe-less array/screw electrodes,
grouped into an EEG and an EMG ``ElectrodeGroup`` on a plain ``Device`` (no
``Probe``). The series carries an ``.electrodes`` region over all channels. This
is all core ``pynwb.ecephys`` -- no extension is required to exercise the
``common_eeg`` ingestion path.
"""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from pynwb import NWBHDF5IO, NWBFile
from pynwb.device import Device
from pynwb.ecephys import ElectricalSeries
from pynwb.file import Subject

EEG_SERIES_NAME = "ElectricalSeries_EEG"
EEG_RATE = 250.4  # Hz, matching the DANDI:001888 telemetry rate
# Channel names mirror the real montage; the "_R"/"_L" suffix drives hemisphere.
EEG_CHANNELS = ["S1_Tr_R", "M2_Fra_R", "V1_M_L", "S1Hl_L"]
EMG_CHANNELS = ["EMG_R"]
N_CHANNELS = len(EEG_CHANNELS) + len(EMG_CHANNELS)


def _new_nwb(identifier: str) -> NWBFile:
    return NWBFile(
        session_description="eeg fixture",
        identifier=identifier,
        session_start_time=datetime(
            2025, 10, 2, 11, 48, tzinfo=ZoneInfo("America/Los_Angeles")
        ),
        subject=Subject(subject_id="GRIN2B-129", species="Rattus norvegicus"),
        experimenter=["Test, Experimenter"],
        lab="Test Lab",
        institution="Test Institution",
    )


def build_eeg(nwb: NWBFile, n_time: int = 500) -> NWBFile:
    """Add a probe-less EEG/EMG acquisition ElectricalSeries to ``nwb``."""
    device = Device(
        name="TainiTecWirelessEEG",
        description="16-ch wireless EEG/EMG telemetry system",
    )
    nwb.add_device(device)

    eeg_group = nwb.create_electrode_group(
        name="EEGArray",
        description="chronic EEG electrode array",
        location="cortex",
        device=device,
    )
    emg_group = nwb.create_electrode_group(
        name="EMGArray",
        description="chronic EMG electrodes",
        location="muscle",
        device=device,
    )

    # Custom columns mirroring the real electrodes table (channel_name,
    # hemisphere, rel_x, rel_y). Standard location/group/group_name/filtering
    # columns come for free from add_electrode.
    for col in ("channel_name", "hemisphere"):
        nwb.add_electrode_column(name=col, description=col)
    for col in ("rel_x", "rel_y"):
        nwb.add_electrode_column(name=col, description=f"{col} (um)")

    for i, name in enumerate(EEG_CHANNELS):
        nwb.add_electrode(
            group=eeg_group,
            location="cortex",
            filtering="none",
            channel_name=name,
            hemisphere="R" if name.endswith("_R") else "L",
            rel_x=float(i),
            rel_y=float(i),
        )
    for name in EMG_CHANNELS:
        nwb.add_electrode(
            group=emg_group,
            location="muscle",
            filtering="none",
            channel_name=name,
            hemisphere="R" if name.endswith("_R") else "L",
            rel_x=np.nan,
            rel_y=np.nan,
        )

    region = nwb.create_electrode_table_region(
        region=list(range(N_CHANNELS)),
        description="all EEG + EMG channels",
    )
    # Deterministic synthetic trace (fixed seed -> reproducible object_id-independent
    # content). volts, continuous acquisition at EEG_RATE.
    data = (
        np.random.RandomState(0)
        .standard_normal((n_time, N_CHANNELS))
        .astype("float32")
    )
    nwb.add_acquisition(
        ElectricalSeries(
            name=EEG_SERIES_NAME,
            data=data,
            electrodes=region,
            starting_time=0.0,
            rate=EEG_RATE,
        )
    )
    return nwb


def write(path, builder=build_eeg, identifier=None, **kwargs) -> Path:
    """Build a fresh NWBFile, apply ``builder``, and write it to ``path``."""
    path = Path(path)
    nwb = _new_nwb(identifier or path.stem)
    builder(nwb, **kwargs)
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return path
