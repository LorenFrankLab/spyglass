"""Builders for synthetic chronic-EEG NWB fixtures (core pynwb, no ndx extension).

Mirrors the structure of the Gonzalez-Sulser chronic-EEG recordings in
``DANDI:001888``: a raw multi-channel ``ElectricalSeries`` in
``nwbfile.acquisition`` whose channels are probe-less array/screw electrodes,
grouped into an EEG and an EMG ``ElectrodeGroup`` on a plain ``Device`` (no
``Probe``). The series carries an ``.electrodes`` region over its channels. This
is all core ``pynwb.ecephys`` -- no extension is required to exercise the
``common_eeg`` ingestion path.

``build_eeg`` is the canonical happy-path fixture; the other builders vary one
axis (explicit timestamps, non-consecutive electrode ids, region/column-count
mismatch) to exercise the ingestion edge paths.
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
N_TIME = 500  # samples in the synthetic trace
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


def _add_channels(nwb: NWBFile, electrode_ids=None) -> None:
    """Add the device, EEG/EMG groups, and per-channel electrodes.

    ``electrode_ids``: optional length-``N_CHANNELS`` list of explicit row ids
    (to exercise non-consecutive ids); default lets pynwb assign ``0..N-1``.
    """
    device = Device(
        name="TainiTecWirelessEEG",
        description="14-ch wireless EEG/EMG telemetry system",
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
    for col in ("channel_name", "hemisphere"):
        nwb.add_electrode_column(name=col, description=col)
    for col in ("rel_x", "rel_y"):
        nwb.add_electrode_column(name=col, description=f"{col} (um)")

    channels = [(eeg_group, "cortex", n) for n in EEG_CHANNELS] + [
        (emg_group, "muscle", n) for n in EMG_CHANNELS
    ]
    for i, (group, location, name) in enumerate(channels):
        kwargs = dict(
            group=group,
            location=location,
            filtering="none",
            channel_name=name,
            hemisphere="R" if name.endswith("_R") else "L",
            rel_x=float(i) if location == "cortex" else np.nan,
            rel_y=float(i) if location == "cortex" else np.nan,
        )
        if electrode_ids is not None:
            kwargs["id"] = int(electrode_ids[i])
        nwb.add_electrode(**kwargs)


def _add_series(nwb, region, n_cols, *, use_timestamps=False, n_time=N_TIME):
    """Add one EEG ``ElectricalSeries`` to ``acquisition``.

    ``region``: positional indices for the ``.electrodes`` region. ``n_cols``:
    number of trace columns (equals ``len(region)`` for a well-formed file;
    differ to exercise the mismatch guard). Deterministic content (fixed seed).
    """
    data = (
        np.random.RandomState(0)
        .standard_normal((n_time, n_cols))
        .astype("float32")
    )
    reg = nwb.create_electrode_table_region(
        region=list(region), description="EEG/EMG channels"
    )
    kwargs = dict(name=EEG_SERIES_NAME, data=data, electrodes=reg)
    if use_timestamps:
        kwargs["timestamps"] = (np.arange(n_time) / EEG_RATE).astype("float64")
    else:
        kwargs["starting_time"] = 0.0
        kwargs["rate"] = EEG_RATE
    nwb.add_acquisition(ElectricalSeries(**kwargs))


def build_eeg(nwb: NWBFile, n_time: int = N_TIME) -> NWBFile:
    """Canonical fixture: rate-based series, region over all channels."""
    _add_channels(nwb)
    _add_series(nwb, range(N_CHANNELS), N_CHANNELS, n_time=n_time)
    return nwb


def build_eeg_timestamps(nwb: NWBFile, n_time: int = N_TIME) -> NWBFile:
    """Explicit-``timestamps`` series (no ``rate``)."""
    _add_channels(nwb)
    _add_series(
        nwb, range(N_CHANNELS), N_CHANNELS, use_timestamps=True, n_time=n_time
    )
    return nwb


def build_eeg_noncontiguous(nwb: NWBFile, n_time: int = N_TIME) -> NWBFile:
    """Non-consecutive electrode ids + a permuted, subset region.

    ids are ``10..14``; the region is ``[2, 0, 4]`` -- so a correct
    position->id translation is the only way to recover the right electrodes.
    """
    _add_channels(nwb, electrode_ids=[10, 11, 12, 13, 14])
    _add_series(nwb, [2, 0, 4], 3, n_time=n_time)
    return nwb


def build_eeg_col_mismatch(nwb: NWBFile, n_time: int = N_TIME) -> NWBFile:
    """Region length != trace column count (a malformed/hand-edited file).

    The region covers all channels but the trace has one extra column, so the
    region_index -> column correspondence is broken.
    """
    _add_channels(nwb)
    _add_series(nwb, range(N_CHANNELS), N_CHANNELS + 1, n_time=n_time)
    return nwb


def write(path, builder=build_eeg, identifier=None, **kwargs) -> Path:
    """Build a fresh NWBFile, apply ``builder``, and write it to ``path``."""
    path = Path(path)
    nwb = _new_nwb(identifier or path.stem)
    builder(nwb, **kwargs)
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return path
