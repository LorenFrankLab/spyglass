"""Convert MEArec ground-truth simulations into Spyglass-ingestible NWB files.

The output is structurally identical to a ``trodes_to_nwb``-produced NWB so that
Spyglass's ``insert_sessions`` ingests it end to end: one ``ndx_franklab_novela``
``Probe`` device, one ``NwbElectrodeGroup`` for the whole probe, an electrode
table carrying the novela columns ``Electrode.make()`` consumes, and an
``ElectricalSeries`` named ``"e-series"``. A ground-truth ``units`` table -- not
part of normal ``trodes_to_nwb`` output -- is added so the simulation's planted
spikes can be imported with ``ImportedSpikeSorting`` and used as a sort-accuracy
oracle.

The electrode/probe layout mirrors the Frank-lab reference probe metadata
(``trodes_to_nwb`` ``device_metadata/probe_metadata``). The recording traces are
read through ``neuroconv``'s ``MEArecRecordingInterface``; the ground-truth
spike trains and unit metadata are read through ``MEArec`` directly. The NWB is
assembled with ``pynwb`` rather than written by ``neuroconv`` because
``neuroconv``'s generic ecephys writer cannot emit the ``ndx_franklab_novela``
probe structure Spyglass ingestion requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Polymer probe constants, from the trodes_to_nwb reference probe metadata
# ``device_metadata/probe_metadata/128c-4s6mm6cm-15um-26um-sl.yml``: a
# 128-channel LLNL polyimide probe, 4 shanks x 32 contacts, 26 um within-shank
# vertical pitch, 15 um contacts. The shank x-offsets are the irregular spacing
# recorded in that metadata file.
POLYMER_PROBE_TYPE = "128c-4s6mm6cm-15um-26um-sl"
_POLYMER_DESCRIPTION = "128 channel polyimide probe"
_POLYMER_CONTACT_SIZE_UM = 15.0
_POLYMER_N_SHANKS = 4
_POLYMER_CONTACTS_PER_SHANK = 32
_POLYMER_PITCH_UM = 26.0
_POLYMER_SHANK_X_UM = (0.0, 350.0, 774.0, 1124.0)

# Neuropixels-like simulated probe: a single dense shank with a two-column
# staggered contact layout, used for dense-probe sorter coverage.
NEUROPIXELS_PROBE_TYPE = "neuropixels-128-sim"
_NP_DESCRIPTION = "128 channel simulated Neuropixels-style probe"
_NP_CONTACT_SIZE_UM = 12.0
_NP_N_CONTACTS = 128
_NP_COLUMN_X_UM = (0.0, 32.0)
_NP_ROW_PITCH_UM = 20.0

# nwbinspector importance levels that block fixture use.
_BLOCKING_INSPECTOR_LEVELS = frozenset({"ERROR", "PYNWB_VALIDATION", "CRITICAL"})


@dataclass(frozen=True)
class ProbeContact:
    """One probe contact.

    Attributes
    ----------
    electrode_id : int
        Global contact ID within the probe (``0 .. n_contacts - 1``).
    shank_id : int
        Shank index the contact sits on.
    rel_x, rel_y, rel_z : float
        Contact position relative to the probe, in micrometers.
    """

    electrode_id: int
    shank_id: int
    rel_x: float
    rel_y: float
    rel_z: float


@dataclass(frozen=True)
class ProbeLayout:
    """Geometry of a simulated probe shared by fixture generation and conversion.

    The fixture generator turns this into a ``probeinterface`` probe for MEArec;
    the converter turns the same object into the ``ndx_franklab_novela`` probe
    structure written to NWB. Keeping one definition guarantees the MEArec
    channel order matches the NWB electrode order.
    """

    probe_type: str
    description: str
    contact_size_um: float
    contact_side_numbering: bool
    contacts: tuple[ProbeContact, ...]

    @property
    def n_contacts(self) -> int:
        """Number of contacts on the probe."""
        return len(self.contacts)

    @property
    def n_shanks(self) -> int:
        """Number of distinct shanks on the probe."""
        return len({contact.shank_id for contact in self.contacts})

    def positions_um(self) -> np.ndarray:
        """Return contact ``(x, y)`` positions in micrometers, contact order.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_contacts, 2)``.
        """
        return np.array(
            [(c.rel_x, c.rel_y) for c in self.contacts], dtype=float
        )


def polymer_probe_layout() -> ProbeLayout:
    """Return the 128-channel Frank-lab polymer probe layout.

    Returns
    -------
    ProbeLayout
        4 shanks x 32 contacts, 26 um within-shank pitch.
    """
    contacts: list[ProbeContact] = []
    for shank_id in range(_POLYMER_N_SHANKS):
        for within in range(_POLYMER_CONTACTS_PER_SHANK):
            contacts.append(
                ProbeContact(
                    electrode_id=shank_id * _POLYMER_CONTACTS_PER_SHANK
                    + within,
                    shank_id=shank_id,
                    rel_x=_POLYMER_SHANK_X_UM[shank_id],
                    rel_y=-_POLYMER_PITCH_UM * within,
                    rel_z=0.0,
                )
            )
    return ProbeLayout(
        probe_type=POLYMER_PROBE_TYPE,
        description=_POLYMER_DESCRIPTION,
        contact_size_um=_POLYMER_CONTACT_SIZE_UM,
        contact_side_numbering=True,
        contacts=tuple(contacts),
    )


def neuropixels_probe_layout() -> ProbeLayout:
    """Return the 128-channel simulated Neuropixels-style probe layout.

    Returns
    -------
    ProbeLayout
        A single shank with a two-column staggered contact grid.
    """
    contacts: list[ProbeContact] = []
    for electrode_id in range(_NP_N_CONTACTS):
        column = electrode_id % 2
        row = electrode_id // 2
        contacts.append(
            ProbeContact(
                electrode_id=electrode_id,
                shank_id=0,
                rel_x=_NP_COLUMN_X_UM[column],
                rel_y=-_NP_ROW_PITCH_UM * row,
                rel_z=0.0,
            )
        )
    return ProbeLayout(
        probe_type=NEUROPIXELS_PROBE_TYPE,
        description=_NP_DESCRIPTION,
        contact_size_um=_NP_CONTACT_SIZE_UM,
        contact_side_numbering=True,
        contacts=tuple(contacts),
    )


@dataclass
class _GroundTruth:
    """Ground-truth units extracted from a MEArec recording generator."""

    spike_times: list[np.ndarray]
    positions_um: np.ndarray  # (n_units, 3)
    cell_types: list[str]

    @property
    def n_units(self) -> int:
        """Number of ground-truth units."""
        return len(self.spike_times)


def _read_recording_traces(mearec_h5_path: Path) -> tuple[np.ndarray, float]:
    """Read the recording traces and sampling rate via neuroconv.

    Parameters
    ----------
    mearec_h5_path : pathlib.Path
        MEArec ``.h5`` recording file.

    Returns
    -------
    traces : numpy.ndarray
        Array of shape ``(n_samples, n_channels)`` in microvolts.
    sampling_frequency : float
        Sampling rate in Hz.
    """
    from neuroconv.datainterfaces import MEArecRecordingInterface

    interface = MEArecRecordingInterface(file_path=str(mearec_h5_path))
    recording = interface.recording_extractor
    sampling_frequency = float(recording.get_sampling_frequency())
    try:
        traces = recording.get_traces(return_in_uV=True)
    except (TypeError, ValueError):
        # Extractor without a uV gain: fall back to native units.
        traces = recording.get_traces()
    if traces.dtype != np.float32:
        traces = traces.astype(np.float32, copy=False)
    return traces, sampling_frequency


def _read_ground_truth(mearec_h5_path: Path) -> _GroundTruth:
    """Read planted spike trains and unit metadata via MEArec.

    Parameters
    ----------
    mearec_h5_path : pathlib.Path
        MEArec ``.h5`` recording file.

    Returns
    -------
    _GroundTruth
    """
    import MEArec

    recgen = MEArec.load_recordings(
        str(mearec_h5_path),
        return_h5_objects=False,
        load=["spiketrains"],
        verbose=False,
    )

    spike_times: list[np.ndarray] = []
    cell_types: list[str] = []
    for spiketrain in recgen.spiketrains:
        times_s = np.asarray(spiketrain.rescale("s").magnitude, dtype=float)
        spike_times.append(times_s)
        annotations = getattr(spiketrain, "annotations", {}) or {}
        cell_types.append(str(annotations.get("cell_type", "unknown")))

    n_units = len(spike_times)
    locations = getattr(recgen, "template_locations", None)
    positions = np.full((n_units, 3), np.nan, dtype=float)
    if locations is not None:
        locations = np.asarray(locations, dtype=float)
        # Drifting recordings carry a position per drift step; use the first.
        if locations.ndim == 3:
            locations = locations[:, 0, :]
        if locations.ndim == 2 and locations.shape[0] == n_units:
            positions[:, : locations.shape[1]] = locations[
                :, : positions.shape[1]
            ]

    return _GroundTruth(
        spike_times=spike_times,
        positions_um=positions,
        cell_types=cell_types,
    )


def _build_nwbfile(fixture_name: str, session_start: datetime):
    """Build an empty NWBFile with synthetic Frank-lab-style session metadata.

    Parameters
    ----------
    fixture_name : str
        Fixture name, used to derive the session ID.
    session_start : datetime.datetime
        Timezone-aware session start time.

    Returns
    -------
    pynwb.NWBFile
    """
    from pynwb import NWBFile
    from pynwb.file import Subject

    nwbfile = NWBFile(
        session_description=(
            "MEArec-simulated ground-truth recording for spike-sorting "
            "validation"
        ),
        identifier=f"mearec-{fixture_name}",
        session_start_time=session_start,
        experimenter=["Synthetic, MEArec"],
        lab="Loren Frank Lab",
        institution="UCSF",
        experiment_description=(
            "MEArec-simulated ground-truth recording for v2 validation"
        ),
        session_id=f"mearec_{fixture_name}",
        keywords=["spike sorting", "simulation", "ground truth"],
        source_script="spyglass.spikesorting.v2._fixtures.mearec_to_nwb",
        source_script_file_name="mearec_to_nwb.py",
    )
    nwbfile.subject = Subject(
        subject_id="synthetic_001",
        description="Synthetic subject for MEArec ground-truth validation",
        genotype="wt/wt",
        sex="U",
        species="Mus musculus",
        date_of_birth=datetime(2023, 1, 1, tzinfo=timezone.utc),
        weight="0.025 kg",
    )
    return nwbfile


def _add_probe_and_electrodes(
    nwbfile,
    layout: ProbeLayout,
    targeted_location: str,
) -> None:
    """Add the novela probe, electrode group, and electrode table.

    Mirrors ``trodes_to_nwb.convert_yaml.add_electrode_groups``: one
    ``NwbElectrodeGroup`` for the whole probe, an ``ndx_franklab_novela.Probe``
    with one ``Shank`` per shank and a ``ShanksElectrode`` per contact, and the
    novela electrode columns ``Electrode.make()`` consumes.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        File under construction.
    layout : ProbeLayout
        Probe geometry.
    targeted_location : str
        Brain region recorded as the electrode group ``location``; Spyglass
        ingestion turns this into the default ``BrainRegion`` for every
        electrode.
    """
    from ndx_franklab_novela import (
        NwbElectrodeGroup,
        Probe,
        Shank,
        ShanksElectrode,
    )

    probe = Probe(
        id=0,
        name="probe 0",
        probe_type=layout.probe_type,
        units="um",
        probe_description=layout.description,
        contact_side_numbering=layout.contact_side_numbering,
        contact_size=layout.contact_size_um,
    )
    electrode_group = NwbElectrodeGroup(
        name="0",
        description=layout.description,
        location=targeted_location,
        targeted_location=targeted_location,
        targeted_x=0.0,
        targeted_y=0.0,
        targeted_z=0.0,
        units="um",
        device=probe,
    )
    nwbfile.add_electrode_group(electrode_group)

    contacts_by_shank: dict[int, list[ProbeContact]] = {}
    for contact in layout.contacts:
        contacts_by_shank.setdefault(contact.shank_id, []).append(contact)

    for shank_id in sorted(contacts_by_shank):
        shank = Shank(name=str(shank_id))
        for contact in contacts_by_shank[shank_id]:
            shank.add_shanks_electrode(
                ShanksElectrode(
                    name=str(contact.electrode_id),
                    rel_x=float(contact.rel_x),
                    rel_y=float(contact.rel_y),
                    rel_z=float(contact.rel_z),
                )
            )
        probe.add_shank(shank)
    nwbfile.add_device(probe)

    for contact in layout.contacts:
        nwbfile.add_electrode(
            location=targeted_location,
            group=electrode_group,
            rel_x=float(contact.rel_x),
            rel_y=float(contact.rel_y),
            rel_z=float(contact.rel_z),
            x=0.0,
            y=0.0,
            z=0.0,
            imp=0.0,
            filtering="none",
        )

    # Novela columns consumed by Electrode.make(): probe_shank, probe_electrode,
    # bad_channel, ref_elect_id. ref_elect_id is -1 (no reference electrode).
    nwbfile.electrodes.add_column(
        name="probe_shank",
        description="The shank of the probe this channel is located on",
        data=[int(c.shank_id) for c in layout.contacts],
    )
    nwbfile.electrodes.add_column(
        name="probe_electrode",
        description="The ID of this electrode with respect to the probe",
        data=[int(c.electrode_id) for c in layout.contacts],
    )
    nwbfile.electrodes.add_column(
        name="bad_channel",
        description="True if noisy or disconnected",
        data=[False] * layout.n_contacts,
    )
    nwbfile.electrodes.add_column(
        name="ref_elect_id",
        description="Experimenter selected reference electrode id",
        data=[-1] * layout.n_contacts,
    )


def _add_raw_ephys(nwbfile, traces: np.ndarray, sampling_frequency: float):
    """Add the ``"e-series"`` ElectricalSeries holding the recording traces.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        File under construction; electrodes must already be added.
    traces : numpy.ndarray
        Array of shape ``(n_samples, n_channels)`` in microvolts.
    sampling_frequency : float
        Sampling rate in Hz.
    """
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from pynwb.ecephys import ElectricalSeries

    n_samples, n_channels = traces.shape
    electrode_region = nwbfile.create_electrode_table_region(
        region=list(range(n_channels)),
        description="electrodes used in raw e-series recording",
    )
    wrapped = H5DataIO(
        data=traces,
        compression="gzip",
        chunks=(min(16384, n_samples), n_channels),
    )
    # MEArec traces are in microvolts; conversion maps the stored values to
    # volts, matching the trodes_to_nwb convention.
    eseries = ElectricalSeries(
        name="e-series",
        data=wrapped,
        electrodes=electrode_region,
        starting_time=0.0,
        rate=float(sampling_frequency),
        conversion=1e-6,
    )
    nwbfile.add_acquisition(eseries)


def _add_ground_truth_units(nwbfile, ground_truth: _GroundTruth) -> None:
    """Add the planted ground-truth ``units`` table.

    This table is not produced by ``trodes_to_nwb`` (which writes raw data
    only); it carries the simulation's known spikes so they can be imported and
    used as a sort-accuracy oracle.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        File under construction.
    ground_truth : _GroundTruth
        Planted spike trains and per-unit metadata.
    """
    nwbfile.add_unit_column(
        name="position_x", description="Ground-truth soma x position (um)"
    )
    nwbfile.add_unit_column(
        name="position_y", description="Ground-truth soma y position (um)"
    )
    nwbfile.add_unit_column(
        name="position_z", description="Ground-truth soma z position (um)"
    )
    nwbfile.add_unit_column(
        name="cell_type", description="Ground-truth simulated cell type"
    )
    nwbfile.add_unit_column(
        name="is_ground_truth",
        description="True for planted ground-truth units",
    )
    for unit_id in range(ground_truth.n_units):
        position = ground_truth.positions_um[unit_id]
        nwbfile.add_unit(
            id=unit_id,
            spike_times=ground_truth.spike_times[unit_id],
            position_x=float(position[0]),
            position_y=float(position[1]),
            position_z=float(position[2]),
            cell_type=ground_truth.cell_types[unit_id],
            is_ground_truth=True,
        )


def _validate_with_nwbinspector(out_nwb_path: Path) -> None:
    """Run NWBInspector and raise if any blocking findings are reported.

    Parameters
    ----------
    out_nwb_path : pathlib.Path
        The written NWB file.

    Raises
    ------
    RuntimeError
        If NWBInspector reports any ERROR, PYNWB_VALIDATION, or CRITICAL
        finding.
    """
    from nwbinspector import inspect_nwbfile

    blocking = [
        message
        for message in inspect_nwbfile(nwbfile_path=str(out_nwb_path))
        if getattr(message.importance, "name", "") in _BLOCKING_INSPECTOR_LEVELS
    ]
    if blocking:
        details = "\n".join(
            f"  [{m.importance.name}] {m.check_function_name}: {m.message}"
            for m in blocking
        )
        raise RuntimeError(
            f"NWBInspector reported {len(blocking)} blocking finding(s) for "
            f"{out_nwb_path}:\n{details}\n"
            "The converter produced an NWB that would fail Spyglass ingestion."
        )


def mearec_to_spyglass_nwb(
    mearec_h5_path: Path | str,
    out_nwb_path: Path | str,
    *,
    fixture_name: str,
    probe_layout: ProbeLayout,
    targeted_location: str = "Unknown",
) -> None:
    """Convert a MEArec recording into a Spyglass-ingestible NWB file.

    Parameters
    ----------
    mearec_h5_path : pathlib.Path or str
        MEArec ``.h5`` recording file to convert.
    out_nwb_path : pathlib.Path or str
        Destination NWB path. Overwritten if it exists.
    fixture_name : str
        Fixture name, recorded as the NWB ``session_id`` and used to derive
        the file ``identifier``.
    probe_layout : ProbeLayout
        Probe geometry the recording was simulated against. The MEArec
        channel count must match ``layout.n_contacts``.
    targeted_location : str, optional
        Brain region recorded as the single ``NwbElectrodeGroup.location``
        and used at ingestion as every electrode's default region. The
        ``trodes_to_nwb``-compatible NWB cannot encode per-shank regions;
        multi-region tracing is set up after ingestion by overriding
        ``Electrode.region_id`` per ``probe_shank``.

    Raises
    ------
    ValueError
        If the MEArec channel count does not match the layout.
    RuntimeError
        If NWBInspector reports a blocking finding on the written file.
    """
    mearec_h5_path = Path(mearec_h5_path)
    out_nwb_path = Path(out_nwb_path)

    traces, sampling_frequency = _read_recording_traces(mearec_h5_path)
    if traces.shape[1] != probe_layout.n_contacts:
        raise ValueError(
            f"MEArec recording {mearec_h5_path} has {traces.shape[1]} "
            f"channels, but the {probe_layout.probe_type} layout has "
            f"{probe_layout.n_contacts} contacts. Generate the recording "
            "with the matching probe geometry."
        )
    ground_truth = _read_ground_truth(mearec_h5_path)

    nwbfile = _build_nwbfile(
        fixture_name=fixture_name,
        session_start=datetime(2023, 6, 22, 12, 0, 0, tzinfo=timezone.utc),
    )
    _add_probe_and_electrodes(nwbfile, probe_layout, targeted_location)
    _add_raw_ephys(nwbfile, traces, sampling_frequency)
    _add_ground_truth_units(nwbfile, ground_truth)

    out_nwb_path.parent.mkdir(parents=True, exist_ok=True)
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(out_nwb_path), mode="w") as io:
        io.write(nwbfile)

    _validate_with_nwbinspector(out_nwb_path)
