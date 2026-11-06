"""Convert MEArec ground-truth simulations into Spyglass-ingestible NWB files.

The output matches ``trodes_to_nwb``'s structure so ``insert_sessions``
ingests it. Planted units go in a sidecar ``ProcessingModule("ground_truth")``
(see :func:`get_ground_truth_units_table`). The NWB is assembled with
``pynwb`` because ``neuroconv``'s writer cannot emit the
``ndx_franklab_novela`` probe structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

# From trodes_to_nwb ``probe_metadata/128c-4s6mm6cm-15um-26um-sl.yml``:
# 4 shanks x 32 contacts, 26 um pitch, irregular shank x-offsets.
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

# Frank-lab ``tetrode_12.5`` (trodes_to_nwb probe_metadata): 4 wires in a
# ±6.25 µm square; sparse-probe sorter coverage.
TETRODE_PROBE_TYPE = "tetrode_12.5"
_TETRODE_DESCRIPTION = "four wire electrode"
_TETRODE_CONTACT_SIZE_UM = 12.5
_TETRODE_HALF_PITCH_UM = 6.25

# nwbinspector importance levels that block fixture use.
_BLOCKING_INSPECTOR_LEVELS = frozenset(
    {"ERROR", "PYNWB_VALIDATION", "CRITICAL"}
)


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

    One definition keeps the MEArec channel order equal to the NWB
    electrode order.
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

    Contacts lie in the XY plane (``rel_z=0``), as MEAutility requires.
    Shanks are offset along ``rel_x``; contacts run down ``rel_y`` so
    contact 0 is the most superficial.

    Returns
    -------
    ProbeLayout
        4 shanks x 32 contacts, 26 um within-shank pitch, in the XY
        plane (``rel_z=0``).
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


def tetrode_probe_layout() -> ProbeLayout:
    """Return the Frank-lab ``tetrode_12.5`` probe layout.

    One shank, 4 contacts in a ±6.25 µm square (trodes_to_nwb
    ``probe_metadata/tetrode_12.5.yml``). The YAML's ``rel_y`` and ``rel_z``
    are swapped because MEAutility requires ``rel_z=0`` (a nonzero
    ``rel_z`` raises ``TypeError`` there).

    Returns
    -------
    ProbeLayout
        1 shank × 4 contacts, square geometry in the XY plane.
    """
    half = _TETRODE_HALF_PITCH_UM
    # YAML id order: 0 (+x, +y), 1 (-x, +y), 2 (-x, -y), 3 (+x, -y).
    positions = [
        (+half, +half, 0.0),
        (-half, +half, 0.0),
        (-half, -half, 0.0),
        (+half, -half, 0.0),
    ]
    contacts = tuple(
        ProbeContact(
            electrode_id=eid,
            shank_id=0,
            rel_x=rx,
            rel_y=ry,
            rel_z=rz,
        )
        for eid, (rx, ry, rz) in enumerate(positions)
    )
    return ProbeLayout(
        probe_type=TETRODE_PROBE_TYPE,
        description=_TETRODE_DESCRIPTION,
        contact_size_um=_TETRODE_CONTACT_SIZE_UM,
        contact_side_numbering=True,
        contacts=contacts,
    )


# MEArec ``cell_type`` codes (excitatory / inhibitory).
CellType = Literal["E", "I"]

_VALID_CELL_TYPES: frozenset[str] = frozenset({"E", "I"})


def _normalize_cell_type(raw: object) -> CellType:
    """Normalize a MEArec ``cell_type`` annotation to the canonical set.

    Strips whitespace and upper-cases. Raises ``ValueError`` outside
    ``{"E", "I"}``.
    """
    normalized = str(raw).strip().upper()
    if normalized not in _VALID_CELL_TYPES:
        raise ValueError(
            f"Unrecognized MEArec cell_type {raw!r} (normalized to "
            f"{normalized!r}); expected one of {sorted(_VALID_CELL_TYPES)}."
        )
    return normalized  # type: ignore[return-value]


@dataclass
class _GroundTruth:
    """Ground-truth units extracted from a MEArec recording generator.

    Attributes
    ----------
    spike_times : list of numpy.ndarray
        Per-unit planted spike times in seconds.
    positions_um : numpy.ndarray
        Per-unit soma positions of shape ``(n_units, 3)`` in micrometers.
    cell_types : list of CellType
        Per-unit normalized cell-type code, one of ``{"E", "I"}``.
    """

    spike_times: list[np.ndarray]
    positions_um: np.ndarray  # (n_units, 3)
    cell_types: list[CellType]

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
    # A gainless float extractor does not raise on return_in_uV=True; it
    # returns unscaled values, so check explicitly.
    if not recording.has_scaleable_traces():
        raise RuntimeError(
            f"MEArec extractor for {mearec_h5_path.name!r} has no "
            "microvolt conversion (`gain_to_uV` / `offset_to_uV` unset), "
            "so its traces cannot be interpreted as microvolts. Set the "
            "extractor's gain (e.g. `recording.set_channel_gains(...)` / "
            "`set_channel_offsets(...)`) before writing the fixture."
        )
    traces = recording.get_traces(return_in_uV=True)
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
        Planted spike times in seconds, per-unit soma positions of shape
        ``(n_units, 3)`` in micrometers, and normalized cell types.
    """
    import MEArec

    # ``templates`` loads the soma positions (otherwise NaN);
    # ``load_waveforms=False`` skips the waveform array.
    recgen = MEArec.load_recordings(
        str(mearec_h5_path),
        return_h5_objects=False,
        load=["spiketrains", "templates"],
        load_waveforms=False,
        verbose=False,
    )

    spike_times: list[np.ndarray] = []
    cell_types: list[CellType] = []
    for idx, spiketrain in enumerate(recgen.spiketrains):
        times_s = np.asarray(spiketrain.rescale("s").magnitude, dtype=float)
        spike_times.append(times_s)
        annotations = getattr(spiketrain, "annotations", {}) or {}
        if "cell_type" not in annotations:
            raise ValueError(
                f"MEArec ground-truth spiketrain {idx} in "
                f"{mearec_h5_path.name!r} has no 'cell_type' annotation; a "
                "MEArec GT fixture must carry it (no silent 'unknown' "
                "fallback)."
            )
        cell_types.append(_normalize_cell_type(annotations["cell_type"]))

    n_units = len(spike_times)
    locations = getattr(recgen, "template_locations", None)
    if locations is None:
        raise ValueError(
            f"MEArec recording {mearec_h5_path.name!r} has no "
            "template_locations; per-unit ground-truth positions are "
            "required. Load with load=['spiketrains', 'templates'] so the "
            "locations array is present (got None)."
        )
    locations = np.asarray(locations, dtype=float)
    # Drifting recordings carry a position per drift step; use the first.
    if locations.ndim == 3:
        locations = locations[:, 0, :]
    if (
        locations.ndim != 2
        or locations.shape[0] != n_units
        or locations.shape[1] > 3
    ):
        raise ValueError(
            f"MEArec template_locations for {mearec_h5_path.name!r} has "
            f"shape {locations.shape}; expected a 2-D array whose first "
            f"dimension equals n_units ({n_units}) and at most 3 spatial "
            "columns. Refusing to write all-NaN or silently-truncated "
            "ground-truth positions."
        )
    positions = np.zeros((n_units, 3), dtype=float)
    positions[:, : locations.shape[1]] = locations

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
        An empty NWBFile carrying only synthetic session and subject
        metadata (no electrodes, ephys, or behavior).
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
        source_script="tests.spikesorting.v2.fixtures.mearec_to_nwb",
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
    channel_names: list[str] | None = None,
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
    channel_names : list of str, optional
        When given, add a ``channel_name`` electrodes column (one per
        contact, electrode order), which SpikeInterface uses as the channel
        id as in Frank-lab NWBs. Omitted by default, so ids fall back to
        ``electrode_id``. Must have ``layout.n_contacts`` entries.
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

    if channel_names is not None:
        if len(channel_names) != layout.n_contacts:
            raise ValueError(
                f"channel_names has {len(channel_names)} entries but the "
                f"{layout.probe_type} layout has {layout.n_contacts} "
                "contacts; supply exactly one name per contact."
            )
        nwbfile.electrodes.add_column(
            name="channel_name",
            description="Per-channel string name (Frank-lab production NWBs)",
            data=list(channel_names),
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
    # Traces are in µV; conversion to volts follows trodes_to_nwb.
    eseries = ElectricalSeries(
        name="e-series",
        data=wrapped,
        electrodes=electrode_region,
        starting_time=0.0,
        rate=float(sampling_frequency),
        conversion=1e-6,
    )
    nwbfile.add_acquisition(eseries)


#: Sidecar ``ProcessingModule`` holding the planted ground-truth ``Units``.
GROUND_TRUTH_PROCESSING_MODULE = "ground_truth"

#: Name of the ``Units`` table within the sidecar processing module.
GROUND_TRUTH_UNITS_NAME = "units"


def get_ground_truth_units_table(nwbfile):
    """Return the sidecar ground-truth ``Units`` table, or ``None``.

    ``None`` when the file has no ground truth (e.g. a real lab NWB).

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        An opened NWB file.

    Returns
    -------
    pynwb.misc.Units or None
        The planted ground-truth ``Units`` table when present.
    """
    processing = getattr(nwbfile, "processing", None)
    if processing is None:
        return None
    module = processing.get(GROUND_TRUTH_PROCESSING_MODULE)
    if module is None:
        return None
    return module.data_interfaces.get(GROUND_TRUTH_UNITS_NAME)


def _add_ground_truth_units(nwbfile, ground_truth: _GroundTruth) -> None:
    """Add the planted ground-truth units in a sidecar processing module.

    Keeps ``nwbfile.units`` free:

    - writing a sorting to ``nwbfile.units`` fails if it is already set
    - readers of ``nwbfile.units`` would mistake planted units for sorter
      output

    Read them with :func:`get_ground_truth_units_table`.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        File under construction.
    ground_truth : _GroundTruth
        Planted spike trains and per-unit metadata.
    """
    import pynwb.misc

    units = pynwb.misc.Units(
        name=GROUND_TRUTH_UNITS_NAME,
        description="Planted ground-truth spike trains from MEArec.",
    )
    units.add_column(
        name="position_x", description="Ground-truth soma x position (um)"
    )
    units.add_column(
        name="position_y", description="Ground-truth soma y position (um)"
    )
    units.add_column(
        name="position_z", description="Ground-truth soma z position (um)"
    )
    units.add_column(
        name="cell_type", description="Ground-truth simulated cell type"
    )
    units.add_column(
        name="is_ground_truth",
        description="True for planted ground-truth units",
    )
    for unit_id in range(ground_truth.n_units):
        position = ground_truth.positions_um[unit_id]
        units.add_unit(
            id=unit_id,
            spike_times=ground_truth.spike_times[unit_id],
            position_x=float(position[0]),
            position_y=float(position[1]),
            position_z=float(position[2]),
            cell_type=ground_truth.cell_types[unit_id],
            is_ground_truth=True,
        )

    module = nwbfile.create_processing_module(
        name=GROUND_TRUTH_PROCESSING_MODULE,
        description=(
            "MEArec-simulated ground-truth units kept out of "
            "``nwbfile.units`` so the canonical units slot remains "
            "free for downstream sorter outputs."
        ),
    )
    module.add(units)


def _add_behavior_stub(
    nwbfile, n_samples: int, sampling_frequency: float
) -> None:
    """Add a minimal ``behavior`` processing module + flat ``Position``.

    ``PositionSource.make`` raises ``KeyError`` without
    ``processing["behavior"]``. The constant x=0, y=0 trace marks "no real
    trajectory" and gives interval-list generation timestamps to use.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        File under construction.
    n_samples : int
        Number of ephys samples; the position trace is decimated from
        this count.
    sampling_frequency : float
        Ephys sampling rate in Hz, used to build the position time-base.
    """
    import pynwb.behavior

    behavior_mod = nwbfile.create_processing_module(
        name="behavior", description="Synthetic behavior stub"
    )
    # ~100 Hz is enough. Explicit timestamps: master spyglass reads
    # ``timestamps[0]`` and ignores ``starting_time``+``rate``.
    target_hz = 100.0
    decim = max(1, int(round(sampling_frequency / target_hz)))
    n_pos = n_samples // decim
    timestamps = (np.arange(n_pos, dtype=np.float64) * decim) / float(
        sampling_frequency
    )
    spatial = pynwb.behavior.SpatialSeries(
        name="position",
        data=np.zeros((n_pos, 2), dtype=np.float32),
        reference_frame="synthetic",
        timestamps=timestamps,
        description=(
            "Flat constant-zero position trace at ~100 Hz; the "
            "MEArec fixture has no real trajectory but the "
            "synthetic series lets downstream Spyglass position / "
            "interval ingestion complete without special-casing."
        ),
    )
    position = pynwb.behavior.Position(spatial_series=spatial)
    behavior_mod.add(position)


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
    channel_names: list[str] | None = None,
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
        Brain region for the single ``NwbElectrodeGroup.location``, used as
        every electrode's default region. Per-shank regions are set after
        ingestion via ``Electrode.region_id``.
    channel_names : list of str, optional
        Per-contact ``channel_name`` electrodes column; see
        ``_add_probe_and_electrodes``.

    Raises
    ------
    ValueError
        If the MEArec channel count does not match the layout, or if the
        ground-truth read fails because of missing or invalid MEArec
        annotations (absent ``cell_type``, an unrecognized cell type, or
        missing/misshaped ``template_locations``).
    RuntimeError
        If the MEArec traces lack a microvolt conversion, or if
        NWBInspector reports a blocking finding on the written file.
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
    _add_probe_and_electrodes(
        nwbfile, probe_layout, targeted_location, channel_names=channel_names
    )
    _add_raw_ephys(nwbfile, traces, sampling_frequency)
    _add_ground_truth_units(nwbfile, ground_truth)
    _add_behavior_stub(nwbfile, traces.shape[0], sampling_frequency)

    out_nwb_path.parent.mkdir(parents=True, exist_ok=True)
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(out_nwb_path), mode="w") as io:
        io.write(nwbfile)

    _validate_with_nwbinspector(out_nwb_path)
