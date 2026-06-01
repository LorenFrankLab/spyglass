"""Convert MEArec ground-truth simulations into Spyglass-ingestible NWB files.

The output is structurally identical to a ``trodes_to_nwb``-produced NWB so that
Spyglass's ``insert_sessions`` ingests it end to end: one ``ndx_franklab_novela``
``Probe`` device, one ``NwbElectrodeGroup`` for the whole probe, an electrode
table carrying the novela columns ``Electrode.make()`` consumes, and an
``ElectricalSeries`` named ``"e-series"``. A ground-truth ``Units`` table -- not
part of normal ``trodes_to_nwb`` output -- is added in a SIDECAR
``ProcessingModule("ground_truth")`` so the simulation's planted spikes are
readable via :func:`get_ground_truth_units_table` for sort-accuracy oracles.
Keeping planted units OUT of ``nwbfile.units`` leaves the canonical units slot
free for real sorter outputs (the v1 baseline-capture path writes to
``nwbfile.units`` directly and would fail HDMF's "units already set" guard
otherwise).

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
from typing import Literal

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

# Single-tetrode probe matching the Frank-lab ``tetrode_12.5`` metadata
# at https://github.com/LorenFrankLab/trodes_to_nwb/blob/main/src/trodes_to_nwb/device_metadata/probe_metadata/tetrode_12.5.yml :
# 4 wires in a square, ±6.25 µm in x and z (rel_y = 0). One shank,
# one sort group, 4 channels. Used as the "sparse-probe sorter
# coverage" axis (vs the polymer 128c-4s).
TETRODE_PROBE_TYPE = "tetrode_12.5"
_TETRODE_DESCRIPTION = "four wire electrode"
_TETRODE_CONTACT_SIZE_UM = 12.5
_TETRODE_HALF_PITCH_UM = 6.25

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

    **Axis convention note** (mirrors :func:`tetrode_probe_layout`):
    the probe is laid out in the XY plane with ``rel_z=0`` for every
    contact, the frame MEArec/MEAutility require (probe normal = +Z; a
    contact with ``rel_z≠0`` raises ``TypeError: 'NoneType' object is
    not subscriptable`` deep inside MEAutility because
    ``electrode.normal`` is never computed). Within that plane the four
    shanks are offset along ``rel_x`` (``_POLYMER_SHANK_X_UM``) and the
    32 contacts of each shank run down ``rel_y`` at the 26 µm
    within-shank pitch (negative-going so contact 0 is the most
    superficial). Spyglass reads ``rel_x``/``rel_y``/``rel_z``
    symmetrically, so the committed NWB stores these positions verbatim.

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

    Mirrors the canonical metadata at
    https://github.com/LorenFrankLab/trodes_to_nwb/blob/main/src/trodes_to_nwb/device_metadata/probe_metadata/tetrode_12.5.yml :
    one shank with 4 wire-electrode contacts arranged in a square
    at ±6.25 µm pitch. 12.5 µm contact size, ``contact_side_numbering = true``.

    **Axis convention note**: trodes_to_nwb's YAML uses ``rel_y=0`` with
    varying ``rel_z`` (probe in XZ plane, normal = +Y). MEArec/MEAutility
    expect probes in the XY plane (``rel_z=0``, normal = +Z) -- e.g.
    :func:`polymer_probe_layout` and :func:`neuropixels_probe_layout`
    both follow this. A probe with ``rel_z≠0`` raises
    ``TypeError: 'NoneType' object is not subscriptable`` deep inside
    MEAutility because ``electrode.normal`` is never computed. We
    therefore swap the YAML's ``rel_y`` and ``rel_z``: same 2D square
    geometry, axes renamed to MEArec's expected frame. The committed
    NWB writes the same channel positions either way (Spyglass reads
    ``rel_x``/``rel_y``/``rel_z`` symmetrically).

    Single sort group (sort_group_id = 0), 4 channels. The narrow
    spatial extent means only units within ~30 µm of the tetrode
    contribute spikes, which is the realistic Frank-lab tetrode-
    drive behavior.

    Returns
    -------
    ProbeLayout
        1 shank × 4 contacts, square geometry in the XY plane.
    """
    half = _TETRODE_HALF_PITCH_UM
    # Order matches the YAML's id sequence: 0 (+x, +y), 1 (-x, +y),
    # 2 (-x, -y), 3 (+x, -y). rel_z=0 to put the probe in the XY plane
    # (MEArec convention) -- see docstring's "Axis convention note".
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


CellType = Literal["E", "I"]
"""MEArec's ground-truth ``cell_type`` annotation vocabulary.

MEArec tags each ground-truth spiketrain with ``cell_type`` in
``{"E", "I"}`` (excitatory / inhibitory; the single-letter codes are what
MEArec writes into the recording ``.h5``, verified across every committed
fixture). v2 types the ground-truth column against this set and normalizes
case/whitespace drift via :func:`_normalize_cell_type` so a stray
``"e "`` cannot silently become a distinct category.
"""

_VALID_CELL_TYPES: frozenset[str] = frozenset({"E", "I"})


def _normalize_cell_type(raw: object) -> CellType:
    """Normalize a MEArec ``cell_type`` annotation to the canonical set.

    Strips surrounding whitespace and upper-cases (MEArec writes ``"E"`` /
    ``"I"``), then validates against ``{"E", "I"}``. Raises ``ValueError``
    on a value outside the set (annotation drift / typo) rather than
    silently coercing it to a fallback.
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
    """Ground-truth units extracted from a MEArec recording generator."""

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
    # ``has_scaleable_traces()`` is False exactly when neither gain_to_uV
    # nor offset_to_uV is set. Check it explicitly: for a FLOAT extractor
    # with no gain, ``get_traces(return_in_uV=True)`` does NOT raise (it
    # returns the floats as if already microvolts), so relying on a raise
    # would let gainless float fixtures through as mislabeled uV.
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
    """
    import MEArec

    # ``templates`` brings in ``template_locations`` (the per-unit soma
    # x/y/z) without loading the heavy waveform array (``load_waveforms=
    # False``). Required so the GT sidecar can carry real position_x/y/z
    # columns; loading only ``spiketrains`` leaves
    # ``recgen.template_locations is None`` and all positions write as
    # NaN. ``load_waveforms=False`` keeps memory bounded -- only the
    # locations array is needed.
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


#: Name of the sidecar ``ProcessingModule`` that holds the planted
#: ground-truth ``Units`` table in fixtures generated by this module.
#: Kept here as a module-level constant so test code reads from the
#: same name without hard-coding string literals.
GROUND_TRUTH_PROCESSING_MODULE = "ground_truth"

#: Name of the ``Units`` table within the sidecar processing module.
GROUND_TRUTH_UNITS_NAME = "units"


def get_ground_truth_units_table(nwbfile):
    """Return the sidecar ground-truth ``Units`` table, or ``None``.

    Centralises the read path so consumers (parity tests, GT-accuracy
    tests, baseline-capture) never have to know the sidecar layout.
    Returns ``None`` if the fixture has no ground-truth (a real lab
    NWB, or a synthetic fixture written by older code).

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

    The planted units are *simulation metadata*, not a real spike-sorting
    output, so they live in a ``ProcessingModule`` named
    ``"ground_truth"`` containing a ``Units`` table named ``"units"``.
    This keeps the top-level ``nwbfile.units`` attribute free, which
    matters because:

    - master Spyglass's v1 ``SpikeSorting._write_sorting_to_nwb`` calls
      ``nwbf.units = pynwb.misc.Units(...)`` on the analysis NWB
      inherited from the source; if the source ships planted units the
      assignment fails with ``AttributeError: can't set attribute
      'units' -- already set``, blocking the v1 baseline-capture
      workflow used by the v1↔v2 parity gate.
    - ``ImportedSpikeSorting.insert_from_nwbfile`` historically reads
      ``nwbfile.units``; leaving GT there silently pollutes the
      v2-merge dispatcher with rows that look like real sorter output.

    Consumers read via :func:`get_ground_truth_units_table` rather than
    poking the processing module directly so the sidecar layout can
    evolve in one place.

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

    Spyglass's ``populate_all_common`` runs ``PositionSource.make`` which
    calls ``get_position_obj``: that helper unconditionally accesses
    ``nwbfile.processing["behavior"]`` and crashes with ``KeyError`` if
    the module is absent. Even a behavior module that contains no
    ``SpatialSeries`` would satisfy the dict-access; we go a step
    further and add a flat (constant) position trace so downstream
    interval-list generation (``raw data valid times``) has something
    to bind against without giving the simulator a real trajectory it
    never had.

    The trace is ``(n_samples, 2)`` at the recording's sampling rate,
    matching the ephys time-base byte for byte. Constant x=0, y=0
    explicitly signals "no real trajectory" to anyone reading the
    file. Spyglass parses this through its standard Position path,
    so the fixture works under master spyglass + SI 0.99 (the v1
    parity capture environment) as well as the v2 ingestion paths.
    """
    import pynwb.behavior

    behavior_mod = nwbfile.create_processing_module(
        name="behavior", description="Synthetic behavior stub"
    )
    # Downsample to ~100 Hz to avoid duplicating the ephys time-base
    # at every sample; downstream Spyglass position parsing only
    # needs a sane (timestamps, x, y) shape, not ephys-resolution
    # position. Master spyglass also reads ``timestamps[0]`` directly
    # (it does NOT honor ``starting_time``+``rate``) so explicit
    # timestamps are required for compatibility.
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
    _add_behavior_stub(nwbfile, traces.shape[0], sampling_frequency)

    out_nwb_path.parent.mkdir(parents=True, exist_ok=True)
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(out_nwb_path), mode="w") as io:
        io.write(nwbfile)

    _validate_with_nwbinspector(out_nwb_path)
