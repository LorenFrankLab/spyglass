"""Persisted-geometry and semantic round-trip contracts for the v2 writer.

``write_nwb_artifact`` streams the preprocessed recording into an
``AnalysisNwbfile``. SpikeInterface rebuilds a reloaded recording's channel
locations from that file's electrodes rows (``rel_x``/``rel_y``/``rel_z`` --
see ``NwbRecordingExtractor._fetch_locations_and_groups``), so the geometry the
writer persists IS the geometry every later stage sees. These tests pin that
the writer writes the recording's *normalized 2D* geometry (constant z) into
exactly the rows the ElectricalSeries references, and that a reload -- including
one served by ``_rebuild_nwb_artifact`` -- reproduces the recording the writer
was handed.

The fixtures here are built to discriminate rather than to be convenient:
contacts lie in the x-z plane (so an un-normalized x-y projection would
collapse them), electrode ids are non-contiguous AND out of row order (so an
id-as-row-index bug mis-points the region), the raw series carries a uniform
non-unit gain with a non-zero offset (so a dropped conversion shows up in
microvolts), the sort interval has two disjoint pieces, and the artifact is
long enough that the writer streams it in several chunks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------

# A deliberately non-Frank-lab probe type: ``maybe_apply_tetrode_geometry``
# patches only 4-channel ``tetrode_12.5`` groups, and these fixtures must
# exercise the plane-normalization path rather than the tetrode repair.
_XZ_PROBE_TYPE = "v2_xz_test_probe"
_CONTACT_PITCH_UM = 30.0
_SAMPLING_FREQUENCY = 1000.0

# Uniform, non-unit calibration: microvolts per count and a non-zero DC
# offset. ElectricalSeries ``conversion``/``offset`` are in volts.
_GAIN_UV_PER_COUNT = 0.195
_OFFSET_UV = 5.0

# Electrode ids: non-contiguous and NOT in electrodes-table row order, so the
# id -> row mapping the writer performs is load-bearing (row ``k`` does not
# hold electrode ``k``, and the ids skip values).
#
# The ids a sort group uses must still land on ASCENDING rows: SpikeInterface
# 0.104.3 reads an ElectricalSeries' electrode columns with a single h5py
# fancy index (``NwbRecordingExtractor._fetch_locations_and_groups``), and
# h5py rejects a non-monotonic one ("Indexing elements must be in increasing
# order"). That is a reader limitation independent of anything here, so the
# fixtures put each exercised group's ids on increasing rows while keeping the
# table as a whole permuted.
_SMALL_ELECTRODE_IDS = (3, 30, 5, 40, 9, 12, 17, 21)
# Rows 0, 2, 4, 5 -- ascending rows, non-contiguous ids, and id != row for
# every one of them.
_SMALL_GROUP_IDS = (3, 5, 9, 12)


def _xz_positions(n_contacts: int) -> np.ndarray:
    """Contact positions in the x-z plane, ``(n_contacts, 3)`` micrometres.

    ``rel_y`` is constant (0) and ``rel_x`` repeats every other contact, so the
    x-y projection SpikeInterface reaches for first collapses the contacts into
    coincident pairs while the x-z projection separates all of them. This is
    the real Frank-lab tetrode failure mode, reproduced without a tetrode.
    """
    index = np.arange(n_contacts)
    return np.column_stack(
        [
            _CONTACT_PITCH_UM * (index % 2),
            np.zeros(n_contacts),
            -_CONTACT_PITCH_UM * (index // 2),
        ]
    ).astype(float)


def _write_xz_probe_nwb(
    out_path,
    *,
    electrode_ids,
    n_samples: int,
    seed: int,
    fixture_name: str,
):
    """Write an ingestible Spyglass NWB whose contacts lie in the x-z plane.

    Mirrors the ``trodes_to_nwb`` layout Spyglass ingestion expects (one
    ``ndx_franklab_novela`` probe device, one electrode group, the novela
    electrode columns) but takes explicit, unordered electrode ids and writes
    the raw series with a uniform non-unit ``conversion`` and a non-zero
    ``offset``.

    Parameters
    ----------
    out_path : pathlib.Path or str
        Destination NWB path.
    electrode_ids : sequence of int
        Electrode ids in electrodes-table ROW order. Row ``k`` gets id
        ``electrode_ids[k]``.
    n_samples : int
        Number of samples in the raw ``ElectricalSeries``.
    seed : int
        Seed for the synthetic int16 traces.
    fixture_name : str
        Recorded as the NWB ``session_id`` / identifier stem.

    Returns
    -------
    pathlib.Path
        ``out_path`` (written).
    """
    import pynwb
    from ndx_franklab_novela import (
        NwbElectrodeGroup,
        Probe,
        Shank,
        ShanksElectrode,
    )

    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import _build_nwbfile

    electrode_ids = [int(e) for e in electrode_ids]
    positions = _xz_positions(len(electrode_ids))

    nwbfile = _build_nwbfile(
        fixture_name=fixture_name,
        session_start=datetime(2024, 3, 1, tzinfo=timezone.utc),
    )
    probe = Probe(
        id=0,
        name="probe 0",
        probe_type=_XZ_PROBE_TYPE,
        units="um",
        probe_description="synthetic x-z plane probe",
        contact_side_numbering=True,
        contact_size=12.5,
    )
    electrode_group = NwbElectrodeGroup(
        name="0",
        description="synthetic x-z plane probe",
        location="CA1",
        targeted_location="CA1",
        targeted_x=0.0,
        targeted_y=0.0,
        targeted_z=0.0,
        units="um",
        device=probe,
    )
    nwbfile.add_electrode_group(electrode_group)

    shank = Shank(name="0")
    for eid, (rel_x, rel_y, rel_z) in zip(electrode_ids, positions):
        shank.add_shanks_electrode(
            ShanksElectrode(
                name=str(eid),
                rel_x=float(rel_x),
                rel_y=float(rel_y),
                rel_z=float(rel_z),
            )
        )
    probe.add_shank(shank)
    nwbfile.add_device(probe)

    for eid, (rel_x, rel_y, rel_z) in zip(electrode_ids, positions):
        nwbfile.add_electrode(
            id=int(eid),
            location="CA1",
            group=electrode_group,
            rel_x=float(rel_x),
            rel_y=float(rel_y),
            rel_z=float(rel_z),
            x=0.0,
            y=0.0,
            z=0.0,
            imp=0.0,
            filtering="none",
        )
    n_contacts = len(electrode_ids)
    nwbfile.electrodes.add_column(
        name="probe_shank",
        description="The shank of the probe this channel is located on",
        data=[0] * n_contacts,
    )
    nwbfile.electrodes.add_column(
        name="probe_electrode",
        description="The ID of this electrode with respect to the probe",
        data=[int(e) for e in electrode_ids],
    )
    nwbfile.electrodes.add_column(
        name="bad_channel",
        description="True if noisy or disconnected",
        data=[False] * n_contacts,
    )
    nwbfile.electrodes.add_column(
        name="ref_elect_id",
        description="Experimenter selected reference electrode id",
        data=[-1] * n_contacts,
    )

    rng = np.random.default_rng(seed)
    traces = rng.integers(
        -2000, 2000, size=(int(n_samples), n_contacts), dtype=np.int16
    )
    nwbfile.add_acquisition(
        pynwb.ecephys.ElectricalSeries(
            name="e-series",
            data=traces,
            electrodes=nwbfile.create_electrode_table_region(
                region=list(range(n_contacts)),
                description="electrodes used in raw e-series recording",
            ),
            starting_time=0.0,
            rate=_SAMPLING_FREQUENCY,
            # uV -> V: a uniform, non-unit calibration with a real DC offset.
            conversion=_GAIN_UV_PER_COUNT * 1e-6,
            offset=_OFFSET_UV * 1e-6,
        )
    )

    out_path = Path(out_path)
    with pynwb.NWBHDF5IO(str(out_path), mode="w") as io:
        io.write(nwbfile)
    return out_path


def _ingest_synthetic_nwb(path) -> str:
    """Ingest a freshly synthesized NWB, replacing any earlier copy of it.

    ``copy_and_insert_nwb`` skips the copy when the raw directory already holds
    a file of that name. These fixtures are synthesized per run (not downloaded
    assets), so a stale copy from an earlier run would silently be ingested
    instead of the file the test just built.
    """
    from spyglass.settings import raw_dir
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    path = Path(path)
    for stale in (path.name, get_nwb_copy_filename(path.name)):
        (Path(raw_dir) / stale).unlink(missing_ok=True)
    return copy_and_insert_nwb(path)


@pytest.fixture(scope="session")
def xz_probe_session(dj_conn, tmp_path_factory):
    """Ingest a short 8-electrode x-z session; yield its session key."""
    path = _write_xz_probe_nwb(
        tmp_path_factory.mktemp("xz_probe") / "v2_xz_probe.nwb",
        electrode_ids=_SMALL_ELECTRODE_IDS,
        n_samples=int(5 * _SAMPLING_FREQUENCY),
        seed=7,
        fixture_name="v2_xz_probe",
    )
    yield {"nwb_file_name": _ingest_synthetic_nwb(path)}


def _analysis_electrodes(analysis_file_name):
    """Return the analysis file's ``(rel_x, rel_y, rel_z)`` by electrode id.

    Reads the persisted electrodes table directly (h5py, no SpikeInterface) so
    the assertion is about the bytes on disk rather than about any reader's
    interpretation of them.
    """
    import h5py

    from spyglass.common.common_nwbfile import AnalysisNwbfile

    abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
    with h5py.File(abs_path, "r") as handle:
        group = handle["/general/extracellular_ephys/electrodes"]
        ids = [int(e) for e in group["id"][:]]
        coords = np.column_stack(
            [group[column][:] for column in ("rel_x", "rel_y", "rel_z")]
        )
    return dict(zip(ids, coords))


def _drop_analysis_file(analysis_file_name) -> None:
    """Unlink an analysis file that was written but never registered."""
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    Path(AnalysisNwbfile.get_abs_path(analysis_file_name)).unlink(
        missing_ok=True
    )


# ---------------------------------------------------------------------------
# 1. The writer persists the recording's 2D geometry -- and only its own rows
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.database
def test_persist_geometry_writes_series_rows_only(xz_probe_session):
    """The writer stamps the recording's 2D locations onto the electrodes rows
    the ElectricalSeries references, leaves every other row at its raw value,
    and the geometry it writes reaches the content fingerprint.

    Without this the reloaded recording is rebuilt from the PARENT file's x-z
    coordinates: SpikeInterface projects those to x-y, the contacts collapse,
    and the in-memory normalization the sort ran with is lost.
    """
    import spikeinterface.core as si_core

    from spyglass.spikesorting.v2.recording import Recording

    nwb_file_name = xz_probe_session["nwb_file_name"]
    raw_geometry = dict(
        zip(_SMALL_ELECTRODE_IDS, _xz_positions(len(_SMALL_ELECTRODE_IDS)))
    )
    # A recording over four of the eight electrodes, in ascending id order
    # (what ``select_sort_group_channels`` produces) -- which is neither the
    # electrodes-table row order nor a contiguous id run. The locations are
    # the x-z plane those four rows normalize to.
    channel_ids = list(_SMALL_GROUP_IDS)
    locations = np.array(
        [[0.0, 0.0], [0.0, -30.0], [0.0, -60.0], [30.0, -60.0]], dtype=float
    )
    # One trace array, reused by both writes: the second write differs from
    # the first ONLY in channel locations, so a hash difference can come from
    # nothing else.
    traces = np.random.default_rng(0).integers(
        -500, 500, size=(500, len(channel_ids)), dtype=np.int16
    )

    def _build(locations):
        recording = si_core.NumpyRecording(
            [traces],
            sampling_frequency=_SAMPLING_FREQUENCY,
            channel_ids=channel_ids,
        )
        recording.set_channel_gains(_GAIN_UV_PER_COUNT)
        recording.set_channel_offsets(_OFFSET_UV)
        recording.set_channel_locations(locations)
        return recording

    first_name, _, first_hash = Recording._write_nwb_artifact(
        _build(locations),
        nwb_file_name,
        filtering_description="persist-geometry probe",
    )
    try:
        persisted = _analysis_electrodes(first_name)

        for eid, position in zip(channel_ids, locations):
            np.testing.assert_array_equal(
                persisted[eid],
                np.array([position[0], position[1], 0.0]),
                err_msg=(
                    f"electrode {eid} is referenced by the series, so its "
                    "persisted rel_x/rel_y/rel_z must be the recording's "
                    "normalized 2D position with rel_z = 0"
                ),
            )
        untouched = set(_SMALL_ELECTRODE_IDS) - set(channel_ids)
        for eid in sorted(untouched):
            np.testing.assert_array_equal(
                persisted[eid],
                raw_geometry[eid],
                err_msg=(
                    f"electrode {eid} is NOT in the series region; the writer "
                    "must not touch its geometry"
                ),
            )

        # Same traces, different geometry -> a different scientific identity.
        shifted = locations + np.array([100.0, 0.0])
        second_name, _, second_hash = Recording._write_nwb_artifact(
            _build(shifted),
            nwb_file_name,
            filtering_description="persist-geometry probe",
        )
        try:
            assert second_hash != first_hash, (
                "the persisted geometry must feed the content fingerprint; "
                "two writes that differ only in channel locations hashed the "
                "same"
            )
        finally:
            _drop_analysis_file(second_name)
    finally:
        _drop_analysis_file(first_name)
