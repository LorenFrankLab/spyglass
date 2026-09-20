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

import h5py
import numpy as np
import pytest

from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------

# Deliberately non-Frank-lab probe types: ``maybe_apply_tetrode_geometry``
# patches only 4-channel ``tetrode_12.5`` groups, and these fixtures must
# exercise the plane-normalization path rather than the tetrode repair. One
# name per fixture -- ``ProbeType`` is keyed on the name and its shank count
# differs between them.
_SMALL_PROBE_TYPE = "v2_xz_test_probe_1shank"
_ROUNDTRIP_PROBE_TYPE = "v2_xz_test_probe_2shank"
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
_SMALL_CONTACTS = tuple((eid, 0) for eid in _SMALL_ELECTRODE_IDS)

# The full-pipeline fixture: two shanks interleaved row by row, so the table's
# ids run 5, 60, 9, 61, ... -- permuted and non-contiguous -- while each
# shank's own ids still ascend with its rows.
_ROUNDTRIP_GROUP_IDS = (5, 9, 14, 22, 30, 38, 47, 55)
_ROUNDTRIP_OTHER_IDS = (60, 61, 62, 63, 64, 65, 66, 67)
_ROUNDTRIP_CONTACTS = tuple(
    contact
    for pair in zip(
        ((eid, 0) for eid in _ROUNDTRIP_GROUP_IDS),
        ((eid, 1) for eid in _ROUNDTRIP_OTHER_IDS),
    )
    for contact in pair
)
# Raw span and the two disjoint sort intervals carved out of it (seconds).
_ROUNDTRIP_RAW_SECONDS = 720.0
_ROUNDTRIP_INTERVALS = ((10.0, 350.0), (370.0, 710.0))
_ROUNDTRIP_PARAMS_NAME = "_pytest_xz_roundtrip"


def _xz_geometry(contacts) -> dict:
    """Map each electrode id to its ``(rel_x, rel_y, rel_z)`` micrometres.

    ``rel_y`` is constant (0) and ``rel_x`` repeats every other contact within
    a shank, so the x-y projection SpikeInterface reaches for first collapses
    the contacts into coincident pairs while the x-z projection separates all
    of them. This is the real Frank-lab tetrode failure mode, reproduced
    without a tetrode. Shanks are offset along ``rel_x`` so they never overlap.

    Parameters
    ----------
    contacts : sequence of (int, int)
        ``(electrode_id, shank_id)`` in electrodes-table row order.

    Returns
    -------
    dict
        ``{electrode_id: numpy.ndarray of shape (3,)}``.
    """
    within_shank: dict[int, int] = {}
    geometry = {}
    for electrode_id, shank_id in contacts:
        index = within_shank.get(shank_id, 0)
        within_shank[shank_id] = index + 1
        geometry[int(electrode_id)] = np.array(
            [
                300.0 * shank_id + _CONTACT_PITCH_UM * (index % 2),
                0.0,
                -_CONTACT_PITCH_UM * (index // 2),
            ],
            dtype=float,
        )
    return geometry


def _write_xz_probe_nwb(
    out_path,
    *,
    contacts,
    n_samples: int,
    seed: int,
    fixture_name: str,
    probe_type: str,
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
    contacts : sequence of (int, int)
        ``(electrode_id, shank_id)`` in electrodes-table ROW order, so row
        ``k`` gets id ``contacts[k][0]``.
    n_samples : int
        Number of samples in the raw ``ElectricalSeries``.
    seed : int
        Seed for the synthetic int16 traces.
    fixture_name : str
        Recorded as the NWB ``session_id`` / identifier stem.
    probe_type : str
        ``Probe.probe_type`` for the synthetic device. Distinct per fixture:
        Spyglass keys ``ProbeType`` on the name, so two fixtures sharing one
        name but differing in shank count collide on ingestion.

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

    contacts = [(int(eid), int(shank)) for eid, shank in contacts]
    geometry = _xz_geometry(contacts)

    nwbfile = _build_nwbfile(
        fixture_name=fixture_name,
        session_start=datetime(2024, 3, 1, tzinfo=timezone.utc),
    )
    probe = Probe(
        id=0,
        name="probe 0",
        probe_type=probe_type,
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

    for shank_id in sorted({shank for _, shank in contacts}):
        shank = Shank(name=str(shank_id))
        for eid, contact_shank in contacts:
            if contact_shank != shank_id:
                continue
            rel_x, rel_y, rel_z = geometry[eid]
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

    for eid, _ in contacts:
        rel_x, rel_y, rel_z = geometry[eid]
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
    n_contacts = len(contacts)
    nwbfile.electrodes.add_column(
        name="probe_shank",
        description="The shank of the probe this channel is located on",
        data=[shank for _, shank in contacts],
    )
    nwbfile.electrodes.add_column(
        name="probe_electrode",
        description="The ID of this electrode with respect to the probe",
        data=[eid for eid, _ in contacts],
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
        contacts=_SMALL_CONTACTS,
        n_samples=int(5 * _SAMPLING_FREQUENCY),
        seed=7,
        fixture_name="v2_xz_probe",
        probe_type=_SMALL_PROBE_TYPE,
    )
    yield {"nwb_file_name": _ingest_synthetic_nwb(path)}


@pytest.fixture(scope="session")
def xz_roundtrip_session(dj_conn, tmp_path_factory):
    """Ingest the long two-shank x-z session the round-trip test sorts.

    Long enough that the writer streams the artifact in several chunks (see
    ``test_recording_semantic_round_trip``), which is why it is session-scoped:
    synthesizing and ingesting it is the heaviest step in this module.
    """
    path = _write_xz_probe_nwb(
        tmp_path_factory.mktemp("xz_roundtrip") / "v2_xz_roundtrip.nwb",
        contacts=_ROUNDTRIP_CONTACTS,
        n_samples=int(_ROUNDTRIP_RAW_SECONDS * _SAMPLING_FREQUENCY),
        seed=11,
        fixture_name="v2_xz_roundtrip",
        probe_type=_ROUNDTRIP_PROBE_TYPE,
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
    raw_geometry = _xz_geometry(_SMALL_CONTACTS)
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


# ---------------------------------------------------------------------------
# 2. Semantic round trip: populate -> read -> rebuild -> read
# ---------------------------------------------------------------------------


def _sort_group_for(nwb_file_name: str, electrode_id: int) -> int:
    """Return the sort group that owns ``electrode_id``."""
    from spyglass.spikesorting.v2.recording import SortGroupV2

    return int(
        (
            SortGroupV2.SortGroupElectrode
            & {
                "nwb_file_name": nwb_file_name,
                "electrode_id": int(electrode_id),
            }
        ).fetch1("sort_group_id")
    )


def _replay(recording, requests, *, return_in_uV=True):
    """Yield ``recording``'s traces for each recorded writer request."""
    for start, stop, channel_ids in requests:
        yield recording.get_traces(
            segment_index=0,
            channel_ids=channel_ids,
            start_frame=start,
            end_frame=stop,
            return_in_uV=return_in_uV,
        )


@pytest.mark.slow
@pytest.mark.database
@pytest.mark.pipeline
def test_recording_semantic_round_trip(xz_roundtrip_session, monkeypatch):
    """The artifact reproduces the recording the writer was handed -- and so
    does the rebuild that replaces a deleted artifact.

    Pins the whole persisted surface, not just the traces: the wall-clock
    timestamps, the channel-id order, the normalized 2D geometry, and the
    microvolt values. SpikeInterface rebuilds channel locations from the
    persisted electrodes rows, so an un-persisted normalization shows up here
    as an x-y collapse.

    The microvolt reference is read from the pre-write lazy recording at the
    writer's OWN chunk boundaries. A single whole-recording request is not a
    valid reference: the lazy bandpass filters each request over a short margin
    of context, so its interior values depend on where the request boundaries
    fall. The test asserts that too, so the reason for the chunked reference is
    documented rather than assumed.
    """
    from spyglass.common import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import _nwb_iterators as iterators_module
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = xz_roundtrip_session["nwb_file_name"]
    initialize_v2_defaults()
    LabTeam.insert1(
        {
            "team_name": "v2_xz_team",
            "team_description": "v2 persisted-geometry tests",
        },
        skip_duplicates=True,
    )
    # A 1 kHz fixture cannot use the shipped 300-6000 Hz default (6000 Hz is
    # past Nyquist). Everything else is the schema default.
    PreprocessingParameters().insert1(
        {
            "preprocessing_params_name": _ROUNDTRIP_PARAMS_NAME,
            "params": PreprocessingParamsSchema.model_validate(
                {"bandpass_filter": {"freq_min": 100.0, "freq_max": 400.0}}
            ).model_dump(),
        },
        skip_duplicates=True,
    )
    if not (SortGroupV2 & {"nwb_file_name": nwb_file_name}):
        SortGroupV2.set_group_by_shank(
            nwb_file_name=nwb_file_name, reference_mode="global_median"
        )
    interval_list_name = "v2_xz_two_intervals"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
            "valid_times": np.asarray(_ROUNDTRIP_INTERVALS, dtype=float),
            "pipeline": "v2_xz_two_intervals",
        },
        skip_duplicates=True,
    )
    pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": _sort_group_for(
                nwb_file_name, _ROUNDTRIP_GROUP_IDS[0]
            ),
            "interval_list_name": interval_list_name,
            "preprocessing_params_name": _ROUNDTRIP_PARAMS_NAME,
            "team_name": "v2_xz_team",
        }
    )

    # Capture the recording the writer is handed, and the exact trace requests
    # the chunk iterator issues while streaming it.
    captured: dict = {}
    real_write = Recording._write_nwb_artifact
    requests: list = []
    real_iterator = iterators_module.SpikeInterfaceRecordingDataChunkIterator

    class _RecordingRequests(real_iterator):
        def _get_data(self, selection):
            requests.append(
                (
                    selection[0].start,
                    selection[0].stop,
                    list(self.channel_ids[selection[1]]),
                )
            )
            return super()._get_data(selection)

    def _capture_write(**kwargs):
        captured.update(kwargs)
        return real_write(**kwargs)

    monkeypatch.setattr(
        Recording, "_write_nwb_artifact", staticmethod(_capture_write)
    )
    monkeypatch.setattr(
        iterators_module,
        "SpikeInterfaceRecordingDataChunkIterator",
        _RecordingRequests,
    )

    (Recording & pk).super_delete(warn=False, force_masters=True)
    Recording.populate(pk, reserve_jobs=False)
    monkeypatch.undo()

    source = captured["recording"]
    source_timestamps = np.asarray(captured["timestamps_override"], float)
    write_requests = list(requests)
    row = (Recording & pk).fetch1()

    assert len(write_requests) >= 3, (
        "the fixture must be long enough that the writer streams it in "
        f"several chunks; it issued {len(write_requests)} get_traces requests"
    )
    assert len({(start, stop) for start, stop, _ in write_requests}) >= 3, (
        "the writer's requests must span at least three distinct sample "
        "ranges, so the chunked reference is genuinely chunk-boundary "
        f"dependent (got {sorted({(s, e) for s, e, _ in write_requests})})"
    )
    assert source.get_num_segments() == 1
    assert source.get_num_samples() == len(
        source_timestamps
    ), "the persisted timestamps must cover every written sample"

    # The selection's two disjoint pieces survive as one gap in the persisted
    # wall clock: the artifact's samples are contiguous, its timestamps are
    # not.
    sample_period = 1.0 / _SAMPLING_FREQUENCY
    first_interval, second_interval = _ROUNDTRIP_INTERVALS
    gaps = np.flatnonzero(np.diff(source_timestamps) > 1.5 * sample_period)
    assert len(gaps) == 1, (
        "the two selected intervals must leave exactly one wall-clock gap in "
        f"the persisted timestamps, found {len(gaps)}"
    )
    assert source_timestamps[0] == pytest.approx(
        first_interval[0], abs=sample_period
    )
    assert source_timestamps[gaps[0]] == pytest.approx(
        first_interval[1], abs=sample_period
    )
    assert source_timestamps[gaps[0] + 1] == pytest.approx(
        second_interval[0], abs=sample_period
    )
    assert source_timestamps[-1] == pytest.approx(
        second_interval[1], abs=sample_period
    )
    # ``duration_s`` on the row is the gap-EXCLUDING saved span.
    selected_seconds = sum(stop - start for start, stop in _ROUNDTRIP_INTERVALS)
    assert float(row["duration_s"]) == pytest.approx(
        selected_seconds, abs=10 * sample_period
    )
    assert int(row["n_channels"]) == source.get_num_channels()
    assert float(row["sampling_frequency"]) == pytest.approx(
        _SAMPLING_FREQUENCY
    )

    def _assert_matches(reloaded, label):
        np.testing.assert_array_equal(
            np.asarray(reloaded.get_channel_ids()),
            np.asarray(source.get_channel_ids()),
            err_msg=f"{label}: channel-id order changed across the write",
        )
        np.testing.assert_array_equal(
            np.asarray(reloaded.get_channel_locations(), dtype=float),
            np.asarray(source.get_channel_locations(), dtype=float),
            err_msg=(
                f"{label}: the reloaded 2D geometry is not the normalized "
                "geometry the sort ran on -- the writer did not persist it"
            ),
        )
        np.testing.assert_array_equal(
            np.asarray(reloaded.get_times(), dtype=float),
            source_timestamps,
            err_msg=f"{label}: persisted wall-clock timestamps changed",
        )
        for expected, got in zip(
            _replay(source, write_requests), _replay(reloaded, write_requests)
        ):
            np.testing.assert_allclose(
                got,
                expected,
                rtol=0.0,
                atol=1e-6,
                err_msg=(
                    f"{label}: microvolt traces diverged from the pre-write "
                    "recording read at the writer's own chunk boundaries"
                ),
            )

    first_load = Recording().get_recording(pk)
    _assert_matches(first_load, "first load")

    # A whole-recording request is NOT a valid reference: the lazy bandpass
    # pulls only a short margin of context around each request, so its interior
    # values move when the request boundaries move.
    whole = first_load.get_traces(return_in_uV=True)
    unchunked = source.get_traces(return_in_uV=True)
    assert np.max(np.abs(whole - unchunked)) > 1e-6, (
        "a single whole-recording request reproduced the chunked write "
        "exactly, so this test is not actually pinning the chunk-boundary "
        "dependence it claims to"
    )
    del whole, unchunked

    # The rebuild path serves the same science from the same row.
    Path(
        AnalysisNwbfile.get_abs_path(
            row["analysis_file_name"], from_schema=True
        )
    ).unlink()
    _assert_matches(Recording().get_recording(pk), "rebuilt load")
    assert (Recording & pk).fetch1("content_hash") == row[
        "content_hash"
    ], "the rebuild must reproduce the stored content hash"


# ---------------------------------------------------------------------------
# 3. Tracked follow-up: specific reference drops the per-channel calibration
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.database
@pytest.mark.xfail(
    strict=True,
    reason=(
        "tracked follow-up: a 'specific' reference subtracts RAW counts and "
        "then zeroes the channel offsets, so with unequal per-channel offsets "
        "and no bandpass to remove the DC the per-channel calibration "
        "(offset_i - offset_ref) is lost before the writer's uniform-offset "
        "guard can see it"
    ),
)
def test_specific_reference_physical_units_oracle(xz_probe_session):
    """Reloaded microvolts must equal the reference subtraction done in
    PHYSICAL units, not in raw counts.

    The oracle is built from the raw counts and the per-channel calibration --
    ``(raw_i * gain_i + offset_i) - (raw_ref * gain_ref + offset_ref)`` -- not
    from the preprocessed in-memory recording, which already carries the loss.
    """
    import spikeinterface.core as si_core

    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )
    from spyglass.spikesorting.v2._recording_nwb import read_recording_nwb
    from spyglass.spikesorting.v2._recording_preprocessing import (
        apply_spatial_preprocessing,
    )
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.recording import (
        _ELECTRICAL_SERIES_PATH,
        Recording,
    )

    nwb_file_name = xz_probe_session["nwb_file_name"]
    channel_ids = [3, 5, 9]  # rows 0, 2, 4; electrode 9 is the reference
    reference_electrode_id = 9
    # Uniform gain (a single ElectricalSeries conversion can carry it) but
    # UNEQUAL per-channel offsets, which it cannot.
    offsets_uv = np.array([5.0, 44.0, 200.0])
    raw = np.random.default_rng(3).integers(
        -400, 400, size=(400, len(channel_ids)), dtype=np.int16
    )
    recording = si_core.NumpyRecording(
        [raw],
        sampling_frequency=_SAMPLING_FREQUENCY,
        channel_ids=channel_ids,
    )
    recording.set_channel_gains(_GAIN_UV_PER_COUNT)
    recording.set_channel_offsets(offsets_uv)
    recording.set_channel_locations(
        np.array([[0.0, 0.0], [0.0, -30.0], [0.0, -60.0]])
    )

    referenced, _ = apply_spatial_preprocessing(
        recording,
        reference_mode="specific",
        reference_electrode_id=reference_electrode_id,
        # ``bandpass_filter=None``: no filter runs, so nothing removes the DC
        # the offsets describe.
        validated=PreprocessingParamsSchema.model_validate(
            {"bandpass_filter": None}
        ),
    )

    kept = [0, 1]  # electrodes 3 and 5; electrode 9 was the reference
    oracle = (raw[:, kept] * _GAIN_UV_PER_COUNT + offsets_uv[kept]) - (
        raw[:, [2]] * _GAIN_UV_PER_COUNT + offsets_uv[2]
    )

    analysis_file_name, _, _ = Recording._write_nwb_artifact(
        referenced,
        nwb_file_name,
        filtering_description="no filter, specific reference",
    )
    try:
        reloaded = read_recording_nwb(
            AnalysisNwbfile.get_abs_path(analysis_file_name),
            electrical_series_path=_ELECTRICAL_SERIES_PATH,
        )
        np.testing.assert_allclose(
            reloaded.get_traces(return_in_uV=True),
            oracle,
            rtol=0.0,
            atol=1e-6,
        )
    finally:
        _drop_analysis_file(analysis_file_name)


# ---------------------------------------------------------------------------
# 4. A parent NWB without rel_* columns still gets the geometry (no DB)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_geometry_columns_are_created_when_absent(tmp_path):
    """The writer's persistence step works on an electrodes table that never
    carried ``rel_x``/``rel_y``/``rel_z``.

    ``AnalysisNwbfile().create`` exports the parent's electrodes table
    verbatim, so a parent written without probe-relative positions leaves the
    normalized geometry nowhere to land. hdmf accepts a new column on a table
    that is already on disk (the file is open in append mode), which is what
    the writer relies on; this pins that, and that only the referenced rows
    end up with a position at all -- every other row stays ``NaN``.

    Exercises the two file-level helpers directly: they touch no database, and
    the branch is unreachable from the Frank-lab-shaped fixtures above (their
    parents all carry the columns).
    """
    import pynwb

    from spyglass.spikesorting.v2._recording_nwb import (
        _ensure_relative_position_columns,
        _persist_channel_geometry,
    )

    path = tmp_path / "no_rel_columns.nwb"
    nwbfile = pynwb.NWBFile(
        session_description="electrodes table without rel_* columns",
        identifier="no-rel-columns",
        session_start_time=datetime(2024, 3, 1, tzinfo=timezone.utc),
    )
    device = nwbfile.create_device(name="probe0")
    group = nwbfile.create_electrode_group(
        name="0", description="g", location="CA1", device=device
    )
    for electrode_id in _SMALL_ELECTRODE_IDS:
        nwbfile.add_electrode(id=int(electrode_id), location="CA1", group=group)
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)

    with pynwb.NWBHDF5IO(str(path), mode="a", load_namespaces=True) as io:
        reopened = io.read()
        assert "rel_x" not in reopened.electrodes.colnames
        _ensure_relative_position_columns(reopened)
        io.write(reopened)

    rows = [2, 4, 5]
    locations = np.array([[0.0, -30.0], [0.0, -60.0], [30.0, -60.0]])
    # Raises on its own read-back if the columns did not take.
    _persist_channel_geometry(str(path), rows, locations)

    with pynwb.NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
        table = io.read().electrodes
        persisted = np.column_stack(
            [table[column][:] for column in ("rel_x", "rel_y", "rel_z")]
        )
    np.testing.assert_array_equal(
        persisted[rows], np.column_stack([locations, np.zeros(len(rows))])
    )
    # Rows the series does not reference stay NaN: the created columns record
    # "no geometry known here", not a contact sitting at the origin.
    untouched = [i for i in range(len(_SMALL_ELECTRODE_IDS)) if i not in rows]
    assert np.isnan(persisted[untouched]).all()
    # Created columns are double precision, so a coordinate that is not
    # exactly representable in float32 survives them unchanged.
    with h5py.File(path, "r") as handle:
        group = handle["/general/extracellular_ephys/electrodes"]
        for column in ("rel_x", "rel_y", "rel_z"):
            assert group[column].dtype == np.float64


# ---------------------------------------------------------------------------
# 4b. A parent whose rel_* columns are narrower than float64
# ---------------------------------------------------------------------------

# Coordinates chosen so that none of them is exactly representable in
# float32: 1000.1 lands on 1000.0999755859375 there (an error of 2.4e-5 um,
# 24x the writer's 1e-6 um read-back tolerance).
_NARROW_ELECTRODE_IDS = (3, 5, 9, 12)
_NARROW_LOCATIONS = np.array(
    [
        [1000.1, 0.0],
        [1000.1, -30.3],
        [1030.7, 0.0],
        [1030.7, -30.3],
    ]
)


def _write_narrow_geometry_nwb(path):
    """Write an NWB whose ``rel_x``/``rel_y`` are float32, ``rel_z`` float64.

    Real files carry mixed-precision ``rel_*`` columns, and
    ``AnalysisNwbfile().create`` exports the parent's electrodes table
    verbatim -- a pynwb export preserves each column's on-disk dtype -- so the
    analysis file the writer stamps inherits that float32 destination. An x-z
    sort group then projects its float64 ``rel_z`` into the float32 ``rel_y``.
    """
    import pynwb

    nwbfile = pynwb.NWBFile(
        session_description="mixed-precision rel_* columns",
        identifier="narrow-rel-columns",
        session_start_time=datetime(2024, 3, 1, tzinfo=timezone.utc),
    )
    device = nwbfile.create_device(name="probe0")
    group = nwbfile.create_electrode_group(
        name="0", description="g", location="CA1", device=device
    )
    for index, electrode_id in enumerate(_NARROW_ELECTRODE_IDS):
        nwbfile.add_electrode(
            id=int(electrode_id),
            location="CA1",
            group=group,
            rel_x=float(index),
            rel_y=0.0,
            rel_z=float(-index),
        )
    nwbfile.add_acquisition(
        pynwb.ecephys.ElectricalSeries(
            name="e-series",
            data=np.zeros((32, len(_NARROW_ELECTRODE_IDS)), dtype=np.int16),
            electrodes=nwbfile.create_electrode_table_region(
                region=list(range(len(_NARROW_ELECTRODE_IDS))),
                description="electrodes used in raw e-series recording",
            ),
            starting_time=0.0,
            rate=_SAMPLING_FREQUENCY,
            conversion=_GAIN_UV_PER_COUNT * 1e-6,
        )
    )
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)

    # pynwb writes every rel_* column float64; narrow two of them the way a
    # writer that declared float32 would have.
    with h5py.File(path, "a") as handle:
        group = handle["/general/extracellular_ephys/electrodes"]
        for column in ("rel_x", "rel_y"):
            dataset = group[column]
            values = dataset[:]
            attrs = dict(dataset.attrs)
            del group[column]
            narrowed = group.create_dataset(
                column, data=values.astype(np.float32), dtype=np.float32
            )
            for name, value in attrs.items():
                narrowed.attrs[name] = value
    return path


@pytest.mark.unit
def test_geometry_is_persisted_at_double_precision(tmp_path):
    """A float32 destination column must not truncate the sort's geometry.

    ``group[column][:] = values`` casts to the destination dtype, so an x-z
    group whose float64 ``rel_z`` is projected into a float32 ``rel_y`` lost
    2.4e-5 um per coordinate -- 24x the 1e-6 um the writer verifies against --
    and the read-back check failed the whole write on a perfectly valid file.
    The persisted geometry is what SpikeInterface rebuilds a reloaded
    recording's channel locations from, so it has to hold the coordinates the
    sort actually ran with, not a truncation of them.
    """
    import pynwb

    from spyglass.spikesorting.v2._recording_nwb import (
        _persist_channel_geometry,
        read_recording_nwb,
    )

    path = _write_narrow_geometry_nwb(tmp_path / "narrow_rel_columns.nwb")
    with h5py.File(path, "r") as handle:
        group = handle["/general/extracellular_ephys/electrodes"]
        assert group["rel_x"].dtype == np.float32
        assert group["rel_y"].dtype == np.float32
        assert group["rel_z"].dtype == np.float64
    assert not np.array_equal(
        _NARROW_LOCATIONS, _NARROW_LOCATIONS.astype(np.float32)
    ), "the fixture coordinates must not be float32-exact"

    rows = list(range(len(_NARROW_ELECTRODE_IDS)))
    _persist_channel_geometry(str(path), rows, _NARROW_LOCATIONS)

    with h5py.File(path, "r") as handle:
        group = handle["/general/extracellular_ephys/electrodes"]
        for column in ("rel_x", "rel_y", "rel_z"):
            assert group[column].dtype == np.float64
        persisted = np.column_stack(
            [group[column][:] for column in ("rel_x", "rel_y", "rel_z")]
        )
    np.testing.assert_allclose(
        persisted,
        np.column_stack([_NARROW_LOCATIONS, np.zeros(len(rows))]),
        rtol=0.0,
        atol=1e-6,
    )

    # The widened columns are still a readable DynamicTable: pynwb resolves
    # the table (and the series' region through it), and SpikeInterface
    # rebuilds the recording's locations from it.
    with pynwb.NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
        table = io.read().electrodes
        assert {"rel_x", "rel_y", "rel_z"}.issubset(table.colnames)
        np.testing.assert_array_equal(
            np.asarray(table["rel_x"][:], dtype=float), persisted[:, 0]
        )

    reloaded = read_recording_nwb(
        str(path), electrical_series_path="acquisition/e-series"
    )
    np.testing.assert_allclose(
        np.asarray(reloaded.get_channel_locations(), dtype=float),
        _NARROW_LOCATIONS,
        rtol=0.0,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 5. Unnormalized 3D geometry is refused, not projected
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.database
def test_write_refuses_unnormalized_3d_geometry(xz_probe_session):
    """A recording whose channel locations are still 3D is rejected before the
    write, rather than having SpikeInterface's default x-y projection
    persisted for it.

    ``get_channel_locations()`` defaults to ``axes="xy"``, so a 3D recording
    does not fail -- it silently loses z. For the x-z geometry these fixtures
    carry, that projection is exactly the contact collapse normalization
    exists to prevent, and the concatenated-recording writer has no upstream
    ``assert_unique_contact_positions`` to catch it. The refusal is what makes
    the writer safe for that second caller.
    """
    import spikeinterface.core as si_core

    from spyglass.settings import analysis_dir
    from spyglass.spikesorting.v2.recording import Recording

    channel_ids = list(_SMALL_GROUP_IDS)
    raw_geometry = _xz_geometry(_SMALL_CONTACTS)
    recording = si_core.NumpyRecording(
        [np.zeros((200, len(channel_ids)), dtype=np.int16)],
        sampling_frequency=_SAMPLING_FREQUENCY,
        channel_ids=channel_ids,
    )
    recording.set_channel_gains(_GAIN_UV_PER_COUNT)
    recording.set_channel_offsets(_OFFSET_UV)
    recording.set_property(
        "location", np.array([raw_geometry[eid] for eid in channel_ids])
    )
    assert recording.has_3d_locations()
    # Distinct in x-z, coincident in the x-y projection SI would take.
    assert len(np.unique(recording.get_channel_locations(), axis=0)) < len(
        channel_ids
    )

    nwb_file_name = xz_probe_session["nwb_file_name"]
    staged_dir = Path(analysis_dir) / Path(nwb_file_name).stem
    before = set(staged_dir.glob("*.nwb"))
    with pytest.raises(ValueError, match="still carries 3D channel locations"):
        Recording._write_nwb_artifact(
            recording,
            nwb_file_name,
            filtering_description="3D geometry probe",
        )
    assert (
        set(staged_dir.glob("*.nwb")) == before
    ), "the refused write must not leave a staged artifact behind"
