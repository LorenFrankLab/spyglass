"""Shared helpers for ingesting NWB files into the isolated test database.

Modern-spike-sorting fixture generation, the v1 baseline-capture script, and
the fixture round-trip test all need the same three-step pattern: copy the
NWB into the isolated raw directory, run ``insert_sessions``, and look up the
``Nwbfile`` row Spyglass created. Centralising it here keeps the deterministic
copy naming and the ``reinsert=True`` choice in one place.

This helper is intentionally version-tolerant on the ``reinsert`` kwarg
because ``baseline_capture.py`` is designed to run under master spyglass
(SI 0.99) where ``insert_sessions`` predates the ``reinsert`` parameter;
the v2 parity test then consumes the captured baseline under the current
spyglass + SI 0.104.
"""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import numpy as np


def copy_and_insert_nwb(
    nwb_source: Path | str, dest_name: str | None = None
) -> str:
    """Copy an NWB file into the test raw directory and ingest it.

    Parameters
    ----------
    nwb_source : pathlib.Path or str
        Source NWB file. Copied (not linked) into ``$SPYGLASS_RAW_DIR``.
    dest_name : str, optional
        Basename to copy the source to (and ingest under) instead of the
        source's own basename. Use this to ingest the same fixture under a
        DISTINCT ``nwb_file_name`` so that one module's session cleanup /
        ``reinsert`` cannot cascade-delete rows another fixture depends on.
        Must end in ``.nwb``.

    Returns
    -------
    str
        The Spyglass ``nwb_file_name``: the basename with the trailing-
        underscore copy suffix Spyglass uses
        (``get_nwb_copy_filename``).
    """
    from spyglass.data_import import insert_sessions
    from spyglass.settings import raw_dir
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    nwb_source = Path(nwb_source)
    ingest_name = dest_name or nwb_source.name
    raw_target = Path(raw_dir) / ingest_name
    if not raw_target.exists():
        shutil.copy(nwb_source, raw_target)
    kwargs = {"raise_err": True}
    if "reinsert" in inspect.signature(insert_sessions).parameters:
        kwargs["reinsert"] = True
    else:
        # Master spyglass's ``insert_sessions`` lacks the ``reinsert``
        # parameter and silently no-ops on a duplicate Nwbfile row.
        # Emulate the v2-branch reinsert path: drop any existing row
        # for the target name (cascades through downstream tables) so
        # ``populate_all_common`` actually runs.
        from spyglass.common.common_nwbfile import Nwbfile

        target_copy = get_nwb_copy_filename(ingest_name)
        (Nwbfile() & {"nwb_file_name": target_copy}).delete(safemode=False)
    insert_sessions(ingest_name, **kwargs)
    return get_nwb_copy_filename(ingest_name)


def clear_curations_for(sorting_key) -> None:
    """Delete every ``CurationV2`` row for a sorting plus its merge masters.

    DataJoint refuses to drop a part row whose master is still present, so
    walk from the ``SpikeSortingOutput`` merge master down before dropping
    the ``CurationV2`` rows. Shared single implementation for conftest's
    curation fixture and the v2 test modules (previously copied -- and
    drifted -- across several of them).

    Parameters
    ----------
    sorting_key : dict
        Restriction selecting the sorting whose curations to drop (e.g.
        ``{"sorting_id": ...}`` or a full curation PK).
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    for mid in (SpikeSortingOutput.CurationV2 & sorting_key).fetch("merge_id"):
        (SpikeSortingOutput & {"merge_id": mid}).super_delete(warn=False)
    (CurationV2 & sorting_key).super_delete(warn=False)


def configure_v2_run_inputs(
    nwb_file_name: str,
    team_name: str,
    *,
    interval_list_name: str = "raw data valid times",
    team_description: str = "",
) -> dict:
    """Seed defaults + LabTeam + sort group (no populate); return run inputs.

    Idempotent ensure-exists setup shared by the v2 pipeline test modules:
    inserts the default Lookup rows and the LabTeam, builds one sort group per
    shank if absent, and returns the ``run_v2_pipeline`` input dict keyed on
    the lowest ``sort_group_id``. Does NOT call ``populate``.

    Parameters
    ----------
    nwb_file_name : str
        Ingested session to configure.
    team_name : str
        LabTeam to ensure exists and own the sort.
    interval_list_name : str
        Interval to sort. Default ``"raw data valid times"``.
    team_description : str
        Description for the LabTeam row.

    Returns
    -------
    dict
        ``{nwb_file_name, sort_group_id, interval_list_name, team_name}``.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import SortGroupV2

    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": team_name, "team_description": team_description},
        skip_duplicates=True,
    )
    session_key = {"nwb_file_name": nwb_file_name}
    if not (SortGroupV2 & session_key):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & session_key).fetch("sort_group_id"))[0]
    )
    return {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": interval_list_name,
        "team_name": team_name,
    }


def _clean_session_v2(session_key):
    """Cascade-aware cleanup of every v2 row for a session.

    The v2 source-polymorphic source-part pattern leaves
    ``ArtifactDetectionSelection`` and ``SortingSelection`` MASTERS with no
    direct FK to upstream tables (only their ``RecordingSource`` PARTS
    carry the FK). DataJoint's cascade can't traverse that gap: a
    ``super_delete(SortGroupV2 & session_key)`` raises ``Attempt to
    delete part table ... before deleting from its master`` once the
    cascade reaches ``ArtifactDetectionSelection.RecordingSource`` because
    DataJoint refuses to drop a part without its master.

    Tests historically worked around this with an order-of-declaration
    convention (SortGroup tests run before tests that populate
    ArtifactDetectionSelection, so the cascade chain stays empty). Anything that
    runs the suite in a different order (``-k``, parallel sharding,
    rerun-failed) tripped the same DataJoint error. This helper makes
    the cleanup order-independent by walking the dependency graph
    leaves-first and deleting source-polymorphic masters explicitly
    before their upstream tables.

    Parameters
    ----------
    session_key
        Dict containing at least ``nwb_file_name``. All v2 rows tied to
        this session are dropped.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
        SharedArtifactGroup,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    # Step 1: drop any merge-master rows whose CurationV2 part points
    # to a sorting derived from this session. The merge insert wraps
    # the part FK in a transaction with the master, so cleanup must
    # take the master first to satisfy the master-before-part rule.
    rec_keys = (RecordingSelection & session_key).fetch("KEY", as_dict=True)
    if rec_keys:
        sorting_keys = (
            SortingSelection.RecordingSource
            & [{"recording_id": r["recording_id"]} for r in rec_keys]
        ).fetch("KEY", as_dict=True)
        if sorting_keys:
            merge_ids = (SpikeSortingOutput.CurationV2 & sorting_keys).fetch(
                "merge_id"
            )
            for mid in merge_ids:
                (SpikeSortingOutput & {"merge_id": mid}).super_delete(
                    warn=False
                )
            (CurationV2 & sorting_keys).super_delete(warn=False)
            (Sorting & sorting_keys).super_delete(warn=False)
            # Step 2: drop SortingSelection masters BEFORE removing
            # Recording, otherwise the Recording cascade tries to
            # delete the orphan RecordingSource part and fails.
            (SortingSelection & sorting_keys).super_delete(warn=False)

        # Step 3: same pattern for ArtifactDetection / ArtifactDetectionSelection.
        artifact_keys = (
            ArtifactDetectionSelection.RecordingSource
            & [{"recording_id": r["recording_id"]} for r in rec_keys]
        ).fetch("KEY", as_dict=True)
        if artifact_keys:
            (ArtifactDetection & artifact_keys).super_delete(warn=False)
            (ArtifactDetectionSelection & artifact_keys).super_delete(
                warn=False
            )

    # Step 4: SharedArtifactGroup tables. The master FK's Session
    # and the Member part FK's Recording, so prior shared-group
    # rows pointing at THIS session's Recording rows must be
    # cleaned before we drop Recording -- otherwise DJ refuses to
    # cascade through the part-without-master constraint. Delete
    # master + Member explicitly (master first satisfies the
    # master-before-part rule).
    shared_groups = (SharedArtifactGroup & session_key).fetch(
        "KEY", as_dict=True
    )
    if shared_groups:
        # force_masters=True: the cascade reaches the source-polymorphic
        # ``ArtifactDetectionSelection.SharedGroupSource`` part, whose master
        # is ``ArtifactDetectionSelection`` (not ``SharedArtifactGroup``); without it
        # DataJoint raises "delete part before master". Mirrors the other
        # master-before-part super_deletes here (the Recording/RecordingSource
        # analog).
        (SharedArtifactGroup & shared_groups).super_delete(
            warn=False, force_masters=True
        )

    # Step 5: now the cascade is unblocked -- Recording, SortGroupV2
    # can be deleted normally. We super_delete each so a leftover
    # IntervalList row from a prior aborted test is also picked up.
    (Recording & rec_keys).super_delete(warn=False) if rec_keys else None
    (RecordingSelection & session_key).super_delete(warn=False)
    (SortGroupV2 & session_key).super_delete(warn=False)


def _synthetic_artifact_recording():
    """8-channel, 90 000-sample (3 s @ 30 kHz) recording with two planted
    artifact runs -- one common-mode amplitude burst, one single-channel
    z-score outlier -- plus heterogeneous gains so the µV scaling matters.
    """
    import spikeinterface as si

    fs = 30_000.0
    n_samples = 90_000
    n_channels = 8
    rng = np.random.default_rng(0)
    # Small baseline noise that never trips either threshold.
    traces = rng.normal(0.0, 2.0, size=(n_samples, n_channels)).astype(
        np.float32
    )
    # Common-mode amplitude burst across all channels (trips amplitude).
    traces[20_000:20_300, :] += 300.0
    # Single-channel transient (trips the across-channel z-score).
    traces[60_000:60_120, 3] += 400.0
    rec = si.NumpyRecording(traces_list=[traces], sampling_frequency=fs)
    # Heterogeneous gains: the chunk worker and the reference must apply the
    # SAME per-channel gain, so a wrong gain broadcast surfaces as inequality.
    rec.set_channel_gains([1.0, 0.5, 2.0, 0.25, 1.0, 1.5, 0.8, 1.2])
    return rec


def clean_session_groups_for_owner(owner: str) -> None:
    """Cascade-delete every SessionGroup (and its concat lineage) for an owner.

    Walks the concat lineage leaves-first so the source-polymorphic
    ``SortingSelection``/merge masters are removed before their upstream
    tables, mirroring :func:`_clean_session_v2`. Idempotent: a missing row at
    any step is a no-op. Used by the chronic fixture's setup/teardown and by
    tests that build their own groups.

    Parameters
    ----------
    owner : str
        ``session_group_owner`` whose groups (and any concat recordings,
        concat-backed sortings, curations, and merge rows derived from them)
        are dropped.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
        SessionGroup,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    owner_key = {"session_group_owner": owner}
    concat_keys = (ConcatenatedRecordingSelection & owner_key).fetch(
        "KEY", as_dict=True
    )
    if concat_keys:
        concat_id_restr = [
            {"concat_recording_id": c["concat_recording_id"]}
            for c in concat_keys
        ]
        sort_keys = (
            SortingSelection.ConcatenatedRecordingSource & concat_id_restr
        ).fetch("KEY", as_dict=True)
        if sort_keys:
            for mid in (SpikeSortingOutput.CurationV2 & sort_keys).fetch(
                "merge_id"
            ):
                (SpikeSortingOutput & {"merge_id": mid}).super_delete(
                    warn=False
                )
            (CurationV2 & sort_keys).super_delete(warn=False)
            (Sorting & sort_keys).super_delete(warn=False)
            (SortingSelection & sort_keys).super_delete(warn=False)
        (ConcatenatedRecording & concat_id_restr).super_delete(warn=False)
        (ConcatenatedRecordingSelection & owner_key).super_delete(warn=False)
    (SessionGroup & owner_key).super_delete(warn=False)


def synthesize_minirec_nwb(
    out_path,
    *,
    session_start,
    fixture_name,
    seed,
    duration_s=5.0,
    sampling_frequency=30_000.0,
    num_units=3,
    firing_rates=15.0,
):
    """Write a short single-tetrode synthetic recording to a Spyglass NWB.

    Synthesizes one 4-channel SpikeInterface ground-truth recording and
    writes it into the same Frank-lab-style NWB layout the MEArec fixtures
    use (tetrode probe + electrodes + a single ``ElectricalSeries``), so the
    file ingests through ``insert_sessions`` like a real session. The probe
    geometry is the fixed ``tetrode_probe_layout()`` for EVERY call, so two
    files synthesized this way carry byte-identical channel positions and can
    be concatenated; only ``seed`` / ``firing_rates`` vary the planted spikes.

    Parameters
    ----------
    out_path : pathlib.Path or str
        Destination NWB path.
    session_start : datetime.datetime
        Timezone-aware session start time (the recording DATE the
        ``SessionGroup`` multi-day gate derives from ``Session``).
    fixture_name : str
        Recorded as the NWB ``session_id`` / identifier.
    seed : int
        Seed for the synthetic recording (different seeds -> different
        planted spikes).
    duration_s : float
        Recording duration in seconds. Default 5.0.
    sampling_frequency : float
        Sampling rate in Hz. Default 30 kHz (Frank-lab standard).
    num_units : int
        Planted ground-truth units. Default 3.
    firing_rates : float
        Mean firing rate (Hz) for the planted units. Default 15.0.

    Returns
    -------
    pathlib.Path
        ``out_path`` (written).
    """
    from pathlib import Path

    import pynwb
    from spikeinterface.core import generate_ground_truth_recording

    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        _add_probe_and_electrodes,
        _add_raw_ephys,
        _build_nwbfile,
        tetrode_probe_layout,
    )

    layout = tetrode_probe_layout()
    recording, _sorting = generate_ground_truth_recording(
        durations=[float(duration_s)],
        sampling_frequency=float(sampling_frequency),
        num_channels=layout.n_contacts,
        num_units=num_units,
        generate_sorting_kwargs={
            "firing_rates": firing_rates,
            "refractory_period_ms": 4.0,
        },
        seed=seed,
    )
    # Traces are returned in the recording's native microvolt scale; the NWB
    # ElectricalSeries carries ``conversion=1e-6`` (uV -> V), matching the
    # trodes_to_nwb convention the MEArec fixtures and Spyglass ingestion use.
    traces = recording.get_traces(return_in_uV=True)

    nwbfile = _build_nwbfile(
        fixture_name=fixture_name, session_start=session_start
    )
    _add_probe_and_electrodes(
        nwbfile, layout=layout, targeted_location="CA1"
    )
    _add_raw_ephys(
        nwbfile, traces=traces, sampling_frequency=float(sampling_frequency)
    )

    out_path = Path(out_path)
    with pynwb.NWBHDF5IO(str(out_path), mode="w") as io:
        io.write(nwbfile)
    return out_path


def write_two_eseries_nwb(
    out_path,
    *,
    first_rate=30_000.0,
    second_rate=25_000.0,
    n_channels=4,
    n_samples=20,
):
    """Write an NWB with TWO acquisition ``ElectricalSeries`` objects.

    The first series is rate-based (``starting_time`` + ``rate``) and the
    second carries an explicit ``timestamps`` vector, so a caller can assert
    that raw-source resolution returns the *matched* series' own path AND
    timestamp mode -- not whatever the acquisition iteration happens to yield
    first. Both series share one electrodes table; their per-series
    ``object_id`` values (the identifiers a ``Raw`` row would store) are
    returned so the caller can pin resolution to either one.

    Parameters
    ----------
    out_path : pathlib.Path or str
        Destination NWB path.
    first_rate, second_rate : float
        Sampling rates (Hz) for the rate-based first series and the
        explicit-timestamp second series. Distinct so a caller can tell which
        series was read from the resolved sampling rate.
    n_channels, n_samples : int
        Electrode count and per-series sample count.

    Returns
    -------
    dict
        ``{"first_series": (name, object_id), "second_series": (name,
        object_id)}`` -- the in-file acquisition name and NWB object id of
        each written ``ElectricalSeries``.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    import pynwb

    nwbfile = pynwb.NWBFile(
        session_description="two-ElectricalSeries raw-source fixture",
        identifier="two-eseries-fixture",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    device = nwbfile.create_device(name="probe0")
    electrode_group = nwbfile.create_electrode_group(
        name="0", description="test group", location="hpc", device=device
    )
    for eid in range(n_channels):
        nwbfile.add_electrode(id=eid, location="hpc", group=electrode_group)

    rng = np.random.default_rng(0)
    first = pynwb.ecephys.ElectricalSeries(
        name="first_series",
        data=rng.normal(size=(n_samples, n_channels)).astype("float32"),
        electrodes=nwbfile.create_electrode_table_region(
            region=list(range(n_channels)), description="first"
        ),
        starting_time=0.0,
        rate=float(first_rate),
    )
    second = pynwb.ecephys.ElectricalSeries(
        name="second_series",
        data=(
            rng.normal(size=(n_samples, n_channels)) + 1_000.0
        ).astype("float32"),
        electrodes=nwbfile.create_electrode_table_region(
            region=list(range(n_channels)), description="second"
        ),
        timestamps=np.arange(n_samples, dtype=float) / float(second_rate),
    )
    nwbfile.add_acquisition(first)
    nwbfile.add_acquisition(second)
    object_ids = {
        "first_series": ("first_series", first.object_id),
        "second_series": ("second_series", second.object_id),
    }

    out_path = Path(out_path)
    with pynwb.NWBHDF5IO(str(out_path), mode="w") as io:
        io.write(nwbfile)
    return object_ids


def _plant_concat_sorting_selection(sid):
    """Land a minimal concat-source ``SortingSelection`` (no ``RecordingSource``).

    ``insert_selection`` now accepts a concat source, but it requires a real,
    populated ``ConcatenatedRecording`` (the part's FK target). This bypass
    plants a ``ConcatenatedRecordingSource`` part pointing at a fake
    ``concat_recording_id`` via FK-checks-off, so source-dispatch tests that
    don't need a materialized concat recording (e.g. the concat brain-region
    raise, the key_source membership check) stay cheap. The caller cleans up
    ``SortingSelection & {sid}``.
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    SorterParameters.insert_default()
    SortingSelection.insert1(
        {
            "sorting_id": sid,
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        },
        allow_direct_insert=True,
    )
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        SortingSelection.ConcatenatedRecordingSource.insert1(
            {"sorting_id": sid, "concat_recording_id": uuid.uuid4()},
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")
