"""Shared helpers for ingesting NWB files into the isolated test database.

``copy_and_insert_nwb`` tolerates an ``insert_sessions`` without ``reinsert``
because ``baseline_capture.py`` runs under master spyglass (SI 0.99).
"""

from __future__ import annotations

import filecmp
import inspect
import os
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
        Basename to copy and ingest under instead of the source's basename.
        A distinct ``nwb_file_name`` keeps one module's cleanup or
        ``reinsert`` from cascade-deleting another fixture's rows. Must end
        in ``.nwb``.

    Returns
    -------
    str
        The Spyglass ``nwb_file_name`` (``get_nwb_copy_filename``).
    """
    from spyglass.data_import import insert_sessions
    from spyglass.settings import raw_dir
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    nwb_source = Path(nwb_source)
    ingest_name = dest_name or nwb_source.name
    raw_target = Path(raw_dir) / ingest_name
    # Close the path-keyed NWB cache, then replace (not rewrite) the copy:
    # HDF5 matches open files by inode and would reuse a stale handle.
    if not (raw_target.exists() and filecmp.cmp(nwb_source, raw_target)):
        from spyglass.utils.nwb_helper_fn import close_nwb_files

        close_nwb_files()
        staged = raw_target.with_name(raw_target.name + ".partial")
        shutil.copy2(nwb_source, staged)
        os.replace(staged, raw_target)
    kwargs = {"raise_err": True}
    if "reinsert" in inspect.signature(insert_sessions).parameters:
        kwargs["reinsert"] = True
    else:
        # Master spyglass no-ops on a duplicate Nwbfile row; delete it so
        # ``populate_all_common`` runs.
        from spyglass.common.common_nwbfile import Nwbfile

        target_copy = get_nwb_copy_filename(ingest_name)
        (Nwbfile() & {"nwb_file_name": target_copy}).delete(safemode=False)
    insert_sessions(ingest_name, **kwargs)
    return get_nwb_copy_filename(ingest_name)


def drop_unitmatch_selections_for(sorting_restriction) -> None:
    """Delete every ``UnitMatchSelection`` pinning a curation of these sorts.

    A selection's ``Input`` part references ``CurationV2``, and DataJoint
    refuses to cascade a curation delete into a part ahead of its master, so
    the selections go first. They reference their ``SessionGroup`` only as
    provenance, so deleting a group does not remove them.

    Parameters
    ----------
    sorting_restriction : dict or list of dict
        Restriction on ``sorting_id`` (extra curation fields narrow it).
    """
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    pinned = (UnitMatchSelection.Input & sorting_restriction).fetch(
        "unitmatch_id"
    )
    if len(pinned):
        (
            UnitMatchSelection & [{"unitmatch_id": u} for u in set(pinned)]
        ).super_delete(warn=False, safemode=False)


def clear_curations_for(sorting_key) -> None:
    """Delete every ``CurationV2`` row for a sorting plus its merge masters.

    Merge masters go first because DataJoint refuses to drop a part row
    whose master is still present.

    Parameters
    ----------
    sorting_key : dict
        Restriction selecting the sorting whose curations to drop (e.g.
        ``{"sorting_id": ...}`` or a full curation PK).
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )
    from spyglass.spikesorting.v2.curation import CurationV2

    drop_unitmatch_selections_for(sorting_key)
    for mid in (SpikeSortingOutput.ConcatMemberCuration & sorting_key).fetch(
        "merge_id"
    ):
        (SpikeSortingOutput & {"merge_id": mid}).super_delete(
            warn=False, safemode=False
        )
    (ConcatMemberCuration & sorting_key).super_delete(
        warn=False, safemode=False
    )
    for mid in (SpikeSortingOutput.CurationV2 & sorting_key).fetch("merge_id"):
        (SpikeSortingOutput & {"merge_id": mid}).super_delete(
            warn=False, safemode=False
        )
    (CurationV2 & sorting_key).super_delete(warn=False, safemode=False)


def configure_v2_run_inputs(
    nwb_file_name: str,
    team_name: str,
    *,
    interval_list_name: str = "raw data valid times",
    team_description: str = "",
) -> dict:
    """Seed defaults + LabTeam + sort group (no populate); return run inputs.

    Idempotent. Builds one sort group per shank if absent and keys the
    returned ``run_v2_pipeline`` inputs on the lowest ``sort_group_id``.

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

    Deletes leaves-first, removing source-polymorphic masters
    (``SortingSelection``, merge tables) before their upstream tables. A plain
    ``super_delete(SortGroupV2 & session_key)`` fails because DataJoint
    refuses to drop a source or merge part before its master.

    Parameters
    ----------
    session_key
        Dict containing at least ``nwb_file_name``. All v2 rows tied to
        this session are dropped.
    """
    from spyglass.common import IntervalList
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
        SharedArtifactGroup,
        SharedGroupArtifactDetection,
        SharedGroupArtifactSelection,
    )
    from spyglass.spikesorting.v2.artifact_output import (
        ArtifactDetectionOutput,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    # Step 1: sorting-derived rows, then SortingSelection masters (before
    # Recording, per master-before-part).
    rec_keys = (RecordingSelection & session_key).fetch("KEY", as_dict=True)
    split_artifact_keys = (
        (
            RecordingArtifactSelection
            & [{"recording_id": r["recording_id"]} for r in rec_keys]
        ).fetch("KEY", as_dict=True)
        if rec_keys
        else []
    )
    shared_groups = (SharedArtifactGroup & session_key).fetch(
        "KEY", as_dict=True
    )
    shared_artifact_keys = (
        (SharedGroupArtifactSelection & shared_groups).fetch(
            "KEY", as_dict=True
        )
        if shared_groups
        else []
    )
    # FreeTable cascades bypass the detections' delete override, so collect
    # owned IntervalList keys while the ownership parts exist.
    owned_intervals = []
    for owner, keys in (
        (RecordingArtifactDetection.RemovedInterval, split_artifact_keys),
        (SharedGroupArtifactDetection.RemovedInterval, shared_artifact_keys),
    ):
        if keys:
            owned_intervals.extend(
                (IntervalList & (owner & keys)).fetch("KEY", as_dict=True)
            )
    if rec_keys:
        sorting_keys = (
            SortingSelection.RecordingSource
            & [{"recording_id": r["recording_id"]} for r in rec_keys]
        ).fetch("KEY", as_dict=True)
        if sorting_keys:
            drop_unitmatch_selections_for(sorting_keys)
            merge_ids = (SpikeSortingOutput.CurationV2 & sorting_keys).fetch(
                "merge_id"
            )
            for mid in merge_ids:
                (SpikeSortingOutput & {"merge_id": mid}).super_delete(
                    warn=False
                )
            (CurationV2 & sorting_keys).super_delete(warn=False)
            (Sorting & sorting_keys).super_delete(warn=False)
            # Step 2: masters before Recording, whose cascade would hit
            # the orphan RecordingSource part.
            (SortingSelection & sorting_keys).super_delete(warn=False)

        # Step 3: artifact merge master first, so the Recording cascade
        # never reaches its RecordingSource part.
        if split_artifact_keys:
            art_merge_ids = (
                ArtifactDetectionOutput.RecordingSource & split_artifact_keys
            ).fetch("merge_id")
            if len(art_merge_ids):
                (
                    ArtifactDetectionOutput
                    & [{"merge_id": m} for m in art_merge_ids]
                ).super_delete(warn=False, force_masters=True)
            (RecordingArtifactDetection & split_artifact_keys).super_delete(
                warn=False
            )
            (RecordingArtifactSelection & split_artifact_keys).super_delete(
                warn=False
            )

    # Step 4: SharedArtifactGroup before Recording (its Member part FKs
    # Recording).
    if shared_groups:
        # force_masters: the cascade reaches parts of other masters.
        (SharedArtifactGroup & shared_groups).super_delete(
            warn=False, force_masters=True
        )

    # Step 5: Recording and SortGroupV2, then abandoned artifact intervals.
    (Recording & rec_keys).super_delete(warn=False) if rec_keys else None
    (RecordingSelection & session_key).super_delete(warn=False)
    (SortGroupV2 & session_key).super_delete(warn=False)

    # Keep intervals another detection still owns; delete_quick never
    # cascades into surviving rows.
    for interval_key in owned_intervals:
        if (RecordingArtifactDetection.RemovedInterval & interval_key) or (
            SharedGroupArtifactDetection.RemovedInterval & interval_key
        ):
            continue
        (IntervalList & interval_key).delete_quick()


def _synthetic_artifact_recording(offsets=None):
    """8-channel, 3 s @ 30 kHz recording with three planted artifact bursts.

    - common-mode amplitude burst on all channels
    - single-channel burst that trips amplitude and z-score at any gain
    - single-channel burst over a 50 µV threshold in raw counts but not
      after gain (180 raw * 0.25 µV/count = 45 µV)

    Per-channel gains differ, so a wrong gain broadcast shows up as a
    mismatch between the chunked scan and an in-memory reference. Both
    ``gain_to_uV`` and ``offset_to_uV`` are set, as the SI NWB reader does;
    SI's ``return_in_uV=True`` silently skips scaling if either is missing.

    Parameters
    ----------
    offsets : list[float] or None, optional
        Per-channel ``offset_to_uV`` values applied via
        ``set_channel_offsets``. Default ``None`` uses all-zero offsets.
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
    # Single-channel transient, large enough to trip amplitude AND z-score
    # regardless of gain (channel 3, gain 0.25 -> 100 µV peak).
    traces[60_000:60_120, 3] += 400.0
    # 180 raw * 0.25 µV/count = 45 µV: under a 50 µV threshold only if gain
    # is applied.
    traces[75_000:75_050, 3] += 180.0
    rec = si.NumpyRecording(traces_list=[traces], sampling_frequency=fs)
    # Heterogeneous gains expose a wrong gain broadcast.
    rec.set_channel_gains([1.0, 0.5, 2.0, 0.25, 1.0, 1.5, 0.8, 1.2])
    if offsets is None:
        offsets = [0.0] * n_channels
    rec.set_channel_offsets(offsets)
    return rec


def clean_session_groups_for_owner(owner: str) -> None:
    """Cascade-delete every SessionGroup (and its concat lineage) for an owner.

    Walks the concat lineage leaves-first, like :func:`_clean_session_v2`.
    Idempotent.

    Parameters
    ----------
    owner : str
        ``session_group_owner`` whose groups (and any concat recordings,
        concat-backed sortings, curations, and merge rows derived from them)
        are dropped.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )
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
            drop_unitmatch_selections_for(sort_keys)
            for mid in (
                SpikeSortingOutput.ConcatMemberCuration & sort_keys
            ).fetch("merge_id"):
                (SpikeSortingOutput & {"merge_id": mid}).super_delete(
                    warn=False, safemode=False
                )
            (ConcatMemberCuration & sort_keys).super_delete(
                warn=False, safemode=False
            )
            for mid in (SpikeSortingOutput.CurationV2 & sort_keys).fetch(
                "merge_id"
            ):
                (SpikeSortingOutput & {"merge_id": mid}).super_delete(
                    warn=False, safemode=False
                )
            (CurationV2 & sort_keys).super_delete(warn=False, safemode=False)
            (Sorting & sort_keys).super_delete(warn=False, safemode=False)
            # Drop SortingSelection masters BEFORE ConcatenatedRecording so its
            # cascade never hits the orphan ConcatenatedRecordingSource part.
            (SortingSelection & sort_keys).super_delete(
                warn=False, safemode=False
            )
        (ConcatenatedRecording & concat_id_restr).super_delete(
            warn=False, safemode=False
        )
        (ConcatenatedRecordingSelection & owner_key).super_delete(
            warn=False, safemode=False
        )
    (SessionGroup & owner_key).super_delete(warn=False, safemode=False)


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

    Uses the MEArec fixtures' NWB layout so it ingests like a real session.
    Every call uses the same ``tetrode_probe_layout()``, so outputs can be
    concatenated; only ``seed`` and ``firing_rates`` vary the spikes.

    Parameters
    ----------
    out_path : pathlib.Path or str
        Destination NWB path.
    session_start : datetime.datetime
        Timezone-aware session start time (the recording DATE the
        ``SessionGroup`` same-day check derives from ``Session``).
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

    from tests.spikesorting.v2.fixtures.mearec_to_nwb import (
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
    # Traces are in µV; ``conversion=1e-6`` follows trodes_to_nwb.
    traces = recording.get_traces(return_in_uV=True)

    nwbfile = _build_nwbfile(
        fixture_name=fixture_name, session_start=session_start
    )
    _add_probe_and_electrodes(nwbfile, layout=layout, targeted_location="CA1")
    _add_raw_ephys(
        nwbfile, traces=traces, sampling_frequency=float(sampling_frequency)
    )

    out_path = Path(out_path)
    with pynwb.NWBHDF5IO(str(out_path), mode="w") as io:
        io.write(nwbfile)
    return out_path


def zero_raw_electrode_geometry(nwb_path):
    """Zero the electrodes table's ``rel_x``/``rel_y``/``rel_z`` in place.

    Mimics Frank-lab sessions with no contact positions, so every contact
    collapses to one point until ``maybe_apply_tetrode_geometry`` runs.
    Probe metadata is left alone: ``Probe.Electrode`` is keyed by probe type,
    so a zeroed ``tetrode_12.5`` would collide with the canonical one.

    Parameters
    ----------
    nwb_path : pathlib.Path or str
        NWB file to edit IN PLACE.

    Returns
    -------
    pathlib.Path
        ``nwb_path``.
    """
    import h5py

    nwb_path = Path(nwb_path)
    with h5py.File(str(nwb_path), "a") as h5:
        electrodes = h5["general/extracellular_ephys/electrodes"]
        for column in ("rel_x", "rel_y", "rel_z"):
            electrodes[column][:] = 0.0
    return nwb_path


def inject_raw_nan(
    nwb_path,
    *,
    frame_index: int,
    channel_index: int,
    series_name: str = "e-series",
):
    """Overwrite one raw ``ElectricalSeries`` sample with NaN, in place.

    The raw dtype must be float; callers check first, since an integer
    dtype silently wraps NaN.

    Parameters
    ----------
    nwb_path : pathlib.Path or str
        NWB file to edit IN PLACE. Must not be registered as an ``Nwbfile``
        row yet; edit a copy, then ingest it.
    frame_index : int
        Row (sample) index into ``data`` to corrupt.
    channel_index : int
        Column (channel) index into ``data`` to corrupt.
    series_name : str
        Acquisition ``ElectricalSeries`` name. Default ``"e-series"``, as
        written by ``fixtures/mearec_to_nwb.py``.

    Returns
    -------
    pathlib.Path
        ``nwb_path``.
    """
    import h5py

    nwb_path = Path(nwb_path)
    with h5py.File(str(nwb_path), "a") as h5:
        data = h5[f"acquisition/{series_name}/data"]
        if not np.issubdtype(data.dtype, np.floating):
            raise ValueError(
                "inject_raw_nan: acquisition "
                f"'{series_name}' data dtype is {data.dtype}, which cannot "
                "represent NaN."
            )
        data[frame_index, channel_index] = np.nan
    return nwb_path


def rename_probe_type(nwb_path, new_probe_type: str):
    """Relabel every ``ndx_franklab_novela.Probe`` device in place.

    ``Probe`` is keyed by probe type, so a second ``tetrode_12.5`` geometry
    raises ``DuplicateError``. Relabelling gives the copy private device rows
    but skips the ``tetrode_12.5`` geometry repair.

    Parameters
    ----------
    nwb_path : pathlib.Path or str
        NWB file to edit IN PLACE.
    new_probe_type : str
        Replacement ``probe_type`` (also becomes the Spyglass ``probe_id``).

    Returns
    -------
    pathlib.Path
        ``nwb_path``.
    """
    import h5py

    nwb_path = Path(nwb_path)
    with h5py.File(str(nwb_path), "a") as h5:

        def _rename(_name, obj):
            if obj.attrs.get("neurodata_type") == "Probe":
                obj.attrs["probe_type"] = new_probe_type

        h5["general/devices"].visititems(_rename)
    return nwb_path


def write_two_eseries_nwb(
    out_path,
    *,
    first_rate=30_000.0,
    second_rate=25_000.0,
    n_channels=4,
    n_samples=20,
):
    """Write an NWB with TWO acquisition ``ElectricalSeries`` objects.

    The first series is rate-based and the second has explicit
    ``timestamps``, so callers can check that resolution returns the matched
    series' path and timestamp mode, not the first one iterated.

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
        object_id)}``: acquisition name and NWB object id per series.
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
        data=(rng.normal(size=(n_samples, n_channels)) + 1_000.0).astype(
            "float32"
        ),
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


def write_processed_recording_nwb(
    out_path,
    *,
    traces,
    timestamps,
    rel_positions,
    conversion=1e-6,
    offset=0.0,
    filtering="bandpass filter 300-6000 Hz",
    channel_ids=None,
    series_name="ProcessedElectricalSeries",
):
    """Write a minimal analysis-style NWB with one preprocessed ElectricalSeries.

    Mirrors what ``write_nwb_artifact`` persists, so the content fingerprint
    can be tested without a database.

    Parameters
    ----------
    out_path : pathlib.Path or str
        Destination NWB path.
    traces : array-like, shape (n_frames, n_channels)
        Unscaled trace samples written to the ElectricalSeries ``data``.
    timestamps : array-like, shape (n_frames,)
        Explicit wall-clock timestamps (seconds).
    rel_positions : array-like, shape (n_channels, 2)
        Per-channel ``(rel_x, rel_y)`` probe geometry (µm).
    conversion, offset : float
        ElectricalSeries volts-per-count and volts offset.
    filtering : str
        ElectricalSeries ``filtering`` provenance string.
    channel_ids : sequence of int, optional
        Electrode ids in row order; defaults to ``range(n_channels)``.
    series_name : str
        Acquisition ElectricalSeries name.

    Returns
    -------
    tuple
        ``(out_path, electrical_series_path)`` where the path is
        ``f"acquisition/{series_name}"``.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    import pynwb

    traces = np.asarray(traces)
    timestamps = np.asarray(timestamps, dtype=float)
    rel_positions = np.asarray(rel_positions, dtype=float)
    n_frames, n_channels = traces.shape

    nwbfile = pynwb.NWBFile(
        session_description="processed-recording fingerprint fixture",
        identifier="processed-recording-fixture",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    device = nwbfile.create_device(name="probe0")
    electrode_group = nwbfile.create_electrode_group(
        name="0", description="test group", location="hpc", device=device
    )
    ids = (
        list(channel_ids)
        if channel_ids is not None
        else list(range(n_channels))
    )
    for i, eid in enumerate(ids):
        nwbfile.add_electrode(
            id=int(eid),
            location="hpc",
            group=electrode_group,
            rel_x=float(rel_positions[i, 0]),
            rel_y=float(rel_positions[i, 1]),
        )
    region = nwbfile.create_electrode_table_region(
        region=list(range(n_channels)), description="all"
    )
    series = pynwb.ecephys.ElectricalSeries(
        name=series_name,
        data=traces,
        electrodes=region,
        timestamps=timestamps,
        conversion=float(conversion),
        offset=float(offset),
        filtering=filtering,
    )
    nwbfile.add_acquisition(series)

    out_path = Path(out_path)
    with pynwb.NWBHDF5IO(str(out_path), mode="w") as io:
        io.write(nwbfile)
    return out_path, f"acquisition/{series_name}"


def _plant_concat_sorting_selection(sid):
    """Land a minimal concat-source ``SortingSelection`` via FK-checks-off.

    Points the source part at a fake ``concat_recording_id`` so
    source-dispatch tests skip building a real ``ConcatenatedRecording``.
    The caller cleans up ``SortingSelection & {sid}``.
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    SorterParameters.insert_default()
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        # dj.Table.insert bypasses SelectionMasterInsertGuard.
        dj.Table.insert(
            SortingSelection(),
            [
                {
                    "sorting_id": sid,
                    "sorter": "clusterless_thresholder",
                    "sorter_params_name": "default",
                }
            ],
        )
        dj.Table.insert(
            SortingSelection.ConcatenatedRecordingSource(),
            [{"sorting_id": sid, "concat_recording_id": uuid.uuid4()}],
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")


def load_effective_recording(traces, **mask_kwargs):
    """Open a sort's effective traces by analysis file name (no self-heal).

    ``mask_kwargs`` (``artifact_valid_times`` / ``artifact_detection_id`` /
    ``recording_id``) pass through to ``read_effective_recording``.
    """
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._recording.source import (
        read_effective_recording,
    )

    return read_effective_recording(
        AnalysisNwbfile.get_abs_path(traces.row["analysis_file_name"]),
        traces,
        **mask_kwargs,
    )


def build_reference_merged_analyzer(
    curation_key, waveform_recipe: str, role: str, *, analyzer_folder
):
    """Build a committed merged curation's analyzer from scratch.

    Reference for the published analyzer, written to ``analyzer_folder``
    outside the analyzer cache.
    """
    from spyglass.spikesorting.v2._curation import analyzer as resolver
    from spyglass.spikesorting.v2._sorting.analyzer import (
        reconstruct_recording_and_sorting,
    )
    from spyglass.spikesorting.v2._sorting.fetch import (
        fetch_sorter_analyzer_inputs,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    row = resolver._resolve_curation_row(curation_key)
    sorting_key = {"sorting_id": row["sorting_id"]}
    recording, _ = reconstruct_recording_and_sorting(Sorting(), sorting_key)
    sorter_row, job_kwargs = fetch_sorter_analyzer_inputs(sorting_key)
    return resolver._build_resolved_merged_analyzer(
        CurationV2.get_merged_sorting(
            {**sorting_key, "curation_id": int(row["curation_id"])}
        ),
        recording,
        sorting_id=row["sorting_id"],
        sorter_row=sorter_row,
        job_kwargs=job_kwargs,
        waveform_params=dict(
            resolver._resolve_recipe(waveform_recipe, role)["params"]
        ),
        statistics_spans=Sorting().get_statistics_spans(sorting_key),
        role=role,
        analyzer_folder=analyzer_folder,
    )


def matched_unit_ids(curation, policy="v2_unflagged_single_units"):
    """Unit ids a matching run over ``curation`` includes under ``policy``.

    ``curation`` is anything ``resolve_unit_selection`` accepts.
    """
    import numpy as np

    from spyglass.spikesorting.v2.analysis_selection import (
        ensure_v2_unit_selection_policies,
        resolve_unit_selection,
    )

    ensure_v2_unit_selection_policies()
    return np.asarray(
        resolve_unit_selection(curation, policy=policy).included_unit_ids,
        dtype=int,
    )
