"""NWB artifact I/O behind ``Recording``.

``read_recording_nwb`` opens an explicit series with lazy timestamp access,
including when workers or analyzers reconstruct the extractor.
``compute_recording_artifact`` runs the preprocessing pipeline from the raw
NWB to a staged artifact (staging through ``Recording._write_nwb_artifact``),
and ``recording_provenance_table`` builds the source-lineage table it embeds.
``rebuild_nwb_artifact`` regenerates a missing artifact under its lock and
installs it only if the rebuild reproduces the stored content fingerprint;
``rebuild_concat_nwb_artifact`` does the same for a ``ConcatenatedRecording``
artifact, re-running that table's ``make_fetch`` and ``make_compute``, and
``rebuild_motion_corrected_artifact`` for a ``MotionCorrectedRecording`` by
reapplying its saved motion.
``write_nwb_artifact`` streams the preprocessed traces and the wall-clock
timestamps vector into an ``AnalysisNwbfile`` for ``Recording.make_compute``
(and the rebuild path), then hashes the persisted file for the cache contract.
``install_rebuilt_recording`` installs verified single/concatenated recording
rebuilds and reconciles their tracked byte checksums, and
``ensure_artifact_file`` is the shared self-heal that triggers them.
The table threads already-fetched DB state in (the tri-part
``make_fetch``/``make_compute``/``make_insert`` contract forbids DB I/O inside
compute), and the file row is registered by the caller inside its DataJoint
transaction, so this write path stays a thin file-write service.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import. SpikeInterface and NWB I/O dependencies are imported
inside the functions that use them. ``write_nwb_artifact``
touches the DB / DataJoint at CALL time via lazy imports (``AnalysisNwbfile``
path resolution + file create); ``rebuild_nwb_artifact`` also reads the
``Recording`` row and re-runs its ``make_fetch``,
``rebuild_concat_nwb_artifact`` and ``rebuild_motion_corrected_artifact`` read
the ``ConcatenatedRecording`` / ``MotionCorrectedRecording`` row and re-run its
``make_fetch`` / ``make_compute``, and
``clear_recompute_deleted_flag`` updates ``RecordingArtifactRecompute``.
Recording carrier types and staged-file cleanup live in DB-free modules.
The writer receives its dependencies explicitly and never imports its
owning table module.

STAGING IS THE ONE DB ACCESS A TRI-PART ``make_compute`` MAY KEEP. Its inputs
belong in ``make_fetch``, whose result DataJoint re-checks inside the insert
transaction; a cached trace file is resolved there with :func:`stored_traces`
(rebuilt if missing) and read in compute with :func:`read_stored_traces`.
Staging an output file still goes through ``AnalysisNwbfile().create`` and
``AnalysisNwbfile.get_abs_path``, which read the ``Nwbfile`` /
``AnalysisNwbfile`` tables to mint and locate the file: in
``write_nwb_artifact`` here, in ``_units_nwb.write_sorting_units_nwb``, and in
the ``UnitMatch`` and ``CurationEvaluation`` writers. This is accepted (a
DB-free ``create`` would be a ``spyglass.common`` change); the staged file is
registered only in ``make_insert``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, NamedTuple

if TYPE_CHECKING:
    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )

from spyglass.spikesorting.v2._recording.types import RecordingArtifactResult
from spyglass.spikesorting.v2._storage.staged_outputs import (
    unlink_staged_analysis_file as _unlink_staged_analysis_file,
)

_ELECTRICAL_SERIES_NAME = "ProcessedElectricalSeries"
_ELECTRICAL_SERIES_PATH = f"acquisition/{_ELECTRICAL_SERIES_NAME}"


class StoredTraces(NamedTuple):
    """A cached trace artifact resolved for a read that needs no DB.

    A tri-part ``make_fetch`` builds it with :func:`stored_traces` (after the
    self-heal) and ``make_compute`` opens it with :func:`read_stored_traces`.
    Strings only, so DataJoint's hash of the fetched inputs is the same on
    both of its fetches; ``content_hash`` puts the row's content under that
    check.

    Attributes
    ----------
    abs_path : str
        Absolute path of the artifact's analysis NWB (present on disk).
    electrical_series_path : str
        The row's in-file path of the persisted ``ElectricalSeries``.
    content_hash : str
        The row's content fingerprint.
    """

    abs_path: str
    electrical_series_path: str
    content_hash: str


def read_recording_nwb(
    path, *, electrical_series_path: str, load_time_vector: bool = True
):
    """Read an explicit NWB series without materializing its timestamps.

    SI 0.104.3's default HDF5 backend reads the entire timestamp dataset to
    estimate the sampling rate. Its PyNWB backend reads only a short prefix
    and retains the dataset for lazy time access. Preserve that choice in
    worker/analyzer serialization too: this SI version omits ``use_pynwb``
    from the extractor's reconstruction kwargs.
    """
    import spikeinterface.extractors as se

    recording = se.read_nwb_recording(
        str(path),
        electrical_series_path=electrical_series_path,
        load_time_vector=load_time_vector,
        use_pynwb=True,
    )
    from spyglass.spikesorting.v2._storage.spikeinterface import (
        retain_pynwb_reader,
    )

    retain_pynwb_reader(recording)
    return recording


def install_rebuilt_recording(
    temp_abs: str, canonical_abs: str, analysis_file_name: str
) -> None:
    """Install a verified rebuild and refresh its tracked byte checksum.

    The caller holds the recording's artifact lock and has checked that the
    temp's content fingerprint matches the stored recording. Replace the file
    atomically, refresh its checksum, then verify that it resolves. On failure,
    remove the temp or installed file so the next read can retry the rebuild.
    """
    import os
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile

    try:
        os.replace(temp_abs, canonical_abs)
    except Exception:
        Path(temp_abs).unlink(missing_ok=True)
        raise
    try:
        AnalysisNwbfile()._resolve_external(analysis_file_name)
        AnalysisNwbfile.get_abs_path(analysis_file_name)
    except Exception:
        Path(canonical_abs).unlink(missing_ok=True)
        raise


#: ``{analysis_file_name: (abs_path, file_identity)}`` for each cached trace
#: artifact this process resolved through ``AnalysisNwbfile.get_abs_path``
#: (which checksums the file) outside a transaction; ``file_identity`` is
#: :func:`_file_identity` at that time. See :func:`ensure_artifact_file`.
_VERIFIED_ARTIFACT_PATHS: dict[str, tuple[str, tuple[int, int, int]]] = {}


def _file_identity(abs_path: str) -> tuple[int, int, int] | None:
    """``(inode, size, mtime_ns)`` of a file, or ``None`` if it is absent."""
    import os

    try:
        stat = os.stat(abs_path)
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def ensure_artifact_file(table, key: dict, analysis_file_name: str) -> str:
    """Absolute path of a cached trace artifact, rebuilt first if missing.

    The one self-heal every trace-artifact table shares: when the file is
    gone, ``table()._rebuild_nwb_artifact(key)`` restores it (a locked,
    content-verified rebuild; the DataJoint row is never deleted), and the
    path is resolved again.

    ``AnalysisNwbfile.get_abs_path`` checksums the whole file, about 1.3 s per
    GiB (measured on a 1 GiB file with a warm page cache, dominated by
    DataJoint's ``uuid_from_file``). DataJoint runs a tri-part ``make_fetch``
    twice, the second time inside the insert transaction, so that checksum
    would run twice per populate, once while the transaction is open. A
    resolution outside a transaction always goes through ``get_abs_path``.
    Inside a transaction, a file this process already resolved that way and
    whose inode, size and modification time are unchanged reuses that result:
    in a populate, that is the first ``make_fetch``'s check of the same file.
    A rebuilt or rewritten file differs and is checked again.

    Parameters
    ----------
    table : type
        The owning table class (``Recording``, ``ConcatenatedRecording`` or
        ``MotionCorrectedRecording``); it must define
        ``_rebuild_nwb_artifact(key)``.
    key : dict
        The artifact row's primary key.
    analysis_file_name : str
        The row's ``analysis_file_name``.

    Returns
    -------
    str
        Absolute path of the (present) artifact file.
    """
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile

    in_transaction = AnalysisNwbfile().connection.in_transaction
    if in_transaction and analysis_file_name in _VERIFIED_ARTIFACT_PATHS:
        abs_path, identity = _VERIFIED_ARTIFACT_PATHS[analysis_file_name]
        if _file_identity(abs_path) == identity:
            return abs_path

    abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
    if not Path(abs_path).exists():
        table()._rebuild_nwb_artifact(key)
        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
    identity = _file_identity(abs_path)
    if not in_transaction and identity is not None:
        _VERIFIED_ARTIFACT_PATHS[analysis_file_name] = (abs_path, identity)
    return abs_path


def stored_traces(table, key: dict, row: dict) -> StoredTraces:
    """Self-heal a cached trace artifact and resolve it for a DB-free read.

    Idempotent: a missing file is rebuilt on the first call, so a second call
    (DataJoint's in-transaction re-fetch) finds it and returns equal values.

    Parameters
    ----------
    table : type
        The owning table class; see :func:`ensure_artifact_file`.
    key : dict
        The artifact row's primary key.
    row : dict
        The artifact row (``analysis_file_name``, ``electrical_series_path``,
        ``content_hash``).

    Returns
    -------
    StoredTraces
    """
    return StoredTraces(
        abs_path=ensure_artifact_file(table, key, row["analysis_file_name"]),
        electrical_series_path=row["electrical_series_path"],
        content_hash=row["content_hash"],
    )


def open_persisted_traces(abs_path: str, electrical_series_path: str):
    """Open a persisted trace artifact's ``ElectricalSeries``; no DB access.

    Reads the stored ``electrical_series_path`` (authoritative, not an
    auto-detect hint) and annotates ``is_filtered=True``: the persisted traces
    are already bandpass-filtered and referenced, so a downstream
    SpikeInterface consumer must not filter them again.

    Parameters
    ----------
    abs_path : str
        Absolute path of the artifact's analysis NWB.
    electrical_series_path : str
        The row's stored ``electrical_series_path``.

    Returns
    -------
    si.BaseRecording
    """
    recording = read_recording_nwb(
        abs_path, electrical_series_path=electrical_series_path
    )
    recording.annotate(is_filtered=True)
    return recording


def read_stored_traces(traces: StoredTraces):
    """Open resolved stored traces with :func:`open_persisted_traces`.

    Parameters
    ----------
    traces : StoredTraces

    Returns
    -------
    si.BaseRecording
    """
    return open_persisted_traces(traces.abs_path, traces.electrical_series_path)


def _remove_partial_artifact(
    analysis_file_name: str, existing_analysis_file_name: str | None
) -> None:
    """Best-effort cleanup of a partial artifact after a failed write.

    NEVER unlink a canonical artifact. An in-place rebuild
    (``existing_analysis_file_name`` set) writes to the *canonical* slot, so
    deleting it on failure would destroy the very artifact being rebuilt; it is
    left in place. The production rebuild temp-stages
    (``existing_analysis_file_name is None``), so only that freshly created temp
    file is removed. The unlink is best-effort -- a cleanup failure is logged,
    not raised, so it cannot mask the original error.
    """
    from spyglass.spikesorting.v2._storage.staged_outputs import (
        unlink_staged_analysis_file,
    )

    unlink_staged_analysis_file(
        analysis_file_name,
        context="Recording._write_nwb_artifact",
        existing_analysis_file_name=existing_analysis_file_name,
    )


# Probe-relative contact position columns of the NWB electrodes table. These
# are what SpikeInterface rebuilds a reloaded recording's channel locations
# from (``NwbRecordingExtractor._fetch_locations_and_groups``), and what the
# content fingerprint hashes as the artifact's geometry component.
_RELATIVE_POSITION_COLUMNS = ("rel_x", "rel_y", "rel_z")

# Contact positions are micrometres. 1e-6 um is far below any real contact
# pitch and matches the rounding the geometry helpers in
# ``_recording_geometry`` use to decide "same contact", so it is the tolerance
# for the persisted-versus-requested read-back.
_POSITION_TOLERANCE_UM = 1e-6


def _ensure_relative_position_columns(nwbfile) -> None:
    """Give the analysis file's electrodes table its ``rel_*`` columns.

    ``AnalysisNwbfile().create`` exports the parent NWB's electrodes table
    verbatim, so a parent written without probe-relative contact positions
    yields an analysis file with nowhere to put the normalized geometry.
    hdmf 4.3 accepts a new column on a table that was already written (the
    file is open in append mode), so the missing columns are created here
    BEFORE ``io.write``. The post-write pass then fills the rows the
    ElectricalSeries actually references; every other row is left ``NaN``,
    which records "no geometry known for this contact" rather than asserting
    a contact at the origin.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        The analysis file, opened in append mode and not yet written.
    """
    electrodes = nwbfile.electrodes
    n_rows = len(electrodes.id)
    for column in _RELATIVE_POSITION_COLUMNS:
        if column in electrodes.colnames:
            continue
        electrodes.add_column(
            name=column,
            description=(
                f"the {column[-1]} coordinate of this contact relative to "
                "the probe, in micrometers"
            ),
            data=[float("nan")] * n_rows,
        )


def _widen_position_column_to_float64(group, column) -> None:
    """Recreate a narrower-than-float64 ``rel_*`` dataset as float64.

    ``group[column][:] = values`` casts to the DESTINATION dtype, so a float32
    column silently truncates the geometry it is handed (1000.1 um is stored
    as 1000.0999755859375 -- an error of 2.4e-5 um, far outside the 1e-6 um
    read-back tolerance below). Column dtypes come from the parent NWB:
    ``AnalysisNwbfile().create`` exports the parent's electrodes table and a
    pynwb export preserves each column's on-disk dtype, so a parent written
    with float32 ``rel_*`` hands the writer a float32 destination. Widening
    the destination keeps the persisted geometry equal to the geometry the
    sort ran with, for every file, rather than only for coordinates that
    happen to be exactly representable in the parent's dtype.

    h5py cannot change a dataset's dtype in place, so the dataset is read,
    unlinked and recreated with the same name, shape, layout and ATTRIBUTES.
    The attributes are what keep the file readable: hdmf identifies a
    ``VectorData`` column by its ``neurodata_type``/``namespace`` and tracks
    it by ``object_id``, so dropping them would orphan the column. (The
    unlinked dataset's bytes are not reclaimed by HDF5; for a handful of
    electrode rows that is a few hundred bytes.)

    Parameters
    ----------
    group : h5py.Group
        The open ``/general/extracellular_ephys/electrodes`` group.
    column : str
        Name of the ``rel_*`` dataset to widen. A no-op when it is already
        float64.
    """
    import numpy as np

    if group[column].dtype == np.float64:
        return
    from spyglass.utils.h5py_helper_fn import convert_dataset_type

    convert_dataset_type(group, column, "float64")


def _persist_channel_geometry(
    analysis_abs_path: str, row_indices, locations
) -> None:
    """Stamp the recording's normalized 2D geometry onto its electrodes rows.

    Writes ``rel_x``/``rel_y`` from the recording's 2D channel locations and
    ``rel_z = 0`` into the electrodes-table rows the ElectricalSeries
    references, then reads them back and verifies them. Rows outside the
    region keep whatever the parent NWB held -- or ``NaN``, when the column
    was created here because the parent carried no contact positions at all.

    The two persisted axes are the CHOSEN PLANE's axes, not necessarily the
    original x and y: for an x-z sort group ``rel_y`` holds what
    ``Probe.Electrode`` calls ``rel_z``, and the persisted ``rel_z`` is 0. A
    diff of an artifact's ``rel_*`` against ``Probe.Electrode``'s will show
    that relabeling; the geometry itself is unchanged.

    This runs after ``io.write`` (the ElectricalSeries and its region are on
    disk) and before the content fingerprint, which hashes exactly these rows.
    h5py rather than pynwb because the datasets already exist and only a few
    of their elements change.

    Coordinates are persisted at DOUBLE precision: a destination column the
    parent NWB wrote narrower than float64 is recreated as float64 first
    (:func:`_widen_position_column_to_float64`), because assigning into it
    would otherwise round every coordinate to the parent's precision.
    Columns this writer had to create are float64 already.

    Parameters
    ----------
    analysis_abs_path : str
        Absolute path to the just-written analysis NWB file.
    row_indices : sequence of int
        Electrodes-table ROW indices the series references, in the recording's
        channel order (the region ``electrode_table_region`` built).
    locations : array_like
        ``(n_channels, 2)`` normalized contact positions, in the recording's
        channel order.

    Raises
    ------
    ValueError
        If ``locations`` is not ``(n_channels, 2)`` or does not have one row
        per referenced electrode.
    RuntimeError
        If the persisted coordinates do not read back as written.
    """
    import h5py
    import numpy as np

    positions = np.asarray(locations, dtype=float)
    rows = np.asarray(row_indices, dtype=int)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            "write_nwb_artifact: the recording's channel locations must be "
            f"2D after geometry normalization, got shape {positions.shape}."
        )
    if len(positions) != len(rows):
        raise ValueError(
            "write_nwb_artifact: the recording has "
            f"{len(positions)} channel locations but its ElectricalSeries "
            f"references {len(rows)} electrodes; the persisted geometry would "
            "be misaligned with the series."
        )
    expected = np.column_stack([positions, np.zeros(len(rows))])

    with h5py.File(analysis_abs_path, "a") as handle:
        group = handle["/general/extracellular_ephys/electrodes"]
        for axis, column in enumerate(_RELATIVE_POSITION_COLUMNS):
            # The destination dtype is the parent NWB's; widen it first so
            # the assignment below cannot truncate what it is handed.
            _widen_position_column_to_float64(group, column)
            values = group[column][:]
            values[rows] = expected[:, axis]
            group[column][:] = values

    with h5py.File(analysis_abs_path, "r") as handle:
        group = handle["/general/extracellular_ephys/electrodes"]
        persisted = np.column_stack(
            [group[column][:][rows] for column in _RELATIVE_POSITION_COLUMNS]
        )
    if not np.allclose(
        persisted, expected, rtol=0.0, atol=_POSITION_TOLERANCE_UM
    ):
        raise RuntimeError(
            "write_nwb_artifact: the electrodes table did not retain the "
            f"normalized geometry (wrote {expected.tolist()}, read back "
            f"{persisted.tolist()}). A reload would rebuild the recording "
            "from coordinates the sort never saw."
        )


def write_nwb_artifact(
    recording,
    nwb_file_name: str,
    existing_analysis_file_name: str | None = None,
    timestamps_override=None,
    *,
    filtering_description: str,
    provenance_tables=None,
    description: str | None = None,
) -> tuple[str, str, str]:
    """Write the preprocessed recording into an ``AnalysisNwbfile``.

    Streams the ``(n_samples, n_channels)`` trace array and the
    ``(n_samples,)`` timestamps vector into the ElectricalSeries
    via HDMF's ``GenericDataChunkIterator`` with a channel-count-scaled
    buffer (~30 s of data, capped at 5 GB; see ``write_buffer_gb``).
    Without streaming, a 30 kHz x 128 ch x 1 h recording
    (~110 GB float64) would have to materialize in RAM before the
    NWB write, which OOMs on any lab workstation.

    The electrodes rows the ElectricalSeries references are stamped with the
    recording's NORMALIZED 2D geometry -- ``rel_x``/``rel_y`` from
    ``recording.get_channel_locations()`` and a constant ``rel_z = 0`` -- so a
    reload's x-y projection is exactly the geometry the sort saw. SpikeInterface
    rebuilds channel locations from those columns, so persisting the parent
    NWB's raw 3D coordinates instead would let an x-z probe collapse back into
    coincident contacts on every read. A recording whose locations are still 3D
    is REFUSED rather than projected: the caller must normalize it to a plane
    first. The persisted ``rel_x``/``rel_y`` are the two axes of the plane that
    normalization chose, so for an x-z sort group ``rel_y`` holds the original
    ``Probe.Electrode`` ``rel_z``. Rows outside the series region keep the
    parent NWB's values, or ``NaN`` where a ``rel_*`` column had to be created
    because the parent carried none.

    Returns ``(analysis_file_name, electrical_series_object_id,
    content_hash)``. The ``content_hash`` is the
    :func:`recording_content_fingerprint` aggregate computed **after**
    the write by reading the persisted ElectricalSeries back from its
    known absolute path -- the recording's reproducible scientific
    identity, not a whole-file byte digest -- so a content-identical
    rebuild reproduces it (see :mod:`spyglass.spikesorting.v2._recording.fingerprint`).

    Writes the file to disk only; the caller registers the
    ``AnalysisNwbfile`` row inside its DataJoint transaction so
    the file registration and the table row commit atomically.

    Parameters
    ----------
    recording : si.BaseRecording
        The preprocessed recording to materialize.
    nwb_file_name : str
        Parent NWB filename (passed to ``AnalysisNwbfile().create``).
    existing_analysis_file_name : str, optional
        When set, write into the existing slot (the recompute /
        rebuild path) rather than minting a new analysis file.
    timestamps_override : array-like, optional
        Pre-computed persisted timestamps, shape ``(n_samples,)``. May be a
        lazy timestamp vector; the chunk iterator materializes only requested
        slices. ``None`` lets the helper concatenate per-segment
        ``recording.get_times()`` via :func:`_get_recording_timestamps`.
    filtering_description : str
        Keyword-only. Provenance string written to
        ``ElectricalSeries.filtering`` describing the preprocessing
        steps that actually ran (from :func:`filtering_description`).
    provenance_tables : list of hdmf.common.DynamicTable, optional
        Keyword-only. Pre-built provenance tables (from
        :mod:`._nwb_provenance`) embedded as NWB scratch alongside the
        ElectricalSeries so the artifact is self-describing. ``None``
        (default) writes no provenance and leaves the data path unchanged.
        Scratch does not enter the ``content_hash`` fingerprint.
    description : str, optional
        Keyword-only. ``ElectricalSeries.description``; ``None`` (default)
        describes a pre-motion preprocessed recording of ``nwb_file_name``.
        The description does not enter the ``content_hash`` fingerprint.
    """
    import numpy as np
    import pynwb

    from spyglass.spikesorting.v2._storage.iterators import (
        SpikeInterfaceRecordingDataChunkIterator,
        TimestampsDataChunkIterator,
    )
    from spyglass.spikesorting.v2._core.signal_math import (
        assert_positive_sampling_frequency,
    )
    from spyglass.spikesorting.v2._core.signal_math import (
        _get_recording_timestamps,
    )
    from spyglass.spikesorting.v2._storage.metadata import (
        electrode_table_region,
        resolve_conversion_and_offset,
        write_buffer_gb,
    )

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._storage.recompute import combined_hash
    from spyglass.spikesorting.v2._recording.fingerprint import (
        recording_content_fingerprint,
    )
    from spyglass.utils import logger

    # ``AnalysisNwbfile().create`` writes a stub file to disk
    # before we open it for the streaming write. Track the
    # filename from the first byte on disk and unlink on any
    # failure between here and the post-write hash, so a
    # partial / aborted write never outlives this call.
    analysis_file_name = AnalysisNwbfile().create(
        nwb_file_name=nwb_file_name,
        recompute_file_name=existing_analysis_file_name,
        restrict_permission=True,  # 0o644, not world-writable 0o666
    )
    try:
        analysis_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name,
            from_schema=bool(existing_analysis_file_name),
        )

        # Traces are written unscaled (return_in_uV=False), so the
        # ElectricalSeries must carry gain (as ``conversion``) AND offset
        # (as ``offset``) to recover real volts on readback:
        # ``volts = raw * conversion + offset``. The resolver rejects
        # heterogeneous gain/offset and non-positive gain so a per-channel
        # gain or a missing offset cannot silently corrupt the scaling.
        conversion, es_offset = resolve_conversion_and_offset(recording)

        # Geometry preconditions, settled before anything streams: the
        # electrodes rows this write stamps are what SpikeInterface rebuilds a
        # reloaded recording's channel locations from, so a recording with no
        # geometry -- or with geometry that has not been reduced to a plane --
        # must not reach the write at all.
        #
        # ``get_channel_locations()`` defaults to ``axes="xy"``, so a still-3D
        # recording would be silently PROJECTED rather than rejected, and the
        # persisted x-y projection of an x-z probe is exactly the collapse the
        # normalization exists to prevent. Reject it here rather than rely on
        # the caller: ``Recording.make_compute`` normalizes and asserts
        # distinct positions upstream, but the concatenated-recording writer
        # assembles its recording from reloaded member artifacts and has no
        # such upstream check.
        if not recording.has_channel_location():
            raise ValueError(
                "write_nwb_artifact: the recording carries no contact "
                "positions, so the artifact would persist no geometry. "
                "Populate Probe.Electrode rel_x/rel_y/rel_z for this sort "
                "group's electrodes."
            )
        if (
            recording.get_property("location") is not None
            and recording.has_3d_locations()
        ):
            raise ValueError(
                "write_nwb_artifact: the recording still carries 3D channel "
                "locations; normalize them to a 2D plane before writing "
                "(normalize_channel_locations). Persisting SpikeInterface's "
                "default x-y projection of a 3D geometry would silently "
                "collapse contacts that are distinct only in z."
            )

        # The data iterator drives ``recording.get_traces(...)``
        # per chunk and never materializes the whole array. The
        # timestamps iterator wraps a 1D vector; resolve through
        # ``_get_recording_timestamps`` so multi-segment NWBs and
        # persisted-timestamps overrides both flow through correctly.
        sampling_frequency = assert_positive_sampling_frequency(
            recording.get_sampling_frequency(), context="write_nwb_artifact: "
        )
        timestamps = _get_recording_timestamps(
            recording, override=timestamps_override
        )
        # Bound the buffers to ~a fixed duration of data so a narrow sort
        # group (e.g. a 4-ch tetrode) does not buffer the whole recording
        # in one 5 GB chunk; wide groups stay capped at 5 GB.
        data_iterator = SpikeInterfaceRecordingDataChunkIterator(
            recording=recording,
            return_in_uV=False,
            buffer_gb=write_buffer_gb(
                recording.get_num_channels(), sampling_frequency
            ),
        )
        timestamps_iterator = TimestampsDataChunkIterator(
            timestamps=timestamps,
            buffer_gb=write_buffer_gb(1, sampling_frequency),
        )

        with pynwb.NWBHDF5IO(
            path=analysis_abs_path, mode="a", load_namespaces=True
        ) as io:
            nwbfile = io.read()
            # The normalized geometry has to land in rel_x/rel_y/rel_z; a
            # parent NWB that never carried those columns gets them here,
            # NaN-filled, while the file is still open for writing.
            _ensure_relative_position_columns(nwbfile)
            # ``recording.get_channel_ids()`` are spyglass electrode ids;
            # map them to electrodes-table ROW INDICES (not raw ids) so a
            # non-contiguous / reordered electrodes table does not silently
            # mis-point the ElectricalSeries at the wrong electrodes.
            table_region = electrode_table_region(
                nwbfile,
                recording.get_channel_ids(),
                "Sort group electrodes",
            )
            geometry_rows = [int(row) for row in table_region.data]
            series = pynwb.ecephys.ElectricalSeries(
                name=_ELECTRICAL_SERIES_NAME,
                data=data_iterator,
                electrodes=table_region,
                timestamps=timestamps_iterator,
                filtering=filtering_description,
                description=(
                    description
                    or f"Pre-motion preprocessed recording from "
                    f"{nwb_file_name} for spike sorting"
                ),
                conversion=conversion,
                offset=es_offset,
            )
            nwbfile.add_acquisition(series)
            for table in provenance_tables or ():
                nwbfile.add_scratch(table)
            object_id = nwbfile.acquisition[_ELECTRICAL_SERIES_NAME].object_id
            io.write(nwbfile)

        # Persist the geometry the sort actually ran on. SpikeInterface
        # rebuilds a reloaded recording's channel locations from these rows,
        # so without this the reload silently reverts to the parent NWB's raw
        # 3D coordinates -- an x-z tetrode collapses again under SI's x-y
        # projection and the in-memory normalization is lost. Before the
        # fingerprint below, which hashes these same rows as the artifact's
        # geometry component. ``axes="xy"`` is explicit: the guards above
        # already established that these locations ARE 2D, so this is an
        # identity selection, not a projection.
        _persist_channel_geometry(
            analysis_abs_path,
            geometry_rows,
            recording.get_channel_locations(axes="xy"),
        )

        # Fingerprint the persisted file (read back from the known abs path,
        # never the checksum-validating get_abs_path) so the identity reflects
        # the recording's reproducible SCIENCE -- traces, timestamps, geometry,
        # scaling -- not the whole-file byte digest. A content-identical rebuild
        # reproduces this hash even though its bytes differ.
        content_hash = combined_hash(
            recording_content_fingerprint(
                analysis_abs_path,
                electrical_series_path=f"acquisition/{_ELECTRICAL_SERIES_NAME}",
            )
        )
    except Exception:
        # Any write/hash failure: remove the partial analysis file before
        # re-raising so a half-written artifact never lingers -- but NEVER a
        # canonical artifact (see ``_remove_partial_artifact``).
        _remove_partial_artifact(
            analysis_file_name, existing_analysis_file_name
        )
        raise

    return analysis_file_name, object_id, content_hash


def recording_provenance_table(
    *,
    recording_id,
    raw_object_id,
    preprocessing_params_name,
    sort_group_id,
    reference_mode,
    bad_channel_handling,
):
    """Build the recording source-provenance scratch table.

    Re-emits, into the artifact NWB, the source lineage ``make_fetch``
    resolved from the DB so the file is interpretable without the database:
    the raw source object id, the recording id, the preprocessing recipe,
    the sort group, the resolved reference mode, the bad-channel handling,
    and the producing SpikeInterface version. Returns a one-element list
    (``write_nwb_artifact``'s ``provenance_tables`` contract).
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2._storage.provenance import (
        RECORDING_PROVENANCE,
        build_provenance_table,
    )

    return [
        build_provenance_table(
            RECORDING_PROVENANCE,
            {
                "recording_id": str(recording_id),
                "raw_object_id": str(raw_object_id),
                "preprocessing_params_name": str(preprocessing_params_name),
                "sort_group_id": int(sort_group_id),
                "reference_mode": str(reference_mode),
                "bad_channel_handling": str(bad_channel_handling),
                "spikeinterface_version": si.__version__,
            },
        )
    ]


def compute_recording_artifact(
    *,
    raw_path: str,
    raw_object_id: str,
    nwb_file_name: str,
    interval_list_name: str,
    channel_ids: list,
    reference_mode: str,
    reference_electrode_id: int | None,
    sort_valid_times,
    raw_valid_times,
    preprocessing_params: PreprocessingParamsSchema,
    probe_types: tuple,
    electrode_group_names: tuple,
    bad_channel_ids: tuple = (),
    existing_analysis_file_name: str | None = None,
    provenance_tables=None,
    writer: Callable | None = None,
) -> RecordingArtifactResult:
    """Open raw NWB, run preprocessing, stream to AnalysisNwbfile.

    The stage order is: read -> channel-select -> normalize the channel
    geometry -> temporal preprocessing (phase-shift, bandpass) on the
    CONTINUOUS recording -> time restriction -> spatial preprocessing
    (bad-channel interpolation, reference) -> tetrode geometry repair ->
    distinct-position check. Filtering before the restriction is what
    keeps a selected interval free of the transient a concatenation join
    would otherwise inject.

    Pipeline body shared between ``Recording.make_compute`` and
    ``Recording._rebuild_nwb_artifact``; both stage a fresh, unregistered file
    (``existing_analysis_file_name=None``). The rebuild path installs that
    staged file into the canonical slot via ``os.replace`` only after a
    verified ``content_hash`` match -- it does not overwrite in place.

    Parameters
    ----------
    writer : callable, optional
        Artifact writer with the ``write_nwb_artifact`` signature. Defaults
        to that service; the table can supply its configured writer.
    raw_path : str
        Absolute path to the raw NWB file to read.
    raw_object_id : str
        NWB object id of the raw acquisition ElectricalSeries
        (``Raw.raw_object_id``); selects the source series to read.
    nwb_file_name : str
        Name of the session's raw NWB file.
    interval_list_name : str
        ``IntervalList`` name selecting the sort interval.
    channel_ids : list
        Sorted electrode ids for the sort group.
    reference_mode : str
        Referencing mode (e.g. ``'none'``, ``'specific'``).
    reference_electrode_id : int or None
        Reference electrode id; non-``None`` only for ``'specific'``.
    sort_valid_times : numpy.ndarray
        Requested sort interval ``valid_times``, shape
        ``(n_intervals, 2)`` in seconds.
    raw_valid_times : numpy.ndarray
        Raw data ``valid_times``, shape ``(n_intervals, 2)`` in
        seconds.
    preprocessing_params : PreprocessingParamsSchema
        Validated preprocessing parameters.
    probe_types : tuple
        Per-channel ``probe_type`` for the sort group.
    electrode_group_names : tuple
        Per-channel ``electrode_group_name`` for the sort group.
    bad_channel_ids : tuple, optional
        Interior bad channels to re-include on the ``interpolate``
        path; defaults to ``()`` (empty, the ``remove`` path).
    existing_analysis_file_name : str or None, optional
        When ``None`` (default), stage a fresh ``AnalysisNwbfile``;
        otherwise write into this existing file in place, which is left in
        place (never unlinked) if the write fails. Every current caller --
        populate, rebuild and recompute -- passes ``None``.
    provenance_tables : optional
        Source-lineage scratch table(s) to embed in the file (see
        :func:`recording_provenance_table`), passed to the writer.

    Returns
    -------
    RecordingArtifactResult
        ``(analysis_file_name, object_id, content_hash, saved_start,
        saved_end, sampling_frequency, n_channels, duration_s)`` -- the
        metadata needed for the ``RecordingComputed`` boxing.
        ``saved_start``/``saved_end``/``duration_s`` describe the
        PERSISTED timestamps (the override when set).

    Notes
    -----
    Cleanup contract: ``_write_nwb_artifact`` either writes a full file or
    raises before any registration. On a write/hash failure a freshly
    staged partial file is unlinked before the error propagates; an existing
    file named by ``existing_analysis_file_name`` is never unlinked. When the
    file was freshly staged (every current caller), a half-written artifact
    never outlives a failed compute.

    A rebuild whose content drifted is rejected by the caller
    (``RecordingContentDriftError``) and the canonical slot is never
    written. The rebuild runs only when the cache file is already absent,
    so no valid cache is lost, and the row's ``content_hash`` plus the raw
    NWB let the next ``get_recording`` try again.
    """
    from spyglass.spikesorting.v2._recording.geometry import (
        assert_unique_contact_positions,
        maybe_apply_tetrode_geometry,
        normalize_channel_locations,
    )
    from spyglass.spikesorting.v2._recording.preprocessing import (
        apply_spatial_preprocessing,
        apply_temporal_preprocessing,
    )

    # Aliased: the bare name is the local ``filtering_description`` below.
    from spyglass.spikesorting.v2._recording.preprocessing import (
        filtering_description as _filtering_description_svc,
    )
    from spyglass.spikesorting.v2._recording.restriction import (
        restrict_recording,
        select_sort_group_channels,
    )
    from spyglass.spikesorting.v2._core.signal_math import (
        _get_recording_timestamps,
    )
    from spyglass.utils.nwb_helper_fn import raw_eseries_path_and_timestamp_mode

    # Name the exact raw acquisition, including when the file also has LFP.
    # Rate-based raw ElectricalSeries can reconstruct selected timestamps
    # lazily from (t_start, sampling_frequency, frame index). Explicit
    # timestamp series may be irregular, so retain their explicit vector.
    raw_series_path, load_time_vector = raw_eseries_path_and_timestamp_mode(
        raw_path, object_id=raw_object_id
    )
    recording = read_recording_nwb(
        raw_path,
        load_time_vector=load_time_vector,
        electrical_series_path=raw_series_path,
    )
    sampling_frequency = float(recording.get_sampling_frequency())

    # Channel-slice first, then filter each continuous acquisition span, and only
    # then restrict in time. Restricting first would hand the lazy bandpass
    # a concatenation of the selected intervals, and its margin would be
    # read across the artificial joins -- so every interval edge, and a
    # short interval in its entirety, would be filter transient rather than
    # signal. Real timestamp gaps are split before filtering, so margins cannot
    # cross missing acquisition data. A frame slice of a filter pulls
    # that margin from the continuous parent instead, so each retained
    # sample is filtered with its true temporal context. The spatial steps
    # are per-sample across channels, so they are unaffected by the joins
    # and run afterwards, on the restricted recording only.
    recording = select_sort_group_channels(
        recording,
        nwb_file_abs_path=raw_path,
        sort_group_channel_ids=channel_ids,
        reference_mode=reference_mode,
        reference_electrode_id=reference_electrode_id,
        bad_channel_handling=preprocessing_params.bad_channel_handling,
        bad_channel_ids=bad_channel_ids,
    )
    # Choose the plane from the contacts that STAY on the sort surface.
    # The slice above also carries the ``specific`` reference, which
    # ``apply_spatial_preprocessing`` subtracts and drops; because
    # ``Probe.Electrode`` rel_* are per probe TYPE, a reference on another
    # probe of the same type duplicates a member's raw position and would
    # veto every plane for a group that is distinct without it.
    # The interior bad channels the ``interpolate`` path re-includes DO
    # stay; the reference is excluded even when it is one of them (which
    # is also how ``apply_spatial_preprocessing`` treats it).
    retained = {int(c) for c in channel_ids}
    if preprocessing_params.bad_channel_handling == "interpolate":
        retained |= {int(c) for c in bad_channel_ids}
    if reference_mode == "specific":
        retained -= {int(reference_electrode_id)}
    retained_channel_ids = sorted(retained)
    recording = normalize_channel_locations(
        recording, channel_ids=retained_channel_ids
    )
    recording, temporal_steps = apply_temporal_preprocessing(
        recording, preprocessing_params
    )
    recording, timestamps_override, n_selected_intervals = restrict_recording(
        recording=recording,
        nwb_file_name=nwb_file_name,
        interval_list_name=interval_list_name,
        sort_valid_times=sort_valid_times,
        raw_valid_times=raw_valid_times,
        min_segment_length=preprocessing_params.min_segment_length,
    )
    recording, spatial_steps = apply_spatial_preprocessing(
        recording,
        reference_mode=reference_mode,
        reference_electrode_id=reference_electrode_id,
        validated=preprocessing_params,
        bad_channel_handling=preprocessing_params.bad_channel_handling,
        bad_channel_ids=bad_channel_ids,
    )
    applied_steps = {**temporal_steps, **spatial_steps}
    recording = maybe_apply_tetrode_geometry(
        recording=recording,
        probe_types=probe_types,
        electrode_group_names=electrode_group_names,
        sort_group_channel_ids=channel_ids,
    )
    assert_unique_contact_positions(recording)

    # Provenance string for the persisted ElectricalSeries, built from the
    # steps ACTUALLY applied (the ``applied_steps`` report): a fixed
    # "Bandpass filter + common reference" string would misdescribe the
    # saved artifact for the no_filter preset or reference_mode='none'
    # (DANDI / archival), and a requested-but-skipped phase-shift must not
    # be listed.
    filtering_description = _filtering_description_svc(
        preprocessing_params.bandpass_filter, reference_mode, applied_steps
    )

    analysis_file_name = None
    try:
        (
            analysis_file_name,
            object_id,
            content_hash,
        ) = (write_nwb_artifact if writer is None else writer)(
            recording=recording,
            nwb_file_name=nwb_file_name,
            existing_analysis_file_name=existing_analysis_file_name,
            timestamps_override=timestamps_override,
            filtering_description=filtering_description,
            provenance_tables=provenance_tables,
        )
        # For a single contiguous interval, derive saved
        # start/end/duration from the persisted override
        # so the row matches the cached timestamps rather than the
        # frame-slice's uncorrected ``get_times()``. For a
        # multi-interval concat the override is the gap-spanning
        # wall-clock envelope; its span would over-count by the gaps
        # and break the truncation guard, so use the concat's own
        # gap-excluded times there.
        if n_selected_intervals == 1:
            saved_times = _get_recording_timestamps(
                recording, override=timestamps_override
            )
            saved_start = float(saved_times[0])
            saved_end = float(saved_times[-1])
        else:
            # ``concatenate_recordings(ignore_times=True)`` gives a
            # synthetic contiguous 0-based time axis. Avoid materializing
            # that full vector just to read the first/last value.
            n_samples = int(recording.get_num_samples(segment_index=0))
            saved_start = 0.0
            saved_end = (
                0.0 if n_samples == 0 else (n_samples - 1) / sampling_frequency
            )
        n_channels = int(recording.get_num_channels())
        duration_s = float(saved_end - saved_start)
    except Exception:
        # Unlink only a file this call staged fresh. A file named by
        # ``existing_analysis_file_name`` is an existing artifact written in
        # place; it is never unlinked here (``write_nwb_artifact`` leaves it
        # in place too).
        if (
            existing_analysis_file_name is None
            and analysis_file_name is not None
        ):
            _unlink_staged_analysis_file(
                analysis_file_name,
                context="Recording._compute_recording_artifact",
            )
        raise

    return RecordingArtifactResult(
        analysis_file_name=analysis_file_name,
        object_id=object_id,
        content_hash=content_hash,
        sampling_frequency=sampling_frequency,
        saved_start=saved_start,
        saved_end=saved_end,
        n_channels=n_channels,
        duration_s=duration_s,
    )


def rebuild_nwb_artifact(table, key) -> None:
    """Rebuild a missing recording artifact -- locked, atomic, reconciled.

    Locked + atomic-publish: acquire
    ``recording_artifact_lock(recording_id)``, double-check the file is
    still missing under the lock (a peer may have rebuilt while we waited),
    then rebuild to a PRIVATE temp file on the same filesystem as the
    canonical slot, fingerprint it, and only on a ``content_hash`` match
    ``os.replace`` it into the slot and refresh the DataJoint ``~external``
    byte checksum. A rebuild whose fingerprint diverges from the stored
    ``content_hash`` (SpikeInterface/BLAS drift, an edited raw NWB, or
    changed upstream inputs) raises ``RecordingContentDriftError`` and never
    touches the canonical slot -- drifted bytes are never served.

    Cleanup contract (all-or-nothing): the temp is unlinked on any failure;
    if ``os.replace`` ran but the checksum refresh then failed, the
    canonical is unlinked to return the slot to the missing state for the
    next (locked) ``get_recording``. Ordering is load-bearing -- the atomic
    ``os.replace`` precedes ``_resolve_external``.

    Calls ``make_fetch`` to re-derive every DB input -- the same fetch the
    populate path uses -- so a rebuild cannot drift from the original
    write's inputs. Safe to call directly (it takes the lock itself) and
    from ``get_recording`` (which does not hold the lock).

    Parameters
    ----------
    table : Recording
        A ``Recording`` instance. ``make_fetch`` and
        ``_compute_recording_artifact`` are called on it so patched
        methods take effect.
    key : dict
        Restriction selecting a single ``Recording`` row.
    """
    import time
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._recording.fingerprint import (
        recording_artifact_lock,
    )
    from spyglass.spikesorting.v2.exceptions import (
        RecordingContentDriftError,
    )
    from spyglass.utils import logger

    row = (table & key).fetch1()
    recording_id = row["recording_id"]
    analysis_file_name = row["analysis_file_name"]
    canonical_abs = AnalysisNwbfile.get_abs_path(analysis_file_name)

    with recording_artifact_lock(recording_id):
        # Double-checked: a peer rebuilt (or the file was never gone) while
        # we waited for the lock -- nothing to do. Two readers cannot both
        # rebuild.
        if Path(canonical_abs).exists():
            return

        started = time.monotonic()
        logger.info(
            "Recording.get_recording: cache miss for "
            f"{analysis_file_name!r} (reason=missing cache); rebuilding "
            "the preprocessed artifact..."
        )
        fetched = table.make_fetch(key)
        # Rebuild to a FRESH, unregistered temp file -- same analysis dir
        # (and filesystem) as the canonical slot, so the install is an
        # atomic rename. ``_compute_recording_artifact`` returns the temp's
        # readback content fingerprint as ``content_hash``.
        rebuilt = table._compute_recording_artifact(
            raw_path=fetched.raw_path,
            raw_object_id=fetched.raw_object_id,
            nwb_file_name=fetched.sel["nwb_file_name"],
            interval_list_name=fetched.sel["interval_list_name"],
            channel_ids=fetched.channel_ids,
            reference_mode=fetched.reference_mode,
            reference_electrode_id=fetched.reference_electrode_id,
            sort_valid_times=fetched.sort_valid_times,
            raw_valid_times=fetched.raw_valid_times,
            preprocessing_params=fetched.preprocessing_params,
            probe_types=fetched.probe_types,
            electrode_group_names=fetched.electrode_group_names,
            bad_channel_ids=fetched.bad_channel_ids,
            existing_analysis_file_name=None,  # fresh temp, not the slot
            provenance_tables=recording_provenance_table(
                recording_id=fetched.sel["recording_id"],
                raw_object_id=fetched.raw_object_id,
                preprocessing_params_name=fetched.sel[
                    "preprocessing_params_name"
                ],
                sort_group_id=fetched.sel["sort_group_id"],
                reference_mode=fetched.reference_mode,
                bad_channel_handling=(
                    fetched.preprocessing_params.bad_channel_handling
                ),
            ),
        )
        temp_abs = AnalysisNwbfile.get_abs_path(rebuilt.analysis_file_name)

        if rebuilt.content_hash != row["content_hash"]:
            Path(temp_abs).unlink(missing_ok=True)
            raise RecordingContentDriftError(
                "Recording._rebuild_nwb_artifact: rebuilt content_hash "
                f"{rebuilt.content_hash} does not match the stored "
                f"content_hash {row['content_hash']} for "
                f"{analysis_file_name!r}. The current environment no "
                "longer reproduces this recording (e.g. a SpikeInterface/"
                "BLAS upgrade, an edited raw NWB, or changed upstream "
                "inputs). The canonical artifact was NOT modified. Recover "
                "by restoring a backup of the artifact, rerunning the "
                "recompute under the original environment, or deleting and "
                "repopulating the Recording row (and its downstream)."
            )

        install_rebuilt_recording(temp_abs, canonical_abs, analysis_file_name)

        logger.info(
            "Recording.get_recording: rebuilt + reconciled "
            f"{analysis_file_name!r} in "
            f"{time.monotonic() - started:.1f}s"
        )

    # Best-effort (outside the all-or-nothing block): clear a stale
    # RecordingArtifactRecompute deleted=1 flag now the file is back on
    # disk. File tracking is presence-aware, so a failed clear cannot hide
    # the rebuilt file -- this only keeps the flag accurate.
    clear_recompute_deleted_flag(recording_id)


def rebuild_motion_corrected_artifact(table, key) -> None:
    """Rebuild a missing corrected artifact from the SAVED motion.

    Locked on the corrected recording, double-checked under the lock,
    then ``make_fetch`` / ``make_compute`` write a fresh temp artifact:
    the saved estimate is reapplied, never estimated again, under the
    installed SpikeInterface even if it differs from the one the
    selection was made with. Only a temp
    whose ``content_hash`` equals the stored one is installed
    (``install_rebuilt_recording``); otherwise it is removed,
    ``RecordingContentDriftError`` is raised and the canonical slot is
    left untouched. A selection whose interpolation recipe now resolves
    differently, or whose application algorithm version changed, cannot
    reproduce the stored traces: ``RecordingContentDriftError`` names
    the missing file and the repair before anything is computed.

    Parameters
    ----------
    table : MotionCorrectedRecording
        A ``MotionCorrectedRecording`` instance; ``make_fetch`` and
        ``make_compute`` are called on it.
    key : dict
        Restriction selecting a single ``MotionCorrectedRecording`` row.
    """
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._motion.estimation import (
        motion_corrected_recording_artifact_lock,
        resolve_interpolation_params,
    )
    from spyglass.spikesorting.v2._motion.compute import (
        stale_corrected_selection_fields,
    )
    from spyglass.spikesorting.v2.exceptions import (
        RecordingContentDriftError,
    )
    from spyglass.utils import logger

    row = (table & key).fetch1()
    analysis_file_name = row["analysis_file_name"]
    canonical_abs = AnalysisNwbfile.get_abs_path(analysis_file_name)
    with motion_corrected_recording_artifact_lock(
        row["motion_corrected_recording_id"]
    ):
        if Path(canonical_abs).exists():
            return
        logger.info(
            "MotionCorrectedRecording.get_recording: cache miss for "
            f"{analysis_file_name!r}; reapplying the saved motion..."
        )
        master_key = {
            "motion_corrected_recording_id": row[
                "motion_corrected_recording_id"
            ]
        }
        fetched = table.make_fetch(master_key)
        # The content hash is the guard (as for Recording), so a rebuild
        # under another SpikeInterface version is allowed and installed
        # only if it reproduces the stored traces. A changed recipe
        # resolution or application algorithm cannot reproduce them.
        stale = stale_corrected_selection_fields(
            fetched.selection,
            resolve_interpolation_params(fetched.interpolation_params),
            check_spikeinterface_version=False,
        )
        if stale:
            raise RecordingContentDriftError(
                "MotionCorrectedRecording._rebuild_nwb_artifact: the "
                f"corrected recording file {analysis_file_name!r} is "
                "missing and cannot be rebuilt: its selection is stale "
                f"({'; '.join(stale)}), so reapplying the saved motion "
                "would not reproduce the stored traces. Restore the file "
                "from a backup, or delete this MotionCorrectedRecording "
                "row and everything made from it (sorts, curations) and "
                "repopulate them from a new "
                "MotionCorrectedRecordingSelection."
            )
        computed = table.make_compute(
            master_key,
            *fetched,
            allow_spikeinterface_version_change=True,
        )
        if computed.content_hash != row["content_hash"]:
            _unlink_staged_analysis_file(
                computed.analysis_file_name,
                context="MotionCorrectedRecording._rebuild_nwb_artifact",
            )
            raise RecordingContentDriftError(
                "MotionCorrectedRecording._rebuild_nwb_artifact: rebuilt "
                f"content_hash {computed.content_hash} does not match the "
                f"stored content_hash {row['content_hash']} for "
                f"{analysis_file_name!r}. The current environment no "
                "longer reproduces this corrected recording (e.g. a "
                "SpikeInterface/BLAS upgrade). The canonical artifact was "
                "NOT modified. Recover by restoring a backup or deleting "
                "and repopulating the MotionCorrectedRecording row."
            )
        install_rebuilt_recording(
            AnalysisNwbfile.get_abs_path(computed.analysis_file_name),
            canonical_abs,
            analysis_file_name,
        )


def clear_recompute_deleted_flag(recording_id) -> None:
    """Best-effort clear of stale ``RecordingArtifactRecompute.deleted``.

    After an on-demand rebuild restores the file, clear any ``deleted=1``
    recompute rows for this recording so the flag stays semantically true
    ("intentionally removed and still absent"). Best-effort: a failed clear
    is logged, never raised, and cannot hide the rebuilt file because file
    tracking is presence-aware.
    """
    from spyglass.utils import logger

    try:
        from spyglass.spikesorting.v2.recompute import (
            RecordingArtifactRecompute,
            RecordingArtifactVersions,
        )

        versions = RecordingArtifactVersions & {"recording_id": recording_id}
        flagged = (RecordingArtifactRecompute & versions & "deleted=1").fetch(
            "KEY", as_dict=True
        )
        for flagged_key in flagged:
            RecordingArtifactRecompute.update1({**flagged_key, "deleted": 0})
    except Exception as exc:  # pragma: no cover -- best-effort accuracy
        logger.warning(
            "Recording._rebuild_nwb_artifact: could not clear stale "
            f"deleted flag for recording_id={recording_id}: {exc!r}"
        )


def rebuild_concat_nwb_artifact(table, key) -> None:
    """Rebuild a missing concat artifact -- locked, atomic, content-verified.

    The concat analog of ``Recording._rebuild_nwb_artifact``. Acquire
    ``concat_recording_artifact_lock(concat_recording_id)``, double-check the
    file is still missing under the lock (a peer may have rebuilt while we
    waited), then re-run the materialization (``make_fetch`` -> ``make_compute``
    -- which also re-verifies the frozen member set against the live
    recordings) to a FRESH temp analysis file, fingerprint it, and only on a
    ``content_hash`` match ``os.replace`` it into the canonical slot and
    refresh the DataJoint ``~external`` byte checksum. A rebuild whose
    fingerprint diverges from the stored ``content_hash`` raises
    ``RecordingContentDriftError`` and never touches the canonical slot --
    drifted bytes are never served.

    Cleanup contract (all-or-nothing): the temp is unlinked on any failure;
    if ``os.replace`` ran but the checksum refresh then failed, the canonical
    is unlinked to return the slot to the missing state for the next (locked)
    ``get_recording``. Ordering is load-bearing -- the atomic ``os.replace``
    precedes ``_resolve_external``.

    An irreproducible rebuild (e.g. a changed SpikeInterface or NWB read
    path) surfaces loudly as ``RecordingContentDriftError`` rather than
    silently serving different bytes (the fingerprint's trace rounding
    absorbs sub-µV noise).
    Safe to call directly (it takes the lock itself) and from
    ``get_recording`` (which does not hold the lock).

    Parameters
    ----------
    table : ConcatenatedRecording
        A ``ConcatenatedRecording`` instance; ``make_fetch`` and
        ``make_compute`` are called on it.
    key : dict
        Restriction selecting a single ``ConcatenatedRecording`` row.
    """
    from pathlib import Path

    import numpy as np

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._recording.concat import (
        concat_recording_artifact_lock,
    )
    from spyglass.spikesorting.v2.exceptions import (
        RecordingContentDriftError,
    )
    from spyglass.utils import logger

    row = (table & key).fetch1()
    concat_recording_id = row["concat_recording_id"]
    analysis_file_name = row["analysis_file_name"]
    canonical_abs = AnalysisNwbfile.get_abs_path(analysis_file_name)

    with concat_recording_artifact_lock(concat_recording_id):
        # Double-checked: a peer rebuilt (or the file was never gone) while
        # we waited for the lock -- nothing to do.
        if Path(canonical_abs).exists():
            return

        logger.info(
            "ConcatenatedRecording.get_recording: cache miss for "
            f"{analysis_file_name!r} (reason=missing cache); rebuilding "
            "the concatenated artifact..."
        )
        fetched = table.make_fetch(key)
        # make_compute writes a FRESH (unregistered) temp analysis file and
        # returns its readback content fingerprint as ``content_hash``.
        computed = table.make_compute(key, *fetched)
        temp_abs = AnalysisNwbfile.get_abs_path(computed.analysis_file_name)

        if computed.content_hash != row["content_hash"]:
            _unlink_staged_analysis_file(
                computed.analysis_file_name,
                context="ConcatenatedRecording._rebuild_nwb_artifact",
            )
            raise RecordingContentDriftError(
                "ConcatenatedRecording._rebuild_nwb_artifact: rebuilt "
                f"content_hash {computed.content_hash} does not match the "
                f"stored content_hash {row['content_hash']} for "
                f"{analysis_file_name!r}. The current environment no longer "
                "reproduces this concatenated recording (e.g. a "
                "SpikeInterface/BLAS upgrade or a changed member recording). "
                "The canonical artifact "
                "was NOT modified. Recover by restoring a backup, rerunning "
                "under the original environment, or deleting and repopulating "
                "the ConcatenatedRecording row (and its downstream)."
            )
        # The traces fingerprint does not include the stored spans:
        # downstream sorts estimate noise from the statistics spans and
        # motion estimation reads the continuity spans and their first
        # and last timestamps, so a rebuild must reproduce them exactly.
        drifted = [
            name
            for name, shape in (
                ("statistics_spans", (-1, 2)),
                ("continuity_spans", (-1, 2)),
                ("continuity_start_s", (-1,)),
                ("continuity_end_s", (-1,)),
            )
            if not np.array_equal(
                np.asarray(getattr(computed, name)).reshape(shape),
                np.asarray(row[name]).reshape(shape),
            )
        ]
        if drifted:
            _unlink_staged_analysis_file(
                computed.analysis_file_name,
                context="ConcatenatedRecording._rebuild_nwb_artifact",
            )
            raise RecordingContentDriftError(
                "ConcatenatedRecording._rebuild_nwb_artifact: rebuilt "
                f"{drifted} do not match the stored values for "
                f"{analysis_file_name!r}. The canonical artifact was NOT "
                "modified. Delete and repopulate the "
                "ConcatenatedRecording row (and its downstream)."
            )

        install_rebuilt_recording(temp_abs, canonical_abs, analysis_file_name)
