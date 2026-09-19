"""NWB artifact I/O behind ``Recording``.

``read_recording_nwb`` opens an explicit series with lazy timestamp access,
including when workers or analyzers reconstruct the extractor.
``write_nwb_artifact`` streams the preprocessed traces and the wall-clock
timestamps vector into an ``AnalysisNwbfile`` for ``Recording.make_compute``
(and the rebuild path), then hashes the persisted file for the cache contract.
``install_rebuilt_recording`` installs verified single/concatenated recording
rebuilds and reconciles their tracked byte checksums.
The table threads already-fetched DB state in (the tri-part
``make_fetch``/``make_compute``/``make_insert`` contract forbids DB I/O inside
compute), and the file row is registered by the caller inside its DataJoint
transaction, so this write path stays a thin file-write service.

Why this lives in its own module rather than in ``recording.py``:
``recording.py`` is a DataJoint *schema* module -- importing it activates
``dj.schema(...)`` and the source-part dependencies. The NWB-write logic needs
none of that at import, so ``Recording`` becomes a thin orchestrator (fetch ->
call these -> insert / verify). Same "thin DataJoint shell over pure/IO
services" direction as ``_artifact_compute`` / ``_selection_identity`` /
``_analyzer_cache`` / ``_curation_transforms`` / ``_units_nwb`` /
``_sorting_dispatch``.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: all SpikeInterface / numpy / pynwb / spyglass
dependencies are imported lazily inside the function. ``write_nwb_artifact``
touches the DB / DataJoint at CALL time via lazy imports (``AnalysisNwbfile``
path resolution + file create). It also lazily imports the
``_ELECTRICAL_SERIES_NAME`` constant from ``recording`` at call time -- by then
``recording`` is fully imported, so there is no import cycle.
"""

from __future__ import annotations


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
    recording._kwargs["use_pynwb"] = True
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


def raw_eseries_path_and_timestamp_mode(
    nwb_file_abs_path: str, raw_object_id: str
) -> tuple[str, bool]:
    """Return the raw ElectricalSeries' in-file path + timestamp mode.

    Resolves the acquisition ``ElectricalSeries`` whose NWB ``object_id``
    equals ``raw_object_id`` -- the exact object the common ``Raw`` row was
    ingested from. A file can hold more than one acquisition
    ``ElectricalSeries`` (and repacking/copying can reorder acquisition
    iteration), so selecting by object id reads the intended raw signal rather
    than whichever series comes first.

    Rate-based ElectricalSeries store ``starting_time`` + ``rate`` and do not
    need SpikeInterface to load a full time vector. Timestamp-based series carry
    a ``timestamps`` dataset and must preserve that explicit vector to avoid
    treating irregular/dropped-sample timing as affine.

    Parameters
    ----------
    nwb_file_abs_path : str
        Absolute path to the raw NWB file.
    raw_object_id : str
        NWB object id of the raw acquisition ElectricalSeries (the
        ``Raw.raw_object_id`` recorded at ingest).

    Returns
    -------
    (path, uses_explicit_timestamps) : tuple of (str, bool)
        In-file path (e.g. ``"acquisition/e-series"``) of the matched series
        and whether it stores an explicit ``timestamps`` vector.

    Raises
    ------
    ValueError
        If no acquisition ElectricalSeries with ``object_id == raw_object_id``
        is present in the file (fail closed rather than read a different
        series).
    """
    import h5py

    with h5py.File(nwb_file_abs_path, "r") as nwb_file:
        acquisition = nwb_file.get("acquisition")
        if acquisition is not None:
            for name, obj in acquisition.items():
                neurodata_type = obj.attrs.get("neurodata_type", b"")
                if isinstance(neurodata_type, bytes):
                    neurodata_type = neurodata_type.decode()
                if neurodata_type != "ElectricalSeries":
                    continue
                object_id = obj.attrs.get("object_id", b"")
                if isinstance(object_id, bytes):
                    object_id = object_id.decode()
                if object_id == raw_object_id:
                    return f"acquisition/{name}", "timestamps" in obj
    raise ValueError(
        f"No acquisition ElectricalSeries with object_id={raw_object_id!r} "
        f"found in {nwb_file_abs_path}."
    )


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
    import pathlib

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.utils import logger

    if existing_analysis_file_name:
        logger.error(
            "Recording._write_nwb_artifact: in-place rebuild of canonical "
            f"artifact {analysis_file_name!r} failed; leaving it in place "
            "(refusing to unlink a canonical artifact on failure)."
        )
        return
    try:
        _abs = AnalysisNwbfile.get_abs_path(
            analysis_file_name, from_schema=False
        )
        pathlib.Path(_abs).unlink(missing_ok=True)
    except Exception as cleanup_exc:  # pragma: no cover -- defensive
        logger.error(
            "Recording._write_nwb_artifact: failed to clean up partial "
            f"analysis file {analysis_file_name!r}: {cleanup_exc!r}"
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


def _persist_channel_geometry(
    analysis_abs_path: str, row_indices, locations
) -> None:
    """Stamp the recording's normalized 2D geometry onto its electrodes rows.

    Writes ``rel_x``/``rel_y`` from the recording's 2D channel locations and
    ``rel_z = 0`` into the electrodes-table rows the ElectricalSeries
    references, then reads them back and verifies them. Rows outside the
    region keep whatever the parent NWB held -- or ``NaN``, when the column
    was created here because the parent carried no contact positions at all.

    This runs after ``io.write`` (the ElectricalSeries and its region are on
    disk) and before the content fingerprint, which hashes exactly these rows.
    h5py rather than pynwb because the datasets already exist and only a few
    of their elements change.

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
    first. Rows outside the series region keep the parent NWB's values, or
    ``NaN`` where a ``rel_*`` column had to be created because the parent
    carried none.

    Returns ``(analysis_file_name, electrical_series_object_id,
    content_hash)``. The ``content_hash`` is the
    :func:`recording_content_fingerprint` aggregate computed **after**
    the write by reading the persisted ElectricalSeries back from its
    known absolute path -- the recording's reproducible scientific
    identity, not a whole-file byte digest -- so a content-identical
    rebuild reproduces it (see :mod:`._recording_fingerprint`).

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
    """
    import numpy as np
    import pynwb

    from spyglass.spikesorting.v2._nwb_iterators import (
        SpikeInterfaceRecordingDataChunkIterator,
        TimestampsDataChunkIterator,
    )
    from spyglass.spikesorting.v2._signal_math import (
        assert_positive_sampling_frequency,
    )
    from spyglass.spikesorting.v2.utils import (
        _get_recording_timestamps,
        electrode_table_region,
        resolve_conversion_and_offset,
        write_buffer_gb,
    )

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._recompute import combined_hash
    from spyglass.spikesorting.v2._recording_fingerprint import (
        recording_content_fingerprint,
    )
    from spyglass.spikesorting.v2.recording import _ELECTRICAL_SERIES_NAME
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
            sampling_frequency=sampling_frequency,
            buffer_gb=write_buffer_gb(1, sampling_frequency),
        )

        with pynwb.NWBHDF5IO(
            path=analysis_abs_path, mode="a", load_namespaces=True
        ) as io:
            nwbfile = io.read()
            # The normalized geometry has to land in rel_x/rel_y/rel_z; a
            # parent NWB that never carried those columns gets them here,
            # zero-filled, while the file is still open for writing.
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
                    f"Pre-motion preprocessed recording from "
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
