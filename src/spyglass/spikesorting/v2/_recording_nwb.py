"""NWB-write service behind ``Recording``.

``write_nwb_artifact`` streams the preprocessed traces and the wall-clock
timestamps vector into an ``AnalysisNwbfile`` for ``Recording.make_compute``
(and the rebuild path), then hashes the persisted file for the cache contract.
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
            # ``recording.get_channel_ids()`` are spyglass electrode ids;
            # map them to electrodes-table ROW INDICES (not raw ids) so a
            # non-contiguous / reordered electrodes table does not silently
            # mis-point the ElectricalSeries at the wrong electrodes.
            table_region = electrode_table_region(
                nwbfile,
                recording.get_channel_ids(),
                "Sort group electrodes",
            )
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
