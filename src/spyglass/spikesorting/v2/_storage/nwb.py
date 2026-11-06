"""NWB trace reading, preprocessing, streaming writes and geometry persistence.

DB-free at import. Writers receive resolved inputs and never import their
table module. Allocating a file with ``AnalysisNwbfile().create`` reads
DataJoint metadata, the accepted exception to DB-free ``make_compute``; the
file is registered only in ``make_insert``. Self-heal and rebuilds live in
``_storage.rebuilds``.
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


def _stored_trace_dtype(dtype):
    """dtype a recording artifact stores ``dtype`` traces as.

    float64 is stored as float32, halving file size and reads; its rounding
    (~3e-7 uV rms) is far below the 1 uV raw step. Integer traces keep their
    dtype.
    """
    import numpy as np

    dtype = np.dtype(dtype)
    return np.dtype(np.float32) if dtype == np.float64 else dtype


class StoredTraces(NamedTuple):
    """A cached trace artifact resolved for a read that needs no DB.

    Strings only, so DataJoint's hash of the fetched inputs matches across
    its two fetches; ``content_hash`` brings the row's content into that
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

    SI 0.104.3's default HDF5 backend reads every timestamp; its PyNWB backend
    does not. SI omits ``use_pynwb`` from reconstruction kwargs, so it is
    re-added for workers and analyzers.
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


def open_persisted_traces(abs_path: str, electrical_series_path: str):
    """Open a persisted trace artifact's ``ElectricalSeries``; no DB access.

    Annotates ``is_filtered=True`` so SpikeInterface does not filter again.

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

    Never unlinks a canonical artifact (``existing_analysis_file_name`` set).
    Cleanup failures are logged, not raised.
    """
    _unlink_staged_analysis_file(
        analysis_file_name,
        context="Recording._write_nwb_artifact",
        existing_analysis_file_name=existing_analysis_file_name,
    )


# SpikeInterface rebuilds a reloaded recording's channel locations from these;
# the content fingerprint hashes them.
_RELATIVE_POSITION_COLUMNS = ("rel_x", "rel_y", "rel_z")

# Matches the "same contact" rounding in ``_recording.geometry``.
_POSITION_TOLERANCE_UM = 1e-6


def _ensure_relative_position_columns(nwbfile) -> None:
    """Give the analysis file's electrodes table its ``rel_*`` columns.

    The parent NWB may lack them. Must run before ``io.write``. Unreferenced
    rows stay ``NaN`` ("unknown"), not the origin.

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

    Assignment casts to the parent NWB's column dtype, so a float32 column
    would truncate geometry beyond the read-back tolerance. The dataset is
    recreated with its attributes, which hdmf needs to identify the column.

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

    Writes ``rel_x``/``rel_y`` (the chosen plane's axes, so an x-z group's
    ``rel_y`` is ``Probe.Electrode``'s ``rel_z``) and ``rel_z = 0`` to the
    referenced rows as float64, then verifies the read-back. Runs after
    ``io.write`` and before the content fingerprint.

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

    Streams traces and timestamps in buffers of a few seconds. Stamps the
    referenced electrodes rows with the recording's 2D geometry, which
    SpikeInterface reloads from. The caller registers the file.

    Returns
    -------
    tuple of str
        ``(analysis_file_name, electrical_series_object_id, content_hash)``.
        ``content_hash`` is :func:`recording_content_fingerprint` of the
        written file, so a content-identical rebuild reproduces it.

    Raises
    ------
    ValueError
        If the recording has no channel locations or still has 3D locations.

    Parameters
    ----------
    recording : si.BaseRecording
        The preprocessed recording to materialize.
    nwb_file_name : str
        Parent NWB filename (passed to ``AnalysisNwbfile().create``).
    existing_analysis_file_name : str, optional
        When set, write into this existing file instead of a new one.
    timestamps_override : array-like, optional
        Persisted timestamps, shape ``(n_samples,)``; may be lazy. ``None``
        uses :func:`_get_recording_timestamps`.
    filtering_description : str
        Keyword-only. ``ElectricalSeries.filtering``: the preprocessing steps
        that ran.
    provenance_tables : list of hdmf.common.DynamicTable, optional
        Keyword-only. Provenance tables added as NWB scratch; not part of
        ``content_hash``.
    description : str, optional
        Keyword-only. ``ElectricalSeries.description``; ``None`` (default)
        describes a pre-motion preprocessed recording of ``nwb_file_name``.
        Not part of ``content_hash``.
    """
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

    # ``create`` writes a stub to disk; any later failure removes it.
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

        conversion, es_offset = resolve_conversion_and_offset(recording)

        # A 3D recording would be silently projected to x-y, collapsing an
        # x-z probe. The concat writer has no upstream check, so check here.
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

        sampling_frequency = assert_positive_sampling_frequency(
            recording.get_sampling_frequency(), context="write_nwb_artifact: "
        )
        timestamps = _get_recording_timestamps(
            recording, override=timestamps_override
        )
        # HDMF grows all-channel buffers along time to about the square root
        # of this ~30 s budget (~8 s at 32 ch).
        stored_dtype = _stored_trace_dtype(recording.get_dtype())
        data_iterator = SpikeInterfaceRecordingDataChunkIterator(
            recording=recording,
            return_in_uV=False,
            buffer_gb=write_buffer_gb(
                recording.get_num_channels(),
                sampling_frequency,
                itemsize=stored_dtype.itemsize,
            ),
            dtype=stored_dtype,
        )
        timestamps_iterator = TimestampsDataChunkIterator(
            timestamps=timestamps,
            buffer_gb=write_buffer_gb(1, sampling_frequency),
        )

        with pynwb.NWBHDF5IO(
            path=analysis_abs_path, mode="a", load_namespaces=True
        ) as io:
            nwbfile = io.read()
            _ensure_relative_position_columns(nwbfile)
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

        # Without this a reload reverts to the parent NWB's 3D coordinates.
        # The locations are already 2D, so ``axes="xy"`` does not project.
        _persist_channel_geometry(
            analysis_abs_path,
            geometry_rows,
            recording.get_channel_locations(axes="xy"),
        )

        # Content, not bytes, so a rebuild reproduces it.
        content_hash = combined_hash(
            recording_content_fingerprint(
                analysis_abs_path,
                electrical_series_path=_ELECTRICAL_SERIES_PATH,
            )
        )
    except Exception:
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

    Records the source lineage so the file is interpretable without the
    database. Returns a one-element list for ``provenance_tables``.
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

    Order: read, channel-select, normalize geometry, temporal preprocessing
    on the continuous recording, time restriction, spatial preprocessing,
    tetrode geometry repair, distinct-position check. Filtering before the
    restriction keeps join transients out of selected intervals.

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
        otherwise write into this file in place, never unlinking it on
        failure.
    provenance_tables : optional
        Source-lineage scratch table(s) to embed in the file (see
        :func:`recording_provenance_table`), passed to the writer.

    Returns
    -------
    RecordingArtifactResult
        ``(analysis_file_name, object_id, content_hash, saved_start,
        saved_end, sampling_frequency, n_channels, duration_s)``. The saved
        times describe the persisted timestamps.
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

    # Name the raw series explicitly (the file may also hold LFP). Explicit
    # timestamps may be irregular, so they are kept.
    raw_series_path, load_time_vector = raw_eseries_path_and_timestamp_mode(
        raw_path, object_id=raw_object_id
    )
    recording = read_recording_nwb(
        raw_path,
        load_time_vector=load_time_vector,
        electrical_series_path=raw_series_path,
    )
    sampling_frequency = float(recording.get_sampling_frequency())

    # Filter before restricting in time: a filter over joined intervals reads
    # its margin across the joins, turning interval edges into transients.
    recording = select_sort_group_channels(
        recording,
        nwb_file_abs_path=raw_path,
        sort_group_channel_ids=channel_ids,
        reference_mode=reference_mode,
        reference_electrode_id=reference_electrode_id,
        bad_channel_handling=preprocessing_params.bad_channel_handling,
        bad_channel_ids=bad_channel_ids,
    )
    # Choose the plane from retained contacts only. A ``specific`` reference
    # on another probe of the same type duplicates a member's position.
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

    # Describe only the steps that actually ran.
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
        # Multiple intervals: the override's span would count the gaps, so
        # use the concatenation's gap-free time axis.
        if n_selected_intervals == 1:
            saved_times = _get_recording_timestamps(
                recording, override=timestamps_override
            )
            saved_start = float(saved_times[0])
            saved_end = float(saved_times[-1])
        else:
            # ``ignore_times=True`` gives a 0-based axis; avoid materializing it.
            n_samples = int(recording.get_num_samples(segment_index=0))
            saved_start = 0.0
            saved_end = (
                0.0 if n_samples == 0 else (n_samples - 1) / sampling_frequency
            )
        n_channels = int(recording.get_num_channels())
        duration_s = float(saved_end - saved_start)
    except Exception:
        # Never unlink an existing artifact written in place.
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
