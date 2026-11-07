"""Units-NWB read/write IO for v2 sorts.

Shared by ``Sorting`` and ``CurationV2``. Only ``write_curated_units_nwb``
queries the database, for the source sort, before writing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

from spyglass.spikesorting._numerical import (
    finite_intervals,
    finite_scalar,
    finite_vector,
    integer_scalar,
    integer_vector,
)

SPIKE_SAMPLE_INDEX_COLUMN = "spike_sample_index"


def read_units_columns(abs_path, columns, *, unit_ids=None) -> tuple:
    """Open a units NWB once and read the requested per-unit columns.

    Selected units' required columns are validated together, including aligned
    spike-time/frame lengths. Only the requested columns are returned.

    Parameters
    ----------
    abs_path : str or pathlib.Path
        Absolute path to the v2 units NWB file.
    columns : sequence of str
        Units columns to read, any of ``"spike_times"`` (float seconds),
        ``SPIKE_SAMPLE_INDEX_COLUMN`` (int64 frames) and ``"obs_intervals"``
        (float ``(n, 2)`` seconds).
    unit_ids : iterable of int, optional
        Read only these units; an id absent from the table is skipped. Default
        reads every unit.

    Returns
    -------
    tuple
        One ``{unit_id (int): np.ndarray}`` per entry of ``columns``, in order.
        Every entry is ``{}`` for an empty or absent Units table. A populated
        v2 table must contain spike times, sample frames, and observation
        intervals; missing columns raise ``ValueError``.
    """
    import numpy as np
    import pynwb

    wanted = (
        None
        if unit_ids is None
        else set(integer_vector(list(unit_ids), name="unit_id").tolist())
    )
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        nwbf = io.read()
        units = nwbf.units
        if units is None or len(units) == 0:
            return tuple({} for _ in columns)
        required = {"spike_times", SPIKE_SAMPLE_INDEX_COLUMN, "obs_intervals"}
        missing = sorted(required.difference(units.colnames))
        if missing:
            raise ValueError(
                f"Invalid v2 Units NWB {str(abs_path)!r}: populated Units table "
                f"is missing required columns {missing}."
            )
        identifiers = integer_vector(units.id[:], name="unit_id")
        if len(np.unique(identifiers)) != len(identifiers):
            raise ValueError(
                f"Invalid v2 Units NWB {str(abs_path)!r}: unit_id must be unique."
            )
        out = [{} for _ in columns]
        for row_ind, uid in enumerate(identifiers):
            uid = int(uid)
            if wanted is not None and uid not in wanted:
                continue
            context = f"v2 Units NWB {str(abs_path)!r}, unit_id={uid}"
            times = finite_vector(
                units["spike_times"][row_ind], name=f"{context} spike_times"
            )
            frames = integer_vector(
                units[SPIKE_SAMPLE_INDEX_COLUMN][row_ind],
                name=f"{context} spike_sample_index",
                nonnegative=True,
            )
            if len(times) != len(frames):
                raise ValueError(
                    f"{context}: spike_times and spike_sample_index must have matching lengths."
                )
            values = {
                "spike_times": times,
                SPIKE_SAMPLE_INDEX_COLUMN: frames,
                "obs_intervals": finite_intervals(
                    units["obs_intervals"][row_ind],
                    name=f"{context} obs_intervals",
                ),
            }
            for column, result in zip(columns, out, strict=True):
                result[uid] = values[column]
        return tuple(out)


def read_units_abs_spike_times(abs_path) -> dict:
    """Return ``{unit_id(int): abs_spike_times(np.ndarray seconds)}``.

    ``{}`` for an empty or absent Units table.
    """
    return read_units_columns(abs_path, ("spike_times",))[0]


def read_units_spike_sample_indices(abs_path) -> dict:
    """Return ``{unit_id: spike_sample_index}`` from a current v2 Units NWB.

    Avoids reading the recording's timestamps. ``{}`` for an empty or absent
    Units table.
    """
    return read_units_columns(abs_path, (SPIKE_SAMPLE_INDEX_COLUMN,))[0]


def read_units_abs_times_and_sample_indices(abs_path, *, unit_ids=None):
    """Open the units NWB ONCE; return ``(abs_times, sample_indices, obs)``.

    ``obs`` is ``{unit_id: (n, 2) obs_intervals}``; the curated writer carries
    it forward so exported rates use the right denominator. ``unit_ids``
    restricts the read, as in :func:`read_units_columns`.
    """
    return read_units_columns(
        abs_path,
        ("spike_times", SPIKE_SAMPLE_INDEX_COLUMN, "obs_intervals"),
        unit_ids=unit_ids,
    )


def curation_source_unit_ids(kept_unit_to_contributors, apply_merge):
    """Return the source unit ids ``write_curated_units_nwb`` will actually read.

    Must mirror ``_write_curated_units_nwb_body``'s reads. A merged unit reads
    its contributors; a fresh merged id is never read from the source.

    Parameters
    ----------
    kept_unit_to_contributors : dict[int, list[int]]
        ``{kept_unit_id: [source contributor ids]}`` for the curation.
    apply_merge : bool
        Whether proposed merges are applied (subset read) or previewed (read
        all).

    Returns
    -------
    set[int] or None
        The source unit ids to read, or ``None`` to read every unit.
    """
    if not apply_merge:
        return None
    needed: set[int] = set()
    for kept_uid, contribs in kept_unit_to_contributors.items():
        if len(contribs) > 1:
            needed.update(
                integer_vector(
                    list(contribs), name="merge member unit_id"
                ).tolist()
            )
        else:
            needed.add(integer_scalar(kept_uid, name="unit_id"))
    return needed


def numpysorting_from_sample_indices(sample_indices, fs):
    """Build a ``NumpySorting`` directly from stored sample frames."""
    import spikeinterface as si

    fs = finite_scalar(fs, name="sampling_frequency", positive=True)
    units_dict = {
        integer_scalar(uid, name="unit_id"): integer_vector(
            frames, name="spike_sample_index", nonnegative=True
        )
        for uid, frames in sample_indices.items()
    }
    return si.NumpySorting.from_unit_dict([units_dict], sampling_frequency=fs)


def sorting_from_units_nwb(abs_path, sampling_frequency):
    """Read a current v2 Units NWB using its stored source-recording frames.

    No DB access or timestamp reads.
    """
    return numpysorting_from_sample_indices(
        read_units_spike_sample_indices(abs_path), sampling_frequency
    )


class StoredUnits(NamedTuple):
    """A Units NWB path and source rate for DB-free frame readback."""

    abs_path: str
    sampling_frequency: float


def read_stored_units(units: StoredUnits):
    """Open resolved stored sample frames as a ``NumpySorting``; no DB access."""
    return sorting_from_units_nwb(units.abs_path, units.sampling_frequency)


def _integer_keyed_mapping(values, *, name):
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping of integer unit IDs.")
    return {
        integer_scalar(uid, name="unit_id"): value
        for uid, value in values.items()
    }


def _validated_spike_mappings(abs_times, sample_indices):
    times_by_unit = _integer_keyed_mapping(abs_times, name="spike_times")
    frames_by_unit = _integer_keyed_mapping(
        sample_indices, name="spike_sample_index"
    )
    if times_by_unit.keys() != frames_by_unit.keys():
        raise ValueError(
            "spike_times and spike_sample_index must have matching unit IDs."
        )
    for uid in times_by_unit:
        times = finite_vector(
            times_by_unit[uid], name=f"unit_id={uid} spike_times"
        )
        frames = integer_vector(
            frames_by_unit[uid],
            name=f"unit_id={uid} spike_sample_index",
            nonnegative=True,
        )
        if len(times) != len(frames):
            raise ValueError(
                f"unit_id={uid}: spike_times and spike_sample_index must have matching lengths."
            )
        times_by_unit[uid], frames_by_unit[uid] = times, frames
    return times_by_unit, frames_by_unit


def build_lazy_merged_sorting_from_samples(
    abs_times, sample_indices, units_to_merge, fs, *, delta_s
):
    """Reconstruct a lazily-merged sorting using stored frames.

    Deduplicates in absolute time so recording gaps are respected; the kept
    events' frames come through the same mask.
    """
    abs_times, sample_indices = _validated_spike_mappings(
        abs_times, sample_indices
    )
    units_to_merge = [
        integer_vector(list(group), name="merge member unit_id").tolist()
        for group in units_to_merge
    ]
    units_dict: dict = {}
    merged_members = {u for g in units_to_merge for u in g}
    for uid, frames in sample_indices.items():
        if uid not in merged_members:
            units_dict[uid] = frames

    next_id = max(abs_times, default=-1) + 1
    for contribs in units_to_merge:
        _times, frames = _dedup_merged_spike_times_and_frames(
            [abs_times[u] for u in contribs],
            [sample_indices[u] for u in contribs],
            delta_s,
        )
        units_dict[next_id] = frames
        next_id += 1
    return numpysorting_from_sample_indices(units_dict, fs)


def _dedup_merged_spike_times_and_frames(times_list, frames_list, delta_s):
    """Return deduplicated ``(times, frames)`` with both arrays aligned."""
    import numpy as np

    times_list, frames_list = list(times_list), list(frames_list)
    if len(times_list) != len(frames_list):
        raise ValueError(
            "Merged unit spike-time and sample-index contributor lists must have matching lengths."
        )
    delta_s = finite_scalar(delta_s, name="delta_s", nonnegative=True)
    time_arrays = [finite_vector(t, name="spike_times") for t in times_list]
    frame_arrays = [
        integer_vector(f, name="spike_sample_index", nonnegative=True)
        for f in frames_list
    ]
    for times, frames in zip(time_arrays, frame_arrays, strict=True):
        if times.shape != frames.shape:
            raise ValueError(
                "Merged unit spike times and sample indices must have matching "
                f"shapes; got {times.shape} and {frames.shape}."
            )
    concat_times = (
        np.concatenate(time_arrays)
        if time_arrays
        else np.asarray([], dtype=float)
    )
    concat_frames = (
        np.concatenate(frame_arrays)
        if frame_arrays
        else np.asarray([], dtype=np.int64)
    )
    if concat_times.size == 0:
        return concat_times, concat_frames
    order = concat_times.argsort(kind="mergesort")
    times_sorted = concat_times[order]
    frames_sorted = concat_frames[order]
    membership = np.concatenate(
        [np.full(arr.shape, i) for i, arr in enumerate(time_arrays)]
    )[order]
    keep = np.nonzero(
        (np.diff(times_sorted) > delta_s) | (np.diff(membership) == 0)
    )[0]
    keep = np.concatenate([[0], keep + 1])
    return times_sorted[keep], frames_sorted[keep]


def abs_spike_times_dataframe(abs_times):
    """Build a DataFrame (index=unit_id) of absolute spike-time arrays.

    Parameters
    ----------
    abs_times : dict[int, np.ndarray]
        ``{unit_id: absolute spike times (seconds)}``.

    Returns
    -------
    pandas.DataFrame
        A single ``spike_times`` column of per-unit arrays, indexed by
        ``unit_id``.
    """
    import pandas as pd

    abs_times = _integer_keyed_mapping(abs_times, name="spike_times")
    unit_ids = list(abs_times)
    return pd.DataFrame(
        {
            "spike_times": [
                finite_vector(abs_times[u], name=f"unit_id={u} spike_times")
                for u in unit_ids
            ]
        },
        index=pd.Index(unit_ids, name="unit_id"),
    )


def empty_spike_times_dataframe():
    """Build an empty spike-times DataFrame for zero-unit sorts.

    Returns
    -------
    pandas.DataFrame
        An empty ``spike_times`` column, with an integer ``unit_id``
        index.
    """
    import pandas as pd

    return pd.DataFrame(
        {"spike_times": []},
        index=pd.Index([], name="unit_id", dtype=int),
    )


def _sample_indices_to_times_by_unit(recording, sample_indices_by_unit):
    """Map stored sample frames to absolute times without full-vector allocation."""
    import numpy as np

    sample_indices_by_unit = _integer_keyed_mapping(
        sample_indices_by_unit, name="spike_sample_index"
    )
    n_samples = integer_scalar(
        recording.get_num_samples(segment_index=0),
        name="recording n_samples",
        nonnegative=True,
    )
    for uid, frames in sample_indices_by_unit.items():
        frames = integer_vector(
            frames, name=f"unit_id={uid} spike_sample_index"
        )
        if np.any((frames < 0) | (frames >= n_samples)):
            raise ValueError(
                f"spike_sample_index contains frame(s) outside the recording range [0, {n_samples})."
            )
        sample_indices_by_unit[uid] = frames

    def read_times(frames):
        times = finite_vector(
            recording.sample_index_to_time(frames, segment_index=0),
            name="spike_times",
        )
        if len(times) != len(frames):
            raise ValueError(
                "spike_times and spike_sample_index must have matching lengths."
            )
        return times

    if not recording.has_time_vector(segment_index=0):
        # Rate-based: ``frames / fs + t_start`` computed by SpikeInterface.
        return {
            uid: read_times(frames)
            for uid, frames in sample_indices_by_unit.items()
        }

    from spyglass.spikesorting.v2._core.signal_math import _segment_times_at

    return {
        int(uid): finite_vector(
            _segment_times_at(recording, frames), name="spike_times"
        )
        for uid, frames in sample_indices_by_unit.items()
    }


def recording_timestamps(recording_row):
    """Return the full timestamp vector of the upstream Recording.

    Timestamps keep gaps between disjoint sort intervals, so
    ``t_start + i/fs`` would be wrong. Reads no traces.

    Parameters
    ----------
    recording_row : dict
        The upstream Recording row, carrying ``analysis_file_name`` and
        ``electrical_series_path``.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        The recording's wall-clock timestamps, in seconds (float64).
    """
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    return read_series_timestamps(
        AnalysisNwbfile.get_abs_path(recording_row["analysis_file_name"]),
        recording_row["electrical_series_path"],
    )


def read_series_timestamps(abs_path, electrical_series_path):
    """Read a persisted ``ElectricalSeries``' full timestamp vector; no DB.

    Parameters
    ----------
    abs_path : str
        Absolute path of the analysis NWB holding the series.
    electrical_series_path : str
        The stored in-file path of the series (its last component names the
        acquisition entry).

    Returns
    -------
    np.ndarray, shape (n_samples,)
        The series' wall-clock timestamps, in seconds (float64).
    """
    import numpy as np
    import pynwb

    series_name = electrical_series_path.rsplit("/", 1)[-1]
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        nwbf = io.read()
        series = nwbf.acquisition[series_name]
        return np.asarray(series.timestamps[:], dtype=np.float64)


#: Sorting-provenance field: artifact-free half-open frame spans that never
#: cross a join, as ``[start, end]`` pairs.
STATISTICS_SPANS_FIELD = "statistics_spans"


def read_sorting_statistics_spans(
    abs_path, *, sorting_id
) -> list[tuple[int, int]]:
    """Read the statistics spans persisted in a sorting units NWB.

    Parameters
    ----------
    abs_path : str or pathlib.Path
        Absolute path to the sort's units NWB.
    sorting_id : str
        The sort, named in the error message.

    Returns
    -------
    list[tuple[int, int]]
        Sorted half-open frame spans of the sorted recording.

    Raises
    ------
    RuntimeError
        If the file has no persisted spans. The whole recording is not a
        safe fallback.
    ValueError
        If the spans are not ordered, non-overlapping and non-empty.
    """
    from spyglass.spikesorting.v2._storage.provenance import (
        SORTING_PROVENANCE,
        read_provenance_values,
    )

    try:
        values = read_provenance_values(str(abs_path), SORTING_PROVENANCE)
    except KeyError:  # no sorting-provenance scratch table at all
        values = {}
    if STATISTICS_SPANS_FIELD not in values:
        raise RuntimeError(
            f"Sorting sorting_id={str(sorting_id)!r} has no persisted "
            f"statistics spans in its units NWB {str(abs_path)!r}; the file "
            "is incomplete or was not written by the v2 Sorting writer. Delete "
            "and repopulate this Sorting (and its downstream) so noise and "
            "whitening statistics are estimated from its artifact-free spans."
        )
    result = []
    previous_end = 0
    for pair in values[STATISTICS_SPANS_FIELD]:
        span = integer_vector(pair, name="statistics_spans", nonnegative=True)
        if len(span) != 2 or span[1] <= span[0] or span[0] < previous_end:
            raise ValueError(
                "statistics_spans requires ordered, nonoverlapping positive frame intervals."
            )
        result.append((int(span[0]), int(span[1])))
        previous_end = span[1]
    return result


def _add_sample_frame_column(nwbf, frame_trains):
    """Write aligned ragged frames, including units whose trains are all empty.

    Added after the unit rows; otherwise HDMF turns appended empty trains into
    a 2D array.
    """
    import numpy as np

    frame_trains = [
        integer_vector(frames, name="spike_sample_index", nonnegative=True)
        for frames in frame_trains
    ]
    nwbf.add_unit_column(
        name=SPIKE_SAMPLE_INDEX_COLUMN,
        description=(
            "Sample indices into the sorted recording, aligned with absolute "
            "spike_times for frame-based readback without the full timestamp vector."
        ),
        data=frame_trains,
        index=True,
    )
    # Keep the flattened target int64 even when every row is empty.
    nwbf.units[SPIKE_SAMPLE_INDEX_COLUMN].target.transform(
        lambda frames: np.asarray(frames, dtype=np.int64)
    )


def write_sorting_units_nwb(
    sorting,
    recording,
    nwb_file_name,
    obs_intervals=None,
    *,
    unit_metadata=None,
    source_provenance,
):
    """Write a fresh AnalysisNwbfile containing only the v2 Units table.

    Spike times are absolute, comparable with the Recording's valid times.
    ``spike_sample_index`` lets ``get_sorting`` skip the recording's
    timestamps. Every row carries ``obs_intervals`` (default: the recording's
    contiguous spans) and ``curation_label="uncurated"``.
    """
    from spyglass.spikesorting.v2._storage.provenance import (
        validate_sorting_provenance,
    )

    validate_sorting_provenance(source_provenance)
    from spyglass.spikesorting.v2._storage.staged_outputs import (
        staged_analysis_file,
    )

    # The caller's staging cleanup does not know this file's name.
    with staged_analysis_file(
        nwb_file_name, context="write_sorting_units_nwb"
    ) as analysis_file_name:
        return _write_sorting_units_nwb_body(
            analysis_file_name=analysis_file_name,
            sorting=sorting,
            recording=recording,
            obs_intervals=obs_intervals,
            unit_metadata=unit_metadata,
            source_provenance=source_provenance,
        )


def _write_sorting_units_nwb_body(
    *,
    analysis_file_name,
    sorting,
    recording,
    obs_intervals,
    unit_metadata=None,
    source_provenance,
):
    """Fill the staged sort-units ``AnalysisNwbfile`` (no cleanup on failure).

    Returns ``(analysis_file_name, units_object_id)``. ``unit_metadata``
    (``{unit_id: {peak_amplitude_uv, peak_electrode_id, n_spikes,
    brain_region}}``) adds the ``Sorting.Unit`` values as columns.
    """
    import pynwb

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._core.signal_math import (
        base_intervals_and_gaps,
    )
    from spyglass.spikesorting.v2._storage.provenance import (
        SORTING_PROVENANCE,
        build_provenance_table,
    )

    analysis_abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)

    sample_indices_by_unit = {
        int(unit_id): integer_vector(
            sorting.get_unit_spike_train(unit_id=unit_id),
            name="spike_sample_index",
            nonnegative=True,
        )
        for unit_id in integer_vector(sorting.unit_ids, name="unit_id")
    }
    spike_times_by_unit = _sample_indices_to_times_by_unit(
        recording, sample_indices_by_unit
    )
    sampling_frequency = finite_scalar(
        recording.get_sampling_frequency(),
        name="sampling_frequency",
        positive=True,
    )
    if obs_intervals is None:
        # No artifact pass: every recorded sample was observed. Split at gaps
        # so a disjoint recording's duration excludes them.
        obs_intervals_arr = finite_intervals(
            base_intervals_and_gaps(
                recording, sampling_frequency
            ).base_intervals,
            name="obs_intervals",
        )
    else:
        obs_intervals_arr = finite_intervals(
            obs_intervals, name="obs_intervals"
        )

    with pynwb.NWBHDF5IO(
        path=analysis_abs_path, mode="a", load_namespaces=True
    ) as io:
        nwbf = io.read()
        # Scalar here so ``== "uncurated"`` works; curation writes a ragged
        # list. Columns must exist before ``add_unit`` passes them.
        if len(sorting.unit_ids) > 0:
            nwbf.add_unit_column(
                name="curation_label",
                description=(
                    'Curation label scalar; ``"uncurated"`` at '
                    "sort time, refined to a per-unit label list "
                    "by CurationV2.insert_curation."
                ),
            )
            if unit_metadata is not None:
                for name, desc in (
                    ("peak_amplitude_uv", "Peak template amplitude (uV)."),
                    (
                        "peak_electrode_id",
                        "Peak channel spyglass electrode id.",
                    ),
                    ("n_spikes", "Number of spikes in the unit."),
                    ("brain_region", "Brain region of the peak electrode."),
                ):
                    nwbf.add_unit_column(name=name, description=desc)
        for unit_id in sorting.unit_ids:
            unit_id = int(unit_id)
            spike_times = spike_times_by_unit[unit_id]
            unit_kwargs = dict(
                spike_times=spike_times,
                id=unit_id,
                obs_intervals=obs_intervals_arr,
                curation_label="uncurated",
            )
            if unit_metadata is not None:
                meta = unit_metadata[unit_id]
                unit_kwargs.update(
                    peak_amplitude_uv=float(meta["peak_amplitude_uv"]),
                    peak_electrode_id=int(meta["peak_electrode_id"]),
                    n_spikes=int(meta["n_spikes"]),
                    brain_region=str(meta["brain_region"] or ""),
                )
            nwbf.add_unit(**unit_kwargs)
        if len(sorting.unit_ids) > 0:
            _add_sample_frame_column(
                nwbf,
                [sample_indices_by_unit[int(uid)] for uid in sorting.unit_ids],
            )
        # pynwb leaves ``units`` None for a zero-unit sort.
        if nwbf.units is None:
            nwbf.units = pynwb.misc.Units(
                name="units",
                description="Empty units table (sorter found zero units).",
            )
        units_object_id = nwbf.units.object_id
        nwbf.add_scratch(
            build_provenance_table(SORTING_PROVENANCE, source_provenance)
        )
        io.write(nwbf)

    # The caller registers the file in its insert transaction.
    return analysis_file_name, units_object_id


def write_curated_units_nwb(
    sorting_id,
    kept_unit_to_contributors: dict,
    apply_merge: bool,
    labels: dict,
    *,
    source_units_abs_path: str | None = None,
    curation_header: dict,
    merge_group_rows: list[dict] | None = None,
) -> tuple[str, str, str, dict]:
    """Write the curated-units NWB.

    Returns ``(analysis_file_name, units_object_id, nwb_file_name,
    n_spikes_by_uid)``; ``n_spikes_by_uid`` counts the stored, deduplicated
    trains.

    ``source_units_abs_path`` is ``None`` for a root curation (reads the
    ``Sorting`` units NWB) or the parent curation's units NWB, whose unit ids
    ``kept_unit_to_contributors`` then uses.

    With ``apply_merge=True`` a merged unit is the deduplicated union of its
    contributors, with id ``max(source unit_ids) + 1`` onward in ascending
    min-contributor order (matching ``build_curated_unit_rows``); survivors
    come first in source order. With ``apply_merge=False`` (preview) every
    source unit is written unchanged.
    """
    from spyglass.spikesorting.v2._storage.provenance import (
        validate_curation_header,
    )

    validate_curation_header(curation_header)
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.sorting import Sorting

    # Same parent NWB as the Sorting.
    nwb_file_name = Sorting.resolve_anchor_nwb_file_name(
        {"sorting_id": sorting_id}
    )

    if source_units_abs_path is None:
        src_abs_path = AnalysisNwbfile.get_abs_path(
            (Sorting & {"sorting_id": sorting_id}).fetch1("analysis_file_name")
        )
    else:
        src_abs_path = source_units_abs_path
    abs_times_by_uid, sample_indices_by_uid, obs_intervals_by_uid = (
        read_units_abs_times_and_sample_indices(
            src_abs_path,
            unit_ids=curation_source_unit_ids(
                kept_unit_to_contributors, apply_merge
            ),
        )
    )

    from spyglass.spikesorting.v2._storage.staged_outputs import (
        staged_analysis_file,
    )

    # Curation stages outside its transaction, so its cleanup misses this file.
    with staged_analysis_file(
        nwb_file_name, context="write_curated_units_nwb"
    ) as analysis_file_name:
        return _write_curated_units_nwb_body(
            analysis_file_name=analysis_file_name,
            nwb_file_name=nwb_file_name,
            kept_unit_to_contributors=kept_unit_to_contributors,
            apply_merge=apply_merge,
            labels=labels,
            abs_times_by_uid=abs_times_by_uid,
            sample_indices_by_uid=sample_indices_by_uid,
            obs_intervals_by_uid=obs_intervals_by_uid,
            curation_header=curation_header,
            merge_group_rows=merge_group_rows,
        )


def _curated_obs_intervals(
    kept_uid, contribs, apply_merge, obs_intervals_by_uid
):
    """Per-unit ``obs_intervals`` for one curated/kept unit.

    A merged unit gets the intersection of its contributors' windows; other
    units keep their own.
    """
    from spyglass.spikesorting.v2._core.signal_math import (
        intersect_interval_sets,
    )

    if apply_merge and len(contribs) > 1:
        return intersect_interval_sets(
            [obs_intervals_by_uid[int(u)] for u in contribs]
        )
    return obs_intervals_by_uid[int(kept_uid)]


def _write_curated_units_nwb_body(
    *,
    analysis_file_name,
    nwb_file_name,
    kept_unit_to_contributors,
    apply_merge,
    labels,
    abs_times_by_uid,
    sample_indices_by_uid,
    obs_intervals_by_uid,
    curation_header,
    merge_group_rows=None,
):
    """Fill the staged curated-units ``AnalysisNwbfile`` (no cleanup on error).

    Returns ``(analysis_file_name, units_object_id, nwb_file_name,
    n_spikes_by_uid)``. ``obs_intervals_by_uid`` is
    ``{unit_id: (n, 2) array}``; SI quality metrics do not use it.
    """
    import numpy as np
    import pynwb

    from spyglass.spikesorting.v2._storage.provenance import (
        validate_curation_header,
    )

    validate_curation_header(curation_header)
    if sample_indices_by_uid is None or obs_intervals_by_uid is None:
        raise ValueError(
            "Curated v2 Units require sample indices and observation intervals."
        )
    abs_times_by_uid, sample_indices_by_uid = _validated_spike_mappings(
        abs_times_by_uid, sample_indices_by_uid
    )
    obs_intervals_by_uid = _integer_keyed_mapping(
        obs_intervals_by_uid, name="obs_intervals"
    )
    if obs_intervals_by_uid.keys() != abs_times_by_uid.keys():
        raise ValueError(
            "obs_intervals and spike_times must have matching unit IDs."
        )
    obs_intervals_by_uid = {
        uid: finite_intervals(intervals, name=f"unit_id={uid} obs_intervals")
        for uid, intervals in obs_intervals_by_uid.items()
    }
    kept_unit_to_contributors = {
        integer_scalar(uid, name="unit_id"): integer_vector(
            list(contributors), name="merge member unit_id"
        ).tolist()
        for uid, contributors in kept_unit_to_contributors.items()
    }
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._core.enums import CurationLabel
    from spyglass.spikesorting.v2._core.signal_math import _MERGE_DEDUP_DELTA_MS

    analysis_abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)

    with pynwb.NWBHDF5IO(
        path=analysis_abs_path, mode="a", load_namespaces=True
    ) as io:
        nwbf = io.read()
        if apply_merge:
            write_specs = []
            for kept_uid, contribs in kept_unit_to_contributors.items():
                if len(contribs) > 1:
                    # Same dedup as get_merged_sorting, so stored equals
                    # previewed.
                    spike_times, spike_indices = (
                        _dedup_merged_spike_times_and_frames(
                            [abs_times_by_uid[int(u)] for u in contribs],
                            [sample_indices_by_uid[int(u)] for u in contribs],
                            _MERGE_DEDUP_DELTA_MS / 1000.0,
                        )
                    )
                else:
                    spike_times = abs_times_by_uid[int(kept_uid)]
                    spike_indices = sample_indices_by_uid[int(kept_uid)]
                obs = _curated_obs_intervals(
                    kept_uid, contribs, apply_merge, obs_intervals_by_uid
                )
                write_specs.append(
                    (int(kept_uid), spike_times, spike_indices, obs)
                )
        else:
            write_specs = [
                (
                    int(uid),
                    abs_times_by_uid[int(uid)],
                    sample_indices_by_uid[int(uid)],
                    _curated_obs_intervals(
                        uid, [uid], apply_merge, obs_intervals_by_uid
                    ),
                )
                for uid in sorted(abs_times_by_uid)
            ]
        # Post-dedup counts keep ``Unit.n_spikes`` equal to the stored train.
        n_spikes_by_uid = {
            int(uid): int(len(spike_times))
            for uid, spike_times, _spike_indices, _obs in write_specs
        }

        if write_specs:
            # Labels are added as a ragged column after the units so pynwb
            # can infer its dtype.
            all_labels: list[list[str]] = []
            for unit_id, spike_times, spike_indices, obs in write_specs:
                lbl_list = labels.get(int(unit_id), [])
                label_list = [CurationLabel.normalize(lbl) for lbl in lbl_list]
                all_labels.append(label_list)
                unit_kwargs = {
                    "spike_times": np.asarray(spike_times, dtype=np.float64),
                    "id": int(unit_id),
                }
                unit_kwargs["obs_intervals"] = np.asarray(obs, dtype=np.float64)
                nwbf.add_unit(**unit_kwargs)
            _add_sample_frame_column(nwbf, [spec[2] for spec in write_specs])
            # pynwb cannot infer the dtype of all-empty label lists.
            if any(all_labels):
                nwbf.add_unit_column(
                    name="curation_label",
                    description=(
                        "Curation label list from "
                        "CurationV2.insert_curation; one entry "
                        "per label, empty list if unlabeled. "
                        "Indexed (ragged) column."
                    ),
                    data=all_labels,
                    index=True,
                )
        else:
            # Empty curation: pynwb would leave ``units`` None.
            nwbf.units = pynwb.misc.Units(
                name="units",
                description=("Empty units table (curation kept zero units)."),
            )
        units_object_id = nwbf.units.object_id
        # Lineage mirrors CurationV2.MergeGroup (raw contributors), real merges
        # only.
        from collections import defaultdict

        from spyglass.spikesorting.v2._storage.provenance import (
            CURATION_MERGE_LINEAGE,
            CURATION_PROVENANCE,
            build_long_provenance_table,
            build_provenance_table,
        )

        nwbf.add_scratch(
            build_provenance_table(CURATION_PROVENANCE, curation_header)
        )
        contributors_by_kept: dict[int, list[int]] = defaultdict(list)
        for row in merge_group_rows or ():
            contributors_by_kept[int(row["unit_id"])].append(
                int(row["contributor_unit_id"])
            )
        merge_lineage_rows = [
            {"kept_unit_id": kept, "contributor_unit_id": contributor}
            for kept, contributors in contributors_by_kept.items()
            if len(contributors) > 1
            for contributor in contributors
        ]
        nwbf.add_scratch(
            build_long_provenance_table(
                CURATION_MERGE_LINEAGE,
                merge_lineage_rows,
                [("kept_unit_id", int), ("contributor_unit_id", int)],
            )
        )
        io.write(nwbf)

    # The caller registers the file in its insert transaction.
    return (
        analysis_file_name,
        units_object_id,
        nwb_file_name,
        n_spikes_by_uid,
    )
