"""Units-NWB read/write IO for v2 sorts.

These functions are the units-NWB IO core behind ``Sorting`` and
``CurationV2``: reading a units NWB's stored ABSOLUTE spike times,
reading the Spyglass-side sample-frame column, falling back to
absolute-time -> frame mapping for older/manual files, writing the
pre-curation sorting-units NWB (``write_sorting_units_nwb``), and writing
the post-curation curated-units NWB (``write_curated_units_nwb``). Most take
already-resolved paths / SpikeInterface objects / fetched row dicts and do
pure pynwb IO;
``write_curated_units_nwb`` is the exception -- it resolves the source
sort itself (``Sorting`` / ``SortingSelection`` / ``RecordingSelection``
fetches) before writing, so ``CurationV2.insert_curation`` stays a thin
orchestrator.

Why this lives in its own module rather than in ``sorting.py``:
``sorting.py`` is a DataJoint *schema* module -- importing it activates
``dj.schema(...)`` and the source-part / merge dependencies. The units-NWB
IO needs none of that, so ``Sorting`` becomes a thin orchestrator (fetch
row -> resolve path -> call these -> insert) and ``CurationV2`` can reach
the SHARED readback helpers here instead of reaching into ``Sorting``'s
private methods. Same "thin DataJoint shell over pure/IO services"
direction as ``_artifact_compute`` / ``_selection_identity`` /
``_analyzer_cache`` / ``_curation_transforms``.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: like ``_analyzer_cache``, the only DataJoint
dependency (``AnalysisNwbfile`` for path resolution / file creation) is
imported lazily at call time. The IO itself is pynwb against the
filesystem, not the database.
"""

from __future__ import annotations

SPIKE_SAMPLE_INDEX_COLUMN = "spike_sample_index"


def read_units_abs_spike_times(abs_path) -> dict:
    """Return ``{unit_id(int): abs_spike_times(np.ndarray seconds)}``.

    Reads the stored absolute spike times directly from the v2 Units table's
    indexed ``spike_times`` column, so callers get the persisted wall-clock
    values exactly -- no affine round-trip and no full DataFrame materialization.
    Returns ``{}`` for an empty/absent Units table.

    Parameters
    ----------
    abs_path : str or pathlib.Path
        Absolute path to the v2 units NWB file.

    Returns
    -------
    dict
        ``{unit_id (int): absolute spike times (np.ndarray of seconds)}``;
        ``{}`` for an empty or absent Units table.
    """
    import numpy as np
    import pynwb

    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        nwbf = io.read()
        units = nwbf.units
        if units is None or len(units) == 0:
            return {}
        unit_ids = np.asarray(units.id[:], dtype=int)
        spike_times = units["spike_times"]
        return {
            int(uid): np.asarray(spike_times[row_ind], dtype=float)
            for row_ind, uid in enumerate(unit_ids)
        }


def read_units_spike_sample_indices(abs_path) -> dict | None:
    """Return ``{unit_id: spike_sample_index}`` or ``None`` if absent.

    New v2 units NWBs store sample frames alongside absolute ``spike_times`` so
    Spyglass readback can reconstruct ``NumpySorting`` objects without reading
    the upstream recording's full timestamp vector. ``None`` is the compatibility
    signal for older/manual units NWBs that lack the column; callers then fall
    back to absolute-time mapping.
    """
    import numpy as np
    import pynwb

    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        nwbf = io.read()
        units = nwbf.units
        if units is None or len(units) == 0:
            return {}
        if SPIKE_SAMPLE_INDEX_COLUMN not in units.colnames:
            return None
        unit_ids = np.asarray(units.id[:], dtype=int)
        sample_indices = units[SPIKE_SAMPLE_INDEX_COLUMN]
        return {
            int(uid): np.asarray(sample_indices[row_ind], dtype=np.int64)
            for row_ind, uid in enumerate(unit_ids)
        }


def numpysorting_from_sample_indices(sample_indices, fs):
    """Build a ``NumpySorting`` directly from stored sample frames."""
    import numpy as np
    import spikeinterface as si

    units_dict = {
        int(uid): np.asarray(frames, dtype=np.int64)
        for uid, frames in sample_indices.items()
    }
    return si.NumpySorting.from_unit_dict([units_dict], sampling_frequency=fs)


def numpysorting_from_abs_times(abs_times, recording_row, fs):
    """Build a ``NumpySorting`` from absolute spike times.

    Maps each unit's absolute spike times to recording frame indices
    with ``np.searchsorted`` against the recording's (possibly
    gap-preserving) timestamps. Searching the actual timestamp vector
    is correct across wall-clock gaps, where an affine ``t_start + i/fs``
    inverse would land on the wrong frame.

    Parameters
    ----------
    abs_times : dict[int, np.ndarray]
        ``{unit_id: absolute spike times (seconds)}`` for each unit.
    recording_row : dict
        The upstream Recording row, used to read the persisted
        timestamp vector via :func:`recording_timestamps`.
    fs : float
        Sampling frequency of the recording, in Hz.

    Returns
    -------
    si.NumpySorting
        A sorting whose per-unit spike trains are frame indices into
        the recording.
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2.utils import _spike_times_to_frames

    recording_times = recording_timestamps(recording_row)
    n_samples = int(recording_times.size)
    units_dict = {
        uid: _spike_times_to_frames(recording_times, st, n_samples, uid)
        for uid, st in abs_times.items()
    }
    return si.NumpySorting.from_unit_dict([units_dict], sampling_frequency=fs)


def build_lazy_merged_sorting_from_samples(
    abs_times, sample_indices, units_to_merge, fs, *, delta_s
):
    """Reconstruct a lazily-merged sorting using stored frames.

    Deduplication still happens in absolute time so disjoint-recording gaps are
    respected. The kept absolute-time events carry their aligned sample frames
    through the same mask, avoiding a full recording timestamp-vector read.
    """
    import numpy as np

    units_dict: dict = {}
    merged_members = {int(u) for g in units_to_merge for u in g}
    for uid, frames in sample_indices.items():
        if int(uid) not in merged_members:
            units_dict[int(uid)] = np.asarray(frames, dtype=np.int64)

    next_id = max(int(u) for u in abs_times) + 1
    for contribs in units_to_merge:
        _times, frames = _dedup_merged_spike_times_and_frames(
            [np.asarray(abs_times[int(u)], dtype=float) for u in contribs],
            [
                np.asarray(sample_indices[int(u)], dtype=np.int64)
                for u in contribs
            ],
            delta_s,
        )
        units_dict[next_id] = frames
        next_id += 1
    return numpysorting_from_sample_indices(units_dict, fs)


def build_lazy_merged_sorting(
    abs_times, units_to_merge, timestamps, fs, *, delta_s
):
    """Reconstruct the lazily-merged ``NumpySorting`` from absolute times.

    Pure compute (no DB, no NWB IO) -- the merge-aware sibling of
    ``numpysorting_from_abs_times``. Applies a curation's PROPOSED merges
    (an ``apply_merge=False`` preview) without re-running the sort, in
    ABSOLUTE time so disjoint-recording wall-clock gaps are respected: the
    units NWB frames are contiguous across an excluded gap, so a frame-space
    dedup would wrongly drop a chunk-1-last / chunk-2-first pair that is
    seconds apart in real time.

    Parameters
    ----------
    abs_times : dict[int, np.ndarray]
        ``{unit_id: absolute spike times (seconds)}`` for every original
        unit (as read from the curated units NWB).
    units_to_merge : list[list[int]]
        Each inner list is a merge group's contributor unit ids; only
        multi-contributor groups (``len > 1``) belong here -- the caller
        filters the 1-element self-entries out.
    timestamps : np.ndarray
        The recording's (possibly gap-preserving) wall-clock timestamps.
    fs : float
        Sampling frequency of the recording.
    delta_s : float
        Coincidence window (seconds) for cross-unit duplicate removal.

    Returns
    -------
    si.NumpySorting
        Non-merged units keep their own abs times mapped to frames
        (identical to ``numpysorting_from_abs_times`` / ``get_sorting``);
        each merge group is abs-time-deduped with
        ``_dedup_merged_spike_times`` (the SAME helper the
        ``apply_merge=True`` staged path uses, so the previewed train is
        identical to the stored one) and assigned a fresh
        ``max(unit_ids) + 1`` id in ``units_to_merge`` order (matching the
        prior SI ``MergeUnitsSorting`` id assignment).
    """
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2._signal_math import (
        _dedup_merged_spike_times,
        _spike_times_to_frames,
    )

    n_samples = int(np.asarray(timestamps).size)
    merged_members = {int(u) for g in units_to_merge for u in g}
    units_dict: dict = {}
    # Non-merged units: map their own absolute times to frames (the same
    # mapping numpysorting_from_abs_times / get_sorting applies).
    for uid, st in abs_times.items():
        if int(uid) not in merged_members:
            units_dict[int(uid)] = _spike_times_to_frames(
                timestamps, np.asarray(st), n_samples, int(uid)
            )
    # Each merge group -> abs-time-deduped train mapped to frames, under a
    # fresh ``max(unit_ids) + 1`` id (in units_to_merge order).
    next_id = max(int(u) for u in abs_times) + 1
    for contribs in units_to_merge:
        deduped_abs = _dedup_merged_spike_times(
            [np.asarray(abs_times[int(u)]) for u in contribs], delta_s
        )
        units_dict[next_id] = _spike_times_to_frames(
            timestamps, deduped_abs, n_samples, next_id
        )
        next_id += 1
    return si.NumpySorting.from_unit_dict([units_dict], sampling_frequency=fs)


def _dedup_merged_spike_times_and_frames(times_list, frames_list, delta_s):
    """Return deduplicated ``(times, frames)`` with both arrays aligned."""
    import numpy as np

    time_arrays = [np.asarray(t, dtype=float) for t in times_list]
    frame_arrays = [np.asarray(f, dtype=np.int64) for f in frames_list]
    for times, frames in zip(time_arrays, frame_arrays):
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

    unit_ids = list(abs_times)
    return pd.DataFrame(
        {"spike_times": [abs_times[u] for u in unit_ids]},
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


def _recording_has_explicit_time_vector(recording) -> bool:
    """Return whether ``recording`` stores explicit timestamps."""
    try:
        return (
            recording.get_time_info(segment_index=0).get("time_vector")
            is not None
        )
    except AttributeError:
        # Compatibility with any older/custom extractor lacking get_time_info:
        # treat it as explicit so callers use the slice-based exact path.
        return True


def _sample_indices_to_times_by_unit(recording, sample_indices_by_unit):
    """Map stored sample frames to absolute times without full-vector allocation."""
    import numpy as np

    fs = float(recording.get_sampling_frequency())
    if not _recording_has_explicit_time_vector(recording):
        try:
            t_start = recording.get_start_time(segment_index=0)
        except TypeError:
            t_start = recording.get_start_time()
        except AttributeError:
            t_start = 0.0
        t_start = 0.0 if t_start is None else float(t_start)
        return {
            int(uid): np.asarray(frames, dtype=np.int64) / fs + t_start
            for uid, frames in sample_indices_by_unit.items()
        }

    n_samples = int(recording.get_num_samples(segment_index=0))

    def lookup(frames):
        frames = np.asarray(frames, dtype=np.int64)
        if frames.size == 0:
            return np.asarray([], dtype=np.float64)
        if np.any((frames < 0) | (frames >= n_samples)):
            raise ValueError(
                "spike_sample_index contains frame(s) outside the recording "
                f"range [0, {n_samples})."
            )
        # sample_index_to_time maps the sparse spike frames -> absolute times
        # directly without the caller materializing the full timestamp vector.
        return np.asarray(
            recording.sample_index_to_time(frames, segment_index=0),
            dtype=np.float64,
        )

    return {
        int(uid): lookup(frames)
        for uid, frames in sample_indices_by_unit.items()
    }


def _base_intervals_from_recording(recording, fs):
    """Return recorded time chunks without materializing full timestamps."""
    import numpy as np

    n_samples = int(recording.get_num_samples(segment_index=0))
    if n_samples == 0:
        return []
    if not _recording_has_explicit_time_vector(recording):
        try:
            t_start = recording.get_start_time(segment_index=0)
        except TypeError:
            t_start = recording.get_start_time()
        except AttributeError:
            t_start = 0.0
        t_start = 0.0 if t_start is None else float(t_start)
        return [[t_start, t_start + (n_samples - 1) / float(fs)]]

    sample_period = 1.0 / float(fs)
    chunk_size = max(1, int(round(float(fs))))
    intervals = []
    current_start = None
    prev_time = None
    for start_frame in range(0, n_samples, chunk_size):
        end_frame = min(n_samples, start_frame + chunk_size)
        # Bounded get_times() slices explicit time vectors in SI, so this caps
        # each gap-scan allocation to ~one second of samples.
        times = np.asarray(
            recording.get_times(
                segment_index=0,
                start_frame=start_frame,
                end_frame=end_frame,
            ),
            dtype=np.float64,
        )
        if times.size == 0:
            continue
        if current_start is None:
            current_start = float(times[0])
        elif float(times[0]) - prev_time > 1.5 * sample_period:
            intervals.append([float(current_start), float(prev_time)])
            current_start = float(times[0])
        gaps = np.flatnonzero(np.diff(times) > 1.5 * sample_period)
        for gap_idx in gaps:
            intervals.append([float(current_start), float(times[gap_idx])])
            current_start = float(times[gap_idx + 1])
        prev_time = float(times[-1])
    if current_start is not None:
        intervals.append([float(current_start), float(prev_time)])
    return intervals


def recording_timestamps(recording_row):
    """Return the full timestamp vector of the upstream Recording.

    Reads the persisted ``ElectricalSeries`` timestamps -- which for
    disjoint sort intervals are gap-preserving (non-uniform). The SI
    readback in ``get_sorting`` maps absolute spike times back to
    frames via ``np.searchsorted`` against this vector; the affine
    ``t_start + i/fs`` assumption is wrong across wall-clock gaps.
    Reads only the timestamps dataset (not the traces), so it is far
    lighter than loading the full SI recording.

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
    import numpy as np
    import pynwb

    from spyglass.common.common_nwbfile import AnalysisNwbfile

    abs_path = AnalysisNwbfile.get_abs_path(recording_row["analysis_file_name"])
    series_name = recording_row["electrical_series_path"].rsplit("/", 1)[-1]
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        nwbf = io.read()
        series = nwbf.acquisition[series_name]
        return np.asarray(series.timestamps[:], dtype=np.float64)


def write_sorting_units_nwb(
    sorting, recording, nwb_file_name, obs_intervals=None
):
    """Write a fresh AnalysisNwbfile containing only the v2 Units table.

    Spike times are stored in the recording's absolute timeline so downstream
    consumers can compare directly against the Recording's IntervalList
    valid_times. Spyglass also stores ``spike_sample_index`` (frame indices into
    the sorted recording) so ``get_sorting`` can reconstruct SI objects without
    reading the full recording timestamp vector.
    ``AnalysisNwbfile().create`` already strips any parent ``/units``
    from the analysis NWB so the sort outputs are the only Units rows
    in the file (addresses #1437).

    Every unit row carries ``obs_intervals`` (the artifact-
    removed valid-time window the sort observed) and a
    ``curation_label`` placeholder list (``["uncurated"]``), so
    external readers that grep for either column on a pre-curation
    NWB find them. ``obs_intervals`` defaults to the recording's full
    timestamp envelope when no artifact mask was applied
    (``obs_intervals=None``).
    """
    import numpy as np
    import pynwb

    from spyglass.common.common_nwbfile import AnalysisNwbfile

    analysis_file_name = AnalysisNwbfile().create(nwb_file_name=nwb_file_name)
    analysis_abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)

    sample_indices_by_unit = {
        int(unit_id): np.asarray(
            sorting.get_unit_spike_train(unit_id=unit_id), dtype=np.int64
        )
        for unit_id in sorting.unit_ids
    }
    spike_times_by_unit = _sample_indices_to_times_by_unit(
        recording, sample_indices_by_unit
    )
    sampling_frequency = float(recording.get_sampling_frequency())
    if obs_intervals is None:
        # ``obs_intervals is None`` is the "no artifact-detection pass" case:
        # the artifact-detection pass is optional (an ArtifactDetectionSource
        # part is zero-or-one; no part / artifact_detection_id=None means no
        # masking), so there is no artifact-removed IntervalList to read. The
        # recorded window(s) ARE the correct obs_intervals then -- the
        # sort observed every recorded sample. Split at wall-clock
        # discontinuities so a DISJOINT recording reports one interval
        # per recorded chunk rather than a single envelope spanning the
        # gaps (which would inflate the observation duration). For a
        # contiguous recording this collapses to a single
        # ``[t0, t_end]``, unchanged.
        obs_intervals_arr = np.asarray(
            _base_intervals_from_recording(recording, sampling_frequency)
        )
    else:
        obs_intervals_arr = np.asarray(obs_intervals)

    with pynwb.NWBHDF5IO(
        path=analysis_abs_path, mode="a", load_namespaces=True
    ) as io:
        nwbf = io.read()
        # ``curation_label`` is a scalar ``"uncurated"`` at sort
        # time, so external readers can do
        # ``nwb.units["curation_label"][i] == "uncurated"`` -- an
        # equality check that would silently fail against a list.
        # ``CurationV2.insert_curation`` rewrites this to the
        # indexed ragged-list shape at post-curation time. The
        # pre-vs-post shape discontinuity is intentional.
        #
        # ``add_unit_column`` must be declared BEFORE any
        # ``add_unit`` call that passes the column as a kwarg;
        # pynwb rejects the kwarg as "extra keys" otherwise.
        # The scalar shape uses no ``index=True``.
        if len(sorting.unit_ids) > 0:
            nwbf.add_unit_column(
                name="curation_label",
                description=(
                    'Curation label scalar; ``"uncurated"`` at '
                    "sort time, refined to a per-unit label list "
                    "by CurationV2.insert_curation."
                ),
            )
            nwbf.add_unit_column(
                name=SPIKE_SAMPLE_INDEX_COLUMN,
                description=(
                    "Sample indices into the sorted recording, one array per "
                    "unit. Stored alongside absolute spike_times so Spyglass "
                    "can reconstruct frame-based NumpySorting objects without "
                    "reading the full recording timestamp vector."
                ),
                index=True,
            )
        for unit_id in sorting.unit_ids:
            unit_id = int(unit_id)
            spike_indices = sample_indices_by_unit[unit_id]
            spike_times = spike_times_by_unit[unit_id]
            nwbf.add_unit(
                spike_times=spike_times,
                id=unit_id,
                obs_intervals=obs_intervals_arr,
                curation_label="uncurated",
                spike_sample_index=spike_indices,
            )
        # pynwb leaves ``nwbf.units = None`` if no add_unit() was
        # called, so a zero-unit sort would crash on .object_id.
        # Initialize an empty Units table explicitly.
        if nwbf.units is None:
            nwbf.units = pynwb.misc.Units(
                name="units",
                description="Empty units table (sorter found zero units).",
            )
        units_object_id = nwbf.units.object_id
        io.write(nwbf)

    # The AnalysisNwbfile DB-row registration (.add) is deliberately
    # NOT done here -- ``Sorting.make`` registers it inside its
    # ``transaction_or_noop`` block so the row rolls back atomically
    # if any of the master / Unit-part inserts fail.
    return analysis_file_name, units_object_id


def write_curated_units_nwb(
    sorting_id,
    kept_unit_to_contributors: dict,
    apply_merge: bool,
    labels: dict,
) -> tuple[str, str, str, dict]:
    """Write the curated-units NWB.

    Returns ``(analysis_file_name, units_object_id, nwb_file_name,
    n_spikes_by_uid)`` where ``n_spikes_by_uid`` maps each written
    unit_id to the length of its STORED (post cross-unit dedup) spike
    train, so the caller can override ``CurationV2.Unit.n_spikes`` and
    keep the ``n_spikes == len(get_sorting train)`` invariant.

    With ``apply_merge=True`` the kept unit's spike train is the
    sorted union of its contributors' spike trains and its id is a
    fresh ``max(source unit_ids) + 1`` assigned in ascending
    min-contributor order (so the lazy ``get_merged_sorting`` preview
    assigns matching ids -- see ``build_curated_unit_rows``); the
    absorbed contributors are dropped from both the NWB and
    ``CurationV2.Unit``. Surviving source units are written first (in
    source order) and merged ids are appended in that same
    ascending-min order.

    With ``apply_merge=False`` (preview) every original unit is
    written 1:1 -- contributors included -- so the proposed merge
    can be reviewed before committing; the merge structure lives in
    ``CurationV2.MergeGroup`` and is reconstructed by
    ``get_merged_sorting`` on demand.
    """
    import numpy as np
    import pynwb

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._enums import CurationLabel
    from spyglass.spikesorting.v2._signal_math import _MERGE_DEDUP_DELTA_MS
    from spyglass.spikesorting.v2.sorting import Sorting
    from spyglass.spikesorting.v2.utils import _dedup_merged_spike_times

    # Anchor the curated-units NWB to the same parent as the Sorting (the sort's
    # own session, or the first SessionGroup.Member for a concat source). The
    # curated absolute spike times are read from the Sorting units NWB below
    # (source-agnostic), so only the parent-file anchor differs by source kind.
    nwb_file_name = Sorting.resolve_anchor_nwb_file_name(
        {"sorting_id": sorting_id}
    )

    # Source the pre-curation units' ABSOLUTE spike times plus Spyglass's
    # sample-frame sidecar straight from the Sorting units NWB. Absolute seconds
    # keep the curated NWB interoperable and gap-correct; sample frames let
    # Spyglass reconstruct sortings without reading the full recording timeline.
    src_abs_path = AnalysisNwbfile.get_abs_path(
        (Sorting & {"sorting_id": sorting_id}).fetch1("analysis_file_name")
    )
    abs_times_by_uid = read_units_abs_spike_times(src_abs_path)
    sample_indices_by_uid = read_units_spike_sample_indices(src_abs_path)

    analysis_file_name = AnalysisNwbfile().create(nwb_file_name=nwb_file_name)
    analysis_abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)

    with pynwb.NWBHDF5IO(
        path=analysis_abs_path, mode="a", load_namespaces=True
    ) as io:
        nwbf = io.read()
        # Add the ``curation_label`` column ONLY when at least one
        # unit will be written. Adding a column ahead of any
        # ``add_unit`` call would create an empty Units table whose
        # ``curation_label`` column has no rows; pynwb's writer
        # then fails dtype inference at ``io.write`` with
        # "Cannot infer dtype of empty list or tuple". For the
        # empty-curation case (zero kept units after filtering, or
        # the contrived all-merged-away case) we initialize a
        # bare ``pynwb.misc.Units`` without the column so the
        # write succeeds.
        # Resolve which units get written + their spike trains:
        #   apply_merge=True  -> kept units; a merged head gets the
        #     concatenated contributor trains.
        #   apply_merge=False -> every original unit 1:1 (preview);
        #     proposed merges stay in MergeGroup for lazy application
        #     via get_merged_sorting.
        if apply_merge:
            write_specs = []
            for kept_uid, contribs in kept_unit_to_contributors.items():
                if len(contribs) > 1:
                    # Membership-aware 0.4 ms dedup of cross-unit
                    # double-detections (a neuron's refractory period
                    # makes any sub-0.4 ms cross-unit pair one physical
                    # spike). Uses the same dedup as the lazy
                    # get_merged_sorting, so the stored
                    # (apply_merge=True) train equals the previewed one.
                    if sample_indices_by_uid is not None:
                        spike_times, spike_indices = (
                            _dedup_merged_spike_times_and_frames(
                                [abs_times_by_uid[int(u)] for u in contribs],
                                [
                                    sample_indices_by_uid[int(u)]
                                    for u in contribs
                                ],
                                _MERGE_DEDUP_DELTA_MS / 1000.0,
                            )
                        )
                    else:
                        spike_times = _dedup_merged_spike_times(
                            [abs_times_by_uid[int(u)] for u in contribs],
                            _MERGE_DEDUP_DELTA_MS / 1000.0,
                        )
                        spike_indices = None
                else:
                    spike_times = abs_times_by_uid[int(kept_uid)]
                    spike_indices = (
                        None
                        if sample_indices_by_uid is None
                        else sample_indices_by_uid[int(kept_uid)]
                    )
                write_specs.append((int(kept_uid), spike_times, spike_indices))
        else:
            write_specs = [
                (
                    int(uid),
                    abs_times_by_uid[int(uid)],
                    (
                        None
                        if sample_indices_by_uid is None
                        else sample_indices_by_uid[int(uid)]
                    ),
                )
                for uid in sorted(abs_times_by_uid)
            ]
        # ``n_spikes`` per written unit is the length of its STORED
        # (post-dedup) train, so the invariant
        # ``CurationV2.Unit.n_spikes == len(get_sorting train)`` holds
        # even after cross-unit dedup removes double-detections. The
        # caller overrides ``unit_rows`` with this map.
        n_spikes_by_uid = {
            int(uid): int(len(spike_times))
            for uid, spike_times, _spike_indices in write_specs
        }

        if write_specs:
            # ``curation_label`` is written as an ``index=True``
            # (ragged) column with a per-unit list of label
            # strings. External readers do
            # ``list(nwb_sorting.get('curation_label', []))`` and
            # expect a list per unit -- they would misparse a
            # comma-separated string by splitting on every
            # character. The ``CurationV2.UnitLabel`` docstring
            # already described this shape.
            #
            # Call ``add_unit(...)`` for every unit FIRST, then add
            # the column with ``data=label_values`` AFTER -- this
            # gives pynwb a full per-unit list-of-lists to infer
            # dtype from. Pre-declaring the column and passing labels
            # per ``add_unit`` makes pynwb fail dtype inference when
            # all labels happen to be empty (the no-labels case).
            if sample_indices_by_uid is not None:
                nwbf.add_unit_column(
                    name=SPIKE_SAMPLE_INDEX_COLUMN,
                    description=(
                        "Sample indices into the curated recording, one array "
                        "per unit. Mirrors spike_times for Spyglass frame-based "
                        "readback without reading the full timestamp vector."
                    ),
                    index=True,
                )
            all_labels: list[list[str]] = []
            for unit_id, spike_times, spike_indices in write_specs:
                lbl_list = labels.get(int(unit_id), [])
                label_list = [CurationLabel.normalize(lbl) for lbl in lbl_list]
                all_labels.append(label_list)
                unit_kwargs = {
                    "spike_times": np.asarray(spike_times, dtype=np.float64),
                    "id": int(unit_id),
                }
                if sample_indices_by_uid is not None:
                    unit_kwargs[SPIKE_SAMPLE_INDEX_COLUMN] = np.asarray(
                        spike_indices, dtype=np.int64
                    )
                nwbf.add_unit(**unit_kwargs)
            # Only add the column when at least one unit
            # carries a non-empty label list. pynwb's dtype
            # inference fails on an all-empty list-of-lists
            # ("Cannot infer dtype of empty list"); the
            # column-missing case is handled by downstream
            # readers via ``nwb_sorting.get('curation_label',
            # [])``.
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
            # Empty curation: initialize an empty Units table so
            # ``.object_id`` is defined and ``io.write`` does not
            # try to infer a dtype for any column.
            nwbf.units = pynwb.misc.Units(
                name="units",
                description=("Empty units table (curation kept zero units)."),
            )
        units_object_id = nwbf.units.object_id
        io.write(nwbf)

    # The AnalysisNwbfile DB-row registration (.add) is deliberately
    # NOT done here -- the caller does it inside its transaction
    # block so the row rolls back atomically if any of the
    # CurationV2 / Unit / UnitLabel / merge-insert steps fail.
    # The file on disk is the only side effect left to clean up on
    # rollback.
    return (
        analysis_file_name,
        units_object_id,
        nwb_file_name,
        n_spikes_by_uid,
    )
