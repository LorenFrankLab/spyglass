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


def read_units_abs_times_and_sample_indices(abs_path, *, unit_ids=None):
    """Open the units NWB ONCE; return ``(abs_times, sample_indices, obs)``.

    Combined reader for callers that need these columns (curated-units write +
    lazy-merge preview), so a single ``NWBHDF5IO`` open replaces several.
    ``abs_times`` is ``{unit_id: abs_spike_times}`` (``{}`` for an empty/absent
    Units table); ``sample_indices`` is ``{unit_id: spike_sample_index}``,
    ``None`` when the ``spike_sample_index`` column is absent (legacy/manual
    files), or ``{}`` for an empty Units table; ``obs`` is
    ``{unit_id: obs_intervals}`` (the per-unit ``(n, 2)`` observation window),
    ``None`` when the ``obs_intervals`` column is absent (legacy files), or
    ``{}`` for an empty Units table. The curated writer carries ``obs`` forward
    so a curated export keeps the correct observation window (CNEP-1); without
    it, NWB-only firing-rate / presence-ratio / duration denominators over a
    curated export silently assume the full session.

    ``unit_ids`` (optional iterable of int): when given, read ONLY those units'
    spike trains / sample frames / obs rather than every unit -- so a curation
    that keeps a subset of a large multi-day sort never materializes the
    discarded units' data. ``None`` (the default) reads every unit, for the
    preview / full-write paths that need all of them (see
    ``curation_source_unit_ids``). A requested id absent from the table is
    skipped: the caller's kept-set is authoritative, and a genuinely-missing
    source unit surfaces as a downstream KeyError exactly as it did before this
    filter existed.
    """
    import numpy as np
    import pynwb

    wanted = None if unit_ids is None else {int(u) for u in unit_ids}
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        nwbf = io.read()
        units = nwbf.units
        if units is None or len(units) == 0:
            return {}, {}, {}
        rows = [
            (row_ind, int(uid))
            for row_ind, uid in enumerate(np.asarray(units.id[:], dtype=int))
            if wanted is None or int(uid) in wanted
        ]
        spike_times = units["spike_times"]
        abs_times = {
            uid: np.asarray(spike_times[row_ind], dtype=float)
            for row_ind, uid in rows
        }
        if SPIKE_SAMPLE_INDEX_COLUMN in units.colnames:
            sample_col = units[SPIKE_SAMPLE_INDEX_COLUMN]
            sample_indices = {
                uid: np.asarray(sample_col[row_ind], dtype=np.int64)
                for row_ind, uid in rows
            }
        else:
            sample_indices = None
        # ``obs_intervals`` is a built-in (optional) NWB Units column; legacy /
        # hand-written files may omit it -> ``None``.
        if "obs_intervals" in units.colnames:
            obs_col = units["obs_intervals"]
            obs = {
                uid: np.asarray(obs_col[row_ind], dtype=float)
                for row_ind, uid in rows
            }
        else:
            obs = None
        return abs_times, sample_indices, obs


def curation_source_unit_ids(kept_unit_to_contributors, apply_merge):
    """Return the source unit ids ``write_curated_units_nwb`` will actually read.

    Mirrors ``_write_curated_units_nwb_body``'s access so the curated-units write
    reads only the kept subset rather than every source unit (a large multi-day
    sort otherwise materializes the discarded units' trains too). With
    ``apply_merge=True`` a multi-contributor kept unit reads each contributor's
    train and a single-contributor kept unit reads its own (by its kept id == the
    surviving source id); the fresh merged-head id is never read from the source
    file. With ``apply_merge=False`` every original unit is written 1:1
    (preview), so this returns ``None`` -> read all.

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
            needed.update(int(u) for u in contribs)
        else:
            needed.add(int(kept_uid))
    return needed


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


def _sample_indices_to_times_by_unit(recording, sample_indices_by_unit):
    """Map stored sample frames to absolute times without full-vector allocation."""
    import numpy as np

    from spyglass.spikesorting.v2._signal_math import (
        _recording_has_explicit_time_vector,
    )

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
        # ``sample_index_to_time`` indexes the h5py-/mmap-backed timestamp
        # vector, whose fancy-indexing requires STRICTLY INCREASING indices.
        # A unit's spike frames are not guaranteed sorted (and may repeat), so
        # map the unique-sorted frames and broadcast the times back into the
        # original frame order -- still sparse, no full-vector materialization.
        order = np.argsort(frames, kind="stable")
        uniq, inverse = np.unique(frames[order], return_inverse=True)
        uniq_times = np.asarray(
            recording.sample_index_to_time(uniq, segment_index=0),
            dtype=np.float64,
        )
        out = np.empty(frames.shape, dtype=np.float64)
        out[order] = uniq_times[inverse]
        return out

    return {
        int(uid): lookup(frames)
        for uid, frames in sample_indices_by_unit.items()
    }


def _base_intervals_from_recording(recording, fs):
    """Return recorded time chunks without materializing full timestamps.

    The chunked/affine scan now lives in ``_signal_math.base_intervals_and_gaps``
    (which generalizes it to also emit the inter-chunk gap frame indices the
    artifact path needs); this writer only needs the per-chunk base intervals.
    """
    from spyglass.spikesorting.v2._signal_math import base_intervals_and_gaps

    return base_intervals_and_gaps(recording, fs).base_intervals


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
    sorting,
    recording,
    nwb_file_name,
    obs_intervals=None,
    *,
    unit_metadata=None,
    source_provenance=None,
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
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    analysis_file_name = AnalysisNwbfile().create(nwb_file_name=nwb_file_name)
    # ``create`` already wrote a stub file to disk; if any step below raises
    # before the caller registers the AnalysisNwbfile row, unlink that orphan
    # (mirrors the recording writer -- the caller's staging cleanup only knows
    # the analyzer folder, not this file's name).
    try:
        return _write_sorting_units_nwb_body(
            analysis_file_name=analysis_file_name,
            sorting=sorting,
            recording=recording,
            obs_intervals=obs_intervals,
            unit_metadata=unit_metadata,
            source_provenance=source_provenance,
        )
    except Exception:
        from spyglass.spikesorting.v2.recording import (
            _unlink_staged_analysis_file,
        )

        _unlink_staged_analysis_file(
            analysis_file_name, context="write_sorting_units_nwb"
        )
        raise


def _write_sorting_units_nwb_body(
    *,
    analysis_file_name,
    sorting,
    recording,
    obs_intervals,
    unit_metadata=None,
    source_provenance=None,
):
    """Fill the staged sort-units ``AnalysisNwbfile`` (no cleanup on failure).

    Split out of :func:`write_sorting_units_nwb` so the staged-file cleanup-on-
    error wrapper there stays a thin try/except. Returns
    ``(analysis_file_name, units_object_id)``.

    ``unit_metadata`` (``{unit_id: {peak_amplitude_uv, peak_electrode_id,
    n_spikes, brain_region}}``) adds the matching per-unit columns -- the SAME
    values used for ``Sorting.Unit`` (computed once). ``source_provenance``
    (from :mod:`._nwb_provenance`) is embedded as a scratch header so the file
    is interpretable without the DB.
    """
    import numpy as np
    import pynwb

    from spyglass.common.common_nwbfile import AnalysisNwbfile

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
            if unit_metadata is not None:
                # Per-unit metadata mirroring Sorting.Unit (computed once),
                # so an NWB-only reader has peak channel / amplitude / count /
                # region without the DB.
                for name, desc in (
                    ("peak_amplitude_uv", "Peak template amplitude (uV)."),
                    ("peak_electrode_id", "Peak channel spyglass electrode id."),
                    ("n_spikes", "Number of spikes in the unit."),
                    ("brain_region", "Brain region of the peak electrode."),
                ):
                    nwbf.add_unit_column(name=name, description=desc)
        for unit_id in sorting.unit_ids:
            unit_id = int(unit_id)
            spike_indices = sample_indices_by_unit[unit_id]
            spike_times = spike_times_by_unit[unit_id]
            unit_kwargs = dict(
                spike_times=spike_times,
                id=unit_id,
                obs_intervals=obs_intervals_arr,
                curation_label="uncurated",
                spike_sample_index=spike_indices,
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
        # pynwb leaves ``nwbf.units = None`` if no add_unit() was
        # called, so a zero-unit sort would crash on .object_id.
        # Initialize an empty Units table explicitly.
        if nwbf.units is None:
            nwbf.units = pynwb.misc.Units(
                name="units",
                description="Empty units table (sorter found zero units).",
            )
        units_object_id = nwbf.units.object_id
        if source_provenance is not None:
            from spyglass.spikesorting.v2._nwb_provenance import (
                SORTING_PROVENANCE,
                build_provenance_table,
            )

            nwbf.add_scratch(
                build_provenance_table(SORTING_PROVENANCE, source_provenance)
            )
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
    *,
    source_units_abs_path: str | None = None,
    curation_header: dict | None = None,
) -> tuple[str, str, str, dict]:
    """Write the curated-units NWB.

    Returns ``(analysis_file_name, units_object_id, nwb_file_name,
    n_spikes_by_uid)`` where ``n_spikes_by_uid`` maps each written
    unit_id to the length of its STORED (post cross-unit dedup) spike
    train, so the caller can override ``CurationV2.Unit.n_spikes`` and
    keep the ``n_spikes == len(get_sorting train)`` invariant.

    ``source_units_abs_path`` selects WHERE the source spike trains are read
    from. ``None`` (a root curation) reads the raw ``Sorting`` units NWB, so
    ``kept_unit_to_contributors`` is in the raw-sort unit namespace. A child
    curation passes its PARENT curation's units NWB; the kept/contributor ids
    are then in the parent's unit namespace, so a merged-parent id composes
    correctly and absorbed raw contributors are not resurrected.

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
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.sorting import Sorting

    # Anchor the curated-units NWB to the same parent as the Sorting (the sort's
    # own session, or the first SessionGroup.Member for a concat source). The
    # curated absolute spike times are read from the Sorting units NWB below
    # (source-agnostic), so only the parent-file anchor differs by source kind.
    nwb_file_name = Sorting.resolve_anchor_nwb_file_name(
        {"sorting_id": sorting_id}
    )

    # Source the units' ABSOLUTE spike times plus Spyglass's sample-frame
    # sidecar. A root curation reads the raw Sorting units NWB; a child reads
    # its PARENT curation's units NWB (``source_units_abs_path``) so stored
    # frames, dedup, and obs_intervals compose from the actual parent state.
    # Absolute seconds keep the curated NWB interoperable and gap-correct;
    # sample frames let Spyglass reconstruct sortings without reading the full
    # recording timeline.
    if source_units_abs_path is None:
        src_abs_path = AnalysisNwbfile.get_abs_path(
            (Sorting & {"sorting_id": sorting_id}).fetch1("analysis_file_name")
        )
    else:
        src_abs_path = source_units_abs_path
    # Read ONLY the source units this curation will write: kept singletons +
    # merge contributors for apply_merge=True, every unit for the apply_merge=
    # False preview (see curation_source_unit_ids). A large multi-day sort
    # otherwise materializes the discarded units' spike trains here too.
    abs_times_by_uid, sample_indices_by_uid, obs_intervals_by_uid = (
        read_units_abs_times_and_sample_indices(
            src_abs_path,
            unit_ids=curation_source_unit_ids(
                kept_unit_to_contributors, apply_merge
            ),
        )
    )

    analysis_file_name = AnalysisNwbfile().create(nwb_file_name=nwb_file_name)
    # ``create`` staged a stub file; unlink it if the write below fails before
    # the caller registers the AnalysisNwbfile row (curation stages OUTSIDE its
    # rows-transaction try, so its cleanup does not otherwise cover this file).
    try:
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
        )
    except Exception:
        from spyglass.spikesorting.v2.recording import (
            _unlink_staged_analysis_file,
        )

        _unlink_staged_analysis_file(
            analysis_file_name, context="write_curated_units_nwb"
        )
        raise


def _curated_obs_intervals(kept_uid, contribs, apply_merge, obs_intervals_by_uid):
    """Per-unit ``obs_intervals`` for one curated/kept unit, or ``None``.

    ``None`` when the source NWB carried no ``obs_intervals`` column (legacy).
    A merged unit (``apply_merge`` with >1 contributor) gets the INTERSECTION of
    its contributors' windows -- the conservative choice, so a unit is reported
    observed only where EVERY contributor was observed; in practice every unit of
    one sort shares the same window, so the intersection equals that shared
    window. A singleton / preview unit keeps its own window.
    """
    if obs_intervals_by_uid is None:
        return None
    from spyglass.spikesorting.v2._signal_math import intersect_interval_sets

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
    curation_header=None,
):
    """Fill the staged curated-units ``AnalysisNwbfile`` (no cleanup on error).

    Split out of :func:`write_curated_units_nwb` so its staged-file cleanup-on-
    error wrapper stays a thin try/except. Returns ``(analysis_file_name,
    units_object_id, nwb_file_name, n_spikes_by_uid)``. ``obs_intervals_by_uid``
    (``{unit_id: (n, 2) array}`` or ``None`` for a legacy source) carries the
    per-unit observation window forward so a curated export keeps the correct
    firing-rate / presence-ratio / duration denominator (CNEP-1).
    """
    import numpy as np
    import pynwb

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._enums import CurationLabel
    from spyglass.spikesorting.v2._signal_math import _MERGE_DEDUP_DELTA_MS
    from spyglass.spikesorting.v2.utils import _dedup_merged_spike_times

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
                    (
                        None
                        if sample_indices_by_uid is None
                        else sample_indices_by_uid[int(uid)]
                    ),
                    _curated_obs_intervals(
                        uid, [uid], apply_merge, obs_intervals_by_uid
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
            for uid, spike_times, _spike_indices, _obs in write_specs
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
            for unit_id, spike_times, spike_indices, obs in write_specs:
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
                # CNEP-1: carry the per-unit observation window forward so a
                # curated export keeps the correct firing-rate / presence-ratio
                # / duration denominator. Omitted only for a legacy source NWB
                # that had no obs_intervals column (obs is None).
                if obs is not None:
                    unit_kwargs["obs_intervals"] = np.asarray(
                        obs, dtype=np.float64
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
        # Self-describing provenance: the curation header (identity/source) and
        # the kept->contributor merge lineage (matching CurationV2.MergeGroup),
        # flagged applied (apply_merge=True) or proposed (apply_merge=False), so
        # the merge structure travels in the file, not only in the DB.
        from spyglass.spikesorting.v2._nwb_provenance import (
            CURATION_MERGE_LINEAGE,
            CURATION_PROVENANCE,
            build_long_provenance_table,
            build_provenance_table,
        )

        if curation_header is not None:
            nwbf.add_scratch(
                build_provenance_table(CURATION_PROVENANCE, curation_header)
            )
        merge_lineage_rows = [
            {
                "kept_unit_id": int(kept_uid),
                "contributor_unit_id": int(contributor),
                "applied": bool(apply_merge),
            }
            for kept_uid, contribs in kept_unit_to_contributors.items()
            if len(contribs) > 1
            for contributor in contribs
        ]
        nwbf.add_scratch(
            build_long_provenance_table(
                CURATION_MERGE_LINEAGE,
                merge_lineage_rows,
                [
                    ("kept_unit_id", int),
                    ("contributor_unit_id", int),
                    ("applied", bool),
                ],
            )
        )
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
