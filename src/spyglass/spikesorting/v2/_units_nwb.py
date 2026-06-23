"""Units-NWB read/write IO for v2 sorts.

These functions are the units-NWB IO core behind ``Sorting`` and
``CurationV2``: reading a units NWB's stored ABSOLUTE spike times,
mapping those times back to recording frames, reading the recording's
persisted timestamp vector, writing the pre-curation sorting-units NWB
(``write_sorting_units_nwb``), and writing the post-curation curated-units
NWB (``write_curated_units_nwb``). Most take already-resolved paths /
SpikeInterface objects / fetched row dicts and do pure pynwb IO;
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


def read_units_abs_spike_times(abs_path) -> dict:
    """Return ``{unit_id(int): abs_spike_times(np.ndarray seconds)}``.

    Reads the stored absolute spike times directly from a v2 units
    NWB (``nwbf.units.to_dataframe()``), so callers get the persisted
    wall-clock values exactly -- no affine round-trip. Returns ``{}``
    for an empty/absent Units table.

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
        if nwbf.units is None or len(nwbf.units) == 0:
            return {}
        units_df = nwbf.units.to_dataframe()
    return {
        int(uid): np.asarray(st, dtype=float)
        for uid, st in units_df["spike_times"].items()
    }


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

    Spike times are stored in the recording's absolute timeline
    (``timestamps[sample_index]``) so downstream consumers can compare
    directly against the Recording's IntervalList valid_times.
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

    timestamps = recording.get_times()
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
        from spyglass.spikesorting.v2.utils import (
            _base_intervals_from_timestamps,
        )

        obs_intervals_arr = np.asarray(
            _base_intervals_from_timestamps(
                timestamps, recording.get_sampling_frequency()
            )
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
        for unit_id in sorting.unit_ids:
            spike_indices = sorting.get_unit_spike_train(unit_id=unit_id)
            # Map sample indices into the recording's wall-clock so
            # the stored spike times match Recording.get_times()
            # exactly.
            spike_times = timestamps[spike_indices]
            nwbf.add_unit(
                spike_times=spike_times,
                id=int(unit_id),
                obs_intervals=obs_intervals_arr,
                curation_label="uncurated",
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
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2.utils import _dedup_merged_spike_times

    # Anchor the curated-units NWB to the same parent as the Sorting: the
    # sort's own session for a single-recording source, or the FIRST
    # SessionGroup.Member for a concat source. The curated absolute spike times
    # are read from the Sorting units NWB below (source-agnostic), so only the
    # parent-file anchor differs between the two source kinds.
    source = SortingSelection.resolve_source({"sorting_id": sorting_id})
    if source.kind == "recording":
        nwb_file_name = (
            RecordingSelection & {"recording_id": source.key["recording_id"]}
        ).fetch1("nwb_file_name")
    else:  # concatenated_recording
        _anchor_recording_id, nwb_file_name, _preproc = (
            Sorting._resolve_concat_anchor(source.key)
        )

    # Source the pre-curation units' ABSOLUTE spike times straight
    # from the Sorting units NWB. Reading absolute seconds (not a
    # frame-based SI object) keeps the curated NWB's stored times
    # exact and gap-correct for disjoint recordings -- a frame->time
    # round-trip through a NumpySorting (t_start=0) would drop the
    # absolute offset and the wall-clock gaps.
    src_abs_path = AnalysisNwbfile.get_abs_path(
        (Sorting & {"sorting_id": sorting_id}).fetch1("analysis_file_name")
    )
    abs_times_by_uid = read_units_abs_spike_times(src_abs_path)

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
                    spike_times = _dedup_merged_spike_times(
                        [abs_times_by_uid[int(u)] for u in contribs],
                        _MERGE_DEDUP_DELTA_MS / 1000.0,
                    )
                else:
                    spike_times = abs_times_by_uid[int(kept_uid)]
                write_specs.append((int(kept_uid), spike_times))
        else:
            write_specs = [
                (int(uid), abs_times_by_uid[int(uid)])
                for uid in sorted(abs_times_by_uid)
            ]
        # ``n_spikes`` per written unit is the length of its STORED
        # (post-dedup) train, so the invariant
        # ``CurationV2.Unit.n_spikes == len(get_sorting train)`` holds
        # even after cross-unit dedup removes double-detections. The
        # caller overrides ``unit_rows`` with this map.
        n_spikes_by_uid = {
            int(uid): int(len(spike_times)) for uid, spike_times in write_specs
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
            all_labels: list[list[str]] = []
            for unit_id, spike_times in write_specs:
                lbl_list = labels.get(int(unit_id), [])
                label_list = [
                    CurationLabel.normalize(lbl) for lbl in lbl_list
                ]
                all_labels.append(label_list)
                nwbf.add_unit(
                    spike_times=np.asarray(spike_times, dtype=np.float64),
                    id=int(unit_id),
                )
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
