"""Pure and SpikeInterface helpers behind the concatenated-recording cache.

DB-free at import; SpikeInterface is imported lazily.
"""

from __future__ import annotations


def member_recording_selection_key(
    member: dict, preprocessing_params_name: str
) -> dict:
    """Build the ``RecordingSelection`` key for one ``SessionGroup.Member``.

    A changed sort group can leave several selections for this key; restrict
    on the member's ``recording_input_hash`` to get one.

    Parameters
    ----------
    member : dict
        A ``SessionGroup.Member`` row (or member dict) carrying
        ``nwb_file_name``, ``sort_group_id``, ``interval_list_name``,
        ``team_name``.
    preprocessing_params_name : str
        The shared preprocessing recipe on the concat selection.

    Returns
    -------
    dict
        The ``RecordingSelection`` key for that member.
    """
    return {
        "nwb_file_name": member["nwb_file_name"],
        "sort_group_id": member["sort_group_id"],
        "interval_list_name": member["interval_list_name"],
        "preprocessing_params_name": preprocessing_params_name,
        "team_name": member["team_name"],
    }


def member_split_key(member: dict) -> tuple:
    """Hashable per-member key for ``split_sorting_by_session`` output.

    All four fields are needed: ``create_group`` permits the same spatial
    member under two teams.

    Parameters
    ----------
    member : dict
        A ``SessionGroup.Member`` row carrying ``nwb_file_name``,
        ``sort_group_id``, ``interval_list_name``, ``team_name``.

    Returns
    -------
    tuple[str, int, str, str]
        The hashable member identity key.
    """
    return (
        member["nwb_file_name"],
        int(member["sort_group_id"]),
        member["interval_list_name"],
        member["team_name"],
    )


#: Per-member identity fields hashed by :func:`member_set_hash`. Content hashes
#: are verified at rebuild, not hashed, so a reproducing rebuild keeps the id.
MEMBER_SNAPSHOT_LOGICAL_FIELDS = (
    "member_index",
    "nwb_file_name",
    "sort_group_id",
    "interval_list_name",
    "team_name",
    "recording_id",
    "artifact_detection_id",
)


def member_set_hash(snapshot_rows: list[dict]) -> str:
    """Return the content-addressed hash of an ordered concat member set.

    Hashes :data:`MEMBER_SNAPSHOT_LOGICAL_FIELDS` in ``member_index`` order,
    so a changed member set mints a new ``concat_recording_id``. Content
    drift is caught at rebuild (``ConcatMemberDriftError``).

    Parameters
    ----------
    snapshot_rows : list of dict
        Member-snapshot rows, each carrying at least
        :data:`MEMBER_SNAPSHOT_LOGICAL_FIELDS`. Order is irrelevant (rows are
        sorted by ``member_index``); extra keys (e.g. ``recording_content_hash``)
        are ignored.

    Returns
    -------
    str
        A 64-character lowercase hex SHA-256 digest of the ordered member set.
    """
    from spyglass.spikesorting.v2._core.selection_identity import (
        canonical_identity,
        sha256_json,
    )

    ordered = sorted(snapshot_rows, key=lambda row: int(row["member_index"]))
    return sha256_json(
        [
            canonical_identity(
                {field: row[field] for field in MEMBER_SNAPSHOT_LOGICAL_FIELDS}
            )
            for row in ordered
        ]
    )


def concat_recording_artifact_lock(concat_recording_id):
    """Return the cross-process lock for one concat artifact's slot.

    See :func:`._storage.analyzer_cache.artifact_slot_lock`.

    Parameters
    ----------
    concat_recording_id
        The concat whose canonical artifact the caller will mutate.

    Returns
    -------
    filelock.FileLock
        An unacquired lock.
    """
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        artifact_slot_lock,
    )

    return artifact_slot_lock("concat_recording", concat_recording_id)


def cumulative_member_boundaries(
    num_samples_per_member: list[int],
) -> list[int]:
    """Return the cumulative end-sample boundary for each member.

    Member ``i`` occupies ``[boundaries[i-1], boundaries[i])``, with 0 as
    the start of member 0.

    Parameters
    ----------
    num_samples_per_member : list[int]
        Per-member sample counts, ordered by ``member_index``.

    Returns
    -------
    list[int]
        Cumulative end-sample boundaries, same length as the input.
    """
    import numpy as np

    counts = [int(n) for n in num_samples_per_member]
    return np.cumsum(counts, dtype=np.int64).tolist()


def split_unit_spike_trains(
    unit_spike_trains: dict,
    boundaries: list[int],
    *,
    total_n_samples: int | None = None,
) -> list[dict]:
    """Slice concat-frame spike trains into per-member local frames.

    Every unit id appears in every member, possibly empty. Every spike must
    land in exactly one member, or this raises.

    Parameters
    ----------
    unit_spike_trains : dict[int, numpy.ndarray]
        ``{unit_id: concat-frame spike indices}`` for the concatenated sort.
    boundaries : list[int]
        Cumulative end-sample boundaries from
        :func:`cumulative_member_boundaries` (one per member, ordered by
        ``member_index``). Must be non-empty and strictly increasing.
    total_n_samples : int, optional
        The concatenated recording's sample count. When supplied, the final
        boundary must equal it (the boundary set covers the full recording).

    Returns
    -------
    list[dict[int, numpy.ndarray]]
        One ``{unit_id: local-frame spike indices}`` dict per member, in
        ``member_index`` order.

    Raises
    ------
    ConcatSplitError
        If the boundaries are empty / not strictly increasing, do not cover
        ``total_n_samples``, or any input spike frame falls outside
        ``[0, boundaries[-1])``.
    """
    import numpy as np

    from spyglass.spikesorting.v2.exceptions import ConcatSplitError

    if not boundaries:
        raise ConcatSplitError(
            "split_unit_spike_trains: no member boundaries; a concatenated "
            "recording must have at least one member boundary to split back."
        )
    previous = 0
    for index, end in enumerate(boundaries):
        end = int(end)
        if end <= previous:
            raise ConcatSplitError(
                "split_unit_spike_trains: member boundaries must be strictly "
                f"increasing, but boundary {index} ({end}) is not greater than "
                f"the previous boundary ({previous}). Overlapping or empty "
                "member intervals would assign a spike to two members or none."
            )
        previous = end
    total = int(boundaries[-1])
    if total_n_samples is not None and total != int(total_n_samples):
        raise ConcatSplitError(
            "split_unit_spike_trains: the final member boundary "
            f"({total}) does not equal the concatenated recording's sample "
            f"count ({int(total_n_samples)}); the boundary set does not cover "
            "the whole recording, so trailing spikes would be dropped."
        )

    dropped = {}
    for unit_id, frames in unit_spike_trains.items():
        frames = np.asarray(frames)
        out_of_range = int(np.count_nonzero((frames < 0) | (frames >= total)))
        if out_of_range:
            dropped[int(unit_id)] = out_of_range
    if dropped:
        n_dropped = sum(dropped.values())
        raise ConcatSplitError(
            f"split_unit_spike_trains: {n_dropped} spike(s) fall outside the "
            f"concatenated frame range [0, {total}) and would be dropped "
            f"(per-unit out-of-range counts: {dropped}). Every concat-frame "
            "spike must map to exactly one member; check the sorting frame "
            "alignment and the MemberBoundary set."
        )

    per_member: list[dict] = []
    start = 0
    for end in boundaries:
        end = int(end)
        member_units: dict = {}
        for unit_id, frames in unit_spike_trains.items():
            frames = np.asarray(frames)
            in_member = frames[(frames >= start) & (frames < end)] - start
            member_units[unit_id] = in_member.astype(np.int64, copy=False)
        per_member.append(member_units)
        start = end
    return per_member


def split_spike_frames_by_spans(
    unit_spike_trains: dict, spans: list
) -> list[dict]:
    """Slice a sort's spike frames by its constituent recordings' frame spans.

    Spans must be contiguous from frame 0; every spike must land in exactly
    one span, or this raises.

    Parameters
    ----------
    unit_spike_trains : dict[int, numpy.ndarray]
        ``{unit_id: spike frames in the sort's frame space}``.
    spans : list of (int, int)
        Half-open ``[start_sample, end_sample)`` frame span of each
        constituent recording, in recording order.

    Returns
    -------
    list[dict[int, numpy.ndarray]]
        One ``{unit_id: local-frame spike indices}`` dict per span (frames
        relative to the span's start), every unit id in every span.

    Raises
    ------
    ConcatSplitError
        If the spans are empty or not contiguous from frame 0, or a spike
        falls outside them (see :func:`split_unit_spike_trains`).
    """
    from spyglass.spikesorting.v2.exceptions import ConcatSplitError

    if not spans:
        raise ConcatSplitError(
            "split_spike_frames_by_spans: no recording spans to split by."
        )
    expected_start = 0
    for index, (start, end) in enumerate(spans):
        if int(start) != expected_start:
            raise ConcatSplitError(
                "split_spike_frames_by_spans: recording spans must be "
                f"contiguous from frame 0, but span {index} starts at "
                f"{int(start)} instead of {expected_start}."
            )
        expected_start = int(end)
    return split_unit_spike_trains(
        unit_spike_trains, [int(end) for _start, end in spans]
    )


def member_spike_times(
    local_frames_by_unit: dict, timestamps, *, context: str
) -> dict:
    """Map member-local spike frames onto the member recording's own clock.

    Unlike the synthetic concatenation timeline, this keeps real gaps.

    Parameters
    ----------
    local_frames_by_unit : dict[int, numpy.ndarray]
        ``{unit_id: member-local spike frames}`` (one entry of
        :func:`split_unit_spike_trains`).
    timestamps : numpy.ndarray, shape (n_member_samples,)
        The member ``Recording``'s timestamps, in seconds.
    context : str
        Names the caller and member in the error message.

    Returns
    -------
    dict[int, numpy.ndarray]
        ``{unit_id: spike times in seconds}``; a unit without spikes in the
        member keeps an empty array.

    Raises
    ------
    ValueError
        If a local frame falls outside ``[0, len(timestamps))``.
    """
    import numpy as np

    bounds = {
        int(unit_id): (int(np.min(frames)), int(np.max(frames)))
        for unit_id, frames in local_frames_by_unit.items()
        if len(frames)
        and (int(np.min(frames)) < 0 or int(np.max(frames)) >= len(timestamps))
    }
    if bounds:
        raise ValueError(
            f"{context}: local spike frames fall outside the member's "
            f"timestamp vector of length {len(timestamps)}: {bounds}."
        )
    return {
        int(unit_id): timestamps[np.asarray(frames, dtype=np.int64)]
        for unit_id, frames in local_frames_by_unit.items()
    }


def electrode_signature_from_rows(
    electrode_rows: list[dict], region_by_key: dict
) -> tuple:
    """Build a member sort group's electrode/region signature.

    Keyed by ``(electrode_group_name, electrode_id)`` because
    ``electrode_id`` can repeat across electrode groups.

    Parameters
    ----------
    electrode_rows : list of dict
        Every sort-group electrode, each carrying ``electrode_group_name`` and
        ``electrode_id``.
    region_by_key : dict
        ``{(electrode_group_name, electrode_id): region_name}``. An electrode
        absent from the map (no registered region) maps to ``None``.

    Returns
    -------
    tuple
        Deterministically ordered tuple of
        ``(electrode_group_name, electrode_id, region)`` triples.
    """
    return tuple(
        sorted(
            (
                str(row["electrode_group_name"]),
                int(row["electrode_id"]),
                region_by_key.get(
                    (str(row["electrode_group_name"]), int(row["electrode_id"]))
                ),
            )
            for row in electrode_rows
        )
    )


def _sampling_frequency_tolerance(reference, recording) -> float:
    """Bound rate roundoff from explicit timestamps, without scanning the clock.

    NWB readers infer the rate from the first 1000 timestamp differences,
    which can shift by one ULP of the clock's absolute offset. Capped at one
    frame of drift over either recording.
    """
    import numpy as np

    from spyglass.spikesorting.v2._core.signal_math import _segment_times_at

    uncertainty = 0.0
    drift_limits = []
    for candidate in (reference, recording):
        fs = float(candidate.get_sampling_frequency())
        n_samples = int(candidate.get_num_samples(segment_index=0))
        if n_samples:
            drift_limits.append(fs / n_samples)
        if n_samples < 2 or not candidate.has_time_vector(segment_index=0):
            continue
        times = _segment_times_at(candidate, [0, min(n_samples, 1000) - 1])
        precision = float(np.max(np.spacing(np.abs(times))))
        period = 1.0 / fs
        uncertainty += (
            fs * precision / (period - precision)
            if precision < period
            else float("inf")
        )
    if drift_limits:
        uncertainty = min(uncertainty, *drift_limits)
    return max(1e-9, uncertainty)


def assert_concat_compatible(recordings: list) -> None:
    """Reject member recordings that cannot be concatenated channel-for-channel.

    Compares every member to the first. SI's ``concatenate_recordings``
    fails opaquely on channel ids and does not check geometry.

    Parameters
    ----------
    recordings : list of si.BaseRecording
        Per-member preprocessed recordings, ordered by ``member_index``.

    Raises
    ------
    ValueError
        If the list is empty, or any member's channel ids (or count), sampling
        frequency, sample dtype, channel gains/offsets, or channel geometry
        differs from the first member's.
    """
    import numpy as np

    if not recordings:
        raise ValueError(
            "build_concatenated_recording: no member recordings to "
            "concatenate."
        )

    def _locations(recording):
        # Synthetic recordings may lack probe locations.
        if not recording.has_channel_location():
            return None
        return np.asarray(recording.get_channel_locations())

    def _scaling(recording, kind):
        getter = (
            recording.get_channel_gains
            if kind == "gain"
            else recording.get_channel_offsets
        )
        values = getter()
        return None if values is None else np.asarray(values)

    def _assert_array_matches(reference, value, index, *, noun, share_clause):
        if (reference is None) != (value is None):
            raise ValueError(
                f"build_concatenated_recording: member {index} {noun} presence "
                "differs from member 0 (one has it, the other does not); "
                f"{share_clause}"
            )
        if reference is not None and (
            value.shape != reference.shape or not np.allclose(value, reference)
        ):
            raise ValueError(
                f"build_concatenated_recording: member {index} {noun} differs "
                f"from member 0; {share_clause}"
            )

    reference = recordings[0]
    reference_ids = list(reference.get_channel_ids())
    reference_locations = _locations(reference)
    reference_fs = float(reference.get_sampling_frequency())
    reference_dtype = reference.get_dtype()
    reference_gains = _scaling(reference, "gain")
    reference_offsets = _scaling(reference, "offset")
    for index, recording in enumerate(recordings[1:], start=1):
        channel_ids = list(recording.get_channel_ids())
        if channel_ids != reference_ids:
            raise ValueError(
                f"build_concatenated_recording: member {index} has channel ids "
                f"{channel_ids} but member 0 has {reference_ids}; concatenated "
                "members must share channel ids (the same sort group / probe "
                "layout across sessions)."
            )
        fs = float(recording.get_sampling_frequency())
        # Rates inferred from explicit timestamps vary with float precision.
        tolerance = (
            1e-9
            if abs(fs - reference_fs) <= 1e-9
            else _sampling_frequency_tolerance(reference, recording)
        )
        if not np.isclose(fs, reference_fs, rtol=0, atol=tolerance):
            raise ValueError(
                f"build_concatenated_recording: member {index} sampling "
                f"frequency {fs} differs from member 0's {reference_fs}; "
                "concatenated members must share a sampling frequency so the "
                "stitched timeline is uniform."
            )
        if recording.get_dtype() != reference_dtype:
            raise ValueError(
                f"build_concatenated_recording: member {index} sample dtype "
                f"{recording.get_dtype()} differs from member 0's "
                f"{reference_dtype}; concatenated members must share a dtype."
            )
        _assert_array_matches(
            reference_gains,
            _scaling(recording, "gain"),
            index,
            noun="channel gains",
            share_clause=(
                "concatenated members must share per-channel gains so traces "
                "are combined on one scale."
            ),
        )
        _assert_array_matches(
            reference_offsets,
            _scaling(recording, "offset"),
            index,
            noun="channel offsets",
            share_clause=(
                "concatenated members must share per-channel offsets so traces "
                "are combined on one scale."
            ),
        )
        _assert_array_matches(
            reference_locations,
            _locations(recording),
            index,
            noun="channel geometry",
            share_clause=(
                "concatenated members must share probe geometry so "
                "cross-session waveforms align channel-for-channel."
            ),
        )


def mask_member_recordings(recordings, member_valid_times):
    """Mask each member and map its excluded frames into the concat timeline.

    A ``None`` valid-times entry means no artifact detection for that member.
    Masked samples must read 0 uV, so if any member has a nonzero channel
    offset, every member is converted with ``scale_to_uV`` to keep one scale
    and dtype. Compatibility is checked before that conversion.
    """
    import spikeinterface.preprocessing as sip

    from spyglass.spikesorting.v2._sorting.artifact_mask import (
        artifact_frame_ranges,
        has_nonzero_offset,
        silence_frame_ranges,
    )

    assert_concat_compatible(recordings)
    if any(has_nonzero_offset(r) for r in recordings):
        recordings = [sip.scale_to_uV(r) for r in recordings]
    masked, concat_ranges = [], []
    offset = 0
    for recording, valid_times in zip(
        recordings, member_valid_times, strict=True
    ):
        ranges = (
            []
            if valid_times is None
            else artifact_frame_ranges(recording, valid_times)
        )
        masked.append(silence_frame_ranges(recording, ranges))
        concat_ranges.extend(
            (start + offset, end + offset) for start, end in ranges
        )
        offset += recording.get_num_samples()
    return masked, concat_ranges


def concat_continuity(member_recordings, member_sample_counts):
    """Continuity spans of a concatenation, from its members' own timestamps.

    Member-internal gaps and every member join are boundaries. Must run on
    the members before concatenation replaces their clocks.

    Parameters
    ----------
    member_recordings : list[si.BaseRecording]
        Per-member recordings carrying their real timestamps (as loaded,
        before ``concatenate_recordings(..., ignore_times=True)``), ordered by
        ``member_index``.
    member_sample_counts : list[int]
        Per-member sample counts, same order.

    Returns
    -------
    _sorting.artifact_mask.Continuity
        Concat-frame spans with each span's first and last timestamp on its
        member's clock.
    """
    from spyglass.spikesorting.v2._sorting.artifact_mask import (
        Continuity,
        continuity_from_timestamps,
    )

    ends = cumulative_member_boundaries(member_sample_counts)
    spans: list[tuple[int, int]] = []
    start_s: list[float] = []
    end_s: list[float] = []
    for recording, offset in zip(
        member_recordings, [0, *ends[:-1]], strict=True
    ):
        member = continuity_from_timestamps(recording)
        spans.extend((offset + a, offset + b) for a, b in member.spans)
        start_s.extend(member.start_s)
        end_s.extend(member.end_s)
    return Continuity(spans=spans, start_s=start_s, end_s=end_s)


def concat_span_arrays(
    member_recordings, member_sample_counts, artifact_ranges
):
    """Continuity and statistics spans of a concatenation, as stored arrays.

    ``member_recordings`` must be the members as loaded, before
    concatenation (see :func:`concat_continuity`).

    Parameters
    ----------
    member_recordings : list[si.BaseRecording]
        Per-member recordings carrying their real timestamps, ordered by
        ``member_index``.
    member_sample_counts : list[int]
        Per-member sample counts, same order.
    artifact_ranges : list[tuple[int, int]]
        Half-open concat frame ranges masked as artifact (from
        :func:`mask_member_recordings`).

    Returns
    -------
    continuity_spans : np.ndarray
        ``(n, 2)`` int64 half-open concat frame ranges of uninterrupted
        acquisition, split at every member join and member-internal gap.
    continuity_start_s, continuity_end_s : np.ndarray
        ``(n,)`` float64 first / last timestamp of each continuity span on its
        member's own clock.
    statistics_spans : np.ndarray
        ``(m, 2)`` int64 half-open concat frame ranges that are artifact-free
        and lie inside a single continuity span.
    """
    import numpy as np

    from spyglass.spikesorting.v2._sorting.artifact_mask import statistics_spans

    continuity = concat_continuity(member_recordings, member_sample_counts)
    continuity_spans = np.asarray(continuity.spans, dtype=np.int64).reshape(
        -1, 2
    )
    continuity_start_s = np.asarray(continuity.start_s, dtype=np.float64)
    continuity_end_s = np.asarray(continuity.end_s, dtype=np.float64)
    statistics = np.asarray(
        statistics_spans(
            sum(member_sample_counts), artifact_ranges, continuity.spans
        ),
        dtype=np.int64,
    ).reshape(-1, 2)
    return continuity_spans, continuity_start_s, continuity_end_s, statistics


def observation_intervals(n_samples, sampling_frequency, artifact_ranges):
    """Return kept concat intervals in seconds from half-open frame ranges."""
    import numpy as np

    from spyglass.spikesorting.v2._sorting.artifact_mask import (
        complement_frame_ranges,
    )

    intervals = complement_frame_ranges(artifact_ranges, n_samples)
    return (
        np.asarray(intervals, dtype=float).reshape(-1, 2) / sampling_frequency
    )


def concat_provenance_tables(
    *,
    concat_recording_id,
    preprocessing_params_name: str,
    anchor_nwb_file_name: str,
    member_plan: list[dict],
    member_sample_counts: list[int],
    boundaries: list[int],
    artifact_ranges,
    obs_intervals,
) -> list:
    """Provenance tables embedded in the concatenated recording's NWB file.

    A header and the ordered member map, so ``split_sorting_by_session`` is
    reconstructable from the file alone.

    Parameters
    ----------
    concat_recording_id : uuid.UUID or str
        The concat being written.
    preprocessing_params_name : str
        The members' shared preprocessing recipe.
    anchor_nwb_file_name : str
        The first member's NWB file, which parents the analysis file.
    member_plan : list[dict]
        Per-member plan dicts from ``make_fetch`` (``member_index``,
        ``recording_pk``, ``nwb_file_name``, ``interval_list_name``,
        ``artifact_detection_id``), ordered by ``member_index``.
    member_sample_counts : list[int]
        Per-member sample counts, same order.
    boundaries : list[int]
        Cumulative member end samples (:func:`cumulative_member_boundaries`).
    artifact_ranges : list[tuple[int, int]]
        Half-open concat frame ranges masked as artifact.
    obs_intervals : np.ndarray
        ``(n, 2)`` kept intervals on the concat timeline, in seconds.

    Returns
    -------
    list
        The header table and the long member table, in write order.
    """
    from spyglass.spikesorting.v2._storage.provenance import (
        CONCAT_MEMBER_COLUMNS,
        CONCAT_MEMBERS,
        CONCAT_PROVENANCE,
        build_long_provenance_table,
        build_provenance_table,
    )

    member_rows = []
    concat_start = 0
    for plan, n_samples, cum_end in zip(
        member_plan, member_sample_counts, boundaries
    ):
        member_rows.append(
            {
                "member_index": int(plan["member_index"]),
                "recording_id": str(plan["recording_pk"]["recording_id"]),
                "nwb_file_name": plan["nwb_file_name"],
                "interval_list_name": plan["interval_list_name"],
                "artifact_detection_id": plan["artifact_detection_id"]
                or "none",
                "start_sample": 0,
                "end_sample": int(n_samples),
                "concat_start_sample": int(concat_start),
                "concat_end_sample": int(cum_end),
            }
        )
        concat_start = int(cum_end)
    return [
        build_provenance_table(
            CONCAT_PROVENANCE,
            {
                "concat_recording_id": str(concat_recording_id),
                "preprocessing_params_name": preprocessing_params_name,
                "anchor_nwb_file_name": anchor_nwb_file_name,
                "n_members": len(member_plan),
                "artifact_frame_ranges": artifact_ranges,
                "obs_intervals": obs_intervals.tolist(),
            },
        ),
        build_long_provenance_table(
            CONCAT_MEMBERS, member_rows, CONCAT_MEMBER_COLUMNS
        ),
    ]


def build_concatenated_recording(recordings: list):
    """Concatenate per-member recordings into one mono-segment recording.

    Wall-clock gaps are dropped (``ignore_times=True``). Traces are
    unchanged; only a constant third contact coordinate is dropped.

    Parameters
    ----------
    recordings : list of si.BaseRecording
        Per-member preprocessed recordings, ordered by ``member_index``. Must
        share channel ids and geometry.

    Returns
    -------
    si.BaseRecording
        The concatenated recording, with ``sum(n_samples_i)`` samples and the
        members' channels.
    """
    from spikeinterface.core import concatenate_recordings

    assert_concat_compatible(recordings)

    from spyglass.spikesorting.v2._recording.geometry import (
        flatten_planar_geometry,
    )

    # Permit exactly the rate roundoff the compatibility check accepted.
    reference_fs = recordings[0].get_sampling_frequency()
    sampling_frequency_max_diff = max(
        abs(recording.get_sampling_frequency() - reference_fs)
        for recording in recordings
    )
    concatenated = concatenate_recordings(
        recordings,
        ignore_times=True,
        sampling_frequency_max_diff=sampling_frequency_max_diff,
    )
    flatten_planar_geometry(concatenated)
    return concatenated
