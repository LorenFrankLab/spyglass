"""Pure + SpikeInterface helpers behind the concatenated-recording cache.

The concat math and the SpikeInterface concatenate call, kept out of the
``session_group`` schema module so ``ConcatenatedRecording`` only fetches,
calls these, writes, and inserts.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import; the SpikeInterface dependency is imported lazily inside
:func:`build_concatenated_recording`. The sample-boundary / back-mapping math
is pure (stdlib + numpy) so it is unit-testable without a database or a real
recording.
"""

from __future__ import annotations


def member_recording_selection_key(
    member: dict, preprocessing_params_name: str
) -> dict:
    """Build the ``RecordingSelection`` key for one ``SessionGroup.Member``.

    A member contributes ``nwb_file_name`` / ``sort_group_id`` /
    ``interval_list_name`` / ``team_name``; the concat preprocessing recipe is
    shared across all members, so it is injected rather than read from the
    member row. The five returned fields are exactly the ``RecordingSelection``
    logical identity, so the concat selection-time precondition, the
    materializer, and the sort-time anchor all resolve a member's cached
    ``Recording`` through one key shape.

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

    Returns the full member identity
    ``(nwb_file_name, sort_group_id, interval_list_name, team_name)`` -- the
    same fields (minus the shared preprocessing recipe) that resolve a member's
    ``RecordingSelection``. Keying on all four means no two distinct members
    collide: ``sort_group_id`` separates two shanks of the same NWB/interval,
    and ``team_name`` separates the mixed-team case (``create_group`` permits
    the same spatial member under two teams, since its duplicate guard also
    keys on ``team_name``). The full member dict is not hashable, so this tuple
    is the addressable per-session identity.

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


#: The per-member logical-identity fields folded into ``concat_recording_id``
#: via :func:`member_set_hash`. Ordered by ``member_index`` so the concatenation
#: order is part of identity. ``recording_content_hash`` / sample counts are
#: deliberately EXCLUDED: member content is verified at materialize/rebuild
#: against the stored snapshot, not folded into identity (a member rebuild that
#: reproduces its content must not fork the concat id).
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

    Hashes only the LOGICAL identity of each member
    (:data:`MEMBER_SNAPSHOT_LOGICAL_FIELDS`), canonicalized and ordered by
    ``member_index``, into a 64-char SHA-256 hex digest. This is folded into
    ``concat_recording_id`` so that two ``SessionGroup``\\s identical in name and
    params but differing in their ordered member set mint DIFFERENT concat ids --
    and so a later edit to ``SessionGroup.Member`` produces a new id rather than
    silently reusing an existing concat over a changed member set.

    The per-member ``recording_content_hash`` (and any sample-count field) is
    excluded: member content is verification, not identity. Content drift is
    caught at materialize/rebuild by comparing the live ``Recording`` against the
    stored snapshot (``ConcatMemberDriftError``), the same separation
    ``Recording`` keeps between its selection-logical ``recording_id`` and its
    post-write ``content_hash``.

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
    import hashlib
    import json

    from spyglass.spikesorting.v2._core.selection_identity import (
        canonical_identity,
    )

    # Reuse the single v2 identity canonicalization (UUID<->str, numpy<->int
    # collapse) so the folded member-set hash can never drift from the rest of
    # the deterministic-id system. ``canonical_identity`` sorts keys, so each
    # member's logical fields canonicalize order-independently; we order the
    # MEMBERS by member_index so the concatenation order is part of the hash.
    ordered = sorted(snapshot_rows, key=lambda row: int(row["member_index"]))
    payload = json.dumps(
        [
            canonical_identity(
                {field: row[field] for field in MEMBER_SNAPSHOT_LOGICAL_FIELDS}
            )
            for row in ordered
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def concat_recording_artifact_lock(concat_recording_id, *, timeout: float = -1):
    """Return a cross-process lock serializing one concat artifact's slot.

    The concat analog of :func:`spyglass.spikesorting.v2._recording.fingerprint.recording_artifact_lock`:
    a per-``concat_recording_id`` ``filelock.FileLock`` so a rebuild
    (``ConcatenatedRecording.get_recording`` read-repair /
    ``_rebuild_nwb_artifact``) of the *same* concat can never interleave with
    another -- no unlink racing a write, no reader seeing a half-written HDF5.
    Different concats stay free to run in parallel. A distinct filename prefix
    (``concat_recording_*``) keeps it from colliding with the single-session
    recording lock even if a ``recording_id`` and a ``concat_recording_id`` ever
    shared a UUID.

    The lock file lives under the shared analyzer/lock root
    (:func:`._analyzer_cache.analyzer_cache_root`), a stable per-install path, so
    all workers must resolve the same file. Multi-host deployments require
    cross-host POSIX file locking on that mount, as documented for the recording
    and analyzer locks. Lock-acquisition errors propagate.

    Parameters
    ----------
    concat_recording_id
        The concat whose canonical artifact the caller will mutate.
    timeout : float, optional
        Seconds to wait before raising ``filelock.Timeout``. Default ``-1``
        blocks indefinitely (serialize-don't-fail); the lock releases when the
        holding process exits, so a crashed job cannot wedge the next one.

    Returns
    -------
    filelock.FileLock
        An unacquired lock; use it as a context manager or call ``.acquire()``.
    """
    from filelock import FileLock

    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        analyzer_cache_root,
    )

    root = analyzer_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    return FileLock(
        str(root / f"concat_recording_{concat_recording_id}.artifact.lock"),
        timeout=timeout,
    )


def cumulative_member_boundaries(
    num_samples_per_member: list[int],
) -> list[int]:
    """Return the cumulative end-sample boundary for each member.

    The boundary for member ``i`` is the number of samples in the
    concatenated recording up to and INCLUDING member ``i`` -- i.e. the
    exclusive end frame of that member's span. Member ``i`` therefore occupies
    concat frames ``[boundaries[i-1], boundaries[i])`` (with ``boundaries[-1]``
    read as 0 for member 0). The final boundary equals the total sample count.

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

    Maps each unit's spike frames (indices into the concatenated recording)
    back into each member's LOCAL sample frame: member ``i`` keeps the frames
    in ``[start_i, end_i)`` and shifts them by ``-start_i`` so they index that
    member's own recording from sample 0. Unit ids are preserved across every
    member (a unit absent from a member's span gets an empty array).

    Spike conservation is enforced, not assumed. The members partition
    ``[0, boundaries[-1])`` into disjoint, contiguous half-open intervals, so
    every concat-frame spike must land in exactly one member. This raises
    ``ConcatSplitError`` rather than silently dropping a spike when the
    boundaries are not strictly increasing (overlapping / empty intervals), when
    ``total_n_samples`` is given and the final boundary does not equal it (a
    boundary set that does not cover the whole recording), or when any input
    frame falls outside ``[0, boundaries[-1])``.

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
    # Strictly increasing from the implicit start 0 -> disjoint, non-empty,
    # contiguous member intervals covering [0, boundaries[-1]).
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

    # Per-spike conservation: reject (rather than silently drop) any frame
    # outside the covered range [0, total). Within range, the strictly-
    # increasing boundaries guarantee each frame maps to exactly one member.
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

    A sort of one recording has one span, ``[0, n_samples)``; a sort of a
    concatenation has one span per member, each starting where the previous
    one ended. The spans must be contiguous from frame 0, so they are exactly
    the member boundaries :func:`split_unit_spike_trains` splits by, and its
    per-spike conservation applies: every spike lands in exactly one span, or
    this raises.

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

    A member's local frame ``k`` is sample ``k`` of that member's
    ``Recording``, so its time is ``timestamps[k]`` on the member's original
    clock (gaps between members and inside a member are kept, unlike the
    synthetic concatenation timeline).

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

    The signature identifies the physical electrode space a member contributes
    so two members of the same implant share a signature while members on
    different probes, sort groups, or regions diverge. Each electrode is keyed
    by ``(electrode_group_name, electrode_id)`` -- NOT ``electrode_id`` alone --
    because the ``Electrode`` primary key is
    ``(nwb_file_name, electrode_group_name, electrode_id)`` and the same
    ``electrode_id`` can repeat across electrode groups; collapsing on the id
    would let two physically distinct probes with reused ids and matching
    regions pass as one electrode space.

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

    NWB readers infer the rate from the median of the first 1000 timestamp
    differences. Subtracting two rounded timestamps can change that period
    by one timestamp ULP; the corresponding reciprocal error grows with the
    clock's absolute offset. Rate-based clocks need no such allowance. Cap
    the combined uncertainty at one frame of drift over either recording.
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

    SI's ``concatenate_recordings`` already requires identical channel ids, but
    it fails deep in the stitch with an opaque message and does NOT check probe
    geometry. Cross-session concatenation additionally needs identical geometry
    so a unit's waveform footprint is the same across members. This front-loads
    both checks (against the first member, the deterministic anchor) with a
    clear message naming the first offending member, so an incompatible
    SessionGroup fails fast in ``ConcatenatedRecording.make`` rather than mid-
    stitch or, worse, silently anchoring to the first member's geometry.

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
        # Cached Recording artifacts (the production input) always carry probe
        # locations from the NWB electrodes table; bare synthetic recordings may
        # not. Return None when geometry is unavailable so a probe-less member
        # is not falsely flagged -- but a member that HAS geometry while another
        # does not is still surfaced below.
        if not recording.has_channel_location():
            return None
        return np.asarray(recording.get_channel_locations())

    def _scaling(recording, kind):
        # ``gain_to_uV`` / ``offset_to_uV`` are optional; a recording without
        # them returns None. Mirror the geometry presence handling so a member
        # that HAS scaling while another does not is surfaced.
        getter = (
            recording.get_channel_gains
            if kind == "gain"
            else recording.get_channel_offsets
        )
        values = getter()
        return None if values is None else np.asarray(values)

    def _assert_array_matches(reference, value, index, *, noun, share_clause):
        # Shared presence-XOR + shape/allclose check for the optional per-channel
        # arrays (geometry, gains, offsets): a member that HAS the array while
        # another does not, or whose values differ, is surfaced.
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
        # Cached NWBs carry explicit timestamps. SI's inferred rate varies
        # with their float precision, so equal acquisition rates can reload
        # slightly differently when the selected epochs start at different
        # offsets. Allow only that numerical uncertainty, bounded to one
        # frame of accumulated drift; rate-based clocks stay effectively exact.
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

    Uses the standalone mask's frame mapping, including disjoint timestamps.
    The result scales with interval count rather than recording duration.
    ``None`` explicitly means no artifact detection for that member.

    Masked samples must read 0 uV. When any member keeps a nonzero channel
    offset (an unfiltered, unreferenced source; see
    :func:`~spyglass.spikesorting.v2._sorting.artifact_mask.has_nonzero_offset`),
    every member, masked or not, is presented as float32 microvolts with a
    unit calibration (SpikeInterface ``scale_to_uV``), so the concatenation
    is on one scale and one dtype even for offsets the compatibility check
    treats as equal (``allclose``, e.g. 0 and 1e-9). Otherwise every member
    is masked in its stored units, so its samples are unchanged. The members
    are checked for compatibility (:func:`assert_concat_compatible`) before
    that conversion, which would otherwise give members with different
    offsets or gains the same unit calibration.
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

    Each member's spans come from its persisted timestamps
    (``continuity_from_timestamps``), so a member-internal wall-clock gap is a
    boundary too; they are offset into concat frames by the cumulative
    member sample counts (the same basis as
    :func:`cumulative_member_boundaries`), so every member join is a boundary
    as well, however short the real gap. The concatenation itself replaces the
    members' timestamps with one synthetic clock, so these must be taken from
    the members before they are concatenated.

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
    _sorting_artifact_mask.Continuity
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

    The continuity spans come from :func:`concat_continuity`, so
    ``member_recordings`` must be the members as loaded: their persisted
    timestamps still carry each member's own gaps, which the concatenation
    replaces with one synthetic timeline. The statistics spans are the
    artifact-free parts of the continuity spans
    (:func:`~spyglass.spikesorting.v2._sorting.artifact_mask.statistics_spans`).

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

    Self-describing provenance: a header and the ordered member map with
    per-member frame boundaries, so ``split_sorting_by_session`` is
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

    Stitches the ordered per-member ``Recording`` artifacts into one
    mono-segment recording with a continuous (uniform) timeline -- the members
    are independent sessions, so their wall-clock gaps are dropped
    (``ignore_times=True``) and the result is a synthetic continuous recording
    a sorter can consume as one piece. The traces are the members' traces
    unchanged (no motion correction, no whitening); only the constant third
    contact coordinate is dropped (see
    :func:`~spyglass.spikesorting.v2._recording.geometry.flatten_planar_geometry`).

    Parameters
    ----------
    recordings : list of si.BaseRecording
        Per-member preprocessed recordings, ordered by ``member_index``. Must
        share channel ids and geometry (enforced upstream by reusing the same
        sort group / probe layout across members).

    Returns
    -------
    si.BaseRecording
        The concatenated recording, with ``sum(n_samples_i)`` samples and the
        members' channels.
    """
    from spikeinterface.core import concatenate_recordings

    # Fail fast (and clearly) on incompatible members BEFORE the stitch, and
    # before the result anchors to the first member's NWB geometry downstream.
    assert_concat_compatible(recordings)

    from spyglass.spikesorting.v2._recording.geometry import (
        flatten_planar_geometry,
    )

    # SI independently requires equal rates by default. The compatibility
    # check above has already bounded any timestamp-inference roundoff, so
    # permit exactly the accepted difference when stitching the members.
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
