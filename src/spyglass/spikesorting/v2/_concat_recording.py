"""Pure + SpikeInterface helpers behind the concatenated-recording cache.

``ConcatenatedRecording`` is a DataJoint *schema* module: importing it activates
``dj.schema(...)`` and the source-part dependencies. The concat math and the
SpikeInterface concatenate/motion-correct calls need none of that at import, so
they live here and the table becomes a thin orchestrator (fetch -> call these ->
write -> insert). Same "thin DataJoint shell over pure/IO services" direction as
``_recording_nwb`` / ``_selection_identity`` / ``_signal_math``.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import; the SpikeInterface dependency is imported lazily inside
:func:`build_concatenated_recording`. The motion-preset resolution and the
sample-boundary / back-mapping math are pure (stdlib + numpy) so they are
unit-testable without a database or a real recording.
"""

from __future__ import annotations

#: The Spyglass ``"auto"`` motion-correction alias resolves to this
#: SpikeInterface preset for same-day groups. It is NOT passed through to
#: ``correct_motion`` (SI has no ``"auto"`` preset). The RESOLVED preset (this
#: value, not the ``"auto"`` alias) is what ``ConcatenatedRecording`` persists in
#: its ``motion_preset`` column, so each row records what actually ran.
AUTO_SAME_DAY_PRESET = "rigid_fast"


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

    from spyglass.spikesorting.v2._selection_identity import (
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

    The concat analog of :func:`._recording_fingerprint.recording_artifact_lock`:
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
    all workers on a machine contend on the same file. As with the recording
    lock this serializes processes on ONE machine; it does not coordinate across
    hosts sharing an NFS mount.

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

    from spyglass.spikesorting.v2._analyzer_cache import analyzer_cache_root

    root = analyzer_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    return FileLock(
        str(root / f"concat_recording_{concat_recording_id}.artifact.lock"),
        timeout=timeout,
    )


def resolve_motion_correction(
    motion_params: dict, *, is_multi_day: bool
) -> tuple[str | None, dict]:
    """Resolve a motion-correction params blob to an SI preset + kwargs.

    Translates the Spyglass-only ``"auto"`` alias before any SpikeInterface
    call: ``"auto"`` maps to :data:`AUTO_SAME_DAY_PRESET` for a same-day group
    and is REJECTED for a multi-day group (cross-day drift needs a deliberate
    preset choice, not a silent default). ``"none"`` means "skip motion
    correction" and returns ``(None, {})``. Any other value is an explicit
    SpikeInterface preset and is passed through unchanged.

    Parameters
    ----------
    motion_params : dict
        The ``MotionCorrectionParameters.params`` blob, carrying ``preset``
        and ``preset_kwargs``.
    is_multi_day : bool
        Whether the underlying ``SessionGroup`` spans two or more dates.

    Returns
    -------
    tuple[str | None, dict]
        ``(preset, preset_kwargs)``. ``preset`` is ``None`` when motion
        correction is skipped (``preset="none"``); otherwise it is the
        SpikeInterface preset name to pass to ``correct_motion``.

    Raises
    ------
    ValueError
        If ``preset="auto"`` on a multi-day group (choose an explicit,
        non-``auto`` preset), or if ``preset`` is missing from the blob.
    """
    if "preset" not in motion_params:
        raise ValueError(
            "resolve_motion_correction: motion_params has no 'preset' key; "
            "expected a validated MotionCorrectionParameters.params blob."
        )
    preset = motion_params["preset"]
    preset_kwargs = dict(motion_params.get("preset_kwargs") or {})

    if preset == "none":
        return None, {}
    if preset == "auto":
        if is_multi_day:
            raise ValueError(
                "Motion-correction preset 'auto' is single-day only; this "
                "concat group spans multiple dates. Choose an explicit "
                "non-'auto' preset (e.g. 'dredge_fast') for multi-day "
                "concatenation, though sort-then-match remains the "
                "recommended cross-day workflow."
            )
        return AUTO_SAME_DAY_PRESET, preset_kwargs
    return preset, preset_kwargs


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
    boundaries: list[int] = []
    running = 0
    for n in num_samples_per_member:
        running += int(n)
        boundaries.append(running)
    return boundaries


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
        try:
            return np.asarray(recording.get_channel_locations())
        except Exception:
            return None

    def _scaling(recording, kind):
        # ``gain_to_uV`` / ``offset_to_uV`` are optional; a recording without
        # them returns None. Mirror the geometry presence handling so a member
        # that HAS scaling while another does not is surfaced.
        getter = (
            recording.get_channel_gains
            if kind == "gain"
            else recording.get_channel_offsets
        )
        try:
            values = getter()
        except Exception:
            return None
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
        # Effectively exact (within float epsilon): a stitched timeline maps
        # sample frames to spike times, so even a sub-Hz fs drift (e.g. 30000.0
        # vs 30000.3) silently mis-times every spike. ``np.isclose``'s default
        # rtol=1e-5 would accept that, so pin rtol=0 with a tiny atol.
        if not np.isclose(fs, reference_fs, rtol=0, atol=1e-9):
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


def build_concatenated_recording(
    recordings: list,
    *,
    motion_preset: str | None,
    preset_kwargs: dict | None = None,
    job_kwargs: dict | None = None,
):
    """Concatenate per-member recordings and optionally motion-correct them.

    Stitches the ordered per-member ``Recording`` artifacts into one
    mono-segment recording with a continuous (uniform) timeline -- the members
    are independent sessions, so their wall-clock gaps are dropped
    (``ignore_times=True``) and the result is a synthetic continuous recording
    a sorter can consume as one piece. When ``motion_preset`` is not ``None``,
    ``correct_motion`` runs on that single concatenated segment (SI's
    ``estimate_motion`` is single-segment only, which is exactly what
    concatenation produces). Motion correction is run BEFORE any whitening:
    whitening stays a sorter/analyzer concern, so the returned recording is
    motion-corrected but unwhitened.

    ``output_motion`` / ``output_motion_info`` are pinned ``False`` so
    ``correct_motion`` returns just the corrected recording (the MVP concat
    cache does not persist motion trajectories); the params schema already
    forbids those kwargs in ``preset_kwargs``.

    Parameters
    ----------
    recordings : list of si.BaseRecording
        Per-member preprocessed recordings, ordered by ``member_index``. Must
        share channel ids and geometry (enforced upstream by reusing the same
        sort group / probe layout across members).
    motion_preset : str or None
        Resolved SpikeInterface ``correct_motion`` preset, or ``None`` to skip
        motion correction. Resolve the Spyglass ``"auto"`` alias via
        :func:`resolve_motion_correction` before calling.
    preset_kwargs : dict, optional
        Extra kwargs forwarded to ``correct_motion`` (per-step ``*_kwargs``).
    job_kwargs : dict, optional
        Resolved SpikeInterface job kwargs (``n_jobs`` etc.) forwarded to
        ``correct_motion``.

    Returns
    -------
    si.BaseRecording
        The concatenated (and, unless skipped, motion-corrected) recording.
    """
    from spikeinterface.core import concatenate_recordings

    # Fail fast (and clearly) on incompatible members BEFORE the stitch, and
    # before the result anchors to the first member's NWB geometry downstream.
    assert_concat_compatible(recordings)

    concatenated = concatenate_recordings(recordings, ignore_times=True)
    if motion_preset is None:
        return concatenated

    from spikeinterface.core.job_tools import job_keys
    from spikeinterface.preprocessing import correct_motion

    # The resolved ``job_kwargs`` may carry ONLY SpikeInterface job kwargs
    # (n_jobs, chunk_duration, progress_bar, ...). ``correct_motion`` has
    # top-level params BEFORE its ``**job_kwargs`` -- ``folder`` / ``overwrite``
    # / ``output_motion`` / ``output_motion_info`` (the side-artifact + return-
    # type contract the concat cache forbids) and ``detect_kwargs`` /
    # ``estimate_motion_kwargs`` / ... (motion parameters that belong in
    # ``preset_kwargs``, which IS part of the concat identity). A non-job key in
    # ``job_kwargs`` would silently bind one of those, bypassing the persistence
    # contract or changing the motion outside the content hash, so reject it.
    resolved_job_kwargs = dict(job_kwargs or {})
    non_job_keys = sorted(set(resolved_job_kwargs) - set(job_keys))
    if non_job_keys:
        raise ValueError(
            "build_concatenated_recording: motion job_kwargs carries non-job "
            f"key(s) {non_job_keys}; only SpikeInterface job kwargs "
            f"{sorted(job_keys)} are allowed there. Motion per-step kwargs "
            "(detect_kwargs, estimate_motion_kwargs, ...) belong in "
            "MotionCorrectionParameters.preset_kwargs; side-artifact / output "
            "kwargs (folder / overwrite / output_motion / output_motion_info) "
            "are unsupported (the concat cache does not persist them)."
        )

    # Merge into ONE kwargs dict (resolved job kwargs win on conflict, per the
    # job-kwargs resolution contract) so an overlapping key -- e.g. ``n_jobs``
    # in both ``preset_kwargs`` and the resolved ``job_kwargs`` -- does not
    # raise ``TypeError: got multiple values for keyword`` from a double splat.
    motion_kwargs = {**(preset_kwargs or {}), **resolved_job_kwargs}
    return correct_motion(
        concatenated,
        preset=motion_preset,
        output_motion=False,
        output_motion_info=False,
        **motion_kwargs,
    )
