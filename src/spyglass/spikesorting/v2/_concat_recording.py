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
#: ``correct_motion`` (SI has no ``"auto"`` preset).
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


def cumulative_member_boundaries(num_samples_per_member: list[int]) -> list[int]:
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
    unit_spike_trains: dict, boundaries: list[int]
) -> list[dict]:
    """Slice concat-frame spike trains into per-member local frames.

    Maps each unit's spike frames (indices into the concatenated recording)
    back into each member's LOCAL sample frame: member ``i`` keeps the frames
    in ``[start_i, end_i)`` and shifts them by ``-start_i`` so they index that
    member's own recording from sample 0. Unit ids are preserved across every
    member (a unit absent from a member's span gets an empty array).

    Parameters
    ----------
    unit_spike_trains : dict[int, numpy.ndarray]
        ``{unit_id: concat-frame spike indices}`` for the concatenated sort.
    boundaries : list[int]
        Cumulative end-sample boundaries from
        :func:`cumulative_member_boundaries` (one per member, ordered by
        ``member_index``).

    Returns
    -------
    list[dict[int, numpy.ndarray]]
        One ``{unit_id: local-frame spike indices}`` dict per member, in
        ``member_index`` order.
    """
    import numpy as np

    per_member: list[dict] = []
    start = 0
    for end in boundaries:
        member_units: dict = {}
        for unit_id, frames in unit_spike_trains.items():
            frames = np.asarray(frames)
            in_member = frames[(frames >= start) & (frames < end)] - start
            member_units[unit_id] = in_member.astype(np.int64, copy=False)
        per_member.append(member_units)
        start = end
    return per_member


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
        If the list is empty, or any member's channel ids (or count) or channel
        geometry differs from the first member's.
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

    reference = recordings[0]
    reference_ids = list(reference.get_channel_ids())
    reference_locations = _locations(reference)
    for index, recording in enumerate(recordings[1:], start=1):
        channel_ids = list(recording.get_channel_ids())
        if channel_ids != reference_ids:
            raise ValueError(
                f"build_concatenated_recording: member {index} has channel ids "
                f"{channel_ids} but member 0 has {reference_ids}; concatenated "
                "members must share channel ids (the same sort group / probe "
                "layout across sessions)."
            )
        locations = _locations(recording)
        if (reference_locations is None) != (locations is None):
            raise ValueError(
                f"build_concatenated_recording: member {index} channel geometry "
                "presence differs from member 0 (one has probe locations, the "
                "other does not); concatenated members must share probe "
                "geometry."
            )
        if reference_locations is not None and (
            locations.shape != reference_locations.shape
            or not np.allclose(locations, reference_locations)
        ):
            raise ValueError(
                f"build_concatenated_recording: member {index} channel geometry "
                "differs from member 0; concatenated members must share probe "
                "geometry so cross-session waveforms align channel-for-channel."
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
