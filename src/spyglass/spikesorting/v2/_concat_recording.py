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

    Returns ``(nwb_file_name, sort_group_id, interval_list_name)``. The
    ``sort_group_id`` is in the key -- not just ``(nwb_file_name,
    interval_list_name)`` -- so two members from the same NWB file and interval
    but DIFFERENT sort groups map to distinct keys instead of one silently
    overwriting the other. (The full member dict, which also carries
    ``team_name`` / ``member_index``, is not hashable as a dict; these three
    fields are the spatial identity a per-session sorting is addressed by.)

    Parameters
    ----------
    member : dict
        A ``SessionGroup.Member`` row carrying ``nwb_file_name``,
        ``sort_group_id``, ``interval_list_name``.

    Returns
    -------
    tuple[str, int, str]
        The hashable member key.
    """
    return (
        member["nwb_file_name"],
        int(member["sort_group_id"]),
        member["interval_list_name"],
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

    concatenated = concatenate_recordings(recordings, ignore_times=True)
    if motion_preset is None:
        return concatenated

    from spikeinterface.preprocessing import correct_motion

    return correct_motion(
        concatenated,
        preset=motion_preset,
        output_motion=False,
        output_motion_info=False,
        **(preset_kwargs or {}),
        **(job_kwargs or {}),
    )
