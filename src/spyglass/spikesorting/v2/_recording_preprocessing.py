"""Pre-motion preprocessing services behind ``Recording``.

These functions apply the pre-motion preprocessing stack (optional ADC
phase-shift, then the bandpass filter, then bad-channel interpolation, then
common reference -- no whitening) for ``Recording.make_compute`` (and the
rebuild path), and build the ``ElectricalSeries.filtering`` provenance string
from the steps that actually ran. The table threads already-fetched DB state in
(the tri-part ``make_fetch``/``make_compute``/``make_insert`` contract forbids
DB I/O inside compute), so the preprocessing hot path here is DB-free.

Why this lives in its own module rather than in ``recording.py``:
``recording.py`` is a DataJoint *schema* module -- importing it activates
``dj.schema(...)`` and the source-part dependencies. The preprocessing logic
needs none of that at import, so ``Recording`` becomes a thin orchestrator
(fetch -> call these -> insert / verify). Same "thin DataJoint shell over
pure/IO services" direction as ``_artifact_compute`` / ``_selection_identity``
/ ``_analyzer_cache`` / ``_curation_transforms`` / ``_units_nwb`` /
``_sorting_dispatch``.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: all SpikeInterface / numpy / spyglass dependencies are
imported lazily inside the functions. Neither function here touches the DB at
CALL time: ``apply_pre_motion_preprocessing`` runs purely on the in-memory
SpikeInterface recording and ``filtering_description`` is pure string
formatting.
"""

from __future__ import annotations


def apply_pre_motion_preprocessing(
    recording,
    reference_mode: str,
    reference_electrode_id: int | None,
    sort_group_channel_ids: list,
    validated,
    bad_channel_handling: str = "remove",
    bad_channel_ids=(),
):
    """Apply the pre-motion preprocessing stack (phase-shift, filter, ref).

    Optional ADC phase-shift runs FIRST, then the bandpass filter
    (temporal), then referencing (spatial). Bandpass-before-reference is
    the signal-processing-preferred order and intentionally diverges from
    v1, which referenced first. The two are not commutative on the
    global-median branch (the per-sample median is non-linear), so v2 and v1
    differ numerically there; on the ``specific`` / ``none`` paths the steps
    are linear and commute, so output is identical to the old order.
    Whitening stays deferred to the sorter stage so motion correction never
    sees whitened data (SpikeInterface docs flag whitening as destructive for
    motion estimators).

    Takes a pre-validated ``PreprocessingParamsSchema`` instance
    so the DB read happens once in ``make_fetch`` -- this method
    is called from inside ``make_compute`` where the tri-part
    contract forbids further DB I/O.

    Returns ``(recording, applied_steps)`` where ``applied_steps`` reports the
    runtime facts a caller cannot derive from the params alone -- currently
    ``{"phase_shift": bool}``, whether the gated phase-shift actually ran (it
    is skipped when the recording lacks ``inter_sample_shift``).
    ``filtering_description`` consumes it so provenance can distinguish a
    phase-shift that ran from one that was requested but skipped. Bandpass and
    reference have no skip path (they run iff their params are set), so they
    are not reported -- the params are ground truth for them.

    Parameters
    ----------
    recording : si.BaseRecording
        The time- and channel-restricted recording to preprocess.
    reference_mode : str
        One of ``"none"``, ``"global_median"``, or ``"specific"``.
    reference_electrode_id : int or None
        Electrode id subtracted on the ``"specific"`` path and dropped
        from the surface afterward; ignored otherwise.
    sort_group_channel_ids : list
        Spyglass electrode ids of the sort group's declared members.
    validated : PreprocessingParamsSchema
        Pre-validated preprocessing params (phase-shift, bandpass,
        common-reference operator) read once in ``make_fetch``.
    bad_channel_handling : str, optional
        ``"remove"`` (default) or ``"interpolate"``. Only
        ``"interpolate"`` acts here, filling the interior bad channels.
    bad_channel_ids : sequence of int, optional
        Interior curated-bad electrode ids to interpolate on the
        ``"interpolate"`` path. Default ``()``.
    """
    import numpy as _np
    import spikeinterface.preprocessing as sip

    from spyglass.utils import logger

    applied_steps: dict = {}

    # 0. ADC phase-shift (multiplexed-ADC / Neuropixels) -- runs FIRST,
    #    before the bandpass. Valid only when the recording carries an
    #    ``inter_sample_shift`` property (``sip.phase_shift`` raises without
    #    it); skip (loudly) otherwise so enabling it on non-multiplexed
    #    acquisition is a safe no-op rather than a hard failure. The AIND
    #    "phase-shift first" order is a deliberate choice, not a correctness
    #    requirement (a fractional-sample delay and a zero-phase bandpass are
    #    both LTI per channel and commute, modulo edge margins).
    applied_steps["phase_shift"] = False
    if validated.phase_shift is not None:
        if recording.get_property("inter_sample_shift") is not None:
            recording = sip.phase_shift(
                recording, margin_ms=validated.phase_shift.margin_ms
            )
            applied_steps["phase_shift"] = True
        else:
            logger.warning(
                "apply_pre_motion_preprocessing: phase_shift requested but the "
                "recording has no 'inter_sample_shift' property (not a "
                "multiplexed-ADC acquisition); skipping phase-shift."
            )

    # 1. Bandpass filter. ``bandpass_filter=None`` is the "no_filter"
    #    preset -- skip the step entirely rather than passing a wide band
    #    that still filters.
    if validated.bandpass_filter is not None:
        # freq_max passes the Pydantic schema (which cannot know the
        # recording's sampling rate) but a value at/above Nyquist (fs/2) fails
        # deep inside scipy's filter design; reject it here with a clear
        # message (shared with suggest_bad_channels).
        from spyglass.spikesorting.v2._signal_math import (
            assert_freq_max_below_nyquist,
        )

        assert_freq_max_below_nyquist(
            validated.bandpass_filter.freq_max,
            recording.get_sampling_frequency(),
            context="apply_pre_motion_preprocessing: ",
        )
        recording = sip.bandpass_filter(
            recording,
            freq_min=validated.bandpass_filter.freq_min,
            freq_max=validated.bandpass_filter.freq_max,
            dtype=_np.float64,
        )

    # 1b. Bad-channel handling: between filter and reference (matches the
    #     IBL/AIND destripe order). Only ``interpolate`` does anything -- it
    #     fills the interior curated-bad channels ``restrict_recording``
    #     re-included. On ``remove`` those channels were never re-added, so
    #     ``bad_channel_ids`` is empty here and this is a no-op (today's
    #     behavior). The ``specific`` reference is present only for subtraction
    #     (dropped after ``common_reference``) and is never a handling target.
    if bad_channel_handling not in ("remove", "interpolate"):
        raise ValueError(
            "apply_pre_motion_preprocessing: invalid bad_channel_handling "
            f"{bad_channel_handling!r}; use 'remove' or 'interpolate'."
        )
    ref_id = (
        int(reference_electrode_id) if reference_mode == "specific" else None
    )
    to_interpolate: list[int] = []
    if bad_channel_handling == "interpolate":
        present = {int(c) for c in recording.get_channel_ids()}
        requested = {int(c) for c in bad_channel_ids}
        # ``restrict_recording`` was told to slice these in; a missing one means
        # the slice contract broke -- fail loud rather than silently leaving an
        # interior bad channel unfilled (mirrors the specific-reference guard
        # below).
        missing = requested - present
        if missing:
            raise RuntimeError(
                "apply_pre_motion_preprocessing: interior bad channels "
                f"{sorted(missing)} are absent from the recording surface; "
                "restrict_recording must slice them in for interpolation."
            )
        to_interpolate = sorted(requested - {ref_id})
    if to_interpolate:
        # interpolation needs channel locations; fail clearly if absent rather
        # than letting SI raise opaquely deep in the kriging call.
        # ``has_channel_location()`` is the right predicate -- SI's
        # ``get_channel_locations()`` *raises* (does not return None) when no
        # probe geometry is set.
        if not recording.has_channel_location():
            raise ValueError(
                "apply_pre_motion_preprocessing: interpolate requires channel "
                "locations on the recording, but none are set (no probe "
                "geometry). Use bad_channel_handling='remove', or ensure the "
                "NWB / probe carries electrode positions."
            )
        recording = sip.interpolate_bad_channels(recording, to_interpolate)
    applied_steps["bad_channels"] = {"interpolated": to_interpolate}

    # 2. Reference the filtered signal.
    if reference_mode == "specific":
        recording = sip.common_reference(
            recording,
            reference="single",
            ref_channel_ids=[int(reference_electrode_id)],
            dtype=_np.float64,
        )
        # Drop the reference channel from the recording surface so
        # the sorter only sees the actual sort-group channels
        # (restrict_recording included it solely for this step).
        # Its presence here is an invariant: restrict_recording always
        # slices the specific reference in, and bandpass / common_reference
        # preserve channel ids, so a missing ref means an upstream contract
        # broke. Fail loud rather than silently shipping a reference channel
        # into the sort.
        channel_ids = [int(c) for c in recording.get_channel_ids()]
        if int(reference_electrode_id) not in channel_ids:
            raise RuntimeError(
                "apply_pre_motion_preprocessing: 'specific' reference "
                f"electrode {int(reference_electrode_id)} is absent after "
                f"referencing (channels={channel_ids}); cannot drop it from "
                "the sort surface. restrict_recording must slice the "
                "reference channel in for subtraction."
            )
        recording = recording.remove_channels([int(reference_electrode_id)])
    elif reference_mode == "global_median":
        # A global reference on a single-channel group zeroes the signal: the
        # per-sample median (or mean) across one channel IS that channel, so
        # subtracting it yields all zeros -- a recording that sorts to nothing
        # with no error. Fail loud instead.
        n_channels = len(recording.get_channel_ids())
        if n_channels < 2:
            raise ValueError(
                "apply_pre_motion_preprocessing: 'global_median' reference on "
                f"a {n_channels}-channel sort group zeroes the signal (the "
                "median/mean across one channel is that channel itself). Use "
                "reference_mode='none' for unitrodes, or omit_unitrode=True "
                "when creating the sort group."
            )
        recording = sip.common_reference(
            recording,
            reference="global",
            operator=validated.common_reference.operator,
            dtype=_np.float64,
        )
    elif reference_mode != "none":
        raise ValueError(
            "Recording.make: invalid reference_mode "
            f"{reference_mode!r}. Use 'none', 'global_median', or "
            "'specific'."
        )
    return recording, applied_steps


def filtering_description(
    bandpass_filter, reference_mode: str, applied_steps: dict
) -> str:
    """``ElectricalSeries.filtering`` provenance from steps ACTUALLY run.

    Built from the preprocessing that actually ran so the persisted NWB
    metadata does not claim a step that did not happen -- e.g. the
    ``no_filter`` preset (``bandpass_filter`` None), ``reference_mode='none'``,
    or a phase-shift that was requested but skipped because the recording
    lacks an ``inter_sample_shift`` property. The phase-shift claim is driven
    by ``applied_steps`` (the report from ``apply_pre_motion_preprocessing``),
    not the params, precisely so a requested-but-skipped phase-shift is not
    falsely listed. Important for archival / DANDI export; the string is
    descriptive only and is not read back internally. Steps are listed in the
    order the runtime APPLIES them -- phase-shift first, then bandpass filter,
    then bad-channel interpolation, then common reference (see
    ``apply_pre_motion_preprocessing``) -- since that order is non-commutative
    on the global-median branch.

    The bad-channel interpolation claim is driven by ``applied_steps`` (the
    actual count that ran), not the params, and is appended ONLY when that count
    is > 0 -- so the default ``remove`` path (count 0) leaves the string
    byte-for-byte unchanged (this string is persisted into
    ``ElectricalSeries.filtering`` and hashed into ``cache_hash``).
    """
    steps = []
    if applied_steps.get("phase_shift"):
        steps.append("phase-shift (ADC)")
    if bandpass_filter is not None:
        steps.append(
            f"bandpass filter {bandpass_filter.freq_min:g}-"
            f"{bandpass_filter.freq_max:g} Hz"
        )
    n_interp = len(
        applied_steps.get("bad_channels", {}).get("interpolated", [])
    )
    if n_interp:
        steps.append(
            f"interpolate {n_interp} bad channel{'s' if n_interp != 1 else ''}"
        )
    if reference_mode != "none":
        steps.append(f"common reference ({reference_mode})")
    return "; ".join(steps) if steps else "none (raw, no preprocessing)"
