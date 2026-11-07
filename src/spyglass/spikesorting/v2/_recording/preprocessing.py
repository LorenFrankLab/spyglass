"""Pre-motion preprocessing services behind ``Recording``.

The caller applies the stack in this order (no whitening):

1. ``apply_temporal_preprocessing`` (phase-shift, bandpass) on the continuous
   recording, so filters do not ring at artificial interval joins.
2. ``restrict_recording``.
3. ``apply_spatial_preprocessing`` (bad-channel interpolation, reference),
   which is per-sample and unaffected by joins.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from spyglass.spikesorting.v2._core.signal_math import (
    assert_freq_max_below_nyquist,
    base_intervals_and_gaps,
)
from spyglass.utils import logger


def _split_acquisition_spans(recording):
    """Isolate real timestamp gaps before operations with temporal support."""
    from spikeinterface import append_recordings

    from spyglass.spikesorting.v2._recording.acquisition_spans import (
        AcquisitionSpanRecording,
    )

    spans = []
    split = False
    for index in range(recording.get_num_segments()):
        segment = recording.select_segments([index])
        if not segment.has_time_vector():
            spans.append(segment)
            continue
        gaps = base_intervals_and_gaps(segment).gap_after
        if not len(gaps):
            spans.append(segment)
            continue
        split = True
        edges = np.r_[0, gaps + 1, segment.get_num_samples()]
        spans.extend(
            AcquisitionSpanRecording(segment, start, stop)
            for start, stop in pairwise(edges)
        )
    return append_recordings(spans) if split else recording


def apply_temporal_preprocessing(recording, validated):
    """Apply the temporal preprocessing steps (phase-shift, then bandpass).

    Call on the continuous channel-sliced recording, before time
    restriction. Real acquisition gaps are split into separate segments
    first so filter margins never cross missing data. Bandpass precedes
    the global-median reference, which does not commute with it.

    Returns ``(recording, applied_steps)``; ``applied_steps`` is
    ``{"phase_shift": bool}``, since phase-shift is skipped without
    ``inter_sample_shift``.

    Parameters
    ----------
    recording : si.BaseRecording
        The channel-sliced, time-CONTINUOUS recording to preprocess.
    validated : PreprocessingParamsSchema
        Pre-validated preprocessing params (phase-shift, bandpass) read once
        in ``make_fetch``.

    Returns
    -------
    tuple
        ``(recording, applied_steps)``.
    """
    import spikeinterface.preprocessing as sip

    applied_steps: dict = {}

    if (
        validated.phase_shift is not None
        or validated.bandpass_filter is not None
    ):
        recording = _split_acquisition_spans(recording)

    # sip.phase_shift raises without inter_sample_shift, so warn and skip.
    applied_steps["phase_shift"] = False
    if validated.phase_shift is not None:
        if recording.get_property("inter_sample_shift") is not None:
            recording = sip.phase_shift(
                recording, margin_ms=validated.phase_shift.margin_ms
            )
            applied_steps["phase_shift"] = True
        else:
            logger.warning(
                "apply_temporal_preprocessing: phase_shift requested but the "
                "recording has no 'inter_sample_shift' property (not a "
                "multiplexed-ADC acquisition); skipping phase-shift."
            )

    if validated.bandpass_filter is not None:
        # The schema cannot know fs; scipy fails opaquely at/above Nyquist.
        assert_freq_max_below_nyquist(
            validated.bandpass_filter.freq_max,
            recording.get_sampling_frequency(),
            context="apply_temporal_preprocessing: ",
        )
        recording = sip.bandpass_filter(
            recording,
            freq_min=validated.bandpass_filter.freq_min,
            freq_max=validated.bandpass_filter.freq_max,
            dtype=np.float64,
        )

    return recording, applied_steps


def apply_spatial_preprocessing(
    recording,
    reference_mode: str,
    reference_electrode_id: int | None,
    validated,
    bad_channel_handling: str = "remove",
    bad_channel_ids=(),
):
    """Apply the spatial preprocessing steps (bad-channel fill, reference).

    Call on the time-restricted recording. Returns ``(recording,
    applied_steps)``; ``applied_steps`` is ``{"bad_channels":
    {"interpolated": [...]}}``.

    Parameters
    ----------
    recording : si.BaseRecording
        The channel- and time-restricted, temporally preprocessed recording.
    reference_mode : str
        One of ``"none"``, ``"global_median"``, or ``"specific"``.
    reference_electrode_id : int or None
        Electrode id subtracted on the ``"specific"`` path and dropped
        from the surface afterward; ignored otherwise.
    validated : PreprocessingParamsSchema
        Pre-validated preprocessing params (common-reference operator) read
        once in ``make_fetch``.
    bad_channel_handling : str, optional
        ``"remove"`` (default) or ``"interpolate"``. Only
        ``"interpolate"`` acts here, filling the interior bad channels.
    bad_channel_ids : sequence of int, optional
        Interior curated-bad electrode ids to interpolate on the
        ``"interpolate"`` path. Default ``()``.

    Returns
    -------
    tuple
        ``(recording, applied_steps)``.

    Raises
    ------
    ValueError
        If referencing would mix channels with unequal or non-finite current
        gains/offsets, or non-positive gains. SI references raw counts, so
        every contributing channel must share its calibration.
    """
    import spikeinterface.preprocessing as sip

    applied_steps: dict = {}

    # Between filter and reference, as in the IBL/AIND destripe order.
    if bad_channel_handling not in ("remove", "interpolate"):
        raise ValueError(
            "apply_spatial_preprocessing: invalid bad_channel_handling "
            f"{bad_channel_handling!r}; use 'remove' or 'interpolate'."
        )
    ref_id = (
        int(reference_electrode_id) if reference_mode == "specific" else None
    )
    to_interpolate: list[int] = []
    if bad_channel_handling == "interpolate":
        present = {int(c) for c in recording.get_channel_ids()}
        requested = {int(c) for c in bad_channel_ids}
        missing = requested - present
        if missing:
            raise RuntimeError(
                "apply_spatial_preprocessing: interior bad channels "
                f"{sorted(missing)} are absent from the recording surface; "
                "select_sort_group_channels must slice them in for "
                "interpolation."
            )
        to_interpolate = sorted(requested - {ref_id})
    if to_interpolate:
        # SI would raise opaquely deep in the kriging call.
        if not recording.has_channel_location():
            raise ValueError(
                "apply_spatial_preprocessing: interpolate requires channel "
                "locations on the recording, but none are set (no probe "
                "geometry). Use bad_channel_handling='remove', or ensure the "
                "NWB / probe carries electrode positions."
            )
        recording = sip.interpolate_bad_channels(recording, to_interpolate)
    applied_steps["bad_channels"] = {"interpolated": to_interpolate}

    # SI subtracts raw counts, so every contributing channel must share its
    # calibration. Check before the reference is dropped or offsets zeroed.
    if reference_mode in ("specific", "global_median"):
        for property_name, label in (
            ("gain_to_uV", "channel gains"),
            ("offset_to_uV", "channel offsets"),
        ):
            values = recording.get_property(property_name)
            if values is None:
                continue
            values = np.asarray(values)
            positive_gain = property_name != "gain_to_uV" or np.all(values > 0)
            if (
                not np.all(np.isfinite(values))
                or np.unique(values).size != 1
                or not positive_gain
            ):
                requirement = (
                    "uniform finite positive"
                    if property_name == "gain_to_uV"
                    else "uniform finite"
                )
                raise ValueError(
                    f"apply_spatial_preprocessing: {reference_mode!r} reference "
                    f"requires {requirement} {label} across every contributing "
                    "channel, including any specific reference electrode; raw-count "
                    "subtraction cannot preserve this per-channel calibration. "
                    "Verify the recording's gain/offset metadata before referencing."
                )
    if reference_mode == "specific":
        recording = sip.common_reference(
            recording,
            reference="single",
            ref_channel_ids=[int(reference_electrode_id)],
            dtype=np.float64,
        )
        # The reference was sliced in only for subtraction.
        channel_ids = [int(c) for c in recording.get_channel_ids()]
        if int(reference_electrode_id) not in channel_ids:
            raise RuntimeError(
                "apply_spatial_preprocessing: 'specific' reference "
                f"electrode {int(reference_electrode_id)} is absent after "
                f"referencing (channels={channel_ids}); cannot drop it from "
                "the sort surface. select_sort_group_channels must slice the "
                "reference channel in for subtraction."
            )
        recording = recording.remove_channels([int(reference_electrode_id)])
        # Referencing cancels the uniform DC offset, but SI keeps the parent
        # offset. set_channel_offsets is in place and returns None.
        recording.set_channel_offsets(0.0)
    elif reference_mode == "global_median":
        # A global reference on one channel zeroes the signal.
        n_channels = len(recording.get_channel_ids())
        if n_channels < 2:
            raise ValueError(
                "apply_spatial_preprocessing: 'global_median' reference on "
                f"a {n_channels}-channel sort group zeroes the signal (the "
                "median/mean across one channel is that channel itself). Use "
                "reference_mode='none' for unitrodes, or omit_unitrode=True "
                "when creating the sort group."
            )
        recording = sip.common_reference(
            recording,
            reference="global",
            operator=validated.common_reference.operator,
            dtype=np.float64,
        )
        # See the 'specific' branch.
        recording.set_channel_offsets(0.0)
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

    Phase-shift and interpolation come from ``applied_steps`` (the merged
    reports of both ``apply_*_preprocessing`` calls), not the params. Steps
    are listed in application order. The string is hashed into
    ``content_hash``.
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
