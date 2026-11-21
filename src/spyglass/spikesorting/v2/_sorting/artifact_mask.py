"""Artifact-mask service behind ``Sorting``."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from spyglass.spikesorting.v2._core.signal_math import (
    artifact_frame_ranges,
    boundary_spans_from_timestamps,
    complement_frame_ranges,
    normalize_spans,
    spans_cover_recording,
)
from spyglass.spikesorting.v2._storage.spikeinterface import (
    use_pickle_recording_serialization,
)
from spyglass.utils import logger
from spyglass.utils.interval import Interval


def apply_artifact_mask(
    recording, valid_times, *, artifact_detection_id=None, recording_id=None
):
    """Mask excluded periods lazily, preserving frames and timestamps."""
    ranges = artifact_frame_ranges(
        recording,
        valid_times,
        artifact_detection_id=artifact_detection_id,
        recording_id=recording_id,
    )
    return silence_frame_ranges(recording, ranges)


def has_nonzero_offset(recording) -> bool:
    """Say whether a recording keeps a nonzero channel calibration offset.

    A stored zero reads as 0 uV only when every offset is 0. Filtering and
    referencing zero the offset.

    Parameters
    ----------
    recording : si.BaseRecording

    Returns
    -------
    bool
    """
    offsets = recording.get_channel_offsets()
    return offsets is not None and bool(np.any(offsets != 0))


def recording_with_zero_offset(recording):
    """Present a recording so that a stored zero reads as 0 uV.

    Returned unchanged unless it has a nonzero offset, in which case it is
    converted to float32 µV with ``scale_to_uV``.

    Parameters
    ----------
    recording : si.BaseRecording
        The recording to be masked.

    Returns
    -------
    si.BaseRecording
        ``recording`` itself, or its float32 microvolt view.
    """
    if not has_nonzero_offset(recording):
        return recording
    import spikeinterface.preprocessing as sip

    return sip.scale_to_uV(recording)


def silence_frame_ranges(recording, frame_ranges):
    """Silence validated half-open frame ranges to 0 uV.

    Zeroed lazily. A recording with a nonzero offset is first converted
    (:func:`recording_with_zero_offset`); others keep their dtype and
    calibration.

    Parameters
    ----------
    recording : si.BaseRecording
        A single-segment recording.
    frame_ranges : sequence of (int, int)
        Validated, sorted, half-open ``[start, end)`` frame ranges.

    Returns
    -------
    si.BaseRecording
        The silenced recording, or ``recording`` when there are no ranges.
    """
    import spikeinterface.preprocessing as sip

    if not len(frame_ranges):
        return recording
    recording = recording_with_zero_offset(recording)

    # Lazy and O(n_ranges) memory; remove_artifacts also mis-masks a boundary
    # frame of a padded window in SI 0.104.
    masked = sip.silence_periods(
        recording,
        list_periods=[frame_ranges],
        mode="zeros",
    )

    # SilencedPeriodsRecording claims JSON support but cannot reload from it.
    use_pickle_recording_serialization(masked)
    return masked


def fill_masked_frames_with_noise(recording, spans, *, noise_levels, seed):
    """Fill the frames outside ``spans`` with seeded Gaussian noise.

    For sorters that estimate their own noise and whitening from the traces,
    where zeros would bias both low. Valid frames are returned unchanged.

    Parameters
    ----------
    recording : si.BaseRecording
        A single-segment recording whose frames outside ``spans`` are zero
        (the masked sort input).
    spans : list[tuple[int, int]]
        Statistics spans: sorted, half-open frames of artifact-free data.
    noise_levels : numpy.ndarray
        ``(n_channels,)`` per-channel noise standard deviation in the
        recording's stored (raw) units.
    seed : int
        Seed of the noise; the same seed reloads the same fill.

    Returns
    -------
    si.BaseRecording
        The filled recording (float32 for an integer input), or
        ``recording`` when ``spans`` cover it.
    """
    import spikeinterface.preprocessing as sip

    n_samples = int(recording.get_num_samples())
    if spans_cover_recording(spans, n_samples):
        return recording
    if seed is None:
        raise ValueError(
            "fill_masked_frames_with_noise: seed is required; an unseeded "
            "fill differs when a sorter worker reloads the recording."
        )
    # SI sizes its noise generator from get_duration(), which can floor one
    # frame short; frames past it keep the zeros already there.
    n_noise = int(recording.get_duration() * recording.get_sampling_frequency())
    periods = [
        (start, min(end, n_noise))
        for start, end in complement_frame_ranges(spans, n_samples)
        if start < n_noise
    ]
    if not periods:
        return recording
    if recording.get_dtype().kind in "iu":
        # SI generates noise only in float dtypes.
        recording = sip.astype(recording, "float32")
    filled = sip.silence_periods(
        recording,
        list_periods=[periods],
        mode="noise",
        noise_levels=np.asarray(noise_levels, dtype=np.float64),
        seed=int(seed),
    )
    use_pickle_recording_serialization(filled)
    return filled


class Continuity(NamedTuple):
    """Continuity spans and the real timestamps that bound each one.

    Attributes
    ----------
    spans : list[tuple[int, int]]
        Half-open frame spans of uninterrupted acquisition.
    start_s : list[float]
        Each span's first timestamp on its own acquisition clock (s).
    end_s : list[float]
        Each span's last timestamp on that clock (s): the time of frame
        ``b - 1``, not ``start + (b - a) / fs``, which can drift.
    """

    spans: list[tuple[int, int]]
    start_s: list[float]
    end_s: list[float]


def continuity_from_timestamps(recording) -> Continuity:
    """Continuity spans of ``recording`` with their first and last timestamps.

    Boundaries are read with scalar lookups, not the full time vector.

    Parameters
    ----------
    recording : si.BaseRecording
        Single-segment recording carrying its persisted timestamps.

    Returns
    -------
    Continuity
    """
    spans = boundary_spans_from_timestamps(recording)
    return Continuity(
        spans=spans,
        start_s=[float(recording.sample_index_to_time(a)) for a, _ in spans],
        end_s=[float(recording.sample_index_to_time(b - 1)) for _, b in spans],
    )


def statistics_spans(
    n_samples: int,
    excluded_ranges: list[tuple[int, int]],
    boundary_spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Artifact-free frame spans that each lie inside a single boundary span.

    Adjacent output spans are never merged; the edge between them is a join.

    Parameters
    ----------
    n_samples : int
        Total number of frames.
    excluded_ranges : list[tuple[int, int]]
        Half-open artifact-masked frame ranges (need not be sorted/merged).
    boundary_spans : list[tuple[int, int]]
        Disjoint half-open frame spans that never cross a join (e.g. from
        ``boundary_spans_from_timestamps`` / ``_recording.concat.
        concat_continuity``); adjacent spans may touch.

    Returns
    -------
    list[tuple[int, int]]
        Sorted, half-open statistics spans.

    Raises
    ------
    ValueError
        If no artifact-free frame lies inside any boundary span.
    """
    n_samples = int(n_samples)
    valid = complement_frame_ranges(excluded_ranges, n_samples)
    valid_samples = sum(b - a for a, b in valid)
    out = [
        (int(a), int(b))
        for a, b in Interval(valid, warn=False).intersect(boundary_spans).times
    ]
    if not out:
        raise ValueError(
            "statistics_spans: no artifact-free samples inside any "
            "acquisition span."
        )
    masked_fraction = (
        (n_samples - valid_samples) / n_samples if n_samples else 0.0
    )
    logger.info(
        "statistics_spans: masked_fraction=%.4f across %d statistics span(s) "
        "(n_samples=%d).",
        masked_fraction,
        len(out),
        n_samples,
    )
    return out


def corrected_statistics_spans(
    recording,
    row: dict,
    *,
    source_n_samples: int,
    concat_statistics_spans,
    obs_intervals,
    artifact_detection_id,
    recording_id,
) -> list[tuple[int, int]]:
    """Return a corrected recording's statistics spans after checking them.

    The row's copied spans must equal those derived from the corrected
    traces and the sort's mask (single recording) or the concatenation's
    stored spans, and the frame count must match the source's.

    Parameters
    ----------
    recording : si.BaseRecording
        The loaded corrected recording.
    row : dict
        Its ``MotionCorrectedRecording`` row (``motion_corrected_recording_id``,
        ``n_samples``, ``statistics_spans`` ``(n, 2)``).
    source_n_samples : int
        The source's frame count.
    concat_statistics_spans : numpy.ndarray or None
        A concat source's stored ``(n, 2)`` spans; ``None`` for a single
        recording.
    obs_intervals : numpy.ndarray or None
        The sort's artifact-removed valid times, ``(n_intervals, 2)`` in
        seconds, or ``None`` without an artifact detection.
    artifact_detection_id : uuid.UUID or None
        The sort's pinned detection (single recording only).
    recording_id : uuid.UUID or None
        The single-recording source, for error messages.

    Returns
    -------
    list[tuple[int, int]]
        The half-open frame spans.

    Raises
    ------
    ValueError
        If the frame count or the spans disagree with the source's.
    """

    n_samples = int(recording.get_num_samples())
    if not n_samples == int(row["n_samples"]) == int(source_n_samples):
        raise ValueError(
            "Sorting: motion-corrected recording "
            f"{row['motion_corrected_recording_id']} has {n_samples} "
            f"frames (row: {int(row['n_samples'])}); its source has "
            f"{int(source_n_samples)}."
        )
    spans = normalize_spans(row["statistics_spans"])
    if concat_statistics_spans is None:
        excluded = []
        if artifact_detection_id is not None:
            excluded = artifact_frame_ranges(
                recording,
                obs_intervals,
                artifact_detection_id=artifact_detection_id,
                recording_id=recording_id,
            )
        expected = normalize_spans(
            statistics_spans(
                n_samples,
                excluded,
                boundary_spans_from_timestamps(recording),
            )
        )
    else:
        expected = normalize_spans(concat_statistics_spans)
    if spans != expected:
        raise ValueError(
            "Sorting: motion-corrected recording "
            f"{row['motion_corrected_recording_id']} carries statistics "
            f"spans {spans}, but its source and the sort's mask give "
            f"{expected}."
        )
    return spans


def sample_span_data(
    recording,
    spans: list[tuple[int, int]],
    *,
    target_samples: int,
    max_piece: int,
    seed,
    return_in_uV: bool,
):
    """Randomly sample traces from ``recording`` without crossing a span edge.

    When ``target_samples`` covers every valid frame, each span is read
    once. Otherwise each span's quota (largest-remainder share of its
    length) is filled with random pieces of at most ``max_piece`` frames,
    returned in start-frame order.

    Parameters
    ----------
    recording : si.BaseRecording
    spans : list[tuple[int, int]]
        Half-open frame spans to sample from (e.g. ``statistics_spans``).
    target_samples : int
        Keyword-only. Requested row budget.
    max_piece : int
        Keyword-only. Maximum contiguous frames read per random draw.
    seed : int
        Keyword-only. Seeds ``numpy.random.default_rng``.
    return_in_uV : bool
        Keyword-only. Forwarded to ``recording.get_traces``.

    Returns
    -------
    numpy.ndarray
        ``(min(target_samples, total_valid), n_channels)`` traces.
    """
    rng = np.random.default_rng(seed)
    lengths = np.array([b - a for a, b in spans], dtype=np.int64)
    total_valid = int(lengths.sum())
    effective_target = min(int(target_samples), total_valid)
    if effective_target == total_valid:
        data = np.concatenate(
            [
                recording.get_traces(
                    start_frame=a, end_frame=b, return_in_uV=return_in_uV
                )
                for a, b in spans
            ],
            axis=0,
        )
        return data
    # Quotas sum to effective_target and never exceed a span's length.
    raw = effective_target * lengths / total_valid
    quota = np.minimum(np.floor(raw).astype(np.int64), lengths)
    for i in np.argsort(-(raw - np.floor(raw))):
        if quota.sum() >= effective_target:
            break
        if quota[i] < lengths[i]:
            quota[i] += 1
    pieces: list[tuple[int, np.ndarray]] = []
    for (a, b), q in zip(spans, quota):
        remaining = int(q)
        while remaining > 0:
            piece = min(max_piece, remaining, b - a)
            s = int(rng.integers(a, b - piece + 1))
            pieces.append(
                (
                    s,
                    recording.get_traces(
                        start_frame=s,
                        end_frame=s + piece,
                        return_in_uV=return_in_uV,
                    ),
                )
            )
            remaining -= piece
    pieces.sort(key=lambda item: item[0])
    data = np.concatenate([piece for _, piece in pieces], axis=0)
    return data


def sorting_statistics_spans(
    recording,
    traces,
    *,
    source_kind,
    recording_id,
    artifact_detection_id,
    obs_intervals,
    concat_statistics_spans,
    source_n_samples,
):
    """Mask a sort's input and resolve its statistics spans.

    Statistics spans are the artifact-free, join-free frame ranges that
    noise and whitening estimates sample from.

    - Motion-corrected: persisted masked; spans checked against its row.
    - Single recording: masked here; boundaries from its timestamps.
    - Concatenation: masks and spans were stored upstream.

    Parameters
    ----------
    recording : spikeinterface.BaseRecording
        The sort input, loaded from the effective traces.
    traces : EffectiveTraces
        The sort's effective traces (``kind``, ``row``,
        ``apply_artifact_mask``).
    source_kind : str
        ``"recording"`` or ``"concatenated_recording"``.
    recording_id : str
        The sort's anchor ``recording_id``.
    artifact_detection_id : str or None
        The sort's artifact detection, if any.
    obs_intervals : numpy.ndarray or None
        Artifact-removed valid times, shape ``(n_intervals, 2)`` in seconds.
    concat_statistics_spans : numpy.ndarray or None
        A concat source's stored spans; ``None`` for a single recording.
    source_n_samples : int or None
        The source's frame count a corrected recording must keep.

    Returns
    -------
    tuple
        ``(recording, statistics_spans)``: the recording to sort (silenced
        over the artifact frames of a masked single recording, otherwise
        unchanged) and the spans as ``(start_frame, end_frame)`` pairs.
    """
    if traces.kind == "motion_corrected_recording":
        spans = corrected_statistics_spans(
            recording,
            traces.row,
            source_n_samples=source_n_samples,
            concat_statistics_spans=concat_statistics_spans,
            obs_intervals=obs_intervals,
            artifact_detection_id=artifact_detection_id,
            recording_id=(recording_id if source_kind == "recording" else None),
        )
    elif source_kind == "recording":
        boundary_spans = boundary_spans_from_timestamps(recording)
        excluded_ranges = []
        if traces.apply_artifact_mask:
            excluded_ranges = artifact_frame_ranges(
                recording,
                obs_intervals,
                artifact_detection_id=artifact_detection_id,
                recording_id=recording_id,
            )
            recording = silence_frame_ranges(recording, excluded_ranges)
        spans = statistics_spans(
            recording.get_num_samples(), excluded_ranges, boundary_spans
        )
    else:

        spans = normalize_spans(concat_statistics_spans)
    return recording, spans
