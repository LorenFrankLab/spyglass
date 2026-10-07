"""Reads behind ``MotionEstimate.report``.

:func:`motion_estimate_report` is the body of ``MotionEstimate.report``: it
fetches one estimate's stored arrays, where a concatenation's members join
and, given a corrected recording, its border mode and removed channels;
:func:`read_trace_window` reads original and corrected traces over a
window. Both hand what they read to ``_motion_report``, which summarizes and
draws it.

Imports without the DB layer: the DataJoint tables are imported inside the
functions.
"""

from __future__ import annotations

import uuid

import numpy as np


def id_restriction(value, field: str):
    """Restrict by ``field`` given its value or any mapping holding it.

    A ``uuid.UUID`` or ``str`` becomes ``{field: value}``; a mapping holding
    ``field`` (e.g. a pipeline receipt) keeps only that field; anything else
    is returned unchanged as a restriction.
    """
    from collections.abc import Mapping

    if isinstance(value, (uuid.UUID, str)):
        return {field: value}
    if isinstance(value, Mapping) and field in value:
        return {field: value[field]}
    return value


def motion_estimate_report(
    table,
    key,
    corrected_key=None,
    trace_window_s=None,
    *,
    trace_channel_ids=None,
):
    """Summarize and plot one estimate before (or after) applying it.

    The body of ``MotionEstimate.report`` (see its docstring for the
    arguments, returns and errors). ``table`` is a ``MotionEstimate``
    instance.
    """
    from spyglass.spikesorting.v2 import _motion, _motion_report
    from spyglass.spikesorting.v2.motion import (
        _ESTIMATION_CLOCK_COLUMNS,
        MotionCorrectedRecording,
        MotionCorrectedRecordingSelection,
        MotionEstimateSelection,
        MotionInterpolationParameters,
        _estimation_clock_of,
    )
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    if trace_window_s is not None and corrected_key is None:
        raise ValueError(
            "MotionEstimate.report: trace_window_s compares original and "
            "corrected traces; pass corrected_key too."
        )
    key = id_restriction(key, "motion_estimate_id")
    estimate_key = (table & key).fetch1("KEY")
    row = (
        (table & estimate_key)
        .proj(
            "motion",
            "resolved_params",
            "statistics_spans",
            "peaks_per_continuity_span",
            "channel_ids",
            "channel_locations",
            *_ESTIMATION_CLOCK_COLUMNS.values(),
        )
        .fetch1()
    )
    lineage = MotionEstimateSelection.resolve_source(estimate_key)
    member_join_frames = ()
    if lineage.kind == "concatenated_recording":
        ends = (ConcatenatedRecording.MemberBoundary & lineage.key).fetch(
            "end_sample", order_by="member_index"
        )
        member_join_frames = tuple(int(end) for end in ends[:-1])

    corrected = None
    border_mode = None
    if corrected_key is not None:
        corrected = (
            MotionCorrectedRecording
            & id_restriction(corrected_key, "motion_corrected_recording_id")
        ).fetch1()
        corrected_key = {
            name: corrected[name]
            for name in MotionCorrectedRecording.primary_key
        }
        selection = (MotionCorrectedRecordingSelection & corrected_key).fetch1()
        if str(selection["motion_estimate_id"]) != str(
            estimate_key["motion_estimate_id"]
        ):
            raise ValueError(
                "MotionEstimate.report: corrected recording "
                f"{corrected_key['motion_corrected_recording_id']} was "
                f"made from motion estimate "
                f"{selection['motion_estimate_id']}, not "
                f"{estimate_key['motion_estimate_id']}."
            )
        border_mode = _motion.resolve_interpolation_params(
            (
                MotionInterpolationParameters
                & {
                    "motion_interpolation_params_name": selection[
                        "motion_interpolation_params_name"
                    ]
                }
            ).fetch1("params")
        )["border_mode"]

    motion = _motion.motion_from_storage_dict(row["motion"])
    clock = _estimation_clock_of(row)
    inputs = _motion_report.MotionReportInputs(
        motion=motion,
        clock=clock,
        statistics_spans=row["statistics_spans"],
        peaks_per_continuity_span=row["peaks_per_continuity_span"],
        channel_ids=list(row["channel_ids"]),
        channel_locations=row["channel_locations"],
        max_gap_s=float(row["resolved_params"]["max_gap_s"]),
        member_join_frames=member_join_frames,
        border_mode=border_mode,
        removed_channel_ids=(
            None
            if corrected is None
            else list(corrected["removed_channel_ids"])
        ),
    )
    trace_window = None
    if trace_window_s is not None:
        trace_window = read_trace_window(
            estimate_key,
            corrected_key,
            corrected,
            clock,
            motion.dim,
            trace_window_s,
            trace_channel_ids,
        )
    summary = _motion_report.motion_report_summary(inputs)
    summary["motion_estimate_id"] = estimate_key["motion_estimate_id"]
    summary["motion_corrected_recording_id"] = (
        None
        if corrected_key is None
        else corrected_key["motion_corrected_recording_id"]
    )
    return (
        _motion_report.plot_motion_report(inputs, trace_window, summary),
        summary,
    )


def read_trace_window(
    estimate_key,
    corrected_key,
    corrected,
    clock,
    depth_dim,
    trace_window_s,
    trace_channel_ids,
):
    """Original and corrected traces of a few channels over a window.

    The window's source-clock times are mapped to frames with each
    span's affine map (:func:`._motion_report.frame_of_source_time`);
    the corrected recording has the source's frames. Both are read in
    microvolts (:func:`._motion.recording_in_microvolts`).
    """
    from spyglass.spikesorting.v2 import _motion, _motion_report
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecording,
        _estimate_source,
    )

    start_frame, end_frame = (
        int(f)
        for f in _motion_report.frame_of_source_time(
            clock, np.asarray(trace_window_s, dtype=np.float64)
        )
    )
    if end_frame <= start_frame:
        raise ValueError(
            f"MotionEstimate.report: trace_window_s {trace_window_s} "
            "selects no frame of the source."
        )
    kept = list(corrected["channel_ids"])
    if trace_channel_ids is None:
        trace_channel_ids = _motion_report.default_trace_channels(
            kept, corrected["channel_locations"], depth_dim
        )
    missing = [c for c in trace_channel_ids if c not in kept]
    if missing:
        raise ValueError(
            f"MotionEstimate.report: trace channels {missing} are not in "
            f"corrected recording "
            f"{corrected_key['motion_corrected_recording_id']} (its "
            f"channels: {kept})."
        )
    table, lineage, _ = _estimate_source(estimate_key["motion_estimate_id"])
    frames = {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "channel_ids": list(trace_channel_ids),
    }
    original = _motion.recording_in_microvolts(
        table().get_recording(lineage.key)
    ).get_traces(**frames)
    corrected_traces = _motion.recording_in_microvolts(
        MotionCorrectedRecording().get_recording(corrected_key)
    ).get_traces(**frames)
    return _motion_report.TraceWindow(
        source_time_s=_motion_report.source_time_of_frames(
            clock, np.arange(start_frame, end_frame)
        ),
        channel_ids=list(trace_channel_ids),
        original_uv=original,
        corrected_uv=corrected_traces,
    )
