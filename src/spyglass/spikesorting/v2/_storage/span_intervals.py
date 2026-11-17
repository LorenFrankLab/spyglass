"""Span tables in the concatenated and motion-corrected trace NWB files.

Standard ``TimeIntervals`` in ``nwbfile.intervals``, so a reader with plain
pynwb can tell which samples are zero because of a mask and where sessions
were joined. They copy the row's span columns, which the code reads; the
content fingerprint hashes only the ``ElectricalSeries`` and ignores them.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pynwb
from hdmf.common import VectorData, VectorIndex
from pynwb.epoch import TimeIntervals

from spyglass.spikesorting.v2._core.signal_math import (
    _segment_times_at,
    complement_frame_ranges,
)
from spyglass.spikesorting.v2._storage.nwb import _ELECTRICAL_SERIES_PATH
from spyglass.spikesorting.v2._storage.provenance import concrete_vectordata

STATISTICS_SPANS_INTERVALS = "spyglass_v2_statistics_spans"
CONTINUITY_SPANS_INTERVALS = "spyglass_v2_continuity_spans"
CONCAT_MEMBER_INTERVALS = "spyglass_v2_concat_member_spans"
SPAN_INTERVAL_NAMES = (
    STATISTICS_SPANS_INTERVALS,
    CONTINUITY_SPANS_INTERVALS,
    CONCAT_MEMBER_INTERVALS,
)

_TIMES_TEXT = (
    "start_sample and end_sample are sample indices into this file's "
    f"{_ELECTRICAL_SERIES_PATH}, end exclusive. start_time and stop_time "
    "are the timestamps of the first and last sample of the row, taken from "
    "that series' timestamps"
)
_CONCAT_CLOCK_TEXT = (
    ", which for a concatenation are a synthetic timeline that starts at "
    "0 s and leaves out the gaps between member sessions."
)
_SINGLE_CLOCK_TEXT = "."


def _frame_columns(clock_recording, spans) -> list:
    """``start_time``, ``stop_time``, ``start_sample`` and ``end_sample``."""
    spans = np.asarray(spans, dtype=np.int64).reshape(-1, 2)
    times = _segment_times_at(
        clock_recording, np.column_stack([spans[:, 0], spans[:, 1] - 1])
    ).reshape(-1, 2)
    return [
        concrete_vectordata(
            "start_time",
            times[:, 0],
            np.float64,
            description="time of the row's first sample, in seconds",
        ),
        concrete_vectordata(
            "stop_time",
            times[:, 1],
            np.float64,
            description="time of the row's last sample, in seconds",
        ),
        concrete_vectordata(
            "start_sample",
            spans[:, 0],
            np.int64,
            description="index of the row's first sample in the series",
        ),
        concrete_vectordata(
            "end_sample",
            spans[:, 1],
            np.int64,
            description=(
                "index one past the row's last sample in the series (exclusive)"
            ),
        ),
    ]


def _ragged_intervals_columns(name, per_row, *, description) -> list:
    """A ragged ``(n_i, 2)`` float64 column and its index, one entry per row."""
    arrays = [np.asarray(v, dtype=np.float64).reshape(-1, 2) for v in per_row]
    data = VectorData(
        name=name,
        description=description,
        data=(
            np.concatenate(arrays)
            if arrays
            else np.empty((0, 2), dtype=np.float64)
        ),
    )
    return [
        data,
        VectorIndex(
            name=f"{name}_index",
            data=np.cumsum([len(a) for a in arrays], dtype=np.int64),
            target=data,
        ),
    ]


def span_interval_tables(
    clock_recording,
    *,
    statistics_spans,
    continuity_spans,
    continuity_start_s,
    continuity_end_s,
    estimation_start_s=None,
    concat_members: list[dict] | None = None,
    is_concat: bool,
) -> list[TimeIntervals]:
    """Build the span tables a trace file carries.

    Parameters
    ----------
    clock_recording : si.BaseRecording
        A recording whose timestamps are the ones the series stores.
    statistics_spans : array-like
        ``(n, 2)`` half-open artifact-free sample ranges.
    continuity_spans : array-like
        ``(m, 2)`` half-open sample ranges of uninterrupted acquisition.
    continuity_start_s, continuity_end_s : array-like
        ``(m,)`` first and last time of each continuity span on the clock it
        was recorded on, in seconds.
    estimation_start_s : array-like, optional
        ``(m,)`` start of each continuity span on a motion estimate's
        estimation clock, in seconds, for a motion-corrected file.
    concat_members : list of dict, optional
        One ``{"member_index", "nwb_file_name", "interval_list_name",
        "start_sample", "end_sample", "valid_times"}`` per member of a
        concatenation, in order; ``valid_times`` is ``(k, 2)`` kept intervals
        on the member's own clock. Adds the member table.
    is_concat : bool
        The series' timestamps are a concatenation's synthetic timeline.

    Returns
    -------
    list of pynwb.epoch.TimeIntervals
    """
    clock_text = _CONCAT_CLOCK_TEXT if is_concat else _SINGLE_CLOCK_TEXT
    continuity_columns = _frame_columns(clock_recording, continuity_spans) + [
        concrete_vectordata(
            "acquisition_start_time",
            continuity_start_s,
            np.float64,
            description=(
                "time of the row's first sample on its acquisition clock, "
                "in seconds"
            ),
        ),
        concrete_vectordata(
            "acquisition_stop_time",
            continuity_end_s,
            np.float64,
            description=(
                "time of the row's last sample on its acquisition clock, "
                "in seconds"
            ),
        ),
    ]
    estimation_text = ""
    if estimation_start_s is not None:
        continuity_columns.append(
            concrete_vectordata(
                "estimation_start_time",
                estimation_start_s,
                np.float64,
                description=(
                    "time of the row's first sample on the motion-estimation "
                    "clock, in seconds"
                ),
            )
        )
        estimation_text = (
            " estimation_start_time is the row's start on the clock the "
            "motion estimate was computed on, where each row advances one "
            "sample period per sample and gaps between rows are capped."
        )
    tables = [
        TimeIntervals(
            name=STATISTICS_SPANS_INTERVALS,
            description=(
                "Each row is a run of samples that is free of artifacts and "
                "lies inside one continuity span. Samples outside every row "
                "were set to zero by the artifact mask; merging rows that "
                "touch gives the observed (unmasked) time. "
                + _TIMES_TEXT
                + clock_text
            ),
            columns=_frame_columns(clock_recording, statistics_spans),
        ),
        TimeIntervals(
            name=CONTINUITY_SPANS_INTERVALS,
            description=(
                "Each row is a run of samples recorded without interruption; "
                "rows split at every gap in the acquisition timestamps"
                + (
                    " and at every join between member sessions"
                    if is_concat
                    else ""
                )
                + ". "
                + _TIMES_TEXT
                + clock_text
                + " acquisition_start_time and acquisition_stop_time are the "
                "same two samples' times on the clock they were recorded on"
                + (" (the member session's own clock)." if is_concat else ".")
                + estimation_text
            ),
            columns=continuity_columns,
        ),
    ]
    if concat_members is not None:
        tables.append(
            TimeIntervals(
                name=CONCAT_MEMBER_INTERVALS,
                description=(
                    "Each row is one member session of the concatenation, in "
                    "order; the member's samples are the rows' sample range. "
                    + _TIMES_TEXT
                    + _CONCAT_CLOCK_TEXT
                    + " valid_times lists the member's kept (artifact-free) "
                    "intervals as [start, stop] pairs in seconds on the "
                    "member session's own clock."
                ),
                columns=_frame_columns(
                    clock_recording,
                    [
                        (m["start_sample"], m["end_sample"])
                        for m in concat_members
                    ],
                )
                + [
                    concrete_vectordata(
                        "member_index",
                        [m["member_index"] for m in concat_members],
                        np.int64,
                        description="position of the member in the order",
                    ),
                    concrete_vectordata(
                        "nwb_file_name",
                        [m["nwb_file_name"] for m in concat_members],
                        object,
                        description="the member session's NWB file name",
                    ),
                    concrete_vectordata(
                        "interval_list_name",
                        [m["interval_list_name"] for m in concat_members],
                        object,
                        description=(
                            "the member's interval list in that session"
                        ),
                    ),
                ]
                + _ragged_intervals_columns(
                    "valid_times",
                    [m["valid_times"] for m in concat_members],
                    description=(
                        "the member's kept intervals on its own clock, "
                        "[start, stop] in seconds"
                    ),
                ),
            )
        )
    return tables


def read_span_intervals(abs_path: str) -> dict:
    """Read the span tables a trace file carries, with plain pynwb.

    Parameters
    ----------
    abs_path : str
        Absolute path of the trace NWB file.

    Returns
    -------
    dict[str, pandas.DataFrame]
        The tables present, by name; empty for a file written without them.
    """
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        intervals = io.read().intervals or {}
        return {
            name: intervals[name].to_dataframe()
            for name in SPAN_INTERVAL_NAMES
            if name in intervals
        }


class TraceFileSpans(NamedTuple):
    """The span facts a trace file holds; ``None`` where it holds none.

    Attributes
    ----------
    statistics_spans, continuity_spans : numpy.ndarray or None
        ``(n, 2)`` int64 half-open sample ranges.
    continuity_start_s, continuity_end_s : numpy.ndarray or None
        ``(n,)`` acquisition-clock times of each continuity span's first and
        last sample.
    estimation_start_s : numpy.ndarray or None
        ``(n,)`` start of each continuity span on the motion-estimation clock.
    member_end_samples : numpy.ndarray or None
        Each concat member's exclusive end sample, in member order.
    member_valid_times : list of numpy.ndarray or None
        Each concat member's ``(k, 2)`` kept intervals on its own clock.
    observed_spans : numpy.ndarray or None
        ``(n, 2)`` merged unmasked sample ranges of a concatenation.
    """

    statistics_spans: np.ndarray | None
    continuity_spans: np.ndarray | None
    continuity_start_s: np.ndarray | None
    continuity_end_s: np.ndarray | None
    estimation_start_s: np.ndarray | None
    member_end_samples: np.ndarray | None
    member_valid_times: list | None
    observed_spans: np.ndarray | None


def _spans(frame, start="start_sample", end="end_sample"):
    return frame[[start, end]].to_numpy(dtype=np.int64).reshape(-1, 2)


def _merge_touching(spans) -> np.ndarray:
    """Merge half-open ranges that touch, in order."""
    merged: list[list[int]] = []
    for start, end in np.asarray(spans, dtype=np.int64).reshape(-1, 2):
        if merged and merged[-1][1] == start:
            merged[-1][1] = int(end)
        else:
            merged.append([int(start), int(end)])
    return np.asarray(merged, dtype=np.int64).reshape(-1, 2)


def read_trace_file_spans(abs_path: str) -> TraceFileSpans:
    """Read a concat or motion-corrected file's spans, in any file generation.

    The standard ``TimeIntervals`` tables are read when present. Files written
    before them hold the motion spans in the scratch objects
    ``MOTION_CORRECTION_PROVENANCE`` and ``MOTION_CONTINUITY_SPANS``, and a
    concat's observed time as the header's ``artifact_frame_ranges``; those
    are read instead. The database row stays the copy the code reads.

    Parameters
    ----------
    abs_path : str
        Absolute path of the trace NWB file.

    Returns
    -------
    TraceFileSpans
    """
    from spyglass.spikesorting.v2._storage.provenance import (
        CONCAT_MEMBERS,
        CONCAT_PROVENANCE,
        MOTION_CONTINUITY_SPANS,
        MOTION_CORRECTION_PROVENANCE,
        _provenance_values_from_frame,
    )

    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        intervals = nwbfile.intervals or {}
        tables = {
            name: intervals[name].to_dataframe()
            for name in SPAN_INTERVAL_NAMES
            if name in intervals
        }
        scratch = {
            name: nwbfile.scratch[name].to_dataframe()
            for name in (
                MOTION_CORRECTION_PROVENANCE,
                MOTION_CONTINUITY_SPANS,
                CONCAT_PROVENANCE,
                CONCAT_MEMBERS,
            )
            if name in nwbfile.scratch
        }
        n_samples = int(
            nwbfile.acquisition[
                _ELECTRICAL_SERIES_PATH.rsplit("/", 1)[-1]
            ].data.shape[0]
        )

    statistics = continuity = start_s = end_s = estimation = None
    member_ends = member_valid = observed = None
    if STATISTICS_SPANS_INTERVALS in tables:
        statistics = _spans(tables[STATISTICS_SPANS_INTERVALS])
    elif MOTION_CORRECTION_PROVENANCE in scratch:
        header = _provenance_values_from_frame(
            scratch[MOTION_CORRECTION_PROVENANCE]
        )
        if "statistics_spans" in header:
            statistics = np.asarray(
                header["statistics_spans"], dtype=np.int64
            ).reshape(-1, 2)
    if CONTINUITY_SPANS_INTERVALS in tables:
        frame = tables[CONTINUITY_SPANS_INTERVALS]
        continuity = _spans(frame)
        start_s = frame["acquisition_start_time"].to_numpy(np.float64)
        end_s = frame["acquisition_stop_time"].to_numpy(np.float64)
        if "estimation_start_time" in frame:
            estimation = frame["estimation_start_time"].to_numpy(np.float64)
    if MOTION_CONTINUITY_SPANS in scratch:
        frame = scratch[MOTION_CONTINUITY_SPANS].sort_values("span_index")
        if continuity is None:
            continuity = _spans(frame)
            start_s = frame["source_start_s"].to_numpy(np.float64)
            end_s = frame["source_end_s"].to_numpy(np.float64)
        if estimation is None:
            estimation = frame["estimation_start_s"].to_numpy(np.float64)
    if CONCAT_MEMBER_INTERVALS in tables:
        frame = tables[CONCAT_MEMBER_INTERVALS]
        member_ends = frame["end_sample"].to_numpy(np.int64)
        if "valid_times" in frame:
            member_valid = [
                np.asarray(v, dtype=np.float64).reshape(-1, 2)
                for v in frame["valid_times"]
            ]
    elif CONCAT_MEMBERS in scratch:
        frame = scratch[CONCAT_MEMBERS].sort_values("member_index")
        member_ends = frame["concat_end_sample"].to_numpy(np.int64)
    if CONCAT_PROVENANCE in scratch:
        if statistics is not None:
            observed = _merge_touching(statistics)
        else:
            header = _provenance_values_from_frame(scratch[CONCAT_PROVENANCE])
            observed = np.asarray(
                complement_frame_ranges(
                    header["artifact_frame_ranges"], n_samples
                ),
                dtype=np.int64,
            ).reshape(-1, 2)
    return TraceFileSpans(
        statistics_spans=statistics,
        continuity_spans=continuity,
        continuity_start_s=start_s,
        continuity_end_s=end_s,
        estimation_start_s=estimation,
        member_end_samples=member_ends,
        member_valid_times=member_valid,
        observed_spans=observed,
    )


def span_interval_mismatches(
    abs_path: str,
    *,
    statistics_spans,
    continuity_spans,
    continuity_start_s=None,
    continuity_end_s=None,
    member_end_samples=None,
    member_valid_times=None,
) -> list[str]:
    """Name the span facts in a file that differ from the row's columns.

    Reads the file with :func:`read_trace_file_spans`. A fact the file does
    not hold, or the caller does not pass, is not compared, so a file of any
    generation passes when what it holds matches.

    Parameters
    ----------
    abs_path : str
        Absolute path of the trace NWB file.
    statistics_spans, continuity_spans : array-like
        The row's ``(n, 2)`` span columns.
    continuity_start_s, continuity_end_s : array-like, optional
        The row's ``(n,)`` acquisition times of each continuity span.
    member_end_samples : sequence of int, optional
        Each member's ``MemberBoundary.end_sample``, in member order.
    member_valid_times : sequence of array-like, optional
        Each member's ``MemberBoundary.member_valid_times``, in member order.

    Returns
    -------
    list of str
        The name of each differing fact (a ``TraceFileSpans`` field).
    """
    stored = read_trace_file_spans(abs_path)
    expected = {
        "statistics_spans": np.asarray(statistics_spans).reshape(-1, 2),
        "continuity_spans": np.asarray(continuity_spans).reshape(-1, 2),
        "continuity_start_s": continuity_start_s,
        "continuity_end_s": continuity_end_s,
        "member_end_samples": member_end_samples,
    }
    mismatched = []
    for name, values in expected.items():
        held = getattr(stored, name)
        if values is None or held is None:
            continue
        values = np.asarray(values, dtype=held.dtype).reshape(
            -1, *held.shape[1:]
        )
        if not np.array_equal(held, values):
            mismatched.append(name)
    if member_valid_times is not None and stored.member_valid_times is not None:
        rows = [
            np.asarray(v, dtype=np.float64).reshape(-1, 2)
            for v in member_valid_times
        ]
        if len(rows) != len(stored.member_valid_times) or not all(
            np.array_equal(a, b)
            for a, b in zip(rows, stored.member_valid_times, strict=True)
        ):
            mismatched.append("member_valid_times")
    return mismatched
