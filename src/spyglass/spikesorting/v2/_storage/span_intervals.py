"""Span tables in the concatenated and motion-corrected trace NWB files.

Standard ``TimeIntervals`` in ``nwbfile.intervals``, so a reader with plain
pynwb can tell which samples are zero because of a mask and where sessions
were joined. They copy the row's span columns, which the code reads; the
content fingerprint hashes only the ``ElectricalSeries`` and ignores them.
"""

from __future__ import annotations

import numpy as np
import pynwb
from pynwb.epoch import TimeIntervals

from spyglass.spikesorting.v2._core.signal_math import _segment_times_at
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


def span_interval_tables(
    clock_recording,
    *,
    statistics_spans,
    continuity_spans,
    continuity_start_s,
    continuity_end_s,
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
    concat_members : list of dict, optional
        One ``{"member_index", "nwb_file_name", "interval_list_name",
        "start_sample", "end_sample"}`` per member of a concatenation, in
        order. Adds the member table.
    is_concat : bool
        The series' timestamps are a concatenation's synthetic timeline.

    Returns
    -------
    list of pynwb.epoch.TimeIntervals
    """
    clock_text = _CONCAT_CLOCK_TEXT if is_concat else _SINGLE_CLOCK_TEXT
    tables = [
        TimeIntervals(
            name=STATISTICS_SPANS_INTERVALS,
            description=(
                "Each row is a run of samples that is free of artifacts and "
                "lies inside one continuity span. Samples outside every row "
                "were set to zero by the artifact mask. "
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
            ),
            columns=_frame_columns(clock_recording, continuity_spans)
            + [
                concrete_vectordata(
                    "acquisition_start_time",
                    continuity_start_s,
                    np.float64,
                    description=(
                        "time of the row's first sample on its acquisition "
                        "clock, in seconds"
                    ),
                ),
                concrete_vectordata(
                    "acquisition_stop_time",
                    continuity_end_s,
                    np.float64,
                    description=(
                        "time of the row's last sample on its acquisition "
                        "clock, in seconds"
                    ),
                ),
            ],
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
                ],
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


def span_interval_mismatches(
    abs_path: str,
    *,
    statistics_spans,
    continuity_spans,
    continuity_start_s=None,
    continuity_end_s=None,
    member_end_samples=None,
) -> list[str]:
    """Name the span tables in a file that differ from the row's columns.

    A table the file lacks is not compared, so a file written before the
    tables existed passes.

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

    Returns
    -------
    list of str
        ``"<table>.<column>"`` for each differing column.
    """
    expected = {
        STATISTICS_SPANS_INTERVALS: {
            "spans": np.asarray(statistics_spans, dtype=np.int64)
        },
        CONTINUITY_SPANS_INTERVALS: {
            "spans": np.asarray(continuity_spans, dtype=np.int64),
            "acquisition_start_time": continuity_start_s,
            "acquisition_stop_time": continuity_end_s,
        },
    }
    if member_end_samples is not None:
        ends = np.asarray(member_end_samples, dtype=np.int64)
        expected[CONCAT_MEMBER_INTERVALS] = {
            "spans": np.column_stack([np.r_[0, ends[:-1]], ends])
        }
    mismatched = []
    for name, frame in read_span_intervals(abs_path).items():
        for column, values in expected.get(name, {}).items():
            if values is None:
                continue
            if column == "spans":
                stored = frame[["start_sample", "end_sample"]].to_numpy()
                values = np.asarray(values).reshape(-1, 2)
            else:
                stored = frame[column].to_numpy()
                values = np.asarray(values).reshape(-1)
            if not np.array_equal(stored, values):
                mismatched.append(f"{name}.{column}")
    return mismatched
