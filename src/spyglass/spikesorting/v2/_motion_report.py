"""Summary and figure of one saved motion estimate, from its stored arrays.

``motion_report_summary`` condenses a ``MotionEstimate`` row (and optionally
the ``MotionCorrectedRecording`` made from it) into a dict: displacement size,
time range, kept peaks per continuity span with the spans that kept none,
the masked fraction, every gap / member join with its capped length, and the
contacts the displacement moves past the probe's ends. ``plot_motion_report``
draws the same inputs as one matplotlib figure. ``MotionEstimate.report``
fetches the inputs; nothing here reads the database or a file.

A continuity span that kept no peak is corrected from the estimator's
temporal prior alone; both the summary and the figure name such spans.

Times are on the source's own acquisition clock (s). Inside continuity span
``i`` a frame ``s`` is placed at ``t_i + (s - a_i) / fs * r_i``, the affine
map :func:`._motion.displacement_on_source_clock` uses for the temporal bins
(``r_i`` scales the span's nominal duration onto its real extent); the
timestamps' own jitter inside a span is not shown.

DB-FREE AT IMPORT. matplotlib is imported inside the plotting function.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from spyglass.spikesorting.v2._motion import (
    EstimationClock,
    displacement_on_source_clock,
    motion_max_abs_displacement_um,
    spans_without_evidence,
)
from spyglass.spikesorting.v2._sorting_artifact_mask import (
    complement_frame_ranges,
)

#: Okabe-Ito colors (colorblind-safe) used by :func:`plot_motion_report`.
_COLORS = {
    "line": "#0072B2",
    "original": "#000000",
    "corrected": "#0072B2",
    "masked": "#999999",
    "gap": "#56B4E9",
    "join": "#CC79A7",
    "no_evidence": "#D55E00",
    "evidence": "#009E73",
    "extrapolated": "#E69F00",
    "removed": "#D55E00",
    "contact": "#999999",
}

#: Diverging colormap for nonrigid displacement (ColorBrewer, colorblind-safe).
_DISPLACEMENT_CMAP = "RdBu_r"


class MotionReportInputs(NamedTuple):
    """The stored arrays of one estimate that the report reads.

    Attributes
    ----------
    motion : spikeinterface.core.motion.Motion
        The single-segment estimate; its temporal bins are on ``clock``.
    clock : _motion.EstimationClock
        The estimate's time map.
    statistics_spans : numpy.ndarray
        ``(n, 2)`` int64 half-open frame ranges of the valid samples used;
        every other frame was masked.
    peaks_per_continuity_span : numpy.ndarray
        ``(n_spans,)`` int64 kept peaks per continuity span.
    channel_ids : list
        Estimation channel ids in recording order.
    channel_locations : numpy.ndarray
        ``(n_channels, 2)`` float64 contact positions (um).
    max_gap_s : float
        The recipe's cap on each gap between continuity spans (s).
    member_join_frames : tuple of int
        Frames where one concatenation member ends and the next begins;
        empty for a single recording. A continuity join at one of these
        frames is a member join, any other is an acquisition gap.
    border_mode : str or None
        The corrected recording's interpolation ``border_mode``; ``None``
        when no corrected recording is reported.
    removed_channel_ids : list or None
        The corrected recording's ``removed_channel_ids``; ``None`` when no
        corrected recording is reported.
    """

    motion: object
    clock: EstimationClock
    statistics_spans: np.ndarray
    peaks_per_continuity_span: np.ndarray
    channel_ids: list
    channel_locations: np.ndarray
    max_gap_s: float
    member_join_frames: tuple = ()
    border_mode: str | None = None
    removed_channel_ids: list | None = None


class TraceWindow(NamedTuple):
    """Original and corrected traces of a few channels over one window.

    Attributes
    ----------
    source_time_s : numpy.ndarray
        ``(n_frames,)`` source-clock time of each frame (s).
    channel_ids : list
        The ``n_channels`` channels shown, in display order.
    original_uv : numpy.ndarray
        ``(n_frames, n_channels)`` source traces (uV).
    corrected_uv : numpy.ndarray
        ``(n_frames, n_channels)`` corrected traces (uV).
    """

    source_time_s: np.ndarray
    channel_ids: list
    original_uv: np.ndarray
    corrected_uv: np.ndarray


def _span_scales(clock: EstimationClock) -> np.ndarray:
    """``(n_spans,)`` real extent over nominal duration of each span."""
    fs = clock.sampling_frequency
    nominal = (clock.spans[:, 1] - clock.spans[:, 0]) / fs
    real = clock.source_end_s + 1.0 / fs - clock.source_start_s
    return real / nominal


def _span_of_frames(clock: EstimationClock, frames) -> np.ndarray:
    """Continuity span holding each frame (clipped to the first / last)."""
    return np.clip(
        np.searchsorted(clock.spans[:, 0], frames, side="right") - 1,
        0,
        len(clock.spans) - 1,
    )


def source_time_of_frames(clock: EstimationClock, frames) -> np.ndarray:
    """Source-clock times (s) of frames, by each span's affine map.

    Parameters
    ----------
    clock : _motion.EstimationClock
    frames : int or array_like of int
        Source frames.

    Returns
    -------
    numpy.ndarray
        ``t_i + (s - a_i) / fs * r_i`` for the span ``i`` holding each frame
        ``s``, the shape of ``frames``.
    """
    frames = np.asarray(frames)
    span = _span_of_frames(clock, frames)
    return (
        clock.source_start_s[span]
        + (frames - clock.spans[span, 0])
        / clock.sampling_frequency
        * _span_scales(clock)[span]
    )


def frame_of_source_time(clock: EstimationClock, time_s) -> np.ndarray:
    """Nearest frames to source-clock times (inverse of the affine map).

    A time in a gap between two spans maps to the end of the earlier span
    (its exclusive end frame); a time before the first span maps to frame 0.

    Parameters
    ----------
    clock : _motion.EstimationClock
    time_s : float or array_like of float
        Source-clock times (s).

    Returns
    -------
    numpy.ndarray
        int64 frames in ``[0, n_samples]``, the shape of ``time_s``.
    """
    time_s = np.asarray(time_s, dtype=np.float64)
    span = np.clip(
        np.searchsorted(clock.source_start_s, time_s, side="right") - 1,
        0,
        len(clock.spans) - 1,
    )
    offset = np.round(
        (time_s - clock.source_start_s[span])
        * clock.sampling_frequency
        / _span_scales(clock)[span]
    )
    return np.clip(
        clock.spans[span, 0] + offset,
        clock.spans[span, 0],
        clock.spans[span, 1],
    ).astype(np.int64)


def _moved_positions(motion, channel_locations):
    """Each contact's position along the motion axis in every temporal bin.

    Returns
    -------
    positions : numpy.ndarray
        ``(n_channels,)`` unmoved positions along ``motion.direction`` (um).
    moved : numpy.ndarray
        ``(n_channels, n_temporal_bins)`` positions plus the displacement at
        each temporal bin (um).
    """
    positions = np.asarray(channel_locations, dtype=np.float64)[:, motion.dim]
    displacement = motion.get_displacement_at_time_and_depth(
        times_s=motion.temporal_bins_s[0], locations_um=positions, grid=True
    )
    return positions, positions[:, None] + displacement


def border_channel_ids(motion, channel_ids, channel_locations) -> list:
    """Contacts the displacement moves past the probe's ends in some bin.

    SpikeInterface's ``interpolate_motion`` criterion
    (``sortingcomponents/motion/motion_interpolation.py:364-383``, SI
    0.104.3): a contact is inside when its position plus the displacement at
    every temporal bin stays within the extent of the contact positions
    along the motion axis. ``border_mode="remove_channels"`` drops the
    contacts that are not; ``"force_extrapolate"`` keeps them and
    extrapolates their traces.

    Parameters
    ----------
    motion : spikeinterface.core.motion.Motion
        Single-segment estimate.
    channel_ids : list
        ``n_channels`` channel ids in recording order.
    channel_locations : numpy.ndarray
        ``(n_channels, 2)`` contact positions (um).

    Returns
    -------
    list
        The border channel ids, in recording order.
    """
    positions, moved = _moved_positions(motion, channel_locations)
    low, high = positions.min(), positions.max()
    inside = (moved.clip(low, high) == moved).all(axis=1)
    return [c for c, keep in zip(channel_ids, inside) if not keep]


def _masked_intervals(inputs: MotionReportInputs) -> list[dict]:
    """Masked frame ranges split at continuity joins, with source times."""
    clock = inputs.clock
    n_samples = int(clock.spans[-1, 1])
    masked = complement_frame_ranges(
        [tuple(s) for s in np.asarray(inputs.statistics_spans).reshape(-1, 2)],
        n_samples,
    )
    intervals = []
    for start, end in masked:
        for a, b in clock.spans:
            lo, hi = max(start, int(a)), min(end, int(b))
            if lo < hi:
                first, last = source_time_of_frames(clock, [lo, hi - 1])
                intervals.append(
                    {
                        "start_frame": lo,
                        "end_frame": hi,
                        "source_start_s": float(first),
                        "source_end_s": float(last),
                    }
                )
    return intervals


def _gaps(inputs: MotionReportInputs) -> list[dict]:
    """Every join between consecutive continuity spans.

    The real gap is ``g_i = t_{i+1} - (u_i + 1 / fs)`` clamped at 0, as
    :func:`._motion.build_estimation_clock` measures it; its length on the
    estimation clock is read from the stored clock, ``e_{i+1} - e_i - (b_i -
    a_i) / fs``, so it is the gap the estimate was actually computed with.
    The clock shortened ``g_i`` to ``max_gap_s`` when it exceeds that cap
    (``capped``).
    """
    clock = inputs.clock
    fs = clock.sampling_frequency
    nominal = (clock.spans[:, 1] - clock.spans[:, 0]) / fs
    on_clock = np.diff(clock.estimation_start_s) - nominal[:-1]
    joins = {int(f) for f in inputs.member_join_frames}
    gaps = []
    for i in range(len(clock.spans) - 1):
        source_gap = max(
            float(clock.source_start_s[i + 1])
            - (float(clock.source_end_s[i]) + 1.0 / fs),
            0.0,
        )
        frame = int(clock.spans[i, 1])
        gaps.append(
            {
                "after_span": i,
                "frame": frame,
                "kind": (
                    "member_join" if frame in joins else "acquisition_gap"
                ),
                "source_gap_s": source_gap,
                "estimation_gap_s": float(on_clock[i]),
                "capped": source_gap > float(inputs.max_gap_s),
            }
        )
    return gaps


def motion_report_summary(inputs: MotionReportInputs) -> dict:
    """Summarize one saved motion estimate.

    Parameters
    ----------
    inputs : MotionReportInputs

    Returns
    -------
    dict
        - ``direction``, ``rigid`` (one spatial window), ``n_temporal_bins``,
          ``n_spatial_bins``.
        - ``max_abs_displacement_um`` and ``rms_displacement_um`` over every
          temporal bin (gap bins included) and spatial window; a non-finite
          displacement propagates.
        - ``source_start_s`` / ``source_end_s``: first and last timestamp;
          ``n_samples``, ``sampling_frequency``.
        - ``continuity_spans``: one ``{"span_index", "start_frame",
          "end_frame", "source_start_s", "source_end_s", "n_peaks_kept"}``
          per span; ``spans_without_evidence``: the spans that kept no peak
          (:func:`._motion.spans_without_evidence`), corrected from the
          temporal prior alone.
        - ``masked_fraction`` of frames outside the statistics spans and
          ``masked_intervals`` (``{"start_frame", "end_frame",
          "source_start_s", "source_end_s"}``, first and last masked frame's
          time, split at continuity joins).
        - ``gaps``: one ``{"after_span", "frame", "kind", "source_gap_s",
          "estimation_gap_s", "capped"}`` per continuity join (``kind``
          ``"member_join"`` or ``"acquisition_gap"``); ``n_acquisition_gaps``,
          ``n_member_joins``, ``max_gap_s`` and ``capped_gap_s`` (the real
          lengths of the gaps shortened to ``max_gap_s``).
        - ``border_channel_ids`` (:func:`border_channel_ids`), and for a
          corrected recording ``border_mode``, ``removed_channel_ids`` and
          ``extrapolated_channel_ids`` (the border channels under
          ``force_extrapolate``, none under ``remove_channels``); these three
          are ``None`` without one.
    """
    motion, clock = inputs.motion, inputs.clock
    displacement = np.asarray(motion.displacement[0], dtype=np.float64)
    peaks = np.asarray(inputs.peaks_per_continuity_span, dtype=np.int64)
    if len(peaks) != len(clock.spans):
        raise ValueError(
            f"Motion report: {len(peaks)} evidence counts for "
            f"{len(clock.spans)} continuity spans."
        )
    n_samples = int(clock.spans[-1, 1])
    statistics = np.asarray(inputs.statistics_spans, dtype=np.int64).reshape(
        -1, 2
    )
    gaps = _gaps(inputs)
    border = border_channel_ids(
        motion, list(inputs.channel_ids), inputs.channel_locations
    )
    extrapolated = None
    if inputs.border_mode is not None:
        extrapolated = (
            border if inputs.border_mode == "force_extrapolate" else []
        )
    return {
        "direction": str(motion.direction),
        "rigid": displacement.shape[1] == 1,
        "n_temporal_bins": int(displacement.shape[0]),
        "n_spatial_bins": int(displacement.shape[1]),
        "max_abs_displacement_um": motion_max_abs_displacement_um(motion),
        "rms_displacement_um": float(np.sqrt(np.mean(displacement**2))),
        "source_start_s": float(clock.source_start_s[0]),
        "source_end_s": float(clock.source_end_s[-1]),
        "n_samples": n_samples,
        "sampling_frequency": float(clock.sampling_frequency),
        "continuity_spans": [
            {
                "span_index": i,
                "start_frame": int(clock.spans[i, 0]),
                "end_frame": int(clock.spans[i, 1]),
                "source_start_s": float(clock.source_start_s[i]),
                "source_end_s": float(clock.source_end_s[i]),
                "n_peaks_kept": int(peaks[i]),
            }
            for i in range(len(clock.spans))
        ],
        "spans_without_evidence": spans_without_evidence(peaks, clock),
        "masked_fraction": 1.0
        - float(np.sum(statistics[:, 1] - statistics[:, 0])) / n_samples,
        "masked_intervals": _masked_intervals(inputs),
        "gaps": gaps,
        "n_acquisition_gaps": sum(g["kind"] == "acquisition_gap" for g in gaps),
        "n_member_joins": sum(g["kind"] == "member_join" for g in gaps),
        "max_gap_s": float(inputs.max_gap_s),
        "capped_gap_s": [g["source_gap_s"] for g in gaps if g["capped"]],
        "border_channel_ids": border,
        "border_mode": inputs.border_mode,
        "removed_channel_ids": (
            None
            if inputs.removed_channel_ids is None
            else list(inputs.removed_channel_ids)
        ),
        "extrapolated_channel_ids": extrapolated,
    }


def default_trace_channels(
    channel_ids, channel_locations, depth_dim: int, n_channels: int = 4
) -> list:
    """The ``n_channels`` contacts nearest the middle of the probe's depth.

    Parameters
    ----------
    channel_ids : list
        Candidate channel ids.
    channel_locations : numpy.ndarray
        ``(len(channel_ids), 2)`` contact positions (um).
    depth_dim : int
        Column of ``channel_locations`` along the motion axis.
    n_channels : int, optional
        How many to return. Defaults to 4.

    Returns
    -------
    list
        Channel ids ordered by depth.
    """
    depth = np.asarray(channel_locations, dtype=np.float64)[:, depth_dim]
    middle = (depth.min() + depth.max()) / 2.0
    nearest = np.argsort(np.abs(depth - middle), kind="stable")[:n_channels]
    return [channel_ids[i] for i in nearest[np.argsort(depth[nearest])]]


def _labeled(label: str, used: set) -> str:
    """``label`` the first time, then matplotlib's no-legend label."""
    if label in used:
        return "_nolegend_"
    used.add(label)
    return label


def _draw_displacement(ax, fig, inputs: MotionReportInputs, t0: float) -> None:
    """Panel (a): displacement over source time (heatmap or one line)."""
    clock = inputs.clock
    on_source = displacement_on_source_clock(inputs.motion, clock)
    fs = clock.sampling_frequency
    scales = _span_scales(clock)
    bins = np.asarray(inputs.motion.temporal_bins_s[0], dtype=np.float64)
    bin_s = float(np.median(np.diff(bins))) if len(bins) > 1 else 1.0
    rigid = on_source.displacement_um.shape[1] == 1
    finite = on_source.displacement_um[np.isfinite(on_source.displacement_um)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 0.0
    vmax = vmax if vmax > 0 else 1.0
    mesh = None
    for i in range(len(clock.spans)):
        in_span = np.flatnonzero(on_source.continuity_span == i)
        if not len(in_span):
            continue
        times = on_source.source_time_s[in_span] - t0
        values = on_source.displacement_um[in_span]
        if rigid:
            ax.plot(times, values[:, 0], color=_COLORS["line"], lw=1.2)
            continue
        span_start = clock.source_start_s[i] - t0
        span_end = clock.source_end_s[i] + 1.0 / fs - t0
        half = bin_s * scales[i] / 2.0
        edges = np.concatenate(
            [
                [times[0] - half],
                (times[1:] + times[:-1]) / 2.0,
                [times[-1] + half],
            ]
        ).clip(span_start, span_end)
        depth = on_source.spatial_bins_um
        depth_edges = (
            np.concatenate(
                [
                    [depth[0] - (depth[1] - depth[0]) / 2.0],
                    (depth[1:] + depth[:-1]) / 2.0,
                    [depth[-1] + (depth[-1] - depth[-2]) / 2.0],
                ]
            )
            if len(depth) > 1
            else np.array([depth[0] - 0.5, depth[0] + 0.5])
        )
        mesh = ax.pcolormesh(
            edges,
            depth_edges,
            values.T,
            cmap=_DISPLACEMENT_CMAP,
            vmin=-vmax,
            vmax=vmax,
            shading="flat",
        )
    if rigid:
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_ylabel("displacement (µm)")
        ax.set_title("(a) rigid displacement over acquisition time")
    else:
        ax.set_ylabel(f"depth along {inputs.motion.direction} (µm)")
        ax.set_title("(a) nonrigid displacement over acquisition time")
        if mesh is not None:
            fig.colorbar(mesh, ax=ax, label="displacement (µm)", pad=0.01)


def _draw_timeline(ax, inputs: MotionReportInputs, summary, t0: float) -> None:
    """Panel (b): masked intervals, gaps / member joins, spans without
    evidence, on three bands of one source-time strip."""
    clock = inputs.clock
    fs = clock.sampling_frequency
    used: set = set()
    for interval in summary["masked_intervals"]:
        ax.axvspan(
            interval["source_start_s"] - t0,
            interval["source_end_s"] + 1.0 / fs - t0,
            ymin=0.0,
            ymax=1 / 3,
            color=_COLORS["masked"],
            alpha=0.6,
            lw=0,
            label=_labeled("masked (not evidence)", used),
        )
    for gap in summary["gaps"]:
        i = gap["after_span"]
        start = clock.source_end_s[i] + 1.0 / fs - t0
        end = clock.source_start_s[i + 1] - t0
        is_join = gap["kind"] == "member_join"
        color = _COLORS["join"] if is_join else _COLORS["gap"]
        label = "member join" if is_join else "acquisition gap"
        ax.axvspan(
            start,
            max(end, start),
            ymin=1 / 3,
            ymax=2 / 3,
            facecolor=color,
            edgecolor=color,
            alpha=0.5,
            hatch="//" if gap["capped"] else None,
            label=_labeled(label, used),
        )
        ax.axvline(
            start,
            ymin=1 / 3,
            ymax=2 / 3,
            color=color,
            lw=1.5,
            label="_nolegend_",
        )
        if gap["capped"]:
            ax.text(
                (start + end) / 2.0,
                0.5,
                f"capped to {inputs.max_gap_s:g} s",
                ha="center",
                va="center",
                fontsize=7,
            )
    for span in summary["spans_without_evidence"]:
        ax.axvspan(
            span["source_start_s"] - t0,
            span["source_end_s"] + 1.0 / fs - t0,
            ymin=2 / 3,
            ymax=1.0,
            color=_COLORS["no_evidence"],
            alpha=0.7,
            lw=0,
            label=_labeled("span without evidence", used),
        )
    ax.set_ylim(0, 1)
    ax.set_yticks([1 / 6, 0.5, 5 / 6])
    ax.set_yticklabels(["masked", "gaps / joins", "no evidence"], fontsize=8)
    ax.set_xlabel("source time from first sample (s)")
    ax.set_title(
        "(b) masked intervals, gaps and member joins (hatched: capped)"
    )
    if used:
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7)


def _draw_evidence(ax, summary) -> None:
    """Panel (c): kept peaks per continuity span, zero-evidence spans
    highlighted."""
    counts = np.array(
        [s["n_peaks_kept"] for s in summary["continuity_spans"]], dtype=float
    )
    index = np.arange(len(counts))
    empty = counts == 0
    colors = [
        _COLORS["no_evidence"] if e else _COLORS["evidence"] for e in empty
    ]
    ax.bar(index, counts, color=colors, label="_nolegend_")
    if empty.any():
        ax.scatter(
            index[empty],
            np.zeros(int(empty.sum())),
            marker="x",
            s=60,
            color=_COLORS["no_evidence"],
            zorder=3,
            label="no evidence: temporal prior only",
        )
        for i in index[empty]:
            ax.annotate(
                "no evidence",
                (i, 0),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=_COLORS["no_evidence"],
            )
        ax.legend(loc="upper right", fontsize=7)
    ax.set_xticks(index)
    for tick, e in zip(ax.get_xticklabels(), empty):
        if e:
            tick.set_color(_COLORS["no_evidence"])
    ax.set_xlabel("continuity span")
    ax.set_ylabel("kept peaks")
    ax.set_title("(c) evidence per continuity span")


def _draw_border_channels(ax, inputs: MotionReportInputs, summary) -> None:
    """Panel (d): contacts on the depth axis, the range each moves over, and
    the border contacts (removed or extrapolated)."""
    positions, moved = _moved_positions(inputs.motion, inputs.channel_locations)
    # Contacts side by side in depth order, so the range each one moves over
    # stays visible on a single-column probe.
    across = np.empty(len(positions))
    across[np.argsort(positions, kind="stable")] = np.arange(len(positions))
    ax.vlines(
        across,
        np.nanmin(moved, axis=1),
        np.nanmax(moved, axis=1),
        color=_COLORS["contact"],
        lw=1.0,
        label="moved position range",
    )
    ax.scatter(
        across, positions, s=12, color=_COLORS["contact"], label="contact"
    )
    ax.axhline(positions.min(), color="0.3", lw=0.6, ls="--")
    ax.axhline(
        positions.max(), color="0.3", lw=0.6, ls="--", label="probe extent"
    )
    ids = list(inputs.channel_ids)
    removed = set(summary["removed_channel_ids"] or [])
    border = [
        ids.index(c) for c in summary["border_channel_ids"] if c not in removed
    ]
    if border:
        ax.scatter(
            across[border],
            positions[border],
            s=40,
            marker="o",
            facecolor="none",
            edgecolor=_COLORS["extrapolated"],
            lw=1.5,
            label=(
                "extrapolated"
                if summary["border_mode"] == "force_extrapolate"
                else "border (moves off probe)"
            ),
        )
    removed_index = [ids.index(c) for c in ids if c in removed]
    if removed_index:
        ax.scatter(
            across[removed_index],
            positions[removed_index],
            s=50,
            marker="x",
            color=_COLORS["removed"],
            lw=1.5,
            label="removed",
        )
    ax.set_xlabel("contact (in depth order)")
    ax.set_ylabel(f"depth along {inputs.motion.direction} (µm)")
    ax.set_title("(d) border channels")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7)


def _draw_traces(
    ax, window: TraceWindow, clock: EstimationClock, t0: float
) -> None:
    """Panel (e): original vs corrected traces, one offset row per
    channel, broken (a NaN sample) where a continuity span starts inside
    the window so no line bridges a gap."""
    original = np.asarray(window.original_uv, dtype=np.float64)
    corrected = np.asarray(window.corrected_uv, dtype=np.float64)
    both = np.concatenate([original, corrected], axis=0)
    spread = float(np.nanmax(np.ptp(both, axis=0))) if both.size else 0.0
    spacing = 1.2 * spread if spread > 0 else 1.0
    times = np.asarray(window.source_time_s, dtype=np.float64)
    breaks = np.searchsorted(times, clock.source_start_s[1:], side="left")
    breaks = breaks[(breaks > 0) & (breaks < len(times))]
    times = np.insert(times, breaks, np.nan) - t0
    original = np.insert(original, breaks, np.nan, axis=0)
    corrected = np.insert(corrected, breaks, np.nan, axis=0)
    offsets = spacing * np.arange(len(window.channel_ids))
    for k in range(len(window.channel_ids)):
        ax.plot(
            times,
            original[:, k] + offsets[k],
            color=_COLORS["original"],
            lw=0.6,
            alpha=0.6,
            label="original" if k == 0 else "_nolegend_",
        )
        ax.plot(
            times,
            corrected[:, k] + offsets[k],
            color=_COLORS["corrected"],
            lw=0.6,
            label="corrected" if k == 0 else "_nolegend_",
        )
    ax.set_yticks(offsets)
    ax.set_yticklabels([str(c) for c in window.channel_ids], fontsize=7)
    ax.set_xlabel("source time from first sample (s)")
    ax.set_ylabel(f"channel (rows {spacing:.0f} µV apart)")
    ax.set_title("(e) original vs corrected traces (µV)")
    ax.legend(loc="upper right", fontsize=7)


def plot_motion_report(
    inputs: MotionReportInputs, trace_window: TraceWindow | None = None
):
    """Draw one saved motion estimate as a multi-panel figure.

    Panels (each axes' ``get_label()`` in brackets):

    - (a) ``[displacement]``: displacement over source time -- a heatmap over
      depth for a nonrigid estimate, one line for a rigid one; each span is
      drawn on its own, so gaps stay empty.
    - (b) ``[timeline]``: masked intervals, acquisition gaps and member joins
      (capped gaps hatched and labeled), and the spans without evidence, on
      the same time axis as (a).
    - (c) ``[evidence]``: kept peaks per continuity span; a span that kept
      none is marked "no evidence" (corrected from the temporal prior only).
    - (d) ``[border_channels]``: contacts in depth order on the depth axis,
      with the range each moves over and the probe's extent; border contacts
      circled, removed ones crossed.
    - (e) ``[traces]``: original vs corrected traces, only when
      ``trace_window`` is given; broken where a continuity span starts.

    Parameters
    ----------
    inputs : MotionReportInputs
    trace_window : TraceWindow, optional
        Traces for panel (e).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    summary = motion_report_summary(inputs)
    t0 = summary["source_start_s"]
    mosaic = [
        ["displacement", "border_channels"],
        ["timeline", "border_channels"],
        ["evidence", "border_channels"],
    ]
    heights = [3.0, 1.2, 1.8]
    if trace_window is not None:
        mosaic.append(["traces", "traces"])
        heights.append(2.5)
    fig, axes = plt.subplot_mosaic(
        mosaic,
        figsize=(12, 2.2 * sum(heights) / 1.5),
        height_ratios=heights,
        width_ratios=[4.0, 1.3],
        layout="constrained",
    )
    axes["timeline"].sharex(axes["displacement"])
    for name, ax in axes.items():
        ax.set_label(name)
    _draw_displacement(axes["displacement"], fig, inputs, t0)
    axes["displacement"].tick_params(labelbottom=False)
    _draw_timeline(axes["timeline"], inputs, summary, t0)
    _draw_evidence(axes["evidence"], summary)
    _draw_border_channels(axes["border_channels"], inputs, summary)
    if trace_window is not None:
        _draw_traces(axes["traces"], trace_window, inputs.clock, t0)
    n_empty = len(summary["spans_without_evidence"])
    fig.suptitle(
        f"Motion estimate: max |displacement| "
        f"{summary['max_abs_displacement_um']:.1f} µm, RMS "
        f"{summary['rms_displacement_um']:.1f} µm\n"
        f"{n_empty} of {len(summary['continuity_spans'])} continuity span(s) "
        "without evidence"
        + (" (corrected from the temporal prior alone)" if n_empty else "")
    )
    return fig
