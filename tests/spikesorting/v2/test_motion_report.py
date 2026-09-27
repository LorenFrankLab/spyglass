"""DB-free tests for the motion report summary and figure.

The inputs are built by hand: three continuity spans at 1 kHz with a 5 s
acquisition gap (capped to 2 s on the estimation clock) and a 0.5 s member
join, one masked interval, a middle span that kept no peak, and eight
contacts 20 um apart. Every expected value below follows from those numbers.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytestmark = pytest.mark.unit

FS = 1000.0
T0 = 100.0
MAX_GAP_S = 2.0
# Span 0: frames [0, 5000), 100.000-104.999 s. Span 1: [5000, 8000), 110.000-
# 112.999 s (real gap 5.0 s, capped to 2.0). Span 2: [8000, 14000), 113.500-
# 119.499 s (real gap 0.5 s, a member join). Estimation clock starts
# 100, 107, 110.5; it ends at 116.5, so 1 s bins give 16 centers
# 100.5 ... 115.5, two of them (105.5, 106.5) inside the capped gap.
SPANS = np.array([[0, 5000], [5000, 8000], [8000, 14000]], dtype=np.int64)
SOURCE_START_S = np.array([100.0, 110.0, 113.5])
SOURCE_END_S = np.array([104.999, 112.999, 119.499])
STATISTICS = np.array([[0, 3600], [5000, 8000], [8000, 14000]], dtype=np.int64)
PEAKS_PER_SPAN = np.array([120, 0, 45], dtype=np.int64)
# Squares sum to 1600 over 16 bins: RMS 10 um; max |d| 30 um.
RIGID = np.array(
    [0, 10, -10, 30, 0, 0, 0, 0, 10, 10, 10, -10, 10, 0, 0, 0], dtype=float
)
CHANNEL_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
LOCATIONS = np.column_stack([np.zeros(8), 20.0 * np.arange(8)])


def _clock():
    from spyglass.spikesorting.v2._motion import build_estimation_clock

    return build_estimation_clock(
        SPANS, SOURCE_START_S, SOURCE_END_S, FS, MAX_GAP_S
    )


def _motion(nonrigid: bool = False):
    from spikeinterface.core.motion import Motion

    bins = 100.5 + np.arange(16, dtype=float)
    if nonrigid:
        displacement = np.column_stack([RIGID, -RIGID / 2])
        return Motion(displacement, bins, np.array([35.0, 105.0]))
    return Motion(RIGID[:, None], bins, np.array([70.0]))


def _inputs(nonrigid: bool = False, **extra):
    from spyglass.spikesorting.v2._motion_report import MotionReportInputs

    return MotionReportInputs(
        motion=_motion(nonrigid),
        clock=_clock(),
        statistics_spans=STATISTICS,
        peaks_per_continuity_span=PEAKS_PER_SPAN,
        channel_ids=CHANNEL_IDS,
        channel_locations=LOCATIONS,
        max_gap_s=MAX_GAP_S,
        member_join_frames=(8000,),
        **extra,
    )


def _axes(fig) -> dict:
    return {ax.get_label(): ax for ax in fig.axes}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_estimation_clock_of_the_constructed_inputs():
    """Guard the hand-derived clock the other expectations rest on."""
    from spyglass.spikesorting.v2._motion import displacement_on_source_clock

    clock = _clock()
    np.testing.assert_allclose(clock.estimation_start_s, [100.0, 107.0, 110.5])
    on_source = displacement_on_source_clock(_motion(), clock)
    assert np.flatnonzero(on_source.in_gap).tolist() == [5, 6]


def test_summary_values_of_a_rigid_estimate():
    """Displacement size, time range, evidence, masking, gaps and border
    channels equal the values the constructed inputs imply."""
    from spyglass.spikesorting.v2._motion_report import motion_report_summary

    summary = motion_report_summary(_inputs())

    assert summary["rigid"] is True
    assert summary["direction"] == "y"
    assert (summary["n_temporal_bins"], summary["n_spatial_bins"]) == (16, 1)
    assert summary["max_abs_displacement_um"] == 30.0
    assert summary["rms_displacement_um"] == pytest.approx(10.0, abs=1e-12)
    assert summary["source_start_s"] == 100.0
    assert summary["source_end_s"] == 119.499
    assert summary["n_samples"] == 14000
    assert summary["sampling_frequency"] == FS

    assert [s["n_peaks_kept"] for s in summary["continuity_spans"]] == [
        120,
        0,
        45,
    ]
    assert summary["spans_without_evidence"] == [
        {
            "span_index": 1,
            "start_frame": 5000,
            "end_frame": 8000,
            "source_start_s": 110.0,
            "source_end_s": 112.999,
        }
    ]

    # Frames [3600, 5000) are masked: 1400 of 14000.
    assert summary["masked_fraction"] == pytest.approx(0.1, abs=1e-12)
    [masked] = summary["masked_intervals"]
    assert (masked["start_frame"], masked["end_frame"]) == (3600, 5000)
    assert masked["source_start_s"] == pytest.approx(103.6, abs=1e-9)
    assert masked["source_end_s"] == pytest.approx(104.999, abs=1e-9)

    gap, join = summary["gaps"]
    assert (gap["after_span"], gap["frame"], gap["kind"]) == (
        0,
        5000,
        "acquisition_gap",
    )
    assert gap["source_gap_s"] == pytest.approx(5.0, abs=1e-9)
    assert gap["estimation_gap_s"] == MAX_GAP_S
    assert gap["capped"] is True
    assert (join["after_span"], join["frame"], join["kind"]) == (
        1,
        8000,
        "member_join",
    )
    assert join["source_gap_s"] == pytest.approx(0.5, abs=1e-9)
    assert join["estimation_gap_s"] == pytest.approx(0.5, abs=1e-9)
    assert join["capped"] is False
    assert (summary["n_acquisition_gaps"], summary["n_member_joins"]) == (1, 1)
    assert summary["capped_gap_s"] == [pytest.approx(5.0, abs=1e-9)]
    assert summary["max_gap_s"] == MAX_GAP_S

    # Contacts at 0..140 um; the displacement reaches -10 and +30 um: the
    # contact at 0 leaves below, those at 120 and 140 above.
    assert summary["border_channel_ids"] == [0, 6, 7]
    assert summary["border_mode"] is None
    assert summary["removed_channel_ids"] is None
    assert summary["extrapolated_channel_ids"] is None


@pytest.mark.parametrize(
    "border_mode, removed, extrapolated",
    [
        ("remove_channels", [0, 6, 7], []),
        ("force_extrapolate", [], [0, 6, 7]),
    ],
)
def test_summary_names_removed_or_extrapolated_channels(
    border_mode, removed, extrapolated
):
    from spyglass.spikesorting.v2._motion_report import motion_report_summary

    summary = motion_report_summary(
        _inputs(border_mode=border_mode, removed_channel_ids=removed)
    )
    assert summary["border_mode"] == border_mode
    assert summary["removed_channel_ids"] == removed
    assert summary["extrapolated_channel_ids"] == extrapolated


def test_summary_of_a_nonrigid_estimate():
    from spyglass.spikesorting.v2._motion_report import motion_report_summary

    summary = motion_report_summary(_inputs(nonrigid=True))
    assert summary["rigid"] is False
    assert summary["n_spatial_bins"] == 2
    assert summary["max_abs_displacement_um"] == 30.0
    # Second window is -d/2: squares sum to 1600 + 400 over 32 values.
    assert summary["rms_displacement_um"] == pytest.approx(
        np.sqrt(2000 / 32), abs=1e-12
    )


def test_evidence_counts_must_match_the_spans():
    from spyglass.spikesorting.v2._motion_report import (
        MotionReportInputs,
        motion_report_summary,
    )

    inputs = _inputs()._replace(peaks_per_continuity_span=np.array([1, 2]))
    assert isinstance(inputs, MotionReportInputs)
    with pytest.raises(ValueError, match="2 evidence counts for 3"):
        motion_report_summary(inputs)


def test_frame_and_source_time_maps_invert_each_other():
    from spyglass.spikesorting.v2._motion_report import (
        frame_of_source_time,
        source_time_of_frames,
    )

    clock = _clock()
    frames = np.array([0, 4999, 5000, 7999, 8000, 13999])
    times = source_time_of_frames(clock, frames)
    np.testing.assert_allclose(
        times, [100.0, 104.999, 110.0, 112.999, 113.5, 119.499], atol=1e-9
    )
    np.testing.assert_array_equal(frame_of_source_time(clock, times), frames)
    # A time inside the gap maps to the end of the earlier span.
    assert int(frame_of_source_time(clock, 107.0)) == 5000


def test_time_maps_follow_a_span_whose_timestamps_run_fast():
    """Span 1's timestamps advance 1.001 s per nominal second (10 000 frames
    at 1 kHz over 10.01 s): frame 10 000, 5 000 frames into the span, is at
    110 + 5 * 1.001 s, not the nominal 115 s, and the inverse map undoes
    that scale."""
    from spyglass.spikesorting.v2._motion import build_estimation_clock
    from spyglass.spikesorting.v2._motion_report import (
        frame_of_source_time,
        source_time_of_frames,
    )

    clock = build_estimation_clock(
        [[0, 5000], [5000, 15000]],
        [100.0, 110.0],
        [104.999, 110.0 + 10.01 - 1.0 / FS],
        FS,
        MAX_GAP_S,
    )
    assert float(source_time_of_frames(clock, 10000)) == pytest.approx(
        115.005, abs=1e-9
    )
    assert int(frame_of_source_time(clock, 115.005)) == 10000
    # The nominal 115 s is 5 ms early: 4995 frames into the span.
    assert int(frame_of_source_time(clock, 115.0)) == 9995
    # The nominal-rate span 0 is unaffected.
    assert float(source_time_of_frames(clock, 2500)) == pytest.approx(
        102.5, abs=1e-9
    )


def test_default_trace_channels_are_the_middle_of_the_probe():
    from spyglass.spikesorting.v2._motion_report import default_trace_channels

    assert default_trace_channels(CHANNEL_IDS, LOCATIONS, 1) == [2, 3, 4, 5]


@pytest.mark.parametrize("nonrigid", [False, True])
def test_figure_panels_and_labels(nonrigid):
    """Panels (a)-(d) render with titles and unit labels; a rigid estimate
    draws one line per span, a nonrigid one a heatmap with a colorbar."""
    from matplotlib.collections import QuadMesh

    from spyglass.spikesorting.v2._motion_report import plot_motion_report

    fig = plot_motion_report(_inputs(nonrigid=nonrigid))
    axes = _axes(fig)
    assert {
        "displacement",
        "timeline",
        "evidence",
        "border_channels",
    } <= set(axes)
    assert "traces" not in axes
    displacement = axes["displacement"]
    meshes = [c for c in displacement.collections if isinstance(c, QuadMesh)]
    if nonrigid:
        assert "nonrigid" in displacement.get_title()
        assert displacement.get_ylabel() == "depth along y (µm)"
        # One heatmap per continuity span, so the gap stays empty.
        assert len(meshes) == 3
        assert axes["<colorbar>"].get_ylabel() == "displacement (µm)"
    else:
        assert "rigid" in displacement.get_title()
        assert displacement.get_ylabel() == "displacement (µm)"
        assert not meshes
        # One line per continuity span (plus the zero line).
        span_lines = [
            line for line in displacement.get_lines() if len(line.get_xdata())
        ]
        assert len(span_lines) == 4
    assert axes["timeline"].get_xlabel() == "source time from first sample (s)"
    assert axes["evidence"].get_xlabel() == "continuity span"
    assert axes["evidence"].get_ylabel() == "kept peaks"
    assert axes["border_channels"].get_ylabel() == "depth along y (µm)"
    assert axes["border_channels"].get_xlabel() == "contact (in depth order)"
    assert (
        "1 of 3 continuity span(s) without evidence" in fig._suptitle.get_text()
    )


def test_figure_flags_the_span_without_evidence():
    """The evidence panel marks span 1, its tick is highlighted, and the
    timeline shades exactly that span's source-time extent."""
    from spyglass.spikesorting.v2._motion_report import (
        _COLORS,
        plot_motion_report,
    )

    axes = _axes(plot_motion_report(_inputs()))
    evidence = axes["evidence"]
    assert [t.get_text() for t in evidence.texts] == ["no evidence"]
    assert evidence.texts[0].xy == (1, 0)
    highlighted = [
        t.get_color() == _COLORS["no_evidence"]
        for t in evidence.get_xticklabels()
    ]
    assert highlighted == [False, True, False]
    bars = [p.get_facecolor() for p in evidence.patches]
    assert matplotlib.colors.to_hex(bars[1]) == _COLORS["no_evidence"].lower()
    assert matplotlib.colors.to_hex(bars[0]) == _COLORS["evidence"].lower()

    timeline = axes["timeline"]
    shaded = [
        p for p in timeline.patches if p.get_label() == "span without evidence"
    ]
    assert len(shaded) == 1
    start = shaded[0].get_x()
    # Span 1 runs from 10.0 s to one sample after 12.999 s, from the first
    # sample at 100 s.
    assert (start, start + shaded[0].get_width()) == (
        pytest.approx(10.0, abs=1e-9),
        pytest.approx(13.0, abs=1e-9),
    )
    labels = {p.get_label() for p in timeline.patches}
    assert {"masked (not evidence)", "acquisition gap", "member join"} <= labels
    hatched = [p for p in timeline.patches if p.get_hatch()]
    assert len(hatched) == 1
    assert [t.get_text() for t in timeline.texts] == ["capped to 2 s"]


def test_figure_marks_removed_and_extrapolated_channels():
    from spyglass.spikesorting.v2._motion_report import plot_motion_report

    removed = _axes(
        plot_motion_report(
            _inputs(
                border_mode="remove_channels", removed_channel_ids=[0, 6, 7]
            )
        )
    )["border_channels"]
    labels = [c.get_label() for c in removed.collections]
    assert "removed" in labels and "extrapolated" not in labels
    [crosses] = [c for c in removed.collections if c.get_label() == "removed"]
    np.testing.assert_array_equal(
        crosses.get_offsets()[:, 1], [0.0, 120.0, 140.0]
    )

    extrapolated = _axes(
        plot_motion_report(
            _inputs(border_mode="force_extrapolate", removed_channel_ids=[])
        )
    )["border_channels"]
    [circles] = [
        c for c in extrapolated.collections if c.get_label() == "extrapolated"
    ]
    np.testing.assert_array_equal(
        circles.get_offsets()[:, 1], [0.0, 120.0, 140.0]
    )


def test_figure_draws_trace_windows_when_given():
    """Panel (e) overlays original and corrected traces per channel, labeled
    in microvolts."""
    from spyglass.spikesorting.v2._motion_report import (
        TraceWindow,
        plot_motion_report,
    )

    time_s = 110.0 + np.arange(100) / FS
    original = np.tile(np.sin(np.arange(100) / 5.0)[:, None], (1, 2)) * 50.0
    window = TraceWindow(
        source_time_s=time_s,
        channel_ids=[3, 4],
        original_uv=original,
        corrected_uv=original * 0.5,
    )
    axes = _axes(plot_motion_report(_inputs(), trace_window=window))
    traces = axes["traces"]
    assert "µV" in traces.get_title() and "µV" in traces.get_ylabel()
    assert [t.get_text() for t in traces.get_yticklabels()] == ["3", "4"]
    assert len(traces.get_lines()) == 4
    np.testing.assert_allclose(
        traces.get_lines()[0].get_xdata(), time_s - 100.0
    )
    assert {line.get_label() for line in traces.get_lines()} >= {
        "original",
        "corrected",
    }
