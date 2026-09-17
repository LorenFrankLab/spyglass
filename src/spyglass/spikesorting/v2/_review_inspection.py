"""Bounded overview raster and explicit detailed inspection views."""

from __future__ import annotations

import numpy as np


def defer_time_views(analyzer, display):
    """Defer large overview payloads without changing the requested sampling."""
    duration = analyzer.get_num_samples() / analyzer.sampling_frequency
    counts = np.asarray(
        list(analyzer.sorting.count_num_spikes_per_unit().values())
    )
    return any(
        (
            counts
            if (limit := display.point_limit(kind, duration)) is None
            else np.minimum(counts, limit)
        ).sum()
        > display.max_initial_points
        for kind in ("raster", "amplitudes")
    )


def timeline_context(timeline, time_range=None):
    """Describe the time coordinates and only the relevant original spans."""
    if timeline is None:
        return "Times are recording-relative seconds."
    basis = (
        "synthetic concatenated"
        if timeline["concatenated"]
        else "recording-relative"
    )
    lines = [
        f"Times are **{basis} seconds**. Red bands mark excluded time, not observed silence. Manual exclusions use **original session seconds**.",
        "",
        "| Session | Display seconds [start, stop) | Original seconds [start, stop) |",
        "| --- | --- | --- |",
    ]
    for name, start, stop, original_start, original_stop in timeline[
        "mappings"
    ]:
        if time_range is not None and (
            stop <= time_range[0] or start >= time_range[1]
        ):
            continue
        lines.append(
            f"| {name} | {start:g}–{stop:g} | {original_start:g}–{original_stop:g} |"
        )
    return "\n".join(lines)


def _with_exclusion_strip(view, analyzer, timeline, time_range):
    import figpack.views as fv

    if timeline is None:
        return view
    start, stop = time_range or (
        0,
        analyzer.get_num_samples() / analyzer.sampling_frequency,
    )
    strip = fv.TimeseriesGraph(
        y_range=[0, 1],
        y_label="Usable time",
        hide_y_gridlines=True,
        hide_nav_toolbar=True,
        hide_time_axis_labels=True,
    )
    strip.add_line_series(
        name="Recording", t=[start, stop], y=[0.5, 0.5], color="#238636"
    )
    excluded = timeline["excluded"]
    selected = excluded[(excluded[:, 1] > start) & (excluded[:, 0] < stop)]
    strip.add_interval_series(
        name="Excluded",
        t_start=np.maximum(selected[:, 0], start),
        t_end=np.minimum(selected[:, 1], stop),
        color="#d33",
        alpha=0.5,
    )
    for name, first, end, _, _ in timeline["mappings"]:
        if start < first < stop:
            strip.add_line_series(
                name=name, t=[first, first], y=[0, 1], color="#666"
            )
    return fv.Box(
        direction="vertical",
        items=[
            fv.LayoutItem(view, stretch=1),
            fv.LayoutItem(
                strip,
                min_size=60,
                max_size=60,
                title="Usable time: red = excluded; lines = session/gap boundaries",
            ),
        ],
    )


def raster_view(
    analyzer, *, max_spikes_per_unit, unit_ids=None, time_range=None
):
    """Build a raster on the analyzer's recording-relative timeline.

    Overview samples span the whole recording. A requested time window retains
    every spike in that window; it is useful for checking a sampled overview.
    """
    from figpack_spike_sorting.views import RasterPlot, RasterPlotItem

    sorting = analyzer.sorting
    fs = sorting.sampling_frequency
    start, stop = (
        (0.0, analyzer.get_num_samples() / fs)
        if time_range is None
        else time_range
    )
    if not 0 <= start < stop <= analyzer.get_num_samples() / fs:
        raise ValueError(
            "time_range must be within the recording, with start < stop."
        )
    ids = sorting.unit_ids if unit_ids is None else unit_ids
    plots = []
    for unit_id in ids:
        # Read a coarse frame window, then compare seconds below: multiplying
        # a sample-aligned time by fs can round just above the sample index.
        frames = sorting.get_unit_spike_train(
            unit_id,
            start_frame=int(np.floor(start * fs)),
            end_frame=int(np.ceil(stop * fs)) + 1,
        )
        if (
            time_range is None
            and max_spikes_per_unit is not None
            and len(frames) > max_spikes_per_unit
        ):
            frames = frames[
                np.linspace(0, len(frames) - 1, max_spikes_per_unit, dtype=int)
            ]
        times = frames / fs
        if time_range is not None:
            times = times[(times >= start) & (times < stop)]
        plots.append(
            RasterPlotItem(unit_id=int(unit_id), spike_times_sec=times)
        )
    return RasterPlot(start_time_sec=start, end_time_sec=stop, plots=plots)


def amplitude_view(analyzer, display, unit_ids):
    """Read only sampled amplitudes and use the same relative clock as raster."""
    from figpack_spike_sorting.views import SpikeAmplitudes, SpikeAmplitudesItem

    from spyglass.spikesorting.v2._analyzer_cache import (
        analyzer_extension_array,
    )

    amplitudes = analyzer_extension_array(
        analyzer, "spike_amplitudes", "amplitudes"
    )
    sorting = analyzer.sorting
    duration = analyzer.get_num_samples() / analyzer.sampling_frequency
    limit = display.point_limit("amplitudes", duration)
    indices = sorting.get_spike_vector_to_indices()[0]
    spikes = sorting.to_spike_vector()
    random = np.random.default_rng(display.amplitude_sampling_seed)
    plots = []
    for unit in unit_ids:
        selected = indices[unit]
        if limit is not None and len(selected) > limit:
            selected = selected[
                random.choice(len(selected), size=limit, replace=False)
            ]
        plots.append(
            SpikeAmplitudesItem(
                unit_id=int(unit),
                spike_times_sec=spikes["sample_index"][selected]
                / analyzer.sampling_frequency,
                spike_amplitudes=amplitudes[selected],
            )
        )
    return SpikeAmplitudes(start_time_sec=0, end_time_sec=duration, plots=plots)


def inspection_view(
    analyzer,
    display,
    *,
    unit_ids=None,
    time_range=None,
    timeline=None,
    deferred=False,
    displayed_unit_properties=None,
    extra_unit_properties=None,
    min_similarity_for_correlograms=None,
):
    """Compose linked views directly, without cloning the scientific analyzer.

    Extra properties are aligned to the full analyzer; only requested unit
    rows and similarity pairs are included in the selector. Deferred time
    views never invoke the amplitude widget or read raster spike trains.
    """
    import figpack.views as fv
    import spikeinterface.widgets as sw
    from figpack_spike_sorting.views import UnitSimilarityScore
    from spikeinterface.widgets.utils_figpack import generate_unit_table_view

    ids = analyzer.unit_ids if unit_ids is None else np.asarray(unit_ids)
    indices = analyzer.sorting.ids_to_indices(ids)
    similarity = analyzer.get_extension("template_similarity").get_data()
    scores = [
        UnitSimilarityScore(
            unit_id1=int(u), unit_id2=int(v), similarity=float(similarity[i, j])
        )
        for u, i in zip(ids, indices)
        for v, j in zip(ids, indices)
    ]
    properties = (
        [
            "firing_rate",
            "num_spikes",
            "x",
            "y",
            "amplitude_median",
            "snr",
            "rp_violations",
        ]
        if displayed_unit_properties is None
        else list(displayed_unit_properties)
    )
    properties = list(
        dict.fromkeys([*properties, *(extra_unit_properties or {})])
    )
    selector = generate_unit_table_view(
        analyzer,
        properties,
        similarity_scores=scores,
        extra_unit_properties=extra_unit_properties,
    )
    rows = {row.unit_id: row for row in selector.rows}
    selector.rows = [rows[unit] for unit in ids]
    widget_options = {
        "unit_ids": ids,
        "backend": "figpack",
        "hide_unit_selector": True,
        "generate_url": False,
        "display": False,
    }
    waveforms = sw.plot_unit_templates(analyzer, **widget_options).view
    geometry = sw.plot_unit_locations(analyzer, **widget_options).view
    correlograms = sw.plot_crosscorrelograms(
        analyzer,
        min_similarity_for_correlograms=min_similarity_for_correlograms,
        **widget_options,
    ).view
    amplitudes = None
    if not deferred:
        amplitudes = amplitude_view(analyzer, display, ids)

    notice = fv.Markdown(
        "**Time-based overview not loaded.** Its requested sampling budget exceeds "
        f"{display.max_initial_points:,} points. Select units and use **Inspect selected units / pairs** "
        "to load their evidence; enter a window for an exact raster. "
        "A hosted/static figure requires `review.inspect_units(...)` in Python, or "
        "an explicit smaller display cap when creating the review. This is not an empty spike train."
    )
    raster = (
        notice
        if deferred
        else raster_view(
            analyzer,
            max_spikes_per_unit=display.point_limit(
                "raster",
                analyzer.get_num_samples() / analyzer.sampling_frequency,
            ),
            unit_ids=unit_ids,
            time_range=time_range,
        )
    )
    autocorrelograms = sw.plot_autocorrelograms(analyzer, **widget_options).view
    raster_title = "Raster (overview)"
    if deferred:
        raster_title = "Raster (not loaded)"
    elif time_range is not None:
        raster_title = "Raster (all spikes in window)"
    tabs = fv.TabLayout(
        items=[
            fv.TabLayoutItem(
                waveforms,
                label="Waveforms",
            ),
            fv.TabLayoutItem(
                (
                    notice
                    if deferred
                    else _with_exclusion_strip(
                        amplitudes, analyzer, timeline, None
                    )
                ),
                label="Spike amplitudes",
            ),
            fv.TabLayoutItem(autocorrelograms, label="Autocorrelograms"),
            fv.TabLayoutItem(correlograms, label="Cross-correlograms"),
            fv.TabLayoutItem(geometry, label="Electrode geometry"),
            fv.TabLayoutItem(
                (
                    raster
                    if deferred
                    else _with_exclusion_strip(
                        raster, analyzer, timeline, time_range
                    )
                ),
                label=raster_title,
            ),
        ]
    )
    tabs.items.append(
        fv.TabLayoutItem(
            fv.Markdown(
                timeline_context(timeline, time_range)
                + "\n\n"
                + display.describe()
                + "\n\n`observed_*` metrics use usable time. Raw SI duration metrics (firing_rate, presence_ratio, contamination ratios, firing_range) retain SI's full-timeline definitions."
            ),
            label="Time and sampling",
        )
    )
    return fv.Splitter(
        direction="horizontal",
        split_pos=0.38,
        item1=fv.LayoutItem(selector),
        item2=fv.LayoutItem(tabs),
    )


def add_trace_inspection(
    analyzer, summary, unit_ids, time_range, *, timeline=None
):
    """Add a bounded spike-on-trace figure to a focused inspection bundle."""
    import io

    import figpack.views as fv
    import matplotlib.pyplot as plt
    import spikeinterface.widgets as sw

    if time_range is None or time_range[1] - time_range[0] > 10:
        raise ValueError("Choose a trace window of at most 10 seconds.")
    # SI's trace widget uses the recording clock (and reads an entire stored
    # timestamp vector). Give this display a lightweight frame-based wrapper,
    # so relative windows select the same frames as the raster, even across
    # absolute session timestamps/gaps. The source recording stays unchanged.
    original = analyzer.recording
    original_sorting = analyzer.sorting
    relative = original.select_channels(original.channel_ids)
    relative.reset_times()
    relative_sorting = original_sorting.select_units(original_sorting.unit_ids)
    relative_sorting.register_recording(relative)
    analyzer.set_temporary_recording(relative)
    analyzer.sorting = relative_sorting
    try:
        widget = sw.plot_spikes_on_traces(
            analyzer,
            unit_ids=unit_ids,
            time_range=time_range,
            backend="matplotlib",
        )
    finally:
        analyzer.set_temporary_recording(original)
        analyzer.sorting = original_sorting
    if timeline is not None:
        for start, stop in timeline["excluded"]:
            if start < time_range[1] and stop > time_range[0]:
                widget.ax.axvspan(
                    max(start, time_range[0]),
                    min(stop, time_range[1]),
                    color="red",
                    alpha=0.2,
                )
    buffer = io.BytesIO()
    widget.figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(widget.figure)
    summary.item2.view.items.append(
        fv.TabLayoutItem(fv.Image(buffer.getvalue()), label="Spikes on traces")
    )
