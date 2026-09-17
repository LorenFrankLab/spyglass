"""Sampling is explicit; windowed rasters retain exact spikes."""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("figpack") is None,
    reason="requires the spikesorting-v2-curation extra (figpack)",
)


def test_raster_budget_spans_full_recording_and_window_keeps_every_spike():
    import spikeinterface.core as si

    from spyglass.spikesorting.v2._review_inspection import raster_view

    recording = si.NumpyRecording(np.zeros((10000, 2)), 1000)
    recording.set_dummy_probe_from_locations(np.array([[0, 0], [0, 20]]))
    sorting = si.NumpySorting.from_unit_dict({1: np.arange(0, 10000, 10)}, 1000)
    analyzer = si.create_sorting_analyzer(sorting, recording, sparse=False)
    sampled = raster_view(analyzer, max_spikes_per_unit=10)
    assert len(sampled.plots[0].spike_times_sec) == 10
    assert sampled.plots[0].spike_times_sec[0] == 0
    assert sampled.plots[0].spike_times_sec[-1] == 9.99
    window = raster_view(analyzer, max_spikes_per_unit=10, time_range=(5, 6))
    np.testing.assert_allclose(
        window.plots[0].spike_times_sec, np.arange(500, 600) / 100
    )


@pytest.mark.parametrize(
    "sampling_frequency, frames",
    [(20000, [700, 1000, 1400]), (30000, [510, 750, 1020])],
)
def test_raster_window_includes_start_and_excludes_stop(
    sampling_frequency, frames
):
    import spikeinterface.core as si

    from spyglass.spikesorting.v2._review_inspection import raster_view

    recording = si.NumpyRecording(np.zeros((2000, 2)), sampling_frequency)
    recording.set_dummy_probe_from_locations(np.array([[0, 0], [0, 20]]))
    frames = np.asarray(frames)
    sorting = si.NumpySorting.from_unit_dict({1: frames}, sampling_frequency)
    analyzer = si.create_sorting_analyzer(sorting, recording, sparse=False)
    times = frames / sampling_frequency
    window = raster_view(
        analyzer,
        max_spikes_per_unit=1,
        time_range=(times[0], times[-1]),
    )
    np.testing.assert_allclose(window.plots[0].spike_times_sec, times[:-1])


def test_hour_budget_defers_large_overviews_without_truncating_focused_units():
    import spikeinterface.core as si

    from spyglass.spikesorting.v2._review_inspection import (
        defer_time_views,
        raster_view,
    )
    from spyglass.spikesorting.v2._review_profile import ReviewDisplayOptions

    duration, fs = 3600, 1000
    recording = si.NumpyRecording(
        np.zeros((duration * fs, 1), dtype=np.int16), fs
    )
    recording.set_dummy_probe_from_locations(np.array([[0, 0]]))
    # Eight 100 Hz trains exceed the initial point budget. Focused loading
    # still grants each requested unit its v1-compatible 180,000 points.
    trains = {i: np.arange(i, duration * fs, 10) for i in range(8)}
    sorting = si.NumpySorting.from_unit_dict(trains, fs)
    analyzer = si.create_sorting_analyzer(sorting, recording, sparse=False)
    display = ReviewDisplayOptions()
    assert defer_time_views(analyzer, display)
    budget = display.point_limit("raster", duration)
    first = raster_view(analyzer, max_spikes_per_unit=budget, unit_ids=[0])
    again = raster_view(analyzer, max_spikes_per_unit=budget, unit_ids=[0])
    assert len(first.plots[0].spike_times_sec) == 180000
    assert first.plots[0].spike_times_sec[-1] == pytest.approx(3599.99)
    np.testing.assert_array_equal(
        first.plots[0].spike_times_sec, again.plots[0].spike_times_sec
    )
    small = display.from_mapping(
        {"max_raster_spikes_per_unit": 100, "max_amplitudes_per_unit": 100}
    )
    assert not defer_time_views(analyzer, small)


def test_trace_window_uses_relative_frames_without_reading_full_timestamps(
    monkeypatch,
):
    from types import SimpleNamespace

    import spikeinterface.core as si
    import spikeinterface.widgets as sw

    from spyglass.spikesorting.v2._review_inspection import (
        add_trace_inspection,
    )

    recording, sorting = si.generate_ground_truth_recording(
        durations=[2.0], num_channels=4, num_units=1, seed=6
    )
    fs = recording.sampling_frequency
    times = 100 + np.arange(recording.get_num_samples()) / fs
    times[times >= 101] += 10  # a disjoint session timeline
    recording.set_times(times)
    analyzer = si.create_sorting_analyzer(sorting, recording, sparse=False)
    analyzer.compute(
        ["random_spikes", "waveforms", "templates", "unit_locations"]
    )
    window = (1.1, 1.2)
    original_plot = sw.plot_spikes_on_traces

    def plot(*args, **kwargs):
        widget = original_plot(*args, **kwargs)
        np.testing.assert_allclose(widget.ax.get_xlim(), window)
        return widget

    def no_full_times(*args, **kwargs):
        raise AssertionError(
            "A bounded trace request must not load the full time vector"
        )

    monkeypatch.setattr(recording, "get_times", no_full_times)
    monkeypatch.setattr(sw, "plot_spikes_on_traces", plot)
    summary = SimpleNamespace(
        item2=SimpleNamespace(view=SimpleNamespace(items=[]))
    )
    original_sorting = analyzer.sorting
    add_trace_inspection(analyzer, summary, analyzer.unit_ids, window)
    # The display wrapper must leave the scientific recording clock intact.
    assert analyzer.recording is recording
    assert analyzer.sorting is original_sorting
    assert recording.get_start_time() == 100
    assert len(summary.item2.view.items) == 1


@pytest.fixture(scope="module")
def inspection_analyzer_folder(tmp_path_factory):
    import spikeinterface.core as si

    recording, sorting = si.generate_ground_truth_recording(
        durations=[2.0],
        num_channels=4,
        num_units=3,
        seed=8,
        generate_sorting_kwargs={"firing_rates": 40},
    )
    sorting = sorting.rename_units([0, 1, 2])
    folder = tmp_path_factory.mktemp("inspection") / "analyzer"
    analyzer = si.create_sorting_analyzer(
        sorting, recording, format="binary_folder", folder=folder, sparse=False
    )
    analyzer.compute(
        [
            "random_spikes",
            "waveforms",
            "templates",
            "noise_levels",
            "spike_amplitudes",
            "correlograms",
            "unit_locations",
            "template_similarity",
        ],
        n_jobs=1,
        progress_bar=False,
    )
    return folder


def test_focused_inspection_keeps_cached_arrays_lazy_and_filters_selector(
    inspection_analyzer_folder, monkeypatch
):
    from spyglass.spikesorting.v2._analyzer_cache import load_analyzer_folder
    from spyglass.spikesorting.v2._review_inspection import inspection_view
    from spyglass.spikesorting.v2._review_profile import ReviewDisplayOptions

    analyzer = load_analyzer_folder(inspection_analyzer_folder)
    waveforms = analyzer.get_extension("waveforms").data["waveforms"]
    assert isinstance(waveforms, np.memmap)

    def no_clone(*args, **kwargs):
        raise AssertionError("inspection must not copy an analyzer")

    monkeypatch.setattr(analyzer, "select_units", no_clone)
    selected = [analyzer.unit_ids[2], analyzer.unit_ids[0]]
    view = inspection_view(
        analyzer,
        ReviewDisplayOptions(max_amplitudes_per_unit=7),
        unit_ids=selected,
        time_range=(0.5, 0.6),
        displayed_unit_properties=[],
        extra_unit_properties={"official_snr": np.array([5.0, 6.0, 7.0])},
    )
    assert [row.unit_id for row in view.item1.view.rows] == selected
    assert [
        float(row.values["official_snr"]) for row in view.item1.view.rows
    ] == [7, 5]
    assert {
        (score.unit_id1, score.unit_id2)
        for score in view.item1.view.similarity_scores
    } == {(u, v) for u in selected for v in selected}
    assert analyzer.get_extension("waveforms").data["waveforms"] is waveforms
    assert "spike_amplitudes" not in analyzer.extensions
    amplitudes = view.item2.view.items[1].view
    assert [item.unit_id for item in amplitudes.plots] == selected
    assert all(len(item.spike_times_sec) == 7 for item in amplitudes.plots)
    raster = view.item2.view.items[5].view
    for plot in raster.plots:
        times = (
            analyzer.sorting.get_unit_spike_train(plot.unit_id)
            / analyzer.sampling_frequency
        )
        np.testing.assert_allclose(
            plot.spike_times_sec, times[(times >= 0.5) & (times < 0.6)]
        )


def test_deferred_inspection_never_loads_amplitudes_or_builds_raster(
    inspection_analyzer_folder, monkeypatch
):
    from spyglass.spikesorting.v2 import _review_inspection as inspection
    from spyglass.spikesorting.v2._analyzer_cache import load_analyzer_folder
    from spyglass.spikesorting.v2._review_profile import ReviewDisplayOptions

    analyzer = load_analyzer_folder(inspection_analyzer_folder)

    def unexpected(*args, **kwargs):
        raise AssertionError("deferred data must not be read")

    monkeypatch.setattr(inspection, "amplitude_view", unexpected)
    monkeypatch.setattr(inspection, "raster_view", unexpected)
    view = inspection.inspection_view(
        analyzer, ReviewDisplayOptions(), deferred=True
    )
    assert "spike_amplitudes" not in analyzer.extensions
    assert view.item2.view.items[5].label == "Raster (not loaded)"


def test_amplitude_sampling_is_repeatable_and_uses_relative_times(
    inspection_analyzer_folder,
):
    from spyglass.spikesorting.v2._analyzer_cache import load_analyzer_folder
    from spyglass.spikesorting.v2._review_inspection import amplitude_view
    from spyglass.spikesorting.v2._review_profile import ReviewDisplayOptions

    analyzer = load_analyzer_folder(inspection_analyzer_folder)
    analyzer.recording.set_times(
        100
        + np.arange(analyzer.get_num_samples()) / analyzer.sampling_frequency
    )
    display = ReviewDisplayOptions(max_amplitudes_per_unit=7)
    unit = analyzer.unit_ids[1]
    first = amplitude_view(analyzer, display, [unit]).plots[0]
    second = amplitude_view(analyzer, display, [unit]).plots[0]
    np.testing.assert_array_equal(first.spike_times_sec, second.spike_times_sec)
    np.testing.assert_array_equal(
        first.spike_amplitudes, second.spike_amplitudes
    )
    assert (first.spike_times_sec < 2).all()
    all_amplitudes = analyzer.get_extension("spike_amplitudes").get_data(
        outputs="by_unit"
    )[0][unit]
    frames = analyzer.sorting.get_unit_spike_train(unit)
    for time, amplitude in zip(first.spike_times_sec, first.spike_amplitudes):
        index = np.argmin(abs(frames / analyzer.sampling_frequency - time))
        assert amplitude == pytest.approx(all_amplitudes[index])


def test_cache_inventory_reads_parameters_without_loading_arrays(
    inspection_analyzer_folder, monkeypatch
):
    from spyglass.spikesorting.v2._analyzer_cache import load_analyzer_folder
    from spyglass.spikesorting.v2._curation_analyzer import _extension_inventory

    analyzer = load_analyzer_folder(inspection_analyzer_folder)
    assert set(analyzer.extensions) == {"waveforms"}

    def no_load(*args, **kwargs):
        raise AssertionError("cache validation must not load extension arrays")

    monkeypatch.setattr(analyzer, "load_extension", no_load)
    inventory = _extension_inventory(analyzer, "display", exact=False)
    assert set(inventory) == set(analyzer.get_saved_extension_names())
    assert set(analyzer.extensions) == {"waveforms"}


def test_disk_copy_retains_unloaded_extensions_and_expert_si_operations(
    inspection_analyzer_folder, tmp_path, monkeypatch
):
    from spyglass.spikesorting.v2._analyzer_cache import (
        analyzer_extension_array,
        copy_analyzer_folder,
        load_analyzer_extensions,
        load_analyzer_folder,
    )

    original = load_analyzer_folder(inspection_analyzer_folder)
    saved = set(original.get_saved_extension_names())

    def no_load(*args, **kwargs):
        raise AssertionError("copying a cache must not load extension arrays")

    monkeypatch.setattr(original, "load_extension", no_load)
    working = copy_analyzer_folder(original, tmp_path / "copy")
    assert set(working.get_saved_extension_names()) == saved
    assert set(working.extensions) == {"waveforms"}
    np.testing.assert_array_equal(
        working.recording.get_traces(start_frame=0, end_frame=100),
        original.recording.get_traces(start_frame=0, end_frame=100),
    )
    before = float(
        analyzer_extension_array(original, "spike_amplitudes", "amplitudes")[0]
    )
    load_analyzer_extensions(working)
    assert set(working.extensions) == saved
    assert isinstance(
        working.get_extension("waveforms").data["waveforms"], np.memmap
    )
    selected = working.select_units([working.unit_ids[0]], format="memory")
    assert set(selected.extensions) == saved
    working.get_extension("spike_amplitudes").data["amplitudes"][0] += 1
    working.get_extension("spike_amplitudes").save()
    assert (
        float(
            analyzer_extension_array(
                original, "spike_amplitudes", "amplitudes"
            )[0]
        )
        == before
    )
    # Metadata was rebased to the new folder, so reopening needs no override.
    reopened = load_analyzer_folder(working.folder)
    assert reopened.has_recording()
    np.testing.assert_array_equal(
        reopened.recording.get_traces(start_frame=0, end_frame=100),
        original.recording.get_traces(start_frame=0, end_frame=100),
    )


def test_expert_cache_load_still_recovers_corrupt_extension_data(
    inspection_analyzer_folder, tmp_path, monkeypatch
):
    from contextlib import nullcontext

    from spyglass.spikesorting.v2 import _analyzer_cache as cache
    from spyglass.spikesorting.v2._sorting_analyzer import (
        _load_analyzer_folder_or_rebuild,
    )
    from spyglass.spikesorting.v2.exceptions import AnalyzerFolderInvalidError

    original = cache.load_analyzer_folder(inspection_analyzer_folder)
    folder = tmp_path / "corrupt"
    cache.copy_analyzer_folder(original, folder)
    (folder / "extensions/templates/average.npy").write_bytes(b"invalid array")
    monkeypatch.setattr(
        cache, "analyzer_cache_lock", lambda _key: nullcontext()
    )
    rebuilds = []

    def rebuild():
        rebuilds.append(True)
        cache.copy_analyzer_folder(original, folder)

    options = {
        "folder": folder,
        "rebuild_fn": rebuild,
        "recipe_label": "test",
        "sorting_id": "test",
    }
    with pytest.raises(AnalyzerFolderInvalidError):
        _load_analyzer_folder_or_rebuild(**options, rebuild=False)
    recovered = _load_analyzer_folder_or_rebuild(**options, rebuild=True)
    assert rebuilds == [True]
    np.testing.assert_array_equal(
        recovered.get_extension("templates").get_data(),
        original.get_extension("templates").get_data(),
    )
