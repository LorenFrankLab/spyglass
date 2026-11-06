"""Tests for the recording-stage service helpers.

Restriction and save-expectation accounting, lazy timestamps, raw
ElectricalSeries timestamp-mode detection and the preprocessing report.
No DB.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize("timestamp_mode", ["explicit", "rate"])
@pytest.mark.parametrize("channel_names", [False, True])
def test_nwb_reader_is_lazy_and_preserves_metadata(
    tmp_path, monkeypatch, timestamp_mode, channel_names
):
    """Reader and serialized reload preserve traces, timing and channel metadata."""
    import h5py
    import pynwb
    import spikeinterface as si
    import spikeinterface.extractors as se

    from spyglass.spikesorting.v2._storage.nwb import read_recording_nwb
    from tests.spikesorting.v2._ingest_helpers import (
        write_processed_recording_nwb,
    )

    fs, n_frames = 30_000.0, 20_000
    times = np.arange(n_frames) / fs + 2.0
    times[10_000:] += 0.5  # Explicit timing must preserve the gap.
    path, series_path = write_processed_recording_nwb(
        tmp_path / "reader.nwb",
        traces=np.arange(n_frames * 4, dtype=np.float32).reshape(-1, 4),
        timestamps=times,
        rel_positions=[[0, 0], [0, 20], [10, 40], [10, 60]],
        channel_ids=[10, 30, 20, 50],
        conversion=2e-6,
        offset=1e-6,
    )
    with h5py.File(path, "a") as file:
        series = file[series_path]
        series.create_dataset("channel_conversion", data=[1.0, 2.0, 3.0, 4.0])
        if timestamp_mode == "rate":
            del series["timestamps"]
            start = series.create_dataset("starting_time", data=2.0)
            start.attrs["rate"] = fs
            start.attrs["unit"] = "seconds"
    if channel_names:
        with pynwb.NWBHDF5IO(str(path), "a") as io:
            nwb = io.read()
            nwb.add_electrode_column(
                "channel_name", "channel identifiers", data=["a", "b", "c", "d"]
            )
            io.write(nwb)

    original = se.read_nwb_recording(
        str(path), electrical_series_path=series_path, load_time_vector=True
    )
    dataset_getitem = h5py.Dataset.__getitem__

    def bounded_getitem(dataset, index):
        if dataset.name.endswith("/timestamps") and isinstance(index, slice):
            assert len(range(*index.indices(len(dataset)))) <= 1000
        return dataset_getitem(dataset, index)

    with monkeypatch.context() as guard:
        guard.setattr(h5py.Dataset, "__getitem__", bounded_getitem)
        recording = read_recording_nwb(path, electrical_series_path=series_path)
        reloaded = si.load(recording.to_dict())

    for candidate in (recording, reloaded):
        assert (
            candidate.get_sampling_frequency()
            == original.get_sampling_frequency()
        )
        assert candidate.get_dtype() == original.get_dtype()
        np.testing.assert_array_equal(
            candidate.get_channel_ids(), original.get_channel_ids()
        )
        assert set(candidate.get_property_keys()) == set(
            original.get_property_keys()
        )
        for name in original.get_property_keys():
            np.testing.assert_array_equal(
                candidate.get_property(name), original.get_property(name)
            )
        for scaled in (False, True):
            np.testing.assert_array_equal(
                candidate.get_traces(
                    start_frame=9950, end_frame=10050, return_in_uV=scaled
                ),
                original.get_traces(
                    start_frame=9950, end_frame=10050, return_in_uV=scaled
                ),
            )
        frames = np.array([0, 9999, 10000, n_frames - 1])
        np.testing.assert_array_equal(
            candidate.sample_index_to_time(frames),
            original.sample_index_to_time(frames),
        )


@pytest.mark.parametrize("failure", [None, "replace", "refresh", "verify"])
def test_rebuilt_recording_install_leaves_retryable_state(
    tmp_path, monkeypatch, failure
):
    """Publish verified bytes, or leave no partial file after an install failure."""
    import os
    import sys
    from types import ModuleType

    from spyglass.spikesorting.v2._storage.rebuilds import (
        install_rebuilt_recording,
    )

    temp = tmp_path / "rebuilt.nwb"
    canonical = tmp_path / "canonical.nwb"
    temp.write_bytes(b"verified recording")
    events = []

    def reached(stage):
        events.append(stage)
        if failure == stage:
            raise RuntimeError(f"injected {stage} failure")

    class AnalysisNwbfile:
        def _resolve_external(self, name):
            assert name == canonical.name
            assert not temp.exists()
            assert canonical.read_bytes() == b"verified recording"
            reached("refresh")

        @staticmethod
        def get_abs_path(name):
            assert name == canonical.name
            reached("verify")
            return str(canonical)

    module = ModuleType("spyglass.common.common_nwbfile")
    module.AnalysisNwbfile = AnalysisNwbfile
    monkeypatch.setitem(sys.modules, module.__name__, module)
    replace = os.replace

    def install(src, dst):
        reached("replace")
        replace(src, dst)

    monkeypatch.setattr(os, "replace", install)
    # The stand-in file is not HDF5; equal ids leave it unopened.
    same_object_id = {"rebuilt_object_id": "id", "stored_object_id": "id"}
    if failure:
        with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
            install_rebuilt_recording(
                str(temp), str(canonical), canonical.name, **same_object_id
            )
        assert not canonical.exists()
    else:
        install_rebuilt_recording(
            str(temp), str(canonical), canonical.name, **same_object_id
        )
        assert canonical.read_bytes() == b"verified recording"
        assert events == ["replace", "refresh", "verify"]
    assert not temp.exists()


def test_truncation_tolerance_scales_with_interval_count():
    from spyglass.spikesorting.v2._recording.restriction import (
        truncation_tolerance,
    )

    fs = 30000.0
    assert truncation_tolerance(1, fs) == pytest.approx(2.5 / fs)
    assert truncation_tolerance(20, fs) == pytest.approx(21.5 / fs)


def test_save_expectation_sums_disjoint_intended_intervals():
    """Disjoint intended epochs: the expected duration is the sum of every
    epoch and the interval count is preserved."""
    from spyglass.spikesorting.v2._recording.restriction import (
        compute_recording_save_expectation,
    )

    intended = np.array([[0.0, 1.0], [2.0, 3.5]])
    sort = np.array([[0.0, 1.0], [2.0, 3.5]])
    exp = compute_recording_save_expectation(intended, sort, 0.0)

    assert exp.n_intended_intervals == 2
    assert exp.expected_saved_total == pytest.approx(2.5)  # 1.0 + 1.5
    assert exp.requested_saved_total == pytest.approx(2.5)
    assert exp.over_request == pytest.approx(0.0)


def test_save_expectation_drops_sub_min_segment_slivers():
    """An epoch shorter than min_segment_length is excluded from the request
    total, so it is not flagged as over-request."""
    from spyglass.spikesorting.v2._recording.restriction import (
        compute_recording_save_expectation,
    )

    intended = np.array([[0.0, 1.0]])
    sort = np.array(
        [[0.0, 1.0], [2.0, 2.05]]
    )  # second epoch is a 0.05 s sliver
    exp = compute_recording_save_expectation(intended, sort, 0.1)

    assert exp.n_intended_intervals == 1
    assert exp.expected_saved_total == pytest.approx(1.0)
    assert exp.requested_saved_total == pytest.approx(1.0)  # sliver excluded
    assert exp.over_request == pytest.approx(0.0)


def test_save_expectation_flags_request_past_raw_coverage():
    """A sort interval past the raw recording is clipped to raw;
    over_request is the dropped span."""
    from spyglass.spikesorting.v2._recording.restriction import (
        compute_recording_save_expectation,
    )

    intended = np.array([[0.0, 6.0]])  # intersect clipped [0, 10] to raw [0, 6]
    sort = np.array([[0.0, 10.0]])
    exp = compute_recording_save_expectation(intended, sort, 0.001)

    assert exp.n_intended_intervals == 1
    assert exp.expected_saved_total == pytest.approx(6.0)  # clipped to raw
    assert exp.requested_saved_total == pytest.approx(10.0)  # full request
    assert exp.over_request == pytest.approx(4.0)  # 10 - 6 past coverage


def test_regular_interval_consolidation_matches_timestamp_search():
    """The lazy regular-grid path matches the searchsorted path without
    materializing the full timestamp vector."""
    from spyglass.spikesorting.v2._recording.restriction import (
        _consolidate_regular_intervals,
    )
    from tests.spikesorting.v2._interval_references import (
        consolidate_intervals,
    )

    timestamps = 10.0 + np.arange(12, dtype=float) / 2.0
    intervals = np.array([[11.0, 12.0], [13.0, 13.5], [12.5, 12.5]])

    expected = consolidate_intervals(intervals, timestamps)
    out = _consolidate_regular_intervals(
        intervals,
        n_samples=len(timestamps),
        sampling_frequency=2.0,
        t_start=10.0,
    )

    np.testing.assert_array_equal(out, expected)


def test_regular_interval_consolidation_snaps_grid_boundaries():
    """Boundaries placed exactly on sample lines must not lose the edge sample
    to float round-off.

    At ``fs=30000`` / ``t_start=17.0``, ``(t0 + k/fs - t0) * fs`` misses
    ``k`` by ~1e-11, so unsnapped ``ceil``/``floor`` drop the edge samples.
    """
    from spyglass.spikesorting.v2._recording.restriction import (
        _consolidate_regular_intervals,
    )
    from tests.spikesorting.v2._interval_references import (
        consolidate_intervals,
    )

    fs = 30000.0
    t_start = 17.0
    n_samples = 12
    timestamps = t_start + np.arange(n_samples, dtype=float) / fs
    # Sample-aligned boundaries whose float products land just off the integer.
    intervals = np.array([[t_start + 4 / fs, t_start + 9 / fs]])

    expected = consolidate_intervals(intervals, timestamps)
    out = _consolidate_regular_intervals(
        intervals,
        n_samples=n_samples,
        sampling_frequency=fs,
        t_start=t_start,
    )

    np.testing.assert_array_equal(out, expected)
    # Pin the frames so both paths cannot move together.
    np.testing.assert_array_equal(out, np.array([[4, 10]]))


def test_lazy_timestamp_override_indexes_and_slices_regular_grid():
    """Lazy timestamp overrides expose array-like indexing while allocating only
    requested slices."""
    from spyglass.spikesorting.v2._recording.restriction import (
        _lazy_timestamp_override,
    )

    override = _lazy_timestamp_override(
        np.array([[4, 7], [10, 12]]),
        sampling_frequency=2.0,
        t_start=10.0,
    )

    assert override.shape == (5,)
    assert override[0] == pytest.approx(12.0)  # frame 4 at 2 Hz from t=10
    assert override[-1] == pytest.approx(15.5)  # frame 11
    np.testing.assert_allclose(override[1:4], np.array([12.5, 13.0, 15.0]))


def test_lazy_timestamp_override_rejects_fancy_indexing_clearly():
    """Lazy timestamp overrides support only int/slice indexing; other
    indexing raises."""
    from spyglass.spikesorting.v2._recording.restriction import (
        _lazy_timestamp_override,
    )

    override = _lazy_timestamp_override(
        np.array([[0, 3]]),
        sampling_frequency=1.0,
        t_start=0.0,
    )

    with pytest.raises(TypeError, match="integer and slice indexing"):
        override[np.array([True, False, True])]


def test_get_recording_timestamps_preserves_lazy_override():
    """A lazy override must not be forced through np.asarray before the NWB
    chunk iterator has a chance to stream it."""
    from spyglass.spikesorting.v2._recording.restriction import (
        _lazy_timestamp_override,
    )
    from spyglass.spikesorting.v2._core.signal_math import (
        _get_recording_timestamps,
    )

    override = _lazy_timestamp_override(
        np.array([[0, 3]]),
        sampling_frequency=1.0,
        t_start=5.0,
    )

    assert _get_recording_timestamps(None, override=override) is override


def test_raw_eseries_timestamp_mode_detects_rate_vs_explicit(tmp_path):
    """Rate-based raw series can skip eager SI time-vector loading; explicit
    timestamp series cannot."""
    import h5py

    from spyglass.utils.nwb_helper_fn import (
        raw_eseries_path_and_timestamp_mode,
    )

    def _write(path, *, explicit):
        with h5py.File(path, "w") as nwb_file:
            acq = nwb_file.create_group("acquisition")
            series = acq.create_group("e-series")
            series.attrs["neurodata_type"] = "ElectricalSeries"
            series.attrs["object_id"] = "raw-obj"
            if explicit:
                series.create_dataset("timestamps", data=np.arange(3.0))
            else:
                starting_time = series.create_dataset("starting_time", data=0.0)
                starting_time.attrs["rate"] = 30000.0

    rate_path = tmp_path / "rate.nwb"
    explicit_path = tmp_path / "explicit.nwb"
    _write(rate_path, explicit=False)
    _write(explicit_path, explicit=True)

    assert raw_eseries_path_and_timestamp_mode(str(rate_path), "raw-obj") == (
        "acquisition/e-series",
        False,
    )
    assert raw_eseries_path_and_timestamp_mode(
        str(explicit_path), "raw-obj"
    ) == (
        "acquisition/e-series",
        True,
    )


def test_raw_eseries_path_resolves_by_object_id(tmp_path):
    """Raw-source resolution uses the series matching ``raw_object_id`` (the
    ``Raw`` row's object), not the first acquisition ElectricalSeries.

    Returns that series' own path and timestamp mode.
    """
    from spyglass.utils.nwb_helper_fn import (
        raw_eseries_path_and_timestamp_mode,
    )
    from tests.spikesorting.v2._ingest_helpers import write_two_eseries_nwb

    path = tmp_path / "two_eseries.nwb"
    object_ids = write_two_eseries_nwb(path)
    first_name, first_obj = object_ids["first_series"]
    second_name, second_obj = object_ids["second_series"]

    # Pin to the SECOND series' object id -> its path + explicit-timestamp mode.
    assert raw_eseries_path_and_timestamp_mode(str(path), second_obj) == (
        f"acquisition/{second_name}",
        True,
    )
    # Pin to the FIRST series' object id -> its path + rate-based mode.
    assert raw_eseries_path_and_timestamp_mode(str(path), first_obj) == (
        f"acquisition/{first_name}",
        False,
    )

    # An absent object id raises instead of falling back to the first series.
    with pytest.raises(ValueError, match="object_id"):
        raw_eseries_path_and_timestamp_mode(str(path), "no-such-object-id")


def test_lazy_regular_path_matches_eager_on_nonzero_start_recording():
    """The lazy regular-grid path reproduces the eager ``get_times()`` path
    byte-for-byte on a real SI recording whose ``t_start`` is non-zero.

    Pins SpikeInterface's timestamp origin; the smoke fixtures all start at
    0.0. Two disjoint intervals cover the multi-interval path.
    """
    from spikeinterface.core import NumpyRecording

    from spyglass.spikesorting.v2._recording.restriction import (
        _consolidate_regular_intervals,
        _lazy_timestamp_override,
        _recording_num_frames,
        _recording_start_time,
    )
    from tests.spikesorting.v2._interval_references import (
        consolidate_intervals,
    )

    fs, n_frames, t_start = 2.0, 12, 10.0
    rec = NumpyRecording(
        traces_list=[np.zeros((n_frames, 2), dtype="float32")],
        sampling_frequency=fs,
        t_starts=[t_start],
    )
    # A regular recording must route to the lazy path with the true t_start.
    assert rec.has_time_vector(segment_index=0) is False
    assert _recording_start_time(rec) == pytest.approx(t_start)
    assert _recording_num_frames(rec) == n_frames

    valid_times = np.array([[10.5, 11.5], [13.0, 14.5]])
    eager_times = rec.get_times(segment_index=0)

    eager_frames = consolidate_intervals(valid_times, eager_times)
    lazy_frames = _consolidate_regular_intervals(
        valid_times,
        n_samples=_recording_num_frames(rec),
        sampling_frequency=fs,
        t_start=_recording_start_time(rec),
    )
    np.testing.assert_array_equal(lazy_frames, eager_frames)

    eager_override = np.concatenate(
        [eager_times[int(s) : int(e)] for s, e in eager_frames]
    )
    lazy_override = _lazy_timestamp_override(
        lazy_frames, sampling_frequency=fs, t_start=t_start
    )
    # Slice path the NWB chunk iterator uses; byte-identical because the
    # timestamps feed the cache hash.
    np.testing.assert_array_equal(lazy_override[:], eager_override)
    # First/last reads are the truncation-guard accessors.
    assert float(lazy_override[0]) == float(eager_override[0])
    assert float(lazy_override[-1]) == float(eager_override[-1])


def test_filtering_description_lists_only_steps_that_ran():
    from types import SimpleNamespace

    from spyglass.spikesorting.v2._recording.preprocessing import (
        filtering_description,
    )

    bp = SimpleNamespace(freq_min=300, freq_max=6000)
    no_ps = {"phase_shift": False}
    assert (
        filtering_description(None, "none", no_ps)
        == "none (raw, no preprocessing)"
    )
    assert (
        filtering_description(bp, "none", no_ps)
        == "bandpass filter 300-6000 Hz"
    )
    assert filtering_description(None, "global_median", no_ps) == (
        "common reference (global_median)"
    )
    # Bandpass first, then reference.
    assert filtering_description(bp, "global_median", no_ps) == (
        "bandpass filter 300-6000 Hz; common reference (global_median)"
    )

    # Phase-shift is listed first, and only when the report says it ran.
    ran = {"phase_shift": True}
    assert filtering_description(bp, "global_median", ran) == (
        "phase-shift (ADC); bandpass filter 300-6000 Hz; "
        "common reference (global_median)"
    )
    assert "phase-shift" not in filtering_description(
        bp, "global_median", no_ps
    )


@pytest.mark.parametrize("disjoint", [False, True])
def test_explicit_restriction_keeps_hdf5_timestamps_lazy(
    tmp_path, monkeypatch, disjoint
):
    import h5py

    from spyglass.spikesorting.v2._storage.nwb import read_recording_nwb
    from spyglass.spikesorting.v2._recording.restriction import (
        restrict_recording_times,
    )
    from tests.spikesorting.v2._ingest_helpers import (
        write_processed_recording_nwb,
    )

    fs, count = 30000, 120000
    times = 5 + np.arange(count) / fs
    times[60000:] += 10
    traces = np.arange(count * 2, dtype=np.float32).reshape(-1, 2)
    path, series = write_processed_recording_nwb(
        tmp_path / "restrict.nwb",
        traces=traces,
        timestamps=times,
        rel_positions=[[0, 0], [0, 20]],
    )
    recording = read_recording_nwb(path, electrical_series_path=series)
    spans = [(1000, 119999)] if not disjoint else [(1000, 5000), (65000, 90000)]
    intervals = [[times[start], times[stop - 1]] for start, stop in spans]
    getitem = h5py.Dataset.__getitem__
    as_array = h5py.Dataset.__array__
    reads = []

    def bounded_getitem(dataset, item):
        if dataset.name.endswith("/timestamps"):
            reads.append(item)
            if isinstance(item, slice):
                assert len(range(*item.indices(len(dataset)))) <= fs
        return getitem(dataset, item)

    def no_full_array(dataset, *args, **kwargs):
        assert not dataset.name.endswith(
            "/timestamps"
        ), "materialized full timestamp dataset"
        return as_array(dataset, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", bounded_getitem)
    monkeypatch.setattr(h5py.Dataset, "__array__", no_full_array)
    selected, override, n_intervals = restrict_recording_times(
        recording, intervals
    )
    expected_times = np.concatenate(
        [times[start:stop] for start, stop in spans]
    )
    assert n_intervals == len(spans)
    assert isinstance(recording.get_time_info()["time_vector"], h5py.Dataset)
    assert any(isinstance(item, slice) for item in reads)
    for first in range(0, len(override), 1000):
        np.testing.assert_array_equal(
            override[first : first + 1000], expected_times[first : first + 1000]
        )
    np.testing.assert_array_equal(override[::-100], expected_times[::-100])
    np.testing.assert_array_equal(
        selected.get_traces(),
        np.concatenate([traces[start:stop] for start, stop in spans]),
    )


def test_restriction_preserves_segment_boundaries_and_rejects_backward_time():
    import spikeinterface.core as si

    from spyglass.spikesorting.v2._recording.restriction import (
        restrict_recording_times,
    )

    recording = si.NumpyRecording([np.zeros((100, 1)), np.ones((100, 1))], 100)
    recording.set_times(np.arange(100) / 100, segment_index=0)
    recording.set_times(5 + np.arange(100) / 100, segment_index=1)
    selected, times, count = restrict_recording_times(recording, [[0.5, 5.5]])
    assert count == 2
    np.testing.assert_array_equal(
        times[:], np.r_[np.arange(50, 100) / 100, 5 + np.arange(51) / 100]
    )
    assert selected.get_num_samples() == 101
    recording.set_times(np.arange(100) / 100, segment_index=1)
    with pytest.raises(ValueError, match="backward across segments"):
        restrict_recording_times(recording, [[0, 6]])


# ---------------------------------------------------------------------------
# Filter before restriction, so retained samples are not filtered across
# artificial joins.
# ---------------------------------------------------------------------------

_DRIFT_HZ = 0.3
_DRIFT_UV = 200.0
_FS = 30_000.0
# Not multiples of the drift half-period (1.667 s), where a join has no step.
_INTERVALS = [[2.4, 6.4], [9.3, 9.3015], [12.7, 18.9]]


def _drift_recording(duration_s, n_channels=2, seed=0):
    """Synthetic recording plus an explicit slow drift, gains pinned to 1.

    The drift makes a join visible after a 600 Hz high-pass. Unit gains, so
    values are in uV.
    """
    import spikeinterface as si
    from spikeinterface.core import NumpyRecording

    base = si.generate_recording(
        num_channels=n_channels,
        sampling_frequency=_FS,
        durations=[duration_s],
        seed=seed,
    )
    traces = np.asarray(base.get_traces(), dtype=np.float64)
    times = np.arange(traces.shape[0], dtype=np.float64) / _FS
    traces += (_DRIFT_UV * np.sin(2 * np.pi * _DRIFT_HZ * times))[:, None]
    recording = NumpyRecording([traces], sampling_frequency=_FS)
    recording.set_channel_gains(1.0)
    recording.set_channel_offsets(0.0)
    return recording


def _bandpass_params(freq_min=600.0, freq_max=6000.0):
    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )

    return PreprocessingParamsSchema.model_validate(
        {
            "phase_shift": None,
            "bandpass_filter": {"freq_min": freq_min, "freq_max": freq_max},
            "common_reference": {"operator": "median"},
            "min_segment_length": 0.0015,
            "bad_channel_handling": "remove",
        }
    )


def _selected_frames(recording, intervals):
    """(start, stop) frames of each selected interval on the regular grid."""
    from spyglass.spikesorting.v2._recording.restriction import (
        _consolidate_regular_intervals,
    )

    return [
        (int(start), int(stop))
        for start, stop in _consolidate_regular_intervals(
            np.asarray(intervals, dtype=float),
            n_samples=recording.get_num_samples(segment_index=0),
            sampling_frequency=_FS,
            t_start=0.0,
        )
    ]


def _rms(values):
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


@pytest.mark.unit
def test_restriction_output_unchanged_by_reorder():
    """Restricting a filtered recording selects exactly the same frames.

    SpikeInterface preprocessors copy the parent's time kwargs.
    """
    from spyglass.spikesorting.v2._recording.preprocessing import (
        apply_temporal_preprocessing,
    )
    from spyglass.spikesorting.v2._recording.restriction import (
        restrict_recording_times,
    )

    recording = _drift_recording(20.0)
    filtered, _steps = apply_temporal_preprocessing(
        recording, _bandpass_params()
    )

    assert filtered.get_num_samples() == recording.get_num_samples()
    assert filtered.has_time_vector(0) == recording.has_time_vector(0)
    np.testing.assert_array_equal(filtered.get_times(), recording.get_times())

    raw_sel, raw_times, raw_n = restrict_recording_times(recording, _INTERVALS)
    filtered_sel, filtered_times, filtered_n = restrict_recording_times(
        filtered, _INTERVALS
    )

    assert raw_n == filtered_n == len(_INTERVALS)
    np.testing.assert_array_equal(
        np.asarray(raw_times), np.asarray(filtered_times)
    )
    assert list(filtered_sel.get_channel_ids()) == list(
        raw_sel.get_channel_ids()
    )
    assert filtered_sel.get_num_samples() == raw_sel.get_num_samples()

    # Per interval: compensating shifts would keep the total.
    for interval in _INTERVALS:
        raw_one, raw_one_times, _ = restrict_recording_times(
            recording, [interval]
        )
        filtered_one, filtered_one_times, _ = restrict_recording_times(
            filtered, [interval]
        )
        assert filtered_one.get_num_samples() == raw_one.get_num_samples()
        np.testing.assert_array_equal(
            np.asarray(raw_one_times), np.asarray(filtered_one_times)
        )


@pytest.mark.unit
def test_restricted_traces_match_continuously_filtered_reference():
    """Filter-then-restrict reproduces the continuously filtered signal.

    The target is the same filter over the whole recording, sampled at the
    selected frames. The tolerance allows ~1e-4 uV from SpikeInterface's
    per-request filter margin.
    """
    import scipy.signal
    import spikeinterface.preprocessing as sip

    from spyglass.spikesorting.v2._recording.preprocessing import (
        apply_temporal_preprocessing,
    )
    from spyglass.spikesorting.v2._recording.restriction import (
        restrict_recording_times,
    )

    recording = _drift_recording(20.0)
    validated = _bandpass_params()
    frames = _selected_frames(recording, _INTERVALS)

    # Production order: filter the continuous recording, then restrict.
    filtered, _ = apply_temporal_preprocessing(recording, validated)
    filter_first, _times, n_intervals = restrict_recording_times(
        filtered, _INTERVALS
    )
    assert n_intervals == len(frames)

    # Reverse order, built from the primitives: restrict, then filter the
    # concatenation.
    restricted, _t, _n = restrict_recording_times(recording, _INTERVALS)
    restrict_first, _ = apply_temporal_preprocessing(restricted, validated)

    reference = sip.bandpass_filter(
        recording,
        freq_min=validated.bandpass_filter.freq_min,
        freq_max=validated.bandpass_filter.freq_max,
        dtype=np.float64,
    )

    # Independent scipy pass with SpikeInterface's coefficients, so a margin
    # bug in the filter under test cannot cancel out.
    sos = scipy.signal.iirfilter(
        5,
        [
            validated.bandpass_filter.freq_min,
            validated.bandpass_filter.freq_max,
        ],
        fs=_FS,
        analog=False,
        btype="bandpass",
        ftype="butter",
        output="sos",
    )
    scipy_reference = scipy.signal.sosfiltfilt(
        sos,
        np.asarray(recording.get_traces(return_in_uV=True), dtype=np.float64),
        axis=0,
    )

    offset = 0
    for index, (start, stop) in enumerate(frames):
        count = stop - start
        ref = np.asarray(
            reference.get_traces(
                start_frame=start, end_frame=stop, return_in_uV=True
            ),
            dtype=np.float64,
        )
        filter_first_traces = np.asarray(
            filter_first.get_traces(
                start_frame=offset,
                end_frame=offset + count,
                return_in_uV=True,
            ),
            dtype=np.float64,
        )
        restrict_first_traces = np.asarray(
            restrict_first.get_traces(
                start_frame=offset,
                end_frame=offset + count,
                return_in_uV=True,
            ),
            dtype=np.float64,
        )
        offset += count

        rms_ref = _rms(ref)
        bound = 1e-3 * rms_ref
        filter_first_max = float(np.max(np.abs(filter_first_traces - ref)))
        restrict_first_max = float(np.max(np.abs(restrict_first_traces - ref)))
        assert filter_first_max <= bound, (
            f"interval {index} ({count} samples): max abs error {filter_first_max:.4g} "
            f"uV exceeds {bound:.4g} uV"
        )
        assert abs(_rms(filter_first_traces) - rms_ref) / rms_ref <= 1e-3
        # The join transient lives at the edges.
        assert np.max(np.abs(filter_first_traces[0] - ref[0])) <= bound
        assert np.max(np.abs(filter_first_traces[-1] - ref[-1])) <= bound
        assert restrict_first_max > 100 * bound, (
            f"interval {index}: restrict-then-filter's max abs error "
            f"{restrict_first_max:.4g} "
            f"uV should dwarf {bound:.4g} uV -- the test no longer "
            "discriminates the two orders"
        )

        # Same bound against the independent scipy pass.
        scipy_ref = scipy_reference[start:stop]
        scipy_bound = 1e-3 * _rms(scipy_ref)
        scipy_max = float(np.max(np.abs(filter_first_traces - scipy_ref)))
        assert scipy_max <= scipy_bound, (
            f"interval {index}: max abs error {scipy_max:.4g} uV against the "
            f"scipy whole-trace reference exceeds {scipy_bound:.4g} uV"
        )


# Fraction of the sample period: irregular, but under the 1.5/fs gap threshold.
_CLOCK_JITTER_FRACTION = 0.2


def _explicit_clock_recording(duration_s, seed=11):
    """``_drift_recording`` carrying an explicit, slightly irregular clock.

    Per-frame timestamps, as in Frank-lab raw files, take the
    ``frames_for_times`` branch of ``restrict_recording_times``.
    """
    recording = _drift_recording(duration_s)
    n_samples = recording.get_num_samples(segment_index=0)
    rng = np.random.default_rng(seed)
    steps = (1.0 / _FS) * (
        1.0 + _CLOCK_JITTER_FRACTION * (2.0 * rng.random(n_samples - 1) - 1.0)
    )
    times = np.concatenate([[0.0], np.cumsum(steps)])
    recording.set_times(times, segment_index=0, with_warning=False)
    return recording


def _explicit_clock_frames(recording, intervals):
    """Frames ``restrict_recording_times`` selects on an explicit clock."""
    from spyglass.spikesorting.v2._core.signal_math import frames_for_times

    bounds = np.asarray(intervals, dtype=float).reshape(-1, 2)
    starts = frames_for_times(recording, bounds[:, 0])
    stops = frames_for_times(recording, bounds[:, 1], side="right")
    return [
        (int(start), int(stop))
        for start, stop in zip(starts, stops)
        if start < stop
    ]


@pytest.mark.unit
def test_restricted_traces_match_reference_on_explicit_clock():
    """The same reference agreement, on the explicit-timestamp clock.

    The filtered recording keeps the parent's explicit time vector, so the
    binary-search branch selects the same frames.
    """
    import scipy.signal
    import spikeinterface.preprocessing as sip

    from spyglass.spikesorting.v2._recording.preprocessing import (
        apply_temporal_preprocessing,
    )
    from spyglass.spikesorting.v2._recording.restriction import (
        restrict_recording_times,
    )

    recording = _explicit_clock_recording(20.0)
    validated = _bandpass_params()
    assert recording.has_time_vector(segment_index=0) is True
    assert np.all(np.diff(recording.get_times()) > 0)

    filtered, _ = apply_temporal_preprocessing(recording, validated)
    assert filtered.has_time_vector(segment_index=0) is True
    np.testing.assert_array_equal(filtered.get_times(), recording.get_times())

    # Same frames on the filtered and raw recordings.
    frames = _explicit_clock_frames(recording, _INTERVALS)
    assert frames == _explicit_clock_frames(filtered, _INTERVALS)
    assert len(frames) == len(_INTERVALS)
    # Not the regular-grid answer: this is a genuinely irregular clock.
    assert frames != _selected_frames(recording, _INTERVALS)

    filter_first, _times, n_intervals = restrict_recording_times(
        filtered, _INTERVALS
    )
    assert n_intervals == len(frames)
    assert filter_first.get_num_samples() == sum(
        stop - start for start, stop in frames
    )
    restricted, _t, n_raw = restrict_recording_times(recording, _INTERVALS)
    assert n_raw == n_intervals
    assert restricted.get_num_samples() == filter_first.get_num_samples()
    restrict_first, _ = apply_temporal_preprocessing(restricted, validated)

    reference = sip.bandpass_filter(
        recording,
        freq_min=validated.bandpass_filter.freq_min,
        freq_max=validated.bandpass_filter.freq_max,
        dtype=np.float64,
    )
    # Independent pass with SpikeInterface's coefficients.
    sos = scipy.signal.iirfilter(
        5,
        [
            validated.bandpass_filter.freq_min,
            validated.bandpass_filter.freq_max,
        ],
        fs=_FS,
        analog=False,
        btype="bandpass",
        ftype="butter",
        output="sos",
    )
    scipy_reference = scipy.signal.sosfiltfilt(
        sos,
        np.asarray(recording.get_traces(return_in_uV=True), dtype=np.float64),
        axis=0,
    )

    offset = 0
    for index, (start, stop) in enumerate(frames):
        count = stop - start
        ref = np.asarray(
            reference.get_traces(
                start_frame=start, end_frame=stop, return_in_uV=True
            ),
            dtype=np.float64,
        )
        filter_first_traces = np.asarray(
            filter_first.get_traces(
                start_frame=offset,
                end_frame=offset + count,
                return_in_uV=True,
            ),
            dtype=np.float64,
        )
        restrict_first_traces = np.asarray(
            restrict_first.get_traces(
                start_frame=offset,
                end_frame=offset + count,
                return_in_uV=True,
            ),
            dtype=np.float64,
        )
        offset += count

        rms_ref = _rms(ref)
        bound = 1e-3 * rms_ref
        filter_first_max = float(np.max(np.abs(filter_first_traces - ref)))
        assert filter_first_max <= bound, (
            f"interval {index} ({count} samples): max abs error {filter_first_max:.4g} "
            f"uV exceeds {bound:.4g} uV"
        )
        assert abs(_rms(filter_first_traces) - rms_ref) / rms_ref <= 1e-3
        assert np.max(np.abs(filter_first_traces[0] - ref[0])) <= bound
        assert np.max(np.abs(filter_first_traces[-1] - ref[-1])) <= bound
        assert (
            float(np.max(np.abs(restrict_first_traces - ref))) > 100 * bound
        ), (
            f"interval {index}: restrict-then-filter is too close to the "
            "reference to discriminate the two orders on an explicit clock"
        )

        scipy_ref = scipy_reference[start:stop]
        scipy_bound = 1e-3 * _rms(scipy_ref)
        scipy_max = float(np.max(np.abs(filter_first_traces - scipy_ref)))
        assert scipy_max <= scipy_bound, (
            f"interval {index}: max abs error {scipy_max:.4g} uV against the "
            f"scipy whole-trace reference exceeds {scipy_bound:.4g} uV"
        )


@pytest.mark.unit
def test_sliver_after_highpass_matches_continuous_filter():
    """A 1.5 ms sliver is pure filter transient under restrict-then-filter;
    filter-then-restrict recovers the continuously filtered signal.

    A 1.5 ms sliver is shorter than the 600 Hz high-pass settling time.
    """
    from spyglass.spikesorting.v2._recording.preprocessing import (
        apply_temporal_preprocessing,
    )
    from spyglass.spikesorting.v2._recording.restriction import (
        restrict_recording_times,
    )
    import spikeinterface.preprocessing as sip

    intervals = [[5.0, 20.0], [40.4, 40.4015], [70.0, 110.0]]
    recording = _drift_recording(120.0)
    validated = _bandpass_params(freq_min=600.0, freq_max=6000.0)
    frames = _selected_frames(recording, intervals)
    sliver_start, sliver_stop = frames[1]
    count = sliver_stop - sliver_start
    assert count == 46  # 1.5 ms at 30 kHz, inclusive of both endpoints
    offset = frames[0][1] - frames[0][0]

    filtered, _ = apply_temporal_preprocessing(recording, validated)
    filter_first, _times, _n = restrict_recording_times(filtered, intervals)
    restricted, _t, _n2 = restrict_recording_times(recording, intervals)
    restrict_first, _ = apply_temporal_preprocessing(restricted, validated)
    reference = sip.bandpass_filter(
        recording,
        freq_min=validated.bandpass_filter.freq_min,
        freq_max=validated.bandpass_filter.freq_max,
        dtype=np.float64,
    )

    ref = np.asarray(
        reference.get_traces(
            start_frame=sliver_start,
            end_frame=sliver_stop,
            return_in_uV=True,
        ),
        dtype=np.float64,
    )
    filter_first_traces = np.asarray(
        filter_first.get_traces(
            start_frame=offset, end_frame=offset + count, return_in_uV=True
        ),
        dtype=np.float64,
    )
    restrict_first_traces = np.asarray(
        restrict_first.get_traces(
            start_frame=offset, end_frame=offset + count, return_in_uV=True
        ),
        dtype=np.float64,
    )

    rms_ref = _rms(ref)
    assert abs(_rms(filter_first_traces) - rms_ref) / rms_ref <= 0.01
    assert float(np.max(np.abs(filter_first_traces - ref))) < 1.0
    assert float(np.max(np.abs(restrict_first_traces - ref))) > 10.0
