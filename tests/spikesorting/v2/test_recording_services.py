"""Tests for the recording-stage service helpers.

Covers restriction / truncation tolerance and save-expectation accounting
(``_recording_restriction``), the lazy regular-grid timestamp path and its
parity with the eager ``get_times()`` path, the raw-ElectricalSeries
timestamp-mode detection (``_recording_nwb``), and the preprocessing
filtering-description report (``_recording_preprocessing``). All pure /
NWB-IO -- no DB.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_truncation_tolerance_scales_with_interval_count():
    from spyglass.spikesorting.v2._recording_restriction import (
        truncation_tolerance,
    )

    fs = 30000.0
    assert truncation_tolerance(1, fs) == pytest.approx(2.5 / fs)
    assert truncation_tolerance(20, fs) == pytest.approx(21.5 / fs)


def test_save_expectation_sums_disjoint_intended_intervals():
    """Disjoint intended epochs: the expected duration is the sum of every
    epoch and the interval count is preserved."""
    from spyglass.spikesorting.v2._recording_restriction import (
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
    """A requested epoch shorter than min_segment_length is excluded from the
    request total (the intersect already dropped it from the intended set), so
    it is not flagged as over-request -- it was intentionally dropped."""
    from spyglass.spikesorting.v2._recording_restriction import (
        compute_recording_save_expectation,
    )

    # The intersect already dropped the 0.05 s sliver from the intended set.
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
    """A sort interval running past the raw recording is clipped to raw in the
    intended set; over_request is the dropped past-coverage span (the warning
    trigger)."""
    from spyglass.spikesorting.v2._recording_restriction import (
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
    """The lazy regular-grid path must match the established searchsorted path
    without materializing the full timestamp vector."""
    from spyglass.spikesorting.v2._recording_restriction import (
        _consolidate_regular_intervals,
    )
    from spyglass.spikesorting.v2.utils import _consolidate_intervals

    timestamps = 10.0 + np.arange(12, dtype=float) / 2.0
    intervals = np.array([[11.0, 12.0], [13.0, 13.5], [12.5, 12.5]])

    expected = _consolidate_intervals(intervals, timestamps)
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

    At ``fs=30000`` / ``t_start=17.0`` the products ``(t0 + k/fs - t0) * fs``
    drift off the integer ``k`` by ~1e-11 (e.g. ``4.000000000026`` and
    ``8.999999999979``). Without snapping, ``ceil`` pushes the start sample
    forward and ``floor`` pulls the stop sample back, dropping the first and
    last samples relative to the searchsorted timestamp path. The snapped
    rate-based path must agree with ``_consolidate_intervals`` exactly.
    """
    from spyglass.spikesorting.v2._recording_restriction import (
        _consolidate_regular_intervals,
    )
    from spyglass.spikesorting.v2.utils import _consolidate_intervals

    fs = 30000.0
    t_start = 17.0
    n_samples = 12
    timestamps = t_start + np.arange(n_samples, dtype=float) / fs
    # Sample-aligned boundaries whose float products land just off the integer.
    intervals = np.array([[t_start + 4 / fs, t_start + 9 / fs]])

    expected = _consolidate_intervals(intervals, timestamps)
    out = _consolidate_regular_intervals(
        intervals,
        n_samples=n_samples,
        sampling_frequency=fs,
        t_start=t_start,
    )

    np.testing.assert_array_equal(out, expected)
    # Pin the intended frames so a future regression to the searchsorted path
    # can't silently move both sides together.
    np.testing.assert_array_equal(out, np.array([[4, 10]]))


def test_lazy_timestamp_override_indexes_and_slices_regular_grid():
    """Lazy timestamp overrides expose array-like indexing while allocating only
    requested slices."""
    from spyglass.spikesorting.v2._recording_restriction import (
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
    """Lazy timestamp overrides intentionally support only the current streaming
    consumers' int/slice indexing; unsupported indexing should fail explicitly.
    """
    from spyglass.spikesorting.v2._recording_restriction import (
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
    from spyglass.spikesorting.v2._recording_restriction import (
        _lazy_timestamp_override,
    )
    from spyglass.spikesorting.v2.utils import _get_recording_timestamps

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

    from spyglass.spikesorting.v2._recording_nwb import (
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
    """Raw-source resolution pins to the series matching ``raw_object_id`` --
    the exact object the common ``Raw`` row points at -- NOT the first
    acquisition ElectricalSeries.

    An NWB can hold more than one acquisition ``ElectricalSeries`` (a repacked
    or copied file can also reorder acquisition iteration). The previous
    resolver returned whichever series came first, so v2 could silently
    preprocess and sort a different signal than the ``RecordingSelection``
    lineage implies. This pins resolution to the object id and asserts the
    matched series' OWN path and timestamp mode are returned: the rate-based
    first series vs. the explicit-timestamp second series.
    """
    from tests.spikesorting.v2._ingest_helpers import write_two_eseries_nwb

    from spyglass.spikesorting.v2._recording_nwb import (
        raw_eseries_path_and_timestamp_mode,
    )

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

    # An object id absent from the file's acquisition series fails loudly
    # rather than silently falling back to the first series.
    with pytest.raises(ValueError, match="object_id"):
        raw_eseries_path_and_timestamp_mode(str(path), "no-such-object-id")


def test_lazy_regular_path_matches_eager_on_nonzero_start_recording():
    """The lazy regular-grid path reproduces the eager ``get_times()`` path
    byte-for-byte on a real SI recording whose ``t_start`` is non-zero.

    Pins SpikeInterface's affine timestamp origin (``get_start_time`` /
    ``get_times``): a future SI change that shifted the regular-grid origin --
    or a recording read that dropped ``starting_time`` -- would diverge here
    rather than silently persisting wall-clock timestamps that start at the
    wrong offset (the smoke fixtures all start at 0.0, so nothing else catches
    it). Two disjoint intervals also exercise the multi-interval concat path.
    """
    from spikeinterface.core import NumpyRecording

    from spyglass.spikesorting.v2._recording_restriction import (
        _consolidate_regular_intervals,
        _lazy_timestamp_override,
        _recording_has_explicit_time_vector,
        _recording_num_frames,
        _recording_start_time,
    )
    from spyglass.spikesorting.v2.utils import _consolidate_intervals

    fs, n_frames, t_start = 2.0, 12, 10.0
    rec = NumpyRecording(
        traces_list=[np.zeros((n_frames, 2), dtype="float32")],
        sampling_frequency=fs,
        t_starts=[t_start],
    )
    # A regular recording must route to the lazy path with the true t_start.
    assert _recording_has_explicit_time_vector(rec) is False
    assert _recording_start_time(rec) == pytest.approx(t_start)
    assert _recording_num_frames(rec) == n_frames

    valid_times = np.array([[10.5, 11.5], [13.0, 14.5]])
    eager_times = rec.get_times(segment_index=0)

    eager_frames = _consolidate_intervals(valid_times, eager_times)
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
    # Materialize via the slice path the NWB chunk iterator actually uses, and
    # require byte-identity (the persisted timestamps feed the cache hash).
    np.testing.assert_array_equal(lazy_override[:], eager_override)
    # First/last reads are the truncation-guard accessors.
    assert float(lazy_override[0]) == float(eager_override[0])
    assert float(lazy_override[-1]) == float(eager_override[-1])


def test_explicit_time_vector_recording_is_not_linearized():
    """A recording carrying an explicit (irregular) time vector takes the eager
    path -- the affine lazy reconstruction would silently overwrite irregular
    wall-clock timestamps with a regular grid.
    """
    from spikeinterface.core import NumpyRecording

    from spyglass.spikesorting.v2._recording_restriction import (
        _recording_has_explicit_time_vector,
    )

    fs, n_frames, t_start = 2.0, 12, 10.0
    rec = NumpyRecording(
        traces_list=[np.zeros((n_frames, 2), dtype="float32")],
        sampling_frequency=fs,
    )
    # A jittered, strictly-increasing vector that is NOT an affine grid.
    irregular = t_start + np.cumsum(
        np.full(n_frames, 1.0 / fs) + np.linspace(0.0, 0.05, n_frames)
    )
    rec.set_times(irregular, segment_index=0)
    assert _recording_has_explicit_time_vector(rec) is True


def test_filtering_description_lists_only_steps_that_ran():
    from types import SimpleNamespace

    from spyglass.spikesorting.v2._recording_preprocessing import (
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
    # Bandpass first, then reference (the non-commutative apply order).
    assert filtering_description(bp, "global_median", no_ps) == (
        "bandpass filter 300-6000 Hz; common reference (global_median)"
    )

    # Phase-shift is listed ONLY when the report says it ran, and prepended
    # first (the apply order). A requested-but-skipped phase-shift (report
    # False) must NOT appear -- the report-driven honesty check.
    ran = {"phase_shift": True}
    assert filtering_description(bp, "global_median", ran) == (
        "phase-shift (ADC); bandpass filter 300-6000 Hz; "
        "common reference (global_median)"
    )
    assert "phase-shift" not in filtering_description(
        bp, "global_median", no_ps
    )
