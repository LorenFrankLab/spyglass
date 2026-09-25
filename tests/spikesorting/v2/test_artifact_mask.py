"""``apply_artifact_mask`` rejects non-finite / out-of-envelope times.

The complement walk silently clips ``valid_times`` to the recording envelope
and a NaN slips through the ``<`` / sorted ordering checks (every NaN compare is
False), so a non-finite or out-of-recording-envelope interval would under-mask
instead of failing loudly. These guards fire BEFORE the complement walk.

Hermetic -- in-memory NumpyRecording, no DB.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


def _recording(n_samples=30_000, n_channels=4, fs=30_000.0):
    import spikeinterface as si

    return si.NumpyRecording(
        [np.zeros((n_samples, n_channels), dtype=np.float32)],
        sampling_frequency=fs,
    )


def _serializable_recording(n_channels=4, duration=1.0, fs=30_000.0):
    """A JSON-serializable, non-zero base recording (parametric noise).

    Unlike ``_recording`` (an in-memory ``NumpyRecording``, which is not
    JSON-serializable), this is needed by the serialization regression test so
    the masked recording's own json flag -- not the base's -- decides the dump
    format.
    """
    from spikeinterface.core import generate_recording

    return generate_recording(
        num_channels=n_channels,
        durations=[duration],
        sampling_frequency=fs,
        seed=0,
    )


@pytest.mark.parametrize(
    "valid_times",
    [
        [[0.0, np.nan]],
        [[np.inf, 0.5]],
        [[0.0, 0.5], [0.6, np.nan]],
    ],
)
def test_mask_rejects_nonfinite(valid_times):
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        apply_artifact_mask,
    )

    with pytest.raises(ValueError, match="non-finite"):
        apply_artifact_mask(_recording(), np.array(valid_times, dtype=float))


@pytest.mark.parametrize(
    "valid_times",
    [
        [[-1.0, 0.5]],  # starts before the first sample
        [[0.0, 100.0]],  # ends well past the last sample (e.g. ms vs s)
    ],
)
def test_mask_rejects_out_of_envelope(valid_times):
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        apply_artifact_mask,
    )

    with pytest.raises(ValueError, match="envelope"):
        apply_artifact_mask(_recording(), np.array(valid_times, dtype=float))


def test_mask_rejects_multi_segment():
    """A multi-segment recording is a caller error, rejected with a clear message.

    ``apply_artifact_mask`` reads ``segment_index=0`` and passes a
    single-segment ``list_periods`` to ``silence_periods``. The v2 sort
    recording is always a mono-segment concatenated timeline
    (``concatenate_recordings``), so >1 segment signals an upstream construction
    error -- fail loudly here instead of with a cryptic ``IndexError`` from deep
    in SpikeInterface.
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        apply_artifact_mask,
    )

    two_seg = si.NumpyRecording(
        [
            np.zeros((30_000, 4), dtype=np.float32),
            np.zeros((30_000, 4), dtype=np.float32),
        ],
        sampling_frequency=30_000.0,
    )
    assert two_seg.get_num_segments() == 2
    with pytest.raises(ValueError, match="single-segment"):
        apply_artifact_mask(two_seg, np.array([[0.0, 0.5]], dtype=float))


def test_mask_accepts_in_envelope_finite_times():
    """A finite, in-envelope mask still works (the guards don't over-reject)."""
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        apply_artifact_mask,
    )

    rec = _recording()
    # Keep almost the whole recording (mask only the last ~1 ms tail) so the
    # frame-fraction guard does not fire; the call must return a recording.
    out = apply_artifact_mask(rec, np.array([[0.0, 0.999]], dtype=float))
    assert out.get_num_samples(segment_index=0) == rec.get_num_samples(
        segment_index=0
    )


def _roundtrip_as_run_sorter_would(recording, folder):
    """Serialize + reload ``recording`` exactly as SpikeInterface's
    ``run_sorter`` does in ``basesorter.setup_recording`` / reload: dump to
    JSON when ``check_serializability("json")`` is truthy, else pickle, then
    ``load`` it back. Returns the reloaded recording (raises if the reload
    fails, which is the behavior under test).
    """
    from pathlib import Path

    from spikeinterface.core import load

    folder = Path(folder)
    if recording.check_serializability("json"):
        rec_file = folder / "spikeinterface_recording.json"
        recording.dump_to_json(rec_file)
    elif recording.check_serializability("pickle"):
        rec_file = folder / "spikeinterface_recording.pickle"
        recording.dump_to_pickle(rec_file)
    else:  # pragma: no cover - defensive; a masked recording is always one of these
        raise AssertionError(
            "recording is neither json- nor pickle-serializable"
        )
    return load(rec_file, base_folder=folder)


def test_masked_recording_survives_run_sorter_serialization(tmp_path):
    """A masked recording with multiple artifact periods must survive the
    serialize + reload that ``run_sorter`` performs.

    ``apply_artifact_mask`` returns a SpikeInterface ``SilencedPeriodsRecording``
    whose artifact intervals live in ``_kwargs["periods"]`` as a *structured*
    numpy array. That array cannot survive a JSON round-trip (JSON has no
    structured-array type), yet the recording reports ``check_serializability(
    "json") == True``, so ``run_sorter`` dumps it to
    ``spikeinterface_recording.json`` and the reload raises
    ``ValueError: periods must be a np.array with dtype ...`` -- the sort fails
    only when artifact detection actually flags intervals (Sorting.populate on
    real data). Regression for that failure.
    """
    from spikeinterface.preprocessing.silence_periods import (
        SilencedPeriodsRecording,
    )

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        apply_artifact_mask,
    )

    # A JSON-serializable, non-zero base recording. This matters: the bug only
    # manifests when the base is JSON-serializable, so the masked recording's
    # own (buggy) json flag decides the dump format. A NumpyRecording is NOT
    # JSON-serializable and would hide the bug by forcing the pickle path.
    rec = _serializable_recording()
    # Kept intervals whose complement leaves SEVERAL small artifact gaps
    # (~0.20-0.22 s, ~0.40-0.42 s, ~0.60-0.62 s) -- multiple artifact periods,
    # well under the 50% frame-fraction guard.
    valid_times = np.array(
        [[0.0, 0.2], [0.22, 0.4], [0.42, 0.6], [0.62, 0.999]], dtype=float
    )
    masked = apply_artifact_mask(rec, valid_times)
    assert isinstance(masked, SilencedPeriodsRecording)

    # run_sorter's serialize + reload must not raise ...
    reloaded = _roundtrip_as_run_sorter_would(masked, tmp_path)
    assert isinstance(reloaded, SilencedPeriodsRecording)

    # ... and the masking must be preserved through the round-trip: two
    # SEPARATE artifact gaps read back all-zero (proving multiple artifacts
    # were masked), while a kept region retains its non-zero signal.
    fs = rec.get_sampling_frequency()
    first_gap = reloaded.get_traces(
        start_frame=int(0.21 * fs), end_frame=int(0.21 * fs) + 100
    )
    second_gap = reloaded.get_traces(
        start_frame=int(0.61 * fs), end_frame=int(0.61 * fs) + 100
    )
    kept = reloaded.get_traces(
        start_frame=int(0.10 * fs), end_frame=int(0.10 * fs) + 100
    )
    assert np.all(first_gap == 0)
    assert np.all(second_gap == 0)
    assert np.any(kept != 0)


# ---------------------------------------------------------------------------
# Statistics-span helpers (DB-free): complement/boundary/statistics spans,
# and the span-respecting samplers built on top of them.
# ---------------------------------------------------------------------------


def _gapped_recording(
    seg_lengths, *, fs=1000.0, gap_s=1.0, n_channels=2, seed=0
):
    """A NumpyRecording whose persisted timestamps have one wall-clock gap
    per segment boundary (the gapped-recording idiom from
    ``test_concat_artifacts.py``). ``seg_lengths`` are frame counts; frames
    stay contiguous, only the persisted TIMES jump between segments.
    """
    import spikeinterface as si

    n = sum(seg_lengths)
    rng = np.random.default_rng(seed)
    traces = rng.standard_normal((n, n_channels)).astype("float64")
    rec = si.NumpyRecording([traces], sampling_frequency=fs)
    chunks = []
    t0 = 0.0
    for length in seg_lengths:
        chunks.append(t0 + np.arange(length) / fs)
        t0 = chunks[-1][-1] + gap_s
    rec.set_times(np.concatenate(chunks))
    return rec


def _frame_indexed_recording(n_samples, *, fs=1000.0, n_channels=2):
    """A NumpyRecording whose channel 0 holds the frame index, so a returned
    row identifies exactly which frame it came from.
    """
    import spikeinterface as si

    traces = np.zeros((n_samples, n_channels), dtype="float64")
    traces[:, 0] = np.arange(n_samples)
    if n_channels > 1:
        traces[:, 1] = -np.arange(n_samples)
    return si.NumpyRecording([traces], sampling_frequency=fs)


def _contiguous_runs(frame_ids):
    """Split a 1-D int array of frame indices into maximal half-open
    ``[start, end)`` runs of consecutive integers, in order of appearance.
    """
    runs = []
    start = 0
    for i in range(1, len(frame_ids) + 1):
        if i == len(frame_ids) or frame_ids[i] != frame_ids[i - 1] + 1:
            runs.append((int(frame_ids[start]), int(frame_ids[i - 1]) + 1))
            start = i
    return runs


def _mixture_mad(weights_sigmas, *, lo=0.0, hi=50.0, iterations=200):
    """Numerically solve for the MAD ``m`` of a zero-mean Gaussian mixture.

    The mixture is symmetric about 0, so its median is 0 and the MAD is the
    smallest ``m >= 0`` solving ``sum_i w_i * (2*Phi(m/sigma_i) - 1) = 0.5``
    (``Phi`` the standard normal CDF), found by bisection.
    """

    def mass_within(m):
        return sum(
            w * math.erf(m / (sigma * math.sqrt(2.0)))
            for w, sigma in weights_sigmas
        )

    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if mass_within(mid) < 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def test_complement_frame_ranges_basic():
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        complement_frame_ranges,
    )

    assert complement_frame_ranges([(100, 200), (500, 700)], 1000) == [
        (0, 100),
        (200, 500),
        (700, 1000),
    ]


def test_complement_frame_ranges_merges_overlapping_and_unsorted():
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        complement_frame_ranges,
    )

    # Unsorted, overlapping ((100,200) & (150,210)), and adjacent
    # ((500,700) & (700,705)) excluded ranges must all merge first.
    excluded = [(500, 700), (150, 210), (100, 200), (700, 705)]
    assert complement_frame_ranges(excluded, 1000) == [
        (0, 100),
        (210, 500),
        (705, 1000),
    ]


def test_complement_frame_ranges_empty_excluded():
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        complement_frame_ranges,
    )

    assert complement_frame_ranges([], 1000) == [(0, 1000)]


def test_boundary_spans_from_timestamps_splits_at_gaps():
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        boundary_spans_from_timestamps,
    )

    rec = _gapped_recording((100, 100, 100))
    assert boundary_spans_from_timestamps(rec) == [
        (0, 100),
        (100, 200),
        (200, 300),
    ]


def test_boundary_spans_from_timestamps_continuous_is_single_span():
    import spikeinterface as si

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        boundary_spans_from_timestamps,
    )

    rec = si.NumpyRecording(
        [np.zeros((300, 2), dtype="float32")], sampling_frequency=1000.0
    )
    assert boundary_spans_from_timestamps(rec) == [(0, 300)]


def test_concat_boundary_spans_offsets_members():
    import spikeinterface as si

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        concat_boundary_spans,
    )

    member0 = si.NumpyRecording(
        [np.zeros((100, 2), dtype="float32")], sampling_frequency=1000.0
    )
    member1 = _gapped_recording((80, 120))
    spans = concat_boundary_spans([member0, member1], [0, 100])
    assert spans == [(0, 100), (100, 180), (180, 300)]


def test_statistics_spans_intersect_boundaries_and_artifacts():
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        statistics_spans,
    )

    boundary = [(0, 500), (500, 1000)]
    assert statistics_spans(1000, [(450, 550)], boundary) == [
        (0, 450),
        (550, 1000),
    ]


def test_statistics_spans_keeps_adjacent_boundary_spans_unmerged():
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        statistics_spans,
    )

    boundary = [(0, 500), (500, 1000)]
    assert statistics_spans(1000, [], boundary) == [(0, 500), (500, 1000)]


def test_statistics_spans_all_excluded_raises():
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        statistics_spans,
    )

    with pytest.raises(ValueError, match="no artifact-free samples"):
        statistics_spans(1000, [(0, 1000)], [(0, 1000)])


def test_statistics_spans_logs_masked_fraction_and_span_count(caplog):
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        statistics_spans,
    )

    with caplog.at_level("INFO", logger="spyglass"):
        spans = statistics_spans(1000, [(450, 550)], [(0, 500), (500, 1000)])
    assert spans == [(0, 450), (550, 1000)]
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "masked_fraction=0.1000" in messages
    assert "across 2 statistics span" in messages


def test_spans_cover_recording():
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        spans_cover_recording,
    )

    assert spans_cover_recording(None, 1000)
    assert spans_cover_recording([(0, 1000)], 1000)
    assert not spans_cover_recording([(0, 999)], 1000)
    assert not spans_cover_recording([(0, 500), (500, 1000)], 1000)


def test_sample_span_data_never_crosses_a_span_boundary():
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        sample_span_data,
    )

    rec = _frame_indexed_recording(1000)
    spans = [(0, 300), (600, 1000)]
    data = sample_span_data(
        rec, spans, target_samples=150, max_piece=37, seed=0, return_in_uV=False
    )
    frame_ids = data[:, 0].astype(np.int64)
    for start, end in _contiguous_runs(frame_ids):
        containing = [(a, b) for a, b in spans if a <= start and end <= b]
        assert containing, f"run [{start}, {end}) crosses a span boundary"
        a, b = containing[0]
        assert end - start <= b - a


def test_sample_span_data_uses_short_spans():
    import spikeinterface as si

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        sample_span_data,
    )

    fs = 1000.0
    sigma = 3.5
    n = 10_000  # 10 s
    rng = np.random.default_rng(1)
    traces = (sigma * rng.standard_normal((n, 1))).astype("float64")
    rec = si.NumpyRecording([traces], sampling_frequency=fs)
    # 20 spans of 400 ms (400 samples) each, 500 samples apart -> 8 s valid
    # inside the 10 s recording.
    spans = [(k * 500, k * 500 + 400) for k in range(20)]
    assert sum(b - a for a, b in spans) == 8000

    reference = np.concatenate(
        [rec.get_traces(start_frame=a, end_frame=b) for a, b in spans], axis=0
    )
    reference_mad = np.median(np.abs(reference - np.median(reference)))

    data = sample_span_data(
        rec,
        spans,
        target_samples=4000,
        max_piece=400,
        seed=2,
        return_in_uV=False,
    )
    assert data.shape[0] == 4000
    sampled_mad = np.median(np.abs(data - np.median(data)))
    assert sampled_mad == pytest.approx(reference_mad, rel=0.02)


def test_sample_span_data_weights_by_length():
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        sample_span_data,
    )

    rec = _frame_indexed_recording(1000)
    spans = [(0, 900), (900, 1000)]
    data = sample_span_data(
        rec,
        spans,
        target_samples=100,
        max_piece=1000,
        seed=3,
        return_in_uV=False,
    )
    assert data.shape[0] == 100
    frame_ids = data[:, 0].astype(np.int64)
    n_from_big = int(np.sum(frame_ids < 900))
    n_from_small = int(np.sum(frame_ids >= 900))
    assert n_from_big == 90
    assert n_from_small == 10
    assert n_from_big == 9 * n_from_small


def test_sample_span_data_budget_exceeds_valid_data():
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        sample_span_data,
    )

    rec = _frame_indexed_recording(9000)
    spans = [(0, 5000), (5000, 8000)]
    expected = np.concatenate(
        [rec.get_traces(start_frame=a, end_frame=b) for a, b in spans], axis=0
    )
    data = sample_span_data(
        rec,
        spans,
        target_samples=10_000,
        max_piece=500,
        seed=4,
        return_in_uV=False,
    )
    assert data.shape[0] == 8000
    np.testing.assert_array_equal(data, expected)


def test_sample_span_data_exact_quotas_heterogeneous_noise():
    import spikeinterface as si

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        sample_span_data,
    )

    # A higher fs keeps the 9 s / 100 ms durations exact while giving the
    # empirical MAD estimator enough samples for a tight (<2%) comparison
    # against the closed-form mixture MAD -- at fs=1000 (9000/100 samples)
    # the estimator's own sampling noise is comparable to the 2% tolerance.
    fs = 10_000.0
    rng = np.random.default_rng(15)
    big_len = 90_000  # 9 s, sigma=1
    small_len = 1_000  # 100 ms each, sigma=5, x10
    n_small = 10
    n = big_len + n_small * small_len
    traces = np.empty((n, 2), dtype="float64")
    traces[:, 1] = np.arange(n)  # frame index, for provenance
    traces[:big_len, 0] = rng.standard_normal(big_len) * 1.0
    spans = [(0, big_len)]
    cursor = big_len
    for _ in range(n_small):
        traces[cursor : cursor + small_len, 0] = (
            rng.standard_normal(small_len) * 5.0
        )
        spans.append((cursor, cursor + small_len))
        cursor += small_len
    rec = si.NumpyRecording([traces], sampling_frequency=fs)

    target_samples = 50_000
    data = sample_span_data(
        rec,
        spans,
        target_samples=target_samples,
        max_piece=5_000,
        seed=8,
        return_in_uV=False,
    )
    assert data.shape[0] == target_samples

    frame_ids = data[:, 1].astype(np.int64)
    n_from_big = int(np.sum(frame_ids < big_len))
    n_from_small = int(np.sum(frame_ids >= big_len))
    assert n_from_big == 45_000  # exact quota: 50000 * 90000 / 100000
    assert n_from_small == 5_000  # exact quota: 50000 * 10000 / 100000

    # Expected pooled MAD computed numerically from the exact-quota mixture
    # (90% sigma=1, 10% sigma=5) via `_mixture_mad`, not from the
    # length-weighted mean of the sigmas -- which is not the mixture MAD.
    expected_mad = _mixture_mad([(0.9, 1.0), (0.1, 5.0)])
    pooled = data[:, 0]
    sample_mad = np.median(np.abs(pooled - np.median(pooled)))
    assert sample_mad == pytest.approx(expected_mad, rel=0.02)


def test_snippet_sampler_respects_length_per_snippet():
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        sample_span_snippet_starts,
    )

    nsamples = 40
    short_span = (0, nsamples - 1)  # one frame too short to admit a snippet
    long_span = (1000, 1000 + 200)
    spans = [short_span, long_span]

    starts = sample_span_snippet_starts(
        spans, nsamples=nsamples, n_snippets=300, seed=7
    )
    assert starts.shape == (300,)
    assert starts.dtype == np.int64
    assert np.array_equal(starts, np.sort(starts))
    assert np.all(starts >= long_span[0])
    assert np.all(starts + nsamples <= long_span[1])

    with pytest.raises(ValueError, match="no span admits"):
        sample_span_snippet_starts(
            [(0, nsamples - 1)], nsamples=nsamples, n_snippets=10, seed=8
        )


def test_no_piece_crosses_selection_join_when_unmasked():
    from spikeinterface.core.recording_tools import get_random_recording_slices

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        boundary_spans_from_timestamps,
        sample_span_data,
        sample_span_snippet_starts,
        statistics_spans,
    )

    fs = 1000.0
    n = 20_000
    boundary = 10_000
    timed_rec = _gapped_recording((boundary, n - boundary), fs=fs, gap_s=1.0)
    frame_rec = _frame_indexed_recording(n, fs=fs)
    frame_rec.set_times(timed_rec.get_times())

    spans = boundary_spans_from_timestamps(timed_rec)
    assert spans == [(0, boundary), (boundary, n)]
    full_spans = statistics_spans(n, [], spans)
    assert full_spans == spans

    # 200 pieces of 40 frames each (quota 4000/span / max_piece 40).
    data = sample_span_data(
        frame_rec,
        full_spans,
        target_samples=8000,
        max_piece=40,
        seed=9,
        return_in_uV=False,
    )
    assert data.shape[0] == 8000
    frame_ids = data[:, 0].astype(np.int64)
    for start, end in _contiguous_runs(frame_ids):
        assert not (
            start < boundary < end
        ), f"run [{start}, {end}) straddles the join at {boundary}"

    starts = sample_span_snippet_starts(
        full_spans, nsamples=40, n_snippets=200, seed=10
    )
    assert starts.shape == (200,)
    assert not np.any((starts < boundary) & (starts + 40 > boundary))

    # Contrast (not a pass condition of the new code): SpikeInterface's own
    # random-chunk sampler ignores internal joins and DOES straddle the
    # boundary on this same recording -- proving the fixture discriminates.
    slices = get_random_recording_slices(
        timed_rec, num_chunks_per_segment=20, chunk_duration="500ms", seed=11
    )
    assert any(start < boundary < end for _, start, end in slices), (
        "fixture does not discriminate: SI's own sampler never straddled "
        "the join"
    )


def test_sample_span_data_single_full_span_matches_si_chunks():
    import spikeinterface as si
    from spikeinterface.core.recording_tools import get_random_data_chunks

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        sample_span_data,
    )

    fs = 1000.0
    n = 50_000
    rng = np.random.default_rng(12)
    traces = rng.standard_normal((n, 3)).astype("float64")
    rec = si.NumpyRecording([traces], sampling_frequency=fs)

    chunk_size = int(0.5 * fs)
    target_samples = 20 * chunk_size
    seed = 123

    ours = sample_span_data(
        rec,
        [(0, n)],
        target_samples=target_samples,
        max_piece=chunk_size,
        seed=seed,
        return_in_uV=False,
    )
    theirs = get_random_data_chunks(rec, return_in_uV=False, seed=seed)
    np.testing.assert_array_equal(ours, theirs)


def test_apply_artifact_mask_returns_ranges_matching_silenced_samples():
    import spikeinterface as si

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        apply_artifact_mask,
        artifact_frame_ranges,
    )

    fs = 1000.0
    n = 5000
    rng = np.random.default_rng(13)
    # Non-zero, non-constant traces so a masked (zeroed) sample is
    # unambiguous against the original signal.
    traces = (1.0 + rng.standard_normal((n, 2))).astype("float64")
    rec = si.NumpyRecording([traces], sampling_frequency=fs)
    valid_times = np.array([[0.0, 1.0], [1.5, 3.0], [3.5, 4.999]])

    ranges = artifact_frame_ranges(rec, valid_times)
    masked = apply_artifact_mask(rec, valid_times)
    masked_traces = masked.get_traces()

    zeroed_mask = np.all(masked_traces == 0, axis=1)
    expected_mask = np.zeros(n, dtype=bool)
    for a, b in ranges:
        expected_mask[a:b] = True
    np.testing.assert_array_equal(zeroed_mask, expected_mask)
    # Every kept sample keeps exactly its original, non-zero value.
    np.testing.assert_array_equal(
        masked_traces[~expected_mask], traces[~expected_mask]
    )
