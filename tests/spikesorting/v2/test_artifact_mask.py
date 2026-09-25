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
