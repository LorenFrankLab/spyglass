"""Tests for the ``_signal_math`` pure scientific helpers.

Covers the boundary invariant guards (positive sampling frequency,
monotonic timestamps) wired into the frame-mapping functions
(``_consolidate_intervals`` / ``_spike_times_to_frames`` /
``_base_intervals_from_timestamps``) and the absolute-time merge dedup
(``_dedup_merged_spike_times``) that the lazy units-NWB merge is built on.
All pure numpy -- no DB, no NWB IO.
"""

from __future__ import annotations

import pytest


def test_assert_positive_sampling_frequency_rejects_nonpositive_or_nonfinite():
    """A finite positive fs passes (and is returned as float); 0 / negative /
    NaN / inf raise -- a bad fs silently yields an inf/NaN sample period and
    mis-maps every frame."""
    from spyglass.spikesorting.v2._signal_math import (
        assert_positive_sampling_frequency,
    )

    assert assert_positive_sampling_frequency(30000) == 30000.0
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite positive"):
            assert_positive_sampling_frequency(bad, context="ctx: ")


def test_assert_monotonic_timestamps_rejects_empty_and_backward_steps():
    """A non-decreasing vector (duplicates allowed) and a single sample pass;
    an empty vector or a backward step raise -- searchsorted-based frame
    mapping would otherwise silently mis-slice the recording."""
    import numpy as np

    from spyglass.spikesorting.v2._signal_math import (
        assert_monotonic_timestamps,
    )

    assert_monotonic_timestamps(np.array([0.0, 1.0, 1.0, 2.0]))
    assert_monotonic_timestamps(np.array([5.0]))
    with pytest.raises(ValueError, match="empty"):
        assert_monotonic_timestamps(np.array([]), context="ctx: ")
    with pytest.raises(ValueError, match="monotonic"):
        assert_monotonic_timestamps(np.array([0.0, 2.0, 1.0]), context="ctx: ")
    # A NaN must NOT slip through: ``np.diff`` across NaN is NaN and
    # ``NaN < 0`` is False, so a NaN-masked vector would otherwise pass the
    # backward-step check yet mis-slice via searchsorted (NaN sorts as +inf).
    with pytest.raises(ValueError, match="non-finite"):
        assert_monotonic_timestamps(
            np.array([0.0, 1.0, np.nan, 0.5, 2.0]), context="ctx: "
        )


def test_consolidate_intervals_rejects_nonmonotonic_timestamps():
    """The frame mapping is ``searchsorted``-based, so a backward step in the
    timestamp vector would silently mis-slice; reject it at the boundary."""
    import numpy as np

    from spyglass.spikesorting.v2._signal_math import _consolidate_intervals

    with pytest.raises(ValueError, match="monotonic"):
        _consolidate_intervals(
            np.array([[0.0, 1.0]]), np.array([0.0, 2.0, 1.0])
        )


def test_spike_times_to_frames_rejects_nonmonotonic_recording_times():
    """``_spike_times_to_frames`` searchsorts spike times into the recording
    timeline; a backward step there mis-maps every spike frame."""
    import numpy as np

    from spyglass.spikesorting.v2._signal_math import _spike_times_to_frames

    with pytest.raises(ValueError, match="monotonic"):
        _spike_times_to_frames(
            np.array([0.0, 2.0, 1.0]), np.array([0.5]), 3, unit_id=0
        )


def test_base_intervals_from_timestamps_rejects_nonpositive_fs():
    """A zero/negative sampling frequency makes the gap threshold ``1.5/fs``
    infinite/negative, silently collapsing or exploding the chunk split."""
    import numpy as np

    from spyglass.spikesorting.v2._signal_math import (
        _base_intervals_from_timestamps,
    )

    with pytest.raises(ValueError, match="finite positive"):
        _base_intervals_from_timestamps(np.array([0.0, 1.0, 2.0]), 0.0)


def test_dedup_merged_spike_times_drops_cross_unit_coincidences():
    """Cross-contributor spikes within delta collapse; far ones are kept."""
    from spyglass.spikesorting.v2._signal_math import (
        _dedup_merged_spike_times,
    )

    # unit A=[0.0], unit B=[0.0001] -> 0.1 ms apart, different units -> dedup.
    near = _dedup_merged_spike_times([[0.0], [0.0001]], delta_s=0.4e-3)
    assert near.tolist() == [0.0]
    # unit A=[0.0], unit B=[0.001] -> 1 ms apart -> both kept.
    far = _dedup_merged_spike_times([[0.0], [0.001]], delta_s=0.4e-3)
    assert far.tolist() == [0.0, 0.001]


def test_dedup_merged_spike_times_keeps_within_unit_close_pair():
    """A close pair from the SAME contributor is a real event, kept.

    The membership guard mirrors SI's ``(diff>delta)|(diff(membership)==0)``:
    a sub-delta pair is dropped only when the two spikes came from DIFFERENT
    contributors (a cross-unit double-detection), never within one unit.
    """
    from spyglass.spikesorting.v2._signal_math import (
        _dedup_merged_spike_times,
    )

    out = _dedup_merged_spike_times([[0.0, 0.0001]], delta_s=0.4e-3)
    assert out.tolist() == [0.0, 0.0001]


def test_dedup_merged_spike_times_empty():
    from spyglass.spikesorting.v2._signal_math import (
        _dedup_merged_spike_times,
    )

    assert _dedup_merged_spike_times([], delta_s=0.4e-3).tolist() == []


# --------------------------------------------------------------------------- #
# Chunked / affine timestamp helpers: output-identical to the full-vector path.
#
# ``frames_for_times`` / ``base_intervals_and_gaps`` / ``timestamp_fingerprint``
# read the recording timestamps lazily via ``sample_index_to_time`` instead of
# materializing ``get_times()``. These pin that the outputs are IDENTICAL to the
# full-vector references for rate-based and explicit (contiguous and disjoint)
# recordings -- so a future SpikeInterface change to ``sample_index_to_time``
# semantics fails CI loudly rather than silently shifting frames.
# --------------------------------------------------------------------------- #
FS = 30000.0


def _explicit_recording(times, fs=FS):
    """In-memory SI recording carrying an explicit time_vector."""
    import warnings

    import numpy as np
    import spikeinterface as si

    times = np.asarray(times, dtype=np.float64)
    rec = si.NumpyRecording(
        traces_list=[np.zeros((times.size, 1), dtype="float32")],
        sampling_frequency=fs,
    )
    with warnings.catch_warnings():  # set_times warns "not recommended"; fine here
        warnings.simplefilter("ignore")
        rec.set_times(times, segment_index=0)
    return rec


def _rate_based_recording(n, fs=FS):
    import spikeinterface.core as sc

    return sc.generate_recording(
        num_channels=1, sampling_frequency=fs, durations=[n / fs]
    )


def _recording_for_kind(kind, n, fs=FS):
    import numpy as np

    if kind == "rate":
        return _rate_based_recording(n, fs)
    ts = np.arange(n, dtype=np.float64) / fs
    if kind == "disjoint":  # two wall-clock gaps from disjoint sort intervals
        ts[n // 3:] += 5.0
        ts[2 * n // 3:] += 11.0
    return _explicit_recording(ts, fs)


@pytest.mark.parametrize("kind", ["rate", "contiguous", "disjoint"])
def test_frames_for_times_matches_full_vector_searchsorted(kind):
    """``frames_for_times`` == ``searchsorted(get_times(), t, "left")`` exactly,
    including at exact grid points, +/- 1 ns, out-of-range, and disjoint-gap
    times -- the property the artifact-mask complement walk depends on."""
    import numpy as np

    from spyglass.spikesorting.v2._signal_math import frames_for_times

    n = 200_000
    rec = _recording_for_kind(kind, n)
    full = rec.get_times()
    queries = np.concatenate([
        full[[0, 1, n // 3, n // 2, n - 2, n - 1]],
        full[[0, n // 2, n - 1]] + 1e-9,
        full[[0, n // 2, n - 1]] - 1e-9,
        # midway between the last pre-gap and first post-gap sample: for the
        # disjoint recording this lands INSIDE the dead wall-clock gap (the
        # most disjoint-specific binary-search query); harmless for the others.
        [(full[n // 3 - 1] + full[n // 3]) / 2.0],
        [full[0] - 1.0, full[-1] + 1.0],
    ])
    np.testing.assert_array_equal(
        frames_for_times(rec, queries),
        np.searchsorted(full, queries, side="left"),
    )


@pytest.mark.parametrize("kind", ["rate", "contiguous", "disjoint"])
def test_base_intervals_and_gaps_matches_full_vector(kind):
    """``base_intervals_and_gaps`` reproduces the full-vector base intervals AND
    the ``np.diff`` gap frame indices, without materializing ``get_times()``."""
    import numpy as np

    from spyglass.spikesorting.v2._signal_math import (
        _base_intervals_from_timestamps,
        base_intervals_and_gaps,
    )

    n = 200_000
    rec = _recording_for_kind(kind, n)
    full = rec.get_times()
    base_ref = np.asarray(_base_intervals_from_timestamps(full, FS))
    gap_ref = np.flatnonzero(np.diff(full) > 1.5 / FS)

    base, gap = base_intervals_and_gaps(rec, FS)
    np.testing.assert_array_equal(np.asarray(base), base_ref)
    np.testing.assert_array_equal(gap, gap_ref)
    expected_gaps = {"rate": 0, "contiguous": 0, "disjoint": 2}[kind]
    assert gap.size == expected_gaps


def test_base_intervals_and_gaps_gap_at_scan_chunk_boundary():
    """A wall-clock gap landing exactly on a scan-chunk seam exercises the
    cross-chunk-boundary branch of ``base_intervals_and_gaps`` (the
    ``float(times[0]) - prev_time`` check that emits ``gap_after = start_frame -
    1``), which interior-gap recordings never hit. The chunked scan steps in
    ``round(fs)``-frame chunks, so a gap at frame ``chunk_size - 1`` straddles
    two scan chunks. Compared against the exact full-vector reference, plus one
    interior gap so both gap branches run in one recording."""
    import numpy as np

    from spyglass.spikesorting.v2._signal_math import (
        _base_intervals_from_timestamps,
        base_intervals_and_gaps,
    )

    chunk_size = max(1, int(round(FS)))  # the scan's per-chunk frame count
    n = 4 * chunk_size  # 4 scan chunks
    ts = np.arange(n, dtype=np.float64) / FS
    seam_gap = chunk_size  # gap between frame chunk_size-1 (chunk 0's last) and
    #                        frame chunk_size (chunk 1's first) -> the seam
    interior_gap = 2 * chunk_size + chunk_size // 2  # interior to chunk 2
    ts[seam_gap:] += 5.0
    ts[interior_gap:] += 11.0
    rec = _explicit_recording(ts)

    base_ref = np.asarray(_base_intervals_from_timestamps(ts, FS))
    gap_ref = np.flatnonzero(np.diff(ts) > 1.5 / FS)
    assert gap_ref.tolist() == [seam_gap - 1, interior_gap - 1]  # sanity

    base, gap = base_intervals_and_gaps(rec, FS)
    np.testing.assert_array_equal(np.asarray(base), base_ref)
    np.testing.assert_array_equal(gap, gap_ref)


def test_timestamp_fingerprint_matches_array_equal_semantics():
    """Equal timestamps -> equal digest; any byte difference (a tiny shift OR a
    different length) -> different digest -- the bounded-memory replacement for
    ``np.array_equal`` over two full vectors in the shared-artifact-group check."""
    import numpy as np

    from spyglass.spikesorting.v2._signal_math import timestamp_fingerprint

    n = 100_000
    ts = np.arange(n, dtype=np.float64) / FS
    fp_a = timestamp_fingerprint(_explicit_recording(ts))
    fp_identical = timestamp_fingerprint(_explicit_recording(ts.copy()))
    fp_shifted = timestamp_fingerprint(_explicit_recording(ts + 1e-6))
    fp_shorter = timestamp_fingerprint(_explicit_recording(ts[:-1]))

    assert fp_a == fp_identical and len(fp_a) == 32
    assert fp_a != fp_shifted
    assert fp_a != fp_shorter
    # identical-timestamp equality matches np.array_equal on the full vectors
    assert np.array_equal(
        _explicit_recording(ts).get_times(),
        _explicit_recording(ts.copy()).get_times(),
    )

    # Rate-based agreement: the fingerprint also runs on a rate-based recording
    # (affine sample_index_to_time, no explicit time_vector) and agrees with an
    # explicit recording carrying its exact times -- so a rate-based member
    # cannot spuriously mismatch a byte-identical explicit one.
    rate_rec = _rate_based_recording(n)
    assert timestamp_fingerprint(rate_rec) == timestamp_fingerprint(
        _explicit_recording(rate_rec.get_times())
    )
