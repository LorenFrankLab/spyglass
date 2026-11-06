"""Tests for the ``_signal_math`` helpers: input guards and frame mapping
(no DB, no NWB IO).
"""

from __future__ import annotations

import pytest


def test_assert_positive_sampling_frequency_rejects_nonpositive_or_nonfinite():
    """A finite positive fs passes (returned as float); 0, negative, NaN and
    inf raise."""
    from spyglass.spikesorting.v2._core.signal_math import (
        assert_positive_sampling_frequency,
    )

    assert assert_positive_sampling_frequency(30000) == 30000.0
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite positive"):
            assert_positive_sampling_frequency(bad, context="ctx: ")


def test_assert_monotonic_timestamps_rejects_empty_and_backward_steps():
    """A non-decreasing vector (duplicates allowed) and a single sample pass;
    an empty vector or a backward step raise."""
    import numpy as np

    from spyglass.spikesorting.v2._core.signal_math import (
        assert_monotonic_timestamps,
    )

    assert_monotonic_timestamps(np.array([0.0, 1.0, 1.0, 2.0]))
    assert_monotonic_timestamps(np.array([5.0]))
    with pytest.raises(ValueError, match="empty"):
        assert_monotonic_timestamps(np.array([]), context="ctx: ")
    with pytest.raises(ValueError, match="monotonic"):
        assert_monotonic_timestamps(np.array([0.0, 2.0, 1.0]), context="ctx: ")
    # NaN must raise: ``NaN < 0`` is False, so a diff check alone passes it.
    with pytest.raises(ValueError, match="non-finite"):
        assert_monotonic_timestamps(
            np.array([0.0, 1.0, np.nan, 0.5, 2.0]), context="ctx: "
        )


def test_assert_artifact_frame_fraction_guards_pathological_overmasking():
    """An artifact fraction over the bound raises
    ``ArtifactFractionExceededError``; at or under it, or with
    ``n_samples == 0``, it is a no-op."""
    from spyglass.spikesorting.v2._core.signal_math import (
        _MAX_ARTIFACT_FRAME_FRACTION,
        assert_artifact_frame_fraction,
    )
    from spyglass.spikesorting.v2.exceptions import (
        ArtifactFractionExceededError,
    )

    n = 1000
    bound = int(_MAX_ARTIFACT_FRAME_FRACTION * n)
    # At/under the bound: no-op (returns None).
    assert assert_artifact_frame_fraction(0, n) is None
    assert assert_artifact_frame_fraction(bound, n) is None
    # Over the bound: raises, surfacing the realized fraction and the context.
    with pytest.raises(ArtifactFractionExceededError, match="ctx: "):
        assert_artifact_frame_fraction(bound + 1, n, context="ctx: ")
    # Degenerate n_samples == 0: no fraction to form -> no-op.
    assert assert_artifact_frame_fraction(0, 0) is None


# --------------------------------------------------------------------------- #
# Lazy timestamp helpers (via ``sample_index_to_time``) match the full
# ``get_times()`` vector for rate-based and explicit recordings.
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
    with (
        warnings.catch_warnings()
    ):  # set_times warns "not recommended"; fine here
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
        ts[n // 3 :] += 5.0
        ts[2 * n // 3 :] += 11.0
    return _explicit_recording(ts, fs)


@pytest.mark.parametrize("kind", ["rate", "contiguous", "disjoint"])
def test_frames_for_times_matches_full_vector_searchsorted(kind):
    """``frames_for_times`` == ``searchsorted(get_times(), t, "left")`` exactly,
    including at grid points, +/- 1 ns, out of range and in gaps."""
    import numpy as np

    from spyglass.spikesorting.v2._core.signal_math import frames_for_times

    n = 200_000
    rec = _recording_for_kind(kind, n)
    full = rec.get_times()
    queries = np.concatenate(
        [
            full[[0, 1, n // 3, n // 2, n - 2, n - 1]],
            full[[0, n // 2, n - 1]] + 1e-9,
            full[[0, n // 2, n - 1]] - 1e-9,
            # Midway across the gap: inside it for the disjoint recording.
            [(full[n // 3 - 1] + full[n // 3]) / 2.0],
            [full[0] - 1.0, full[-1] + 1.0],
        ]
    )
    np.testing.assert_array_equal(
        frames_for_times(rec, queries),
        np.searchsorted(full, queries, side="left"),
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_frames_for_times_rejects_non_finite(bad):
    """A NaN/Inf query time raises; finite out-of-range times clamp."""
    import numpy as np

    from spyglass.spikesorting.v2._core.signal_math import frames_for_times

    rec = _recording_for_kind("rate", 1000)
    with pytest.raises(ValueError, match="non-finite"):
        frames_for_times(rec, np.array([0.001, bad]))


@pytest.mark.parametrize("kind", ["rate", "contiguous", "disjoint"])
def test_base_intervals_and_gaps_matches_full_vector(kind):
    """``base_intervals_and_gaps`` reproduces the full-vector base intervals AND
    the ``np.diff`` gap frame indices, without materializing ``get_times()``."""
    import numpy as np

    from spyglass.spikesorting.v2._core.signal_math import (
        base_intervals_and_gaps,
    )
    from tests.spikesorting.v2._interval_references import (
        base_intervals_from_timestamps,
    )

    n = 200_000
    rec = _recording_for_kind(kind, n)
    full = rec.get_times()
    base_ref = np.asarray(base_intervals_from_timestamps(full, FS))
    gap_ref = np.flatnonzero(np.diff(full) > 1.5 / FS)

    base, gap = base_intervals_and_gaps(rec, FS)
    np.testing.assert_array_equal(np.asarray(base), base_ref)
    np.testing.assert_array_equal(gap, gap_ref)
    expected_gaps = {"rate": 0, "contiguous": 0, "disjoint": 2}[kind]
    assert gap.size == expected_gaps


def test_base_intervals_and_gaps_gap_at_scan_chunk_boundary():
    """A gap on a scan-chunk seam and one inside a chunk both match the
    full-vector reference. The scan steps in ``round(fs)``-frame chunks."""
    import numpy as np

    from spyglass.spikesorting.v2._core.signal_math import (
        base_intervals_and_gaps,
    )
    from tests.spikesorting.v2._interval_references import (
        base_intervals_from_timestamps,
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

    base_ref = np.asarray(base_intervals_from_timestamps(ts, FS))
    gap_ref = np.flatnonzero(np.diff(ts) > 1.5 / FS)
    assert gap_ref.tolist() == [seam_gap - 1, interior_gap - 1]  # sanity

    base, gap = base_intervals_and_gaps(rec, FS)
    np.testing.assert_array_equal(np.asarray(base), base_ref)
    np.testing.assert_array_equal(gap, gap_ref)


def test_base_intervals_and_gaps_rejects_nonpositive_fs():
    """A zero or negative sampling frequency raises."""
    import numpy as np

    from spyglass.spikesorting.v2._core.signal_math import (
        base_intervals_and_gaps,
    )

    rec = _explicit_recording(np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="finite positive"):
        base_intervals_and_gaps(rec, 0.0)


def test_timestamp_fingerprint_matches_array_equal_semantics():
    """Equal timestamps give equal digests; a tiny shift or a different
    length gives a different digest, as ``np.array_equal`` would.
    """
    import numpy as np

    from spyglass.spikesorting.v2._core.signal_math import timestamp_fingerprint

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

    # A rate-based recording matches an explicit one with the same times.
    rate_rec = _rate_based_recording(n)
    assert timestamp_fingerprint(rate_rec) == timestamp_fingerprint(
        _explicit_recording(rate_rec.get_times())
    )
