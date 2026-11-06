"""Artifact-mask input validation and artifact valid-times structure.

- ``apply_artifact_mask`` rejects malformed valid_times instead of
  under-masking.
- ``detect_artifacts`` returns sorted, disjoint, in-bounds valid_times of at
  least ``min_length_s`` that exclude artifacts and never span a wall-clock
  gap.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.spikesorting.v2._ingest_helpers import _synthetic_artifact_recording


def _rec(traces, fs=30000.0, gain=1.0, times=None):
    import spikeinterface as si

    n_ch = traces.shape[1]
    rec = si.NumpyRecording(
        traces_list=[traces.astype("float32")], sampling_frequency=fs
    )
    rec.set_channel_gains([gain] * n_ch)
    rec.set_channel_offsets([0.0] * n_ch)
    if times is not None:
        rec.set_times(np.asarray(times, dtype=float))
    return rec


def _artifact_params(**overrides):
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )

    base = dict(
        detect=True,
        amplitude_threshold_uv=1000.0,
        zscore_threshold=None,
        proportion_above_threshold=1.0,
        removal_window_ms=1.0,
        join_window_ms=1.0,
        min_length_s=1.0,
    )
    base.update(overrides)
    return ArtifactDetectionParamsSchema.model_validate(base)


# --------------------------------------------------------------------------- #
# apply_artifact_mask input validation
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_rejects_empty_valid_times():
    from spyglass.spikesorting.v2.exceptions import (
        EmptyArtifactValidTimesError,
    )
    from spyglass.spikesorting.v2._sorting.artifact_mask import (
        apply_artifact_mask,
    )

    rec = _rec(np.zeros((100, 2)))
    with pytest.raises(EmptyArtifactValidTimesError):
        apply_artifact_mask(rec, np.empty((0, 2)))


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_rejects_wrong_shape():
    from spyglass.spikesorting.v2._sorting.artifact_mask import (
        apply_artifact_mask,
    )

    rec = _rec(np.zeros((100, 2)))
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        apply_artifact_mask(rec, np.array([0.0, 1.0, 2.0]))


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_rejects_end_before_start():
    from spyglass.spikesorting.v2._sorting.artifact_mask import (
        apply_artifact_mask,
    )

    rec = _rec(np.zeros((100, 2)))
    with pytest.raises(ValueError, match="end precedes its start"):
        apply_artifact_mask(rec, np.array([[0.0, 0.001], [0.005, 0.003]]))


@pytest.mark.usefixtures("dj_conn")
@pytest.mark.parametrize(
    "valid_times",
    [
        [[0.002, 0.003], [0.000, 0.001]],  # unsorted by start
        [[0.000, 0.002], [0.001, 0.003]],  # overlapping
    ],
)
def test_apply_artifact_mask_rejects_unsorted_or_overlapping(valid_times):
    from spyglass.spikesorting.v2._sorting.artifact_mask import (
        apply_artifact_mask,
    )

    rec = _rec(np.zeros((100, 2)))
    with pytest.raises(ValueError, match="sorted by start"):
        apply_artifact_mask(rec, np.array(valid_times))


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_zeros_complement_frames():
    """The mask zeros exactly the complement of ``valid_times`` and leaves
    the kept frames untouched."""
    from spyglass.spikesorting.v2._sorting.artifact_mask import (
        apply_artifact_mask,
    )

    fs = 30000.0
    rec = _rec(np.ones((10, 2), dtype="float32"), fs=fs)
    # Keep [0, 3/fs) and [6/fs, end] -> artifact frames 3,4,5. The end must
    # stay within one sample of the last sample (9/fs) or the guard rejects it.
    valid_times = np.array([[0.0, 3.0 / fs], [6.0 / fs, 10.0 / fs]])
    out = apply_artifact_mask(rec, valid_times).get_traces()
    assert np.all(out[3:6] == 0.0), "artifact (complement) frames not zeroed"
    assert np.all(out[0:3] == 1.0), "kept frames before the artifact altered"
    assert np.all(out[6:10] == 1.0), "kept frames after the artifact altered"


# --------------------------------------------------------------------------- #
# detect_artifacts output structure (contiguous)
# --------------------------------------------------------------------------- #


def _assert_valid_times_well_formed(vt, t0, t_end, min_length_s):
    assert vt.ndim == 2 and vt.shape[1] == 2, f"want (n,2); got {vt.shape}"
    starts, ends = vt[:, 0], vt[:, 1]
    assert np.all(ends >= starts), "each interval must have end >= start"
    assert np.all(np.diff(starts) >= 0), "intervals must be start-sorted"
    assert np.all(starts[1:] >= ends[:-1] - 1e-12), "intervals must be disjoint"
    assert (
        starts.min() >= t0 - 1e-9 and ends.max() <= t_end + 1e-9
    ), "intervals must lie within the recording bounds"
    assert np.all(
        (ends - starts) >= min_length_s - 1e-9
    ), f"every interval must be >= min_length_s={min_length_s}"


@pytest.mark.usefixtures("dj_conn")
def test_detect_artifacts_output_structure_contiguous():
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts

    fs = 30000.0
    n, n_ch = 120000, 4  # 4 s
    traces = np.zeros((n, n_ch), dtype="float32")
    art_frame = 60000  # ~2.0 s, interior so both sides exceed min_length
    traces[art_frame : art_frame + 10, :] = 5000.0  # 5000 uV >> 1000 thresh
    rec = _rec(traces, fs=fs)
    params = _artifact_params()

    vt = detect_artifacts(rec, params, job_kwargs=None)
    times = rec.get_times()
    _assert_valid_times_well_formed(
        vt, times[0], times[-1], params.min_length_s
    )

    # The detected artifact (~2.0 s) is excluded from every valid interval.
    art_t = times[art_frame]
    covered = np.any((vt[:, 0] <= art_t) & (art_t <= vt[:, 1]))
    assert not covered, "the detected artifact must not be inside a valid time"


# --------------------------------------------------------------------------- #
# detect_artifacts on a DISJOINT recording (a chunk-boundary artifact
# IS masked, valid_times never span the gap)
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("dj_conn")
def test_detect_artifacts_disjoint_masks_chunk_boundary_artifact():
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts

    fs = 30000.0
    per_chunk = 60000  # 2 s each
    # Two chunks with a 5 s wall-clock gap between them.
    t_a = np.arange(per_chunk) / fs
    t_b = 5.0 + np.arange(per_chunk) / fs
    times = np.concatenate([t_a, t_b])
    n = times.size
    traces = np.zeros((n, 4), dtype="float32")
    # Artifact on chunk A's last 10 frames; removal_window_ms widens it left.
    traces[per_chunk - 10 : per_chunk, :] = 5000.0
    rec = _rec(traces, fs=fs, times=times)
    params = _artifact_params()

    vt = detect_artifacts(rec, params, job_kwargs=None)
    _assert_valid_times_well_formed(
        vt, times[0], times[-1], params.min_length_s
    )

    # The chunk-boundary artifact frame (chunk A's final sample) is excluded.
    art_t = times[per_chunk - 1]
    assert not np.any((vt[:, 0] <= art_t) & (art_t <= vt[:, 1])), (
        "a chunk-boundary artifact must be masked (the chunk-final-sample "
        "edge is unreachable with removal_window_ms>0)"
    )
    # No valid interval spans the inter-chunk gap [2 s, 5 s].
    spans_gap = np.any((vt[:, 0] < 2.5) & (vt[:, 1] > 4.5))
    assert not spans_gap, "valid_times must never span an inter-chunk gap"


# --------------------------------------------------------------------------- #
# interval recovery: artifacts planted at KNOWN times/amplitudes are removed
# over [planted +/- removal_window/2], not merely "detection did not crash"
# --------------------------------------------------------------------------- #


def _removed_intervals(valid_times, t0, t_end):
    """Complement of ``valid_times`` within ``[t0, t_end]``.

    Returns the removed intervals as an ``(n, 2)`` array. ``valid_times``
    must be start-sorted and disjoint.
    """
    removed = []
    cursor = t0
    for start, end in valid_times:
        if start > cursor:
            removed.append([cursor, start])
        cursor = max(cursor, end)
    if cursor < t_end:
        removed.append([cursor, t_end])
    return np.asarray(removed)


@pytest.mark.usefixtures("dj_conn")
def test_detect_artifacts_recovers_planted_interval_times():
    """Two artifacts planted at KNOWN times and a KNOWN amplitude are recovered
    as two removed intervals, each tightly bracketing its planted epoch.

    Checks removal-window expansion, joining and the complement in seconds.
    """
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts

    fs = 30000.0
    n, n_ch = 150000, 4  # 5.0 s
    removal_window_ms = 1.0
    traces = np.zeros((n, n_ch), dtype="float32")
    # Two 10-frame artifacts ~2 s apart (>> join window), 5000 uV on every
    # channel. Inclusive frame spans [a0, a1].
    planted_frames = [(45000, 45009), (105000, 105009)]  # ~1.5 s and ~3.5 s
    for a0, a1 in planted_frames:
        traces[a0 : a1 + 1, :] = 5000.0
    rec = _rec(traces, fs=fs)
    params = _artifact_params(removal_window_ms=removal_window_ms)

    vt = detect_artifacts(rec, params, job_kwargs=None)
    times = rec.get_times()
    _assert_valid_times_well_formed(
        vt, times[0], times[-1], params.min_length_s
    )

    removed = _removed_intervals(vt, times[0], times[-1])
    assert removed.shape[0] == len(planted_frames), (
        f"expected {len(planted_frames)} recovered artifact intervals, got "
        f"{removed.shape[0]}: {removed.tolist()} -- the two well-separated "
        "artifacts must not be merged into one nor split into more"
    )

    # Expected bounds from the planted frames and the param, not the
    # detector's frame math: half a removal window on each side.
    half_window_s = removal_window_ms * 1e-3 / 2.0
    sample = 1.0 / fs
    tol = 3 * sample  # grid snapping + the half-open [start, end) +1 sample
    for (a0, a1), (rem_start, rem_end) in zip(planted_frames, removed):
        planted_start, planted_end = times[a0], times[a1]
        assert rem_start <= planted_start and rem_end >= planted_end, (
            f"planted [{planted_start:.5f}, {planted_end:.5f}] not contained "
            f"in removed [{rem_start:.5f}, {rem_end:.5f}]"
        )
        # Tight: not over-removed.
        assert rem_start == pytest.approx(
            planted_start - half_window_s, abs=tol
        )
        assert rem_end == pytest.approx(planted_end + half_window_s, abs=tol)


# --------------------------------------------------------------------------- #
# detect_artifacts degenerate-configuration guards
# --------------------------------------------------------------------------- #


def test_detect_artifacts_warns_when_min_length_drops_all_valid_time(caplog):
    """When ``min_length_s`` drops every valid sliver, detection returns
    empty ``valid_times`` and warns, rather than failing later at sort time."""
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts

    fs = 30000.0
    n, n_ch = 15000, 2  # 0.5 s: every kept sliver is < min_length_s=1.0
    traces = np.zeros((n, n_ch), dtype="float32")
    traces[n // 2, :] = 5000.0  # one artifact frame, both channels
    rec = _rec(traces, fs=fs)
    params = _artifact_params(min_length_s=1.0)  # amplitude_threshold_uv=1000

    with caplog.at_level("WARNING", logger="spyglass"):
        vt = detect_artifacts(rec, params, context=" for unit-test")
    assert vt.shape == (0, 2)
    assert any("no valid time" in r.message.lower() for r in caplog.records), (
        "expected an empty-valid-times warning; got "
        f"{[r.message for r in caplog.records]}"
    )


def test_detect_artifacts_keeps_short_chunk_no_artifact_touched():
    """``min_length_s`` drops slivers that artifact removal cuts, not a clean
    recorded chunk the preprocessing recipe kept: a short chunk survives the
    same whether or not an artifact lands in another chunk."""
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts

    fs = 1000.0
    # Chunks [0, 10), [20, 20.5) and [30, 40) s split by wall-clock gaps.
    times = np.concatenate(
        [
            np.arange(10000) / fs,
            20 + np.arange(500) / fs,
            30 + np.arange(10000) / fs,
        ]
    )
    params = _artifact_params(min_length_s=1.0)  # amplitude_threshold_uv=1000

    def valid_times(with_artifact):
        traces = np.zeros((times.size, 4), dtype="float32")
        if with_artifact:
            traces[15000:15010] = 5000.0  # inside the last chunk
        return detect_artifacts(_rec(traces, fs=fs, times=times), params)

    short_chunk = [times[10000], times[10499]]
    for with_artifact in (False, True):
        vt = valid_times(with_artifact)
        assert any(
            np.allclose(row, short_chunk) for row in vt
        ), f"with_artifact={with_artifact}: short chunk dropped from {vt}"


def test_detect_artifacts_raises_on_single_channel_zscore_only():
    """A z-score-only config raises on one channel, where the across-channel
    z-score is identically zero."""
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts
    from spyglass.spikesorting.v2.exceptions import (
        InsufficientZScoreChannelsError,
    )

    rec = _rec(np.zeros((3000, 1), dtype="float32"))
    params = _artifact_params(amplitude_threshold_uv=None, zscore_threshold=3.0)
    with pytest.raises(InsufficientZScoreChannelsError):
        detect_artifacts(rec, params)


def test_detect_artifacts_raises_on_two_channel_zscore_only():
    """A z-score-only config raises on two channels, where the
    across-channel z-score is always +/-1."""
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts
    from spyglass.spikesorting.v2.exceptions import (
        InsufficientZScoreChannelsError,
    )

    rec = _rec(np.zeros((3000, 2), dtype="float32"))
    params = _artifact_params(amplitude_threshold_uv=None, zscore_threshold=3.0)
    with pytest.raises(InsufficientZScoreChannelsError):
        detect_artifacts(rec, params)


def test_detect_artifacts_warns_zscore_inert_on_single_channel(caplog):
    """With ``amplitude_threshold_uv`` also set, single-channel z-score is
    inert (not fatal); warn that only the amplitude detector will fire."""
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts

    rec = _rec(np.zeros((3000, 1), dtype="float32"))
    params = _artifact_params(
        amplitude_threshold_uv=1000.0, zscore_threshold=3.0
    )
    with caplog.at_level("WARNING", logger="spyglass"):
        detect_artifacts(rec, params)
    assert any("inert" in r.message.lower() for r in caplog.records)


def test_detect_artifacts_warns_proportion_rounds_to_all_channels(caplog):
    """On two channels, ``proportion_above_threshold=0.7`` rounds up to both
    channels; a warning states the realized requirement."""
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts

    rec = _rec(np.zeros((3000, 2), dtype="float32"))
    params = _artifact_params(
        proportion_above_threshold=0.7, amplitude_threshold_uv=1000.0
    )
    with caplog.at_level("WARNING", logger="spyglass"):
        detect_artifacts(rec, params)
    assert any(
        "all" in r.message.lower() and "channel" in r.message.lower()
        for r in caplog.records
    )


def test_detect_artifacts_no_proportion_warning_on_tetrode(caplog):
    """On a tetrode, ``0.7`` -> 3 of 4 channels, so no 'rounds up' warning."""
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts

    rec = _rec(np.zeros((3000, 4), dtype="float32"))
    params = _artifact_params(
        proportion_above_threshold=0.7, amplitude_threshold_uv=1000.0
    )
    with caplog.at_level("WARNING", logger="spyglass"):
        detect_artifacts(rec, params)
    assert not any("rounds up" in r.message.lower() for r in caplog.records)


def test_detect_artifacts_raises_when_most_frames_flagged():
    """A detector that flags more than half the recording raises
    ``ArtifactFractionExceededError`` before building a per-frame index."""
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts
    from spyglass.spikesorting.v2.exceptions import (
        ArtifactFractionExceededError,
    )

    # Every frame exceeds 1 uV on both channels.
    rec = _rec(np.full((2000, 2), 1000.0, dtype="float32"))
    params = _artifact_params(
        amplitude_threshold_uv=1.0, proportion_above_threshold=1.0
    )
    with pytest.raises(ArtifactFractionExceededError, match="2000 of 2000"):
        detect_artifacts(rec, params, context=" for unit-test")


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_raises_when_valid_times_keep_almost_nothing():
    """A ``valid_times`` keeping only a sliver raises
    ``ArtifactFractionExceededError`` before expanding the complement."""
    from spyglass.spikesorting.v2._sorting.artifact_mask import (
        apply_artifact_mask,
    )
    from spyglass.spikesorting.v2.exceptions import (
        ArtifactFractionExceededError,
    )

    rec = _rec(np.zeros((3000, 2), dtype="float32"))  # 0.1 s @ 30 kHz
    # Keep only the first ~1 ms (~99% complement).
    valid_times = np.array([[0.0, 0.001]])
    with pytest.raises(ArtifactFractionExceededError, match="%"):
        apply_artifact_mask(rec, valid_times)


# --------------------------------------------------------------------------- #
# Chunked artifact detection + the artifact-mask complement walker.
# --------------------------------------------------------------------------- #


def _in_memory_artifact_frames_reference(recording, validated):
    """Whole-recording, in-memory artifact-frame scan (test-only oracle).

    Same per-frame math as the chunked production scan over one full
    ``get_traces`` load. Returns ascending flagged frame indices.
    """
    traces = recording.get_traces(return_in_uV=False)
    # SI's return_in_uV=True transform: float32 gain and offset. No
    # has_scaleable_traces() branch; both are always set here and in NWB reads.
    gains = recording.get_channel_gains().astype(np.float32)
    offsets = recording.get_channel_offsets().astype(np.float32)
    traces_uv = traces.astype(np.float32) * gains[None, :] + offsets[None, :]
    absolute = np.abs(traces_uv)

    n_channels = traces.shape[1]
    n_required = int(np.ceil(validated.proportion_above_threshold * n_channels))

    if validated.amplitude_threshold_uv is not None:
        above_amp = absolute > validated.amplitude_threshold_uv
    else:
        above_amp = np.zeros_like(absolute, dtype=bool)
    if validated.zscore_threshold is not None:
        ch_mean = traces_uv.mean(axis=1, keepdims=True)
        ch_std = traces_uv.std(axis=1, keepdims=True) + 1e-12
        zscores = np.abs((traces_uv - ch_mean) / ch_std)
        above_z = zscores > validated.zscore_threshold
    else:
        above_z = np.zeros_like(absolute, dtype=bool)

    if (
        validated.amplitude_threshold_uv is not None
        and validated.zscore_threshold is not None
    ):
        channel_hit = above_amp | above_z
    elif validated.amplitude_threshold_uv is not None:
        channel_hit = above_amp
    else:
        channel_hit = above_z

    return (channel_hit.sum(axis=1) >= n_required).nonzero()[0]


# Offsets on channels 0, 2 and 4 only; channel 3 (the z-score burst's
# channel) stays at 0 so its window depends on the others' offsets.
_ARTIFACT_CHANNEL_OFFSETS_UV = [1000.0, 0.0, 1000.0, 0.0, 1000.0, 0.0, 0.0, 0.0]

# One outlier among N channels reaches at most |z| = sqrt(N - 1) ~= 2.65.
# The threshold sits below that and above the all-channel burst's ~1.91, so
# only the single-channel bursts trip z.
_ARTIFACT_ZSCORE_THRESHOLD = 2.2


@pytest.mark.parametrize(
    "amplitude_threshold_uv,zscore_threshold,channel_offsets_uv,"
    "proportion_above_threshold",
    [
        (50.0, None, None, 0.5),  # amplitude-only branch
        # z-score-only: proportion 0.125 -> n_required=1, so one outlier
        # channel can flag a frame.
        (None, _ARTIFACT_ZSCORE_THRESHOLD, None, 0.125),
        # OR-combined: the 75_000:75_050 burst (45 µV) trips z but not
        # amplitude, so ignoring z when amplitude is set would drop it.
        (50.0, _ARTIFACT_ZSCORE_THRESHOLD, None, 0.125),
        (50.0, None, _ARTIFACT_CHANNEL_OFFSETS_UV, 0.5),  # amplitude+offsets
        # amplitude + gain: a wrong gain on channel 3 flags 75_000:75_050
        # (180 raw > 50).
        (50.0, None, None, 0.125),
    ],
)
def test_chunked_artifact_matches_in_memory_reference(
    dj_conn,
    amplitude_threshold_uv,
    zscore_threshold,
    channel_offsets_uv,
    proportion_above_threshold,
):
    """The chunked ``scan_artifact_frames`` matches the in-memory reference
    frame for frame, independent of chunk boundaries, for amplitude, z-score,
    OR-combined, offset and gain cases.
    """
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2._artifacts.intervals import (
        scan_artifact_frames,
    )

    rec = _synthetic_artifact_recording(offsets=channel_offsets_uv)
    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=amplitude_threshold_uv,
        zscore_threshold=zscore_threshold,
        proportion_above_threshold=proportion_above_threshold,
        removal_window_ms=1.0,
        join_window_ms=0.0,
        min_length_s=0.001,
    )

    reference = _in_memory_artifact_frames_reference(rec, validated)
    assert reference.size > 0, (
        "This parametrization's reference flagged no frames at all -- a "
        "vacuous case that compares an empty set to an empty set and "
        "cannot catch a regression."
    )

    # Many small chunks (~0.1 s each → ~30 chunks) exercises chunk seams.
    chunked = scan_artifact_frames(
        rec, validated, job_kwargs={"n_jobs": 1, "chunk_duration": "0.1s"}
    )
    # A single chunk spanning the whole recording == the in-memory path.
    single = scan_artifact_frames(
        rec, validated, job_kwargs={"n_jobs": 1, "chunk_size": 90_000}
    )

    # scan returns run ranges; expand them to per-frame sets.
    assert np.array_equal(_expand_runs(chunked), reference), (
        "Chunked artifact runs diverge from the in-memory reference; "
        "chunking changed which frames are flagged."
    )
    assert np.array_equal(
        _expand_runs(single), reference
    ), "Single-chunk pass should reproduce the in-memory reference exactly."


def _expand_runs(runs):
    """Expand ``(start, end_inclusive)`` runs to the ascending per-frame array."""
    if runs.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([np.arange(s, e + 1) for s, e in runs])


def test_scan_artifact_frames_returns_bounded_runs(dj_conn):
    """``scan_artifact_frames`` returns run ranges, not one index per flagged
    frame, so memory is O(n_runs)."""
    from spyglass.spikesorting.v2._artifacts.intervals import (
        scan_artifact_frames,
    )

    traces = np.zeros((5000, 2), dtype="float32")
    traces[:2000] = 1000.0  # first 40% (< 50% guard) flagged, contiguous
    rec = _rec(traces)
    validated = _artifact_params(
        amplitude_threshold_uv=1.0, proportion_above_threshold=1.0
    )
    runs = scan_artifact_frames(
        rec, validated, job_kwargs={"n_jobs": 1, "chunk_duration": "0.05s"}
    )
    assert runs.ndim == 2 and runs.shape[1] == 2
    assert runs.shape[0] <= 5
    assert np.array_equal(_expand_runs(runs), np.arange(2000))


def test_split_runs_at_gaps_splits_only_inside_runs():
    """``_split_runs_at_gaps`` splits a run only at a gap strictly inside it
    and is a no-op without gaps."""
    from spyglass.spikesorting.v2._artifacts.intervals import (
        _split_runs_at_gaps,
    )

    runs = np.array([[10, 20], [30, 35]], dtype=np.int64)
    # gap 15 inside [10,20] -> split; gap 20 at the run's trailing edge -> no
    # split; gap 25 between runs -> no split.
    gap_after = np.array([15, 20, 25], dtype=np.int64)
    assert _split_runs_at_gaps(runs, gap_after).tolist() == [
        [10, 15],
        [16, 20],
        [30, 35],
    ]
    assert np.array_equal(
        _split_runs_at_gaps(runs, np.array([], dtype=np.int64)), runs
    )


def test_artifact_job_kwargs_propagate_to_executor(dj_conn, monkeypatch):
    """Per-row ``job_kwargs`` reach ``ChunkRecordingExecutor``."""
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2._artifacts.intervals import (
        scan_artifact_frames,
    )

    rec = _synthetic_artifact_recording()
    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=50.0,
        zscore_threshold=None,
        proportion_above_threshold=0.5,
        removal_window_ms=1.0,
        join_window_ms=0.0,
        min_length_s=0.001,
    )

    seen = {}

    import spikeinterface.core.job_tools as jt

    real_executor = jt.ChunkRecordingExecutor

    class _SpyExecutor(real_executor):
        def __init__(self, *args, **kwargs):
            seen["n_jobs"] = kwargs.get("n_jobs")
            seen["chunk_duration"] = kwargs.get("chunk_duration")
            super().__init__(*args, **kwargs)

        def run(self, *args, **kwargs):
            # Only the constructor kwargs matter; a NumpyRecording cannot
            # go to a real process pool.
            return []

    # The scan imports ChunkRecordingExecutor at call time; patch the source.
    monkeypatch.setattr(jt, "ChunkRecordingExecutor", _SpyExecutor)

    scan_artifact_frames(
        rec,
        validated,
        job_kwargs={"n_jobs": 2, "chunk_duration": "0.5s"},
    )
    assert seen["n_jobs"] == 2, (
        "Row job_kwargs n_jobs=2 did not reach ChunkRecordingExecutor; the "
        "job_kwargs blob is still dead weight."
    )
    assert seen["chunk_duration"] == "0.5s"


def test_artifact_scan_chunked_by_default(dj_conn, monkeypatch):
    """With NO job_kwargs the scan still chunks (chunk_duration='1s').

    Without a chunk-size key ``ChunkRecordingExecutor`` reads the whole
    recording at once, so direct callers must still get bounded memory.
    """
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2._artifacts.intervals import (
        scan_artifact_frames,
    )

    rec = _synthetic_artifact_recording()
    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=50.0,
        zscore_threshold=None,
        proportion_above_threshold=0.5,
        removal_window_ms=1.0,
        join_window_ms=0.0,
        min_length_s=0.001,
    )

    seen = {}

    import spikeinterface.core.job_tools as jt

    real_executor = jt.ChunkRecordingExecutor

    class _SpyExecutor(real_executor):
        def __init__(self, *args, **kwargs):
            seen["chunk_duration"] = kwargs.get("chunk_duration")
            seen["chunk_size"] = kwargs.get("chunk_size")
            super().__init__(*args, **kwargs)

    # The scan imports ChunkRecordingExecutor at call time; patch the source.
    monkeypatch.setattr(jt, "ChunkRecordingExecutor", _SpyExecutor)

    scan_artifact_frames(rec, validated)
    assert seen["chunk_duration"] == "1s", (
        "default scan did not chunk: ChunkRecordingExecutor got "
        f"chunk_duration={seen.get('chunk_duration')!r} / "
        f"chunk_size={seen.get('chunk_size')!r}, so it would load the whole "
        "recording in one chunk (the memory cost chunking avoids)."
    )


@pytest.mark.slow
def test_artifact_scan_multiprocess_worker_path_runs(dj_conn, tmp_path):
    """The n_jobs>1 worker path actually EXECUTES and matches n_jobs=1.

    Workers rehydrate the recording from ``to_dict()`` with SI 0.104's
    ``si.load``.
    """

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2._artifacts.intervals import (
        scan_artifact_frames,
    )

    # Save to disk: a NumpyRecording is not dict-serializable.
    saved = _synthetic_artifact_recording().save(folder=tmp_path / "rec")
    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=50.0,
        zscore_threshold=None,
        proportion_above_threshold=0.5,
        removal_window_ms=1.0,
        join_window_ms=0.0,
        min_length_s=0.001,
    )

    single = scan_artifact_frames(
        saved, validated, job_kwargs={"n_jobs": 1, "chunk_duration": "0.5s"}
    )
    multi = scan_artifact_frames(
        saved, validated, job_kwargs={"n_jobs": 2, "chunk_duration": "0.5s"}
    )
    # Non-vacuous: the bursts are detected.
    assert len(single) > 0
    assert np.array_equal(single, multi), (
        "n_jobs=2 worker path produced different frames than n_jobs=1 (or "
        "failed to rehydrate the to_dict() recording via si.load)."
    )


@pytest.mark.slow
def test_artifact_detection_peak_memory_bounded_by_chunk_size(
    dj_conn, tmp_path
):
    """Peak memory of the chunked scan is bounded by chunk size, not by
    the full recording length.

    A 5-minute, 32-channel recording (~4.6 GB working set in memory),
    scanned in 1 s chunks, must peak under 256 MB in ``tracemalloc``.
    """
    import tracemalloc

    import spikeinterface as si

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2._artifacts.intervals import (
        scan_artifact_frames,
    )

    fs = 30_000.0
    n_channels = 32
    n_samples = int(fs * 300)  # 5 minutes
    # Memmap-backed traces so building the fixture itself does not dominate
    # the measured peak; the scan reads it chunk by chunk.
    arr = np.memmap(
        tmp_path / "memory_probe.raw",
        dtype=np.float32,
        mode="w+",
        shape=(n_samples, n_channels),
    )
    arr[:] = 0.0
    arr[1_000_000:1_000_300, :] = 300.0  # one planted burst
    arr.flush()
    rec = si.NumpyRecording(traces_list=[arr], sampling_frequency=fs)
    rec.set_channel_gains([1.0] * n_channels)

    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=50.0,
        zscore_threshold=None,
        proportion_above_threshold=0.5,
        removal_window_ms=1.0,
        join_window_ms=0.0,
        min_length_s=0.001,
    )

    tracemalloc.start()
    try:
        runs = scan_artifact_frames(
            rec, validated, job_kwargs={"n_jobs": 1, "chunk_duration": "1s"}
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    np.testing.assert_array_equal(
        _expand_runs(runs), np.arange(1_000_000, 1_000_300)
    )

    full_working_set = 4 * n_samples * n_channels * 4  # ~4.6 GB
    ceiling = 256 * 1024 * 1024  # 256 MB
    assert peak < ceiling, (
        f"Chunked scan peaked at {peak / 1e6:.1f} MB, above the "
        f"{ceiling / 1e6:.0f} MB ceiling; chunking is not bounding memory "
        f"(full-recording working set would be ~{full_working_set / 1e9:.1f} "
        "GB)."
    )


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_empty_valid_times_raises():
    """Empty valid_times raises instead of silently zeroing all.

    ``EmptyArtifactValidTimesError`` names the ``artifact_detection_id``
    and ``recording_id``.
    """
    import numpy as np_mod
    import spikeinterface.core as sc

    from spyglass.spikesorting.v2.exceptions import (
        EmptyArtifactValidTimesError,
    )
    from spyglass.spikesorting.v2._sorting.artifact_mask import (
        apply_artifact_mask,
    )

    rec = sc.generate_recording(
        num_channels=4, durations=[0.5], sampling_frequency=30000.0
    )
    with pytest.raises(EmptyArtifactValidTimesError) as excinfo:
        apply_artifact_mask(
            rec,
            np_mod.zeros((0, 2)),
            artifact_detection_id="art-123",
            recording_id="rec-456",
        )
    msg = str(excinfo.value)
    assert "art-123" in msg and "rec-456" in msg

    # Raised before any masking.
    assert not np_mod.array_equal(
        rec.get_traces(), np_mod.zeros_like(rec.get_traces())
    )


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_unsorted_valid_times_raises():
    """Unsorted / overlapping valid_times is rejected (strict input).

    Otherwise the complement walker would silently under-mask.
    """
    import numpy as np_mod
    import spikeinterface.core as sc

    from spyglass.spikesorting.v2._sorting.artifact_mask import (
        apply_artifact_mask,
    )

    rec = sc.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=30000.0
    )

    # Unsorted by start time.
    with pytest.raises(ValueError, match="sorted by start"):
        apply_artifact_mask(rec, np_mod.array([[0.6, 0.8], [0.1, 0.3]]))
    # Sorted by start but overlapping.
    with pytest.raises(ValueError, match="non-overlapping"):
        apply_artifact_mask(rec, np_mod.array([[0.1, 0.5], [0.3, 0.8]]))
    masked = apply_artifact_mask(rec, np_mod.array([[0.1, 0.9]]))
    assert masked.get_num_samples() == rec.get_num_samples()


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_full_coverage_short_circuits():
    """When ``valid_times`` covers the whole recording, the mask is a
    no-op and the original recording object is returned unmodified.
    """
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2._sorting.artifact_mask import (
        apply_artifact_mask,
    )

    rec = si.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=30_000.0
    )
    ts = rec.get_times()

    full = np.asarray([[ts[0], ts[-1]]])
    out_full = apply_artifact_mask(rec, full)
    assert out_full is rec, (
        "full-coverage valid_times must short-circuit to the original "
        "recording (no remove_artifacts wrapper)"
    )

    # Contrast: an artifact gap returns a wrapped recording.
    partial = np.asarray([[ts[0], ts[len(ts) // 2]]])
    out_partial = apply_artifact_mask(rec, partial)
    assert out_partial is not rec


# --------------------------------------------------------------------------- #
# Gain-aware µV artifact detection: peaks at a non-unity gain, thresholds that
# straddle the converted value.
# --------------------------------------------------------------------------- #


def _planted_recording(gain=0.195, n_ch=4, n_frames=200):
    """SI recording: zeros + a 3000-count peak (f50) and 2000-count (f100).

    At ``gain = 0.195`` µV/count the peaks are 585 µV and 390 µV. Returns
    the recording with that gain stored on every channel.
    """
    import spikeinterface as si

    traces = np.zeros((n_frames, n_ch), dtype=np.float32)
    traces[50, :] = 3000.0  # -> 585 µV at gain 0.195
    traces[100, :] = 2000.0  # -> 390 µV at gain 0.195
    rec = si.NumpyRecording([traces], sampling_frequency=30000.0)
    rec.set_channel_gains([gain] * n_ch)
    rec.set_channel_offsets([0.0] * n_ch)
    return rec, n_frames


def _flagged_frames(rec, n_frames, amplitude_threshold_uv):
    """Run the amplitude-only chunk detector over the whole recording.

    Every channel must exceed the threshold; z-score is off.
    """
    from spyglass.spikesorting.v2._artifacts.compute import (
        _compute_artifact_chunk,
        _init_artifact_worker,
    )

    ctx = _init_artifact_worker(
        rec,
        zscore_threshold=None,
        amplitude_threshold_uv=amplitude_threshold_uv,
        proportion_above_threshold=1.0,
    )
    runs = _compute_artifact_chunk(
        segment_index=0,
        start_frame=0,
        end_frame=n_frames,
        worker_ctx=ctx,
    )
    # Expand (start, end_inclusive) runs to frames.
    return _expand_runs(runs)


@pytest.mark.usefixtures("dj_conn")
def test_gain_conversion_detects_peak_above_uv_threshold():
    """Threshold 500 µV: only the 585 µV peak (frame 50) is flagged.

    The 390 µV bump (frame 100) is below threshold. Exact frame set.
    """
    rec, n = _planted_recording(gain=0.195)
    flagged = _flagged_frames(rec, n, amplitude_threshold_uv=500.0)
    np.testing.assert_array_equal(flagged, np.array([50]))


@pytest.mark.usefixtures("dj_conn")
def test_gain_conversion_no_detect_when_uv_below_threshold():
    """Threshold 700 µV: nothing is flagged (585 µV after gain, 3000 counts
    before)."""
    rec, n = _planted_recording(gain=0.195)
    flagged = _flagged_frames(rec, n, amplitude_threshold_uv=700.0)
    assert flagged.tolist() == [], (
        "frame 50 (3000 counts = 585 µV) must not exceed a 700 µV threshold; "
        f"got {flagged.tolist()} -- gain conversion was not applied."
    )


@pytest.mark.usefixtures("dj_conn")
def test_gain_conversion_low_threshold_flags_both_peaks():
    """Threshold 300 µV: both the 585 µV and 390 µV peaks are flagged."""
    rec, n = _planted_recording(gain=0.195)
    flagged = _flagged_frames(rec, n, amplitude_threshold_uv=300.0)
    np.testing.assert_array_equal(flagged, np.array([50, 100]))


# --------------------------------------------------------------------------- #
# Non-finite traces raise; a NaN compares False against every threshold.
# --------------------------------------------------------------------------- #


def test_compute_artifact_chunk_raises_on_single_nan_sample():
    """One NaN sample (on one channel) makes the chunk detector raise a
    ``ValueError`` naming the chunk's frame range and that channel's
    non-finite count, instead of silently returning zero flagged runs."""
    from spyglass.spikesorting.v2._artifacts.compute import (
        _compute_artifact_chunk,
        _init_artifact_worker,
    )

    traces = np.zeros((100, 3), dtype="float32")
    traces[42, 1] = np.nan  # channel 1 only
    rec = _rec(traces)
    ctx = _init_artifact_worker(
        rec,
        zscore_threshold=None,
        amplitude_threshold_uv=1000.0,
        proportion_above_threshold=1.0,
    )
    with pytest.raises(ValueError) as excinfo:
        _compute_artifact_chunk(
            segment_index=0, start_frame=0, end_frame=100, worker_ctx=ctx
        )
    msg = str(excinfo.value)
    assert "segment 0" in msg, msg
    assert "[0, 100)" in msg, msg
    assert "channel 1: 1" in msg, msg
    # Channels with zero non-finite samples are not named.
    assert "channel 0:" not in msg and "channel 2:" not in msg, msg


def test_compute_artifact_chunk_raises_on_all_nan_chunk():
    """An all-NaN chunk raises with every channel's non-finite count."""
    from spyglass.spikesorting.v2._artifacts.compute import (
        _compute_artifact_chunk,
        _init_artifact_worker,
    )

    traces = np.full((50, 2), np.nan, dtype="float32")
    rec = _rec(traces)
    ctx = _init_artifact_worker(
        rec,
        zscore_threshold=None,
        amplitude_threshold_uv=1000.0,
        proportion_above_threshold=1.0,
    )
    with pytest.raises(ValueError) as excinfo:
        _compute_artifact_chunk(
            segment_index=0, start_frame=0, end_frame=50, worker_ctx=ctx
        )
    msg = str(excinfo.value)
    assert "[0, 50)" in msg, msg
    assert "channel 0: 50" in msg, msg
    assert "channel 1: 50" in msg, msg


def test_compute_artifact_chunk_finite_recording_unaffected():
    """A fully-finite chunk is unaffected by the finiteness check."""
    from spyglass.spikesorting.v2._artifacts.compute import (
        _compute_artifact_chunk,
        _init_artifact_worker,
    )

    traces = np.zeros((100, 2), dtype="float32")
    traces[10, :] = 5000.0
    rec = _rec(traces)
    ctx = _init_artifact_worker(
        rec,
        zscore_threshold=None,
        amplitude_threshold_uv=1000.0,
        proportion_above_threshold=1.0,
    )
    runs = _compute_artifact_chunk(
        segment_index=0, start_frame=0, end_frame=100, worker_ctx=ctx
    )
    assert runs.tolist() == [[10, 10]]


def test_scan_artifact_frames_raises_on_nan_at_n_jobs_1():
    """The finiteness ``ValueError`` reaches the caller through
    ``scan_artifact_frames`` at ``n_jobs=1``."""
    from spyglass.spikesorting.v2._artifacts.intervals import (
        scan_artifact_frames,
    )

    traces = np.zeros((3000, 2), dtype="float32")
    traces[1500, 0] = np.nan
    rec = _rec(traces)
    validated = _artifact_params(amplitude_threshold_uv=1000.0)
    with pytest.raises(ValueError, match=r"non-finite"):
        scan_artifact_frames(
            rec,
            validated,
            job_kwargs={"n_jobs": 1, "chunk_duration": "0.05s"},
        )


@pytest.mark.slow
def test_scan_artifact_frames_raises_on_nan_at_n_jobs_2_process_pool(
    tmp_path,
):
    """The same ``ValueError`` and message reach the caller from a real
    ``n_jobs=2`` process pool."""
    from spyglass.spikesorting.v2._artifacts.intervals import (
        scan_artifact_frames,
    )

    traces = np.zeros((3000, 2), dtype="float32")
    traces[1500, 0] = np.nan
    rec = _rec(traces)
    # Save to disk: a NumpyRecording is not dict-serializable.
    saved = rec.save(folder=tmp_path / "rec")
    validated = _artifact_params(amplitude_threshold_uv=1000.0)
    with pytest.raises(ValueError, match=r"non-finite"):
        scan_artifact_frames(
            saved,
            validated,
            job_kwargs={
                "n_jobs": 2,
                "pool_engine": "process",
                "chunk_duration": "0.05s",
            },
        )


# --------------------------------------------------------------------------- #
# Timestamp helpers read h5py-backed timestamps in slices, never the full
# float64 vector ``get_times()`` caches (~824 MB for 1 h @ 30 kHz).
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_timestamp_helpers_peak_memory_bounded_vs_get_times(tmp_path):
    """The chunked timestamp helpers peak far below ``get_times()`` on an
    h5py-backed recording; they never materialize the full timestamp
    vector."""
    import gc
    import tracemalloc
    from datetime import datetime, timezone

    import pynwb
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from pynwb.ecephys import ElectricalSeries

    from spyglass.spikesorting.v2._storage.nwb import read_recording_nwb
    from spyglass.spikesorting.v2._core.signal_math import (
        base_intervals_and_gaps,
        timestamp_fingerprint,
    )

    fs = 30000.0
    n = 4_000_000  # 32 MB float64 timestamp vector
    path = tmp_path / "lazy_rec.nwb"
    nwbf = pynwb.NWBFile("d", "id", datetime(2020, 1, 1, tzinfo=timezone.utc))
    dev = nwbf.create_device("probe")
    grp = nwbf.create_electrode_group("g", "d", "loc", dev)
    for _ in range(2):
        nwbf.add_electrode(
            x=0.0,
            y=0.0,
            z=0.0,
            imp=0.0,
            location="loc",
            filtering="f",
            group=grp,
            group_name="g",
        )
    region = nwbf.create_electrode_table_region([0, 1], "all")
    ts = np.arange(n, dtype=np.float64) / fs
    es = ElectricalSeries(
        name="es",
        data=H5DataIO(np.zeros((n, 2), np.float32), chunks=(32768, 2)),
        electrodes=region,
        timestamps=H5DataIO(ts, chunks=(32768,)),
    )
    nwbf.add_acquisition(es)
    with pynwb.NWBHDF5IO(path=str(path), mode="w") as io:
        io.write(nwbf)

    def _peak(op):
        # Include construction: the backend must not eagerly read timestamps
        # before the chunked helper even runs. Reopen to isolate SI's caches.
        gc.collect()
        tracemalloc.start()
        rec = read_recording_nwb(
            str(path),
            electrical_series_path="acquisition/es",
        )
        keep = op(rec)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del keep, rec
        return peak

    full_bytes = n * 8
    peak_get_times = _peak(lambda r: r.get_times())
    peak_base = _peak(lambda r: base_intervals_and_gaps(r, fs))
    peak_fp = _peak(lambda r: timestamp_fingerprint(r))

    assert peak_get_times > 0.5 * full_bytes, (
        f"get_times() should materialize ~{full_bytes} bytes; "
        f"peaked at {peak_get_times}"
    )
    assert peak_base < 0.25 * full_bytes, (
        f"base_intervals_and_gaps peaked at {peak_base} bytes -- not chunked? "
        f"(full vector is {full_bytes} bytes)"
    )
    assert peak_fp < 0.25 * full_bytes, (
        f"timestamp_fingerprint peaked at {peak_fp} bytes -- not chunked? "
        f"(full vector is {full_bytes} bytes)"
    )
