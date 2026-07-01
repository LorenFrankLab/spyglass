"""Artifact-mask input validation + artifact valid-times output structure.

Locks two contracts a silent error would corrupt:
* ``Sorting._apply_artifact_mask`` REJECTS malformed valid_times (empty,
  wrong-shape, end<start, unsorted/overlapping) instead of under-masking;
* ``ArtifactDetection._detect_artifacts`` returns valid_times that are
  start-sorted, non-overlapping, within the recording bounds, >= min_length_s,
  exclude the detected artifact, and never span an inter-chunk wall-clock gap
  (the disjoint case also proves a chunk-boundary artifact IS masked -- the L1
  edge is unreachable because removal_window_ms>0 widens every artifact).

Plus an L2 pin: the clamp-vs-raise boundary of ``_spike_times_to_frames``.
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
# A. _apply_artifact_mask input validation
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_rejects_empty_valid_times():
    from spyglass.spikesorting.v2.exceptions import (
        EmptyArtifactValidTimesError,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    rec = _rec(np.zeros((100, 2)))
    with pytest.raises(EmptyArtifactValidTimesError):
        Sorting._apply_artifact_mask(rec, np.empty((0, 2)))


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_rejects_wrong_shape():
    from spyglass.spikesorting.v2.sorting import Sorting

    rec = _rec(np.zeros((100, 2)))
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        Sorting._apply_artifact_mask(rec, np.array([0.0, 1.0, 2.0]))


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_rejects_end_before_start():
    from spyglass.spikesorting.v2.sorting import Sorting

    rec = _rec(np.zeros((100, 2)))
    with pytest.raises(ValueError, match="end precedes its start"):
        Sorting._apply_artifact_mask(
            rec, np.array([[0.0, 0.001], [0.005, 0.003]])
        )


@pytest.mark.usefixtures("dj_conn")
@pytest.mark.parametrize(
    "valid_times",
    [
        [[0.002, 0.003], [0.000, 0.001]],  # unsorted by start
        [[0.000, 0.002], [0.001, 0.003]],  # overlapping
    ],
)
def test_apply_artifact_mask_rejects_unsorted_or_overlapping(valid_times):
    from spyglass.spikesorting.v2.sorting import Sorting

    rec = _rec(np.zeros((100, 2)))
    with pytest.raises(ValueError, match="sorted by start"):
        Sorting._apply_artifact_mask(rec, np.array(valid_times))


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_zeros_complement_frames():
    """The mask zeros exactly the complement of ``valid_times`` (the artifact
    frames) and leaves the kept frames untouched. Pins the masking semantics so
    the interval-native silence_periods path zeros the same samples the prior
    per-frame remove_artifacts path did."""
    from spyglass.spikesorting.v2.sorting import Sorting

    fs = 30000.0
    rec = _rec(np.ones((10, 2), dtype="float32"), fs=fs)
    # Keep [0, 3/fs) and [6/fs, end] -> the artifact complement is frames 3,4,5.
    valid_times = np.array([[0.0, 3.0 / fs], [6.0 / fs, 1.0]])
    out = Sorting._apply_artifact_mask(rec, valid_times).get_traces()
    assert np.all(out[3:6] == 0.0), "artifact (complement) frames not zeroed"
    assert np.all(out[0:3] == 1.0), "kept frames before the artifact altered"
    assert np.all(out[6:10] == 1.0), "kept frames after the artifact altered"


# --------------------------------------------------------------------------- #
# B. _detect_artifacts output structure (contiguous)
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
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    fs = 30000.0
    n, n_ch = 120000, 4  # 4 s
    traces = np.zeros((n, n_ch), dtype="float32")
    art_frame = 60000  # ~2.0 s, interior so both sides exceed min_length
    traces[art_frame : art_frame + 10, :] = 5000.0  # 5000 uV >> 1000 thresh
    rec = _rec(traces, fs=fs)
    params = _artifact_params()

    vt = ArtifactDetection._detect_artifacts(rec, params, job_kwargs=None)
    times = rec.get_times()
    _assert_valid_times_well_formed(
        vt, times[0], times[-1], params.min_length_s
    )

    # The detected artifact (~2.0 s) is excluded from every valid interval.
    art_t = times[art_frame]
    covered = np.any((vt[:, 0] <= art_t) & (art_t <= vt[:, 1]))
    assert not covered, "the detected artifact must not be inside a valid time"


# --------------------------------------------------------------------------- #
# C. _detect_artifacts on a DISJOINT recording (L1-safe: chunk-boundary
#    artifact IS masked, valid_times never span the gap)
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("dj_conn")
def test_detect_artifacts_disjoint_masks_chunk_boundary_artifact():
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    fs = 30000.0
    per_chunk = 60000  # 2 s each
    # Two chunks with a 5 s wall-clock gap between them.
    t_a = np.arange(per_chunk) / fs
    t_b = 5.0 + np.arange(per_chunk) / fs
    times = np.concatenate([t_a, t_b])
    n = times.size
    traces = np.zeros((n, 4), dtype="float32")
    # Artifact on the LAST 10 frames of chunk A (ends exactly on the chunk's
    # final sample -- the L1 edge). With removal_window_ms>0 it is widened to
    # the left, so it stays multi-sample and is masked correctly.
    traces[per_chunk - 10 : per_chunk, :] = 5000.0
    rec = _rec(traces, fs=fs, times=times)
    params = _artifact_params()

    vt = ArtifactDetection._detect_artifacts(rec, params, job_kwargs=None)
    _assert_valid_times_well_formed(
        vt, times[0], times[-1], params.min_length_s
    )

    # The chunk-boundary artifact frame (chunk A's final sample) is excluded.
    art_t = times[per_chunk - 1]
    assert not np.any((vt[:, 0] <= art_t) & (art_t <= vt[:, 1])), (
        "a chunk-boundary artifact must be masked (L1 edge is unreachable "
        "with removal_window_ms>0)"
    )
    # No valid interval spans the inter-chunk gap [2 s, 5 s].
    spans_gap = np.any((vt[:, 0] < 2.5) & (vt[:, 1] > 4.5))
    assert not spans_gap, "valid_times must never span an inter-chunk gap"


# --------------------------------------------------------------------------- #
# E. interval recovery: artifacts planted at KNOWN times/amplitudes are removed
#    over [planted +/- removal_window/2], not merely "detection did not crash"
# --------------------------------------------------------------------------- #


def _removed_intervals(valid_times, t0, t_end):
    """Complement of ``valid_times`` within ``[t0, t_end]``.

    The removed (artifact) intervals, as an ``(n, 2)`` array -- exactly what
    ``Sorting._apply_artifact_mask`` excises before a sort. ``valid_times`` is
    assumed start-sorted and disjoint (the detector's contract, checked
    separately by ``_assert_valid_times_well_formed``).
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

    The interval-recovery counterpart to the frame-level gain tests
    (``test_gain_conversion_*`` in this module): it pins detected-vs-planted at
    the *seconds*
    level, so a regression in the removal-window expansion, the join logic, or
    the valid_times complement that mislocates, splits, merges, or over-/under-
    removes an artifact is caught -- not just "populate did not crash".
    """
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    fs = 30000.0
    n, n_ch = 150000, 4  # 5.0 s
    removal_window_ms = 1.0
    traces = np.zeros((n, n_ch), dtype="float32")
    # Two 10-frame artifacts ~2 s apart (>> the join window) at a known
    # amplitude (5000 uV >> the 1000 uV threshold, so every channel is
    # flagged). Inclusive frame spans [a0, a1].
    planted_frames = [(45000, 45009), (105000, 105009)]  # ~1.5 s and ~3.5 s
    for a0, a1 in planted_frames:
        traces[a0 : a1 + 1, :] = 5000.0
    rec = _rec(traces, fs=fs)
    params = _artifact_params(removal_window_ms=removal_window_ms)

    vt = ArtifactDetection._detect_artifacts(rec, params, job_kwargs=None)
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

    # Each removed interval brackets its planted epoch, widened by ~half the
    # removal window on each side (the documented expansion). Expressed in
    # physical units (seconds) from the planted frames and the param -- NOT by
    # re-running the detector's frame math -- so the assertion is independent
    # of the implementation under test.
    half_window_s = removal_window_ms * 1e-3 / 2.0
    sample = 1.0 / fs
    tol = 3 * sample  # grid snapping + the half-open [start, end) +1 sample
    for (a0, a1), (rem_start, rem_end) in zip(planted_frames, removed):
        planted_start, planted_end = times[a0], times[a1]
        # The entire planted artifact is inside the removed interval.
        assert rem_start <= planted_start and rem_end >= planted_end, (
            f"planted [{planted_start:.5f}, {planted_end:.5f}] not contained "
            f"in removed [{rem_start:.5f}, {rem_end:.5f}]"
        )
        # ...and the removal is TIGHT (half a window each side), not a giant
        # over-removed chunk -- localization is correct.
        assert rem_start == pytest.approx(
            planted_start - half_window_s, abs=tol
        )
        assert rem_end == pytest.approx(planted_end + half_window_s, abs=tol)


# --------------------------------------------------------------------------- #
# D. L2 pin: _spike_times_to_frames clamp-vs-raise boundary
# --------------------------------------------------------------------------- #


def test_spike_times_to_frames_clamp_vs_raise_boundary():
    """Within ~2 sample periods past the end -> clamp (FP rounding); beyond ->
    raise (genuine alignment/units error). Pins the 2*dt tolerance."""
    from spyglass.spikesorting.v2.utils import _spike_times_to_frames

    fs = 30000.0
    timestamps = np.arange(100) / fs
    n = timestamps.size
    dt = 1.0 / fs

    # 1.5 sample periods past -> within tolerance -> clamp to last frame.
    frames = _spike_times_to_frames(
        timestamps, np.array([timestamps[50], timestamps[-1] + 1.5 * dt]), n, 1
    )
    assert frames.tolist() == [50, 99], f"within-tol must clamp; got {frames}"

    # 3 sample periods past -> beyond tolerance -> raise.
    with pytest.raises(ValueError, match="alignment/units error"):
        _spike_times_to_frames(
            timestamps, np.array([timestamps[-1] + 3.0 * dt]), n, 1
        )


# --------------------------------------------------------------------------- #
# E. detect_artifacts degenerate-configuration guards
# --------------------------------------------------------------------------- #


def test_detect_artifacts_warns_when_min_length_drops_all_valid_time(caplog):
    """A short recording whose only valid slivers fall below ``min_length_s``
    returns empty ``valid_times`` -- but must WARN at detection time (mirroring
    the zero-artifacts warning), so the operator sees the cause here rather
    than three stages later as an ``EmptyArtifactValidTimesError`` at sort
    time."""
    from spyglass.spikesorting.v2._artifact_intervals import detect_artifacts

    fs = 30000.0
    n, n_ch = 15000, 2  # 0.5 s -- every kept sliver is < min_length_s=1.0
    traces = np.zeros((n, n_ch), dtype="float32")
    traces[n // 2, :] = 5000.0  # one artifact frame, both channels
    rec = _rec(traces, fs=fs)
    params = _artifact_params(min_length_s=1.0)  # amplitude_threshold_uv=1000

    with caplog.at_level("WARNING"):
        vt = detect_artifacts(rec, params, context=" for unit-test")
    assert vt.shape == (0, 2)
    assert any("no valid time" in r.message.lower() for r in caplog.records), (
        "expected an empty-valid-times warning; got "
        f"{[r.message for r in caplog.records]}"
    )


def test_detect_artifacts_raises_on_single_channel_zscore_only():
    """The z-score is computed ACROSS channels within a frame; on a 1-channel
    group it is identically zero, so a z-score-only config would silently
    detect nothing. Raise instead of returning a misleadingly-clean window."""
    from spyglass.spikesorting.v2._artifact_intervals import detect_artifacts
    from spyglass.spikesorting.v2.exceptions import (
        InsufficientZScoreChannelsError,
    )

    rec = _rec(np.zeros((3000, 1), dtype="float32"))
    params = _artifact_params(amplitude_threshold_uv=None, zscore_threshold=3.0)
    with pytest.raises(InsufficientZScoreChannelsError):
        detect_artifacts(rec, params)


def test_detect_artifacts_raises_on_two_channel_zscore_only():
    """The across-channel z-score is also degenerate on TWO channels: for any
    two distinct values it is identically +/-1 regardless of amplitude, so a
    z-score-only config on a stereotrode flags every frame (threshold < 1) or
    none (threshold > 1) -- never amplitude-sensitively. Raise, same as the
    single-channel case."""
    from spyglass.spikesorting.v2._artifact_intervals import detect_artifacts
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
    from spyglass.spikesorting.v2._artifact_intervals import detect_artifacts

    rec = _rec(np.zeros((3000, 1), dtype="float32"))
    params = _artifact_params(
        amplitude_threshold_uv=1000.0, zscore_threshold=3.0
    )
    with caplog.at_level("WARNING"):
        detect_artifacts(rec, params)
    assert any("inert" in r.message.lower() for r in caplog.records)


def test_detect_artifacts_warns_proportion_rounds_to_all_channels(caplog):
    """On a 2-channel (stereotrode) group, ``proportion_above_threshold=0.7``
    rounds up via ``ceil`` to requiring BOTH channels -- silently stricter
    than the nominal 70%. Warn so the operator sees the realized
    requirement."""
    from spyglass.spikesorting.v2._artifact_intervals import detect_artifacts

    rec = _rec(np.zeros((3000, 2), dtype="float32"))
    params = _artifact_params(
        proportion_above_threshold=0.7, amplitude_threshold_uv=1000.0
    )
    with caplog.at_level("WARNING"):
        detect_artifacts(rec, params)
    assert any(
        "all" in r.message.lower() and "channel" in r.message.lower()
        for r in caplog.records
    )


def test_detect_artifacts_no_proportion_warning_on_tetrode(caplog):
    """On a tetrode (4 ch), ``0.7`` -> ``ceil(2.8)=3`` of 4 -- a genuine
    majority, not silently promoted to all-channels -- so no 'rounds up'
    warning fires."""
    from spyglass.spikesorting.v2._artifact_intervals import detect_artifacts

    rec = _rec(np.zeros((3000, 4), dtype="float32"))
    params = _artifact_params(
        proportion_above_threshold=0.7, amplitude_threshold_uv=1000.0
    )
    with caplog.at_level("WARNING"):
        detect_artifacts(rec, params)
    assert not any("rounds up" in r.message.lower() for r in caplog.records)


def test_detect_artifacts_raises_when_most_frames_flagged():
    """A misconfigured (too-loose) detector that flags more than half the
    recording raises ``ArtifactFractionExceededError`` BEFORE materializing a
    per-frame index for every flagged sample (and the slow per-frame join) --
    instead of allocating an O(n_samples) array on a pathological config."""
    from spyglass.spikesorting.v2._artifact_intervals import detect_artifacts
    from spyglass.spikesorting.v2.exceptions import (
        ArtifactFractionExceededError,
    )

    # Every sample sits far above a 1 uV amplitude threshold on both channels,
    # so the detector flags all 2000 frames (> 50% of the recording).
    rec = _rec(np.full((2000, 2), 1000.0, dtype="float32"))
    params = _artifact_params(
        amplitude_threshold_uv=1.0, proportion_above_threshold=1.0
    )
    with pytest.raises(ArtifactFractionExceededError, match="2000 of 2000"):
        detect_artifacts(rec, params, context=" for unit-test")


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_raises_when_valid_times_keep_almost_nothing():
    """A ``valid_times`` keeping only a sliver makes the artifact complement
    span most of the recording; the mask raises ``ArtifactFractionExceededError``
    BEFORE expanding the complement to a per-frame index array."""
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        apply_artifact_mask,
    )
    from spyglass.spikesorting.v2.exceptions import (
        ArtifactFractionExceededError,
    )

    rec = _rec(np.zeros((3000, 2), dtype="float32"))  # 0.1 s @ 30 kHz
    # Keep only the first ~1 ms; the complement (~99%) would expand to ~2970
    # artifact frames (> 50% of 3000).
    valid_times = np.array([[0.0, 0.001]])
    with pytest.raises(ArtifactFractionExceededError, match="%"):
        apply_artifact_mask(rec, valid_times)


# --------------------------------------------------------------------------- #
# F. Chunked artifact detection + the artifact-mask complement walker.
#
# Covers the chunked ``_scan_artifact_frames`` (frame-identical to the frozen
# in-memory reference, job_kwargs propagation, default chunking, multiprocess
# worker path, bounded peak memory) and ``_apply_artifact_mask`` input
# strictness (empty / unsorted valid_times raise; full coverage short-circuits).
# --------------------------------------------------------------------------- #


def _in_memory_artifact_frames_reference(recording, validated):
    """Frozen copy of the pre-port full-in-memory artifact-frame scan.

    This reproduces, verbatim, the per-frame detection math that lived in
    ``ArtifactDetection._detect_artifacts`` before the chunked port replaced
    the full-recording ``get_traces`` load with a chunked
    ``ChunkRecordingExecutor`` pass. It exists ONLY in the test suite as the
    equivalence oracle: the chunked port must produce frame-identical output
    to this reference on the same recording. It is intentionally NOT importable
    from production -- the in-memory path is deleted there, not kept as a
    fallback.

    Returns the ascending ndarray of flagged frame indices (the
    ``frames_above`` array the interval-building code consumes).
    """
    traces = recording.get_traces(return_in_uV=False)
    gains = recording.get_channel_gains()
    traces_uv = traces.astype(np.float32) * gains[None, :]
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


@pytest.mark.parametrize(
    "amplitude_threshold_uv,zscore_threshold",
    [
        (50.0, None),  # amplitude-only branch
        (None, 6.0),  # z-score-only branch
        (50.0, 6.0),  # OR-combined branch
    ],
)
def test_chunked_artifact_matches_in_memory_reference(
    dj_conn, amplitude_threshold_uv, zscore_threshold
):
    """The chunked ``_scan_artifact_frames`` produces frame-identical
    output to the frozen full-in-memory reference, and is invariant to chunk
    boundaries.

    This is the gating equivalence evidence: the chunked port replaced the
    single full-recording ``get_traces`` call with a per-chunk
    ``ChunkRecordingExecutor`` pass to bound peak memory. The port is correct
    only if the flagged frame set is unchanged. The per-frame across-channel
    z-score depends solely on that frame's columns, so chunk boundaries
    (which split the time axis) cannot change which frames are flagged --
    this test pins that property across all three detection branches.
    """
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    rec = _synthetic_artifact_recording()
    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=amplitude_threshold_uv,
        zscore_threshold=zscore_threshold,
        proportion_above_threshold=0.5,
        removal_window_ms=1.0,
        join_window_ms=0.0,
        min_length_s=0.001,
    )

    reference = _in_memory_artifact_frames_reference(rec, validated)

    # Many small chunks (~0.1 s each → ~30 chunks) exercises chunk seams.
    chunked = ArtifactDetection._scan_artifact_frames(
        rec, validated, job_kwargs={"n_jobs": 1, "chunk_duration": "0.1s"}
    )
    # A single chunk spanning the whole recording == the in-memory path.
    single = ArtifactDetection._scan_artifact_frames(
        rec, validated, job_kwargs={"n_jobs": 1, "chunk_size": 90_000}
    )

    # scan returns run RANGES; expand them to the per-frame set the reference
    # holds. Equivalence is "the runs cover exactly the flagged frames", across
    # chunk seams (chunked) and as a single chunk (single).
    assert np.array_equal(_expand_runs(chunked), reference), (
        "Chunked artifact runs diverge from the in-memory reference; the "
        "ChunkRecordingExecutor port changed which frames are flagged."
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
    """``scan_artifact_frames`` returns run-length ranges, not one index per
    flagged frame, so the reviewer's "large but under the 50% guard" case (a
    heavily-but-contiguously flagged recording) stays O(n_runs) in memory
    instead of O(n_bad_frames). Here 40% of the recording is flagged in one
    contiguous block -> a couple of per-chunk runs (the cross-chunk join happens
    in detect_artifacts), and expanding them recovers exactly the flagged
    frames."""
    from spyglass.spikesorting.v2._artifact_intervals import (
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
    # (n_runs, 2) ranges, NOT a 2000-long per-frame array.
    assert runs.ndim == 2 and runs.shape[1] == 2
    # 2000 contiguous flagged frames across ~2 chunks -> a couple of runs.
    assert runs.shape[0] <= 5
    # The runs cover exactly the flagged frames (0..1999).
    assert np.array_equal(_expand_runs(runs), np.arange(2000))


def test_split_runs_at_gaps_splits_only_inside_runs():
    """A fixed-size chunk can straddle a wall-clock gap, so a flagged run may
    span one. ``_split_runs_at_gaps`` splits a run at a gap STRICTLY inside it,
    leaves a gap at the trailing edge and a gap between runs alone, and is a
    no-op when there are no gaps -- so the joined spans match the old per-frame
    gap-aware behavior."""
    from spyglass.spikesorting.v2._artifact_intervals import (
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
    # No gaps -> unchanged.
    assert np.array_equal(
        _split_runs_at_gaps(runs, np.array([], dtype=np.int64)), runs
    )


def test_artifact_job_kwargs_propagate_to_executor(dj_conn, monkeypatch):
    """Per-row ``job_kwargs`` reach ``ChunkRecordingExecutor``.

    The stored ``job_kwargs`` blob was dead weight under the in-memory scan.
    The chunked port wires it through ``_resolved_job_kwargs`` into the
    executor. This test patches the executor constructor and asserts an
    ``n_jobs=2`` override (a recognized SI job key) is observed.
    """
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

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
            # The test asserts the constructor received the row's
            # job_kwargs; it does not need the worker pool to execute (an
            # in-memory NumpyRecording cannot round-trip through
            # ``to_dict()``/``load_extractor`` for a real multi-process pool).
            return []

    # ``scan_artifact_frames`` (now in ``_artifact_intervals``) lazy-imports
    # ``ChunkRecordingExecutor`` from its SI source at call time, so patch the
    # source module -- the ``from spikeinterface.core.job_tools import
    # ChunkRecordingExecutor`` inside the scan binds this spy at call time.
    monkeypatch.setattr(jt, "ChunkRecordingExecutor", _SpyExecutor)

    ArtifactDetection._scan_artifact_frames(
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

    ChunkRecordingExecutor with no chunk-size key processes the whole
    recording in a single chunk -- the exact full-traces memory profile the
    restoration removed. Any direct ``_detect_artifacts`` caller (not just the
    production ``make_compute`` path that merges SI's global job kwargs) must
    still be bounded, so ``_scan_artifact_frames`` defaults to a 1 s chunk.
    """
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

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

    # ``scan_artifact_frames`` (now in ``_artifact_intervals``) lazy-imports
    # ``ChunkRecordingExecutor`` from its SI source at call time, so patch the
    # source module -- the ``from spikeinterface.core.job_tools import
    # ChunkRecordingExecutor`` inside the scan binds this spy at call time.
    monkeypatch.setattr(jt, "ChunkRecordingExecutor", _SpyExecutor)

    # No job_kwargs at all -- the bare default path.
    ArtifactDetection._scan_artifact_frames(rec, validated)
    assert seen["chunk_duration"] == "1s", (
        "default scan did not chunk: ChunkRecordingExecutor got "
        f"chunk_duration={seen.get('chunk_duration')!r} / "
        f"chunk_size={seen.get('chunk_size')!r}, so it would load the whole "
        "recording in one chunk (the memory trap chunking removed)."
    )


@pytest.mark.slow
def test_artifact_scan_multiprocess_worker_path_runs(dj_conn, tmp_path):
    """The n_jobs>1 worker path actually EXECUTES and matches n_jobs=1.

    For n_jobs>1 the recording is passed to each worker as a ``to_dict()`` blob
    and re-hydrated inside ``_init_artifact_worker``. v1 used SI 0.99's
    ``si.load_extractor``; SI 0.104 renamed it to ``si.load``, so the worker
    would ``AttributeError`` if it still called the old name. The propagation
    test only inspects constructor kwargs and returns before workers run, so it
    cannot catch this. This test runs the multiprocess path end-to-end on a
    serializable (saved-to-disk binary) recording and asserts the flagged
    frames equal the n_jobs=1 result -- proving the worker rehydrates and the
    chunked output is worker-count-invariant.
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    # _synthetic_artifact_recording is an in-memory NumpyRecording (not
    # dict-serializable); save it to a binary folder so to_dict()/si.load()
    # round-trips, the way a real cached preprocessed Recording does.
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

    single = ArtifactDetection._scan_artifact_frames(
        saved, validated, job_kwargs={"n_jobs": 1, "chunk_duration": "0.5s"}
    )
    multi = ArtifactDetection._scan_artifact_frames(
        saved, validated, job_kwargs={"n_jobs": 2, "chunk_duration": "0.5s"}
    )
    # The planted bursts must actually be detected, else "equal" is vacuous.
    assert len(single) > 0
    assert np.array_equal(single, multi), (
        "n_jobs=2 worker path produced different frames than n_jobs=1 (or "
        "failed to rehydrate the to_dict() recording via si.load)."
    )


@pytest.mark.slow
def test_artifact_detection_peak_memory_bounded_by_chunk_size(dj_conn):
    """Peak memory of the chunked scan is bounded by chunk size, not by
    the full recording length.

    Builds a 5-minute, 32-channel recording (~1.15 GB if materialized in
    float32 once; ~4.6 GB at the in-memory path's ~4× working-set factor) and
    scans it with a 1-second chunk. Peak *additional* allocation measured by
    ``tracemalloc`` must stay far below the full-recording working set -- it is
    bounded by ``~4 × chunk_frames × n_channels × 4 bytes`` plus the flagged
    frame-index arrays. Asserts the peak stays under a generous 256 MB ceiling
    that the full-recording load would blow through.
    """
    import tracemalloc

    import spikeinterface as si

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    fs = 30_000.0
    n_channels = 32
    n_samples = int(fs * 300)  # 5 minutes
    # Memmap-backed traces so building the fixture itself does not dominate
    # the measured peak; the scan reads it chunk by chunk.
    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
    tmp.close()
    arr = np.memmap(
        tmp.name, dtype=np.float32, mode="w+", shape=(n_samples, n_channels)
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
    ArtifactDetection._scan_artifact_frames(
        rec, validated, job_kwargs={"n_jobs": 1, "chunk_duration": "1s"}
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

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

    When the artifact pass keeps zero seconds, the complement walker
    would mask the WHOLE recording to zeros and the sort would run over
    all-zeros, emitting a misleading "zero units" result. The fix raises
    ``EmptyArtifactValidTimesError`` naming the ``artifact_detection_id`` and
    ``recording_id`` so the user re-runs detection or overrides the
    selection -- a loud failure, not a silent blanked recording.
    """
    import numpy as np_mod
    import spikeinterface.core as sc

    from spyglass.spikesorting.v2.exceptions import (
        EmptyArtifactValidTimesError,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    rec = sc.generate_recording(
        num_channels=4, durations=[0.5], sampling_frequency=30000.0
    )
    with pytest.raises(EmptyArtifactValidTimesError) as excinfo:
        Sorting._apply_artifact_mask(
            rec,
            np_mod.zeros((0, 2)),
            artifact_detection_id="art-123",
            recording_id="rec-456",
        )
    msg = str(excinfo.value)
    assert "art-123" in msg and "rec-456" in msg

    # The recording is NOT zeroed -- the raise happens before any masking.
    assert not np_mod.array_equal(
        rec.get_traces(), np_mod.zeros_like(rec.get_traces())
    )


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_unsorted_valid_times_raises():
    """Unsorted / overlapping valid_times is rejected (strict input).

    The complement walker assumes monotonic, disjoint intervals; an
    unsorted or overlapping list would silently under-mask (leave
    artifact frames in the recording). This chooses strict input +
    a clear raise rather than silently sorting/merging.
    """
    import numpy as np_mod
    import spikeinterface.core as sc

    from spyglass.spikesorting.v2.sorting import Sorting

    rec = sc.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=30000.0
    )

    # Unsorted by start time.
    with pytest.raises(ValueError, match="sorted by start"):
        Sorting._apply_artifact_mask(
            rec, np_mod.array([[0.6, 0.8], [0.1, 0.3]])
        )
    # Sorted by start but overlapping.
    with pytest.raises(ValueError, match="non-overlapping"):
        Sorting._apply_artifact_mask(
            rec, np_mod.array([[0.1, 0.5], [0.3, 0.8]])
        )
    # A single well-formed interval still masks normally (no raise).
    masked = Sorting._apply_artifact_mask(rec, np_mod.array([[0.1, 0.9]]))
    assert masked.get_num_samples() == rec.get_num_samples()


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_full_coverage_short_circuits():
    """When ``valid_times`` covers the whole recording, the mask is a
    no-op and the original recording object is returned unmodified.

    ``frame_ranges`` ends up empty (no artifact gaps), so ``_apply_artifact_
    mask`` returns ``recording`` itself -- no ``remove_artifacts`` wrapper.
    The contrast call (valid_times dropping the second half) returns a
    DIFFERENT object, proving the short-circuit is meaningful.
    """
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2.sorting import Sorting

    rec = si.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=30_000.0
    )
    ts = rec.get_times()

    full = np.asarray([[ts[0], ts[-1]]])
    out_full = Sorting._apply_artifact_mask(rec, full)
    assert out_full is rec, (
        "full-coverage valid_times must short-circuit to the original "
        "recording (no remove_artifacts wrapper)"
    )

    # Contrast: dropping the second half leaves an artifact gap -> a wrapped
    # recording, not the original object.
    partial = np.asarray([[ts[0], ts[len(ts) // 2]]])
    out_partial = Sorting._apply_artifact_mask(rec, partial)
    assert out_partial is not rec


# --------------------------------------------------------------------------- #
# G. Gain-aware µV artifact detection (hermetic absolute-outcome).
#
# ``artifact._compute_artifact_chunk`` scales raw counts to microvolts with
# the recording's stored per-channel gains (``traces_uv = traces * gains``)
# before comparing against ``amplitude_threshold_uv``. Every other direct
# artifact test uses ``gain = 1.0``, where counts and µV are numerically
# identical, so the conversion itself is never exercised. These tests plant a
# peak at a KNOWN non-unity gain and assert the absolute detect / no-detect
# outcome at thresholds that straddle the converted value -- a result that
# only holds if the gain is actually applied. They request ``dj_conn`` because
# importing ``artifact`` declares its ``@schema`` tables (a live connection),
# even though the detection math is pure / synthetic; they are not ``slow``.
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

    ``proportion_above_threshold = 1.0`` requires every channel to exceed the
    threshold; ``zscore_threshold = None`` disables the z-score detector so the
    outcome is governed purely by the µV amplitude comparison.
    """
    from spyglass.spikesorting.v2.artifact import (
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
    # The worker returns (start, end_inclusive) runs; expand to the per-frame
    # set these gain-conversion tests assert on.
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
    """Threshold 700 µV: NOTHING is flagged -- proves the gain is applied.

    The peak is 3000 counts -> 585 µV < 700, so no frame is flagged. If the
    conversion were skipped (counts compared directly), 3000 > 700 would
    wrongly flag frame 50. The empty result is the gain-correctness signal.
    """
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
# H. The timestamp helpers stay LAZY on an explicit (h5py-backed) recording.
#
# ``Recording.get_recording`` loads the cached preprocessed NWB with
# ``load_time_vector=True`` -- the timestamps are a lazy h5py object until a
# consumer forces them. The prior artifact path called ``recording.get_times()``
# which materializes (and caches) the whole float64 vector (8 bytes/sample, ~824
# MB for 1 h @ 30 kHz). ``base_intervals_and_gaps`` / ``timestamp_fingerprint``
# read it in ~1 s slices via ``sample_index_to_time`` instead. This guards that
# they never trigger the full materialization -- if a future SpikeInterface made
# ``sample_index_to_time`` eager, this fails loudly instead of silently
# regressing peak memory.
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_timestamp_helpers_peak_memory_bounded_vs_get_times(tmp_path):
    """The chunked timestamp helpers peak far below ``get_times()`` on an
    explicit (h5py-backed) recording -- they never materialize the full
    ``n_samples`` float64 vector."""
    import gc
    import tracemalloc
    from datetime import datetime, timezone

    import pynwb
    import spikeinterface.extractors as se
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from pynwb.ecephys import ElectricalSeries

    from spyglass.spikesorting.v2._signal_math import (
        base_intervals_and_gaps,
        timestamp_fingerprint,
    )

    fs = 30000.0
    n = 4_000_000  # 32 MB float64 timestamp vector -- clear materialize signal
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
        # recording built OUTSIDE the window (production: get_recording builds
        # it once), re-read per op so get_times()'s caching cannot pollute.
        rec = se.read_nwb_recording(
            str(path),
            electrical_series_path="acquisition/es",
            load_time_vector=True,
        )
        gc.collect()
        tracemalloc.start()
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
