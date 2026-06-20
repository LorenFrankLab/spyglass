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

    The interval-recovery counterpart to the frame-level gain test
    (``test_artifact_gain.py``): it pins detected-vs-planted at the *seconds*
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
    from spyglass.spikesorting.v2.exceptions import SingleChannelZScoreError

    rec = _rec(np.zeros((3000, 1), dtype="float32"))
    params = _artifact_params(amplitude_threshold_uv=None, zscore_threshold=3.0)
    with pytest.raises(SingleChannelZScoreError):
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
