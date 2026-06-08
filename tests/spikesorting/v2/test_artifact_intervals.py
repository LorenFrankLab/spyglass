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
        amplitude_thresh_uV=1000.0,
        zscore_thresh=None,
        proportion_above_thresh=1.0,
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
