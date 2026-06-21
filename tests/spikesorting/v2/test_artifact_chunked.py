"""Chunked artifact detection and the artifact-mask complement walker.

Covers the chunked ``_scan_artifact_frames`` (frame-identical to the frozen
in-memory reference, job_kwargs propagation, default chunking, multiprocess
worker path, bounded peak memory) and ``_apply_artifact_mask`` input strictness
(empty / unsorted valid_times raise; full coverage short-circuits).
"""

from __future__ import annotations

import numpy as np
import pytest


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


def _synthetic_artifact_recording():
    """8-channel, 90 000-sample (3 s @ 30 kHz) recording with two planted
    artifact runs -- one common-mode amplitude burst, one single-channel
    z-score outlier -- plus heterogeneous gains so the µV scaling matters.
    """
    import spikeinterface as si

    fs = 30_000.0
    n_samples = 90_000
    n_channels = 8
    rng = np.random.default_rng(0)
    # Small baseline noise that never trips either threshold.
    traces = rng.normal(0.0, 2.0, size=(n_samples, n_channels)).astype(
        np.float32
    )
    # Common-mode amplitude burst across all channels (trips amplitude).
    traces[20_000:20_300, :] += 300.0
    # Single-channel transient (trips the across-channel z-score).
    traces[60_000:60_120, 3] += 400.0
    rec = si.NumpyRecording(traces_list=[traces], sampling_frequency=fs)
    # Heterogeneous gains: the chunk worker and the reference must apply the
    # SAME per-channel gain, so a wrong gain broadcast surfaces as inequality.
    rec.set_channel_gains([1.0, 0.5, 2.0, 0.25, 1.0, 1.5, 0.8, 1.2])
    return rec


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

    assert np.array_equal(chunked, reference), (
        "Chunked artifact frames diverge from the in-memory reference; the "
        "ChunkRecordingExecutor port changed which frames are flagged."
    )
    assert np.array_equal(
        single, reference
    ), "Single-chunk pass should reproduce the in-memory reference exactly."


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
