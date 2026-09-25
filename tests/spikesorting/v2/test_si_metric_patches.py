"""DB-free tests for the SpikeInterface nn_noise_overlap sparse-analyzer shim.

These build tiny in-memory SI analyzers (no database, no fixtures) and exercise
the vendored fix directly, so the SI bug and its correction are guarded cheaply
and in isolation from the heavy end-to-end curation-evaluation test.
"""

import numpy as np
import pytest


def _sparse_whitened_analyzer(num_channels, num_units):
    """A sparse, spatially-whitened analyzer with median templates + PCA.

    Mirrors how the v2 metric analyzer is built (sparse + whitened + the pinned
    PCA params), which is the configuration that trips SI's bug.
    """
    import spikeinterface.full as sf
    import spikeinterface.preprocessing as spre
    from spikeinterface.core import generate_ground_truth_recording

    rec, sort = generate_ground_truth_recording(
        durations=[30.0],
        sampling_frequency=30000.0,
        num_channels=num_channels,
        num_units=num_units,
        seed=0,
    )
    rec = spre.whiten(rec, dtype="float32")
    analyzer = sf.create_sorting_analyzer(
        sort, rec, format="memory", sparse=True
    )
    analyzer.compute(
        ["random_spikes", "noise_levels", "templates", "waveforms"]
    )
    analyzer.compute("templates", operators=["average", "std", "median"])
    analyzer.compute(
        "principal_components",
        n_components=5,
        mode="by_channel_local",
        whiten=True,
        dtype="float32",
    )
    return analyzer


@pytest.mark.parametrize("num_channels", [32, 64])
def test_patched_nn_noise_overlap_is_finite_on_sparse_many_channel(
    num_channels,
):
    """The vendored fix returns a finite value where upstream raises (-> NaN).

    Unpatched SI derives the peak channel from the dense median template but
    indexes the sparse noise cluster with it, so on a sparse analyzer whose peak
    channel exceeds the sparse channel count it raises IndexError (swallowed as
    NaN by SI's per-unit ``except``). The fix sparsifies the median first.
    """
    from spyglass.spikesorting.v2._si_metric_patches import (
        _nn_noise_overlap_sparse_fixed,
    )

    analyzer = _sparse_whitened_analyzer(num_channels, 3)
    value = _nn_noise_overlap_sparse_fixed(analyzer, analyzer.unit_ids[0])
    assert np.isfinite(value)


def test_patch_is_idempotent_and_installs_the_fix():
    """patch_nn_noise_overlap_sparsity swaps in the fix and is a no-op twice."""
    import spikeinterface.metrics.quality.pca_metrics as pm

    from spyglass.spikesorting.v2._si_metric_patches import (
        _nn_noise_overlap_sparse_fixed,
        patch_nn_noise_overlap_sparsity,
    )

    patch_nn_noise_overlap_sparsity()
    assert pm.nearest_neighbors_noise_overlap is _nn_noise_overlap_sparse_fixed
    # Second call must not re-wrap or raise.
    patch_nn_noise_overlap_sparsity()
    assert pm.nearest_neighbors_noise_overlap is _nn_noise_overlap_sparse_fixed


# ---------- nn noise cluster drawn from the statistics spans -----------------

_FS = 30_000.0
# SI's default waveform window (1 ms before + 2 ms after) at 30 kHz.
_NSAMPLES = 90
_N_SNIPPETS = 1000


def _masked_frame_indexed_recording():
    """A 45%-masked recording whose channel 0 is the frame index.

    Channel 0 holds each frame's own index (never masked), so every drawn
    snippet's frames are read back exactly. Channels 1-2 hold non-zero
    Gaussian noise on retained frames and +/-800 uV transients inside the
    excluded ranges. The statistics spans also split one long retained stretch
    at a join, so a snippet straddling that join lies in valid frames yet
    crosses a span boundary.

    Returns
    -------
    recording : spikeinterface.core.NumpyRecording
    spans : list[tuple[int, int]]
        Statistics spans (artifact-free, split at the join).
    excluded : numpy.ndarray
        ``(n_samples,)`` bool, True on excluded frames.
    join : int
        The join frame.
    """
    from spikeinterface.core import NumpyRecording

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        complement_frame_ranges,
        statistics_spans,
    )
    from tests.spikesorting.v2._masked_statistics_helpers import (
        TRANSIENT_UV,
        excluded_ranges,
    )

    n_samples = 120_000
    rng = np.random.default_rng(0)
    traces = np.empty((n_samples, 3), dtype="float64")
    traces[:, 0] = np.arange(n_samples)
    traces[:, 1:] = rng.normal(0.0, 10.0, size=(n_samples, 2))
    ranges = excluded_ranges(n_samples, 0.45)
    excluded = np.zeros(n_samples, dtype=bool)
    for start, end in ranges:
        excluded[start:end] = True
        traces[start:end, 1:] = TRANSIENT_UV * rng.choice(
            [-1.0, 1.0], size=(end - start, 2)
        )
    longest = max(
        complement_frame_ranges(ranges, n_samples), key=lambda s: s[1] - s[0]
    )
    join = (longest[0] + longest[1]) // 2
    spans = statistics_spans(n_samples, ranges, [(0, join), (join, n_samples)])
    recording = NumpyRecording([traces], sampling_frequency=_FS)
    return recording, spans, excluded, join


def _si_default_noise_cluster(recording):
    """SI's own noise-cluster draw, as upstream ``nn_noise_overlap`` makes it."""
    from spikeinterface.core.recording_tools import get_random_data_chunks

    chunks = get_random_data_chunks(
        recording,
        return_in_uV=False,
        num_chunks_per_segment=_N_SNIPPETS,
        chunk_size=_NSAMPLES,
        seed=0,
    )
    return np.reshape(chunks, (_N_SNIPPETS, _NSAMPLES, -1))


def test_nn_noise_cluster_excludes_masked_samples():
    """Every noise snippet is contiguous frames inside ONE statistics span.

    Each snippet's frames are recovered from the frame-index channel: they
    must be consecutive, lie wholly inside a single span (so none overlaps an
    excluded range and none crosses the join), and carry non-zero data. SI's
    own draw on the same recording is shown to hit excluded frames, so the
    fixture can tell the two apart.
    """
    from spyglass.spikesorting.v2._si_metric_patches import (
        _draw_noise_cluster,
        noise_cluster_spans,
    )

    recording, spans, excluded, join = _masked_frame_indexed_recording()
    with noise_cluster_spans(spans):
        noise = _draw_noise_cluster(
            recording,
            n_snippets=_N_SNIPPETS,
            nsamples=_NSAMPLES,
            seed=0,
            return_in_uV=False,
        )

    assert noise.shape == (_N_SNIPPETS, _NSAMPLES, 3)
    frames = noise[:, :, 0].astype(np.int64)
    starts = frames[:, 0]
    assert np.array_equal(
        frames, starts[:, None] + np.arange(_NSAMPLES)[None, :]
    ), "a snippet is not a contiguous run of frames"
    assert np.all(np.diff(starts) >= 0), "snippets are not in start order"
    assert not excluded[frames].any(), "a snippet overlaps an excluded range"
    span_starts = np.array([a for a, _ in spans])
    span_ends = np.array([b for _, b in spans])
    owner = np.searchsorted(span_starts, starts, side="right") - 1
    assert np.all(
        starts + _NSAMPLES <= span_ends[owner]
    ), "a snippet straddles a statistics-span boundary"
    assert not np.any((starts < join) & (starts + _NSAMPLES > join))
    data = noise[:, :, 1:]
    assert np.all(np.any(data != 0, axis=(1, 2))), "an all-zero snippet"
    assert np.abs(data).max() < 800.0

    # The fixture discriminates: SI's uniform draw lands in excluded frames.
    si_frames = _si_default_noise_cluster(recording)[:, :, 0].astype(np.int64)
    assert excluded[si_frames].any()


@pytest.mark.parametrize("cover", ["none", "whole_recording"])
def test_nn_noise_cluster_unchanged_when_spans_cover_recording(cover):
    """No spans, or one span over the recording, is SI's own draw exactly."""
    from spyglass.spikesorting.v2._si_metric_patches import (
        _draw_noise_cluster,
        noise_cluster_spans,
    )

    recording, _, _, _ = _masked_frame_indexed_recording()
    spans = None if cover == "none" else [(0, recording.get_num_samples())]
    with noise_cluster_spans(spans):
        noise = _draw_noise_cluster(
            recording,
            n_snippets=_N_SNIPPETS,
            nsamples=_NSAMPLES,
            seed=0,
            return_in_uV=False,
        )
    expected = _si_default_noise_cluster(recording)
    assert noise.dtype == expected.dtype
    assert np.array_equal(noise, expected)


def test_nn_noise_cluster_warns_when_no_span_fits_a_snippet(caplog):
    """No span long enough for one snippet: a warning names the snippet
    length and the longest span, then the error propagates (SI's caller
    turns it into a silent NaN, so the warning is the only trace)."""
    from spyglass.spikesorting.v2._si_metric_patches import (
        _draw_noise_cluster,
        noise_cluster_spans,
    )

    recording, _, _, _ = _masked_frame_indexed_recording()
    with caplog.at_level("WARNING"):
        with noise_cluster_spans([(0, 50), (100, 180)]):
            with pytest.raises(ValueError, match="no span admits"):
                _draw_noise_cluster(
                    recording,
                    n_snippets=_N_SNIPPETS,
                    nsamples=_NSAMPLES,
                    seed=0,
                    return_in_uV=False,
                )
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert f"{_NSAMPLES} frames" in message
    assert "80 frames" in message and "(100, 180)" in message


def test_noise_cluster_spans_reset_on_exit_and_on_error():
    """The spans are visible only inside the block, and reset on an error."""
    from spyglass.spikesorting.v2._si_metric_patches import (
        _NOISE_CLUSTER_SPANS,
        noise_cluster_spans,
    )

    assert _NOISE_CLUSTER_SPANS.get() is None
    with noise_cluster_spans([(0, 10), (20, 40)]):
        assert _NOISE_CLUSTER_SPANS.get() == ((0, 10), (20, 40))
        with noise_cluster_spans([(5, 9)]):
            assert _NOISE_CLUSTER_SPANS.get() == ((5, 9),)
        assert _NOISE_CLUSTER_SPANS.get() == ((0, 10), (20, 40))
    assert _NOISE_CLUSTER_SPANS.get() is None

    with pytest.raises(RuntimeError, match="boom"):
        with noise_cluster_spans([(0, 10)]):
            raise RuntimeError("boom")
    assert _NOISE_CLUSTER_SPANS.get() is None


@pytest.fixture(scope="module")
def small_ground_truth():
    """(traces, probe, sorting) of a 20 s, 8-channel, 4-unit recording."""
    import spikeinterface as si

    recording, sorting = si.generate_ground_truth_recording(
        durations=[20.0],
        sampling_frequency=_FS,
        num_channels=8,
        num_units=4,
        seed=0,
    )
    return recording.get_traces(), recording.get_probe(), sorting


def _nn_noise_overlap_by_fill(small_ground_truth):
    """``nn_noise_overlap`` per excluded-sample fill, via the patched metric.

    The excluded frames (30% of each of 12 blocks) are overwritten with
    zeros, +/-10 mV, or NaN. Spikes whose waveform window touches an excluded
    frame are dropped (the sorter never sees masked frames), so only the
    noise cluster can read excluded samples. The analyzer is built like the
    metric recipe: span-whitened, sparse, ``return_in_uV=False``.

    Returns
    -------
    dict
        ``{(fill, with_spans): (n_units,) nn_noise_overlap}`` for each fill
        and with / without the statistics spans set.
    """
    import spikeinterface as si
    from spikeinterface.metrics.quality import compute_quality_metrics

    from spyglass.spikesorting.v2._si_metric_patches import (
        noise_cluster_spans,
        patch_nn_noise_overlap_sparsity,
    )
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        statistics_spans,
    )
    from spyglass.spikesorting.v2._sorting_dispatch import pinned_whiten
    from tests.spikesorting.v2._masked_statistics_helpers import (
        numpy_recording,
    )

    traces, probe, sorting = small_ground_truth
    n_samples = traces.shape[0]
    block = n_samples // 12
    length = int(0.30 * block)
    ranges = [
        (i * block + 1_000, i * block + 1_000 + length) for i in range(12)
    ]
    spans = statistics_spans(n_samples, ranges, [(0, n_samples)])
    near_excluded = np.zeros(n_samples, dtype=bool)
    for start, end in ranges:
        near_excluded[max(0, start - 100) : end + 100] = True
    samples, labels = [], []
    for unit_index, unit_id in enumerate(sorting.unit_ids):
        train = sorting.get_unit_spike_train(unit_id)
        train = train[~near_excluded[train]]
        samples.append(train)
        labels.append(np.full(train.size, unit_index))
    kept = si.NumpySorting.from_samples_and_labels(
        [np.concatenate(samples)], [np.concatenate(labels)], _FS
    )
    assert min(len(t) for t in samples) >= 50

    patch_nn_noise_overlap_sparsity()
    results = {}
    for fill in ("zeros", "10mV", "nan"):
        filled = traces.copy()
        for start, end in ranges:
            if fill == "zeros":
                filled[start:end] = 0.0
            elif fill == "10mV":
                sign = np.where(np.arange(end - start) % 2 == 0, 1.0, -1.0)
                filled[start:end] = (10_000.0 * sign)[:, None]
            else:
                filled[start:end] = np.nan
        recording = pinned_whiten(
            numpy_recording(filled, probe), random_seed=0, spans=spans
        )
        analyzer = si.create_sorting_analyzer(
            kept, recording, format="memory", sparse=True, return_in_uV=False
        )
        analyzer.compute("random_spikes", seed=0)
        analyzer.compute("waveforms")
        analyzer.compute("templates", operators=["average", "median"])
        analyzer.compute(
            "principal_components",
            n_components=5,
            mode="by_channel_local",
            whiten=True,
            dtype="float32",
        )
        for with_spans in (True, False):
            with noise_cluster_spans(spans if with_spans else None):
                metrics = compute_quality_metrics(
                    analyzer,
                    metric_names=["nn_advanced"],
                    metric_params={
                        "nn_advanced": {
                            "seed": 0,
                            "n_neighbors": 5,
                            "n_components": 7,
                        }
                    },
                    skip_pc_metrics=False,
                    delete_existing_metrics=True,
                    n_jobs=1,
                )
            results[fill, with_spans] = metrics["nn_noise_overlap"].to_numpy(
                dtype=float
            )
    return results


@pytest.mark.medium
def test_nn_noise_overlap_invariant_to_excluded_sample_values(
    small_ground_truth,
):
    """Through the real patched metric, excluded samples cannot move it.

    With the statistics spans set, ``nn_noise_overlap`` is finite and
    bit-identical whether the excluded frames hold zeros, +/-10 mV or NaN.
    Without them, SI's uniform noise draw reads those frames and the metric
    changes, so the comparison is sensitive to a leak.
    """
    values = _nn_noise_overlap_by_fill(small_ground_truth)
    reference = values["zeros", True]
    assert np.all(np.isfinite(reference))
    for fill in ("10mV", "nan"):
        assert np.array_equal(values[fill, True], reference), fill

    assert not np.array_equal(values["10mV", False], values["zeros", False])
    assert not np.all(np.isfinite(values["nan", False]))
