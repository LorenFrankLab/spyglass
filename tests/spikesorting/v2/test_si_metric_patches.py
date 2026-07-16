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
