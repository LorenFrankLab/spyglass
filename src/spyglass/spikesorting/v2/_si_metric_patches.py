"""Local fix for a SpikeInterface ``nn_noise_overlap`` bug on sparse analyzers.

SI's ``nearest_neighbors_noise_overlap`` (``spikeinterface.metrics.quality.
pca_metrics``) sparsifies the per-unit noise cluster to the unit's sparse channel
set, but derives the peak channel from the **dense** median template: it only
sparsifies the median waveform on the ``if not sorting_analyzer.is_sparse()``
branch. On a sparse analyzer whose peak channel index (full space) exceeds a
unit's sparse channel count, indexing the sparse noise cluster with that dense
index raises ``IndexError`` -- which SI's per-unit ``except: = np.nan`` swallows,
so every ``nn_noise_overlap`` is silently NaN.

The v2 metric analyzer is built ``sparse=True``, so on any multi-channel probe
the shipped auto-curation rules that threshold ``nn_noise_overlap`` would never
fire. This module installs a corrected copy of the function over the SI module
symbol (median waveform sparsified for sparse analyzers too). Validated against
spikeinterface 0.104.3; the same bug is present on SI ``main`` at time of writing.

The corrected metric must run in-process (``n_jobs=1``): SI parallelises
``nn_advanced`` with a ``ProcessPoolExecutor`` whose spawned workers re-import SI
and would not see this monkeypatch, so the caller forces ``n_jobs=1`` for the
PC/NN metric computation. Remove this shim once a fixed SpikeInterface is adopted.
"""

from __future__ import annotations

import spikeinterface as si
import spikeinterface.metrics.quality.pca_metrics as _pm

from spyglass.utils import logger

#: SI version families whose ``nearest_neighbors_noise_overlap`` carries the
#: sparse bug this module patches. Revisit (and delete the shim) on SI upgrade.
_VALIDATED_SI_PREFIXES = ("0.104",)

_PATCH_FLAG = "_spyglass_v2_nn_noise_overlap_sparsity_patched"


def _nn_noise_overlap_sparse_fixed(
    sorting_analyzer,
    this_unit_id,
    n_spikes_all_units=None,
    fr_all_units=None,
    max_spikes=1000,
    min_spikes=10,
    min_fr=0.0,
    n_neighbors=5,
    n_components=10,
    radius_um=100,
    peak_sign="neg",
    seed=None,
):
    """``nearest_neighbors_noise_overlap`` with the sparse-median fix.

    Faithful copy of SI 0.104.3's function; the ONLY change is that the median
    waveform -- which always comes back DENSE from
    ``templates.get_data(operator='median')`` -- is sparsified to the unit's
    channels for sparse analyzers too, so the peak-channel index matches the
    (sparse) noise cluster's channel axis instead of overrunning it.
    """
    from sklearn.decomposition import IncrementalPCA

    np = _pm.np
    warnings = _pm.warnings

    rng = np.random.default_rng(seed=seed)

    waveforms_ext = sorting_analyzer.get_extension("waveforms")
    assert (
        waveforms_ext is not None
    ), "nn_noise_overlap needs extension 'waveforms'"
    templates_ext = sorting_analyzer.get_extension("templates")
    assert (
        templates_ext is not None
    ), "nn_noise_overlap needs extension 'templates'"

    try:
        templates_ext.get_data(operator="median")
    except KeyError:
        warnings.warn(
            "nn_noise_overlap needs 'templates' computed with the 'median' "
            "operator; run sorting_analyzer.compute('templates', "
            "operators=['average', 'median'])."
        )

    if n_spikes_all_units is None:
        n_spikes_all_units = _pm.compute_num_spikes(sorting_analyzer)
    if fr_all_units is None:
        fr_all_units = _pm.compute_firing_rates(sorting_analyzer)

    if n_spikes_all_units[this_unit_id] < min_spikes:
        return np.nan
    if fr_all_units[this_unit_id] < min_fr:
        return np.nan

    nsamples = waveforms_ext.nbefore + waveforms_ext.nafter
    recording = sorting_analyzer.recording
    noise_cluster = _pm.get_random_data_chunks(
        recording,
        return_in_uV=sorting_analyzer.return_in_uV,
        num_chunks_per_segment=max_spikes,
        chunk_size=nsamples,
        seed=seed,
    )
    noise_cluster = np.reshape(noise_cluster, (max_spikes, nsamples, -1))

    waveforms = waveforms_ext.get_waveforms_one_unit(
        unit_id=this_unit_id, force_dense=False
    ).copy()

    if waveforms.shape[0] > max_spikes:
        wf_ind = rng.choice(waveforms.shape[0], max_spikes, replace=False)
        waveforms = waveforms[wf_ind]
        n_snippets = max_spikes
    elif waveforms.shape[0] < max_spikes:
        noise_ind = rng.choice(
            noise_cluster.shape[0], waveforms.shape[0], replace=False
        )
        noise_cluster = noise_cluster[noise_ind]
        n_snippets = waveforms.shape[0]
    else:
        n_snippets = max_spikes

    if sorting_analyzer.is_sparse():
        sparsity = sorting_analyzer.sparsity
    else:
        sparsity = _pm.compute_sparsity(
            sorting_analyzer,
            method="radius",
            peak_sign=peak_sign,
            radius_um=radius_um,
        )
    channels = sparsity.unit_id_to_channel_indices[this_unit_id]
    noise_cluster = noise_cluster[:, :, channels]

    all_templates = templates_ext.get_data(operator="median")
    this_unit_index = sorting_analyzer.sorting.id_to_index(this_unit_id)
    median_waveform = all_templates[this_unit_index, :, :]

    # FIX vs upstream: ``median_waveform`` is ALWAYS dense (from
    # ``get_data(operator='median')``), so sparsify it to the unit's channels
    # unconditionally. Upstream only sparsified it on the ``not is_sparse()``
    # branch, leaving ``chmax`` in dense space while ``noise_cluster`` is sparse.
    median_waveform = median_waveform[:, channels]
    if not sorting_analyzer.is_sparse():
        waveforms = waveforms[:, :, channels]

    tmax, chmax = np.unravel_index(
        np.argmax(np.abs(median_waveform)), median_waveform.shape
    )
    weights = np.asarray(
        [noise_clip[tmax, chmax] for noise_clip in noise_cluster]
    )
    weights = weights / np.sum(weights)
    weighted_noise_snippet = np.sum(
        weights * noise_cluster.swapaxes(0, 2), axis=2
    ).swapaxes(0, 1)

    for snippet in range(n_snippets):
        waveforms[snippet, :, :] = _pm._subtract_clip_component(
            waveforms[snippet, :, :], weighted_noise_snippet
        )
        noise_cluster[snippet, :, :] = _pm._subtract_clip_component(
            noise_cluster[snippet, :, :], weighted_noise_snippet
        )

    all_snippets = np.concatenate(
        [
            waveforms.reshape((n_snippets, -1)),
            noise_cluster.reshape((n_snippets, -1)),
        ],
        axis=0,
    )
    pca = IncrementalPCA(n_components=n_components)
    pca.partial_fit(all_snippets)
    projected_snippets = pca.transform(all_snippets)
    return 1 - _pm._compute_isolation(
        projected_snippets[:n_snippets, :],
        projected_snippets[n_snippets:, :],
        n_neighbors,
    )


def patch_nn_noise_overlap_sparsity() -> None:
    """Idempotently install the sparse-median ``nn_noise_overlap`` fix.

    No-op if already applied. If the running SI version is outside the validated
    family, logs a warning (and still applies) so a future SI change surfaces.
    Callers must run the PC/NN metric with ``n_jobs=1`` -- see module docstring.
    """
    if getattr(_pm, _PATCH_FLAG, False):
        return
    if not si.__version__.startswith(_VALIDATED_SI_PREFIXES):
        logger.warning(
            "spikesorting v2: SpikeInterface %s is outside the validated set "
            "%s for the nn_noise_overlap sparsity patch; applying anyway -- "
            "re-verify or remove this shim.",
            si.__version__,
            _VALIDATED_SI_PREFIXES,
        )
    _pm.nearest_neighbors_noise_overlap = _nn_noise_overlap_sparse_fixed
    setattr(_pm, _PATCH_FLAG, True)
