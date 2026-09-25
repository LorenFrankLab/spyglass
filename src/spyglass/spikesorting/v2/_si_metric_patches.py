"""Local fixes for SpikeInterface quality metrics on sparse or masked sorts.

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

The corrected copy also draws its noise cluster only from the sort's
statistics spans (artifact-free frame ranges that never cross a recording
join) when a caller supplies them with :func:`noise_cluster_spans`. SI draws
the noise snippets uniformly over the whole recording, so on a masked sort
they land in zeroed artifact frames or straddle a join. Without spans (or with
one span covering the recording) the draw is SI's own, unchanged.

The corrected metric must run in-process (``n_jobs=1``): SI parallelises
``nn_advanced`` with a ``ProcessPoolExecutor`` whose spawned workers re-import SI
and would not see this monkeypatch (nor the spans, which live in a
``ContextVar`` of the calling context), so the caller forces ``n_jobs=1`` for the
PC/NN metric computation. Remove this shim once a fixed SpikeInterface is adopted.

``sd_ratio`` divides by a noise std that, on a masked sort, is estimated from
the statistics spans (``cache_span_noise_levels``), but SI's correction for
the unit's own template variance counts the unit's spikes over every sample,
masked ones included, so it under-subtracts and biases ``sd_ratio`` low.
:func:`patch_sd_ratio_statistics_spans` takes that correction over the spikes
and samples inside the spans set by :func:`noise_cluster_spans`; without
spans (or with one covering the recording) it is SI's function, unchanged.
"""

from __future__ import annotations

import contextlib
import contextvars

import spikeinterface as si
import spikeinterface.metrics.quality.misc_metrics as _mm
import spikeinterface.metrics.quality.pca_metrics as _pm

from spyglass.utils import logger

#: SI version families whose ``nearest_neighbors_noise_overlap`` and
#: ``compute_sd_ratio`` this module patches. Revisit (and delete the shims)
#: on SI upgrade.
_VALIDATED_SI_PREFIXES = ("0.104",)

_PATCH_FLAG = "_spyglass_v2_nn_noise_overlap_sparsity_patched"
_SD_RATIO_PATCH_FLAG = "_spyglass_v2_sd_ratio_statistics_spans_patched"

#: Statistics spans the nn noise cluster is drawn from and ``sd_ratio``'s
#: template correction counts over, as a tuple of half-open ``(start, end)``
#: frame pairs of the analyzer's recording; ``None`` keeps SI's
#: whole-recording behavior. Set only through ``noise_cluster_spans``.
_NOISE_CLUSTER_SPANS: contextvars.ContextVar[
    tuple[tuple[int, int], ...] | None
] = contextvars.ContextVar("nn_noise_cluster_spans", default=None)


@contextlib.contextmanager
def noise_cluster_spans(spans):
    """Restrict the span-aware metrics to ``spans`` inside this block.

    The nn noise cluster is drawn only from ``spans``, and ``sd_ratio``'s
    template correction counts spikes and samples only inside them.

    Parameters
    ----------
    spans : list[tuple[int, int]] or None
        Half-open statistics spans in the frame coordinates of the measured
        analyzer's recording (the display and metric analyzers are built
        on the same frames). ``None`` (or one span covering the recording)
        keeps SpikeInterface's own behavior.

    Notes
    -----
    The value lives in a ``contextvars.ContextVar``: it reaches the metric
    only when SI computes it in this thread (``n_jobs=1``), and the previous
    value is restored when the block exits, including on an exception.
    """
    value = None if spans is None else tuple((int(a), int(b)) for a, b in spans)
    token = _NOISE_CLUSTER_SPANS.set(value)
    try:
        yield
    finally:
        _NOISE_CLUSTER_SPANS.reset(token)


def _draw_noise_cluster(recording, *, n_snippets, nsamples, seed, return_in_uV):
    """Random fixed-length noise snippets for ``nn_noise_overlap``.

    Without statistics spans set (or with one span covering the recording)
    this is SI 0.104.3's own draw, unchanged: ``get_random_data_chunks`` with
    starts uniform in ``[0, n_samples - nsamples]``, sorted. With spans set
    by :func:`noise_cluster_spans`, starts are drawn uniformly over the
    positions whose whole snippet lies inside one span
    (``sample_span_snippet_starts``), so no snippet reads an excluded frame
    or crosses a join. ``seed`` is passed through with SI's semantics
    (``numpy.random.default_rng(seed)``; ``None`` is unseeded).

    Parameters
    ----------
    recording : spikeinterface.BaseRecording
        The analyzer's (single-segment) recording.
    n_snippets : int
        Keyword-only. Number of snippets.
    nsamples : int
        Keyword-only. Snippet length in frames.
    seed : int or None
        Keyword-only. Seed of the start draw.
    return_in_uV : bool
        Keyword-only. Forwarded to ``recording.get_traces``.

    Returns
    -------
    numpy.ndarray
        ``(n_snippets, nsamples, n_channels)`` snippets in start order.
    """
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        sample_span_snippet_starts,
        spans_cover_recording,
    )

    np = _pm.np
    spans = _NOISE_CLUSTER_SPANS.get()
    if spans_cover_recording(spans, recording.get_num_samples()):
        noise_cluster = _pm.get_random_data_chunks(
            recording,
            return_in_uV=return_in_uV,
            num_chunks_per_segment=n_snippets,
            chunk_size=nsamples,
            seed=seed,
        )
        return np.reshape(noise_cluster, (n_snippets, nsamples, -1))
    try:
        starts = sample_span_snippet_starts(
            list(spans), nsamples=nsamples, n_snippets=n_snippets, seed=seed
        )
    except ValueError:
        # SI's nn_noise_overlap caller catches every exception and returns
        # NaN without a log, so this warning is the only trace of why.
        longest = max(spans, key=lambda span: span[1] - span[0], default=(0, 0))
        logger.warning(
            "nn_noise_overlap: no statistics span fits a noise snippet of "
            f"{nsamples} frames; the longest span is {longest} "
            f"({longest[1] - longest[0]} frames). SpikeInterface reports "
            "this unit's nn_noise_overlap as NaN."
        )
        raise
    return np.stack(
        [
            recording.get_traces(
                start_frame=int(start),
                end_frame=int(start) + nsamples,
                return_in_uV=return_in_uV,
            )
            for start in starts
        ]
    )


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

    Faithful copy of SI 0.104.3's function with two changes: the median
    waveform -- which always comes back DENSE from
    ``templates.get_data(operator='median')`` -- is sparsified to the unit's
    channels for sparse analyzers too, so the peak-channel index matches the
    (sparse) noise cluster's channel axis instead of overrunning it; and the
    noise cluster comes from ``_draw_noise_cluster``, which honours the
    statistics spans set by ``noise_cluster_spans``.
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
    # Change vs upstream: drawn from the statistics spans when a caller set
    # them (``noise_cluster_spans``); otherwise SI's own draw, unchanged.
    noise_cluster = _draw_noise_cluster(
        sorting_analyzer.recording,
        n_snippets=max_spikes,
        nsamples=nsamples,
        seed=seed,
        return_in_uV=sorting_analyzer.return_in_uV,
    )

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


def _template_corrected_sd_ratio(
    sorting_analyzer,
    *,
    n_spikes,
    total_samples,
    unit_ids=None,
    periods=None,
    censored_period_ms=4.0,
    correct_for_drift=True,
    peak_sign="neg",
    **job_kwargs,
):
    """SI 0.104.3's ``sd_ratio`` with its template correction over a population.

    SI's ``compute_sd_ratio`` divides the amplitude spread by
    ``sqrt(std_noise**2 - template_variance)``, the noise std less the
    variance a unit's own non-overlapping template adds to its extremum
    channel, with ``p = len(template) * n_spikes / get_total_samples()``.
    Here SI computes the uncorrected ratio (``correct_for_template_itself=
    False``: same censoring, drift correction and NaN / 0.0 edge cases) and
    the correction is applied to it with SI's formula and inputs -- the same
    cached ``std`` noise level, extremum channel and dense template -- except
    that ``p`` is taken over the given population::

        p = len(template) * n_spikes[unit_id] / total_samples

    The result equals SI's corrected value up to floating-point rounding
    when ``n_spikes`` and ``total_samples`` are SI's own.

    Parameters
    ----------
    sorting_analyzer : spikeinterface.SortingAnalyzer
        Needs the ``templates`` and ``spike_amplitudes`` extensions.
    n_spikes : dict
        Keyword-only. ``{unit_id: int}`` spikes of the population the noise
        std was measured on.
    total_samples : int
        Keyword-only. Samples in that population.
    unit_ids, periods, censored_period_ms, correct_for_drift, peak_sign
        Keyword-only. SI's ``compute_sd_ratio`` arguments, with its defaults.
    **job_kwargs
        SI's noise-level kwargs (e.g. ``random_slices_kwargs``), forwarded
        to ``get_noise_levels`` as SI forwards them.

    Returns
    -------
    dict
        ``{unit_id: float}`` corrected ``sd_ratio``.
    """
    from spikeinterface.core import get_noise_levels

    np = _mm.np
    uncorrected = _mm.compute_sd_ratio(
        sorting_analyzer,
        unit_ids=unit_ids,
        periods=periods,
        censored_period_ms=censored_period_ms,
        correct_for_drift=correct_for_drift,
        correct_for_template_itself=False,
        peak_sign=peak_sign,
        **job_kwargs,
    )
    # SI's call above cached this std on the recording (or read the cached
    # span std), so this is the std it divided by.
    noise_levels = get_noise_levels(
        sorting_analyzer.recording,
        return_in_uV=sorting_analyzer.return_in_uV,
        method="std",
        **{**job_kwargs, "progress_bar": False},
    )
    best_channels = _mm.get_template_extremum_channel(
        sorting_analyzer, outputs="index", peak_sign=peak_sign
    )
    templates_array = _mm.get_dense_templates_array(
        sorting_analyzer, return_in_uV=sorting_analyzer.return_in_uV
    )
    corrected = {}
    for unit_id, ratio in uncorrected.items():
        best_channel = best_channels[unit_id]
        unit_index = sorting_analyzer.sorting.id_to_index(unit_id)
        template = templates_array[unit_index, :, best_channel]
        # Changed vs SI: the population the noise std was measured on.
        p = len(template) * n_spikes[unit_id] / total_samples
        template_variance = (
            p * np.mean(template**2) - p**2 * np.mean(template) ** 2
        )
        std_noise = noise_levels[best_channel]
        # (unit_std / std_noise) * std_noise / corrected std is SI's
        # unit_std / corrected std. SI's NaN (no spikes) stays NaN and its
        # 0.0 (one spike) stays 0.0 whenever the corrected std is real.
        corrected[unit_id] = (
            ratio * std_noise / np.sqrt(std_noise**2 - template_variance)
        )
    return corrected


def _span_spike_counts(sorting, spans):
    """``{unit_id: int}`` spikes whose frame lies inside a half-open span."""
    np = _mm.np
    spike_vector = sorting.to_spike_vector()
    frames = spike_vector["sample_index"]
    starts = np.array([a for a, _ in spans], dtype=np.int64)
    ends = np.array([b for _, b in spans], dtype=np.int64)
    owner = np.searchsorted(starts, frames, side="right") - 1
    inside = (owner >= 0) & (frames < ends[np.maximum(owner, 0)])
    counts = np.bincount(
        spike_vector["unit_index"][inside], minlength=len(sorting.unit_ids)
    )
    return {
        unit_id: int(count) for unit_id, count in zip(sorting.unit_ids, counts)
    }


def _sd_ratio_statistics_spans(
    sorting_analyzer,
    unit_ids=None,
    periods=None,
    censored_period_ms=4.0,
    correct_for_drift=True,
    correct_for_template_itself=True,
    peak_sign="neg",
    **job_kwargs,
):
    """``compute_sd_ratio`` whose template correction uses the spans.

    Same signature as SI 0.104.3's ``compute_sd_ratio``. Without statistics
    spans set by :func:`noise_cluster_spans` (or with one span covering the
    recording), or with ``correct_for_template_itself=False``, this is SI's
    function, called unchanged. With spans, the noise std it divides by is
    the std of the span samples (``cache_span_noise_levels``), so the
    template correction takes ``p`` over the same population: the unit's
    spikes whose frame lies inside a span, over the total span samples.
    Counting a spike by its frame -- the sample index SI aligns its waveform
    on -- selects the spikes whose waveforms the span samples contain; a
    spike within a waveform length of a span edge contributes part of its
    waveform either way, an edge effect SI's non-overlapping-template
    approximation already neglects. The spans are in the frame
    coordinates of ``sorting_analyzer``'s recording.
    """
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        spans_cover_recording,
    )

    sd_ratio_kwargs = dict(
        unit_ids=unit_ids,
        periods=periods,
        censored_period_ms=censored_period_ms,
        correct_for_drift=correct_for_drift,
        peak_sign=peak_sign,
    )
    spans = _NOISE_CLUSTER_SPANS.get()
    if not correct_for_template_itself or spans_cover_recording(
        spans, sorting_analyzer.get_total_samples()
    ):
        return _mm.compute_sd_ratio(
            sorting_analyzer,
            correct_for_template_itself=correct_for_template_itself,
            **sd_ratio_kwargs,
            **job_kwargs,
        )
    return _template_corrected_sd_ratio(
        sorting_analyzer,
        n_spikes=_span_spike_counts(sorting_analyzer.sorting, spans),
        total_samples=sum(end - start for start, end in spans),
        **sd_ratio_kwargs,
        **job_kwargs,
    )


def patch_sd_ratio_statistics_spans() -> None:
    """Idempotently route SI's ``sd_ratio`` metric through the spans.

    SI's quality-metrics extension runs each metric as
    ``metric_class.metric_function(...)``, so the ``SDRatio`` class
    attribute is replaced; the module-level ``compute_sd_ratio`` stays
    SI's own. The metric runs in the calling thread, so the spans set by
    :func:`noise_cluster_spans` reach it. Warns (and still applies) outside
    the validated SI versions.
    """
    if getattr(_mm.SDRatio, _SD_RATIO_PATCH_FLAG, False):
        return
    if not si.__version__.startswith(_VALIDATED_SI_PREFIXES):
        logger.warning(
            "spikesorting v2: SpikeInterface %s is outside the validated set "
            "%s for the sd_ratio statistics-span patch; applying anyway -- "
            "re-verify or remove this shim.",
            si.__version__,
            _VALIDATED_SI_PREFIXES,
        )
    _mm.SDRatio.metric_function = _sd_ratio_statistics_spans
    setattr(_mm.SDRatio, _SD_RATIO_PATCH_FLAG, True)
