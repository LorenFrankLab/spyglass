"""Local fixes for SpikeInterface 0.104 quality metrics on sparse or masked sorts.

- ``nn_noise_overlap``: SI indexes the sparse noise cluster with a dense peak
  channel, so on a sparse analyzer a value is NaN or uses the wrong channel.
  The patched
  copy sparsifies the median waveform and draws noise only from the
  statistics spans set by :func:`noise_cluster_spans`.
- ``sd_ratio``: the template correction counts spikes and samples only inside
  those spans, matching the span-based noise std.
- :func:`isolated_si_metric_defaults`: SI writes each compute's params into
  class-level defaults; this restores them.

Each ``patch_*`` is a context manager: SI's own function is restored when the
last active block exits. Patched metrics must run with ``n_jobs=1``: spawned
workers would not see the patches or the spans ``ContextVar``. Remove once a
fixed SI is adopted.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import threading

import spikeinterface as si
import spikeinterface.metrics.quality.misc_metrics as _mm
import spikeinterface.metrics.quality.pca_metrics as _pm
import numpy as np
from spikeinterface.metrics.quality import ComputeQualityMetrics

from spyglass.utils import logger
from spyglass.spikesorting.v2._core.signal_math import (
    frames_with_window_in_one_span,
)
from spyglass.spikesorting.v2._sorting.artifact_mask import (
    sample_span_snippet_starts,
    spans_cover_recording,
)

#: SI versions these patches are validated against.
_VALIDATED_SI_PREFIXES = ("0.104",)

#: ``(id(target), attr) -> [active blocks, SI's original]``.
_ACTIVE_PATCHES: dict[tuple[int, str], list] = {}
_ACTIVE_PATCHES_LOCK = threading.Lock()

#: Serializes Spyglass threads around SI's class-level metric params and
#: ``warnings.catch_warnings`` (process-wide state). Acquire after
#: ``_storage.analyzer_cache.analyzer_cache_lock``; take no other lock inside.
SI_METRIC_STATE_LOCK = threading.RLock()

#: Half-open ``(start, end)`` statistics spans in analyzer frames; ``None``
#: keeps SI's behavior. Set only through ``noise_cluster_spans``.
_NOISE_CLUSTER_SPANS: contextvars.ContextVar[
    tuple[tuple[int, int], ...] | None
] = contextvars.ContextVar("nn_noise_cluster_spans", default=None)

#: Seeded noise draws cached within a ``noise_cluster_spans`` block. SI passes
#: every unit the same seed, so each unit would re-read the same snippets.
_NOISE_CLUSTER_DRAWS: contextvars.ContextVar[dict | None] = (
    contextvars.ContextVar("nn_noise_cluster_draws", default=None)
)

#: Threads per nn neighbor search. Set only through ``isolation_workers``.
_ISOLATION_WORKERS: contextvars.ContextVar[int] = contextvars.ContextVar(
    "nn_isolation_workers", default=1
)

#: Snippets this close are read in one ``get_traces`` call. On time-tall,
#: few-channel HDF5 chunks one call costs about a second of 30 kHz data.
_SNIPPET_READ_MAX_GAP_FRAMES = 30_000

#: Upper bound on one grouped read, in bytes of the recording's traces.
_SNIPPET_READ_MAX_BYTES = 64 * 2**20


@contextlib.contextmanager
def noise_cluster_spans(spans):
    """Restrict the span-aware metrics to ``spans`` inside this block.

    A seeded noise draw is read once per block and shared read-only.

    Parameters
    ----------
    spans : list[tuple[int, int]] or None
        Half-open statistics spans in the analyzer recording's frames.
        ``None`` (or one span covering the recording) keeps SI's behavior.

    Notes
    -----
    Stored in a ``ContextVar``, so it reaches only metrics computed in this
    thread (``n_jobs=1``).
    """
    value = None if spans is None else tuple((int(a), int(b)) for a, b in spans)
    token = _NOISE_CLUSTER_SPANS.set(value)
    draws_token = _NOISE_CLUSTER_DRAWS.set({})
    try:
        yield
    finally:
        _NOISE_CLUSTER_DRAWS.reset(draws_token)
        _NOISE_CLUSTER_SPANS.reset(token)


@contextlib.contextmanager
def isolated_si_metric_defaults():
    """Keep SpikeInterface's metric defaults unchanged across this block.

    SI 0.104.3 ``update()``s the metric classes' own ``metric_params`` with
    each caller's kwargs, so a later row omitting a kwarg would inherit an
    earlier row's value. Each class holds a deep copy inside the block and
    gets its original back on exit. Holds :data:`SI_METRIC_STATE_LOCK`.
    """
    with SI_METRIC_STATE_LOCK, _si_metric_param_copies():
        yield


@contextlib.contextmanager
def _si_metric_param_copies():
    """Swap in copies of SI's metric-param dicts; restore them on exit.

    Unlocked; the caller must hold :data:`SI_METRIC_STATE_LOCK`.
    """
    from spikeinterface.metrics.template import ComputeTemplateMetrics

    metric_classes = {
        metric
        for extension in (ComputeQualityMetrics, ComputeTemplateMetrics)
        for metric in extension.metric_list
    }
    own_params = {
        metric: metric.__dict__.get("metric_params")
        for metric in metric_classes
    }
    for metric in metric_classes:
        metric.metric_params = copy.deepcopy(metric.metric_params)
    try:
        yield
    finally:
        for metric, params in own_params.items():
            if params is None:
                del metric.metric_params
            else:
                metric.metric_params = params


def _draw_noise_cluster(recording, *, n_snippets, nsamples, seed, return_in_uV):
    """Random noise snippets for ``nn_noise_overlap``, reused within a block.

    Seeded draws are cached within a :func:`noise_cluster_spans` block;
    ``seed=None`` draws anew, as SI does.
    """
    draws = _NOISE_CLUSTER_DRAWS.get()
    if draws is None or seed is None:
        return _draw_noise_cluster_uncached(
            recording,
            n_snippets=n_snippets,
            nsamples=nsamples,
            seed=seed,
            return_in_uV=return_in_uV,
        )
    key = (id(recording), int(n_snippets), int(nsamples), seed, return_in_uV)
    cached = draws.get(key)
    # Holding the recording keeps its id from being reused.
    if cached is None or cached[0] is not recording:
        noise = _draw_noise_cluster_uncached(
            recording,
            n_snippets=n_snippets,
            nsamples=nsamples,
            seed=seed,
            return_in_uV=return_in_uV,
        )
        noise.setflags(write=False)
        cached = draws[key] = (recording, noise)
    return cached[1]


def _read_snippets(recording, starts, *, nsamples, return_in_uV):
    """Read ``nsamples``-frame snippets at sorted ``starts``.

    Snippets within ``_SNIPPET_READ_MAX_GAP_FRAMES`` of each other are read
    in one ``get_traces`` call (up to ``_SNIPPET_READ_MAX_BYTES``) and
    sliced.

    Returns
    -------
    numpy.ndarray
        ``(len(starts), nsamples, n_channels)`` snippets in start order.
    """
    np = _pm.np
    starts = np.asarray(starts, dtype=np.int64)
    max_frames = max(
        nsamples,
        _SNIPPET_READ_MAX_BYTES
        // (recording.get_dtype().itemsize * recording.get_num_channels()),
    )
    snippets = []
    first = 0
    while first < len(starts):
        last = first
        block_start = int(starts[first])
        while (
            last + 1 < len(starts)
            and starts[last + 1] - (starts[last] + nsamples)
            <= _SNIPPET_READ_MAX_GAP_FRAMES
            and starts[last + 1] + nsamples - block_start <= max_frames
        ):
            last += 1
        block = recording.get_traces(
            start_frame=block_start,
            end_frame=int(starts[last]) + nsamples,
            return_in_uV=return_in_uV,
        )
        offsets = starts[first : last + 1] - block_start
        snippets.append(
            block[offsets[:, None] + np.arange(nsamples)[None, :], :]
        )
        first = last + 1
    return np.concatenate(snippets, axis=0)


def _draw_noise_cluster_uncached(
    recording, *, n_snippets, nsamples, seed, return_in_uV
):
    """Random fixed-length noise snippets for ``nn_noise_overlap``.

    Without spans this is SI 0.104.3's draw. With spans, each snippet lies
    wholly inside one span, so none reads an excluded frame or crosses a
    join.

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
        # SI turns any exception into a silent NaN; this log is the only trace.
        longest = max(spans, key=lambda span: span[1] - span[0], default=(0, 0))
        logger.warning(
            "nn_noise_overlap: no statistics span fits a noise snippet of "
            f"{nsamples} frames; the longest span is {longest} "
            f"({longest[1] - longest[0]} frames). SpikeInterface reports "
            "this unit's nn_noise_overlap as NaN."
        )
        raise
    return _read_snippets(
        recording, starts, nsamples=nsamples, return_in_uV=return_in_uV
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

    Copy of SI 0.104.3's function with two changes, marked inline: the dense
    median waveform is always sparsified, and the noise cluster honors the
    statistics spans.
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
    # Change vs SI: honors the statistics spans.
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

    # Change vs SI: the median template is always dense, so always sparsify.
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


@contextlib.contextmanager
def _patched(target, attr: str, replacement, *, description: str):
    """Set ``target.<attr>`` to ``replacement`` while any block is active.

    Outside the validated SI family, logs a warning naming ``description``
    and still applies, so a future SI change surfaces.
    """
    key = (id(target), attr)
    with _ACTIVE_PATCHES_LOCK:
        entry = _ACTIVE_PATCHES.get(key)
        if entry is None:
            if not si.__version__.startswith(_VALIDATED_SI_PREFIXES):
                logger.warning(
                    "spikesorting v2: SpikeInterface %s is outside the "
                    "validated set %s for the %s; applying anyway -- "
                    "re-verify or remove this shim.",
                    si.__version__,
                    _VALIDATED_SI_PREFIXES,
                    description,
                )
            entry = _ACTIVE_PATCHES[key] = [0, getattr(target, attr)]
            setattr(target, attr, replacement)
        entry[0] += 1
    try:
        yield
    finally:
        with _ACTIVE_PATCHES_LOCK:
            entry[0] -= 1
            if entry[0] == 0:
                setattr(target, attr, entry[1])
                del _ACTIVE_PATCHES[key]


def patch_nn_noise_overlap_sparsity():
    """Apply the sparse-median ``nn_noise_overlap`` fix inside the block.

    Callers must run the PC/NN metric with ``n_jobs=1``.
    """
    return _patched(
        _pm,
        "nearest_neighbors_noise_overlap",
        _nn_noise_overlap_sparse_fixed,
        description="nn_noise_overlap sparsity patch",
    )


@contextlib.contextmanager
def isolation_workers(n_workers: int):
    """Run :func:`_compute_isolation_kdtree` searches on ``n_workers`` threads."""
    token = _ISOLATION_WORKERS.set(int(n_workers))
    try:
        yield
    finally:
        _ISOLATION_WORKERS.reset(token)


def _compute_isolation_kdtree(
    pcs_target_unit, pcs_other_unit, n_neighbors: int
):
    """SI 0.104.3's ``_compute_isolation`` on a threaded scipy k-d tree.

    Same exact search as SI's single-threaded scikit-learn tree, but faster
    and multithreaded. Only ties between identical PC features can order
    differently, as arbitrarily as in SI. Each spike's own row is excluded as
    scikit-learn's ``kneighbors()`` does.

    Parameters
    ----------
    pcs_target_unit : numpy.ndarray
        ``(n_spikes_target, n_features)`` PC features of the target unit.
    pcs_other_unit : numpy.ndarray
        ``(n_spikes_other, n_features)`` PC features of the other cluster.
    n_neighbors : int
        Neighbors per spike; reduced to ``n_spikes - 1`` when larger.

    Returns
    -------
    float
        Fraction of neighbors that share each spike's cluster, in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``n_neighbors`` equals ``n_spikes``, as scikit-learn raises in SI;
        SI's callers turn it into a NaN metric.
    """
    from scipy.spatial import cKDTree

    n_spikes_target = pcs_target_unit.shape[0]
    n_spikes_other = pcs_other_unit.shape[0]
    pcs_concat = np.concatenate((pcs_target_unit, pcs_other_unit), axis=0)
    label_concat = np.concatenate(
        (np.zeros(n_spikes_target), np.ones(n_spikes_other))
    )
    n_spikes = len(label_concat)
    n_neighbors_adjusted = (
        n_spikes - 1 if n_neighbors > n_spikes else n_neighbors
    )
    if n_neighbors_adjusted >= n_spikes:
        raise ValueError(
            f"n_neighbors={n_neighbors_adjusted} must be less than the "
            f"{n_spikes} spikes searched"
        )
    _, neighbors = cKDTree(pcs_concat).query(
        pcs_concat,
        k=n_neighbors_adjusted + 1,
        workers=_ISOLATION_WORKERS.get(),
    )
    neighbors = np.asarray(neighbors).reshape(n_spikes, -1)
    keep = neighbors != np.arange(n_spikes)[:, None]
    keep[np.all(keep, axis=1), 0] = False
    membership_ind = neighbors[keep].reshape(n_spikes, n_neighbors_adjusted)
    target_nn_in_target = np.sum(
        label_concat[membership_ind[:n_spikes_target]] == 0
    )
    other_nn_in_other = np.sum(
        label_concat[membership_ind[n_spikes_target:]] == 1
    )
    return (
        (target_nn_in_target + other_nn_in_other)
        / (n_spikes_target + n_spikes_other)
        / n_neighbors_adjusted
    )


def patch_isolation_kdtree():
    """Use :func:`_compute_isolation_kdtree` over SI's inside the block.

    Both nn metrics look ``_compute_isolation`` up on the module at call time.
    """
    return _patched(
        _pm,
        "_compute_isolation",
        _compute_isolation_kdtree,
        description="k-d tree isolation patch",
    )


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

    SI computes the uncorrected ratio; SI's template correction is then
    applied with ``p`` taken over the given population::

        p = len(template) * n_spikes[unit_id] / total_samples

    Equals SI's corrected value when ``n_spikes`` and ``total_samples`` are
    SI's own.

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
    # Cached by the call above, so this is the std SI divided by.
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
        # Equals SI's unit_std / corrected std; NaN and 0.0 are preserved.
        corrected[unit_id] = (
            ratio * std_noise / np.sqrt(std_noise**2 - template_variance)
        )
    return corrected


def _span_spike_counts(sorting, spans):
    """``{unit_id: int}`` spikes whose frame lies inside a half-open span."""
    np = _mm.np
    spike_vector = sorting.to_spike_vector()
    # A one-sample window [frame, frame + 1) lies in a span iff the frame does.
    inside = frames_with_window_in_one_span(
        spike_vector["sample_index"], spans, n_before=0, n_after=1
    )
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

    Same signature as SI 0.104.3's ``compute_sd_ratio``, and SI's function
    unless spans are set and ``correct_for_template_itself``. With spans the
    noise std comes from span samples, so ``p`` counts the unit's spikes whose
    frame lies in a span over the total span samples.
    """
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


def patch_sd_ratio_statistics_spans():
    """Route SI's ``sd_ratio`` metric through the spans inside the block.

    SI calls ``metric_class.metric_function``, so the ``SDRatio`` attribute is
    replaced; module-level ``compute_sd_ratio`` stays SI's.
    """
    return _patched(
        _mm.SDRatio,
        "metric_function",
        _sd_ratio_statistics_spans,
        description="sd_ratio statistics-span patch",
    )
