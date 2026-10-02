"""Sorter dispatch + post-sort trim behind ``Sorting``.

The two sorter-execution paths plus the post-sort cleanup of
``Sorting.make_compute``: the Spyglass clusterless thresholder
(``run_clusterless_thresholder`` with its ``_clusterless_noise_levels``
precedence), an SpikeInterface registered sorter run under a managed scratch
dir (``run_si_sorter`` -- the MATLAB-container carve-out, the external-float64
whitening pin, the scratch-dir failure cleanup, and the global-job-kwargs
restore), and ``remove_excess_spikes`` which trims spikes outside the recording
window. They operate on SpikeInterface objects and already-fetched parameter
rows; the table threads the fetched DB state in (the tri-part
``make_fetch``/``make_compute``/``make_insert`` contract forbids DB I/O inside
compute), so the hot path here is DB-free. Keeping it out of the ``sorting``
schema module lets it be imported and tested without activating a schema.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: all SpikeInterface / numpy / spyglass dependencies are
imported lazily inside the functions, and none touches the DB at call time.
"""

from __future__ import annotations

from typing import NamedTuple

# MATLAB-backed sorters, whose compiled runtime ships only as a container image:
# a ``backend="local"`` row raises
# (``assert_matlab_sorter_has_container_backend``). Unlike v1, the container
# comes from the tracked ``execution_params``, never from the sorter name. The
# set coincides with
# ``_params.sorter._INTERNAL_WHITEN_NO_KWARG_SORTERS`` but encodes a different
# concept (container policy, not whitening); keep them separate.
MATLAB_SORTERS = ("kilosort2_5", "kilosort3", "ironclust")
# Sorters whose ``whiten=True`` the dispatcher routes to ``pinned_whiten``
# (whitening exactly once). Any other sorter's ``whiten`` passes through
# unchanged. The internal-whitening MATLAB sorters reject a truthy ``whiten`` at
# insert, so they never reach this check with ``whiten=True``.
_EXTERNAL_WHITEN_SORTERS = frozenset({"mountainsort4", "mountainsort5"})


def without_random_seed(job_kwargs) -> dict:
    """Return ``job_kwargs`` without the Spyglass-side ``random_seed`` key.

    ``random_seed`` seeds Spyglass's own pins (whitening, noise levels, random
    spikes); SpikeInterface's job-kwarg handling (``fix_job_kwargs``,
    ``SortingAnalyzer.compute``) rejects it. ``None`` gives ``{}``.
    """
    return {k: v for k, v in (job_kwargs or {}).items() if k != "random_seed"}


def _should_external_whiten(sorter: str, sorter_params: dict) -> bool:
    """Whether the dispatcher should route this sort through the runtime's
    external float64 whitening.

    True only for an external-whitening sorter (MountainSort 4/5) carrying a
    truthy ``whiten``. For any other sorter the ``whiten`` value is left in
    ``sorter_params`` and passed through to the sorter unchanged.
    """
    return sorter.lower() in _EXTERNAL_WHITEN_SORTERS and bool(
        sorter_params.get("whiten", False)
    )


# Sorter kwargs that do not survive containerization of the MATLAB sorters
# (they reference host paths / host process settings). Stripped only when one of
# the MATLAB sorters runs on a container backend.
MATLAB_SORTER_STRIP_KWARGS = (
    "tempdir",
    "mp_context",
    "max_threads_per_process",
)


class EffectiveSortConfig(NamedTuple):
    """What one sort actually executes, resolved once from its parameter row.

    The single description of a sort's effective configuration, shared by the
    dispatcher (``run_si_sorter`` / ``run_clusterless_thresholder``), preflight
    (``PreflightReport.effective_config``) and the run receipt
    (``RunResult["sorter_config"]`` via ``describe_run``), so the three cannot
    disagree about what SpikeInterface receives.

    Attributes
    ----------
    sorter
        SpikeInterface sorter name (or ``clusterless_thresholder``).
    scientific_params
        The row's validated ``params`` blob minus ``schema_version`` -- the
        tracked scientific parameters as the row states them.
    si_sorter_params
        Exactly the kwargs handed to ``sis.run_sorter`` (or to
        ``detect_peaks`` for the clusterless path): ``scientific_params`` with
        ``whiten`` forced to ``False`` when the runtime whitens externally, and
        the MATLAB-container strip applied when relevant.
    external_whiten
        Whether the runtime whitens the recording itself (float64, seeded) and
        turns the sorter's internal whitening off.
    random_seed
        The effective seed consumed by the external whitening / clusterless
        noise sampling (``0`` when unset).
    job_kwargs
        The resolved SpikeInterface job kwargs the stage installs via
        ``set_global_job_kwargs`` (``random_seed`` already removed).
    execution_backend
        ``"local"`` / ``"docker"`` / ``"singularity"``.
    container_image
        The pinned image for a container backend, else ``None``.
    """

    sorter: str
    scientific_params: dict
    si_sorter_params: dict
    external_whiten: bool
    random_seed: int
    job_kwargs: dict
    execution_backend: str
    container_image: str | None

    def as_dict(self) -> dict:
        """Plain-dict form for receipts / reports (JSON-friendly values)."""
        return {
            "sorter": self.sorter,
            "scientific_params": dict(self.scientific_params),
            "si_sorter_params": dict(self.si_sorter_params),
            "external_whiten": bool(self.external_whiten),
            "random_seed": int(self.random_seed),
            "job_kwargs": dict(self.job_kwargs),
            "execution_backend": self.execution_backend,
            "container_image": self.container_image,
        }


def resolve_sort_config(
    sorter: str,
    sorter_params,
    *,
    job_kwargs=None,
    execution_params=None,
) -> EffectiveSortConfig:
    """Resolve the effective configuration of one sort (pure, DB-free).

    Parameters
    ----------
    sorter : str
        Sorter name from the ``SorterParameters`` row.
    sorter_params : Mapping
        The row's validated ``params`` blob (``schema_version`` is dropped).
    job_kwargs : Mapping or None
        The ALREADY-RESOLVED job kwargs (``utils._resolved_job_kwargs`` --
        ambient SI globals + ``dj.config`` + the row blob). ``random_seed`` is
        read out of it and removed from the installed job kwargs.
    execution_params : Mapping or None
        The row's ``execution_params`` blob (``None`` -> local).

    Returns
    -------
    EffectiveSortConfig
    """
    from spyglass.spikesorting.v2._params.sorter import (
        validate_execution_params,
    )

    execution = validate_execution_params(execution_params)
    scientific = {
        k: v
        for k, v in dict(sorter_params or {}).items()
        if k != "schema_version"
    }
    random_seed = int((job_kwargs or {}).get("random_seed", 0))
    resolved_jobs = without_random_seed(job_kwargs)
    external_whiten = _should_external_whiten(sorter, scientific)
    si_params = dict(scientific)
    if external_whiten:
        si_params["whiten"] = False
    if sorter.lower() in MATLAB_SORTERS and is_container_backend(execution):
        si_params = {
            k: v
            for k, v in si_params.items()
            if k not in MATLAB_SORTER_STRIP_KWARGS
        }
    return EffectiveSortConfig(
        sorter=sorter,
        scientific_params=scientific,
        si_sorter_params=si_params,
        external_whiten=external_whiten,
        random_seed=random_seed,
        job_kwargs=resolved_jobs,
        execution_backend=execution["backend"],
        container_image=execution["container_image"],
    )


#: Sorter name -> the installed distribution whose version identifies the
#: producing code. SI-internal sorters (``spykingcircus2`` / ``tridesclous2``)
#: and the in-process ``clusterless_thresholder`` have no separate distribution
#: -- their producing version IS ``spikeinterface.__version__`` (recorded
#: separately) -- so they are deliberately absent and resolve to ``None``.
_SORTER_DISTRIBUTIONS: dict[str, str] = {
    "mountainsort4": "mountainsort4",
    "mountainsort5": "mountainsort5",
    "kilosort4": "kilosort",
}


def sorter_distribution_version(sorter: str) -> str | None:
    """Return the installed distribution version of a sorter package, or None.

    ``None`` for SI-internal sorters (``spykingcircus2`` / ``tridesclous2``),
    the in-process ``clusterless_thresholder``, and any mapped sorter whose
    distribution is not installed. Their producing version is
    ``spikeinterface.__version__``, recorded separately; storing ``None`` rather
    than guessing a version keeps the provenance honest.
    """
    import importlib.metadata

    dist = _SORTER_DISTRIBUTIONS.get(sorter)
    if dist is None:
        return None
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def unit_semantics_for_sorter(sorter: str) -> str:
    """Return the unit semantics a sorter produces.

    ``"clusterless_threshold_crossings"`` for the clusterless thresholder, which
    emits every detected peak as ONE pseudo-unit (a threshold-crossing event
    stream, NOT a sorted neuron); ``"sorted_units"`` for any real spike sorter.
    Derived from ``sorter`` (single source of truth) rather than stored, so it
    cannot drift from the sort row. Consuming surfaces (e.g. cross-session
    matching) use it to avoid treating a threshold-crossing pseudo-unit as a
    trackable neuron.
    """
    return (
        "clusterless_threshold_crossings"
        if sorter == "clusterless_thresholder"
        else "sorted_units"
    )


def is_container_backend(execution_params: dict) -> bool:
    """Return ``True`` when the validated execution row selects a container."""
    return execution_params.get("backend") in ("docker", "singularity")


def matlab_container_required_message(sorter: str) -> str:
    """Build the error message for a MATLAB sorter on a local execution row."""
    return (
        f"Sorter {sorter!r} is a MATLAB-backed sorter (Kilosort 2.5/3, "
        "IronClust) whose runtime ships only as a container image; it cannot "
        "run with execution backend 'local'. Insert a SorterParameters row "
        "whose execution_params selects backend='docker' or 'singularity' with "
        "an explicit container_image. (Unlike v1, v2 never infers a "
        "container from the sorter name: execution comes from tracked "
        "provenance, so a local row for this sorter is never silently "
        "containerized.)"
    )


def assert_matlab_sorter_has_container_backend(
    sorter: str, execution_params: dict
) -> None:
    """Raise if a MATLAB-backed sorter is on a local execution backend.

    The explicit execution-policy check (v1 instead picks Singularity from the
    sorter name). Shared by ``run_si_sorter`` (which raises) and
    ``preflight`` (which surfaces the same message as a failed check) so the two
    cannot drift.
    """
    if sorter.lower() in MATLAB_SORTERS and not is_container_backend(
        execution_params
    ):
        raise ValueError(matlab_container_required_message(sorter))


def build_run_sorter_container_kwargs(execution_params: dict) -> dict:
    """Build the container ``run_sorter`` kwargs from a validated execution row.

    Returns ``{}`` for ``backend="local"`` (no container kwargs at all). For a
    container backend, returns the SpikeInterface ``run_sorter`` execution kwargs:
    ``docker_image`` / ``singularity_image`` set to the explicit
    ``container_image``, ``delete_container_files``, and the container-install
    controls (``installation_mode`` always; ``spikeinterface_version`` only when
    pinned; ``extra_requirements`` only when non-empty). SI's public
    ``run_sorter`` signature names only ``docker_image`` / ``singularity_image`` /
    ``delete_container_files``; the install controls reach ``run_sorter_container``
    through ``**sorter_params``. These originate ONLY here (from
    ``execution_params``), never from the scientific sorter ``params`` blob.

    Parameters
    ----------
    execution_params : dict
        A validated ``SorterExecutionParamsSchema`` dump.

    Returns
    -------
    dict
        The container-execution kwargs to splat into ``sis.run_sorter`` (empty
        for local).
    """
    if not is_container_backend(execution_params):
        return {}
    backend = execution_params["backend"]
    image = execution_params["container_image"]
    image_key = "docker_image" if backend == "docker" else "singularity_image"
    kwargs = {
        image_key: image,
        "delete_container_files": execution_params["delete_container_files"],
        # ``installation_mode`` is meaningful for every container backend
        # (including "auto"); pass it explicitly so the run records the choice.
        "installation_mode": execution_params["installation_mode"],
    }
    if execution_params.get("spikeinterface_version") is not None:
        kwargs["spikeinterface_version"] = execution_params[
            "spikeinterface_version"
        ]
    if execution_params.get("extra_requirements"):
        kwargs["extra_requirements"] = execution_params["extra_requirements"]
    return kwargs


#: Random-chunk budget of the span sampler: SpikeInterface's
#: ``get_random_recording_slices`` defaults, 20 chunks of 500 ms per segment
#: (``spikeinterface/core/recording_tools.py:460-468``). The motion estimate
#: records this budget in its resolved configuration.
STATISTICS_SAMPLE_NUM_CHUNKS = 20
STATISTICS_SAMPLE_CHUNK_MS = 500


def _sample_statistics_spans(recording, spans, *, seed, return_in_uV: bool):
    """Random traces from inside ``spans`` with SI's default sample budget.

    SpikeInterface's ``get_random_recording_slices`` defaults to
    :data:`STATISTICS_SAMPLE_NUM_CHUNKS` chunks of
    :data:`STATISTICS_SAMPLE_CHUNK_MS` per segment; the same total budget
    (``20 * int(0.5 * fs)`` rows) in pieces of at most one such chunk is drawn
    here, so one full span reproduces SI's ``get_random_data_chunks`` rows
    exactly.

    Returns
    -------
    numpy.ndarray
        ``(n_rows, n_channels)`` traces, ``n_rows`` at most the budget.
    """
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        sample_span_data,
    )

    chunk = int(
        STATISTICS_SAMPLE_CHUNK_MS / 1000 * recording.get_sampling_frequency()
    )
    return sample_span_data(
        recording,
        spans,
        target_samples=STATISTICS_SAMPLE_NUM_CHUNKS * chunk,
        max_piece=chunk,
        seed=seed,
        return_in_uV=return_in_uV,
    )


def _span_whitening_matrix(recording, spans, *, random_seed: int):
    """SI's ``mode="global"`` whitening matrix from samples inside ``spans``.

    Mirrors ``spikeinterface.preprocessing.whiten.compute_whitening_matrix``
    with ``apply_mean=False`` / ``regularize=False`` / ``eps=None`` (the
    ``sip.whiten`` defaults ``pinned_whiten`` relies on), replacing only the
    sampler: float32 data, uncentered covariance ``data.T @ data / n_rows``,
    SI's data-dependent ``eps``, and SI's own ZCA
    ``compute_whitening_from_covariance``. On one span covering the recording
    the result equals SI's ``W`` exactly.

    Returns
    -------
    numpy.ndarray
        ``(n_channels, n_channels)`` float32 whitening matrix.
    """
    import numpy as np
    from spikeinterface.preprocessing.whiten import (
        compute_whitening_from_covariance,
    )

    data = _sample_statistics_spans(
        recording, spans, seed=random_seed, return_in_uV=False
    ).astype(np.float32)
    cov = data.T @ data
    cov = cov / data.shape[0]
    # SI 0.104.3 ``compute_whitening_matrix`` eps rule for ``eps=None``
    # (spikeinterface/preprocessing/whiten.py:200-205; its docstring's 1e-8 is
    # not what the code does).
    median_data_sqr = np.median(data**2)
    if 0 < median_data_sqr < 1:
        eps = max(1e-16, median_data_sqr * 1e-3)
    else:
        eps = 1e-16
    return compute_whitening_from_covariance(cov, eps)


def pinned_whiten(recording, *, random_seed: int = 0, spans=None):
    """SI external float64 whitening with a pinned covariance seed.

    The single whitening implementation shared by the sorter's external-whiten
    path (``run_si_sorter``) and the whitened metric analyzer build
    (``build_analyzer`` for a ``whiten=True`` recipe). SI PR #3359 (2024-10-25)
    changed ``sip.whiten``'s default from ``seed=0`` to ``seed=None``, making
    the whitening matrix non-deterministic across runs on the same input, so
    Spyglass is the explicit seeder; ``dtype=float64`` matches the sorter path.

    Reusing it for the metric analyzer is NOT a claim that it matches what each
    sorter saw -- MS4/MS5 use this same external whiten, but KS4 whitens
    internally and the clusterless thresholder does not whiten at all. It is
    only the lab's "whiten for cluster-separation (PC/NN) metrics" applied
    consistently, with one seeded implementation rather than two.

    ``spans`` are the statistics spans: half-open frame ranges of artifact-free
    samples that never cross a recording join. Artifact-masked frames are
    zeros, and zeros counted as data shrink the covariance and over-whiten the
    retained signal, so the covariance is estimated only from samples inside
    ``spans`` (same seed, same sample budget and same math as SI's
    ``mode="global"`` whitening). When ``spans`` is ``None`` or one span
    covering the whole recording, SpikeInterface's own whitening runs
    unchanged, so an unmasked continuous recording whitens bit-identically to
    SI.

    Parameters
    ----------
    recording : spikeinterface.BaseRecording
        The recording to whiten.
    random_seed : int, optional
        Seed for the random-chunk covariance estimate. Default 0 (the
        per-row ``job_kwargs={"random_seed": N}`` override flows in here).
    spans : list[tuple[int, int]] or None, optional
        Keyword-only. Statistics spans in ``recording``'s frame coordinates.
        Default ``None`` (the whole recording).
    """
    import numpy as np
    import spikeinterface.preprocessing as sip

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        spans_cover_recording,
    )

    if spans is None or spans_cover_recording(
        spans, recording.get_num_samples()
    ):
        return sip.whiten(recording, dtype=np.float64, seed=random_seed)
    whitening = _span_whitening_matrix(
        recording, spans, random_seed=random_seed
    )
    return sip.whiten(recording, dtype=np.float64, W=whitening, M=None)


def cache_span_noise_levels(
    recording, spans, *, return_in_uV: bool, seed, method: str = "mad"
):
    """Cache per-channel noise levels estimated from ``spans`` only.

    SpikeInterface's ``get_noise_levels`` returns a cached
    ``noise_level_{method}_scaled`` (``return_in_uV=True``) or
    ``noise_level_{method}_raw`` (``False``) recording property whenever one
    is present, whatever its other arguments. Setting that property here
    makes every downstream SI noise estimate of that method on this exact
    object -- the analyzer ``noise_levels`` extension and ``detect_peaks``'s
    MAD threshold (``"mad"``), the ``sd_ratio`` metric's noise (``"std"``)
    -- use samples inside the statistics spans, so artifact-masked zeros do
    not bias the noise low. Preprocessors drop the ``noise_level_*``
    properties, so call this on the final object the consumer receives.

    Samples are drawn with SI's default 20 x 500 ms budget. ``"mad"`` is the
    MAD of the pooled span samples (``median(|x - median(x)|) /
    0.6744897501960817``, SI's scale constant); SI's own estimator instead
    averages per-chunk MADs. ``"std"`` is the population standard deviation
    of the pooled span samples about their pooled mean; SI's own estimator
    instead averages per-chunk standard deviations, each about its chunk's
    own mean.

    Parameters
    ----------
    recording : spikeinterface.BaseRecording
        The exact object whose noise SI will read. Mutated: the property is
        set on it (as SI's own ``get_noise_levels`` would).
    spans : list[tuple[int, int]] or None
        Statistics spans in ``recording``'s frame coordinates. ``None`` or
        one span covering the recording leaves SI's own estimator in charge
        (nothing is cached).
    return_in_uV : bool
        Keyword-only. Estimate on microvolt (``True``) or raw (``False``)
        traces; selects the property key to match the consumer.
    seed : int
        Keyword-only. Seed of the span sampler.
    method : {"mad", "std"}, optional
        Keyword-only. The SI noise method to cache. Default ``"mad"``.

    Returns
    -------
    numpy.ndarray or None
        ``(n_channels,)`` cached noise levels, or ``None`` when nothing was
        cached.
    """
    import numpy as np
    from scipy.stats import median_abs_deviation

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        spans_cover_recording,
    )

    if method not in ("mad", "std"):
        raise ValueError(f"method must be 'mad' or 'std', got {method!r}")
    if spans is None or spans_cover_recording(
        spans, recording.get_num_samples()
    ):
        return None
    data = _sample_statistics_spans(
        recording, spans, seed=seed, return_in_uV=return_in_uV
    )
    if method == "mad":
        noise_levels = median_abs_deviation(data, axis=0, scale="normal")
    else:
        noise_levels = np.std(data, axis=0)
    suffix = "scaled" if return_in_uV else "raw"
    recording.set_property(f"noise_level_{method}_{suffix}", noise_levels)
    return noise_levels


def _clusterless_noise_levels(
    noise_levels: list[float] | None, threshold_unit: str
) -> list[float] | None:
    """Resolve clusterless-thresholder ``noise_levels`` precedence.

    An explicit ``noise_levels`` always wins. Otherwise ``threshold_unit``
    governs how ``detect_threshold`` is interpreted:

    * ``"uv"`` -> ``[1.0]``; the caller (``run_clusterless_thresholder``)
      scales the recording to microvolts (``scale_to_uV``) before
      ``detect_peaks``, so ``detect_threshold`` is a genuine microvolt
      threshold. (For Frank-lab data gain==1 uV/count, so this equals a
      raw-count threshold; for non-unity-gain rigs it converts the threshold
      to true microvolts.)
    * ``"mad"`` -> ``None`` so SpikeInterface estimates per-channel MAD
      and ``detect_threshold`` is a MAD multiplier (scale-relative, so the
      recording is NOT uV-scaled on this path).
    """
    if noise_levels is not None:
        return noise_levels
    return [1.0] if threshold_unit == "uv" else None


def run_clusterless_thresholder(
    sorter_params,
    recording,
    job_kwargs,
    statistics_spans=None,
):
    """Run Spyglass's clusterless-thresholder peak-detection path.

    Not an SI registered sorter; uses
    ``spikeinterface.sortingcomponents.peak_detection.detect_peaks``
    directly and wraps the result in a ``NumpySorting``.

    The per-channel threshold is ``noise_levels[chan] * detect_threshold``
    in the recording's amplitude units. ``threshold_unit="uv"`` (the schema
    default and the production Frank-lab row) scales the recording to
    microvolts with its stored NWB gain (a no-op at 1 uV/count). An explicit
    ``noise_levels`` wins (a singleton is broadcast to ``n_channels``);
    otherwise (:func:`_clusterless_noise_levels`):

    - ``"uv"``: ``[1.0]``, so ``detect_threshold`` is in microvolts.
    - ``"mad"``: SI estimates per-channel MAD and ``detect_threshold`` is a
      MAD multiplier. The MAD is estimated only from samples inside
      ``statistics_spans`` (:func:`cache_span_noise_levels`), so
      artifact-masked zeros do not lower the threshold; without spans, or
      with one span covering the recording, SI's own estimator runs.

    Parameters
    ----------
    sorter_params : dict
        The fetched clusterless-thresholder params row (detect kwargs
        plus the Spyglass-side ``threshold_unit`` knob).
    recording : si.BaseRecording
        The artifact-masked recording to detect peaks on; scaled to
        microvolts on the ``threshold_unit="uv"`` path.
    job_kwargs : dict or None
        Resolved SI job kwargs; the Spyglass-side ``random_seed`` is
        stripped before the ``detect_peaks`` call.
    statistics_spans : list[tuple[int, int]] or None, optional
        Artifact-free frame spans of ``recording`` for the MAD noise
        estimate. Ignored when ``noise_levels`` is explicit or derived
        (``"uv"``). Default ``None`` (the whole recording).

    Returns
    -------
    spikeinterface.NumpySorting
        A single-unit sorting wrapping the detected peak sample
        indices.
    """
    import numpy as np
    import spikeinterface as si
    from spikeinterface.sortingcomponents.peak_detection import (
        detect_peaks,
    )

    from spyglass.spikesorting.v2.utils import _assert_noise_levels_length

    params = dict(sorter_params)
    # SI kwarg rename: SI 0.99 ``local_radius_um`` became
    # ``radius_um`` in 0.101+.
    if "local_radius_um" in params:
        params["radius_um"] = params.pop("local_radius_um")
    # SI 0.104 ``detect_peaks`` rejects stale routing hints
    # via the new ``(method, method_kwargs, job_kwargs)`` shape:
    # ``outputs`` was a Spyglass-only routing hint, and
    # ``random_chunk_kwargs`` was renamed to
    # ``random_slices_kwargs`` and is now managed internally.
    for stale in ("outputs", "random_chunk_kwargs"):
        params.pop(stale, None)

    # ``threshold_unit`` is a Spyglass-side knob, not a detect_peaks kwarg.
    # A row without it (``schema_version`` < 4) falls back to the schema
    # default "uv", so it is scaled to microvolts rather than thresholded in
    # native counts (on Intan 0.195 uV/count, "100" would be ~19.5 uV).
    threshold_unit = params.pop("threshold_unit", "uv")
    # The schema enforces ``Literal["uv", "mad"]`` at insert, but ``update1``
    # and pre-validator rows bypass it and the fetched blob is not
    # re-validated; any non-"uv" value would otherwise silently take the MAD
    # path.
    if threshold_unit not in ("uv", "mad"):
        raise ValueError(
            "clusterless_thresholder: threshold_unit must be 'uv' or "
            f"'mad', got {threshold_unit!r}. A params row written via "
            "update1 or before the SorterParameters validator existed can "
            "carry an invalid value; fix the stored params row."
        )
    # Same bypass as above for the insert validator's MAD-multiplier check:
    # a microvolt-scale threshold read as a MAD multiplier (e.g. 100x MAD)
    # detects almost nothing -- a silent zero-unit sort.
    from spyglass.spikesorting.v2._params.sorter import (
        _MAX_PLAUSIBLE_MAD_MULTIPLIER,
    )

    if (
        threshold_unit == "mad"
        and params.get("noise_levels") is None
        and float(params.get("detect_threshold", 0.0))
        > _MAX_PLAUSIBLE_MAD_MULTIPLIER
    ):
        raise ValueError(
            "clusterless_thresholder: detect_threshold="
            f"{params.get('detect_threshold')} with threshold_unit='mad' "
            "and no noise_levels is an implausibly large MAD multiplier "
            f"(> {_MAX_PLAUSIBLE_MAD_MULTIPLIER}); SpikeInterface would "
            "estimate per-channel MAD and detect almost nothing. Use "
            "threshold_unit='uv' for a microvolt/native threshold, or a "
            "MAD multiplier near 5. (This row likely bypassed the "
            "SorterParameters insert validator via update1 or predates it.)"
        )
    nl_in = _clusterless_noise_levels(
        params.get("noise_levels"), threshold_unit
    )
    if nl_in is None:
        # Absent noise_levels -> detect_peaks estimates per-channel MAD.
        params.pop("noise_levels", None)
        # SI PR #3359 (2024-10-25) made ``get_noise_levels``'s default seed
        # None, so an unpinned MAD varies between runs and can flip ~10-20
        # borderline peaks per shank at a MAD multiplier of 5. Pin it, as
        # ``pinned_whiten`` does; override via the row's
        # ``job_kwargs["random_seed"]``.
        _random_seed = (job_kwargs or {}).get("random_seed", 0)
        params.setdefault("random_slices_kwargs", {"seed": _random_seed})
    else:
        n_channels = recording.get_num_channels()
        # Only a singleton or an n_channels-length array is valid; any other
        # length would mis-index inside SI's ``locally_exclusive``.
        _assert_noise_levels_length(nl_in, n_channels)
        nl = np.asarray(nl_in, dtype=np.float64)
        if nl.size == 1:
            nl = np.full(n_channels, float(nl[0]), dtype=np.float64)
        params["noise_levels"] = nl

    method = params.pop("method", "locally_exclusive")
    detect_job_kwargs = without_random_seed(job_kwargs)
    if threshold_unit == "uv":
        # The stored gain/offset is the NWB ElectricalSeries conversion/offset
        # that se.read_nwb_recording loads onto the recording.
        import spikeinterface.preprocessing as sip

        if recording.get_channel_gains() is None:
            raise ValueError(
                "clusterless_thresholder threshold_unit='uv' requires the "
                "recording to carry channel gains (the NWB ElectricalSeries "
                "conversion); none are set. Use threshold_unit='mad' for a "
                "gain-free relative threshold."
            )
        recording = sip.scale_to_uV(recording)
    if nl_in is None:
        # SI's ``detect_peaks`` asks ``get_noise_levels(recording,
        # return_in_uV=False, ...)`` for the MAD, which returns a cached
        # ``noise_level_mad_raw`` property when present -- so the span MAD is
        # cached on the exact object handed to ``detect_peaks``.
        cache_span_noise_levels(
            recording,
            statistics_spans,
            return_in_uV=False,
            seed=_random_seed,
        )
    detected = detect_peaks(
        recording,
        method=method,
        method_kwargs=params,
        job_kwargs=(detect_job_kwargs or None),
    )
    # SI 0.104 renamed ``from_times_labels`` to
    # ``from_samples_and_labels`` (sample indices); ``detect_peaks``
    # already returns sample indices.
    return si.NumpySorting.from_samples_and_labels(
        samples_list=detected["sample_index"],
        labels_list=np.zeros(len(detected), dtype=np.int32),
        sampling_frequency=recording.get_sampling_frequency(),
    )


def run_si_sorter(
    sorter,
    sorter_params,
    recording,
    sorting_id,
    job_kwargs,
    execution_params=None,
    statistics_spans=None,
):
    """Run an SI registered sorter under a managed scratch dir.

    Scratch is anchored under ``spyglass.settings.temp_dir`` via
    ``tempfile.TemporaryDirectory`` so the dir is cleaned on
    successful exit AND on raise (no scratch dir is leaked).
    For a CONTAINER backend the scratch is ``os.chmod 0o777`` so an SI
    sorter subprocess with a different uid (rootless container, slurm
    scenarios) can write into it; a local run keeps the 0o700 default.

    External float64 whitening: for an external-whitening sorter asking for
    whitening, the runtime whitens at float64 and turns the sorter's own
    whitening off, so the signal is whitened once. The covariance is
    estimated only from samples inside ``statistics_spans`` (see
    ``pinned_whiten``), so artifact-masked zeros never enter it.

    Container execution: the ``execution_params`` row selects the backend.
    A container backend passes the pinned image and install controls
    (``build_run_sorter_container_kwargs``). MATLAB-backed sorters must
    select a container backend (a local row raises), and
    ``MATLAB_SORTER_STRIP_KWARGS`` are stripped when one runs in a
    container.

    Parameters
    ----------
    sorter : str
        SpikeInterface registered sorter name (e.g.
        ``"mountainsort4"``).
    sorter_params : dict
        The fetched sorter params row passed to ``run_sorter``; a
        truthy ``whiten`` is moved to the external float64 path.
    recording : si.BaseRecording
        The artifact-masked recording to sort.
    sorting_id : str
        Sorting identifier; used to name the managed scratch dir.
    job_kwargs : dict or None
        Resolved SI job kwargs installed via
        ``set_global_job_kwargs``; ``random_seed`` is stripped first.
    execution_params : dict or None, optional
        The validated ``SorterExecutionParamsSchema`` dump from
        ``Sorting.make_fetch``. ``None`` resolves to default local execution.
    statistics_spans : list[tuple[int, int]] or None, optional
        Artifact-free frame spans of ``recording`` that the external
        whitening's covariance is estimated from. ``None`` (default) means
        the whole recording.

    Returns
    -------
    spikeinterface.BaseSorting
        The sorter output as returned by ``run_sorter``.
    """
    import os
    import tempfile

    import numpy as np
    import spikeinterface as si
    import spikeinterface.sorters as sis

    from spyglass.settings import temp_dir as spyglass_temp_dir
    from spyglass.spikesorting.v2._params.sorter import (
        validate_execution_params,
    )
    from spyglass.utils import logger

    # Enforce the MATLAB container policy before any scratch/whitening work.
    execution_params = validate_execution_params(execution_params)
    assert_matlab_sorter_has_container_backend(sorter, execution_params)
    container_kwargs = build_run_sorter_container_kwargs(execution_params)
    # The same resolution preflight and the run receipt report.
    config = resolve_sort_config(
        sorter,
        sorter_params,
        job_kwargs=job_kwargs,
        execution_params=execution_params,
    )

    sorter_temp_dir = tempfile.TemporaryDirectory(
        prefix=f"sort_{sorting_id}_",
        dir=spyglass_temp_dir,
    )
    patched_numpy_inf = False
    try:
        # A container process may run as a different uid; a local run keeps
        # TemporaryDirectory's 0o700.
        if is_container_backend(execution_params):
            os.chmod(sorter_temp_dir.name, 0o777)

        # spikeextractors 0.9.11 (pulled in by SI's MS4 wrapper) references
        # ``numpy.Inf``, removed in numpy 2. Restore the alias only for this
        # call (the ``finally`` deletes it) so later ``hasattr(np, "Inf")``
        # probes (some scipy versions) see the same numpy as at import.
        # TODO: drop once spikeextractors stops referencing the alias.
        if sorter.lower() == "mountainsort4" and not hasattr(np, "Inf"):
            np.Inf = np.inf
            patched_numpy_inf = True

        if config.external_whiten:
            # Seeded covariance (default 0, override via the row's
            # ``job_kwargs["random_seed"]``): three seeded MS4 runs gave
            # identical (n_units, median_fr); four unseeded runs gave four
            # different results. ``config.si_sorter_params`` already carries
            # ``whiten=False``.
            recording = pinned_whiten(
                recording,
                random_seed=config.random_seed,
                spans=statistics_spans,
            )

        # Job kwargs reach run_sorter through SI's global state. Splatting them
        # into run_sorter would route them into ``**sorter_params``, which
        # MS4/MS5/KS4 reject (``Invalid parameters: [...]``).
        # ``resolve_sort_config`` already removed ``random_seed``, which
        # ``set_global_job_kwargs`` rejects.
        sj_kwargs = dict(config.job_kwargs)
        previous_global = dict(si.get_global_job_kwargs())
        if sj_kwargs:
            si.set_global_job_kwargs(**sj_kwargs)
        # A CHILD output folder makes SI's container runner write its
        # fixed-name ``in_container_*`` files into the per-sort (chmod-ed)
        # temp dir, not the shared ``spyglass_temp_dir`` where parallel
        # container populates would overwrite each other's files.
        output_folder = os.path.join(sorter_temp_dir.name, "sorter_output")
        run_kwargs = dict(
            sorter_name=sorter,
            recording=recording,
            folder=output_folder,
            remove_existing_folder=True,
            # Empty for local. The reserved-key rule keeps these out of the
            # scientific params, so they cannot collide with
            # **effective_params.
            **container_kwargs,
        )
        effective_params = config.si_sorter_params
        try:
            raw_sorting = sis.run_sorter(**run_kwargs, **effective_params)
            # The returned sorting reads from sorter_temp_dir, which the
            # finally below deletes; copy the spike trains into memory first.
            # with_metadata=True keeps unit properties (SI 0.104.3 defaults to
            # False); copy_spike_vector=True avoids aliasing a memmap into the
            # temp dir.
            return si.NumpySorting.from_sorting(
                raw_sorting, with_metadata=True, copy_spike_vector=True
            )
        finally:
            if sj_kwargs:
                # ``set_global_job_kwargs`` updates rather than replaces, so a
                # key absent from the prior global (e.g. chunk_size) would
                # leak into later populates: reset, then re-apply. A restore
                # failure is logged so it cannot mask a sort exception.
                try:
                    si.reset_global_job_kwargs()
                    si.set_global_job_kwargs(**previous_global)
                except Exception as restore_exc:
                    logger.warning(
                        "Sorting._run_si_sorter: failed to restore SI "
                        f"global job kwargs to {previous_global!r}: "
                        f"{restore_exc!r}. Original sort exception (if "
                        "any) preserved."
                    )
    finally:
        if patched_numpy_inf and hasattr(np, "Inf"):
            del np.Inf
        # Explicit cleanup (not garbage collection) is predictable in pool
        # workers; a cleanup failure (e.g. a stale network-FS lock) is logged
        # so it cannot replace the sort's own exception.
        try:
            sorter_temp_dir.cleanup()
        except Exception as cleanup_exc:
            logger.warning(
                "Sorting._run_si_sorter: sorter_temp_dir cleanup "
                f"failed for sorting_id={sorting_id}: {cleanup_exc!r}. "
                "Original sort exception (if any) preserved."
            )


def remove_excess_spikes(sorting, recording):
    """Drop spikes whose sample index is outside the recording window, then
    drop every unit with zero spikes (in every segment).

    The dropped units are those the sorter already returned empty and those
    whose only spikes fell outside the recording window. Either would
    otherwise survive as a spurious zero-spike unit -- inflating ``n_units``
    and appearing in the units NWB with an empty spike train. Dropping it
    here means ``n_units`` and the persisted units NWB never include it.

    Parameters
    ----------
    sorting : spikeinterface.BaseSorting
        The sorter output to trim.
    recording : si.BaseRecording
        The recording whose sample window bounds valid spike indices.

    Returns
    -------
    spikeinterface.BaseSorting
        The sorting with out-of-window spikes removed and every
        zero-spike unit dropped.
    """
    import numpy as np
    import spikeinterface.curation as sic

    from spyglass.utils import logger

    trimmed = sic.remove_excess_spikes(sorting, recording)
    before_ids = trimmed.unit_ids
    result = trimmed.remove_empty_units()
    dropped = np.setdiff1d(before_ids, result.unit_ids)
    if dropped.size:
        logger.info(
            "Sorting._remove_excess_spikes: dropped unit(s) "
            f"{dropped.tolist()} with zero spikes (empty from the sorter or "
            "emptied by removing spikes outside the recording window)."
        )
    return result
