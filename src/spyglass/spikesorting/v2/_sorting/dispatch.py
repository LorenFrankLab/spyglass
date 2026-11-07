"""Sorter dispatch and post-sort trim behind ``Sorting``."""

from __future__ import annotations

from typing import NamedTuple

# Container-only sorters; the container comes from execution_params.
MATLAB_SORTERS = ("kilosort2_5", "kilosort3", "ironclust")
# Sorters whose whiten=True is routed to pinned_whiten.
_EXTERNAL_WHITEN_SORTERS = frozenset({"mountainsort4", "mountainsort5"})


def without_random_seed(job_kwargs) -> dict:
    """Return ``job_kwargs`` without the Spyglass-side ``random_seed`` key.

    ``random_seed`` seeds Spyglass's own pins (whitening, noise levels, random
    spikes); SpikeInterface's job-kwarg handling (``fix_job_kwargs``,
    ``SortingAnalyzer.compute``) rejects it. ``None`` gives ``{}``.
    """
    return {k: v for k, v in (job_kwargs or {}).items() if k != "random_seed"}


def _should_external_whiten(sorter: str, sorter_params: dict) -> bool:
    """Whether to whiten externally at float64 instead of in the sorter."""
    return sorter.lower() in _EXTERNAL_WHITEN_SORTERS and bool(
        sorter_params.get("whiten", False)
    )


# Host-only kwargs stripped when a MATLAB sorter runs in a container.
MATLAB_SORTER_STRIP_KWARGS = (
    "tempdir",
    "mp_context",
    "max_threads_per_process",
)


class EffectiveSortConfig(NamedTuple):
    """What one sort actually executes, resolved once from its parameter row.

    Shared by the dispatcher, preflight and the run receipt.

    Attributes
    ----------
    sorter
        SpikeInterface sorter name (or ``clusterless_thresholder``).
    scientific_params
        The row's validated ``params`` blob minus ``schema_version``.
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
        Already-resolved job kwargs. ``random_seed`` is read out and
        removed.
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


#: Sorter -> distribution. Absent sorters are versioned by SpikeInterface.
_SORTER_DISTRIBUTIONS: dict[str, str] = {
    "mountainsort4": "mountainsort4",
    "mountainsort5": "mountainsort5",
    "kilosort4": "kilosort",
}


def sorter_distribution_version(sorter: str) -> str | None:
    """Return the installed distribution version of a sorter package, or None.

    ``None`` for sorters without their own distribution or not installed.
    """
    import importlib.metadata

    dist = _SORTER_DISTRIBUTIONS.get(sorter)
    if dist is None:
        return None
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def sort_runtime_versions(sorting, sorter, execution_params=None):
    """Producing runtime versions; container runs never fall back to the host."""
    import spikeinterface as si

    from spyglass.spikesorting.v2._sorting.container import RUNTIME_ANNOTATION

    if is_container_backend(execution_params):
        runtime = sorting.get_annotation(RUNTIME_ANNOTATION)
        if not isinstance(runtime, dict) or not runtime.get(
            "spikeinterface_version"
        ):
            raise RuntimeError(
                "Container sorting has no producing-runtime provenance."
            )
        return runtime["spikeinterface_version"], runtime.get("sorter_version")
    return si.__version__, sorter_distribution_version(sorter)


def unit_semantics_for_sorter(sorter: str) -> str:
    """Return the unit semantics a sorter produces.

    ``"clusterless_threshold_crossings"`` for the clusterless thresholder,
    whose single pseudo-unit is not a neuron; ``"sorted_units"`` otherwise.
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

    Shared by ``run_si_sorter`` and preflight.
    """
    if sorter.lower() in MATLAB_SORTERS and not is_container_backend(
        execution_params
    ):
        raise ValueError(matlab_container_required_message(sorter))


def build_run_sorter_container_kwargs(execution_params: dict) -> dict:
    """Build the container ``run_sorter`` kwargs from a validated execution row.

    Returns ``{}`` for ``backend="local"``. These kwargs come only from
    ``execution_params``, never from the scientific ``params`` blob.

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
        "installation_mode": execution_params["installation_mode"],
    }
    if execution_params.get("spikeinterface_version") is not None:
        kwargs["spikeinterface_version"] = execution_params[
            "spikeinterface_version"
        ]
    if execution_params.get("extra_requirements"):
        kwargs["extra_requirements"] = execution_params["extra_requirements"]
    return kwargs


#: SI's get_random_recording_slices default budget: 20 chunks of 500 ms.
STATISTICS_SAMPLE_NUM_CHUNKS = 20
STATISTICS_SAMPLE_CHUNK_MS = 500


def _sample_statistics_spans(recording, spans, *, seed, return_in_uV: bool):
    """Random traces from inside ``spans`` with SI's default sample budget.

    One full span reproduces SI's ``get_random_data_chunks`` rows exactly.

    Returns
    -------
    numpy.ndarray
        ``(n_rows, n_channels)`` traces, ``n_rows`` at most the budget.
    """
    from spyglass.spikesorting.v2._sorting.artifact_mask import sample_span_data

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

    Mirrors SI's ``compute_whitening_matrix`` with its ``sip.whiten``
    defaults, replacing only the sampler. On one span covering the recording
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
    # SI 0.104.3's eps=None rule (its docstring's 1e-8 is wrong).
    median_data_sqr = np.median(data**2)
    if 0 < median_data_sqr < 1:
        eps = max(1e-16, median_data_sqr * 1e-3)
    else:
        eps = 1e-16
    return compute_whitening_from_covariance(cov, eps)


def pinned_whiten(recording, *, random_seed: int = 0, spans=None):
    """SI external float64 whitening with a pinned covariance seed.

    Shared by the sorter's external-whiten path and the whitened metric
    analyzer. ``sip.whiten`` defaults to ``seed=None`` (non-deterministic),
    so the seed is explicit. The covariance uses only samples inside
    ``spans``, since artifact-masked zeros would over-whiten. Without spans,
    or with one covering the recording, this is SI's own whitening.

    Parameters
    ----------
    recording : spikeinterface.BaseRecording
        The recording to whiten.
    random_seed : int, optional
        Seed for the random-chunk covariance estimate. Default 0.
    spans : list[tuple[int, int]] or None, optional
        Keyword-only. Statistics spans in ``recording``'s frame coordinates.
        Default ``None`` (the whole recording).
    """
    import numpy as np
    import spikeinterface.preprocessing as sip

    from spyglass.spikesorting.v2._sorting.artifact_mask import (
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

    SI's ``get_noise_levels`` returns a cached ``noise_level_{method}_*``
    property when present, so every downstream estimate on this object
    avoids artifact-masked zeros. Preprocessors drop the property, so call
    this on the final object. Unlike SI, which averages per-chunk values,
    this pools the span samples.

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

    from spyglass.spikesorting.v2._sorting.artifact_mask import (
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

    An explicit ``noise_levels`` wins. Otherwise ``"uv"`` gives ``[1.0]``
    (the recording is scaled to µV) and ``"mad"`` gives ``None`` (SI
    estimates per-channel MAD).
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

    Wraps SI's ``detect_peaks`` in a ``NumpySorting``. The per-channel
    threshold is ``noise_levels[chan] * detect_threshold``; see
    :func:`_clusterless_noise_levels`. A singleton ``noise_levels`` is
    broadcast. The MAD is estimated from ``statistics_spans`` only.

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

    from spyglass.spikesorting.v2._core.signal_math import (
        _assert_noise_levels_length,
    )

    params = dict(sorter_params)
    # SI >= 0.101 calls it radius_um.
    if "local_radius_um" in params:
        params["radius_um"] = params.pop("local_radius_um")
    # ``threshold_unit`` is a Spyglass-side knob, not a detect_peaks kwarg.
    threshold_unit = params.pop("threshold_unit")
    # update1 bypasses the insert validator.
    if threshold_unit not in ("uv", "mad"):
        raise ValueError(
            "clusterless_thresholder: threshold_unit must be 'uv' or "
            f"'mad', got {threshold_unit!r}. A params row edited via "
            "update1 bypasses the SorterParameters validator; fix the "
            "stored params row."
        )
    # A µV threshold read as a MAD multiplier silently detects nothing.
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
            "SorterParameters insert validator via update1.)"
        )
    nl_in = _clusterless_noise_levels(
        params.get("noise_levels"), threshold_unit
    )
    if nl_in is None:
        params.pop("noise_levels", None)
        # SI's default seed is None; an unpinned MAD flips borderline peaks.
        _random_seed = (job_kwargs or {}).get("random_seed", 0)
        params.setdefault("random_slices_kwargs", {"seed": _random_seed})
    else:
        n_channels = recording.get_num_channels()
        _assert_noise_levels_length(nl_in, n_channels)
        nl = np.asarray(nl_in, dtype=np.float64)
        if nl.size == 1:
            nl = np.full(n_channels, float(nl[0]), dtype=np.float64)
        params["noise_levels"] = nl

    method = params.pop("method", "locally_exclusive")
    detect_job_kwargs = without_random_seed(job_kwargs)
    if threshold_unit == "uv":
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
        # detect_peaks reads the cached noise_level_mad_raw property.
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

    - Scratch lives under ``spyglass.settings.temp_dir`` and is removed on
      success or failure.
    - External-whitening sorters are whitened once, at float64, by
      :func:`pinned_whiten`.
    - ``execution_params`` selects local or container execution; MATLAB
      sorters require a container.

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

    import spikeinterface as si
    import spikeinterface.sorters as sis

    from spyglass.settings import temp_dir as spyglass_temp_dir
    from spyglass.spikesorting.v2._params.sorter import (
        validate_execution_params,
    )
    from spyglass.utils import logger

    execution_params = validate_execution_params(execution_params)
    assert_matlab_sorter_has_container_backend(sorter, execution_params)
    container_kwargs = build_run_sorter_container_kwargs(execution_params)
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
    try:
        # A container process may run as a different uid.
        if is_container_backend(execution_params):
            os.chmod(sorter_temp_dir.name, 0o777)

        from spyglass.spikesorting.v2._core.si_compat import (
            sorter_runtime_state,
        )

        with sorter_runtime_state(sorter, config.job_kwargs):
            if config.external_whiten:
                recording = pinned_whiten(
                    recording,
                    random_seed=config.random_seed,
                    spans=statistics_spans,
                )

            # Job kwargs go via SI globals (MS4/MS5/KS4 reject them as kwargs).
            # A child folder keeps container files out of the shared temp dir.
            output_folder = os.path.join(sorter_temp_dir.name, "sorter_output")
            run_kwargs = dict(
                sorter_name=sorter,
                recording=recording,
                folder=output_folder,
                remove_existing_folder=True,
                **container_kwargs,
            )
            effective_params = config.si_sorter_params
            if is_container_backend(execution_params):
                from spyglass.spikesorting.v2._sorting.container import (
                    run_sorter_container,
                )

                raw_sorting = run_sorter_container(
                    **run_kwargs, **effective_params
                )
            else:
                raw_sorting = sis.run_sorter(**run_kwargs, **effective_params)
            # Copy into memory before the finally deletes the temp dir.
            return si.NumpySorting.from_sorting(
                raw_sorting, with_metadata=True, copy_spike_vector=True
            )
    finally:
        # Logged so a cleanup failure cannot replace the sort's exception.
        try:
            sorter_temp_dir.cleanup()
        except Exception as cleanup_exc:
            logger.warning(
                "run_si_sorter: sorter_temp_dir cleanup "
                f"failed for sorting_id={sorting_id}: {cleanup_exc!r}. "
                "Original sort exception (if any) preserved."
            )


def remove_excess_spikes(sorting, recording):
    """Drop spikes outside the recording window, then every zero-spike unit.

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
            "remove_excess_spikes: dropped unit(s) "
            f"{dropped.tolist()} with zero spikes (empty from the sorter or "
            "emptied by removing spikes outside the recording window)."
        )
    return result
