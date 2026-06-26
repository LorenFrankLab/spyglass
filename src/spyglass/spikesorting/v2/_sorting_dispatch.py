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
compute), so the hot path here is DB-free.

Why this lives in its own module rather than in ``sorting.py``:
``sorting.py`` is a DataJoint *schema* module -- importing it activates
``dj.schema(...)`` and the source-part / merge dependencies. The sort-time
dispatch needs none of that at import, so ``Sorting`` becomes a thin
orchestrator (fetch -> call these -> insert). Same "thin DataJoint shell over
pure/IO services" direction as ``_artifact_compute`` / ``_selection_identity``
/ ``_analyzer_cache`` / ``_curation_transforms`` / ``_units_nwb`` /
``_sorting_units`` / ``_sorting_artifact_mask`` / ``_sorting_analyzer``.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: all SpikeInterface / numpy / spyglass dependencies are
imported lazily inside the functions, and none touches the DB at call time.
"""

from __future__ import annotations

# MATLAB-backed sorters in SpikeInterface (Kilosort 2.5 / 3, IronClust). Their
# algorithm runtime is a compiled MATLAB binary that ships only as a container
# image, so they CANNOT run on a ``backend="local"`` execution row -- a local
# row for one of these raises a clear error (see
# ``assert_matlab_sorter_has_container_backend``). This replaces the old
# name-based ``singularity_image=True`` auto-fallback: the container backend now
# comes from tracked ``SorterParameters.execution_params`` provenance, never
# from the sorter name alone, so an existing custom Kilosort/IronClust row is
# never silently reinterpreted as local execution.
#
# This set coincides today with
# ``_params.sorter._INTERNAL_WHITEN_NO_KWARG_SORTERS`` but encodes a DIFFERENT
# concept (container-execution policy, not internal whitening); they are kept
# separate on purpose -- see the note there.
MATLAB_SORTERS = ("kilosort2_5", "kilosort3", "ironclust")
# Sorter kwargs that do not survive containerization of the MATLAB sorters
# (they reference host paths / host process settings). Stripped only when one of
# the MATLAB sorters runs on a container backend.
MATLAB_SORTER_STRIP_KWARGS = (
    "tempdir",
    "mp_context",
    "max_threads_per_process",
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
        "an explicit container_image. (The old name-based "
        "singularity_image=True fallback was removed in favor of tracked "
        "execution provenance, so an existing local row for this sorter is no "
        "longer silently containerized.)"
    )


def assert_matlab_sorter_has_container_backend(
    sorter: str, execution_params: dict
) -> None:
    """Raise if a MATLAB-backed sorter is on a local execution backend.

    The explicit execution-policy check that replaced the old name-based
    Singularity auto-fallback. Shared by ``run_si_sorter`` (which raises) and
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


def pinned_whiten(recording, *, random_seed: int = 0):
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

    Parameters
    ----------
    recording : spikeinterface.BaseRecording
        The recording to whiten.
    random_seed : int, optional
        Seed for SI's random-chunk covariance estimate. Default 0 (the
        per-row ``job_kwargs={"random_seed": N}`` override flows in here).
    """
    import numpy as np
    import spikeinterface.preprocessing as sip

    return sip.whiten(recording, dtype=np.float64, seed=random_seed)


def _clusterless_noise_levels(
    noise_levels: list[float] | None, threshold_unit: str
) -> list[float] | None:
    """Resolve clusterless-thresholder ``noise_levels`` precedence.

    An explicit ``noise_levels`` always wins. Otherwise ``threshold_unit``
    governs how ``detect_threshold`` is interpreted:

    * ``"uv"`` -> ``[1.0]``; the caller (``run_clusterless_thresholder``)
      scales the recording to microvolts (``scale_to_uV``) before
      ``detect_peaks``, so ``detect_threshold`` is a genuine microvolt
      threshold. (For Frank-lab data gain==1 uV/count, so this matches the
      old raw-count behavior; for non-unity-gain rigs it is the fix.)
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
):
    """Run Spyglass's clusterless-thresholder peak-detection path.

    Not an SI registered sorter; uses
    ``spikeinterface.sortingcomponents.peak_detection.detect_peaks``
    directly and wraps the result in a ``NumpySorting``.

    ``noise_levels`` handling: when the params row supplies a
    non-``None`` value (e.g. ``default_clusterless`` ships
    ``noise_levels=[1.0]``), SI's ``locally_exclusive`` interprets
    ``detect_threshold`` in the recording's amplitude units. With
    ``threshold_unit="uv"`` this method scales the recording to
    microvolts (``scale_to_uV``, using the stored NWB gain) before
    ``detect_peaks``, so ``detect_threshold`` is a TRUE microvolt
    threshold. For Frank-lab data gain==1 uV/count so this is a no-op;
    for non-unity-gain rigs it is the fix. A scalar (singleton list) is
    broadcast to length ``n_channels``
    because SI's ``locally_exclusive`` indexes ``noise_levels[chan]
    * detect_threshold`` per channel. When the params row omits
    ``noise_levels``, the runtime derives it from ``threshold_unit``:
    ``"uv"`` (the schema default, and the production Frank-lab row)
    derives ``[1.0]`` and scales to microvolts as above; ``"mad"``
    (what the ``smoke_clusterless_5uv`` / synthetic-fixture rows set
    EXPLICITLY) leaves it unset so SI computes per-channel MAD and
    ``detect_threshold`` is a MAD multiplier, which finds peaks on the
    low-amplitude MEArec fixture.

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

    # ``threshold_unit`` is a Spyglass-side knob, not a detect_peaks
    # kwarg: strip it and use it to resolve the noise_levels
    # precedence (explicit noise_levels win; otherwise "uv" -> [1.0]
    # AND the recording is scaled to microvolts below via
    # ``scale_to_uV``, so detect_threshold is a TRUE microvolt
    # threshold; "mad" -> None so SI estimates per-channel MAD and
    # detect_threshold is a MAD multiplier).
    #
    # The fallback is "uv" (matching ClusterlessThresholderSchema's
    # default), NOT "mad": a row missing ``threshold_unit`` -- a legacy
    # pre-v4 row, or a default-shaped row carrying only
    # noise_levels=[1.0] -- is interpreted as a microvolt threshold and
    # scaled, not silently thresholded in native counts (which on Intan
    # 0.195 uV/count data would make "100" ~19.5 uV instead of 100 uV).
    # The runtime fallback must agree with the schema default.
    threshold_unit = params.pop("threshold_unit", "uv")
    # Reject an invalid ``threshold_unit`` LOUDLY rather than silently
    # treating it as MAD. The schema's
    # ``Literal["uv", "mad"]`` enforces this at insert, but ``update1`` and
    # pre-validator rows bypass it, and make/sort-time consumes the fetched
    # blob WITHOUT re-validating. Without this guard ANY non-"uv" value
    # falls through to the MAD path -- ``_clusterless_noise_levels`` returns
    # ``None`` for everything except "uv" -- so a typo like "microvolts" or
    # "UV" would silently change what ``detect_threshold`` means instead of
    # failing.
    if threshold_unit not in ("uv", "mad"):
        raise ValueError(
            "clusterless_thresholder: threshold_unit must be 'uv' or "
            f"'mad', got {threshold_unit!r}. A params row written via "
            "update1 or before the SorterParameters validator existed can "
            "carry an invalid value; fix the stored params row."
        )
    # Defense-in-depth: the SorterParameters insert
    # validator rejects this combo, but it is bypassed by ``update1`` and
    # by rows written before the validator existed, and make/sort-time
    # consumes the fetched blob WITHOUT re-validating. A microvolt-scale
    # detect_threshold left in MAD units with no explicit noise_levels
    # makes SI treat e.g. 100 as a 100x-MAD threshold and detect almost
    # nothing -- a silent zero-unit sort. Re-assert the guard here so it
    # surfaces regardless of how the row reached the detector.
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
        # Drop the key entirely so ``detect_peaks`` falls through to
        # SI's per-channel MAD estimation path.
        params.pop("noise_levels", None)
        # Pin SI's random-chunk sampling inside
        # ``get_noise_levels`` (the path detect_peaks takes when
        # noise_levels is absent) to a deterministic seed.
        # PR #3359 (merged 2024-10-25) changed SI's default from
        # ``seed=0`` to ``seed=None``, making the per-channel MAD
        # non-deterministic across runs on the same input --
        # which at a detect_threshold of 5 (a MAD multiplier, not
        # 5σ) can flip ~10-20 borderline peaks per shank. Per PR
        # #3359's stated principle
        # (*"seed must be explicit and no implicit"*) Spyglass IS
        # the explicit-seeder. Same fix + same user-override
        # mechanism as the ``sip.whiten`` pin in
        # ``run_si_sorter`` -- set ``random_seed`` in the per-
        # row ``SorterParameters.job_kwargs`` blob to override.
        _random_seed = (job_kwargs or {}).get("random_seed", 0)
        params.setdefault("random_slices_kwargs", {"seed": _random_seed})
    else:
        n_channels = recording.get_num_channels()
        # Reject an explicit noise_levels of the wrong length BEFORE
        # broadcasting / indexing it per channel: a singleton is
        # broadcast, an n_channels-length array is used as-is, and
        # any other explicit length is a configuration error (it
        # would otherwise mis-index inside SI's ``locally_exclusive``).
        _assert_noise_levels_length(nl_in, n_channels)
        nl = np.asarray(nl_in, dtype=np.float64)
        if nl.size == 1:
            nl = np.full(n_channels, float(nl[0]), dtype=np.float64)
        params["noise_levels"] = nl

    method = params.pop("method", "locally_exclusive")
    # ``random_seed`` is a Spyglass-side knob (already threaded into
    # ``random_slices_kwargs`` above for the noise_levels=None path);
    # it is NOT a valid SI job kwarg, so leaving it in ``job_kwargs``
    # makes ``detect_peaks`` -> ``fix_job_kwargs`` raise
    # ``AssertionError: random_seed is not a valid job keyword
    # argument``. Strip it before the call.
    detect_job_kwargs = {
        k: v for k, v in (job_kwargs or {}).items() if k != "random_seed"
    }
    if threshold_unit == "uv":
        # Honor the "uv" label: scale the detector's input from raw ADC
        # counts to microvolts using the recording's STORED gain/offset
        # (the NWB ElectricalSeries conversion/offset, reloaded onto the
        # recording's channel gains by se.read_nwb_recording). With
        # noise_levels=[1.0], detect_threshold is then a genuine
        # microvolt threshold. For Frank-lab data (gain==1 uV/count) this
        # is a no-op; for non-unity-gain rigs (e.g. Intan ~0.195) it
        # converts a raw-count threshold into true uV.
        import spikeinterface.preprocessing as sip

        if recording.get_channel_gains() is None:
            raise ValueError(
                "clusterless_thresholder threshold_unit='uv' requires the "
                "recording to carry channel gains (the NWB ElectricalSeries "
                "conversion); none are set. Use threshold_unit='mad' for a "
                "gain-free relative threshold."
            )
        recording = sip.scale_to_uV(recording)
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
):
    """Run an SI registered sorter under a managed scratch dir.

    Scratch is anchored under ``spyglass.settings.temp_dir`` via
    ``tempfile.TemporaryDirectory`` so the dir is cleaned on
    successful exit AND on raise (fixes the tempdir leak).
    ``os.chmod 0o777`` makes it world-writable so SI sorter
    subprocesses with a different uid (rootless container, slurm
    scenarios) can write into it.

    External float64 whitening: if the sorter asks for whitening,
    run it externally at float64 and turn the sorter's internal
    whitening off so we do not whiten twice. Runs AFTER the upstream
    artifact mask was applied in ``Sorting.make_compute`` --
    artifact-masked frames should not bias whitening's covariance
    estimate.

    Container execution: the ``execution_params`` row selects the backend.
    ``backend="local"`` runs the sorter on the host (no container kwargs). A
    container backend passes SI's ``docker_image`` / ``singularity_image`` (the
    explicit pinned image) plus the container-install controls
    (``build_run_sorter_container_kwargs``). MATLAB-backed sorters
    (Kilosort 2.5 / 3, IronClust) MUST select a container backend -- a local
    row raises (``assert_matlab_sorter_has_container_backend``); the old
    name-based ``singularity_image=True`` auto-fallback is gone. The
    ``MATLAB_SORTER_STRIP_KWARGS`` (``tempdir`` / ``mp_context`` /
    ``max_threads_per_process``) are stripped only when a MATLAB sorter runs on
    a container backend (they do not survive containerization).

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

    # Resolve + validate execution provenance (None -> default local), then
    # enforce the MATLAB-sorter container policy BEFORE any scratch/whitening
    # work so a misconfigured local MATLAB row fails fast and clearly.
    execution_params = validate_execution_params(execution_params)
    assert_matlab_sorter_has_container_backend(sorter, execution_params)
    container_kwargs = build_run_sorter_container_kwargs(execution_params)

    sorter_temp_dir = tempfile.TemporaryDirectory(
        prefix=f"sort_{sorting_id}_",
        dir=spyglass_temp_dir,
    )
    patched_numpy_inf = False
    try:
        os.chmod(sorter_temp_dir.name, 0o777)

        # spikeextractors 0.9.11 (a transitive dep of SI's MS4
        # wrapper at ``spikeinterface/sorters/external/mountainsort4.py``)
        # references the removed ``numpy.Inf`` alias at
        # ``spikeextractors/extraction_tools.py:766``; that crashes
        # under numpy >= 2.0. Restore the alias only for the duration
        # of this call and delete it again in the ``finally`` -- a
        # persistent global mutation would make every later module
        # that probes ``hasattr(np, "Inf")`` (some scipy versions)
        # see a different numpy than import time. ``np.inf`` is what
        # ``np.Inf`` always aliased, so the patch is value-safe; only
        # its lifetime is scoped. TODO: drop once SI's MS4 wrapper /
        # spikeextractors stops referencing the removed alias.
        if sorter.lower() == "mountainsort4" and not hasattr(np, "Inf"):
            np.Inf = np.inf
            patched_numpy_inf = True

        if sorter_params.get("whiten", False):
            # Pin SI's random-chunk-based covariance estimate inside
            # ``sip.whiten`` to a deterministic seed (see ``pinned_whiten``).
            # Empirically verified: 3 v2 MS4 runs with seed=0 produce identical
            # (n_units, median_fr); without the pin, 4 runs produced 4 distinct
            # states. User override: set ``random_seed`` in the per-row
            # ``SorterParameters.job_kwargs`` blob (for robustness studies);
            # Spyglass's default 0 makes re-runs of a parameter row reproducible
            # by default.
            _random_seed = (job_kwargs or {}).get("random_seed", 0)
            recording = pinned_whiten(recording, random_seed=_random_seed)
            sorter_params = {**sorter_params, "whiten": False}

        # Resolved job_kwargs (n_jobs, chunk_duration, progress_bar,
        # etc.) install via ``si.set_global_job_kwargs`` and are
        # picked up by ``run_sorter`` through SI's global state.
        # They MUST NOT be splatted into ``run_sorter(**...)``: SI
        # 0.104's ``run_sorter`` signature has ``**sorter_params``
        # as the only catch-all, so any extra kwargs flow straight
        # into the sorter and trip strict per-sorter validators
        # (MS4, MS5, KS4 all raise ``Invalid parameters: [...]``
        # for pool_engine / n_jobs / chunk_duration / progress_bar
        # / mp_context / max_threads_per_worker).
        sj_kwargs = dict(job_kwargs or {})
        # ``random_seed`` is a Spyglass-side knob for SI's
        # random-chunk sampling (consumed by the ``sip.whiten``
        # call above and the clusterless ``random_slices_kwargs``
        # pin); strip before installing as a job kwarg because
        # SI's ``set_global_job_kwargs`` rejects unknown keys.
        sj_kwargs.pop("random_seed", None)
        previous_global = dict(si.get_global_job_kwargs())
        if sj_kwargs:
            si.set_global_job_kwargs(**sj_kwargs)
        # Pass a CHILD output folder, not the temp root, so SI's container
        # runner writes its fixed-name ``in_container_recording.*`` /
        # ``in_container_params.json`` files into ``folder.parent`` ==
        # ``sorter_temp_dir.name`` (unique per sort) rather than the SHARED
        # ``spyglass_temp_dir``. Without this, two parallel container populates
        # would stomp each other's fixed-name files in the common parent. The
        # 0o777 chmod above is on that per-sort parent, so the container's
        # (possibly different-uid) writes still land somewhere world-writable.
        output_folder = os.path.join(sorter_temp_dir.name, "sorter_output")
        run_kwargs = dict(
            sorter_name=sorter,
            recording=recording,
            folder=output_folder,
            remove_existing_folder=True,
            # Container execution kwargs (empty for local). docker_image /
            # singularity_image / delete_container_files bind to run_sorter's
            # named params; installation_mode / spikeinterface_version /
            # extra_requirements flow through **sorter_params into
            # run_sorter_container. They originate ONLY from execution_params --
            # the reserved-key rule keeps them out of the scientific params blob,
            # so there is no collision with **effective_params below.
            **container_kwargs,
        )
        # The MATLAB-sorter kwarg strip is needed only when one of those sorters
        # actually runs in a container; it is gated on the selected backend, not
        # the sorter name alone.
        if sorter.lower() in MATLAB_SORTERS and is_container_backend(
            execution_params
        ):
            effective_params = {
                k: v
                for k, v in sorter_params.items()
                if k not in MATLAB_SORTER_STRIP_KWARGS
            }
        else:
            effective_params = sorter_params
        try:
            raw_sorting = sis.run_sorter(**run_kwargs, **effective_params)
            # R4: run_sorter returns a sorting that READS from sorter_temp_dir,
            # which the outer finally cleans up -- downstream _build_analyzer /
            # _stage_sorting_artifact would then read freed files. Sever the
            # file backing here, while the temp dir still exists, by
            # materializing the (small) spike trains into an in-memory
            # NumpySorting. The return expression evaluates BEFORE either finally
            # runs. with_metadata=True so unit properties/annotations survive
            # (SI 0.104.3 default False drops them); copy_spike_vector=True
            # forces an owned copy of the spike vector (the SI default False can
            # leave it aliasing a memmap into the temp dir for some sorter
            # outputs). Neither flag loads traces.
            return si.NumpySorting.from_sorting(
                raw_sorting, with_metadata=True, copy_spike_vector=True
            )
        finally:
            if sj_kwargs:
                # ``set_global_job_kwargs`` UPDATES (does not
                # replace) the global, so a job kwarg the sort
                # installed that was ABSENT from the prior global
                # (chunk_size / total_memory / chunk_memory are not
                # in SI's default global set) would leak into every
                # later populate. Reset to the baseline first, then
                # re-apply the captured prior global for an exact
                # restore. Guard the restore so a restore failure
                # cannot MASK an in-flight sort exception (classic
                # finally-block masking): if ``run_sorter`` raised,
                # that exception must propagate, not a secondary
                # ``set_global_job_kwargs`` error.
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
        # Undo the scoped ``np.Inf`` patch so the global numpy
        # module is left exactly as the rest of the process saw it.
        if patched_numpy_inf and hasattr(np, "Inf"):
            del np.Inf
        # ``TemporaryDirectory`` auto-cleans on garbage collection,
        # but the explicit ``.cleanup()`` in a ``finally`` makes
        # the cleanup point obvious and survives worker-process
        # exit predictably under the parallel-populate process
        # pool. Catch + log any cleanup error: if the sort itself
        # raised, a cleanup failure (e.g. a stale lock on a network
        # FS) must NOT replace the original sort exception -- that
        # would hide the real failure behind a misleading
        # PermissionError on the tempdir.
        try:
            sorter_temp_dir.cleanup()
        except Exception as cleanup_exc:
            logger.warning(
                "Sorting._run_si_sorter: sorter_temp_dir cleanup "
                f"failed for sorting_id={sorting_id}: {cleanup_exc!r}. "
                "Original sort exception (if any) preserved."
            )


def remove_excess_spikes(sorting, recording):
    """Drop spikes whose sample index is outside the recording window.

    Parameters
    ----------
    sorting : spikeinterface.BaseSorting
        The sorter output to trim.
    recording : si.BaseRecording
        The recording whose sample window bounds valid spike indices.

    Returns
    -------
    spikeinterface.BaseSorting
        The sorting with out-of-window spikes removed.
    """
    import spikeinterface.curation as sic

    return sic.remove_excess_spikes(sorting, recording)
