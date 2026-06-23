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

# Sorters that ship as MATLAB containers in SpikeInterface. These need
# ``singularity_image=True`` and a small kwarg-strip carve-out so the
# default Lookup rows survive containerization. The check is
# name-only; users who insert custom rows for other MATLAB sorters can
# extend this set.
MATLAB_SORTERS = ("kilosort2_5", "kilosort3", "ironclust")
MATLAB_SORTER_STRIP_KWARGS = (
    "tempdir",
    "mp_context",
    "max_threads_per_process",
)


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

    MATLAB sorters (Kilosort 2.5 / 3, IronClust) get
    ``singularity_image=True`` and the strip-kwargs carve-out:
    ``tempdir`` / ``mp_context`` / ``max_threads_per_process`` do
    not survive containerization.

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
    from spyglass.utils import logger

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
        run_kwargs = dict(
            sorter_name=sorter,
            recording=recording,
            folder=sorter_temp_dir.name,
            remove_existing_folder=True,
        )
        if sorter.lower() in MATLAB_SORTERS:
            run_kwargs["singularity_image"] = True
            effective_params = {
                k: v
                for k, v in sorter_params.items()
                if k not in MATLAB_SORTER_STRIP_KWARGS
            }
        else:
            effective_params = sorter_params
        try:
            return sis.run_sorter(**run_kwargs, **effective_params)
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
