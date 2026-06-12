"""Sort-time compute services behind ``Sorting``.

These functions are the compute core of ``Sorting.make_compute``: applying
the artifact mask to a recording, dispatching the sorter (the Spyglass
clusterless thresholder or an SpikeInterface registered sorter under a
managed scratch dir), trimming excess spikes, and building the
``SortingAnalyzer`` cache folder. They operate on SpikeInterface objects
and already-fetched parameter rows; the table class threads the fetched
DB state in (the tri-part ``make_fetch``/``make_compute``/``make_insert``
contract forbids DB I/O inside compute), so the hot path here is DB-free.

Why this lives in its own module rather than in ``sorting.py``:
``sorting.py`` is a DataJoint *schema* module -- importing it activates
``dj.schema(...)`` and the source-part / merge dependencies. The
sort-time compute needs none of that at import, so ``Sorting`` becomes a
thin orchestrator (fetch -> call these -> insert). Same "thin DataJoint
shell over pure/IO services" direction as ``_artifact_compute`` /
``_selection_identity`` / ``_analyzer_cache`` / ``_curation_transforms`` /
``_units_nwb``.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: all SpikeInterface / numpy / spyglass dependencies
are imported lazily inside the functions. ``build_analyzer``'s rebuild
fallback (only when ``sorter_row`` is not threaded in) does a single
``SorterParameters`` / ``SortingSelection`` fetch via a CALL-TIME lazy
import of those table classes from ``sorting`` -- by then ``sorting`` is
fully imported, so there is no import cycle.
"""

from __future__ import annotations

# Sorters that ship as MATLAB containers in SpikeInterface. These need
# ``singularity_image=True`` and a small kwarg-strip carve-out so the
# v1-style default Lookup rows survive containerization. The check is
# name-only; users who insert custom rows for other MATLAB sorters can
# extend this set.
MATLAB_SORTERS = ("kilosort2_5", "kilosort3", "ironclust")
MATLAB_SORTER_STRIP_KWARGS = (
    "tempdir",
    "mp_context",
    "max_threads_per_process",
)


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


def apply_artifact_mask(
    recording, valid_times, *, artifact_id=None, recording_id=None
):
    """Zero out the complement of ``valid_times`` on the recording.

    ``valid_times`` is the artifact-removed (start, end) seconds
    array from the upstream ``IntervalList``; ``make_fetch``
    already fetched it as ``obs_intervals`` so ``make_compute``
    passes it through here instead of re-issuing the DB lookup
    (the tri-part contract forbids DB I/O inside compute).

    ``artifact_id`` / ``recording_id`` are used only to make the
    empty-``valid_times`` error message actionable.

    Raises
    ------
    EmptyArtifactValidTimesError
        If ``valid_times`` is empty -- masking would zero the whole
        recording, so the sort must fail loudly instead of running
        over all-zeros.
    ValueError
        If ``valid_times`` is not an ``(n, 2)`` array, has an
        interval with ``end < start``, or is not sorted-by-start and
        non-overlapping. The complement walker assumes monotonic,
        disjoint input; an unsorted/overlapping list would silently
        under-mask. (The fetched ``obs_intervals`` are monotonic in
        practice; this guards a hand-built curation override. Strict
        input is intentional -- silent sort/merge is deferred.)
    """
    import numpy as _np
    import spikeinterface.preprocessing as sip

    from spyglass.spikesorting.v2.exceptions import (
        EmptyArtifactValidTimesError,
    )

    valid_times = _np.asarray(valid_times, dtype=float)
    if valid_times.size == 0:
        raise EmptyArtifactValidTimesError(
            "Artifact-removed valid_times is empty for "
            f"artifact_id={artifact_id!r}, "
            f"recording_id={recording_id!r}: the artifact pass kept "
            "zero seconds of the recording. Masking would zero the "
            "entire recording and the sort would run over all-zeros. "
            "Re-run ArtifactDetection with looser thresholds or "
            "override the artifact selection."
        )
    if valid_times.ndim != 2 or valid_times.shape[1] != 2:
        raise ValueError(
            "_apply_artifact_mask: valid_times must be an (n, 2) array "
            f"of (start, end) seconds; got shape {valid_times.shape}."
        )
    starts = valid_times[:, 0]
    ends = valid_times[:, 1]
    if _np.any(ends < starts):
        raise ValueError(
            "_apply_artifact_mask: valid_times has an interval whose "
            "end precedes its start; each interval must be "
            "(start <= end)."
        )
    if valid_times.shape[0] > 1 and (
        _np.any(_np.diff(starts) < 0) or _np.any(starts[1:] < ends[:-1])
    ):
        raise ValueError(
            "_apply_artifact_mask: valid_times must be sorted by start "
            "time and non-overlapping (the complement walker assumes "
            f"monotonic, disjoint input); got {valid_times.tolist()!r}. "
            "Sort and merge the intervals before passing them."
        )

    timestamps = recording.get_times()
    # Walk the valid intervals left-to-right, collecting the
    # complement (artifact gaps) as a list of (start_frame,
    # end_frame) pairs. Each pair is materialized via
    # ``np.arange`` and concatenated -- ~100x faster than the
    # equivalent ``list.extend(range(start, end))`` for
    # multi-million-sample recordings (Python-int boxing for
    # every frame is the bottleneck of the loop form).
    frame_ranges: list[tuple[int, int]] = []
    cursor = timestamps[0]
    for vs, ve in valid_times:
        if vs > cursor:
            start = int(_np.searchsorted(timestamps, cursor))
            end = int(_np.searchsorted(timestamps, vs))
            if end > start:
                frame_ranges.append((start, end))
        cursor = max(cursor, ve)
    if cursor < timestamps[-1]:
        start = int(_np.searchsorted(timestamps, cursor))
        end = len(timestamps)
        if end > start:
            frame_ranges.append((start, end))

    # Drop pure inter-chunk-gap ranges. For a DISJOINT recording the
    # gap-respecting valid_times leave a single boundary frame between
    # two chunks; the complement walk emits it as a width-1 range whose
    # successor is a wall-clock discontinuity. That frame is the last
    # real sample of the preceding chunk (valid) -- masking it would
    # zero a good sample per gap. A genuine 1-frame artifact instead
    # has ~1-sample spacing to its neighbor, so it is kept. (A 1-sample
    # artifact landing exactly on a chunk's final sample is the lone
    # uncovered edge; negligible at the chunk boundary.)
    sample_period = 1.0 / float(recording.get_sampling_frequency())
    frame_ranges = [
        (s, e)
        for (s, e) in frame_ranges
        if not (
            e - s == 1
            and e < len(timestamps)
            and (timestamps[e] - timestamps[s]) > 1.5 * sample_period
        )
    ]

    if not frame_ranges:
        return recording

    artifact_frames = _np.concatenate(
        [_np.arange(s, e, dtype=_np.int64) for s, e in frame_ranges]
    )
    # SI's ``list_triggers`` is documented as a list-of-arrays, one
    # per event channel. We pass a single numpy array wrapping ALL
    # artifact frames so the mask zeros every artifact sample
    # exactly once, independent of how SI interprets a bare Python
    # list of scalars in future minor versions.
    #
    # ``ms_before=None, ms_after=None`` is the documented "single
    # sample" mode and uses the ``pad is None`` code path in SI
    # 0.104's RemoveArtifactsRecordingSegment.get_traces (each
    # trigger zeros exactly its own frame). ``ms_before=0.0,
    # ms_after=0.0`` looks equivalent but triggers a boundary
    # bug in SI 0.104 where the first artifact frame in any
    # contiguous run is left unmasked (the ``trig - pad[0] <= 0``
    # branch assigns to an empty slice).
    return sip.remove_artifacts(
        recording=recording,
        list_triggers=[artifact_frames],
        ms_before=None,
        ms_after=None,
        mode="zeros",
    )


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
    threshold -- honoring the label v1 used at ``v1/sorting.py:177``
    but never actually enforced (v1 thresholded in raw counts). For
    Frank-lab data gain==1 uV/count so this is a no-op; for
    non-unity-gain rigs it is the fix. A scalar (singleton list) is
    broadcast to length ``n_channels``
    because SI's ``locally_exclusive`` indexes ``noise_levels[chan]
    * detect_threshold`` per channel. When the params row omits
    ``noise_levels``, the runtime derives it from ``threshold_unit``:
    ``"uv"`` (the schema default, and the production Frank-lab row)
    derives ``[1.0]`` and scales to microvolts as above; ``"mad"``
    (what the ``smoke_clusterless_5uv`` / synthetic-fixture rows set
    EXPLICITLY) leaves it unset so SI computes per-channel MAD and
    ``detect_threshold`` is a MAD multiplier -- which is what the v1
    baseline-capture script relies on to find any peaks on the
    low-amplitude MEArec fixture.
    """
    import numpy as _np
    import spikeinterface as si
    from spikeinterface.sortingcomponents.peak_detection import (
        detect_peaks,
    )

    from spyglass.spikesorting.v2.utils import _assert_noise_levels_length

    params = dict(sorter_params)
    # v1-era kwarg rename: SI 0.99 ``local_radius_um`` became
    # ``radius_um`` in 0.101+.
    if "local_radius_um" in params:
        params["radius_um"] = params.pop("local_radius_um")
    # SI 0.104 ``detect_peaks`` rejects v1's stale routing hints
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
    # pre-v4 row, or a v1-parity default-shaped row carrying only
    # noise_levels=[1.0] -- is interpreted as a microvolt threshold and
    # scaled, not silently thresholded in native counts (which on Intan
    # 0.195 uV/count data would make "100" ~19.5 uV instead of 100 uV).
    # The runtime fallback must agree with the schema default.
    threshold_unit = params.pop("threshold_unit", "uv")
    # Reject an invalid ``threshold_unit`` LOUDLY rather than silently
    # treating it as MAD (audit follow-up P2). The schema's
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
    # Defense-in-depth (audit follow-up): the SorterParameters insert
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
        nl = _np.asarray(nl_in, dtype=_np.float64)
        if nl.size == 1:
            nl = _np.full(n_channels, float(nl[0]), dtype=_np.float64)
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
        # converts a previously raw-count threshold into true uV.
        # Diverges from v1, which always thresholded in raw counts.
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
        labels_list=_np.zeros(len(detected), dtype=_np.int32),
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

    External float64 whitening matches v1 (``v1/sorting.py:428-430``):
    if the sorter asks for whitening, run it externally at float64
    and turn the sorter's internal whitening off so we do not
    whiten twice. Runs AFTER the upstream artifact mask was
    applied in ``Sorting.make_compute`` -- artifact-masked frames
    should not bias whitening's covariance estimate.

    MATLAB sorters (Kilosort 2.5 / 3, IronClust) get
    ``singularity_image=True`` and the strip-kwargs carve-out:
    ``tempdir`` / ``mp_context`` / ``max_threads_per_process`` do
    not survive containerization.
    """
    import os
    import tempfile

    import numpy as _np
    import spikeinterface as _si
    import spikeinterface.sorters as sis

    from spyglass.settings import temp_dir as _spyglass_temp_dir
    from spyglass.utils import logger

    sorter_temp_dir = tempfile.TemporaryDirectory(
        prefix=f"sort_{sorting_id}_",
        dir=_spyglass_temp_dir,
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
        if sorter.lower() == "mountainsort4" and not hasattr(_np, "Inf"):
            _np.Inf = _np.inf
            patched_numpy_inf = True

        if sorter_params.get("whiten", False):
            import spikeinterface.preprocessing as sip

            # Pin SI's random-chunk-based covariance estimate
            # inside ``sip.whiten`` to a deterministic seed.
            # PR #3359 (merged 2024-10-25) changed SI's default
            # from ``seed=0`` to ``seed=None``, making the
            # whitening matrix non-deterministic across runs on
            # the same input. Per the PR author: *"seed must be
            # explicit and no implicit"* -- so Spyglass IS the
            # explicit-seeder. Empirically verified: 3 v2 MS4
            # runs with seed=0 produce identical (n_units,
            # median_fr); without the pin, 4 runs produced 4
            # distinct states.
            #
            # User override: set ``random_seed`` in the per-row
            # ``SorterParameters.job_kwargs`` blob to use a
            # different seed (for robustness studies / variance
            # characterization). Spyglass's default 0 matches v1
            # SI 0.99 behavior so re-runs of a parameter row are
            # reproducible by default.
            _random_seed = (job_kwargs or {}).get("random_seed", 0)
            recording = sip.whiten(
                recording, dtype=_np.float64, seed=_random_seed
            )
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
        previous_global = dict(_si.get_global_job_kwargs())
        if sj_kwargs:
            _si.set_global_job_kwargs(**sj_kwargs)
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
                # restore.
                _si.reset_global_job_kwargs()
                _si.set_global_job_kwargs(**previous_global)
    finally:
        # Undo the scoped ``np.Inf`` patch so the global numpy
        # module is left exactly as the rest of the process saw it.
        if patched_numpy_inf and hasattr(_np, "Inf"):
            del _np.Inf
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
    """Drop spikes whose sample index is outside the recording window."""
    import spikeinterface.curation as sic

    return sic.remove_excess_spikes(sorting, recording)


def build_analyzer(
    sorting,
    recording,
    key,
    *,
    sorter_row=None,
    job_kwargs=None,
    analyzer_folder=None,
):
    """Build the binary-folder SortingAnalyzer + base extensions.

    ``sorter_row`` is the already-fetched ``SorterParameters`` row
    from ``make_fetch``; passing it through avoids three redundant
    DB round-trips during ``make_compute`` (we cannot read inside
    ``make_compute`` per the tri-part contract). The rebuild path
    does not have the row pre-fetched, so it leaves
    ``sorter_row=None`` and pays the DB cost once -- that path is
    rare (missing analyzer folder) and not on the populate hot
    path, so the lookup is acceptable there.

    ``job_kwargs`` is the already-resolved dict from
    ``_resolved_job_kwargs`` -- when ``make_compute`` calls
    ``run_sorter`` and ``build_analyzer`` from the same
    invocation it resolves once and threads through; the rebuild
    path falls back to resolving locally (same DB-read tradeoff
    as ``sorter_row``).

    ``analyzer_folder`` is the resolved cache folder to write. When
    ``None`` (the make_compute path) it is resolved from ``sorting_id``
    and RETURNED for the caller to thread on. Callers that already
    resolved it (``_rebuild_analyzer_folder``, which needs the path for
    its own cleanup) pass it in so the folder built equals the folder
    they clean up -- one resolution, no config-drift edge.
    """
    import shutil as _shutil

    import spikeinterface as si

    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2.utils import _resolved_job_kwargs
    from spyglass.utils import logger

    folder = (
        analyzer_folder
        if analyzer_folder is not None
        else analyzer_path(key["sorting_id"])
    )

    # Zero-unit short-circuit BEFORE any I/O or DB fetch:
    # ``create_sorting_analyzer(sparse=True)`` -> ``estimate_sparsity``
    # -> ``random_spikes_selection`` crashes on ``np.concatenate([])``
    # for empty sortings. ``_populate_unit_part`` iterates an empty
    # ``sorting.unit_ids`` and writes zero Unit rows; the Sorting
    # master row commits with ``n_units=0``. Return the (not yet
    # created) folder path for the row; ``Sorting.get_analyzer``
    # raises ``ZeroUnitAnalyzerError`` for a zero-unit sort rather
    # than trying to load this never-built folder.
    if sorting.get_num_units() == 0:
        logger.warning(
            "Sorting._build_analyzer: sorting_id="
            f"{key.get('sorting_id')!r} has zero units; skipping "
            "analyzer build. Check ``detect_threshold`` / "
            "artifact masking if you expected non-zero output."
        )
        return folder

    folder.parent.mkdir(parents=True, exist_ok=True)

    if sorter_row is None:
        from spyglass.spikesorting.v2.sorting import (
            SorterParameters,
            SortingSelection,
        )

        sorter_row = (
            SorterParameters
            & ((SortingSelection & key).proj("sorter", "sorter_params_name"))
        ).fetch1()
    if job_kwargs is None:
        job_kwargs = _resolved_job_kwargs(sorter_row["job_kwargs"])

    try:
        analyzer = si.create_sorting_analyzer(
            sorting=sorting,
            recording=recording,
            sparse=True,
            format="binary_folder",
            folder=folder,
            return_in_uV=True,
            overwrite=True,
        )
        # ``random_seed`` is a Spyglass-side knob (consumed by the sorter
        # and the whitening pin in ``run_si_sorter``), not a valid
        # ``SortingAnalyzer.compute`` keyword -- SI raises
        # "please remove {'random_seed'}". Strip it here, mirroring the
        # detect_peaks path. It stays in ``job_kwargs`` upstream so the
        # sorter/whitening still read the seed.
        analyzer_job_kwargs = {
            k: v for k, v in job_kwargs.items() if k != "random_seed"
        }
        analyzer.compute(
            ["random_spikes", "noise_levels", "templates", "waveforms"],
            extension_params={
                "random_spikes": {
                    "max_spikes_per_unit": 500,
                    "method": "uniform",
                    # Pin the stochastic spike subsampling. When a
                    # unit has more than ``max_spikes_per_unit`` (500)
                    # spikes, ``random_spikes`` draws a uniform random
                    # subset; the SI 0.104 extension defaults
                    # ``seed=None`` (verified against
                    # ``ComputeRandomSpikes._set_params``), so an
                    # unseeded build selects a different subset each
                    # time -- and the persisted ``peak_amplitude_uv``
                    # / peak channel (computed from the subset's
                    # templates) drifts across rebuilds of the same
                    # sort. Defaults to 0 but honors the per-row
                    # ``job_kwargs={"random_seed": N}`` override, the
                    # same knob the whitening / noise_levels pins read
                    # in ``run_si_sorter`` / ``run_clusterless_
                    # thresholder`` (lines using
                    # ``(job_kwargs or {}).get("random_seed", 0)``) --
                    # so a user changing the seed gets a consistent
                    # seed across the sort AND the analyzer subsample.
                    "seed": (job_kwargs or {}).get("random_seed", 0),
                },
                "waveforms": {"ms_before": 1.0, "ms_after": 2.0},
            },
            **analyzer_job_kwargs,
        )
    except Exception:
        try:
            if folder.exists():
                _shutil.rmtree(folder, ignore_errors=False)
        except Exception as cleanup_exc:  # pragma: no cover -- defensive
            logger.error(
                "Sorting._build_analyzer: failed to remove partial "
                f"analyzer folder {folder!r}: {cleanup_exc!r}"
            )
        raise
    return folder
