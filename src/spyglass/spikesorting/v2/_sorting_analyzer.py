"""``SortingAnalyzer`` cache access/build/rebuild behind ``Sorting``.

``build_analyzer`` builds the binary-folder ``SortingAnalyzer`` and its base
extensions (``random_spikes`` / ``noise_levels`` / ``templates`` /
``waveforms``) for ``Sorting.make_compute`` and the rebuild path -- with the
deterministic-seed pins, the 3D->2D probe projection, the zero-unit
short-circuit, and partial-folder cleanup on failure. The table threads the
fetched ``SorterParameters`` row and resolved job kwargs in (the tri-part
``make_fetch``/``make_compute``/``make_insert`` contract forbids DB I/O inside
compute). ``load_or_rebuild_analyzer`` and ``rebuild_analyzer_folder`` keep the
cache-miss / reconstruction policy out of ``sorting.py`` so future analyzer
extension growth can be added behind the same boundary.

Why this lives in its own module rather than in ``sorting.py``:
``sorting.py`` is a DataJoint *schema* module -- importing it activates
``dj.schema(...)`` and the source-part / merge dependencies. The analyzer build
needs none of that at import, so ``Sorting`` becomes a thin orchestrator. Same
"thin DataJoint shell over pure/IO services" direction as ``_artifact_compute``
/ ``_selection_identity`` / ``_analyzer_cache`` / ``_curation_transforms`` /
``_units_nwb`` / ``_sorting_units`` / ``_sorting_artifact_mask`` /
``_sorting_dispatch``.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: all SpikeInterface / spyglass dependencies are imported
lazily inside the function. ``build_analyzer``'s rebuild fallback (only when
``sorter_row`` is not threaded in) does a single ``SorterParameters`` /
``SortingSelection`` fetch via a CALL-TIME lazy import of those table classes
from ``sorting`` -- by then ``sorting`` is fully imported, so there is no import
cycle.
"""

from __future__ import annotations


def ensure_extensions(
    analyzer, names, *, job_kwargs=None, extension_params=None
):
    """Compute only the SortingAnalyzer extensions not already present.

    Idempotent: an already-present extension is never recomputed (recomputing a
    parent cascade-deletes its children and rewrites template-derived values).
    ``random_seed`` is stripped from ``job_kwargs`` because it is an extension
    param, not a ``ChunkRecordingExecutor`` job kwarg (mirrors the detect_peaks
    / build_analyzer convention).

    Parameters
    ----------
    analyzer : si.SortingAnalyzer
        The analyzer to add extensions to (modified in place / on disk).
    names : list of str
        Extension names to ensure are present.
    job_kwargs : dict, optional
        Resolved concurrency kwargs forwarded to ``analyzer.compute``.
    extension_params : dict, optional
        Per-extension parameter dicts (e.g. ``{"principal_components": {...}}``),
        passed through to ``analyzer.compute(extension_params=...)``. Filtered to
        the extensions actually being added, so a param for an already-present
        extension is dropped (SI rejects params for extensions it isn't
        computing). Use to PIN an extension's params explicitly rather than rely
        on SI's defaults.

    Returns
    -------
    list of str
        The extensions actually computed (already-present ones are skipped).
    """
    compute_kwargs = {
        k: v for k, v in (job_kwargs or {}).items() if k != "random_seed"
    }
    to_add = [name for name in names if not analyzer.has_extension(name)]
    if to_add:
        params = {
            k: v
            for k, v in (extension_params or {}).items()
            if k in to_add
        }
        analyzer.compute(
            to_add, extension_params=params or None, **compute_kwargs
        )
    return to_add


def resolve_display_waveform_params_name(sorting_table, sorting_id) -> str:
    """Return a sort's stored display ``waveform_params_name``.

    Reads ``Sorting.display_waveform_params_name`` -- resolved from the source
    preprocessing recipe and persisted once at sort time, never re-derived --
    so every later rebuild / cache-miss load of the display analyzer resolves
    the SAME recipe (and the same ``peak_amplitude_uv``).
    """
    return (sorting_table & {"sorting_id": sorting_id}).fetch1(
        "display_waveform_params_name"
    )


def fetch_waveform_params(waveform_params_name: str) -> dict:
    """Resolve a waveform-recipe NAME to its tracked, validated params blob.

    The single resolver every name-aware path uses (``Sorting.make_fetch`` and
    the cache-miss rebuild / recompute paths) -- ``build_analyzer`` itself never
    resolves a bare name to params (no DB I/O per the tri-part contract).

    Resolution is **strict**: the params come from the tracked
    ``AnalyzerWaveformParameters`` DB row, so a sort can never build / rebuild
    an analyzer whose waveform parameters are not recorded and queryable in the
    database (the v1-like provenance goal). A missing row raises a clear,
    actionable error rather than silently falling back to hardcoded / catalog
    defaults -- run ``initialize_v2_defaults()`` (or
    ``AnalyzerWaveformParameters.insert_default()``) to install the shipped
    region rows, or insert the custom row, first.
    """
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    row = AnalyzerWaveformParameters & {
        "waveform_params_name": waveform_params_name
    }
    if not row:
        # ValueError (not KeyError): KeyError.__str__ wraps its arg in repr(),
        # which mangles this multi-line actionable message at the moment a user
        # needs to read it.
        raise ValueError(
            "AnalyzerWaveformParameters: no row named "
            f"{waveform_params_name!r}. The analyzer waveform parameters must "
            "be tracked in the database (provenance); install the shipped "
            "region rows with initialize_v2_defaults() / "
            "AnalyzerWaveformParameters.insert_default(), or insert the custom "
            "row, before resolving its analyzer."
        )
    return dict(row.fetch1("params"))


def _load_analyzer_folder_or_rebuild(
    folder, *, rebuild, rebuild_fn, recipe_label, sorting_id
):
    """Load an analyzer folder; rebuild via ``rebuild_fn`` on missing/invalid.

    The shared folder load/invalid-rebuild/missing-rebuild policy behind both
    :func:`load_or_rebuild_analyzer` (DB-coupled rebuild that re-fetches the
    sorting) and :func:`load_or_rebuild_analyzer_from_resolved` (DB-free rebuild
    from already-resolved inputs). ``rebuild_fn`` is the zero-arg callable that
    (re)builds ``folder`` from the canonical sorting; the two callers differ
    only in how they source that build, not in the load/clean/raise policy.

    ``recipe_label`` identifies the analyzer recipe in error messages (the
    recipe name for the DB-coupled path, the folder for the resolved path);
    ``sorting_id`` is for the same messages. ``rebuild=False`` raises
    ``AnalyzerFolderMissingError`` / ``AnalyzerFolderInvalidError`` instead of
    rebuilding (the recompute audit observes the reclaimed/corrupt state).
    """
    import shutil

    import spikeinterface as si

    from spyglass.spikesorting.v2.exceptions import (
        AnalyzerFolderInvalidError,
        AnalyzerFolderMissingError,
    )

    if folder.exists():
        try:
            return si.load_sorting_analyzer(folder)
        except Exception as exc:
            message = (
                "Sorting.get_analyzer: analyzer folder for "
                f"sorting_id={sorting_id!r}, recipe={recipe_label!r} "
                f"exists but could not be loaded ({folder}). The regeneratable "
                "analyzer cache is invalid, likely from a killed build, partial "
                "cleanup, or out-of-band corruption."
            )
            if not rebuild:
                raise AnalyzerFolderInvalidError(
                    message
                    + " The no-rebuild loader surfaces this state for the "
                    "recompute audit instead of rebuilding."
                ) from exc
            from spyglass.utils import logger

            logger.warning(
                f"{message} Removing it and rebuilding. Original error: {exc!r}"
            )
            try:
                shutil.rmtree(folder, ignore_errors=False)
            except Exception as cleanup_exc:
                raise AnalyzerFolderInvalidError(
                    message
                    + " Failed to remove the invalid folder before rebuilding; "
                    "manual cleanup is required."
                ) from cleanup_exc

    if not folder.exists():
        if not rebuild:
            raise AnalyzerFolderMissingError(
                "Sorting.get_analyzer(rebuild=False): analyzer folder for "
                f"sorting_id={sorting_id!r}, recipe={recipe_label!r} is "
                f"absent ({folder}). The regeneratable analyzer cache was removed "
                "out of band; the no-rebuild loader surfaces the missing state "
                "for the recompute audit instead of rebuilding. Use the default "
                "rebuild=True path to reconstruct it."
            )
        rebuild_fn()
    return si.load_sorting_analyzer(folder)


def load_or_rebuild_analyzer(
    sorting_table, key, waveform_params_name=None, *, rebuild=True
):
    """Return the SortingAnalyzer for ``key``, rebuilding the cache if needed.

    Parameters
    ----------
    sorting_table
        ``Sorting`` table/relation instance. Passed in so this service module
        does not import the schema module at import time.
    key : dict
        Restriction selecting a single ``Sorting`` row.
    waveform_params_name : str, optional
        The analyzer recipe to load. ``None`` (the default) resolves the sort's
        stored DISPLAY recipe (``Sorting.display_waveform_params_name``) -- a
        deterministic, well-defined default (the sort's OWN display analyzer),
        not a silent cross-recipe reuse. A caller needing the whitened metric
        recipe passes its name explicitly.
    rebuild : bool, optional
        If ``True`` (the default), a missing or invalid analyzer folder is
        rebuilt in place from the canonical stored sorting (the self-healing
        cache). If ``False``, a missing folder raises
        ``AnalyzerFolderMissingError`` and an unloadable folder raises
        ``AnalyzerFolderInvalidError`` -- the recompute audit uses this so it can
        OBSERVE a missing/reclaimed/corrupt analyzer rather than silently
        rebuild-then-hash it.

    Returns
    -------
    spikeinterface.SortingAnalyzer
        Loaded analyzer. With ``rebuild=True`` a missing or invalid folder is
        rebuilt in place from the canonical stored sorting first.

    Raises
    ------
    ZeroUnitAnalyzerError
        If the selected sort has zero units; SI cannot build a valid analyzer
        over an empty sorting.
    AnalyzerFolderMissingError
        If ``rebuild=False`` and the (units-bearing) sort's analyzer folder is
        absent on disk.
    AnalyzerFolderInvalidError
        If ``rebuild=False`` and the analyzer folder exists but cannot be loaded.
    """
    from spyglass.spikesorting.v2.exceptions import ZeroUnitAnalyzerError

    # Resolve the canonical sorting_id from the matched row rather than
    # assuming ``key`` literally carries "sorting_id" -- a general
    # restriction (e.g. {"object_id": ...} or any single-row selector)
    # must work too. ``fetch1`` enforces that the restriction selects
    # exactly one Sorting row, so the resolved id is unambiguous.
    sorting_id, n_units = (sorting_table & key).fetch1(
        "sorting_id", "n_units"
    )
    if int(n_units) == 0:
        raise ZeroUnitAnalyzerError(
            "Sorting.get_analyzer: sorting_id="
            f"{sorting_id!r} has zero units; no "
            "SortingAnalyzer exists (SI cannot build one over zero "
            "units). Use get_sorting() if you only need the empty "
            "unit list, or re-sort with a lower detect_threshold."
        )

    if waveform_params_name is None:
        waveform_params_name = resolve_display_waveform_params_name(
            sorting_table, sorting_id
        )

    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    folder = analyzer_path(sorting_id, waveform_params_name)
    return _load_analyzer_folder_or_rebuild(
        folder,
        rebuild=rebuild,
        rebuild_fn=lambda: rebuild_analyzer_folder(
            sorting_table,
            {"sorting_id": sorting_id},
            waveform_params_name=waveform_params_name,
        ),
        recipe_label=waveform_params_name,
        sorting_id=sorting_id,
    )


def load_or_rebuild_analyzer_from_resolved(
    *,
    sorting_id,
    n_units,
    analyzer_folder,
    waveform_params,
    recording,
    sorting,
    sorter_row,
    job_kwargs,
    rebuild=True,
):
    """Load (or rebuild) a canonical analyzer folder from resolved inputs.

    The DB-FREE counterpart of :func:`load_or_rebuild_analyzer`: the caller's
    ``make_fetch`` already resolved ``sorting_id`` / ``n_units`` / the analyzer
    cache folder / the recipe params blob, plus the recording + canonical
    ``sorting`` + ``SorterParameters`` row + job kwargs a cache-miss rebuild
    needs, so this runs inside a tri-part ``make_compute`` with NO DB access.
    Same folder load / invalid-rebuild / missing-rebuild policy as the
    DB-coupled loader (the shared :func:`_load_analyzer_folder_or_rebuild`), but
    the rebuild path calls :func:`build_analyzer` with the passed
    recording/sorting instead of re-fetching them through
    ``rebuild_analyzer_folder``.

    Parameters
    ----------
    sorting_id : str
        The sort whose analyzer is loaded (folder identity + messages).
    n_units : int
        The sort's unit count; ``0`` raises ``ZeroUnitAnalyzerError`` (SI cannot
        build an analyzer over zero units), mirroring the DB-coupled loader.
    analyzer_folder : pathlib.Path
        The canonical cache folder (``analyzer_path(sorting_id, recipe_name)``)
        resolved by the caller -- carries the recipe identity.
    waveform_params : dict
        The resolved ``AnalyzerWaveformParameters`` blob for the recipe.
    recording : si.BaseRecording
        The (artifact-masked) recording, used only on a cache-miss rebuild.
    sorting : si.BaseSorting
        The canonical sorting reconstructed from the units NWB, used only on a
        cache-miss rebuild.
    sorter_row : dict
        The fetched ``SorterParameters`` row for the rebuild.
    job_kwargs : dict
        Resolved job kwargs for the rebuild.
    rebuild : bool, optional
        ``True`` (default) rebuilds a missing/invalid folder; ``False`` raises
        (parity with ``load_or_rebuild_analyzer``).

    Returns
    -------
    spikeinterface.SortingAnalyzer
        The loaded analyzer.
    """
    from spyglass.spikesorting.v2.exceptions import ZeroUnitAnalyzerError

    if int(n_units) == 0:
        raise ZeroUnitAnalyzerError(
            "load_or_rebuild_analyzer_from_resolved: sorting_id="
            f"{sorting_id!r} has zero units; no SortingAnalyzer exists (SI "
            "cannot build one over zero units)."
        )
    return _load_analyzer_folder_or_rebuild(
        analyzer_folder,
        rebuild=rebuild,
        rebuild_fn=lambda: build_analyzer(
            sorting,
            recording,
            {"sorting_id": sorting_id},
            sorter_row=sorter_row,
            job_kwargs=job_kwargs,
            analyzer_folder=analyzer_folder,
            waveform_params=waveform_params,
        ),
        recipe_label=analyzer_folder.name,
        sorting_id=sorting_id,
    )


def reconstruct_recording_for_sorting_from_resolved(
    *,
    recording_row,
    source_kind,
    artifact_valid_times=None,
    artifact_detection_id=None,
    recording_id=None,
):
    """Reconstruct a sort's (artifact-masked) recording from resolved inputs.

    The DB-FREE half of :func:`reconstruct_recording_and_sorting`: every DB
    input (the cached recording / concat row, the artifact id + valid_times) is
    resolved by the caller's ``make_fetch`` and threaded in, so this runs inside
    a tri-part ``make_compute`` with NO DB access. Reads the cached preprocessed
    recording NWB directly (the same series ``Recording().get_recording`` reads),
    annotates ``is_filtered=True``, and applies the artifact mask for a
    single-recording artifact-backed sort -- exactly the recording
    ``build_analyzer`` starts from. A concat source observes the full recording
    (no artifact pass), matching ``reconstruct_recording_and_sorting``.

    Parameters
    ----------
    recording_row : dict
        The cached ``Recording`` (single source) or ``ConcatenatedRecording``
        (concat source) row, carrying ``analysis_file_name`` and
        ``electrical_series_path``.
    source_kind : str
        ``"recording"`` or ``"concatenated_recording"``; only a single-recording
        source has an artifact pass.
    artifact_valid_times : np.ndarray, optional
        The artifact-removed valid-times array (shape ``(n_intervals, 2)``)
        resolved in make_fetch; required when masking applies.
    artifact_detection_id : optional
        ``None`` means no artifact pass; otherwise threaded to
        ``apply_artifact_mask`` (also gates whether the mask is applied).
    recording_id : optional
        Threaded to ``apply_artifact_mask`` for provenance/logging.

    Returns
    -------
    si.BaseRecording
        The (artifact-masked) filtered-annotated recording.
    """
    import spikeinterface.extractors as se

    from spyglass.common.common_nwbfile import AnalysisNwbfile

    abs_path = AnalysisNwbfile.get_abs_path(recording_row["analysis_file_name"])
    recording = se.read_nwb_recording(
        abs_path,
        electrical_series_path=recording_row["electrical_series_path"],
        load_time_vector=True,
    )
    # Match Recording().get_recording: the cached artifact is already bandpass +
    # common-referenced, so annotate is_filtered to stop a downstream SI consumer
    # from re-filtering an already-filtered recording.
    recording.annotate(is_filtered=True)
    if source_kind == "recording" and artifact_detection_id is not None:
        from spyglass.spikesorting.v2._sorting_artifact_mask import (
            apply_artifact_mask,
        )

        recording = apply_artifact_mask(
            recording,
            artifact_valid_times,
            artifact_detection_id=artifact_detection_id,
            recording_id=recording_id,
        )
    return recording


def reconstruct_recording_and_sorting(sorting_table, key):
    """Reconstruct the canonical (artifact-masked) recording + sorting for a sort.

    Resolves the sort's source recording (single or concatenated), applies the
    artifact mask when the sort is artifact-backed, and loads the canonical
    sorting from the units NWB -- exactly the ``(recording, sorting)`` pair
    ``build_analyzer`` starts from (it then 2D-projects + whitens per recipe).
    Touches NO analyzer cache, so the recompute audit can source sorting +
    recording without loading a (possibly reclaimed) analyzer folder -- shared
    with ``rebuild_analyzer_folder`` (the cache-rebuild path) so both stay in
    lockstep. ``key`` must carry a literal ``sorting_id``.

    Parameters
    ----------
    sorting_table
        ``Sorting`` table/relation instance (passed so this module imports no
        schema module at import time).
    key : dict
        Restriction carrying a literal ``sorting_id``.

    Returns
    -------
    tuple
        ``(recording, sorting)`` -- the artifact-masked SI recording and the
        canonical SI sorting.
    """
    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection
    from spyglass.spikesorting.v2.utils import (
        artifact_detection_interval_list_name,
    )

    # Resolve the artifact-detection id from the ArtifactDetectionSource part
    # (the master has no artifact_detection_id FK); without it the mask gate
    # below never fires and an artifact-backed sort's recording would omit the
    # mask, diverging from what Sorting.make wrote.
    artifact_detection_id = SortingSelection.resolve_artifact_detection(key)
    source = SortingSelection.resolve_source(key)
    if source.kind == "recording":
        recording = Recording().get_recording(
            {"recording_id": source.key["recording_id"]}
        )
    else:  # concatenated_recording: runs over the concat cache, no artifact pass
        from spyglass.spikesorting.v2.session_group import (
            ConcatenatedRecording,
        )

        recording = ConcatenatedRecording().get_recording(source.key)
    if source.kind == "recording" and artifact_detection_id is not None:
        nwb_file_name = (
            RecordingSelection
            & {"recording_id": source.key["recording_id"]}
        ).fetch1("nwb_file_name")
        valid_times = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": artifact_detection_interval_list_name(
                    artifact_detection_id
                ),
            }
        ).fetch1("valid_times")
        recording = sorting_table._apply_artifact_mask(
            recording=recording,
            valid_times=valid_times,
            artifact_detection_id=artifact_detection_id,
            recording_id=source.key["recording_id"],
        )
    return recording, sorting_table.get_sorting(key)


def rebuild_analyzer_folder(
    sorting_table, key, waveform_params_name=None
) -> None:
    """Rebuild an analyzer folder for an existing ``Sorting`` row.

    Reloads the canonical sorting from the units NWB so the rebuilt analyzer is
    bit-equivalent to the one ``Sorting.make`` wrote -- not a fresh, possibly
    nondeterministic, sort. Resolves the recipe NAME to its params dict here
    (outside ``make_compute``, where a DB read is allowed) and threads BOTH the
    resolved cache folder and the params dict into ``_build_analyzer`` so the
    rebuilt analyzer is byte-comparable to the cached one for the same recipe.

    ``key`` must carry a literal ``sorting_id`` because the path policy and
    selection fetches are keyed by that id. Public callers should resolve a
    general restriction through :func:`load_or_rebuild_analyzer` first.

    ``waveform_params_name`` is the recipe to rebuild; ``None`` resolves the
    sort's stored display recipe (``Sorting.display_waveform_params_name``).
    """
    import shutil

    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.utils import logger

    # The canonical (artifact-masked) recording + sorting -- the exact pair the
    # recompute audit reconstructs too (shared resolver), so a rebuilt analyzer
    # and a fresh audit build start from identical inputs.
    recording, sorting_obj = reconstruct_recording_and_sorting(
        sorting_table, key
    )
    if waveform_params_name is None:
        waveform_params_name = resolve_display_waveform_params_name(
            sorting_table, key["sorting_id"]
        )
    waveform_params = fetch_waveform_params(waveform_params_name)
    # ``_build_analyzer`` writes a folder to disk; a mid-rebuild failure would
    # otherwise leak a partial scratch folder. Removing it before re-raising
    # keeps the rebuild path's invariant ("analyzer folder reflects the
    # canonical sort") true under failure.
    folder = analyzer_path(key["sorting_id"], waveform_params_name)
    try:
        # Pass the already-resolved folder so the build target equals the path
        # this method cleans up on failure (one resolution), and the resolved
        # params dict so the rebuild uses the SAME recipe the cache stored.
        sorting_table._build_analyzer(
            sorting=sorting_obj,
            recording=recording,
            key=key,
            analyzer_folder=folder,
            waveform_params=waveform_params,
        )
    except Exception:
        # Any build failure: remove the partial analyzer folder before
        # re-raising so a half-built folder is never mistaken for a valid cache.
        # The rmtree is best-effort -- a cleanup failure is logged, not raised,
        # so it cannot mask the original error.
        try:
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=False)
        except Exception as cleanup_exc:  # pragma: no cover -- defensive
            logger.error(
                "Sorting._rebuild_analyzer_folder: failed to remove "
                f"partial analyzer folder {folder!r}: {cleanup_exc!r}"
            )
        raise


def build_analyzer(
    sorting,
    recording,
    key,
    *,
    sorter_row=None,
    job_kwargs=None,
    analyzer_folder=None,
    waveform_params=None,
    extensions=None,
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

    ``analyzer_folder`` is the resolved cache folder to write -- the caller
    computes it via ``analyzer_path(sorting_id, waveform_params_name)`` so the
    folder carries the recipe identity. It is required because that name is not
    recoverable from the params dict alone, and this function does no DB I/O.

    ``waveform_params`` is the RESOLVED analyzer-waveform params dict (the
    ``AnalyzerWaveformParameters`` row's blob: ``ms_before`` / ``ms_after`` /
    ``max_spikes_per_unit`` / ``whiten`` / ``purpose``) the caller already
    fetched -- this function never resolves a bare recipe name to params (no DB
    I/O per the tri-part contract). It is required: ``None`` raises so a caller
    can never silently build the wrong (schema-default) window into a
    recipe-named folder.

    Parameters
    ----------
    sorting : spikeinterface.BaseSorting
        The (excess-trimmed) sorting to build the analyzer from.
    recording : si.BaseRecording
        The recording the sorting was computed on.
    key : dict
        Restriction dict carrying ``sorting_id``; used for the fallback
        ``SorterParameters`` fetch and log messages.
    sorter_row : dict, optional
        Keyword-only. Pre-fetched ``SorterParameters`` row. ``None``
        (the rebuild path) triggers a one-time DB fetch. Default
        ``None``.
    job_kwargs : dict, optional
        Keyword-only. Pre-resolved job kwargs. ``None`` resolves them
        locally from ``sorter_row``. Default ``None``.
    analyzer_folder : pathlib.Path, optional
        Keyword-only. Resolved cache folder to write, carrying the recipe
        identity. Required; ``None`` raises ``ValueError``. Default ``None``.
    waveform_params : dict, optional
        Keyword-only. Resolved analyzer-waveform params blob. Required;
        ``None`` raises ``ValueError`` (a caller must pass the sort's resolved
        recipe, never let the build pick a default). Default ``None``.

    Returns
    -------
    pathlib.Path
        The analyzer cache folder path. For a zero-unit sort the folder
        is returned without being built.

    Raises
    ------
    ValueError
        If ``analyzer_folder`` or ``waveform_params`` is not provided.
    """
    import shutil

    import spikeinterface as si

    from spyglass.spikesorting.v2.utils import _resolved_job_kwargs
    from spyglass.utils import logger

    if analyzer_folder is None:
        raise ValueError(
            "build_analyzer: analyzer_folder is required. Resolve it via "
            "analyzer_path(sorting_id, waveform_params_name) so the cache "
            "folder carries the recipe identity (this function does no DB "
            "I/O and cannot resolve the recipe name itself)."
        )
    folder = analyzer_folder

    if waveform_params is None:
        raise ValueError(
            "build_analyzer: waveform_params is required. Pass the sort's "
            "resolved AnalyzerWaveformParameters blob (via "
            "fetch_waveform_params / make_fetch) so the analyzer is built with "
            "the recipe its folder name records -- this function never picks a "
            "default window."
        )

    # ``whiten`` is part of the analyzer recipe identity (display=False,
    # metric=True). A whitened analyzer is the metric recipe for PC/NN
    # cluster-separation metrics; it is whitened below (after the 2D probe
    # projection, before ``create_sorting_analyzer``) and built with
    # ``return_in_uV=False`` (see below).
    whiten = bool(waveform_params.get("whiten"))

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

    # Project the probe to 2D before building the analyzer. Spyglass electrode
    # geometry is stored in 3D (the z coordinate is typically 0), but several
    # SortingAnalyzer extensions and consumers assume 2D contact positions:
    # ``unit_locations`` (monopolar_triangulation / center_of_mass) and the
    # spikeinterface-gui probe view both raise
    # ``could not broadcast input array from shape (3,) into shape (2,)`` on a
    # 3D probe. The in-plane (x, y) coordinates are unchanged by the projection,
    # so sparsity, templates, and waveforms are unaffected ONLY when the z axis
    # is degenerate (all-zero, as Spyglass stores it). Done here (rather than at
    # recording materialization) so the sort itself still sees the recording
    # untouched.
    probe = recording.get_probe()
    if probe.ndim == 3:
        import numpy as np

        from spyglass.utils import logger

        # Dropping a non-degenerate z axis would shift channel distances (and
        # therefore sparsity), so the "x/y unchanged" guarantee above no longer
        # holds. Frank-lab probes are planar (z=0); warn on a genuinely
        # non-planar probe, which may need a different projection axis.
        z = np.asarray(probe.contact_positions)[:, 2]
        if z.size and not np.allclose(z, z[0]):
            logger.warning(
                "build_analyzer: projecting a 3D probe to 2D (axes='xy'), but "
                "the contact z coordinates are not constant (range "
                f"{float(z.max() - z.min()):.3g} um). Depth geometry is "
                "discarded, which can shift channel distances and sparsity "
                "relative to the 3D probe."
            )
        recording = recording.set_probe(probe.to_2d())

    if whiten:
        # The metric recipe: whiten BEFORE building so PC/NN cluster-separation
        # metrics compute in the decorrelated space. Reuse the sorter's pinned
        # whitening (same seed source) -- one implementation, not two. The
        # display recipe leaves the recording in real voltages.
        from spyglass.spikesorting.v2._sorting_dispatch import pinned_whiten

        recording = pinned_whiten(
            recording, random_seed=(job_kwargs or {}).get("random_seed", 0)
        )

    try:
        analyzer = si.create_sorting_analyzer(
            sorting=sorting,
            recording=recording,
            sparse=True,
            format="zarr",
            folder=folder,
            # Display (unwhitened) -> True: real uV amplitudes. Metric
            # (whitened) -> False: ``sip.whiten`` preserves per-channel gains,
            # so a True readback would re-apply them and partially un-normalize
            # the whitened space for non-uniform gains -- defeating the point of
            # whitening for PC/NN metrics. So ``return_in_uV`` derives from the
            # recipe's ``whiten`` flag, it is not an independent knob.
            return_in_uV=not whiten,
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
        # Window + subsample come from the resolved AnalyzerWaveformParameters
        # row (tracked in the DB), NOT hardcoded -- so the settings that
        # produced each analyzer are recorded and reproducible. The window is
        # region-specific (hippocampus 0.5/0.5, cortex 1.0/2.0) and the
        # subsample is the lab's 20000.
        # Default base set computes ``noise_levels`` (needed by downstream
        # quality/template metrics on the STORED analyzer). The recompute-verify
        # path passes ``extensions=ANALYZER_RECOMPUTE_EXTENSIONS`` (random_spikes
        # / templates / waveforms -- the only extensions it hashes) to skip the
        # unseeded ``noise_levels`` estimate it would compute then discard;
        # noise_levels is not a dependency of templates/waveforms, and the hashed
        # extensions are byte-identical with or without it (verified).
        base_extensions = (
            list(extensions)
            if extensions is not None
            else ["random_spikes", "noise_levels", "templates", "waveforms"]
        )
        extension_params = {
            "random_spikes": {
                "max_spikes_per_unit": int(
                    waveform_params["max_spikes_per_unit"]
                ),
                "method": "uniform",
                # Pin the stochastic spike subsampling. When a unit has more
                # than ``max_spikes_per_unit`` spikes, ``random_spikes`` draws a
                # uniform random subset; the SI 0.104 extension defaults
                # ``seed=None`` (verified against
                # ``ComputeRandomSpikes._set_params``), so an unseeded build
                # selects a different subset each time -- and the persisted
                # ``peak_amplitude_uv`` / peak channel (computed from the
                # subset's templates) drifts across rebuilds of the same sort.
                # Defaults to 0 but honors the per-row
                # ``job_kwargs={"random_seed": N}`` override, the same knob the
                # whitening / noise_levels pins read in ``run_si_sorter`` /
                # ``run_clusterless_thresholder`` -- so a user changing the seed
                # gets a consistent seed across the sort AND the analyzer
                # subsample.
                "seed": (job_kwargs or {}).get("random_seed", 0),
            },
            "waveforms": {
                "ms_before": float(waveform_params["ms_before"]),
                "ms_after": float(waveform_params["ms_after"]),
            },
        }
        analyzer.compute(
            base_extensions,
            # Only pass params for extensions actually being computed (a subset
            # rebuild may omit some); SI rejects params for absent extensions.
            extension_params={
                name: params
                for name, params in extension_params.items()
                if name in base_extensions
            },
            **analyzer_job_kwargs,
        )
    except Exception:
        try:
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=False)
        except Exception as cleanup_exc:  # pragma: no cover -- defensive
            logger.error(
                "Sorting._build_analyzer: failed to remove partial "
                f"analyzer folder {folder!r}: {cleanup_exc!r}"
            )
        raise
    return folder
