"""``SortingAnalyzer`` cache access/build/rebuild behind ``Sorting``.

``build_analyzer`` builds the binary-folder ``SortingAnalyzer`` and its base
extensions (``random_spikes`` / ``noise_levels`` / ``templates`` /
``waveforms``) for ``Sorting.make_compute`` and the rebuild path -- with the
deterministic-seed pins, the 3D->2D probe projection, the zero-unit
short-circuit, and partial-folder cleanup on failure. The table threads the
fetched ``SorterParameters`` row and resolved job kwargs in (the tri-part
``make_fetch``/``make_compute``/``make_insert`` contract forbids DB I/O inside
compute). ``load_or_rebuild_analyzer`` and ``rebuild_analyzer_folder`` hold the
cache-miss / reconstruction policy.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: all SpikeInterface / spyglass dependencies are imported
lazily inside the functions. ``build_analyzer`` accepts resolved inputs only;
the explicit rebuild adapter fetches its inputs before invoking computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from spyglass.spikesorting.v2._recording.source import EffectiveSource

BASE_ANALYZER_EXTENSIONS = (
    "random_spikes",
    "noise_levels",
    "templates",
    "waveforms",
)

# Complete immutable extension set published for an interactive display
# analyzer. Evaluation, curation plots, and the curation-scoped resolver import
# this one tuple so a cached merged analyzer cannot be missing an extension a
# supported display helper expects and then be mutated in place on first use.
STANDARD_DISPLAY_ANALYZER_EXTENSIONS = (
    "spike_amplitudes",
    "correlograms",
    "template_similarity",
    "unit_locations",
    "template_metrics",
)


def ensure_extensions(
    analyzer, names, *, job_kwargs=None, extension_params=None
):
    """Compute only the SortingAnalyzer extensions not already present.

    Idempotent: an already-present extension is never recomputed (recomputing a
    parent cascade-deletes its children and rewrites template-derived values).
    ``random_seed`` is stripped from ``job_kwargs``
    (:func:`._sorting_dispatch.without_random_seed`) because it is an extension
    param, not a ``ChunkRecordingExecutor`` job kwarg.

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
    from spyglass.spikesorting.v2._sorting.dispatch import without_random_seed

    compute_kwargs = without_random_seed(job_kwargs)
    to_add = [name for name in names if not analyzer.has_extension(name)]
    if to_add:
        params = {
            k: v for k, v in (extension_params or {}).items() if k in to_add
        }
        analyzer.compute(
            to_add, extension_params=params or None, **compute_kwargs
        )
    return to_add


def _zero_unit_error(caller: str, sorting_id, *, hint: str = ""):
    """The ``ZeroUnitAnalyzerError`` for a zero-unit sort's analyzer request.

    ``hint`` is appended verbatim after the shared message.
    """
    from spyglass.spikesorting.v2.exceptions import ZeroUnitAnalyzerError

    return ZeroUnitAnalyzerError(
        f"{caller}: sorting_id={sorting_id!r} has zero units; no "
        "SortingAnalyzer exists (SI cannot build one over zero units)."
        f"{hint}"
    )


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
    database (the same provenance v1 keeps in ``WaveformParameters``). A
    missing row raises a clear, actionable error rather than silently falling
    back to hardcoded / catalog defaults -- run ``initialize_v2_defaults()`` (or
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
    folder,
    *,
    rebuild,
    rebuild_fn,
    recipe_label,
    sorting_id,
    load_extensions=True,
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

    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        analyzer_cache_lock,
        load_analyzer_extensions,
        load_analyzer_folder,
    )
    from spyglass.spikesorting.v2.exceptions import (
        AnalyzerFolderInvalidError,
        AnalyzerFolderMissingError,
    )

    def load():
        analyzer = load_analyzer_folder(folder)
        _assert_complete_base_extensions(analyzer)
        return (
            load_analyzer_extensions(analyzer) if load_extensions else analyzer
        )

    # Hold the per-sort lock around the whole load / invalid-cleanup / rebuild
    # region: a reader must not observe the brief move-aside window of a
    # concurrent atomic publish, and an invalid-folder rmtree + rebuild must not
    # race another job. The lock is reentrant, so a caller that already holds it
    # (the CurationEvaluation fast path loads inside its own lock) does not
    # self-deadlock.
    with analyzer_cache_lock(sorting_id):
        if folder.exists():
            try:
                return load()
            except Exception as exc:
                message = (
                    "Sorting.get_analyzer: analyzer folder for "
                    f"sorting_id={sorting_id!r}, recipe={recipe_label!r} "
                    f"exists but could not be loaded ({folder}). The "
                    "regeneratable analyzer cache is invalid, likely from a "
                    "killed build, partial cleanup, or out-of-band corruption."
                )
                if not rebuild:
                    raise AnalyzerFolderInvalidError(
                        message
                        + " The no-rebuild loader surfaces this state for the "
                        "recompute audit instead of rebuilding."
                    ) from exc
                from spyglass.utils import logger

                logger.warning(
                    f"{message} Removing it and rebuilding. Original error: "
                    f"{exc!r}"
                )
                try:
                    shutil.rmtree(folder, ignore_errors=False)
                except Exception as cleanup_exc:
                    raise AnalyzerFolderInvalidError(
                        message + " Failed to remove the invalid folder before "
                        "rebuilding; manual cleanup is required."
                    ) from cleanup_exc

        if not folder.exists():
            if not rebuild:
                raise AnalyzerFolderMissingError(
                    "Sorting.get_analyzer(rebuild=False): analyzer folder for "
                    f"sorting_id={sorting_id!r}, recipe={recipe_label!r} is "
                    f"absent ({folder}). The regeneratable analyzer cache was "
                    "removed out of band; the no-rebuild loader surfaces the "
                    "missing state for the recompute audit instead of "
                    "rebuilding. Use the default rebuild=True path to "
                    "reconstruct it."
                )
            rebuild_fn()
        return load()


def _assert_complete_base_extensions(analyzer) -> None:
    """Reject interrupted canonical builds without eagerly loading arrays.

    SI lists an extension as saved as soon as ``params.json`` exists, before
    computation finishes. Validate both its run status and required arrays so
    a killed build cannot look like a usable cache. The mmap reads inspect
    array headers without registering extensions or reading their payloads.
    Low-level loaders still allow intentionally partial expert analyzers.
    """
    from spikeinterface.core.sortinganalyzer import get_extension_class

    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        analyzer_extension_array,
    )

    missing = set(BASE_ANALYZER_EXTENSIONS) - set(
        analyzer.get_saved_extension_names()
    )
    if missing:
        raise ValueError(
            f"Analyzer is missing required base extensions: {sorted(missing)}."
        )
    data_names = {
        "random_spikes": ("random_spikes_indices",),
        "noise_levels": ("noise_levels",),
        "waveforms": ("waveforms",),
    }
    for name in BASE_ANALYZER_EXTENSIONS:
        extension = get_extension_class(name)(analyzer)
        extension.load_params()
        extension.load_run_info()
        if not (extension.run_info or {}).get("run_completed", False):
            raise ValueError(f"Analyzer base extension {name!r} is incomplete.")
        params = extension.params
        if name == "templates":
            names = tuple(
                (
                    operator
                    if isinstance(operator, str)
                    else f"{operator[0]}_{operator[1]}"
                )
                for operator in params["operators"]
            )
            if not names:
                raise ValueError("Analyzer templates have no operators.")
        else:
            names = data_names[name]
        for data_name in names:
            analyzer_extension_array(analyzer, name, data_name)


def load_or_rebuild_analyzer(
    sorting_table,
    key,
    waveform_params_name=None,
    *,
    rebuild=True,
    load_extensions=True,
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

    load_extensions : bool, optional
        Load all extensions by default, including their data validation inside
        the rebuild boundary. Internal read-only views can pass False and load
        individual arrays on demand.

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
    # ``key`` may be any single-row restriction, not only a sorting_id.
    sorting_id, n_units = (sorting_table & key).fetch1("sorting_id", "n_units")
    if int(n_units) == 0:
        raise _zero_unit_error(
            "Sorting.get_analyzer",
            sorting_id,
            hint=(
                " Use get_sorting() if you only need the empty unit list, or "
                "re-sort with a lower detect_threshold."
            ),
        )

    if waveform_params_name is None:
        waveform_params_name = resolve_display_waveform_params_name(
            sorting_table, sorting_id
        )

    # Validate the recipe BEFORE computing the cache path or touching/rmtree-ing
    # any folder: ``waveform_params_name`` is embedded in the folder name and
    # reaches ``get_analyzer`` as an un-FK'd free string, so a path-like name
    # (``../bad``, ``bad/name``) must be rejected up front, and resolving the
    # params row here fails fast on an unknown recipe rather than mid-rebuild
    # after a folder delete. Both checks run for ``rebuild=True`` and ``False``.
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        analyzer_path,
        assert_path_safe_waveform_params_name,
    )

    assert_path_safe_waveform_params_name(waveform_params_name)
    fetch_waveform_params(waveform_params_name)

    folder = analyzer_path(sorting_id, waveform_params_name)
    return _load_analyzer_folder_or_rebuild(
        folder,
        rebuild=rebuild,
        load_extensions=load_extensions,
        rebuild_fn=lambda: rebuild_analyzer_folder(
            sorting_table,
            {"sorting_id": sorting_id},
            waveform_params_name=waveform_params_name,
        ),
        recipe_label=waveform_params_name,
        sorting_id=sorting_id,
    )


def load_analyzer_folder_no_rebuild(folder, *, recipe_label, sorting_id):
    """Load a resolved analyzer folder without rebuilding it; no DB access.

    The ``rebuild=False`` policy of :func:`load_or_rebuild_analyzer` for a
    folder the caller already resolved, so the recompute inventory can
    observe a missing or corrupt analyzer inside a tri-part ``make_compute``.

    Parameters
    ----------
    folder : pathlib.Path
        The canonical cache folder (``analyzer_path(sorting_id, recipe)``).
    recipe_label : str
        The recipe name, for error messages.
    sorting_id
        The sort, for the cache lock and error messages.

    Returns
    -------
    spikeinterface.SortingAnalyzer
        The loaded analyzer, extensions included.

    Raises
    ------
    AnalyzerFolderMissingError
        If the folder is absent.
    AnalyzerFolderInvalidError
        If the folder exists but cannot be loaded.
    """
    return _load_analyzer_folder_or_rebuild(
        folder,
        rebuild=False,
        rebuild_fn=None,
        recipe_label=recipe_label,
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
    statistics_spans,
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
    statistics_spans : list[tuple[int, int]]
        The sort's persisted statistics spans (``Sorting.get_statistics_spans``,
        resolved by the caller), so a rebuild estimates noise and whitening
        from the same samples as the sort-time build.
    rebuild : bool, optional
        ``True`` (default) rebuilds a missing/invalid folder; ``False`` raises
        (parity with ``load_or_rebuild_analyzer``).

    Returns
    -------
    spikeinterface.SortingAnalyzer
        The loaded analyzer.
    """
    if int(n_units) == 0:
        raise _zero_unit_error(
            "load_or_rebuild_analyzer_from_resolved", sorting_id
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
            statistics_spans=statistics_spans,
        ),
        recipe_label=analyzer_folder.name,
        sorting_id=sorting_id,
    )


class CanonicalRecording(NamedTuple):
    """A sort's canonical recording resolved for a read that needs no DB.

    Built by :func:`resolve_canonical_recording` and opened by
    :func:`read_canonical_recording`.

    Attributes
    ----------
    source : EffectiveSource
        The sort's lineage and effective traces.
    abs_path : str
        The effective traces' file (present on disk).
    artifact_valid_times : np.ndarray or None
        Artifact-removed valid times, shape ``(n_intervals, 2)`` in seconds,
        when the traces must be artifact-masked at load; ``None`` otherwise.
    """

    source: EffectiveSource
    abs_path: str
    artifact_valid_times: object


def resolve_canonical_recording(key) -> CanonicalRecording:
    """Resolve a sort's canonical (artifact-masked) recording; reads the DB.

    Resolves the sort's effective traces
    (``SortingSelection.resolve_effective_source``), reads the artifact
    valid times when the traces require the mask (an artifact-backed single
    recording), and rebuilds a missing traces file.

    Parameters
    ----------
    key : dict
        Restriction carrying a literal ``sorting_id``.

    Returns
    -------
    CanonicalRecording
    """
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    # The effective source carries the artifact-detection id from the
    # ArtifactDetectionSource part (the master has no artifact_detection_id
    # FK); without it an artifact-backed sort's recording would omit the mask,
    # diverging from what Sorting.make wrote. A concat cache already holds its
    # member masks, so its traces are never masked again here.
    source = SortingSelection.resolve_effective_source(key)
    lineage, traces = source
    valid_times = None
    if traces.apply_artifact_mask:
        # Route the artifact mask through the ownership-validated helper -- the
        # same one Sorting.make_fetch uses -- so a rebuilt analyzer never
        # diverges from what Sorting.make wrote.
        from spyglass.spikesorting.v2._artifacts.readers import (
            read_recording_artifact_valid_times,
        )

        valid_times = read_recording_artifact_valid_times(
            lineage.artifact_detection_id,
            (RecordingSelection & lineage.key).fetch1("nwb_file_name"),
            caller="reconstruct_recording_and_sorting",
        )
    return CanonicalRecording(
        source=source,
        abs_path=SortingSelection.ensure_effective_traces(traces),
        artifact_valid_times=valid_times,
    )


def read_canonical_recording(canonical: CanonicalRecording):
    """Open a resolved canonical recording, masking if needed; no DB access.

    Parameters
    ----------
    canonical : CanonicalRecording

    Returns
    -------
    si.BaseRecording
        The effective traces, silenced over the artifact periods when the
        mask applies.
    """
    from spyglass.spikesorting.v2._recording.source import (
        read_effective_recording,
    )

    lineage, traces = canonical.source
    return read_effective_recording(
        canonical.abs_path,
        traces,
        artifact_valid_times=canonical.artifact_valid_times,
        artifact_detection_id=lineage.artifact_detection_id,
        recording_id=lineage.key.get("recording_id"),
    )


def reconstruct_recording_and_sorting(sorting_table, key):
    """Reconstruct the canonical (artifact-masked) recording + sorting for a sort.

    The recording is :func:`resolve_canonical_recording` opened by
    :func:`read_canonical_recording` (effective traces, rebuilt if missing,
    masked when they require it), and the sorting is the canonical one from
    the units NWB -- exactly the ``(recording, sorting)`` pair
    ``build_analyzer`` starts from (it then 2D-projects + whitens per recipe).
    Touches NO analyzer cache, so the recompute audit can source sorting +
    recording without loading a (possibly reclaimed) analyzer folder -- the
    recording halves are shared with the audit and with
    ``rebuild_analyzer_folder`` (the cache-rebuild path) so all stay in
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
    recording = read_canonical_recording(resolve_canonical_recording(key))
    return recording, sorting_table.get_sorting(key)


def rebuild_analyzer_folder(
    sorting_table, key, waveform_params_name=None
) -> None:
    """Rebuild an analyzer folder for an existing ``Sorting`` row.

    Reloads the canonical sorting from the units NWB so the rebuilt analyzer is
    bit-equivalent to the one ``Sorting.make`` wrote -- not a fresh, possibly
    nondeterministic, sort. Resolves the recipe NAME to its params dict here
    (outside ``make_compute``, where a DB read is allowed) and threads BOTH the
    resolved cache folder and the params dict into :func:`build_analyzer` so the
    rebuilt analyzer is byte-comparable to the cached one for the same recipe.

    ``key`` must carry a literal ``sorting_id`` because the path policy and
    selection fetches are keyed by that id. Public callers should resolve a
    general restriction through :func:`load_or_rebuild_analyzer` first.

    ``waveform_params_name`` is the recipe to rebuild; ``None`` resolves the
    sort's stored display recipe (``Sorting.display_waveform_params_name``).
    """
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        analyzer_cache_lock,
        analyzer_path,
        publish_analyzer_atomically,
    )
    from spyglass.spikesorting.v2.recompute import (
        invalidate_sorting_analyzer_inventory,
        repopulate_sorting_analyzer_inventory,
    )

    # The canonical (artifact-masked) recording + sorting -- the exact pair the
    # recompute audit reconstructs too (shared resolver), so a rebuilt analyzer
    # and a fresh audit build start from identical inputs.
    recording, sorting_obj = reconstruct_recording_and_sorting(
        sorting_table, key
    )
    # The spans persisted at sort time, so the rebuilt noise levels and
    # whitening match the build being replaced.
    statistics_spans = sorting_table.get_statistics_spans(key)
    if waveform_params_name is None:
        waveform_params_name = resolve_display_waveform_params_name(
            sorting_table, key["sorting_id"]
        )
    waveform_params = fetch_waveform_params(waveform_params_name)
    from spyglass.spikesorting.v2._sorting.fetch import (
        fetch_sorter_analyzer_inputs,
    )

    sorter_row, job_kwargs = fetch_sorter_analyzer_inputs(key)
    folder = analyzer_path(key["sorting_id"], waveform_params_name)
    # A Versions row and every dependent verdict describe the generation that
    # is about to be replaced. Retire them before publishing so an old
    # legacy/unverifiable result can never remain attached to new bytes. If an
    # inventory existed, restore it immediately for whichever folder survives
    # the atomic publish (new on success, old on failure).
    had_inventory = invalidate_sorting_analyzer_inventory(
        key["sorting_id"], waveform_params_name
    )
    # Publish atomically under the per-sort lock: build into a private temp
    # folder, then move it into the canonical slot. A build failure leaves the
    # existing canonical folder untouched (the publisher cleans only its own
    # temp), so an interrupted rebuild can never replace a valid cache with a
    # half-built one. The lock is reentrant, so a rebuild invoked from the
    # (already-locked) load path does not self-deadlock.
    try:
        with analyzer_cache_lock(key["sorting_id"]):
            publish_analyzer_atomically(
                folder,
                lambda temp_folder: build_analyzer(
                    sorting=sorting_obj,
                    recording=recording,
                    key=key,
                    sorter_row=sorter_row,
                    job_kwargs=job_kwargs,
                    analyzer_folder=temp_folder,
                    waveform_params=waveform_params,
                    statistics_spans=statistics_spans,
                ),
            )
    except Exception:
        if had_inventory:
            # Restore the surviving old folder's inventory, or record MISSING
            # when the load path had already removed an invalid folder, before
            # propagating the original build failure.
            try:
                repopulate_sorting_analyzer_inventory(
                    key["sorting_id"], waveform_params_name
                )
            except Exception:  # noqa: BLE001  # pragma: no cover
                from spyglass.utils import logger

                logger.exception(
                    "Failed to restore SortingAnalyzerVersions after an "
                    "analyzer rebuild failed for sorting_id=%s, recipe=%s.",
                    key["sorting_id"],
                    waveform_params_name,
                )
        raise
    if had_inventory:
        repopulate_sorting_analyzer_inventory(
            key["sorting_id"], waveform_params_name
        )


def build_analyzer(
    sorting,
    recording,
    key,
    *,
    sorter_row,
    job_kwargs,
    analyzer_folder=None,
    waveform_params=None,
    extensions=None,
    statistics_spans=None,
):
    """Build the ``binary_folder`` SortingAnalyzer + base extensions.

    ``binary_folder`` is deliberate (see ``_analyzer_cache``): it is the only
    SI 0.104.3 format whose waveform extraction writes straight into a
    memmapped ``waveforms.npy`` instead of a shared-memory buffer sized for
    the whole waveform volume, so the extraction peak is bounded by the
    worker chunk buffers. The recipe's ``sparsity`` block selects the
    channel-sparsity estimation (or a dense analyzer); every effective value,
    including SI defaults, comes from the tracked row.

    Inside ``make_compute`` (no DB reads) the caller passes the fetched
    ``sorter_row`` and resolved ``job_kwargs``. The rebuild adapter resolves
    the same inputs before calling this function. ``analyzer_folder`` and ``waveform_params``
    are always required, so the folder's recipe name and the params built
    into it cannot disagree.

    Parameters
    ----------
    sorting : spikeinterface.BaseSorting
        The (excess-trimmed) sorting to build the analyzer from.
    recording : si.BaseRecording
        The recording the sorting was computed on.
    key : dict
        Restriction dict carrying ``sorting_id`` for log messages.
    sorter_row : dict
        Keyword-only. Required pre-fetched ``SorterParameters`` row.
    job_kwargs : dict
        Keyword-only. Required pre-resolved execution kwargs. An empty mapping
        is valid; ``None`` is rejected rather than consulting ambient settings.
    analyzer_folder : pathlib.Path, optional
        Keyword-only. Resolved cache folder to write, carrying the recipe
        identity. Required; ``None`` raises ``ValueError``. Default ``None``.
    waveform_params : dict, optional
        Keyword-only. Resolved analyzer-waveform params blob. Required;
        ``None`` raises ``ValueError`` (a caller must pass the sort's resolved
        recipe, never let the build pick a default). Default ``None``.
    statistics_spans : list[tuple[int, int]], optional
        Keyword-only. Artifact-free frame spans of ``recording`` (probe
        projection and whitening preserve frames). The metric recipe's
        whitening covariance and the ``noise_levels`` extension are then
        estimated only from samples inside them, so artifact-masked zeros do
        not bias either. ``None`` (default) or one span covering the
        recording keeps SpikeInterface's own estimators unchanged.

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
    if sorter_row is None or job_kwargs is None:
        raise ValueError(
            "build_analyzer: sorter_row and job_kwargs must be resolved "
            "before computation; this function does no database or "
            "ambient configuration lookup."
        )

    import shutil

    import spikeinterface as si

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

    # ``whiten`` is part of the recipe identity: display=False, metric=True
    # (PC/NN cluster-separation metrics).
    whiten = bool(waveform_params.get("whiten"))
    # Channel sparsity comes from the tracked recipe (``SparsityParams``); a
    # blob without the field means SI's radius/100 um default, which the
    # schema default reproduces exactly.
    from spyglass.spikesorting.v2._params.analyzer_waveform import (
        SparsityParams,
    )

    sparsity_kwargs = SparsityParams.model_validate(
        waveform_params.get("sparsity") or {}
    ).si_create_kwargs()

    # SI's sparse analyzer build crashes on an empty sorting
    # (``np.concatenate([])`` in ``random_spikes_selection``). The folder is
    # never built; ``Sorting.get_analyzer`` raises ZeroUnitAnalyzerError.
    # Checked before the folder's parent is created, so a zero-unit sort does
    # no filesystem I/O here.
    if sorting.get_num_units() == 0:
        logger.warning(
            "build_analyzer: sorting_id="
            f"{key.get('sorting_id')!r} has zero units; skipping "
            "analyzer build. Check ``detect_threshold`` / "
            "artifact masking if you expected non-zero output."
        )
        return folder

    folder.parent.mkdir(parents=True, exist_ok=True)

    # Check contacts before ``get_probe()``: probeinterface's own duplicate-
    # position error names neither the sort nor the table to fix. The check
    # reads channel locations without building a probe, and also catches
    # contacts distinct in 3D that collapse under ``to_2d()`` below.
    # ``require_2d=False`` because a reloaded artifact always carries a 3D
    # location (``rel_z`` is persisted) and is projected here on purpose.
    from spyglass.spikesorting.v2._recording.geometry import (
        assert_unique_contact_positions,
    )

    try:
        assert_unique_contact_positions(recording, require_2d=False)
    except ValueError as exc:
        raise ValueError(
            "build_analyzer: cannot build a SortingAnalyzer for sorting_id="
            f"{key.get('sorting_id')!r} -- its recording's contacts do not "
            "have distinct 2D positions, so SpikeInterface cannot build a "
            "probe for it. Fix Probe.Electrode rel_x/rel_y/rel_z for this "
            "sort group's electrodes and re-populate the Recording. "
            f"({exc})"
        ) from exc

    # Project the probe to 2D: ``unit_locations`` and the spikeinterface-gui
    # probe view raise a (3,)-into-(2,) broadcast error on a 3D probe. Done
    # here, not at recording materialization, so the sort sees the recording
    # untouched.
    probe = recording.get_probe()
    if probe.ndim == 3:
        import numpy as np

        from spyglass.utils import logger

        # Dropping a non-constant z shifts channel distances (and sparsity);
        # Frank-lab probes are planar, so warn.
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
        # Same seeded whitening as the sorter path.
        from spyglass.spikesorting.v2._sorting.dispatch import pinned_whiten

        recording = pinned_whiten(
            recording,
            random_seed=(job_kwargs or {}).get("random_seed", 0),
            spans=statistics_spans,
        )

    # Span noise levels are cached on the FINAL object handed to
    # ``create_sorting_analyzer``: every preprocessor (probe projection,
    # whitening) drops SI's ``noise_level_*`` properties, and the
    # ``noise_levels`` extension reads the cache matching the analyzer's
    # ``return_in_uV`` (``not whiten``; the metric recipe estimates on the
    # whitened traces). Same seed as the extension's own pin below.
    from spyglass.spikesorting.v2._sorting.dispatch import (
        cache_span_noise_levels,
    )

    cache_span_noise_levels(
        recording,
        statistics_spans,
        return_in_uV=not whiten,
        seed=(job_kwargs or {}).get("random_seed", 0),
    )

    try:
        # SI 0.104 loses structured preprocessing parameters (notably artifact
        # periods) in JSON. Set this AFTER probe projection/whitening: cloning
        # an extractor reconstructs it and resets its serialization flags.
        # Analyzer provenance is local, trusted Python data, like its sorting.
        from spyglass.spikesorting.v2._storage.spikeinterface import (
            use_pickle_recording_serialization,
        )

        use_pickle_recording_serialization(recording)
        analyzer = si.create_sorting_analyzer(
            sorting=sorting,
            recording=recording,
            format="binary_folder",
            folder=folder,
            **sparsity_kwargs,
            # Display: real uV. Metric: False, because ``sip.whiten`` keeps
            # per-channel gains and a uV readback would partly un-whiten
            # channels with non-uniform gains.
            return_in_uV=not whiten,
            overwrite=True,
        )
        from spyglass.spikesorting.v2._sorting.dispatch import (
            without_random_seed,
        )

        analyzer_job_kwargs = without_random_seed(job_kwargs)
        # Window and subsample come from the tracked recipe row. The random
        # extensions (random_spikes, noise_levels) are seed-pinned so a rebuild
        # is identical, which the recompute check
        # (``ANALYZER_RECOMPUTE_EXTENSIONS``) relies on.
        base_extensions = list(
            extensions if extensions is not None else BASE_ANALYZER_EXTENSIONS
        )
        extension_params = {
            "random_spikes": {
                "max_spikes_per_unit": int(
                    waveform_params["max_spikes_per_unit"]
                ),
                "method": "uniform",
                # SI 0.104 defaults seed=None; unseeded, the subset (and the
                # persisted peak_amplitude_uv / peak channel derived from its
                # templates) drifts between rebuilds. Same row seed as the
                # sort's whitening / noise pins.
                "seed": (job_kwargs or {}).get("random_seed", 0),
            },
            "noise_levels": {
                # SI samples unseeded random chunks by default, so noise (and
                # SNR-based curation) would drift between builds.
                "random_slices_kwargs": {
                    "seed": (job_kwargs or {}).get("random_seed", 0)
                },
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
                "build_analyzer: failed to remove partial "
                f"analyzer folder {folder!r}: {cleanup_exc!r}"
            )
        raise
    return folder
