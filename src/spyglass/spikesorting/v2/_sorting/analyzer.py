"""``SortingAnalyzer`` cache access/build/rebuild behind ``Sorting``.

DB-free at import. ``build_analyzer`` takes resolved inputs only; the rebuild
adapters fetch theirs first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from spyglass.spikesorting.v2._storage.recompute import (
    BASE_ANALYZER_EXTENSIONS,
)

if TYPE_CHECKING:
    from spyglass.spikesorting.v2._recording.source import EffectiveSource

# One shared set, so no display helper mutates a cached analyzer on first use.
STANDARD_DISPLAY_ANALYZER_EXTENSIONS = (
    "spike_amplitudes",
    "correlograms",
    "template_similarity",
    "unit_locations",
    "template_metrics",
)

# Extensions that read each spike at its template extremum for a
# ``peak_sign``; SI's default ``"neg"`` misreads a positive-going sort.
POLARITY_EXTENSIONS = ("spike_amplitudes", "spike_locations")


def polarity_extension_params(names, peak_sign, extension_params=None):
    """``extension_params`` with the sort's ``peak_sign`` on polarity extensions.

    The sort's polarity overrides a requested ``peak_sign``.
    """
    params = {name: dict(p) for name, p in (extension_params or {}).items()}
    for name in POLARITY_EXTENSIONS:
        if name in names:
            params[name] = {**params.get(name, {}), "peak_sign": peak_sign}
    return params


def ensure_review_template_percentiles(analyzer) -> None:
    """Store the template percentiles the review's waveform view shades with.

    SI's waveform widgets otherwise compute and save them into the shared
    cache on first plot, invalidating it. The recompute hash covers only the
    average templates.
    """
    import inspect

    import spikeinterface.widgets as sw

    shading = (
        inspect.signature(sw.UnitWaveformsWidget.__init__)
        .parameters["templates_percentile_shading"]
        .default
    )
    if (
        shading is None
        or not analyzer.has_extension("waveforms")
        or not analyzer.has_extension("templates")
    ):
        return
    if isinstance(shading, (int, float)):
        shading = (shading, 100 - shading)
    templates = analyzer.get_extension("templates")
    for percentile in shading:
        templates.get_templates(operator="percentile", percentile=percentile)


class _AnalyzerSparsityInputsError(RuntimeError):
    """Unavailable sorter parameters do not establish that a cache is invalid."""


def ensure_extensions(
    analyzer, names, *, peak_sign, job_kwargs=None, extension_params=None
):
    """Compute only the SortingAnalyzer extensions not already present.

    Recomputing a parent would cascade-delete its children. ``random_seed``
    is stripped from ``job_kwargs``.

    Parameters
    ----------
    analyzer : si.SortingAnalyzer
        The analyzer to add extensions to (modified in place / on disk).
    names : list of str
        Extension names to ensure are present.
    peak_sign : {"neg", "pos", "both"}
        The sort's polarity (``resolve_peak_sign``), applied to every
        :data:`POLARITY_EXTENSIONS` member being added.
    job_kwargs : dict, optional
        Resolved concurrency kwargs forwarded to ``analyzer.compute``.
    extension_params : dict, optional
        Per-extension parameter dicts for ``analyzer.compute``, filtered to
        the extensions being added (SI rejects the rest).

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
            k: v
            for k, v in polarity_extension_params(
                to_add, peak_sign, extension_params
            ).items()
            if k in to_add
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

    Persisted at sort time, so every rebuild resolves the same recipe.
    """
    return (sorting_table & {"sorting_id": sorting_id}).fetch1(
        "display_waveform_params_name"
    )


def fetch_waveform_params(waveform_params_name: str) -> dict:
    """Resolve a waveform-recipe NAME to its tracked, validated params blob.

    Strict: a missing ``AnalyzerWaveformParameters`` row raises rather than
    falling back to defaults, so every analyzer's parameters are tracked.
    """
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    row = AnalyzerWaveformParameters & {
        "waveform_params_name": waveform_params_name
    }
    if not row:
        # Not KeyError, whose str() repr-wraps the message.
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
    validate_analyzer=None,
    restore_recording=None,
):
    """Load an analyzer folder; rebuild via ``rebuild_fn`` on missing/invalid.

    - ``rebuild_fn``: zero-arg callable that builds ``folder``.
    - ``recipe_label``, ``sorting_id``: for error messages.
    - ``rebuild=False`` raises ``AnalyzerFolderMissingError`` /
      ``AnalyzerFolderInvalidError`` instead of rebuilding.
    - ``restore_recording``: optional zero-arg callable restoring the trace
      file; a folder missing only its recording is retried once after it.
    """
    import shutil

    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        AnalyzerRecordingUnavailableError,
        analyzer_cache_lock,
        load_analyzer_extensions,
        load_analyzer_folder,
    )
    from spyglass.spikesorting.v2.exceptions import (
        AnalyzerFolderInvalidError,
        AnalyzerFolderMissingError,
    )

    def load(restore=restore_recording):
        try:
            analyzer = load_analyzer_folder(folder)
        except AnalyzerRecordingUnavailableError:
            if restore is None:
                raise
            # Cheaper than a rebuild. Reload in a fresh frame: pyparsing's
            # packrat cache can pin this one and its memmapped waveforms.
            restore()
            return load(restore=None)
        _assert_complete_base_extensions(analyzer)
        if validate_analyzer is not None:
            validate_analyzer(analyzer)
        return (
            load_analyzer_extensions(analyzer) if load_extensions else analyzer
        )

    # Covers a concurrent publish's move-aside window; the lock is reentrant.
    with analyzer_cache_lock(sorting_id):
        if folder.exists():
            try:
                return load()
            except _AnalyzerSparsityInputsError:
                raise
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


def assert_analyzer_matches_recipe(analyzer, waveform_params) -> None:
    """Raise if a cached sort analyzer was built from other recipe content.

    The cache folder is keyed by recipe name, so a recipe re-inserted under
    the same name leaves a stale folder.

    Raises
    ------
    ValueError
        If the cached window, ``max_spikes_per_unit`` or whitening differs
        from ``waveform_params``.
    """
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        analyzer_extension_params,
    )

    expected = {
        ("waveforms", "ms_before"): float(waveform_params["ms_before"]),
        ("waveforms", "ms_after"): float(waveform_params["ms_after"]),
        ("random_spikes", "max_spikes_per_unit"): int(
            waveform_params["max_spikes_per_unit"]
        ),
    }
    for (extension, name), value in expected.items():
        cached = analyzer_extension_params(analyzer, extension).get(name)
        if cached != value:
            raise ValueError(
                f"Analyzer {extension}.{name}={cached!r} does not match its "
                f"recipe's {value!r}; the recipe changed under the same name. "
                "Rebuild the analyzer."
            )
    if bool(analyzer.return_in_uV) == bool(waveform_params.get("whiten")):
        raise ValueError(
            "Analyzer whitening does not match its recipe's "
            f"whiten={bool(waveform_params.get('whiten'))}; the recipe "
            "changed under the same name. Rebuild the analyzer."
        )


def sort_analyzer_validator(waveform_params, sorter_params_fn):
    """Validate a sort-level cache against its recipe, then its polarity."""
    check_polarity = analyzer_sparsity_validator(sorter_params_fn)

    def validate(analyzer):
        assert_analyzer_matches_recipe(analyzer, waveform_params)
        check_polarity(analyzer)

    return validate


def analyzer_sparsity_validator(sorter_params_fn):
    """Check a cached analyzer's polarity against its sorter's.

    Covers sparsity and every computed :data:`POLARITY_EXTENSIONS` member.
    """
    peak_sign = None

    def validate(analyzer):
        nonlocal peak_sign
        polarity_extensions = [
            name for name in POLARITY_EXTENSIONS if analyzer.has_extension(name)
        ]
        if not analyzer.is_sparse() and not polarity_extensions:
            return
        if peak_sign is None:
            try:
                sorter_params = dict(sorter_params_fn() or {})
            except Exception as exc:
                raise _AnalyzerSparsityInputsError(
                    "Could not resolve sorter parameters for analyzer sparsity "
                    "validation; the existing cache has not been invalidated."
                ) from exc
            from spyglass.spikesorting.v2._core.signal_math import (
                resolve_peak_sign,
            )

            peak_sign = resolve_peak_sign(sorter_params)
        from spyglass.spikesorting.v2._storage.analyzer_cache import (
            analyzer_extension_params,
            assert_analyzer_sparsity_polarity,
        )

        assert_analyzer_sparsity_polarity(
            analyzer, expected_peak_sign=peak_sign
        )
        for name in polarity_extensions:
            cached = analyzer_extension_params(analyzer, name).get("peak_sign")
            if cached != peak_sign:
                raise ValueError(
                    f"Analyzer {name} polarity is incompatible with its "
                    f"sorter: cached={cached!r}, expected={peak_sign!r}. "
                    "Rebuild the analyzer to read spikes at their peaks."
                )

    return validate


def _assert_complete_base_extensions(analyzer) -> None:
    """Reject interrupted canonical builds without eagerly loading arrays.

    SI lists an extension as saved once ``params.json`` exists, before it
    finishes. Checks run status and array headers without reading payloads.
    """
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        analyzer_extension_array,
    )
    from spyglass.spikesorting.v2._storage.spikeinterface import (
        saved_extension_metadata,
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
        extension, completed = saved_extension_metadata(analyzer, name)
        if not completed:
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
        ``Sorting`` table/relation instance.
    key : dict
        Restriction selecting a single ``Sorting`` row.
    waveform_params_name : str, optional
        The analyzer recipe to load. ``None`` (the default) uses the sort's
        stored display recipe.
    rebuild : bool, optional
        If ``True`` (the default), a missing or invalid folder is rebuilt
        from the canonical stored sorting. If ``False``, it raises (see
        Raises), so the recompute audit can observe the state.
    load_extensions : bool, optional
        Load all extensions by default; read-only views can pass False and
        load arrays on demand.

    Returns
    -------
    spikeinterface.SortingAnalyzer
        Loaded analyzer.

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

    # Before any folder is touched: the name is an un-FK'd string in a path.
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        analyzer_path,
        assert_path_safe_waveform_params_name,
    )

    assert_path_safe_waveform_params_name(waveform_params_name)
    waveform_params = fetch_waveform_params(waveform_params_name)

    def sorter_params():
        from spyglass.spikesorting.v2._sorting.fetch import (
            fetch_sorter_analyzer_inputs,
        )

        return fetch_sorter_analyzer_inputs({"sorting_id": sorting_id})[0].get(
            "params"
        )

    def restore_recording():
        from spyglass.spikesorting.v2.sorting import SortingSelection

        SortingSelection.ensure_effective_traces(
            SortingSelection.resolve_effective_source(
                {"sorting_id": sorting_id}
            ).traces
        )

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
        validate_analyzer=sort_analyzer_validator(
            waveform_params, sorter_params
        ),
        restore_recording=restore_recording if rebuild else None,
    )


def load_analyzer_folder_no_rebuild(folder, *, recipe_label, sorting_id):
    """Load a resolved analyzer folder without rebuilding it; no DB access.

    The ``rebuild=False`` policy of :func:`load_or_rebuild_analyzer` for an
    already-resolved folder.

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

    The DB-free counterpart of :func:`load_or_rebuild_analyzer`, for use
    inside ``make_compute``.

    Parameters
    ----------
    sorting_id : str
        The sort whose analyzer is loaded (folder identity + messages).
    n_units : int
        The sort's unit count; ``0`` raises ``ZeroUnitAnalyzerError``.
    analyzer_folder : pathlib.Path
        The canonical cache folder (``analyzer_path(sorting_id, recipe_name)``).
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
        The sort's persisted statistics spans, so a rebuild samples noise
        and whitening as the sort-time build did.
    rebuild : bool, optional
        ``True`` (default) rebuilds a missing/invalid folder; ``False`` raises.

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
        validate_analyzer=sort_analyzer_validator(
            waveform_params, lambda: sorter_row.get("params")
        ),
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

    Reads the artifact valid times when the traces require the mask, and
    rebuilds a missing traces file.

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

    source = SortingSelection.resolve_effective_source(key)
    lineage, traces = source
    valid_times = None
    if traces.apply_artifact_mask:
        # Same helper as Sorting.make_fetch, so rebuilds match the sort.
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

    The ``(recording, sorting)`` pair ``build_analyzer`` starts from. Touches
    no analyzer cache.

    Parameters
    ----------
    sorting_table
        ``Sorting`` table/relation instance.
    key : dict
        Restriction carrying a literal ``sorting_id``.

    Returns
    -------
    tuple
        ``(recording, sorting)``: the artifact-masked SI recording and the
        canonical SI sorting.
    """
    recording = read_canonical_recording(resolve_canonical_recording(key))
    return recording, sorting_table.get_sorting(key)


def rebuild_analyzer_folder(
    sorting_table, key, waveform_params_name=None
) -> None:
    """Rebuild an analyzer folder for an existing ``Sorting`` row.

    Uses the canonical sorting from the units NWB, not a fresh sort, so the
    rebuild matches what ``Sorting.make`` wrote. ``key`` must carry a
    literal ``sorting_id``. ``waveform_params_name=None`` uses the stored
    display recipe.
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

    recording, sorting_obj = reconstruct_recording_and_sorting(
        sorting_table, key
    )
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
    # Retire verdicts first so none attaches to new bytes; restore after.
    had_inventory = invalidate_sorting_analyzer_inventory(
        key["sorting_id"], waveform_params_name
    )
    # A failed build leaves the existing folder untouched; lock is reentrant.
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

    ``binary_folder`` is the only SI 0.104.3 format whose waveform
    extraction memmaps instead of allocating the whole volume. Sparsity
    comes from the recipe, with the sorter's polarity. No DB or ambient
    configuration is read.

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
        Keyword-only. Required pre-resolved execution kwargs; may be empty.
    analyzer_folder : pathlib.Path, optional
        Keyword-only. Resolved cache folder to write. Required.
    waveform_params : dict, optional
        Keyword-only. Resolved analyzer-waveform params blob. Required.
    statistics_spans : list[tuple[int, int]], optional
        Keyword-only. Artifact-free frame spans of ``recording``; whitening
        and ``noise_levels`` sample only inside them. ``None`` (default)
        keeps SpikeInterface's own estimators.

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
    seed = job_kwargs.get("random_seed", 0)

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

    # Display recipe: False. Metric recipe (PC/NN metrics): True.
    whiten = bool(waveform_params.get("whiten"))
    # A mismatched polarity can drop a positive sort's peak channel.
    from spyglass.spikesorting.v2._params.analyzer_waveform import (
        SparsityParams,
    )

    sparsity_kwargs = SparsityParams.model_validate(
        waveform_params["sparsity"]
    ).si_create_kwargs()
    if sparsity_kwargs["sparse"]:
        from spyglass.spikesorting.v2._core.signal_math import resolve_peak_sign

        sparsity_kwargs["peak_sign"] = resolve_peak_sign(
            sorter_row.get("params")
        )

    # SI's sparse build crashes on an empty sorting.
    if sorting.get_num_units() == 0:
        logger.warning(
            "build_analyzer: sorting_id="
            f"{key.get('sorting_id')!r} has zero units; skipping "
            "analyzer build. Check ``detect_threshold`` / "
            "artifact masking if you expected non-zero output."
        )
        return folder

    folder.parent.mkdir(parents=True, exist_ok=True)

    # Before get_probe(), whose error names neither the sort nor the fix.
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

    # unit_locations and the GUI probe view fail on a 3D probe.
    probe = recording.get_probe()
    if probe.ndim == 3:
        import numpy as np

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
        from spyglass.spikesorting.v2._sorting.dispatch import pinned_whiten

        recording = pinned_whiten(
            recording,
            random_seed=seed,
            spans=statistics_spans,
        )

    # On the final object: preprocessors drop the cached property.
    from spyglass.spikesorting.v2._sorting.dispatch import (
        cache_span_noise_levels,
    )

    cache_span_noise_levels(
        recording,
        statistics_spans,
        return_in_uV=not whiten,
        seed=seed,
    )

    try:
        # SI 0.104 loses artifact periods in JSON. Set after projection and
        # whitening, which reset the flag.
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
            # sip.whiten keeps gains, so a uV readback would un-whiten.
            return_in_uV=not whiten,
            overwrite=True,
        )
        from spyglass.spikesorting.v2._sorting.dispatch import (
            without_random_seed,
        )

        analyzer_job_kwargs = without_random_seed(job_kwargs)
        # Seeds make rebuilds identical, as the recompute check requires.
        base_extensions = list(
            extensions if extensions is not None else BASE_ANALYZER_EXTENSIONS
        )
        extension_params = {
            "random_spikes": {
                "max_spikes_per_unit": int(
                    waveform_params["max_spikes_per_unit"]
                ),
                "method": "uniform",
                # SI 0.104 defaults to seed=None.
                "seed": seed,
            },
            "noise_levels": {
                "random_slices_kwargs": {"seed": seed},
            },
            "waveforms": {
                "ms_before": float(waveform_params["ms_before"]),
                "ms_after": float(waveform_params["ms_after"]),
            },
        }
        analyzer.compute(
            base_extensions,
            # SI rejects params for extensions not being computed.
            extension_params={
                name: params
                for name, params in extension_params.items()
                if name in base_extensions
            },
            **analyzer_job_kwargs,
        )
        if not whiten:
            ensure_review_template_percentiles(analyzer)
        if sparsity_kwargs["sparse"]:
            from spyglass.spikesorting.v2._storage.analyzer_cache import (
                write_analyzer_sparsity_receipt,
            )

            write_analyzer_sparsity_receipt(
                folder, peak_sign=sparsity_kwargs["peak_sign"]
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
