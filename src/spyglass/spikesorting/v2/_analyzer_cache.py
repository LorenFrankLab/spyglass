"""Path policy for the regeneratable SortingAnalyzer cache.

A SpikeInterface ``SortingAnalyzer`` folder (waveforms, templates, metric
extensions) is large (5-50 GB) regeneratable SCRATCH, not a canonical
database artifact: a valid ``Sorting`` row keeps its FK-guaranteed upstream
``Recording`` / NWB, so the analyzer can always be rebuilt. This module is
the single place that decides WHERE that cache lives, so the location is a
pure function of ``sorting_id`` + one configured root rather than an
absolute path scattered through (or persisted by) the ``Sorting`` table --
which is the path-drift class of bug this replaces.

Resolution (``analyzer_cache_root``):

1. ``dj.config["custom"]["spikesorting_v2_analyzer_dir"]`` when set -- point
   it at shared storage for a persistent cache;
2. otherwise ``Path(temp_dir) / "spikesorting_v2" / "analyzers"`` -- scratch
   semantics under Spyglass's configured temp directory (the default, and
   identical to the path used before this module existed).

Changing the root is an explicit cache-relocation choice: old folders simply
become cache misses (``get_analyzer`` rebuilds into the new root) and can be
cleaned by the operator -- never a stale-row inconsistency.

This module reads ``dj.config`` and ``temp_dir`` but opens no DB connection
and activates no ``dj.schema``; the reads happen at call time so import stays
side-effect free.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def analyzer_cache_root() -> Path:
    """Return the configured root directory for SortingAnalyzer caches.

    ``dj.config["custom"]["spikesorting_v2_analyzer_dir"]`` when truthy,
    else ``Path(temp_dir) / "spikesorting_v2" / "analyzers"``.

    Returns
    -------
    pathlib.Path
        The root directory under which analyzer caches are stored.
    """
    import datajoint as dj

    from spyglass.settings import temp_dir

    custom = dj.config.get("custom") or {}
    configured = custom.get("spikesorting_v2_analyzer_dir")
    if configured:
        return Path(configured)
    return Path(temp_dir) / "spikesorting_v2" / "analyzers"


def analyzer_path(sorting_id, waveform_params_name: str) -> Path:
    """Return the analyzer-cache folder for a ``(sorting_id, recipe)`` pair.

    ``analyzer_cache_root() / f"{sorting_id}__{waveform_params_name}.zarr"``.
    A sort has more than one analyzer recipe (an unwhitened display recipe and
    a whitened metric recipe), so the folder is keyed by both the
    ``sorting_id`` and the ``waveform_params_name`` that produced it -- the two
    never collide. Deterministic in ``(sorting_id, waveform_params_name)`` +
    the configured root, so every code path resolves the same folder without
    the ``Sorting`` row needing to store the path.

    ``waveform_params_name`` is embedded in the folder name, so callers must
    pass a path-safe name; the ``AnalyzerWaveformParameters`` insert guard
    validates it (``^[A-Za-z0-9_]+$``) before any row -- and therefore any
    folder -- can use it. The ``.zarr`` suffix matches the SI ``zarr`` store
    ``create_sorting_analyzer`` writes (SI forces the ``.zarr`` suffix) and lets
    ``load_sorting_analyzer`` auto-detect the format.

    Parameters
    ----------
    sorting_id
        The sorting whose analyzer is cached.
    waveform_params_name : str
        The ``AnalyzerWaveformParameters`` row name that produced the analyzer.

    Returns
    -------
    pathlib.Path
        The analyzer-cache folder path for this ``(sorting_id, recipe)`` pair.
    """
    return (
        analyzer_cache_root() / f"{sorting_id}__{waveform_params_name}.zarr"
    )


def remove_analyzer_cache(sorting_id, *, missing_ok: bool = True) -> bool:
    """Remove ALL analyzer-cache folders for a ``sorting_id``.

    A sort can have several analyzer recipes on disk (display + metric), so
    this removes every ``{sorting_id}__*.zarr`` folder under the cache root --
    deleting the sort orphans every recipe. The glob is anchored on the full
    ``sorting_id`` (a fixed-length UUID) followed by the ``__`` separator, so
    one sort's folders never match another's.

    ``missing_ok=True`` (default) makes the no-folders case a no-op returning
    ``False`` (the common case: zero-unit sorts never wrote one, and the cache
    is regeneratable). ``missing_ok=False`` raises ``FileNotFoundError`` when no
    folder exists. A removal failure (e.g. a permission error) propagates
    rather than being swallowed.

    Parameters
    ----------
    sorting_id
        The sorting whose analyzer-cache folders should be removed.
    missing_ok : bool, optional
        If ``True`` (the default), no matching folder is a no-op. If
        ``False``, it raises ``FileNotFoundError``.

    Returns
    -------
    bool
        ``True`` if at least one folder was removed, ``False`` if none existed
        and ``missing_ok=True``.

    Raises
    ------
    FileNotFoundError
        If no matching folder exists and ``missing_ok=False``.
    """
    root = analyzer_cache_root()
    folders = (
        sorted(root.glob(f"{sorting_id}__*.zarr")) if root.exists() else []
    )
    if not folders:
        if missing_ok:
            return False
        raise FileNotFoundError(root / f"{sorting_id}__*.zarr")
    for folder in folders:
        shutil.rmtree(folder, ignore_errors=False)
    return True


def classify_orphaned_analyzer_folders(
    units_bearing,
    referenced_paths,
    disk_dir_paths,
) -> dict:
    """Classify analyzer-folder leaks into DB-side and disk-side orphans.

    Pure (DB-free) set logic for ``Sorting.find_orphaned_analyzer_folders``; the
    caller gathers the DB / filesystem facts (which sorting_ids bear units,
    whether each computed folder exists, which folders sit under the analyzer
    root) and this function classifies them.

    A **DB-side orphan** is a units-bearing ``Sorting`` row whose computed
    analyzer folder is gone on disk (regeneratable scratch removed out of band);
    reported only, never auto-deleted. A **disk-side orphan** is an on-disk
    folder under the analyzer root that no row references (the row was deleted
    via a path that bypassed the ``Sorting.delete`` override). Input order of
    ``units_bearing`` and ``disk_dir_paths`` is preserved.

    Parameters
    ----------
    units_bearing : iterable of (sorting_id, computed_path, exists)
        The ``n_units > 0`` rows: each ``sorting_id``, its computed analyzer
        path (as a string), and whether that path currently exists on disk.
    referenced_paths : iterable of str
        Computed analyzer paths of EVERY ``Sorting`` row (any ``n_units``); a
        disk folder in this set is referenced and not an orphan.
    disk_dir_paths : iterable of str
        Directory paths found directly under the analyzer root.

    Returns
    -------
    dict
        ``{"db_side": [{"sorting_id", "computed_analyzer_path"}, ...],
        "disk_side": [folder_path_str, ...]}``.
    """
    db_side = [
        {"sorting_id": sorting_id, "computed_analyzer_path": computed_path}
        for sorting_id, computed_path, exists in units_bearing
        if not exists
    ]
    referenced = set(referenced_paths)
    disk_side = [path for path in disk_dir_paths if path not in referenced]
    return {"db_side": db_side, "disk_side": disk_side}
