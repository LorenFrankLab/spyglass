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
import threading
from pathlib import Path

# Memoize one ``FileLock`` instance per lock-file path so a same-thread nested
# acquisition (the read path taking the lock while the compute path already
# holds it) is reentrant rather than a self-deadlock -- see
# ``analyzer_cache_lock``. The registry persists for the process lifetime (one
# small entry per sort touched); ``_ANALYZER_LOCK_REGISTRY_GUARD`` serializes
# the get-or-create so two threads cannot mint two instances for one path.
_ANALYZER_LOCK_REGISTRY: dict = {}
_ANALYZER_LOCK_REGISTRY_GUARD = threading.Lock()


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
    A sort may have more than one analyzer recipe (an unwhitened display recipe
    and a whitened metric recipe built on demand for PC/NN metrics), so the
    folder is keyed by both the ``sorting_id`` and the ``waveform_params_name``
    that produced it -- the two never collide. Deterministic in ``(sorting_id, waveform_params_name)`` +
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


def analyzer_cache_lock(sorting_id, *, timeout: float = -1):
    """Return a cross-process lock serializing analyzer-cache access per sort.

    Every canonical analyzer-cache folder reader/writer/deleter holds this lock:
    the populate build, the self-healing read/rebuild path
    (``Sorting.get_analyzer`` / ``load_or_rebuild_analyzer``), the atomic
    publish, ``remove_analyzer_cache``, the recompute delete gate, and the
    ``CurationEvaluation`` raw-sort fast path (which loads the shared analyzer
    and persists extensions + ``quality_metrics`` into it). Holding the lock
    around the load/mutate/publish region lets ONE job touch a sort's analyzer
    at a time, so a concurrent build never corrupts the shared zarr store and a
    reader never observes the brief move-aside window of an atomic publish.

    The lock is keyed on ``sorting_id`` (which every recipe folder of that sort
    shares -- both the display and the whitened-metric analyzer), so it
    serializes one sort's jobs against each other while leaving jobs for
    DIFFERENT sorts free to run in parallel. The lock file lives under
    ``analyzer_cache_root()`` next to the folders it guards.

    **Reentrant per process.** The read path acquires this lock while the
    compute path already holds it for the same sort (the fast path loads inside
    its own ``with`` block). A fresh ``FileLock`` per call would self-deadlock
    there, so the instance is memoized per lock-file path: filelock's instance
    counter makes a same-thread nested acquisition reentrant (instant), while a
    DIFFERENT thread or a DIFFERENT process still contends on the OS-level lock.
    ``populate(processes=N)`` (the standard parallel case) uses separate
    processes, so cross-job serialization is preserved.

    This serializes processes on ONE machine. It does NOT coordinate across
    machines sharing an NFS-mounted ``spikesorting_v2_analyzer_dir`` --
    ``filelock``'s OS-level lock is local; cross-host populate of the same sort
    would still race. A regeneratable cache makes the worst case a rebuild, not
    data loss.

    Parameters
    ----------
    sorting_id
        The sorting whose analyzer-cache folders the job will access.
    timeout : float, optional
        Seconds to wait for the lock before raising ``filelock.Timeout``.
        Default ``-1`` blocks indefinitely (wait your turn), the intended
        serialize-don't-fail behavior; the lock releases when the holding
        process exits, so a crashed job cannot wedge the next one. Applied when
        the per-sort instance is first created; later callers reuse that
        instance (every internal caller uses the default), and ``.acquire``
        accepts a per-call ``timeout`` override regardless.

    Returns
    -------
    filelock.FileLock
        A memoized, reentrant lock; use it as a context manager or call
        ``.acquire()``.
    """
    from filelock import FileLock

    root = analyzer_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = str(root / f"{sorting_id}.analyzer.lock")
    with _ANALYZER_LOCK_REGISTRY_GUARD:
        lock = _ANALYZER_LOCK_REGISTRY.get(lock_path)
        if lock is None:
            lock = FileLock(lock_path, timeout=timeout)
            _ANALYZER_LOCK_REGISTRY[lock_path] = lock
        return lock


# Backwards-compatible alias: callers predating the cache-wide rename (the
# ``CurationEvaluation`` fast path, the recording-lock docstring cross-ref) used
# ``analyzer_curation_lock`` when the lock guarded only ``AnalyzerCuration``.
analyzer_curation_lock = analyzer_cache_lock


def remove_analyzer_cache(sorting_id, *, missing_ok: bool = True) -> bool:
    """Remove ALL analyzer-cache folders for a ``sorting_id``.

    A sort can have several analyzer recipes on disk (the display recipe today;
    the whitened metric recipe once that build lands), so this removes every
    ``{sorting_id}__*.zarr`` folder under the cache root -- deleting the sort
    orphans every recipe. The glob is anchored on the full
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
    reclaimed_paths=(),
) -> dict:
    """Classify analyzer-folder leaks into DB-side, disk-side, and reclaimed.

    Pure (DB-free) set logic for ``Sorting.find_orphaned_analyzer_folders``; the
    caller gathers the DB / filesystem facts (which sorting_ids bear units,
    whether each computed folder exists, which folders sit under the analyzer
    root, and which missing folders were intentionally reclaimed) and this
    function classifies them.

    A **DB-side orphan** is a units-bearing ``Sorting`` row whose computed
    analyzer folder is gone on disk (regeneratable scratch removed out of band);
    reported only, never auto-deleted. A **reclaimed** folder is the same missing
    row-side path, but with a recompute ``deleted=1`` audit trail showing the
    absence was intentional. A **disk-side orphan** is an on-disk folder under
    the analyzer root that no row references (the row was deleted via a path that
    bypassed the ``Sorting.delete`` override). Input order of ``units_bearing``
    and ``disk_dir_paths`` is preserved.

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
    reclaimed_paths : iterable of str, optional
        Computed analyzer paths intentionally removed by the recompute deletion
        workflow (``deleted=1``). Missing units-bearing rows in this set are
        reported separately from unexpected DB-side orphans.

    Returns
    -------
    dict
        ``{"db_side": [{"sorting_id", "computed_analyzer_path"}, ...],
        "disk_side": [folder_path_str, ...],
        "reclaimed": [{"sorting_id", "computed_analyzer_path"}, ...]}``.
    """
    reclaimed_set = set(reclaimed_paths)
    db_side = []
    reclaimed = []
    for sorting_id, computed_path, exists in units_bearing:
        if exists:
            continue
        row = {"sorting_id": sorting_id, "computed_analyzer_path": computed_path}
        if computed_path in reclaimed_set:
            reclaimed.append(row)
        else:
            db_side.append(row)
    referenced = set(referenced_paths)
    disk_side = [path for path in disk_dir_paths if path not in referenced]
    return {"db_side": db_side, "disk_side": disk_side, "reclaimed": reclaimed}
