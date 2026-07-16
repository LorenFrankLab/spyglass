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

import os
import re
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

# The recipe name is embedded in the analyzer cache folder
# ``{sorting_id}__{waveform_params_name}.zarr`` (see ``analyzer_path``), so it
# must be path-safe -- no separators, dots, or traversal. Validated both at
# insert (``AnalyzerWaveformParameters``) and at load (``get_analyzer`` accepts
# an un-FK'd free-string recipe name). Lives here -- the DB-free owner of the
# analyzer-path policy -- so the load path can validate without importing the
# ``sorting`` schema module.
_WAVEFORM_PARAMS_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def assert_path_safe_waveform_params_name(name) -> None:
    """Reject a ``waveform_params_name`` that is not path-safe.

    It must match ``^[A-Za-z0-9_]+$`` (letters, digits, underscore) because it
    is embedded in the analyzer cache folder name.
    """
    if not _WAVEFORM_PARAMS_NAME_RE.match(str(name)):
        raise ValueError(
            "AnalyzerWaveformParameters: waveform_params_name "
            f"{name!r} is not path-safe; it is embedded in the analyzer "
            "cache folder name, so it must match ^[A-Za-z0-9_]+$ (letters, "
            "digits, underscore)."
        )


def is_canonical_analyzer_folder_name(name: str) -> bool:
    """Whether ``name`` is a canonical analyzer-cache folder name.

    Canonical analyzer folders are ``{sorting_id}__{waveform_params_name}.zarr``
    (see :func:`analyzer_path`): a UUID sorting id, a ``__`` separator, a
    path-safe recipe name, and the ``.zarr`` suffix. The disk-side orphan sweep
    uses this to refuse deleting any directory under a (possibly misconfigured)
    analyzer root that is not a canonical analyzer cache.
    """
    if not name.endswith(".zarr") or "__" not in name:
        return False
    sorting_id, _, recipe = name[: -len(".zarr")].partition("__")
    try:
        uuid.UUID(sorting_id)
    except ValueError:
        return False
    return bool(_WAVEFORM_PARAMS_NAME_RE.match(recipe))


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
    return analyzer_cache_root() / f"{sorting_id}__{waveform_params_name}.zarr"


def analyzer_cache_lock(sorting_id):
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

    The lock blocks indefinitely by default (the intended serialize-don't-fail
    behavior); it releases when the holding process exits, so a crashed job
    cannot wedge the next one. A per-call timeout is available via
    ``analyzer_cache_lock(sorting_id).acquire(timeout=...)`` -- the memoized
    instance has no constructor-level timeout knob because a second caller would
    silently inherit the first's value.

    Parameters
    ----------
    sorting_id
        The sorting whose analyzer-cache folders the job will access.

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
            lock = FileLock(lock_path)
            _ANALYZER_LOCK_REGISTRY[lock_path] = lock
        return lock


# Backwards-compatible alias: callers predating the cache-wide rename (the
# ``CurationEvaluation`` fast path, the recording-lock docstring cross-ref) used
# ``analyzer_curation_lock`` when the lock guarded only ``AnalyzerCuration``.
analyzer_curation_lock = analyzer_cache_lock


def _publish_sibling(canonical_folder, kind: str) -> Path:
    """Return a hidden ``.zarr`` sibling of ``canonical_folder`` for staging.

    The build/move-aside folders MUST sit in the SAME directory as the canonical
    slot (not a sub-directory): a SpikeInterface zarr ``SortingAnalyzer`` stores
    its recording reference as a path RELATIVE to the analyzer folder, so the
    publish ``os.replace`` only preserves that reference -- and therefore the
    ability to (re)compute recording-dependent extensions like
    ``spike_amplitudes`` after a load -- when the temp and the canonical slot are
    at the same depth relative to the recording. A leading ``.`` keeps the
    staging folder hidden from the orphan scan (which skips dotted entries) and
    from ``remove_analyzer_cache``'s ``{sorting_id}__*.zarr`` glob, and the
    ``os.getpid()`` token avoids collisions across concurrent processes.
    """
    parent = canonical_folder.parent
    return parent / (
        f".{canonical_folder.stem}.{kind}-{os.getpid()}{canonical_folder.suffix}"
    )


def _sorting_id_from_analyzer_folder(canonical_folder) -> str:
    """Return the sorting id embedded in a canonical analyzer folder path."""
    folder = Path(canonical_folder)
    if not folder.name.endswith(".zarr") or "__" not in folder.name:
        raise ValueError(
            "publish_analyzer_atomically requires a canonical analyzer-cache "
            "folder named '{sorting_id}__{waveform_params_name}.zarr'; got "
            f"{folder.name!r}."
        )
    sorting_id, _, _recipe = folder.name[: -len(".zarr")].partition("__")
    if not sorting_id or not _WAVEFORM_PARAMS_NAME_RE.match(_recipe):
        raise ValueError(
            "publish_analyzer_atomically requires a canonical analyzer-cache "
            "folder named '{sorting_id}__{waveform_params_name}.zarr' with a "
            f"path-safe waveform_params_name; got {folder.name!r}."
        )
    return sorting_id


def publish_analyzer_atomically(canonical_folder, build_into):
    """Build an analyzer into a private temp folder, then move it into the slot.

    The low-level build (``build_into``) writes into a FRESH private folder that
    is a HIDDEN SIBLING of ``canonical_folder`` (same parent directory, so the
    move is a rename AND the analyzer's relative recording path survives it --
    see :func:`_publish_sibling`); only on success is that temp moved into the
    canonical slot. This is the canonical-cache safety layer that keeps a build
    from writing straight into (and corrupting) the live folder a reader might be
    loading -- distinct from ``build_into`` itself, which simply builds wherever
    it is told (recompute / merged-curation temp analyzers reuse the low-level
    build directly, never this publisher).

    A directory rename is NOT a clean atomic swap: POSIX ``rename(2)`` requires
    the destination to be empty (else ``ENOTEMPTY``), so a rebuild over an
    existing ``.zarr`` cannot ``os.replace`` straight onto it. The publish
    sequence is therefore:

    - canonical **absent**  -> ``os.replace(temp, canonical)``;
    - canonical **present** -> ``os.replace(canonical, trash)``,
      ``os.replace(temp, canonical)``, ``rmtree(trash)``. If the install move
      fails, the original is moved back from trash so the slot is restored; if
      THAT restore also fails (a rare double failure), the trash is LEFT on disk
      (the only surviving copy) for manual recovery -- the slot may be absent
      but is recoverable.

    The brief window where the canonical slot is moved aside is reader-safe
    because this publisher acquires ``analyzer_cache_lock(sorting_id)`` (the read
    path takes the same lock); the directory replace is not itself atomic.
    Callers that already hold the lock remain safe because the lock is
    process-reentrant.
    ``build_into`` that writes no folder (a zero-unit sort short-circuits before
    building) leaves the slot untouched. The temp is always removed; the trash
    is removed only after a successful install (never on the failed-rollback
    path, where it is the surviving copy).

    Parameters
    ----------
    canonical_folder : pathlib.Path
        The cache slot to publish into (``analyzer_path(sorting_id, recipe)``).
    build_into : callable
        ``build_into(temp_folder: pathlib.Path) -> None`` builds the analyzer
        into the given private temp folder (it may legitimately write nothing
        for a zero-unit sort).

    Returns
    -------
    pathlib.Path
        ``canonical_folder``.
    """
    canonical_folder = Path(canonical_folder)
    sorting_id = _sorting_id_from_analyzer_folder(canonical_folder)
    with analyzer_cache_lock(sorting_id):
        return _publish_analyzer_atomically_unlocked(
            canonical_folder, build_into
        )


def _publish_analyzer_atomically_unlocked(canonical_folder, build_into):
    """Publish implementation; caller holds ``analyzer_cache_lock``."""
    canonical_folder.parent.mkdir(parents=True, exist_ok=True)
    temp_folder = _publish_sibling(canonical_folder, "build")
    # Clear any leftover from a crashed prior build before reusing the name.
    shutil.rmtree(temp_folder, ignore_errors=True)
    try:
        build_into(temp_folder)
        if not temp_folder.exists():
            # Nothing built (zero-unit short-circuit): leave the slot as-is.
            return canonical_folder
        if not canonical_folder.exists():
            os.replace(temp_folder, canonical_folder)
            return canonical_folder
        # Rebuild over an existing folder: move it aside, install the temp, then
        # drop the aside copy. Restore it if the install fails so the slot is
        # never left empty.
        trash_folder = _publish_sibling(canonical_folder, "trash")
        shutil.rmtree(trash_folder, ignore_errors=True)
        os.replace(canonical_folder, trash_folder)
        try:
            os.replace(temp_folder, canonical_folder)
        except Exception:
            # Install failed: restore the original from trash so the slot is
            # never left empty. If THIS also fails, leave the trash on disk for
            # manual recovery -- it is the only surviving copy, so it must not
            # be deleted (hence the cleanup is NOT in a ``finally``).
            os.replace(trash_folder, canonical_folder)
            raise
        # Install succeeded: the trash is the now-stale old folder; drop it.
        shutil.rmtree(trash_folder, ignore_errors=True)
        return canonical_folder
    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)


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
        row = {
            "sorting_id": sorting_id,
            "computed_analyzer_path": computed_path,
        }
        if computed_path in reclaimed_set:
            reclaimed.append(row)
        else:
            db_side.append(row)
    referenced = set(referenced_paths)
    disk_side = [path for path in disk_dir_paths if path not in referenced]
    return {"db_side": db_side, "disk_side": disk_side, "reclaimed": reclaimed}
