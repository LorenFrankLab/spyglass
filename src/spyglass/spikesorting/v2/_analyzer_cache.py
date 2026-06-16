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


def analyzer_path(sorting_id) -> Path:
    """Return the canonical analyzer-cache folder for a ``sorting_id``.

    ``analyzer_cache_root() / f"{sorting_id}.analyzer"``. Deterministic in
    ``sorting_id`` + the configured root, so every code path resolves the
    same folder without the ``Sorting`` row needing to store it.

    Returns
    -------
    pathlib.Path
        The analyzer-cache folder path for this ``sorting_id``.
    """
    return analyzer_cache_root() / f"{sorting_id}.analyzer"


def remove_analyzer_cache(sorting_id, *, missing_ok: bool = True) -> bool:
    """Remove a sorting's analyzer-cache folder.

    ``missing_ok=True`` (default) makes an absent folder a no-op returning
    ``False`` (the common case: zero-unit sorts never wrote one, and the
    cache is regeneratable). ``missing_ok=False`` raises ``FileNotFoundError``
    for an absent folder. A removal failure (e.g. a permission error)
    propagates rather than being swallowed.

    Parameters
    ----------
    sorting_id
        The sorting whose analyzer-cache folder should be removed.
    missing_ok : bool, optional
        If ``True`` (the default), an absent folder is a no-op. If
        ``False``, an absent folder raises ``FileNotFoundError``.

    Returns
    -------
    bool
        ``True`` if a folder was removed, ``False`` if none existed and
        ``missing_ok=True``.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist and ``missing_ok=False``.
    """
    folder = analyzer_path(sorting_id)
    if not folder.exists():
        if missing_ok:
            return False
        raise FileNotFoundError(folder)
    shutil.rmtree(folder, ignore_errors=False)
    return True
