"""Execution configuration and reproducible seed resolution.

Compute helpers receive this module's output rather than reading config.
Nothing heavy is imported at module load.
"""

from __future__ import annotations

import functools
import numbers
from collections.abc import Mapping

#: Default thread cap for the nn-metric neighbor search.
_DEFAULT_SEARCH_THREADS_CAP = 16


def _usable_cpus() -> int:
    """Cores this process may run on (its affinity mask where available)."""
    import os

    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on macOS / Windows
        return os.cpu_count() or 1


def _repair_openblas_after_fork() -> None:
    """Rebuild each OpenBLAS thread pool in the parent after a fork.

    SciPy's bundled OpenBLAS 0.3.30 otherwise deadlocks on its next float32 LU
    (e.g. MountainSort5 PCA). Resetting the thread count rebuilds the pool.
    """
    try:
        from threadpoolctl import ThreadpoolController
    except ImportError:
        return
    for lib in ThreadpoolController().lib_controllers:
        if lib.internal_api == "openblas":
            lib.set_num_threads(lib.num_threads)


@functools.cache
def install_openblas_fork_repair() -> None:
    """Run :func:`_repair_openblas_after_fork` after every fork (once per process).

    Repairs only the parent; forked workers must stay single-threaded
    (SpikeInterface's default ``max_threads_per_worker=1``).
    """
    import os

    os.register_at_fork(after_in_parent=_repair_openblas_after_fork)


def _explicit_n_jobs(*row_job_kwargs: dict | None):
    """The ``n_jobs`` a layer set (a later row blob first), or ``None``."""
    import datajoint as dj

    from spyglass.spikesorting.v2._core.si_compat import get_global_job_kwargs

    for blob in reversed(row_job_kwargs):
        if blob and "n_jobs" in blob:
            return blob["n_jobs"]
    custom = (dj.config.get("custom", {}) or {}).get(
        "spikesorting_v2_job_kwargs", {}
    ) or {}
    if "n_jobs" in custom:
        return custom["n_jobs"]
    si_global = get_global_job_kwargs().get("n_jobs", 1)
    return None if si_global == 1 else si_global


def resolve_search_threads(*row_job_kwargs: dict | None) -> int:
    """Threads for the exact neighbor searches of the nn_advanced metrics.

    An explicit ``n_jobs`` (per-row blobs, then
    ``dj.config['custom']['spikesorting_v2_job_kwargs']``, then a non-default SI
    global) wins. Otherwise a quarter of the usable cores, at most
    ``_DEFAULT_SEARCH_THREADS_CAP``.

    Parameters
    ----------
    *row_job_kwargs : dict or None
        The stage's per-row ``job_kwargs`` blobs, in increasing precedence.

    Returns
    -------
    int
        At least 1.
    """
    explicit = _explicit_n_jobs(*row_job_kwargs)
    if explicit is None:
        return max(1, min(_DEFAULT_SEARCH_THREADS_CAP, _usable_cpus() // 4))
    from spikeinterface.core.job_tools import fix_job_kwargs

    return fix_job_kwargs({"n_jobs": explicit})["n_jobs"]


def _ambient_job_kwargs() -> dict:
    """Return the ambient job-kwargs layer: SI globals then dj.config custom.

    Shared by :func:`_resolved_job_kwargs` and :func:`resolve_effective_seed`.
    """
    import datajoint as dj

    from spyglass.spikesorting.v2._core.si_compat import get_global_job_kwargs

    merged = get_global_job_kwargs()
    custom = dj.config.get("custom", {}) or {}
    merged.update(custom.get("spikesorting_v2_job_kwargs", {}) or {})
    return merged


def _resolved_job_kwargs(*row_job_kwargs: dict | None) -> dict:
    """Merge SpikeInterface-global, DataJoint-config, and per-row job kwargs.

    Sources are merged in increasing precedence order: the SpikeInterface
    global defaults, then ``dj.config['custom']['spikesorting_v2_job_kwargs']``,
    then each per-row blob in the order given.

    Parameters
    ----------
    *row_job_kwargs : dict or None
        ``job_kwargs`` blob values from the parameter rows that govern this
        compute stage, in increasing precedence order (a later argument wins
        on key conflict). ``None`` and empty-dict entries are skipped.

    Returns
    -------
    dict
        The merged kwargs, ready to splat into a compute call.
    """
    merged = _ambient_job_kwargs()
    for override in row_job_kwargs:
        if override:
            merged.update(override)
    return merged


@functools.lru_cache(maxsize=None)
def _warn_ambient_seed_once(seed: int) -> None:
    """Emit the ambient-seed warning at most once per distinct seed value."""
    from spyglass.utils import logger

    logger.warning(
        "spikesorting v2: random_seed=%s is supplied via the ambient "
        "SI-global / dj.config['custom']['spikesorting_v2_job_kwargs'] layer, "
        "not a per-row job_kwargs blob. It is used and recorded as the "
        "effective seed, but an ambient seed is process-global rather than "
        "pinned to a parameter row, so the run is harder to reproduce. Prefer "
        "setting random_seed in the per-row job_kwargs.",
        seed,
    )


def resolve_effective_seed(
    *row_job_kwargs: dict | None, reject_ambient_seed: bool = False
) -> int:
    """Return the ``random_seed`` actually used by a v2 compute stage.

    Uses the same merge as :func:`_resolved_job_kwargs`, so the stored seed is
    the seed the sorter or analyzer consumed. Defaults to ``0``. Warns once per
    value when the seed comes only from the ambient SI-global / ``dj.config``
    layer, since that seed is not pinned to a parameter row.

    Parameters
    ----------
    *row_job_kwargs : dict or None
        The per-row ``job_kwargs`` blob(s) governing this stage, in increasing
        precedence order (a later argument wins). ``None`` / empty entries are
        skipped, matching :func:`_resolved_job_kwargs`.
    reject_ambient_seed : bool, optional
        Raise instead of warning on an ambient-only ``random_seed``. The sorting
        stage passes this because an ambient seed is absent from
        ``sorting_id``. Default ``False``.

    Returns
    -------
    int
        The resolved effective random seed (``0`` when unset).

    Raises
    ------
    ValueError
        If the resolved ``random_seed`` is not a non-negative integer (e.g.
        ``"7"``, ``7.9``, or ``-1``), or if ``reject_ambient_seed`` and the seed
        is ambient-only.
    """
    resolved = _resolved_job_kwargs(*row_job_kwargs).get("random_seed", 0)
    # Reject bool (an int subclass); accept numpy integers.
    if (
        isinstance(resolved, bool)
        or not isinstance(resolved, numbers.Integral)
        or resolved < 0
    ):
        raise ValueError(
            "spikesorting v2 random_seed must be a non-negative integer, got "
            f"{resolved!r} ({type(resolved).__name__}). Set an int random_seed "
            "in the parameter row's job_kwargs blob (for a sort, the "
            "SorterParameters row, since the sort rejects an ambient seed)."
        )
    effective = int(resolved)
    per_row_has_seed = any(
        isinstance(blob, Mapping) and "random_seed" in blob
        for blob in row_job_kwargs
    )
    if not per_row_has_seed:
        ambient = _ambient_job_kwargs()
        if "random_seed" in ambient:
            if reject_ambient_seed:
                raise ValueError(
                    "spike sorting requires random_seed to live in the "
                    "SorterParameters job_kwargs (where it is part of the "
                    "sort's identity via sorter_params_name), not the ambient "
                    "dj.config['custom']['spikesorting_v2_job_kwargs'] / "
                    "SI-global layer. An ambient random_seed="
                    f"{int(ambient['random_seed'])} changes the sort output "
                    "WITHOUT changing sorting_id, so a later seed change would "
                    "silently reuse this sort. Move random_seed into the "
                    "SorterParameters row (or clear the ambient seed)."
                )
            _warn_ambient_seed_once(int(ambient["random_seed"]))
    return effective
