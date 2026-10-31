"""Execution configuration and reproducible seed resolution.

This module owns the ambient configuration boundary. Resolved-input compute
helpers receive its output rather than resolving database-backed parameters.
No runtime configuration or SpikeInterface dependency is loaded at import.
"""

from __future__ import annotations

import functools
import numbers
from collections.abc import Mapping

#: Most threads an exact nn-metric neighbor search uses by default (see
#: ``resolve_search_threads``).
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

    SciPy's bundled OpenBLAS (0.3.30) shuts its pool down in its fork handler,
    and its next parallel single-precision LU then deadlocks rebuilding it,
    blocked on ``server_lock``: MountainSort5's float32 PCA hung after
    SpikeInterface's forked workers. Setting a pool's thread count again
    rebuilds it cleanly, in ~2 ms.
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

    A Python-level hook covers ``os.fork`` and multiprocessing's fork start
    method, which SpikeInterface's process pools use. It repairs only the
    parent: multithreaded BLAS inside forked workers hangs even in float64 and
    without this hook, and the same repair in the child did not help, so
    workers must stay single-threaded (SpikeInterface's default
    ``max_threads_per_worker=1``). ``subprocess`` forks below both Python and
    OpenBLAS's fork handler and leaves the pool intact.
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

    An ``n_jobs`` set by a layer (per-row blobs, then
    ``dj.config['custom']['spikesorting_v2_job_kwargs']``, then a changed SI
    global) sets it. Otherwise a quarter of the usable cores, at most
    ``_DEFAULT_SEARCH_THREADS_CAP``: the search is ~75% of curation-evaluation
    metrics and its threads add no memory (on 11 real searches 1 / 4 / 16
    threads took 8.9 / 2.7 / 0.74 s). ``n_jobs`` itself stays at
    SpikeInterface's one worker unless a layer sets it: on a real 20 min epoch
    four worker processes saved only ~6 s, and sorting and the evaluation did
    not get faster.

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

    The process-global precedence stack beneath any per-row blob -- the
    SpikeInterface global defaults overlaid with
    ``dj.config['custom']['spikesorting_v2_job_kwargs']``. Single source of this
    merge so :func:`_resolved_job_kwargs` and :func:`resolve_effective_seed`
    (which needs the ambient layer in isolation to attribute a seed) cannot
    drift.
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
    """Emit the ambient-seed warning at most once per distinct seed value.

    Deduped per ``seed`` value so a repeated populate (or several sort groups
    sharing one ambient seed) does not spam the log, while a genuinely new
    ambient seed is still surfaced. ``cache_clear()`` resets the dedup (used by
    the unit tests).
    """
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

    Resolves the seed through the same precedence the dispatch reads --
    SpikeInterface globals, then ``dj.config['custom']['spikesorting_v2_job_kwargs']``,
    then the per-row ``job_kwargs`` blob(s) -- so the value stored on a computed
    row equals the value the sorter / analyzer consumed. The resolution bottoms
    out on :func:`_resolved_job_kwargs`, the exact merge the seed sites read, so
    the stored seed cannot drift from the used seed. Defaults to ``0``.

    Emits a one-time warning (per distinct seed value) when a ``random_seed``
    arrives via the ambient SI-global / ``dj.config`` layer rather than a
    per-row blob -- a process-global ambient seed already takes effect, but it
    is not pinned to a parameter row, so surfacing it keeps a non-reproducible
    ambient seed visible rather than silent. At the sort-identity boundary
    (``reject_ambient_seed=True``) the same ambient-only seed is a hard error
    instead: a ``Sorting`` whose seed is not folded into its identity (via
    ``sorter_params_name``) could be silently reused after a later seed change.

    Parameters
    ----------
    *row_job_kwargs : dict or None
        The per-row ``job_kwargs`` blob(s) governing this stage, in increasing
        precedence order (a later argument wins). ``None`` / empty entries are
        skipped, matching :func:`_resolved_job_kwargs`.
    reject_ambient_seed : bool, optional
        When ``True``, an ambient-only ``random_seed`` (present in the SI-global
        / ``dj.config`` layer but not in any per-row blob) raises ``ValueError``
        instead of warning. The sorting stage passes this so a stochastic sort
        cannot be created with a seed absent from ``sorting_id``. Default
        ``False`` (warn), for stages where the seed is captured as provenance.

    Returns
    -------
    int
        The resolved effective random seed (``0`` when unset).

    Raises
    ------
    ValueError
        If the resolved ``random_seed`` is not a non-negative integer (e.g.
        ``"7"``, ``7.9``, or ``-1``). The seed sites consume this resolved value
        directly, so a bad seed is rejected here -- BEFORE compute -- rather than
        silently ``int()``-coerced (which would store ``7`` while the sorter saw
        the original object) or deferred to a later SI/NumPy RNG failure. This is
        the call that runs first in ``make_compute``, so a bad seed aborts the
        populate before any row is written. Mirrors the ``seed >= 0`` constraint
        on the UnitMatch bundle-seed schema.
    """
    resolved = _resolved_job_kwargs(*row_job_kwargs).get("random_seed", 0)
    # ``bool`` is an ``int`` subclass but never a valid seed; ``numbers.Integral``
    # accepts Python and numpy integers (which are seed-equivalent to ``int``).
    # SI/NumPy RNG seeds must be non-negative, so reject ``< 0`` here too.
    if (
        isinstance(resolved, bool)
        or not isinstance(resolved, numbers.Integral)
        or resolved < 0
    ):
        raise ValueError(
            "spikesorting v2 random_seed must be a non-negative integer, got "
            f"{resolved!r} ({type(resolved).__name__}). Set an int random_seed "
            "in the per-row job_kwargs blob (or "
            "dj.config['custom']['spikesorting_v2_job_kwargs'])."
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
