"""Process-level helpers for multiprocess and multithreaded compute."""

from __future__ import annotations

import functools
import os


def usable_cpus() -> int:
    """Cores this process may run on (its affinity mask where available)."""
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
    os.register_at_fork(after_in_parent=_repair_openblas_after_fork)
