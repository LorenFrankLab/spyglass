"""Scoped SpikeInterface sorter compatibility state.

SpikeInterface's sorter wrappers read process-global job settings. Protect
their temporary settings and MountainSort 4's NumPy alias with one lock so
Spyglass calls on separate threads cannot overwrite or inherit each other's
runtime state. Process workers remain independent.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager

_SORTER_RUNTIME_LOCK = threading.RLock()


def _reset_sorter_runtime_lock():
    """A forked worker must not inherit another thread's locked mutex."""
    global _SORTER_RUNTIME_LOCK
    _SORTER_RUNTIME_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_sorter_runtime_lock)


def get_global_job_kwargs() -> dict:
    """Read ambient SI settings after any other thread's sorter restores them."""
    import spikeinterface as si

    with _SORTER_RUNTIME_LOCK:
        return dict(si.get_global_job_kwargs())


@contextmanager
def sorter_runtime_state(sorter: str, job_kwargs: dict):
    """Install sorter globals until its output has been materialized.

    All Spyglass sorter calls, including those with empty job overrides, take
    this process-wide lock. Ambient-setting readers take the same lock. The
    lock is reentrant for nested calls and is reset in forked workers. Direct
    third-party SI/NumPy mutations do not participate in this protocol.
    """
    import numpy as np
    import spikeinterface as si

    from spyglass.utils import logger

    with _SORTER_RUNTIME_LOCK:
        previous_global = dict(si.get_global_job_kwargs())
        patched_numpy_inf = sorter.lower() == "mountainsort4" and not hasattr(
            np, "Inf"
        )
        try:
            # spikeextractors 0.9.11 references the alias removed by NumPy 2.
            if patched_numpy_inf:
                np.Inf = np.inf
            if job_kwargs:
                si.set_global_job_kwargs(**job_kwargs)
            yield
        finally:
            if job_kwargs:
                try:
                    # SI updates its dict; resetting also removes new keys.
                    si.reset_global_job_kwargs()
                    si.set_global_job_kwargs(**previous_global)
                except Exception as restore_exc:
                    logger.warning(
                        "run_si_sorter: failed to restore SI "
                        f"global job kwargs to {previous_global!r}: "
                        f"{restore_exc!r}. Original sort exception (if "
                        "any) preserved."
                    )
            if patched_numpy_inf and hasattr(np, "Inf"):
                del np.Inf
