"""SpikeInterface runtime: global job settings, sorter lock, version gate.

SpikeInterface's sorter wrappers read process-global job settings. Protect
their temporary settings and MountainSort 4's NumPy alias with one lock so
Spyglass calls on separate threads cannot overwrite or inherit each other's
runtime state. Process workers remain independent. Code that reproduces
SpikeInterface internals first checks the installed version with
:func:`require_validated_spikeinterface`.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import NamedTuple

import numpy as np

from spyglass.utils import logger

_SORTER_RUNTIME_LOCK = threading.RLock()


class SIReproduction(NamedTuple):
    """One v2 copy or patch of SpikeInterface internals.

    Attributes
    ----------
    reproduces : str
        What the copy fixes or reproduces.
    validated_si : tuple of str
        SpikeInterface ``major.minor`` releases it is validated on.
    probe : str
        How it checks SpikeInterface at run time, or why it does not.
    """

    reproduces: str
    validated_si: tuple[str, ...]
    probe: str


#: Each v2 copy or patch of SpikeInterface internals, validated on its own so
#: a new release can keep one and drop another (``Dependencies.md``).
SI_REPRODUCTIONS: dict[str, SIReproduction] = {
    "nn_noise_overlap_sparsity": SIReproduction(
        reproduces=(
            "nearest_neighbors_noise_overlap with the median template "
            "sparsified (SI picks the peak channel of the dense template, so "
            "sparse analyzers get NaN) and noise drawn from the statistics "
            "spans"
        ),
        validated_si=("0.104",),
        probe=(
            "none: showing the defect needs a sparse analyzer with median "
            "templates (about 2 s), and the span draw has no upstream "
            "counterpart"
        ),
    ),
    "isolation_kdtree": SIReproduction(
        reproduces=(
            "_compute_isolation's exact neighbor search on scipy's threaded "
            "cKDTree instead of scikit-learn's single-threaded tree"
        ),
        validated_si=("0.104",),
        probe="none: a speed change, not a defect; the version is the gate",
    ),
    "sd_ratio_statistics_spans": SIReproduction(
        reproduces=(
            "compute_sd_ratio's template correction, with spikes and samples "
            "counted over the statistics spans"
        ),
        validated_si=("0.104",),
        probe=(
            "none: SI has no statistics spans, so no upstream defect to "
            "detect; checking the copied correction needs an analyzer"
        ),
    ),
    "span_whitening": SIReproduction(
        reproduces=(
            'compute_whitening_matrix (mode="global", its eps rule) on '
            "samples drawn from the statistics spans"
        ),
        validated_si=("0.104",),
        probe=(
            "the first use in a process compares the copy with SI's matrix on "
            "a small synthetic recording (_sorting.dispatch)"
        ),
    ),
    "expected_nan_metric_eligibility": SIReproduction(
        reproduces=(
            "the spike-count and firing-rate conditions under which SI leaves "
            "a quality metric NaN"
        ),
        validated_si=("0.104",),
        probe="none: checking a condition computes SI metrics on an analyzer",
    ),
    "span_motion_estimation": SIReproduction(
        reproduces=(
            "estimate_motion's peak detection, localization and estimation "
            "steps, run on the statistics spans"
        ),
        validated_si=("0.104",),
        probe="none: a check runs peak detection and DREDge (seconds)",
    ),
}


def require_validated_spikeinterface(reproduction: str) -> None:
    """Refuse a reproduction outside the SI versions it is validated on.

    Parameters
    ----------
    reproduction : str
        Key of the reproduction about to run in :data:`SI_REPRODUCTIONS`.

    Raises
    ------
    RuntimeError
        If the installed SpikeInterface is outside the entry's
        ``validated_si``.
    """
    import spikeinterface

    from spyglass.spikesorting._si_compat import installed_si_release

    entry = SI_REPRODUCTIONS[reproduction]
    if installed_si_release() not in entry.validated_si:
        raise RuntimeError(
            f"spikesorting v2: {reproduction} reproduces SpikeInterface "
            f"{', '.join(entry.validated_si)} internals ({entry.reproduces}), "
            f"but SpikeInterface {spikeinterface.__version__} is installed. "
            "Install the pinned version with the 'spikesorting' extra "
            "(spikeinterface==0.104.3)."
        )


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
    import spikeinterface as si

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
