"""Default worker count for v2's parallel stages.

One worker process unless a layer sets ``n_jobs`` (four workers saved ~6 s
on a 20 min, 32-channel epoch). Neighbor-search threads have their own
default. Forked workers must leave SciPy's OpenBLAS usable. No DB.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def no_v2_job_config(monkeypatch):
    """Clear ``dj.config['custom']['spikesorting_v2_job_kwargs']``."""
    import datajoint as dj

    custom = dict(dj.config.get("custom", {}) or {})
    custom.pop("spikesorting_v2_job_kwargs", None)
    monkeypatch.setitem(dj.config, "custom", custom)


def _usable_cpus(monkeypatch, n):
    import os

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(n)))


@pytest.mark.parametrize("cpus", [96, 16, 2])
def test_default_n_jobs_is_one_worker_on_any_machine(
    monkeypatch, no_v2_job_config, cpus
):
    from spyglass.spikesorting.v2._core.job_config import _resolved_job_kwargs

    _usable_cpus(monkeypatch, cpus)
    assert _resolved_job_kwargs()["n_jobs"] == 1


def test_configured_and_row_n_jobs_override_the_default(
    monkeypatch, no_v2_job_config
):
    import datajoint as dj

    from spyglass.spikesorting.v2._core.job_config import _resolved_job_kwargs

    _usable_cpus(monkeypatch, 96)
    assert _resolved_job_kwargs({"n_jobs": 2})["n_jobs"] == 2
    monkeypatch.setitem(
        dj.config, "custom", {"spikesorting_v2_job_kwargs": {"n_jobs": 1}}
    )
    assert _resolved_job_kwargs()["n_jobs"] == 1
    assert _resolved_job_kwargs({"n_jobs": 3})["n_jobs"] == 3


def test_a_changed_spikeinterface_global_n_jobs_is_honored(
    monkeypatch, no_v2_job_config
):
    import spikeinterface as si

    from spyglass.spikesorting.v2._core.job_config import _resolved_job_kwargs

    _usable_cpus(monkeypatch, 96)
    original = dict(si.get_global_job_kwargs())
    si.set_global_job_kwargs(n_jobs=6)
    try:
        assert _resolved_job_kwargs()["n_jobs"] == 6
    finally:
        si.set_global_job_kwargs(**original)


@pytest.mark.parametrize("cpus, expected", [(96, 16), (16, 4), (8, 2), (2, 1)])
def test_default_search_threads_are_a_quarter_of_usable_cores_at_most_16(
    monkeypatch, no_v2_job_config, cpus, expected
):
    """The nn-metric neighbor search thread default is independent of
    ``n_jobs``; threads add no memory (4 / 16 threads: 2.7 / 0.74 s)."""
    from spyglass.spikesorting.v2._core.job_config import (
        resolve_search_threads,
    )

    _usable_cpus(monkeypatch, cpus)
    assert resolve_search_threads() == expected


def test_an_explicit_n_jobs_sets_the_search_threads(
    monkeypatch, no_v2_job_config
):
    import datajoint as dj

    from spyglass.spikesorting.v2._core.job_config import (
        resolve_search_threads,
    )

    _usable_cpus(monkeypatch, 96)
    assert resolve_search_threads({"n_jobs": 3}) == 3
    monkeypatch.setitem(
        dj.config, "custom", {"spikesorting_v2_job_kwargs": {"n_jobs": 1}}
    )
    assert resolve_search_threads() == 1
    assert resolve_search_threads(None, {"n_jobs": 2}) == 2


_FORK_THEN_LU = """
import multiprocessing as mp
import numpy as np
import scipy.linalg
import spyglass.spikesorting.v2  # noqa: F401  (installs the fork repair)


def _noop(value):
    return value


if __name__ == "__main__":
    x = np.random.default_rng(0).normal(size=(4000, 4000)).astype("float32")
    with mp.get_context("fork").Pool(2) as pool:
        pool.map(_noop, range(4))
    scipy.linalg.lu(x, permute_l=True)
    print("lu done")
"""


def test_forked_workers_leave_scipy_float32_lu_usable(tmp_path):
    """Importing v2 keeps float32 LU usable after a fork.

    SciPy's bundled OpenBLAS 0.3.30 otherwise deadlocks on ``server_lock``
    (hit by MountainSort5's PCA in SpikeInterface workers).
    """
    import subprocess
    import sys

    script = tmp_path / "fork_then_lu.py"
    script.write_text(_FORK_THEN_LU)
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("float32 scipy.linalg.lu deadlocked after a fork")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "lu done" in result.stdout
