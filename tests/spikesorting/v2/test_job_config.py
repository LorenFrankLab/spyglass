"""Default worker count for v2's parallel stages.

Measured on a 20 min, 32-channel epoch, results are identical for any n_jobs,
and 4 workers capture most of the speedup (curation evaluation 208 / 114 / 100
/ 90 s at 1 / 4 / 8 / 16 workers), so v2 uses a quarter of the usable cores,
at most 4, unless a layer sets ``n_jobs``. Hermetic -- no DB.
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


@pytest.mark.parametrize("cpus, expected", [(96, 4), (16, 4), (8, 2), (2, 1)])
def test_default_n_jobs_is_a_quarter_of_usable_cores_at_most_four(
    monkeypatch, no_v2_job_config, cpus, expected
):
    from spyglass.spikesorting.v2._core.job_config import _resolved_job_kwargs

    _usable_cpus(monkeypatch, cpus)
    assert _resolved_job_kwargs()["n_jobs"] == expected


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
    """The exact nn-metric neighbor search runs on threads, which add no
    memory: on 11 real searches, 4 / 16 threads took 2.7 / 0.74 s. Its
    default is therefore wider than the worker-process default."""
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
