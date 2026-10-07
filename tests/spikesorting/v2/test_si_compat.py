"""Sorter calls and ambient readers cannot share temporary SI globals."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest


@pytest.mark.parametrize("first_fails", [False, True])
def test_concurrent_sorters_and_ambient_reader_restore_their_own_state(
    tmp_path, monkeypatch, first_fails
):
    """Force a second MS4 call to overlap the first call's temporary state."""
    import numpy as np
    import spikeinterface as si
    import spikeinterface.sorters as sis

    from spyglass import settings
    from spyglass.spikesorting.v2 import _si_compat
    from spyglass.spikesorting.v2._sorting_dispatch import run_si_sorter
    from spyglass.spikesorting.v2.utils import _ambient_job_kwargs

    monkeypatch.setattr(settings, "temp_dir", str(tmp_path))
    monkeypatch.delattr(np, "Inf", raising=False)
    original = dict(si.get_global_job_kwargs())
    first_inside, second_attempting, reader_attempting = (
        Event(),
        Event(),
        Event(),
    )
    release_first, first_exited, second_inside = Event(), Event(), Event()
    recording = si.NumpyRecording(
        np.zeros((100, 4), dtype=np.float32), sampling_frequency=1000.0
    )
    sorting = si.NumpySorting.from_unit_dict(
        {1: np.array([20, 40])}, sampling_frequency=1000.0
    )
    runtime_state = _si_compat.sorter_runtime_state

    def requested_runtime_state(sorter, jobs):
        if jobs["n_jobs"] == 3:
            second_attempting.set()
        return runtime_state(sorter, jobs)

    monkeypatch.setattr(
        _si_compat, "sorter_runtime_state", requested_runtime_state
    )

    def run_sorter(**kwargs):
        jobs = si.get_global_job_kwargs()
        assert np.Inf == np.inf
        if jobs["n_jobs"] == 2:
            first_inside.set()
            assert release_first.wait(5), "first sorter was not released"
            assert si.get_global_job_kwargs()["n_jobs"] == 2
            assert hasattr(np, "Inf"), "a concurrent call removed the alias"
            first_exited.set()
            if first_fails:
                raise RuntimeError("first sorter failed")
        else:
            second_inside.set()
            assert first_exited.is_set(), "second sorter entered before restore"
            assert jobs["n_jobs"] == 3
        return sorting

    monkeypatch.setattr(sis, "run_sorter", run_sorter)

    def run(n_jobs):
        return run_si_sorter(
            "mountainsort4", {}, recording, str(n_jobs), {"n_jobs": n_jobs}
        )

    def read_ambient():
        reader_attempting.set()
        return _ambient_job_kwargs()

    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(run, 2)
        try:
            assert first_inside.wait(5)
            second = pool.submit(run, 3)
            ambient = pool.submit(read_ambient)
            assert second_attempting.wait(5)
            assert reader_attempting.wait(5)
            assert not second_inside.wait(0.1)
            assert (
                not ambient.done()
            ), "ambient reader inherited active settings"
        finally:
            release_first.set()
        if first_fails:
            with pytest.raises(RuntimeError, match="first sorter failed"):
                first.result(timeout=5)
        else:
            np.testing.assert_array_equal(first.result(timeout=5).unit_ids, [1])
        np.testing.assert_array_equal(second.result(timeout=5).unit_ids, [1])
        # dj.config may add settings, but the temporary n_jobs must be absent.
        assert ambient.result(timeout=5)["n_jobs"] == original["n_jobs"]

    assert si.get_global_job_kwargs() == original
    assert not hasattr(np, "Inf")


def test_nested_sorter_state_restores_outer_settings(monkeypatch):
    """Reentrant compatibility scopes restore the enclosing caller's state."""
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2._si_compat import sorter_runtime_state

    monkeypatch.delattr(np, "Inf", raising=False)
    original = dict(si.get_global_job_kwargs())
    with sorter_runtime_state("mountainsort4", {"n_jobs": 2}):
        with sorter_runtime_state("mountainsort4", {"n_jobs": 3}):
            assert si.get_global_job_kwargs()["n_jobs"] == 3
            assert np.Inf == np.inf
        assert si.get_global_job_kwargs()["n_jobs"] == 2
        assert np.Inf == np.inf
    assert si.get_global_job_kwargs() == original
    assert not hasattr(np, "Inf")


def test_existing_numpy_alias_is_preserved(monkeypatch):
    import numpy as np

    from spyglass.spikesorting.v2._si_compat import sorter_runtime_state

    existing_alias = object()
    monkeypatch.setattr(np, "Inf", existing_alias, raising=False)
    with sorter_runtime_state("mountainsort4", {}):
        assert np.Inf is existing_alias
    assert np.Inf is existing_alias
