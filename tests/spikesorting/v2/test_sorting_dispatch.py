"""``Sorting._run_si_sorter`` dispatch invariants.

Covers the MS4 ``numpy.Inf`` global shim teardown, tempdir-cleanup not masking
the real sort exception, SI global ``job_kwargs`` set/restore, and the
MATLAB-sorter Singularity carve-out (with its non-MATLAB contrast).
"""

from __future__ import annotations

import pytest


def _run_si_sorter_with_patched_run_sorter(monkeypatch, run_sorter_impl):
    """Drive Sorting._run_si_sorter with a cheap recording and a patched
    sis.run_sorter, returning (before_global, after_global, result_or_exc).

    Passes a non-empty job_kwargs ({"n_jobs": 2, ...}) so the global
    set/restore path actually runs (it is gated on ``if sj_kwargs``), and
    n_jobs=2 differs from SI's default n_jobs=1 so a missing restore would
    leave the global changed.
    """
    import uuid

    import spikeinterface as si
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import Sorting

    rec = si.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=30_000.0
    )
    monkeypatch.setattr(sis, "run_sorter", run_sorter_impl)

    before = dict(si.get_global_job_kwargs())
    result = Sorting._run_si_sorter(
        "mountainsort5",
        {},
        rec,
        uuid.uuid4(),
        {"n_jobs": 2, "chunk_duration": "2s"},
    )
    after = dict(si.get_global_job_kwargs())
    return before, after, result


@pytest.mark.slow
@pytest.mark.usefixtures("dj_conn")
def test_run_si_sorter_does_not_leak_numpy_inf(monkeypatch):
    """The MS4 ``np.Inf`` shim is scoped and torn down.

    The MS4 wrapper (via spikeextractors) references the numpy-2.0-removed
    ``np.Inf`` alias, so ``_run_si_sorter`` restores it for the MS4 call.
    The restore must be deleted afterward; a persistent global mutation
    would leak a different numpy into every later module that probes
    ``hasattr(np, "Inf")``.

    The shim is set BEFORE ``run_sorter`` is invoked, so the leak is
    observable whether or not MS4 itself is installed (on the CI SI-0.104
    image it is not, and ``run_sorter`` raises -- which the test
    tolerates). ``monkeypatch.delattr`` locks a clean ``np.Inf``-absent
    baseline so the assertion is order-independent.
    """
    import numpy as np_mod
    import spikeinterface.core as sc

    from spyglass.spikesorting.v2.sorting import Sorting

    monkeypatch.delattr(np_mod, "Inf", raising=False)
    assert not hasattr(np_mod, "Inf"), "baseline not clean"

    rec = sc.generate_recording(
        num_channels=4, durations=[0.5], sampling_frequency=30000.0
    )
    try:
        Sorting._run_si_sorter(
            sorter="mountainsort4",
            sorter_params={},
            recording=rec,
            sorting_id="audit-a8-leak-check",
            job_kwargs=None,
        )
    except Exception:
        # MS4 not installed (or sort failure) is fine -- the np.Inf shim
        # already fired before run_sorter, so the leak invariant still
        # applies.
        pass

    assert not hasattr(np_mod, "Inf"), (
        "MS4 path leaked np.Inf globally; the try/finally restore " "regressed."
    )


@pytest.mark.usefixtures("dj_conn")
def test_sorter_tempdir_cleanup_does_not_mask_sort_exception(
    monkeypatch, caplog
):
    """A cleanup failure never replaces the real sort exception.

    If the sort raises AND ``sorter_temp_dir.cleanup()`` also raises
    (e.g. a stale lock on a network FS), the unguarded ``finally`` would
    propagate the cleanup's ``PermissionError`` and hide the sort's
    actual failure. The cleanup error must be caught + logged so the
    caller sees the sort exception.
    """
    import tempfile

    import spikeinterface.core as sc
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import Sorting

    class _SortBoom(RuntimeError):
        pass

    def _boom_run_sorter(*args, **kwargs):
        raise _SortBoom("the sort itself failed")

    def _boom_cleanup(self):
        raise PermissionError("stale lock on tempdir during cleanup")

    monkeypatch.setattr(sis, "run_sorter", _boom_run_sorter)
    monkeypatch.setattr(tempfile.TemporaryDirectory, "cleanup", _boom_cleanup)

    rec = sc.generate_recording(
        num_channels=4, durations=[0.5], sampling_frequency=30000.0
    )
    with caplog.at_level("WARNING"):
        # tridesclous2: non-MS4 (no np.Inf patch), non-MATLAB sorter.
        with pytest.raises(_SortBoom):
            Sorting._run_si_sorter(
                sorter="tridesclous2",
                sorter_params={},
                recording=rec,
                sorting_id="audit-a9-cleanup",
                job_kwargs=None,
            )

    assert any(
        "cleanup failed" in record.getMessage() for record in caplog.records
    ), "cleanup failure was not logged"


def test_run_si_sorter_restores_global_job_kwargs_on_raise(
    dj_conn, monkeypatch
):
    """SI's global job_kwargs are restored after the sort raises.

    ``_run_si_sorter`` installs the per-row job_kwargs into SI's process-global
    state via ``set_global_job_kwargs`` and restores the prior global in a
    ``finally`` (reset-then-reapply, so keys absent from the prior global do
    not leak). A regression removing the restore would leak the mutated global
    (here n_jobs=2 vs the default 1) into every later populate. Force the sort
    to raise and assert the global is byte-for-byte the pre-call state.
    """
    import spikeinterface as si
    import spikeinterface.sorters as sis

    def _boom(*args, **kwargs):
        raise RuntimeError("sorter blew up")

    rec = si.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=30_000.0
    )
    monkeypatch.setattr(sis, "run_sorter", _boom)

    from spyglass.spikesorting.v2.sorting import Sorting

    before = dict(si.get_global_job_kwargs())
    import uuid

    with pytest.raises(RuntimeError, match="sorter blew up"):
        Sorting._run_si_sorter(
            "mountainsort5",
            {},
            rec,
            uuid.uuid4(),
            {"n_jobs": 2, "chunk_duration": "2s"},
        )
    after = dict(si.get_global_job_kwargs())
    assert after == before, (
        "global job_kwargs were not restored after the sort raised; the "
        f"finally-restore leaked state. before={before} after={after}"
    )


def test_run_si_sorter_restores_global_job_kwargs_on_success(
    dj_conn, monkeypatch
):
    """SI's global job_kwargs are restored after a successful sort too."""
    import spikeinterface.sorters as sis

    sentinel = object()
    before, after, result = _run_si_sorter_with_patched_run_sorter(
        monkeypatch, lambda *a, **k: sentinel
    )
    assert result is sentinel
    # n_jobs=2 differs from SI's default n_jobs=1, so the sort genuinely
    # mutates the global mid-run; a removed restore would surface here as
    # after != before (the leaked n_jobs=2).
    assert after == before, (
        "global job_kwargs were not restored after a successful sort; "
        f"before={before} after={after}"
    )


@pytest.mark.usefixtures("dj_conn")
def test_run_si_sorter_matlab_carveout(monkeypatch):
    """A MATLAB sorter gets ``singularity_image=True`` and the
    container-incompatible kwargs stripped.

    The MATLAB carve-out (``kilosort2_5`` / ``kilosort3`` / ``ironclust``)
    forces Singularity containerization and removes the
    ``_MATLAB_SORTER_STRIP_KWARGS`` (``tempdir`` / ``mp_context`` /
    ``max_threads_per_process``) that the container rejects. We capture the
    kwargs ``run_sorter`` actually receives.
    """
    import uuid

    import spikeinterface as si
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import Sorting

    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(sis, "run_sorter", _capture)
    rec = si.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=30_000.0
    )
    sorter_params = {
        "tempdir": "/should/be/stripped",
        "mp_context": "spawn",
        "max_threads_per_process": 4,
        "detect_threshold": 6.0,  # a real param that must survive
    }
    Sorting._run_si_sorter("kilosort2_5", sorter_params, rec, uuid.uuid4(), {})

    assert (
        captured.get("singularity_image") is True
    ), "MATLAB sorter must run under Singularity"
    for stripped in ("tempdir", "mp_context", "max_threads_per_process"):
        assert (
            stripped not in captured
        ), f"{stripped!r} should be stripped for a MATLAB sorter"
    assert (
        captured.get("detect_threshold") == 6.0
    ), "a non-stripped sorter param must reach run_sorter"


@pytest.mark.usefixtures("dj_conn")
def test_run_si_sorter_non_matlab_keeps_kwargs(monkeypatch):
    """Contrast: a non-MATLAB sorter keeps every param and gets no
    Singularity flag -- proving the carve-out is sorter-name-gated."""
    import uuid

    import spikeinterface as si
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import Sorting

    captured = {}
    monkeypatch.setattr(
        sis, "run_sorter", lambda **k: captured.update(k) or object()
    )
    rec = si.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=30_000.0
    )
    Sorting._run_si_sorter(
        "mountainsort5", {"tempdir": "/keep/me"}, rec, uuid.uuid4(), {}
    )
    assert "singularity_image" not in captured
    assert captured.get("tempdir") == "/keep/me"
