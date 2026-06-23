"""``Sorting._run_si_sorter`` dispatch invariants.

Covers the MS4 ``numpy.Inf`` global shim teardown, tempdir-cleanup not masking
the real sort exception, SI global ``job_kwargs`` set/restore, the tracked
container-execution backend (local vs Docker/Singularity run_sorter kwargs +
the MATLAB-sorter container policy), and keeping SI job kwargs off the
run_sorter call.
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
def test_matlab_sorters_require_explicit_container_backend(monkeypatch):
    """MATLAB-backed sorters cannot run on a local execution backend.

    ``kilosort2_5`` / ``kilosort3`` / ``ironclust`` ship only as container
    images, so a default/local execution row raises a clear
    tracked-container-backend message BEFORE ``run_sorter`` is reached -- the old
    name-based ``singularity_image=True`` auto-fallback is gone.
    """
    import uuid

    import spikeinterface as si
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import Sorting

    def _must_not_run(**kwargs):
        raise AssertionError("run_sorter must not be reached for a local MATLAB row")

    monkeypatch.setattr(sis, "run_sorter", _must_not_run)
    rec = si.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=30_000.0
    )
    for sorter in ("kilosort2_5", "kilosort3", "ironclust"):
        # execution_params omitted -> default local -> must raise.
        with pytest.raises(ValueError, match="container"):
            Sorting._run_si_sorter(sorter, {}, rec, uuid.uuid4(), {})


@pytest.mark.usefixtures("dj_conn")
def test_run_si_sorter_passes_container_kwargs(monkeypatch):
    """Container execution rows pass the right SI run_sorter container kwargs.

    A Singularity row for a MATLAB sorter passes ``singularity_image=<image>`` +
    the container-install controls, AND strips the container-incompatible
    ``MATLAB_SORTER_STRIP_KWARGS`` while keeping a real sorter param. A Docker row
    for MS4 passes ``docker_image=<image>``. A local row passes no container
    kwargs.
    """
    import uuid

    import spikeinterface as si
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import Sorting

    captured: dict = {}
    monkeypatch.setattr(
        sis, "run_sorter", lambda **k: captured.update(k) or object()
    )
    rec = si.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=30_000.0
    )

    # Singularity MATLAB row: image + install controls passed; strip applied.
    captured.clear()
    Sorting._run_si_sorter(
        "kilosort2_5",
        {
            "tempdir": "/strip/me",
            "mp_context": "spawn",
            "max_threads_per_process": 4,
            "detect_threshold": 6.0,  # a real param that must survive
        },
        rec,
        uuid.uuid4(),
        {},
        {
            "backend": "singularity",
            "container_image": "ks-image.sif",
            "installation_mode": "pypi",
            "spikeinterface_version": "0.104.3",
        },
    )
    assert captured["singularity_image"] == "ks-image.sif"
    assert "docker_image" not in captured
    assert captured["installation_mode"] == "pypi"
    assert captured["spikeinterface_version"] == "0.104.3"
    assert captured["delete_container_files"] is True
    # The scratch-collision fix: SI's output folder is a per-sort CHILD dir
    # (``.../sorter_output``), so its fixed-name in_container_* files land in the
    # unique temp dir (folder.parent), not the shared temp_dir.
    import os

    assert os.path.basename(captured["folder"]) == "sorter_output"
    for stripped in ("tempdir", "mp_context", "max_threads_per_process"):
        assert stripped not in captured, f"{stripped!r} must be stripped"
    assert captured["detect_threshold"] == 6.0

    # Docker MS4 row: docker_image passed (no MATLAB strip -- MS4 is not MATLAB).
    captured.clear()
    Sorting._run_si_sorter(
        "mountainsort4",
        {"adjacency_radius": 100.0},
        rec,
        uuid.uuid4(),
        {},
        {
            "backend": "docker",
            "container_image": "ms4-image:0.104.3",
            "installation_mode": "no-install",
        },
    )
    assert captured["docker_image"] == "ms4-image:0.104.3"
    assert "singularity_image" not in captured
    assert captured["installation_mode"] == "no-install"
    assert captured["adjacency_radius"] == 100.0

    # Local row: no container kwargs at all.
    captured.clear()
    Sorting._run_si_sorter(
        "mountainsort5", {"tempdir": "/keep/me"}, rec, uuid.uuid4(), {}
    )
    assert "singularity_image" not in captured
    assert "docker_image" not in captured
    assert "installation_mode" not in captured
    assert captured.get("tempdir") == "/keep/me"  # not a MATLAB sorter -> kept


@pytest.mark.usefixtures("dj_conn")
def test_run_si_sorter_keeps_job_kwargs_out_of_sorter_params(monkeypatch):
    """SI job kwargs install via the global; execution kwargs are run kwargs.

    ``n_jobs`` / ``chunk_duration`` must NOT reach ``run_sorter(**...)`` (they
    would trip strict per-sorter validators); they install via
    ``set_global_job_kwargs``. Container execution kwargs DO reach ``run_sorter``.
    """
    import uuid

    import spikeinterface as si
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import Sorting

    captured: dict = {}
    monkeypatch.setattr(
        sis, "run_sorter", lambda **k: captured.update(k) or object()
    )
    rec = si.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=30_000.0
    )
    Sorting._run_si_sorter(
        "mountainsort4",
        {"adjacency_radius": 100.0},
        rec,
        uuid.uuid4(),
        {"n_jobs": 2, "chunk_duration": "2s"},
        {
            "backend": "singularity",
            "container_image": "img.sif",
            "installation_mode": "no-install",
        },
    )
    # Job kwargs route through the SI global, never into run_sorter kwargs.
    assert "n_jobs" not in captured
    assert "chunk_duration" not in captured
    # Execution kwargs DO reach run_sorter.
    assert captured["singularity_image"] == "img.sif"
    assert captured["installation_mode"] == "no-install"


@pytest.mark.medium
def test_v2_recording_chain_is_container_serializable():
    """The v2 recording wrappers stay (json|pickle)-serializable for containers.

    SI's container runner requires the recording passed to ``run_sorter`` to be
    JSON- or pickle-serializable so it can re-materialize it inside the
    container. The v2 sort-time chain wraps the preprocessed recording in an
    artifact mask (``apply_artifact_mask``) and, for a whitening sorter, the
    external float64 whitening (``pinned_whiten``). Assert each wrapper -- and
    their composition -- preserves container-serializability, without Docker or
    a real sort. A regression (e.g. a wrapper holding an unpicklable closure)
    would otherwise only surface deep in a real container populate.
    """
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        apply_artifact_mask,
    )
    from spyglass.spikesorting.v2._sorting_dispatch import pinned_whiten

    def _container_serializable(recording) -> bool:
        return recording.check_serializability(
            "json"
        ) or recording.check_serializability("pickle")

    rec = si.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=30_000.0
    )
    assert _container_serializable(rec), "base recording not serializable"

    # Artifact-masked recording (keep the first half-second).
    masked = apply_artifact_mask(rec, np.array([[0.0, 0.5]]))
    assert _container_serializable(masked), "artifact-masked not serializable"

    # Whitened wrapper (the external float64 whitening path).
    whitened = pinned_whiten(rec)
    assert _container_serializable(whitened), "whitened not serializable"

    # The full sort-time composition: whiten(artifact-mask(recording)).
    composed = pinned_whiten(masked)
    assert _container_serializable(composed), "composed chain not serializable"
