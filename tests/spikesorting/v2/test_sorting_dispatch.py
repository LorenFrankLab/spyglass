"""``Sorting._run_si_sorter`` dispatch invariants.

Covers the MS4 ``numpy.Inf`` global shim teardown, tempdir-cleanup not masking
the real sort exception, SI global ``job_kwargs`` set/restore, the tracked
container-execution backend (local vs Docker/Singularity run_sorter kwargs +
the MATLAB-sorter container policy), and keeping SI job kwargs off the
run_sorter call.
"""

from __future__ import annotations

import pytest


def _tiny_numpy_sorting():
    """A minimal REAL in-memory sorting for ``run_sorter`` stubs.

    ``run_si_sorter`` now materializes the sorter output via
    ``NumpySorting.from_sorting`` (sever the temp-dir file backing before
    cleanup), so a stub must return a real sorting -- a placeholder ``object()``
    no longer satisfies the materialize contract.
    """
    import numpy as np
    import spikeinterface as si

    return si.NumpySorting.from_samples_and_labels(
        samples_list=[np.array([10, 20, 30], dtype=np.int64)],
        labels_list=[np.array([0, 0, 1], dtype=np.int64)],
        sampling_frequency=30_000.0,
    )


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

    import spikeinterface as si

    before, after, result = _run_si_sorter_with_patched_run_sorter(
        monkeypatch, lambda *a, **k: _tiny_numpy_sorting()
    )
    # run_si_sorter materializes the sorter output, so the result is an
    # in-memory NumpySorting rather than the stub object identity.
    assert isinstance(result, si.NumpySorting)
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
        raise AssertionError(
            "run_sorter must not be reached for a local MATLAB row"
        )

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
        sis,
        "run_sorter",
        lambda **k: captured.update(k) or _tiny_numpy_sorting(),
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
        sis,
        "run_sorter",
        lambda **k: captured.update(k) or _tiny_numpy_sorting(),
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
def test_v2_recording_chain_survives_run_sorter_serialization(tmp_path):
    """The v2 recording wrappers survive ``run_sorter``'s serialize + reload.

    SI's sorter/container runner dumps the recording passed to ``run_sorter``
    to ``spikeinterface_recording.json`` when ``check_serializability("json")``
    is truthy (else pickle), then reloads it inside the sorter/container. The
    v2 sort-time chain wraps the preprocessed recording in an artifact mask
    (``apply_artifact_mask``) and, for a whitening sorter, external float64
    whitening (``pinned_whiten``).

    This asserts a real dump -> reload for each wrapper and their composition,
    NOT merely that ``check_serializability`` returns truthy: a wrapper can
    report json-serializable yet fail to reload (SI's
    ``SilencedPeriodsRecording`` stores its ``periods`` as a structured numpy
    array that cannot survive a JSON round-trip -- the flag-only check missed
    exactly that). Without Docker or a real sort.
    """
    import numpy as np
    import spikeinterface as si
    from spikeinterface.core import load

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        apply_artifact_mask,
    )
    from spyglass.spikesorting.v2._sorting_dispatch import pinned_whiten

    def _assert_roundtrips(recording, name):
        # Mirror SI basesorter.setup_recording: JSON when the recording claims
        # json-serializability, else pickle -- then reload. ``load`` raises if
        # the dumped form cannot be reconstructed.
        folder = tmp_path / name
        folder.mkdir()
        if recording.check_serializability("json"):
            rec_file = folder / "spikeinterface_recording.json"
            recording.dump_to_json(rec_file)
        elif recording.check_serializability("pickle"):
            rec_file = folder / "spikeinterface_recording.pickle"
            recording.dump_to_pickle(rec_file)
        else:
            raise AssertionError(
                f"{name}: neither json- nor pickle-serializable"
            )
        load(rec_file, base_folder=folder)

    rec = si.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=30_000.0
    )
    _assert_roundtrips(rec, "base")

    # Artifact-masked recording (keep the first half-second).
    masked = apply_artifact_mask(rec, np.array([[0.0, 0.5]]))
    _assert_roundtrips(masked, "masked")

    # Whitened wrapper (the external float64 whitening path).
    whitened = pinned_whiten(rec)
    _assert_roundtrips(whitened, "whitened")

    # The full sort-time composition: whiten(artifact-mask(recording)).
    composed = pinned_whiten(masked)
    _assert_roundtrips(composed, "composed")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("dj_conn")
def test_run_si_sorter_output_survives_tempdir_cleanup():
    """A real sorter run's output is readable AFTER run_si_sorter returns.

    ``sis.run_sorter`` returns a sorting that READS from the sorter temp dir,
    which ``run_si_sorter`` cleans up in its finally; downstream
    ``_build_analyzer`` / ``_stage_sorting_artifact`` would then read freed
    files. The fix materializes the spike trains into an in-memory
    ``NumpySorting`` before cleanup. Use a real MountainSort5 run (a stub can't
    reproduce the file backing).
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2.sorting import Sorting

    rec, _gt = si.generate_ground_truth_recording(
        durations=[10.0],
        num_channels=4,
        num_units=3,
        seed=0,
        sampling_frequency=30000.0,
    )
    result = Sorting._run_si_sorter(
        "mountainsort5", {}, rec, "r4-tempdir-survival", {}
    )
    # Severed from the temp dir: an in-memory NumpySorting, not file-backed.
    assert isinstance(result, si.NumpySorting)
    # Spike trains still read back after the temp dir is gone.
    unit_ids = list(result.get_unit_ids())
    assert unit_ids, "MS5 should detect the planted units"
    for uid in unit_ids:
        assert result.get_unit_spike_train(uid) is not None


def test_whiten_interception_allowlisted():
    """The external-whitening interception fires only for the curated
    external-whitening sorters (MountainSort 4/5).

    Those sorters deliberately carry ``whiten=True`` and the dispatcher routes
    them through the runtime's external float64 whitening. A generic /
    uncurated sorter's ``whiten`` must be passed through to the sorter
    unchanged, not silently rewritten.
    """
    from spyglass.spikesorting.v2._sorting_dispatch import (
        _should_external_whiten,
    )

    # Curated external-whitening sorters with a truthy whiten ARE intercepted.
    assert _should_external_whiten("mountainsort4", {"whiten": True})
    assert _should_external_whiten("mountainsort5", {"whiten": True})
    assert _should_external_whiten("MountainSort5", {"whiten": True})  # case

    # A generic / uncurated sorter's whiten is passed through unchanged.
    assert not _should_external_whiten("kilosort4", {"whiten": True})
    assert not _should_external_whiten("spykingcircus2", {"whiten": True})

    # MS without a truthy whiten is not intercepted.
    assert not _should_external_whiten("mountainsort5", {"whiten": False})
    assert not _should_external_whiten("mountainsort5", {})


@pytest.mark.medium
def test_ms5_non_default_params_reach_run_sorter_unchanged(monkeypatch):
    """Configured MS5 values reach ``run_sorter`` verbatim; whiten is routed.

    The parameter-to-SI forwarding contract: the long-recording knobs are
    passed through untouched, ``whiten=True`` is intercepted (external float64
    whitening) and handed to SI as ``whiten=False``, and the dispatcher's
    kwargs equal ``resolve_sort_config(...).si_sorter_params`` -- the same
    resolution preflight reports.
    """
    import uuid

    import spikeinterface as si
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2._params.sorter import MountainSort5Schema
    from spyglass.spikesorting.v2._sorting_dispatch import resolve_sort_config
    from spyglass.spikesorting.v2.sorting import Sorting

    captured: dict = {}
    monkeypatch.setattr(
        sis,
        "run_sorter",
        lambda **k: captured.update(k) or _tiny_numpy_sorting(),
    )
    rec = si.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=30_000.0
    )
    params = MountainSort5Schema(
        scheme="3",
        scheme3_block_duration_sec=300,
        scheme2_training_duration_sec=120,
        npca_per_channel=5,
    ).model_dump()
    params.pop("schema_version")
    job_kwargs = {"n_jobs": 2, "random_seed": 3}
    Sorting._run_si_sorter(
        "mountainsort5", params, rec, uuid.uuid4(), job_kwargs
    )
    assert captured["scheme"] == "3"
    assert captured["scheme3_block_duration_sec"] == 300.0
    assert captured["scheme2_training_duration_sec"] == 120.0
    assert captured["npca_per_channel"] == 5
    assert captured["whiten"] is False  # whitened externally, exactly once
    assert "n_jobs" not in captured and "random_seed" not in captured
    config = resolve_sort_config("mountainsort5", params, job_kwargs=job_kwargs)
    forwarded = {
        k: v for k, v in captured.items() if k in config.si_sorter_params
    }
    assert forwarded == config.si_sorter_params
    assert config.external_whiten is True
    assert config.random_seed == 3
    assert config.job_kwargs == {"n_jobs": 2}


def test_remove_excess_spikes_drops_empty_units(caplog):
    """A unit whose only spikes fall outside the recording window is left
    with zero spikes by ``sic.remove_excess_spikes``'s window trim; it must
    then be dropped from the returned sorting entirely (not kept as an empty
    unit). Other units and their spike trains must be unchanged, and a
    sorting with no empty units must keep every unit id.

    Uses sparse, non-contiguous unit ids ([3, 7, 12]) so an index/id mix-up
    in the drop logic would show up as the wrong unit being removed.
    """
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2._sorting_dispatch import (
        remove_excess_spikes,
    )

    fs = 30_000.0
    rec = si.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=fs
    )
    n = rec.get_num_samples()
    assert n == 30_000

    # Unit 3: two in-window spikes (unchanged). Unit 7: its only spike is
    # past the recording window -> left empty by the trim -> must be
    # dropped. Unit 12: one in-window spike (unchanged).
    samples = np.array([100, 200, n + 500, 300], dtype=np.int64)
    labels = np.array([3, 3, 7, 12], dtype=np.int64)
    sorting = si.NumpySorting.from_samples_and_labels(
        samples_list=[samples], labels_list=[labels], sampling_frequency=fs
    )

    with caplog.at_level("INFO"):
        out = remove_excess_spikes(sorting, rec)

    assert list(out.unit_ids) == [3, 12], (
        f"expected unit 7 (left empty by the window trim) to be dropped; "
        f"got unit_ids={list(out.unit_ids)}"
    )
    np.testing.assert_array_equal(out.get_unit_spike_train(3), [100, 200])
    np.testing.assert_array_equal(out.get_unit_spike_train(12), [300])
    assert any(
        "7" in r.message for r in caplog.records
    ), "dropped unit id 7 must be logged"

    # No empty units -> every unit id is kept.
    samples_all_in_window = np.array([100, 200, 300], dtype=np.int64)
    labels_all_in_window = np.array([3, 7, 12], dtype=np.int64)
    sorting_no_empty = si.NumpySorting.from_samples_and_labels(
        samples_list=[samples_all_in_window],
        labels_list=[labels_all_in_window],
        sampling_frequency=fs,
    )
    out_no_empty = remove_excess_spikes(sorting_no_empty, rec)
    assert list(out_no_empty.unit_ids) == [3, 7, 12]


# ---------------------------------------------------------------------------
# Whitening and clusterless noise levels from valid samples only.
#
# The sort stage silences artifact frames with zeros. The whitening covariance
# and the clusterless MAD must draw only from samples inside the statistics
# spans, while an unmasked continuous recording keeps SpikeInterface's own
# path bit-for-bit. Targets come from the clean twin (see
# ``_masked_statistics_helpers``).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clean_ground_truth():
    """(traces, probe, sorting) of the 60 s, 16-channel clean recording."""
    from tests.spikesorting.v2._masked_statistics_helpers import (
        clean_ground_truth as _clean_ground_truth,
    )

    return _clean_ground_truth()


def _applied_whitening(whitened):
    """(W, M) the WhitenRecording actually applies to its traces."""
    segment = whitened._recording_segments[0]
    return segment.W, segment.M


@pytest.mark.medium
def test_pinned_whiten_unmasked_is_bit_identical_to_previous(
    clean_ground_truth,
):
    """No spans, or one span covering the recording, is SI's own whitening.

    Both must equal ``sip.whiten(recording, dtype=float64, seed=s)`` exactly:
    the same ``W`` (and ``M=None``) and the same whitened traces.
    """
    import numpy as np
    import spikeinterface.preprocessing as sip

    from spyglass.spikesorting.v2._sorting_dispatch import pinned_whiten
    from tests.spikesorting.v2._masked_statistics_helpers import (
        numpy_recording,
    )

    traces, probe, _ = clean_ground_truth
    recording = numpy_recording(traces, probe)
    n_samples = recording.get_num_samples()
    seed = 7
    reference = sip.whiten(recording, dtype=np.float64, seed=seed)
    w_ref, m_ref = _applied_whitening(reference)
    assert m_ref is None
    ref_slice = reference.get_traces(start_frame=1_000, end_frame=31_000)

    for spans in (None, [(0, n_samples)]):
        whitened = pinned_whiten(recording, random_seed=seed, spans=spans)
        w, m = _applied_whitening(whitened)
        assert m is None
        assert w.dtype == w_ref.dtype
        assert np.array_equal(w, w_ref), f"W differs for spans={spans}"
        assert np.array_equal(
            whitened.get_traces(start_frame=1_000, end_frame=31_000),
            ref_slice,
        ), f"whitened traces differ for spans={spans}"


@pytest.mark.medium
@pytest.mark.parametrize("scale", [1.0, 1e-3])
def test_span_whitening_matrix_matches_spikeinterface_on_full_span(
    clean_ground_truth, scale
):
    """The span-path W on one full span is exactly SI's ``mode="global"`` W.

    ``scale=1e-3`` puts the median squared sample inside (0, 1), exercising
    SI's data-dependent ``eps`` branch rather than its ``1e-16`` floor.
    """
    import numpy as np
    from spikeinterface.preprocessing.whiten import compute_whitening_matrix

    from spyglass.spikesorting.v2._sorting_dispatch import (
        _span_whitening_matrix,
    )
    from tests.spikesorting.v2._masked_statistics_helpers import (
        numpy_recording,
    )

    traces, probe, _ = clean_ground_truth
    recording = numpy_recording((traces * scale).astype(np.float32), probe)
    n_samples = recording.get_num_samples()
    seed = 7

    w_si, m_si = compute_whitening_matrix(
        recording, "global", {"seed": seed}, apply_mean=False
    )
    assert m_si is None
    w_span = _span_whitening_matrix(
        recording, [(0, n_samples)], random_seed=seed
    )
    assert w_span.dtype == w_si.dtype
    assert np.array_equal(w_span, w_si)


@pytest.mark.medium
def test_whitened_valid_samples_have_unit_variance_under_masking(
    clean_ground_truth,
):
    """30% masked: whitened valid samples have unit standard deviation.

    Measured over ALL valid samples (not the covariance sample), pooled
    across channels. Zeros counted as data shrink the covariance and inflate
    this std (about 1.1 on this recording). Pooled, not per channel: with
    SI's 20 x 500 ms sample budget, per-channel std on this spiking recording
    varies by a few percent with the seed even for SpikeInterface's own
    whitening of the clean twin (0.96-1.02), so a per-channel 2% band is a
    property of the budget, not of masking.
    """
    import numpy as np

    from spyglass.spikesorting.v2._sorting_dispatch import pinned_whiten
    from tests.spikesorting.v2._masked_statistics_helpers import masked_twin

    traces, probe, _ = clean_ground_truth
    masked, spans, _ = masked_twin(traces, probe, 0.30)
    whitened = pinned_whiten(masked, random_seed=0, spans=spans)
    valid = np.concatenate(
        [whitened.get_traces(start_frame=a, end_frame=b) for a, b in spans]
    )
    std = valid.std()
    assert 0.98 <= std <= 1.02, std


@pytest.mark.medium
def test_run_si_sorter_whitens_from_statistics_spans(
    clean_ground_truth, monkeypatch
):
    """The sorter receives a recording whitened from the span covariance.

    ``run_sorter`` is replaced by a stub that records the recording it was
    handed; the whitening itself is the real ``pinned_whiten``.
    """
    import uuid

    import numpy as np
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2._sorting_dispatch import (
        _span_whitening_matrix,
        run_si_sorter,
    )
    from tests.spikesorting.v2._masked_statistics_helpers import masked_twin

    traces, probe, _ = clean_ground_truth
    masked, spans, _ = masked_twin(traces, probe, 0.30)
    received = {}

    def _record_run_sorter(**kwargs):
        received["recording"] = kwargs["recording"]
        return _tiny_numpy_sorting()

    monkeypatch.setattr(sis, "run_sorter", _record_run_sorter)
    run_si_sorter(
        "mountainsort5",
        {"whiten": True},
        masked,
        uuid.uuid4(),
        {"random_seed": 5},
        statistics_spans=spans,
    )
    w, m = _applied_whitening(received["recording"])
    assert m is None
    assert np.array_equal(
        w, _span_whitening_matrix(masked, spans, random_seed=5)
    )
