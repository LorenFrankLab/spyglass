"""Audit-derived correctness & v1-parity regression slice.

Tests here pin behavioral fixes surfaced by the v1↔v2 parity audit that
the original 6-agent review missed. Each test either reproduces a
confirmed bug (raise / no-leak / no-silent-mask) or pins an intentional
v2 design choice against future drift. Where a test locks a v1↔v2
divergence the docstring cites the v1 source so a reviewer can confirm
the choice was deliberate.

Pure-Pydantic tests import only ``_params`` (no DB). Tests that touch a
``dj.schema``-activated runtime module request ``dj_conn`` and import the
table inside the test body (mirroring ``test_v1_parity``).
"""

from __future__ import annotations

import numpy as np
import pytest

from spyglass.spikesorting.v2._params.sorter import MountainSort4Schema

# ---------- A3: MS4 adjacency_radius -1 sentinel ---------------------------


def test_ms4_schema_accepts_adjacency_radius_minus_one():
    """A3: ``adjacency_radius=-1`` is SI's "use all channels" sentinel.

    SpikeInterface's MountainSort4 wrapper documents ``adjacency_radius=-1``
    as "use all channels in the adjacency graph". The earlier
    ``Field(ge=0.0)`` floor rejected the sentinel, so a v1 user porting
    ``{"adjacency_radius": -1}`` was blocked at insert time. ``ge=-1.0``
    plus a validator that rejects the open interval ``(-1, 0)`` accepts
    the sentinel and any non-negative radius while still catching
    nonsensical negatives.
    """
    assert MountainSort4Schema(adjacency_radius=-1).adjacency_radius == -1.0
    assert MountainSort4Schema(adjacency_radius=0).adjacency_radius == 0.0
    assert MountainSort4Schema(adjacency_radius=75.0).adjacency_radius == 75.0

    # The open interval (-1, 0) is meaningless and must be rejected.
    for bad in (-0.5, -0.999, -0.0001):
        with pytest.raises(ValueError, match="adjacency_radius"):
            MountainSort4Schema(adjacency_radius=bad)

    # Below the sentinel is also invalid.
    with pytest.raises(ValueError):
        MountainSort4Schema(adjacency_radius=-2.0)


# ---------- A2: superseded Franklab MS4 v1-name aliases (retracted) --------
#
# v2 briefly shipped back-compat ``SorterParameters`` rows under v1's bare
# ``franklab_tetrode_hippocampus_30KHz`` / ``franklab_probe_ctx_30KHz`` names
# (and the region-encoded ``*_30kHz_ms4`` rows) so ported v1 lookups kept
# resolving. The June 2026 catalog correction supersedes them: the MS4 rows
# are now rate-keyed (``franklab_30khz_ms4_2026_06`` /
# ``franklab_20khz_ms4_2026_06``) and the aliases are removed pre-release. The
# A2 alias-parity test is intentionally gone; the Migration guide records the
# rename instead of an alias.


# ---------- A4: install-gating happens on the insert_default() path --------


@pytest.mark.usefixtures("dj_conn")
def test_sorter_parameters_skips_uninstalled_sorters(monkeypatch):
    """A4: ``insert_default()`` skips uninstalled SI sorters at INSERT time.

    The companion ``_gated_default_rows`` test pins the gating DECISION;
    this one pins the INSERT PATH -- that ``insert_default`` actually
    routes the decision into the table, not just computes it. A
    regression that made ``insert_default`` insert the full
    ``_DEFAULT_CONTENTS`` directly (bypassing the gate) would pass the
    decision test but fail here.

    With ``installed_sorters()`` forced empty, every SI-registered default
    row is gated out while the Spyglass-internal ``clusterless_thresholder``
    row (never gated) still inserts. The skip is asserted on default
    sorters genuinely absent from the box, whose rows are therefore
    guaranteed clean (no earlier test inserted them) -- if the gate
    regressed, those rows would appear.
    """
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import SorterParameters

    real_installed = set(sis.installed_sorters())
    si_default_sorters = {
        row[0]
        for row in SorterParameters._DEFAULT_CONTENTS
        if row[0] not in SorterParameters._NON_SI_SORTERS
    }
    # Canaries: SI default sorters genuinely absent from this box. We
    # delete their rows first (a clean slate the persistent test DB may
    # not give us -- deleting a default Lookup row for an uninstalled,
    # therefore unreferenced, sorter is safe) so that a row reappearing
    # after insert_default unambiguously means the gate was bypassed.
    clean_canaries = si_default_sorters - real_installed
    if not clean_canaries:
        pytest.skip(
            "every SI default sorter is installed; no clean canary row to "
            "verify the gated insert path against"
        )
    for sorter in clean_canaries:
        (SorterParameters & {"sorter": sorter}).delete(safemode=False)

    monkeypatch.setattr(sis, "installed_sorters", lambda: [])
    SorterParameters.insert_default()

    # Non-SI special case -> never gated -> always inserted.
    assert SorterParameters & {
        "sorter": "clusterless_thresholder",
        "sorter_params_name": "default",
    }, "clusterless default row must insert even with no SI sorter installed"

    # Every gated-out SI sorter stays absent: insert_default routed the
    # gating DECISION into the INSERT, it did not insert _DEFAULT_CONTENTS
    # wholesale.
    for sorter in clean_canaries:
        assert not (SorterParameters & {"sorter": sorter}), (
            f"insert_default inserted uninstalled SI sorter {sorter!r} "
            "despite installed_sorters()==[]; the install gate was "
            "bypassed on the insert path."
        )


# ---------- A5: params_schema_version is backfilled; drift still raises -----


@pytest.mark.usefixtures("dj_conn")
def test_sorter_parameters_backfills_missing_schema_version():
    """A5: a custom row may omit ``params_schema_version``; it is backfilled.

    The column default is the sentinel 0 ("unspecified"). The validated
    ``params`` blob already carries the authoritative ``schema_version``
    (4 for clusterless), so ``insert`` backfills the outer column from it
    when the caller leaves it at 0 -- the user does not copy the number by
    hand. An explicitly-supplied version that DISAGREES with the blob still
    trips the drift guard.
    """
    from spyglass.spikesorting.v2.sorting import SorterParameters

    base = {
        "sorter": "clusterless_thresholder",
        "params": {"detect_threshold": 100.0, "threshold_unit": "uv"},
        "job_kwargs": None,
    }

    def _stored_version(name):
        return (
            SorterParameters
            & {
                "sorter": "clusterless_thresholder",
                "sorter_params_name": name,
            }
        ).fetch1("params_schema_version")

    # The three rows are intentionally the SAME content under different names
    # (this test exercises schema-version backfill, not the duplicate-content
    # guard), so each opts out of the guard explicitly.
    # Omitted entirely -> backfilled from the blob's schema_version (4).
    omitted = dict(base, sorter_params_name="audit_a5_missing")
    SorterParameters().insert1(
        omitted, skip_duplicates=True, allow_duplicate_params=True
    )
    assert _stored_version("audit_a5_missing") == 4

    # Explicit sentinel 0 is backfilled the same way.
    sentinel = dict(
        base, sorter_params_name="audit_a5_zero", params_schema_version=0
    )
    SorterParameters().insert1(
        sentinel, skip_duplicates=True, allow_duplicate_params=True
    )
    assert _stored_version("audit_a5_zero") == 4

    # The correct explicit version inserts cleanly and is preserved.
    ok = dict(base, sorter_params_name="audit_a5_ok", params_schema_version=4)
    SorterParameters().insert1(
        ok, skip_duplicates=True, allow_duplicate_params=True
    )
    assert _stored_version("audit_a5_ok") == 4

    # A wrong (non-sentinel) version that disagrees with the blob still
    # trips the drift check -- backfill only fills the sentinel 0.
    drift = dict(
        base, sorter_params_name="audit_a5_drift", params_schema_version=2
    )
    with pytest.raises(ValueError, match="schema_version"):
        SorterParameters().insert1(drift, skip_duplicates=True)


@pytest.mark.usefixtures("dj_conn")
def test_sorter_parameters_rejects_unknown_sorter_name():
    """A typo'd sorter name is rejected at insert, not at populate time.

    ``_get_sorter_schema`` falls back to the permissive generic schema for
    any unknown sorter, so without the guard a typo like ``"mountainSort4"``
    (capital S) would validate cleanly here and fail only later inside
    ``Sorting.populate`` with an opaque SI "sorter not registered" error.
    The insert guard rejects it up front with a message naming the bad
    sorter, while real registered sorters still insert -- including the
    v1 "try any installed SI sorter" escape hatch (a sorter that is in
    ``sis.available_sorters()`` but NOT in the curated ``_SORTER_SCHEMAS``
    set, dispatched through ``GenericSorterParamsSchema``).
    """
    from spyglass.spikesorting.v2.sorting import SorterParameters

    typo = {
        "sorter": "mountainSort4",  # capital S -> not a real SI sorter
        "sorter_params_name": "audit_a5_typo",
        "params": {},
        "job_kwargs": None,
    }
    with pytest.raises(ValueError, match="mountainSort4"):
        SorterParameters().insert1(typo, skip_duplicates=True)

    # A curated v2 sorter is accepted (covered by the _SORTER_SCHEMAS term
    # of the valid_sorters union).
    ok = {
        "sorter": "tridesclous2",
        "sorter_params_name": "audit_a5_real",
        "params": {},
        "job_kwargs": None,
    }
    # The bare-default content matches the shipped ``tridesclous2`` "default"
    # row; this test checks sorter-name acceptance, not the duplicate guard.
    SorterParameters().insert1(
        ok, skip_duplicates=True, allow_duplicate_params=True
    )
    assert SorterParameters & {
        "sorter": "tridesclous2",
        "sorter_params_name": "audit_a5_real",
    }

    # The escape hatch: ``simple`` is a real SI sorter that is NOT in
    # ``_SORTER_SCHEMAS`` and NOT in ``_NON_SI_SORTERS``, so it is accepted
    # ONLY via the ``sis.available_sorters()`` term of the valid_sorters
    # union and validated by ``GenericSorterParamsSchema``. A regression
    # that dropped ``available_sorters()`` from the union would still pass
    # the curated/clusterless cases but break this insert. Its
    # ``params_schema_version`` is backfilled from the generic blob (1),
    # exercising backfill from a non-clusterless sorter.
    escape_hatch = {
        "sorter": "simple",
        "sorter_params_name": "audit_a5_escape_hatch",
        "params": {},
        "job_kwargs": None,
    }
    SorterParameters().insert1(escape_hatch, skip_duplicates=True)
    assert (
        SorterParameters
        & {"sorter": "simple", "sorter_params_name": "audit_a5_escape_hatch"}
    ).fetch1("params_schema_version") == 1


# ---------- A8: MS4 path must not leak the numpy.Inf global ----------------


@pytest.mark.slow
@pytest.mark.usefixtures("dj_conn")
def test_run_si_sorter_does_not_leak_numpy_inf(monkeypatch):
    """A8: the MS4 ``np.Inf`` shim is scoped and torn down.

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


# ---------- A9: tempdir cleanup must not mask the sort exception -----------


@pytest.mark.usefixtures("dj_conn")
def test_sorter_tempdir_cleanup_does_not_mask_sort_exception(
    monkeypatch, caplog
):
    """A9: a cleanup failure never replaces the real sort exception.

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


# ---------- A13: initialize_v2_defaults ships motion presets ---------------


@pytest.mark.usefixtures("dj_conn")
def test_initialize_v2_defaults_installs_motion_correction_presets():
    """A13: ``initialize_v2_defaults()`` installs the motion presets too.

    The helper previously called ``insert_default`` on only three Lookups
    and skipped ``MotionCorrectionParameters``. The motion presets then
    shipped missing, so when the concat consumer lands every run would
    hit a missing-row FK violation unless the user remembered the call
    themselves. ``initialize_v2_defaults()`` must seed all four.
    """
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.session_group import (
        MotionCorrectionParameters,
    )

    initialize_v2_defaults()
    for name in ("none", "auto_default", "rigid_fast_default"):
        assert (
            len(
                MotionCorrectionParameters
                & {"motion_correction_params_name": name}
            )
            == 1
        ), f"motion preset {name!r} missing after initialize_v2_defaults()"


# ---------- A14: MotionCorrectionParameters schema-version drift -----------


@pytest.mark.usefixtures("dj_conn")
def test_motion_correction_parameters_rejects_schema_version_drift():
    """A14: outer/inner schema-version drift is caught at insert time.

    ``MotionCorrectionParameters.insert1`` validated the ``params`` blob
    but never compared the outer ``params_schema_version`` column to the
    blob's inner ``schema_version`` (unlike ``SorterParameters.insert1``).
    A row whose outer says 1 and inner says 2 would land with the two
    disagreeing. ``_assert_schema_version_matches`` now closes the gap.
    """
    from spyglass.spikesorting.v2._params.motion_correction import (
        MotionCorrectionParamsSchema,
    )
    from spyglass.spikesorting.v2.session_group import (
        MotionCorrectionParameters,
    )

    drifted_params = MotionCorrectionParamsSchema(schema_version=2).model_dump()
    drift_row = {
        "motion_correction_params_name": "audit_a14_drift",
        "params": drifted_params,
        "params_schema_version": 1,
        "job_kwargs": None,
    }
    with pytest.raises(ValueError, match="schema_version"):
        MotionCorrectionParameters().insert1(drift_row, skip_duplicates=True)

    # A consistent row still inserts cleanly. The bare-default blob matches
    # the shipped ``"none"`` row, so opt out of the duplicate-content guard
    # (this test checks the schema-version drift guard, not duplicate content).
    ok_params = MotionCorrectionParamsSchema().model_dump()
    ok_row = {
        "motion_correction_params_name": "audit_a14_ok",
        "params": ok_params,
        "params_schema_version": 1,
        "job_kwargs": None,
    }
    MotionCorrectionParameters().insert1(
        ok_row, skip_duplicates=True, allow_duplicate_params=True
    )
    assert MotionCorrectionParameters & {
        "motion_correction_params_name": "audit_a14_ok"
    }


# ---------- A1: empty artifact valid_times must not silently zero ----------


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_empty_valid_times_raises():
    """A1: empty valid_times raises instead of silently zeroing all.

    When the artifact pass keeps zero seconds, the complement walker
    would mask the WHOLE recording to zeros and the sort would run over
    all-zeros, emitting a misleading "zero units" result. The fix raises
    ``EmptyArtifactValidTimesError`` naming the ``artifact_detection_id`` and
    ``recording_id`` so the user re-runs detection or overrides the
    selection -- a loud failure, not a silent blanked recording.
    """
    import numpy as np_mod
    import spikeinterface.core as sc

    from spyglass.spikesorting.v2.exceptions import (
        EmptyArtifactValidTimesError,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    rec = sc.generate_recording(
        num_channels=4, durations=[0.5], sampling_frequency=30000.0
    )
    with pytest.raises(EmptyArtifactValidTimesError) as excinfo:
        Sorting._apply_artifact_mask(
            rec,
            np_mod.zeros((0, 2)),
            artifact_detection_id="art-123",
            recording_id="rec-456",
        )
    msg = str(excinfo.value)
    assert "art-123" in msg and "rec-456" in msg

    # The recording is NOT zeroed -- the raise happens before any masking.
    assert not np_mod.array_equal(
        rec.get_traces(), np_mod.zeros_like(rec.get_traces())
    )


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_unsorted_valid_times_raises():
    """A1: unsorted / overlapping valid_times is rejected (strict input).

    The complement walker assumes monotonic, disjoint intervals; an
    unsorted or overlapping list would silently under-mask (leave
    artifact frames in the recording). This phase chooses strict input +
    a clear raise rather than silently sorting/merging.
    """
    import numpy as np_mod
    import spikeinterface.core as sc

    from spyglass.spikesorting.v2.sorting import Sorting

    rec = sc.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=30000.0
    )

    # Unsorted by start time.
    with pytest.raises(ValueError, match="sorted by start"):
        Sorting._apply_artifact_mask(
            rec, np_mod.array([[0.6, 0.8], [0.1, 0.3]])
        )
    # Sorted by start but overlapping.
    with pytest.raises(ValueError, match="non-overlapping"):
        Sorting._apply_artifact_mask(
            rec, np_mod.array([[0.1, 0.5], [0.3, 0.8]])
        )
    # A single well-formed interval still masks normally (no raise).
    masked = Sorting._apply_artifact_mask(rec, np_mod.array([[0.1, 0.9]]))
    assert masked.get_num_samples() == rec.get_num_samples()


# ---------- A11: Sorting.populate skips concat-source rows -----------------


@pytest.mark.usefixtures("dj_conn")
def test_sorting_make_skips_concat_rows_silently():
    """A11: concat-source selections are excluded from ``populate()``.

    ``make_fetch`` raises ``NotImplementedError`` for the concat path,
    but without a ``key_source`` filter ``populate()`` would still pick up
    concat rows and print that error per row. The antijoin key_source
    drops any selection that has a ``ConcatenatedRecordingSource`` part.

    The concat materializer (``ConcatenatedRecording.make``) is
    NotImplementedError today, so a real concat chain cannot be built;
    we create the ``ConcatenatedRecordingSource`` precondition via a raw
    FK-checks-off bypass (the only way to land the row pre-Phase-3) and
    pin the no-op skip.

    TODO(parent-Phase-3): when ``ConcatenatedRecording.make`` lands,
    remove the key_source antijoin and flip this test to assert the
    concat row IS populated.
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    # clusterless_thresholder/default is always inserted (non-SI, never
    # install-gated) so the master's SorterParameters FK resolves.
    SorterParameters.insert_default()

    sid = uuid.uuid4()
    SortingSelection.insert1(
        {
            "sorting_id": sid,
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        },
        allow_direct_insert=True,
    )
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        SortingSelection.ConcatenatedRecordingSource.insert1(
            {"sorting_id": sid, "concat_recording_id": uuid.uuid4()},
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        # Excluded from key_source...
        assert (
            len(Sorting.key_source & {"sorting_id": sid}) == 0
        ), "concat-source selection leaked into Sorting.key_source"
        # ...and populate restricted to it is a silent no-op (make_fetch's
        # NotImplementedError is never reached).
        Sorting.populate({"sorting_id": sid}, reserve_jobs=False)
        assert len(Sorting & {"sorting_id": sid}) == 0
    finally:
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


# ---------- A10: zero-unit get_analyzer guard fires before path lookup -----


@pytest.mark.usefixtures("dj_conn")
def test_get_analyzer_zero_unit_raises_before_path_lookup():
    """A10: a zero-unit sort raises before any analyzer-folder lookup.

    With Phase B the analyzer cache path is computed from ``sorting_id``
    (not a stored column). ``get_analyzer`` checks ``n_units`` FIRST and
    raises ``ZeroUnitAnalyzerError`` before computing / ``.exists()``-ing
    the path -- so a missing folder never surfaces as a confusing
    ``FileNotFoundError``.
    """
    import datetime as dt
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.exceptions import ZeroUnitAnalyzerError
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    SorterParameters.insert_default()
    sid = uuid.uuid4()
    SortingSelection.insert1(
        {
            "sorting_id": sid,
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        },
        allow_direct_insert=True,
    )

    # A zero-unit Sorting row; its computed analyzer path does NOT exist on
    # disk. The AnalysisNwbfile FK is bypassed -- the zero-unit guard never
    # reads the file or the folder.
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        Sorting.insert1(
            {
                "sorting_id": sid,
                "analysis_file_name": "a10_fake.nwb",
                "object_id": "a10-fake-object-id",
                "n_units": 0,
                "time_of_sort": dt.datetime(2020, 1, 1, 0, 0, 0),
            },
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        with pytest.raises(ZeroUnitAnalyzerError):
            Sorting().get_analyzer({"sorting_id": sid})
    finally:
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


# ---------- A17: chunked artifact detection via ChunkRecordingExecutor -----


def _in_memory_artifact_frames_reference(recording, validated):
    """Frozen copy of the pre-port full-in-memory artifact-frame scan.

    This reproduces, verbatim, the per-frame detection math that lived in
    ``ArtifactDetection._detect_artifacts`` before A17 replaced the
    full-recording ``get_traces`` load with a chunked
    ``ChunkRecordingExecutor`` pass. It exists ONLY in the test suite as the
    equivalence oracle: the chunked port must produce frame-identical output
    to this reference on the same recording. It is intentionally NOT importable
    from production -- the in-memory path is deleted there, not kept as a
    fallback.

    Returns the ascending ndarray of flagged frame indices (the
    ``frames_above`` array the interval-building code consumes).
    """
    traces = recording.get_traces(return_in_uV=False)
    gains = recording.get_channel_gains()
    traces_uv = traces.astype(np.float32) * gains[None, :]
    absolute = np.abs(traces_uv)

    n_channels = traces.shape[1]
    n_required = int(np.ceil(validated.proportion_above_threshold * n_channels))

    if validated.amplitude_threshold_uv is not None:
        above_amp = absolute > validated.amplitude_threshold_uv
    else:
        above_amp = np.zeros_like(absolute, dtype=bool)
    if validated.zscore_threshold is not None:
        ch_mean = traces_uv.mean(axis=1, keepdims=True)
        ch_std = traces_uv.std(axis=1, keepdims=True) + 1e-12
        zscores = np.abs((traces_uv - ch_mean) / ch_std)
        above_z = zscores > validated.zscore_threshold
    else:
        above_z = np.zeros_like(absolute, dtype=bool)

    if (
        validated.amplitude_threshold_uv is not None
        and validated.zscore_threshold is not None
    ):
        channel_hit = above_amp | above_z
    elif validated.amplitude_threshold_uv is not None:
        channel_hit = above_amp
    else:
        channel_hit = above_z

    return (channel_hit.sum(axis=1) >= n_required).nonzero()[0]


def _synthetic_artifact_recording():
    """8-channel, 90 000-sample (3 s @ 30 kHz) recording with two planted
    artifact runs -- one common-mode amplitude burst, one single-channel
    z-score outlier -- plus heterogeneous gains so the µV scaling matters.
    """
    import spikeinterface as si

    fs = 30_000.0
    n_samples = 90_000
    n_channels = 8
    rng = np.random.default_rng(0)
    # Small baseline noise that never trips either threshold.
    traces = rng.normal(0.0, 2.0, size=(n_samples, n_channels)).astype(
        np.float32
    )
    # Common-mode amplitude burst across all channels (trips amplitude).
    traces[20_000:20_300, :] += 300.0
    # Single-channel transient (trips the across-channel z-score).
    traces[60_000:60_120, 3] += 400.0
    rec = si.NumpyRecording(traces_list=[traces], sampling_frequency=fs)
    # Heterogeneous gains: the chunk worker and the reference must apply the
    # SAME per-channel gain, so a wrong gain broadcast surfaces as inequality.
    rec.set_channel_gains([1.0, 0.5, 2.0, 0.25, 1.0, 1.5, 0.8, 1.2])
    return rec


@pytest.mark.parametrize(
    "amplitude_threshold_uv,zscore_threshold",
    [
        (50.0, None),  # amplitude-only branch
        (None, 6.0),  # z-score-only branch
        (50.0, 6.0),  # OR-combined branch
    ],
)
def test_chunked_artifact_matches_in_memory_reference(
    dj_conn, amplitude_threshold_uv, zscore_threshold
):
    """A17: the chunked ``_scan_artifact_frames`` produces frame-identical
    output to the frozen full-in-memory reference, and is invariant to chunk
    boundaries.

    This is the gating equivalence evidence: A17 replaced the single
    full-recording ``get_traces`` call with a per-chunk
    ``ChunkRecordingExecutor`` pass to bound peak memory. The port is correct
    only if the flagged frame set is unchanged. The per-frame across-channel
    z-score depends solely on that frame's columns, so chunk boundaries
    (which split the time axis) cannot change which frames are flagged --
    this test pins that property across all three detection branches.
    """
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    rec = _synthetic_artifact_recording()
    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=amplitude_threshold_uv,
        zscore_threshold=zscore_threshold,
        proportion_above_threshold=0.5,
        removal_window_ms=1.0,
        join_window_ms=0.0,
        min_length_s=0.001,
    )

    reference = _in_memory_artifact_frames_reference(rec, validated)

    # Many small chunks (~0.1 s each → ~30 chunks) exercises chunk seams.
    chunked = ArtifactDetection._scan_artifact_frames(
        rec, validated, job_kwargs={"n_jobs": 1, "chunk_duration": "0.1s"}
    )
    # A single chunk spanning the whole recording == the in-memory path.
    single = ArtifactDetection._scan_artifact_frames(
        rec, validated, job_kwargs={"n_jobs": 1, "chunk_size": 90_000}
    )

    assert np.array_equal(chunked, reference), (
        "Chunked artifact frames diverge from the in-memory reference; the "
        "ChunkRecordingExecutor port changed which frames are flagged."
    )
    assert np.array_equal(
        single, reference
    ), "Single-chunk pass should reproduce the in-memory reference exactly."


def test_artifact_job_kwargs_propagate_to_executor(dj_conn, monkeypatch):
    """A17: per-row ``job_kwargs`` reach ``ChunkRecordingExecutor``.

    The audit found the stored ``job_kwargs`` blob was dead weight under the
    in-memory scan. A17 wires it through ``_resolved_job_kwargs`` into the
    executor. This test patches the executor constructor and asserts an
    ``n_jobs=2`` override (a recognized SI job key) is observed.
    """
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    rec = _synthetic_artifact_recording()
    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=50.0,
        zscore_threshold=None,
        proportion_above_threshold=0.5,
        removal_window_ms=1.0,
        join_window_ms=0.0,
        min_length_s=0.001,
    )

    seen = {}

    import spikeinterface.core.job_tools as jt

    real_executor = jt.ChunkRecordingExecutor

    class _SpyExecutor(real_executor):
        def __init__(self, *args, **kwargs):
            seen["n_jobs"] = kwargs.get("n_jobs")
            seen["chunk_duration"] = kwargs.get("chunk_duration")
            super().__init__(*args, **kwargs)

        def run(self, *args, **kwargs):
            # The test asserts the constructor received the row's
            # job_kwargs; it does not need the worker pool to execute (an
            # in-memory NumpyRecording cannot round-trip through
            # ``to_dict()``/``load_extractor`` for a real multi-process pool).
            return []

    # ``scan_artifact_frames`` (now in ``_artifact_intervals``) lazy-imports
    # ``ChunkRecordingExecutor`` from its SI source at call time, so patch the
    # source module -- the ``from spikeinterface.core.job_tools import
    # ChunkRecordingExecutor`` inside the scan binds this spy at call time.
    monkeypatch.setattr(jt, "ChunkRecordingExecutor", _SpyExecutor)

    ArtifactDetection._scan_artifact_frames(
        rec,
        validated,
        job_kwargs={"n_jobs": 2, "chunk_duration": "0.5s"},
    )
    assert seen["n_jobs"] == 2, (
        "Row job_kwargs n_jobs=2 did not reach ChunkRecordingExecutor; the "
        "job_kwargs blob is still dead weight."
    )
    assert seen["chunk_duration"] == "0.5s"


def test_artifact_scan_chunked_by_default(dj_conn, monkeypatch):
    """A17: with NO job_kwargs the scan still chunks (chunk_duration='1s').

    ChunkRecordingExecutor with no chunk-size key processes the whole
    recording in a single chunk -- the exact full-traces memory profile the
    restoration removed. Any direct ``_detect_artifacts`` caller (not just the
    production ``make_compute`` path that merges SI's global job kwargs) must
    still be bounded, so ``_scan_artifact_frames`` defaults to a 1 s chunk.
    """
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    rec = _synthetic_artifact_recording()
    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=50.0,
        zscore_threshold=None,
        proportion_above_threshold=0.5,
        removal_window_ms=1.0,
        join_window_ms=0.0,
        min_length_s=0.001,
    )

    seen = {}

    import spikeinterface.core.job_tools as jt

    real_executor = jt.ChunkRecordingExecutor

    class _SpyExecutor(real_executor):
        def __init__(self, *args, **kwargs):
            seen["chunk_duration"] = kwargs.get("chunk_duration")
            seen["chunk_size"] = kwargs.get("chunk_size")
            super().__init__(*args, **kwargs)

    # ``scan_artifact_frames`` (now in ``_artifact_intervals``) lazy-imports
    # ``ChunkRecordingExecutor`` from its SI source at call time, so patch the
    # source module -- the ``from spikeinterface.core.job_tools import
    # ChunkRecordingExecutor`` inside the scan binds this spy at call time.
    monkeypatch.setattr(jt, "ChunkRecordingExecutor", _SpyExecutor)

    # No job_kwargs at all -- the bare default path.
    ArtifactDetection._scan_artifact_frames(rec, validated)
    assert seen["chunk_duration"] == "1s", (
        "default scan did not chunk: ChunkRecordingExecutor got "
        f"chunk_duration={seen.get('chunk_duration')!r} / "
        f"chunk_size={seen.get('chunk_size')!r}, so it would load the whole "
        "recording in one chunk (the memory trap A17 removed)."
    )


@pytest.mark.slow
def test_artifact_scan_multiprocess_worker_path_runs(dj_conn, tmp_path):
    """A17: the n_jobs>1 worker path actually EXECUTES and matches n_jobs=1.

    For n_jobs>1 the recording is passed to each worker as a ``to_dict()`` blob
    and re-hydrated inside ``_init_artifact_worker``. v1 used SI 0.99's
    ``si.load_extractor``; SI 0.104 renamed it to ``si.load``, so the worker
    would ``AttributeError`` if it still called the old name. The propagation
    test only inspects constructor kwargs and returns before workers run, so it
    cannot catch this. This test runs the multiprocess path end-to-end on a
    serializable (saved-to-disk binary) recording and asserts the flagged
    frames equal the n_jobs=1 result -- proving the worker rehydrates and the
    chunked output is worker-count-invariant.
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    # _synthetic_artifact_recording is an in-memory NumpyRecording (not
    # dict-serializable); save it to a binary folder so to_dict()/si.load()
    # round-trips, the way a real cached preprocessed Recording does.
    saved = _synthetic_artifact_recording().save(folder=tmp_path / "rec")
    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=50.0,
        zscore_threshold=None,
        proportion_above_threshold=0.5,
        removal_window_ms=1.0,
        join_window_ms=0.0,
        min_length_s=0.001,
    )

    single = ArtifactDetection._scan_artifact_frames(
        saved, validated, job_kwargs={"n_jobs": 1, "chunk_duration": "0.5s"}
    )
    multi = ArtifactDetection._scan_artifact_frames(
        saved, validated, job_kwargs={"n_jobs": 2, "chunk_duration": "0.5s"}
    )
    # The planted bursts must actually be detected, else "equal" is vacuous.
    assert len(single) > 0
    assert np.array_equal(single, multi), (
        "n_jobs=2 worker path produced different frames than n_jobs=1 (or "
        "failed to rehydrate the to_dict() recording via si.load)."
    )


@pytest.mark.slow
def test_artifact_detection_peak_memory_bounded_by_chunk_size(dj_conn):
    """A17: peak memory of the chunked scan is bounded by chunk size, not by
    the full recording length.

    Builds a 5-minute, 32-channel recording (~1.15 GB if materialized in
    float32 once; ~4.6 GB at the in-memory path's ~4× working-set factor) and
    scans it with a 1-second chunk. Peak *additional* allocation measured by
    ``tracemalloc`` must stay far below the full-recording working set -- it is
    bounded by ``~4 × chunk_frames × n_channels × 4 bytes`` plus the flagged
    frame-index arrays. Asserts the peak stays under a generous 256 MB ceiling
    that the full-recording load would blow through.
    """
    import tracemalloc

    import spikeinterface as si

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    fs = 30_000.0
    n_channels = 32
    n_samples = int(fs * 300)  # 5 minutes
    # Memmap-backed traces so building the fixture itself does not dominate
    # the measured peak; the scan reads it chunk by chunk.
    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
    tmp.close()
    arr = np.memmap(
        tmp.name, dtype=np.float32, mode="w+", shape=(n_samples, n_channels)
    )
    arr[:] = 0.0
    arr[1_000_000:1_000_300, :] = 300.0  # one planted burst
    arr.flush()
    rec = si.NumpyRecording(traces_list=[arr], sampling_frequency=fs)
    rec.set_channel_gains([1.0] * n_channels)

    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=50.0,
        zscore_threshold=None,
        proportion_above_threshold=0.5,
        removal_window_ms=1.0,
        join_window_ms=0.0,
        min_length_s=0.001,
    )

    tracemalloc.start()
    ArtifactDetection._scan_artifact_frames(
        rec, validated, job_kwargs={"n_jobs": 1, "chunk_duration": "1s"}
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    full_working_set = 4 * n_samples * n_channels * 4  # ~4.6 GB
    ceiling = 256 * 1024 * 1024  # 256 MB
    assert peak < ceiling, (
        f"Chunked scan peaked at {peak / 1e6:.1f} MB, above the "
        f"{ceiling / 1e6:.0f} MB ceiling; chunking is not bounding memory "
        f"(full-recording working set would be ~{full_working_set / 1e9:.1f} "
        "GB)."
    )


# ---------- A20: tetrode-geometry gate negative cases ----------------------


def _assert_tetrode_gate_noop(
    caplog, probe_types, electrode_group_names, channel_ids, reason_substr
):
    """Call ``_maybe_apply_tetrode_geometry`` with a failing-gate setup and
    assert (a) the recording geometry is untouched and (b) an INFO log names
    the condition that failed.

    The synthetic recording carries SI's default linear probe (contacts at
    y=0,20,40,...); the tetrode patch would replace it with a 12.5 µm square,
    so an unchanged ``get_channel_locations()`` proves the gate no-opped.
    """
    import logging

    import spikeinterface as si

    from spyglass.spikesorting.v2.recording import Recording

    rec = si.generate_recording(
        num_channels=len(channel_ids),
        durations=[1.0],
        sampling_frequency=30_000.0,
    )
    before = rec.get_channel_locations().copy()
    with caplog.at_level(logging.INFO):
        result = Recording._maybe_apply_tetrode_geometry(
            rec, probe_types, electrode_group_names, channel_ids
        )
    after = result.get_channel_locations()
    assert np.array_equal(before, after), (
        "tetrode geometry was applied despite a failed gate condition; the "
        "channel locations changed."
    )
    skip_msgs = [
        r.getMessage()
        for r in caplog.records
        if "_maybe_apply_tetrode_geometry skipped" in r.getMessage()
    ]
    assert any(reason_substr in m for m in skip_msgs), (
        f"expected an INFO log naming the failed condition "
        f"({reason_substr!r}); got skip logs {skip_msgs}."
    )


def test_tetrode_geometry_gate_three_channel(dj_conn, caplog):
    """A20: a 3-channel sort group (e.g. after a bad-channel drop) is not a
    tetrode; the gate no-ops and logs the channel-count condition."""
    _assert_tetrode_gate_noop(
        caplog,
        probe_types=("tetrode_12.5",) * 3,
        electrode_group_names=("g",) * 3,
        channel_ids=[0, 1, 2],
        reason_substr="4 channel",
    )


def test_tetrode_geometry_gate_mixed_probe(dj_conn, caplog):
    """A20: a sort group spanning two probe types is not a single tetrode;
    the gate no-ops and logs the multiple-probe-types condition."""
    _assert_tetrode_gate_noop(
        caplog,
        probe_types=(
            "tetrode_12.5",
            "tetrode_12.5",
            "other_probe",
            "other_probe",
        ),
        electrode_group_names=("g",) * 4,
        channel_ids=[0, 1, 2, 3],
        reason_substr="multiple probe type",
    )


def test_tetrode_geometry_gate_renamed_probe(dj_conn, caplog):
    """A20: a single-probe group whose probe string is not exactly
    ``tetrode_12.5`` (e.g. a renamed ``tetrode_12.5_v2``) is not patched; the
    gate no-ops and logs the probe-type-mismatch condition."""
    _assert_tetrode_gate_noop(
        caplog,
        probe_types=("tetrode_12.5_v2",) * 4,
        electrode_group_names=("g",) * 4,
        channel_ids=[0, 1, 2, 3],
        reason_substr="not tetrode_12.5",
    )


def test_tetrode_geometry_gate_multi_group(dj_conn, caplog):
    """A20: four channels split across two electrode groups is not a single
    tetrode; the gate no-ops and logs the multiple-groups condition."""
    _assert_tetrode_gate_noop(
        caplog,
        probe_types=("tetrode_12.5",) * 4,
        electrode_group_names=("g1", "g1", "g2", "g2"),
        channel_ids=[0, 1, 2, 3],
        reason_substr="multiple electrode group",
    )


# ---------- A19: channel_name resolution on a real-NWB-shape fixture -------


@pytest.mark.parametrize(
    "channel_names", [None, ["ch_a", "ch_b", "ch_c", "ch_d"]]
)
def test_channel_name_resolution_path_real_nwb(
    dj_conn, tmp_path, monkeypatch, channel_names
):
    """A19: ``_spikeinterface_channel_ids`` resolves channel ids from the raw
    NWB ``channel_name`` column when present, and falls back to integer
    ``electrode_id`` when absent.

    The MEArec fixtures omit ``channel_name`` so only the integer-fallback
    branch was exercised; production Frank-lab NWBs carry the column. This
    test builds a 4-contact NWB via the fixture builder's new ``channel_names``
    parameter (injecting the column) and via the default (no column), then
    asserts the resolved SpikeInterface channel ids match each branch's
    expected mapping.
    """
    from datetime import datetime, timezone

    import pynwb

    from spyglass.common import Nwbfile
    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        _add_probe_and_electrodes,
        tetrode_probe_layout,
    )
    from spyglass.spikesorting.v2.recording import Recording

    nwbfile = pynwb.NWBFile(
        session_description="channel_name resolution fixture",
        identifier="a19-chan-name",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    _add_probe_and_electrodes(
        nwbfile,
        tetrode_probe_layout(),
        targeted_location="hpc",
        channel_names=channel_names,
    )
    out = tmp_path / "a19_channel_name_fixture.nwb"
    with pynwb.NWBHDF5IO(str(out), mode="w") as io:
        io.write(nwbfile)

    # _spikeinterface_channel_ids resolves the raw path via Nwbfile; redirect
    # it to our standalone fixture (no ingestion needed for the lookup).
    monkeypatch.setattr(
        Nwbfile, "get_abs_path", staticmethod(lambda *a, **k: str(out))
    )

    spyglass_ids = [0, 1, 2, 3]
    resolved = Recording._spikeinterface_channel_ids(
        "a19_channel_name_fixture.nwb", spyglass_ids
    )

    if channel_names is None:
        assert resolved == [0, 1, 2, 3], (
            "integer-fallback branch must return int electrode_ids; got "
            f"{resolved!r}"
        )
        assert all(isinstance(c, int) for c in resolved)
    else:
        assert resolved == channel_names, (
            "channel_name branch must resolve to the injected string names "
            f"in electrode order; got {resolved!r}"
        )


# ---------- A21: SI version pin + sorter-default snapshot tests ------------
#
# SI is pinned to ==0.104.3 in pyproject.toml. The KS4/MS5/SC2/TDC2/Generic
# v2 schemas use extra="allow" (overview decision #2 -- KS4 stays permissive,
# NOT made strict), so every sorter field the schema does not type falls
# through to SpikeInterface's per-version default at sort time. These
# snapshot tests fail loudly when a SI bump shifts those defaults so the diff
# can be audited against current sort outputs before the pin moves.


# KS4's install-independent SI-wrapper defaults (Kilosort4Sorter
# ._si_default_params). The kilosort-ALGORITHM defaults (Th_universal,
# batch_size, nearest_chans, ...) live in kilosort.parameters.DEFAULT_SETTINGS
# and require the kilosort4 package to be installed; the CI SI-0.104 image does
# NOT install KS4 (get_default_sorter_params('kilosort4') then returns only the
# global job-kwargs + a "not installed" warning -- verified). This snapshot
# pins the SI-controlled wrapper subset (always readable). The algorithm-level
# knobs are pinned by the companion skipif test below, which runs only where
# KS4 is installed.
EXPECTED_KS4_SI_DEFAULTS = {
    "do_CAR": True,
    "invert_sign": False,
    "save_extra_vars": False,
    "save_preprocessed_copy": False,
    "torch_device": "auto",
    "bad_channels": None,
    "clear_cache": False,
    "do_correction": True,
    "skip_kilosort_preprocessing": False,
    "keep_good_only": False,
    "use_binary_file": True,
    "delete_recording_dat": True,
}


# MS5's full default surface (minus the global job-kwargs), SI 0.104.3. The
# v2 MS5 schema types only a subset; the remaining ~8 fields are silently
# stripped/defaulted at sort time (documented in Phase 7's migration guide).
# Snapshotting them here lets that guide name the actual hidden values.
EXPECTED_MS5_DEFAULTS = {
    "scheme": "2",
    "detect_threshold": 5.5,
    "detect_sign": -1,
    "detect_time_radius_msec": 0.5,
    "snippet_T1": 20,
    "snippet_T2": 20,
    "npca_per_channel": 3,
    "npca_per_subdivision": 10,
    "snippet_mask_radius": 250,
    "scheme1_detect_channel_radius": 150,
    "scheme2_phase1_detect_channel_radius": 200,
    "scheme2_detect_channel_radius": 50,
    "scheme2_max_num_snippets_per_training_batch": 200,
    "scheme2_training_duration_sec": 300,
    "scheme2_training_recording_sampling_mode": "uniform",
    "scheme3_block_duration_sec": 1800,
    "freq_min": 300,
    "freq_max": 6000,
    "filter": True,
    "whiten": True,
    "delete_temporary_recording": True,
}

# SpykingCircus2 / TridesClous2 ship with ``extra="allow"`` (untyped) v2
# schemas and EMPTY default-row params, so their effective defaults float
# with the installed SpikeInterface rather than being frozen into the row
# (as v1 did). These snapshots pin SI 0.104.3's defaults so an SI bump
# that would silently change either sorter's behavior surfaces as a test
# failure -- the same guard A21 gives KS4/MS5.
EXPECTED_SC2_DEFAULTS = {
    "apply_motion_correction": True,
    "apply_preprocessing": True,
    "apply_whitening": True,
    "cache_preprocessing": {"memory_limit": 0.5, "mode": "memory"},
    "chunk_preprocessing": {"memory_limit": None},
    "cleaning": {
        "max_jitter_ms": 0.2,
        "mean_sd_ratio_threshold": 3,
        "min_snr": 5,
        "sparsify_threshold": 1,
    },
    "clustering": {"method": "iterative-hdbscan", "method_kwargs": {}},
    "debug": False,
    "detection": {
        "method": "matched_filtering",
        "method_kwargs": {"detect_threshold": 5, "peak_sign": "neg"},
        "pipeline_kwargs": {},
    },
    "deterministic_peaks_detection": False,
    "filtering": {
        "filter_order": 2,
        "freq_max": 7000,
        "freq_min": 150,
        "ftype": "bessel",
    },
    "general": {"ms_after": 1.5, "ms_before": 0.5, "radius_um": 100.0},
    "job_kwargs": {},
    "matching": {
        "method": "circus-omp",
        "method_kwargs": {},
        "pipeline_kwargs": {},
    },
    "merging": {"max_distance_um": 50},
    "min_firing_rate": 0.1,
    "motion_correction": {"preset": "dredge_fast"},
    "multi_units_only": False,
    "seed": 42,
    "selection": {
        "method": "uniform",
        "method_kwargs": {
            "min_n_peaks": 100000,
            "n_peaks_per_channel": 5000,
            "select_per_channel": False,
        },
    },
    "whitening": {"mode": "local", "regularize": False},
}

EXPECTED_TDC2_DEFAULTS = {
    "apply_motion_correction": False,
    "apply_preprocessing": True,
    "cache_preprocessing_mode": "auto",
    "clustering_ms_after": 1.5,
    "clustering_ms_before": 0.5,
    "clustering_recursive_depth": 3,
    "debug": False,
    "detect_threshold": 5.0,
    "detection_radius_um": 150.0,
    "features_radius_um": 120.0,
    "freq_max": 6000.0,
    "freq_min": 150.0,
    "gather_mode": "memory",
    "job_kwargs": {},
    "merge_similarity_lag_ms": 0.5,
    "min_firing_rate": 0.1,
    "motion_correction_preset": "dredge_fast",
    "ms_after": 2.5,
    "ms_before": 1.0,
    "n_pca_features": 6,
    "n_peaks_per_channel": 5000,
    "n_svd_components_per_channel": 5,
    "peak_sign": "neg",
    "preprocessing_dict": None,
    "save_array": True,
    "seed": None,
    "split_radius_um": 60.0,
    "template_max_jitter_ms": 0.2,
    "template_min_snr_ptp": 3.5,
    "template_radius_um": 100.0,
    "template_sparsify_threshold": 1.5,
}


def test_kilosort4_si_defaults_unchanged():
    """A21: SI's install-independent KS4 wrapper defaults match the snapshot.

    Pins ``Kilosort4Sorter._si_default_params`` (the SI-controlled overlay,
    readable without the kilosort4 package). A SI bump that shifts any of these
    surfaces as a test failure rather than a silent change to v2 KS4 sort
    behavior. Diff against the pinned snapshot, decide whether v2's typed-5
    KS4 subset still expresses the right knobs, then update the snapshot and
    the CHANGELOG / SI pin together.
    """
    from spikeinterface.sorters.external.kilosort4 import Kilosort4Sorter

    assert Kilosort4Sorter._si_default_params == EXPECTED_KS4_SI_DEFAULTS, (
        "SI's KS4 wrapper defaults shifted. Diff the change against the "
        "pinned EXPECTED_KS4_SI_DEFAULTS, confirm v2's typed KS4 subset still "
        "covers the right knobs, then update the snapshot, the SI pin in "
        "pyproject.toml, and the CHANGELOG together."
    )


# The kilosort-ALGORITHM defaults the audit named as the silent-drift risk
# ("a SI upgrade changing batch_size or nearest_chans defaults"). Values read
# authoritatively from kilosort 4.1.7's parameters.py (MAIN_PARAMETERS) and
# surfaced top-level by SI's KS4 wrapper. Asserted as a SUBSET (not full-dict
# equality): KS4 is an optional, UNPINNED package, so the full default dict
# varies harmlessly across kilosort patch releases -- pinning the full dict
# would false-fail on unrelated version differences. The subset pins exactly
# the knobs whose drift changes sort outputs.
EXPECTED_KS4_ALGORITHM_DEFAULTS = {
    "Th_universal": 9,
    "Th_learned": 8,
    "batch_size": 60000,
    "nearest_chans": 10,
}


def _kilosort4_installed():
    import spikeinterface.sorters as sis

    return "kilosort4" in sis.installed_sorters()


@pytest.mark.skipif(
    not _kilosort4_installed(),
    reason="kilosort4 not installed; algorithm defaults are unreadable "
    "without the package (see test_kilosort4_si_defaults_unchanged for the "
    "install-independent wrapper-overlay snapshot that always runs).",
)
def test_kilosort4_algorithm_defaults_unchanged():
    """A21: KS4's algorithm-level defaults (the audit's drift risk) match.

    Runs only where kilosort4 is installed (skipped on the CI SI-0.104 image).
    Pins the specific knobs the audit named -- ``Th_universal`` / ``Th_learned``
    / ``batch_size`` / ``nearest_chans`` -- whose silent drift across a SI/KS4
    bump changes sort outputs. Subset assertion (not full-dict equality) so an
    unrelated kilosort patch-version difference does not false-fail; a change
    to ONE of these knobs surfaces loudly. Note: because this is skipped in the
    KS4-less CI lane, continuous protection requires a KS4-enabled CI job
    (infra follow-up, out of Phase 5 scope).
    """
    import spikeinterface.sorters as sis

    actual = sis.get_default_sorter_params("kilosort4")
    observed = {
        k: actual[k] for k in EXPECTED_KS4_ALGORITHM_DEFAULTS if k in actual
    }
    assert observed == EXPECTED_KS4_ALGORITHM_DEFAULTS, (
        "KS4 algorithm defaults shifted (or a knob was renamed/removed). "
        f"expected (subset) {EXPECTED_KS4_ALGORITHM_DEFAULTS}, got {observed}. "
        "Audit the change against current sort outputs, then update this "
        "snapshot + the SI pin + the CHANGELOG together."
    )


def test_ms5_si_defaults_unchanged():
    """A21: SI's MountainSort5 defaults match the snapshot.

    MS5 is installed in the SI-0.104 env, so its full default surface is
    readable. Excludes the global job-kwargs (``job_keys``) so the snapshot
    is independent of any per-session job-kwargs config and reflects only the
    MS5 algorithm defaults -- including the fields v2's MS5 schema silently
    strips (Phase 7 migration guide names them from this snapshot).
    """
    import spikeinterface.sorters as sis
    from spikeinterface.core.job_tools import job_keys

    actual = {
        k: v
        for k, v in sis.get_default_sorter_params("mountainsort5").items()
        if k not in job_keys
    }
    assert actual == EXPECTED_MS5_DEFAULTS, (
        "SI's MountainSort5 defaults shifted. Diff against the pinned "
        "EXPECTED_MS5_DEFAULTS, then update the snapshot, the SI pin in "
        "pyproject.toml, and the CHANGELOG together."
    )


def test_spykingcircus2_si_defaults_unchanged():
    """A21 (extended): SI's SpykingCircus2 defaults match the snapshot.

    SC2's v2 schema is ``extra="allow"`` and ships an EMPTY default-row
    params blob, so its effective defaults come from the installed SI at
    sort time. This pins SI 0.104.3's defaults so an SI bump that would
    silently change SC2's behavior surfaces as a test failure (loosen only
    alongside the SI pin in pyproject.toml + the CHANGELOG).
    """
    import spikeinterface.sorters as sis
    from spikeinterface.core.job_tools import job_keys

    actual = {
        k: v
        for k, v in sis.get_default_sorter_params("spykingcircus2").items()
        if k not in job_keys
    }
    assert actual == EXPECTED_SC2_DEFAULTS, (
        "SI's SpykingCircus2 defaults shifted. Diff against the pinned "
        "EXPECTED_SC2_DEFAULTS, then update the snapshot, the SI pin in "
        "pyproject.toml, and the CHANGELOG together."
    )


def test_tridesclous2_si_defaults_unchanged():
    """A21 (extended): SI's TridesClous2 defaults match the snapshot.

    Same rationale as SpykingCircus2: TDC2's v2 schema is ``extra="allow"``
    with an empty default-row params blob, so the effective defaults float
    with the installed SI. Pin SI 0.104.3 here so a bump is a deliberate,
    audited change.
    """
    import spikeinterface.sorters as sis
    from spikeinterface.core.job_tools import job_keys

    actual = {
        k: v
        for k, v in sis.get_default_sorter_params("tridesclous2").items()
        if k not in job_keys
    }
    assert actual == EXPECTED_TDC2_DEFAULTS, (
        "SI's TridesClous2 defaults shifted. Diff against the pinned "
        "EXPECTED_TDC2_DEFAULTS, then update the snapshot, the SI pin in "
        "pyproject.toml, and the CHANGELOG together."
    )


# ---------- A22: analyzer-folder disk-leak audit job -----------------------


def _insert_bypassed_sorting_row(sid, *, n_units):
    """Insert a minimal Sorting row via FK-checks-off bypass.

    Building a real Sorting row needs the full sort pipeline; for the
    disk-leak audit we only need rows carrying a ``sorting_id`` +
    ``n_units`` (the analyzer cache path is computed from ``sorting_id``, not
    stored), so bypass the AnalysisNwbfile FK like the A10 pin does. Caller
    cleans up via the SortingSelection cascade.
    """
    import datetime as dt

    import datajoint as dj

    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    SorterParameters.insert_default()
    SortingSelection.insert1(
        {
            "sorting_id": sid,
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        },
        allow_direct_insert=True,
    )
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        Sorting.insert1(
            {
                "sorting_id": sid,
                "analysis_file_name": "a22_fake.nwb",
                "object_id": "a22-fake-object-id",
                "n_units": n_units,
                "time_of_sort": dt.datetime(2020, 1, 1, 0, 0, 0),
            },
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")


def test_find_orphaned_analyzer_folders_db_side(dj_conn):
    """A22: a Sorting row (n_units > 0) whose analyzer folder is gone on disk
    is reported as a DB-side orphan -- and nothing is auto-deleted."""
    import uuid

    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    sid = uuid.uuid4()
    folder = analyzer_path(sid)  # never written on disk
    assert not folder.exists()
    _insert_bypassed_sorting_row(sid, n_units=3)

    try:
        report = Sorting.find_orphaned_analyzer_folders(dry_run=True)
        db_ids = {str(r["sorting_id"]) for r in report["db_side"]}
        assert str(sid) in db_ids, (
            "a units-bearing row with a missing analyzer folder must be "
            "reported as a DB-side orphan"
        )
        # dry_run must not delete the row.
        assert Sorting & {"sorting_id": sid}, (
            "find_orphaned_analyzer_folders(dry_run=True) deleted a DB row; "
            "it must only report"
        )
    finally:
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


def test_find_orphaned_analyzer_folders_disk_side(dj_conn, tmp_path):
    """A22: a stray on-disk analyzer folder referenced by no Sorting row is
    reported as a disk-side orphan."""
    import uuid

    from spyglass.spikesorting.v2.sorting import Sorting
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    analyzer_root = analyzer_path("x").parent
    analyzer_root.mkdir(parents=True, exist_ok=True)
    stray = analyzer_root / f"a22_disk_orphan_{uuid.uuid4()}.analyzer"
    stray.mkdir()
    try:
        report = Sorting.find_orphaned_analyzer_folders(dry_run=True)
        assert str(stray) in report["disk_side"], (
            "an on-disk analyzer folder with no referencing Sorting row must "
            "be reported as a disk-side orphan"
        )
    finally:
        import shutil

        shutil.rmtree(stray, ignore_errors=True)


def test_find_orphaned_analyzer_folders_zero_unit_carveout(dj_conn):
    """A22: a zero-unit Sorting row is NOT a DB-side orphan even though its
    computed analyzer cache path does not exist on disk.

    ``_build_analyzer`` short-circuits before writing a folder and
    ``get_analyzer`` raises ``ZeroUnitAnalyzerError`` before reading the path,
    so a missing folder for a zero-unit row is expected, not a leak. The
    carve-out is keyed on ``n_units == 0`` (the cache path is computed from
    ``sorting_id``, not stored, so there is no column value to match on).
    """
    import uuid

    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    sid = uuid.uuid4()
    folder = analyzer_path(sid)  # never written on disk
    assert not folder.exists()
    _insert_bypassed_sorting_row(sid, n_units=0)

    try:
        report = Sorting.find_orphaned_analyzer_folders(dry_run=True)
        db_ids = {str(r["sorting_id"]) for r in report["db_side"]}
        assert str(sid) not in db_ids, (
            "a zero-unit row whose folder is (legitimately) absent must NOT "
            "be reported as a DB-side orphan -- the n_units==0 carve-out "
            "regressed"
        )
    finally:
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


# ---------- A23: _run_si_sorter global job_kwargs restore ------------------


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


def test_run_si_sorter_restores_global_job_kwargs_on_raise(
    dj_conn, monkeypatch
):
    """A23: SI's global job_kwargs are restored after the sort raises.

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
    """A23: SI's global job_kwargs are restored after a successful sort too."""
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


# ===========================================================================
# A25: artifact-detection invariant + lookup branches.
#
# SharedArtifactGroup.insert_group cross-session / cross-frequency guards,
# ArtifactDetectionSelection.insert_selection duplicate + missing-Lookup-row
# diagnostics, the empty sliver-filter return, and the already-gone
# IntervalList cleanup. The insert_group session/frequency checks and the
# duplicate check all fire BEFORE any recording load, so every
# precondition is built via the contained FK-checks-off raw bypass (the
# established A10/A11/A22 pattern) -- no real populate needed.
# ===========================================================================


def _plant_fake_recording(recording_id, nwb_file_name, sampling_frequency):
    """Insert a minimal ``Recording`` + ``RecordingSelection`` pair via the
    FK-checks-off bypass.

    ``SharedArtifactGroup.insert_group`` reads ``RecordingSelection.
    nwb_file_name`` (session check) and ``Recording.sampling_frequency``
    (frequency check) and only requires the ``Recording`` row to *exist*
    for the populated-check -- all before it ever loads the recording. So
    a fake pair with the right two scalar fields is enough to drive both
    guards without a real populate. Returns the ``recording_id``.
    """
    import datajoint as dj

    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection

    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        RecordingSelection.insert1(
            {
                "recording_id": recording_id,
                "nwb_file_name": nwb_file_name,
                "sort_group_id": 0,
                "interval_list_name": "raw data valid times",
                "preprocessing_params_name": "default",
                "team_name": "v2_a25_team",
            },
            allow_direct_insert=True,
        )
        Recording.insert1(
            {
                "recording_id": recording_id,
                "analysis_file_name": "a25_fake.nwb",
                "electrical_series_path": "/fake/es",
                "object_id": "a25-fake-object-id",
                "n_channels": 4,
                "sampling_frequency": sampling_frequency,
                "duration_s": 60.0,
                "cache_hash": "0" * 64,
            },
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")
    return recording_id


def _drop_fake_recording(recording_id):
    """Tear down a ``_plant_fake_recording`` pair (parts-first)."""
    import datajoint as dj

    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection

    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        (Recording & {"recording_id": recording_id}).delete_quick()
        (RecordingSelection & {"recording_id": recording_id}).delete_quick()
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")


@pytest.mark.usefixtures("dj_conn")
def test_shared_artifact_group_rejects_cross_session_members():
    """A25: members from two sessions raise, and no master row is left.

    A shared artifact pass writes IntervalList rows keyed by
    ``(nwb_file_name, interval_list_name)``, so mixing sessions makes the
    artifact-removed valid times undefined. The guard fires before the
    master insert, so the table must be empty for this group name after
    the raise (transactional rollback / pre-insert raise).
    """
    import uuid

    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup

    rid_a = _plant_fake_recording(uuid.uuid4(), "session_a_.nwb", 30000.0)
    rid_b = _plant_fake_recording(uuid.uuid4(), "session_b_.nwb", 30000.0)
    group_name = "v2_a25_cross_session"
    try:
        with pytest.raises(ValueError, match="members span 2 sessions"):
            SharedArtifactGroup.insert_group(
                group_name,
                [{"recording_id": rid_a}, {"recording_id": rid_b}],
            )
        assert (
            len(
                SharedArtifactGroup & {"shared_artifact_group_name": group_name}
            )
            == 0
        ), "master row not rolled back after cross-session raise"
    finally:
        _drop_fake_recording(rid_a)
        _drop_fake_recording(rid_b)


@pytest.mark.usefixtures("dj_conn")
def test_shared_artifact_group_rejects_frequency_mismatch():
    """A25: members with differing sampling frequencies raise.

    ``si.aggregate_channels`` requires identical fs; a typo in upstream
    preproc could otherwise produce a mismatch that only crashes opaquely
    deep inside SI at populate time. Both members share one session so the
    session guard passes and the frequency guard is the one that fires.
    """
    import uuid

    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup

    rid_a = _plant_fake_recording(uuid.uuid4(), "session_freq_.nwb", 30000.0)
    rid_b = _plant_fake_recording(uuid.uuid4(), "session_freq_.nwb", 25000.0)
    try:
        with pytest.raises(ValueError, match="differing sampling frequencies"):
            SharedArtifactGroup.insert_group(
                "v2_a25_freq_mismatch",
                [{"recording_id": rid_a}, {"recording_id": rid_b}],
            )
    finally:
        _drop_fake_recording(rid_a)
        _drop_fake_recording(rid_b)


@pytest.mark.usefixtures("dj_conn")
def test_shared_artifact_group_rejects_timestamp_mismatch(monkeypatch):
    """A25: equal-length same-session members still need equal timestamps.

    The two fake recordings share session, sampling frequency, sample count,
    and dtype. Only their wall-clock vectors differ. Without the exact
    timestamp guard, ``si.aggregate_channels`` would stack non-aligned frames
    and the resulting artifact intervals could mask the wrong times.
    """
    import uuid

    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup
    from spyglass.spikesorting.v2.recording import Recording

    rid_a = _plant_fake_recording(uuid.uuid4(), "session_times_.nwb", 30000.0)
    rid_b = _plant_fake_recording(uuid.uuid4(), "session_times_.nwb", 30000.0)

    class _FakeRecording:
        def __init__(self, times):
            self._times = np.asarray(times, dtype=np.float64)

        def get_num_samples(self):
            return len(self._times)

        def get_dtype(self):
            return "float32"

        def get_num_segments(self):
            return 1

        def get_times(self):
            return self._times

    base_times = np.arange(8, dtype=np.float64) / 30000.0

    def _fake_get_recording(self, key):
        if str(key["recording_id"]) == str(rid_a):
            return _FakeRecording(base_times)
        return _FakeRecording(base_times + 10.0)

    monkeypatch.setattr(Recording, "get_recording", _fake_get_recording)

    try:
        with pytest.raises(ValueError, match="differing exact timestamps"):
            SharedArtifactGroup.insert_group(
                "v2_a25_timestamp_mismatch",
                [{"recording_id": rid_a}, {"recording_id": rid_b}],
            )
        assert (
            len(
                SharedArtifactGroup
                & {"shared_artifact_group_name": "v2_a25_timestamp_mismatch"}
            )
            == 0
        )
    finally:
        _drop_fake_recording(rid_a)
        _drop_fake_recording(rid_b)


@pytest.mark.usefixtures("dj_conn")
def test_artifact_detection_selection_raises_duplicate_selection_error():
    """A25: two master rows for one (params, source) raise
    ``DuplicateSelectionError``.

    ``insert_selection`` is find-existing-or-insert; if a prior direct
    bypass left two masters sharing the same ``artifact_detection_params_name`` and
    ``recording_id``, the find step sees >1 and must raise the integrity
    error rather than silently picking one. Plant the duplicate masters +
    matching ``RecordingSource`` parts via the FK-checks-off bypass (the
    only way to land the otherwise-impossible state).
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.artifact import ArtifactDetectionSelection
    from spyglass.spikesorting.v2.exceptions import DuplicateSelectionError

    params_name = "v2_a25_dup_params"
    rec_id = uuid.uuid4()
    aid1, aid2 = uuid.uuid4(), uuid.uuid4()
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        for aid in (aid1, aid2):
            ArtifactDetectionSelection.insert1(
                {
                    "artifact_detection_id": aid,
                    "artifact_detection_params_name": params_name,
                },
                allow_direct_insert=True,
            )
            ArtifactDetectionSelection.RecordingSource.insert1(
                {"artifact_detection_id": aid, "recording_id": rec_id},
                allow_direct_insert=True,
            )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        with pytest.raises(DuplicateSelectionError, match="master rows"):
            ArtifactDetectionSelection.insert_selection(
                {
                    "recording_id": rec_id,
                    "artifact_detection_params_name": params_name,
                }
            )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (
                ArtifactDetectionSelection.RecordingSource
                & {"recording_id": rec_id}
            ).delete_quick()
            for aid in (aid1, aid2):
                (
                    ArtifactDetectionSelection & {"artifact_detection_id": aid}
                ).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")


@pytest.mark.usefixtures("dj_conn")
def test_artifact_detection_selection_requires_artifact_detection_params_name():
    """A25: a key without ``artifact_detection_params_name`` raises naming the field.

    The source key alone is insufficient -- the master row needs the
    params FK. The guard names the missing field so the notebook user can
    fix the call in one step.
    """
    import uuid

    from spyglass.spikesorting.v2.artifact import ArtifactDetectionSelection

    with pytest.raises(ValueError, match="artifact_detection_params_name"):
        ArtifactDetectionSelection.insert_selection(
            {"recording_id": uuid.uuid4()}
        )


@pytest.mark.usefixtures("dj_conn")
def test_artifact_detection_selection_missing_lookup_row_diagnostic():
    """A25: a missing ``ArtifactDetectionParameters`` row raises a
    diagnostic ValueError, not an opaque FK IntegrityError.

    ``_ensure_lookup_row_exists`` pre-checks the params FK target and, when
    absent, names the Lookup table and the ``insert_default()`` path. The
    source key has no matching master yet (find returns empty), so control
    reaches the lookup pre-check.
    """
    import uuid

    from spyglass.spikesorting.v2.artifact import ArtifactDetectionSelection

    with pytest.raises(ValueError) as excinfo:
        ArtifactDetectionSelection.insert_selection(
            {
                "recording_id": uuid.uuid4(),
                "artifact_detection_params_name": "v2_a25_no_such_params_row",
            }
        )
    message = str(excinfo.value)
    assert "ArtifactDetectionParameters" in message
    assert "insert_default" in message


@pytest.mark.usefixtures("dj_conn")
def test_detect_artifacts_empty_sliver_filter_returns_empty():
    """A25: when ``min_length_s`` filters out every kept interval,
    ``_detect_artifacts`` returns ``np.empty((0, 2))``.

    Detection finds artifact frames (so the early all-valid return at the
    zero-frames branch is NOT taken), the complement is built, then a huge
    ``min_length_s`` drops every surviving valid sliver. The return must be
    a real (0, 2) float array -- the shape downstream ``_apply_artifact_
    mask`` expects -- not an empty 1-D array. Pairs with Phase 4 A1, where
    that empty array drives ``_apply_artifact_mask``'s raise.
    """
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    rec = _synthetic_artifact_recording()
    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=50.0,  # trips on the planted 300 µV burst
        zscore_threshold=None,
        proportion_above_threshold=0.5,
        removal_window_ms=1.0,
        join_window_ms=0.0,
        min_length_s=1e9,  # larger than the whole recording -> all filtered
    )

    out = ArtifactDetection._detect_artifacts(rec, validated)
    assert out.shape == (
        0,
        2,
    ), f"expected an empty (0, 2) array, got shape {out.shape}"
    # dtype matches the recording's timestamp dtype (float64); a 1-D
    # np.empty(0) or an int array would break the downstream mask walker.
    assert out.dtype == rec.get_times().dtype


@pytest.mark.usefixtures("dj_conn")
def test_artifact_detection_delete_tolerates_already_gone_interval_list():
    """A25: ``ArtifactDetection.delete`` does not raise when the paired
    IntervalList rows were already removed.

    The override deletes the master then cleans up the IntervalList rows it
    resolved; the ``len(rows) == 0`` guard skips a restriction whose rows
    are already gone. We pre-delete the IntervalList rows, then delete the
    detection, and assert no raise + the master is gone.

    Builds a DEDICATED selection on a planted recording (FK-off) and
    inserts a zero-row ``ArtifactDetection`` master directly, so the shared
    populated fixture's artifact row is untouched.
    """
    import uuid

    import datajoint as dj

    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.utils import (
        artifact_detection_interval_list_name,
    )

    nwb = "a25_delete_session_.nwb"
    rec_id = _plant_fake_recording(uuid.uuid4(), nwb, 30000.0)
    aid = uuid.uuid4()
    params_name = "v2_a25_delete_params"
    ilist_name = artifact_detection_interval_list_name(aid)
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        ArtifactDetectionSelection.insert1(
            {
                "artifact_detection_id": aid,
                "artifact_detection_params_name": params_name,
            },
            allow_direct_insert=True,
        )
        ArtifactDetectionSelection.RecordingSource.insert1(
            {"artifact_detection_id": aid, "recording_id": rec_id},
            allow_direct_insert=True,
        )
        ArtifactDetection.insert1(
            {"artifact_detection_id": aid}, allow_direct_insert=True
        )
        # The IntervalList row the delete would resolve and try to clean.
        IntervalList.insert1(
            {
                "nwb_file_name": nwb,
                "interval_list_name": ilist_name,
                "valid_times": np.empty((0, 2)),
            },
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        # Pre-delete the IntervalList row so delete() hits the already-gone
        # (len == 0) branch.
        (
            IntervalList
            & {
                "nwb_file_name": nwb,
                "interval_list_name": ilist_name,
            }
        ).delete_quick()

        # Must not raise even though the cleanup target is already gone.
        (ArtifactDetection & {"artifact_detection_id": aid}).delete(
            safemode=False
        )
        assert len(ArtifactDetection & {"artifact_detection_id": aid}) == 0
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (ArtifactDetection & {"artifact_detection_id": aid}).delete_quick()
            (
                ArtifactDetectionSelection.RecordingSource
                & {"artifact_detection_id": aid}
            ).delete_quick()
            (
                ArtifactDetectionSelection & {"artifact_detection_id": aid}
            ).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")
        _drop_fake_recording(rec_id)


# ===========================================================================
# A26: sorting.py untested branches.
#
# Concat-source ambiguity, peak-channel mismatch, analyzer rebuild, zero-unit
# get_sorting, unit-id int cast, make_compute Mode-A cleanup, the missing
# SorterParameters diagnostic, the MATLAB-sorter carve-out, the empty
# artifact-mask short-circuit, the singleton noise_levels broadcast, and the
# delete-safemode passthrough. Excludes A1/A8/A9/A10/A11 (already in the
# ledger). Light tests use raw bypass / cheap synthetic recordings; the few
# that need a real analyzer reuse the shared ``populated_sorting`` fixture.
# ===========================================================================


def _plant_concat_sorting_selection(sid):
    """Land a concat-source ``SortingSelection`` (no ``RecordingSource``).

    Mirrors the A11 precondition: ``insert_selection`` rejects concat today,
    so the only way to get a ``ConcatenatedRecordingSource`` part is the
    FK-checks-off bypass. The caller cleans up ``SortingSelection & {sid}``.
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    SorterParameters.insert_default()
    SortingSelection.insert1(
        {
            "sorting_id": sid,
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        },
        allow_direct_insert=True,
    )
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        SortingSelection.ConcatenatedRecordingSource.insert1(
            {"sorting_id": sid, "concat_recording_id": uuid.uuid4()},
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")


@pytest.mark.usefixtures("dj_conn")
def test_sorting_get_unit_brain_regions_concat_raises_without_anchor():
    """A26: a concat-backed sort raises ``ConcatBrainRegionAmbiguousError``
    unless ``allow_anchor_member=True``.

    A concat unit's peak channel maps to one Electrode row per member
    session, so per-session regions are ambiguous without cross-session
    matching (not in this build). The default refuses; the opt-in returns
    anchor-member regions.
    """
    import uuid

    from spyglass.spikesorting.v2.exceptions import (
        ConcatBrainRegionAmbiguousError,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    sid = uuid.uuid4()
    _plant_concat_sorting_selection(sid)
    try:
        with pytest.raises(ConcatBrainRegionAmbiguousError):
            Sorting().get_unit_brain_regions(
                {"sorting_id": sid}, allow_anchor_member=False
            )
    finally:
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


@pytest.mark.usefixtures("dj_conn")
def test_sorting_get_unit_brain_regions_concat_anchor_member_df(
    populated_sorting,
):
    """A26: ``allow_anchor_member=True`` returns the
    ``unit_brain_region_df`` DataFrame labeled ``anchor_member``.

    The return is the DataFrame (NOT a ``SourceResolution`` dataclass --
    that type lives inside ``make_fetch`` dispatch, not on this accessor).
    Non-vacuous: one ``Sorting.Unit`` row is planted (copied from the
    populated fixture so its Electrode FK resolves through BrainRegion), so
    the frame has a real row carrying ``region_resolution == 'anchor_member'``.
    """
    import datetime as dt
    import uuid

    import datajoint as dj
    import pandas as pd

    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2.utils import SourceResolution

    template_unit = (Sorting.Unit & populated_sorting).fetch(as_dict=True)[0]

    sid = uuid.uuid4()
    _plant_concat_sorting_selection(sid)
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        Sorting.insert1(
            {
                "sorting_id": sid,
                "analysis_file_name": "a26_concat_fake.nwb",
                "object_id": "a26-concat-object-id",
                "n_units": 1,
                "time_of_sort": dt.datetime(2020, 1, 1),
            },
            allow_direct_insert=True,
        )
        unit_row = {**template_unit, "sorting_id": sid, "unit_id": 0}
        Sorting.Unit.insert1(unit_row, allow_direct_insert=True)
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        result = Sorting().get_unit_brain_regions(
            {"sorting_id": sid}, allow_anchor_member=True
        )
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, SourceResolution)
        assert "region_resolution" in result.columns
        assert len(result) == 1, "anchor-member df should carry the one unit"
        assert (result["region_resolution"] == "anchor_member").all()
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (Sorting.Unit & {"sorting_id": sid}).delete_quick()
            (Sorting & {"sorting_id": sid}).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


@pytest.mark.usefixtures("dj_conn")
def test_sorting_selection_missing_sorter_params_diagnostic():
    """A26: a missing ``SorterParameters`` row raises a diagnostic ValueError.

    ``_ensure_lookup_row_exists`` translates the would-be FK IntegrityError
    into a message naming ``SorterParameters`` and ``insert_default()``. The
    recording source has no matching master, so control reaches the
    lookup pre-check.
    """
    import uuid

    from spyglass.spikesorting.v2.sorting import SortingSelection

    with pytest.raises(ValueError) as excinfo:
        SortingSelection.insert_selection(
            {
                "recording_id": uuid.uuid4(),
                "sorter": "mountainsort5",
                "sorter_params_name": "a26_no_such_sorter_row",
            }
        )
    message = str(excinfo.value)
    assert "SorterParameters" in message
    assert "insert_default" in message


@pytest.mark.usefixtures("dj_conn")
def test_sorting_selection_rejects_cross_recording_artifact_detection_source():
    """A26: a sort cannot link an artifact detected on another recording.

    Both recordings are in the same session, so the old interval lookup by
    ``nwb_file_name`` + artifact interval name could succeed and apply the
    wrong artifact mask. ``insert_selection`` must reject the mismatch before
    writing ``SortingSelection.ArtifactDetectionSource``.
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    rid_sort = _plant_fake_recording(
        uuid.uuid4(), "session_cross_artifact_.nwb", 30000.0
    )
    rid_artifact = _plant_fake_recording(
        uuid.uuid4(), "session_cross_artifact_.nwb", 30000.0
    )
    artifact_detection_id = uuid.uuid4()
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        ArtifactDetectionSelection.insert1(
            {
                "artifact_detection_id": artifact_detection_id,
                "artifact_detection_params_name": "v2_a26_cross_artifact_params",
            },
            allow_direct_insert=True,
        )
        ArtifactDetectionSelection.RecordingSource.insert1(
            {
                "artifact_detection_id": artifact_detection_id,
                "recording_id": rid_artifact,
            },
            allow_direct_insert=True,
        )
        ArtifactDetection.insert1(
            {"artifact_detection_id": artifact_detection_id},
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        SorterParameters.insert_default()
        with pytest.raises(ValueError, match="belongs to recording_id"):
            SortingSelection.insert_selection(
                {
                    "recording_id": rid_sort,
                    "sorter": "clusterless_thresholder",
                    "sorter_params_name": "default",
                    "artifact_detection_id": artifact_detection_id,
                }
            )
        assert (
            len(
                SortingSelection.ArtifactDetectionSource
                & {"artifact_detection_id": artifact_detection_id}
            )
            == 0
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            sort_keys = (
                SortingSelection.RecordingSource & {"recording_id": rid_sort}
            ).fetch("KEY", as_dict=True)
            if sort_keys:
                (
                    SortingSelection.ArtifactDetectionSource & sort_keys
                ).delete_quick()
                (
                    SortingSelection.RecordingSource
                    & {"recording_id": rid_sort}
                ).delete_quick()
                (SortingSelection & sort_keys).delete_quick()
            (
                ArtifactDetection
                & {"artifact_detection_id": artifact_detection_id}
            ).delete_quick()
            (
                ArtifactDetectionSelection.RecordingSource
                & {"artifact_detection_id": artifact_detection_id}
            ).delete_quick()
            (
                ArtifactDetectionSelection
                & {"artifact_detection_id": artifact_detection_id}
            ).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")
        _drop_fake_recording(rid_sort)
        _drop_fake_recording(rid_artifact)


def _stub_recording_with_2d_probe():
    """Recording stub whose probe is already planar (``ndim == 2``).

    ``_build_analyzer`` projects the probe to 2D (via ``recording.get_probe()``)
    before building the analyzer; this test stubs the analyzer factory, so the
    recording only needs a planar probe for the projection step to be skipped.
    """

    class _Probe:
        ndim = 2

    class _Recording:
        def get_probe(self):
            return _Probe()

    return _Recording()


def test_build_analyzer_cleans_partial_folder_when_create_fails(
    monkeypatch, tmp_path
):
    """A26: partial SortingAnalyzer folders are removed on build failure."""
    import uuid

    import spikeinterface as si

    from spyglass.spikesorting.v2 import utils as utils_mod
    from spyglass.spikesorting.v2.sorting import Sorting

    analyzer_folder = tmp_path / "partial.analyzer"
    monkeypatch.setattr(
        "spyglass.spikesorting.v2._analyzer_cache.analyzer_path",
        lambda sorting_id: analyzer_folder,
    )

    class _OneUnitSorting:
        def get_num_units(self):
            return 1

    def _raise_after_creating_folder(**kwargs):
        folder = kwargs["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "partial.bin").write_bytes(b"partial")
        raise RuntimeError("create_sorting_analyzer boom")

    monkeypatch.setattr(
        si, "create_sorting_analyzer", _raise_after_creating_folder
    )

    with pytest.raises(RuntimeError, match="create_sorting_analyzer boom"):
        Sorting._build_analyzer(
            sorting=_OneUnitSorting(),
            recording=_stub_recording_with_2d_probe(),
            key={"sorting_id": uuid.uuid4()},
            sorter_row={"job_kwargs": None},
            job_kwargs={},
        )
    assert not analyzer_folder.exists()


@pytest.mark.usefixtures("dj_conn")
def test_run_si_sorter_matlab_carveout(monkeypatch):
    """A26: a MATLAB sorter gets ``singularity_image=True`` and the
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
    """A26 (contrast): a non-MATLAB sorter keeps every param and gets no
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


@pytest.mark.usefixtures("dj_conn")
def test_clusterless_singleton_noise_levels_broadcast(monkeypatch):
    """A26: a singleton ``noise_levels=[1.0]`` is broadcast to length
    ``n_channels`` before ``detect_peaks``.

    SI's ``locally_exclusive`` indexes ``noise_levels[chan] *
    detect_threshold`` per channel, so a length-1 array would read only
    channel 0 (the v1 multi-channel clusterless bug). The runtime broadcasts
    the scalar to one value per channel. We capture the ``method_kwargs``
    that reach ``detect_peaks`` and assert the length + fill.
    """
    import numpy as _np
    import spikeinterface as si
    import spikeinterface.sortingcomponents.peak_detection as pd_mod

    from spyglass.spikesorting.v2.sorting import Sorting

    n_channels = 6
    rec = si.generate_recording(
        num_channels=n_channels,
        durations=[1.0],
        sampling_frequency=30_000.0,
    )
    # threshold_unit="uv" now scales the recording to uV (scale_to_uV),
    # which requires channel gains/offsets to be set.
    rec.set_channel_gains([1.0] * n_channels)
    rec.set_channel_offsets([0.0] * n_channels)

    captured = {}

    def _capture_detect(recording, *, method, method_kwargs, job_kwargs):
        captured["noise_levels"] = method_kwargs.get("noise_levels")
        # Return an empty detection so the wrapper builds an empty sorting.
        return _np.zeros(
            0, dtype=[("sample_index", "int64"), ("channel_index", "int64")]
        )

    monkeypatch.setattr(pd_mod, "detect_peaks", _capture_detect)
    # The runtime imports detect_peaks at call time from this module path.
    monkeypatch.setattr(
        "spikeinterface.sortingcomponents.peak_detection.detect_peaks",
        _capture_detect,
    )

    Sorting._run_clusterless_thresholder(
        sorter_params={"noise_levels": [1.0], "threshold_unit": "uv"},
        recording=rec,
        job_kwargs=None,
    )

    nl = captured["noise_levels"]
    assert nl is not None
    nl = _np.asarray(nl)
    assert nl.shape == (n_channels,), (
        f"singleton noise_levels must broadcast to n_channels={n_channels}, "
        f"got shape {nl.shape}"
    )
    assert _np.allclose(nl, 1.0)


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_full_coverage_short_circuits():
    """A26: when ``valid_times`` covers the whole recording, the mask is a
    no-op and the original recording object is returned unmodified.

    ``frame_ranges`` ends up empty (no artifact gaps), so ``_apply_artifact_
    mask`` returns ``recording`` itself -- no ``remove_artifacts`` wrapper.
    The contrast call (valid_times dropping the second half) returns a
    DIFFERENT object, proving the short-circuit is meaningful.
    """
    import numpy as _np
    import spikeinterface as si

    from spyglass.spikesorting.v2.sorting import Sorting

    rec = si.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=30_000.0
    )
    ts = rec.get_times()

    full = _np.asarray([[ts[0], ts[-1]]])
    out_full = Sorting._apply_artifact_mask(rec, full)
    assert out_full is rec, (
        "full-coverage valid_times must short-circuit to the original "
        "recording (no remove_artifacts wrapper)"
    )

    # Contrast: dropping the second half leaves an artifact gap -> a wrapped
    # recording, not the original object.
    partial = _np.asarray([[ts[0], ts[len(ts) // 2]]])
    out_partial = Sorting._apply_artifact_mask(rec, partial)
    assert out_partial is not rec


@pytest.mark.usefixtures("dj_conn")
def test_get_sorting_zero_unit_returns_empty_numpysorting(populated_sorting):
    """A26: a zero-unit sort returns an empty single-segment ``NumpySorting``.

    The zero-unit branch short-circuits before any NWB read and returns an
    empty ``NumpySorting`` at the recording's sampling frequency (reading an
    empty units table would otherwise fail). Built on the fixture's real
    recording (the branch still reads ``sampling_frequency`` from it) via a
    planted zero-unit Sorting row.
    """
    import datetime as dt
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    SorterParameters.insert_default()
    sid = uuid.uuid4()
    SortingSelection.insert1(
        {
            "sorting_id": sid,
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        },
        allow_direct_insert=True,
    )
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        SortingSelection.RecordingSource.insert1(
            {"sorting_id": sid, "recording_id": recording_id},
            allow_direct_insert=True,
        )
        Sorting.insert1(
            {
                "sorting_id": sid,
                "analysis_file_name": "a26_zero_fake.nwb",
                "object_id": "a26-zero-object-id",
                "n_units": 0,
                "time_of_sort": dt.datetime(2020, 1, 1),
            },
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        sorting = Sorting().get_sorting({"sorting_id": sid})
        assert len(sorting.unit_ids) == 0
        assert sorting.get_num_segments() == 1
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (Sorting & {"sorting_id": sid}).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


@pytest.mark.usefixtures("dj_conn")
def test_get_sorting_dataframe_casts_unit_ids_to_python_int(populated_sorting):
    """A26: ``get_sorting(as_dataframe=True)`` indexes by Python ``int``.

    v2 writes integer unit_ids; the DataFrame path casts ``int(uid)`` so the
    index matches v1's ``nwb.units.to_dataframe()`` shape. Non-vacuous: the
    SI sorting object (a ``NumpySorting``) yields numpy integers, so the cast
    genuinely changes the type.
    """
    from spyglass.spikesorting.v2.sorting import Sorting

    raw = Sorting().get_sorting(populated_sorting)
    assert len(raw.unit_ids) >= 1, "fixture sort must have units"
    # The SI sorting object's unit_ids are numpy integers -- the cast is not
    # a no-op.
    assert all(
        isinstance(uid, np.integer) for uid in raw.unit_ids
    ), "precondition: NumpySorting yields numpy unit_ids"

    df = Sorting().get_sorting(populated_sorting, as_dataframe=True)
    assert all(
        type(uid) is int for uid in df.index
    ), "DataFrame index unit_ids must be Python int, not numpy scalars"


@pytest.mark.usefixtures("dj_conn")
def test_populate_unit_part_peak_channel_not_in_sort_group(
    populated_sorting, monkeypatch
):
    """A26: a peak channel absent from the sort group raises RuntimeError.

    ``_populate_unit_part`` resolves each unit's peak channel to a
    ``SortGroupV2.SortGroupElectrode`` row; a channel id outside the group
    is a recording/sort-group mismatch and must fail loudly. We monkeypatch
    the extremum-channel lookup to return an out-of-group id and call the
    helper with the fixture's real analyzer.
    """
    from spikeinterface.core import template_tools

    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    nwb_file_name = (
        RecordingSelection & {"recording_id": recording_id}
    ).fetch1("nwb_file_name")
    # The transient analyzer folder _populate_unit_part loads (built by the
    # populate that created populated_sorting; resolved from sorting_id).
    analyzer_folder = analyzer_path(populated_sorting["sorting_id"])
    sorting = Sorting().get_sorting(populated_sorting)

    bad_channel = 10_000_000
    original_extremum_channel = template_tools.get_template_extremum_channel

    def _bad_peak_channels(analyzer, **kwargs):
        # ``_populate_unit_part`` resolves peak channels with ``outputs="id"``
        # (and a configured ``peak_sign``); ``get_template_extremum_amplitude``
        # calls this internally with ``peak_sign``/``mode`` but NOT
        # ``outputs="id"``. Both carry ``peak_sign``, so discriminate on
        # ``outputs`` -- only the unit-attribution call returns the planted
        # out-of-group channel; delegate the amplitude path to the real fn.
        if kwargs.get("outputs") != "id":
            return original_extremum_channel(analyzer, **kwargs)
        return {uid: bad_channel for uid in sorting.unit_ids}

    monkeypatch.setattr(
        template_tools, "get_template_extremum_channel", _bad_peak_channels
    )

    with pytest.raises(RuntimeError, match=str(bad_channel)):
        Sorting._populate_unit_part(
            sorting,
            recording_id,
            nwb_file_name,
            dict(populated_sorting),
            analyzer_folder,
        )


@pytest.mark.slow
@pytest.mark.usefixtures("dj_conn")
def test_rebuild_analyzer_folder_recreates_on_missing(populated_sorting):
    """A26: ``get_analyzer`` rebuilds the analyzer folder when it is gone.

    The analyzer folder is regeneratable scratch; ``get_analyzer`` rebuilds
    it in place (without deleting the DataJoint row) and the rebuilt analyzer
    must carry the same ``unit_ids`` as the original. Reuses the fixture
    (non-destructive -- the folder is restored by the rebuild).
    """
    import shutil

    from spyglass.spikesorting.v2.sorting import Sorting
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    folder = analyzer_path(populated_sorting["sorting_id"])
    original = Sorting().get_analyzer(populated_sorting)
    original_unit_ids = list(original.unit_ids)
    del original  # release the handle before rmtree

    shutil.rmtree(folder)
    assert not folder.exists()

    try:
        rebuilt = Sorting().get_analyzer(populated_sorting)
        assert folder.exists(), "rebuild did not recreate the analyzer folder"
        assert (
            list(rebuilt.unit_ids) == original_unit_ids
        ), "rebuilt analyzer unit_ids diverge from the original sort"
    finally:
        # The folder belongs to the shared package fixture; if anything above
        # failed after the rmtree, restore it so later fixture consumers do
        # not cascade-fail on a missing analyzer.
        if not folder.exists():
            try:
                Sorting().get_analyzer(populated_sorting)
            except Exception:
                pass


@pytest.mark.usefixtures("dj_conn")
def test_rebuild_analyzer_folder_concat_source_not_implemented():
    """A26: ``_rebuild_analyzer_folder`` raises ``NotImplementedError`` for a
    concat-source row.

    The rebuild path resolves the sorting's source and delegates to the concat
    stub, which is not implemented today. This pins the cross-method invariant
    (the rebuild correctly routes to the concat path) rather than a generic
    "method raises" -- the concat branch fires before any analyzer/folder I/O,
    so a planted concat ``SortingSelection`` is sufficient.
    """
    import uuid

    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    sid = uuid.uuid4()
    _plant_concat_sorting_selection(sid)
    try:
        with pytest.raises(NotImplementedError):
            Sorting()._rebuild_analyzer_folder({"sorting_id": sid})
    finally:
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


def _fresh_unit_producing_selection(populated_sorting):
    """Build a fresh MS5 ``SortingSelection`` on the fixture's
    recording+artifact (NOT yet populated); return its ``{"sorting_id"}``.

    The destructive delete / Mode-A tests need a sort that actually yields
    units so ``_build_analyzer`` writes an analyzer folder on disk -- the
    clusterless ``default`` row finds zero peaks on the MEArec smoke fixture
    (no folder, vacuous assertions). MS5 produces units.

    The selection is ARTIFACT-FREE (no ``artifact_detection_id``) so it is a DISTINCT
    row from the package fixture's artifact-backed MS5 sort -- otherwise
    ``insert_selection`` (find-existing-or-insert) would return the shared
    fixture's sort and a destructive test would delete shared state. The
    caller owns populate + teardown.
    """
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    SorterParameters.insert_default()
    return SortingSelection.insert_selection(
        {
            "recording_id": recording_id,
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
        }
    )


@pytest.mark.slow
@pytest.mark.parametrize("safemode_arg", [None, False])
@pytest.mark.usefixtures("dj_conn")
def test_sorting_delete_removes_analyzer_folder(
    populated_sorting, safemode_arg
):
    """A26: ``Sorting.delete`` removes the on-disk analyzer folder for both
    ``safemode=None`` (default) and ``safemode=False`` (explicit pass-through).

    The analyzer cache path is not DataJoint-tracked (computed from
    ``sorting_id``), so the override must rmtree it after the cascade.
    Populates a dedicated MS5 sort per parametrization so the shared fixture
    is untouched.
    """
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    sort_pk = _fresh_unit_producing_selection(populated_sorting)
    Sorting.populate(sort_pk, reserve_jobs=False)
    folder = analyzer_path(sort_pk["sorting_id"])
    try:
        assert folder.exists(), "fresh sort should have an analyzer folder"
        (Sorting & sort_pk).delete(safemode=safemode_arg)
        assert (
            not folder.exists()
        ), f"analyzer folder not removed on delete(safemode={safemode_arg})"
        assert len(Sorting & sort_pk) == 0
    finally:
        # Selection row is independent of the Sorting master; clean it up.
        (SortingSelection & sort_pk).delete(safemode=False)
        if folder.exists():
            import shutil

            shutil.rmtree(folder, ignore_errors=True)


@pytest.mark.slow
@pytest.mark.usefixtures("dj_conn")
def test_make_compute_mode_a_cleanup_on_write_failure(
    populated_sorting, monkeypatch
):
    """A26: a ``_write_units_nwb`` failure after ``_build_analyzer`` removes
    the analyzer folder and inserts no row (make_compute Mode-A cleanup).

    Once ``_build_analyzer`` has written the folder, any later failure in
    ``make_compute`` must rmtree it (DataJoint never calls ``make_insert``
    after the raise). We build a fresh unit-producing (MS5) selection with
    ``_write_units_nwb`` patched to raise, then assert the folder is gone and
    no Sorting row landed.
    """
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    sort_pk = _fresh_unit_producing_selection(populated_sorting)
    folder = analyzer_path(sort_pk["sorting_id"])

    def _boom_write(self, **kwargs):
        raise RuntimeError("units NWB write blew up")

    monkeypatch.setattr(Sorting, "_write_units_nwb", _boom_write)

    try:
        with pytest.raises(Exception, match="units NWB write blew up"):
            Sorting.populate(sort_pk, reserve_jobs=False)
        assert not folder.exists(), (
            "Mode-A cleanup did not remove the analyzer folder after the "
            "units-NWB write failed"
        )
        assert len(Sorting & sort_pk) == 0, "no row should be inserted"
    finally:
        (SortingSelection & sort_pk).delete(safemode=False)
        if folder.exists():
            import shutil

            shutil.rmtree(folder, ignore_errors=True)


# ===========================================================================
# A30: parity pins for intentional-justified divergences that lacked a test.
#
# Without these pins a future refactor "fixing" them as drift would silently
# regress to v1 behavior. All behavioral: schema validation, shipped default
# rows, table column shape, and the runtime defensive strip.
# ===========================================================================


def test_ms4_schema_freq_band_defaults():
    """A30: MS4 schema ships ``freq_min=600`` / ``freq_max=6000``.

    These match v1's tetrode preset; the docstring records the choice but no
    test pinned it. A drift back to SI's bare MS4 defaults (no band) would
    silently change the filtered band the sorter sees.
    """
    schema = MountainSort4Schema()
    assert schema.freq_min == 600.0
    assert schema.freq_max == 6000.0


def test_ms4_schema_detect_threshold_float_and_positive():
    """A30: MS4 ``detect_threshold`` is a positive float (int coerced, 0 rejected)."""
    import pydantic

    coerced = MountainSort4Schema(detect_threshold=3)
    assert coerced.detect_threshold == 3.0
    assert isinstance(coerced.detect_threshold, float)
    with pytest.raises(pydantic.ValidationError):
        MountainSort4Schema(detect_threshold=0)  # gt=0 floor


def test_clusterless_schema_peak_sign_accepts_documented_values():
    """A30: the clusterless schema accepts ``neg`` / ``pos`` / ``both`` and
    rejects an unknown peak_sign."""
    import pydantic

    from spyglass.spikesorting.v2._params.sorter import (
        ClusterlessThresholderSchema,
    )

    for sign in ("neg", "pos", "both"):
        assert ClusterlessThresholderSchema(peak_sign=sign).peak_sign == sign
    with pytest.raises(pydantic.ValidationError):
        ClusterlessThresholderSchema(peak_sign="unknown")


def test_clusterless_schema_rejects_stale_fields():
    """A30: the clusterless schema forbids v1's stale ``outputs`` /
    ``random_chunk_kwargs`` (extra='forbid')."""
    import pydantic

    from spyglass.spikesorting.v2._params.sorter import (
        ClusterlessThresholderSchema,
    )

    for stale in ({"outputs": "sorting"}, {"random_chunk_kwargs": {}}):
        with pytest.raises(pydantic.ValidationError):
            ClusterlessThresholderSchema(**stale)


def test_common_reference_params_operator_knob():
    """A30: ``CommonReferenceParams.operator`` accepts both documented values."""
    import pydantic

    from spyglass.spikesorting.v2._params.preprocessing import (
        CommonReferenceParams,
    )

    for op in ("median", "average"):
        assert CommonReferenceParams(operator=op).operator == op
    with pytest.raises(pydantic.ValidationError):
        CommonReferenceParams(operator="rms")


def test_curation_source_enum_members():
    """A30: ``CurationSource`` carries ``manual`` / ``analyzer_curation`` /
    ``figpack`` (the set ``insert_curation`` coerces against).

    Today only ``manual`` is exercised end-to-end; pinning the other two
    members guards against a refactor dropping them (which would make a valid
    curation_source raise at the insert boundary).
    """
    from spyglass.spikesorting.v2.utils import CurationSource

    for value in ("manual", "analyzer_curation", "figpack"):
        assert CurationSource(value).value == value
    with pytest.raises(ValueError):
        CurationSource("not_a_member")


@pytest.mark.usefixtures("dj_conn")
def test_clusterless_default_row_ships_noise_levels_one():
    """A30: the shipped ``clusterless_thresholder/default`` row carries
    ``noise_levels=[1.0]``.

    Regression guard for the 1,400x divergence bug: v1 read detect_threshold
    in raw uV (noise_levels=[1.0]); a drift to None would reinterpret it as a
    MAD multiplier.
    """
    from spyglass.spikesorting.v2.sorting import SorterParameters

    SorterParameters.insert_default()
    params = (
        SorterParameters
        & {"sorter": "clusterless_thresholder", "sorter_params_name": "default"}
    ).fetch1("params")
    assert params["noise_levels"] == [1.0]
    # The shipped uv row carries detect_threshold=100 (the production
    # clusterless threshold) via the schema default, paired with its explicit
    # threshold_unit='uv' (audit finding #7).
    assert params["detect_threshold"] == 100.0


def test_clusterless_schema_default_is_production_uv():
    """The clusterless schema's bare default is the production/real-data
    threshold (100 uV under the default 'uv' unit), and a microvolt-scale
    threshold explicitly left in MAD units is rejected (audit finding #7).
    The OLD default ``(detect_threshold=100, threshold_unit='mad')`` was a
    100x-MAD threshold that silently detected almost nothing.
    """
    import pydantic

    from spyglass.spikesorting.v2._params.sorter import (
        ClusterlessThresholderSchema,
    )

    # Bare default: the real-data 100 uV threshold ('uv' derives
    # noise_levels=[1.0] at runtime), self-consistent with detect_threshold.
    bare = ClusterlessThresholderSchema()
    assert bare.threshold_unit == "uv"
    assert bare.detect_threshold == 100.0

    # A microvolt-scale threshold explicitly left in MAD units (no
    # noise_levels override) is rejected with a helpful message.
    with pytest.raises(pydantic.ValidationError, match="MAD multiplier"):
        ClusterlessThresholderSchema(
            detect_threshold=100.0, threshold_unit="mad"
        )

    # A sane MAD multiplier (the simulation fixture's regime) is accepted.
    mad = ClusterlessThresholderSchema(
        detect_threshold=5.0, threshold_unit="mad"
    )
    assert mad.threshold_unit == "mad" and mad.detect_threshold == 5.0

    # An explicit noise_levels override bypasses the guard even in MAD mode
    # (the documented advanced-override path is deliberately untouched).
    override = ClusterlessThresholderSchema(
        detect_threshold=100.0, threshold_unit="mad", noise_levels=[2.0]
    )
    assert override.noise_levels == [2.0]


@pytest.mark.usefixtures("dj_conn")
def test_filtering_description_reflects_actual_steps():
    """The persisted ``ElectricalSeries.filtering`` provenance is built from
    the preprocessing steps that ACTUALLY ran, not the old hardcoded
    "Bandpass filter + common reference" that misdescribed the no_filter /
    reference_mode='none' artifact (audit finding #2).
    """
    from spyglass.spikesorting.v2._params.preprocessing import (
        BandpassFilterParams,
    )
    from spyglass.spikesorting.v2.recording import Recording

    bp = BandpassFilterParams(freq_min=300.0, freq_max=6000.0)
    no_ps = {"phase_shift": False}

    # No preprocessing at all -> must not claim a filter or a reference step.
    none = Recording._filtering_description(None, "none", no_ps)
    assert none == "none (raw, no preprocessing)"
    assert "bandpass" not in none.lower()
    assert "common reference" not in none.lower()

    # Bandpass only (reference_mode='none') -> no reference claim.
    bp_only = Recording._filtering_description(bp, "none", no_ps)
    assert "bandpass filter 300-6000 Hz" in bp_only
    assert "common reference" not in bp_only

    # Bandpass + common reference -> both steps named, in the RUNTIME APPLY
    # order (bandpass first, then reference -- the order is non-commutative on
    # the global-median branch, so the provenance must track the apply order).
    both = Recording._filtering_description(bp, "global_median", no_ps)
    assert "bandpass filter 300-6000 Hz" in both
    assert "common reference (global_median)" in both
    assert both.index("bandpass filter") < both.index(
        "common reference"
    ), "provenance must list bandpass before reference (the apply order)"

    # Phase-shift is named ONLY when the applied-step report says it ran, and
    # listed first; a requested-but-skipped phase-shift (report False) is not
    # claimed -- the provenance tracks what RAN, not what was requested.
    with_ps = Recording._filtering_description(
        bp, "global_median", {"phase_shift": True}
    )
    assert with_ps.startswith("phase-shift (ADC); bandpass filter 300-6000 Hz")
    assert "phase-shift" not in both


@pytest.mark.usefixtures("dj_conn")
def test_clusterless_runtime_rejects_bypassed_mad_footgun():
    """Defense-in-depth at SORT time: a clusterless params blob that bypassed
    the SorterParameters insert validator (written via ``update1`` or
    predating the validator) is still caught -- a microvolt-scale
    detect_threshold left in MAD units with no noise_levels raises rather
    than running a silent ~zero-detection sort (audit follow-up). The
    runtime consumes the fetched blob without re-validating the schema, so
    the guard must live in ``_run_clusterless_thresholder`` too.
    """
    import spikeinterface.core as sc

    from spyglass.spikesorting.v2.sorting import Sorting

    rec = sc.generate_recording(
        num_channels=4, durations=[0.5], sampling_frequency=30000.0
    )
    # The footgun combo: detect_threshold=100 in MAD units, no noise_levels.
    with pytest.raises(ValueError, match="MAD multiplier"):
        Sorting._run_clusterless_thresholder(
            sorter_params={"detect_threshold": 100.0, "threshold_unit": "mad"},
            recording=rec,
            job_kwargs=None,
        )


@pytest.mark.usefixtures("dj_conn")
def test_to_int_unit_id_raises_typed_error_on_non_integer():
    """A sorter unit_id that does not convert to int raises the typed
    ``NonIntegerUnitIDError`` (a ValueError subclass), naming the offending id
    and the remap guidance -- not a bare ``int()`` ValueError or a silent
    coercion (audit test-hardening #10).
    """
    from spyglass.spikesorting.v2.exceptions import NonIntegerUnitIDError
    from spyglass.spikesorting.v2.sorting import _to_int_unit_id

    # Convertible ids pass through.
    assert _to_int_unit_id(3) == 3
    assert _to_int_unit_id("5") == 5

    # A non-convertible label raises the TYPED error, naming the bad id.
    with pytest.raises(NonIntegerUnitIDError, match="noise_3"):
        _to_int_unit_id("noise_3")

    # It remains a ValueError subclass so existing ``except ValueError``
    # handlers still catch it.
    assert issubclass(NonIntegerUnitIDError, ValueError)


@pytest.mark.usefixtures("dj_conn")
def test_sorting_master_has_no_analyzer_folder_column():
    """Phase B: the ``Sorting`` master no longer declares an
    ``analyzer_folder`` column (the analyzer cache path is computed from
    ``sorting_id``, not persisted); ``n_units`` is still a column."""
    from spyglass.spikesorting.v2.sorting import Sorting

    attrs = Sorting.heading.attributes
    assert "analyzer_folder" not in attrs
    assert "n_units" in attrs


@pytest.mark.usefixtures("dj_conn")
def test_clusterless_runtime_strips_stale_fields(monkeypatch):
    """A30: the clusterless runtime defensively strips ``outputs`` /
    ``random_chunk_kwargs`` from a params row that slipped past the Pydantic
    gate (e.g. a raw ``dj.insert1``).

    ``detect_peaks`` rejects those keys; the runtime pops them before the call
    so a stale row does not crash the sort. We capture ``method_kwargs`` and
    assert the stale keys are gone.
    """
    import numpy as _np
    import spikeinterface as si
    import spikeinterface.sortingcomponents.peak_detection as pd_mod

    from spyglass.spikesorting.v2.sorting import Sorting

    rec = si.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=30_000.0
    )
    # threshold_unit="uv" requires the recording to carry channel gains
    # (it scales to uV); generate_recording has none. Unity gain/offset make
    # the uv scaling a no-op, so this test exercises field-stripping (its
    # actual purpose) rather than the gains precondition.
    rec.set_channel_gains(1.0)
    rec.set_channel_offsets(0.0)
    captured = {}

    def _capture_detect(recording, *, method, method_kwargs, job_kwargs):
        captured["method_kwargs"] = dict(method_kwargs)
        return _np.zeros(
            0, dtype=[("sample_index", "int64"), ("channel_index", "int64")]
        )

    monkeypatch.setattr(
        "spikeinterface.sortingcomponents.peak_detection.detect_peaks",
        _capture_detect,
    )
    monkeypatch.setattr(pd_mod, "detect_peaks", _capture_detect)

    # Stale fields that the Pydantic schema would forbid, planted as if a raw
    # insert bypassed validation.
    Sorting._run_clusterless_thresholder(
        sorter_params={
            "detect_threshold": 100.0,
            "threshold_unit": "uv",
            "outputs": "sorting",
            "random_chunk_kwargs": {"num_chunks_per_segment": 5},
        },
        recording=rec,
        job_kwargs=None,
    )
    mk = captured["method_kwargs"]
    assert "outputs" not in mk
    assert "random_chunk_kwargs" not in mk


@pytest.mark.slow
@pytest.mark.usefixtures("dj_conn")
def test_obs_intervals_recorded_windows_fallback(populated_sorting):
    """A30/A16: ``obs_intervals=None`` (no artifact pass) writes the
    recorded window(s); an artifact-backed sort writes the artifact
    IntervalList's valid_times.

    The Units NWB carries ``obs_intervals`` per unit for downstream
    firing-rate windows. When no artifact mask was applied the fallback is
    the recording's recorded window(s) -- split at wall-clock
    discontinuities, so a contiguous recording yields a single interval
    (asserted here) and a DISJOINT recording yields one per chunk (asserted
    in ``test_obs_intervals_no_artifact_respects_disjoint_gap``). When an
    artifact pass exists the obs_intervals come from its IntervalList (the
    make_fetch path), not the fallback. We populate an artifact-FREE MS5
    sort for the fallback and reuse the artifact-backed fixture for the
    IntervalList path.
    """
    import pynwb

    from spyglass.common import IntervalList
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2.utils import (
        artifact_detection_interval_list_name,
    )

    def _first_unit_obs_intervals(sort_pk):
        analysis_file_name = (Sorting & sort_pk).fetch1("analysis_file_name")
        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        with pynwb.NWBHDF5IO(
            path=abs_path, mode="r", load_namespaces=True
        ) as io:
            nwbf = io.read()
            return np.asarray(nwbf.units["obs_intervals"][0])

    # --- artifact-FREE sort: full-envelope fallback -----------------------
    free_pk = _fresh_unit_producing_selection(populated_sorting)
    try:
        Sorting.populate(free_pk, reserve_jobs=False)
        recording_id = SortingSelection.resolve_source(free_pk).key[
            "recording_id"
        ]
        rec = Recording().get_recording({"recording_id": recording_id})
        times = rec.get_times()
        obs = _first_unit_obs_intervals(free_pk)
        assert obs.shape == (1, 2), (
            "no-artifact sort over a CONTIGUOUS recording must observe one "
            "recorded interval (base intervals collapse to the envelope)"
        )
        assert abs(obs[0][0] - float(times[0])) < 1e-6
        assert abs(obs[0][1] - float(times[-1])) < 1e-6
    finally:
        (Sorting & free_pk).delete(safemode=False)
        (SortingSelection & free_pk).delete(safemode=False)

    # --- artifact-backed fixture: obs_intervals from the IntervalList -----
    artifact_detection_id = SortingSelection.resolve_artifact_detection(
        populated_sorting
    )
    assert (
        artifact_detection_id is not None
    ), "fixture sort should be artifact-backed"
    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    nwb_file_name = (
        RecordingSelection & {"recording_id": recording_id}
    ).fetch1("nwb_file_name")
    valid_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": artifact_detection_interval_list_name(
                artifact_detection_id
            ),
        }
    ).fetch1("valid_times")
    obs = _first_unit_obs_intervals(populated_sorting)
    np.testing.assert_allclose(obs, np.asarray(valid_times), rtol=0, atol=1e-6)


# ===========================================================================
# A31: session_group invariants (replaces the rejected NotImplementedError
# stub tests, per the project decision). Behavioral / schema-shape only.
# The A13/A14-dependent tests are LIVE here -- both merged in Phase 4.
# (Member<->LabTeam consistency is deliberately NOT tested: the part FKs to
# LabTeam and master independently, with no insert-time agreement check.)
# ===========================================================================


@pytest.mark.usefixtures("dj_conn")
def test_session_group_member_index_unique_within_master():
    """A31: two members sharing ``(group, member_index)`` collide.

    ``member_index`` is part of the ``Member`` primary key, so a second row
    with the same group + index raises a duplicate-key error regardless of the
    other fields. Built via the FK-checks-off bypass (the upstream Session /
    SortGroupV2 / IntervalList FKs are irrelevant to the PK uniqueness check).
    """
    import datajoint as dj

    from spyglass.spikesorting.v2.session_group import SessionGroup

    owner, group = "a31_team", "a31_group"
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        SessionGroup.insert1(
            {
                "session_group_owner": owner,
                "session_group_name": group,
                "description": "a31 uniqueness probe",
            },
            allow_direct_insert=True,
        )
        member = {
            "session_group_owner": owner,
            "session_group_name": group,
            "member_index": 0,
            "nwb_file_name": "a31_x_.nwb",
            "sort_group_id": 0,
            "interval_list_name": "raw data valid times",
            "team_name": owner,
        }
        SessionGroup.Member.insert1(member, allow_direct_insert=True)
        # Same (group, member_index), different downstream fields -> collision.
        dup = {**member, "sort_group_id": 1}
        with pytest.raises(dj.errors.DuplicateError):
            SessionGroup.Member.insert1(dup, allow_direct_insert=True)
    finally:
        try:
            (
                SessionGroup.Member
                & {
                    "session_group_owner": owner,
                    "session_group_name": group,
                }
            ).delete_quick()
            (
                SessionGroup
                & {
                    "session_group_owner": owner,
                    "session_group_name": group,
                }
            ).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")


@pytest.mark.usefixtures("dj_conn")
def test_concatenated_recording_has_total_duration_s_column():
    """A31: ``ConcatenatedRecording`` declares ``total_duration_s``.

    Semantic check via the heading (not a source-string match). Pairs with the
    Phase 7 CHANGELOG entry naming the column-name divergence
    (``total_duration_s`` vs ``Recording.duration_s``).
    """
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    assert "total_duration_s" in ConcatenatedRecording.heading.attributes


@pytest.mark.usefixtures("dj_conn")
def test_motion_correction_parameters_validates_params_blob():
    """A31: ``MotionCorrectionParameters.insert1`` Pydantic-validates ``params``.

    The table-level wiring (not just the standalone schema tests) must reject a
    bogus blob -- here an unknown ``preset`` against the ``MotionPreset``
    Literal.
    """
    import pydantic

    from spyglass.spikesorting.v2.session_group import (
        MotionCorrectionParameters,
    )

    with pytest.raises(pydantic.ValidationError):
        MotionCorrectionParameters().insert1(
            {
                "motion_correction_params_name": "a31_bogus_preset",
                "params": {"preset": "not_a_real_preset"},
                "params_schema_version": 1,
                "job_kwargs": None,
            }
        )


# A31's two remaining items -- the ``_assert_schema_version_matches`` drift
# raise (A14) and ``initialize_v2_defaults`` shipping the motion presets (A13)
# -- are already covered LIVE above by the Phase 4 tests
# ``test_motion_correction_parameters_rejects_schema_version_drift`` and
# ``test_initialize_v2_defaults_installs_motion_correction_presets``. They are
# not duplicated here.


@pytest.mark.usefixtures("dj_conn")
def test_sorting_computed_matches_make_insert_signature():
    """The tri-part dispatch unpacks ``SortingComputed`` POSITIONALLY into
    ``make_insert(key, *computed)``, so the NamedTuple field order is a wire
    contract. Pin that it matches ``make_insert``'s parameter order -- a
    reorder would silently mis-bind the several str-adjacent slots
    (analysis_file_name / units_object_id / recording_id / nwb_file_name)
    without a TypeError.
    """
    import inspect

    from spyglass.spikesorting.v2.sorting import Sorting, SortingComputed

    params = list(inspect.signature(Sorting.make_insert).parameters)
    assert params[0] == "self"
    assert params[1] == "key"
    assert tuple(params[2:]) == SortingComputed._fields


@pytest.mark.slow
@pytest.mark.integration
def test_changed_analyzer_root_causes_miss_and_rebuild(
    populated_sorting, restore_custom_config, tmp_path
):
    """Phase B headline behavior, end-to-end: relocating the analyzer root
    (``spikesorting_v2_analyzer_dir``) makes the previously-built folder a
    cache MISS, and ``get_analyzer`` rebuilds into the NEW root -- never a
    stale-row inconsistency. The two halves (config override + rebuild-on-
    miss) were only covered separately before.
    """
    import shutil

    import datajoint as dj
    import spikeinterface as si

    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2.sorting import Sorting

    sid = populated_sorting["sorting_id"]
    original = analyzer_path(sid)
    assert original.exists(), "populated sort should have an analyzer folder"

    new_root = tmp_path / "relocated_analyzers"
    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(new_root)
    relocated = analyzer_path(sid)
    assert relocated != original, "root override must change the resolved path"
    assert not relocated.exists(), "fresh root -> cache miss, not a stale row"
    try:
        analyzer = Sorting().get_analyzer(populated_sorting)
        assert isinstance(analyzer, si.SortingAnalyzer)
        assert relocated.exists(), "get_analyzer must rebuild into the NEW root"
        # The original folder is untouched: a relocation is a cache miss, not
        # a row inconsistency.
        assert original.exists()
    finally:
        shutil.rmtree(new_root, ignore_errors=True)
    # restore_custom_config resets spikesorting_v2_analyzer_dir on teardown so
    # the package-scoped populated_sorting fixture resolves to `original` again.
