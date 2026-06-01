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


# ---------- A2: Franklab MS4 v1-name back-compat aliases -------------------


@pytest.mark.usefixtures("dj_conn")
def test_franklab_ms4_v1_alias_rows_present():
    """A2: v1's bare ``30KHz`` preset names still resolve on v2.

    v2 renamed the Franklab MS4 presets to lowercase-k + ``_ms4``
    (``v1/sorting.py:158,163`` -> ``franklab_*_30kHz_ms4``), silently
    breaking v1 code that looks the row up by its old name. The catalog
    ships back-compat alias rows carrying the IDENTICAL validated params
    blob, so both the new and old names resolve to the same parameters.
    """
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import SorterParameters

    catalog = {
        (row[0], row[1]): row for row in SorterParameters._DEFAULT_CONTENTS
    }
    pairs = [
        (
            "franklab_tetrode_hippocampus_30kHz_ms4",
            "franklab_tetrode_hippocampus_30KHz",
        ),
        ("franklab_probe_ctx_30kHz_ms4", "franklab_probe_ctx_30KHz"),
    ]
    for ms4_name, alias_name in pairs:
        ms4_row = catalog[("mountainsort4", ms4_name)]
        alias_row = catalog[("mountainsort4", alias_name)]
        # params blob (index 2) and schema_version (index 3) identical.
        assert alias_row[2] == ms4_row[2], (
            f"alias {alias_name!r} params blob diverged from {ms4_name!r}"
        )
        assert alias_row[3] == ms4_row[3]

    # When MS4 is installed the aliases actually land in the table, so a
    # v1-style lookup by the old name resolves. (On the CI SI-0.104 image
    # MS4 is not installed and insert_default skips every MS4 row -- the
    # catalog assertion above is the install-independent invariant.)
    if "mountainsort4" in sis.installed_sorters():
        SorterParameters().insert_default()
        for _, alias_name in pairs:
            assert (
                SorterParameters
                & {
                    "sorter": "mountainsort4",
                    "sorter_params_name": alias_name,
                }
            ), f"alias row {alias_name!r} did not insert with MS4 present"


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
    assert (
        SorterParameters
        & {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        }
    ), "clusterless default row must insert even with no SI sorter installed"

    # Every gated-out SI sorter stays absent: insert_default routed the
    # gating DECISION into the INSERT, it did not insert _DEFAULT_CONTENTS
    # wholesale.
    for sorter in clean_canaries:
        assert not (SorterParameters & {"sorter": sorter}), (
            f"insert_default inserted uninstalled SI sorter {sorter!r} "
            "despite installed_sorters()==[]; the install gate was "
            "bypassed on the insert path."
        )


# ---------- A5: params_schema_version must be supplied ---------------------


@pytest.mark.usefixtures("dj_conn")
def test_sorter_parameters_rejects_missing_schema_version():
    """A5: a custom row omitting ``params_schema_version`` is rejected.

    The column default is the sentinel 0 ("unspecified"). A clusterless
    row whose validated ``params`` carries ``schema_version=4`` but whose
    outer column silently defaulted to 0 (or the stale 1) would mis-tag
    the blob. ``insert1`` requires the caller to pass the version
    explicitly and the error names both the sorter and the expected
    version.
    """
    from spyglass.spikesorting.v2.sorting import SorterParameters

    base = {
        "sorter": "clusterless_thresholder",
        "params": {"detect_threshold": 100.0, "threshold_unit": "uv"},
        "job_kwargs": None,
    }

    # Omitted entirely -> raise naming the field, the sorter, and the
    # expected inner version (4 for clusterless).
    omitted = dict(base, sorter_params_name="audit_a5_missing")
    with pytest.raises(ValueError) as excinfo:
        SorterParameters().insert1(omitted, skip_duplicates=True)
    msg = str(excinfo.value)
    assert "params_schema_version" in msg
    assert "clusterless_thresholder" in msg
    assert "4" in msg

    # Explicit sentinel 0 is rejected the same way.
    sentinel = dict(
        base, sorter_params_name="audit_a5_zero", params_schema_version=0
    )
    with pytest.raises(ValueError, match="params_schema_version"):
        SorterParameters().insert1(sentinel, skip_duplicates=True)

    # The correct explicit version inserts cleanly.
    ok = dict(
        base, sorter_params_name="audit_a5_ok", params_schema_version=4
    )
    SorterParameters().insert1(ok, skip_duplicates=True)
    assert (
        SorterParameters
        & {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "audit_a5_ok",
        }
    )

    # A wrong (non-sentinel) version that disagrees with the blob still
    # trips the existing drift check.
    drift = dict(
        base, sorter_params_name="audit_a5_drift", params_schema_version=2
    )
    with pytest.raises(ValueError, match="schema_version"):
        SorterParameters().insert1(drift, skip_duplicates=True)


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
        "MS4 path leaked np.Inf globally; the try/finally restore "
        "regressed."
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
    monkeypatch.setattr(
        tempfile.TemporaryDirectory, "cleanup", _boom_cleanup
    )

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

    # A consistent row still inserts cleanly.
    ok_params = MotionCorrectionParamsSchema().model_dump()
    ok_row = {
        "motion_correction_params_name": "audit_a14_ok",
        "params": ok_params,
        "params_schema_version": 1,
        "job_kwargs": None,
    }
    MotionCorrectionParameters().insert1(ok_row, skip_duplicates=True)
    assert (
        MotionCorrectionParameters
        & {"motion_correction_params_name": "audit_a14_ok"}
    )


# ---------- A1: empty artifact valid_times must not silently zero ----------


@pytest.mark.usefixtures("dj_conn")
def test_apply_artifact_mask_empty_valid_times_raises():
    """A1: empty valid_times raises instead of silently zeroing all.

    When the artifact pass keeps zero seconds, the complement walker
    would mask the WHOLE recording to zeros and the sort would run over
    all-zeros, emitting a misleading "zero units" result. The fix raises
    ``EmptyArtifactValidTimesError`` naming the ``artifact_id`` and
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
            artifact_id="art-123",
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
    masked = Sorting._apply_artifact_mask(
        rec, np_mod.array([[0.1, 0.9]])
    )
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
        assert len(Sorting.key_source & {"sorting_id": sid}) == 0, (
            "concat-source selection leaked into Sorting.key_source"
        )
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

    ``analyzer_folder`` is NOT-null ``varchar(255)``, so a zero-unit row
    still stores its would-be path. ``get_analyzer`` checks ``n_units``
    FIRST and raises ``ZeroUnitAnalyzerError`` before computing /
    ``.exists()``-ing the path -- so a missing folder never surfaces as a
    confusing ``FileNotFoundError``. (The audit's "phantom path" framing
    was disproved on re-read; this pins the existing guard.)
    """
    import datetime as dt
    import os
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

    # A zero-unit Sorting row whose analyzer_folder holds a would-be path
    # that does NOT exist on disk. The AnalysisNwbfile FK is bypassed --
    # the zero-unit guard never reads the file or the folder.
    fake_folder = "/nonexistent/analyzer/zero_unit_a10_pin"
    assert not os.path.exists(fake_folder)
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        Sorting.insert1(
            {
                "sorting_id": sid,
                "analysis_file_name": "a10_fake.nwb",
                "object_id": "a10-fake-object-id",
                "analyzer_folder": fake_folder,
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
    n_required = int(np.ceil(validated.proportion_above_thresh * n_channels))

    if validated.amplitude_thresh_uV is not None:
        above_amp = absolute > validated.amplitude_thresh_uV
    else:
        above_amp = np.zeros_like(absolute, dtype=bool)
    if validated.zscore_thresh is not None:
        ch_mean = traces_uv.mean(axis=1, keepdims=True)
        ch_std = traces_uv.std(axis=1, keepdims=True) + 1e-12
        zscores = np.abs((traces_uv - ch_mean) / ch_std)
        above_z = zscores > validated.zscore_thresh
    else:
        above_z = np.zeros_like(absolute, dtype=bool)

    if (
        validated.amplitude_thresh_uV is not None
        and validated.zscore_thresh is not None
    ):
        channel_hit = above_amp | above_z
    elif validated.amplitude_thresh_uV is not None:
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
    "amplitude_thresh_uV,zscore_thresh",
    [
        (50.0, None),  # amplitude-only branch
        (None, 6.0),  # z-score-only branch
        (50.0, 6.0),  # OR-combined branch
    ],
)
def test_chunked_artifact_matches_in_memory_reference(
    dj_conn, amplitude_thresh_uV, zscore_thresh
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
        amplitude_thresh_uV=amplitude_thresh_uV,
        zscore_thresh=zscore_thresh,
        proportion_above_thresh=0.5,
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
    assert np.array_equal(single, reference), (
        "Single-chunk pass should reproduce the in-memory reference exactly."
    )


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
        amplitude_thresh_uV=50.0,
        zscore_thresh=None,
        proportion_above_thresh=0.5,
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

    monkeypatch.setattr(
        "spyglass.spikesorting.v2.artifact.ChunkRecordingExecutor",
        _SpyExecutor,
    )

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
    arr[1_000_000 : 1_000_300, :] = 300.0  # one planted burst
    arr.flush()
    rec = si.NumpyRecording(traces_list=[arr], sampling_frequency=fs)
    rec.set_channel_gains([1.0] * n_channels)

    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_thresh_uV=50.0,
        zscore_thresh=None,
        proportion_above_thresh=0.5,
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
