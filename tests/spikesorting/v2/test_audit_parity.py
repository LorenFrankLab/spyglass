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
