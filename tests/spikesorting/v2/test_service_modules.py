"""Direct service-level tests for the DB-free spikesorting-v2 modules.

Phase C extracted the table-class internals into dependency-light service
modules. These tests exercise the extracted pure / IO functions DIRECTLY
(not through the DataJoint table classes) and assert the load-bearing
property the whole extraction rests on: the ``_*`` service modules import
without pulling in the DB layer (``spyglass.common`` or a v2 *schema*
module), so a cold import opens no DB connection.

Most tests here are genuinely DB-free (no ``dj_conn`` fixture): the pure
functions operate on numpy / SpikeInterface objects only. The single
exception is the ``read_artifact_removed_intervals`` validation-order pin,
which must import the ``artifact`` schema module and therefore needs the DB
session.

Covers the newest helpers specifically called out for direct coverage:
``_units_nwb.build_lazy_merged_sorting`` and the ``_artifact_intervals``
pure functions (``scan_artifact_frames`` / ``detect_artifacts`` /
``build_artifact_interval_rows``).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import uuid

import numpy as np
import pytest

# Service modules that MUST be importable without the DataJoint DB layer.
_DB_FREE_SERVICE_MODULES = [
    "_analyzer_cache",
    "_artifact_compute",
    "_artifact_intervals",
    "_curation_transforms",
    "_enums",
    "_recording_materialization",
    "_selection_identity",
    "_signal_math",
    "_sorting_compute",
    "_units_nwb",
]


# --------------------------------------------------------------------------- #
# Cold-import checks: DB-free modules open no DB connection at import.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("modname", _DB_FREE_SERVICE_MODULES)
def test_service_module_imports_without_db_layer(modname):
    """Each ``_*`` service module cold-imports with no DB-layer dependency.

    Importing a v2 *schema* module (artifact / sorting / recording /
    curation) activates ``dj.schema(...)`` and the ``from spyglass.common
    import ...`` side effect, both of which open a DB connection AT IMPORT.
    A service module must do neither, so a fresh interpreter (the view a
    spawned ``n_jobs>1`` worker or a DB-isolated compute node gets) can
    import it and run its pure / IO logic without a database. Cold-import
    in a subprocess and assert it pulled in neither ``spyglass.common`` nor
    any v2 schema module. Hermetic -- no DB, no container.
    """
    probe = textwrap.dedent(f"""
        import sys

        import spyglass.spikesorting.v2.{modname}  # noqa: F401

        schema_mods = {{
            "spyglass.spikesorting.v2.artifact",
            "spyglass.spikesorting.v2.sorting",
            "spyglass.spikesorting.v2.recording",
            "spyglass.spikesorting.v2.curation",
        }}
        leaked = sorted(
            m
            for m in sys.modules
            if m.startswith("spyglass.common") or m in schema_mods
        )
        assert not leaked, "cold-import pulled in DB-layer modules: " + repr(
            leaked
        )
        """)
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"{modname} must import without a DB dependency\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


# --------------------------------------------------------------------------- #
# _units_nwb.build_lazy_merged_sorting (pure compute; no DB / no NWB IO)
# --------------------------------------------------------------------------- #


def _unit_train(sorting, unit_id):
    return sorting.get_unit_spike_train(unit_id=unit_id).tolist()


def test_build_lazy_merged_sorting_assigns_fresh_id_and_keeps_non_merged():
    """Non-merged units pass through; the merge group gets ``max(id)+1``.

    A merge group whose contributors are farther apart than ``delta_s``
    keeps BOTH spikes; the non-merged unit's frames are the plain
    ``searchsorted`` mapping of its own abs times (identical to
    ``numpysorting_from_abs_times`` / ``get_sorting``).
    """
    from spyglass.spikesorting.v2._units_nwb import build_lazy_merged_sorting

    fs = 1000.0
    timestamps = np.arange(100, dtype=float) / fs  # 0.000 .. 0.099 s
    abs_times = {
        0: timestamps[[10, 20]],  # non-merged
        1: timestamps[[30]],
        2: timestamps[[31]],  # 1 ms from unit 1 -> > delta, both kept
    }
    merged = build_lazy_merged_sorting(
        abs_times,
        units_to_merge=[[1, 2]],
        timestamps=timestamps,
        fs=fs,
        delta_s=0.4e-3,
    )

    assert sorted(int(u) for u in merged.get_unit_ids()) == [0, 3]
    assert _unit_train(merged, 0) == [10, 20]
    # Fresh id 3 = max(0,1,2)+1; both contributor spikes survive (> delta).
    assert _unit_train(merged, 3) == [30, 31]


def test_build_lazy_merged_sorting_dedups_coincident_cross_unit_spikes():
    """Coincident spikes from DIFFERENT contributors collapse to one."""
    from spyglass.spikesorting.v2._units_nwb import build_lazy_merged_sorting

    fs = 1000.0
    timestamps = np.arange(100, dtype=float) / fs
    abs_times = {
        1: timestamps[[30]],
        2: timestamps[[30]],  # exactly coincident, different unit -> dedup
    }
    merged = build_lazy_merged_sorting(
        abs_times,
        units_to_merge=[[1, 2]],
        timestamps=timestamps,
        fs=fs,
        delta_s=0.4e-3,
    )
    assert sorted(int(u) for u in merged.get_unit_ids()) == [3]
    assert _unit_train(merged, 3) == [30]


def test_build_lazy_merged_sorting_absolute_time_dedup_respects_gaps():
    """Cross-gap, frame-adjacent spikes are NOT deduped (absolute time).

    On a disjoint recording the units-NWB frames are contiguous across an
    excluded wall-clock gap. Two spikes one frame apart (49, 50) but ~10 s
    apart in real time must BOTH survive -- a frame-space dedup would
    wrongly drop one. This pins the gap-correctness of the abs-time merge.
    """
    from spyglass.spikesorting.v2._units_nwb import build_lazy_merged_sorting

    fs = 1000.0
    # Two 50-sample chunks separated by a 10 s wall-clock gap.
    timestamps = np.concatenate(
        [
            np.arange(50, dtype=float) / fs,
            10.0 + np.arange(50, dtype=float) / fs,
        ]
    )
    abs_times = {
        1: timestamps[[49]],  # last sample of chunk 1 (0.049 s)
        2: timestamps[[50]],  # first sample of chunk 2 (10.000 s)
    }
    merged = build_lazy_merged_sorting(
        abs_times,
        units_to_merge=[[1, 2]],
        timestamps=timestamps,
        fs=fs,
        delta_s=0.4e-3,
    )
    # Frame-adjacent but seconds apart -> both kept.
    assert _unit_train(merged, 3) == [49, 50]


def test_build_lazy_merged_sorting_fresh_ids_in_group_order():
    """Multiple groups get consecutive ``max(id)+1`` ids in input order."""
    from spyglass.spikesorting.v2._units_nwb import build_lazy_merged_sorting

    fs = 1000.0
    timestamps = np.arange(100, dtype=float) / fs
    abs_times = {
        5: timestamps[[10]],
        2: timestamps[[12]],
        8: timestamps[[14]],
        3: timestamps[[16]],
    }
    merged = build_lazy_merged_sorting(
        abs_times,
        units_to_merge=[[5, 2], [8, 3]],
        timestamps=timestamps,
        fs=fs,
        delta_s=0.4e-3,
    )
    # next_id = max(5,2,8,3)+1 = 9; groups assigned in units_to_merge order.
    assert sorted(int(u) for u in merged.get_unit_ids()) == [9, 10]
    assert _unit_train(merged, 9) == [10, 12]
    assert _unit_train(merged, 10) == [14, 16]


# --------------------------------------------------------------------------- #
# _artifact_intervals pure functions
# --------------------------------------------------------------------------- #


def _rec(traces, fs=1000.0, gain=1.0, times=None):
    import spikeinterface as si

    n_ch = traces.shape[1]
    rec = si.NumpyRecording(
        traces_list=[traces.astype("float32")], sampling_frequency=fs
    )
    rec.set_channel_gains([gain] * n_ch)
    rec.set_channel_offsets([0.0] * n_ch)
    if times is not None:
        rec.set_times(np.asarray(times, dtype=float))
    return rec


def _artifact_params(**overrides):
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )

    # removal_window_ms / min_length_s are gt=0 in the schema; keep them
    # small so the 1-frame transient widens by ~1 frame and the ~0.049 s
    # valid intervals on a 0.1 s recording survive the min-length filter.
    base = dict(
        detect=True,
        amplitude_thresh_uV=1000.0,
        zscore_thresh=None,
        proportion_above_thresh=1.0,
        removal_window_ms=0.5,
        join_window_ms=0.0,
        min_length_s=0.001,
    )
    base.update(overrides)
    return ArtifactDetectionParamsSchema.model_validate(base)


def test_detect_artifacts_detect_false_returns_full_window():
    """``detect=False`` returns the recorded window untouched (no DB)."""
    from spyglass.spikesorting.v2._artifact_intervals import detect_artifacts

    traces = np.zeros((100, 2), dtype="float32")
    rec = _rec(traces)
    vt = detect_artifacts(rec, _artifact_params(detect=False))
    assert vt.shape == (1, 2)
    assert vt[0, 0] == pytest.approx(0.0)
    # One interval spanning the recording.
    assert vt[0, 1] >= 0.099 - 1e-9


def test_scan_artifact_frames_flags_transient_only():
    """``scan_artifact_frames`` flags the artifact frames and nothing else."""
    from spyglass.spikesorting.v2._artifact_intervals import (
        scan_artifact_frames,
    )

    traces = np.zeros((100, 2), dtype="float32")
    traces[50:52, :] = 5000.0  # large transient on every channel
    rec = _rec(traces)
    flagged = scan_artifact_frames(rec, _artifact_params())
    assert set(int(f) for f in flagged) == {50, 51}


def test_detect_artifacts_excludes_transient_window():
    """``detect_artifacts`` carves the artifact out of the valid times."""
    from spyglass.spikesorting.v2._artifact_intervals import detect_artifacts

    traces = np.zeros((100, 2), dtype="float32")
    traces[50:52, :] = 5000.0
    rec = _rec(traces)
    vt = detect_artifacts(rec, _artifact_params())
    # The artifact instant (~0.050 s) is in no returned valid interval, but
    # quiet instants near the ends are.
    art_t = 50 / 1000.0
    assert not any(lo <= art_t < hi for lo, hi in vt)
    assert any(lo <= 0.0 <= hi for lo, hi in vt)
    assert any(lo <= 0.090 <= hi for lo, hi in vt)


def test_build_artifact_interval_rows_single_recording_fallback():
    """Empty ``per_member`` -> one row keyed by the master nwb_file_name."""
    from spyglass.spikesorting.v2._artifact_intervals import (
        build_artifact_interval_rows,
    )

    art_id = uuid.UUID("00000000-0000-0000-0000-0000000000aa")
    valid_times = np.array([[0.0, 1.0], [2.0, 3.0]])
    rows = build_artifact_interval_rows(
        {"artifact_id": art_id},
        valid_times,
        nwb_file_name="sess.nwb",
        per_member_nwb_files=(),
    )
    assert len(rows) == 1
    (row,) = rows
    assert row["nwb_file_name"] == "sess.nwb"
    assert row["interval_list_name"] == f"artifact_{art_id}"
    assert row["pipeline"] == "spikesorting_artifact_v2"
    np.testing.assert_array_equal(row["valid_times"], valid_times)


def test_build_artifact_interval_rows_one_row_per_member():
    """A shared-group source writes one row per distinct member nwb file."""
    from spyglass.spikesorting.v2._artifact_intervals import (
        build_artifact_interval_rows,
    )

    art_id = uuid.UUID("00000000-0000-0000-0000-0000000000bb")
    valid_times = np.array([[0.0, 1.0]])
    rows = build_artifact_interval_rows(
        {"artifact_id": art_id},
        valid_times,
        nwb_file_name="a.nwb",
        per_member_nwb_files=("a.nwb", "b.nwb"),
    )
    assert [r["nwb_file_name"] for r in rows] == ["a.nwb", "b.nwb"]
    # Same interval name + valid_times across members.
    assert {r["interval_list_name"] for r in rows} == {f"artifact_{art_id}"}
    for r in rows:
        np.testing.assert_array_equal(r["valid_times"], valid_times)


# --------------------------------------------------------------------------- #
# Validation-order pin (needs the artifact schema module -> DB session).
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("dj_conn")
def test_read_artifact_removed_intervals_missing_artifact_id_raises_value_error():
    """Missing ``artifact_id`` raises a clear ValueError before source-resolve.

    The guard runs BEFORE ``ArtifactSelection.resolve_source(key)``, so a key
    with no ``artifact_id`` surfaces this targeted message rather than a
    source-resolution / ``SchemaBypassError`` from walking the source parts.
    """
    from spyglass.spikesorting.v2._artifact_intervals import (
        read_artifact_removed_intervals,
    )

    with pytest.raises(ValueError, match="must include 'artifact_id'"):
        read_artifact_removed_intervals({})
