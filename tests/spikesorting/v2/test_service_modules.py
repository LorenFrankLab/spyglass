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
    "bad_channels",
    "_curation_transforms",
    "_enums",
    "_lookup_validation",
    "_nwb_metadata_helpers",
    "_pipeline_presets",
    "_recipe_catalog",
    "_recording_materialization",
    "_reference_resolution",
    "_selection_identity",
    "_selection_plan",
    "_signal_math",
    "_sort_group_planning",
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


# Shared frame grid for the lazy-merge cases: 100 samples at 1 kHz
# (0.000 .. 0.099 s). The gap case overrides ``timestamps``.
_FS = 1000.0
_TS = np.arange(100, dtype=float) / _FS


def _lazy_merge(abs_times, units_to_merge, timestamps=_TS):
    from spyglass.spikesorting.v2._units_nwb import build_lazy_merged_sorting

    return build_lazy_merged_sorting(
        abs_times,
        units_to_merge=units_to_merge,
        timestamps=timestamps,
        fs=_FS,
        delta_s=0.4e-3,
    )


def test_build_lazy_merged_sorting_assigns_fresh_id_and_keeps_non_merged():
    """Non-merged units pass through; the merge group gets ``max(id)+1``.

    A merge group whose contributors are farther apart than ``delta_s``
    keeps BOTH spikes; the non-merged unit's frames are the plain
    ``searchsorted`` mapping of its own abs times (identical to
    ``numpysorting_from_abs_times`` / ``get_sorting``).
    """
    merged = _lazy_merge(
        {
            0: _TS[[10, 20]],  # non-merged
            1: _TS[[30]],
            2: _TS[[31]],  # 1 ms from unit 1 -> > delta, both kept
        },
        [[1, 2]],
    )
    assert sorted(int(u) for u in merged.get_unit_ids()) == [0, 3]
    assert _unit_train(merged, 0) == [10, 20]
    # Fresh id 3 = max(0,1,2)+1; both contributor spikes survive (> delta).
    assert _unit_train(merged, 3) == [30, 31]


def test_build_lazy_merged_sorting_dedups_coincident_cross_unit_spikes():
    """Coincident spikes from DIFFERENT contributors collapse to one."""
    merged = _lazy_merge(
        {
            1: _TS[[30]],
            2: _TS[[30]],  # exactly coincident, different unit -> dedup
        },
        [[1, 2]],
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
    # Two 50-sample chunks separated by a 10 s wall-clock gap.
    gap_ts = np.concatenate(
        [
            np.arange(50, dtype=float) / _FS,
            10.0 + np.arange(50, dtype=float) / _FS,
        ]
    )
    merged = _lazy_merge(
        {
            1: gap_ts[[49]],  # last sample of chunk 1 (0.049 s)
            2: gap_ts[[50]],  # first sample of chunk 2 (10.000 s)
        },
        [[1, 2]],
        timestamps=gap_ts,
    )
    # Frame-adjacent but seconds apart -> both kept.
    assert _unit_train(merged, 3) == [49, 50]


def test_build_lazy_merged_sorting_fresh_ids_in_group_order():
    """Multiple groups get consecutive ``max(id)+1`` ids in input order."""
    merged = _lazy_merge(
        {5: _TS[[10]], 2: _TS[[12]], 8: _TS[[14]], 3: _TS[[16]]},
        [[5, 2], [8, 3]],
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
        amplitude_threshold_uv=1000.0,
        zscore_threshold=None,
        proportion_above_threshold=1.0,
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
        {"artifact_detection_id": art_id},
        valid_times,
        nwb_file_name="sess.nwb",
        per_member_nwb_files=(),
    )
    assert len(rows) == 1
    (row,) = rows
    assert row["nwb_file_name"] == "sess.nwb"
    assert row["interval_list_name"] == f"artifact_detection_{art_id}"
    assert row["pipeline"] == "spikesorting_artifact_detection_v2"
    np.testing.assert_array_equal(row["valid_times"], valid_times)


def test_build_artifact_interval_rows_one_row_per_member():
    """A shared-group source writes one row per distinct member nwb file."""
    from spyglass.spikesorting.v2._artifact_intervals import (
        build_artifact_interval_rows,
    )

    art_id = uuid.UUID("00000000-0000-0000-0000-0000000000bb")
    valid_times = np.array([[0.0, 1.0]])
    rows = build_artifact_interval_rows(
        {"artifact_detection_id": art_id},
        valid_times,
        nwb_file_name="a.nwb",
        per_member_nwb_files=("a.nwb", "b.nwb"),
    )
    assert [r["nwb_file_name"] for r in rows] == ["a.nwb", "b.nwb"]
    # Same interval name + valid_times across members.
    assert {r["interval_list_name"] for r in rows} == {
        f"artifact_detection_{art_id}"
    }
    for r in rows:
        np.testing.assert_array_equal(r["valid_times"], valid_times)


# --------------------------------------------------------------------------- #
# Validation-order pin (needs the artifact schema module -> DB session).
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("dj_conn")
def test_read_artifact_removed_intervals_missing_artifact_detection_id_raises_value_error():
    """Missing ``artifact_detection_id`` raises a clear ValueError before source-resolve.

    The guard runs BEFORE ``ArtifactDetectionSelection.resolve_source(key)``, so a key
    with no ``artifact_detection_id`` surfaces this targeted message rather than a
    source-resolution / ``SchemaBypassError`` from walking the source parts.
    """
    from spyglass.spikesorting.v2._artifact_intervals import (
        read_artifact_removed_intervals,
    )

    with pytest.raises(
        ValueError, match="must include 'artifact_detection_id'"
    ):
        read_artifact_removed_intervals({})


# --------------------------------------------------------------------------- #
# _signal_math merge dedup (the abs-time core build_lazy_merged_sorting uses)
# --------------------------------------------------------------------------- #


def test_dedup_merged_spike_times_drops_cross_unit_coincidences():
    """Cross-contributor spikes within delta collapse; far ones are kept."""
    from spyglass.spikesorting.v2._signal_math import (
        _dedup_merged_spike_times,
    )

    # unit A=[0.0], unit B=[0.0001] -> 0.1 ms apart, different units -> dedup.
    near = _dedup_merged_spike_times([[0.0], [0.0001]], delta_s=0.4e-3)
    assert near.tolist() == [0.0]
    # unit A=[0.0], unit B=[0.001] -> 1 ms apart -> both kept.
    far = _dedup_merged_spike_times([[0.0], [0.001]], delta_s=0.4e-3)
    assert far.tolist() == [0.0, 0.001]


def test_dedup_merged_spike_times_keeps_within_unit_close_pair():
    """A close pair from the SAME contributor is a real event, kept.

    The membership guard mirrors SI's ``(diff>delta)|(diff(membership)==0)``:
    a sub-delta pair is dropped only when the two spikes came from DIFFERENT
    contributors (a cross-unit double-detection), never within one unit.
    """
    from spyglass.spikesorting.v2._signal_math import (
        _dedup_merged_spike_times,
    )

    out = _dedup_merged_spike_times([[0.0, 0.0001]], delta_s=0.4e-3)
    assert out.tolist() == [0.0, 0.0001]


def test_dedup_merged_spike_times_empty():
    from spyglass.spikesorting.v2._signal_math import (
        _dedup_merged_spike_times,
    )

    assert _dedup_merged_spike_times([], delta_s=0.4e-3).tolist() == []


# --------------------------------------------------------------------------- #
# _sorting_compute contracts (pure precedence + input-validation guards)
# --------------------------------------------------------------------------- #


def test_clusterless_noise_levels_precedence():
    """Explicit noise_levels win; else uv->[1.0], mad->None."""
    from spyglass.spikesorting.v2._sorting_compute import (
        _clusterless_noise_levels,
    )

    assert _clusterless_noise_levels([2.0, 3.0], "uv") == [2.0, 3.0]
    assert _clusterless_noise_levels([2.0, 3.0], "mad") == [2.0, 3.0]
    assert _clusterless_noise_levels(None, "uv") == [1.0]
    assert _clusterless_noise_levels(None, "mad") is None


def test_apply_artifact_mask_rejects_malformed_valid_times():
    """The complement walker rejects inputs that would silently under-mask."""
    from spyglass.spikesorting.v2._sorting_compute import apply_artifact_mask
    from spyglass.spikesorting.v2.exceptions import (
        EmptyArtifactValidTimesError,
    )

    rec = _rec(np.zeros((100, 2), dtype="float32"))
    with pytest.raises(EmptyArtifactValidTimesError):
        apply_artifact_mask(rec, np.empty((0, 2)))
    with pytest.raises(ValueError):
        apply_artifact_mask(rec, np.array([0.0, 1.0, 2.0]))  # not (n, 2)
    with pytest.raises(ValueError):
        apply_artifact_mask(rec, np.array([[1.0, 0.0]]))  # end < start
    with pytest.raises(ValueError):
        apply_artifact_mask(
            rec, np.array([[0.0, 0.05], [0.02, 0.08]])
        )  # overlapping / unsorted


# --------------------------------------------------------------------------- #
# _recording_materialization contracts (pure)
# --------------------------------------------------------------------------- #


def test_truncation_tolerance_scales_with_interval_count():
    from spyglass.spikesorting.v2._recording_materialization import (
        truncation_tolerance,
    )

    fs = 30000.0
    assert truncation_tolerance(1, fs) == pytest.approx(2.5 / fs)
    assert truncation_tolerance(20, fs) == pytest.approx(21.5 / fs)


def test_save_expectation_sums_disjoint_intended_intervals():
    """Disjoint intended epochs: the expected duration is the sum of every
    epoch and the interval count is preserved."""
    from spyglass.spikesorting.v2._recording_materialization import (
        compute_recording_save_expectation,
    )

    intended = np.array([[0.0, 1.0], [2.0, 3.5]])
    sort = np.array([[0.0, 1.0], [2.0, 3.5]])
    exp = compute_recording_save_expectation(intended, sort, 0.0)

    assert exp.n_intended_intervals == 2
    assert exp.expected_saved_total == pytest.approx(2.5)  # 1.0 + 1.5
    assert exp.requested_saved_total == pytest.approx(2.5)
    assert exp.over_request == pytest.approx(0.0)


def test_save_expectation_drops_sub_min_segment_slivers():
    """A requested epoch shorter than min_segment_length is excluded from the
    request total (the intersect already dropped it from the intended set), so
    it is not flagged as over-request -- it was intentionally dropped."""
    from spyglass.spikesorting.v2._recording_materialization import (
        compute_recording_save_expectation,
    )

    # The intersect already dropped the 0.05 s sliver from the intended set.
    intended = np.array([[0.0, 1.0]])
    sort = np.array([[0.0, 1.0], [2.0, 2.05]])  # second epoch is a 0.05 s sliver
    exp = compute_recording_save_expectation(intended, sort, 0.1)

    assert exp.n_intended_intervals == 1
    assert exp.expected_saved_total == pytest.approx(1.0)
    assert exp.requested_saved_total == pytest.approx(1.0)  # sliver excluded
    assert exp.over_request == pytest.approx(0.0)


def test_save_expectation_flags_request_past_raw_coverage():
    """A sort interval running past the raw recording is clipped to raw in the
    intended set; over_request is the dropped past-coverage span (the warning
    trigger)."""
    from spyglass.spikesorting.v2._recording_materialization import (
        compute_recording_save_expectation,
    )

    intended = np.array([[0.0, 6.0]])  # intersect clipped [0, 10] to raw [0, 6]
    sort = np.array([[0.0, 10.0]])
    exp = compute_recording_save_expectation(intended, sort, 0.001)

    assert exp.n_intended_intervals == 1
    assert exp.expected_saved_total == pytest.approx(6.0)  # clipped to raw
    assert exp.requested_saved_total == pytest.approx(10.0)  # full request
    assert exp.over_request == pytest.approx(4.0)  # 10 - 6 past coverage


# --------------------------------------------------------------------------- #
# _sorting_compute.build_sorting_unit_rows contracts (pure)
# --------------------------------------------------------------------------- #


def test_build_sorting_unit_rows_constructs_rows_from_peak_metadata():
    """One ``Sorting.Unit`` row per unit, carrying the peak channel's Electrode
    FK fields, the peak amplitude (as float), and the precomputed spike count,
    merged onto the base key."""
    from spyglass.spikesorting.v2._sorting_compute import (
        build_sorting_unit_rows,
    )

    electrode_by_id = {
        10: {
            "nwb_file_name": "x.nwb",
            "electrode_group_name": "0",
            "electrode_id": 10,
            "ignored_extra_column": "dropped",
        },
        20: {
            "nwb_file_name": "x.nwb",
            "electrode_group_name": "1",
            "electrode_id": 20,
        },
    }
    rows = build_sorting_unit_rows(
        unit_ids=[0, 1],
        peak_channels={0: 10, 1: 20},
        peak_amplitudes={0: 50.0, 1: 30.5},
        n_spikes_by_unit={0: 100, 1: 40},
        electrode_by_id=electrode_by_id,
        key={"sorting_id": "s"},
        sort_group_id=0,
        nwb_file_name="x.nwb",
    )

    assert rows == [
        {
            "sorting_id": "s",
            "unit_id": 0,
            "nwb_file_name": "x.nwb",
            "electrode_group_name": "0",
            "electrode_id": 10,
            "peak_amplitude_uv": 50.0,
            "n_spikes": 100,
        },
        {
            "sorting_id": "s",
            "unit_id": 1,
            "nwb_file_name": "x.nwb",
            "electrode_group_name": "1",
            "electrode_id": 20,
            "peak_amplitude_uv": 30.5,
            "n_spikes": 40,
        },
    ]


def test_build_sorting_unit_rows_rejects_peak_channel_outside_sort_group():
    """A unit whose peak channel is not in the sort group's electrode map is a
    channel-id mismatch -- raise rather than build an invalid Electrode FK."""
    from spyglass.spikesorting.v2._sorting_compute import (
        build_sorting_unit_rows,
    )

    with pytest.raises(RuntimeError, match="not in sort group"):
        build_sorting_unit_rows(
            unit_ids=[0],
            peak_channels={0: 99},  # 99 not in electrode_by_id
            peak_amplitudes={0: 50.0},
            n_spikes_by_unit={0: 100},
            electrode_by_id={10: {"nwb_file_name": "x.nwb"}},
            key={"sorting_id": "s"},
            sort_group_id=0,
            nwb_file_name="x.nwb",
        )


def test_build_sorting_unit_rows_raises_typed_error_on_non_integer_unit_id():
    """A sorter unit id that does not convert to int raises the typed
    NonIntegerUnitIDError (not a bare ValueError)."""
    from spyglass.spikesorting.v2._sorting_compute import (
        build_sorting_unit_rows,
    )
    from spyglass.spikesorting.v2.exceptions import NonIntegerUnitIDError

    with pytest.raises(NonIntegerUnitIDError):
        build_sorting_unit_rows(
            unit_ids=["noise_3"],
            peak_channels={"noise_3": 10},
            peak_amplitudes={"noise_3": 50.0},
            n_spikes_by_unit={"noise_3": 100},
            electrode_by_id={10: {"nwb_file_name": "x.nwb"}},
            key={"sorting_id": "s"},
            sort_group_id=0,
            nwb_file_name="x.nwb",
        )


# --------------------------------------------------------------------------- #
# _analyzer_cache.classify_orphaned_analyzer_folders contracts (pure)
# --------------------------------------------------------------------------- #


def test_classify_orphans_reports_units_bearing_rows_with_missing_folder():
    """A units-bearing row whose computed analyzer folder is gone on disk is a
    DB-side orphan; a row whose folder still exists is not."""
    from spyglass.spikesorting.v2._analyzer_cache import (
        classify_orphaned_analyzer_folders,
    )

    result = classify_orphaned_analyzer_folders(
        units_bearing=[
            ("s1", "/cache/s1", False),  # folder gone -> DB-side orphan
            ("s2", "/cache/s2", True),  # folder present -> not an orphan
        ],
        referenced_paths={"/cache/s1", "/cache/s2"},
        disk_dir_paths=[],
    )

    assert result["db_side"] == [
        {"sorting_id": "s1", "computed_analyzer_path": "/cache/s1"}
    ]
    assert result["disk_side"] == []


def test_classify_orphans_reports_unreferenced_disk_folders():
    """An on-disk folder under the analyzer root that no row references is a
    disk-side orphan; a referenced folder is kept. Input order is preserved."""
    from spyglass.spikesorting.v2._analyzer_cache import (
        classify_orphaned_analyzer_folders,
    )

    result = classify_orphaned_analyzer_folders(
        units_bearing=[],
        referenced_paths={"/cache/s1"},
        disk_dir_paths=["/cache/s1", "/cache/leaked"],  # leaked not referenced
    )

    assert result["db_side"] == []
    assert result["disk_side"] == ["/cache/leaked"]


def test_classify_orphans_keeps_referenced_folders_and_present_rows():
    """A row with a present, referenced folder is in neither orphan class."""
    from spyglass.spikesorting.v2._analyzer_cache import (
        classify_orphaned_analyzer_folders,
    )

    result = classify_orphaned_analyzer_folders(
        units_bearing=[("s1", "/cache/s1", True)],
        referenced_paths={"/cache/s1"},
        disk_dir_paths=["/cache/s1"],
    )

    assert result == {"db_side": [], "disk_side": []}


def test_filtering_description_lists_only_steps_that_ran():
    from types import SimpleNamespace

    from spyglass.spikesorting.v2._recording_materialization import (
        filtering_description,
    )

    bp = SimpleNamespace(freq_min=300, freq_max=6000)
    no_ps = {"phase_shift": False}
    assert (
        filtering_description(None, "none", no_ps)
        == "none (raw, no preprocessing)"
    )
    assert (
        filtering_description(bp, "none", no_ps)
        == "bandpass filter 300-6000 Hz"
    )
    assert filtering_description(None, "global_median", no_ps) == (
        "common reference (global_median)"
    )
    # Bandpass first, then reference (the non-commutative apply order).
    assert filtering_description(bp, "global_median", no_ps) == (
        "bandpass filter 300-6000 Hz; common reference (global_median)"
    )

    # Phase-shift is listed ONLY when the report says it ran, and prepended
    # first (the apply order). A requested-but-skipped phase-shift (report
    # False) must NOT appear -- the report-driven honesty check.
    ran = {"phase_shift": True}
    assert filtering_description(bp, "global_median", ran) == (
        "phase-shift (ADC); bandpass filter 300-6000 Hz; "
        "common reference (global_median)"
    )
    assert "phase-shift" not in filtering_description(
        bp, "global_median", no_ps
    )


# --------------------------------------------------------------------------- #
# _units_nwb edge cases (dataframes pure; readback via pynwb, no DB)
# --------------------------------------------------------------------------- #


def test_empty_spike_times_dataframe_shape():
    from spyglass.spikesorting.v2._units_nwb import (
        empty_spike_times_dataframe,
    )

    df = empty_spike_times_dataframe()
    assert list(df.columns) == ["spike_times"]
    assert df.index.name == "unit_id"
    assert len(df) == 0


def test_abs_spike_times_dataframe_roundtrip():
    from spyglass.spikesorting.v2._units_nwb import abs_spike_times_dataframe

    df = abs_spike_times_dataframe(
        {0: np.array([1.0, 2.0]), 3: np.array([5.0])}
    )
    assert df.index.name == "unit_id"
    assert list(df.index) == [0, 3]
    np.testing.assert_array_equal(df.loc[0, "spike_times"], [1.0, 2.0])
    np.testing.assert_array_equal(df.loc[3, "spike_times"], [5.0])
    assert len(abs_spike_times_dataframe({})) == 0


def _write_units_nwb(path, unit_spike_times):
    """Write a minimal NWB; add a Units table only if spike trains given."""
    from datetime import datetime, timezone

    import pynwb

    nwbfile = pynwb.NWBFile(
        session_description="test",
        identifier="test-units-nwb",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    for st in unit_spike_times:
        nwbfile.add_unit(spike_times=list(st))
    with pynwb.NWBHDF5IO(path=str(path), mode="w") as io:
        io.write(nwbfile)


def test_read_units_abs_spike_times_populated(tmp_path):
    from spyglass.spikesorting.v2._units_nwb import (
        read_units_abs_spike_times,
    )

    p = tmp_path / "units.nwb"
    _write_units_nwb(p, [[0.1, 0.2, 0.3], [1.5]])
    out = read_units_abs_spike_times(str(p))
    assert set(out) == {0, 1}
    np.testing.assert_allclose(out[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(out[1], [1.5])


def test_read_units_abs_spike_times_empty(tmp_path):
    """No Units table -> {} (the zero-unit-sort edge case)."""
    from spyglass.spikesorting.v2._units_nwb import (
        read_units_abs_spike_times,
    )

    p = tmp_path / "no_units.nwb"
    _write_units_nwb(p, [])  # no add_unit -> nwbf.units is None
    assert read_units_abs_spike_times(str(p)) == {}
