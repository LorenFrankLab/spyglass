"""Tests for the ``_units_nwb`` units-NWB build / read helpers.

Covers the lazy abs-time merge (``build_lazy_merged_sorting``), the pure
spike-times dataframe builders, and the pynwb readback path. All of these
are DB-free: the compute operates on numpy / SpikeInterface objects and
the readback goes through pynwb against a temp NWB file -- no DataJoint.
"""

from __future__ import annotations

import numpy as np
import pytest


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


def test_numpysorting_from_sample_indices_roundtrip():
    from spyglass.spikesorting.v2._units_nwb import (
        numpysorting_from_sample_indices,
    )

    sorting = numpysorting_from_sample_indices(
        {5: np.array([1, 4, 9]), 8: np.array([], dtype=np.int64)}, _FS
    )
    assert sorted(int(u) for u in sorting.get_unit_ids()) == [5, 8]
    assert _unit_train(sorting, 5) == [1, 4, 9]
    assert _unit_train(sorting, 8) == []


def test_build_lazy_merged_sorting_from_samples_avoids_timestamp_mapping():
    """Merged preview keeps frames already stored in the Units NWB."""
    from spyglass.spikesorting.v2._units_nwb import (
        build_lazy_merged_sorting_from_samples,
    )

    sorting = build_lazy_merged_sorting_from_samples(
        {
            1: np.array([0.010]),
            2: np.array([0.0102]),  # within 0.4 ms, different unit -> dedup
            3: np.array([0.020]),
        },
        {
            1: np.array([10], dtype=np.int64),
            2: np.array([11], dtype=np.int64),
            3: np.array([20], dtype=np.int64),
        },
        units_to_merge=[[1, 2]],
        fs=_FS,
        delta_s=0.4e-3,
    )
    assert sorted(int(u) for u in sorting.get_unit_ids()) == [3, 4]
    assert _unit_train(sorting, 3) == [20]
    # The first contributor's frame is retained by the same stable sort / keep
    # mask as _dedup_merged_spike_times.
    assert _unit_train(sorting, 4) == [10]


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


def _write_units_nwb_with_samples(path, rows):
    """Write ``[(unit_id, spike_times, sample_indices), ...]``."""
    from datetime import datetime, timezone

    import pynwb

    from spyglass.spikesorting.v2._units_nwb import SPIKE_SAMPLE_INDEX_COLUMN

    nwbfile = pynwb.NWBFile(
        session_description="test",
        identifier="test-units-nwb-samples",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    if rows:
        nwbfile.add_unit_column(
            SPIKE_SAMPLE_INDEX_COLUMN,
            "sample frames",
            index=True,
        )
    for unit_id, spike_times, sample_indices in rows:
        nwbfile.add_unit(
            id=unit_id,
            spike_times=list(spike_times),
            spike_sample_index=np.asarray(sample_indices, dtype=np.int64),
        )
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


def test_read_units_spike_sample_indices_populated(tmp_path):
    from spyglass.spikesorting.v2._units_nwb import (
        read_units_spike_sample_indices,
    )

    p = tmp_path / "units_samples.nwb"
    _write_units_nwb_with_samples(
        p,
        [
            (4, [0.1, 0.2, 0.3], [10, 20, 30]),
            (9, [], []),
        ],
    )
    out = read_units_spike_sample_indices(str(p))
    assert set(out) == {4, 9}
    np.testing.assert_array_equal(out[4], [10, 20, 30])
    np.testing.assert_array_equal(out[9], [])


def test_read_units_spike_sample_indices_missing_column(tmp_path):
    from spyglass.spikesorting.v2._units_nwb import (
        read_units_spike_sample_indices,
    )

    p = tmp_path / "legacy_units.nwb"
    _write_units_nwb(p, [[0.1, 0.2]])
    assert read_units_spike_sample_indices(str(p)) is None


def test_read_units_abs_times_and_sample_indices_matches_single_readers(
    tmp_path,
):
    """The single-open combined reader returns exactly what the two
    single-column readers return -- with the sample column, without it, and for
    an empty Units table."""
    from spyglass.spikesorting.v2._units_nwb import (
        read_units_abs_spike_times,
        read_units_abs_times_and_sample_indices,
        read_units_spike_sample_indices,
    )

    # (a) with the sample_sample_index column
    p_full = tmp_path / "with_samples.nwb"
    _write_units_nwb_with_samples(
        p_full, [(4, [0.1, 0.2, 0.3], [10, 20, 30]), (9, [], [])]
    )
    abs_c, samp_c, _obs_c = read_units_abs_times_and_sample_indices(str(p_full))
    abs_s = read_units_abs_spike_times(str(p_full))
    samp_s = read_units_spike_sample_indices(str(p_full))
    assert set(abs_c) == set(abs_s)
    assert all(np.array_equal(abs_c[u], abs_s[u]) for u in abs_s)
    assert all(np.array_equal(samp_c[u], samp_s[u]) for u in samp_s)

    # (b) legacy file without the column -> sample_indices is None
    p_legacy = tmp_path / "legacy.nwb"
    _write_units_nwb(p_legacy, [[0.1, 0.2]])
    abs_l, samp_l, _obs_l = read_units_abs_times_and_sample_indices(
        str(p_legacy)
    )
    assert samp_l is None
    assert np.array_equal(abs_l[0], read_units_abs_spike_times(str(p_legacy))[0])

    # (c) empty Units table -> ({}, {}, {})
    p_empty = tmp_path / "empty.nwb"
    _write_units_nwb(p_empty, [])
    assert read_units_abs_times_and_sample_indices(str(p_empty)) == (
        {},
        {},
        {},
    )


def test_read_units_abs_times_and_sample_indices_filters_to_requested_units(
    tmp_path,
):
    """``unit_ids`` reads ONLY the requested units' trains, so a curation that
    keeps a subset of a large sort never materializes the discarded units.
    ``None`` reads every unit; a requested id absent from the table is skipped
    (the caller's kept-set is authoritative)."""
    from spyglass.spikesorting.v2._units_nwb import (
        read_units_abs_times_and_sample_indices,
    )

    p = tmp_path / "three_units.nwb"
    _write_units_nwb_with_samples(
        p,
        [
            (4, [0.1, 0.2], [10, 20]),
            (7, [0.3, 0.4, 0.5], [30, 40, 50]),
            (9, [1.5], [99]),
        ],
    )

    # Subset: only units 4 and 9 are read (unit 7 is never materialized).
    abs_sub, samp_sub, _obs_sub = read_units_abs_times_and_sample_indices(
        str(p), unit_ids={4, 9}
    )
    assert set(abs_sub) == {4, 9}
    assert set(samp_sub) == {4, 9}
    np.testing.assert_array_equal(samp_sub[4], [10, 20])
    np.testing.assert_array_equal(samp_sub[9], [99])

    # None reads every unit (unchanged default behavior).
    abs_all, _, _ = read_units_abs_times_and_sample_indices(
        str(p), unit_ids=None
    )
    assert set(abs_all) == {4, 7, 9}

    # A requested id absent from the table is skipped, not an error.
    abs_missing, _, _ = read_units_abs_times_and_sample_indices(
        str(p), unit_ids={4, 999}
    )
    assert set(abs_missing) == {4}


def test_curation_source_unit_ids_selects_kept_and_contributors():
    """The source units ``write_curated_units_nwb`` actually reads: with
    ``apply_merge=True`` a single-contributor kept unit reads its own train
    (by kept id) and a multi-contributor kept unit reads its contributors'
    (NOT the fresh merged head id) -- exactly mirroring the write body's access,
    so dropped units are never read. ``apply_merge=False`` writes every original
    unit 1:1, so it returns ``None`` (read all)."""
    from spyglass.spikesorting.v2._units_nwb import curation_source_unit_ids

    # 2 and 5 kept as-is; 11 is a fresh merged head of contributors 3 and 7.
    kept = {2: [2], 5: [5], 11: [3, 7]}
    assert curation_source_unit_ids(kept, apply_merge=True) == {2, 5, 3, 7}
    assert curation_source_unit_ids(kept, apply_merge=False) is None


class _FakeRecording:
    def __init__(self, times, fs=_FS, explicit=True):
        self.times = np.asarray(times, dtype=float)
        self.fs = fs
        self.explicit = explicit
        self.sample_calls = []
        self.time_slice_calls = []

    def get_sampling_frequency(self):
        return self.fs

    def get_time_info(self, segment_index=0):
        return {"time_vector": self.times if self.explicit else None}

    def get_start_time(self, segment_index=0):
        return float(self.times[0]) if self.times.size else 0.0

    def get_num_samples(self, segment_index=0):
        return int(self.times.size)

    def sample_index_to_time(self, sample_ind, segment_index=0):
        sample_ind = np.asarray(sample_ind, dtype=np.int64)
        self.sample_calls.append(sample_ind)
        return self.times[sample_ind]

    def get_times(self, segment_index=0, start_frame=None, end_frame=None):
        # SI 0.104.3's BaseRecording.get_times takes NO frame bounds -- a
        # frame-bounded call raises TypeError there. Model that so this double
        # cannot stand in for an API real SI lacks: a bounded call is a hard
        # error, and the bounds-less call is recorded so the tests can assert
        # production never reads the full vector this way (it uses
        # sample_index_to_time instead).
        if start_frame is not None or end_frame is not None:
            raise TypeError(
                "spikeinterface 0.104.3 BaseRecording.get_times() takes no "
                "start_frame/end_frame; production must use "
                "sample_index_to_time for bounded reads."
            )
        self.time_slice_calls.append((start_frame, end_frame))
        return self.times


def test_sample_indices_to_times_maps_only_requested_frames():
    """frame->time mapping goes through sample_index_to_time with ONLY the spike
    frames (one request per unit), never the full timeline."""
    from spyglass.spikesorting.v2._units_nwb import (
        _sample_indices_to_times_by_unit,
    )

    times = np.concatenate(
        [np.arange(1000, dtype=float) / _FS, 10.0 + np.arange(1000) / _FS]
    )
    rec = _FakeRecording(times, explicit=True)
    out = _sample_indices_to_times_by_unit(
        rec,
        {
            1: np.array([2, 1002], dtype=np.int64),
            2: np.array([1003], dtype=np.int64),
        },
    )
    np.testing.assert_allclose(out[1], [times[2], times[1002]])
    np.testing.assert_allclose(out[2], [times[1003]])
    # Exactly the requested spike frames are mapped (one call per unit), so a
    # long recording's full timeline is never materialized.
    assert [c.tolist() for c in rec.sample_calls] == [[2, 1002], [1003]]
    assert rec.time_slice_calls == []


def test_base_intervals_from_recording_detects_gaps():
    """Gap detection yields one interval per contiguous run from bounded
    timestamp chunks."""
    from spyglass.spikesorting.v2._units_nwb import (
        _base_intervals_from_recording,
    )

    times = np.concatenate(
        [np.arange(5, dtype=float) / _FS, 10.0 + np.arange(4) / _FS]
    )
    rec = _FakeRecording(times, explicit=True)
    assert _base_intervals_from_recording(rec, _FS) == [
        [0.0, 0.004],
        [10.0, 10.003],
    ]
    # One chunk (chunk_size == round(fs) == 1000) covers the 9 samples, mapped
    # via sample_index_to_time over the frame range -- NOT frame-bounded
    # get_times (which SI 0.104.3 does not support).
    assert len(rec.sample_calls) == 1
    assert rec.sample_calls[0].tolist() == list(range(9))
    assert rec.time_slice_calls == []


# ---------- staged-file cleanup on write failure ----------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_write_sorting_units_nwb_unlinks_staged_file_on_failure(
    populated_sorting, monkeypatch
):
    """If the units-NWB write fails after ``AnalysisNwbfile().create`` staged the
    file, the writer unlinks that orphan before propagating.

    Mirrors the recording writer's cleanup contract. Without it the partial,
    unregistered AnalysisNwbfile would leak on disk (the Sorting/curation
    staging cleanup only knows the analyzer folder / the registration step, not
    this file's name). The curated writer shares the identical wrapper. Fault is
    injected into the post-create body so the real ``create`` + cleanup path
    runs end to end.
    """
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import _units_nwb
    from spyglass.spikesorting.v2.sorting import Sorting

    nwb_file_name = Sorting.resolve_anchor_nwb_file_name(populated_sorting)

    staged = {}

    def _boom(*, analysis_file_name, **kwargs):
        staged["name"] = analysis_file_name
        # The file must exist on disk at this point (create staged it).
        assert Path(AnalysisNwbfile.get_abs_path(analysis_file_name)).exists()
        raise RuntimeError("simulated units-NWB write failure")

    monkeypatch.setattr(_units_nwb, "_write_sorting_units_nwb_body", _boom)

    with pytest.raises(RuntimeError, match="simulated units-NWB write failure"):
        _units_nwb.write_sorting_units_nwb(
            sorting=None, recording=None, nwb_file_name=nwb_file_name
        )

    # The staged orphan was unlinked by the writer's except block.
    assert "name" in staged
    assert not Path(
        AnalysisNwbfile.get_abs_path(staged["name"])
    ).exists()


@pytest.mark.slow
@pytest.mark.integration
def test_units_nwb_carries_per_unit_and_source_metadata(planted_two_unit_sort):
    """The sort-time Units NWB carries per-unit metadata matching Sorting.Unit
    plus a source-provenance scratch, so the file is interpretable standalone.

    Per-unit columns (peak_amplitude_uv / peak_electrode_id / n_spikes /
    brain_region) come from the SAME rows used for Sorting.Unit.insert (computed
    once), and a header records the sort source / sorter / params / recipe /
    versions / seed.
    """
    import pynwb

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.common.common_region import BrainRegion
    from spyglass.spikesorting.v2._nwb_provenance import (
        SORTING_PROVENANCE,
        read_provenance_values,
    )
    from spyglass.spikesorting.v2.recording import Electrode
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    sort = planted_two_unit_sort
    row = (Sorting & sort).fetch1()
    abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])

    expected = {
        int(r["unit_id"]): r
        for r in ((Sorting.Unit & sort) * Electrode * BrainRegion).fetch(
            "unit_id",
            "peak_amplitude_uv",
            "n_spikes",
            "electrode_id",
            "region_name",
            as_dict=True,
        )
    }
    assert expected, "planted sort must have units to compare"

    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        units = io.read().units.to_dataframe()

    for unit_id, exp in expected.items():
        got = units.loc[unit_id]
        # Sorting.Unit.peak_amplitude_uv is a single-precision DataJoint float;
        # compare with tolerance, not equality.
        assert float(got["peak_amplitude_uv"]) == pytest.approx(
            float(exp["peak_amplitude_uv"]), rel=1e-4
        )
        assert int(got["peak_electrode_id"]) == int(exp["electrode_id"])
        assert int(got["n_spikes"]) == int(exp["n_spikes"])
        assert got["brain_region"] == exp["region_name"]

    prov = read_provenance_values(abs_path, SORTING_PROVENANCE)
    recording_id = SortingSelection.resolve_source(sort).key["recording_id"]
    artifact_detection_id = SortingSelection.resolve_artifact_detection(sort)
    sel = (SortingSelection & sort).fetch1()
    assert prov["recording_id"] == str(recording_id)
    assert prov["concat_recording_id"] is None
    assert prov["sorter"] == "mountainsort5"
    # The named sorter recipe + execution backend are recorded, not just the
    # scientific params blob (execution_params can change the sorter output).
    assert prov["sorter_params_name"] == sel["sorter_params_name"]
    assert isinstance(prov["sorter_params"], dict)
    assert isinstance(prov["execution_params"], dict)
    assert prov["artifact_detection_id"] == str(artifact_detection_id)
    assert (
        prov["display_waveform_params_name"]
        == row["display_waveform_params_name"]
    )
    assert prov["effective_random_seed"] == int(row["effective_random_seed"])
    assert prov["spikeinterface_version"] == row["spikeinterface_version"]
    assert prov["sorter_version"] == row["sorter_version"]
