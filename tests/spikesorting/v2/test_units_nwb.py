"""Tests for the ``_units_nwb`` units-NWB build / read helpers.

Covers the lazy abs-time merge (``build_lazy_merged_sorting``), the pure
spike-times dataframe builders, and the pynwb readback path. All of these
are DB-free: the compute operates on numpy / SpikeInterface objects and
the readback goes through pynwb against a temp NWB file -- no DataJoint.
"""

from __future__ import annotations

import numpy as np


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
        self.time_slice_calls.append((start_frame, end_frame))
        start = 0 if start_frame is None else int(start_frame)
        stop = self.times.size if end_frame is None else int(end_frame)
        return self.times[start:stop]


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
