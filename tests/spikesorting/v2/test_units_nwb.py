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
