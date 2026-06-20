"""Unit tests for the pure SortGroupV2 planning helpers.

``SortGroupV2.set_group_by_shank`` /
``set_group_by_electrode_table_column`` are thin orchestrators over three
pure (DataJoint-free) helpers extracted for testability:

* ``_plan_sort_groups_by_shank`` / ``_plan_sort_groups_by_column`` --
  enumerate groupings and resolve each group's reference, returning
  ``(proposed, skipped)``.
* ``_build_sort_group_rows`` -- materialize resolved plans into the
  master + part row lists the transaction inserts.

These tests drive those helpers directly with fabricated NumPy structured
arrays (mimicking ``Electrode().fetch()``) -- no database fixture, no
``populate``. The module import still activates the schema, so the
``dj_conn`` fixture only needs to be live; the planners themselves issue
no queries.
"""

from __future__ import annotations

import numpy as np
import pytest

from spyglass.spikesorting.v2._sort_group_planning import (
    _SortGroupPlan,
    _build_sort_group_rows,
    _plan_sort_groups_by_column,
    _plan_sort_groups_by_shank,
)


def _electrodes(rows: list[dict]) -> np.ndarray:
    """Build an ``Electrode``-like structured array from ``rows``.

    Each dict supplies ``electrode_id``, ``electrode_group_name``,
    ``probe_shank`` and ``original_reference_electrode`` (the four columns
    the planners read). ``-1`` is the v1 "no reference" sentinel.
    """
    dtype = [
        ("electrode_id", "i8"),
        ("electrode_group_name", "U16"),
        ("probe_shank", "i8"),
        ("original_reference_electrode", "i8"),
    ]
    arr = np.zeros(len(rows), dtype=dtype)
    for i, row in enumerate(rows):
        for key, value in row.items():
            arr[key][i] = value
    return arr


def _shank(group, shank, eid, ref=-1):
    return {
        "electrode_id": eid,
        "electrode_group_name": group,
        "probe_shank": shank,
        "original_reference_electrode": ref,
    }


# ---------- _build_sort_group_rows -----------------------------------------


def test_build_sort_group_rows_orders_master_then_parts():
    """One master row per plan, then its part rows, in plan order."""
    plans = [
        _SortGroupPlan(0, "none", None, [("0", 1), ("0", 2)]),
        _SortGroupPlan(1, "specific", 7, [("1", 3)]),
    ]
    master, part = _build_sort_group_rows("sess_.nwb", plans)
    assert master == [
        {
            "nwb_file_name": "sess_.nwb",
            "sort_group_id": 0,
            "reference_mode": "none",
            "reference_electrode_id": None,
        },
        {
            "nwb_file_name": "sess_.nwb",
            "sort_group_id": 1,
            "reference_mode": "specific",
            "reference_electrode_id": 7,
        },
    ]
    assert part == [
        {
            "nwb_file_name": "sess_.nwb",
            "sort_group_id": 0,
            "electrode_group_name": "0",
            "electrode_id": 1,
        },
        {
            "nwb_file_name": "sess_.nwb",
            "sort_group_id": 0,
            "electrode_group_name": "0",
            "electrode_id": 2,
        },
        {
            "nwb_file_name": "sess_.nwb",
            "sort_group_id": 1,
            "electrode_group_name": "1",
            "electrode_id": 3,
        },
    ]


def test_build_sort_group_rows_coerces_electrode_id_to_int():
    """NumPy integer electrode ids become plain Python ints in part rows."""
    plans = [_SortGroupPlan(0, "none", None, [("0", np.int64(5))])]
    _, part = _build_sort_group_rows("s.nwb", plans)
    assert part[0]["electrode_id"] == 5
    assert type(part[0]["electrode_id"]) is int


# ---------- _plan_sort_groups_by_shank -------------------------------------


def test_plan_by_shank_basic_two_groups_inherit_none():
    """Two electrode groups, one shank each, inherit a 'none' reference."""
    electrodes = _electrodes(
        [_shank("0", 0, eid) for eid in (1, 2, 3, 4)]
        + [_shank("1", 0, eid) for eid in (5, 6, 7, 8)]
    )
    proposed, skipped = _plan_sort_groups_by_shank(
        electrodes,
        electrodes,
        "s.nwb",
        omit_ref_electrode_group=False,
        omit_unitrode=True,
        references=None,
        override_pair=None,
    )
    assert skipped == []
    assert [p[0] for p in proposed] == ["0", "1"]
    assert [list(p[1]) for p in proposed] == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert [(p[2], p[3]) for p in proposed] == [("none", None), ("none", None)]


def test_plan_by_shank_omits_single_channel_unitrode():
    """A one-channel shank is skipped with reason 'unitrode'."""
    electrodes = _electrodes(
        [_shank("0", 0, eid) for eid in (1, 2)] + [_shank("1", 0, 3)]
    )
    proposed, skipped = _plan_sort_groups_by_shank(
        electrodes,
        electrodes,
        "s.nwb",
        omit_ref_electrode_group=False,
        omit_unitrode=True,
        references=None,
        override_pair=None,
    )
    assert [p[0] for p in proposed] == ["0"]
    assert skipped == [
        {"electrode_group_name": "1", "shank": 0, "reason": "unitrode"}
    ]


def test_plan_by_shank_references_missing_key_raises():
    """A `references` mapping missing a producing group raises ValueError."""
    electrodes = _electrodes([_shank("0", 0, eid) for eid in (1, 2)])
    with pytest.raises(ValueError, match="missing from `references`"):
        _plan_sort_groups_by_shank(
            electrodes,
            electrodes,
            "s.nwb",
            omit_ref_electrode_group=False,
            omit_unitrode=True,
            references={"other": -1},
            override_pair=None,
        )


def test_plan_by_shank_override_pair_forces_mode():
    """`override_pair` forces the mode regardless of configured reference."""
    # Configured ref is -2 (global_median); override forces 'none'.
    electrodes = _electrodes(
        [_shank("0", 0, eid, ref=-2) for eid in (1, 2)]
    )
    proposed, skipped = _plan_sort_groups_by_shank(
        electrodes,
        electrodes,
        "s.nwb",
        omit_ref_electrode_group=False,
        omit_unitrode=True,
        references=None,
        override_pair=("none", None),
    )
    assert skipped == []
    assert [(p[2], p[3]) for p in proposed] == [("none", None)]


def test_plan_by_shank_omit_ref_electrode_group_skips_owning_group():
    """omit_ref_electrode_group skips the group containing its own reference."""
    # Group "0" references electrode 4, which lives in group "0".
    electrodes = _electrodes(
        [_shank("0", 0, eid, ref=4) for eid in (1, 2, 3, 4)]
        + [_shank("1", 0, eid) for eid in (5, 6, 7, 8)]
    )
    proposed, skipped = _plan_sort_groups_by_shank(
        electrodes,
        electrodes,
        "s.nwb",
        omit_ref_electrode_group=True,
        omit_unitrode=True,
        references=None,
        override_pair=None,
    )
    assert [p[0] for p in proposed] == ["1"]
    assert skipped == [
        {
            "electrode_group_name": "0",
            "shank": 0,
            "reason": "reference_electrode_group",
        }
    ]


def test_plan_by_shank_specific_reference_member_raises():
    """A 'specific' reference that is a group member fails loudly."""
    electrodes = _electrodes(
        [_shank("0", 0, eid, ref=4) for eid in (1, 2, 3, 4)]
    )
    with pytest.raises(ValueError, match="also a sort-group channel"):
        _plan_sort_groups_by_shank(
            electrodes,
            electrodes,
            "s.nwb",
            omit_ref_electrode_group=False,
            omit_unitrode=True,
            references=None,
            override_pair=None,
        )


def test_plan_by_shank_no_groups_survive_raises():
    """All-unitrode input produces no sort groups -> ValueError."""
    electrodes = _electrodes([_shank("0", 0, 1), _shank("1", 0, 2)])
    with pytest.raises(ValueError, match="no sort groups produced"):
        _plan_sort_groups_by_shank(
            electrodes,
            electrodes,
            "s.nwb",
            omit_ref_electrode_group=False,
            omit_unitrode=True,
            references=None,
            override_pair=None,
        )


def test_plan_by_shank_references_mapping_resolves_specific():
    """A `references` entry resolves to a 'specific' cross-group reference."""
    # Group "0"'s members are referenced to electrode 7, which lives in a
    # different group ("9"), so it validates and is not a member.
    electrodes = _electrodes([_shank("0", 0, eid) for eid in (1, 2, 3)])
    all_electrodes = _electrodes(
        [_shank("0", 0, eid) for eid in (1, 2, 3)] + [_shank("9", 0, 7)]
    )
    proposed, skipped = _plan_sort_groups_by_shank(
        electrodes,
        all_electrodes,
        "s.nwb",
        omit_ref_electrode_group=False,
        omit_unitrode=True,
        references={"0": 7},
        override_pair=None,
    )
    assert skipped == []
    assert [(p[0], p[2], p[3]) for p in proposed] == [("0", "specific", 7)]


def test_plan_by_shank_global_median_auto_derived():
    """A configured -2 reference auto-derives to ('global_median', None)."""
    electrodes = _electrodes(
        [_shank("0", 0, eid, ref=-2) for eid in (1, 2, 3, 4)]
    )
    proposed, skipped = _plan_sort_groups_by_shank(
        electrodes,
        electrodes,
        "s.nwb",
        omit_ref_electrode_group=False,
        omit_unitrode=True,
        references=None,
        override_pair=None,
    )
    assert skipped == []
    assert [(p[2], p[3]) for p in proposed] == [("global_median", None)]


# ---------- _plan_sort_groups_by_column ------------------------------------


def test_plan_by_column_basic_two_groups():
    """Grouping by probe_shank yields one proposed entry per group."""
    electrodes = _electrodes(
        [_shank("0", 0, eid) for eid in (1, 2)]
        + [_shank("0", 1, eid) for eid in (3, 4)]
    )
    proposed, skipped = _plan_sort_groups_by_column(
        electrodes,
        electrodes,
        "s.nwb",
        "probe_shank",
        "probe_shank",
        groups=[[0], [1]],
        sort_group_ids=[0, 1],
        omit_unitrode=True,
        override_pair=None,
    )
    assert skipped == []
    assert [p[0] for p in proposed] == [0, 1]
    assert [list(p[1]["electrode_id"]) for p in proposed] == [[1, 2], [3, 4]]
    assert [(p[2], p[3]) for p in proposed] == [("none", None), ("none", None)]


def test_plan_by_column_empty_match_raises():
    """A group matching no electrodes fails loudly."""
    electrodes = _electrodes([_shank("0", 0, eid) for eid in (1, 2)])
    with pytest.raises(ValueError, match="matched no electrodes"):
        _plan_sort_groups_by_column(
            electrodes,
            electrodes,
            "s.nwb",
            "probe_shank",
            "probe_shank",
            groups=[[99]],
            sort_group_ids=[0],
            omit_unitrode=True,
            override_pair=None,
        )


def test_plan_by_column_unitrode_skip_leaves_id_gap():
    """A skipped unitrode group leaves a gap in the assigned ids."""
    electrodes = _electrodes(
        [_shank("0", 0, 1)] + [_shank("0", 1, eid) for eid in (2, 3)]
    )
    proposed, skipped = _plan_sort_groups_by_column(
        electrodes,
        electrodes,
        "s.nwb",
        "probe_shank",
        "probe_shank",
        groups=[[0], [1]],
        sort_group_ids=[0, 1],
        omit_unitrode=True,
        override_pair=None,
    )
    # Group with id 0 was a unitrode -> skipped; id 1 survives (gap at 0).
    assert [p[0] for p in proposed] == [1]
    assert skipped == [{"sort_group_id": 0, "reason": "unitrode"}]


def test_plan_by_column_no_groups_survive_raises():
    """All-unitrode column groups produce nothing -> ValueError."""
    electrodes = _electrodes([_shank("0", 0, 1)])
    with pytest.raises(ValueError, match="no sort groups"):
        _plan_sort_groups_by_column(
            electrodes,
            electrodes,
            "s.nwb",
            "probe_shank",
            "probe_shank",
            groups=[[0]],
            sort_group_ids=[0],
            omit_unitrode=True,
            override_pair=None,
        )
