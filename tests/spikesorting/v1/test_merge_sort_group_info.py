"""Tests for ``SpikeSortingOutput.get_sort_group_info`` electrode coverage.

Issue #1394: ``get_sort_group_info`` reported a single electrode per sort
group, which is not what the method name promises. It now returns every
electrode of each sort group, with no opt-out -- see the PR #1678 review
discussion. These tests pin that return shape, the merge-id join that is the
whole point of the method, and the signature parity the dispatch relies on.
"""

from inspect import signature

import datajoint as dj
import pytest


@pytest.fixture(scope="session")
def merge_sort_group_key(spike_v1, spike_merge, pop_spike_merge):
    """Primary key of the SortGroup behind the ``pop_spike_merge`` fixture."""
    sorting_id = (spike_merge.CurationV1 & pop_spike_merge).fetch1("sorting_id")
    recording_id = (
        spike_v1.SpikeSortingSelection & {"sorting_id": sorting_id}
    ).fetch1("recording_id")
    row = (
        spike_v1.SpikeSortingRecordingSelection & {"recording_id": recording_id}
    ).fetch1()
    yield {
        "nwb_file_name": row["nwb_file_name"],
        "sort_group_id": row["sort_group_id"],
    }


@pytest.fixture(scope="session")
def merge_electrode_ids(spike_v1, merge_sort_group_key):
    """All electrode ids in the sort group, asserted to be more than one.

    The minirec fixture's ``SortGroup.set_group_by_shank`` places the four
    channels of a ``tetrode_12.5`` probe into ``sort_group_id`` 0. Asserting
    that here keeps the multi-electrode tests from passing vacuously on a
    single-electrode sort group.
    """
    ids = sorted(
        (spike_v1.SortGroup.SortGroupElectrode & merge_sort_group_key).fetch(
            "electrode_id"
        )
    )
    assert len(ids) > 1, (
        "Test precondition failed: fixture sort group has "
        f"{len(ids)} electrode(s); cannot prove multi-electrode behavior."
    )
    yield ids


def test_merge_sort_group_info_returns_all_electrodes(
    spike_merge, pop_spike_merge, merge_electrode_ids
):
    """Every electrode of the sort group is returned."""
    info = spike_merge.get_sort_group_info(pop_spike_merge)

    returned = sorted(info.fetch("electrode_id"))
    assert returned == merge_electrode_ids, (
        "get_sort_group_info should return every electrode in the sort "
        f"group. Expected {merge_electrode_ids}, got {returned}"
    )


def test_merge_sort_group_info_keeps_merge_id(
    spike_merge, pop_spike_merge, merge_electrode_ids
):
    """The merge-id join survives the multi-electrode return."""
    info = spike_merge.get_sort_group_info(pop_spike_merge)

    assert isinstance(info, dj.expression.QueryExpression), (
        f"get_sort_group_info returned {type(info)}, not a DataJoint "
        "expression"
    )
    assert (
        "merge_id" in info.heading.names
    ), "get_sort_group_info dropped the merge_id column"

    merge_ids = set(info.fetch("merge_id"))
    assert merge_ids == {pop_spike_merge["merge_id"]}, (
        "Every electrode row should carry the queried merge id; got "
        f"{merge_ids}"
    )
    assert len(info) == len(merge_electrode_ids), (
        "Joining merge ids should not duplicate or drop electrode rows; "
        f"expected {len(merge_electrode_ids)} rows, got {len(info)}"
    )


def test_merge_sort_group_info_takes_only_key():
    """No electrode-coverage opt-out, on the merge table or its sources.

    ``get_sort_group_info`` dispatches through ``source_class_dict``, so the
    merge table and every source that defines the method must agree on the
    signature. Reintroducing a coverage flag on one of them would fail only
    at runtime, for that one source.
    """
    from spyglass.spikesorting.spikesorting_merge import (
        SpikeSortingOutput,
        source_class_dict,
    )

    targets = {"SpikeSortingOutput": SpikeSortingOutput}
    targets.update(
        {
            name: source
            for name, source in source_class_dict.items()
            if getattr(source, "get_sort_group_info", None) is not None
        }
    )

    for name, target in targets.items():
        params = signature(target.get_sort_group_info).parameters
        assert "all_electrodes" not in params, (
            f"{name}.get_sort_group_info reintroduced all_electrodes; the "
            "method returns every electrode unconditionally"
        )
        assert list(params) == ["key"], (
            f"{name}.get_sort_group_info takes {list(params)}, breaking the "
            "merge dispatch, which passes only a key"
        )
