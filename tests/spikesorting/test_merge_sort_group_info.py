"""Tests for ``SpikeSortingOutput.get_sort_group_info`` electrode coverage.

Issue #1394: ``get_sort_group_info`` reported a single electrode per sort
group. The source tables (``CurationV1`` and ``CuratedSpikeSorting``) gained
an ``all_electrodes`` opt-in, but the merge table -- the entry point used by
the ``10_Spike_SortingV1`` tutorial -- did not forward it, so users of the
merge table could not opt in. These tests pin the passthrough, the unchanged
default, and the merge-id join that is the whole point of the method.
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


def test_merge_sort_group_info_default_unchanged(spike_merge, pop_spike_merge):
    """Default call still returns one representative electrode row."""
    info = spike_merge.get_sort_group_info(pop_spike_merge)

    assert len(info) == 1, (
        "Default SpikeSortingOutput.get_sort_group_info should return a "
        f"single row per sort group; got {len(info)}"
    )
    assert (
        info.fetch1("merge_id") == pop_spike_merge["merge_id"]
    ), "Default get_sort_group_info lost the merge id it is meant to join"


def test_merge_sort_group_info_explicit_default_matches(
    spike_merge, pop_spike_merge
):
    """``all_electrodes=False`` reproduces the default return exactly."""
    implicit = spike_merge.get_sort_group_info(pop_spike_merge).fetch(
        as_dict=True, order_by="electrode_id"
    )
    explicit = spike_merge.get_sort_group_info(
        pop_spike_merge, all_electrodes=False
    ).fetch(as_dict=True, order_by="electrode_id")

    assert (
        implicit == explicit
    ), "all_electrodes=False must reproduce the default return exactly"


def test_merge_sort_group_info_all_electrodes(
    spike_merge, pop_spike_merge, merge_electrode_ids
):
    """``all_electrodes=True`` returns every electrode of the sort group."""
    info = spike_merge.get_sort_group_info(pop_spike_merge, all_electrodes=True)

    returned = sorted(info.fetch("electrode_id"))
    assert returned == merge_electrode_ids, (
        "all_electrodes=True should return every electrode in the sort "
        f"group. Expected {merge_electrode_ids}, got {returned}"
    )


def test_merge_sort_group_info_all_electrodes_keeps_merge_id(
    spike_merge, pop_spike_merge, merge_electrode_ids
):
    """The merge-id join survives the ``all_electrodes=True`` path."""
    info = spike_merge.get_sort_group_info(pop_spike_merge, all_electrodes=True)

    assert isinstance(info, dj.expression.QueryExpression), (
        f"get_sort_group_info returned {type(info)}, not a DataJoint "
        "expression"
    )
    assert (
        "merge_id" in info.heading.names
    ), "all_electrodes=True dropped the merge_id column"

    merge_ids = set(info.fetch("merge_id"))
    assert merge_ids == {pop_spike_merge["merge_id"]}, (
        "Every electrode row should carry the queried merge id; got "
        f"{merge_ids}"
    )
    assert len(info) == len(merge_electrode_ids), (
        "Joining merge ids should not duplicate or drop electrode rows; "
        f"expected {len(merge_electrode_ids)} rows, got {len(info)}"
    )


def test_merge_sort_group_info_signature():
    """The merge method exposes the same opt-in as its source tables."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    params = signature(SpikeSortingOutput.get_sort_group_info).parameters
    assert (
        "all_electrodes" in params
    ), "SpikeSortingOutput.get_sort_group_info is missing all_electrodes"
    assert (
        params["all_electrodes"].default is False
    ), "all_electrodes must default to False on the merge table"


def test_merge_source_classes_accept_all_electrodes():
    """Every dispatch target that defines the method accepts the kwarg.

    ``get_sort_group_info`` dispatches through ``source_class_dict``. A
    source class that defines the method without ``all_electrodes`` would
    fail only at runtime, for that one source.
    """
    from spyglass.spikesorting.spikesorting_merge import source_class_dict

    for name, source in source_class_dict.items():
        method = getattr(source, "get_sort_group_info", None)
        if method is None:  # source does not support the method at all
            continue
        params = signature(method).parameters
        assert "all_electrodes" in params, (
            f"{name}.get_sort_group_info cannot accept all_electrodes, so "
            "the merge passthrough would fail for this source"
        )
        assert (
            params["all_electrodes"].default is False
        ), f"{name}.get_sort_group_info must default all_electrodes to False"
