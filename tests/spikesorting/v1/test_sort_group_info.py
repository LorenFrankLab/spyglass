"""Tests for ``get_sort_group_info`` electrode coverage.

Issue #1394: ``get_sort_group_info`` fetched a single electrode per sort
group, so a sort group spanning many electrodes was silently described by one
of them. It now returns every electrode of the group, with no opt-out -- see
the PR #1678 review discussion. These tests pin that return shape and the
v0/v1 signature parity.
"""

import datajoint as dj
import pytest

# Expected row for electrode 0 of the ``pop_curation`` fixture's sort group.
# The group now returns one row per electrode; this pins the column values of
# a single known row so a change in the joins cannot pass unnoticed.
EXPECTED_ELECTRODE_0_ROW = {
    "bad_channel": "False",
    "curation_id": 0,
    "electrode_group_name": "0",
    "electrode_id": 0,
    "filtering": "None",
    "impedance": 0.0,
    "merges_applied": 0,
    "name": "0",
    "nwb_file_name": "minirec20230622_.nwb",
    "original_reference_electrode": 0,
    "parent_curation_id": -1,
    "probe_electrode": 0,
    "probe_id": "tetrode_12.5",
    "probe_shank": 0,
    "region_id": 1,
    "sort_group_id": 0,
    "subregion_name": None,
    "subsubregion_name": None,
    "x": 0.0,
    "x_warped": 0.0,
    "y": 0.0,
    "y_warped": 0.0,
    "z": 0.0,
    "z_warped": 0.0,
}

# Fields shared by every electrode of the group. The rest (electrode_id, name,
# probe_electrode, ...) legitimately vary row to row.
GROUP_LEVEL_FIELDS = [
    "bad_channel",
    "curation_id",
    "electrode_group_name",
    "filtering",
    "impedance",
    "merges_applied",
    "nwb_file_name",
    "parent_curation_id",
    "probe_id",
    "probe_shank",
    "region_id",
    "sort_group_id",
]


@pytest.fixture(scope="session")
def sort_group_key(spike_v1, pop_curation):
    """Primary key of the SortGroup behind the ``pop_curation`` fixture."""
    recording_id = (
        (spike_v1.CurationV1 & pop_curation) * spike_v1.SpikeSortingSelection()
    ).fetch1("recording_id")
    row = (
        spike_v1.SpikeSortingRecordingSelection & {"recording_id": recording_id}
    ).fetch1()
    yield {
        "nwb_file_name": row["nwb_file_name"],
        "sort_group_id": row["sort_group_id"],
    }


@pytest.fixture(scope="session")
def sort_group_electrode_ids(spike_v1, sort_group_key):
    """All electrode ids in the sort group, asserted to be more than one.

    The minirec fixture uses a ``tetrode_12.5`` probe, so ``sort_group_id``
    0 spans four electrodes. Asserting here keeps the multi-electrode tests
    from silently passing on a single-electrode group.
    """
    ids = sorted(
        (spike_v1.SortGroup.SortGroupElectrode & sort_group_key).fetch(
            "electrode_id"
        )
    )
    assert len(ids) > 1, (
        "Test precondition failed: fixture sort group has "
        f"{len(ids)} electrode(s); cannot prove multi-electrode behavior."
    )
    yield ids


def test_sort_group_info_returns_every_electrode(
    spike_v1, pop_curation, sort_group_electrode_ids
):
    """Every electrode in the sort group is returned."""
    info = spike_v1.CurationV1.get_sort_group_info(pop_curation)

    returned = sorted(info.fetch("electrode_id"))
    assert returned == sort_group_electrode_ids, (
        "get_sort_group_info should return every electrode in the sort "
        f"group. Expected {sort_group_electrode_ids}, got {returned}"
    )


def test_sort_group_info_row_values(spike_v1, pop_curation):
    """Electrode 0's row carries the values the joins have always produced."""
    info = spike_v1.CurationV1.get_sort_group_info(pop_curation)

    row = (info & {"electrode_id": 0}).fetch1()
    for k, v in EXPECTED_ELECTRODE_0_ROW.items():
        assert row[k] == v, f"get_sort_group_info changed value for {k}"


def test_sort_group_info_group_fields_agree(spike_v1, pop_curation):
    """Group-level fields are identical across the electrode rows."""
    info = spike_v1.CurationV1.get_sort_group_info(pop_curation)

    for row in info.fetch(as_dict=True):
        for k in GROUP_LEVEL_FIELDS:
            assert (
                row[k] == EXPECTED_ELECTRODE_0_ROW[k]
            ), f"unexpected group-level value for {k}"


def test_sort_group_info_returns_dj_expression(spike_v1, pop_curation):
    """The return is still a restrictable DataJoint expression."""
    info = spike_v1.CurationV1.get_sort_group_info(pop_curation)

    assert isinstance(info, dj.expression.QueryExpression), (
        f"get_sort_group_info returned {type(info)}, "
        "not a DataJoint expression"
    )
    # still usable as a query: restriction must not raise
    assert len(info & {"electrode_id": 0}) == 1


def test_v0_sort_group_info_signature_mirrors_v1():
    """V0 and V1 stay interchangeable: both take only a key."""
    from inspect import signature

    from spyglass.spikesorting.v0.spikesorting_curation import (
        CuratedSpikeSorting,
    )
    from spyglass.spikesorting.v1.curation import CurationV1

    v0_params = signature(CuratedSpikeSorting.get_sort_group_info).parameters
    v1_params = signature(CurationV1.get_sort_group_info).parameters

    assert list(v0_params) == list(v1_params) == ["key"], (
        "v0 and v1 get_sort_group_info must take only a key; got "
        f"{list(v0_params)} and {list(v1_params)}"
    )
