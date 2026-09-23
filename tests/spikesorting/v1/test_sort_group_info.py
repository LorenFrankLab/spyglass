"""Tests for ``get_sort_group_info`` electrode coverage.

Issue #1394: ``get_sort_group_info`` fetched a single electrode per sort
group, so a sort group spanning many electrodes was silently described by
one of them. These tests pin the historical (default) behavior and cover
the new ``all_electrodes`` opt-in that returns every electrode.
"""

import datajoint as dj
import pytest

# Expected single-electrode row for the ``pop_curation`` fixture, matching
# the historical return of ``CurationV1.get_sort_group_info``. Used to prove
# the default path is byte-for-byte unchanged.
EXPECTED_DEFAULT_ROW = {
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


def test_sort_group_info_default_is_one_electrode(spike_v1, pop_curation):
    """Default call is unchanged: one electrode row describes the group."""
    info = spike_v1.CurationV1.get_sort_group_info(pop_curation)

    assert len(info) == 1, (
        "Default get_sort_group_info should return a single row per sort "
        f"group; got {len(info)}"
    )

    row = info.fetch1()
    for k, v in EXPECTED_DEFAULT_ROW.items():
        assert row[k] == v, f"Default get_sort_group_info changed value for {k}"


def test_sort_group_info_default_kwarg_is_explicit_default(
    spike_v1, pop_curation
):
    """``all_electrodes=False`` is identical to omitting the argument."""
    implicit = spike_v1.CurationV1.get_sort_group_info(pop_curation).fetch(
        as_dict=True, order_by="electrode_id"
    )
    explicit = spike_v1.CurationV1.get_sort_group_info(
        pop_curation, all_electrodes=False
    ).fetch(as_dict=True, order_by="electrode_id")

    assert (
        implicit == explicit
    ), "all_electrodes=False must reproduce the default return exactly"


def test_sort_group_info_all_electrodes(
    spike_v1, pop_curation, sort_group_electrode_ids
):
    """``all_electrodes=True`` yields every electrode in the sort group."""
    info = spike_v1.CurationV1.get_sort_group_info(
        pop_curation, all_electrodes=True
    )

    returned = sorted(info.fetch("electrode_id"))
    assert returned == sort_group_electrode_ids, (
        "all_electrodes=True should return every electrode in the sort "
        f"group. Expected {sort_group_electrode_ids}, got {returned}"
    )

    # Group-level fields must still agree with the single-electrode row;
    # per-electrode fields (electrode_id, name, probe_electrode, ...) vary.
    group_level = [
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
    for row in info.fetch(as_dict=True):
        for k in group_level:
            assert (
                row[k] == EXPECTED_DEFAULT_ROW[k]
            ), f"all_electrodes=True unexpected value for {k}"


def test_sort_group_info_returns_dj_expression(spike_v1, pop_curation):
    """Both modes still return a restrictable DataJoint expression."""
    for kwargs in ({}, {"all_electrodes": True}):
        info = spike_v1.CurationV1.get_sort_group_info(pop_curation, **kwargs)
        assert isinstance(info, dj.expression.QueryExpression), (
            f"get_sort_group_info({kwargs}) returned {type(info)}, "
            "not a DataJoint expression"
        )
        # still usable as a query: restriction must not raise
        assert len(info & {"electrode_id": 0}) == 1


def test_v0_sort_group_info_signature_mirrors_v1():
    """V0 exposes the same opt-in, so the two APIs stay interchangeable."""
    from inspect import signature

    from spyglass.spikesorting.v0.spikesorting_curation import (
        CuratedSpikeSorting,
    )
    from spyglass.spikesorting.v1.curation import CurationV1

    v0_params = signature(CuratedSpikeSorting.get_sort_group_info).parameters
    v1_params = signature(CurationV1.get_sort_group_info).parameters

    assert "all_electrodes" in v0_params, (
        "CuratedSpikeSorting.get_sort_group_info is missing the "
        "all_electrodes opt-in added to CurationV1"
    )
    assert (
        v0_params["all_electrodes"].default
        == v1_params["all_electrodes"].default
        is False
    ), "all_electrodes must default to False in both v0 and v1"
