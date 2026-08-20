import pytest
from datajoint import U as dj_U


@pytest.fixture
def region_dict():
    yield dict(region_name="test_region")


@pytest.fixture
def brain_region(common):
    """Yield BrainRegion, clearing test rows (region_id > 1) before and after."""
    brain_region = common.common_region.BrainRegion()
    (brain_region & "region_id > 1").delete(safemode=False)
    yield brain_region
    (brain_region & "region_id > 1").delete(safemode=False)


def _max_region_id(brain_region):
    return dj_U().aggr(brain_region, n="max(region_id)").fetch1("n") or 0


def test_region_add(brain_region, region_dict):
    """fetch_add inserts a new region with a fresh autoincrement id.

    The fixture clears test rows before/after, so the test owns its data. We
    assert the new id is *greater than* the prior max — always true for a
    fresh autoincrement id — rather than exactly ``max + 1`` (which fails once
    MySQL's counter has advanced past deleted rows, the reason this test was
    previously skipped without teardown).
    """
    pre_max = _max_region_id(brain_region)
    region_id = brain_region.fetch_add(
        **region_dict,
        subregion_name="test_subregion_add",
        subsubregion_name="test_subsubregion_add",
    )
    assert region_id > pre_max, "fetch_add did not assign a fresh id"

    row = (brain_region & {"region_id": region_id}).fetch1()
    assert row["region_name"] == "test_region"
    assert row["subregion_name"] == "test_subregion_add"
    assert row["subsubregion_name"] == "test_subsubregion_add"


def test_region_fetch_add_idempotent(brain_region):
    """Repeat fetch_add returns the same id and does not re-insert."""
    first = brain_region.fetch_add(region_name="test_region_idempotent")
    count = len(brain_region)
    second = brain_region.fetch_add(region_name="test_region_idempotent")

    assert second == first
    assert (
        len(brain_region) == count
    ), "fetch_add re-inserted an existing region"
