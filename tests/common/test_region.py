import pytest
from datajoint import U as dj_U


@pytest.fixture
def brain_region(common):
    yield common.common_region.BrainRegion()


def test_region_add(brain_region):
    next_id = (
        dj_U().aggr(brain_region, n="max(region_id)").fetch1("n") or 0
    ) + 1
    region_id = brain_region.fetch_add(
        region_name="test_region_add",
        subregion_name="test_subregion_add",
        subsubregion_name="test_subsubregion_add",
    )
    assert (
        region_id == next_id
    ), "Region.fetch_add() should autincrement region_id."
