import pytest

from ..conftest import TEARDOWN


@pytest.fixture
def region_dict():
    yield dict(region_name="test_region")


@pytest.fixture
def brain_region(common, region_dict):
    brain_region = common.common_region.BrainRegion()
    (brain_region & "region_id > 1").delete(safemode=False)
    yield brain_region
    (brain_region & "region_id > 1").delete(safemode=False)


@pytest.mark.skipif(not TEARDOWN, reason="No teardown: no test autoincrement")
def test_region_add(brain_region, region_dict):
    before = set(brain_region.fetch("region_id"))
    region_id = brain_region.fetch_add(
        **region_dict,
        subregion_name="test_subregion_add",
        subsubregion_name="test_subsubregion_add",
    )
    # fetch_add autoincrements: the new region gets a fresh id larger than any
    # existing one. Assert that directly rather than assuming the AUTO_INCREMENT
    # counter equals max(region_id)+1 -- it does not once earlier rows have been
    # inserted and deleted (InnoDB does not reclaim the counter).
    assert region_id not in before, "fetch_add should create a new region_id."
    assert region_id == max(
        brain_region.fetch("region_id")
    ), "the new region should hold the largest region_id."
