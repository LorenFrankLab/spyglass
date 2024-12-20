import pytest
from datajoint import U as dj_U

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
    next_id = (
        dj_U().aggr(brain_region, n="max(region_id)").fetch1("n") or 0
    ) + 1
    region_id = brain_region.fetch_add(
        **region_dict,
        subregion_name="test_subregion_add",
        subsubregion_name="test_subsubregion_add",
    )
    assert (
        region_id == next_id
    ), "Region.fetch_add() should autoincrement region_id."
