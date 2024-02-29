import pytest

from tests.conftest import VERBOSE


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy")
def test_deprecation_factory(caplog, common):
    common.common_position.TrackGraph()
    assert (
        "Deprecation:" in caplog.text
    ), "No deprecation warning logged on migrated table."
