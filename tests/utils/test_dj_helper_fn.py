import pytest

from tests.conftest import VERBOSE


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy")
def test_deprecation_factory(caplog, common):
    common.common_position.TrackGraph()
    assert (
        "Deprecation:" in caplog.text
    ), "No deprecation warning logged on migrated table."


def test_str_to_bool():
    """Test str_to_bool function handles various string inputs correctly.
    
    This test verifies that str_to_bool properly converts string values
    to booleans, which is essential for fixing issue #1528 where string
    "false" was incorrectly evaluated as True.
    """
    from spyglass.utils.dj_helper_fn import str_to_bool
    
    # Test truthy strings (case insensitive)
    assert str_to_bool("true") is True
    assert str_to_bool("True") is True
    assert str_to_bool("TRUE") is True
    assert str_to_bool("t") is True
    assert str_to_bool("T") is True
    assert str_to_bool("yes") is True
    assert str_to_bool("YES") is True
    assert str_to_bool("y") is True
    assert str_to_bool("Y") is True
    assert str_to_bool("1") is True
    
    # Test falsy strings
    assert str_to_bool("false") is False
    assert str_to_bool("False") is False
    assert str_to_bool("FALSE") is False
    assert str_to_bool("no") is False
    assert str_to_bool("n") is False
    assert str_to_bool("0") is False
    assert str_to_bool("") is False
    assert str_to_bool("random_string") is False
    
    # Test non-string inputs
    assert str_to_bool(True) is True
    assert str_to_bool(False) is False
    assert str_to_bool(None) is False
    assert str_to_bool(0) is False
    assert str_to_bool(1) is True
