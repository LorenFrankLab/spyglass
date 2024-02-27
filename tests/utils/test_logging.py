import sys

import pytest

from spyglass.utils.logging import excepthook
from tests.conftest import VERBOSE


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy.")
def test_existing_excepthook(caplog):
    try:
        raise ValueError("Test exception")
    except ValueError:
        excepthook(*sys.exc_info())

    assert "Uncaught exception" in caplog.text, "No warning issued."
