import sys

import pytest

from tests.conftest import VERBOSE


@pytest.fixture(scope="module")
def excepthook():
    from spyglass.utils.logging import excepthook

    return excepthook


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy.")
def test_existing_excepthook(caplog, excepthook):
    try:
        raise ValueError("Test exception")
    except ValueError:
        excepthook(*sys.exc_info())

    assert "Uncaught exception" in caplog.text, "No warning issued."
