"""Local conftest for offline (no-DB) unit tests.

Shadows the root conftest's ``mini_insert`` autouse session fixture with a
no-op so that ``test_connection_off.py`` can run without a Docker/MySQL
container.  All other root fixtures (``verbose_context``, etc.) are harmless
and continue to apply normally.
"""

import pytest


@pytest.fixture(autouse=True, scope="session")
def mini_insert():
    """No-op override: offline tests need no DB setup."""
    yield
