"""
Shared fixtures for spikesorting tests (both v0 and v1).

This file contains common fixtures used across v0 and v1 spikesorting tests,
reducing duplication and maintenance burden.
"""

import pytest


@pytest.fixture(scope="session")
def burst_params_key_shared():
    """Shared burst parameters key for both v0 and v1.

    This fixture is identical in v0 and v1, so we consolidate it here.
    Both v0 and v1 conftest files can import this to avoid duplication.
    """
    yield dict(burst_params_name="default")
