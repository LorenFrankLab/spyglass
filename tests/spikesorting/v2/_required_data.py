"""Skip a test for missing data, unless the run requires its fixtures."""

from __future__ import annotations

import os
from typing import NoReturn

import pytest

REQUIRE_FIXTURES_ENV = "SPYGLASS_V2_REQUIRE_FIXTURES"


def skip_unless_fixtures_required(reason: str) -> NoReturn:
    """Skip with ``reason``, or fail when ``SPYGLASS_V2_REQUIRE_FIXTURES`` is set.

    A run that declares required fixtures must exercise them, so data they
    should provide but do not is a failure there, not a silent skip.
    """
    if os.environ.get(REQUIRE_FIXTURES_ENV, "").split():
        pytest.fail(
            f"{reason} ({REQUIRE_FIXTURES_ENV} is set, so this does not skip)",
            pytrace=False,
        )
    pytest.skip(reason)
