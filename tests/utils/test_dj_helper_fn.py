import subprocess
import sys

import datajoint as dj
import pytest

from tests.conftest import VERBOSE


def test_settings_import_no_db_connection():
    """Importing SpyglassConfig must not trigger a database connection.

    Regression test for https://github.com/LorenFrankLab/spyglass/issues/...
    where `from spyglass.settings import SpyglassConfig` caused a premature
    connection attempt even before a valid database was configured.
    """
    code = "\n".join(
        [
            "import datajoint as dj",
            "dj.config['database.host'] = 'invalid_host_that_does_not_exist'",
            "dj.config['connection.init_function'] = None",
            "from spyglass.settings import SpyglassConfig",
            "print('SUCCESS')",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "SUCCESS" in result.stdout, (
        "Importing SpyglassConfig triggered a database connection attempt.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy")
def test_deprecation_factory(caplog, common):
    common.common_position.TrackGraph()
    assert (
        "Deprecation:" in caplog.text
    ), "No deprecation warning logged on migrated table."


@pytest.fixture(scope="module")
def str_to_bool():
    from spyglass.utils.dj_helper_fn import str_to_bool

    return str_to_bool


@pytest.mark.parametrize(
    "input_str, expected",
    [
        # Test truthy strings (case insensitive)
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("t", True),
        ("T", True),
        ("yes", True),
        ("YES", True),
        ("y", True),
        ("Y", True),
        ("1", True),
        # Test falsy strings
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("no", False),
        ("n", False),
        ("0", False),
        ("", False),
        ("random_string", False),
        # Test non-string inputs
        (True, True),
        (False, False),
        (None, False),
        (0, False),
        (1, True),
    ],
)
def test_str_to_bool(str_to_bool, input_str, expected):
    """Test str_to_bool function handles various string inputs correctly.

    This test verifies that str_to_bool properly converts string values
    to booleans, which is essential for fixing issue #1528 where string
    "false" was incorrectly evaluated as True.
    """
    assert (
        str_to_bool(input_str) == expected
    ), f"Expected {expected} for input '{input_str}'"
