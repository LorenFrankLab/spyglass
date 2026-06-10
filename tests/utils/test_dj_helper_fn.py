import subprocess
import sys
from pathlib import Path

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


# -- _write_external_checksum ----------------------------------------


def test_write_external_checksum_updates_row(tmp_path):
    """Helper writes size + contents_hash and calls update1 with the key."""
    from unittest.mock import MagicMock, patch

    from spyglass.utils.dj_helper_fn import _write_external_checksum

    filepath = tmp_path / "stub.nwb"
    filepath.write_bytes(b"abc")
    ext_tbl = MagicMock()
    key = {"hash": "old", "filepath": "stub.nwb"}

    with patch(
        "spyglass.utils.dj_helper_fn.dj.hash.uuid_from_file",
        return_value="new_hash",
    ):
        _write_external_checksum(filepath, ext_tbl, key)

    ext_tbl.update1.assert_called_once()
    written = ext_tbl.update1.call_args[0][0]
    assert written["size"] == 3
    assert written["contents_hash"] == "new_hash"
    assert written["filepath"] == "stub.nwb"


def test_write_external_checksum_skips_admin_check(tmp_path):
    """Helper does not call LabMember.check_admin_privilege."""
    from unittest.mock import MagicMock, patch

    from spyglass.utils.dj_helper_fn import _write_external_checksum

    filepath = tmp_path / "stub.nwb"
    filepath.write_bytes(b"x")
    ext_tbl = MagicMock()

    with (
        patch("spyglass.common.LabMember") as mock_lm,
        patch(
            "spyglass.utils.dj_helper_fn.dj.hash.uuid_from_file",
            return_value="h",
        ),
    ):
        _write_external_checksum(filepath, ext_tbl, {"filepath": "stub.nwb"})

    mock_lm.return_value.check_admin_privilege.assert_not_called()


def test_resolve_external_table_still_requires_admin():
    """_resolve_external_table must still gate on admin privilege."""
    from unittest.mock import patch

    from spyglass.utils.dj_helper_fn import _resolve_external_table

    with patch("spyglass.common.LabMember") as mock_lm:
        mock_lm.return_value.check_admin_privilege.side_effect = (
            PermissionError("admin only")
        )
        with pytest.raises(PermissionError, match="admin only"):
            _resolve_external_table("/tmp/x.nwb", "x.nwb", "analysis")


def test_update_analysis_file_checksum_resolves_and_writes(tmp_path):
    """Helper resolves rel_path against AnalysisNwbfile, searches all
    registered external tables, and writes via
    _write_external_checksum (no admin gate)."""
    from unittest.mock import MagicMock, patch

    from spyglass.utils.dj_helper_fn import _update_analysis_file_checksum

    real_file = tmp_path / "file.nwb"
    real_file.write_bytes(b"payload")
    ext_key = {"filepath": "file.nwb"}
    ext_tbl = MagicMock()
    ext_tbl.__and__ = MagicMock(
        return_value=MagicMock(
            __bool__=MagicMock(return_value=True),
            fetch1=MagicMock(return_value=ext_key),
        )
    )

    with (
        patch("spyglass.common.common_nwbfile.AnalysisNwbfile") as mock_anwb,
        patch(
            "spyglass.common.common_nwbfile.AnalysisRegistry"
        ) as mock_registry,
        patch(
            "spyglass.utils.dj_helper_fn._write_external_checksum"
        ) as mock_write,
        patch("spyglass.common.LabMember") as mock_lm,
    ):
        inst = mock_anwb.return_value
        inst._analysis_dir = str(tmp_path)
        mock_registry.return_value.get_externals.return_value = [ext_tbl]

        _update_analysis_file_checksum(str(real_file))

    mock_write.assert_called_once()
    args = mock_write.call_args[0]
    assert Path(args[0]) == real_file
    assert args[1] is ext_tbl
    assert args[2] == ext_key
    mock_lm.return_value.check_admin_privilege.assert_not_called()


def test_update_analysis_file_checksum_warns_on_zero_matches(tmp_path):
    """Helper logs a warning (does not raise) when the file is registered
    in zero external tables — avoids an opaque DataJointError after
    the file has already been swapped on disk."""
    from unittest.mock import MagicMock, patch

    from spyglass.utils.dj_helper_fn import _update_analysis_file_checksum

    real_file = tmp_path / "file.nwb"
    real_file.write_bytes(b"payload")
    ext_tbl = MagicMock()
    ext_tbl.__and__ = MagicMock(
        return_value=MagicMock(__bool__=MagicMock(return_value=False))
    )

    with (
        patch("spyglass.common.common_nwbfile.AnalysisNwbfile") as mock_anwb,
        patch(
            "spyglass.common.common_nwbfile.AnalysisRegistry"
        ) as mock_registry,
        patch(
            "spyglass.utils.dj_helper_fn._write_external_checksum"
        ) as mock_write,
        patch("spyglass.utils.dj_helper_fn.logger") as mock_logger,
    ):
        inst = mock_anwb.return_value
        inst._analysis_dir = str(tmp_path)
        mock_registry.return_value.get_externals.return_value = [ext_tbl]

        _update_analysis_file_checksum(str(real_file))

    mock_write.assert_not_called()
    mock_logger.warning.assert_called_once()
