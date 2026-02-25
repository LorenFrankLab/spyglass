from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import VERBOSE


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy")
def test_deprecation_factory(caplog, common):
    common.common_position.TrackGraph()
    assert (
        "Deprecation:" in caplog.text
    ), "No deprecation warning logged on migrated table."


# ---------------------------------------------------------------------------
# Helpers for _resolve_external_table tests
# ---------------------------------------------------------------------------


def _make_external_mock(num_entries, fetch1_result=None):
    """Return a mock external table restricted to `num_entries` rows."""
    mock = MagicMock()
    mock.__bool__.return_value = bool(num_entries)
    mock.__len__.return_value = num_entries
    mock.__and__.return_value = mock  # supports `& file_restr`
    if fetch1_result is not None:
        mock.fetch1.return_value = dict(fetch1_result)
    return mock


def _make_common_mocks(file_size=42, file_hash="abc123"):
    """Return mocks shared across tests."""
    lab_member_mock = MagicMock()
    lab_member_instance = MagicMock()
    lab_member_instance.check_admin_privilege = MagicMock(return_value=None)
    lab_member_mock.return_value = lab_member_instance

    path_mock = MagicMock()
    path_mock.return_value.stat.return_value.st_size = file_size

    dj_mock = MagicMock()
    dj_mock.hash.uuid_from_file.return_value = file_hash

    return lab_member_mock, path_mock, dj_mock


def _import_resolve():
    from spyglass.utils.dj_helper_fn import _resolve_external_table

    return _resolve_external_table


# ---------------------------------------------------------------------------
# Tests — analysis location
# ---------------------------------------------------------------------------


def test_analysis_no_externals_warns(caplog):
    """When no external tables have a matching entry, log a warning."""
    empty_ext = _make_external_mock(0)
    lab_member_mock, path_mock, dj_mock = _make_common_mocks()

    registry_mock = MagicMock()
    registry_mock.return_value.get_externals.return_value = [empty_ext]
    schema_mock = MagicMock()

    resolve = _import_resolve()
    with patch("spyglass.common.LabMember", lab_member_mock), patch(
        "spyglass.common.common_nwbfile.AnalysisRegistry", registry_mock
    ), patch("spyglass.common.common_nwbfile.schema", schema_mock), patch(
        "spyglass.utils.dj_helper_fn.Path", path_mock
    ), patch(
        "spyglass.utils.dj_helper_fn.dj", dj_mock
    ):
        with caplog.at_level("WARNING"):
            resolve("/fake/path/file.nwb", "file.nwb", location="analysis")

    assert "No entries found" in caplog.text


def test_analysis_multiple_entries_raises():
    """When an external table has >1 matching entry, raise ValueError."""
    multi_ext = _make_external_mock(2)
    lab_member_mock, path_mock, dj_mock = _make_common_mocks()

    registry_mock = MagicMock()
    registry_mock.return_value.get_externals.return_value = [multi_ext]
    schema_mock = MagicMock()

    resolve = _import_resolve()
    with patch("spyglass.common.LabMember", lab_member_mock), patch(
        "spyglass.common.common_nwbfile.AnalysisRegistry", registry_mock
    ), patch("spyglass.common.common_nwbfile.schema", schema_mock), patch(
        "spyglass.utils.dj_helper_fn.Path", path_mock
    ), patch(
        "spyglass.utils.dj_helper_fn.dj", dj_mock
    ):
        with pytest.raises(ValueError, match="Multiple entries"):
            resolve("/fake/path/file.nwb", "file.nwb", location="analysis")


def test_analysis_single_entry_updates():
    """A single matching entry should trigger update1 on the external table."""
    base_key = {
        "filepath": "/fake/path/file.nwb",
        "size": 0,
        "contents_hash": "old",
    }
    single_ext = _make_external_mock(1, fetch1_result=base_key)
    lab_member_mock, path_mock, dj_mock = _make_common_mocks()

    registry_mock = MagicMock()
    registry_mock.return_value.get_externals.return_value = [single_ext]
    schema_mock = MagicMock()

    resolve = _import_resolve()
    with patch("spyglass.common.LabMember", lab_member_mock), patch(
        "spyglass.common.common_nwbfile.AnalysisRegistry", registry_mock
    ), patch("spyglass.common.common_nwbfile.schema", schema_mock), patch(
        "spyglass.utils.dj_helper_fn.Path", path_mock
    ), patch(
        "spyglass.utils.dj_helper_fn.dj", dj_mock
    ):
        resolve("/fake/path/file.nwb", "file.nwb", location="analysis")

    single_ext.update1.assert_called_once()
    updated_key = single_ext.update1.call_args[0][0]
    assert updated_key["size"] == 42
    assert updated_key["contents_hash"] == "abc123"


def test_analysis_skips_empty_externals():
    """Empty externals are skipped; only the non-empty one triggers update."""
    base_key = {
        "filepath": "/fake/path/file.nwb",
        "size": 0,
        "contents_hash": "old",
    }
    empty_ext = _make_external_mock(0)
    single_ext = _make_external_mock(1, fetch1_result=base_key)
    lab_member_mock, path_mock, dj_mock = _make_common_mocks()

    registry_mock = MagicMock()
    registry_mock.return_value.get_externals.return_value = [empty_ext, single_ext]
    schema_mock = MagicMock()

    resolve = _import_resolve()
    with patch("spyglass.common.LabMember", lab_member_mock), patch(
        "spyglass.common.common_nwbfile.AnalysisRegistry", registry_mock
    ), patch("spyglass.common.common_nwbfile.schema", schema_mock), patch(
        "spyglass.utils.dj_helper_fn.Path", path_mock
    ), patch(
        "spyglass.utils.dj_helper_fn.dj", dj_mock
    ):
        resolve("/fake/path/file.nwb", "file.nwb", location="analysis")

    empty_ext.update1.assert_not_called()
    single_ext.update1.assert_called_once()


# ---------------------------------------------------------------------------
# Tests — raw location
# ---------------------------------------------------------------------------


def _make_raw_schema_mock(raw_ext):
    """Build a schema mock whose external["raw"] returns raw_ext."""
    schema_mock = MagicMock()
    schema_mock.external.__getitem__.return_value = raw_ext
    # Also make `& file_restr` return raw_ext itself
    raw_ext.__and__.return_value = raw_ext
    return schema_mock


def test_raw_no_entries_warns(caplog):
    """When raw external has no matching entry, log a warning and return."""
    empty_raw = _make_external_mock(0)
    lab_member_mock, path_mock, dj_mock = _make_common_mocks()
    schema_mock = _make_raw_schema_mock(empty_raw)
    registry_mock = MagicMock()

    resolve = _import_resolve()
    with patch("spyglass.common.LabMember", lab_member_mock), patch(
        "spyglass.common.common_nwbfile.AnalysisRegistry", registry_mock
    ), patch("spyglass.common.common_nwbfile.schema", schema_mock), patch(
        "spyglass.utils.dj_helper_fn.Path", path_mock
    ), patch(
        "spyglass.utils.dj_helper_fn.dj", dj_mock
    ):
        with caplog.at_level("WARNING"):
            resolve("/fake/path/file.nwb", "file.nwb", location="raw")

    assert "No entries found" in caplog.text


def test_raw_multiple_entries_raises():
    """When raw external has >1 matching entry, raise ValueError."""
    multi_raw = _make_external_mock(2)
    lab_member_mock, path_mock, dj_mock = _make_common_mocks()
    schema_mock = _make_raw_schema_mock(multi_raw)
    registry_mock = MagicMock()

    resolve = _import_resolve()
    with patch("spyglass.common.LabMember", lab_member_mock), patch(
        "spyglass.common.common_nwbfile.AnalysisRegistry", registry_mock
    ), patch("spyglass.common.common_nwbfile.schema", schema_mock), patch(
        "spyglass.utils.dj_helper_fn.Path", path_mock
    ), patch(
        "spyglass.utils.dj_helper_fn.dj", dj_mock
    ):
        with pytest.raises(ValueError, match="Multiple entries"):
            resolve("/fake/path/file.nwb", "file.nwb", location="raw")


def test_raw_single_entry_updates():
    """A single matching raw entry should trigger update1."""
    base_key = {
        "filepath": "/fake/path/file.nwb",
        "size": 0,
        "contents_hash": "old",
    }
    single_raw = _make_external_mock(1, fetch1_result=base_key)
    lab_member_mock, path_mock, dj_mock = _make_common_mocks()
    schema_mock = _make_raw_schema_mock(single_raw)
    registry_mock = MagicMock()

    resolve = _import_resolve()
    with patch("spyglass.common.LabMember", lab_member_mock), patch(
        "spyglass.common.common_nwbfile.AnalysisRegistry", registry_mock
    ), patch("spyglass.common.common_nwbfile.schema", schema_mock), patch(
        "spyglass.utils.dj_helper_fn.Path", path_mock
    ), patch(
        "spyglass.utils.dj_helper_fn.dj", dj_mock
    ):
        resolve("/fake/path/file.nwb", "file.nwb", location="raw")

    single_raw.update1.assert_called_once()
    updated_key = single_raw.update1.call_args[0][0]
    assert updated_key["size"] == 42
    assert updated_key["contents_hash"] == "abc123"
