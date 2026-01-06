"""Tests for common_file_tracking.py - AnalysisFileIssues functionality.

Tests custom analysis table support for file tracking, including:
1. check1_file() - Single file checking
2. check_files() - Batch file checking per table
3. get_tbl() - Finding downstream tables with file references
4. show_downstream() - Showing affected downstream tables
5. Integration with AnalysisNwbfile.check_all_files()
"""

from pathlib import Path
from unittest.mock import patch

import datajoint as dj
import pytest

# Mark all tests in this module to run after other tests
pytestmark = pytest.mark.order("last")


# ================================ FIXTURES ===================================


@pytest.fixture(scope="module")
def file_tracking_module(dj_conn):
    """Import and return the file tracking module."""
    from spyglass.common.common_file_tracking import AnalysisFileIssues

    yield AnalysisFileIssues


@pytest.fixture(scope="module")
def analysis_file_issues(file_tracking_module, teardown):
    """Create AnalysisFileIssues instance and clean up after tests."""
    issues_table = file_tracking_module()

    yield issues_table

    # Cleanup: delete all test entries
    if teardown:
        issues_table.delete_quick()


@pytest.fixture(scope="module")
def custom_analysis_with_files(custom_config, dj_conn, common_nwbfile):
    """Create custom analysis table with test files."""
    from spyglass.utils.dj_mixin import SpyglassAnalysis

    prefix = custom_config
    schema = dj.schema(f"{prefix}_nwbfile")

    _ = common_nwbfile.AnalysisRegistry().unblock_new_inserts()

    @schema
    class AnalysisNwbfile(SpyglassAnalysis, dj.Manual):
        definition = """This definition is managed by SpyglassAnalysis"""

    @schema
    class FileConsumer(dj.Manual):
        """Test table that references analysis files."""

        definition = """
        id: int auto_increment
        -> AnalysisNwbfile
        """

    yield AnalysisNwbfile(), FileConsumer()


@pytest.fixture(scope="module")
def test_analysis_file(mini_copy_name, custom_analysis_with_files, teardown):
    """Create a test analysis file."""
    analysis_tbl, _ = custom_analysis_with_files
    analysis_file_name = analysis_tbl.create(mini_copy_name)

    yield analysis_file_name, analysis_tbl

    # Cleanup
    if teardown:
        file_path = analysis_tbl.get_abs_path(analysis_file_name)
        if Path(file_path).exists():
            Path(file_path).unlink()


# ================================ TESTS ===================================


def test_table_structure(analysis_file_issues):
    """Test that AnalysisFileIssues has the correct structure."""
    assert analysis_file_issues.table_name == "analysis_file_issues"

    # Check primary key
    primary_key = analysis_file_issues.primary_key
    assert "full_table_name" in primary_key
    assert "analysis_file_name" in primary_key

    # Check secondary attributes
    attrs = analysis_file_issues.heading.names
    assert "on_disk" in attrs
    assert "can_read" in attrs
    assert "issue" in attrs
    assert "table" in attrs


def test_check1_file_existing(
    analysis_file_issues, test_analysis_file, teardown
):
    """Test check1_file() with an existing, valid file."""
    analysis_file_name, analysis_tbl = test_analysis_file

    # Clear any previous entries for this file
    key = {
        "full_table_name": analysis_tbl.full_table_name,
        "analysis_file_name": analysis_file_name,
    }
    if teardown:
        (analysis_file_issues & key).delete_quick()

    # Check the file - should not find issues
    issue_found = analysis_file_issues.check1_file(
        analysis_tbl, analysis_file_name
    )

    # Valid file should not be inserted
    assert not issue_found
    assert len(analysis_file_issues & key) == 0


def test_check_files_with_valid_file(
    analysis_file_issues, test_analysis_file, teardown
):
    """Test check_files() with valid files returns 0."""
    analysis_file_name, analysis_tbl = test_analysis_file

    # Clear previous entries for this table
    if teardown:
        (
            analysis_file_issues
            & {"full_table_name": analysis_tbl.full_table_name}
        ).delete_quick()

    # Run check_files - should find no issues with valid file
    issue_count = analysis_file_issues.check_files(analysis_tbl)

    # Valid files should return 0 issues
    assert issue_count == 0


def test_get_tbl_method_exists(analysis_file_issues):
    """Test that get_tbl method exists and is callable."""
    assert hasattr(analysis_file_issues, "get_tbl")
    assert callable(analysis_file_issues.get_tbl)


def test_show_downstream_no_issues(analysis_file_issues, capsys):
    """Test show_downstream() with no issues."""
    # Call with restriction that matches nothing
    result = analysis_file_issues.show_downstream(
        restriction={"analysis_file_name": "definitely_nonexistent_file.nwb"}
    )

    captured = capsys.readouterr()
    assert "No issues found" in captured.out
    assert result is None


def test_integration_check_all_files_method_exists(common_nwbfile):
    """Test that check_all_files() method exists on AnalysisNwbfile."""
    analysis_nwbfile = common_nwbfile.AnalysisNwbfile()

    # Check method exists
    assert hasattr(analysis_nwbfile, "check_all_files")
    assert callable(analysis_nwbfile.check_all_files)


def test_integration_check_all_files_returns_dict(common_nwbfile):
    """Test that check_all_files() returns a dictionary."""
    # Run the check (may take a while with real data)
    results = common_nwbfile.AnalysisNwbfile().check_all_files()

    # Should return a dictionary
    assert isinstance(results, dict)

    # Should have at least the common table
    assert len(results) > 0


# ======================= NEGATIVE TEST CASES ==========================


def test_check1_file_missing_from_disk(
    analysis_file_issues, test_analysis_file, teardown
):
    """Test check1_file() when file doesn't exist on disk (line 123-127)."""
    analysis_file_name, analysis_tbl = test_analysis_file

    # Get the path of the test file and delete it temporarily
    test_file_path = Path(analysis_tbl.get_abs_path(analysis_file_name))

    # Rename the file temporarily to simulate it being missing
    temp_path = test_file_path.with_suffix(".nwb.tmp")
    test_file_path.rename(temp_path)

    key = {
        "full_table_name": analysis_tbl.full_table_name,
        "analysis_file_name": analysis_file_name,
    }

    # Clear any previous entries for this file
    if teardown:
        (analysis_file_issues & key).delete_quick()

    try:
        # Check the file - should detect it's missing
        issue_found = analysis_file_issues.check1_file(
            analysis_tbl, analysis_file_name
        )

        # Should find an issue
        assert issue_found is True

        # Should insert an entry with on_disk=False, can_read=False
        issue_entry = (analysis_file_issues & key).fetch1()
        assert not issue_entry["on_disk"]
        assert not issue_entry["can_read"]
        assert "path not found" in issue_entry["issue"]

    finally:
        # Restore the file
        temp_path.rename(test_file_path)

        # Cleanup issue entry
        if teardown:
            (analysis_file_issues & key).delete_quick()


def test_check1_file_filenotfound_error(
    analysis_file_issues, test_analysis_file, teardown
):
    """Test check1_file() when get_abs_path raises FileNotFoundError (line 128-130)."""
    analysis_file_name, analysis_tbl = test_analysis_file

    key = {
        "full_table_name": analysis_tbl.full_table_name,
        "analysis_file_name": analysis_file_name,
    }

    # Clear any previous entries for this file
    if teardown:
        (analysis_file_issues & key).delete_quick()

    # Mock get_abs_path to raise FileNotFoundError
    error_msg = "File path could not be constructed"
    with patch.object(
        analysis_tbl, "get_abs_path", side_effect=FileNotFoundError(error_msg)
    ):
        # Check the file - should catch FileNotFoundError
        issue_found = analysis_file_issues.check1_file(
            analysis_tbl, analysis_file_name
        )

    # Should find an issue
    assert issue_found is True

    # Should insert an entry with on_disk=False, can_read=False
    issue_entry = (analysis_file_issues & key).fetch1()
    assert not issue_entry["on_disk"]
    assert not issue_entry["can_read"]
    assert error_msg in issue_entry["issue"]

    # Cleanup
    if teardown:
        (analysis_file_issues & key).delete_quick()


def test_check1_file_checksum_error(
    analysis_file_issues, test_analysis_file, teardown
):
    """Test check1_file() when DataJointError is raised for checksum mismatch (line 131-138)."""
    analysis_file_name, analysis_tbl = test_analysis_file

    key = {
        "full_table_name": analysis_tbl.full_table_name,
        "analysis_file_name": analysis_file_name,
    }

    # Clear any previous entries for this file
    if teardown:
        (analysis_file_issues & key).delete_quick()

    # Mock get_abs_path to raise DataJointError (simulating checksum mismatch)
    error_msg = "Checksum mismatch for file"

    # Also mock get_tbl to return a dummy table name
    with (
        patch.object(
            analysis_tbl,
            "get_abs_path",
            side_effect=dj.DataJointError(error_msg),
        ),
        patch.object(
            analysis_file_issues, "get_tbl", return_value="dummy_child_table"
        ),
    ):
        # Check the file - should catch DataJointError
        issue_found = analysis_file_issues.check1_file(
            analysis_tbl, analysis_file_name
        )

    # Should find an issue
    assert issue_found is True

    # Should insert an entry with on_disk=True, can_read=False (checksum error)
    issue_entry = (analysis_file_issues & key).fetch1()
    assert issue_entry["on_disk"]  # Should be truthy (1 in MySQL)
    assert not issue_entry["can_read"]  # Should be falsy (0 in MySQL)
    assert error_msg in issue_entry["issue"]
    # Should also capture the downstream table name
    assert issue_entry["table"] == "dummy_child_table"

    # Cleanup
    if teardown:
        (analysis_file_issues & key).delete_quick()
