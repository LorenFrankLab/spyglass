"""Tests for common_file_tracking.py - AnalysisFileIssues functionality.

Tests custom analysis table support for file tracking, including:
1. check1_file() - Single file checking
2. check_files() - Batch file checking per table
3. get_tbl() - Finding downstream tables with file references
4. show_downstream() - Showing affected downstream tables
5. Integration with AnalysisNwbfile.check_all_files()
6. _check_path() - Thread-safe existence and checksum check
7. _batch_resolve_paths() - Batch path/hash resolution from external table
8. _get_recompute_deleted() - Fetching recompute-deleted file names
9. resolve_table_refs() - Populating the table field post-insertion
10. check_files() - limit, deleted_files, and checksum detection
"""

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    from spyglass.common import Nwbfile  # noqa: F401
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
    # create() only writes to disk; add() registers it in the main table so
    # check_files() can find it via fetch("analysis_file_name").
    analysis_tbl.add(mini_copy_name, analysis_file_name)

    yield analysis_file_name, analysis_tbl

    # Cleanup
    if teardown:
        (
            analysis_tbl & {"analysis_file_name": analysis_file_name}
        ).delete_quick()
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
    # Always clear before test to avoid stale entries from prior --no-teardown runs
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
    # Always clear before test to avoid stale entries from prior --no-teardown runs
    (
        analysis_file_issues & {"full_table_name": analysis_tbl.full_table_name}
    ).delete_quick()

    # Run check_files - should find no issues with the specific valid file.
    # Restrict to this file only: the table is shared across test modules, so
    # checking all entries would include files from other tests that may be
    # missing (e.g. after --no-teardown reruns).
    restricted_tbl = analysis_tbl & {"analysis_file_name": analysis_file_name}
    issue_count = analysis_file_issues.check_files(restricted_tbl)

    # Valid files should return 0 issues
    assert issue_count == 0


def test_get_tbl_method_exists(analysis_file_issues):
    """Test that get_tbl method exists and is callable."""
    assert hasattr(analysis_file_issues, "get_tbl")
    assert callable(analysis_file_issues.get_tbl)


def test_show_downstream_no_issues(analysis_file_issues):
    """Test show_downstream() with no issues."""
    # Call with restriction that matches nothing
    result = analysis_file_issues.show_downstream(
        restriction={"analysis_file_name": "definitely_nonexistent_file.nwb"}
    )

    assert isinstance(result, list) and not result, "Should be an empty list"


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

    # Always clear before test to avoid stale entries from prior --no-teardown runs
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

    # Always clear before test to avoid stale entries from prior --no-teardown runs
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

    # Always clear before test to avoid stale entries from prior --no-teardown runs
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


# ======================= _check_path (unit tests, no DB) ==========================


def test_check_path_deleted_file(file_tracking_module):
    """_check_path returns None for files in the deleted_files set."""
    result = file_tracking_module._check_path(
        fname="/irrelevant/path.nwb",
        analysis_file_name="path.nwb",
        full_table_name="`test`.`table`",
        deleted_files={"path.nwb"},
        hash_map={},
    )
    assert result is None


def test_check_path_missing_file(file_tracking_module, tmp_path):
    """_check_path returns a not-on-disk issue dict when file is absent."""
    missing = str(tmp_path / "missing.nwb")
    result = file_tracking_module._check_path(
        fname=missing,
        analysis_file_name="missing.nwb",
        full_table_name="`test`.`table`",
        deleted_files=set(),
        hash_map={},
    )
    assert result is not None
    assert result["on_disk"] is False
    assert result["can_read"] is False
    assert "path not found" in result["issue"]


def test_check_path_existing_no_hash(file_tracking_module, tmp_path):
    """_check_path returns None when file exists and no hash is stored."""
    f = tmp_path / "file.nwb"
    f.write_bytes(b"content")
    result = file_tracking_module._check_path(
        fname=str(f),
        analysis_file_name="file.nwb",
        full_table_name="`test`.`table`",
        deleted_files=set(),
        hash_map={},
    )
    assert result is None


def test_check_path_checksum_match(file_tracking_module, tmp_path):
    """_check_path returns None when file exists and hash matches stored value."""
    from datajoint.hash import uuid_from_file

    f = tmp_path / "file.nwb"
    f.write_bytes(b"content")
    result = file_tracking_module._check_path(
        fname=str(f),
        analysis_file_name="file.nwb",
        full_table_name="`test`.`table`",
        deleted_files=set(),
        hash_map={"file.nwb": uuid_from_file(str(f))},
    )
    assert result is None


def test_check_path_checksum_mismatch(file_tracking_module, tmp_path):
    """_check_path returns a checksum issue dict when stored hash doesn't match."""
    f = tmp_path / "file.nwb"
    f.write_bytes(b"content")
    result = file_tracking_module._check_path(
        fname=str(f),
        analysis_file_name="file.nwb",
        full_table_name="`test`.`table`",
        deleted_files=set(),
        hash_map={"file.nwb": uuid.uuid4()},  # wrong hash
    )
    assert result is not None
    assert result["on_disk"] is True
    assert result["can_read"] is False
    assert "checksum mismatch" in result["issue"]


# ======================= _batch_resolve_paths (unit tests) ==========================


def test_batch_resolve_paths_registered(file_tracking_module, tmp_path):
    """_batch_resolve_paths resolves paths from the external table."""
    mock_tbl = MagicMock()
    mock_tbl._analysis_dir = str(tmp_path)
    mock_tbl._get_analysis_file_paths.return_value = [
        "session_abc/session_abc_123.nwb"
    ]
    mock_tbl._ext_tbl.__and__.return_value.fetch.return_value = [
        {"filepath": "session_abc/session_abc_123.nwb", "contents_hash": None}
    ]
    path_map, hash_map = file_tracking_module._batch_resolve_paths(
        mock_tbl, ["session_abc_123.nwb"]
    )
    assert path_map["session_abc_123.nwb"] == str(
        tmp_path / "session_abc" / "session_abc_123.nwb"
    )
    assert "session_abc_123.nwb" not in hash_map  # NULL hash excluded


def test_batch_resolve_paths_unregistered_fallback(
    file_tracking_module, tmp_path
):
    """_batch_resolve_paths generates a subdirectory path for unregistered files."""
    mock_tbl = MagicMock()
    mock_tbl._analysis_dir = str(tmp_path)
    fname = "session_abc_123.nwb"
    mock_tbl._get_analysis_file_paths.return_value = [f"session_abc/{fname}"]
    mock_tbl._ext_tbl.__and__.return_value.fetch.return_value = []

    path_map, _ = file_tracking_module._batch_resolve_paths(mock_tbl, [fname])
    expected_subdir = fname[: fname.rfind("_")]
    assert path_map[fname] == str(tmp_path / expected_subdir / fname)


def test_batch_resolve_paths_hash_map(file_tracking_module, tmp_path):
    """_batch_resolve_paths populates hash_map for non-NULL contents_hash."""
    stored = uuid.uuid4()
    mock_tbl = MagicMock()
    mock_tbl._analysis_dir = str(tmp_path)
    mock_tbl._get_analysis_file_paths.return_value = [
        "session_abc/session_abc_123.nwb"
    ]
    mock_tbl._ext_tbl.__and__.return_value.fetch.return_value = [
        {"filepath": "session_abc/session_abc_123.nwb", "contents_hash": stored}
    ]
    _, hash_map = file_tracking_module._batch_resolve_paths(
        mock_tbl, ["session_abc_123.nwb"]
    )
    assert hash_map["session_abc_123.nwb"] == stored


# ======================= _get_recompute_deleted ==========================


def test_get_recompute_deleted_returns_set(file_tracking_module):
    """_get_recompute_deleted always returns a set, even on import failure."""
    with patch.dict(
        "sys.modules", {"spyglass.spikesorting.v1.recompute": None}
    ):
        result = file_tracking_module._get_recompute_deleted()
    assert isinstance(result, set)


# ======================= resolve_table_refs ==========================


def test_resolve_table_refs_no_null_entries(analysis_file_issues):
    """resolve_table_refs returns 0 when no entries have NULL table field."""
    result = analysis_file_issues.resolve_table_refs(
        {"analysis_file_name": "definitely_nonexistent_file.nwb"}
    )
    assert result == 0


def test_resolve_table_refs_populates_table(
    analysis_file_issues, test_analysis_file, teardown
):
    """resolve_table_refs populates the table field for matching children."""
    analysis_file_name, analysis_tbl = test_analysis_file

    key = {
        "full_table_name": analysis_tbl.full_table_name,
        "analysis_file_name": analysis_file_name,
    }
    (analysis_file_issues & key).delete_quick()

    # Insert an issue with table=NULL
    analysis_file_issues.insert1(
        dict(key, on_disk=False, can_read=False, issue="test for resolve")
    )
    assert (analysis_file_issues & key).fetch1("table") is None

    # Mock AnalysisRegistry and children so the child "matches" the file
    mock_child = MagicMock()
    mock_child.full_table_name = "`test`.`child_table`"
    # Configure the MagicMock's __and__ operator so (child & file_keys).fetch(...)
    # returns the analysis_file_name as expected by resolve_table_refs().
    mock_child.__and__.return_value.fetch.return_value = [analysis_file_name]

    with patch(
        "spyglass.common.common_file_tracking.AnalysisRegistry"
    ) as mock_registry:
        mock_registry.return_value.get_class.return_value = MagicMock(
            return_value=MagicMock(
                children=MagicMock(return_value=[mock_child])
            )
        )
        updated = analysis_file_issues.resolve_table_refs(key)

    assert updated == 1
    assert (analysis_file_issues & key).fetch1(
        "table"
    ) == "`test`.`child_table`"

    if teardown:
        (analysis_file_issues & key).delete_quick()


# ======================= check_files new behaviors ==========================


def test_check_files_limit(analysis_file_issues, test_analysis_file):
    """check_files(limit=0) checks no files and returns 0."""
    _, analysis_tbl = test_analysis_file
    restricted = analysis_tbl & {"analysis_file_name": "nonexistent_file.nwb"}
    result = analysis_file_issues.check_files(restricted, limit=0)
    assert result == 0


def test_check_files_skips_deleted_files(
    analysis_file_issues, test_analysis_file, tmp_path, teardown
):
    """check_files skips files present in deleted_files set."""
    analysis_file_name, analysis_tbl = test_analysis_file

    key = {
        "full_table_name": analysis_tbl.full_table_name,
        "analysis_file_name": analysis_file_name,
    }
    (analysis_file_issues & key).delete_quick()

    # Move file away so it would normally be flagged as missing
    file_path = Path(analysis_tbl.get_abs_path(analysis_file_name))
    temp_path = file_path.with_suffix(".nwb.tmp")
    file_path.rename(temp_path)

    try:
        restricted = analysis_tbl & {"analysis_file_name": analysis_file_name}
        count = analysis_file_issues.check_files(
            restricted,
            deleted_files={
                analysis_file_name
            },  # tell check_files it was deleted
        )
        assert count == 0
        assert len(analysis_file_issues & key) == 0
    finally:
        temp_path.rename(file_path)


def test_check_files_checksum_mismatch(
    analysis_file_issues, test_analysis_file, teardown
):
    """check_files records a checksum issue when the stored hash doesn't match."""
    analysis_file_name, analysis_tbl = test_analysis_file

    key = {
        "full_table_name": analysis_tbl.full_table_name,
        "analysis_file_name": analysis_file_name,
    }
    (analysis_file_issues & key).delete_quick()

    wrong_hash = uuid.uuid4()
    restricted = analysis_tbl & {"analysis_file_name": analysis_file_name}

    abs_path = analysis_tbl.get_abs_path(analysis_file_name)
    with patch.object(
        type(analysis_file_issues),
        "_batch_resolve_paths",
        new=staticmethod(
            lambda *a: (
                {analysis_file_name: abs_path},
                {analysis_file_name: wrong_hash},
            )
        ),
    ):
        count = analysis_file_issues.check_files(restricted)

    assert count == 1
    entry = (analysis_file_issues & key).fetch1()
    assert entry["on_disk"] is True or entry["on_disk"] == 1
    assert "checksum mismatch" in entry["issue"]

    if teardown:
        (analysis_file_issues & key).delete_quick()
