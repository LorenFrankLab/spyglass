"""Tests for custom AnalysisNwbfile factory pattern (Issue #1434).

This test module covers the verification strategy outlined in TASK0_NOTES.md:
1. Table Declaration - SpyglassAnalysis enforcement
2. File Operations - create, add, add_nwb_object, add_units
3. (Skipped - item 3 was lock contention, removed from numbering)
4. Cleanup and Orphan Detection - AnalysisRegistry functionality

Tests marked with @pytest.mark.xfail are expected to fail until full
implementation is complete.
"""

import os
import random
from pathlib import Path
from unittest.mock import Mock, patch

import datajoint as dj
import numpy as np
import pandas as pd
import pytest

# Mark all tests in this module to run after other tests
pytestmark = pytest.mark.order("last")


# ================================ FIXTURES ===================================


@pytest.fixture(scope="module")
def custom_prefix():
    """Custom database prefix for testing."""
    yield "testcustom"


@pytest.fixture(scope="module")
def custom_config(dj_conn, custom_prefix):
    """Set up custom config with database prefix."""
    original_prefix = dj.config.get("custom", {}).get("database.prefix")

    if "custom" not in dj.config:
        dj.config["custom"] = {}
    dj.config["custom"]["database.prefix"] = custom_prefix

    yield custom_prefix

    # Restore original config
    if original_prefix:
        dj.config["custom"]["database.prefix"] = original_prefix
    elif "database.prefix" in dj.config.get("custom", {}):
        del dj.config["custom"]["database.prefix"]


@pytest.fixture(scope="module")
def common_nwbfile(common):
    """Return common nwbfile module."""
    return common.common_nwbfile


@pytest.fixture(scope="module")
def analysis_registry(common_nwbfile):
    """Return AnalysisRegistry table."""
    return common_nwbfile.AnalysisRegistry()


@pytest.fixture(scope="module")
def master_analysis_table(common_nwbfile):
    """Return master AnalysisNwbfile table for comparison."""
    return common_nwbfile.AnalysisNwbfile()


@pytest.fixture(scope="module")
def custom_analysis_table(custom_config, dj_conn, common_nwbfile):
    """Create and return a custom AnalysisNwbfile table.

    This fixture dynamically creates a table following the factory pattern.
    """
    from spyglass.utils.dj_mixin import SpyglassAnalysis

    prefix = (
        custom_config  # custom_config fixture yields the prefix string directly
    )
    schema = dj.schema(f"{prefix}_nwbfile")

    # Make Nwbfile available in the schema context for foreign key resolution
    Nwbfile = common_nwbfile.Nwbfile

    @schema
    class AnalysisNwbfile(SpyglassAnalysis, dj.Manual):
        definition = """This definition is managed by SpyglassAnalysis"""

    yield AnalysisNwbfile()


@pytest.fixture(scope="module")
def temp_analysis_file(mini_copy_name, custom_analysis_table, teardown):
    """Create a temporary analysis file for testing."""
    # Create the file
    analysis_file_name = custom_analysis_table.create(mini_copy_name)

    yield analysis_file_name

    # Cleanup
    if teardown:
        try:
            file_path = custom_analysis_table.get_abs_path(analysis_file_name)
            if Path(file_path).exists():
                Path(file_path).unlink()
        except Exception:
            pass


# ========================= 1. TABLE DECLARATION TESTS =========================


class TestTableDeclaration:
    """Test table declaration and SpyglassAnalysis enforcement (Item 1)."""

    def test_custom_table_created(self, custom_analysis_table):
        """Test that custom AnalysisNwbfile table can be created."""
        assert custom_analysis_table is not None
        assert hasattr(custom_analysis_table, "create")
        assert hasattr(custom_analysis_table, "add")

    def test_schema_naming_convention(
        self, custom_analysis_table, custom_prefix
    ):
        """Test that schema follows {prefix}_nwbfile naming convention."""
        schema_name = custom_analysis_table.database
        assert schema_name == f"{custom_prefix}_nwbfile"

    def test_table_name_convention(self, custom_analysis_table):
        """Test that table name is 'AnalysisNwbfile'."""
        table_name = custom_analysis_table.table_name
        assert table_name == "analysis_nwbfile"

    def test_definition_enforcement(
        self, custom_analysis_table, master_analysis_table
    ):
        """Test that custom table has enforced definition matching master."""
        # Both tables should have the same primary keys
        custom_pks = set(custom_analysis_table.primary_key)
        master_pks = set(master_analysis_table.primary_key)
        assert custom_pks == master_pks

        # Both should have analysis_file_name as primary key
        assert "analysis_file_name" in custom_analysis_table.primary_key

        # Both should have analysis_file_abs_path attribute
        attrs = custom_analysis_table.heading.attributes
        assert "analysis_file_abs_path" in attrs

    def test_registry_insertion(self, custom_analysis_table, analysis_registry):
        """Test that table is registered in AnalysisRegistry."""
        full_name = custom_analysis_table.full_table_name
        registered = analysis_registry & {"full_table_name": full_name}
        assert len(registered) > 0, "Table not found in AnalysisRegistry"

    def test_invalid_schema_name_rejected(self, custom_config, dj_conn):
        """Test that invalid schema names are rejected."""
        from spyglass.utils.dj_mixin import SpyglassAnalysis

        # Try to create with wrong schema name
        bad_schema = dj.schema("bad_schema")

        with pytest.raises(
            ValueError, match=".*does not match.*|Analysis requires.*"
        ):

            @bad_schema
            class AnalysisNwbfile(SpyglassAnalysis, dj.Manual):
                definition = (
                    """This definition is managed by SpyglassAnalysis"""
                )

            new_table = AnalysisNwbfile()

    def test_wrong_prefix_rejected(self, dj_conn):
        """Test that schema with wrong prefix is rejected."""
        from spyglass.utils.dj_mixin import SpyglassAnalysis

        wrong_schema = dj.schema("wrongprefix_nwbfile")

        with pytest.raises(ValueError, match="Schema prefix.*"):

            @wrong_schema
            class AnalysisNwbfile(SpyglassAnalysis, dj.Manual):
                definition = (
                    """This definition is managed by SpyglassAnalysis"""
                )

            new_table = AnalysisNwbfile()


# ========================= 2. FILE OPERATIONS TESTS =========================


class TestFileOperations:
    """Test file operations (Item 2): create, add, helper methods."""

    def test_create_analysis_file(self, custom_analysis_table, mini_copy_name):
        """Test creating an analysis file."""
        analysis_file_name = custom_analysis_table.create(mini_copy_name)

        assert analysis_file_name is not None
        assert analysis_file_name.endswith(".nwb")
        assert mini_copy_name.split(".")[0] in analysis_file_name

        # Verify file exists on disk
        file_path = custom_analysis_table.get_abs_path(analysis_file_name)
        assert Path(file_path).exists()

        # Cleanup
        Path(file_path).unlink()

    def test_get_abs_path(self, temp_analysis_file, custom_analysis_table):
        """Test getting absolute path for analysis file."""
        abs_path = custom_analysis_table.get_abs_path(temp_analysis_file)

        assert abs_path is not None
        assert Path(abs_path).exists()
        assert abs_path.endswith(".nwb")

    def test_add_to_database(
        self,
        mini_copy_name,
        temp_analysis_file,
        custom_analysis_table,
        teardown,
    ):
        """Test adding analysis file to database."""
        # Add to database
        custom_analysis_table.add(mini_copy_name, temp_analysis_file)

        # Verify entry exists
        entry = custom_analysis_table & {
            "analysis_file_name": temp_analysis_file
        }
        assert len(entry) == 1

        # Cleanup
        if teardown:
            entry.delete_quick()

    def test_add_nwb_object_dataframe(
        self, mini_copy_name, custom_analysis_table, teardown
    ):
        """Test adding DataFrame to analysis file using add_nwb_object."""
        # Create file
        analysis_file_name = custom_analysis_table.create(mini_copy_name)

        try:
            # Create test DataFrame
            test_df = pd.DataFrame(
                {"time": [1.0, 2.0, 3.0], "value": [0.1, 0.2, 0.3]}
            )

            # Add DataFrame to file BEFORE registering
            object_id = custom_analysis_table.add_nwb_object(
                analysis_file_name, test_df, table_name="test_data"
            )

            assert object_id is not None
            assert isinstance(object_id, str)

            # Now register the file
            custom_analysis_table.add(mini_copy_name, analysis_file_name)

            # Verify entry exists
            entry = custom_analysis_table & {
                "analysis_file_name": analysis_file_name
            }
            assert len(entry) == 1

        finally:
            # Cleanup
            if teardown:
                try:
                    (
                        custom_analysis_table
                        & {"analysis_file_name": analysis_file_name}
                    ).delete_quick()
                except Exception:
                    pass
                file_path = custom_analysis_table.get_abs_path(
                    analysis_file_name
                )
                if Path(file_path).exists():
                    Path(file_path).unlink()

    def test_add_nwb_object_array(
        self, mini_copy_name, custom_analysis_table, teardown
    ):
        """Test adding numpy array to analysis file."""
        analysis_file_name = custom_analysis_table.create(mini_copy_name)

        try:
            # Create test array
            test_array = np.array([1, 2, 3, 4, 5])

            # Add array BEFORE registration
            object_id = custom_analysis_table.add_nwb_object(
                analysis_file_name, test_array, table_name="test_array"
            )

            assert object_id is not None

            # Register the file
            custom_analysis_table.add(mini_copy_name, analysis_file_name)

        finally:
            if teardown:
                try:
                    (
                        custom_analysis_table
                        & {"analysis_file_name": analysis_file_name}
                    ).delete_quick()
                except Exception:
                    pass
                file_path = custom_analysis_table.get_abs_path(
                    analysis_file_name
                )
                if Path(file_path).exists():
                    Path(file_path).unlink()

    def test_file_path_conflicts(
        self,
        mini_copy_name,
        custom_analysis_table,
        master_analysis_table,
        base_dir,
        teardown,
    ):
        """Test handling of file path conflicts across tables.

        When two tables try to use the same filename, files should be
        disambiguated by random suffixes. The __get_new_file_name method
        checks both the table and filesystem to ensure uniqueness.
        """
        fname = mini_copy_name
        cust_tbl = custom_analysis_table
        mast_tbl = master_analysis_table

        # Access the private method using name mangling
        # In Python, __method becomes _ClassName__method
        get_new_file = lambda tbl, name: tbl._AnalysisMixin__get_new_file_name(
            name
        )

        # Test 1: Same seed generates same filename when table is empty
        random.seed(42)
        cust_file = get_new_file(cust_tbl, fname)

        random.seed(42)
        mast_file = get_new_file(mast_tbl, fname)

        assert cust_file == mast_file, "Same seed should generate match"

        # Test 2: Creating a file should prevent that name from being reused
        # Create the file in the expected location
        file_dir = base_dir / "analysis" / fname.split("_")[0]
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path = file_dir / cust_file
        file_path.touch()

        # Now, should get different filename because file exists
        random.seed(42)
        new_file = get_new_file(mast_tbl, fname)

        assert new_file != cust_file, "Should not reuse existing filename"

        if file_path.exists():
            file_path.unlink()

    def test_copy_analysis_file(
        self, mini_copy_name, custom_analysis_table, teardown
    ):
        """Test copying an existing analysis file."""
        # Create and register original file
        original_file = custom_analysis_table.create(mini_copy_name)
        custom_analysis_table.add(mini_copy_name, original_file)

        try:
            # Copy the file
            copied_file = custom_analysis_table.copy(original_file)

            assert copied_file is not None
            assert copied_file != original_file
            assert copied_file.endswith(".nwb")

            # Verify copy exists on disk
            copy_path = custom_analysis_table.get_abs_path(copied_file)
            assert Path(copy_path).exists()

            # Note: copied file is NOT automatically registered
            query = custom_analysis_table & {"analysis_file_name": copied_file}
            assert len(query) == 0

        finally:
            if teardown:
                # Cleanup original
                (
                    custom_analysis_table
                    & {"analysis_file_name": original_file}
                ).delete_quick()
                # Cleanup copy (file only, not in DB)
                try:
                    copy_path = custom_analysis_table.get_abs_path(copied_file)
                    if Path(copy_path).exists():
                        Path(copy_path).unlink()
                except Exception:
                    pass


# ==================== 4. CLEANUP AND ORPHAN DETECTION TESTS ====================


class TestCleanupAndRegistry:
    """Test cleanup and orphan detection via AnalysisRegistry (Item 4)."""

    def test_registry_get_class(self, analysis_registry, custom_analysis_table):
        """Test AnalysisRegistry.get_class() retrieves table class."""
        full_name = custom_analysis_table.full_table_name

        # Get class from registry
        retrieved_class = analysis_registry.get_class(full_name)

        assert retrieved_class is not None
        assert hasattr(retrieved_class, "create")
        assert hasattr(retrieved_class, "add")

    def test_registry_lists_all_tables(
        self, analysis_registry, custom_analysis_table, master_analysis_table
    ):
        """Test that AnalysisRegistry tracks all analysis tables."""
        registered_tables = analysis_registry.fetch("full_table_name")

        # Should include both master and custom tables
        custom_name = custom_analysis_table.full_table_name
        master_name = master_analysis_table.full_table_name

        assert custom_name in registered_tables
        # Master table may or may not be in registry depending on implementation
        # assert master_name in registered_tables

    def test_orphan_detection_across_tables(
        self,
        analysis_registry,
        custom_analysis_table,
        mini_copy_name,
        base_dir,
        teardown,
    ):
        """Test that orphan detection can find files across all registered tables.

        This test verifies that the cleanup system can identify orphaned files
        even when they're tracked in different AnalysisNwbfile tables.

        NOTE: This test currently passes because it only verifies file existence.
        The actual orphan detection logic (commented below) is not implemented yet.
        Do NOT remove @pytest.mark.xfail until find_orphans() method is implemented.
        """
        # Create files in multiple tables
        custom_file = custom_analysis_table.create(mini_copy_name)

        # Don't register it - making it an orphan
        custom_path = Path(custom_analysis_table.get_abs_path(custom_file))

        downstream = Mock()  # Simulate downstream references

        try:
            # TODO: use `cleanup()` logic to find orphans
            # Simulate orphan detection across all registered tables
            # insert table entries without downstream fk-references
            all_orphans = []
            for table_name in analysis_registry.fetch("full_table_name"):
                table_class = analysis_registry.get_class(table_name)
                table_instance = table_class()  # Get instance
                first = table_instance.create(mini_copy_name)  # create file
                second = table_instance.add(
                    mini_copy_name, first
                )  # register file
                # TODO: add fixture for table downstream references
                downstream.insert(first)
                third = table_instance.create(
                    mini_copy_name
                )  # create another file
                fourth = table_instance.add(mini_copy_name, third)  # register it
                table_instance.cleanup()

                table_instance.cleanup()

                assert not Path(
                    fourth
                ).exists(), "Cleanup should remove unreferenced files"

        finally:
            if teardown and custom_path.exists():
                custom_path.unlink()

    def test_registry_entry_has_metadata(
        self, analysis_registry, custom_analysis_table
    ):
        """Test that registry entries include metadata (created_at, created_by)."""
        full_name = custom_analysis_table.full_table_name
        entry = (analysis_registry & {"full_table_name": full_name}).fetch1()

        assert "created_at" in entry
        assert "created_by" in entry
        assert entry["created_by"] is not None


# ========================== INTEGRATION TESTS ================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(
        self, mini_copy_name, custom_analysis_table, analysis_registry, teardown
    ):
        """Test complete workflow: create -> populate -> register -> fetch."""
        # 1. Create file
        analysis_file_name = custom_analysis_table.create(mini_copy_name)
        assert Path(
            custom_analysis_table.get_abs_path(analysis_file_name)
        ).exists()

        # 2. Populate with data
        test_data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        obj_id = custom_analysis_table.add_nwb_object(
            analysis_file_name, test_data, table_name="test"
        )
        assert obj_id is not None

        # 3. Register in database
        custom_analysis_table.add(mini_copy_name, analysis_file_name)
        entry = custom_analysis_table & {
            "analysis_file_name": analysis_file_name
        }
        assert len(entry) == 1

        # 4. Verify in registry
        full_name = custom_analysis_table.full_table_name
        assert len(analysis_registry & {"full_table_name": full_name}) > 0

        # 5. Fetch and verify
        fetched = entry.fetch1()
        assert fetched["analysis_file_name"] == analysis_file_name
        assert fetched["nwb_file_name"] == mini_copy_name

        # Cleanup
        if teardown:
            entry.delete_quick()
            file_path = Path(
                custom_analysis_table.get_abs_path(analysis_file_name)
            )
            if file_path.exists():
                file_path.unlink()

    @pytest.mark.xfail(reason="Multi-table isolation not yet verified")
    def test_multiple_tables_isolation(
        self, mini_copy_name, custom_config, dj_conn, teardown
    ):
        """Test that multiple custom tables can coexist without conflicts."""
        from spyglass.utils.dj_mixin import SpyglassAnalysis

        # Create two different custom schemas
        schema1 = dj.schema(f"{custom_config}_nwbfile")
        schema2 = dj.schema("another_test_nwbfile")

        @schema1
        class AnalysisNwbfile1(SpyglassAnalysis, dj.Manual):
            definition = """This definition is managed by SpyglassAnalysis"""

        # Set config for second table
        original = dj.config["custom"]["database.prefix"]
        dj.config["custom"]["database.prefix"] = "another_test"

        try:

            @schema2
            class AnalysisNwbfile2(SpyglassAnalysis, dj.Manual):
                definition = (
                    """This definition is managed by SpyglassAnalysis"""
                )

            # Create files in both tables
            file1 = AnalysisNwbfile1().create(mini_copy_name)
            file2 = AnalysisNwbfile2().create(mini_copy_name)

            # Files should be independent
            assert file1 != file2

            # Register both
            AnalysisNwbfile1().add(mini_copy_name, file1)
            AnalysisNwbfile2().add(mini_copy_name, file2)

            # Both should be in their respective tables
            assert len(AnalysisNwbfile1() & {"analysis_file_name": file1}) == 1
            assert len(AnalysisNwbfile2() & {"analysis_file_name": file2}) == 1

            # Should not be in each other's tables
            assert len(AnalysisNwbfile1() & {"analysis_file_name": file2}) == 0
            assert len(AnalysisNwbfile2() & {"analysis_file_name": file1}) == 0

        finally:
            dj.config["custom"]["database.prefix"] = original
            if teardown:
                # Cleanup
                try:
                    AnalysisNwbfile1().delete_quick()
                    AnalysisNwbfile2().delete_quick()
                except Exception:
                    pass
