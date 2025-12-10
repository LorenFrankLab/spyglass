"""Tests for custom AnalysisNwbfile factory pattern (Issue #1434).

1. Table Declaration - SpyglassAnalysis enforcement
2. File Operations - create, add, add_nwb_object, add_units
3. Cleanup and Orphan Detection - AnalysisRegistry functionality
"""

import random
from pathlib import Path

import datajoint as dj
import numpy as np
import pandas as pd
import pytest

# Mark all tests in this module to run after other tests
pytestmark = pytest.mark.order("last")


# ================================ FIXTURES ===================================
# Common fixtures are now in conftest.py


@pytest.fixture(scope="module")
def custom_analysis_tables(custom_config, dj_conn, common_nwbfile):
    """Create and return custom AnalysisNwbfile table with downstream table.

    This fixture extends the base custom_analysis_table fixture from conftest.py
    by adding a CustomDownstream table for testing FK relationships.
    """
    from spyglass.utils.dj_mixin import SpyglassAnalysis

    prefix = custom_config
    schema = dj.schema(f"{prefix}_nwbfile")

    # Make Nwbfile available in the schema context for foreign key resolution
    Nwbfile = common_nwbfile.Nwbfile  # noqa F401
    _ = common_nwbfile.AnalysisRegistry().unblock_new_inserts()

    @schema
    class AnalysisNwbfile(SpyglassAnalysis, dj.Manual):
        definition = """This definition is managed by SpyglassAnalysis"""

    @schema
    class CustomDownstream(dj.Manual):
        definition = """
        foreign_id: int auto_increment
        -> AnalysisNwbfile
        """

        def insert_by_name(self, fname):
            super().insert1(dict(analysis_file_name=fname))

    yield AnalysisNwbfile(), CustomDownstream()


@pytest.fixture(scope="module")
def temp_analysis_file(mini_copy_name, custom_analysis_table, teardown):
    """Create a temporary analysis file for testing."""
    # Create the file
    analysis_file_name = custom_analysis_table.create(mini_copy_name)

    yield analysis_file_name

    # Cleanup
    if teardown:
        file_path = custom_analysis_table.get_abs_path(analysis_file_name)
        if Path(file_path).exists():
            Path(file_path).unlink()


# ========================= 1. TABLE DECLARATION TESTS =========================


class TestTableDeclaration:
    """Test table declaration and SpyglassAnalysis enforcement."""

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

            _ = AnalysisNwbfile()

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

            _ = AnalysisNwbfile()


# ========================= 2. FILE OPERATIONS TESTS =========================


class TestFileOperations:
    """Test file operations: create, add, helper methods."""

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
        analysis_dict = dict(analysis_file_name=analysis_file_name)

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
            entry = custom_analysis_table & analysis_dict
            assert len(entry) == 1

        finally:
            # Cleanup
            if teardown:
                (custom_analysis_table & analysis_dict).delete_quick()
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
        analysis_dict = dict(analysis_file_name=analysis_file_name)

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
                (custom_analysis_table & analysis_dict).delete_quick()
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

        # Test 1: Same seed generates different file bc file now exists
        random.seed(42)
        cust_file = get_new_file(cust_tbl, fname)

        random.seed(42)
        mast_file = get_new_file(mast_tbl, fname)

        assert cust_file != mast_file, "Same seed should not match"

        # Test 2: Grabbing a file name means it exists on disk as empty file
        file_dir = base_dir / "analysis" / fname.split("_")[0]
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path = file_dir / cust_file

        assert file_path.exists(), "File created when got file name"

        if file_path.exists():
            file_path.unlink()

    def test_copy_analysis_file(
        self, mini_copy_name, custom_analysis_table, teardown
    ):
        """Test copying an existing analysis file."""
        # Create and register original file
        original_file = custom_analysis_table.create(mini_copy_name)
        custom_analysis_table.add(mini_copy_name, original_file)

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

        if teardown:
            (
                custom_analysis_table & {"analysis_file_name": original_file}
            ).delete_quick()
            copy_path = custom_analysis_table.get_abs_path(copied_file)
            if Path(copy_path).exists():
                Path(copy_path).unlink()


# ==================== 4. CLEANUP AND ORPHAN DETECTION TESTS ===================


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

        custom_name = custom_analysis_table.full_table_name
        assert custom_name in registered_tables, "Custom table not in registry"

    def test_orphan_detection_across_tables(
        self,
        analysis_registry,
        custom_analysis_tables,
        mini_copy_name,
        base_dir,
        teardown,
        common,
        mock_create,
    ):
        """Test orphan detection across all registered tables.

        Case 1: File is created, but not added to analysis table (null).
        Case 2: File is created and added, but no downstream fk-ref (orphan).
        Case 3: File is created, added, and has downstream fk-ref (valid).
        Case 4: Valid file is copied to master table without downstream fk-ref
               (export master-orphan, but custom valid).

        This test verifies that the cleanup system can identify orphaned files
        even when they're tracked in different AnalysisNwbfile tables.

        Case 4 is relevant for ExportSelection.File usecase
        """

        master_table = common.common_nwbfile.AnalysisNwbfile()
        custom_table, downstream = custom_analysis_tables

        # Use mock_create fixture for fast file creation
        mock_create(custom_table)

        # Case 1: Create null file
        null_file = custom_table.create(mini_copy_name)
        null_fp = custom_table.get_abs_path(null_file)

        # Case 2: Create orphan file
        orph_file = custom_table.create(mini_copy_name)
        orph_fp = custom_table.get_abs_path(orph_file)
        custom_table.add(mini_copy_name, orph_file)

        # Case 3: Create valid file with downstream reference
        valid_file = custom_table.create(mini_copy_name)
        valid_fp = custom_table.get_abs_path(valid_file)
        custom_table.add(mini_copy_name, valid_file)
        downstream.insert_by_name(valid_file)

        # Case 4: Create valid file, copy to master table without downstream
        export_file = custom_table.create(mini_copy_name)
        export_fp = custom_table.get_abs_path(export_file)
        custom_table.add(mini_copy_name, export_file)
        downstream.insert_by_name(export_file)
        custom_table._copy_to_common(export_file)

        # TODO: use `cleanup()` logic to find orphans
        # Simulate orphan detection across all registered tables
        # insert table entries without downstream fk-references
        master_table.cleanup()

        assert not Path(null_fp).exists(), "Null file should be deleted"

        has_orph = custom_table.file_like(orph_file)
        assert not bool(has_orph), "Orphan should be detected"
        assert not Path(orph_fp).exists(), "Orphan file should be deleted"
        assert Path(valid_fp).exists(), "Valid file should remain"
        assert Path(export_fp).exists(), "Exported valid file should remain"

    def test_registry_entry_has_metadata(
        self, analysis_registry, custom_analysis_table
    ):
        """Test that registry entries include metadata."""
        full_name = custom_analysis_table.full_table_name
        entry = (analysis_registry & {"full_table_name": full_name}).fetch1()

        assert "created_at" in entry
        assert "created_by" in entry
        assert entry["created_by"] is not None

    def test_get_externals(
        self, analysis_registry, custom_analysis_table, master_analysis_table
    ):
        """Test AnalysisRegistry.get_externals() returns ExternalTable objects.

        Verifies that get_externals() can:
        1. Discover all registered analysis schemas
        2. Create ExternalTable objects for each schema
        3. Return valid ExternalTable instances with correct database names
        """
        # Get external tables for all registered schemas
        externals = analysis_registry.get_externals()

        # Should return a list
        assert isinstance(externals, list)

        # Should have at least one external (for custom table schema)
        assert len(externals) > 0

        # Each item should be an ExternalTable instance
        for ext in externals:
            assert isinstance(ext, dj.external.ExternalTable)
            # Should have a database attribute
            assert hasattr(ext, "database")
            # Database should end with _nwbfile
            assert ext.database.endswith("_nwbfile")

        # Get database names from externals
        external_dbs = {ext.database for ext in externals}

        # Extract database from custom table
        custom_db = custom_analysis_table.database

        # Custom table's database should have an external
        assert custom_db in external_dbs


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

    def test_multiple_tables_isolation(
        self,
        mini_copy_name,
        custom_config,
        dj_conn,
        teardown,
        common,
        mock_create,
    ):
        """Test that multiple custom tables can coexist without conflicts.

        This test verifies that the factory pattern allows multiple teams
        to have their own isolated AnalysisNwbfile tables without conflicts.
        """
        from spyglass.utils.dj_mixin import SpyglassAnalysis

        # For foreign key resolution
        Nwbfile = common.common_nwbfile.Nwbfile  # noqa F401

        # Create two different custom schemas
        schema1 = dj.schema(f"{custom_config}_nwbfile")

        # Save original config before creating any tables
        original = dj.config["custom"]["database.prefix"]

        @schema1
        class AnalysisNwbfile1(SpyglassAnalysis, dj.Manual):
            definition = """This definition is managed by SpyglassAnalysis"""

        # Set config for second table (must match schema name prefix)
        dj.config["custom"]["database.prefix"] = "anothertest"
        schema2 = dj.schema("anothertest_nwbfile")

        @schema2
        class AnalysisNwbfile2(SpyglassAnalysis, dj.Manual):
            definition = """This definition is managed by SpyglassAnalysis"""

        # Use mock_create for fast file creation
        mock_create(AnalysisNwbfile1())
        mock_create(AnalysisNwbfile2())

        # Create files in both tables
        file1 = AnalysisNwbfile1().create(mini_copy_name)
        file2 = AnalysisNwbfile2().create(mini_copy_name)

        # Files should be independent
        assert file1 != file2

        # Register both
        AnalysisNwbfile1().add(mini_copy_name, file1)
        AnalysisNwbfile2().add(mini_copy_name, file2)

        # Both should be in their respective tables
        f_dict1 = {"analysis_file_name": file1}
        f_dict2 = {"analysis_file_name": file2}
        assert len(AnalysisNwbfile1() & f_dict1) == 1
        assert len(AnalysisNwbfile2() & f_dict2) == 1

        # Should not be in each other's tables
        assert len(AnalysisNwbfile1() & f_dict2) == 0
        assert len(AnalysisNwbfile2() & f_dict1) == 0

        # Always cleanup registry entries (even if teardown=False)
        from spyglass.common.common_nwbfile import AnalysisRegistry

        names = tuple(
            (AnalysisNwbfile1.full_table_name, AnalysisNwbfile2.full_table_name)
        )
        (AnalysisRegistry & f"full_table_name IN {names}").delete_quick()

        # Restore original config
        dj.config["custom"]["database.prefix"] = original

        AnalysisNwbfile1().delete_quick()
        AnalysisNwbfile2().delete_quick()
