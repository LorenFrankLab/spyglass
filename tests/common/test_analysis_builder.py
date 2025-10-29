"""Tests for AnalysisFileBuilder context manager

Tests the builder's lifecycle enforcement (CREATE → POPULATE → REGISTER),
state machine validation, and integration with AnalysisMixin.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Mark all tests in this module to run after other tests
pytestmark = pytest.mark.order("last")


# ======================= 1. BASIC FUNCTIONALITY TESTS =======================


class TestBasicFunctionality:
    """Test basic builder functionality: creation, registration, paths."""

    def test_auto_registration_happy_path(
        self, master_analysis_table, mini_copy_name, mock_create, teardown
    ):
        """Test that builder auto-registers file on successful exit."""
        table = master_analysis_table
        mock_create(table)

        # Create file using builder
        with table.build(mini_copy_name) as builder:
            analysis_file = builder.analysis_file_name
            # File should be created but not registered yet
            assert analysis_file is not None
            assert Path(builder.get_path()).exists()

        # After exit, file should be auto-registered
        entry = table & {"analysis_file_name": analysis_file}
        assert len(entry) == 1, "File should be auto-registered"

        # Cleanup
        if teardown:
            entry.delete_quick()
            Path(table.get_abs_path(analysis_file)).unlink(missing_ok=True)

    def test_manual_registration(
        self, master_analysis_table, mini_copy_name, mock_create, teardown
    ):
        """Test that manual register() call works (idempotent)."""
        table = master_analysis_table
        mock_create(table)

        with table.build(mini_copy_name) as builder:
            analysis_file = builder.analysis_file_name

            # Call register() manually before exit
            builder.register()

            # Should be registered now
            entry = table & {"analysis_file_name": analysis_file}
            assert len(entry) == 1

            # Calling register() again should be idempotent (no error)
            builder.register()  # Second call does nothing

        # Should still be registered after exit
        entry = table & {"analysis_file_name": analysis_file}
        assert len(entry) == 1

        # Cleanup
        if teardown:
            entry.delete_quick()
            Path(table.get_abs_path(analysis_file)).unlink(missing_ok=True)

    def test_file_creation(
        self, master_analysis_table, mini_copy_name, mock_create, teardown
    ):
        """Test that builder creates file with correct name."""
        table = master_analysis_table
        mock_create(table)

        with table.build(mini_copy_name) as builder:
            analysis_file = builder.analysis_file_name

            # File should have correct naming pattern
            assert analysis_file is not None
            assert analysis_file.endswith(".nwb")
            assert mini_copy_name.split(".")[0] in analysis_file

            # Parent file name should be stored
            assert builder.nwb_file_name == mini_copy_name

        # Cleanup
        if teardown:
            (table & {"analysis_file_name": analysis_file}).delete_quick()
            Path(table.get_abs_path(analysis_file)).unlink(missing_ok=True)

    def test_path_retrieval(
        self, master_analysis_table, mini_copy_name, mock_create, teardown
    ):
        """Test that get_path() returns correct absolute path."""
        table = master_analysis_table
        mock_create(table)

        with table.build(mini_copy_name) as builder:
            analysis_file = builder.analysis_file_name
            path = builder.get_path()

            # Path should be absolute and exist
            assert path is not None
            assert os.path.isabs(path)
            assert Path(path).exists()

            # Path should match table's get_abs_path()
            expected_path = table.get_abs_path(analysis_file)
            assert path == expected_path

        # Cleanup
        if teardown:
            (table & {"analysis_file_name": analysis_file}).delete_quick()
            Path(table.get_abs_path(analysis_file)).unlink(missing_ok=True)


# ========================= 2. HELPER METHOD TESTS ============================


class TestHelperMethods:
    """Test builder helper methods: add_nwb_object, add_units, etc."""

    def test_add_nwb_object_before_registration(
        self,
        master_analysis_table,
        mini_copy_name,
        common,
        load_config,
        teardown,
    ):
        """Test add_nwb_object() works before registration."""
        table = master_analysis_table

        with table.build(mini_copy_name) as builder:
            analysis_file = builder.analysis_file_name

            # Add a DataFrame using builder
            test_df = pd.DataFrame(
                {"time": [1.0, 2.0, 3.0], "value": [0.1, 0.2, 0.3]}
            )
            object_id = builder.add_nwb_object(test_df, "test_data")

            # Should return object ID
            assert object_id is not None
            assert isinstance(object_id, str)

        # File should be registered after exit
        entry = table & {"analysis_file_name": analysis_file}
        assert len(entry) == 1

        # Cleanup
        if teardown:
            entry.delete_quick()
            Path(table.get_abs_path(analysis_file)).unlink(missing_ok=True)

    def test_add_units_before_registration(
        self,
        master_analysis_table,
        mini_copy_name,
        common,
        load_config,
        teardown,
    ):
        """Test add_units() works before registration."""
        table = master_analysis_table

        with table.build(mini_copy_name) as builder:
            analysis_file = builder.analysis_file_name

            # Add units using builder
            units = {0: np.array([1.0, 2.0, 3.0])}
            units_valid_times = {0: np.array([[0.0, 10.0]])}
            units_sort_interval = {0: np.array([0.0, 10.0])}

            units_id, waveforms_id = builder.add_units(
                units, units_valid_times, units_sort_interval
            )

            # Should return object IDs
            assert units_id is not None
            assert isinstance(units_id, str)

        # File should be registered after exit
        entry = table & {"analysis_file_name": analysis_file}
        assert len(entry) == 1

        # Cleanup
        if teardown:
            entry.delete_quick()
            Path(table.get_abs_path(analysis_file)).unlink(missing_ok=True)

    def test_multiple_helper_calls(
        self,
        master_analysis_table,
        mini_copy_name,
        common,
        load_config,
        teardown,
    ):
        """Test multiple helper calls in sequence."""
        table = master_analysis_table

        with table.build(mini_copy_name) as builder:
            analysis_file = builder.analysis_file_name

            # Add multiple objects
            test_df1 = pd.DataFrame({"x": [1, 2, 3]})
            test_df2 = pd.DataFrame({"y": [4, 5, 6]})

            obj_id1 = builder.add_nwb_object(test_df1, "data1")
            obj_id2 = builder.add_nwb_object(test_df2, "data2")

            # Both should succeed
            assert obj_id1 is not None
            assert obj_id2 is not None
            assert obj_id1 != obj_id2

        # File should be registered
        entry = table & {"analysis_file_name": analysis_file}
        assert len(entry) == 1

        # Cleanup
        if teardown:
            entry.delete_quick()
            Path(table.get_abs_path(analysis_file)).unlink(missing_ok=True)

    def test_open_for_write_before_registration(
        self,
        master_analysis_table,
        mini_copy_name,
        common,
        load_config,
        teardown,
    ):
        """Test open_for_write() works before registration."""
        table = master_analysis_table

        with table.build(mini_copy_name) as builder:
            analysis_file = builder.analysis_file_name

            # Open file for writing
            with builder.open_for_write() as io:
                nwbf = io.read()
                # Should be able to read the NWB file
                assert nwbf is not None

        # File should be registered after exit
        entry = table & {"analysis_file_name": analysis_file}
        assert len(entry) == 1

        # Cleanup
        if teardown:
            entry.delete_quick()
            Path(table.get_abs_path(analysis_file)).unlink(missing_ok=True)


# ======================== 3. STATE ENFORCEMENT TESTS =========================


class TestStateEnforcement:
    """Test state machine validation and error messages."""

    def test_error_on_add_nwb_object_after_registration(
        self, master_analysis_table, mini_copy_name, mock_create, teardown
    ):
        """Test that add_nwb_object() fails after registration."""
        table = master_analysis_table
        mock_create(table)

        with table.build(mini_copy_name) as builder:
            analysis_file = builder.analysis_file_name

            # Register early
            builder.register()

            # Try to add object after registration
            test_df = pd.DataFrame({"x": [1, 2, 3]})

            with pytest.raises(ValueError, match="REGISTERED"):
                builder.add_nwb_object(test_df, "data")

        # Cleanup
        if teardown:
            (table & {"analysis_file_name": analysis_file}).delete_quick()
            Path(table.get_abs_path(analysis_file)).unlink(missing_ok=True)

    def test_error_on_add_units_after_registration(
        self, master_analysis_table, mini_copy_name, mock_create, teardown
    ):
        """Test that add_units() fails after registration."""
        table = master_analysis_table
        mock_create(table)

        with table.build(mini_copy_name) as builder:
            analysis_file = builder.analysis_file_name

            # Register early
            builder.register()

            # Try to add units after registration
            units = {0: np.array([1.0, 2.0, 3.0])}
            units_valid_times = {0: np.array([[0.0, 10.0]])}
            units_sort_interval = {0: np.array([0.0, 10.0])}

            with pytest.raises(ValueError, match="REGISTERED"):
                builder.add_units(units, units_valid_times, units_sort_interval)

        # Cleanup
        if teardown:
            (table & {"analysis_file_name": analysis_file}).delete_quick()
            Path(table.get_abs_path(analysis_file)).unlink(missing_ok=True)

    def test_error_on_open_for_write_after_registration(
        self, master_analysis_table, mini_copy_name, mock_create, teardown
    ):
        """Test that open_for_write() fails after registration."""
        table = master_analysis_table
        mock_create(table)

        with table.build(mini_copy_name) as builder:
            analysis_file = builder.analysis_file_name

            # Register early
            builder.register()

            # Try to open file after registration
            with pytest.raises(ValueError, match="REGISTERED"):
                with builder.open_for_write() as io:
                    pass

        # Cleanup
        if teardown:
            (table & {"analysis_file_name": analysis_file}).delete_quick()
            Path(table.get_abs_path(analysis_file)).unlink(missing_ok=True)

    def test_error_on_operations_before_create(
        self, master_analysis_table, mini_copy_name
    ):
        """Test that helper methods fail before entering context."""
        table = master_analysis_table

        # Create builder but don't enter context
        builder = table.build(mini_copy_name)

        # Try to use methods before __enter__
        test_df = pd.DataFrame({"x": [1, 2, 3]})

        with pytest.raises(ValueError, match="before entering context"):
            builder.add_nwb_object(test_df, "data")

        with pytest.raises(ValueError, match="File not yet created"):
            builder.get_path()

    def test_idempotent_registration(
        self, master_analysis_table, mini_copy_name, mock_create, teardown
    ):
        """Test that register() can be called multiple times (idempotent)."""
        table = master_analysis_table
        mock_create(table)

        with table.build(mini_copy_name) as builder:
            analysis_file = builder.analysis_file_name

            # Call register() multiple times
            builder.register()
            builder.register()  # Should not raise error
            builder.register()  # Should not raise error

            # Should only be one entry
            entry = table & {"analysis_file_name": analysis_file}
            assert len(entry) == 1

        # Cleanup
        if teardown:
            entry.delete_quick()
            Path(table.get_abs_path(analysis_file)).unlink(missing_ok=True)


# ======================= 4. EXCEPTION HANDLING TESTS ========================


class TestExceptionHandling:
    """Test exception handling and cleanup logging."""

    def test_file_not_registered_on_exception(
        self, master_analysis_table, mini_copy_name, mock_create, teardown
    ):
        """Test that file is NOT registered when exception occurs."""
        table = master_analysis_table
        mock_create(table)

        analysis_file = None

        try:
            with table.build(mini_copy_name) as builder:
                analysis_file = builder.analysis_file_name

                # Simulate error during population
                raise ValueError("Simulated error")
        except ValueError:
            pass  # Expected

        # File should NOT be registered
        if analysis_file:
            entry = table & {"analysis_file_name": analysis_file}
            assert len(entry) == 0, "File should not be registered on exception"

            # Cleanup
            if teardown:
                Path(table.get_abs_path(analysis_file)).unlink(missing_ok=True)

    def test_failed_file_logged_for_cleanup(
        self,
        master_analysis_table,
        mini_copy_name,
        mock_create,
        teardown,
        caplog,
    ):
        """Test that failed files are logged for cleanup detection."""
        table = master_analysis_table
        mock_create(table)

        analysis_file = None

        import logging

        caplog.set_level(logging.WARNING)

        try:
            with table.build(mini_copy_name) as builder:
                analysis_file = builder.analysis_file_name

                # Simulate error
                raise RuntimeError("Test error")
        except RuntimeError:
            pass  # Expected

        # Check that warning was logged
        assert any(
            "not registered due to exception" in record.message
            for record in caplog.records
        ), "Should log warning about unregistered file"

        # Cleanup
        if teardown and analysis_file:
            Path(table.get_abs_path(analysis_file)).unlink(missing_ok=True)

    def test_no_auto_register_on_exception(
        self, master_analysis_table, mini_copy_name, mock_create, teardown
    ):
        """Test that auto-registration does not happen on exception."""
        table = master_analysis_table
        mock_create(table)

        analysis_file = None

        # Test with multiple types of exceptions
        for exc_class in [ValueError, RuntimeError, KeyError]:
            try:
                with table.build(mini_copy_name) as builder:
                    analysis_file = builder.analysis_file_name
                    raise exc_class("Test")
            except exc_class:
                pass

            # Should not be registered
            if analysis_file:
                entry = table & {"analysis_file_name": analysis_file}
                assert len(entry) == 0

                # Cleanup
                if teardown:
                    Path(table.get_abs_path(analysis_file)).unlink(
                        missing_ok=True
                    )

    def test_manual_register_after_exception(
        self, master_analysis_table, mini_copy_name, mock_create, teardown
    ):
        """Test that manual register() can be called in exception handler."""
        table = master_analysis_table
        mock_create(table)

        analysis_file = None

        # Simulate recovery after exception
        with table.build(mini_copy_name) as builder:
            analysis_file = builder.analysis_file_name

            # Even if we have an error, we can manually register if recovered
            try:  # Some operation that might fail
                raise RuntimeError("Simulated failure")
            except Exception:  # Recovery logic here...
                pass

            # Manually register after recovery
            builder.register()

        # Should be registered
        entry = table & {"analysis_file_name": analysis_file}
        assert len(entry) == 1

        # Cleanup
        if teardown:
            entry.delete_quick()
            Path(table.get_abs_path(analysis_file)).unlink(missing_ok=True)


# ========================= 5. INTEGRATION TESTS ==============================


class TestIntegration:
    """Integration tests with master and custom tables."""

    def test_with_master_table(
        self,
        master_analysis_table,
        mini_copy_name,
        common,
        load_config,
        teardown,
    ):
        """Test builder works with master AnalysisNwbfile table."""
        table = master_analysis_table

        with table.build(mini_copy_name) as builder:
            analysis_file = builder.analysis_file_name

            # Add test data
            test_df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
            obj_id = builder.add_nwb_object(test_df, "test")

            assert obj_id is not None

        # Verify registration in master table
        entry = table & {"analysis_file_name": analysis_file}
        assert len(entry) == 1

        # Cleanup
        if teardown:
            entry.delete_quick()
            Path(table.get_abs_path(analysis_file)).unlink(missing_ok=True)

    def test_with_custom_table(
        self,
        custom_analysis_table,
        mini_copy_name,
        common,
        load_config,
        teardown,
    ):
        """Test builder works with custom AnalysisNwbfile table."""
        table = custom_analysis_table

        with table.build(mini_copy_name) as builder:
            analysis_file = builder.analysis_file_name

            # Add test data
            test_array = np.array([1, 2, 3, 4, 5])
            obj_id = builder.add_nwb_object(test_array, "test_array")

            assert obj_id is not None

        # Verify registration in custom table
        entry = table & {"analysis_file_name": analysis_file}
        assert len(entry) == 1

        # Cleanup
        if teardown:
            entry.delete_quick()
            Path(table.get_abs_path(analysis_file)).unlink(missing_ok=True)

    def test_full_workflow(
        self,
        master_analysis_table,
        mini_copy_name,
        common,
        load_config,
        teardown,
    ):
        """Test complete workflow: create → populate → register → fetch."""
        table = master_analysis_table

        # 1. Create and populate
        with table.build(mini_copy_name) as builder:
            analysis_file = builder.analysis_file_name

            # 2. Add multiple objects
            df1 = pd.DataFrame({"time": [1.0, 2.0], "value": [0.1, 0.2]})
            df2 = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

            obj_id1 = builder.add_nwb_object(df1, "timeseries")
            obj_id2 = builder.add_nwb_object(df2, "position")

            assert obj_id1 is not None
            assert obj_id2 is not None

        # 3. Verify registration
        entry = table & {"analysis_file_name": analysis_file}
        assert len(entry) == 1

        # 4. Fetch and verify
        fetched = entry.fetch1()
        assert fetched["analysis_file_name"] == analysis_file
        assert fetched["nwb_file_name"] == mini_copy_name

        # 5. Verify file exists
        file_path = Path(table.get_abs_path(analysis_file))
        assert file_path.exists()

        # Cleanup
        if teardown:
            entry.delete_quick()
            file_path.unlink(missing_ok=True)

    def test_with_real_nwb_data(
        self,
        master_analysis_table,
        mini_copy_name,
        common,
        load_config,
        teardown,
    ):
        """Test builder with actual NWB file creation (not mocked)."""
        table = master_analysis_table

        # Use real create() method (no mock)
        with table.build(mini_copy_name) as builder:
            analysis_file = builder.analysis_file_name

            # Verify it's a real NWB file
            file_path = Path(builder.get_path())
            assert file_path.exists()
            assert file_path.suffix == ".nwb"

            # Add real data
            test_df = pd.DataFrame(
                {
                    "time": [1.0, 2.0, 3.0, 4.0],
                    "x": [0.1, 0.2, 0.3, 0.4],
                    "y": [1.1, 1.2, 1.3, 1.4],
                }
            )
            obj_id = builder.add_nwb_object(test_df, "position_data")
            assert obj_id is not None

        # File should be registered
        entry = table & {"analysis_file_name": analysis_file}
        assert len(entry) == 1

        # Cleanup
        if teardown:
            entry.delete_quick()
            Path(table.get_abs_path(analysis_file)).unlink(missing_ok=True)
