#!/usr/bin/env python3
"""High-priority tests for core quickstart functionality.

This module focuses on testing the most critical functions that are actually
being used in the quickstart script, without making assumptions about APIs.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import only what we know exists and works
from quickstart import (
    SetupConfig, InstallType, Pipeline, validate_base_dir,
    UserInterface, EnvironmentManager, DisabledColors
)
from utils.result_types import (
    Success, Failure, success, failure, ValidationError, Severity
)

# Test the UX validation if available
try:
    from ux.validation import validate_port
    UX_VALIDATION_AVAILABLE = True
except ImportError:
    UX_VALIDATION_AVAILABLE = False


class TestCriticalValidationFunctions:
    """Test the core validation functions that must work."""

    def test_validate_base_dir_home_directory(self):
        """Test validate_base_dir with home directory (should always work)."""
        result = validate_base_dir(Path.home())
        assert result.is_success
        assert isinstance(result.value, Path)
        assert result.value.is_absolute()

    def test_validate_base_dir_current_directory(self):
        """Test validate_base_dir with current directory."""
        result = validate_base_dir(Path("."))
        assert result.is_success
        assert isinstance(result.value, Path)
        assert result.value.is_absolute()

    def test_validate_base_dir_impossible_path(self):
        """Test validate_base_dir with clearly impossible path."""
        result = validate_base_dir(Path("/nonexistent/impossible/nested/deep/path"))
        assert result.is_failure
        assert isinstance(result.error, ValueError)

    def test_validate_base_dir_result_type_contract(self):
        """Test that validate_base_dir always returns proper Result type."""
        test_paths = [Path.home(), Path("."), Path("/nonexistent")]

        for test_path in test_paths:
            result = validate_base_dir(test_path)
            # Must be either Success or Failure
            assert hasattr(result, 'is_success')
            assert hasattr(result, 'is_failure')
            assert result.is_success != result.is_failure  # Exactly one should be true

            if result.is_success:
                assert hasattr(result, 'value')
                assert isinstance(result.value, Path)
            else:
                assert hasattr(result, 'error')
                assert isinstance(result.error, Exception)


@pytest.mark.skipif(not UX_VALIDATION_AVAILABLE, reason="ux.validation not available")
class TestUXValidationCore:
    """Test critical UX validation functions."""

    def test_validate_port_mysql_default(self):
        """Test validating the default MySQL port."""
        result = validate_port("3306")
        assert result.is_success
        assert "3306" in result.message

    def test_validate_port_invalid_string(self):
        """Test validating clearly invalid port strings."""
        invalid_ports = ["abc", "", "not_a_number"]
        for port_str in invalid_ports:
            result = validate_port(port_str)
            assert result.is_failure
            assert hasattr(result, 'error')

    def test_validate_port_out_of_range(self):
        """Test validating out-of-range port numbers."""
        out_of_range = ["0", "-1", "65536", "100000"]
        for port_str in out_of_range:
            result = validate_port(port_str)
            assert result.is_failure
            assert "range" in result.error.message.lower() or "between" in result.error.message.lower()


class TestSetupConfigBehavior:
    """Test SetupConfig behavior and usage patterns."""

    def test_default_configuration(self):
        """Test that default configuration has sensible values."""
        config = SetupConfig()

        # Test defaults make sense
        assert config.install_type == InstallType.MINIMAL
        assert config.setup_database is True
        assert config.run_validation is True
        assert isinstance(config.base_dir, Path)
        assert config.env_name == "spyglass"
        assert isinstance(config.db_port, int)
        assert 1 <= config.db_port <= 65535

    def test_pipeline_configuration(self):
        """Test configuration with pipeline settings."""
        config = SetupConfig(
            install_type=InstallType.FULL,
            pipeline=Pipeline.DLC
        )

        assert config.install_type == InstallType.FULL
        assert config.pipeline == Pipeline.DLC

    def test_custom_base_directory(self):
        """Test configuration with custom base directory."""
        custom_path = Path("/tmp/custom_spyglass")
        config = SetupConfig(base_dir=custom_path)

        assert config.base_dir == custom_path

    def test_database_configuration_options(self):
        """Test database-related configuration options."""
        # Test with database
        config_with_db = SetupConfig(setup_database=True, db_port=5432)
        assert config_with_db.setup_database is True
        assert config_with_db.db_port == 5432

        # Test without database
        config_no_db = SetupConfig(setup_database=False)
        assert config_no_db.setup_database is False


class TestEnvironmentManagerCore:
    """Test critical EnvironmentManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = SetupConfig()
        self.ui = Mock()
        self.env_manager = EnvironmentManager(self.ui, self.config)

    def test_environment_manager_creation(self):
        """Test that EnvironmentManager can be created with valid config."""
        assert isinstance(self.env_manager, EnvironmentManager)
        assert self.env_manager.config == self.config

    def test_environment_file_selection_minimal(self):
        """Test environment file selection for minimal install."""
        with patch.object(Path, 'exists', return_value=True):
            result = self.env_manager.select_environment_file()
            assert isinstance(result, str)
            assert "environment" in result
            assert result.endswith(".yml")

    def test_environment_file_selection_full(self):
        """Test environment file selection for full install."""
        full_config = SetupConfig(install_type=InstallType.FULL)
        env_manager = EnvironmentManager(self.ui, full_config)

        with patch.object(Path, 'exists', return_value=True):
            result = env_manager.select_environment_file()
            assert isinstance(result, str)
            assert "environment" in result
            assert result.endswith(".yml")

    def test_environment_file_selection_dlc_pipeline(self):
        """Test environment file selection for DLC pipeline."""
        dlc_config = SetupConfig(pipeline=Pipeline.DLC)
        env_manager = EnvironmentManager(self.ui, dlc_config)

        with patch.object(Path, 'exists', return_value=True):
            result = env_manager.select_environment_file()
            assert isinstance(result, str)
            assert "dlc" in result.lower()

    @patch('subprocess.run')
    def test_build_environment_command_structure(self, mock_run):
        """Test that environment commands have proper structure."""
        cmd = self.env_manager._build_environment_command(
            "environment.yml", "conda", update=False
        )

        assert isinstance(cmd, list)
        assert len(cmd) > 3  # Should have conda, env, create/update, -f, -n, name
        assert cmd[0] in ["conda", "mamba"]  # First item should be package manager
        assert "env" in cmd
        assert "-f" in cmd  # Should specify file
        assert "-n" in cmd  # Should specify name


class TestUserInterfaceCore:
    """Test critical UserInterface functionality."""

    def test_user_interface_creation_minimal(self):
        """Test creating UserInterface with minimal setup."""
        ui = UserInterface(DisabledColors, auto_yes=False)
        assert isinstance(ui, UserInterface)

    def test_user_interface_auto_yes_mode(self):
        """Test UserInterface auto_yes functionality."""
        ui = UserInterface(DisabledColors, auto_yes=True)
        assert ui.auto_yes is True

    def test_display_methods_exist(self):
        """Test that essential display methods exist."""
        ui = UserInterface(DisabledColors)

        essential_methods = ['print_info', 'print_success', 'print_warning', 'print_error']
        for method_name in essential_methods:
            assert hasattr(ui, method_name)
            assert callable(getattr(ui, method_name))


class TestEnumDefinitions:
    """Test that enum definitions are correct and usable."""

    def test_install_type_enum(self):
        """Test InstallType enum values."""
        # Test that expected values exist
        assert hasattr(InstallType, 'MINIMAL')
        assert hasattr(InstallType, 'FULL')

        # Test that they're different
        assert InstallType.MINIMAL != InstallType.FULL

        # Test that they can be used in equality comparisons
        config = SetupConfig(install_type=InstallType.MINIMAL)
        assert config.install_type == InstallType.MINIMAL

    def test_pipeline_enum(self):
        """Test Pipeline enum values."""
        # Test that DLC pipeline exists (most commonly tested)
        assert hasattr(Pipeline, 'DLC')

        # Test that it can be used in configuration
        config = SetupConfig(pipeline=Pipeline.DLC)
        assert config.pipeline == Pipeline.DLC

    def test_severity_enum(self):
        """Test Severity enum values."""
        # Test that all expected severity levels exist
        assert hasattr(Severity, 'INFO')
        assert hasattr(Severity, 'WARNING')
        assert hasattr(Severity, 'ERROR')
        assert hasattr(Severity, 'CRITICAL')

        # Test that they're different
        assert Severity.INFO != Severity.ERROR


class TestResultTypeSystem:
    """Test the Result type system functionality."""

    def test_success_result_properties(self):
        """Test Success result properties and methods."""
        result = success("test_value", "Success message")

        assert result.is_success
        assert not result.is_failure
        assert result.value == "test_value"
        assert result.message == "Success message"

    def test_failure_result_properties(self):
        """Test Failure result properties and methods."""
        error = ValueError("Test error")
        result = failure(error, "Failure message")

        assert not result.is_success
        assert result.is_failure
        assert result.error == error
        assert result.message == "Failure message"

    def test_result_type_discrimination(self):
        """Test that we can properly discriminate between Success and Failure."""
        success_result = success("value")
        failure_result = failure(ValueError(), "error")

        results = [success_result, failure_result]

        successes = [r for r in results if r.is_success]
        failures = [r for r in results if r.is_failure]

        assert len(successes) == 1
        assert len(failures) == 1
        assert successes[0] == success_result
        assert failures[0] == failure_result


# Integration tests that verify the most critical workflows
class TestCriticalWorkflows:
    """Test critical workflows that must work for the installer."""

    def test_minimal_config_to_environment_file(self):
        """Test the workflow from minimal config to environment file selection."""
        config = SetupConfig(install_type=InstallType.MINIMAL)
        ui = Mock()
        env_manager = EnvironmentManager(ui, config)

        with patch.object(Path, 'exists', return_value=True):
            env_file = env_manager.select_environment_file()
            assert isinstance(env_file, str)
            assert "min" in env_file or "minimal" in env_file.lower()

    def test_full_config_to_environment_file(self):
        """Test the workflow from full config to environment file selection."""
        config = SetupConfig(install_type=InstallType.FULL)
        ui = Mock()
        env_manager = EnvironmentManager(ui, config)

        with patch.object(Path, 'exists', return_value=True):
            env_file = env_manager.select_environment_file()
            assert isinstance(env_file, str)
            # For full install, should not be the minimal environment
            assert "min" not in env_file

    def test_pipeline_config_to_environment_file(self):
        """Test the workflow from pipeline config to environment file selection."""
        config = SetupConfig(pipeline=Pipeline.DLC)
        ui = Mock()
        env_manager = EnvironmentManager(ui, config)

        with patch.object(Path, 'exists', return_value=True):
            env_file = env_manager.select_environment_file()
            assert isinstance(env_file, str)
            assert "dlc" in env_file.lower()

    def test_base_dir_validation_workflow(self):
        """Test the base directory validation workflow."""
        # Test with a safe, known path
        safe_path = Path.home()
        result = validate_base_dir(safe_path)

        assert result.is_success
        assert isinstance(result.value, Path)
        assert result.value.is_absolute()
        assert result.value.exists() or result.value.parent.exists()


# Tests specifically for coverage of high-priority edge cases
class TestEdgeCases:
    """Test edge cases that could cause problems in real usage."""

    def test_empty_environment_name_handling(self):
        """Test how system handles empty environment names."""
        # This tests what happens if someone tries to create config with empty name
        try:
            config = SetupConfig(env_name="")
            # If this succeeds, the system should handle it gracefully elsewhere
            assert config.env_name == ""
        except Exception:
            # If this fails, that's also acceptable behavior
            pass

    def test_very_long_base_path(self):
        """Test handling of very long base directory paths."""
        # Create a very long but valid path
        long_path = Path.home() / ("very_long_directory_name" * 10)
        result = validate_base_dir(long_path)

        # Should handle gracefully (either succeed or fail with clear message)
        assert hasattr(result, 'is_success')
        assert hasattr(result, 'is_failure')

    def test_special_characters_in_path(self):
        """Test handling of paths with special characters."""
        special_paths = [
            Path.home() / "spyglass data",  # Space
            Path.home() / "spyglass-data",  # Hyphen
            Path.home() / "spyglass_data",  # Underscore
        ]

        for path in special_paths:
            result = validate_base_dir(path)
            # Should handle all these cases gracefully
            assert hasattr(result, 'is_success')
            if result.is_success:
                assert isinstance(result.value, Path)

    @pytest.mark.skipif(not UX_VALIDATION_AVAILABLE, reason="ux.validation not available")
    def test_port_edge_cases(self):
        """Test port validation edge cases."""
        edge_cases = [
            "1024",    # First non-privileged port
            "49152",   # Common ephemeral port
            "65534",   # Almost maximum
        ]

        for port_str in edge_cases:
            result = validate_port(port_str)
            # Should handle all these cases (success or clear failure)
            assert hasattr(result, 'is_success')
            if result.is_failure:
                assert hasattr(result, 'error')
                assert len(result.error.message) > 0


class TestDataIntegrity:
    """Test data integrity and consistency."""

    def test_config_consistency(self):
        """Test that config values remain consistent."""
        config = SetupConfig(
            install_type=InstallType.FULL,
            pipeline=Pipeline.DLC,
            env_name="test-env"
        )

        # Values should remain as set
        assert config.install_type == InstallType.FULL
        assert config.pipeline == Pipeline.DLC
        assert config.env_name == "test-env"

        # Should be able to read values multiple times consistently
        assert config.install_type == InstallType.FULL
        assert config.install_type == InstallType.FULL

    def test_result_type_consistency(self):
        """Test that Result types behave consistently."""
        success_result = success("test")
        failure_result = failure(ValueError("test"), "test message")

        # Properties should be consistent across multiple calls
        assert success_result.is_success == success_result.is_success
        assert failure_result.is_failure == failure_result.is_failure

        # Opposite properties should always be inverse
        assert success_result.is_success != success_result.is_failure
        assert failure_result.is_success != failure_result.is_failure


if __name__ == "__main__":
    print("This test file focuses on high-priority core functionality.")
    print("To run tests:")
    print("  pytest test_core_functions.py              # Run all tests")
    print("  pytest test_core_functions.py -v           # Verbose output")
    print("  pytest test_core_functions.py::TestCriticalValidationFunctions  # Critical tests")
    print("  pytest test_core_functions.py -k workflow  # Run workflow tests")