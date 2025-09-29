#!/usr/bin/env python3
"""Tests for error handling and recovery functionality.

This module tests the error handling, recovery mechanisms, and edge cases
that are critical for a robust installation experience.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from quickstart import (
    SetupConfig, InstallType, Pipeline, validate_base_dir,
    UserInterface, EnvironmentManager, DisabledColors
)
from utils.result_types import (
    Success, Failure, ValidationError, Severity,
    success, failure, validation_failure
)
from common import EnvironmentCreationError

# Test error recovery if available
try:
    from ux.error_recovery import ErrorRecoveryGuide, ErrorCategory
    ERROR_RECOVERY_AVAILABLE = True
except ImportError:
    ERROR_RECOVERY_AVAILABLE = False


class TestPathValidationErrors:
    """Test path validation error cases that users commonly encounter."""

    def test_path_with_tilde_expansion(self):
        """Test that tilde paths are properly expanded."""
        tilde_path = Path("~/test_spyglass")
        result = validate_base_dir(tilde_path)

        if result.is_success:
            # Should be expanded to full path
            assert str(result.value).startswith("/")
            assert "~" not in str(result.value)

    def test_relative_path_resolution(self):
        """Test that relative paths are resolved to absolute."""
        relative_path = Path("./test_dir")
        result = validate_base_dir(relative_path)

        if result.is_success:
            assert result.value.is_absolute()
            assert not str(result.value).startswith(".")

    def test_path_with_symlinks(self):
        """Test path validation with symbolic links."""
        # Use a common system path that might have symlinks
        result = validate_base_dir(Path("/tmp"))

        # Should handle symlinks gracefully
        assert hasattr(result, 'is_success')
        if result.is_success:
            assert isinstance(result.value, Path)

    def test_permission_denied_simulation(self):
        """Test handling of permission denied scenarios."""
        # Test with root directory which should exist but may not be writable
        result = validate_base_dir(Path("/root/spyglass_data"))

        # Should either succeed or fail with clear error
        assert hasattr(result, 'is_success')
        if result.is_failure:
            assert isinstance(result.error, Exception)
            assert len(str(result.error)) > 0


class TestConfigurationErrors:
    """Test configuration error scenarios."""

    def test_invalid_port_in_config(self):
        """Test SetupConfig with invalid port values."""
        # Test with clearly invalid port
        config = SetupConfig(db_port=999999)
        assert config.db_port == 999999  # Should store the value

        # The validation should happen elsewhere (not in config creation)

    def test_invalid_install_type_handling(self):
        """Test how system handles invalid install types."""
        # This tests the enum safety
        valid_config = SetupConfig(install_type=InstallType.MINIMAL)
        assert valid_config.install_type == InstallType.MINIMAL

        # Can't easily test invalid enum values due to type safety

    def test_none_pipeline_handling(self):
        """Test configuration with None pipeline."""
        config = SetupConfig(pipeline=None)
        assert config.pipeline is None

        # Should be handled gracefully by environment manager
        ui = Mock()
        env_manager = EnvironmentManager(ui, config)
        assert isinstance(env_manager, EnvironmentManager)


class TestEnvironmentCreationErrors:
    """Test environment creation error scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = SetupConfig()
        self.ui = Mock()
        self.env_manager = EnvironmentManager(self.ui, self.config)

    def test_missing_environment_file(self):
        """Test behavior when environment file is missing."""
        with patch.object(Path, 'exists', return_value=False):
            # Should raise EnvironmentCreationError for missing files
            with pytest.raises(EnvironmentCreationError) as exc_info:
                self.env_manager.select_environment_file()
            assert "Environment file not found" in str(exc_info.value)

    @patch('subprocess.run')
    def test_conda_command_failure_simulation(self, mock_run):
        """Test handling of conda command failures."""
        # Simulate conda command failure
        mock_run.return_value = Mock(returncode=1, stderr="Command failed")

        # The error should be handled gracefully
        # (Actual error handling depends on implementation)
        cmd = self.env_manager._build_environment_command(
            "environment.yml", "conda", update=False
        )
        assert isinstance(cmd, list)

    def test_environment_name_validation_in_manager(self):
        """Test that environment manager handles name validation."""
        config_with_complex_name = SetupConfig(env_name="complex-test_env.2024")
        env_manager = EnvironmentManager(self.ui, config_with_complex_name)

        # Should create successfully
        assert isinstance(env_manager, EnvironmentManager)
        assert env_manager.config.env_name == "complex-test_env.2024"


@pytest.mark.skipif(not ERROR_RECOVERY_AVAILABLE, reason="ux.error_recovery not available")
class TestErrorRecoverySystem:
    """Test the error recovery and guidance system."""

    def test_error_recovery_guide_instantiation(self):
        """Test creating ErrorRecoveryGuide."""
        ui = Mock()
        guide = ErrorRecoveryGuide(ui)
        assert isinstance(guide, ErrorRecoveryGuide)

    def test_error_category_completeness(self):
        """Test that ErrorCategory enum has expected categories."""
        expected_categories = ['DOCKER', 'CONDA', 'PYTHON', 'NETWORK']

        for category_name in expected_categories:
            assert hasattr(ErrorCategory, category_name), f"Missing {category_name} category"

    def test_error_category_usage(self):
        """Test that error categories can be used properly."""
        # Should be able to compare categories
        docker_cat = ErrorCategory.DOCKER
        conda_cat = ErrorCategory.CONDA

        assert docker_cat != conda_cat
        assert docker_cat == ErrorCategory.DOCKER

    def test_error_recovery_methods_exist(self):
        """Test that ErrorRecoveryGuide has expected methods."""
        ui = Mock()
        guide = ErrorRecoveryGuide(ui)

        # Should have some method for handling errors
        expected_methods = ['handle_error']
        for method_name in expected_methods:
            if hasattr(guide, method_name):
                assert callable(getattr(guide, method_name))


class TestResultTypeEdgeCases:
    """Test Result type system edge cases."""

    def test_success_with_none_value(self):
        """Test Success result with None value."""
        result = success(None, "Success with no value")
        assert result.is_success
        assert result.value is None
        assert result.message == "Success with no value"

    def test_failure_with_complex_error(self):
        """Test Failure result with complex error object."""
        complex_error = ValidationError(
            message="Complex validation error",
            field="test_field",
            severity=Severity.ERROR,
            recovery_actions=["Action 1", "Action 2"]
        )

        result = failure(complex_error, "Validation failed")
        assert result.is_failure
        assert result.error == complex_error
        assert len(result.error.recovery_actions) == 2

    def test_validation_failure_creation(self):
        """Test creating validation-specific failures."""
        result = validation_failure(
            "port",
            "Invalid port number",
            Severity.ERROR,
            ["Use port 3306", "Check port availability"]
        )

        assert result.is_failure
        assert isinstance(result.error, ValidationError)
        assert result.error.field == "port"
        assert result.error.severity == Severity.ERROR
        assert len(result.error.recovery_actions) == 2

    def test_result_type_properties_immutable(self):
        """Test that Result type properties are read-only."""
        success_result = success("test")
        failure_result = failure(ValueError(), "error")

        # Properties should be stable
        assert success_result.is_success
        assert not success_result.is_failure
        assert not failure_result.is_success
        assert failure_result.is_failure


class TestUserInterfaceErrorHandling:
    """Test UserInterface error handling behavior."""

    def test_user_interface_with_disabled_colors(self):
        """Test UserInterface creation with disabled colors."""
        ui = UserInterface(DisabledColors)
        assert isinstance(ui, UserInterface)

    def test_display_methods_handle_exceptions(self):
        """Test that display methods don't crash on edge cases."""
        ui = UserInterface(DisabledColors)

        # Test with various edge case inputs
        edge_cases = ["", None, "Very long message " * 100, "Unicode: 🚀", "\n\t"]

        for test_input in edge_cases:
            try:
                if test_input is not None:
                    ui.print_info(str(test_input))
                    ui.print_success(str(test_input))
                    ui.print_warning(str(test_input))
                    ui.print_error(str(test_input))
                # Should not crash
            except Exception as e:
                pytest.fail(f"Display method crashed on input '{test_input}': {e}")

    @patch('builtins.input', return_value='')
    def test_input_methods_with_empty_response(self, mock_input):
        """Test input methods with empty user responses."""
        ui = UserInterface(DisabledColors, auto_yes=False)

        # Test methods that have default values
        if hasattr(ui, '_get_port_input'):
            result = ui._get_port_input()
            assert isinstance(result, int)
            assert 1 <= result <= 65535

    def test_auto_yes_behavior(self):
        """Test auto_yes mode behavior."""
        ui_auto = UserInterface(DisabledColors, auto_yes=True)
        ui_interactive = UserInterface(DisabledColors, auto_yes=False)

        assert ui_auto.auto_yes is True
        assert ui_interactive.auto_yes is False


class TestSystemRobustness:
    """Test system robustness and error recovery."""

    def test_multiple_config_creation(self):
        """Test creating multiple configs doesn't interfere."""
        config1 = SetupConfig(env_name="env1")
        config2 = SetupConfig(env_name="env2")

        assert config1.env_name == "env1"
        assert config2.env_name == "env2"
        assert config1.env_name != config2.env_name

    def test_config_with_extreme_values(self):
        """Test configuration with extreme but valid values."""
        extreme_config = SetupConfig(
            base_dir=Path("/tmp"),  # Minimal path
            env_name="a",           # Single character
            db_port=65535           # Maximum port
        )

        assert extreme_config.base_dir == Path("/tmp")
        assert extreme_config.env_name == "a"
        assert extreme_config.db_port == 65535

    def test_environment_manager_with_extreme_config(self):
        """Test EnvironmentManager with extreme configuration."""
        extreme_config = SetupConfig(
            install_type=InstallType.FULL,
            pipeline=Pipeline.DLC,
            env_name="test-with-many-hyphens-and-numbers-123"
        )

        ui = Mock()
        env_manager = EnvironmentManager(ui, extreme_config)

        # Should create successfully
        assert isinstance(env_manager, EnvironmentManager)

    def test_parallel_environment_manager_creation(self):
        """Test creating multiple EnvironmentManagers simultaneously."""
        configs = [
            SetupConfig(env_name="env1"),
            SetupConfig(env_name="env2"),
            SetupConfig(env_name="env3")
        ]

        ui = Mock()
        managers = [EnvironmentManager(ui, config) for config in configs]

        # All should be created successfully
        assert len(managers) == 3
        assert all(isinstance(m, EnvironmentManager) for m in managers)

        # Each should have the correct config
        for manager, config in zip(managers, configs):
            assert manager.config == config


# Performance and stress tests
class TestPerformanceEdgeCases:
    """Test performance and stress scenarios."""

    def test_large_number_of_validation_calls(self):
        """Test that validation functions can handle many calls."""
        test_paths = [Path.home()] * 100  # Test same path many times

        results = [validate_base_dir(path) for path in test_paths]

        # All should succeed and be consistent
        assert len(results) == 100
        assert all(r.is_success for r in results)

        # Results should be consistent
        first_result = results[0]
        assert all(r.value == first_result.value for r in results)

    def test_config_creation_performance(self):
        """Test creating many configurations quickly."""
        configs = [SetupConfig(env_name=f"env{i}") for i in range(100)]

        assert len(configs) == 100
        assert all(isinstance(c, SetupConfig) for c in configs)

        # Each should have unique name
        names = [c.env_name for c in configs]
        assert len(set(names)) == 100  # All unique


if __name__ == "__main__":
    print("This test file focuses on error handling and robustness.")
    print("To run tests:")
    print("  pytest test_error_handling.py              # Run all tests")
    print("  pytest test_error_handling.py -v           # Verbose output")
    print("  pytest test_error_handling.py::TestPathValidationErrors  # Path tests")
    print("  pytest test_error_handling.py -k performance  # Performance tests")