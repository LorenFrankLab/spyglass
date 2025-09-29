#!/usr/bin/env python3
"""Pytest tests for Spyglass quickstart script.

This replaces the unittest-based tests with pytest conventions as per CLAUDE.md.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from quickstart import (
    SetupConfig, InstallType, Pipeline,
    UserInterface, EnvironmentManager,
    validate_base_dir, DisabledColors
)


class TestSetupConfig:
    """Test the SetupConfig dataclass."""

    def test_default_values(self):
        """Test that SetupConfig has sensible defaults."""
        config = SetupConfig()

        assert config.install_type == InstallType.MINIMAL
        assert config.setup_database is True
        assert config.run_validation is True
        assert config.base_dir == Path.home() / "spyglass_data"
        assert config.env_name == "spyglass"
        assert config.db_port == 3306
        assert config.auto_yes is False

    def test_custom_values(self):
        """Test that SetupConfig accepts custom values."""
        config = SetupConfig(
            install_type=InstallType.FULL,
            setup_database=False,
            base_dir=Path("/custom/path"),
            env_name="my-env",
            db_port=3307,
            auto_yes=True
        )

        assert config.install_type == InstallType.FULL
        assert config.setup_database is False
        assert config.base_dir == Path("/custom/path")
        assert config.env_name == "my-env"
        assert config.db_port == 3307
        assert config.auto_yes is True


class TestValidation:
    """Test validation functions."""

    def test_validate_base_dir_valid(self):
        """Test base directory validation with valid path."""
        # Use home directory which should exist
        result = validate_base_dir(Path.home())
        assert result.is_success
        assert result.value == Path.home().resolve()

    def test_validate_base_dir_nonexistent_parent(self):
        """Test base directory validation with nonexistent parent."""
        result = validate_base_dir(Path("/nonexistent/path/subdir"))
        assert result.is_failure
        assert isinstance(result.error, ValueError)


class TestUserInterface:
    """Test UserInterface class methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ui = UserInterface(DisabledColors, auto_yes=False)

    def test_display_methods_exist(self):
        """Test that display methods exist and are callable."""
        assert callable(self.ui.print_info)
        assert callable(self.ui.print_success)
        assert callable(self.ui.print_warning)
        assert callable(self.ui.print_error)

    @patch('builtins.input', return_value='')
    def test_get_port_input_default(self, mock_input):
        """Test that get_port_input returns default when no input provided."""
        result = self.ui._get_port_input()
        assert result == 3306


class TestIntegration:
    """Test integration between components."""

    def test_complete_config_creation(self):
        """Test creating a complete configuration."""
        config = SetupConfig(
            install_type=InstallType.FULL,
            pipeline=Pipeline.DLC,
            setup_database=True,
            run_validation=True,
            base_dir=Path("/tmp/spyglass")
        )

        # Test that all components can be instantiated with this config
        ui = UserInterface(DisabledColors)
        env_manager = EnvironmentManager(ui, config)

        # Verify they're created successfully
        assert isinstance(ui, UserInterface)
        assert isinstance(env_manager, EnvironmentManager)


class TestEnvironmentManager:
    """Test EnvironmentManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = SetupConfig()
        self.ui = Mock()
        self.env_manager = EnvironmentManager(self.ui, self.config)

    def test_select_environment_file_minimal(self):
        """Test environment file selection for minimal install."""
        with patch.object(Path, 'exists', return_value=True):
            result = self.env_manager.select_environment_file()
            assert result == "environment-min.yml"

    def test_select_environment_file_full(self):
        """Test environment file selection for full install."""
        self.config = SetupConfig(install_type=InstallType.FULL)
        self.env_manager = EnvironmentManager(self.ui, self.config)

        with patch.object(Path, 'exists', return_value=True):
            result = self.env_manager.select_environment_file()
            assert result == "environment.yml"

    def test_select_environment_file_pipeline_dlc(self):
        """Test environment file selection for DLC pipeline."""
        self.config = SetupConfig(install_type=InstallType.MINIMAL, pipeline=Pipeline.DLC)
        self.env_manager = EnvironmentManager(self.ui, self.config)

        with patch.object(Path, 'exists', return_value=True):
            result = self.env_manager.select_environment_file()
            assert result == "environment_dlc.yml"

    @patch('os.path.exists', return_value=True)
    @patch('subprocess.run')
    def test_create_environment_command(self, mock_run, mock_exists):
        """Test that create_environment builds correct command."""
        # Test environment creation command
        cmd = self.env_manager._build_environment_command(
            "environment.yml", "conda", update=False
        )

        assert cmd[0] == "conda"
        assert "env" in cmd
        assert "create" in cmd
        assert "-f" in cmd
        assert "-n" in cmd
        assert self.config.env_name in cmd


# Pytest fixtures for shared resources
@pytest.fixture
def mock_ui():
    """Fixture for a mock UI object."""
    ui = Mock()
    ui.print_info = Mock()
    ui.print_success = Mock()
    ui.print_error = Mock()
    ui.print_warning = Mock()
    return ui


@pytest.fixture
def default_config():
    """Fixture for default SetupConfig."""
    return SetupConfig()


@pytest.fixture
def full_config():
    """Fixture for full installation SetupConfig."""
    return SetupConfig(install_type=InstallType.FULL)


# Parametrized tests for comprehensive coverage
@pytest.mark.parametrize("install_type,expected_file", [
    (InstallType.MINIMAL, "environment-min.yml"),
    (InstallType.FULL, "environment.yml"),
])
def test_environment_file_selection(install_type, expected_file, mock_ui):
    """Test environment file selection for different install types."""
    config = SetupConfig(install_type=install_type)
    env_manager = EnvironmentManager(mock_ui, config)

    with patch.object(Path, 'exists', return_value=True):
        result = env_manager.select_environment_file()
        assert result == expected_file


@pytest.mark.parametrize("path,should_succeed", [
    (Path.home(), True),
    (Path("/nonexistent/deeply/nested/path"), False),
])
def test_validate_base_dir_parametrized(path, should_succeed):
    """Parametrized test for base directory validation."""
    result = validate_base_dir(path)
    assert result.is_success == should_succeed
    if should_succeed:
        assert result.value == path.resolve()
    else:
        assert result.is_failure
        assert result.error is not None


# Skip tests that require Docker/conda when not available
@pytest.mark.skipif(not Path("/usr/local/bin/docker").exists() and not Path("/usr/bin/docker").exists(),
                    reason="Docker not available")
def test_docker_operations():
    """Test Docker operations when Docker is available."""
    from core.docker_operations import check_docker_available
    result = check_docker_available()
    # This test will only run if Docker is available
    assert result is not None


if __name__ == "__main__":
    # Provide helpful information for running tests
    print("This test file uses pytest. To run tests:")
    print("  pytest test_quickstart_pytest.py              # Run all tests")
    print("  pytest test_quickstart_pytest.py -v           # Verbose output")
    print("  pytest test_quickstart_pytest.py::TestValidation  # Run specific class")
    print("  pytest test_quickstart_pytest.py -k validate  # Run tests matching 'validate'")
    print("\nInstall pytest if needed: pip install pytest")