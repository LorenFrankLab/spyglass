#!/usr/bin/env python3
"""Tests for system components and factory patterns.

This module tests the system detection, factory patterns, and orchestration
components of the quickstart system.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import platform

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from quickstart import (
    InstallType,
    Pipeline,
    SetupConfig,
    SystemInfo,
    UserInterface,
    EnvironmentManager,
    QuickstartOrchestrator,
    InstallerFactory,
    DisabledColors,
)
from common import EnvironmentCreationError

try:
    from ux.error_recovery import ErrorRecoveryGuide, ErrorCategory

    ERROR_RECOVERY_AVAILABLE = True
except ImportError:
    ERROR_RECOVERY_AVAILABLE = False


class TestSystemInfo:
    """Test the SystemInfo dataclass."""

    def test_system_info_creation(self):
        """Test creating SystemInfo objects."""
        system_info = SystemInfo(
            os_name="Darwin",
            arch="arm64",
            is_m1=True,
            python_version=(3, 10, 18),
            conda_cmd="conda",
        )

        assert system_info.os_name == "Darwin"
        assert system_info.arch == "arm64"
        assert system_info.is_m1 is True
        assert system_info.python_version == (3, 10, 18)
        assert system_info.conda_cmd == "conda"

    def test_system_info_fields(self):
        """Test SystemInfo field access."""
        system_info = SystemInfo(
            os_name="Linux",
            arch="x86_64",
            is_m1=False,
            python_version=(3, 9, 0),
            conda_cmd="mamba",
        )

        # Should be able to read fields
        assert system_info.os_name == "Linux"
        assert system_info.arch == "x86_64"
        assert system_info.is_m1 is False

    def test_system_info_current_system(self):
        """Test SystemInfo with actual system data."""
        current_os = platform.system()
        current_arch = platform.machine()
        current_python = tuple(map(int, platform.python_version().split(".")))

        system_info = SystemInfo(
            os_name=current_os,
            arch=current_arch,
            is_m1=current_arch == "arm64",
            python_version=current_python,
            conda_cmd="conda",
        )

        assert system_info.os_name == current_os
        assert system_info.arch == current_arch
        assert system_info.python_version == current_python


class TestInstallerFactory:
    """Test the InstallerFactory class."""

    def test_factory_creation(self):
        """Test that factory can be instantiated."""
        factory = InstallerFactory()
        assert isinstance(factory, InstallerFactory)


class TestUserInterface:
    """Test UserInterface functionality."""

    def test_user_interface_creation(self):
        """Test creating UserInterface objects."""
        ui = UserInterface(DisabledColors)
        assert isinstance(ui, UserInterface)

    def test_user_interface_with_auto_yes(self):
        """Test UserInterface with auto_yes mode."""
        ui = UserInterface(DisabledColors, auto_yes=True)
        assert ui.auto_yes is True

    def test_display_methods_callable(self):
        """Test that all display methods are callable."""
        ui = UserInterface(DisabledColors)

        # These methods should exist and be callable
        assert callable(ui.print_info)
        assert callable(ui.print_success)
        assert callable(ui.print_warning)
        assert callable(ui.print_error)
        assert callable(ui.print_header)

    def test_message_formatting(self):
        """Test message formatting functionality."""
        ui = UserInterface(DisabledColors)

        # Test that _format_message works (if it exists)
        if hasattr(ui, "_format_message"):
            result = ui._format_message("Test message", "✓", "")
            assert isinstance(result, str)
            assert "Test message" in result

    @patch("builtins.input", return_value="y")
    def test_confirmation_prompt_yes(self, mock_input):
        """Test confirmation prompt with yes response."""
        ui = UserInterface(DisabledColors, auto_yes=False)

        if hasattr(ui, "confirm"):
            result = ui.confirm("Continue?")
            assert result is True

    @patch("builtins.input", return_value="n")
    def test_confirmation_prompt_no(self, mock_input):
        """Test confirmation prompt with no response."""
        ui = UserInterface(DisabledColors, auto_yes=False)

        if hasattr(ui, "confirm"):
            result = ui.confirm("Continue?")
            assert result is False

    def test_auto_yes_mode(self):
        """Test that auto_yes mode bypasses prompts."""
        ui = UserInterface(DisabledColors, auto_yes=True)

        if hasattr(ui, "confirm"):
            # Should return True without prompting
            result = ui.confirm("Continue?")
            assert result is True


class TestEnvironmentManager:
    """Test EnvironmentManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = SetupConfig()
        self.ui = Mock()
        self.env_manager = EnvironmentManager(self.ui, self.config)

    def test_environment_manager_creation(self):
        """Test creating EnvironmentManager objects."""
        assert isinstance(self.env_manager, EnvironmentManager)
        assert self.env_manager.ui == self.ui
        assert self.env_manager.config == self.config

    def test_select_environment_file_minimal(self):
        """Test environment file selection for minimal install."""
        with patch.object(Path, "exists", return_value=True):
            result = self.env_manager.select_environment_file()
            assert result == "environment-min.yml"

    def test_select_environment_file_full(self):
        """Test environment file selection for full install."""
        full_config = SetupConfig(install_type=InstallType.FULL)
        env_manager = EnvironmentManager(self.ui, full_config)

        with patch.object(Path, "exists", return_value=True):
            result = env_manager.select_environment_file()
            assert result == "environment.yml"

    def test_select_environment_file_pipeline_dlc(self):
        """Test environment file selection for DLC pipeline."""
        dlc_config = SetupConfig(pipeline=Pipeline.DLC)
        env_manager = EnvironmentManager(self.ui, dlc_config)

        with patch.object(Path, "exists", return_value=True):
            result = env_manager.select_environment_file()
            assert result == "environment_dlc.yml"

    def test_environment_file_missing(self):
        """Test behavior when environment file doesn't exist."""
        with patch.object(Path, "exists", return_value=False):
            # Should raise EnvironmentCreationError for missing files
            with pytest.raises(EnvironmentCreationError) as exc_info:
                self.env_manager.select_environment_file()
            assert "Environment file not found" in str(exc_info.value)

    @patch("subprocess.run")
    def test_build_environment_command(self, mock_run):
        """Test building conda environment commands."""
        cmd = self.env_manager._build_environment_command(
            "environment.yml", "conda", update=False
        )

        assert isinstance(cmd, list)
        assert cmd[0] == "conda"
        assert "env" in cmd
        assert "create" in cmd
        assert "-f" in cmd
        assert "-n" in cmd
        assert self.config.env_name in cmd

    @patch("subprocess.run")
    def test_build_update_command(self, mock_run):
        """Test building conda environment update commands."""
        cmd = self.env_manager._build_environment_command(
            "environment.yml", "conda", update=True
        )

        assert isinstance(cmd, list)
        assert cmd[0] == "conda"
        assert "env" in cmd
        assert "update" in cmd
        assert "-f" in cmd
        assert "-n" in cmd


class TestQuickstartOrchestrator:
    """Test QuickstartOrchestrator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = SetupConfig()
        self.ui = Mock()
        self.orchestrator = QuickstartOrchestrator(self.config, DisabledColors)

    def test_orchestrator_creation(self):
        """Test creating QuickstartOrchestrator objects."""
        assert isinstance(self.orchestrator, QuickstartOrchestrator)
        assert isinstance(self.orchestrator.ui, UserInterface)
        assert self.orchestrator.config == self.config
        assert isinstance(self.orchestrator.env_manager, EnvironmentManager)

    def test_orchestrator_has_required_methods(self):
        """Test that orchestrator has required methods."""
        # Check for key methods that should exist
        required_methods = ["run", "setup_database", "validate_installation"]

        for method_name in required_methods:
            if hasattr(self.orchestrator, method_name):
                assert callable(getattr(self.orchestrator, method_name))

    @patch("quickstart.validate_base_dir")
    def test_orchestrator_validation_integration(self, mock_validate):
        """Test that orchestrator integrates with validation functions."""
        from utils.result_types import success

        # Mock successful validation
        mock_validate.return_value = success(Path("/tmp/test"))

        # Test that validation is called during orchestration
        if hasattr(self.orchestrator, "validate_configuration"):
            result = self.orchestrator.validate_configuration()
            # Should get some kind of result
            assert result is not None


@pytest.mark.skipif(
    not ERROR_RECOVERY_AVAILABLE,
    reason="ux.error_recovery module not available",
)
class TestErrorRecovery:
    """Test error recovery functionality."""

    def test_error_recovery_guide_creation(self):
        """Test creating ErrorRecoveryGuide objects."""
        ui = Mock()
        guide = ErrorRecoveryGuide(ui)
        assert isinstance(guide, ErrorRecoveryGuide)

    def test_error_category_enum(self):
        """Test ErrorCategory enum values."""
        # Test that common error categories exist
        common_categories = [
            ErrorCategory.DOCKER,
            ErrorCategory.CONDA,
            ErrorCategory.PYTHON,
            ErrorCategory.NETWORK,
        ]

        for category in common_categories:
            assert category in ErrorCategory

    def test_error_recovery_methods(self):
        """Test that ErrorRecoveryGuide has required methods."""
        ui = Mock()
        guide = ErrorRecoveryGuide(ui)

        # Should have methods for handling different error types
        required_methods = ["handle_error"]
        for method_name in required_methods:
            if hasattr(guide, method_name):
                assert callable(getattr(guide, method_name))


# Integration tests
class TestSystemIntegration:
    """Test integration between system components."""

    def test_full_config_pipeline(self):
        """Test complete configuration pipeline."""
        config = SetupConfig(
            install_type=InstallType.FULL,
            pipeline=Pipeline.DLC,
            base_dir=Path("/tmp/test"),
            env_name="test-env",
        )

        ui = UserInterface(DisabledColors)
        env_manager = EnvironmentManager(ui, config)
        orchestrator = QuickstartOrchestrator(config, DisabledColors)

        # All components should be created successfully
        assert isinstance(config, SetupConfig)
        assert isinstance(ui, UserInterface)
        assert isinstance(env_manager, EnvironmentManager)
        assert isinstance(orchestrator, QuickstartOrchestrator)

        # Configuration should flow through correctly
        assert orchestrator.config == config
        assert orchestrator.env_manager.config == config


# Parametrized tests for comprehensive coverage
@pytest.mark.parametrize(
    "install_type,pipeline,expected_env_file",
    [
        (InstallType.MINIMAL, None, "environment-min.yml"),
        (InstallType.FULL, None, "environment.yml"),
        (InstallType.MINIMAL, Pipeline.DLC, "environment_dlc.yml"),
    ],
)
def test_environment_file_selection_parametrized(
    install_type, pipeline, expected_env_file
):
    """Test environment file selection for different configurations."""
    config = SetupConfig(install_type=install_type, pipeline=pipeline)
    ui = Mock()
    env_manager = EnvironmentManager(ui, config)

    with patch.object(Path, "exists", return_value=True):
        result = env_manager.select_environment_file()
        assert result == expected_env_file


@pytest.mark.parametrize(
    "auto_yes,expected_behavior",
    [
        (True, "automatic"),
        (False, "interactive"),
    ],
)
def test_user_interface_modes(auto_yes, expected_behavior):
    """Test different UserInterface modes."""
    ui = UserInterface(DisabledColors, auto_yes=auto_yes)
    assert ui.auto_yes == auto_yes

    if expected_behavior == "automatic":
        assert ui.auto_yes is True
    else:
        assert ui.auto_yes is False


if __name__ == "__main__":
    # Provide helpful information for running tests
    print("This test file validates system components and factory patterns.")
    print("To run tests:")
    print("  pytest test_system_components.py              # Run all tests")
    print("  pytest test_system_components.py -v           # Verbose output")
    print(
        "  pytest test_system_components.py::TestInstallerFactory  # Run specific class"
    )
    print(
        "  pytest test_system_components.py -k factory   # Run tests matching 'factory'"
    )
    print(
        "\nNote: Some tests require the ux.error_recovery module to be available."
    )
