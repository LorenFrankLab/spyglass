#!/usr/bin/env python
"""
Basic unit tests for quickstart.py refactored architecture.

These tests demonstrate the improved testability of the refactored code.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from quickstart import (
    SetupConfig, InstallType, Pipeline,
    UserInterface, SystemDetector, EnvironmentManager,
    validate_base_dir, DisabledColors
)


class TestSetupConfig(unittest.TestCase):
    """Test the SetupConfig dataclass."""

    def test_config_creation(self):
        """Test that SetupConfig can be created with all parameters."""
        config = SetupConfig(
            install_type=InstallType.MINIMAL,
            pipeline=None,
            setup_database=True,
            run_validation=True,
            base_dir=Path("/tmp/test")
        )

        self.assertEqual(config.install_type, InstallType.MINIMAL)
        self.assertIsNone(config.pipeline)
        self.assertTrue(config.setup_database)
        self.assertTrue(config.run_validation)
        self.assertEqual(config.base_dir, Path("/tmp/test"))


class TestValidation(unittest.TestCase):
    """Test validation functions."""

    def test_validate_base_dir_valid(self):
        """Test base directory validation with valid path."""
        # Use home directory which should exist
        result = validate_base_dir(Path.home())
        self.assertEqual(result, Path.home().resolve())

    def test_validate_base_dir_nonexistent_parent(self):
        """Test base directory validation with nonexistent parent."""
        with self.assertRaises(ValueError):
            validate_base_dir(Path("/nonexistent/path/subdir"))


class TestUserInterface(unittest.TestCase):
    """Test UserInterface class methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.ui = UserInterface(DisabledColors)

    def test_format_message(self):
        """Test message formatting."""
        result = self.ui._format_message("Test", "✓", "")
        self.assertIn("✓", result)
        self.assertIn("Test", result)

    @patch('builtins.input')
    def test_get_host_input_default(self, mock_input):
        """Test host input with default value."""
        mock_input.return_value = ""  # Empty input should use default
        result = self.ui._get_host_input()
        self.assertEqual(result, "localhost")

    @patch('builtins.input')
    def test_get_port_input_valid(self, mock_input):
        """Test port input with valid value."""
        mock_input.return_value = "5432"
        result = self.ui._get_port_input()
        self.assertEqual(result, 5432)

    @patch('builtins.input')
    def test_get_port_input_default(self, mock_input):
        """Test port input with default value."""
        mock_input.return_value = ""  # Empty input should use default
        result = self.ui._get_port_input()
        self.assertEqual(result, 3306)


class TestSystemDetector(unittest.TestCase):
    """Test SystemDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.ui = Mock()
        self.detector = SystemDetector(self.ui)

    @patch('platform.system')
    @patch('platform.machine')
    def test_detect_system_macos(self, mock_machine, mock_system):
        """Test system detection for macOS."""
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "x86_64"

        system_info = self.detector.detect_system()

        self.assertEqual(system_info.os_name, "macOS")  # SystemDetector returns 'macOS' not 'Darwin'
        self.assertEqual(system_info.arch, "x86_64")
        self.assertFalse(system_info.is_m1)


class TestIntegration(unittest.TestCase):
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
        detector = SystemDetector(ui)
        env_manager = EnvironmentManager(ui, config)

        # Verify they're created successfully
        self.assertIsInstance(ui, UserInterface)
        self.assertIsInstance(detector, SystemDetector)
        self.assertIsInstance(env_manager, EnvironmentManager)


class TestEnvironmentManager(unittest.TestCase):
    """Test EnvironmentManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.ui = Mock()
        self.config = SetupConfig(
            install_type=InstallType.MINIMAL,
            pipeline=None,
            setup_database=False,
            run_validation=False,
            base_dir=Path("/tmp/test")
        )
        self.env_manager = EnvironmentManager(self.ui, self.config)

    def test_select_environment_file_minimal(self):
        """Test environment file selection for minimal install."""
        # Mock the environment file existence check
        with patch.object(Path, 'exists', return_value=True):
            result = self.env_manager.select_environment_file()
            self.assertEqual(result, "environment-min.yml")

    def test_select_environment_file_full(self):
        """Test environment file selection for full install."""
        self.config = SetupConfig(
            install_type=InstallType.FULL,
            pipeline=None,
            setup_database=False,
            run_validation=False,
            base_dir=Path("/tmp/test")
        )
        env_manager = EnvironmentManager(self.ui, self.config)

        # Mock the environment file existence check
        with patch.object(Path, 'exists', return_value=True):
            result = env_manager.select_environment_file()
            self.assertEqual(result, "environment.yml")

    def test_select_environment_file_pipeline(self):
        """Test environment file selection for specific pipeline."""
        self.config = SetupConfig(
            install_type=InstallType.FULL,  # Use FULL instead of non-existent PIPELINE
            pipeline=Pipeline.DLC,
            setup_database=False,
            run_validation=False,
            base_dir=Path("/tmp/test")
        )
        env_manager = EnvironmentManager(self.ui, self.config)

        # Mock the environment file existence check
        with patch.object(Path, 'exists', return_value=True):
            result = env_manager.select_environment_file()
            self.assertEqual(result, "environment_dlc.yml")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)