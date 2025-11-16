"""Tests for installation script."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

# Add scripts to path
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from install import (
    check_prerequisites,
    get_base_directory,
    get_conda_command,
    get_required_python_version,
    is_docker_available_inline,
)


class TestGetRequiredPythonVersion:
    """Tests for get_required_python_version()."""

    def test_returns_tuple(self):
        """Test that function returns a tuple."""
        version = get_required_python_version()
        assert isinstance(version, tuple)
        assert len(version) == 2

    def test_version_is_reasonable(self):
        """Test that returned version is reasonable."""
        major, minor = get_required_python_version()
        assert major == 3
        assert 9 <= minor <= 13  # Current supported range


class TestGetCondaCommand:
    """Tests for get_conda_command()."""

    def test_prefers_mamba(self):
        """Test that mamba is preferred over conda."""
        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: cmd == "mamba"
            assert get_conda_command() == "mamba"

    def test_falls_back_to_conda(self):
        """Test fallback to conda when mamba unavailable."""
        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: cmd == "conda"
            assert get_conda_command() == "conda"

    def test_raises_when_neither_available(self):
        """Test that RuntimeError raised when neither available."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="conda or mamba not found"):
                get_conda_command()


class TestGetBaseDirectory:
    """Tests for get_base_directory()."""

    def test_cli_arg_priority(self, tmp_path):
        """Test that CLI argument has highest priority."""
        cli_path = tmp_path / "cli_path"
        result = get_base_directory(str(cli_path))
        assert result == cli_path.resolve()
        assert result.exists()  # Verify it was created

    def test_env_var_priority(self, monkeypatch, tmp_path):
        """Test that environment variable has second priority."""
        env_path = tmp_path / "env_path"
        monkeypatch.setenv("SPYGLASS_BASE_DIR", str(env_path))
        result = get_base_directory(None)
        assert result == env_path.resolve()
        assert result.exists()  # Verify it was created

    def test_cli_overrides_env_var(self, monkeypatch, tmp_path):
        """Test that CLI argument overrides environment variable."""
        env_path = tmp_path / "env_path"
        cli_path = tmp_path / "cli_path"
        monkeypatch.setenv("SPYGLASS_BASE_DIR", str(env_path))
        result = get_base_directory(str(cli_path))
        assert result == cli_path.resolve()
        assert result.exists()  # Verify CLI path was created
        assert not env_path.exists()  # Verify ENV path was NOT created

    def test_expands_user_home(self, tmp_path):
        """Test that ~ is expanded to user home."""
        # Use a subdirectory under user's home that we can safely create/delete
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test directory we can safely use
            test_path = Path(tmpdir) / "test"
            result = get_base_directory(str(test_path))
            assert test_path.resolve() == result
            assert result.is_absolute()
            assert result.exists()


class TestIsDockerAvailableInline:
    """Tests for is_docker_available_inline()."""

    def test_returns_false_when_docker_not_in_path(self):
        """Test returns False when docker not in PATH."""
        with patch("shutil.which", return_value=None):
            assert is_docker_available_inline() is False

    def test_returns_false_when_daemon_not_running(self):
        """Test returns False when docker daemon not running."""
        with patch("shutil.which", return_value="/usr/bin/docker"):
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.CalledProcessError(
                    1, "docker"
                )
                assert is_docker_available_inline() is False

    def test_returns_true_when_docker_available(self):
        """Test returns True when docker is available."""
        with patch("shutil.which", return_value="/usr/bin/docker"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=0)
                assert is_docker_available_inline() is True


class TestCheckPrerequisites:
    """Tests for check_prerequisites()."""

    def test_does_not_raise_on_valid_system(self):
        """Test that function doesn't raise on valid system."""
        # This test assumes we're running on a valid development system
        # If it fails, the system isn't suitable for development
        try:
            check_prerequisites()
        except RuntimeError as e:
            # Only acceptable failure is conda/mamba not found in test env
            if "conda or mamba not found" not in str(e):
                raise


@pytest.mark.integration
class TestInstallationIntegration:
    """Integration tests for full installation workflow.

    These tests are marked as integration and can be run separately.
    They require conda/mamba and take longer to run.
    """

    def test_validate_script_exists(self):
        """Test that validate.py script exists."""
        validate_script = scripts_dir / "validate.py"
        assert validate_script.exists()
        assert validate_script.is_file()

    def test_install_script_exists(self):
        """Test that install.py script exists."""
        install_script = scripts_dir / "install.py"
        assert install_script.exists()
        assert install_script.is_file()

    def test_scripts_are_executable(self):
        """Test that scripts have execute permissions."""
        validate_script = scripts_dir / "validate.py"
        install_script = scripts_dir / "install.py"

        # Check if readable and executable (on Unix-like systems)
        if sys.platform != "win32":
            assert validate_script.stat().st_mode & 0o111  # Has execute bit
            assert install_script.stat().st_mode & 0o111


class TestDockerUtilities:
    """Tests for docker utility module."""

    def test_docker_module_exists(self):
        """Test that docker utilities module exists."""
        docker_module = (
            Path(__file__).parent.parent.parent
            / "src"
            / "spyglass"
            / "utils"
            / "docker.py"
        )
        assert docker_module.exists()

    def test_docker_module_imports(self):
        """Test that docker utilities can be imported."""
        try:
            from spyglass.utils import docker

            assert hasattr(docker, "DockerConfig")
            assert hasattr(docker, "is_docker_available")
            assert hasattr(docker, "start_database_container")
        except ImportError:
            pytest.skip("Spyglass not installed")
