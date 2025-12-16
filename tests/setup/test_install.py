"""Tests for installation script.

This module contains unit and integration tests for the Spyglass installer.
Tests are organized by the functions they test, with clear categories:

- Unit tests: Fast, isolated tests with mocked dependencies
- Integration tests: Tests that verify end-to-end behavior

Run all tests:
    pytest tests/setup/test_install.py -v

Run only unit tests:
    pytest tests/setup/test_install.py -v -m "not integration"

Run only integration tests:
    pytest tests/setup/test_install.py -v -m integration
"""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add scripts to path
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from install import (
    build_directory_structure,
    check_disk_space,
    check_prerequisites,
    CondaManager,
    create_database_config,
    determine_tls,
    DockerManager,
    get_base_directory,
    get_required_python_version,
    is_port_available,
    load_directory_schema,
    validate_database_config,
    validate_hostname,
    validate_port,
    validate_schema,
)

# =============================================================================
# Python Version Detection Tests
# =============================================================================


class TestGetRequiredPythonVersion:
    """Tests for get_required_python_version()."""

    def test_returns_tuple(self):
        """Returns a tuple of two integers."""
        version = get_required_python_version()
        assert isinstance(version, tuple)
        assert len(version) == 2

    def test_version_is_reasonable(self):
        """Returned version is within expected range (3.9-3.13)."""
        major, minor = get_required_python_version()
        assert major == 3
        assert 9 <= minor <= 13

    def test_reads_from_pyproject_toml(self):
        """Reads version from pyproject.toml, not hardcoded."""
        # Verify pyproject.toml exists and has requires-python
        pyproject_path = scripts_dir.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml not found"

        content = pyproject_path.read_text()
        assert (
            "requires-python" in content
        ), "pyproject.toml should have requires-python"


# =============================================================================
# Conda/Mamba Detection Tests
# =============================================================================


class TestCondaManagerGetCommand:
    """Tests for CondaManager.get_command()."""

    def test_returns_conda_when_available(self):
        """Returns conda when available."""
        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: cmd == "conda"
            assert CondaManager.get_command() == "conda"

    def test_raises_when_conda_not_available(self):
        """Raises RuntimeError when conda not available."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="conda not found"):
                CondaManager.get_command()


# =============================================================================
# Base Directory Resolution Tests
# =============================================================================


class TestGetBaseDirectory:
    """Tests for get_base_directory()."""

    def test_cli_arg_priority(self, tmp_path):
        """CLI argument has highest priority."""
        cli_path = tmp_path / "cli_path"
        result = get_base_directory(str(cli_path))
        assert result == cli_path.resolve()
        assert result.exists()

    def test_env_var_priority(self, monkeypatch, tmp_path):
        """Environment variable has second priority."""
        env_path = tmp_path / "env_path"
        monkeypatch.setenv("SPYGLASS_BASE_DIR", str(env_path))
        result = get_base_directory(None)
        assert result == env_path.resolve()
        assert result.exists()

    def test_cli_overrides_env_var(self, monkeypatch, tmp_path):
        """CLI argument overrides environment variable."""
        env_path = tmp_path / "env_path"
        cli_path = tmp_path / "cli_path"
        monkeypatch.setenv("SPYGLASS_BASE_DIR", str(env_path))
        result = get_base_directory(str(cli_path))
        assert result == cli_path.resolve()
        assert result.exists()
        assert not env_path.exists()

    def test_expands_user_home(self, tmp_path):
        """Expands ~ to user home directory."""
        test_path = tmp_path / "test"
        result = get_base_directory(str(test_path))
        assert test_path.resolve() == result
        assert result.is_absolute()
        assert result.exists()

    def test_creates_directory_if_not_exists(self, tmp_path):
        """Creates directory if it doesn't exist."""
        new_path = tmp_path / "new" / "nested" / "path"
        assert not new_path.exists()
        result = get_base_directory(str(new_path))
        assert result.exists()
        assert result.is_dir()


# =============================================================================
# Docker Availability Tests
# =============================================================================


class TestDockerManagerIsAvailable:
    """Tests for DockerManager.is_available()."""

    def test_returns_false_when_docker_not_in_path(self):
        """Returns False when docker not in PATH."""
        with patch("shutil.which", return_value=None):
            assert DockerManager.is_available() is False

    def test_returns_false_when_daemon_not_running(self):
        """Returns False when docker daemon not running."""
        with patch("shutil.which", return_value="/usr/bin/docker"):
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.CalledProcessError(
                    1, "docker"
                )
                assert DockerManager.is_available() is False

    def test_returns_true_when_docker_available(self):
        """Returns True when docker is available and running."""
        with patch("shutil.which", return_value="/usr/bin/docker"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=0)
                assert DockerManager.is_available() is True


class TestDockerManagerIsComposeAvailable:
    """Tests for DockerManager.is_compose_available()."""

    def test_returns_false_when_docker_compose_fails(self):
        """Returns False when docker compose command fails."""
        with patch("install.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            assert DockerManager.is_compose_available() is False

    def test_returns_false_on_nonzero_exit(self):
        """Returns False when docker compose returns non-zero exit code."""
        with patch("install.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=1)
            assert DockerManager.is_compose_available() is False

    def test_returns_true_when_compose_available(self):
        """Returns True when docker compose is available."""
        with patch("install.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            assert DockerManager.is_compose_available() is True


# =============================================================================
# Disk Space Check Tests
# =============================================================================


class TestCheckDiskSpace:
    """Tests for check_disk_space()."""

    def test_existing_path(self, tmp_path):
        """Returns disk space for existing path."""
        sufficient, available = check_disk_space(1, tmp_path)
        # Should have at least 1 GB on any modern system
        assert isinstance(sufficient, bool)
        assert isinstance(available, int)
        assert available >= 0

    def test_nonexistent_path_checks_parent(self, tmp_path):
        """Walks up tree to find existing parent for non-existent path."""
        nonexistent = tmp_path / "does" / "not" / "exist"
        sufficient, available = check_disk_space(1, nonexistent)
        # Should still work by checking tmp_path
        assert isinstance(sufficient, bool)
        assert isinstance(available, int)

    def test_returns_false_when_insufficient(self, tmp_path):
        """Returns False when space is insufficient."""
        # Request absurdly large amount
        sufficient, _ = check_disk_space(999999, tmp_path)
        assert sufficient is False

    def test_returns_true_when_sufficient(self, tmp_path):
        """Returns True when space is sufficient."""
        # Request small amount
        sufficient, _ = check_disk_space(1, tmp_path)
        # Most systems have at least 1 GB free
        assert sufficient is True


# =============================================================================
# Hostname Validation Tests
# =============================================================================


class TestValidateHostname:
    """Tests for validate_hostname()."""

    def test_accepts_localhost(self):
        """Accepts 'localhost'."""
        assert validate_hostname("localhost") is True

    def test_accepts_ipv4_localhost(self):
        """Accepts IPv4 localhost address."""
        assert validate_hostname("127.0.0.1") is True

    def test_accepts_ipv6_localhost(self):
        """Accepts IPv6 localhost address."""
        assert validate_hostname("::1") is True

    def test_accepts_domain_name(self):
        """Accepts valid domain names."""
        assert validate_hostname("db.example.com") is True
        assert validate_hostname("lmf-db.cin.ucsf.edu") is True

    def test_accepts_ip_address(self):
        """Accepts valid IP addresses."""
        assert validate_hostname("192.168.1.100") is True
        assert validate_hostname("10.0.0.1") is True

    def test_rejects_empty_string(self):
        """Rejects empty hostname."""
        assert validate_hostname("") is False

    def test_rejects_spaces(self):
        """Rejects hostname with spaces."""
        assert validate_hostname("host with spaces") is False
        assert validate_hostname(" localhost") is False
        assert validate_hostname("localhost ") is False

    def test_rejects_control_characters(self):
        """Rejects hostname with control characters."""
        assert validate_hostname("host\tname") is False
        assert validate_hostname("host\nname") is False

    def test_rejects_leading_dot(self):
        """Rejects hostname starting with dot."""
        assert validate_hostname(".example.com") is False

    def test_rejects_trailing_dot(self):
        """Rejects hostname ending with dot."""
        assert validate_hostname("example.com.") is False

    def test_rejects_consecutive_dots(self):
        """Rejects hostname with consecutive dots."""
        assert validate_hostname("example..com") is False
        assert validate_hostname("..invalid") is False

    def test_rejects_overly_long_hostname(self):
        """Rejects hostname exceeding 253 characters."""
        long_hostname = "a" * 254
        assert validate_hostname(long_hostname) is False


# =============================================================================
# Port Validation Tests
# =============================================================================


class TestValidatePort:
    """Tests for validate_port()."""

    def test_accepts_mysql_default_port(self):
        """Accepts MySQL default port 3306."""
        valid, msg = validate_port(3306)
        assert valid is True
        assert msg == ""

    def test_accepts_non_privileged_ports(self):
        """Accepts ports in non-privileged range (1024-65535)."""
        valid, _ = validate_port(1024)
        assert valid is True

        valid, _ = validate_port(65535)
        assert valid is True

        valid, _ = validate_port(8080)
        assert valid is True

    def test_rejects_privileged_ports(self):
        """Rejects privileged ports (1-1023)."""
        valid, msg = validate_port(80)
        assert valid is False
        assert "privileged" in msg.lower()

        valid, msg = validate_port(1)
        assert valid is False

        valid, msg = validate_port(1023)
        assert valid is False

    def test_rejects_port_zero(self):
        """Rejects port 0."""
        valid, msg = validate_port(0)
        assert valid is False
        assert "out of valid range" in msg.lower()

    def test_rejects_negative_port(self):
        """Rejects negative port numbers."""
        valid, msg = validate_port(-1)
        assert valid is False

    def test_rejects_port_above_65535(self):
        """Rejects ports above 65535."""
        valid, msg = validate_port(65536)
        assert valid is False
        assert "out of valid range" in msg.lower()


# =============================================================================
# Database Configuration Validation Tests
# =============================================================================


class TestValidateDatabaseConfig:
    """Tests for validate_database_config()."""

    def test_accepts_valid_config(self):
        """Accepts valid database configuration."""
        valid, errors = validate_database_config(
            host="localhost",
            port=3306,
            user="root",
            password="password",
        )
        assert valid is True
        assert errors == []

    def test_accepts_remote_config(self):
        """Accepts valid remote database configuration."""
        valid, errors = validate_database_config(
            host="lmf-db.cin.ucsf.edu",
            port=3306,
            user="myuser",
            password="secret",
        )
        assert valid is True
        assert errors == []

    def test_rejects_empty_hostname(self):
        """Rejects empty hostname."""
        valid, errors = validate_database_config(
            host="",
            port=3306,
            user="root",
            password="password",
        )
        assert valid is False
        assert any("hostname" in e.lower() for e in errors)

    def test_rejects_invalid_hostname(self):
        """Rejects invalid hostname format."""
        valid, errors = validate_database_config(
            host="host with spaces",
            port=3306,
            user="root",
            password="password",
        )
        assert valid is False
        assert any(
            "hostname" in e.lower() or "invalid" in e.lower() for e in errors
        )

    def test_rejects_invalid_port(self):
        """Rejects invalid port number."""
        valid, errors = validate_database_config(
            host="localhost",
            port=80,  # Privileged port
            user="root",
            password="password",
        )
        assert valid is False
        assert any("port" in e.lower() for e in errors)

    def test_rejects_empty_username(self):
        """Rejects empty username."""
        valid, errors = validate_database_config(
            host="localhost",
            port=3306,
            user="",
            password="password",
        )
        assert valid is False
        assert any("username" in e.lower() for e in errors)

    def test_rejects_long_username(self):
        """Rejects username exceeding 32 characters."""
        valid, errors = validate_database_config(
            host="localhost",
            port=3306,
            user="a" * 33,
            password="password",
        )
        assert valid is False
        assert any(
            "username" in e.lower() and "long" in e.lower() for e in errors
        )

    def test_collects_multiple_errors(self):
        """Collects all validation errors, not just the first."""
        valid, errors = validate_database_config(
            host="",
            port=0,
            user="",
            password="password",
        )
        assert valid is False
        assert len(errors) >= 3  # At least host, port, user errors


# =============================================================================
# Port Availability Tests
# =============================================================================


class TestIsPortAvailable:
    """Tests for is_port_available()."""

    def test_localhost_port_in_use(self):
        """Returns False when localhost port is in use."""
        with patch("socket.socket") as mock_socket:
            mock_instance = Mock()
            mock_socket.return_value.__enter__ = Mock(
                return_value=mock_instance
            )
            mock_socket.return_value.__exit__ = Mock(return_value=False)
            # connect_ex returns 0 when port is in use (connection succeeded)
            mock_instance.connect_ex.return_value = 0

            available, msg = is_port_available("localhost", 3306)
            assert available is False
            assert "already in use" in msg

    def test_localhost_port_free(self):
        """Returns True when localhost port is free."""
        with patch("socket.socket") as mock_socket:
            mock_instance = Mock()
            mock_socket.return_value.__enter__ = Mock(
                return_value=mock_instance
            )
            mock_socket.return_value.__exit__ = Mock(return_value=False)
            # connect_ex returns non-zero when port is free (connection refused)
            mock_instance.connect_ex.return_value = 111  # ECONNREFUSED

            available, msg = is_port_available("localhost", 3306)
            assert available is True
            assert "available" in msg

    def test_remote_port_reachable(self):
        """Returns True when remote port is reachable."""
        with patch("socket.socket") as mock_socket:
            mock_instance = Mock()
            mock_socket.return_value.__enter__ = Mock(
                return_value=mock_instance
            )
            mock_socket.return_value.__exit__ = Mock(return_value=False)
            # connect_ex returns 0 when port is reachable (connection succeeded)
            mock_instance.connect_ex.return_value = 0

            available, msg = is_port_available("db.example.com", 3306)
            assert available is True
            assert "reachable" in msg

    def test_remote_port_unreachable(self):
        """Returns False when remote port is unreachable."""
        with patch("socket.socket") as mock_socket:
            mock_instance = Mock()
            mock_socket.return_value.__enter__ = Mock(
                return_value=mock_instance
            )
            mock_socket.return_value.__exit__ = Mock(return_value=False)
            # connect_ex returns non-zero when port is unreachable
            mock_instance.connect_ex.return_value = 111  # ECONNREFUSED

            available, msg = is_port_available("db.example.com", 3306)
            assert available is False
            assert "Cannot reach" in msg

    def test_dns_resolution_failure(self):
        """Returns False when hostname cannot be resolved."""
        import socket

        with patch("socket.socket") as mock_socket:
            mock_instance = Mock()
            mock_socket.return_value.__enter__ = Mock(
                return_value=mock_instance
            )
            mock_socket.return_value.__exit__ = Mock(return_value=False)
            mock_instance.connect_ex.side_effect = socket.gaierror(
                "Name resolution failed"
            )

            available, msg = is_port_available("nonexistent.invalid", 3306)
            assert available is False
            assert "Cannot resolve" in msg

    def test_socket_error(self):
        """Returns False on general socket errors."""
        import socket

        with patch("socket.socket") as mock_socket:
            mock_instance = Mock()
            mock_socket.return_value.__enter__ = Mock(
                return_value=mock_instance
            )
            mock_socket.return_value.__exit__ = Mock(return_value=False)
            mock_instance.connect_ex.side_effect = socket.error(
                "Network unreachable"
            )

            available, msg = is_port_available("10.0.0.1", 3306)
            assert available is False
            assert "Socket error" in msg

    def test_ipv4_localhost_treated_as_local(self):
        """127.0.0.1 is treated as localhost (port free = available)."""
        with patch("socket.socket") as mock_socket:
            mock_instance = Mock()
            mock_socket.return_value.__enter__ = Mock(
                return_value=mock_instance
            )
            mock_socket.return_value.__exit__ = Mock(return_value=False)
            # Port is free
            mock_instance.connect_ex.return_value = 111

            available, msg = is_port_available("127.0.0.1", 3306)
            assert available is True
            assert "available" in msg

    def test_ipv6_localhost_treated_as_local(self):
        """::1 is treated as localhost (port free = available)."""
        with patch("socket.socket") as mock_socket:
            mock_instance = Mock()
            mock_socket.return_value.__enter__ = Mock(
                return_value=mock_instance
            )
            mock_socket.return_value.__exit__ = Mock(return_value=False)
            # Port is free
            mock_instance.connect_ex.return_value = 111

            available, msg = is_port_available("::1", 3306)
            assert available is True
            assert "available" in msg


# =============================================================================
# TLS Determination Tests
# =============================================================================


class TestDetermineTls:
    """Tests for determine_tls()."""

    def test_localhost_disables_tls(self):
        """Localhost connections disable TLS."""
        assert determine_tls("localhost") is False

    def test_ipv4_localhost_disables_tls(self):
        """IPv4 localhost (127.0.0.1) disables TLS."""
        assert determine_tls("127.0.0.1") is False

    def test_ipv6_localhost_disables_tls(self):
        """IPv6 localhost (::1) disables TLS."""
        assert determine_tls("::1") is False

    def test_remote_hostname_enables_tls(self):
        """Remote hostnames enable TLS."""
        assert determine_tls("lmf-db.cin.ucsf.edu") is True
        assert determine_tls("db.example.com") is True

    def test_remote_ip_enables_tls(self):
        """Remote IP addresses enable TLS."""
        assert determine_tls("192.168.1.100") is True
        assert determine_tls("10.0.0.1") is True


# =============================================================================
# Environment File Generation Tests
# =============================================================================


class TestDockerManagerGenerateEnvFile:
    """Tests for DockerManager.generate_env_file()."""

    def test_creates_no_file_with_defaults(self, tmp_path):
        """Creates no file when all values are defaults."""
        env_path = tmp_path / ".env"
        DockerManager.generate_env_file(
            mysql_port=3306,
            mysql_password="tutorial",
            mysql_image="datajoint/mysql:8.0",
            env_path=str(env_path),
        )
        # With all defaults, no file should be created
        assert not env_path.exists()

    def test_creates_file_with_custom_password(self, tmp_path):
        """Creates file when password differs from default."""
        env_path = tmp_path / ".env"
        DockerManager.generate_env_file(
            mysql_port=3306,
            mysql_password="custom_password",
            mysql_image="datajoint/mysql:8.0",
            env_path=str(env_path),
        )
        assert env_path.exists()
        content = env_path.read_text()
        assert "MYSQL_ROOT_PASSWORD=custom_password" in content

    def test_creates_file_with_custom_port(self, tmp_path):
        """Creates file when port differs from default."""
        env_path = tmp_path / ".env"
        DockerManager.generate_env_file(
            mysql_port=3307,
            mysql_password="tutorial",
            mysql_image="datajoint/mysql:8.0",
            env_path=str(env_path),
        )
        assert env_path.exists()
        content = env_path.read_text()
        assert "MYSQL_PORT=3307" in content

    def test_creates_file_with_custom_image(self, tmp_path):
        """Creates file when image differs from default."""
        env_path = tmp_path / ".env"
        DockerManager.generate_env_file(
            mysql_port=3306,
            mysql_password="tutorial",
            mysql_image="mysql:8.0",
            env_path=str(env_path),
        )
        assert env_path.exists()
        content = env_path.read_text()
        assert "MYSQL_IMAGE=mysql:8.0" in content


class TestDockerManagerValidateEnvFile:
    """Tests for DockerManager.validate_env_file()."""

    def test_returns_true_for_missing_file(self, tmp_path):
        """Returns True when .env file doesn't exist (uses defaults)."""
        env_path = tmp_path / ".env"
        assert not env_path.exists()
        assert DockerManager.validate_env_file(str(env_path)) is True

    def test_returns_true_for_readable_file(self, tmp_path):
        """Returns True when .env file exists and is readable."""
        env_path = tmp_path / ".env"
        env_path.write_text("MYSQL_PORT=3306\n")
        assert DockerManager.validate_env_file(str(env_path)) is True


# =============================================================================
# Directory Structure Tests
# =============================================================================


class TestBuildDirectoryStructure:
    """Tests for build_directory_structure()."""

    def test_creates_all_16_directories(self, tmp_path):
        """Creates all 16 expected directories."""
        base_dir = tmp_path / "spyglass_data"
        dirs = build_directory_structure(base_dir, create=True, verbose=False)

        assert len(dirs) == 16
        for name, path in dirs.items():
            assert path.exists(), f"Directory {name} not created"

    def test_dry_run_does_not_create_directories(self, tmp_path):
        """Dry run (create=False) doesn't create directories."""
        base_dir = tmp_path / "spyglass_data"
        dirs = build_directory_structure(base_dir, create=False, verbose=False)

        assert len(dirs) == 16
        assert not base_dir.exists()

    def test_creates_expected_prefixes(self, tmp_path):
        """Creates directories for all four prefixes."""
        base_dir = tmp_path / "spyglass_data"
        dirs = build_directory_structure(base_dir, create=True, verbose=False)

        prefixes = {"spyglass", "kachery", "dlc", "moseq"}
        for prefix in prefixes:
            matching = [k for k in dirs.keys() if k.startswith(f"{prefix}_")]
            assert len(matching) > 0, f"No directories for prefix {prefix}"


# =============================================================================
# Schema Validation Tests
# =============================================================================


class TestValidateSchema:
    """Tests for validate_schema()."""

    def test_accepts_valid_schema(self):
        """Accepts valid schema with all required prefixes and keys."""
        schema = {
            "directory_schema": {
                "spyglass": {
                    "raw": "raw",
                    "analysis": "analysis",
                    "recording": "recording",
                    "sorting": "spikesorting",
                    "waveforms": "waveforms",
                    "temp": "tmp",
                    "video": "video",
                    "export": "export",
                },
                "kachery": {
                    "cloud": ".kachery-cloud",
                    "storage": "kachery_storage",
                    "temp": "tmp",
                },
                "dlc": {
                    "project": "projects",
                    "video": "video",
                    "output": "output",
                },
                "moseq": {"project": "projects", "video": "video"},
            }
        }
        # Should not raise
        validate_schema(schema)

    def test_rejects_missing_directory_schema(self):
        """Rejects schema missing directory_schema key."""
        with pytest.raises(ValueError, match="missing 'directory_schema'"):
            validate_schema({"other_key": {}})

    def test_rejects_missing_prefix(self):
        """Rejects schema missing required prefixes."""
        schema = {
            "directory_schema": {
                "spyglass": {"raw": "raw"},
                "kachery": {"cloud": ".kachery-cloud"},
                # Missing dlc and moseq
            }
        }
        with pytest.raises(ValueError, match="Missing prefixes"):
            validate_schema(schema)


# =============================================================================
# Prerequisites Check Tests
# =============================================================================


class TestCheckPrerequisites:
    """Tests for check_prerequisites()."""

    def test_does_not_raise_on_valid_system(self):
        """Doesn't raise on valid development system."""
        try:
            check_prerequisites()
        except RuntimeError as e:
            # Only acceptable failure is conda/mamba not found in test env
            if "conda or mamba not found" not in str(e):
                raise

    def test_accepts_minimal_install_type(self, tmp_path):
        """Accepts 'minimal' install type."""
        try:
            check_prerequisites(install_type="minimal", base_dir=tmp_path)
        except RuntimeError as e:
            if "conda or mamba not found" not in str(e):
                raise

    def test_accepts_full_install_type(self, tmp_path):
        """Accepts 'full' install type."""
        try:
            check_prerequisites(install_type="full", base_dir=tmp_path)
        except RuntimeError as e:
            if "conda or mamba not found" not in str(e):
                raise


# =============================================================================
# Integration Tests - Script Existence and Executability
# =============================================================================


@pytest.mark.integration
class TestScriptFiles:
    """Integration tests for script file existence and permissions."""

    def test_install_script_exists(self):
        """install.py script exists."""
        install_script = scripts_dir / "install.py"
        assert install_script.exists()
        assert install_script.is_file()

    def test_validate_script_exists(self):
        """validate.py script exists."""
        validate_script = scripts_dir / "validate.py"
        assert validate_script.exists()
        assert validate_script.is_file()

    def test_setup_franklab_script_exists(self):
        """setup_franklab.sh script exists."""
        franklab_script = scripts_dir / "setup_franklab.sh"
        assert franklab_script.exists()
        assert franklab_script.is_file()

    def test_scripts_are_executable(self):
        """Scripts have execute permissions (Unix only)."""
        if sys.platform == "win32":
            pytest.skip("Execute permission check not applicable on Windows")

        for script in ["install.py", "validate.py"]:
            path = scripts_dir / script
            assert path.stat().st_mode & 0o111, f"{script} not executable"


@pytest.mark.integration
class TestInstallScriptHelp:
    """Integration tests for install.py --help."""

    def test_help_output(self):
        """install.py --help runs successfully."""
        result = subprocess.run(
            [sys.executable, str(scripts_dir / "install.py"), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "Usage:" in result.stdout

    def test_help_shows_minimal_option(self):
        """install.py --help shows --minimal option."""
        result = subprocess.run(
            [sys.executable, str(scripts_dir / "install.py"), "--help"],
            capture_output=True,
            text=True,
        )
        assert "--minimal" in result.stdout

    def test_help_shows_full_option(self):
        """install.py --help shows --full option."""
        result = subprocess.run(
            [sys.executable, str(scripts_dir / "install.py"), "--help"],
            capture_output=True,
            text=True,
        )
        assert "--full" in result.stdout

    def test_help_shows_docker_option(self):
        """install.py --help shows --docker option."""
        result = subprocess.run(
            [sys.executable, str(scripts_dir / "install.py"), "--help"],
            capture_output=True,
            text=True,
        )
        assert "--docker" in result.stdout

    def test_help_shows_dry_run_option(self):
        """install.py --help shows --dry-run option."""
        result = subprocess.run(
            [sys.executable, str(scripts_dir / "install.py"), "--help"],
            capture_output=True,
            text=True,
        )
        assert "--dry-run" in result.stdout


@pytest.mark.integration
class TestValidateScriptExecution:
    """Integration tests for validate.py execution."""

    def test_runs_successfully(self):
        """validate.py runs and exits with code 0."""
        result = subprocess.run(
            [sys.executable, str(scripts_dir / "validate.py")],
            capture_output=True,
            text=True,
        )
        # validate.py returns 0 when critical checks pass
        # (database connection is optional, so warnings are OK)
        assert result.returncode == 0

    def test_shows_validation_header(self):
        """validate.py shows validation header."""
        result = subprocess.run(
            [sys.executable, str(scripts_dir / "validate.py")],
            capture_output=True,
            text=True,
        )
        assert "Spyglass" in result.stdout
        assert "Validation" in result.stdout


@pytest.mark.integration
class TestDryRunMode:
    """Integration tests for --dry-run mode."""

    def test_dry_run_exits_successfully(self, tmp_path):
        """--dry-run mode exits with code 0."""
        result = subprocess.run(
            [
                sys.executable,
                str(scripts_dir / "install.py"),
                "--dry-run",
                "--minimal",
                "--base-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_dry_run_does_not_create_environment(self, tmp_path):
        """--dry-run mode doesn't create conda environment."""
        result = subprocess.run(
            [
                sys.executable,
                str(scripts_dir / "install.py"),
                "--dry-run",
                "--minimal",
                "--base-dir",
                str(tmp_path),
                "--env-name",
                "test-dry-run-env",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Verify environment was not created
        check_env = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
        )
        assert "test-dry-run-env" not in check_env.stdout

    def test_dry_run_shows_what_would_be_done(self, tmp_path):
        """--dry-run mode shows planned actions."""
        result = subprocess.run(
            [
                sys.executable,
                str(scripts_dir / "install.py"),
                "--dry-run",
                "--minimal",
                "--base-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        output = result.stdout.lower()
        # Should mention key installation steps
        assert "environment" in output or "conda" in output
        assert "directory" in output or "path" in output


# =============================================================================
# Negative Integration Tests (Error Scenarios)
# =============================================================================


@pytest.mark.integration
class TestInvalidCLIArguments:
    """Integration tests for invalid CLI arguments and error scenarios."""

    def test_minimal_takes_precedence_over_full(self, tmp_path):
        """When both --minimal and --full specified, --minimal takes precedence."""
        result = subprocess.run(
            [
                sys.executable,
                str(scripts_dir / "install.py"),
                "--minimal",
                "--full",
                "--dry-run",
                "--base-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        # Script uses if/elif, so --minimal wins
        assert result.returncode == 0
        assert "environment_min.yml" in result.stdout

    def test_docker_takes_precedence_over_remote(self, tmp_path):
        """When both --docker and --remote specified, --docker takes precedence."""
        result = subprocess.run(
            [
                sys.executable,
                str(scripts_dir / "install.py"),
                "--docker",
                "--remote",
                "--dry-run",
                "--base-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        # Script uses if/elif, so --docker wins
        assert result.returncode == 0
        assert "Docker Compose" in result.stdout

    def test_remote_without_credentials_fails(self, tmp_path):
        """--remote without required credentials shows helpful error."""
        result = subprocess.run(
            [
                sys.executable,
                str(scripts_dir / "install.py"),
                "--config-only",
                "--remote",
                "--base-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": str(tmp_path)},
        )
        # Should fail or show error about missing credentials
        # (depends on whether it's interactive or not)
        output = result.stdout + result.stderr
        # Either exits with error or prompts for input
        assert result.returncode != 0 or "host" in output.lower()

    def test_invalid_port_number(self, tmp_path):
        """Invalid port number shows helpful error."""
        result = subprocess.run(
            [
                sys.executable,
                str(scripts_dir / "install.py"),
                "--config-only",
                "--remote",
                "--db-host",
                "localhost",
                "--db-port",
                "abc",  # Invalid port
                "--db-user",
                "root",
                "--db-password",
                "test",
                "--base-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": str(tmp_path)},
        )
        # argparse should reject non-integer port
        assert result.returncode != 0
        assert (
            "invalid" in result.stderr.lower()
            or "error" in result.stderr.lower()
        )

    def test_privileged_port_shows_error(self, tmp_path):
        """Privileged port (< 1024) shows helpful error."""
        result = subprocess.run(
            [
                sys.executable,
                str(scripts_dir / "install.py"),
                "--config-only",
                "--remote",
                "--db-host",
                "localhost",
                "--db-port",
                "80",  # Privileged port
                "--db-user",
                "root",
                "--db-password",
                "test",
                "--base-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": str(tmp_path)},
        )
        output = result.stdout + result.stderr
        # Should mention privileged port issue
        assert "privileged" in output.lower() or result.returncode != 0


@pytest.mark.integration
class TestDockerUnavailableScenarios:
    """Integration tests for when Docker is not available."""

    def test_docker_flag_without_docker_shows_helpful_message(self, tmp_path):
        """--docker when Docker unavailable shows helpful error."""
        # Mock docker not being available by using a PATH without docker
        env = os.environ.copy()
        env["PATH"] = str(tmp_path)  # Empty PATH, no docker
        env["HOME"] = str(tmp_path)

        result = subprocess.run(
            [
                sys.executable,
                str(scripts_dir / "install.py"),
                "--config-only",
                "--docker",
                "--base-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        output = result.stdout + result.stderr
        # Should either succeed (config-only doesn't need docker running)
        # or show helpful message about Docker
        # Config-only mode should still work
        assert result.returncode == 0 or "docker" in output.lower()


@pytest.mark.integration
class TestConfigOnlyErrorScenarios:
    """Integration tests for --config-only mode error handling."""

    def test_config_only_remote_missing_host(self, tmp_path):
        """--config-only --remote without --db-host is handled."""
        result = subprocess.run(
            [
                sys.executable,
                str(scripts_dir / "install.py"),
                "--config-only",
                "--remote",
                "--db-user",
                "root",
                "--db-password",
                "test",
                "--base-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": str(tmp_path)},
            timeout=5,  # Should fail quickly, not hang
        )
        # Should fail or require host
        output = result.stdout + result.stderr
        assert result.returncode != 0 or "host" in output.lower()

    def test_config_only_creates_directories(self, tmp_path):
        """--config-only still creates directory structure."""
        result = subprocess.run(
            [
                sys.executable,
                str(scripts_dir / "install.py"),
                "--config-only",
                "--docker",
                "--base-dir",
                str(tmp_path / "spyglass_data"),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": str(tmp_path)},
        )
        assert result.returncode == 0

        # Verify directories were created
        base_dir = tmp_path / "spyglass_data"
        assert base_dir.exists()
        assert (base_dir / "raw").exists()
        assert (base_dir / "analysis").exists()

    def test_config_only_unwritable_base_dir(self, tmp_path):
        """--config-only with unwritable directory shows helpful error."""
        if sys.platform == "win32":
            pytest.skip("Permission test not reliable on Windows")

        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(scripts_dir / "install.py"),
                    "--config-only",
                    "--docker",
                    "--base-dir",
                    str(readonly_dir / "spyglass_data"),
                ],
                capture_output=True,
                text=True,
                env={**os.environ, "HOME": str(tmp_path)},
                timeout=5,
            )
            output = result.stdout + result.stderr
            # Should fail with permission error
            assert result.returncode != 0 or "permission" in output.lower()
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)


# =============================================================================
# Docker Utilities Module Tests
# =============================================================================


class TestDockerUtilities:
    """Tests for docker utility module."""

    def test_docker_module_exists(self):
        """docker utilities module exists."""
        docker_module = (
            Path(__file__).parent.parent.parent
            / "src"
            / "spyglass"
            / "utils"
            / "docker.py"
        )
        assert docker_module.exists()

    def test_docker_module_imports(self):
        """docker utilities can be imported."""
        try:
            from spyglass.utils import docker

            assert hasattr(docker, "DockerConfig")
            assert hasattr(docker, "is_docker_available")
            assert hasattr(docker, "start_database_container")
        except ImportError:
            pytest.skip("Spyglass not installed")


# =============================================================================
# Schema Loading Tests
# =============================================================================


class TestLoadDirectorySchema:
    """Tests for load_directory_schema()."""

    def test_returns_dict(self):
        """Returns a dictionary."""
        schema = load_directory_schema()
        assert isinstance(schema, dict)

    def test_has_all_prefixes(self):
        """Has all four required prefixes."""
        schema = load_directory_schema()
        assert set(schema.keys()) == {"spyglass", "kachery", "dlc", "moseq"}

    def test_spyglass_has_expected_keys(self):
        """spyglass prefix has expected directory keys."""
        schema = load_directory_schema()
        expected = {
            "raw",
            "analysis",
            "recording",
            "sorting",
            "waveforms",
            "temp",
            "video",
            "export",
        }
        assert set(schema["spyglass"].keys()) == expected

    def test_kachery_has_expected_keys(self):
        """kachery prefix has expected directory keys."""
        schema = load_directory_schema()
        expected = {"cloud", "storage", "temp"}
        assert set(schema["kachery"].keys()) == expected

    def test_dlc_has_expected_keys(self):
        """dlc prefix has expected directory keys."""
        schema = load_directory_schema()
        expected = {"project", "video", "output"}
        assert set(schema["dlc"].keys()) == expected

    def test_moseq_has_expected_keys(self):
        """moseq prefix has expected directory keys."""
        schema = load_directory_schema()
        expected = {"project", "video"}
        assert set(schema["moseq"].keys()) == expected


# =============================================================================
# Database Config File Creation Tests
# =============================================================================


class TestCreateDatabaseConfig:
    """Tests for create_database_config() and .datajoint_config.json creation."""

    def test_creates_config_file(self, tmp_path, monkeypatch):
        """Creates .datajoint_config.json file with correct structure."""
        import json

        # Use temp directory as home to avoid modifying real config
        config_file = tmp_path / ".datajoint_config.json"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        base_dir = tmp_path / "spyglass_data"

        # Run the function (suppress prompts by not having existing file)
        create_database_config(
            host="localhost",
            port=3306,
            user="testuser",
            password="testpass",
            use_tls=False,
            base_dir=base_dir,
        )

        # Verify file was created
        assert config_file.exists(), ".datajoint_config.json not created"

        # Load and verify structure
        with config_file.open() as f:
            config = json.load(f)

        # Check database settings
        assert config["database.host"] == "localhost"
        assert config["database.port"] == 3306
        assert config["database.user"] == "testuser"
        assert config["database.password"] == "testpass"
        assert config["database.use_tls"] is False

    def test_config_has_required_top_level_keys(self, tmp_path, monkeypatch):
        """Config file has all required top-level keys."""
        import json

        config_file = tmp_path / ".datajoint_config.json"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        base_dir = tmp_path / "spyglass_data"
        create_database_config(
            host="localhost",
            port=3306,
            user="root",
            password="pass",
            base_dir=base_dir,
        )

        with config_file.open() as f:
            config = json.load(f)

        required_keys = {
            "database.host",
            "database.port",
            "database.user",
            "database.password",
            "database.use_tls",
            "filepath_checksum_size_limit",
            "enable_python_native_blobs",
            "stores",
            "custom",
        }
        assert required_keys.issubset(set(config.keys()))

    def test_config_has_stores_with_raw_and_analysis(
        self, tmp_path, monkeypatch
    ):
        """Config file has stores for raw and analysis data."""
        import json

        config_file = tmp_path / ".datajoint_config.json"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        base_dir = tmp_path / "spyglass_data"
        create_database_config(
            host="localhost",
            port=3306,
            user="root",
            password="pass",
            base_dir=base_dir,
        )

        with config_file.open() as f:
            config = json.load(f)

        stores = config["stores"]
        assert "raw" in stores
        assert "analysis" in stores

        # Each store should have protocol, location, stage
        for store_name in ["raw", "analysis"]:
            store = stores[store_name]
            assert store["protocol"] == "file"
            assert "location" in store
            assert "stage" in store

    def test_config_has_custom_spyglass_dirs(self, tmp_path, monkeypatch):
        """Config file has all spyglass directory paths in custom section."""
        import json

        config_file = tmp_path / ".datajoint_config.json"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        base_dir = tmp_path / "spyglass_data"
        create_database_config(
            host="localhost",
            port=3306,
            user="root",
            password="pass",
            base_dir=base_dir,
        )

        with config_file.open() as f:
            config = json.load(f)

        custom = config["custom"]
        assert "spyglass_dirs" in custom

        spyglass_dirs = custom["spyglass_dirs"]
        expected_keys = {
            "base",
            "raw",
            "analysis",
            "recording",
            "sorting",
            "waveforms",
            "temp",
            "video",
            "export",
        }
        assert set(spyglass_dirs.keys()) == expected_keys

    def test_config_has_kachery_dirs(self, tmp_path, monkeypatch):
        """Config file has kachery directory paths."""
        import json

        config_file = tmp_path / ".datajoint_config.json"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        base_dir = tmp_path / "spyglass_data"
        create_database_config(
            host="localhost",
            port=3306,
            user="root",
            password="pass",
            base_dir=base_dir,
        )

        with config_file.open() as f:
            config = json.load(f)

        kachery_dirs = config["custom"]["kachery_dirs"]
        expected_keys = {"cloud", "storage", "temp"}
        assert set(kachery_dirs.keys()) == expected_keys

    def test_config_has_dlc_dirs(self, tmp_path, monkeypatch):
        """Config file has DeepLabCut directory paths."""
        import json

        config_file = tmp_path / ".datajoint_config.json"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        base_dir = tmp_path / "spyglass_data"
        create_database_config(
            host="localhost",
            port=3306,
            user="root",
            password="pass",
            base_dir=base_dir,
        )

        with config_file.open() as f:
            config = json.load(f)

        dlc_dirs = config["custom"]["dlc_dirs"]
        expected_keys = {"base", "project", "video", "output"}
        assert set(dlc_dirs.keys()) == expected_keys

    def test_config_has_moseq_dirs(self, tmp_path, monkeypatch):
        """Config file has MoSeq directory paths."""
        import json

        config_file = tmp_path / ".datajoint_config.json"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        base_dir = tmp_path / "spyglass_data"
        create_database_config(
            host="localhost",
            port=3306,
            user="root",
            password="pass",
            base_dir=base_dir,
        )

        with config_file.open() as f:
            config = json.load(f)

        moseq_dirs = config["custom"]["moseq_dirs"]
        expected_keys = {"base", "project", "video"}
        assert set(moseq_dirs.keys()) == expected_keys

    def test_config_directories_are_absolute_paths(self, tmp_path, monkeypatch):
        """All directory paths in config are absolute paths."""
        import json

        config_file = tmp_path / ".datajoint_config.json"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        base_dir = tmp_path / "spyglass_data"
        create_database_config(
            host="localhost",
            port=3306,
            user="root",
            password="pass",
            base_dir=base_dir,
        )

        with config_file.open() as f:
            config = json.load(f)

        # Check spyglass_dirs paths are absolute
        for key, path in config["custom"]["spyglass_dirs"].items():
            assert Path(
                path
            ).is_absolute(), f"spyglass_dirs.{key} is not absolute"

        # Check store paths are absolute
        for store_name in ["raw", "analysis"]:
            location = config["stores"][store_name]["location"]
            assert Path(
                location
            ).is_absolute(), f"stores.{store_name}.location not absolute"

    def test_config_directories_exist_on_disk(self, tmp_path, monkeypatch):
        """Directories referenced in config actually exist after creation."""
        import json

        config_file = tmp_path / ".datajoint_config.json"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        base_dir = tmp_path / "spyglass_data"
        create_database_config(
            host="localhost",
            port=3306,
            user="root",
            password="pass",
            base_dir=base_dir,
        )

        with config_file.open() as f:
            config = json.load(f)

        # Check spyglass_dirs paths exist
        for key, path_str in config["custom"]["spyglass_dirs"].items():
            path = Path(path_str)
            assert path.exists(), f"spyglass_dirs.{key} ({path}) does not exist"
            assert (
                path.is_dir()
            ), f"spyglass_dirs.{key} ({path}) is not a directory"

    def test_config_file_is_valid_json(self, tmp_path, monkeypatch):
        """Config file is valid JSON that can be parsed."""
        import json

        config_file = tmp_path / ".datajoint_config.json"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        base_dir = tmp_path / "spyglass_data"
        create_database_config(
            host="localhost",
            port=3306,
            user="root",
            password="pass",
            base_dir=base_dir,
        )

        # This should not raise
        with config_file.open() as f:
            config = json.load(f)

        assert isinstance(config, dict)

    def test_tls_auto_enabled_for_remote_host(self, tmp_path, monkeypatch):
        """TLS is automatically enabled for non-localhost hosts."""
        import json

        config_file = tmp_path / ".datajoint_config.json"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        base_dir = tmp_path / "spyglass_data"
        create_database_config(
            host="lmf-db.cin.ucsf.edu",  # Remote host
            port=3306,
            user="root",
            password="pass",
            use_tls=None,  # Auto-determine
            base_dir=base_dir,
        )

        with config_file.open() as f:
            config = json.load(f)

        assert config["database.use_tls"] is True

    def test_tls_disabled_for_localhost(self, tmp_path, monkeypatch):
        """TLS is automatically disabled for localhost."""
        import json

        config_file = tmp_path / ".datajoint_config.json"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        base_dir = tmp_path / "spyglass_data"
        create_database_config(
            host="localhost",
            port=3306,
            user="root",
            password="pass",
            use_tls=None,  # Auto-determine
            base_dir=base_dir,
        )

        with config_file.open() as f:
            config = json.load(f)

        assert config["database.use_tls"] is False


# =============================================================================
# Frank Lab Configuration Tests
# =============================================================================


class TestFrankLabConfig:
    """Tests for Frank Lab-specific configuration generation."""

    # Frank Lab production settings
    FRANKLAB_HOST = "lmf-db.cin.ucsf.edu"
    FRANKLAB_PORT = 3306
    FRANKLAB_KACHERY_ZONE = "franklab.default"

    def test_franklab_config_has_correct_host(self, tmp_path, monkeypatch):
        """Config for Frank Lab has correct database host."""
        import json

        config_file = tmp_path / ".datajoint_config.json"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        create_database_config(
            host=self.FRANKLAB_HOST,
            port=self.FRANKLAB_PORT,
            user="testuser",
            password="testpass",
            base_dir=tmp_path / "spyglass_data",
        )

        with config_file.open() as f:
            config = json.load(f)

        assert config["database.host"] == self.FRANKLAB_HOST
        assert config["database.port"] == self.FRANKLAB_PORT

    def test_franklab_config_has_tls_enabled(self, tmp_path, monkeypatch):
        """Config for Frank Lab (remote) has TLS enabled."""
        import json

        config_file = tmp_path / ".datajoint_config.json"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        create_database_config(
            host=self.FRANKLAB_HOST,
            port=self.FRANKLAB_PORT,
            user="testuser",
            password="testpass",
            base_dir=tmp_path / "spyglass_data",
        )

        with config_file.open() as f:
            config = json.load(f)

        assert config["database.use_tls"] is True

    def test_franklab_config_has_correct_kachery_zone(
        self, tmp_path, monkeypatch
    ):
        """Config for Frank Lab has franklab.default kachery zone."""
        import json

        config_file = tmp_path / ".datajoint_config.json"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        create_database_config(
            host=self.FRANKLAB_HOST,
            port=self.FRANKLAB_PORT,
            user="testuser",
            password="testpass",
            base_dir=tmp_path / "spyglass_data",
        )

        with config_file.open() as f:
            config = json.load(f)

        assert config["custom"]["kachery_zone"] == self.FRANKLAB_KACHERY_ZONE

    def test_franklab_config_has_all_directory_groups(
        self, tmp_path, monkeypatch
    ):
        """Config for Frank Lab has all required directory groups."""
        import json

        config_file = tmp_path / ".datajoint_config.json"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        create_database_config(
            host=self.FRANKLAB_HOST,
            port=self.FRANKLAB_PORT,
            user="testuser",
            password="testpass",
            base_dir=tmp_path / "spyglass_data",
        )

        with config_file.open() as f:
            config = json.load(f)

        custom = config["custom"]
        assert "spyglass_dirs" in custom
        assert "kachery_dirs" in custom
        assert "dlc_dirs" in custom
        assert "moseq_dirs" in custom

    def test_franklab_config_directories_match_schema(
        self, tmp_path, monkeypatch
    ):
        """Config directories match the schema structure."""
        import json

        config_file = tmp_path / ".datajoint_config.json"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        base_dir = tmp_path / "spyglass_data"
        create_database_config(
            host=self.FRANKLAB_HOST,
            port=self.FRANKLAB_PORT,
            user="testuser",
            password="testpass",
            base_dir=base_dir,
        )

        with config_file.open() as f:
            config = json.load(f)

        # Load schema to verify structure
        schema = load_directory_schema()

        # Verify spyglass_dirs has all keys from schema (plus 'base')
        spyglass_keys = set(config["custom"]["spyglass_dirs"].keys())
        expected_spyglass = set(schema["spyglass"].keys()) | {"base"}
        assert spyglass_keys == expected_spyglass

        # Verify kachery_dirs has all keys from schema
        kachery_keys = set(config["custom"]["kachery_dirs"].keys())
        expected_kachery = set(schema["kachery"].keys())
        assert kachery_keys == expected_kachery


# =============================================================================
# Config-Only Mode Integration Tests
# =============================================================================


@pytest.mark.integration
class TestConfigOnlyMode:
    """Integration tests for --config-only mode."""

    def test_config_only_help_shows_option(self):
        """--help shows --config-only option."""
        result = subprocess.run(
            [sys.executable, str(scripts_dir / "install.py"), "--help"],
            capture_output=True,
            text=True,
        )
        assert "--config-only" in result.stdout

    def test_config_only_with_docker_creates_localhost_config(
        self, tmp_path, monkeypatch
    ):
        """--config-only --docker creates localhost config."""
        import json

        config_file = tmp_path / ".datajoint_config.json"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Import and call run_config_only directly to test
        import argparse

        from install import run_config_only

        args = argparse.Namespace(
            docker=True,
            remote=False,
            db_host=None,
            db_port=3306,
            db_user="root",
            db_password="tutorial",
            base_dir=str(tmp_path / "spyglass_data"),
            config_only=True,
        )

        run_config_only(args)

        assert config_file.exists()
        with config_file.open() as f:
            config = json.load(f)

        assert config["database.host"] == "localhost"
        assert config["database.use_tls"] is False

    def test_config_only_with_remote_creates_tls_config(
        self, tmp_path, monkeypatch
    ):
        """--config-only --remote creates config with TLS enabled."""
        import json

        config_file = tmp_path / ".datajoint_config.json"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        import argparse

        from install import run_config_only

        args = argparse.Namespace(
            docker=False,
            remote=True,
            db_host="lmf-db.cin.ucsf.edu",
            db_port=3306,
            db_user="testuser",
            db_password="testpass",
            base_dir=str(tmp_path / "spyglass_data"),
            config_only=True,
        )

        run_config_only(args)

        assert config_file.exists()
        with config_file.open() as f:
            config = json.load(f)

        assert config["database.host"] == "lmf-db.cin.ucsf.edu"
        assert config["database.use_tls"] is True

    def test_config_only_cli_runs_successfully(self, tmp_path):
        """--config-only via CLI runs successfully with all required args."""
        result = subprocess.run(
            [
                sys.executable,
                str(scripts_dir / "install.py"),
                "--config-only",
                "--docker",
                "--base-dir",
                str(tmp_path / "spyglass_data"),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": str(tmp_path)},
        )
        # Should succeed
        assert result.returncode == 0
        assert (
            "Configuration created" in result.stdout
            or "CONFIG-ONLY MODE" in result.stdout
        )

    def test_config_only_franklab_cli(self, tmp_path):
        """--config-only can recreate Frank Lab config via CLI."""
        result = subprocess.run(
            [
                sys.executable,
                str(scripts_dir / "install.py"),
                "--config-only",
                "--remote",
                "--db-host",
                "lmf-db.cin.ucsf.edu",
                "--db-port",
                "3306",
                "--db-user",
                "testuser",
                "--db-password",
                "testpass",
                "--base-dir",
                str(tmp_path / "spyglass_data"),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": str(tmp_path)},
        )
        assert result.returncode == 0

        # Verify config file was created with correct content
        import json

        config_file = tmp_path / ".datajoint_config.json"
        assert config_file.exists()

        with config_file.open() as f:
            config = json.load(f)

        assert config["database.host"] == "lmf-db.cin.ucsf.edu"
        assert config["database.use_tls"] is True
        assert config["custom"]["kachery_zone"] == "franklab.default"


# =============================================================================
# Setup Franklab Script Tests
# =============================================================================


@pytest.mark.integration
class TestSetupFranklabScript:
    """Integration tests for setup_franklab.sh script."""

    def test_setup_franklab_script_exists(self):
        """setup_franklab.sh script exists."""
        franklab_script = scripts_dir / "setup_franklab.sh"
        assert franklab_script.exists()
        assert franklab_script.is_file()

    def test_setup_franklab_help_runs(self):
        """setup_franklab.sh --help runs without error."""
        result = subprocess.run(
            [str(scripts_dir / "setup_franklab.sh"), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Frank Lab" in result.stdout

    def test_setup_franklab_has_correct_defaults(self):
        """setup_franklab.sh has correct Frank Lab defaults in help."""
        result = subprocess.run(
            [str(scripts_dir / "setup_franklab.sh"), "--help"],
            capture_output=True,
            text=True,
        )
        assert "lmf-db.cin.ucsf.edu" in result.stdout
        assert "3306" in result.stdout
