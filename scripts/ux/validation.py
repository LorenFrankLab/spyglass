"""Enhanced input validation with user-friendly error messages.

Replaces boolean validation functions with Result-returning validators
that provide actionable error messages, as recommended in REVIEW.md.
"""

import os
import re
import socket
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse

# Import from utils (using absolute path within scripts)
import sys

scripts_dir = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_dir))

from utils.result_types import (
    ValidationResult,
    validation_success,
    validation_failure,
    Severity,
)


class PortValidator:
    """Validator for network port numbers."""

    @staticmethod
    def validate(value: str) -> ValidationResult:
        """Validate port number input.

        Args:
            value: Port number as string

        Returns:
            ValidationResult with specific error message if invalid

        Example:
            >>> result = PortValidator.validate("3306")
            >>> assert result.is_success

            >>> result = PortValidator.validate("99999")
            >>> assert result.is_failure
            >>> assert "must be between" in result.error.message
        """
        if not value or not value.strip():
            return validation_failure(
                field="port",
                message="Port number is required",
                severity=Severity.ERROR,
                recovery_actions=["Enter a port number between 1 and 65535"],
            )

        try:
            port = int(value.strip())
        except ValueError:
            return validation_failure(
                field="port",
                message="Port must be a valid integer",
                severity=Severity.ERROR,
                recovery_actions=[
                    "Enter a numeric port number (e.g., 3306)",
                    "Common ports: 3306 (MySQL), 5432 (PostgreSQL)",
                ],
            )

        if not (1 <= port <= 65535):
            return validation_failure(
                field="port",
                message=f"Port {port} must be between 1 and 65535",
                severity=Severity.ERROR,
                recovery_actions=[
                    "Use standard MySQL port: 3306",
                    "Choose an available port above 1024",
                    "Check for port conflicts with: lsof -i :PORT",
                ],
            )

        # Check for well-known ports that might cause issues
        if port < 1024:
            return validation_failure(
                field="port",
                message=f"Port {port} is a system/privileged port",
                severity=Severity.WARNING,
                recovery_actions=[
                    "Use port 3306 (standard MySQL port)",
                    "Choose a port above 1024 to avoid permission issues",
                ],
            )

        return validation_success(f"Port {port} is valid")


class PathValidator:
    """Validator for file and directory paths."""

    @staticmethod
    def validate_directory_path(
        value: str, must_exist: bool = False, create_if_missing: bool = False
    ) -> ValidationResult:
        """Validate directory path input.

        Args:
            value: Directory path as string
            must_exist: Whether directory must already exist
            create_if_missing: Whether to create directory if it doesn't exist

        Returns:
            ValidationResult with path validation details
        """
        if not value or not value.strip():
            return validation_failure(
                field="directory_path",
                message="Directory path is required",
                severity=Severity.ERROR,
                recovery_actions=["Enter a valid directory path"],
            )

        try:
            path = Path(value.strip()).expanduser().resolve()
        except (OSError, ValueError) as e:
            return validation_failure(
                field="directory_path",
                message=f"Invalid path format: {e}",
                severity=Severity.ERROR,
                recovery_actions=[
                    "Use absolute paths (e.g., /home/user/spyglass)",
                    "Avoid special characters in path names",
                    "Use ~ for home directory (e.g., ~/spyglass)",
                ],
            )

        # Check for path traversal attempts
        if ".." in str(path):
            return validation_failure(
                field="directory_path",
                message="Path traversal detected (contains '..')",
                severity=Severity.ERROR,
                recovery_actions=[
                    "Use absolute paths without '..' components",
                    "Specify direct path to target directory",
                ],
            )

        # Check if path exists
        if must_exist and not path.exists():
            return validation_failure(
                field="directory_path",
                message=f"Directory does not exist: {path}",
                severity=Severity.ERROR,
                recovery_actions=[
                    f"Create directory: mkdir -p {path}",
                    "Check path spelling and permissions",
                    "Use an existing directory",
                ],
            )

        # Check if parent exists (for creation)
        if not path.exists() and not path.parent.exists():
            return validation_failure(
                field="directory_path",
                message=f"Parent directory does not exist: {path.parent}",
                severity=Severity.ERROR,
                recovery_actions=[
                    f"Create parent directory: mkdir -p {path.parent}",
                    "Choose a path with existing parent directory",
                ],
            )

        # Check permissions
        if path.exists():
            if not path.is_dir():
                return validation_failure(
                    field="directory_path",
                    message=f"Path exists but is not a directory: {path}",
                    severity=Severity.ERROR,
                    recovery_actions=[
                        "Choose a different path",
                        "Remove the existing file if not needed",
                    ],
                )

            if not os.access(path, os.W_OK):
                return validation_failure(
                    field="directory_path",
                    message=f"No write permission for directory: {path}",
                    severity=Severity.ERROR,
                    recovery_actions=[
                        f"Fix permissions: chmod u+w {path}",
                        "Choose a directory you have write access to",
                        "Run with appropriate user permissions",
                    ],
                )

        return validation_success(f"Directory path '{path}' is valid")

    @staticmethod
    def validate_base_directory(
        value: str, min_space_gb: float = 10.0
    ) -> ValidationResult:
        """Validate base directory for Spyglass installation.

        Args:
            value: Base directory path
            min_space_gb: Minimum required space in GB

        Returns:
            ValidationResult with space and permission checks
        """
        # First validate as regular directory
        path_result = PathValidator.validate_directory_path(
            value, must_exist=False
        )
        if path_result.is_failure:
            return path_result

        path = Path(value).expanduser().resolve()

        # Check available disk space
        try:
            import shutil

            _, _, available_bytes = shutil.disk_usage(
                path.parent if path.exists() else path.parent
            )
            available_gb = available_bytes / (1024**3)

            if available_gb < min_space_gb:
                return validation_failure(
                    field="base_directory",
                    message=f"Insufficient disk space: {available_gb:.1f}GB available, {min_space_gb}GB required",
                    severity=Severity.ERROR,
                    recovery_actions=[
                        "Free up disk space by deleting unnecessary files",
                        "Choose a different location with more space",
                        "Use minimal installation to reduce space requirements",
                    ],
                )

            space_warning_threshold = min_space_gb * 1.5
            if available_gb < space_warning_threshold:
                return validation_failure(
                    field="base_directory",
                    message=f"Low disk space: {available_gb:.1f}GB available (recommended: {space_warning_threshold:.1f}GB+)",
                    severity=Severity.WARNING,
                    recovery_actions=[
                        "Consider freeing up more space for sample data",
                        "Monitor disk usage during installation",
                    ],
                )

        except (OSError, ValueError) as e:
            return validation_failure(
                field="base_directory",
                message=f"Cannot check disk space: {e}",
                severity=Severity.WARNING,
                recovery_actions=[
                    "Ensure you have sufficient space (~10GB minimum)",
                    "Check disk usage manually with: df -h",
                ],
            )

        return validation_success(
            f"Base directory '{path}' is valid with {available_gb:.1f}GB available"
        )


class HostValidator:
    """Validator for database host addresses."""

    @staticmethod
    def validate(value: str) -> ValidationResult:
        """Validate database host input.

        Args:
            value: Host address as string

        Returns:
            ValidationResult with host validation details
        """
        if not value or not value.strip():
            return validation_failure(
                field="host",
                message="Host address is required",
                severity=Severity.ERROR,
                recovery_actions=[
                    "Enter a host address (e.g., localhost, 192.168.1.100)"
                ],
            )

        host = value.strip()

        # Check for valid hostname/IP format
        if not HostValidator._is_valid_hostname(
            host
        ) and not HostValidator._is_valid_ip(host):
            return validation_failure(
                field="host",
                message=f"Invalid host format: {host}",
                severity=Severity.ERROR,
                recovery_actions=[
                    "Use localhost for local database",
                    "Use valid IP address (e.g., 192.168.1.100)",
                    "Use valid hostname (e.g., database.example.com)",
                ],
            )

        # Warn about localhost alternatives
        if host.lower() in ["127.0.0.1", "::1"]:
            return validation_failure(
                field="host",
                message=f"Using {host} (consider 'localhost' for clarity)",
                severity=Severity.INFO,
                recovery_actions=["Use 'localhost' for local connections"],
            )

        return validation_success(f"Host '{host}' is valid")

    @staticmethod
    def _is_valid_hostname(hostname: str) -> bool:
        """Check if string is a valid hostname."""
        if len(hostname) > 253:
            return False

        # Remove trailing dot
        if hostname.endswith("."):
            hostname = hostname[:-1]

        # Check each label
        allowed = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$")
        labels = hostname.split(".")

        return all(allowed.match(label) for label in labels)

    @staticmethod
    def _is_valid_ip(ip: str) -> bool:
        """Check if string is a valid IP address."""
        try:
            socket.inet_aton(ip)
            return True
        except socket.error:
            return False


class EnvironmentNameValidator:
    """Validator for conda environment names."""

    @staticmethod
    def validate(value: str) -> ValidationResult:
        """Validate conda environment name.

        Args:
            value: Environment name as string

        Returns:
            ValidationResult with name validation details
        """
        if not value or not value.strip():
            return validation_failure(
                field="environment_name",
                message="Environment name is required",
                severity=Severity.ERROR,
                recovery_actions=[
                    "Enter a valid environment name (e.g., spyglass)"
                ],
            )

        name = value.strip()

        # Check for valid conda environment name format
        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            return validation_failure(
                field="environment_name",
                message="Environment name contains invalid characters",
                severity=Severity.ERROR,
                recovery_actions=[
                    "Use only letters, numbers, underscores, and hyphens",
                    "Example: spyglass, spyglass_v1, my-analysis",
                ],
            )

        # Check length
        if len(name) > 50:
            return validation_failure(
                field="environment_name",
                message=f"Environment name too long ({len(name)} chars, max 50)",
                severity=Severity.ERROR,
                recovery_actions=["Use a shorter environment name"],
            )

        # Warn about reserved names
        reserved_names = ["base", "root", "conda", "python", "pip"]
        if name.lower() in reserved_names:
            return validation_failure(
                field="environment_name",
                message=f"'{name}' is a reserved name",
                severity=Severity.WARNING,
                recovery_actions=[
                    "Use a different name (e.g., spyglass, my_analysis)",
                    "Avoid reserved conda/python names",
                ],
            )

        return validation_success(f"Environment name '{name}' is valid")


# Convenience functions for common validations
def validate_port(port_str: str) -> ValidationResult:
    """Validate port number string."""
    return PortValidator.validate(port_str)


def validate_directory(
    path_str: str, must_exist: bool = False
) -> ValidationResult:
    """Validate directory path string."""
    return PathValidator.validate_directory_path(path_str, must_exist)


def validate_base_directory(
    path_str: str, min_space_gb: float = 10.0
) -> ValidationResult:
    """Validate base directory with space requirements."""
    return PathValidator.validate_base_directory(path_str, min_space_gb)


def validate_host(host_str: str) -> ValidationResult:
    """Validate database host string."""
    return HostValidator.validate(host_str)


def validate_environment_name(name_str: str) -> ValidationResult:
    """Validate conda environment name string."""
    return EnvironmentNameValidator.validate(name_str)
