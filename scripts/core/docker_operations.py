"""Pure functions for Docker database operations.

Extracted from quickstart.py setup_docker_database() to separate business
logic from I/O operations, as recommended in REVIEW.md.
"""

import subprocess
import socket
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass

# Import from utils (using absolute path within scripts)
import sys
scripts_dir = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_dir))

from utils.result_types import (
    Result, success, failure, DockerResult, DockerError
)


@dataclass(frozen=True)
class DockerConfig:
    """Configuration for Docker database setup."""
    container_name: str = "spyglass-db"
    image: str = "datajoint/mysql:8.0"
    port: int = 3306
    password: str = "tutorial"
    mysql_port: int = 3306


@dataclass(frozen=True)
class DockerContainerInfo:
    """Information about Docker container state."""
    name: str
    exists: bool
    running: bool
    port_mapping: str


def build_docker_run_command(config: DockerConfig) -> List[str]:
    """Build docker run command from configuration.

    Pure function - no side effects, easy to test.

    Args:
        config: Docker configuration

    Returns:
        List of command arguments for docker run

    Example:
        >>> config = DockerConfig(port=3307)
        >>> cmd = build_docker_run_command(config)
        >>> assert "-p 3307:3306" in " ".join(cmd)
    """
    port_mapping = f"{config.port}:{config.mysql_port}"

    return [
        "docker", "run", "-d",
        "--name", config.container_name,
        "-p", port_mapping,
        "-e", f"MYSQL_ROOT_PASSWORD={config.password}",
        config.image
    ]


def build_docker_pull_command(config: DockerConfig) -> List[str]:
    """Build docker pull command from configuration.

    Pure function - no side effects.

    Args:
        config: Docker configuration

    Returns:
        List of command arguments for docker pull
    """
    return ["docker", "pull", config.image]


def build_mysql_ping_command(config: DockerConfig) -> List[str]:
    """Build MySQL ping command for readiness check.

    Pure function - no side effects.

    Args:
        config: Docker configuration

    Returns:
        List of command arguments for MySQL ping
    """
    return [
        "docker", "exec", config.container_name,
        "mysqladmin", "-uroot", f"-p{config.password}", "ping"
    ]


def check_docker_available() -> DockerResult:
    """Check if Docker is available in PATH.

    Returns:
        Result indicating Docker availability
    """
    import shutil

    if not shutil.which("docker"):
        return failure(
            DockerError(
                operation="check_availability",
                docker_available=False,
                daemon_running=False,
                permission_error=False
            ),
            "Docker is not installed or not in PATH",
            recovery_actions=[
                "Install Docker from: https://docs.docker.com/engine/install/",
                "Make sure docker command is in your PATH"
            ]
        )

    return success(True, "Docker command found")


def check_docker_daemon_running() -> DockerResult:
    """Check if Docker daemon is running.

    Returns:
        Result indicating daemon status
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            return success(True, "Docker daemon is running")
        else:
            return failure(
                DockerError(
                    operation="check_daemon",
                    docker_available=True,
                    daemon_running=False,
                    permission_error="permission denied" in result.stderr.lower()
                ),
                "Docker daemon is not running",
                recovery_actions=[
                    "Start Docker Desktop application (macOS/Windows)",
                    "Run: sudo systemctl start docker (Linux)",
                    "Check Docker Desktop is running and accessible"
                ]
            )

    except subprocess.TimeoutExpired:
        return failure(
            DockerError(
                operation="check_daemon",
                docker_available=True,
                daemon_running=False,
                permission_error=False
            ),
            "Docker daemon check timed out",
            recovery_actions=[
                "Check if Docker Desktop is starting up",
                "Restart Docker Desktop",
                "Check system resources and Docker configuration"
            ]
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return failure(
            DockerError(
                operation="check_daemon",
                docker_available=True,
                daemon_running=False,
                permission_error="permission" in str(e).lower()
            ),
            f"Failed to check Docker daemon: {e}",
            recovery_actions=[
                "Verify Docker installation",
                "Check Docker permissions",
                "Restart Docker service"
            ]
        )


def check_port_available(port: int) -> DockerResult:
    """Check if specified port is available.

    Args:
        port: Port number to check

    Returns:
        Result indicating port availability
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('localhost', port))

            if result == 0:
                return failure(
                    DockerError(
                        operation="check_port",
                        docker_available=True,
                        daemon_running=True,
                        permission_error=False
                    ),
                    f"Port {port} is already in use",
                    recovery_actions=[
                        f"Use a different port with --db-port (e.g., --db-port {port + 1})",
                        f"Stop service using port {port}",
                        "Check what's running on the port with: lsof -i :3306"
                    ]
                )
            else:
                return success(True, f"Port {port} is available")

    except Exception as e:
        return failure(
            DockerError(
                operation="check_port",
                docker_available=True,
                daemon_running=True,
                permission_error=False
            ),
            f"Failed to check port availability: {e}",
            recovery_actions=[
                "Check network configuration",
                "Try a different port number"
            ]
        )


def get_container_info(container_name: str) -> DockerResult:
    """Get information about Docker container.

    Args:
        container_name: Name of container to check

    Returns:
        Result containing container information
    """
    try:
        # Check if container exists
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return failure(
                DockerError(
                    operation="list_containers",
                    docker_available=True,
                    daemon_running=False,
                    permission_error=False
                ),
                "Failed to list Docker containers",
                recovery_actions=[
                    "Check Docker daemon is running",
                    "Verify Docker permissions"
                ]
            )

        exists = container_name in result.stdout

        if not exists:
            container_info = DockerContainerInfo(
                name=container_name,
                exists=False,
                running=False,
                port_mapping=""
            )
            return success(container_info, f"Container '{container_name}' does not exist")

        # Check if container is running
        running_result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10
        )

        running = container_name in running_result.stdout

        container_info = DockerContainerInfo(
            name=container_name,
            exists=True,
            running=running,
            port_mapping=""  # Could be enhanced to parse port mapping
        )

        status = "running" if running else "stopped"
        return success(container_info, f"Container '{container_name}' exists and is {status}")

    except subprocess.TimeoutExpired:
        return failure(
            DockerError(
                operation="get_container_info",
                docker_available=True,
                daemon_running=False,
                permission_error=False
            ),
            "Timeout checking container status",
            recovery_actions=[
                "Check Docker daemon responsiveness",
                "Restart Docker if needed"
            ]
        )
    except Exception as e:
        return failure(
            DockerError(
                operation="get_container_info",
                docker_available=True,
                daemon_running=True,
                permission_error=False
            ),
            f"Failed to get container info: {e}",
            recovery_actions=[
                "Check Docker installation",
                "Verify container name is correct"
            ]
        )


def validate_docker_prerequisites(config: DockerConfig) -> List[DockerResult]:
    """Validate all Docker prerequisites.

    Pure function that orchestrates all validation checks.

    Args:
        config: Docker configuration to validate

    Returns:
        List of validation results
    """
    validations = [
        check_docker_available(),
        check_docker_daemon_running(),
        check_port_available(config.port)
    ]

    # Only check container info if Docker is available
    if validations[0].is_success and validations[1].is_success:
        container_result = get_container_info(config.container_name)
        validations.append(container_result)

    return validations


def assess_docker_readiness(validations: List[DockerResult]) -> DockerResult:
    """Assess overall Docker readiness from validation results.

    Pure function - takes validation results, returns assessment.

    Args:
        validations: List of validation results

    Returns:
        Overall readiness assessment
    """
    failures = [v for v in validations if v.is_failure]

    if not failures:
        return success(True, "Docker is ready for database setup")

    # Categorize failures
    critical_failures = []
    recoverable_failures = []

    for failure_result in failures:
        if failure_result.error.operation in ["check_availability", "check_daemon"]:
            critical_failures.append(failure_result)
        else:
            recoverable_failures.append(failure_result)

    if critical_failures:
        # Combine error messages and recovery actions
        messages = [f.message for f in critical_failures]
        all_actions = []
        for f in critical_failures:
            all_actions.extend(f.recovery_actions)

        return failure(
            DockerError(
                operation="overall_assessment",
                docker_available=len([f for f in critical_failures
                                    if f.error.operation == "check_availability"]) == 0,
                daemon_running=len([f for f in critical_failures
                                  if f.error.operation == "check_daemon"]) == 0,
                permission_error=any(f.error.permission_error for f in critical_failures)
            ),
            f"Critical Docker issues: {'; '.join(messages)}",
            recovery_actions=list(dict.fromkeys(all_actions))  # Remove duplicates
        )

    elif recoverable_failures:
        # Non-critical issues that can be worked around
        messages = [f.message for f in recoverable_failures]
        all_actions = []
        for f in recoverable_failures:
            all_actions.extend(f.recovery_actions)

        return success(
            True,
            f"Docker ready with minor issues: {'; '.join(messages)}"
        )

    return success(True, "Docker is ready")


