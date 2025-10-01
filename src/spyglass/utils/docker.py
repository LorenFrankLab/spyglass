"""Docker utilities for Spyglass database setup.

This module provides utilities for managing MySQL database containers via Docker.
These utilities are used by:
1. Testing infrastructure (tests/container.py)
2. Post-installation database management
3. NOT for the installer (installer uses inline code to avoid circular dependency)
"""

import subprocess
import shutil
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class DockerConfig:
    """Docker container configuration for MySQL database."""

    container_name: str = "spyglass-db"
    image: str = "datajoint/mysql:8.0"
    port: int = 3306
    password: str = "tutorial"


def is_docker_available() -> bool:
    """Check if Docker is installed and daemon is running.

    Returns:
        True if Docker is available, False otherwise
    """
    if not shutil.which("docker"):
        return False

    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def container_exists(container_name: str) -> bool:
    """Check if a Docker container exists.

    Args:
        container_name: Name of the container to check

    Returns:
        True if container exists, False otherwise
    """
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    return container_name in result.stdout


def start_database_container(config: Optional[DockerConfig] = None) -> None:
    """Start MySQL database container.

    Args:
        config: Docker configuration (uses defaults if None)

    Raises:
        RuntimeError: If Docker is not available or container fails to start
    """
    if config is None:
        config = DockerConfig()

    if not is_docker_available():
        raise RuntimeError(
            "Docker is not available. Install from: "
            "https://docs.docker.com/get-docker/"
        )

    # Check if container already exists
    if container_exists(config.container_name):
        # Start existing container
        subprocess.run(
            ["docker", "start", config.container_name],
            check=True,
        )
    else:
        # Pull image first (better UX - shows progress)
        subprocess.run(["docker", "pull", config.image], check=True)

        # Create and start new container
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                config.container_name,
                "-p",
                f"{config.port}:3306",
                "-e",
                f"MYSQL_ROOT_PASSWORD={config.password}",
                config.image,
            ],
            check=True,
        )

    # Wait for MySQL to be ready
    wait_for_mysql(config)


def wait_for_mysql(config: Optional[DockerConfig] = None, timeout: int = 60) -> None:
    """Wait for MySQL to be ready to accept connections.

    Args:
        config: Docker configuration (uses defaults if None)
        timeout: Maximum time to wait in seconds

    Raises:
        TimeoutError: If MySQL does not become ready within timeout
    """
    if config is None:
        config = DockerConfig()

    for attempt in range(timeout // 2):
        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    config.container_name,
                    "mysqladmin",
                    "-uroot",
                    f"-p{config.password}",
                    "ping",
                ],
                capture_output=True,
                timeout=5,
            )

            if result.returncode == 0:
                return  # Success!

        except subprocess.TimeoutExpired:
            pass

        if attempt < (timeout // 2) - 1:
            time.sleep(2)

    raise TimeoutError(
        f"MySQL did not become ready within {timeout}s. "
        f"Check logs: docker logs {config.container_name}"
    )


def stop_database_container(config: Optional[DockerConfig] = None) -> None:
    """Stop MySQL database container.

    Args:
        config: Docker configuration (uses defaults if None)
    """
    if config is None:
        config = DockerConfig()

    try:
        subprocess.run(
            ["docker", "stop", config.container_name],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        # Container may not be running, that's okay
        pass


def remove_database_container(config: Optional[DockerConfig] = None) -> None:
    """Remove MySQL database container.

    Args:
        config: Docker configuration (uses defaults if None)
    """
    if config is None:
        config = DockerConfig()

    try:
        subprocess.run(
            ["docker", "rm", "-f", config.container_name],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        # Container may not exist, that's okay
        pass
