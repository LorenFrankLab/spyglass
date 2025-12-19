#!/usr/bin/env python3
"""Cross-platform Spyglass installer.

This script automates the Spyglass installation process, reducing setup from
~30 manual steps to 2-3 interactive prompts.

Usage:
    python scripts/install.py              # Interactive mode
    python scripts/install.py --minimal    # Minimal install
    python scripts/install.py --full       # Full install
    python scripts/install.py --docker     # Include database setup
    python scripts/install.py --help       # Show help

Environment Variables:
    SPYGLASS_BASE_DIR - Set base directory (skips prompt)

Exit codes:
    0 - Installation successful
    1 - Installation failed
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

# Color codes for cross-platform output
# Windows fallback uses ASCII symbols instead of empty strings for accessibility
COLORS = (
    {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "blue": "\033[94m",
        "reset": "\033[0m",
    }
    if sys.platform != "win32"
    else {"green": "", "yellow": "", "red": "", "blue": "", "reset": ""}
)

# Status symbols - Windows uses ASCII fallback for compatibility
SYMBOLS = (
    {"success": "✓", "error": "✗", "warning": "⚠", "step": "▶"}
    if sys.platform != "win32"
    else {"success": "[OK]", "error": "[X]", "warning": "[!]", "step": "->"}
)

# System constants
BYTES_PER_GB = 1024**3
LOCALHOST_ADDRESSES = frozenset(["localhost", "127.0.0.1", "::1"])
CURRENT_SCHEMA_VERSION = "1.0.0"  # Config schema version compatibility

# Repository root - works regardless of where script is run from
REPO_ROOT = Path(__file__).parent.parent

# Disk space requirements (GB) - includes packages + working space buffer
# Minimal: ~8 GB packages + ~2 GB working space = 10 GB total
# Full: ~18 GB packages + ~7 GB working space = 25 GB total
DISK_SPACE_REQUIREMENTS = {
    "minimal": {"packages": 8, "total": 10},
    "full": {"packages": 18, "total": 25},
}

# MySQL health check configuration
MYSQL_HEALTH_CHECK_INTERVAL = 2  # seconds
MYSQL_HEALTH_CHECK_ATTEMPTS = 30  # 60 seconds total
MYSQL_HEALTH_CHECK_TIMEOUT = (
    MYSQL_HEALTH_CHECK_ATTEMPTS * MYSQL_HEALTH_CHECK_INTERVAL
)

# Docker configuration
DOCKER_IMAGE_PULL_TIMEOUT = 300  # 5 minutes
DOCKER_STARTUP_TIMEOUT = 60  # 1 minute
DEFAULT_MYSQL_PORT = 3306
DEFAULT_MYSQL_PASSWORD = "tutorial"

# Environment creation time estimates (minutes)
ENV_CREATION_TIME_MINIMAL = 5
ENV_CREATION_TIME_FULL = 15


# Named tuple for database menu options
class DatabaseOption(NamedTuple):
    """Represents a database setup option in the menu.

    Attributes
    ----------
    number : str
        Menu option number (e.g., "1", "2")
    name : str
        Short name of option (e.g., "Docker", "Remote")
    status : str
        Availability status with icon (e.g., "✓ Available", "✗ Not available")
    description : str
        Detailed description for user
    """

    number: str
    name: str
    status: str
    description: str


# =============================================================================
# Console Class - Terminal Output and User Interaction
# =============================================================================


class Console:
    """Handles all terminal output formatting and user interaction.

    Provides consistent colored output, progress messages, and user prompts.
    All methods are static since no instance state is needed.

    The step/done/fail pattern provides tighter output:
        Console.step("Checking prerequisites")
        # ... do work ...
        Console.done()  # prints: "Checking prerequisites... ✓"

    For standalone status messages, use success/warning/error directly.
    """

    @staticmethod
    def step(msg: str) -> None:
        """Print step message without newline, waiting for done/fail."""
        print(f"{msg}... ", end="", flush=True)

    @staticmethod
    def done(msg: str = "") -> None:
        """Complete a step with success checkmark."""
        suffix = f" {msg}" if msg else ""
        print(f"{COLORS['green']}{SYMBOLS['success']}{COLORS['reset']}{suffix}")

    @staticmethod
    def fail(msg: str = "") -> None:
        """Complete a step with failure mark."""
        suffix = f" {msg}" if msg else ""
        print(f"{COLORS['red']}{SYMBOLS['error']}{COLORS['reset']}{suffix}")

    @staticmethod
    def success(msg: str, indent: bool = False) -> None:
        """Print standalone success message (with newline)."""
        prefix = "  " if indent else ""
        print(
            f"{prefix}{COLORS['green']}{SYMBOLS['success']}{COLORS['reset']} {msg}"
        )

    @staticmethod
    def warning(msg: str, indent: bool = False) -> None:
        """Print warning message."""
        prefix = "  " if indent else ""
        print(
            f"{prefix}{COLORS['yellow']}{SYMBOLS['warning']}{COLORS['reset']} {msg}"
        )

    @staticmethod
    def error(msg: str, indent: bool = False) -> None:
        """Print error message."""
        prefix = "  " if indent else ""
        print(
            f"{prefix}{COLORS['red']}{SYMBOLS['error']}{COLORS['reset']} {msg}"
        )

    @staticmethod
    def manual_password_instructions(env_name: str) -> None:
        """Print instructions for manual password change."""
        print("\nYou can change it manually later:")
        print(f"  conda activate {env_name}")
        print("  python -c 'import datajoint as dj; dj.set_password()'")

    @staticmethod
    def prompt_yes_no(prompt: str, default_yes: bool = False) -> bool:
        """Prompt user for yes/no confirmation.

        Parameters
        ----------
        prompt : str
            Question to ask user (without the [Y/n] suffix)
        default_yes : bool
            If True, default is yes when user presses Enter.

        Returns
        -------
        bool
            True if user answered yes, False otherwise
        """
        suffix = "[Y/n]" if default_yes else "[y/N]"
        response = input(f"{prompt} {suffix}: ").strip().lower()
        if not response:
            return default_yes
        return response in ["y", "yes"]

    @staticmethod
    def progress(operation: str, estimated_minutes: int) -> None:
        """Show progress message for long-running operation.

        Unlike step(), this prints on its own line with details below.
        """
        print(f"{operation}...")
        print(f"  Estimated time: ~{estimated_minutes} minute(s)")
        print("  This may take a while - please be patient...")
        if estimated_minutes > 10:
            print("  Tip: This is a good time for a coffee break")


def get_required_python_version() -> Tuple[int, int]:
    """Get required Python version from pyproject.toml.

    Returns
    -------
    tuple of int
        Tuple of (major, minor) version. Falls back to (3, 9) if parsing fails.

    Notes
    -----
    This ensures single source of truth for version requirements.

    INTENTIONAL DUPLICATION: This function is duplicated in both install.py
    and validate.py because validate.py must work standalone before Spyglass
    is installed. Both scripts are designed to run independently without
    importing from each other to avoid path/module complexity.

    If you modify this function, you MUST update it in both files:
    - scripts/install.py (this file)
    - scripts/validate.py

    Future: Consider extracting to scripts/_shared.py if the installer
    becomes a package, but for now standalone scripts are simpler.
    """
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # Fallback for Python 3.9-3.10
        except ImportError:
            return (3, 9)  # Safe fallback

    try:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)

        # Parse ">=3.9,<3.13" format
        requires_python = data["project"]["requires-python"]
        match = re.search(r">=(\d+)\.(\d+)", requires_python)
        if match:
            return (int(match.group(1)), int(match.group(2)))
    except (FileNotFoundError, KeyError, AttributeError, ValueError):
        # Expected errors during parsing - use safe fallback
        pass

    return (3, 9)  # Safe fallback


def check_disk_space(required_gb: int, path: Path) -> Tuple[bool, int]:
    """Check available disk space at given path.

    Walks up directory tree to find existing parent if path doesn't
    exist yet, then checks available disk space.

    Parameters
    ----------
    required_gb : int
        Required disk space in gigabytes
    path : pathlib.Path
        Path to check. If doesn't exist, checks nearest existing parent.

    Returns
    -------
    sufficient : bool
        True if available space >= required space
    available_gb : int
        Available disk space in gigabytes

    Examples
    --------
    >>> sufficient, available = check_disk_space(10, Path("/tmp"))
    >>> if sufficient:
    ...     print(f"OK: {available} GB available")
    """
    # Find existing path to check
    check_path = path
    while not check_path.exists() and check_path != check_path.parent:
        check_path = check_path.parent

    # Get disk usage
    usage = shutil.disk_usage(check_path)
    available_gb = usage.free / BYTES_PER_GB

    return available_gb >= required_gb, int(available_gb)


def check_prerequisites(
    install_type: str = "minimal", base_dir: Optional[Path] = None
) -> None:
    """Check system prerequisites before installation.

    Verifies Python version, conda/mamba availability, and sufficient
    disk space for the selected installation type.

    Parameters
    ----------
    install_type : str, optional
        Installation type - either 'minimal' or 'full' (default: 'minimal')
    base_dir : pathlib.Path, optional
        Base directory where Spyglass data will be stored

    Raises
    ------
    RuntimeError
        If prerequisites are not met (insufficient disk space, etc.)

    Examples
    --------
    >>> check_prerequisites("minimal", Path("/tmp/spyglass_data"))
    """
    print("Checking prerequisites...")

    # Get Python version requirement from pyproject.toml
    min_version = get_required_python_version()

    # Python version
    if sys.version_info < min_version:
        raise RuntimeError(
            f"Python {min_version[0]}.{min_version[1]}+ required, "
            f"found {sys.version_info.major}.{sys.version_info.minor}"
        )
    Console.success(
        f"Python {sys.version_info.major}.{sys.version_info.minor}", indent=True
    )

    # Conda/Mamba
    conda_cmd = CondaManager.get_command()
    Console.success(f"Package manager: {conda_cmd}", indent=True)

    # Git (optional but recommended)
    if not shutil.which("git"):
        Console.warning(
            "Git not found (recommended for development)", indent=True
        )
    else:
        Console.success("Git available", indent=True)

    # Disk space check (if base_dir provided)
    if not base_dir:
        return

    # Use centralized disk space requirements
    space_req = DISK_SPACE_REQUIREMENTS.get(
        install_type, DISK_SPACE_REQUIREMENTS["minimal"]
    )
    required_gb = space_req["total"]
    packages_gb = space_req["packages"]
    buffer_gb = required_gb - packages_gb

    sufficient, available_gb = check_disk_space(required_gb, base_dir)

    if sufficient:
        Console.success(
            f"Disk space: {available_gb} GB available (need {required_gb} GB)",
            indent=True,
        )
        return

    needed_to_free = required_gb - available_gb
    Console.error(
        "Insufficient disk space - installation cannot continue",
        indent=True,
    )
    print(f"    Checking: {base_dir}")
    print(f"    Available: {available_gb} GB")
    print(
        f"    Required:  {required_gb} GB ({install_type}: ~{packages_gb} GB packages + ~{buffer_gb} GB buffer)"
    )
    print(f"    Need to free: {needed_to_free} GB")
    print()
    print("    To fix:")
    print(f"      1. Free at least {needed_to_free} GB in this location")
    print(
        "      2. Choose different directory: python scripts/install.py --base-dir /other/path"
    )
    min_total = DISK_SPACE_REQUIREMENTS["minimal"]["total"]
    print(
        f"      3. Use minimal install (needs {min_total} GB): python scripts/install.py --minimal"
    )
    raise RuntimeError("Insufficient disk space")


# =============================================================================
# CondaManager Class - Conda Environment Operations
# =============================================================================


class CondaManager:
    """Manages conda/mamba environment creation and package installation.

    Attributes
    ----------
    env_name : str
        Name of the conda environment to manage
    """

    def __init__(self, env_name: str = "spyglass"):
        self.env_name = env_name

    @staticmethod
    def get_command() -> str:
        """Get conda/mamba command (required for all environment operations).

        Prefers mamba over conda for faster package resolution.
        """
        if shutil.which("mamba"):
            return "mamba"
        elif shutil.which("conda"):
            return "conda"
        else:
            raise RuntimeError(
                "Neither mamba nor conda found. Install from:\n"
                "  https://github.com/conda-forge/miniforge"
            )

    def exists(self) -> bool:
        """Check if the conda environment already exists."""
        conda_cmd = self.get_command()
        try:
            result = subprocess.run(
                [conda_cmd, "env", "list", "--json"],
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(result.stdout)
            envs = data.get("envs", [])
            names = {Path(env).name for env in envs if env}
            return self.env_name in names
        except (
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            OSError,
        ):
            result = subprocess.run(
                [conda_cmd, "env", "list"], capture_output=True, text=True
            )
            if result.returncode != 0:
                return False
            names = set()
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "*" and len(parts) > 1:
                    names.add(parts[1])
                    continue
                names.add(parts[0])
            return self.env_name in names

    def remove(self) -> None:
        """Remove the conda environment."""
        conda_cmd = self.get_command()
        Console.step(f"Removing existing environment '{self.env_name}'")
        subprocess.run(
            [conda_cmd, "env", "remove", "-n", self.env_name, "-y"], check=True
        )
        Console.done()

    def create(self, env_file: str, force: bool = False) -> None:
        """Create conda environment from file.

        Parameters
        ----------
        env_file : str
            Name of environment YAML file (relative to REPO_ROOT)
        force : bool
            If True, overwrite existing environment without prompting
        """
        estimated_time = (
            ENV_CREATION_TIME_MINIMAL
            if "min" in env_file
            else ENV_CREATION_TIME_FULL
        )
        Console.progress(
            f"Creating environment '{self.env_name}' from {env_file}",
            estimated_time,
        )

        if self.exists():
            if not force:
                if not Console.prompt_yes_no(
                    f"Environment '{self.env_name}' exists. Overwrite?",
                    default_yes=False,
                ):
                    Console.success(
                        f"Using existing environment '{self.env_name}'"
                    )
                    print(
                        "  Package installation will continue (updates if needed)"
                    )
                    print(
                        "  To use a different name, run with: --env-name <name>"
                    )
                    return
            self.remove()

        conda_cmd = self.get_command()
        print("  Installing packages... (this will take several minutes)")

        env_file_path = REPO_ROOT / env_file

        try:
            process = subprocess.Popen(
                [
                    conda_cmd,
                    "env",
                    "create",
                    "-f",
                    str(env_file_path),
                    "-n",
                    self.env_name,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in process.stdout:
                if any(
                    kw in line
                    for kw in ["Solving", "Downloading", "Extracting"]
                ):
                    print(".", end="", flush=True)

            process.wait()
            print()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, conda_cmd
                )

            Console.success(f"Environment '{self.env_name}' created")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "Failed to create environment. Try:\n"
                f"  1. Update {conda_cmd}: {conda_cmd} update {conda_cmd}\n"
                f"  2. Clear cache: {conda_cmd} clean --all\n"
                f"  3. Check {env_file} for conflicts"
            ) from e

    def install_package(self) -> None:
        """Install spyglass package in development mode."""
        Console.progress("Installing spyglass package", 1)

        conda_cmd = self.get_command()
        try:
            subprocess.run(
                [
                    conda_cmd,
                    "run",
                    "-n",
                    self.env_name,
                    "pip",
                    "install",
                    "-e",
                    str(REPO_ROOT),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            Console.success("Spyglass installed")
        except subprocess.CalledProcessError as e:
            error_output = e.stderr if e.stderr else str(e)
            error_lower = error_output.lower()

            # Platform-specific disk space command
            disk_space_cmd = "dir" if sys.platform == "win32" else "df -h"

            if "no space left" in error_lower or "disk" in error_lower:
                primary_cause = "Disk space is full"
                primary_fix = (
                    f"Free up disk space: {disk_space_cmd} shows usage"
                )
            elif "connection" in error_lower or "timeout" in error_lower:
                primary_cause = "Network connection issue"
                primary_fix = "Check internet connection and retry"
            elif "permission" in error_lower or "access" in error_lower:
                primary_cause = "Permission denied"
                primary_fix = "Check write permissions in the install directory"
            else:
                primary_cause = "Package installation failed"
                primary_fix = (
                    f"Update pip: {conda_cmd} run -n {self.env_name} "
                    "pip install --upgrade pip"
                )

            raise RuntimeError(
                f"Failed to install spyglass package.\n\n"
                f"Most likely cause: {primary_cause}\n"
                f"Recommended fix: {primary_fix}\n\n"
                f"If that doesn't help, try these steps:\n"
                f"  1. Update pip: {conda_cmd} run -n {self.env_name} pip install --upgrade pip\n"
                f"  2. Check disk space: {disk_space_cmd}\n"
                f"  3. Check network connection\n"
                f"  4. Retry the installation\n\n"
                f"Error details: {error_output[:500]}"
            ) from e


def get_base_directory(cli_arg: Optional[str] = None) -> Path:
    """Get base directory for Spyglass data with write permission validation.

    Determines base directory using priority: CLI argument > environment
    variable > interactive prompt. Validates that directory can be created
    and written to before returning.

    Parameters
    ----------
    cli_arg : str, optional
        Base directory path from CLI argument. If provided, takes highest
        priority over environment variables and prompts.

    Returns
    -------
    pathlib.Path
        Validated base directory path that is writable

    Raises
    ------
    RuntimeError
        If directory cannot be created or is not writable due to permissions

    Examples
    --------
    >>> # From CLI argument
    >>> base_dir = get_base_directory("/data/spyglass")

    >>> # From environment or prompt
    >>> base_dir = get_base_directory()
    """

    def validate_and_test_write(path: Path) -> Path:
        """Validate directory and test write permissions.

        Parameters
        ----------
        path : pathlib.Path
            Directory path to validate

        Returns
        -------
        pathlib.Path
            Validated directory path

        Raises
        ------
        RuntimeError
            If directory cannot be created or written to
        """
        try:
            # Check if we can create the directory
            path.mkdir(parents=True, exist_ok=True)

            # Test write access
            test_file = path / ".spyglass_write_test"
            test_file.touch()
            test_file.unlink()

            return path

        except PermissionError:
            raise RuntimeError(
                f"Cannot write to base directory: {path}\n"
                f"  Check permissions or choose a different location"
            )
        except OSError as e:
            raise RuntimeError(
                f"Cannot create base directory: {path}\n" f"  Error: {e}"
            )

    # 1. CLI argument (highest priority)
    if cli_arg:
        base_path = Path(cli_arg).expanduser().resolve()
        validated_path = validate_and_test_write(base_path)
        Console.success(f"Using base directory from CLI: {validated_path}")
        return validated_path

    # 2. Environment variable (second priority)
    if base_env := os.getenv("SPYGLASS_BASE_DIR"):
        base_path = Path(base_env).expanduser().resolve()
        validated_path = validate_and_test_write(base_path)
        Console.success(
            f"Using base directory from environment: {validated_path}"
        )
        return validated_path

    # 3. Interactive prompt
    print("\nWhere should Spyglass store data?")
    print("  This will store raw NWB files, analysis results, and video data.")
    print("  Typical usage: 10-100+ GB depending on your experiments.")
    print()
    default = Path.home() / "spyglass_data"
    print(f"  Default: {default}")
    print(
        "  Tip: Set SPYGLASS_BASE_DIR environment variable to skip this prompt"
    )

    while True:
        response = input(f"\nData directory [{default}]: ").strip()

        if not response:
            try:
                validated_path = validate_and_test_write(default)
                Console.success(f"Base directory validated: {validated_path}")
                return validated_path
            except RuntimeError as e:
                Console.error(str(e))
                continue

        try:
            base_path = Path(response).expanduser().resolve()

            # Validate parent exists
            if not base_path.parent.exists():
                Console.error(
                    f"Parent directory does not exist: {base_path.parent}"
                )
                print(
                    "  Please create parent directory first or choose another location"
                )
                continue

            # Warn if directory already exists
            if base_path.exists():
                if not base_path.is_dir():
                    Console.error(
                        f"Path exists but is not a directory: {base_path}"
                    )
                    continue

                if not Console.prompt_yes_no(
                    "Directory exists. Use it?", default_yes=True
                ):
                    continue

            # Validate write permissions
            validated_path = validate_and_test_write(base_path)
            Console.success(f"Base directory validated: {validated_path}")
            return validated_path

        except RuntimeError as e:
            Console.error(str(e))
            continue
        except (ValueError, OSError) as e:
            Console.error(f"Invalid path: {e}")


def prompt_install_type() -> Tuple[str, str]:
    """Interactive prompt for installation type.

    Displays menu of installation options (minimal vs full) and prompts
    user to select one. Returns appropriate environment file and type.

    Parameters
    ----------
    None

    Returns
    -------
    env_file : str
        Path to environment YAML file ("environment_min.yml" or "environment.yml")
    install_type : str
        Installation type identifier ("minimal" or "full")

    Examples
    --------
    >>> env_file, install_type = prompt_install_type()
    >>> print(f"Using {env_file} for {install_type} installation")
    """
    print("\n" + "=" * 60)
    print("Installation Type")
    print("=" * 60)

    # Get disk space values from constants for consistency
    min_pkg = DISK_SPACE_REQUIREMENTS["minimal"]["packages"]
    min_total = DISK_SPACE_REQUIREMENTS["minimal"]["total"]
    full_pkg = DISK_SPACE_REQUIREMENTS["full"]["packages"]
    full_total = DISK_SPACE_REQUIREMENTS["full"]["total"]

    print("\n1. Minimal (Recommended for getting started)")
    print(f"   ├─ Install time: ~{ENV_CREATION_TIME_MINIMAL} minutes")
    print(
        f"   ├─ Disk space: ~{min_pkg} GB packages ({min_total} GB total with buffer)"
    )
    print("   ├─ Includes:")
    print("   │  • Core Spyglass functionality")
    print("   │  • Common data tables")
    print("   │  • Position tracking")
    print("   │  • LFP analysis")
    print("   │  • Basic spike sorting")
    print("   └─ Good for: Learning, basic workflows")

    print("\n2. Full (For advanced analysis)")
    print(f"   ├─ Install time: ~{ENV_CREATION_TIME_FULL} minutes")
    print(
        f"   ├─ Disk space: ~{full_pkg} GB packages ({full_total} GB total with buffer)"
    )
    print("   ├─ Includes: Everything in Minimal, plus:")
    print("   │  • Advanced spike sorting (Kilosort, etc.)")
    print("   │  • Ripple detection")
    print("   │  • Track linearization")
    print("   └─ Good for: Production work, all features")

    print("\nNote: DeepLabCut, Moseq, and some decoding features")
    print("      require separate installation (see docs)")

    # Map choices to (env_file, install_type)
    choice_map = {
        "1": ("environment_min.yml", "minimal"),
        "2": ("environment.yml", "full"),
    }

    while True:
        choice = input("\nChoice [1-2]: ").strip()

        if choice not in choice_map:
            Console.error("Please enter 1 or 2")
            continue

        env_file, install_type = choice_map[choice]
        Console.success(f"Selected: {install_type.capitalize()} installation")
        return env_file, install_type


# =============================================================================
# DockerManager Class - Docker and Docker Compose Operations
# =============================================================================


class DockerManager:
    """Manages Docker and Docker Compose operations for database setup.

    All methods are static since they operate on system state, not instance state.
    These are self-contained because spyglass isn't installed yet when they run.
    """

    @staticmethod
    def is_available() -> bool:
        """Check if Docker is available and daemon is running."""
        if not shutil.which("docker"):
            return False
        try:
            result = subprocess.run(
                ["docker", "info"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def is_compose_available() -> bool:
        """Check if Docker Compose is available."""
        try:
            result = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    @staticmethod
    def get_compose_command() -> list[str]:
        """Get the Docker Compose command prefix."""
        return ["docker", "compose"]

    @staticmethod
    def generate_env_file(
        mysql_port: int = 3306,
        mysql_password: str = "tutorial",
        mysql_image: str = "datajoint/mysql:8.0",
        env_path: Optional[str] = None,
    ) -> None:
        """Generate .env file for Docker Compose.

        Only writes non-default values.

        Parameters
        ----------
        mysql_port : int
            MySQL port (default: 3306)
        mysql_password : str
            MySQL root password (default: "tutorial")
        mysql_image : str
            Docker image to use (default: "datajoint/mysql:8.0")
        env_path : str, optional
            Path to write .env file. If None, writes to REPO_ROOT/.env.
        """
        env_lines = ["# Spyglass Docker Compose Configuration", ""]

        if mysql_password != "tutorial":
            env_lines.append(f"MYSQL_ROOT_PASSWORD={mysql_password}")
        if mysql_port != 3306:
            env_lines.append(f"MYSQL_PORT={mysql_port}")
        if mysql_image != "datajoint/mysql:8.0":
            env_lines.append(f"MYSQL_IMAGE={mysql_image}")

        # If all defaults, don't create file
        if len(env_lines) == 2:
            return

        target_path = Path(env_path) if env_path else REPO_ROOT / ".env"
        with target_path.open("w") as f:
            f.write("\n".join(env_lines) + "\n")

    @staticmethod
    def validate_env_file(env_path: Optional[str] = None) -> bool:
        """Validate .env file is readable (missing is OK).

        Parameters
        ----------
        env_path : str, optional
            Path to .env file. If None, checks REPO_ROOT/.env.
        """
        target_path = Path(env_path) if env_path else REPO_ROOT / ".env"
        if not target_path.exists():
            return True
        try:
            with target_path.open("r") as f:
                f.read()
            return True
        except (OSError, PermissionError):
            return False

    @staticmethod
    def read_env_file(env_path: Optional[str] = None) -> Tuple[int, str]:
        """Read MySQL settings from .env file (missing uses defaults)."""
        target_path = Path(env_path) if env_path else REPO_ROOT / ".env"
        port = DEFAULT_MYSQL_PORT
        password = DEFAULT_MYSQL_PASSWORD
        if not target_path.exists():
            return port, password
        try:
            with target_path.open("r") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    key, _, value = line.partition("=")
                    value = value.strip()
                    if key == "MYSQL_PORT":
                        if not value:
                            raise ValueError("MYSQL_PORT is empty")
                        port = int(value)
                        if port < 1 or port > 65535:
                            raise ValueError(
                                f"MYSQL_PORT {port} is out of range"
                            )
                    elif key == "MYSQL_ROOT_PASSWORD":
                        if not value:
                            raise ValueError("MYSQL_ROOT_PASSWORD is empty")
                        password = value
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid .env file: {e}") from e
        return port, password

    @staticmethod
    def cleanup() -> None:
        """Clean up after failed Docker Compose setup (best-effort)."""
        try:
            compose_cmd = DockerManager.get_compose_command()
            subprocess.run(
                compose_cmd + ["down", "-v"],
                capture_output=True,
                timeout=30,
                cwd=REPO_ROOT,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass


# ============================================================================
# JSON Schema Loading Functions
# ============================================================================
# These functions read from directory_schema.json in src/spyglass/ to ensure
# the installer and settings.py use the same directory structure (single
# source of truth). This avoids code duplication and ensures consistency.


def validate_schema(schema: Dict[str, Any]) -> None:
    """Validate config schema has required structure.

    Validates that the schema has a directory_schema key with at least one
    prefix, and each prefix has at least one subdirectory. The JSON file
    is the single source of truth for directory structure.

    Raises
    ------
    ValueError
        If schema is missing directory_schema or has empty prefixes
    """
    if "directory_schema" not in schema:
        raise ValueError("Schema missing 'directory_schema' key")

    dir_schema = schema["directory_schema"]
    if not dir_schema:
        raise ValueError("Schema 'directory_schema' is empty")

    # Validate each prefix has at least one subdirectory defined
    for prefix, subdirs in dir_schema.items():
        if not isinstance(subdirs, dict) or not subdirs:
            raise ValueError(
                f"Schema prefix '{prefix}' must have at least one subdirectory"
            )


def load_full_schema() -> Dict[str, Any]:
    """Load complete schema including TLS config.

    Returns
    -------
    Dict[str, Any]
        Complete schema with directory_schema and tls sections

    Raises
    ------
    FileNotFoundError
        If directory_schema.json not found
    ValueError
        If schema is invalid
    """
    import json

    schema_path = (
        Path(__file__).parent.parent / "src/spyglass/directory_schema.json"
    )

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Config schema not found: {schema_path}\n"
            f"This file should exist in src/spyglass/ directory."
        )

    try:
        with open(schema_path) as f:
            schema = json.load(f)
    except (OSError, IOError) as e:
        raise ValueError(f"Cannot read {schema_path}: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {schema_path}: {e}")

    if not isinstance(schema, dict):
        raise ValueError(f"Schema should be a dict, got {type(schema)}")

    # Check schema version for compatibility
    schema_version = schema.get("_schema_version")
    if schema_version and schema_version != CURRENT_SCHEMA_VERSION:
        Console.warning(
            f"Schema version mismatch: expected {CURRENT_SCHEMA_VERSION}, "
            f"got {schema_version}. This may cause compatibility issues."
        )

    # Validate schema
    validate_schema(schema)

    return schema


def load_directory_schema() -> Dict[str, Dict[str, str]]:
    """Load directory schema from JSON file (single source of truth).

    Returns
    -------
    Dict[str, Dict[str, str]]
        Directory schema with prefixes (spyglass, kachery, dlc, moseq)

    Raises
    ------
    FileNotFoundError
        If directory_schema.json not found in src/spyglass/
    ValueError
        If schema is invalid or missing required keys
    """
    full_schema = load_full_schema()
    return full_schema["directory_schema"]


def build_directory_structure(
    base_dir: Path,
    schema: Optional[Dict[str, Dict[str, str]]] = None,
    create: bool = True,
    verbose: bool = True,
) -> Dict[str, Path]:
    """Build Spyglass directory structure from base directory.

    Parameters
    ----------
    base_dir : Path
        Base directory for Spyglass data
    schema : Dict[str, Dict[str, str]], optional
        Pre-loaded directory schema. If None, will load from file.
    create : bool, optional
        Whether to create directories if they don't exist, by default True
    verbose : bool, optional
        Whether to print progress feedback, by default True

    Returns
    -------
    Dict[str, Path]
        Mapping of directory names to full paths
    """
    if schema is None:
        schema = load_directory_schema()

    # Validate schema loaded successfully
    if not schema:
        raise ValueError(
            "Directory schema could not be loaded. "
            "Check directory_schema.json exists in src/spyglass/."
        )

    directories = {}

    if verbose and create:
        print(f"Creating Spyglass directory structure in {base_dir}")
        print("  Creating:")

    for prefix, dir_map in schema.items():
        for key, rel_path in dir_map.items():
            full_path = base_dir / rel_path
            directories[f"{prefix}_{key}"] = full_path

            if create:
                full_path.mkdir(parents=True, exist_ok=True)
                if verbose:
                    print(f"    • {rel_path}")

    if verbose and create:
        print(f"  ✓ Created {len(directories)} directories")

    return directories


def determine_tls(host: str, schema: Optional[Dict[str, Any]] = None) -> bool:
    """Automatically determine if TLS should be used.

    Uses smart defaults - no user prompt needed.

    Parameters
    ----------
    host : str
        Database hostname
    schema : Dict[str, Any], optional
        Pre-loaded schema. If None, will load from file.

    Returns
    -------
    bool
        Whether to use TLS
    """
    if schema is None:
        schema = load_full_schema()

    tls_config = schema.get("tls", {})
    localhost_addresses = tls_config.get(
        "localhost_addresses", ["localhost", "127.0.0.1", "::1"]
    )

    # Automatic decision: enable for remote, disable for local
    is_local = host in localhost_addresses
    use_tls = not is_local

    # User-friendly messaging (plain language instead of technical terms)
    if is_local:
        print(
            f"{COLORS['blue']}✓ Connecting to local database at {host}{COLORS['reset']}"
        )
        print("  Security: Using unencrypted connection (safe for localhost)")
    else:
        print(
            f"{COLORS['blue']}✓ Connecting to remote database at {host}{COLORS['reset']}"
        )
        print(
            "  Security: Using encrypted connection (TLS) to protect your data"
        )
        print("  This is required when connecting over a network")

    return use_tls


def create_database_config(
    host: str = "localhost",
    port: int = 3306,
    user: str = "root",
    password: str = "tutorial",
    use_tls: Optional[bool] = None,
    base_dir: Optional[Path] = None,
) -> None:
    """Create complete Spyglass configuration with database and directories.

    This creates a complete DataJoint + Spyglass configuration including:
    - Database connection settings
    - DataJoint external stores
    - Spyglass directory structure (from directory_schema.json)

    Parameters
    ----------
    host : str, optional
        Database host (default: "localhost")
    port : int, optional
        Database port (default: 3306)
    user : str, optional
        Database user (default: "root")
    password : str, optional
        Database password (default: "tutorial")
    use_tls : bool or None, optional
        Whether to use TLS/SSL. If None, automatically determined based on host.
    base_dir : Path or None, optional
        Base directory for Spyglass data. If None, will prompt user.

    Notes
    -----
    Uses JSON for safety (no code injection vulnerability).
    Reads directory structure from directory_schema.json (DRY principle).
    """
    # Get base directory if not provided
    if base_dir is None:
        base_dir = get_base_directory()

    # Load schema once for efficiency (used by both TLS and directory creation)
    full_schema = load_full_schema()
    dir_schema = full_schema["directory_schema"]

    # Auto-determine TLS if not explicitly provided
    if use_tls is None:
        use_tls = determine_tls(host, schema=full_schema)

    # Build directory structure from JSON schema
    dirs = build_directory_structure(
        base_dir, schema=dir_schema, create=True, verbose=True
    )

    # Create complete configuration
    config = {
        # Database connection settings
        "database.host": host,
        "database.port": port,
        "database.user": user,
        "database.password": password,
        "database.use_tls": use_tls,
        "database.reconnect": True,
        "connection.init_function": None,
        "connection.charset": "",
        # DataJoint general settings
        "loglevel": "INFO",
        "safemode": True,
        "fetch_format": "array",
        "display.limit": 12,
        "display.width": 14,
        "display.show_tuple_count": True,
        "add_hidden_timestamp": False,
        # DataJoint performance settings
        "filepath_checksum_size_limit": 1 * 1024**3,  # 1 GB
        "enable_python_native_blobs": True,
        # DataJoint stores for external file storage
        "stores": {
            "raw": {
                "protocol": "file",
                "location": str(dirs["spyglass_raw"]),
                "stage": str(dirs["spyglass_raw"]),
            },
            "analysis": {
                "protocol": "file",
                "location": str(dirs["spyglass_analysis"]),
                "stage": str(dirs["spyglass_analysis"]),
            },
        },
        # Spyglass custom configuration
        "custom": {
            "debug_mode": "false",
            "test_mode": "false",
            "kachery_zone": "franklab.default",
            "spyglass_dirs": {
                "base": str(base_dir),
                "raw": str(dirs["spyglass_raw"]),
                "analysis": str(dirs["spyglass_analysis"]),
                "recording": str(dirs["spyglass_recording"]),
                "sorting": str(dirs["spyglass_sorting"]),
                "waveforms": str(dirs["spyglass_waveforms"]),
                "temp": str(dirs["spyglass_temp"]),
                "video": str(dirs["spyglass_video"]),
                "export": str(dirs["spyglass_export"]),
            },
            "kachery_dirs": {
                "cloud": str(dirs["kachery_cloud"]),
                "storage": str(dirs["kachery_storage"]),
                "temp": str(dirs["kachery_temp"]),
            },
            "dlc_dirs": {
                "base": str(base_dir / "deeplabcut"),
                "project": str(dirs["dlc_project"]),
                "video": str(dirs["dlc_video"]),
                "output": str(dirs["dlc_output"]),
            },
            "moseq_dirs": {
                "base": str(base_dir / "moseq"),
                "project": str(dirs["moseq_project"]),
                "video": str(dirs["moseq_video"]),
            },
        },
    }

    config_file = Path.home() / ".datajoint_config.json"

    # Handle existing config file with better UX
    if config_file.exists():
        Console.warning(f"Configuration file already exists: {config_file}")
        print("\nExisting database settings:")
        try:
            with config_file.open() as f:
                existing = json.load(f)
                existing_host = existing.get("database.host", "unknown")
                existing_port = existing.get("database.port", "unknown")
                existing_user = existing.get("database.user", "unknown")
                print(f"  Database: {existing_host}:{existing_port}")
                print(f"  User: {existing_user}")
        except (OSError, IOError, json.JSONDecodeError, KeyError) as e:
            print(f"  (Unable to read existing config: {e})")

        print("\nOptions:")
        print(
            "  [b] Backup and create new (saves to .datajoint_config.json.backup)"
        )
        print("  [o] Overwrite with new settings")
        print("  [k] Keep existing (cancel installation)")

        choice = input("\nChoice [B/o/k]: ").strip().lower() or "b"

        if choice in ["k", "keep"]:
            Console.warning(
                "Keeping existing configuration. Installation cancelled."
            )
            print("\nTo install with different settings:")
            print(
                "  1. Backup your config: cp ~/.datajoint_config.json ~/.datajoint_config.json.backup"
            )
            print("  2. Run installer again")
            return
        elif choice in ["b", "backup"]:
            backup_file = config_file.with_suffix(".json.backup")
            shutil.copy2(config_file, backup_file)
            Console.success(f"Backed up existing config to {backup_file}")
        elif choice not in ["o", "overwrite"]:
            Console.error("Invalid choice")
            return

    # Save configuration with atomic write and secure permissions
    import tempfile

    # Security warning before saving
    Console.warning(
        "Database password will be stored in plain text in config file.\n"
        "  For production environments:\n"
        "    1. Use environment variable SPYGLASS_DB_PASSWORD\n"
        "    2. File permissions will be restricted automatically\n"
        "    3. Consider database roles with limited privileges"
    )

    # Atomic write: write to temp file, then move
    config_dir = config_file.parent
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=config_dir,
        delete=False,
        prefix=".datajoint_config.tmp",
        suffix=".json",
    ) as tmp_file:
        json.dump(config, tmp_file, indent=2)
        tmp_path = Path(tmp_file.name)

    # Set restrictive permissions (Unix/Linux/macOS only)
    try:
        tmp_path.chmod(0o600)  # Owner read/write only
    except (AttributeError, OSError):
        # Windows doesn't support chmod - permissions handled differently
        pass

    # Atomic move (on same filesystem)
    shutil.move(str(tmp_path), str(config_file))
    Console.success(f"Configuration saved to: {config_file}")
    print("  Permissions: Owner read/write only (secure)")

    # Enhanced success message with next steps
    print()
    Console.success("✓ Spyglass configuration complete!")
    print()
    print("Database connection:")
    print(f"  • Server: {host}:{port}")
    print(f"  • User: {user}")
    tls_status = "Yes" if use_tls else "No (localhost)"
    print(f"  • Encrypted: {tls_status}")
    print()
    print("Data directories:")
    print(f"  • Base: {base_dir}")
    print(f"  • Raw data: {config['custom']['spyglass_dirs']['raw']}")
    print(f"  • Analysis: {config['custom']['spyglass_dirs']['analysis']}")
    print(f"  • ({len(dirs)} directories total)")
    print()
    print("Next steps:")
    print("  1. Activate environment: conda activate spyglass")
    print("  2. Test your installation: python scripts/validate.py")
    print("  3. Start using Spyglass: python -c 'import spyglass'")
    print()
    print("Need help? See: https://lorenfranklab.github.io/spyglass/")


def validate_hostname(hostname: str) -> bool:
    """Validate hostname format to prevent common typos.

    Performs basic validation to catch obvious errors like spaces,
    control characters, multiple consecutive dots, or invalid length.

    Parameters
    ----------
    hostname : str
        Hostname or IP address to validate

    Returns
    -------
    bool
        True if hostname appears valid, False otherwise

    Examples
    --------
    >>> validate_hostname("localhost")
    True
    >>> validate_hostname("db.example.com")
    True
    >>> validate_hostname("host with spaces")
    False
    >>> validate_hostname("..invalid")
    False

    Notes
    -----
    This is intentionally permissive - only catches obvious typos.
    DNS resolution will be the final validation.
    """
    if not hostname:
        return False

    # Reject hostnames with whitespace or control characters
    if any(c.isspace() or ord(c) < 32 for c in hostname):
        return False

    # Reject obvious typos (multiple dots, leading/trailing dots)
    if hostname.startswith(".") or hostname.endswith(".") or ".." in hostname:
        return False

    # Check length (DNS hostname max is 253 characters)
    if len(hostname) > 253:
        return False

    return True


def validate_port(port: int) -> Tuple[bool, str]:
    """Validate port number is in valid range.

    Parameters
    ----------
    port : int
        Port number to validate

    Returns
    -------
    valid : bool
        True if port is valid, False otherwise
    message : str
        Error or warning message

    Notes
    -----
    Valid ports are 1024-65535 (non-privileged).
    Ports 1-1023 require root access on Unix systems.
    """
    if not isinstance(port, int):
        return False, f"Port must be an integer, got {type(port).__name__}"

    if port < 1 or port > 65535:
        return (
            False,
            f"Port {port} is out of valid range (1-65535). MySQL typically uses 3306.",
        )

    if port < 1024:
        return (
            False,
            f"Port {port} is privileged (requires root). Use port 1024-65535.",
        )

    return True, ""


def validate_database_config(
    host: str,
    port: int,
    user: str,
    password: str,
) -> Tuple[bool, list[str]]:
    """Validate database configuration parameters.

    Performs comprehensive validation of database connection parameters
    before attempting to connect or save configuration.

    Parameters
    ----------
    host : str
        Database hostname or IP address
    port : int
        Database port number
    user : str
        Database username
    password : str
        Database password

    Returns
    -------
    valid : bool
        True if all parameters are valid
    errors : list of str
        List of validation error messages (empty if valid)

    Examples
    --------
    >>> valid, errors = validate_database_config("localhost", 3306, "root", "pass")
    >>> if not valid:
    ...     for err in errors:
    ...         print(f"Error: {err}")
    """
    errors = []

    # Validate hostname
    if not host:
        errors.append("Hostname cannot be empty")
    elif not validate_hostname(host):
        errors.append(
            f"Invalid hostname '{host}': cannot contain spaces, "
            "control characters, or consecutive dots"
        )

    # Validate port
    port_valid, port_msg = validate_port(port)
    if not port_valid:
        errors.append(port_msg)

    # Validate username
    if not user:
        errors.append("Username cannot be empty")
    elif len(user) > 32:
        errors.append(f"Username too long ({len(user)} chars, max 32)")
    elif not re.match(r"^[a-zA-Z0-9_@.-]+$", user):
        # MySQL usernames allow alphanumeric, underscore, @, dot, hyphen
        errors.append(
            f"Username '{user}' contains invalid characters.\n"
            "  Valid: letters, numbers, underscore, @, dot, hyphen"
        )

    # Validate password (basic checks only - actual auth is server-side)
    # Defense in depth: warn about shell metacharacters that could cause issues
    if not password:
        errors.append("Password cannot be empty")
    else:
        # Characters that could cause shell escaping issues if mishandled
        shell_metacharacters = ["`", "$", "\\", '"', "'", ";", "&", "|"]
        found_dangerous = [c for c in shell_metacharacters if c in password]
        if found_dangerous:
            # Warning, not error - password might legitimately contain these
            Console.warning(
                f"Password contains special characters: {found_dangerous}\n"
                "  This should work, but if you encounter issues, consider\n"
                "  using a password without: ` $ \\ \" ' ; & |"
            )

    return len(errors) == 0, errors


def is_port_available(host: str, port: int) -> Tuple[bool, str]:
    """Check if port is available or reachable.

    For localhost: Checks if port is free (available for binding)
    For remote hosts: Checks if port is reachable (something listening)

    Parameters
    ----------
    host : str
        Hostname or IP address to check
    port : int
        Port number to check

    Returns
    -------
    available : bool
        True if port is available/reachable, False if blocked/in-use
    message : str
        Description of port status

    Examples
    --------
    >>> available, msg = is_port_available("localhost", 3306)
    >>> if not available:
    ...     print(f"Port issue: {msg}")

    Notes
    -----
    The interpretation differs for localhost vs remote:
    - localhost: False = port in use (good for remote, bad for Docker)
    - remote: False = port unreachable (bad - firewall/wrong port)
    """
    import socket

    try:
        addrinfos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False, f"Cannot resolve hostname: {host}"

    is_local = host in LOCALHOST_ADDRESSES
    any_success = False
    last_error: Optional[Exception] = None

    for family, socktype, proto, _canonname, sockaddr in addrinfos:
        try:
            with socket.socket(family, socktype, proto) as sock:
                sock.settimeout(1)  # 1 second timeout
                if sock.connect_ex(sockaddr) == 0:
                    any_success = True
                    break
        except socket.error as e:
            last_error = e

    # For localhost, we want the port to be FREE (not in use)
    # For remote, we want the port to be IN USE (something listening)
    if is_local:
        if any_success:
            return False, f"Port {port} is already in use on {host}"
        return True, f"Port {port} is available on {host}"

    if any_success:
        return True, f"Port {port} is reachable on {host}"
    if last_error:
        return False, f"Cannot reach {host}:{port} ({last_error})"
    return False, f"Cannot reach {host}:{port} (firewall/wrong port?)"


def prompt_remote_database_config() -> Optional[Dict[str, Any]]:
    """Prompt user for remote database connection details.

    Interactively asks for host, port, user, and password. Uses getpass
    for secure password input. Automatically enables TLS for remote hosts.
    Validates hostname format to prevent typos.

    Parameters
    ----------
    None

    Returns
    -------
    dict or None
        Dictionary with keys: 'host', 'port', 'user', 'password', 'use_tls'
        Returns None if user cancels (Ctrl+C)

    Examples
    --------
    >>> config = prompt_remote_database_config()
    >>> if config:
    ...     print(f"Connecting to {config['host']}:{config['port']}")
    """
    print("\nRemote database configuration:")
    print("  Your lab admin should have provided these credentials.")
    print("  Check your welcome email or contact your admin if unsure.")
    print("  (Press Ctrl+C to cancel)")

    try:
        host = input("  Host (e.g., db.lab.edu): ").strip()

        # Require explicit host for remote database
        if not host:
            Console.error("Host is required for remote database connection")
            print("  Ask your lab admin for the database hostname")
            return None

        # Validate hostname format
        if not validate_hostname(host):
            Console.error(f"Invalid hostname: {host}")
            print("  Hostname cannot contain spaces or invalid characters")
            print("  Valid examples: localhost, db.example.com, 192.168.1.100")
            return None
        port_str = input("  Port [3306]: ").strip() or "3306"
        user = input("  User [root]: ").strip() or "root"

        # Use getpass for password to hide input
        import getpass

        password = getpass.getpass("  Password: ")

        # Parse and validate port
        try:
            port = int(port_str)
        except ValueError:
            Console.error(f"Invalid port: '{port_str}' is not a number")
            return None

        port_valid, port_msg = validate_port(port)
        if not port_valid:
            Console.error(port_msg)
            return None

        # Check if port is reachable
        print(f"  Testing connection to {host}:{port}...")
        port_reachable, port_msg = is_port_available(host, port)

        if host not in LOCALHOST_ADDRESSES and not port_reachable:
            # Remote host, port not reachable
            Console.warning(port_msg)
            print("\n  Possible causes:")
            print("    • Wrong port number (MySQL usually uses 3306)")
            print("    • Firewall blocking connections")
            print("    • Database server not running")
            print("    • Wrong hostname")
            print("\n  Common MySQL ports:")
            print("    • Standard MySQL: 3306")
            print("    • SSH tunnel: Check your tunnel configuration")

            if not Console.prompt_yes_no(
                "\n  Continue anyway?", default_yes=False
            ):
                return None
        elif port_reachable:
            print("  ✓ Port is reachable")

        # Determine TLS based on host (use TLS for non-localhost)
        # This is automatic - no prompt needed. TLS is required for security
        # on remote connections.
        use_tls = host not in LOCALHOST_ADDRESSES

        return {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "use_tls": use_tls,
        }

    except KeyboardInterrupt:
        print("\n")
        Console.warning("Database configuration cancelled")
        return None


def get_database_options() -> Tuple[list[DatabaseOption], bool]:
    """Get available database options based on system capabilities.

    Checks Docker Compose availability and returns menu options.

    Parameters
    ----------
    None

    Returns
    -------
    options : list of DatabaseOption
        List of database option objects for menu display
    compose_available : bool
        True if Docker Compose is available

    Examples
    --------
    >>> options, compose_avail = get_database_options()
    >>> for opt in options:
    ...     print(f"{opt.number}. {opt.name} - {opt.status}")
    """
    options = []

    # Check Docker Compose availability
    compose_available = DockerManager.is_compose_available()

    # Option 1: Remote database (primary use case - joining existing lab)
    # Use clearer terminology: "if you have credentials" instead of "lab members"
    options.append(
        DatabaseOption(
            number="1",
            name="Remote",
            status=f"{SYMBOLS['success']} Recommended if you have database credentials",
            description="Connect to existing database (lab server, cloud, etc.)",
        )
    )

    # Option 2: Docker (trial/development use case)
    if compose_available:
        options.append(
            DatabaseOption(
                number="2",
                name="Docker",
                status=f"{SYMBOLS['success']} Available",
                description="Local trial database (for testing/tutorials)",
            )
        )
    else:
        options.append(
            DatabaseOption(
                number="2",
                name="Docker",
                status=f"{SYMBOLS['error']} Not available",
                description="Requires Docker Desktop",
            )
        )

    # Option 3: Skip setup - add hint about recovery
    options.append(
        DatabaseOption(
            number="3",
            name="Skip",
            status=f"{SYMBOLS['success']} Available",
            description="Configure later (see docs/DATABASE.md or run installer again)",
        )
    )

    return options, compose_available


def prompt_database_setup() -> str:
    """Ask user about database setup preference.

    Displays menu of database setup options with availability status
    and prompts user to choose.

    Parameters
    ----------
    None

    Returns
    -------
    str
        One of: 'compose' (Docker Compose), 'remote' (existing database),
        or 'skip' (configure later)

    Examples
    --------
    >>> choice = prompt_database_setup()
    >>> if choice == "compose":
    ...     setup_database_compose()
    """
    print("\n" + "=" * 60)
    print("Database Setup")
    print("=" * 60)

    options, compose_available = get_database_options()

    print("\nOptions:")
    for opt in options:
        # Color status based on availability (check for success/error symbols)
        if SYMBOLS["error"] in opt.status:
            status_color = COLORS["red"]
        elif SYMBOLS["success"] in opt.status:
            status_color = COLORS["green"]
        else:
            status_color = COLORS["reset"]
        print(
            f"  {opt.number}. {opt.name:20} {status_color}{opt.status}{COLORS['reset']}"
        )
        print(f"      {opt.description}")

    # If Docker not available, guide user
    if not compose_available:
        print(
            f"\n{COLORS['yellow']}{SYMBOLS['warning']}{COLORS['reset']} Docker is not available"
        )
        print("  To enable Docker setup:")
        print(
            "    1. Install Docker Desktop: https://docs.docker.com/get-docker/"
        )
        print("    2. Start Docker Desktop")
        print("    3. Verify: docker compose version")
        print("    4. Re-run installer")

    # Map choices to actions (updated order: Remote first, then Docker)
    choice_map = {
        "1": "remote",
        "2": "compose",
        "3": "skip",
    }

    # Get valid choices
    valid_choices = ["1", "3"]  # Remote and Skip always available
    if compose_available:
        valid_choices.insert(1, "2")  # Insert Docker as option 2 if available

    while True:
        choice = input(f"\nChoice [{'/'.join(valid_choices)}]: ").strip()

        if choice not in choice_map:
            Console.error(f"Please enter {' or '.join(valid_choices)}")
            continue

        # Handle Docker unavailability
        if choice == "2" and not compose_available:
            Console.error("Docker is not available")
            continue

        return choice_map[choice]


def setup_database_compose() -> Tuple[bool, str]:
    """Set up database using Docker Compose.

    Checks Docker Compose availability, generates .env if needed,
    starts services, waits for readiness, and creates configuration file.

    Returns
    -------
    success : bool
        True if database setup succeeded, False otherwise
    reason : str
        Reason for failure or "success"

    Notes
    -----
    This function cannot import from spyglass because spyglass hasn't been
    installed yet. All operations must be inline.

    Uses docker-compose.yml in repository root for configuration.
    Creates .env file only if non-default values are needed.

    Examples
    --------
    >>> success, reason = setup_database_compose()
    >>> if success:
    ...     print("Database ready")
    """
    import time

    Console.step("Checking Docker Compose")
    if not DockerManager.is_compose_available():
        Console.fail()
        return False, "compose_unavailable"
    Console.done()

    try:
        # Generate .env file (only if customizations needed)
        # For now, use all defaults - no .env file needed
        # Future: could prompt for port/password customization
        DockerManager.generate_env_file()

        # Validate .env if it exists
        if not DockerManager.validate_env_file():
            return False, "env_file_invalid"

        env_path = REPO_ROOT / ".env"
        try:
            actual_port, actual_password = DockerManager.read_env_file(
                str(env_path)
            )
        except ValueError as e:
            Console.error(str(e))
            return False, "env_file_invalid"

        # Check if port is available
        port_available, port_msg = is_port_available("localhost", actual_port)
        if not port_available:
            Console.error(port_msg)
            print(f"\n  Port {actual_port} is already in use. Solutions:")

            # Platform-specific guidance
            if sys.platform == "darwin":  # macOS
                print("    1. Stop existing MySQL (if installed):")
                print("       brew services stop mysql")
                print(
                    "       # or: sudo launchctl unload -w /Library/LaunchDaemons/com.mysql.mysql.plist"
                )
                print("    2. Find what's using the port:")
                print(f"       lsof -i :{actual_port}")
            elif sys.platform.startswith("linux"):  # Linux
                print("    1. Stop existing MySQL service:")
                print("       sudo systemctl stop mysql")
                print("       # or: sudo service mysql stop")
                print("    2. Find what's using the port:")
                print(f"       sudo lsof -i :{actual_port}")
                print(f"       # or: sudo netstat -tulpn | grep {actual_port}")
            elif sys.platform == "win32":  # Windows
                print("    1. Stop existing MySQL service:")
                print("       net stop MySQL")
                print("       # or use Services app (services.msc)")
                print("    2. Find what's using the port:")
                print(f"       netstat -ano | findstr :{actual_port}")

            print("    Alternative: Use a different port:")
            if env_path.exists():
                print(f"       Edit {env_path} and set MYSQL_PORT=3307")
            else:
                print("       Create .env file with: MYSQL_PORT=3307")
            print("       (and update DataJoint config to match)")
            return False, "port_in_use"

        # Show what will happen
        print("\n" + "=" * 60)
        print("Docker Database Setup")
        print("=" * 60)
        print("\nThis will:")
        print("  • Download MySQL 8.0 Docker image (~500 MB)")
        print("  • Create a container named 'spyglass-db'")
        print(f"  • Start MySQL on localhost:{actual_port}")
        print("  • Save credentials to ~/.datajoint_config.json")
        print("\nEstimated time: 2-3 minutes")
        print("=" * 60)

        # Get compose command
        compose_cmd = DockerManager.get_compose_command()

        # Pull images first (better UX - shows progress)
        Console.progress("Pulling Docker images", 2)
        result = subprocess.run(
            compose_cmd + ["pull"],
            capture_output=True,
            timeout=300,  # 5 minutes for image pull
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            error_output = result.stderr.decode()
            error_lower = error_output.lower()
            Console.error(f"Failed to pull Docker images: {error_output}")

            # Prioritize causes by likelihood (network issues most common)
            if "no space" in error_lower or "disk" in error_lower:
                print("\n  Most likely cause: Insufficient disk space")
                print("  Fix: Free up space with: docker system prune -a")
            elif "timeout" in error_lower or "connection" in error_lower:
                print("\n  Most likely cause: Network connection issue")
                print("  Fix: Check internet connection and retry")
            else:
                print("\n  Most likely cause: Network or Docker Hub issue")
                print("  Fix: Wait a moment and retry")

            print("\n  Other steps to try:")
            print("    1. Check internet connection")
            print("    2. Check disk space: docker system df")
            print("    3. Retry: docker compose pull")
            print("    4. If persistent, try: docker system prune")
            return False, "pull_failed"

        # Start services
        Console.step("Starting services")
        result = subprocess.run(
            compose_cmd + ["up", "-d"],
            capture_output=True,
            timeout=60,
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            Console.fail()
            error_msg = result.stderr.decode()
            Console.error(f"Failed to start services: {error_msg}")
            DockerManager.cleanup()
            return False, "start_failed"
        Console.done()

        # Wait for MySQL readiness using health check
        Console.step("Waiting for MySQL to be ready")

        for attempt in range(MYSQL_HEALTH_CHECK_ATTEMPTS):
            try:
                # Check if service is healthy
                result = subprocess.run(
                    compose_cmd + ["ps", "--format", "json"],
                    capture_output=True,
                    timeout=5,
                    cwd=REPO_ROOT,
                )

                if result.returncode == 0:
                    # Parse JSON output to check health
                    try:
                        services = json.loads(result.stdout.decode())
                        # Handle both single dict and list of dicts
                        if isinstance(services, dict):
                            services = [services]

                        mysql_service = next(
                            (
                                s
                                for s in services
                                if "mysql" in s.get("Service", "")
                            ),
                            None,
                        )

                        if mysql_service and "healthy" in mysql_service.get(
                            "Health", ""
                        ):
                            Console.done()
                            break
                    except json.JSONDecodeError:
                        # JSON parse failed, retry on next attempt
                        pass

            except subprocess.TimeoutExpired:
                # Command timed out, retry on next attempt
                pass

            if attempt < MYSQL_HEALTH_CHECK_ATTEMPTS - 1:
                print(".", end="", flush=True)
                time.sleep(MYSQL_HEALTH_CHECK_INTERVAL)
        else:
            # Timeout - provide debug info
            Console.fail()
            Console.error(
                f"MySQL did not become ready within {MYSQL_HEALTH_CHECK_TIMEOUT} seconds"
            )
            print("\n  Check logs:")
            print("    docker compose logs mysql")
            DockerManager.cleanup()
            return False, "timeout"

        # Create configuration file matching .env values
        create_database_config(
            host="localhost",
            port=actual_port,
            user="root",
            password=actual_password,
            use_tls=False,
        )

        # Warn if .env exists with custom values
        if env_path.exists():
            Console.warning(
                "Using custom settings from .env file. "
                "DataJoint config updated to match."
            )

        return True, "success"

    except subprocess.CalledProcessError as e:
        Console.error(f"Docker Compose command failed: {e}")
        DockerManager.cleanup()
        return False, str(e)
    except subprocess.TimeoutExpired:
        Console.error("Docker Compose command timed out")
        DockerManager.cleanup()
        return False, "timeout"
    except (OSError, IOError) as e:
        Console.error(f"File system error during Docker setup: {e}")
        DockerManager.cleanup()
        return False, str(e)
    except json.JSONDecodeError as e:
        Console.error(f"Failed to parse Docker Compose output: {e}")
        DockerManager.cleanup()
        return False, "json_parse_error"


def test_database_connection(
    host: str,
    port: int,
    user: str,
    password: str,
    use_tls: bool,
    timeout: int = 10,
) -> Tuple[bool, Optional[str]]:
    """Test database connection before saving configuration.

    Attempts to connect to MySQL database and execute a simple query
    to verify connectivity. Handles graceful fallback if pymysql not
    yet installed.

    Parameters
    ----------
    host : str
        Database hostname or IP address
    port : int
        Database port number (typically 3306)
    user : str
        Database username for authentication
    password : str
        Database password for authentication
    use_tls : bool
        Whether to enable TLS/SSL encryption
    timeout : int, optional
        Connection timeout in seconds (default: 10)

    Returns
    -------
    success : bool
        True if connection succeeded, False otherwise
    error_message : str or None
        Error message if connection failed, None if successful
    """
    try:
        import pymysql
    except ImportError:
        # pymysql not available yet (before pip install)
        Console.warning("Cannot test connection (pymysql not available)")
        print("  Connection will be tested during validation")
        return True, None  # Allow to proceed

    try:
        Console.step("Testing database connection")

        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            connect_timeout=timeout,
            ssl={"ssl": True} if use_tls else None,
        )

        # Test basic operation
        with connection.cursor() as cursor:
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()

        connection.close()
        Console.done(f"MySQL {version[0]}")
        return True, None

    except pymysql.MySQLError as e:
        Console.fail()
        error_msg = f"MySQL error: {e}"
        Console.error(f"Database connection failed: {error_msg}")
        return False, error_msg

    except OSError as e:
        # Network/socket errors
        Console.fail()
        error_msg = f"Network error: {e}"
        Console.error(f"Database connection failed: {error_msg}")
        return False, error_msg

    except Exception as e:
        Console.fail()
        error_msg = f"Unexpected error: {e}"
        Console.error(f"Database connection failed: {error_msg}")
        return False, error_msg


def handle_database_setup_interactive(env_name: str) -> None:
    """Interactive database setup with retry logic.

    Allows user to try different database options if one fails,
    without restarting the entire installation.

    Parameters
    ----------
    env_name : str
        Name of conda environment where DataJoint is installed

    Returns
    -------
    None
    """
    while True:
        db_choice = prompt_database_setup()

        if db_choice == "compose":
            success, reason = setup_database_compose()
            if success:
                break
            else:
                Console.error("Docker setup failed")
                if reason == "compose_unavailable":
                    print("\nDocker is not available.")
                    print("  Option 1: Install Docker Desktop and restart")
                    print("  Option 2: Choose remote database")
                    print("  Option 3: Skip for now")
                else:
                    print(f"  Error: {reason}")

                if not Console.prompt_yes_no(
                    "\nTry different option?", default_yes=True
                ):
                    Console.warning("Skipping database setup")
                    print("  Configure later: docker compose up -d")
                    print("  Or manually: see docs/DATABASE.md")
                    break
                # Loop continues to show menu again

        elif db_choice == "remote":
            success = setup_database_remote(env_name)
            if success:
                break
            # If remote setup returns False (cancelled), loop to menu

        else:  # skip
            Console.warning("Skipping database setup")
            print("  Configure later: docker compose up -d")
            print("  Or manually: see docs/DATABASE.md")
            break


def handle_database_setup_cli(
    env_name: str,
    db_type: str,
    db_host: Optional[str] = None,
    db_port: Optional[int] = None,
    db_user: Optional[str] = None,
    db_password: Optional[str] = None,
) -> None:
    """Handle database setup from CLI arguments.

    Parameters
    ----------
    env_name : str
        Name of conda environment where DataJoint is installed
    db_type : str
        One of: "compose", "docker" (alias for compose), or "remote"
    db_host : str, optional
        Database host for remote connection
    db_port : int, optional
        Database port for remote connection
    db_user : str, optional
        Database user for remote connection
    db_password : str, optional
        Database password for remote connection

    Returns
    -------
    None
    """
    # Treat 'docker' as alias for 'compose' for backward compatibility
    if db_type == "docker":
        db_type = "compose"

    if db_type == "compose":
        success, reason = setup_database_compose()
        if not success:
            Console.error("Docker setup failed")
            if reason == "compose_unavailable":
                Console.warning("Docker not available")
                print("  Install from: https://docs.docker.com/get-docker/")
            else:
                Console.error(f"Error: {reason}")
            print("  You can configure manually later")
    elif db_type == "remote":
        success = setup_database_remote(
            env_name=env_name,
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
        )
        if not success:
            Console.warning("Remote database setup cancelled")
            print("  You can configure manually later")


def change_database_password(
    host: str,
    port: int,
    user: str,
    old_password: str,
    use_tls: bool,
    env_name: str,
) -> Optional[str]:
    """Prompt user to change their database password using DataJoint.

    Interactive password change flow for new lab members who received
    temporary credentials from their admin. Runs inside the conda environment
    where DataJoint is installed.

    Parameters
    ----------
    host : str
        Database hostname
    port : int
        Database port
    user : str
        Database username
    old_password : str
        Current password (temporary from admin)
    use_tls : bool
        Whether TLS is enabled
    env_name : str
        Name of conda environment where DataJoint is installed

    Returns
    -------
    str or None
        New password if changed, None if user skipped or error occurred

    Notes
    -----
    Prompts user for new password, then uses dj.set_password() (running in
    conda env) to change password on MySQL server. All sensitive data
    (host, user, passwords) are passed via environment variables for security
    (not visible in process listings via ps).
    """
    import getpass

    print("\n" + "=" * 60)
    print("Password Change (Recommended for lab members)")
    print("=" * 60)
    print("\nIf you received temporary credentials from your lab admin,")
    print("you should change your password now for security.")
    print()

    if not Console.prompt_yes_no("Change password?", default_yes=True):
        Console.warning("Keeping current password")
        return None

    # Prompt for new password with confirmation
    while True:
        print()
        new_password = getpass.getpass("  New password: ")
        if not new_password:
            Console.error("Password cannot be empty")
            continue

        confirm_password = getpass.getpass("  Confirm password: ")
        if new_password != confirm_password:
            Console.error("Passwords do not match")
            if not Console.prompt_yes_no("  Try again?", default_yes=True):
                return None
            continue

        break

    # Change password using DataJoint in conda environment
    Console.step("Changing password on database server")

    # Build Python code to run inside conda environment
    # All sensitive data (passwords) passed via environment variables for security
    # (not visible in process listings via ps)
    python_code = """\
import os, datajoint as dj
dj.config["database.host"] = os.environ["SPYGLASS_DB_HOST"]
dj.config["database.port"] = int(os.environ["SPYGLASS_DB_PORT"])
dj.config["database.user"] = os.environ["SPYGLASS_DB_USER"]
dj.config["database.password"] = os.environ["SPYGLASS_OLD_PASSWORD"]
if os.environ.get("SPYGLASS_USE_TLS") == "1":
    dj.config["database.use_tls"] = True
dj.conn()
dj.set_password(new_password=os.environ["SPYGLASS_NEW_PASSWORD"], update_config=True)
print("SUCCESS")
"""
    conda_cmd = CondaManager.get_command()

    try:
        env = os.environ.copy()
        env["SPYGLASS_DB_HOST"] = host
        env["SPYGLASS_DB_PORT"] = str(port)
        env["SPYGLASS_DB_USER"] = user
        env["SPYGLASS_OLD_PASSWORD"] = old_password
        env["SPYGLASS_NEW_PASSWORD"] = new_password
        if use_tls:
            env["SPYGLASS_USE_TLS"] = "1"

        result = subprocess.run(
            [conda_cmd, "run", "-n", env_name, "python", "-c", python_code],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0 and "SUCCESS" in result.stdout:
            Console.done()
            return new_password
        else:
            Console.fail()
            Console.error(f"Failed to change password: {result.stderr}")
            Console.manual_password_instructions(env_name)
            return None

    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        OSError,
    ) as e:
        Console.fail()
        Console.error(f"Password change failed: {e}")
        Console.manual_password_instructions(env_name)
        return None


def setup_database_remote(
    env_name: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> bool:
    """Set up remote database connection.

    Prompts for connection details (if not provided), tests the connection,
    optionally changes password for new lab members, and creates configuration
    file if connection succeeds.

    Parameters
    ----------
    env_name : str
        Name of conda environment where DataJoint is installed
    host : str, optional
        Database host (prompts if not provided)
    port : int, optional
        Database port (prompts if not provided)
    user : str, optional
        Database user (prompts if not provided)
    password : str, optional
        Database password (prompts if not provided, checks env var)

    Returns
    -------
    bool
        True if configuration was created, False if cancelled

    Examples
    --------
    >>> if setup_database_remote():
    ...     print("Remote database configured")
    >>> if setup_database_remote(host="db.example.com", user="myuser"):
    ...     print("Non-interactive setup succeeded")
    """
    print("Setting up remote database connection...")

    # If any parameters are missing, prompt interactively
    if host is None or user is None or password is None:
        config = prompt_remote_database_config()
        if config is None:
            return False
    else:
        # Non-interactive mode - use provided parameters
        # (host, user, and password are all not None at this point)

        # Use defaults for optional parameters
        if port is None:
            port = 3306

        # Validate all database configuration parameters
        valid, errors = validate_database_config(host, port, user, password)
        if not valid:
            Console.error("Invalid database configuration:")
            for err in errors:
                print(f"  - {err}")
            return False

        # Check if port is reachable (for remote hosts only)
        if host not in LOCALHOST_ADDRESSES:
            print(f"  Testing connection to {host}:{port}...")
            port_reachable, port_msg = is_port_available(host, port)
            if not port_reachable:
                Console.warning(port_msg)
                print("  Port may be blocked by firewall or wrong port number")
                print("  Continuing anyway (connection test will verify)...")
            else:
                print("  ✓ Port is reachable")

        # Determine TLS based on host
        use_tls = host not in LOCALHOST_ADDRESSES

        config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "use_tls": use_tls,
        }

        print(f"  Connecting to {host}:{port} as {user}")
        if use_tls:
            print("  TLS: enabled")

    host = config["host"]
    port = config["port"]
    user = config["user"]

    # Test connection before saving
    success, _error = test_database_connection(**config)

    if not success:
        Console.error(f"Cannot connect to database: {_error}")
        print()
        print("Most common causes (in order):")
        print("  1. Wrong password - Double check credentials")
        print("  2. Firewall blocking connection")
        print("  3. Database not running")
        print("  4. TLS mismatch")
        print()
        print("Diagnostic steps:")
        print(f"  Test port:  nc -zv {host} {port}")
        print(f"  Test MySQL: mysql -h {host} -P {port} -u {user} -p")
        print()
        print(
            "Need help? See: docs/TROUBLESHOOTING.md#database-connection-fails"
        )
        print()

        if Console.prompt_yes_no(
            "Retry with different settings?", default_yes=False
        ):
            return setup_database_remote(env_name)  # Recursive retry
        Console.warning("Database setup cancelled")
        return False

    # Offer password change for new lab members (only for non-localhost)
    if config["host"] not in LOCALHOST_ADDRESSES:
        new_password = change_database_password(
            host=config["host"],
            port=config["port"],
            user=config["user"],
            old_password=config["password"],
            use_tls=config["use_tls"],
            env_name=env_name,
        )
        # Update config with new password if changed
        if new_password is not None:
            config["password"] = new_password

    # Save configuration
    create_database_config(**config)
    return True


def validate_installation(env_name: str) -> bool:
    """Run validation checks.

    Executes validate.py script in the specified conda environment to
    verify installation success.

    Parameters
    ----------
    env_name : str
        Name of the conda environment to validate

    Returns
    -------
    bool
        True if all critical checks passed, False if any failed

    Notes
    -----
    Prints warnings if validation fails but does not raise exceptions.
    """
    Console.step("Validating installation")

    validate_script = Path(__file__).parent / "validate.py"
    conda_cmd = CondaManager.get_command()

    try:
        subprocess.run(
            [conda_cmd, "run", "-n", env_name, "python", str(validate_script)],
            check=True,
        )
        Console.done()
        return True
    except subprocess.CalledProcessError:
        Console.fail()
        Console.warning("Some optional validation checks did not pass")
        print(
            "\n  Core installation succeeded, but some features may need attention."
        )
        print("  Many warnings are not critical for getting started.")
        print("\n  Common non-critical warnings:")
        print("    - Database connection: Configure later if needed")
        print("    - Optional packages: Install when you need them")
        print("\n  To investigate:")
        print(f"    1. Activate environment: conda activate {env_name}")
        print("    2. Run detailed validation: python scripts/validate.py -v")
        print("    3. See docs/TROUBLESHOOTING.md for specific issues")
        return False


def print_installation_header() -> None:
    """Print installation header banner."""
    print(f"\n{COLORS['blue']}{'='*60}{COLORS['reset']}")
    print(f"{COLORS['blue']}  Spyglass Installation{COLORS['reset']}")
    print(f"{COLORS['blue']}{'='*60}{COLORS['reset']}\n")


def determine_installation_type(args: argparse.Namespace) -> Tuple[str, str]:
    """Determine installation type from CLI args or interactive prompt.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Returns
    -------
    env_file : str
        Path to environment YAML file
    install_type : str
        Installation type identifier ("minimal" or "full")
    """
    if args.minimal:
        return "environment_min.yml", "minimal"
    if args.full:
        return "environment.yml", "full"
    return prompt_install_type()


def setup_environment(
    env_file: str,
    install_type: str,
    base_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Set up conda environment and install spyglass.

    Parameters
    ----------
    env_file : str
        Path to environment YAML file
    install_type : str
        Installation type ("minimal" or "full")
    base_dir : Path
        Base directory for Spyglass data
    args : argparse.Namespace
        Parsed command-line arguments
    """
    check_prerequisites(install_type, base_dir)
    conda = CondaManager(args.env_name)
    conda.create(env_file, force=args.force)
    conda.install_package()


def setup_database(args: argparse.Namespace) -> None:
    """Handle database setup based on CLI args or interactive prompt.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    """
    if args.docker:
        handle_database_setup_cli(args.env_name, "docker")
        return
    if args.remote:
        import os

        db_password = args.db_password or os.environ.get("SPYGLASS_DB_PASSWORD")
        handle_database_setup_cli(
            args.env_name,
            "remote",
            db_host=args.db_host,
            db_port=args.db_port,
            db_user=args.db_user,
            db_password=db_password,
        )
        return
    handle_database_setup_interactive(args.env_name)


def print_completion_message(env_name: str, validation_passed: bool) -> None:
    """Print installation completion message with next steps.

    Parameters
    ----------
    env_name : str
        Name of the conda environment
    validation_passed : bool
        Whether validation checks passed
    """
    print(f"\n{COLORS['green']}{'='*60}{COLORS['reset']}")
    if validation_passed:
        print(f"{COLORS['green']}Installation complete!{COLORS['reset']}")
        print(f"{COLORS['green']}{'='*60}{COLORS['reset']}\n")
    else:
        print(
            f"{COLORS['yellow']}Installation complete with warnings{COLORS['reset']}"
        )
        print(f"{COLORS['yellow']}{'='*60}{COLORS['reset']}\n")
        print("Core installation succeeded but some features may not work.")
        print("Review warnings above and see: docs/TROUBLESHOOTING.md\n")

    print("Next steps:")
    print(f"  1. Activate environment: conda activate {env_name}")
    print("  2. Start tutorial:       jupyter notebook notebooks/")
    print(
        "  3. View documentation:   https://lorenfranklab.github.io/spyglass/"
    )


def run_dry_run(args: argparse.Namespace) -> None:
    """Show what would be done without making changes.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    """
    print("\n" + "=" * 60)
    print("DRY RUN MODE - No changes will be made")
    print("=" * 60 + "\n")

    # Determine installation type
    if args.minimal:
        env_file, install_type = "environment_min.yml", "minimal"
    elif args.full:
        env_file, install_type = "environment.yml", "full"
    else:
        env_file, install_type = "environment_min.yml", "minimal (default)"

    # Determine base directory
    base_dir = (
        args.base_dir
        or os.getenv("SPYGLASS_BASE_DIR")
        or str(Path.home() / "spyglass_data")
    )

    # Determine database setup
    if args.docker:
        db_setup = "Docker Compose (local database)"
    elif args.remote:
        db_host = args.db_host or "<interactive prompt>"
        db_setup = f"Remote database ({db_host}:{args.db_port})"
    else:
        db_setup = "Interactive prompt"

    # Get disk space requirements
    space_req = DISK_SPACE_REQUIREMENTS.get(
        install_type.split()[0], DISK_SPACE_REQUIREMENTS["minimal"]
    )

    print("Would perform the following steps:\n")

    print(f"1. {SYMBOLS['step']} Check prerequisites")
    print(
        f"     Python version: {sys.version_info.major}.{sys.version_info.minor}"
    )
    print(f"     Required disk space: {space_req['total']} GB")
    print()

    print(f"2. {SYMBOLS['step']} Create conda environment")
    print(f"     Environment name: {args.env_name}")
    print(f"     Environment file: {env_file}")
    print(f"     Install type: {install_type}")
    print()

    print(f"3. {SYMBOLS['step']} Install spyglass package")
    print("     Command: pip install -e .")
    print()

    print(f"4. {SYMBOLS['step']} Create directory structure")
    print(f"     Base directory: {base_dir}")
    print(
        "     Subdirectories: raw, analysis, recording, sorting, waveforms, etc."
    )
    print()

    print(f"5. {SYMBOLS['step']} Setup database")
    print(f"     Method: {db_setup}")
    print()

    print(f"6. {SYMBOLS['step']} Create configuration file")
    print(f"     Location: {Path.home() / '.datajoint_config.json'}")
    print()

    if not args.skip_validation:
        print(f"7. {SYMBOLS['step']} Validate installation")
        print("     Run: python scripts/validate.py")
        print()

    print("=" * 60)
    print("To perform installation, run without --dry-run flag")
    print("=" * 60)


def run_config_only(args: argparse.Namespace) -> None:
    """Generate only the .datajoint_config.json file without environment setup.

    This mode is useful for:
    - Regenerating config after manual changes
    - Setting up config on a system where spyglass is already installed
    - Creating config for Frank Lab members on existing environments

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing database and path options
    """
    print("\n" + "=" * 60)
    print("CONFIG-ONLY MODE - Generating configuration file")
    print("=" * 60 + "\n")

    # Get base directory
    base_dir = get_base_directory(args.base_dir)

    # Determine database settings
    if args.docker:
        # Docker mode: localhost with default credentials
        host = "localhost"
        port = args.db_port
        user = args.db_user
        password = args.db_password or os.getenv(
            "SPYGLASS_DB_PASSWORD", DEFAULT_MYSQL_PASSWORD
        )
    elif args.remote or args.db_host:
        # Remote mode: use provided credentials
        host = args.db_host
        if not host:
            host = input("Database host: ").strip()
            if not host:
                raise ValueError("Database host is required for --remote mode")
        port = args.db_port
        user = args.db_user
        password = args.db_password or os.getenv("SPYGLASS_DB_PASSWORD")
        if not password:
            import getpass

            password = getpass.getpass("Database password: ")
            if not password:
                raise ValueError("Database password is required")
    else:
        # Interactive mode
        print("Database configuration:")
        print("  1. Local Docker database (localhost)")
        print("  2. Remote database (e.g., lmf-db.cin.ucsf.edu)")
        choice = input("\nChoice [1/2]: ").strip() or "1"

        if choice == "1":
            host = "localhost"
            port = args.db_port
            user = args.db_user
            password = args.db_password or DEFAULT_MYSQL_PASSWORD
        else:
            host = input("Database host: ").strip()
            port = args.db_port
            user = (
                input(f"Database user [{args.db_user}]: ").strip()
                or args.db_user
            )
            import getpass

            password = getpass.getpass("Database password: ")

    # Validate database config
    valid, errors = validate_database_config(host, port, user, password)
    if not valid:
        for error in errors:
            Console.error(error)
        raise ValueError("Invalid database configuration")

    # Create the configuration
    Console.step(f"Creating configuration for {host}:{port}")
    create_database_config(
        host=host,
        port=port,
        user=user,
        password=password,
        base_dir=base_dir,
    )
    Console.done()

    config_file = Path.home() / ".datajoint_config.json"
    print()
    print("=" * 60)
    Console.success(f"Configuration created: {config_file}")
    print("=" * 60)
    print()
    print("Configuration summary:")
    print(f"  Database: {host}:{port}")
    print(f"  User: {user}")
    print(f"  Base directory: {base_dir}")
    print(f"  TLS: {'enabled' if determine_tls(host) else 'disabled'}")
    print()
    print("To test your configuration:")
    print('  python -c "import datajoint as dj; dj.conn()"')


def run_installation(args: argparse.Namespace) -> None:
    """Main installation flow.

    Orchestrates the complete installation process in a specific order
    to avoid import issues and ensure proper setup.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing installation options

    Notes
    -----
    CRITICAL ORDER:
    1. Get base directory (for disk space check)
    2. Check prerequisites including disk space (no spyglass imports)
    3. Create conda environment (no spyglass imports)
    4. Install spyglass package (pip install -e .)
    5. Setup database (inline code, NO spyglass imports)
    6. Validate (runs IN the new environment, CAN import spyglass)
    """
    # Handle dry-run mode
    if args.dry_run:
        run_dry_run(args)
        return

    # Handle config-only mode
    if args.config_only:
        run_config_only(args)
        return

    print_installation_header()

    # Step 1: Determine installation type
    env_file, install_type = determine_installation_type(args)

    # Step 2: Get base directory (CLI arg > env var > prompt)
    base_dir = get_base_directory(args.base_dir)

    # Step 3: Set up environment and install package
    setup_environment(env_file, install_type, base_dir, args)

    # Step 4: Database setup (inline code - no spyglass imports)
    setup_database(args)

    # Step 5: Validation (runs in new environment, CAN import spyglass)
    validation_passed = True
    if not args.skip_validation:
        validation_passed = validate_installation(args.env_name)

    # Step 6: Show completion message
    print_completion_message(args.env_name, validation_passed)


def main() -> None:
    """Main entry point for Spyglass installer.

    Parses command-line arguments and runs the installation process.
    """
    parser = argparse.ArgumentParser(
        description="Install Spyglass in one command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/install.py                  # Interactive
  python scripts/install.py --minimal        # Minimal install
  python scripts/install.py --full --docker  # Full with local database
  python scripts/install.py --remote         # Connect to remote database
  python scripts/install.py --config-only --remote --db-host lmf-db.cin.ucsf.edu --db-user alice --base-dir ~/data
                                             # Generate config only (no env)

Environment Variables:
  SPYGLASS_BASE_DIR - Set base directory (skips prompt)
        """,
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Install minimal dependencies only",
    )
    parser.add_argument(
        "--full", action="store_true", help="Install all dependencies"
    )
    parser.add_argument(
        "--docker", action="store_true", help="Set up local Docker database"
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Connect to remote database (interactive)",
    )
    parser.add_argument("--db-host", help="Database host (for --remote)")
    parser.add_argument(
        "--db-port",
        type=int,
        default=3306,
        help="Database port (default: 3306)",
    )
    parser.add_argument(
        "--db-user", default="root", help="Database user (default: root)"
    )
    parser.add_argument(
        "--db-password",
        help="Database password (or use SPYGLASS_DB_PASSWORD env var)",
    )
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip validation checks"
    )
    parser.add_argument(
        "--env-name",
        default="spyglass",
        help="Conda environment name (default: spyglass)",
    )
    parser.add_argument(
        "--base-dir",
        help="Base directory for data (overrides SPYGLASS_BASE_DIR)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing environment without prompting",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Only generate .datajoint_config.json (skip environment setup)",
    )

    args = parser.parse_args()

    try:
        run_installation(args)
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        sys.exit(1)
    except RuntimeError as e:
        # Expected errors from our code (prerequisites, validation, etc.)
        Console.error(f"Installation failed: {e}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        # Process execution failures
        Console.error(f"Command failed: {e}")
        print("  Check the output above for details")
        sys.exit(1)
    except (OSError, IOError) as e:
        # File system errors
        Console.error(f"File system error: {e}")
        print("  Check disk space and permissions")
        sys.exit(1)
    except ValueError as e:
        # Configuration/validation errors
        Console.error(f"Invalid configuration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
