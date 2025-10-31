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
COLORS = (
    {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "blue": "\033[94m",
        "reset": "\033[0m",
    }
    if sys.platform != "win32"
    else {k: "" for k in ["green", "yellow", "red", "blue", "reset"]}
)

# System constants
BYTES_PER_GB = 1024**3
LOCALHOST_ADDRESSES = frozenset(["localhost", "127.0.0.1", "::1"])

# Disk space requirements (GB)
DISK_SPACE_REQUIREMENTS = {
    "minimal": 10,
    "full": 25,
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


def print_step(msg: str) -> None:
    """Print installation step message.

    Parameters
    ----------
    msg : str
        Message to display
    """
    print(f"{COLORS['blue']}▶{COLORS['reset']} {msg}")


def print_success(msg: str) -> None:
    """Print success message.

    Parameters
    ----------
    msg : str
        Success message to display
    """
    print(f"{COLORS['green']}✓{COLORS['reset']} {msg}")


def print_warning(msg: str) -> None:
    """Print warning message.

    Parameters
    ----------
    msg : str
        Warning message to display
    """
    print(f"{COLORS['yellow']}⚠{COLORS['reset']} {msg}")


def print_error(msg: str) -> None:
    """Print error message.

    Parameters
    ----------
    msg : str
        Error message to display
    """
    print(f"{COLORS['red']}✗{COLORS['reset']} {msg}")


def show_progress_message(operation: str, estimated_minutes: int) -> None:
    """Show progress message for long-running operation.

    Displays estimated time and user-friendly messages to prevent
    users from thinking the installer has frozen.

    Parameters
    ----------
    operation : str
        Description of the operation being performed
    estimated_minutes : int
        Estimated completion time in minutes

    Returns
    -------
    None

    Examples
    --------
    >>> show_progress_message("Installing packages", 10)
    """
    print_step(operation)
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
    except Exception:
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
    print_step("Checking prerequisites...")

    # Get Python version requirement from pyproject.toml
    min_version = get_required_python_version()

    # Python version
    if sys.version_info < min_version:
        raise RuntimeError(
            f"Python {min_version[0]}.{min_version[1]}+ required, "
            f"found {sys.version_info.major}.{sys.version_info.minor}"
        )
    print_success(f"Python {sys.version_info.major}.{sys.version_info.minor}")

    # Conda/Mamba
    conda_cmd = get_conda_command()
    print_success(f"Package manager: {conda_cmd}")

    # Git (optional but recommended)
    if not shutil.which("git"):
        print_warning("Git not found (recommended for development)")
    else:
        print_success("Git available")

    # Disk space check (if base_dir provided)
    if base_dir:
        # Add buffer: minimal needs ~10GB (8 + 2), full needs ~25GB (18 + 7)
        required_space = {"minimal": 10, "full": 25}
        required_gb = required_space.get(install_type, 10)

        sufficient, available_gb = check_disk_space(required_gb, base_dir)

        if sufficient:
            print_success(
                f"Disk space: {available_gb} GB available (need {required_gb} GB)"
            )
        else:
            print_error("Insufficient disk space!")
            print(f"  Available: {available_gb} GB")
            print(f"  Required:  {required_gb} GB")
            print("  Please free up space or choose a different location")
            raise RuntimeError("Insufficient disk space")


def get_conda_command() -> str:
    """Get conda or mamba command.

    Returns:
        'mamba' if available, else 'conda'

    Raises:
        RuntimeError: If neither conda nor mamba found
    """
    if shutil.which("mamba"):
        return "mamba"
    elif shutil.which("conda"):
        return "conda"
    else:
        raise RuntimeError(
            "conda or mamba not found. Install from:\n"
            "  https://github.com/conda-forge/miniforge"
        )


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
        print_success(f"Using base directory from CLI: {validated_path}")
        return validated_path

    # 2. Environment variable (second priority)
    if base_env := os.getenv("SPYGLASS_BASE_DIR"):
        base_path = Path(base_env).expanduser().resolve()
        validated_path = validate_and_test_write(base_path)
        print_success(
            f"Using base directory from environment: {validated_path}"
        )
        return validated_path

    # 3. Interactive prompt
    print("\nWhere should Spyglass store data?")
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
                print_success(f"Base directory validated: {validated_path}")
                return validated_path
            except RuntimeError as e:
                print_error(str(e))
                continue

        try:
            base_path = Path(response).expanduser().resolve()

            # Validate parent exists
            if not base_path.parent.exists():
                print_error(
                    f"Parent directory does not exist: {base_path.parent}"
                )
                print(
                    "  Please create parent directory first or choose another location"
                )
                continue

            # Warn if directory already exists
            if base_path.exists():
                if not base_path.is_dir():
                    print_error(
                        f"Path exists but is not a directory: {base_path}"
                    )
                    continue

                response = (
                    input("Directory exists. Use it? [Y/n]: ").strip().lower()
                )
                if response in ["n", "no"]:
                    continue

            # Validate write permissions
            validated_path = validate_and_test_write(base_path)
            print_success(f"Base directory validated: {validated_path}")
            return validated_path

        except RuntimeError as e:
            print_error(str(e))
            continue
        except (ValueError, OSError) as e:
            print_error(f"Invalid path: {e}")


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
        Path to environment YAML file ("environment-min.yml" or "environment.yml")
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

    print("\n1. Minimal (Recommended for getting started)")
    print("   ├─ Install time: ~5 minutes")
    print("   ├─ Disk space: ~8 GB")
    print("   ├─ Includes:")
    print("   │  • Core Spyglass functionality")
    print("   │  • Common data tables")
    print("   │  • Position tracking")
    print("   │  • LFP analysis")
    print("   │  • Basic spike sorting")
    print("   └─ Good for: Learning, basic workflows")

    print("\n2. Full (For advanced analysis)")
    print("   ├─ Install time: ~15 minutes")
    print("   ├─ Disk space: ~18 GB")
    print("   ├─ Includes: Everything in Minimal, plus:")
    print("   │  • Advanced spike sorting (Kilosort, etc.)")
    print("   │  • Ripple detection")
    print("   │  • Track linearization")
    print("   └─ Good for: Production work, all features")

    print("\nNote: DeepLabCut, Moseq, and some decoding features")
    print("      require separate installation (see docs)")

    # Map choices to (env_file, install_type)
    choice_map = {
        "1": ("environment-min.yml", "minimal"),
        "2": ("environment.yml", "full"),
    }

    while True:
        choice = input("\nChoice [1-2]: ").strip()

        if choice not in choice_map:
            print_error("Please enter 1 or 2")
            continue

        env_file, install_type = choice_map[choice]
        print_success(f"Selected: {install_type.capitalize()} installation")
        return env_file, install_type


def create_conda_environment(
    env_file: str, env_name: str, force: bool = False
) -> None:
    """Create conda environment from file.

    Parameters
    ----------
    env_file : str
        Path to environment.yml file
    env_name : str
        Name for the environment
    force : bool, optional
        If True, overwrite existing environment without prompting (default: False)

    Raises
    ------
    RuntimeError
        If environment creation fails
    """
    # Estimate time based on environment type
    estimated_time = 5 if "min" in env_file else 15

    show_progress_message(
        f"Creating environment '{env_name}' from {env_file}", estimated_time
    )

    # Check if environment already exists
    result = subprocess.run(
        ["conda", "env", "list"], capture_output=True, text=True
    )

    if env_name in result.stdout:
        if not force:
            response = input(
                f"Environment '{env_name}' exists. Overwrite? [y/N]: "
            )
            if response.lower() not in ["y", "yes"]:
                print_success(f"Using existing environment '{env_name}'")
                print(
                    "  Package installation will continue (updates if needed)"
                )
                print("  To use a different name, run with: --env-name <name>")
                return  # Skip environment creation, use existing

        print_step(f"Removing existing environment '{env_name}'...")
        subprocess.run(
            ["conda", "env", "remove", "-n", env_name, "-y"], check=True
        )

    # Create environment with progress indication
    conda_cmd = get_conda_command()
    print("  Installing packages... (this will take several minutes)")

    try:
        # Use Popen to show real-time progress
        process = subprocess.Popen(
            [conda_cmd, "env", "create", "-f", env_file, "-n", env_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Show dots to indicate progress
        for line in process.stdout:
            if (
                "Solving environment" in line
                or "Downloading" in line
                or "Extracting" in line
            ):
                print(".", end="", flush=True)

        process.wait()
        print()  # New line after dots

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, conda_cmd)

        print_success(f"Environment '{env_name}' created")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to create environment. Try:\n"
            f"  1. Update conda: conda update conda\n"
            f"  2. Clear cache: conda clean --all\n"
            f"  3. Check {env_file} for conflicts"
        ) from e


def install_spyglass_package(env_name: str) -> None:
    """Install spyglass package in development mode.

    Parameters
    ----------
    env_name : str
        Name of the conda environment
    """
    show_progress_message("Installing spyglass package", 1)

    try:
        subprocess.run(
            ["conda", "run", "-n", env_name, "pip", "install", "-e", "."],
            check=True,
        )
        print_success("Spyglass installed")
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Failed to install spyglass package") from e


# Docker operations (inline - cannot import from spyglass before it's installed)


def is_docker_available_inline() -> bool:
    """Check if Docker is available (inline, no imports).

    Checks both that docker command exists and daemon is running.

    Parameters
    ----------
    None

    Returns
    -------
    bool
        True if Docker is available and running, False otherwise

    Notes
    -----
    This is self-contained because spyglass isn't installed yet.
    """
    if not shutil.which("docker"):
        return False

    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def is_docker_compose_available_inline() -> bool:
    """Check if Docker Compose is installed (inline, no imports).

    Returns
    -------
    bool
        True if 'docker compose' command is available, False otherwise

    Notes
    -----
    This is self-contained because spyglass isn't installed yet.
    Checks for modern 'docker compose' (not legacy 'docker-compose').
    """
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_compose_command_inline() -> list[str]:
    """Get the appropriate Docker Compose command (inline, no imports).

    Returns
    -------
    list[str]
        Command prefix for Docker Compose (e.g., ['docker', 'compose'])

    Notes
    -----
    This is self-contained because spyglass isn't installed yet.
    Always returns modern 'docker compose' format.
    """
    return ["docker", "compose"]


def generate_env_file_inline(
    mysql_port: int = 3306,
    mysql_password: str = "tutorial",
    mysql_image: str = "datajoint/mysql:8.0",
    env_path: str = ".env",
) -> None:
    """Generate .env file for Docker Compose (inline, no imports).

    Parameters
    ----------
    mysql_port : int, optional
        MySQL port number (default: 3306)
    mysql_password : str, optional
        MySQL root password (default: 'tutorial')
    mysql_image : str, optional
        Docker image to use (default: 'datajoint/mysql:8.0')
    env_path : str, optional
        Path to write .env file (default: '.env')

    Returns
    -------
    None

    Raises
    ------
    OSError
        If file cannot be written

    Notes
    -----
    This is self-contained because spyglass isn't installed yet.
    Only writes non-default values to keep .env file minimal.
    """
    env_lines = ["# Spyglass Docker Compose Configuration", ""]

    # Only write non-default values
    if mysql_password != "tutorial":
        env_lines.append(f"MYSQL_ROOT_PASSWORD={mysql_password}")
    if mysql_port != 3306:
        env_lines.append(f"MYSQL_PORT={mysql_port}")
    if mysql_image != "datajoint/mysql:8.0":
        env_lines.append(f"MYSQL_IMAGE={mysql_image}")

    # If all defaults, don't create file (compose will use defaults)
    if len(env_lines) == 2:  # Only header lines
        return

    env_path_obj = Path(env_path)
    with env_path_obj.open("w") as f:
        f.write("\n".join(env_lines) + "\n")


def validate_env_file_inline(env_path: str = ".env") -> bool:
    """Validate .env file exists and is readable (inline, no imports).

    Parameters
    ----------
    env_path : str, optional
        Path to .env file (default: '.env')

    Returns
    -------
    bool
        True if file exists and is readable (or doesn't exist, which is OK),
        False if file exists but has issues

    Notes
    -----
    This is self-contained because spyglass isn't installed yet.
    Missing .env file is NOT an error (defaults will be used).
    """
    import os

    # Missing .env is fine - compose uses defaults
    if not os.path.exists(env_path):
        return True

    # If it exists, make sure it's readable
    try:
        env_path_obj = Path(env_path)
        with env_path_obj.open("r") as f:
            f.read()
        return True
    except (OSError, PermissionError):
        return False


def create_database_config(
    host: str = "localhost",
    port: int = 3306,
    user: str = "root",
    password: str = "tutorial",
    use_tls: bool = False,
) -> None:
    """Create DataJoint configuration file.

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
    use_tls : bool, optional
        Whether to use TLS/SSL (default: False)

    Notes
    -----
    Uses JSON for safety (no code injection vulnerability).
    """
    # Use JSON for safety (no code injection)
    dj_config = {
        "database.host": host,
        "database.port": port,
        "database.user": user,
        "database.password": password,
        "database.use_tls": use_tls,
    }

    config_file = Path.home() / ".datajoint_config.json"

    if config_file.exists():
        response = input(f"{config_file} exists. Overwrite? [y/N]: ")
        if response.lower() not in ["y", "yes"]:
            print_warning("Keeping existing configuration")
            return

    with config_file.open("w") as f:
        json.dump(dj_config, f, indent=2)

    print_success(f"Configuration saved to {config_file}")


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
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)  # 1 second timeout
            result = sock.connect_ex((host, port))

            # For localhost, we want the port to be FREE (not in use)
            # For remote, we want the port to be IN USE (something listening)

            if host in LOCALHOST_ADDRESSES:
                # Checking if local port is free for Docker/services
                if result == 0:
                    # Port is in use
                    return False, f"Port {port} is already in use on {host}"
                else:
                    # Port is free
                    return True, f"Port {port} is available on {host}"
            else:
                # Checking if remote port is reachable
                if result == 0:
                    # Port is reachable (good!)
                    return True, f"Port {port} is reachable on {host}"
                else:
                    # Port is not reachable
                    return (
                        False,
                        f"Cannot reach {host}:{port} (firewall/wrong port?)",
                    )

    except socket.gaierror:
        # DNS resolution failed
        return False, f"Cannot resolve hostname: {host}"
    except socket.error as e:
        # Other socket errors
        return False, f"Socket error: {e}"


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
    print("  Enter connection details for your MySQL database")
    print("  (Press Ctrl+C to cancel)")

    try:
        host = input("  Host [localhost]: ").strip() or "localhost"

        # Validate hostname format
        if not validate_hostname(host):
            print_error(f"Invalid hostname: {host}")
            print("  Hostname cannot contain spaces or invalid characters")
            return None
        port_str = input("  Port [3306]: ").strip() or "3306"
        user = input("  User [root]: ").strip() or "root"

        # Use getpass for password to hide input
        import getpass

        password = getpass.getpass("  Password: ")

        # Parse port
        try:
            port = int(port_str)
            if not (1 <= port <= 65535):
                raise ValueError("Port must be between 1 and 65535")
        except ValueError as e:
            print_error(f"Invalid port: {e}")
            return None

        # Check if port is reachable
        print(f"  Testing connection to {host}:{port}...")
        port_reachable, port_msg = is_port_available(host, port)

        if host not in LOCALHOST_ADDRESSES and not port_reachable:
            # Remote host, port not reachable
            print_warning(port_msg)
            print("\n  Possible causes:")
            print("    • Wrong port number (MySQL usually uses 3306)")
            print("    • Firewall blocking connections")
            print("    • Database server not running")
            print("    • Wrong hostname")
            print("\n  Common MySQL ports:")
            print("    • Standard MySQL: 3306")
            print("    • SSH tunnel: Check your tunnel configuration")

            retry = input("\n  Continue anyway? [y/N]: ").strip().lower()
            if retry not in ["y", "yes"]:
                return None
        elif port_reachable:
            print("  ✓ Port is reachable")

        # Determine TLS based on host (use TLS for non-localhost)
        use_tls = host not in LOCALHOST_ADDRESSES

        if use_tls:
            print_warning(f"TLS will be enabled for remote host '{host}'")
            tls_response = input("  Disable TLS? [y/N]: ").strip().lower()
            if tls_response in ["y", "yes"]:
                use_tls = False
                print_warning(
                    "TLS disabled (not recommended for remote connections)"
                )

        return {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "use_tls": use_tls,
        }

    except KeyboardInterrupt:
        print("\n")
        print_warning("Database configuration cancelled")
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
    compose_available = is_docker_compose_available_inline()

    if compose_available:
        options.append(
            DatabaseOption(
                number="1",
                name="Docker",
                status="✓ Available (Recommended)",
                description="Automatic local database setup",
            )
        )
    else:
        options.append(
            DatabaseOption(
                number="1",
                name="Docker",
                status="✗ Not available",
                description="Requires Docker Desktop",
            )
        )

    options.append(
        DatabaseOption(
            number="2",
            name="Remote",
            status="✓ Available",
            description="Connect to existing lab/cloud database",
        )
    )
    options.append(
        DatabaseOption(
            number="3",
            name="Skip",
            status="✓ Available",
            description="Configure manually later",
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
        # Color status based on availability
        status_color = COLORS["green"] if "✓" in opt.status else COLORS["red"]
        print(
            f"  {opt.number}. {opt.name:20} {status_color}{opt.status}{COLORS['reset']}"
        )
        print(f"      {opt.description}")

    # If Docker not available, guide user
    if not compose_available:
        print(f"\n{COLORS['yellow']}⚠{COLORS['reset']} Docker is not available")
        print("  To enable Docker setup:")
        print(
            "    1. Install Docker Desktop: https://docs.docker.com/get-docker/"
        )
        print("    2. Start Docker Desktop")
        print("    3. Verify: docker compose version")
        print("    4. Re-run installer")

    # Map choices to actions
    choice_map = {
        "1": "compose",
        "2": "remote",
        "3": "skip",
    }

    # Get valid choices
    valid_choices = ["2", "3"]  # Remote and Skip always available
    if compose_available:
        valid_choices.insert(0, "1")

    while True:
        choice = input(f"\nChoice [{'/'.join(valid_choices)}]: ").strip()

        if choice not in choice_map:
            print_error(f"Please enter {' or '.join(valid_choices)}")
            continue

        # Handle Docker unavailability
        if choice == "1" and not compose_available:
            print_error("Docker is not available")
            continue

        return choice_map[choice]


def cleanup_failed_compose_setup_inline() -> None:
    """Clean up after failed Docker Compose setup (inline, no imports).

    Stops and removes containers created by Docker Compose if setup fails.
    This ensures a clean state for retry attempts.

    Returns
    -------
    None

    Notes
    -----
    This is self-contained because spyglass isn't installed yet.
    Silently handles errors - cleanup is best-effort.
    """
    try:
        compose_cmd = get_compose_command_inline()
        subprocess.run(
            compose_cmd + ["down", "-v"],
            capture_output=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass  # Best-effort cleanup


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

    print_step("Setting up database with Docker Compose...")

    # Check Docker Compose availability
    if not is_docker_compose_available_inline():
        return False, "compose_unavailable"

    # Check if port 3306 is available
    port = 3306  # Default port (could be customized via .env)
    port_available, port_msg = is_port_available("localhost", port)
    if not port_available:
        print_error(port_msg)
        print("\n  Port 3306 is already in use. Solutions:")

        # Platform-specific guidance
        if sys.platform == "darwin":  # macOS
            print("    1. Stop existing MySQL (if installed):")
            print("       brew services stop mysql")
            print(
                "       # or: sudo launchctl unload -w /Library/LaunchDaemons/com.mysql.mysql.plist"
            )
            print("    2. Find what's using the port:")
            print("       lsof -i :3306")
        elif sys.platform.startswith("linux"):  # Linux
            print("    1. Stop existing MySQL service:")
            print("       sudo systemctl stop mysql")
            print("       # or: sudo service mysql stop")
            print("    2. Find what's using the port:")
            print("       sudo lsof -i :3306")
            print("       # or: sudo netstat -tulpn | grep 3306")
        elif sys.platform == "win32":  # Windows
            print("    1. Stop existing MySQL service:")
            print("       net stop MySQL")
            print("       # or use Services app (services.msc)")
            print("    2. Find what's using the port:")
            print("       netstat -ano | findstr :3306")

        print("    Alternative: Use a different port:")
        print("       Create .env file with: MYSQL_PORT=3307")
        print("       (and update DataJoint config to match)")
        return False, "port_in_use"

    # Show what will happen
    print("\n" + "=" * 60)
    print("Docker Database Setup")
    print("=" * 60)
    print("\nThis will:")
    print("  • Download MySQL 8.0 Docker image (~200 MB)")
    print("  • Create a container named 'spyglass-db'")
    print("  • Start MySQL on localhost:3306")
    print("  • Save credentials to ~/.datajoint_config.json")
    print("\nEstimated time: 2-3 minutes")
    print("=" * 60)

    try:
        # Generate .env file (only if customizations needed)
        # For now, use all defaults - no .env file needed
        # Future: could prompt for port/password customization
        generate_env_file_inline()

        # Validate .env if it exists
        if not validate_env_file_inline():
            return False, "env_file_invalid"

        # Get compose command
        compose_cmd = get_compose_command_inline()

        # Pull images first (better UX - shows progress)
        show_progress_message("Pulling Docker images", 2)
        result = subprocess.run(
            compose_cmd + ["pull"],
            capture_output=True,
            timeout=300,  # 5 minutes for image pull
        )
        if result.returncode != 0:
            print_error(f"Failed to pull images: {result.stderr.decode()}")
            return False, "pull_failed"

        # Start services
        print_step("Starting services...")
        result = subprocess.run(
            compose_cmd + ["up", "-d"],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            error_msg = result.stderr.decode()
            print_error(f"Failed to start services: {error_msg}")
            cleanup_failed_compose_setup_inline()
            return False, "start_failed"

        print_success("Services started")

        # Wait for MySQL readiness using health check
        print_step("Waiting for MySQL to be ready...")
        print("  Checking connection", end="", flush=True)

        for attempt in range(30):  # 60 seconds max
            try:
                # Check if service is healthy
                result = subprocess.run(
                    compose_cmd + ["ps", "--format", "json"],
                    capture_output=True,
                    timeout=5,
                )

                if result.returncode == 0:
                    # Parse JSON output to check health
                    import json

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
                            print()  # New line after dots
                            print_success("MySQL is ready")
                            break
                    except json.JSONDecodeError:
                        pass

            except subprocess.TimeoutExpired:
                pass

            if attempt < 29:
                print(".", end="", flush=True)
                time.sleep(2)
        else:
            # Timeout - provide debug info
            print()
            print_error("MySQL did not become ready within 60 seconds")
            print("\n  Check logs:")
            print("    docker compose logs mysql")
            cleanup_failed_compose_setup_inline()
            return False, "timeout"

        # Read actual port/password from .env if it exists
        import os

        actual_port = port
        actual_password = "tutorial"

        env_path = Path(".env")
        if env_path.exists():
            # Parse .env file to check for custom values
            try:
                with env_path.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("MYSQL_PORT="):
                            actual_port = int(line.split("=", 1)[1])
                        elif line.startswith("MYSQL_ROOT_PASSWORD="):
                            actual_password = line.split("=", 1)[1]
            except (OSError, ValueError):
                pass  # Use defaults if .env parsing fails

        # Create configuration file matching .env values
        create_database_config(
            host="localhost",
            port=actual_port,
            user="root",
            password=actual_password,
            use_tls=False,
        )

        # Warn if .env exists with custom values
        if os.path.exists(".env"):
            print_warning(
                "Using custom settings from .env file. "
                "DataJoint config updated to match."
            )

        return True, "success"

    except subprocess.CalledProcessError as e:
        print_error(f"Docker Compose command failed: {e}")
        cleanup_failed_compose_setup_inline()
        return False, str(e)
    except subprocess.TimeoutExpired:
        print_error("Docker Compose command timed out")
        cleanup_failed_compose_setup_inline()
        return False, "timeout"
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        cleanup_failed_compose_setup_inline()
        return False, str(e)


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

        print_step("Testing database connection...")

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
            print(f"  MySQL version: {version[0]}")

        connection.close()
        print_success("Database connection successful!")
        return True, None

    except ImportError:
        # pymysql not available yet (before pip install)
        print_warning("Cannot test connection (pymysql not available)")
        print("  Connection will be tested during validation")
        return True, None  # Allow to proceed

    except Exception as e:
        error_msg = str(e)
        print_error(f"Database connection failed: {error_msg}")
        return False, error_msg


def handle_database_setup_interactive() -> None:
    """Interactive database setup with retry logic.

    Allows user to try different database options if one fails,
    without restarting the entire installation.

    Parameters
    ----------
    None

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
                print_error("Docker setup failed")
                if reason == "compose_unavailable":
                    print("\nDocker is not available.")
                    print("  Option 1: Install Docker Desktop and restart")
                    print("  Option 2: Choose remote database")
                    print("  Option 3: Skip for now")
                else:
                    print(f"  Error: {reason}")

                retry = input("\nTry different option? [Y/n]: ").strip().lower()
                if retry in ["n", "no"]:
                    print_warning("Skipping database setup")
                    print("  Configure later: docker compose up -d")
                    print("  Or manually: see docs/DATABASE.md")
                    break
                # Loop continues to show menu again

        elif db_choice == "remote":
            success = setup_database_remote()
            if success:
                break
            # If remote setup returns False (cancelled), loop to menu

        else:  # skip
            print_warning("Skipping database setup")
            print("  Configure later: docker compose up -d")
            print("  Or manually: see docs/DATABASE.md")
            break


def handle_database_setup_cli(
    db_type: str,
    db_host: Optional[str] = None,
    db_port: Optional[int] = None,
    db_user: Optional[str] = None,
    db_password: Optional[str] = None,
) -> None:
    """Handle database setup from CLI arguments.

    Parameters
    ----------
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
            print_error("Docker setup failed")
            if reason == "compose_unavailable":
                print_warning("Docker not available")
                print("  Install from: https://docs.docker.com/get-docker/")
            else:
                print_error(f"Error: {reason}")
            print("  You can configure manually later")
    elif db_type == "remote":
        success = setup_database_remote(
            host=db_host, port=db_port, user=db_user, password=db_password
        )
        if not success:
            print_warning("Remote database setup cancelled")
            print("  You can configure manually later")


def setup_database_remote(
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> bool:
    """Set up remote database connection.

    Prompts for connection details (if not provided), tests the connection,
    and creates configuration file if connection succeeds.

    Parameters
    ----------
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
    print_step("Setting up remote database connection...")

    # If any parameters are missing, prompt interactively
    if host is None or user is None or password is None:
        config = prompt_remote_database_config()
        if config is None:
            return False
    else:
        # Non-interactive mode - use provided parameters
        import os

        # Validate hostname format
        if not validate_hostname(host):
            print_error(f"Invalid hostname: {host}")
            print("  Hostname cannot contain spaces or invalid characters")
            return False

        # Check environment variable for password if not provided
        if password is None:
            password = os.environ.get("SPYGLASS_DB_PASSWORD")
            if password is None:
                print_error(
                    "Password required: use --db-password or SPYGLASS_DB_PASSWORD env var"
                )
                return False

        # Use defaults for optional parameters
        if port is None:
            port = 3306

        # Check if port is reachable (for remote hosts only)
        if host not in LOCALHOST_ADDRESSES:
            print(f"  Testing connection to {host}:{port}...")
            port_reachable, port_msg = is_port_available(host, port)
            if not port_reachable:
                print_warning(port_msg)
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

    # Test connection before saving
    success, _error = test_database_connection(**config)

    if not success:
        print("\nConnection test failed. Common issues:")
        print("  • Wrong host/port (check firewall)")
        print("  • Incorrect username/password")
        print("  • Database not accessible from this machine")
        print("  • TLS misconfiguration")

        retry = (
            input("\nRetry with different settings? [y/N]: ").strip().lower()
        )
        if retry in ["y", "yes"]:
            return setup_database_remote()  # Recursive retry
        else:
            print_warning("Database setup cancelled")
            return False

    # Save configuration
    create_database_config(**config)
    return True


def validate_installation(env_name: str) -> None:
    """Run validation checks.

    Executes validate.py script in the specified conda environment to
    verify installation success.

    Parameters
    ----------
    env_name : str
        Name of the conda environment to validate

    Returns
    -------
    None

    Notes
    -----
    Prints warnings if validation fails but does not raise exceptions.
    """
    print_step("Validating installation...")

    validate_script = Path(__file__).parent / "validate.py"

    try:
        subprocess.run(
            ["conda", "run", "-n", env_name, "python", str(validate_script)],
            check=True,
        )
        print_success("Validation passed")
    except subprocess.CalledProcessError:
        print_warning("Some validation checks failed")
        print("  Review errors above and see docs/TROUBLESHOOTING.md")


def run_installation(args) -> None:
    """Main installation flow.

    Orchestrates the complete installation process in a specific order
    to avoid import issues and ensure proper setup.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing installation options

    Returns
    -------
    None

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
    print(f"\n{COLORS['blue']}{'='*60}{COLORS['reset']}")
    print(f"{COLORS['blue']}  Spyglass Installation{COLORS['reset']}")
    print(f"{COLORS['blue']}{'='*60}{COLORS['reset']}\n")

    # Determine installation type
    if args.minimal:
        env_file = "environment-min.yml"
        install_type = "minimal"
    elif args.full:
        env_file = "environment.yml"
        install_type = "full"
    else:
        env_file, install_type = prompt_install_type()

    # 1. Get base directory first (CLI arg > env var > prompt)
    base_dir = get_base_directory(args.base_dir)

    # 2. Check prerequisites with disk space validation (no spyglass imports)
    check_prerequisites(install_type, base_dir)

    # 3. Create environment (no spyglass imports)
    create_conda_environment(env_file, args.env_name, force=args.force)

    # 3. Install package (pip install makes spyglass available)
    install_spyglass_package(args.env_name)

    # 4. Database setup (INLINE CODE - no spyglass imports!)
    #    This happens AFTER spyglass is installed but doesn't use it
    #    because docker operations are self-contained
    if args.docker:
        # Docker explicitly requested via CLI
        handle_database_setup_cli("docker")
    elif args.remote:
        # Remote database explicitly requested via CLI
        # Support non-interactive mode with CLI args or env vars
        import os

        db_password = args.db_password or os.environ.get("SPYGLASS_DB_PASSWORD")
        handle_database_setup_cli(
            "remote",
            db_host=args.db_host,
            db_port=args.db_port,
            db_user=args.db_user,
            db_password=db_password,
        )
    else:
        # Interactive prompt with retry logic
        handle_database_setup_interactive()

    # 5. Validation (runs in new environment, CAN import spyglass)
    if not args.skip_validation:
        validate_installation(args.env_name)

    # Success message
    print(f"\n{COLORS['green']}{'='*60}{COLORS['reset']}")
    print(f"{COLORS['green']}✓ Installation complete!{COLORS['reset']}")
    print(f"{COLORS['green']}{'='*60}{COLORS['reset']}\n")
    print("Next steps:")
    print(f"  1. Activate environment: conda activate {args.env_name}")
    print("  2. Start tutorial:       jupyter notebook notebooks/")
    print(
        "  3. View documentation:   https://lorenfranklab.github.io/spyglass/"
    )


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

    args = parser.parse_args()

    try:
        run_installation(args)
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Installation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
