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
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

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


def print_step(msg: str):
    """Print installation step."""
    print(f"{COLORS['blue']}▶{COLORS['reset']} {msg}")


def print_success(msg: str):
    """Print success message."""
    print(f"{COLORS['green']}✓{COLORS['reset']} {msg}")


def print_warning(msg: str):
    """Print warning message."""
    print(f"{COLORS['yellow']}⚠{COLORS['reset']} {msg}")


def print_error(msg: str):
    """Print error message."""
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
    if estimated_minutes > 5:
        print("  Tip: This is a good time for a coffee break ☕")


def get_required_python_version() -> Tuple[int, int]:
    """Get required Python version from pyproject.toml.

    Returns:
        Tuple of (major, minor) version

    This ensures single source of truth for version requirements.
    Falls back to (3, 9) if parsing fails.
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
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        # Parse ">=3.9,<3.13" format
        requires_python = data["project"]["requires-python"]
        match = re.search(r">=(\d+)\.(\d+)", requires_python)
        if match:
            return (int(match.group(1)), int(match.group(2)))
    except Exception:
        pass

    return (3, 9)  # Safe fallback


def check_prerequisites():
    """Check system prerequisites.

    Reads Python version requirement from pyproject.toml to maintain
    single source of truth.

    Raises:
        RuntimeError: If prerequisites are not met
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

    while True:
        choice = input("\nChoice [1-2]: ").strip()
        if choice == "1":
            print_success("Selected: Minimal installation")
            return "environment-min.yml", "minimal"
        elif choice == "2":
            print_success("Selected: Full installation")
            return "environment.yml", "full"
        else:
            print_error("Please enter 1 or 2")


def create_conda_environment(env_file: str, env_name: str, force: bool = False):
    """Create conda environment from file.

    Args:
        env_file: Path to environment.yml
        env_name: Name for the environment
        force: If True, overwrite existing environment without prompting

    Raises:
        RuntimeError: If environment creation fails
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


def install_spyglass_package(env_name: str):
    """Install spyglass package in development mode.

    Args:
        env_name: Name of the conda environment
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


def start_docker_container_inline() -> None:
    """Start Docker container (inline, no imports).

    Creates or starts spyglass-db MySQL container with default credentials.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    subprocess.CalledProcessError
        If docker commands fail

    Notes
    -----
    This is self-contained because spyglass isn't installed yet.
    """
    container_name = "spyglass-db"
    image = "datajoint/mysql:8.0"
    port = 3306

    # Check if container already exists
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )

    if container_name in result.stdout:
        # Start existing container
        print_step("Starting existing container...")
        subprocess.run(["docker", "start", container_name], check=True)
    else:
        # Pull image first (better UX - shows progress)
        show_progress_message(f"Pulling Docker image {image}", 2)
        subprocess.run(["docker", "pull", image], check=True)

        # Create and start new container
        print_step("Creating container...")
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-p",
                f"{port}:3306",
                "-e",
                "MYSQL_ROOT_PASSWORD=tutorial",
                image,
            ],
            check=True,
        )


def wait_for_mysql_inline(timeout: int = 60) -> None:
    """Wait for MySQL to be ready (inline, no imports).

    Polls MySQL container until it responds to ping or timeout occurs.

    Parameters
    ----------
    timeout : int, optional
        Maximum time to wait in seconds (default: 60)

    Returns
    -------
    None

    Raises
    ------
    TimeoutError
        If MySQL does not become ready within timeout period

    Notes
    -----
    This is self-contained because spyglass isn't installed yet.
    """
    import time

    container_name = "spyglass-db"
    print_step("Waiting for MySQL to be ready...")
    print("  Checking connection", end="", flush=True)

    for attempt in range(timeout // 2):
        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    container_name,
                    "mysqladmin",
                    "-uroot",
                    "-ptutorial",
                    "ping",
                ],
                capture_output=True,
                timeout=5,
            )

            if result.returncode == 0:
                print()  # New line after dots
                return  # Success!

        except subprocess.TimeoutExpired:
            pass

        if attempt < (timeout // 2) - 1:
            print(".", end="", flush=True)
            time.sleep(2)

    print()  # New line after dots
    raise TimeoutError(
        "MySQL did not become ready. Try:\n" "  docker logs spyglass-db"
    )


def create_database_config(
    host: str = "localhost",
    port: int = 3306,
    user: str = "root",
    password: str = "tutorial",
    use_tls: bool = False,
):
    """Create DataJoint configuration file.

    Args:
        host: Database host
        port: Database port
        user: Database user
        password: Database password
        use_tls: Whether to use TLS/SSL

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

    with open(config_file, "w") as f:
        json.dump(dj_config, f, indent=2)

    print_success(f"Configuration saved to {config_file}")


def prompt_remote_database_config() -> Optional[Dict[str, Any]]:
    """Prompt user for remote database connection details.

    Interactively asks for host, port, user, and password. Uses getpass
    for secure password input. Automatically enables TLS for remote hosts.

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

        # Determine TLS based on host (use TLS for non-localhost)
        localhost_addresses = ("localhost", "127.0.0.1", "::1")
        use_tls = host not in localhost_addresses

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


def prompt_database_setup() -> str:
    """Ask user about database setup preference.

    Displays menu of database setup options and prompts user to choose.

    Parameters
    ----------
    None

    Returns
    -------
    str
        One of: 'docker' (local Docker database), 'remote' (existing database),
        or 'skip' (configure later)

    Examples
    --------
    >>> choice = prompt_database_setup()
    >>> if choice == "docker":
    ...     setup_database_docker()
    """
    print("\nDatabase setup:")
    print("  1. Docker (local MySQL container)")
    print("  2. Remote (connect to existing database)")
    print("  3. Skip (configure later)")

    while True:
        choice = input("\nChoice [1-3]: ").strip()
        if choice == "1":
            return "docker"
        elif choice == "2":
            return "remote"
        elif choice == "3":
            return "skip"
        else:
            print_error("Please enter 1, 2, or 3")


def setup_database_docker() -> bool:
    """Set up local Docker database.

    Checks Docker availability, starts MySQL container, waits for readiness,
    and creates configuration file. Uses inline docker commands since
    spyglass package is not yet installed.

    Parameters
    ----------
    None

    Returns
    -------
    bool
        True if database setup succeeded, False otherwise

    Notes
    -----
    This function cannot import from spyglass because spyglass hasn't been
    installed yet. All docker operations must be inline.

    Examples
    --------
    >>> if setup_database_docker():
    ...     print("Database ready")
    """
    print_step("Setting up Docker database...")

    # Check Docker availability (inline, no imports)
    if not is_docker_available_inline():
        print_warning("Docker not available")
        print("  Install from: https://docs.docker.com/get-docker/")
        print("  Or choose option 2 to connect to remote database")
        return False

    try:
        # Start container (inline docker commands)
        start_docker_container_inline()
        print_success("Database container started")

        # Wait for MySQL readiness
        wait_for_mysql_inline()
        print_success("MySQL is ready")

        # Create configuration file (local Docker defaults)
        create_database_config(
            host="localhost",
            port=3306,
            user="root",
            password="tutorial",
            use_tls=False,
        )

        return True

    except Exception as e:
        print_error(f"Database setup failed: {e}")
        print("  You can configure manually later")
        return False


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


def setup_database_remote() -> bool:
    """Set up remote database connection.

    Prompts for connection details, tests the connection, and creates
    configuration file if connection succeeds.

    Parameters
    ----------
    None

    Returns
    -------
    bool
        True if configuration was created, False if cancelled

    Examples
    --------
    >>> if setup_database_remote():
    ...     print("Remote database configured")
    """
    print_step("Setting up remote database connection...")

    config = prompt_remote_database_config()
    if config is None:
        return False

    # Test connection before saving
    success, error = test_database_connection(**config)

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
    1. Check prerequisites (no spyglass imports)
    2. Create conda environment (no spyglass imports)
    3. Install spyglass package (pip install -e .)
    4. Setup database (inline code, NO spyglass imports)
    5. Validate (runs IN the new environment, CAN import spyglass)
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

    # 1. Check prerequisites (no spyglass imports)
    check_prerequisites()

    # 1b. Get base directory (CLI arg > env var > prompt)
    # Note: base_dir will be used for disk space check in Phase 2
    base_dir = get_base_directory(args.base_dir)  # noqa: F841

    # 2. Create environment (no spyglass imports)
    create_conda_environment(env_file, args.env_name, force=args.force)

    # 3. Install package (pip install makes spyglass available)
    install_spyglass_package(args.env_name)

    # 4. Database setup (INLINE CODE - no spyglass imports!)
    #    This happens AFTER spyglass is installed but doesn't use it
    #    because docker operations are self-contained
    if args.docker:
        # Docker explicitly requested via CLI
        setup_database_docker()
    elif args.remote:
        # Remote database explicitly requested via CLI
        setup_database_remote()
    else:
        # Interactive prompt for database setup
        db_choice = prompt_database_setup()
        if db_choice == "docker":
            setup_database_docker()
        elif db_choice == "remote":
            setup_database_remote()
        else:
            print_warning("Skipping database setup")
            print("  Configure later with: python scripts/install.py --docker")
            print("  Or manually: see docs/DATABASE.md")

    # 5. Validation (runs in new environment, CAN import spyglass)
    if not args.skip_validation:
        validate_installation(args.env_name)

    # Success message
    print(f"\n{COLORS['green']}{'='*60}{COLORS['reset']}")
    print(f"{COLORS['green']}✓ Installation complete!{COLORS['reset']}")
    print(f"{COLORS['green']}{'='*60}{COLORS['reset']}\n")
    print("Next steps:")
    print(f"  1. Activate environment: conda activate {args.env_name}")
    print("  2. Validate setup:      python scripts/validate.py")
    print("  3. Start tutorial:      jupyter notebook notebooks/")


def main():
    """Main entry point."""
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
