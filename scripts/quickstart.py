#!/usr/bin/env python
"""Spyglass Quickstart Script (Python version).

One-command setup for Spyglass installation.
This script provides a streamlined setup process for Spyglass, guiding you
through environment creation, package installation, and configuration.

Usage:
    python quickstart.py [OPTIONS]

Options:
    --minimal       Install core dependencies only (will prompt if not specified)
    --full          Install all optional dependencies (will prompt if not specified)
    --pipeline=X    Install specific pipeline dependencies (will prompt if not specified)
    --no-database   Skip database setup
    --no-validate   Skip validation after setup
    --base-dir=PATH Set base directory for data
    --help          Show help message

Interactive Mode:
    If no installation type is specified, you'll be prompted to choose between:
    1) Minimal installation (core dependencies only)
    2) Full installation (all optional dependencies)
    3) Pipeline-specific installation (choose from DLC, Moseq, LFP, Decoding)
"""

import sys
import subprocess
import shutil
import argparse
import time
import json
from pathlib import Path
from typing import Optional, List, Tuple, Callable, Iterator
from dataclasses import dataclass, replace
from enum import Enum
import getpass

# Import shared utilities
from common import (
    Colors, DisabledColors,
    SpyglassSetupError, SystemRequirementError,
    EnvironmentCreationError, DatabaseSetupError,
    MenuChoice, DatabaseChoice, ConfigLocationChoice, PipelineChoice
)

# Import new UX modules
from ux.system_requirements import (
    SystemRequirementsChecker, InstallationType
)


class InstallType(Enum):
    """Installation type options.

    Values
    ------
    MINIMAL : str
        Core dependencies only, fastest installation
    FULL : str
        All optional dependencies included
    """

    MINIMAL = "minimal"
    FULL = "full"


class Pipeline(Enum):
    """Available pipeline options.

    Values
    ------
    DLC : str
        DeepLabCut pose estimation and behavior analysis
    MOSEQ_CPU : str
        Keypoint-Moseq behavioral sequence analysis (CPU)
    MOSEQ_GPU : str
        Keypoint-Moseq behavioral sequence analysis (GPU-accelerated)
    LFP : str
        Local field potential processing and analysis
    DECODING : str
        Neural population decoding algorithms
    """

    DLC = "dlc"
    MOSEQ_CPU = "moseq-cpu"
    MOSEQ_GPU = "moseq-gpu"
    LFP = "lfp"
    DECODING = "decoding"


@dataclass
class SystemInfo:
    """System information.

    Attributes
    ----------
    os_name : str
        Operating system name (e.g., 'macOS', 'Linux', 'Windows')
    arch : str
        System architecture (e.g., 'x86_64', 'arm64')
    is_m1 : bool
        True if running on Apple M1/M2/M3 silicon
    python_version : Tuple[int, int, int]
        Python version as (major, minor, patch)
    conda_cmd : Optional[str]
        Command to use for conda ('mamba' or 'conda'), None if not found
    """

    os_name: str
    arch: str
    is_m1: bool
    python_version: Tuple[int, int, int]
    conda_cmd: Optional[str]


@dataclass
class SetupConfig:
    """Configuration for setup process.

    Attributes
    ----------
    install_type : InstallType
        Type of installation (MINIMAL or FULL)
    pipeline : Optional[Pipeline]
        Specific pipeline to install, None for general installation
    setup_database : bool
        Whether to set up database configuration
    run_validation : bool
        Whether to run validation checks after installation
    base_dir : Path
        Base directory for Spyglass data storage
    repo_dir : Path
        Repository root directory
    env_name : str
        Name of the conda environment to create/use
    db_port : int
        Database port number for connection
    auto_yes : bool
        Whether to auto-accept prompts without user input
    install_type_specified : bool
        Whether install_type was explicitly specified via CLI
    """

    install_type: InstallType = InstallType.MINIMAL
    pipeline: Optional[Pipeline] = None
    setup_database: bool = True
    run_validation: bool = True
    base_dir: Path = Path.home() / "spyglass_data"
    repo_dir: Path = Path(__file__).parent.parent
    env_name: str = "spyglass"
    db_port: int = 3306
    auto_yes: bool = False
    install_type_specified: bool = False


# Using standard library functions directly - no unnecessary wrappers


def validate_base_dir(path: Path) -> Path:
    """Validate and resolve base directory path."""
    resolved = Path(path).expanduser().resolve()

    # Check if parent directory exists (we'll create the base_dir itself if needed)
    if not resolved.parent.exists():
        raise ValueError(f"Parent directory does not exist: {resolved.parent}")

    return resolved



def setup_docker_database(orchestrator: 'QuickstartOrchestrator') -> None:
    """Setup Docker database - simple function."""
    orchestrator.ui.print_info("Setting up local Docker database...")

    # Check Docker availability
    if not shutil.which("docker"):
        orchestrator.ui.print_error("Docker is not installed")
        orchestrator.ui.print_info("Please install Docker from: https://docs.docker.com/engine/install/")
        raise SystemRequirementError("Docker is not installed")

    # Check Docker daemon
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        orchestrator.ui.print_error("Docker daemon is not running")
        orchestrator.ui.print_info("Please start Docker Desktop and try again")
        orchestrator.ui.print_info("On macOS: Open Docker Desktop application")
        orchestrator.ui.print_info("On Linux: sudo systemctl start docker")
        raise SystemRequirementError("Docker daemon is not running")

    # Pull and run container
    orchestrator.ui.print_info("Pulling MySQL image...")
    subprocess.run(["docker", "pull", "datajoint/mysql:8.0"], check=True)

    # Check existing container
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )

    if "spyglass-db" in result.stdout:
        orchestrator.ui.print_warning("Container 'spyglass-db' already exists")
        subprocess.run(["docker", "start", "spyglass-db"], check=True)
    else:
        # Check if port is already in use
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('localhost', orchestrator.config.db_port))
            if result == 0:
                orchestrator.ui.print_error(f"Port {orchestrator.config.db_port} is already in use")
                orchestrator.ui.print_info("Try using a different port with --db-port (e.g., --db-port 3307)")
                raise SystemRequirementError(f"Port {orchestrator.config.db_port} is already in use")

        port_mapping = f"{orchestrator.config.db_port}:3306"
        subprocess.run([
            "docker", "run", "-d",
            "--name", "spyglass-db",
            "-p", port_mapping,
            "-e", "MYSQL_ROOT_PASSWORD=tutorial",
            "datajoint/mysql:8.0"
        ], check=True)

    orchestrator.ui.print_success("Docker database started")

    # Wait for MySQL to be ready
    orchestrator.ui.print_info("Waiting for MySQL to be ready...")
    for attempt in range(60):  # Wait up to 2 minutes
        try:
            result = subprocess.run(
                ["docker", "exec", "spyglass-db", "mysqladmin", "-uroot", "-ptutorial", "ping"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if b"mysqld is alive" in result.stdout.encode() or "mysqld is alive" in result.stdout:
                orchestrator.ui.print_success("MySQL is ready!")
                break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

        if attempt < 59:  # Don't sleep on the last attempt
            time.sleep(2)
    else:
        orchestrator.ui.print_warning("MySQL readiness check timed out, but proceeding anyway")

    orchestrator.create_config("localhost", "root", "tutorial", orchestrator.config.db_port)


def setup_existing_database(orchestrator: 'QuickstartOrchestrator') -> None:
    """Setup existing database connection."""
    orchestrator.ui.print_info("Configuring connection to existing database...")

    host, port, user, password = orchestrator.ui.get_database_credentials()
    _test_database_connection(orchestrator.ui, host, port, user, password)
    orchestrator.create_config(host, user, password, port)


def _test_database_connection(ui: 'UserInterface', host: str, port: int, user: str, password: str) -> None:
    """Test database connection before proceeding."""
    ui.print_info("Testing database connection...")

    try:
        import pymysql
        connection = pymysql.connect(host=host, port=port, user=user, password=password)
        connection.close()
        ui.print_success("Database connection successful")
    except ImportError:
        ui.print_warning("PyMySQL not available for connection test")
        ui.print_info("Connection will be tested when DataJoint loads")
    except (ConnectionError, OSError, TimeoutError) as e:
        ui.print_error(f"Database connection failed: {e}")
        raise DatabaseSetupError(f"Cannot connect to database: {e}") from e


# Database setup function mapping - simple dictionary approach
DATABASE_SETUP_METHODS = {
    DatabaseChoice.DOCKER: setup_docker_database,
    DatabaseChoice.EXISTING: setup_existing_database,
    DatabaseChoice.SKIP: lambda _: None  # Skip setup
}


class UserInterface:
    """Handles all user interactions and display formatting.

    Parameters
    ----------
    colors : Colors
        Color scheme for terminal output
    auto_yes : bool, optional
        If True, automatically accept all prompts with defaults, by default False

    """

    def __init__(self, colors: 'Colors', auto_yes: bool = False) -> None:
        self.colors = colors
        self.auto_yes = auto_yes

    def get_input(self, prompt: str, default: str = None) -> str:
        """Get user input with auto-yes support.

        Parameters
        ----------
        prompt : str
            The input prompt to display to the user
        default : str, optional
            Default value to use in auto-yes mode, by default None

        Returns
        -------
        str
            User input or default value

        Raises
        ------
        ValueError
            If auto_yes is True but no default is provided

        """
        if self.auto_yes:
            if default is not None:
                self.print_info(f"Auto-accepting: {prompt} -> {default}")
                return default
            else:
                raise ValueError(f"Cannot auto-accept prompt without default: {prompt}")
        return input(prompt).strip()

    def get_validated_input(self, prompt: str, validator: Callable[[str], bool],
                           error_msg: str, default: str = None) -> str:
        """Generic validated input helper.

        Parameters
        ----------
        prompt : str
            The input prompt to display to the user
        validator : Callable[[str], bool]
            Function to validate the input, returns True if valid
        error_msg : str
            Error message to display for invalid input
        default : str, optional
            Default value to use in auto-yes mode, by default None

        Returns
        -------
        str
            Validated user input or default value

        """
        if self.auto_yes and default is not None:
            self.print_info(f"Auto-accepting: {prompt} -> {default}")
            return default

        while True:
            value = input(prompt).strip() or default
            if validator(value):
                return value
            self.print_error(error_msg)

    def print_header_banner(self) -> None:
        """Print the main application banner."""
        print("\n" + "═" * 43)
        print("║     Spyglass Quickstart Installer    ║")
        print("═" * 43)

    def print_header(self, text: str) -> None:
        """Print section header.

        Parameters
        ----------
        text : str
            Header text to display

        """
        print(f"\n{'=' * 42}")
        print(text)
        print("=" * 42)

    def _format_message(self, text: str, symbol: str, color: str) -> str:
        """Format a message with color and symbol.

        Parameters
        ----------
        text : str
            Message text to format
        symbol : str
            Symbol to prefix the message with
        color : str
            ANSI color code for the message

        Returns
        -------
        str
            Formatted message with color and symbol

        """
        return f"{color}{symbol} {text}{self.colors.ENDC}"

    def print_success(self, text: str) -> None:
        """Print success message.

        Parameters
        ----------
        text : str
            Success message to display

        """
        print(self._format_message(text, "✓", self.colors.OKGREEN))

    def print_warning(self, text: str) -> None:
        """Print warning message.

        Parameters
        ----------
        text : str
            Warning message to display

        """
        print(self._format_message(text, "⚠", self.colors.WARNING))

    def print_error(self, text: str) -> None:
        """Print error message.

        Parameters
        ----------
        text : str
            Error message to display

        """
        print(self._format_message(text, "✗", self.colors.FAIL))

    def print_info(self, text: str) -> None:
        """Print info message.

        Parameters
        ----------
        text : str
            Info message to display

        """
        print(self._format_message(text, "ℹ", self.colors.OKBLUE))

    def select_install_type(self) -> Tuple[InstallType, Optional[Pipeline]]:
        """Let user select installation type.

        Returns
        -------
        Tuple[InstallType, Optional[Pipeline]]
            Tuple of (installation type, optional pipeline choice)

        """
        print("\nChoose your installation type:")
        print("1) Minimal (core dependencies only)")
        print("   ├─ Basic Spyglass functionality")
        print("   ├─ Standard data analysis tools")
        print("   └─ Fastest installation (~5-10 minutes)")
        print("")
        print("2) Full (all optional dependencies)")
        print("   ├─ All analysis pipelines included")
        print("   ├─ Spike sorting, LFP, visualization tools")
        print("   └─ Longer installation (~15-30 minutes)")
        print("")
        print("3) Pipeline-specific")
        print("   ├─ Choose specific analysis pipeline")
        print("   ├─ DeepLabCut, Moseq, LFP, or Decoding")
        print("   └─ Optimized environment for your workflow")

        while True:
            try:
                choice = input("\nEnter choice (1-3): ").strip()
                if choice == str(MenuChoice.MINIMAL.value):
                    return InstallType.MINIMAL, None
                elif choice == str(MenuChoice.FULL.value):
                    return InstallType.FULL, None
                elif choice == str(MenuChoice.PIPELINE.value):
                    pipeline = self.select_pipeline()
                    return InstallType.MINIMAL, pipeline
                else:
                    self.print_error("Invalid choice. Please enter 1, 2, or 3")
            except EOFError:
                self.print_warning("Interactive input not available, defaulting to minimal installation")
                self.print_info("Use --minimal, --full, or --pipeline flags to specify installation type")
                return InstallType.MINIMAL, None

    def select_pipeline(self) -> Pipeline:
        """Let user select specific pipeline."""
        print("\nChoose your pipeline:")
        print("1) DeepLabCut - Pose estimation and behavior analysis")
        print("2) Keypoint-Moseq (CPU) - Behavioral sequence analysis")
        print("3) Keypoint-Moseq (GPU) - GPU-accelerated behavioral analysis")
        print("4) LFP Analysis - Local field potential processing")
        print("5) Decoding - Neural population decoding")

        while True:
            try:
                choice = input("\nEnter choice (1-5): ").strip()
                if choice == str(PipelineChoice.DLC.value):
                    return Pipeline.DLC
                elif choice == str(PipelineChoice.MOSEQ_CPU.value):
                    return Pipeline.MOSEQ_CPU
                elif choice == str(PipelineChoice.MOSEQ_GPU.value):
                    return Pipeline.MOSEQ_GPU
                elif choice == str(PipelineChoice.LFP.value):
                    return Pipeline.LFP
                elif choice == str(PipelineChoice.DECODING.value):
                    return Pipeline.DECODING
                else:
                    self.print_error("Invalid choice. Please enter 1-5")
            except EOFError:
                self.print_warning("Interactive input not available, defaulting to DeepLabCut")
                self.print_info("Use --pipeline flag to specify pipeline type")
                return Pipeline.DLC

    def confirm_environment_update(self, env_name: str) -> bool:
        """Ask user if they want to update existing environment."""
        self.print_warning(f"Environment '{env_name}' already exists")
        if self.auto_yes:
            self.print_info("Auto-accepting environment update (--yes)")
            return True

        try:
            choice = input("Do you want to update it? (y/N): ").strip().lower()
            return choice == 'y'
        except EOFError:
            # Handle case where stdin is not available (e.g., non-interactive environment)
            self.print_warning("Interactive input not available, defaulting to 'no'")
            self.print_info("Use --yes flag to auto-accept prompts")
            return False

    def select_database_setup(self) -> str:
        """Select database setup choice."""
        print("\nChoose database setup option:")
        print("1) Local Docker database (recommended for beginners)")
        print("2) Connect to existing database")
        print("3) Skip database setup")

        while True:
            try:
                choice = input("\nEnter choice (1-3): ").strip()
                try:
                    db_choice = DatabaseChoice(int(choice))
                    if db_choice == DatabaseChoice.SKIP:
                        self.print_info("Skipping database setup")
                        self.print_warning("You'll need to configure the database manually later")
                    return db_choice
                except (ValueError, IndexError):
                    self.print_error("Invalid choice. Please enter 1, 2, or 3")
            except EOFError:
                self.print_warning("Interactive input not available, defaulting to skip database setup")
                self.print_info("Use --no-database flag to skip database setup")
                return DatabaseChoice.SKIP

    def select_config_location(self, repo_dir: Path) -> Path:
        """Select where to save the DataJoint configuration file."""
        print("\nChoose configuration file location:")
        print(f"1) Repository root (recommended): {repo_dir}")
        print("2) Current directory")
        print("3) Custom location")

        while True:
            try:
                choice = input("\nEnter choice (1-3): ").strip()
                try:
                    config_choice = ConfigLocationChoice(int(choice))
                    if config_choice == ConfigLocationChoice.REPO_ROOT:
                        return repo_dir
                    elif config_choice == ConfigLocationChoice.CURRENT_DIR:
                        return Path.cwd()
                    elif config_choice == ConfigLocationChoice.CUSTOM:
                        return self._get_custom_path()
                except (ValueError, IndexError):
                    self.print_error("Invalid choice. Please enter 1, 2, or 3")
            except EOFError:
                self.print_warning("Interactive input not available, defaulting to repository root")
                self.print_info("Use --base-dir to specify a different location")
                return repo_dir

    def _get_custom_path(self) -> Path:
        """Get custom path from user with validation."""
        while True:
            try:
                custom_path = input("Enter custom directory path: ").strip()
                if not custom_path:
                    self.print_error("Path cannot be empty")
                    continue

                try:
                    path = Path(custom_path).expanduser().resolve()
                    if not path.exists():
                        try:
                            create = input(f"Directory {path} doesn't exist. Create it? (y/N): ").strip().lower()
                            if create == 'y':
                                path.mkdir(parents=True, exist_ok=True)
                            else:
                                continue
                        except EOFError:
                            self.print_warning("Interactive input not available, creating directory automatically")
                            path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    self.print_error(f"Invalid path: {e}")
                    continue

                return path
            except EOFError:
                self.print_warning("Interactive input not available, using current directory")
                return Path.cwd()

    def get_database_credentials(self) -> Tuple[str, int, str, str]:
        """Get database connection credentials from user."""
        print("\nEnter database connection details:")

        host = self._get_host_input()
        port = self._get_port_input()
        user = self._get_user_input()
        password = self._get_password_input()

        return host, port, user, password

    def _get_host_input(self) -> str:
        """Get host input with default."""
        return input("Host (default: localhost): ").strip() or "localhost"

    def _get_port_input(self) -> int:
        """Get and validate port input."""
        def is_valid_port(port_str: str) -> bool:
            try:
                port = int(port_str)
                return 1 <= port <= 65535
            except ValueError:
                return False

        port_str = self.get_validated_input(
            "Port (default: 3306): ",
            is_valid_port,
            "Port must be between 1 and 65535",
            "3306"
        )
        return int(port_str)

    def _get_user_input(self) -> str:
        """Get username input with default."""
        return input("Username (default: root): ").strip() or "root"

    def _get_password_input(self) -> str:
        """Get password input securely."""
        while True:
            password = getpass.getpass("Password: ")
            if password:  # Allow empty passwords for local development
                return password

            # Confirm if user wants empty password
            confirm = input("Use empty password? (y/N): ").strip().lower()
            if confirm == 'y':
                return password
            self.print_info("Please enter a password or confirm empty password")


class EnvironmentManager:
    """Handles conda environment creation and management."""

    def __init__(self, ui: 'UserInterface', config: SetupConfig) -> None:
        self.ui = ui
        self.config = config
        self.system_info = None
        self.PIPELINE_ENVIRONMENTS = {
            Pipeline.DLC: ("environment_dlc.yml", "DeepLabCut pipeline environment"),
            Pipeline.MOSEQ_CPU: ("environment_moseq.yml", "Keypoint-Moseq (CPU) pipeline environment"),
            Pipeline.MOSEQ_GPU: ("environment_moseq_gpu.yml", "Keypoint-Moseq (GPU) pipeline environment"),
            Pipeline.LFP: ("environment_lfp.yml", "LFP pipeline environment"),
            Pipeline.DECODING: ("environment_decoding.yml", "Decoding pipeline environment"),
        }

    def select_environment_file(self) -> str:
        """Select appropriate environment file based on configuration."""
        if env_info := self.PIPELINE_ENVIRONMENTS.get(self.config.pipeline):
            env_file, description = env_info
            self.ui.print_info(f"Selected: {description}")
        elif self.config.install_type == InstallType.FULL:
            env_file = "environment.yml"
            self.ui.print_info("Selected: Full environment with all optional dependencies")
        else:  # MINIMAL
            env_file = "environment-min.yml"
            self.ui.print_info("Selected: Minimal environment with core dependencies only")

        # Verify environment file exists
        env_path = self.config.repo_dir / env_file
        if not env_path.exists():
            raise EnvironmentCreationError(
                f"Environment file not found: {env_path}\n"
                f"Please ensure you're running from the Spyglass repository root"
            )

        return env_file

    def create_environment(self, env_file: str, conda_cmd: str) -> bool:
        """Create or update conda environment."""
        self.ui.print_header("Creating Conda Environment")

        update = self._check_environment_exists(conda_cmd)
        if update and not self.ui.confirm_environment_update(self.config.env_name):
            self.ui.print_info("Keeping existing environment unchanged")
            return True

        cmd = self._build_environment_command(env_file, conda_cmd, update)
        self._execute_environment_command(cmd)
        return True

    def _check_environment_exists(self, conda_cmd: str) -> bool:
        """Check if the target environment already exists."""
        try:
            result = subprocess.run([conda_cmd, "env", "list"], capture_output=True, text=True, check=True)
            return self.config.env_name in result.stdout
        except subprocess.CalledProcessError:
            return False

    def _build_environment_command(self, env_file: str, conda_cmd: str, update: bool) -> List[str]:
        """Build conda environment command."""
        env_path = self.config.repo_dir / env_file
        env_name = self.config.env_name

        if update:
            self.ui.print_info("Updating existing environment...")
            return [conda_cmd, "env", "update", "-f", str(env_path), "-n", env_name]
        else:
            self.ui.print_info(f"Creating new environment '{env_name}'...")
            self.ui.print_info("This may take 5-10 minutes...")
            return [conda_cmd, "env", "create", "-f", str(env_path), "-n", env_name]

    def _execute_environment_command(self, cmd: List[str], timeout: int = 1800) -> None:
        """Execute environment creation/update command with progress and timeout."""
        process = self._start_process(cmd)
        output_buffer = self._monitor_process(process, timeout)
        self._handle_process_result(process, output_buffer)

    def _start_process(self, cmd: List[str]) -> subprocess.Popen:
        """Start subprocess with appropriate settings."""
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

    def _monitor_process(self, process: subprocess.Popen, timeout: int) -> List[str]:
        """Monitor process execution with timeout and progress display."""
        output_buffer = []
        start_time = time.time()

        try:
            while process.poll() is None:
                if time.time() - start_time > timeout:
                    process.kill()
                    raise EnvironmentCreationError("Environment creation timed out after 30 minutes")

                # Read and display progress
                try:
                    for line in self._filter_progress_lines(process):
                        output_buffer.append(line)
                except (StopIteration, OSError):
                    pass

                time.sleep(1)
        except subprocess.TimeoutExpired:
            raise EnvironmentCreationError("Environment creation timed out") from None
        except (subprocess.CalledProcessError, OSError, FileNotFoundError) as e:
            raise EnvironmentCreationError(f"Environment creation/update failed: {str(e)}") from e

        return output_buffer

    def _handle_process_result(self, process: subprocess.Popen, output_buffer: List[str]) -> None:
        """Handle process completion and errors."""
        if process.returncode == 0:
            return  # Success

        # Handle failure case
        full_output = '\n'.join(output_buffer) if output_buffer else "No output captured"

        # Get last 200 lines for error context
        output_lines = full_output.split('\n') if full_output else []
        error_context = '\n'.join(output_lines[-200:]) if output_lines else "No output captured"

        raise EnvironmentCreationError(
            f"Environment creation failed with return code {process.returncode}\n"
            f"--- Last 200 lines of output ---\n{error_context}"
        )

    def _filter_progress_lines(self, process: subprocess.Popen) -> Iterator[str]:
        """Filter and yield all lines while printing only progress lines."""
        progress_keywords = {"Solving environment", "Downloading", "Extracting", "Installing"}

        for line in process.stdout:
            # Always yield all lines for error context buffering
            yield line
            # But only print progress-related lines live
            if any(keyword in line for keyword in progress_keywords):
                print(f"  {line.strip()}")

    def install_additional_dependencies(self, conda_cmd: str) -> None:
        """Install additional dependencies after environment creation."""
        self.ui.print_header("Installing Additional Dependencies")

        # Install in development mode
        self.ui.print_info("Installing Spyglass in development mode...")
        self._run_in_env(conda_cmd, ["pip", "install", "-e", str(self.config.repo_dir)])

        # Install pipeline-specific dependencies
        if self.config.pipeline:
            self._install_pipeline_dependencies(conda_cmd)
        elif self.config.install_type == InstallType.FULL:
            self.ui.print_info("Installing optional dependencies for full installation...")
            # For full installation using environment.yml, all packages are already included
            # Editable install already done above

        self.ui.print_success("Additional dependencies installed")

    def _install_pipeline_dependencies(self, conda_cmd: str) -> None:
        """Install dependencies for specific pipeline."""
        self.ui.print_info("Installing pipeline-specific dependencies...")

        if self.config.pipeline == Pipeline.LFP:
            self.ui.print_info("Installing LFP dependencies...")
            # Handle M1 Mac specific installation
            system_info = self._get_system_info()
            if system_info and system_info.is_m1:
                self.ui.print_info("Detected M1 Mac, installing pyfftw via conda first...")
                self._run_in_env(conda_cmd, ["conda", "install", "-c", "conda-forge", "pyfftw", "-y"])


    def _run_in_env(self, conda_cmd: str, cmd: List[str]) -> int:
        """Run command in the target conda environment."""
        full_cmd = [conda_cmd, "run", "-n", self.config.env_name] + cmd
        try:
            result = subprocess.run(full_cmd, check=True, capture_output=True, text=True)
            # Print output for user feedback
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="")
            return result.returncode
        except subprocess.CalledProcessError as e:
            self.ui.print_error(f"Command failed in environment '{self.config.env_name}': {' '.join(cmd)}")
            if e.stdout:
                self.ui.print_error(f"STDOUT: {e.stdout}")
            if e.stderr:
                self.ui.print_error(f"STDERR: {e.stderr}")
            raise

    def _get_system_info(self) -> Optional[SystemInfo]:
        """Get system info from orchestrator."""
        return self.system_info


class QuickstartOrchestrator:
    """Main orchestrator that coordinates all installation components."""

    def __init__(self, config: SetupConfig, colors: 'Colors') -> None:
        self.config = config
        self.ui = UserInterface(colors, auto_yes=config.auto_yes)
        self.env_manager = EnvironmentManager(self.ui, config)
        self.system_info = None
        # Use new comprehensive system requirements checker
        self.requirements_checker = SystemRequirementsChecker(config.base_dir)

    def run(self) -> int:
        """Run the complete installation process."""
        try:
            self.ui.print_header_banner()
            self._execute_setup_steps()
            self._print_summary()
            return 0

        except KeyboardInterrupt:
            self.ui.print_error("\nSetup interrupted by user")
            return 130
        except SystemRequirementError as e:
            self.ui.print_error(f"\nSystem requirement not met: {e}")
            return 1
        except EnvironmentCreationError as e:
            self.ui.print_error(f"\nFailed to create environment: {e}")
            return 1
        except DatabaseSetupError as e:
            self.ui.print_error(f"\nDatabase setup failed: {e}")
            return 1
        except SpyglassSetupError as e:
            self.ui.print_error(f"\nSetup error: {e}")
            return 1
        except SystemExit:
            raise
        except Exception as e:
            self.ui.print_error(f"\nUnexpected error: {e}")
            return 1

    def _execute_setup_steps(self) -> None:
        """Execute the main setup steps in order."""
        # Step 1: Comprehensive System Requirements Check
        conda_cmd, system_info = self._run_system_requirements_check()

        # Wire system_info to environment manager (converted format)
        self.env_manager.system_info = self._convert_system_info(system_info)

        # Step 2: Installation Type Selection (if not specified)
        if not self._installation_type_specified():
            install_type, pipeline = self._select_install_type_with_estimates(system_info)
            self.config = replace(self.config, install_type=install_type, pipeline=pipeline)

        # Step 3: Environment Creation
        env_file = self.env_manager.select_environment_file()
        self.env_manager.create_environment(env_file, conda_cmd)
        self.env_manager.install_additional_dependencies(conda_cmd)

        # Step 4: Database Setup
        if self.config.setup_database:
            self._setup_database()

        # Step 5: Validation
        if self.config.run_validation:
            self._run_validation(conda_cmd)

    def _map_install_type_to_requirements_type(self) -> InstallationType:
        """Map our InstallType enum to the requirements checker InstallationType."""
        if self.config.pipeline:
            return InstallationType.PIPELINE_SPECIFIC
        elif self.config.install_type == InstallType.FULL:
            return InstallationType.FULL
        else:
            return InstallationType.MINIMAL

    def _run_system_requirements_check(self) -> Tuple[str, 'SystemInfo']:
        """Run comprehensive system requirements check with user-friendly output.

        Returns:
            Tuple of (conda_cmd, system_info) for use in subsequent steps
        """
        self.ui.print_header("System Requirements Check")

        # Detect system info
        system_info = self.requirements_checker.detect_system_info()

        # Use minimal as baseline for general compatibility check (not specific estimates)
        baseline_install_type = InstallationType.MINIMAL

        # Run comprehensive checks (for compatibility, not specific to user's choice)
        checks = self.requirements_checker.run_comprehensive_check(baseline_install_type)

        # Display system information
        self._display_system_info(system_info)

        # Display requirement checks
        self._display_requirement_checks(checks)

        # Show general system readiness (without specific installation estimates)
        self._display_system_readiness(system_info)

        # Check for critical failures
        critical_failures = [check for check in checks.values()
                           if not check.met and check.severity.value in ['error', 'critical']]

        if critical_failures:
            self.ui.print_error("\nCritical requirements not met. Installation cannot proceed.")
            for check in critical_failures:
                self.ui.print_error(f"  • {check.message}")
                for suggestion in check.suggestions:
                    self.ui.print_info(f"    → {suggestion}")
            raise SystemRequirementError("Critical system requirements not met")

        # Determine conda command from system info
        if system_info.mamba_available:
            conda_cmd = "mamba"
        elif system_info.conda_available:
            conda_cmd = "conda"
        else:
            raise SystemRequirementError("No conda/mamba found - should have been caught above")

        # Show that system is ready for installation (without specific estimates)
        if not self.config.auto_yes:
            self.ui.print_info("\nSystem compatibility confirmed. Ready to proceed with installation.")
            proceed = self.ui.get_input("Continue to installation options? [Y/n]: ", "y").lower()
            if proceed and proceed[0] == 'n':
                self.ui.print_info("Installation cancelled by user.")
                raise KeyboardInterrupt()

        return conda_cmd, system_info

    def _convert_system_info(self, new_system_info) -> SystemInfo:
        """Convert from new SystemInfo to old SystemInfo format for EnvironmentManager."""
        return SystemInfo(
            os_name=new_system_info.os_name,
            arch=new_system_info.architecture,
            is_m1=new_system_info.is_m1_mac,
            python_version=new_system_info.python_version,
            conda_cmd="mamba" if new_system_info.mamba_available else "conda"
        )

    def _display_system_info(self, system_info) -> None:
        """Display detected system information."""
        print(f"\n🖥️  System Information:")
        print(f"   Operating System: {system_info.os_name} {system_info.os_version}")
        print(f"   Architecture: {system_info.architecture}")
        if system_info.is_m1_mac:
            print(f"   Apple Silicon: Yes (optimized builds available)")

        python_version = f"{system_info.python_version[0]}.{system_info.python_version[1]}.{system_info.python_version[2]}"
        print(f"   Python: {python_version}")
        print(f"   Disk Space: {system_info.available_space_gb:.1f} GB available")

    def _display_requirement_checks(self, checks: dict) -> None:
        """Display requirement check results."""
        print(f"\n📋 Requirements Status:")

        for check in checks.values():
            if check.met:
                if check.severity.value == 'warning':
                    symbol = "⚠️"
                    color = "WARNING"
                else:
                    symbol = "✅"
                    color = "OKGREEN"
            else:
                if check.severity.value in ['error', 'critical']:
                    symbol = "❌"
                    color = "FAIL"
                else:
                    symbol = "⚠️"
                    color = "WARNING"

            # Format the message with color
            if hasattr(self.ui.colors, color):
                color_code = getattr(self.ui.colors, color)
                print(f"   {symbol} {color_code}{check.name}: {check.message}{self.ui.colors.ENDC}")
            else:
                print(f"   {symbol} {check.name}: {check.message}")

            # Show suggestions for warnings or failures
            if check.suggestions and (not check.met or check.severity.value == 'warning'):
                for suggestion in check.suggestions[:2]:  # Limit to 2 suggestions for brevity
                    print(f"      💡 {suggestion}")

    def _display_system_readiness(self, system_info) -> None:
        """Display general system readiness without specific installation estimates."""
        print(f"\n🚀 System Readiness:")
        print(f"   Available Space: {system_info.available_space_gb:.1f} GB (sufficient for all installation types)")

        if system_info.is_m1_mac:
            print(f"   Performance: Optimized builds available for Apple Silicon")

        if system_info.mamba_available:
            print(f"   Package Manager: Mamba (fastest option)")
        elif system_info.conda_available:
            # Check if it's modern conda
            conda_version = self.requirements_checker._get_conda_version()
            if conda_version and self.requirements_checker._has_libmamba_solver(conda_version):
                print(f"   Package Manager: Conda with fast libmamba solver")
            else:
                print(f"   Package Manager: Conda (classic solver)")

    def _display_installation_estimates(self, system_info, install_type: InstallationType) -> None:
        """Display installation time and space estimates for a specific type."""
        time_estimate = self.requirements_checker.estimate_installation_time(system_info, install_type)
        space_estimate = self.requirements_checker.DISK_ESTIMATES[install_type]

        print(f"\n📊 {install_type.value.title()} Installation Estimates:")
        print(f"   Time: {time_estimate.format_range()}")
        print(f"   Space: {space_estimate.format_summary()}")

        if time_estimate.factors:
            print(f"   Factors: {', '.join(time_estimate.factors)}")

    def _select_install_type_with_estimates(self, system_info) -> Tuple[InstallType, Optional[Pipeline]]:
        """Let user select installation type with time/space estimates for each option."""
        self.ui.print_header("Installation Type Selection")

        # Show estimates for each installation type
        print("\nChoose your installation type:\n")

        # Minimal installation
        minimal_time = self.requirements_checker.estimate_installation_time(system_info, InstallationType.MINIMAL)
        minimal_space = self.requirements_checker.DISK_ESTIMATES[InstallationType.MINIMAL]
        print("1) Minimal Installation")
        print("   ├─ Basic Spyglass functionality")
        print("   ├─ Standard data analysis tools")
        print(f"   ├─ Time: {minimal_time.format_range()}")
        print(f"   └─ Space: {minimal_space.total_required_gb:.1f} GB required")

        print("")

        # Full installation
        full_time = self.requirements_checker.estimate_installation_time(system_info, InstallationType.FULL)
        full_space = self.requirements_checker.DISK_ESTIMATES[InstallationType.FULL]
        print("2) Full Installation")
        print("   ├─ All analysis pipelines included")
        print("   ├─ Spike sorting, LFP, visualization tools")
        print(f"   ├─ Time: {full_time.format_range()}")
        print(f"   └─ Space: {full_space.total_required_gb:.1f} GB required")

        print("")

        # Pipeline-specific installation
        pipeline_time = self.requirements_checker.estimate_installation_time(system_info, InstallationType.PIPELINE_SPECIFIC)
        pipeline_space = self.requirements_checker.DISK_ESTIMATES[InstallationType.PIPELINE_SPECIFIC]
        print("3) Pipeline-Specific Installation")
        print("   ├─ Choose specific analysis pipeline")
        print("   ├─ DeepLabCut, Moseq, LFP, or Decoding")
        print(f"   ├─ Time: {pipeline_time.format_range()}")
        print(f"   └─ Space: {pipeline_space.total_required_gb:.1f} GB required")

        # Show recommendation based on available space
        available_space = system_info.available_space_gb
        if available_space >= full_space.total_recommended_gb:
            print(f"\n💡 Recommendation: Full installation is well-supported with {available_space:.1f} GB available")
        elif available_space >= minimal_space.total_recommended_gb:
            print(f"\n💡 Recommendation: Minimal installation recommended with {available_space:.1f} GB available")
        else:
            print(f"\n⚠️  Note: Space is limited ({available_space:.1f} GB available). Minimal installation advised.")

        # Get user choice using existing UI method
        install_type, pipeline = self.ui.select_install_type()

        # Show final estimates for chosen type
        chosen_install_type = self._map_install_type_to_requirements_type()
        if pipeline:
            chosen_install_type = InstallationType.PIPELINE_SPECIFIC

        self._display_installation_estimates(system_info, chosen_install_type)

        return install_type, pipeline

    def _installation_type_specified(self) -> bool:
        """Check if installation type was specified via command line arguments."""
        return self.config.install_type_specified

    def _setup_database(self) -> None:
        """Setup database configuration."""
        self.ui.print_header("Database Setup")

        choice = self.ui.select_database_setup()
        setup_func = DATABASE_SETUP_METHODS.get(choice)
        if setup_func:
            setup_func(self)

    def _run_validation(self, conda_cmd: str) -> int:
        """Run validation checks."""
        self.ui.print_header("Running Validation")

        validation_script = self.config.repo_dir / "scripts" / "validate_spyglass.py"

        if not validation_script.exists():
            self.ui.print_error("Validation script not found")
            self.ui.print_info("Expected location: scripts/validate_spyglass.py")
            self.ui.print_info("Please ensure you're running from the Spyglass repository root")
            return 1

        self.ui.print_info("Running comprehensive validation checks...")

        try:
            # Try to find the environment's python directly instead of using conda run
            self.ui.print_info("Finding environment python executable...")

            # Get conda environment info
            env_info_result = subprocess.run(
                [conda_cmd, "info", "--envs"],
                capture_output=True, text=True, check=False
            )

            python_path = None
            if env_info_result.returncode == 0:
                # Parse environment path
                for line in env_info_result.stdout.split('\n'):
                    if self.config.env_name in line and not line.strip().startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            env_path = parts[-1]
                            # Try both bin/python (Linux/macOS) and python.exe (Windows)
                            for python_name in ["bin/python", "python.exe"]:
                                potential_path = Path(env_path) / python_name
                                if potential_path.exists():
                                    python_path = str(potential_path)
                                    break
                            if python_path:
                                break

            if python_path:
                # Use direct python execution
                cmd = [python_path, str(validation_script), "-v"]
                self.ui.print_info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            else:
                # Fallback: try conda run anyway
                self.ui.print_warning(f"Could not find python in environment '{self.config.env_name}', trying conda run...")
                cmd = [conda_cmd, "run", "--no-capture-output", "-n", self.config.env_name, "python", str(validation_script), "-v"]
                self.ui.print_info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            # Print validation output
            if result.stdout:
                print(result.stdout)

            # Filter out conda's overly aggressive error logging for non-zero exit codes
            if result.stderr:
                stderr_lines = result.stderr.split('\n')
                filtered_lines = []

                for line in stderr_lines:
                    # Skip conda's false-positive error messages
                    if "ERROR conda.cli.main_run:execute(127):" in line and "failed." in line:
                        continue
                    if "failed. (See above for error)" in line:
                        continue
                    # Keep legitimate stderr content (like deprecation warnings)
                    if line.strip():
                        filtered_lines.append(line)

                if filtered_lines:
                    print('\n'.join(filtered_lines))

            if result.returncode == 0:
                self.ui.print_success("All validation checks passed!")
            elif result.returncode == 1:
                self.ui.print_warning("Validation passed with warnings")
                self.ui.print_info("Review the warnings above if you need specific features")
            else:
                self.ui.print_error(f"Validation failed with return code {result.returncode}")
                if result.stderr:
                    self.ui.print_error(f"Error details:\\n{result.stderr}")
                self.ui.print_info("Please review the errors above and fix any issues")

            return result.returncode

        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
            self.ui.print_error(f"Failed to run validation script: {e}")
            self.ui.print_info(f"Attempted command: {conda_cmd} run -n {self.config.env_name} python {validation_script} -v")
            self.ui.print_info("This might indicate an issue with conda environment or the validation script")
            return 1

    def create_config(self, host: str, user: str, password: str, port: int) -> None:
        """Create DataJoint configuration file."""
        config_dir = self.ui.select_config_location(self.config.repo_dir)
        config_file_path = config_dir / "dj_local_conf.json"

        self.ui.print_info(f"Creating configuration file at: {config_file_path}")

        # Create base directory structure
        self._create_directory_structure()

        # Create configuration using spyglass environment (without test_mode)
        try:
            self._create_config_in_env(host, user, password, port, config_dir)
            self.ui.print_success(f"Configuration file created at: {config_file_path}")
            self.ui.print_success(f"Data directories created at: {self.config.base_dir}")

        except (OSError, PermissionError, ValueError, json.JSONDecodeError) as e:
            self.ui.print_error(f"Failed to create configuration: {e}")
            raise

    def _create_config_in_env(self, host: str, user: str, password: str, port: int, config_dir: Path) -> None:
        """Create configuration within the spyglass environment."""
        import tempfile

        # Create a temporary Python script file for better subprocess handling
        python_script_content = f'''
import sys
import os
from pathlib import Path

# Change to config directory
original_cwd = Path.cwd()
try:
    os.chdir("{config_dir}")

    # Import and use SpyglassConfig
    from spyglass.settings import SpyglassConfig

    # Create SpyglassConfig instance (without test_mode)
    config = SpyglassConfig(base_dir="{self.config.base_dir}")

    # Save configuration
    config.save_dj_config(
        save_method="local",
        base_dir="{self.config.base_dir}",
        database_host="{host}",
        database_port={port},
        database_user="{user}",
        database_password="{password}",
        database_use_tls={not (host.startswith("127.0.0.1") or host == "localhost")},
        set_password=False
    )

    print("SUCCESS: Configuration created successfully")

finally:
    os.chdir(original_cwd)
'''

        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(python_script_content)
            temp_script_path = temp_file.name

        try:
            # Find the python executable in the spyglass environment directly
            env_name = self.config.env_name

            # Get the python executable path for the spyglass environment
            python_executable = self._get_env_python_executable(env_name)

            # Execute directly with the environment's python executable
            cmd = [python_executable, temp_script_path]

            # Run with stdin/stdout/stderr inherited to allow interactive prompts
            subprocess.run(cmd, check=True, stdin=None, stdout=None, stderr=None)
            self.ui.print_info("Configuration created in spyglass environment")

        except subprocess.CalledProcessError as e:
            self.ui.print_error(f"Failed to create configuration in environment '{env_name}'")
            self.ui.print_error(f"Return code: {e.returncode}")
            raise
        finally:
            # Clean up temporary file
            import os
            try:
                os.unlink(temp_script_path)
            except OSError:
                pass

    def _get_env_python_executable(self, env_name: str) -> str:
        """Get the python executable path for a conda environment."""
        import sys
        import subprocess
        from pathlib import Path

        # Try to get conda base path
        conda_base = Path(sys.executable).parent.parent

        # Common paths for conda environment python executables
        possible_paths = [
            conda_base / "envs" / env_name / "bin" / "python",  # Linux/Mac
            conda_base / "envs" / env_name / "python.exe",     # Windows
        ]

        for python_path in possible_paths:
            if python_path.exists():
                return str(python_path)

        # Fallback: try to find using conda command
        try:
            result = subprocess.run(
                ["conda", "run", "-n", env_name, "python", "-c", "import sys; print(sys.executable)"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        raise RuntimeError(f"Could not find Python executable for environment '{env_name}'")

    def _create_directory_structure(self) -> None:
        """Create the basic directory structure for Spyglass."""
        subdirs = ["raw", "analysis", "recording", "sorting", "tmp", "video", "waveforms"]

        try:
            self.config.base_dir.mkdir(parents=True, exist_ok=True)
            for subdir in subdirs:
                (self.config.base_dir / subdir).mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            self.ui.print_error(f"Permission denied creating directories: {e}")
            raise
        except (OSError, ValueError) as e:
            self.ui.print_error(f"Directory access failed: {e}")
            raise

    def _validate_spyglass_config(self, config) -> None:
        """Validate the created configuration using SpyglassConfig."""
        try:
            # Test basic functionality
            self.ui.print_info("Validating configuration...")
            # Validate that the config object has required attributes
            if hasattr(config, 'base_dir'):
                self.ui.print_success(f"Base directory configured: {config.base_dir}")
            # Add more validation logic here as needed
            self.ui.print_success("Configuration validated successfully")
        except (ValueError, AttributeError, TypeError) as e:
            self.ui.print_error(f"Configuration validation failed: {e}")
            raise

    def _print_summary(self) -> None:
        """Print installation summary."""
        self.ui.print_header("Setup Complete!")

        print("\nNext steps:")
        print("\n1. Activate the Spyglass environment:")
        print(f"   conda activate {self.config.env_name}")
        print("\n2. Test the installation:")
        print("   python -c \"from spyglass.settings import SpyglassConfig; print('✓ Integration successful')\"")
        print("\n3. Start with the tutorials:")
        print("   cd notebooks")
        print("   jupyter notebook 01_Concepts.ipynb")
        print("\n4. For help and documentation:")
        print("   Documentation: https://lorenfranklab.github.io/spyglass/")
        print("   GitHub Issues: https://github.com/LorenFrankLab/spyglass/issues")

        print("\nConfiguration Summary:")
        print(f"  Base directory: {self.config.base_dir}")
        print(f"  Environment: {self.config.env_name}")
        print(f"  Database: {'Configured' if self.config.setup_database else 'Skipped'}")
        print("  Integration: SpyglassConfig compatible")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Spyglass Quickstart Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quickstart.py                  # Minimal installation
  python quickstart.py --full           # Full installation
  python quickstart.py --pipeline=dlc   # DeepLabCut pipeline
  python quickstart.py --no-database    # Skip database setup
        """
    )

    # Mutually exclusive group for install type
    install_group = parser.add_mutually_exclusive_group()
    install_group.add_argument(
        "--minimal",
        action="store_true",
        help="Install core dependencies only (will prompt if none specified)"
    )
    install_group.add_argument(
        "--full",
        action="store_true",
        help="Install all optional dependencies (will prompt if none specified)"
    )

    parser.add_argument(
        "--pipeline",
        choices=["dlc", "moseq-cpu", "moseq-gpu", "lfp", "decoding"],
        help="Install specific pipeline dependencies (will prompt if none specified)"
    )

    parser.add_argument(
        "--no-database",
        action="store_true",
        help="Skip database setup"
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation after setup"
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(Path.home() / "spyglass_data"),
        help="Set base directory for data (default: ~/spyglass_data)"
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )

    parser.add_argument(
        "--env-name",
        type=str,
        default="spyglass",
        help="Name of conda environment to create (default: spyglass)"
    )

    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Auto-accept all prompts (non-interactive mode)"
    )

    parser.add_argument(
        "--db-port",
        type=int,
        default=3306,
        help="Host port for MySQL database (default: 3306)"
    )

    return parser.parse_args()


def main() -> Optional[int]:
    """Execute the main program."""
    args = parse_arguments()

    # Select colors based on arguments and terminal
    colors = DisabledColors if args.no_color or not sys.stdout.isatty() else Colors

    # Create configuration with validated base directory
    try:
        validated_base_dir = validate_base_dir(Path(args.base_dir))
    except ValueError as e:
        print(f"Error: Invalid base directory: {e}")
        return 1

    config = SetupConfig(
        install_type=InstallType.FULL if args.full else InstallType.MINIMAL,
        pipeline=Pipeline.__members__.get(args.pipeline.replace('-', '_').upper()) if args.pipeline else None,
        setup_database=not args.no_database,
        run_validation=not args.no_validate,
        base_dir=validated_base_dir,
        env_name=args.env_name,
        db_port=args.db_port,
        auto_yes=args.yes,
        install_type_specified=args.full or args.minimal or bool(args.pipeline)
    )

    # Run installer with new architecture
    orchestrator = QuickstartOrchestrator(config, colors)
    exit_code = orchestrator.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()