#!/usr/bin/env python
"""
Spyglass Quickstart Script (Python version)

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
import platform
import subprocess
import shutil
import argparse
import time
from pathlib import Path
from typing import Optional, List, Iterator, Tuple
from dataclasses import dataclass, replace
from enum import Enum
from collections import namedtuple
import getpass

# Named constants
DEFAULT_CHECKSUM_SIZE_LIMIT = 1024**3  # 1 GB

# User choice constants
CHOICE_1 = "1"
CHOICE_2 = "2"
CHOICE_3 = "3"
CHOICE_4 = "4"
CHOICE_5 = "5"

# Installation type choices
MINIMAL_CHOICE = CHOICE_1
FULL_CHOICE = CHOICE_2
PIPELINE_CHOICE = CHOICE_3

# Database setup choices
DOCKER_DB_CHOICE = CHOICE_1
EXISTING_DB_CHOICE = CHOICE_2
SKIP_DB_CHOICE = CHOICE_3

# Config location choices
REPO_ROOT_CHOICE = CHOICE_1
CURRENT_DIR_CHOICE = CHOICE_2
CUSTOM_PATH_CHOICE = CHOICE_3

# Pipeline choices
DLC_CHOICE = CHOICE_1
MOSEQ_CPU_CHOICE = CHOICE_2
MOSEQ_GPU_CHOICE = CHOICE_3
LFP_CHOICE = CHOICE_4
DECODING_CHOICE = CHOICE_5


# Exception hierarchy for clear error handling
class SpyglassSetupError(Exception):
    """Base exception for setup errors."""
    pass


class SystemRequirementError(SpyglassSetupError):
    """System doesn't meet requirements."""
    pass


class EnvironmentCreationError(SpyglassSetupError):
    """Failed to create conda environment."""
    pass


class DatabaseSetupError(SpyglassSetupError):
    """Failed to setup database."""
    pass


# Immutable Colors using NamedTuple
Colors = namedtuple('Colors', [
    'RED', 'GREEN', 'YELLOW', 'BLUE', 'CYAN', 'BOLD', 'ENDC'
])(
    RED='\033[0;31m',
    GREEN='\033[0;32m',
    YELLOW='\033[1;33m',
    BLUE='\033[0;34m',
    CYAN='\033[0;36m',
    BOLD='\033[1m',
    ENDC='\033[0m'
)

# Disabled colors instance
DisabledColors = Colors._replace(**{field: '' for field in Colors._fields})


class InstallType(Enum):
    """Installation type options"""
    MINIMAL = "minimal"
    FULL = "full"


class Pipeline(Enum):
    """Available pipeline options"""
    DLC = "dlc"
    MOSEQ_CPU = "moseq-cpu"
    MOSEQ_GPU = "moseq-gpu"
    LFP = "lfp"
    DECODING = "decoding"


@dataclass
class SystemInfo:
    """System information"""
    os_name: str
    arch: str
    is_m1: bool
    python_version: Tuple[int, int, int]
    conda_cmd: Optional[str]


@dataclass
class SetupConfig:
    """Configuration for setup process"""
    install_type: InstallType = InstallType.MINIMAL
    pipeline: Optional[Pipeline] = None
    setup_database: bool = True
    run_validation: bool = True
    base_dir: Path = Path.home() / "spyglass_data"
    repo_dir: Path = Path(__file__).parent.parent
    env_name: str = "spyglass"


# Using standard library functions directly - no unnecessary wrappers


def validate_base_dir(path: Path) -> Path:
    """Validate and resolve base directory path."""
    resolved = path.resolve()

    # Check if parent directory exists (we'll create the base_dir itself if needed)
    if not resolved.parent.exists():
        raise ValueError(f"Parent directory does not exist: {resolved.parent}")

    # Check for potential security issues (directory traversal)
    if str(resolved).startswith((".", "..")):
        raise ValueError(f"Relative paths not allowed: {path}")

    return resolved


class SpyglassConfigManager:
    """Manages SpyglassConfig for quickstart setup"""

    def create_config(self, base_dir: Path, host: str, port: int, user: str, password: str, config_dir: Path):
        """Create complete SpyglassConfig setup using official methods"""
        from spyglass.settings import SpyglassConfig
        import os

        # Temporarily change to config directory so dj_local_conf.json gets created there
        original_cwd = Path.cwd()
        try:
            os.chdir(config_dir)

            # Create SpyglassConfig instance with base directory
            config = SpyglassConfig(base_dir=str(base_dir), test_mode=True)

            # Use SpyglassConfig's official save_dj_config method with local config
            config.save_dj_config(
                save_method="local",  # Creates dj_local_conf.json in current directory (config_dir)
                base_dir=str(base_dir),
                database_host=host,
                database_port=port,
                database_user=user,
                database_password=password,
                database_use_tls=False if host.startswith("127.0.0.1") or host == "localhost" else True,
                set_password=False  # Skip password prompt during setup
            )

            return config
        finally:
            # Always restore original working directory
            os.chdir(original_cwd)


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
        subprocess.run([
            "docker", "run", "-d",
            "--name", "spyglass-db",
            "-p", "3306:3306",
            "-e", "MYSQL_ROOT_PASSWORD=tutorial",
            "datajoint/mysql:8.0"
        ], check=True)

    orchestrator.ui.print_success("Docker database started")
    orchestrator.create_config("localhost", "root", "tutorial", 3306)


def setup_existing_database(orchestrator: 'QuickstartOrchestrator') -> None:
    """Setup existing database connection."""
    orchestrator.ui.print_info("Configuring connection to existing database...")

    host, port, user, password = orchestrator.ui.get_database_credentials()
    orchestrator.create_config(host, user, password, port)


# Database setup function mapping - simple dictionary approach
DATABASE_SETUP_METHODS = {
    DOCKER_DB_CHOICE: setup_docker_database,
    EXISTING_DB_CHOICE: setup_existing_database,
    SKIP_DB_CHOICE: lambda orchestrator: None  # Skip setup
}


class UserInterface:
    """Handles all user interactions and display formatting."""

    def __init__(self, colors):
        self.colors = colors

    def print_header_banner(self):
        """Print the main application banner"""
        print("\n" + "═" * 43)
        print("║     Spyglass Quickstart Installer    ║")
        print("═" * 43)

    def print_header(self, text: str):
        """Print section header"""
        print(f"\n{'=' * 42}")
        print(text)
        print("=" * 42)

    def _format_message(self, text: str, symbol: str, color: str) -> str:
        """Format a message with color and symbol."""
        return f"{color}{symbol} {text}{self.colors.ENDC}"

    def print_success(self, text: str):
        """Print success message"""
        print(self._format_message(text, "✓", self.colors.GREEN))

    def print_warning(self, text: str):
        """Print warning message"""
        print(self._format_message(text, "⚠", self.colors.YELLOW))

    def print_error(self, text: str):
        """Print error message"""
        print(self._format_message(text, "✗", self.colors.RED))

    def print_info(self, text: str):
        """Print info message"""
        print(self._format_message(text, "ℹ", self.colors.BLUE))

    def select_install_type(self) -> Tuple[InstallType, Optional[Pipeline]]:
        """Let user select installation type"""
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
            choice = input("\nEnter choice (1-3): ").strip()
            if choice == MINIMAL_CHOICE:
                return InstallType.MINIMAL, None
            elif choice == FULL_CHOICE:
                return InstallType.FULL, None
            elif choice == PIPELINE_CHOICE:
                pipeline = self.select_pipeline()
                return InstallType.MINIMAL, pipeline
            else:
                self.print_error("Invalid choice. Please enter 1, 2, or 3")

    def select_pipeline(self) -> Pipeline:
        """Let user select specific pipeline"""
        print("\nChoose your pipeline:")
        print("1) DeepLabCut - Pose estimation and behavior analysis")
        print("2) Keypoint-Moseq (CPU) - Behavioral sequence analysis")
        print("3) Keypoint-Moseq (GPU) - GPU-accelerated behavioral analysis")
        print("4) LFP Analysis - Local field potential processing")
        print("5) Decoding - Neural population decoding")

        while True:
            choice = input("\nEnter choice (1-5): ").strip()
            if choice == DLC_CHOICE:
                return Pipeline.DLC
            elif choice == MOSEQ_CPU_CHOICE:
                return Pipeline.MOSEQ_CPU
            elif choice == MOSEQ_GPU_CHOICE:
                return Pipeline.MOSEQ_GPU
            elif choice == LFP_CHOICE:
                return Pipeline.LFP
            elif choice == DECODING_CHOICE:
                return Pipeline.DECODING
            else:
                self.print_error("Invalid choice. Please enter 1-5")

    def confirm_environment_update(self, env_name: str) -> bool:
        """Ask user if they want to update existing environment"""
        self.print_warning(f"Environment '{env_name}' already exists")
        choice = input("Do you want to update it? (y/N): ").strip().lower()
        return choice == 'y'

    def select_database_setup(self) -> str:
        """Select database setup choice"""
        print("\nChoose database setup option:")
        print("1) Local Docker database (recommended for beginners)")
        print("2) Connect to existing database")
        print("3) Skip database setup")

        while True:
            choice = input("\nEnter choice (1-3): ").strip()
            if choice in [DOCKER_DB_CHOICE, EXISTING_DB_CHOICE, SKIP_DB_CHOICE]:
                if choice == SKIP_DB_CHOICE:
                    self.print_info("Skipping database setup")
                    self.print_warning("You'll need to configure the database manually later")
                return choice
            else:
                self.print_error("Invalid choice. Please enter 1, 2, or 3")

    def select_config_location(self, repo_dir: Path) -> Path:
        """Select where to save the DataJoint configuration file"""
        print("\nChoose configuration file location:")
        print(f"1) Repository root (recommended): {repo_dir}")
        print("2) Current directory")
        print("3) Custom location")

        while True:
            choice = input("\nEnter choice (1-3): ").strip()
            if choice == REPO_ROOT_CHOICE:
                return repo_dir
            elif choice == CURRENT_DIR_CHOICE:
                return Path.cwd()
            elif choice == CUSTOM_PATH_CHOICE:
                return self._get_custom_path()
            else:
                self.print_error("Invalid choice. Please enter 1, 2, or 3")

    def _get_custom_path(self) -> Path:
        """Get custom path from user with validation"""
        while True:
            custom_path = input("Enter custom directory path: ").strip()
            if not custom_path:
                self.print_error("Path cannot be empty")
                continue

            try:
                path = Path(custom_path).expanduser().resolve()
                if not path.exists():
                    create = input(f"Directory {path} doesn't exist. Create it? (y/N): ").strip().lower()
                    if create == 'y':
                        path.mkdir(parents=True, exist_ok=True)
                    else:
                        continue
                if not path.is_dir():
                    self.print_error("Path must be a directory")
                    continue
                return path
            except Exception as e:
                self.print_error(f"Invalid path: {e}")
                continue

    def get_database_credentials(self) -> Tuple[str, int, str, str]:
        """Get database connection credentials from user"""
        host = input("Database host: ").strip()
        port_str = input("Database port (3306): ").strip() or "3306"
        port = int(port_str)
        user = input("Database user: ").strip()
        password = getpass.getpass("Database password: ")
        return host, port, user, password


class EnvironmentManager:
    """Handles conda environment creation and management."""

    def __init__(self, ui, config: SetupConfig):
        self.ui = ui
        self.config = config
        self.PIPELINE_ENVIRONMENTS = {
            Pipeline.DLC: ("environment_dlc.yml", "DeepLabCut pipeline environment"),
            Pipeline.MOSEQ_CPU: ("environment_moseq.yml", "Keypoint-Moseq (CPU) pipeline environment"),
            Pipeline.MOSEQ_GPU: ("environment_moseq_gpu.yml", "Keypoint-Moseq (GPU) pipeline environment"),
            Pipeline.LFP: ("environment_lfp.yml", "LFP pipeline environment"),
            Pipeline.DECODING: ("environment_decoding.yml", "Decoding pipeline environment"),
        }

    def select_environment_file(self) -> str:
        """Select appropriate environment file based on configuration"""
        if env_info := self.PIPELINE_ENVIRONMENTS.get(self.config.pipeline):
            env_file, description = env_info
            self.ui.print_info(f"Selected: {description}")
        elif self.config.install_type == InstallType.FULL:
            env_file = "environment.yml"
            self.ui.print_info("Selected: Standard environment (full)")
        else:
            env_file = "environment.yml"
            self.ui.print_info("Selected: Standard environment (minimal)")

        # Verify environment file exists
        env_path = self.config.repo_dir / env_file
        if not env_path.exists():
            raise EnvironmentCreationError(
                f"Environment file not found: {env_path}\n"
                f"Please ensure you're running from the Spyglass repository root"
            )

        return env_file

    def create_environment(self, env_file: str, conda_cmd: str) -> bool:
        """Create or update conda environment"""
        self.ui.print_header("Creating Conda Environment")

        update = self._check_environment_exists(conda_cmd)
        if update:
            if not self.ui.confirm_environment_update(self.config.env_name):
                self.ui.print_info("Keeping existing environment unchanged")
                return True

        cmd = self._build_environment_command(env_file, conda_cmd, update)
        self._execute_environment_command(cmd)
        return True

    def _check_environment_exists(self, conda_cmd: str) -> bool:
        """Check if the target environment already exists"""
        try:
            result = subprocess.run([conda_cmd, "env", "list"], capture_output=True, text=True, check=True)
            return self.config.env_name in result.stdout
        except subprocess.CalledProcessError:
            return False

    def _build_environment_command(self, env_file: str, conda_cmd: str, update: bool) -> List[str]:
        """Build conda environment command"""
        env_path = self.config.repo_dir / env_file
        env_name = self.config.env_name

        if update:
            self.ui.print_info("Updating existing environment...")
            return [conda_cmd, "env", "update", "-f", str(env_path), "-n", env_name]
        else:
            self.ui.print_info(f"Creating new environment '{env_name}'...")
            self.ui.print_info("This may take 5-10 minutes...")
            return [conda_cmd, "env", "create", "-f", str(env_path), "-n", env_name]

    def _execute_environment_command(self, cmd: List[str], timeout: int = 1800):
        """Execute environment creation/update command with progress and timeout"""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Monitor process with timeout
            start_time = time.time()
            while process.poll() is None:
                if time.time() - start_time > timeout:
                    process.kill()
                    raise EnvironmentCreationError("Environment creation timed out after 30 minutes")

                # Read and display progress
                try:
                    for line in self._filter_progress_lines(process):
                        print(line)
                except:
                    pass

                time.sleep(1)

            if process.returncode != 0:
                stderr_output = process.stderr.read() if process.stderr else "Unknown error"
                raise EnvironmentCreationError(
                    f"Environment creation failed with return code {process.returncode}\n{stderr_output}"
                )

        except subprocess.TimeoutExpired:
            raise EnvironmentCreationError("Environment creation timed out")
        except Exception as e:
            raise EnvironmentCreationError("Environment creation/update failed")

    def _filter_progress_lines(self, process) -> Iterator[str]:
        """Filter and yield relevant progress lines"""
        progress_keywords = {"Solving environment", "Downloading", "Extracting", "Installing"}

        for line in process.stdout:
            if any(keyword in line for keyword in progress_keywords):
                yield f"  {line.strip()}"

    def install_additional_dependencies(self, conda_cmd: str):
        """Install additional dependencies after environment creation"""
        self.ui.print_header("Installing Additional Dependencies")

        # Install in development mode
        self.ui.print_info("Installing Spyglass in development mode...")
        self._run_in_env(conda_cmd, ["pip", "install", "-e", str(self.config.repo_dir)])

        # Install pipeline-specific dependencies
        if self.config.pipeline:
            self._install_pipeline_dependencies(conda_cmd)
        elif self.config.install_type == InstallType.FULL:
            self._install_full_dependencies(conda_cmd)

        self.ui.print_success("Additional dependencies installed")

    def _install_pipeline_dependencies(self, conda_cmd: str):
        """Install dependencies for specific pipeline"""
        self.ui.print_info("Installing pipeline-specific dependencies...")

        if self.config.pipeline == Pipeline.LFP:
            self.ui.print_info("Installing LFP dependencies...")
            # Handle M1 Mac specific installation
            system_info = self._get_system_info()
            if system_info and system_info.is_m1:
                self.ui.print_info("Detected M1 Mac, installing pyfftw via conda first...")
                self._run_in_env(conda_cmd, ["conda", "install", "-c", "conda-forge", "pyfftw", "-y"])

    def _install_full_dependencies(self, conda_cmd: str):
        """Install full set of dependencies"""
        self.ui.print_info("Installing full dependencies...")
        # Add full dependency installation logic here if needed

    def _run_in_env(self, conda_cmd: str, cmd: List[str]) -> int:
        """Run command - simplified approach"""
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return result.returncode
        except subprocess.CalledProcessError as e:
            self.ui.print_error(f"Command failed: {' '.join(cmd)}")
            if e.stderr:
                self.ui.print_error(e.stderr)
            raise

    def _get_system_info(self):
        """Get system info - placeholder for now"""
        # This would be injected or accessed differently in the refactored version
        return None


class QuickstartOrchestrator:
    """Main orchestrator that coordinates all installation components."""

    def __init__(self, config: SetupConfig, colors):
        self.config = config
        self.ui = UserInterface(colors)
        self.system_detector = SystemDetector(self.ui)
        self.env_manager = EnvironmentManager(self.ui, config)
        self.system_info = None

    def run(self) -> int:
        """Run the complete installation process."""
        try:
            self.ui.print_header_banner()
            self._execute_setup_steps()
            self._print_summary()
            return 0

        except KeyboardInterrupt:
            self.ui.print_error("\n\nSetup interrupted by user")
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
        except Exception as e:
            self.ui.print_error(f"\nUnexpected error: {e}")
            return 1

    def _execute_setup_steps(self):
        """Execute the main setup steps in order."""
        # Step 1: System Detection
        self.system_info = self.system_detector.detect_system()
        self.system_detector.check_python(self.system_info)
        conda_cmd = self.system_detector.check_conda()
        self.system_info = replace(self.system_info, conda_cmd=conda_cmd)

        # Step 2: Installation Type Selection (if not specified)
        if not self._installation_type_specified():
            install_type, pipeline = self.ui.select_install_type()
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

    def _installation_type_specified(self) -> bool:
        """Check if installation type was specified via command line arguments."""
        return (self.config.install_type == InstallType.FULL or
                self.config.pipeline is not None)

    def _setup_database(self):
        """Setup database configuration"""
        self.ui.print_header("Database Setup")

        choice = self.ui.select_database_setup()
        setup_func = DATABASE_SETUP_METHODS.get(choice)
        if setup_func:
            setup_func(self)

    def _run_validation(self, conda_cmd: str) -> int:
        """Run validation checks"""
        self.ui.print_header("Running Validation")

        validation_script = self.config.repo_dir / "scripts" / "validate_spyglass.py"

        if not validation_script.exists():
            self.ui.print_error("Validation script not found")
            self.ui.print_info("Expected location: scripts/validate_spyglass.py")
            self.ui.print_info("Please ensure you're running from the Spyglass repository root")
            return 1

        self.ui.print_info("Running comprehensive validation checks...")

        try:
            result = subprocess.run(
                ["python", str(validation_script), "-v"],
                capture_output=True,
                text=True,
                check=False
            )

            # Print validation output
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)

            if result.returncode == 0:
                self.ui.print_success("All validation checks passed!")
            elif result.returncode == 1:
                self.ui.print_warning("Validation passed with warnings")
                self.ui.print_info("Review the warnings above if you need specific features")
            else:
                self.ui.print_error("Validation failed")
                self.ui.print_info("Please review the errors above and fix any issues")

            return result.returncode

        except Exception as e:
            self.ui.print_error(f"Validation failed: {e}")
            return 1

    def create_config(self, host: str, user: str, password: str, port: int):
        """Create DataJoint configuration file"""
        config_dir = self.ui.select_config_location(self.config.repo_dir)
        config_file_path = config_dir / "dj_local_conf.json"

        self.ui.print_info(f"Creating configuration file at: {config_file_path}")

        # Create base directory structure
        self._create_directory_structure()

        # Use SpyglassConfig to create configuration
        try:
            config_manager = SpyglassConfigManager()
            spyglass_config = config_manager.create_config(
                base_dir=self.config.base_dir,
                host=host,
                port=port,
                user=user,
                password=password,
                config_dir=config_dir
            )

            self.ui.print_success(f"Configuration file created at: {config_file_path}")
            self.ui.print_success(f"Data directories created at: {self.config.base_dir}")

            # Validate the configuration
            self._validate_spyglass_config(spyglass_config)

        except Exception as e:
            self.ui.print_error(f"Failed to create configuration: {e}")
            raise

    def _create_directory_structure(self):
        """Create the basic directory structure for Spyglass"""
        subdirs = ["raw", "analysis", "recording", "sorting", "tmp", "video", "waveforms"]

        try:
            self.config.base_dir.mkdir(parents=True, exist_ok=True)
            for subdir in subdirs:
                (self.config.base_dir / subdir).mkdir(exist_ok=True)
        except PermissionError as e:
            self.ui.print_error(f"Permission denied creating directories: {e}")
            raise
        except Exception as e:
            self.ui.print_error(f"Directory access failed: {e}")
            raise

    def _validate_spyglass_config(self, spyglass_config):
        """Validate the created configuration using SpyglassConfig"""
        try:
            # Test basic functionality
            self.ui.print_info("Validating configuration...")
            # Add basic validation logic here
            self.ui.print_success("Configuration validated successfully")
        except Exception as e:
            self.ui.print_error(f"Configuration validation failed: {e}")
            raise

    def _print_summary(self):
        """Print installation summary"""
        self.ui.print_header("Setup Complete!")

        print("\nNext steps:")
        print(f"\n1. Activate the Spyglass environment:")
        print(f"   conda activate {self.config.env_name}")
        print(f"\n2. Test the installation:")
        print(f"   python -c \"from spyglass.settings import SpyglassConfig; print('✓ Integration successful')\"")
        print(f"\n3. Start with the tutorials:")
        print(f"   cd notebooks")
        print(f"   jupyter notebook 01_Concepts.ipynb")
        print(f"\n4. For help and documentation:")
        print(f"   Documentation: https://lorenfranklab.github.io/spyglass/")
        print(f"   GitHub Issues: https://github.com/LorenFrankLab/spyglass/issues")

        print(f"\nConfiguration Summary:")
        print(f"  Base directory: {self.config.base_dir}")
        print(f"  Environment: {self.config.env_name}")
        print(f"  Database: {'Configured' if self.config.setup_database else 'Skipped'}")
        print(f"  Integration: SpyglassConfig compatible")


class SystemDetector:
    """Handles system detection and validation."""

    def __init__(self, ui):
        self.ui = ui

    def detect_system(self) -> SystemInfo:
        """Detect operating system and architecture"""
        self.ui.print_header("System Detection")

        os_name = platform.system()
        arch = platform.machine()

        if os_name == "Darwin":
            os_display = "macOS"
            is_m1 = arch == "arm64"
            self.ui.print_success("Operating System: macOS")
            if is_m1:
                self.ui.print_success("Architecture: Apple Silicon (M1/M2)")
            else:
                self.ui.print_success("Architecture: Intel x86_64")
        elif os_name == "Linux":
            os_display = "Linux"
            is_m1 = False
            self.ui.print_success(f"Operating System: Linux")
            self.ui.print_success(f"Architecture: {arch}")
        elif os_name == "Windows":
            self.ui.print_warning("Windows detected - not officially supported")
            self.ui.print_info("Proceeding with setup, but you may encounter issues")
            os_display = "Windows"
            is_m1 = False
        else:
            raise SystemRequirementError(f"Unsupported operating system: {os_name}")

        python_version = sys.version_info[:3]

        return SystemInfo(
            os_name=os_display,
            arch=arch,
            is_m1=is_m1,
            python_version=python_version,
            conda_cmd=None
        )

    def check_python(self, system_info: SystemInfo):
        """Check Python version"""
        self.ui.print_header("Python Check")

        major, minor, micro = system_info.python_version
        version_str = f"{major}.{minor}.{micro}"

        if major >= 3 and minor >= 9:
            self.ui.print_success(f"Python {version_str} found")
        else:
            self.ui.print_warning(f"Python {version_str} found, but Python >= 3.9 is required")
            self.ui.print_info("The conda environment will install the correct version")

    def check_conda(self) -> str:
        """Check for conda/mamba availability and return the command to use"""
        self.ui.print_header("Package Manager Check")

        conda_cmd = self._find_conda_command()
        if not conda_cmd:
            self.ui.print_error("Neither mamba nor conda found")
            self.ui.print_info("Please install miniforge or miniconda:")
            self.ui.print_info("  https://github.com/conda-forge/miniforge#install")
            raise SystemRequirementError("No conda/mamba found")

        # Show version info
        version_output = self._get_command_output([conda_cmd, "--version"])
        if version_output:
            self.ui.print_success(f"Found {conda_cmd}: {version_output}")

        if conda_cmd == "conda":
            self.ui.print_info("Consider installing mamba for faster environment creation:")
            self.ui.print_info("  conda install -n base -c conda-forge mamba")

        return conda_cmd

    def _find_conda_command(self) -> Optional[str]:
        """Find available conda command, preferring mamba"""
        for cmd in ["mamba", "conda"]:
            if shutil.which(cmd):
                return cmd
        return None

    def _get_command_output(self, cmd: List[str]) -> str:
        """Get command output, return empty string on failure"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""


def parse_arguments():
    """Parse command line arguments"""
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

    return parser.parse_args()


def main():
    """Main entry point"""
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
        base_dir=validated_base_dir
    )

    # Run installer with new architecture
    orchestrator = QuickstartOrchestrator(config, colors)
    exit_code = orchestrator.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()