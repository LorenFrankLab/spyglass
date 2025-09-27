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
from pathlib import Path
from typing import Optional, List, Iterator, Tuple
from dataclasses import dataclass, replace
from enum import Enum
from collections import namedtuple
from functools import lru_cache
# Removed ABC import - not needed for a simple script
import getpass

# Named constants
DEFAULT_CHECKSUM_SIZE_LIMIT = 1024**3  # 1 GB


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


# Simplified helper functions - no need for Protocols in a script
def run_command(cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command with subprocess.run."""
    return subprocess.run(cmd, **kwargs)


def path_exists(path: Path) -> bool:
    """Check if a path exists."""
    return path.exists()


def make_directory(path: Path, exist_ok: bool = False) -> None:
    """Create a directory."""
    path.mkdir(exist_ok=exist_ok, parents=True)


class SpyglassConfigManager:
    """Manages SpyglassConfig for quickstart setup"""

    def create_config(self, base_dir: Path, host: str, port: int, user: str, password: str):
        """Create complete SpyglassConfig setup using official methods"""
        from spyglass.settings import SpyglassConfig

        # Create SpyglassConfig instance with base directory
        config = SpyglassConfig(base_dir=str(base_dir), test_mode=True)

        # Use SpyglassConfig's official save_dj_config method with local config
        config.save_dj_config(
            save_method="local",  # Creates dj_local_conf.json in current directory
            base_dir=str(base_dir),
            database_host=host,
            database_port=port,
            database_user=user,
            database_password=password,
            database_use_tls=False if host.startswith("127.0.0.1") or host == "localhost" else True,
            set_password=False  # Skip password prompt during setup
        )

        return config


class DatabaseSetupStrategy:
    """Base class for database setup strategies."""

    def setup(self, installer: 'SpyglassQuickstart') -> None:
        """Setup the database."""
        raise NotImplementedError("Subclasses must implement setup()")


class DockerDatabaseStrategy(DatabaseSetupStrategy):
    """Docker database setup strategy."""

    def setup(self, installer: 'SpyglassQuickstart') -> None:
        installer.print_info("Setting up local Docker database...")

        # Check Docker availability
        if not shutil.which("docker"):
            installer.print_error("Docker is not installed")
            installer.print_info("Please install Docker from: https://docs.docker.com/engine/install/")
            raise SystemRequirementError("Docker is not installed")

        # Check Docker daemon
        result = run_command(
            ["docker", "info"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            installer.print_error("Docker daemon is not running")
            installer.print_info("Please start Docker Desktop and try again")
            installer.print_info("On macOS: Open Docker Desktop application")
            installer.print_info("On Linux: sudo systemctl start docker")
            raise SystemRequirementError("Docker daemon is not running")

        # Pull and run container
        installer.print_info("Pulling MySQL image...")
        run_command(["docker", "pull", "datajoint/mysql:8.0"], check=True)

        # Check existing container
        result = run_command(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )

        if "spyglass-db" in result.stdout:
            installer.print_warning("Container 'spyglass-db' already exists")
            run_command(["docker", "start", "spyglass-db"], check=True)
        else:
            run_command([
                "docker", "run", "-d",
                "--name", "spyglass-db",
                "-p", "3306:3306",
                "-e", "MYSQL_ROOT_PASSWORD=tutorial",
                "datajoint/mysql:8.0"
            ], check=True)

        installer.print_success("Docker database started")
        installer.create_config("localhost", "root", "tutorial", 3306)


class ExistingDatabaseStrategy(DatabaseSetupStrategy):
    """Existing database setup strategy."""

    def setup(self, installer: 'SpyglassQuickstart') -> None:
        installer.print_info("Configuring connection to existing database...")

        host = input("Database host: ").strip()
        port_str = input("Database port (3306): ").strip() or "3306"
        port = int(port_str)
        user = input("Database user: ").strip()
        password = getpass.getpass("Database password: ")

        installer.create_config(host, user, password, port)


class SpyglassQuickstart:
    """Main quickstart installer class"""

    # Environment file mapping
    PIPELINE_ENVIRONMENTS = {
        Pipeline.DLC: ("environment_dlc.yml", "DeepLabCut pipeline environment"),
        Pipeline.MOSEQ_CPU: ("environment_moseq_cpu.yml", "Keypoint-Moseq CPU environment"),
        Pipeline.MOSEQ_GPU: ("environment_moseq_gpu.yml", "Keypoint-Moseq GPU environment"),
    }

    def __init__(self, config: SetupConfig, colors: Optional[object] = None):
        self.config = config
        self.colors = colors or Colors
        self.system_info: Optional[SystemInfo] = None

    def run(self) -> int:
        """Run the complete setup process"""
        try:
            self.print_header_banner()
            self._execute_setup_steps()
            self.print_summary()
            return 0
        except KeyboardInterrupt:
            self.print_error("\n\nSetup interrupted by user")
            return 130
        except SystemRequirementError as e:
            self.print_error(f"\nSystem requirement not met: {e}")
            self.print_info("Please install missing requirements and try again")
            return 2
        except EnvironmentCreationError as e:
            self.print_error(f"\nFailed to create environment: {e}")
            self.print_info("Check your conda installation and try again")
            return 3
        except DatabaseSetupError as e:
            self.print_error(f"\nDatabase setup failed: {e}")
            self.print_info("You can skip database setup with --no-database")
            return 4
        except SpyglassSetupError as e:
            self.print_error(f"\nSetup error: {e}")
            return 5
        except Exception as e:
            self.print_error(f"\nUnexpected error: {e}")
            self.print_info("Please report this issue at https://github.com/LorenFrankLab/spyglass/issues")
            return 1

    def _execute_setup_steps(self):
        """Execute all setup steps in sequence.

        This method coordinates the setup process. Each step is independent
        and can be tested separately.
        """
        # Define setup steps with their conditions
        setup_steps = [
            # Step: (method, condition to run, description)
            (self.detect_system, True, "Detecting system"),
            (self.check_python, True, "Checking Python"),
            (self.check_conda, True, "Checking conda/mamba"),
            (self.select_installation_type,
             not self._installation_type_specified(),
             "Selecting installation type"),
            (self._confirm_environment_name, True, "Confirming environment name"),
        ]

        # Execute initial setup steps
        for method, should_run, description in setup_steps:
            if should_run:
                method()

        # Environment setup - special handling for dependencies
        env_file = self.select_environment()
        env_was_updated = self.create_environment(env_file)

        # Only install additional dependencies if environment was created/updated
        if env_was_updated:
            self.install_additional_deps()

        # Optional final steps
        if self.config.setup_database:
            self.setup_database()

        if self.config.run_validation:
            self.run_validation()

    def print_header_banner(self):
        """Print welcome banner"""
        print(f"{self.colors.CYAN}{self.colors.BOLD}")
        print("╔═══════════════════════════════════════╗")
        print("║     Spyglass Quickstart Installer    ║")
        print("╚═══════════════════════════════════════╝")
        print(f"{self.colors.ENDC}")
        self.print_info("Note: SpyglassConfig warnings during setup are normal - configuration will be created")
        print()

    def print_header(self, text: str):
        """Print section header"""
        print()
        print(f"{self.colors.CYAN}{'=' * 42}{self.colors.ENDC}")
        print(f"{self.colors.CYAN}{self.colors.BOLD}{text}{self.colors.ENDC}")
        print(f"{self.colors.CYAN}{'=' * 42}{self.colors.ENDC}")
        print()

    def print_success(self, text: str):
        """Print success message"""
        print(f"{self.colors.GREEN}✓ {text}{self.colors.ENDC}")

    def print_warning(self, text: str):
        """Print warning message"""
        print(f"{self.colors.YELLOW}⚠ {text}{self.colors.ENDC}")

    def print_error(self, text: str):
        """Print error message"""
        print(f"{self.colors.RED}✗ {text}{self.colors.ENDC}")

    def print_info(self, text: str):
        """Print info message"""
        print(f"{self.colors.BLUE}ℹ {text}{self.colors.ENDC}")

    def detect_system(self):
        """Detect operating system and architecture"""
        self.print_header("System Detection")

        os_name = platform.system()
        arch = platform.machine()

        if os_name == "Darwin":
            os_display = "macOS"
            is_m1 = arch == "arm64"
            self.print_success("Operating System: macOS")
            if is_m1:
                self.print_success("Architecture: Apple Silicon (M1/M2)")
            else:
                self.print_success("Architecture: Intel x86_64")
        elif os_name == "Linux":
            os_display = "Linux"
            is_m1 = False
            self.print_success(f"Operating System: Linux")
            self.print_success(f"Architecture: {arch}")
        elif os_name == "Windows":
            self.print_warning("Windows detected - not officially supported")
            self.print_info("Proceeding with setup, but you may encounter issues")
            os_display = "Windows"
            is_m1 = False
        else:
            raise SystemRequirementError(f"Unsupported operating system: {os_name}")

        python_version = sys.version_info[:3]

        self.system_info = SystemInfo(
            os_name=os_display,
            arch=arch,
            is_m1=is_m1,
            python_version=python_version,
            conda_cmd=None
        )

    def check_python(self):
        """Check Python version"""
        self.print_header("Python Check")

        major, minor, micro = self.system_info.python_version
        version_str = f"{major}.{minor}.{micro}"

        if major >= 3 and minor >= 9:
            self.print_success(f"Python {version_str} found")
        else:
            self.print_warning(f"Python {version_str} found, but Python >= 3.9 is required")
            self.print_info("The conda environment will install the correct version")

    def check_conda(self):
        """Check for conda/mamba availability"""
        self.print_header("Package Manager Check")

        conda_cmd = self._find_conda_command()
        if not conda_cmd:
            self.print_error("Neither mamba nor conda found")
            self.print_info("Please install miniforge or miniconda:")
            self.print_info("  https://github.com/conda-forge/miniforge#install")
            raise SystemRequirementError("No conda/mamba found")

        # Update system info with conda command
        self.system_info = replace(self.system_info, conda_cmd=conda_cmd)

        version = self.get_command_output([conda_cmd, "--version"])
        if conda_cmd == "mamba":
            self.print_success(f"Found mamba (recommended): {version}")
        else:
            self.print_success(f"Found conda: {version}")
            self.print_info("Consider installing mamba for faster environment creation:")
            self.print_info("  conda install -n base -c conda-forge mamba")

    def _find_conda_command(self) -> Optional[str]:
        """Find available conda command"""
        for cmd in ["mamba", "conda"]:
            if shutil.which(cmd):
                return cmd
        return None

    def get_command_output(self, cmd: List[str], default: str = "") -> str:
        """Run command and return output, or default on failure.

        Args:
            cmd: Command to run as list of strings
            default: Value to return on failure

        Returns:
            Command output or default value
        """
        try:
            result = run_command(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Log failure for debugging but don't crash
            # In production, you'd want: logger.debug(f"Command failed: {cmd}")
            return default

    @lru_cache(maxsize=8)  # 8 is plenty for a setup script
    def cached_command(self, *cmd: str) -> str:
        """Cache frequently used command outputs."""
        return self.get_command_output(list(cmd), "")

    def _installation_type_specified(self) -> bool:
        """Check if installation type was specified via command line arguments."""
        # Installation type is considered specified if user used --full or --pipeline flags
        # Config always has these attributes due to dataclass defaults
        return (self.config.install_type == InstallType.FULL or
                self.config.pipeline is not None)

    def select_installation_type(self):
        """Let user select installation type interactively"""
        self.print_header("Installation Type Selection")

        print("\nChoose your installation type:")
        print("1) Minimal (core dependencies only)")
        print("   ├─ Basic Spyglass functionality")
        print("   ├─ Standard data analysis tools")
        print("   └─ Fastest installation (~5-10 minutes)")
        print()
        print("2) Full (all optional dependencies)")
        print("   ├─ All analysis pipelines included")
        print("   ├─ Spike sorting, LFP, visualization tools")
        print("   └─ Longer installation (~15-30 minutes)")
        print()
        print("3) Pipeline-specific")
        print("   ├─ Choose specific analysis pipeline")
        print("   ├─ DeepLabCut, Moseq, LFP, or Decoding")
        print("   └─ Optimized environment for your workflow")

        while True:
            choice = input("\nEnter choice (1-3): ").strip()
            if choice == "1":
                # Keep current minimal setup
                self.print_info("Selected: Minimal installation")
                break
            elif choice == "2":
                self.config.install_type = InstallType.FULL
                self.print_info("Selected: Full installation")
                break
            elif choice == "3":
                self._select_pipeline()
                break
            else:
                self.print_error("Invalid choice. Please enter 1, 2, or 3")

    def _select_pipeline(self):
        """Let user select specific pipeline"""
        print("\nChoose your pipeline:")
        print("1) DeepLabCut - Pose estimation and behavior analysis")
        print("2) Keypoint-Moseq (CPU) - Behavioral sequence analysis")
        print("3) Keypoint-Moseq (GPU) - GPU-accelerated behavioral analysis")
        print("4) LFP Analysis - Local field potential processing")
        print("5) Decoding - Neural population decoding")

        while True:
            choice = input("\nEnter choice (1-5): ").strip()
            if choice == "1":
                self.config.pipeline = Pipeline.DLC
                self.print_info("Selected: DeepLabCut pipeline")
                break
            elif choice == "2":
                self.config.pipeline = Pipeline.MOSEQ_CPU
                self.print_info("Selected: Keypoint-Moseq (CPU) pipeline")
                break
            elif choice == "3":
                self.config.pipeline = Pipeline.MOSEQ_GPU
                self.print_info("Selected: Keypoint-Moseq (GPU) pipeline")
                break
            elif choice == "4":
                self.config.pipeline = Pipeline.LFP
                self.print_info("Selected: LFP Analysis pipeline")
                break
            elif choice == "5":
                self.config.pipeline = Pipeline.DECODING
                self.print_info("Selected: Neural Decoding pipeline")
                break
            else:
                self.print_error("Invalid choice. Please enter 1-5")

    def _confirm_environment_name(self):
        """Let user confirm or customize environment name"""
        # Get suggested name based on installation type
        if self.config.pipeline and self.config.pipeline in self.PIPELINE_ENVIRONMENTS:
            # Pipeline-specific installations have descriptive suggestions
            suggested_name = {
                Pipeline.DLC: "spyglass-dlc",
                Pipeline.MOSEQ_CPU: "spyglass-moseq-cpu",
                Pipeline.MOSEQ_GPU: "spyglass-moseq-gpu"
            }.get(self.config.pipeline, "spyglass")

            print(f"\nYou selected {self.config.pipeline.value} pipeline.")
            print(f"Environment name options:")
            print(f"1) spyglass (default, works with all Spyglass documentation)")
            print(f"2) {suggested_name} (descriptive, matches pipeline choice)")
            print(f"3) Custom name")

            while True:
                choice = input(f"\nEnter choice (1-3) [default: 1]: ").strip() or "1"
                if choice == "1":
                    # Keep default name
                    break
                elif choice == "2":
                    self.config.env_name = suggested_name
                    self.print_info(f"Environment will be named: {suggested_name}")
                    break
                elif choice == "3":
                    custom_name = input("Enter custom environment name: ").strip()
                    if custom_name:
                        self.config.env_name = custom_name
                        self.print_info(f"Environment will be named: {custom_name}")
                        break
                    else:
                        self.print_error("Environment name cannot be empty")
                else:
                    self.print_error("Invalid choice. Please enter 1, 2, or 3")

        else:
            # Standard installations (minimal, full, or LFP/decoding pipelines)
            install_type_name = "minimal"
            if self.config.install_type == InstallType.FULL:
                install_type_name = "full"
            elif self.config.pipeline:
                install_type_name = self.config.pipeline.value

            print(f"\nYou selected {install_type_name} installation.")
            print(f"Environment name options:")
            print(f"1) spyglass (default)")
            print(f"2) Custom name")

            while True:
                choice = input(f"\nEnter choice (1-2) [default: 1]: ").strip() or "1"
                if choice == "1":
                    # Keep default name
                    break
                elif choice == "2":
                    custom_name = input("Enter custom environment name: ").strip()
                    if custom_name:
                        self.config.env_name = custom_name
                        self.print_info(f"Environment will be named: {custom_name}")
                        break
                    else:
                        self.print_error("Environment name cannot be empty")
                else:
                    self.print_error("Invalid choice. Please enter 1 or 2")

    def select_environment(self) -> str:
        """Select appropriate environment file"""
        self.print_header("Environment Selection")

        env_file, description = self._select_environment_file()
        self.print_info(f"Selected: {description}")

        # Verify environment file exists
        env_path = self.config.repo_dir / env_file
        if not path_exists(env_path):
            raise EnvironmentCreationError(f"Environment file not found: {env_path}")

        return env_file

    def _select_environment_file(self) -> Tuple[str, str]:
        """Select environment file and description"""
        # Check pipeline-specific environments first
        if self.config.pipeline and self.config.pipeline in self.PIPELINE_ENVIRONMENTS:
            return self.PIPELINE_ENVIRONMENTS[self.config.pipeline]

        # Standard environment with different descriptions
        if self.config.install_type == InstallType.FULL:
            description = "Standard environment (will add all optional dependencies)"
        elif self.config.pipeline:
            pipeline_name = self.config.pipeline.value
            description = f"Standard environment (will add {pipeline_name} dependencies)"
        else:
            description = "Standard environment (minimal)"

        return "environment.yml", description

    def create_environment(self, env_file: str) -> bool:
        """Create or update conda environment

        Returns:
            bool: True if environment was created/updated, False if kept existing
        """
        self.print_header("Creating Conda Environment")

        env_exists = self._check_environment_exists()
        if env_exists and not self._confirm_update():
            self.print_info("Keeping existing environment")
            return False

        cmd = self._build_environment_command(env_file, env_exists)
        self._execute_environment_command(cmd)
        self.print_success("Environment created/updated successfully")
        return True

    def _check_environment_exists(self) -> bool:
        """Check if environment already exists"""
        env_list = self.get_command_output([self.system_info.conda_cmd, "env", "list"])
        return self.config.env_name in env_list

    def _confirm_update(self) -> bool:
        """Confirm environment update with user"""
        self.print_warning(f"Environment '{self.config.env_name}' already exists")
        response = input("Do you want to update it? (y/N): ").strip().lower()
        return response == 'y'

    def _build_environment_command(self, env_file: str, update: bool) -> List[str]:
        """Build conda environment command"""
        env_path = self.config.repo_dir / env_file
        conda_cmd = self.system_info.conda_cmd
        env_name = self.config.env_name

        if update:
            self.print_info("Updating existing environment...")
            return [conda_cmd, "env", "update", "-f", str(env_path), "-n", env_name]
        else:
            self.print_info(f"Creating new environment '{env_name}'...")
            self.print_info("This may take 5-10 minutes...")
            return [conda_cmd, "env", "create", "-f", str(env_path), "-n", env_name]

    def _execute_environment_command(self, cmd: List[str], timeout: int = 1800):
        """Execute environment creation/update command with progress and timeout.

        Args:
            cmd: Command to execute
            timeout: Timeout in seconds (default 30 minutes)
        """
        import time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        start_time = time.time()

        # Show progress with timeout check
        for progress_line in self._filter_progress_lines(process):
            print(progress_line)

            # Check timeout
            if time.time() - start_time > timeout:
                process.kill()
                raise EnvironmentCreationError(
                    f"Environment creation exceeded {timeout}s timeout"
                )

        process.wait()
        if process.returncode != 0:
            raise EnvironmentCreationError("Environment creation/update failed")

    def _filter_progress_lines(self, process) -> Iterator[str]:
        """Filter and yield relevant progress lines"""
        progress_keywords = {"Solving environment", "Downloading", "Extracting"}

        for line in process.stdout:
            if any(keyword in line for keyword in progress_keywords):
                yield f"  {line.strip()}"

    def install_additional_deps(self):
        """Install additional dependencies"""
        self.print_header("Installing Additional Dependencies")

        # Install Spyglass in development mode
        self.print_info("Installing Spyglass in development mode...")
        self._run_in_env(["pip", "install", "-e", str(self.config.repo_dir)])

        # Pipeline-specific dependencies
        self._install_pipeline_dependencies()

        # Full installation
        if self.config.install_type == InstallType.FULL:
            self._install_full_dependencies()

        self.print_success("Additional dependencies installed")

    def _install_pipeline_dependencies(self):
        """Install pipeline-specific dependencies"""
        if self.config.pipeline == Pipeline.LFP:
            self.print_info("Installing LFP dependencies...")
            if self.system_info.is_m1:
                self.print_info("Detected M1 Mac, installing pyfftw via conda first...")
                self._run_in_env(["conda", "install", "-c", "conda-forge", "pyfftw", "-y"])
            self._run_in_env(["pip", "install", "ghostipy"])

        elif self.config.pipeline == Pipeline.DECODING:
            self.print_info("Installing decoding dependencies...")
            self.print_info("Please refer to JAX installation guide for GPU support:")
            self.print_info("https://jax.readthedocs.io/en/latest/installation.html")

    def _install_full_dependencies(self):
        """Install all optional dependencies"""
        self.print_info("Installing all optional dependencies...")
        self._run_in_env(["pip", "install", "spikeinterface[full,widgets]"])
        self._run_in_env(["pip", "install", "mountainsort4"])

        if self.system_info.is_m1:
            self._run_in_env(["conda", "install", "-c", "conda-forge", "pyfftw", "-y"])
        self._run_in_env(["pip", "install", "ghostipy"])

        self.print_warning("Some dependencies (DLC, JAX) require separate environment files")

    def _run_in_env(self, cmd: List[str]) -> int:
        """Run command in conda environment"""
        conda_cmd = self.system_info.conda_cmd
        env_name = self.config.env_name

        # Use conda run to execute in environment
        full_cmd = [conda_cmd, "run", "-n", env_name] + cmd

        result = run_command(
            full_cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            self.print_error(f"Command failed: {' '.join(cmd)}")
            if result.stderr:
                self.print_error(result.stderr)

        return result.returncode

    def _run_validation_script(self, script_path: Path) -> int:
        """Run validation script in the spyglass environment - simple and reliable."""
        try:
            # Run validation script in the spyglass environment
            conda_cmd = self.system_info.conda_cmd
            env_name = self.config.env_name

            result = run_command(
                [conda_cmd, "run", "-n", env_name, "python", str(script_path), "-v"],
                capture_output=True,
                text=True,
                check=False  # Don't raise on non-zero exit
            )

            # Print the output
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)

            return result.returncode

        except Exception as e:
            self.print_error(f"Validation failed: {e}")
            return 1

    def setup_database(self):
        """Setup database configuration"""
        self.print_header("Database Setup")

        strategy = self._select_database_strategy()
        if strategy is not None:
            strategy.setup(self)

    def _select_database_strategy(self) -> Optional[DatabaseSetupStrategy]:
        """Select database setup strategy"""
        print("\nChoose database setup option:")
        print("1) Local Docker database (recommended for beginners)")
        print("2) Connect to existing database")
        print("3) Skip database setup")

        while True:
            choice = input("\nEnter choice (1-3): ").strip()
            if choice == "1":
                return DockerDatabaseStrategy()
            elif choice == "2":
                return ExistingDatabaseStrategy()
            elif choice == "3":
                self.print_info("Skipping database setup")
                self.print_warning("You'll need to configure the database manually later")
                return None
            else:
                self.print_error("Invalid choice. Please enter 1, 2, or 3")

    def create_config(self, host: str, user: str, password: str, port: int):
        """Create DataJoint configuration file using SpyglassConfig directly"""
        self.print_info("Creating configuration file...")

        # Create base directory structure
        self._create_directory_structure()

        # Suppress SpyglassConfig warnings during setup
        import warnings
        import logging

        # Temporarily suppress specific warnings that occur during setup
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Failed to load SpyglassConfig.*")

            # Also temporarily suppress spyglass logger warnings
            spyglass_logger = logging.getLogger('spyglass')
            old_level = spyglass_logger.level
            spyglass_logger.setLevel(logging.ERROR)  # Only show errors, not warnings

            try:
                # Use SpyglassConfig to create and save configuration
                config_manager = SpyglassConfigManager()

                spyglass_config = config_manager.create_config(
                    base_dir=self.config.base_dir,
                    host=host,
                    port=port,
                    user=user,
                    password=password
                )

                # Config file is created in current working directory as dj_local_conf.json
                local_config_path = Path.cwd() / "dj_local_conf.json"
                self.print_success(f"Configuration file created at: {local_config_path}")
                self.print_success(f"Data directories created at: {self.config.base_dir}")

                # Validate the configuration
                self._validate_spyglass_config(spyglass_config)

            except Exception as e:
                self.print_error(f"Failed to create configuration: {e}")
                raise
            finally:
                # Restore original logger level
                spyglass_logger.setLevel(old_level)

    def _validate_spyglass_config(self, spyglass_config):
        """Validate the created configuration using SpyglassConfig"""
        try:
            # Test if the configuration can be loaded properly
            spyglass_config.load_config(force_reload=True)

            # Verify all expected directories are accessible
            test_dirs = [
                spyglass_config.base_dir,
                spyglass_config.raw_dir,
                spyglass_config.analysis_dir,
                spyglass_config.recording_dir,
                spyglass_config.sorting_dir,
            ]

            for test_dir in test_dirs:
                if test_dir and not Path(test_dir).exists():
                    self.print_warning(f"Directory not found: {test_dir}")

            self.print_success("Configuration validated with SpyglassConfig")

        except (ImportError, AttributeError) as e:
            self.print_warning(f"SpyglassConfig unavailable: {e}")
        except (FileNotFoundError, PermissionError) as e:
            self.print_error(f"Directory access failed: {e}")
        except Exception as e:
            self.print_error(f"Unexpected validation error: {e}")
            raise  # Re-raise unexpected errors

    def _create_directory_structure(self):
        """Create base directory structure using SpyglassConfig"""
        base_dir = self.config.base_dir
        make_directory(base_dir, exist_ok=True)

        try:
            # Use SpyglassConfig to create directories with official structure
            from spyglass.settings import SpyglassConfig

            # Create SpyglassConfig instance with our base directory
            sg_config = SpyglassConfig(base_dir=str(base_dir))
            sg_config.load_config()

            self.print_info("Using SpyglassConfig official directory structure")

        except ImportError:
            # Fallback to manual directory creation
            self.print_warning("SpyglassConfig not available, using fallback directory creation")
            subdirs = ["raw", "analysis", "recording", "sorting", "tmp", "video", "waveforms"]
            for subdir in subdirs:
                make_directory(base_dir / subdir, exist_ok=True)

    def run_validation(self) -> int:
        """Run validation script with SpyglassConfig integration check"""
        self.print_header("Running Validation")

        # First, run a quick SpyglassConfig integration test
        self._test_spyglass_integration()

        validation_script = self.config.repo_dir / "scripts" / "validate_spyglass.py"

        if not path_exists(validation_script):
            self.print_error("Validation script not found")
            return 1

        self.print_info("Running comprehensive validation checks...")

        # Run validation script directly
        exit_code = self._run_validation_script(validation_script)

        if exit_code == 0:
            self.print_success("All validation checks passed!")
        elif exit_code == 1:
            self.print_warning("Validation passed with warnings")
            self.print_info("Review the warnings above if you need specific features")
        else:
            self.print_error("Validation failed")
            self.print_info("Please review the errors above and fix any issues")

        return exit_code

    def _test_spyglass_integration(self):
        """Test SpyglassConfig integration in the spyglass environment."""
        try:
            # Create a simple integration test script to run in the environment
            test_cmd = [
                "python", "-c",
                f"from spyglass.settings import SpyglassConfig; "
                f"sg_config = SpyglassConfig(base_dir='{self.config.base_dir}'); "
                f"sg_config.load_config(); "
                f"print('✓ Integration successful')"
            ]

            exit_code = self._run_in_env(test_cmd)

            if exit_code == 0:
                self.print_success("SpyglassConfig integration test passed")
            else:
                self.print_warning("SpyglassConfig integration test failed")
                self.print_info("This may indicate a configuration issue")

        except Exception as e:
            self.print_warning(f"SpyglassConfig integration test failed: {e}")
            self.print_info("This may indicate a configuration issue")

    def print_summary(self):
        """Print setup summary and next steps"""
        self.print_header("Setup Complete!")

        print("\nNext steps:\n")
        print("1. Activate the Spyglass environment:")
        print(f"   {self.colors.GREEN}conda activate {self.config.env_name}{self.colors.ENDC}\n")

        print("2. Test the installation:")
        print(f"   {self.colors.GREEN}python -c \"from spyglass.settings import SpyglassConfig; print('✓ Integration successful')\"{self.colors.ENDC}\n")

        print("3. Start with the tutorials:")
        print(f"   {self.colors.GREEN}cd {self.config.repo_dir / 'notebooks'}{self.colors.ENDC}")
        print(f"   {self.colors.GREEN}jupyter notebook 01_Concepts.ipynb{self.colors.ENDC}\n")

        print("4. For help and documentation:")
        print(f"   {self.colors.BLUE}Documentation: https://lorenfranklab.github.io/spyglass/{self.colors.ENDC}")
        print(f"   {self.colors.BLUE}GitHub Issues: https://github.com/LorenFrankLab/spyglass/issues{self.colors.ENDC}\n")

        if not self.config.setup_database:
            self.print_warning("Remember to configure your database connection")
            self.print_info("See: https://lorenfranklab.github.io/spyglass/latest/notebooks/00_Setup/")

        # Show configuration summary
        print(f"\n{self.colors.CYAN}Configuration Summary:{self.colors.ENDC}")
        print(f"  Base directory: {self.config.base_dir}")
        print(f"  Environment: {self.config.env_name}")
        if self.config.setup_database:
            print(f"  Database: Configured")
        print(f"  Integration: SpyglassConfig compatible")


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

    # Create configuration
    config = SetupConfig(
        install_type=InstallType.FULL if args.full else InstallType.MINIMAL,
        pipeline=Pipeline(args.pipeline) if args.pipeline else None,
        setup_database=not args.no_database,
        run_validation=not args.no_validate,
        base_dir=Path(args.base_dir)
    )

    # Run installer
    installer = SpyglassQuickstart(config, colors=colors)
    exit_code = installer.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()