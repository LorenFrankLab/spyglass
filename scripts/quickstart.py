#!/usr/bin/env python
"""
Spyglass Quickstart Script (Python version)

One-command setup for Spyglass installation.
This script provides a streamlined setup process for Spyglass, guiding you
through environment creation, package installation, and configuration.

Usage:
    python quickstart.py [OPTIONS]

Options:
    --minimal       Install core dependencies only (default)
    --full          Install all optional dependencies
    --pipeline=X    Install specific pipeline dependencies
    --no-database   Skip database setup
    --no-validate   Skip validation after setup
    --base-dir=PATH Set base directory for data
    --help          Show help message
"""

import sys
import json
import platform
import subprocess
import shutil
import argparse
from pathlib import Path
from typing import Optional, List, Protocol, Iterator, Tuple
from dataclasses import dataclass, replace
from enum import Enum
from collections import namedtuple
from contextlib import suppress
from functools import wraps, lru_cache
from abc import ABC, abstractmethod
import getpass

# Named constants
DEFAULT_CHECKSUM_SIZE_LIMIT = 1024**3  # 1 GB


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


# Protocols for dependency injection
class CommandRunner(Protocol):
    """Protocol for command execution"""
    def run(self, cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
        ...


class FileSystem(Protocol):
    """Protocol for file system operations"""
    def exists(self, path: Path) -> bool:
        ...

    def mkdir(self, path: Path, exist_ok: bool = False) -> None:
        ...


# Default implementations
class DefaultCommandRunner:
    """Default command runner implementation"""

    def run(self, cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, **kwargs)


class DefaultFileSystem:
    """Default file system implementation"""

    def exists(self, path: Path) -> bool:
        return path.exists()

    def mkdir(self, path: Path, exist_ok: bool = False) -> None:
        path.mkdir(exist_ok=exist_ok)


# Decorator for safe subprocess execution
def subprocess_handler(default_return=""):
    """Decorator for safe subprocess execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with suppress(subprocess.CalledProcessError, FileNotFoundError):
                return func(*args, **kwargs)
            return default_return
        return wrapper
    return decorator


class ConfigBuilder:
    """Builder for DataJoint configuration integrated with SpyglassConfig"""

    def __init__(self, spyglass_config_factory=None):
        self._config = {}
        self._spyglass_config_factory = spyglass_config_factory or self._default_config_factory
        self._spyglass_config = None

    @staticmethod
    def _default_config_factory():
        """Default factory for SpyglassConfig"""
        try:
            from spyglass.settings import SpyglassConfig
            return SpyglassConfig()
        except ImportError:
            return None

    @property
    def spyglass_config(self):
        """Lazy-load SpyglassConfig only when needed"""
        if self._spyglass_config is None:
            self._spyglass_config = self._spyglass_config_factory()
        return self._spyglass_config

    def database(self, host: str, port: int, user: str, password: str) -> 'ConfigBuilder':
        """Add database configuration"""
        self._config.update({
            "database.host": host,
            "database.port": port,
            "database.user": user,
            "database.password": password,
            "database.reconnect": True,
            "database.use_tls": False,
        })
        return self

    def stores(self, base_dir: Path) -> 'ConfigBuilder':
        """Add store configuration using SpyglassConfig structure"""
        self._config["stores"] = {
            "raw": {
                "protocol": "file",
                "location": str(base_dir / "raw"),
                "stage": str(base_dir / "raw")
            },
            "analysis": {
                "protocol": "file",
                "location": str(base_dir / "analysis"),
                "stage": str(base_dir / "analysis")
            },
        }
        return self

    def spyglass_dirs(self, base_dir: Path) -> 'ConfigBuilder':
        """Add Spyglass directory configuration using official structure"""
        if self.spyglass_config:
            self._add_official_spyglass_config(base_dir)
        else:
            self._add_fallback_spyglass_config(base_dir)
        return self

    def _add_official_spyglass_config(self, base_dir: Path):
        """Add configuration using official SpyglassConfig structure"""
        config = self.spyglass_config

        self._config["custom"] = {
            "spyglass_dirs": self._build_spyglass_dirs(base_dir, config),
            "debug_mode": "false",
            "test_mode": "false",
        }

        self._add_subsystem_configs(base_dir, config)
        self._add_spyglass_defaults()

    def _add_fallback_spyglass_config(self, base_dir: Path):
        """Add fallback configuration when SpyglassConfig unavailable"""
        subdirs = ["raw", "analysis", "recording", "sorting", "tmp", "video", "waveforms"]
        spyglass_config = self._build_dir_config(base_dir, {subdir: subdir for subdir in subdirs})
        spyglass_config["base"] = str(base_dir)

        self._config["custom"] = {
            "spyglass_dirs": spyglass_config,
            "debug_mode": "false",
            "test_mode": "false",
        }

    def _build_dir_config(self, base_dir: Path, dirs_dict: dict) -> dict:
        """Build directory configuration with consistent path conversion"""
        return {subdir: str(base_dir / rel_path)
                for subdir, rel_path in dirs_dict.items()}

    def _build_spyglass_dirs(self, base_dir: Path, config) -> dict:
        """Build core spyglass directory configuration"""
        spyglass_dirs = config.relative_dirs["spyglass"]
        result = self._build_dir_config(base_dir, spyglass_dirs)
        result["base"] = str(base_dir)
        return result

    def _add_subsystem_configs(self, base_dir: Path, config):
        """Add configurations for subsystems (kachery, DLC, moseq)"""
        # Add kachery directories
        kachery_dirs = config.relative_dirs.get("kachery", {})
        self._config["custom"]["kachery_dirs"] = self._build_dir_config(base_dir, kachery_dirs)

        # Add DLC directories
        dlc_dirs = config.relative_dirs.get("dlc", {})
        dlc_base = base_dir / "deeplabcut"
        self._config["custom"]["dlc_dirs"] = {
            "base": str(dlc_base),
            **self._build_dir_config(dlc_base, dlc_dirs)
        }

        # Add Moseq directories
        moseq_dirs = config.relative_dirs.get("moseq", {})
        moseq_base = base_dir / "moseq"
        self._config["custom"]["moseq_dirs"] = {
            "base": str(moseq_base),
            **self._build_dir_config(moseq_base, moseq_dirs)
        }

    def _add_spyglass_defaults(self):
        """Add standard SpyglassConfig defaults"""
        self._config.update({
            "filepath_checksum_size_limit": DEFAULT_CHECKSUM_SIZE_LIMIT,
            "enable_python_native_blobs": True,
        })

    def build(self) -> dict:
        """Build the final configuration with validation"""
        config = self._config.copy()
        self._validate_config(config)
        return config

    def _validate_config(self, config: dict):
        """Validate configuration completeness and consistency"""
        # Check for required keys (some are flat, some are nested)
        required_checks = [
            ("database.host" in config, "database.host"),
            ("stores" in config, "stores")
        ]

        missing = [key for check, key in required_checks if not check]
        if missing:
            raise ValueError(f"Missing required configuration: {missing}")


class DatabaseSetupStrategy(ABC):
    """Abstract base class for database setup strategies"""

    @abstractmethod
    def setup(self, installer: 'SpyglassQuickstart') -> None:
        """Setup the database"""
        pass


class DockerDatabaseStrategy(DatabaseSetupStrategy):
    """Docker database setup strategy"""

    def setup(self, installer: 'SpyglassQuickstart') -> None:
        installer.print_info("Setting up local Docker database...")

        # Check Docker availability
        if not shutil.which("docker"):
            installer.print_error("Docker is not installed")
            installer.print_info("Please install Docker from: https://docs.docker.com/engine/install/")
            raise RuntimeError("Docker is not installed")

        # Check Docker daemon
        result = installer.command_runner.run(
            ["docker", "info"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            installer.print_error("Docker daemon is not running")
            installer.print_info("Please start Docker Desktop and try again")
            installer.print_info("On macOS: Open Docker Desktop application")
            installer.print_info("On Linux: sudo systemctl start docker")
            raise RuntimeError("Docker daemon is not running")

        # Pull and run container
        installer.print_info("Pulling MySQL image...")
        installer.command_runner.run(["docker", "pull", "datajoint/mysql:8.0"], check=True)

        # Check existing container
        result = installer.command_runner.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )

        if "spyglass-db" in result.stdout:
            installer.print_warning("Container 'spyglass-db' already exists")
            installer.command_runner.run(["docker", "start", "spyglass-db"], check=True)
        else:
            installer.command_runner.run([
                "docker", "run", "-d",
                "--name", "spyglass-db",
                "-p", "3306:3306",
                "-e", "MYSQL_ROOT_PASSWORD=tutorial",
                "datajoint/mysql:8.0"
            ], check=True)

        installer.print_success("Docker database started")
        installer.create_config("localhost", "root", "tutorial", 3306)


class ExistingDatabaseStrategy(DatabaseSetupStrategy):
    """Existing database setup strategy"""

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

    def __init__(self, config: SetupConfig, colors: Optional[object] = None,
                 command_runner: Optional[CommandRunner] = None,
                 file_system: Optional[FileSystem] = None):
        self.config = config
        self.colors = colors or Colors
        self.command_runner = command_runner or DefaultCommandRunner()
        self.file_system = file_system or DefaultFileSystem()
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
        except Exception as e:
            self.print_error(f"\nSetup failed: {e}")
            return 1

    def _execute_setup_steps(self):
        """Execute all setup steps"""
        self.detect_system()
        self.check_python()
        self.check_conda()

        env_file = self.select_environment()
        self.create_environment(env_file)
        self.install_additional_deps()

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
        print(f"{self.colors.ENDC}\n")

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
            raise RuntimeError(f"Unsupported operating system: {os_name}")

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
            raise RuntimeError("No conda/mamba found")

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

    @subprocess_handler("")
    def get_command_output(self, cmd: List[str]) -> str:
        """Run command and return output safely"""
        result = self.command_runner.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()

    @lru_cache(maxsize=128)
    def get_cached_command_output(self, cmd_tuple: tuple) -> str:
        """Get cached command output"""
        return self.get_command_output(list(cmd_tuple))

    def select_environment(self) -> str:
        """Select appropriate environment file"""
        self.print_header("Environment Selection")

        env_file, description = self._select_environment_file()
        self.print_info(f"Selected: {description}")

        # Verify environment file exists
        env_path = self.config.repo_dir / env_file
        if not self.file_system.exists(env_path):
            raise FileNotFoundError(f"Environment file not found: {env_path}")

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

    def create_environment(self, env_file: str):
        """Create or update conda environment"""
        self.print_header("Creating Conda Environment")

        env_exists = self._check_environment_exists()
        if env_exists and not self._confirm_update():
            self.print_info("Keeping existing environment")
            return

        cmd = self._build_environment_command(env_file, env_exists)
        self._execute_environment_command(cmd)
        self.print_success("Environment created/updated successfully")

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

    def _execute_environment_command(self, cmd: List[str]):
        """Execute environment creation/update command with progress"""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Show progress
        for progress_line in self._filter_progress_lines(process):
            print(progress_line)

        process.wait()
        if process.returncode != 0:
            raise RuntimeError("Environment creation/update failed")

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

        result = self.command_runner.run(
            full_cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            self.print_error(f"Command failed: {' '.join(cmd)}")
            if result.stderr:
                self.print_error(result.stderr)

        return result.returncode

    def setup_database(self):
        """Setup database configuration"""
        self.print_header("Database Setup")

        strategy = self._select_database_strategy()
        strategy.setup(self)

    def _select_database_strategy(self) -> DatabaseSetupStrategy:
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
        """Create DataJoint configuration file using builder pattern"""
        self.print_info("Creating configuration file...")

        # Create base directory structure
        self._create_directory_structure()

        # Build configuration using integrated ConfigBuilder
        config = (ConfigBuilder()
                 .database(host, port, user, password)
                 .stores(self.config.base_dir)
                 .spyglass_dirs(self.config.base_dir)
                 .build())

        # Save configuration
        config_path = self.config.repo_dir / "dj_local_conf.json"
        with config_path.open('w') as f:
            json.dump(config, f, indent=4)

        self.print_success(f"Configuration file created at: {config_path}")
        self.print_success(f"Data directories created at: {self.config.base_dir}")

        # Validate configuration using SpyglassConfig
        self._validate_spyglass_config()

    def _validate_spyglass_config(self):
        """Validate the created configuration using SpyglassConfig"""
        try:
            from spyglass.settings import SpyglassConfig

            # Test if the configuration can be loaded properly
            sg_config = SpyglassConfig(base_dir=str(self.config.base_dir))
            sg_config.load_config(force_reload=True)

            # Verify all expected directories are accessible
            test_dirs = [
                sg_config.base_dir,
                sg_config.raw_dir,
                sg_config.analysis_dir,
                sg_config.recording_dir,
                sg_config.sorting_dir,
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
        self.file_system.mkdir(base_dir, exist_ok=True)

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
                self.file_system.mkdir(base_dir / subdir, exist_ok=True)

    def run_validation(self) -> int:
        """Run validation script with SpyglassConfig integration check"""
        self.print_header("Running Validation")

        # First, run a quick SpyglassConfig integration test
        self._test_spyglass_integration()

        validation_script = self.config.repo_dir / "scripts" / "validate_spyglass.py"

        if not self.file_system.exists(validation_script):
            self.print_error("Validation script not found")
            return 1

        self.print_info("Running comprehensive validation checks...")

        # Run validation in environment
        exit_code = self._run_in_env(["python", str(validation_script), "-v"])

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
        """Test SpyglassConfig integration as part of validation"""
        try:
            from spyglass.settings import SpyglassConfig

            # Quick integration test
            sg_config = SpyglassConfig(base_dir=str(self.config.base_dir))
            sg_config.load_config()

            self.print_success("SpyglassConfig integration test passed")

        except ImportError:
            self.print_warning("SpyglassConfig not available for integration test")
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
        help="Install core dependencies only (default)"
    )
    install_group.add_argument(
        "--full",
        action="store_true",
        help="Install all optional dependencies"
    )

    parser.add_argument(
        "--pipeline",
        choices=["dlc", "moseq-cpu", "moseq-gpu", "lfp", "decoding"],
        help="Install specific pipeline dependencies"
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