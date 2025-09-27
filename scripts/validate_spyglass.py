#!/usr/bin/env python
"""
Spyglass Installation Validator.

This script validates that Spyglass is properly installed and configured.
It checks prerequisites, core functionality, database connectivity, and
optional dependencies without requiring any data files.

Exit codes:
    0: Success - all checks passed
    1: Warning - setup complete but with warnings
    2: Failure - critical issues found
"""

import sys
import platform
import subprocess
import importlib
import json
from pathlib import Path
from typing import List, NamedTuple, Optional, Dict, Generator
import types
from dataclasses import dataclass
from collections import Counter
from enum import Enum
from contextlib import contextmanager
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Import shared color definitions
from common import Colors, DisabledColors

# Global color palette that can be modified for --no-color support
PALETTE = Colors


class Severity(Enum):
    """Validation result severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ValidationResult:
    """Store validation results for a single check."""
    name: str
    passed: bool
    message: str
    severity: Severity = Severity.ERROR

    def __str__(self) -> str:
        status_symbols = {
            (True, None): f"{PALETTE.OKGREEN}✓{PALETTE.ENDC}",
            (False, Severity.WARNING): f"{PALETTE.WARNING}⚠{PALETTE.ENDC}",
            (False, Severity.ERROR): f"{PALETTE.FAIL}✗{PALETTE.ENDC}",
            (False, Severity.INFO): f"{PALETTE.OKCYAN}ℹ{PALETTE.ENDC}",
        }

        status_key = (self.passed, None if self.passed else self.severity)
        status = status_symbols.get(status_key, status_symbols[(False, Severity.ERROR)])

        return f"  {status} {self.name}: {self.message}"


class DependencyConfig(NamedTuple):
    """Configuration for a dependency check."""
    module: str
    display_name: str
    required: bool = True
    category: str = "core"


# Centralized dependency configuration
DEPENDENCIES = [
    # Core dependencies
    DependencyConfig("datajoint", "DataJoint", True, "core"),
    DependencyConfig("pynwb", "PyNWB", True, "core"),
    DependencyConfig("pandas", "Pandas", True, "core"),
    DependencyConfig("numpy", "NumPy", True, "core"),
    DependencyConfig("matplotlib", "Matplotlib", True, "core"),

    # Optional dependencies
    DependencyConfig("spikeinterface", "Spike Sorting", False, "spikesorting"),
    DependencyConfig("mountainsort4", "MountainSort", False, "spikesorting"),
    DependencyConfig("ghostipy", "LFP Analysis", False, "lfp"),
    DependencyConfig("deeplabcut", "DeepLabCut", False, "position"),
    DependencyConfig("jax", "Decoding (GPU)", False, "decoding"),
    DependencyConfig("figurl", "Visualization", False, "visualization"),
    DependencyConfig("kachery_cloud", "Data Sharing", False, "sharing"),
]


@contextmanager
def import_module_safely(module_name: str) -> Generator[Optional[types.ModuleType], None, None]:
    """Context manager for safe module imports."""
    try:
        module = importlib.import_module(module_name)
        yield module
    except (ImportError, AttributeError, TypeError):
        yield None


class SpyglassValidator:
    """Main validator class for Spyglass installation."""

    def __init__(self, verbose: bool = False, config_file: Optional[str] = None) -> None:
        self.verbose = verbose
        self.config_file = Path(config_file) if config_file else None
        self.results: List[ValidationResult] = []

    def run_all_checks(self) -> int:
        """Run all validation checks and return exit code."""
        print(f"\n{PALETTE.HEADER}{PALETTE.BOLD}Spyglass Installation Validator{PALETTE.ENDC}")
        print("=" * 50)

        # Check prerequisites
        self._run_category_checks("Prerequisites", [
            self.check_python_version,
            self.check_platform,
            self.check_conda_mamba,
        ])

        # Check Spyglass installation
        self._run_category_checks("Spyglass Installation", [
            self.check_spyglass_import,
            lambda: self.check_dependencies("core"),
        ])

        # Check configuration
        self._run_category_checks("Configuration", [
            self.check_datajoint_config,
            self.check_directories,
        ])

        # Check database
        self._run_category_checks("Database Connection", [
            self.check_database_connection,
        ])

        # Check optional dependencies
        self._run_category_checks("Optional Dependencies", [
            lambda: self.check_dependencies(None, required_only=False),
        ])

        # Generate summary
        return self.generate_summary()

    def _run_category_checks(self, category: str, checks: List) -> None:
        """Run a category of checks."""
        print(f"\n{PALETTE.OKCYAN}Checking {category}...{PALETTE.ENDC}")
        for check in checks:
            check()

    def check_python_version(self) -> None:
        """Check Python version is >= 3.9."""
        version = sys.version_info
        version_str = f"Python {version.major}.{version.minor}.{version.micro}"

        if version >= (3, 9):
            self.add_result("Python version", True, version_str)
        else:
            self.add_result(
                "Python version",
                False,
                f"{version_str} found, need >= 3.9",
                Severity.ERROR
            )

    def check_platform(self) -> None:
        """Check operating system compatibility."""
        system = platform.system()
        platform_info = f"{system} {platform.release()}"

        if system in ["Darwin", "Linux"]:
            self.add_result("Operating System", True, platform_info)
        elif system == "Windows":
            self.add_result(
                "Operating System",
                False,
                "Windows is not officially supported",
                Severity.WARNING
            )
        else:
            self.add_result(
                "Operating System",
                False,
                f"Unknown OS: {system}",
                Severity.ERROR
            )

    def check_conda_mamba(self) -> None:
        """Check if conda or mamba is available."""
        for cmd in ["mamba", "conda"]:
            try:
                result = subprocess.run(
                    [cmd, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    version = result.stdout.strip()
                    self.add_result("Package Manager", True, f"{cmd} found: {version}")
                    return
            except (subprocess.SubprocessError, FileNotFoundError):
                continue

        self.add_result(
            "Package Manager",
            False,
            "Neither mamba nor conda found in PATH",
            Severity.WARNING
        )

    def check_spyglass_import(self) -> bool:
        """Check if Spyglass can be imported."""
        with import_module_safely("spyglass") as spyglass:
            if spyglass:
                version = getattr(spyglass, "__version__", "unknown")
                self.add_result("Spyglass Import", True, f"Version {version}")
                return True
            else:
                self.add_result(
                    "Spyglass Import",
                    False,
                    "Cannot import spyglass",
                    Severity.ERROR
                )
                return False

    def check_dependencies(self, category: Optional[str] = None, required_only: bool = True) -> None:
        """Check dependencies, optionally filtered by category."""
        deps = DEPENDENCIES

        if category:
            deps = [d for d in deps if d.category == category]

        if required_only:
            deps = [d for d in deps if d.required]
        else:
            deps = [d for d in deps if not d.required]

        for dep in deps:
            with import_module_safely(dep.module) as mod:
                if mod:
                    version = getattr(mod, "__version__", "unknown")
                    self.add_result(dep.display_name, True, f"Version {version}")
                else:
                    severity = Severity.ERROR if dep.required else Severity.INFO
                    suffix = "" if dep.required else " (optional)"
                    self.add_result(
                        dep.display_name,
                        False,
                        f"Not installed{suffix}",
                        severity
                    )

    def check_datajoint_config(self) -> None:
        """Check DataJoint configuration."""
        with import_module_safely("datajoint") as dj:
            if dj is None:
                self.add_result(
                    "DataJoint Config",
                    False,
                    "DataJoint not installed",
                    Severity.ERROR
                )
                return

            config_file = self._find_config_file()
            if config_file:
                self.add_result("DataJoint Config", True, f"Using config file: {config_file}")
                self._validate_config_file(config_file)
            else:
                if self.config_file:
                    # Explicitly specified config file not found
                    self.add_result(
                        "DataJoint Config",
                        False,
                        f"Specified config file not found: {self.config_file}",
                        Severity.WARNING
                    )
                else:
                    # Show where we looked for config files
                    search_locations = [
                        "DJ_CONFIG_FILE environment variable",
                        "./dj_local_conf.json (current directory)",
                        "~/.datajoint_config.json (home directory)"
                    ]
                    self.add_result(
                        "DataJoint Config",
                        False,
                        f"No config file found. Searched: {', '.join(search_locations)}. Use --config-file to specify location.",
                        Severity.WARNING
                    )

    def _find_config_file(self) -> Optional[Path]:
        """Find DataJoint config file and warn about multiple files."""
        import os

        # If config file explicitly specified, use it
        if self.config_file:
            return self.config_file if self.config_file.exists() else None

        candidates = []

        # Environment variable override (if set)
        dj_config_env = os.environ.get("DJ_CONFIG_FILE", "").strip()
        if dj_config_env:
            candidates.append(Path(dj_config_env))

        # Standard locations
        candidates.extend([
            # Current working directory (quickstart default)
            Path.cwd() / "dj_local_conf.json",
            # Home directory default
            Path.home() / ".datajoint_config.json",
            # Repo root fallback (for quickstart-generated configs)
            Path(__file__).resolve().parent.parent / "dj_local_conf.json",
        ])

        # Find existing files
        existing_files = [p for p in candidates if p.exists()]

        if len(existing_files) > 1:
            # Warn about multiple config files
            self.add_result(
                "Multiple Config Files",
                False,
                f"Found {len(existing_files)} config files: {', '.join(str(f) for f in existing_files)}. Using: {existing_files[0]}",
                Severity.WARNING
            )

        return existing_files[0] if existing_files else None

    def _validate_config_file(self, config_path: Path) -> None:
        """Validate the contents of a config file."""
        try:
            config = json.loads(config_path.read_text())
            if 'custom' in config and 'spyglass_dirs' in config['custom']:
                self.add_result(
                    "Spyglass Config",
                    True,
                    "spyglass_dirs found in config"
                )
            else:
                self.add_result(
                    "Spyglass Config",
                    False,
                    "spyglass_dirs not found in config",
                    Severity.WARNING
                )
        except (json.JSONDecodeError, OSError) as e:
            self.add_result(
                "Config Parse",
                False,
                f"Invalid config: {e}",
                Severity.ERROR
            )

    def check_directories(self) -> None:
        """Check if Spyglass directories are configured and accessible."""
        with import_module_safely("spyglass.settings") as settings_module:
            if settings_module is None:
                self.add_result(
                    "Directory Check",
                    False,
                    "Cannot import SpyglassConfig",
                    Severity.ERROR
                )
                return

            try:
                config = settings_module.SpyglassConfig()
                base_dir = config.base_dir

                if base_dir and Path(base_dir).exists():
                    self.add_result("Base Directory", True, f"Found at {base_dir}")
                    self._check_subdirectories(Path(base_dir))
                else:
                    self.add_result(
                        "Base Directory",
                        False,
                        "Not found or not configured",
                        Severity.WARNING
                    )
            except (OSError, PermissionError, ValueError) as e:
                self.add_result(
                    "Directory Check",
                    False,
                    f"Error: {str(e)}",
                    Severity.ERROR
                )

    def _check_subdirectories(self, base_dir: Path) -> None:
        """Check standard Spyglass subdirectories."""
        subdirs = ['raw', 'analysis', 'recording', 'sorting', 'tmp']

        for subdir in subdirs:
            dir_path = base_dir / subdir
            if dir_path.exists():
                self.add_result(
                    f"{subdir.capitalize()} Directory",
                    True,
                    "Exists",
                    Severity.INFO
                )
            else:
                self.add_result(
                    f"{subdir.capitalize()} Directory",
                    False,
                    "Not found (will be created on first use)",
                    Severity.INFO
                )

    def check_database_connection(self) -> None:
        """Check database connectivity."""
        with import_module_safely("datajoint") as dj:
            if dj is None:
                self.add_result(
                    "Database Connection",
                    False,
                    "DataJoint not installed",
                    Severity.WARNING
                )
                return

            try:
                connection = dj.conn(reset=False)
                if connection.is_connected:
                    # Get connection info from dj.config instead of connection object
                    host = dj.config.get('database.host', 'unknown')
                    port = dj.config.get('database.port', 'unknown')
                    user = dj.config.get('database.user', 'unknown')
                    host_port = f"{host}:{port}"
                    self.add_result(
                        "Database Connection",
                        True,
                        f"Connected to {host_port} as {user}"
                    )
                    self._check_spyglass_tables()
                else:
                    self.add_result(
                        "Database Connection",
                        False,
                        "Not connected",
                        Severity.WARNING
                    )
            except (ConnectionError, OSError, TimeoutError) as e:
                self.add_result(
                    "Database Connection",
                    False,
                    f"Cannot connect: {str(e)}",
                    Severity.WARNING
                )

    def _check_spyglass_tables(self) -> None:
        """Check if Spyglass tables are accessible."""
        with import_module_safely("spyglass.common") as common:
            if common:
                try:
                    common.Session()
                    self.add_result(
                        "Spyglass Tables",
                        True,
                        "Can access Session table"
                    )
                except (AttributeError, ImportError, ConnectionError) as e:
                    self.add_result(
                        "Spyglass Tables",
                        False,
                        f"Cannot access tables: {str(e)}",
                        Severity.WARNING
                    )

    def add_result(self, name: str, passed: bool, message: str,
                   severity: Severity = Severity.ERROR) -> None:
        """Add a validation result."""
        result = ValidationResult(name, passed, message, severity)
        self.results.append(result)

        if self.verbose or not passed:
            print(result)

    def get_summary_stats(self) -> Dict[str, int]:
        """Get validation summary statistics."""
        stats = Counter(total=len(self.results))

        for result in self.results:
            if result.passed:
                stats['passed'] += 1
            else:
                stats[result.severity.value] += 1

        return dict(stats)

    def generate_summary(self) -> int:
        """Generate summary report and return exit code."""
        print(f"\n{PALETTE.HEADER}{PALETTE.BOLD}Validation Summary{PALETTE.ENDC}")
        print("=" * 50)

        stats = self.get_summary_stats()

        print(f"\nTotal checks: {stats.get('total', 0)}")
        print(f"  {PALETTE.OKGREEN}Passed: {stats.get('passed', 0)}{PALETTE.ENDC}")

        warnings = stats.get('warning', 0)
        if warnings > 0:
            print(f"  {PALETTE.WARNING}Warnings: {warnings}{PALETTE.ENDC}")

        errors = stats.get('error', 0)
        if errors > 0:
            print(f"  {PALETTE.FAIL}Errors: {errors}{PALETTE.ENDC}")

        # Determine exit code and final message
        if errors > 0:
            print(f"\n{PALETTE.FAIL}{PALETTE.BOLD}❌ Validation FAILED{PALETTE.ENDC}")
            print("\nPlease address the errors above before proceeding.")
            print("See https://lorenfranklab.github.io/spyglass/latest/notebooks/00_Setup/")
            return 2
        elif warnings > 0:
            print(f"\n{PALETTE.WARNING}{PALETTE.BOLD}⚠️  Validation PASSED with warnings{PALETTE.ENDC}")
            print("\nSpyglass is functional but some optional features may not work.")
            print("Review the warnings above if you need those features.")
            return 1
        else:
            print(f"\n{PALETTE.OKGREEN}{PALETTE.BOLD}✅ Validation PASSED{PALETTE.ENDC}")
            print("\nSpyglass is properly installed and configured!")
            print("You can start with the tutorials in the notebooks directory.")
            return 0


def main() -> None:
    """Execute the validation script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate Spyglass installation and configuration"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show all checks, not just failures"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to DataJoint config file (overrides default search)"
    )

    args = parser.parse_args()

    # Apply --no-color flag
    global PALETTE
    if args.no_color:
        PALETTE = DisabledColors

    validator = SpyglassValidator(verbose=args.verbose, config_file=args.config_file)
    exit_code = validator.run_all_checks()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()