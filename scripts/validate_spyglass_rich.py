#!/usr/bin/env python
"""
Spyglass Installation Validator (Rich UI Version)

A comprehensive validation script with enhanced Rich UI that checks all aspects
of a Spyglass installation with beautiful progress bars, styled output, and
detailed reporting.

This is a demonstration version showing how Rich can enhance the validation experience.

Usage:
    python validate_spyglass_rich.py [options]

Requirements:
    pip install rich

Features:
    - Live progress bars for validation steps
    - Beautiful tables for results summary
    - Expandable tree view for detailed results
    - Color-coded status indicators
    - Interactive result exploration
    - Professional-looking reports
"""

import sys
import os
import platform
import importlib
import json
import warnings
from pathlib import Path
from typing import List, NamedTuple, Optional, Dict, Generator
import types
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager

# Rich imports - graceful fallback if not available
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.tree import Tree
    from rich.text import Text
    from rich.live import Live
    from rich.status import Status
    from rich.layout import Layout
    from rich.align import Align
    from rich.columns import Columns
    from rich.rule import Rule
    from rich import box
    from rich.prompt import Prompt, Confirm
    RICH_AVAILABLE = True
except ImportError:
    print("❌ Rich is not installed. Please install it with: pip install rich")
    print("   This script requires Rich for enhanced UI features.")
    print("   Use the standard validate_spyglass.py script instead, or install Rich:")
    print("   pip install rich")
    sys.exit(1)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Import shared color definitions (fallback if rich not available)
try:
    from common import Colors, DisabledColors
    PALETTE = Colors
except ImportError:
    # Fallback for demonstration
    class DummyColors:
        OKGREEN = FAIL = WARNING = HEADER = BOLD = ENDC = ""
    PALETTE = DummyColors()

# Rich console instance
console = Console()

class Severity(Enum):
    """Validation result severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ValidationResult:
    """Validation result with rich display support."""
    name: str
    passed: bool
    message: str
    severity: Severity = Severity.INFO

    def __str__(self) -> str:
        """Rich-formatted string representation."""
        if self.passed:
            return f"[bold green]✓[/bold green] {self.name}: [green]{self.message}[/green]"
        else:
            if self.severity == Severity.ERROR:
                return f"[bold red]✗[/bold red] {self.name}: [red]{self.message}[/red]"
            elif self.severity == Severity.WARNING:
                return f"[bold yellow]⚠[/bold yellow] {self.name}: [yellow]{self.message}[/yellow]"
            else:
                return f"[bold blue]ℹ[/bold blue] {self.name}: [blue]{self.message}[/blue]"


@dataclass(frozen=True)
class DependencyConfig:
    """Configuration for dependency validation."""
    name: str
    category: str
    required: bool
    import_name: str


# Dependency configurations
DEPENDENCIES = [
    # Core dependencies
    DependencyConfig("Spyglass", "Spyglass Installation", True, "spyglass"),
    DependencyConfig("DataJoint", "Spyglass Installation", True, "datajoint"),
    DependencyConfig("PyNWB", "Spyglass Installation", True, "pynwb"),
    DependencyConfig("Pandas", "Spyglass Installation", True, "pandas"),
    DependencyConfig("NumPy", "Spyglass Installation", True, "numpy"),
    DependencyConfig("Matplotlib", "Spyglass Installation", True, "matplotlib"),

    # Optional dependencies
    DependencyConfig("Spike Sorting", "Optional Dependencies", False, "spikeinterface"),
    DependencyConfig("MountainSort", "Optional Dependencies", False, "mountainsort4"),
    DependencyConfig("LFP Analysis", "Optional Dependencies", False, "ghostipy"),
    DependencyConfig("DeepLabCut", "Optional Dependencies", False, "deeplabcut"),
    DependencyConfig("Decoding (GPU)", "Optional Dependencies", False, "jax"),
    DependencyConfig("Visualization", "Optional Dependencies", False, "figurl"),
    DependencyConfig("Data Sharing", "Optional Dependencies", False, "kachery_cloud"),
]


@contextmanager
def import_module_safely(module_name: str) -> Generator[Optional[types.ModuleType], None, None]:
    """Context manager for safe module imports."""
    try:
        module = importlib.import_module(module_name)
        yield module
    except (ImportError, AttributeError, TypeError):
        yield None


class RichSpyglassValidator:
    """Rich-enhanced Spyglass installation validator."""

    def __init__(self, verbose: bool = False, config_file: str = None):
        self.verbose = verbose
        self.config_file = Path(config_file) if config_file else None
        self.results: List[ValidationResult] = []
        self.console = console

    def run_all_checks(self) -> int:
        """Run all validation checks with rich UI."""
        self.console.print(Panel(
            "[bold blue]Spyglass Installation Validator[/bold blue]\n"
            "[dim]Rich Enhanced Version[/dim]",
            title="🔍 Validation",
            box=box.DOUBLE,
            border_style="blue"
        ))
        self.console.print()

        # Define check categories with progress tracking
        check_categories = [
            ("Prerequisites", [
                ("Python Version", self.check_python_version),
                ("Operating System", self.check_operating_system),
                ("Package Manager", self.check_package_manager)
            ]),
            ("Spyglass Installation", [
                (dep.name, lambda d=dep: self.check_dependency(d))
                for dep in DEPENDENCIES if dep.required
            ]),
            ("Configuration", [
                ("DataJoint Config", self.check_datajoint_config),
                ("Spyglass Config", self.check_spyglass_config),
                ("Directory Structure", self.check_directories)
            ]),
            ("Database Connection", [
                ("Database Connection", self.check_database_connection),
            ]),
            ("Optional Dependencies", [
                (dep.name, lambda d=dep: self.check_dependency(d))
                for dep in DEPENDENCIES if not dep.required
            ])
        ]

        # Run checks with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        ) as progress:

            main_task = progress.add_task("[cyan]Running validation checks...", total=100)

            for category_name, checks in check_categories:
                # Update main progress
                progress.update(main_task, description=f"[cyan]Checking {category_name}...")

                self._run_category_checks_rich(category_name, checks, progress)

                # Advance main progress
                progress.advance(main_task, advance=100 / len(check_categories))

        self.console.print()

        # Generate and display summary
        return self.generate_rich_summary()

    def _run_category_checks_rich(self, category: str, checks: List, progress: Progress):
        """Run a category of checks with rich progress."""
        if not checks:
            return

        # Create subtask for this category
        category_task = progress.add_task(f"[yellow]{category}", total=len(checks))

        for check_name, check_func in checks:
            progress.update(category_task, description=f"[yellow]{category}: {check_name}")

            try:
                check_func()
            except Exception as e:
                self.add_result(
                    check_name,
                    False,
                    f"Check failed: {str(e)}",
                    Severity.ERROR
                )

            progress.advance(category_task)

        # Remove the subtask when done
        progress.remove_task(category_task)

    def add_result(self, name: str, passed: bool, message: str, severity: Severity = Severity.INFO):
        """Add a validation result."""
        result = ValidationResult(name, passed, message, severity)
        self.results.append(result)

        # Show result immediately if verbose
        if self.verbose:
            self.console.print(f"  {result}")

    def check_python_version(self) -> None:
        """Check Python version."""
        version = sys.version_info
        version_str = f"Python {version.major}.{version.minor}.{version.micro}"

        if version >= (3, 9):
            self.add_result("Python version", True, version_str)
        else:
            self.add_result(
                "Python version",
                False,
                f"{version_str} (need ≥3.9)",
                Severity.ERROR
            )

    def check_operating_system(self) -> None:
        """Check operating system compatibility."""
        system = platform.system()
        release = platform.release()

        if system in ["Darwin", "Linux"]:
            self.add_result("Operating System", True, f"{system} {release}")
        else:
            self.add_result(
                "Operating System",
                False,
                f"{system} may not be fully supported",
                Severity.WARNING
            )

    def check_package_manager(self) -> None:
        """Check for conda/mamba availability."""
        import shutil

        for cmd in ["mamba", "conda"]:
            if shutil.which(cmd):
                # Get version
                try:
                    import subprocess
                    result = subprocess.run([cmd, "--version"], capture_output=True, text=True)
                    version = result.stdout.strip()
                    self.add_result("Package Manager", True, f"{cmd} found: {version}")
                    return
                except Exception:
                    self.add_result("Package Manager", True, f"{cmd} found")
                    return

        self.add_result(
            "Package Manager",
            False,
            "Neither conda nor mamba found",
            Severity.ERROR
        )

    def check_dependency(self, dep: DependencyConfig) -> None:
        """Check if a dependency is available."""
        with import_module_safely(dep.import_name) as module:
            if module is not None:
                # Try to get version
                version = getattr(module, "__version__", "unknown version")
                self.add_result(dep.name, True, f"Version {version}")
            else:
                severity = Severity.ERROR if dep.required else Severity.INFO
                message = "Not installed"
                if not dep.required:
                    message += " (optional)"
                self.add_result(dep.name, False, message, severity)

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

            try:
                # Check for config files
                config_files = []
                possible_paths = [
                    Path.cwd() / "dj_local_conf.json",
                    Path.home() / ".datajoint_config.json",
                    Path.cwd() / "dj_local_conf.json"
                ]

                for path in possible_paths:
                    if path.exists():
                        config_files.append(str(path))

                if len(config_files) > 1:
                    self.add_result(
                        "Multiple Config Files",
                        False,
                        f"Found {len(config_files)} config files: {', '.join(config_files)}. Using: {config_files[0]}",
                        Severity.WARNING
                    )

                if config_files:
                    self.add_result(
                        "DataJoint Config",
                        True,
                        f"Using config file: {config_files[0]}"
                    )
                else:
                    self.add_result(
                        "DataJoint Config",
                        False,
                        "No config file found",
                        Severity.WARNING
                    )

            except Exception as e:
                self.add_result(
                    "DataJoint Config",
                    False,
                    f"Config check failed: {str(e)}",
                    Severity.ERROR
                )

    def check_spyglass_config(self) -> None:
        """Check Spyglass configuration."""
        with import_module_safely("spyglass.settings") as settings:
            if settings is None:
                self.add_result(
                    "Spyglass Config",
                    False,
                    "Cannot import SpyglassConfig",
                    Severity.ERROR
                )
                return

            try:
                with import_module_safely("datajoint") as dj:
                    if dj and hasattr(dj, 'config') and 'spyglass_dirs' in dj.config:
                        self.add_result(
                            "Spyglass Config",
                            True,
                            "spyglass_dirs found in config"
                        )
                    else:
                        self.add_result(
                            "Spyglass Config",
                            False,
                            "spyglass_dirs not found in DataJoint config",
                            Severity.WARNING
                        )
            except Exception as e:
                self.add_result(
                    "Spyglass Config",
                    False,
                    f"Config validation failed: {str(e)}",
                    Severity.ERROR
                )

    def check_directories(self) -> None:
        """Check directory structure."""
        with import_module_safely("datajoint") as dj:
            if dj is None:
                return

            try:
                spyglass_dirs = dj.config.get('spyglass_dirs', {})
                if not spyglass_dirs:
                    return

                base_dir = Path(spyglass_dirs.get('base_dir', ''))
                if base_dir.exists():
                    self.add_result("Base Directory", True, f"Found at {base_dir}")

                    # Check common subdirectories
                    subdirs = ['raw', 'analysis', 'recording', 'sorting', 'tmp']
                    for subdir in subdirs:
                        subdir_path = base_dir / subdir
                        if subdir_path.exists():
                            self.add_result(f"{subdir.title()} Directory", True, "Exists")
                        else:
                            self.add_result(
                                f"{subdir.title()} Directory",
                                False,
                                "Not found",
                                Severity.WARNING
                            )
                else:
                    self.add_result(
                        "Base Directory",
                        False,
                        f"Directory {base_dir} does not exist",
                        Severity.ERROR
                    )

            except Exception as e:
                self.add_result(
                    "Directory Check",
                    False,
                    f"Directory check failed: {str(e)}",
                    Severity.ERROR
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
                        "Cannot connect to database",
                        Severity.ERROR
                    )

            except Exception as e:
                self.add_result(
                    "Database Connection",
                    False,
                    f"Connection test failed: {str(e)}",
                    Severity.ERROR
                )

    def _check_spyglass_tables(self) -> None:
        """Check if Spyglass tables are accessible."""
        with import_module_safely("spyglass.common") as common:
            if common is None:
                return

            try:
                # Try to access a basic table
                session_table = getattr(common, 'Session', None)
                if session_table is not None:
                    # Try to describe the table (doesn't require data)
                    session_table.describe()
                    self.add_result("Spyglass Tables", True, "Can access Session table")
                else:
                    self.add_result(
                        "Spyglass Tables",
                        False,
                        "Cannot find Session table",
                        Severity.WARNING
                    )
            except Exception as e:
                self.add_result(
                    "Spyglass Tables",
                    False,
                    f"Table access failed: {str(e)}",
                    Severity.WARNING
                )

    def generate_rich_summary(self) -> int:
        """Generate rich summary with interactive exploration."""
        stats = self.get_summary_stats()

        # Create summary table
        summary_table = Table(title="Validation Summary", box=box.ROUNDED, show_header=True)
        summary_table.add_column("Metric", style="bold cyan")
        summary_table.add_column("Count", style="bold", justify="center")
        summary_table.add_column("Status", justify="center")

        total_checks = stats.get('total', 0)
        passed = stats.get('passed', 0)
        warnings = stats.get('warnings', 0)
        errors = stats.get('errors', 0)

        summary_table.add_row("Total Checks", str(total_checks), "📊")
        summary_table.add_row("Passed", str(passed), "[green]✅[/green]")

        if warnings > 0:
            summary_table.add_row("Warnings", str(warnings), "[yellow]⚠️[/yellow]")

        if errors > 0:
            summary_table.add_row("Errors", str(errors), "[red]❌[/red]")

        self.console.print(summary_table)
        self.console.print()

        # Create detailed results tree
        if self.verbose or errors > 0 or warnings > 0:
            self._show_detailed_results()

        # Overall status
        if errors > 0:
            status_panel = Panel(
                "[bold red]❌ Validation FAILED[/bold red]\n\n"
                "Please address the errors above before proceeding.\n"
                "See [link=https://lorenfranklab.github.io/spyglass/latest/notebooks/00_Setup/]setup documentation[/link] for help.",
                title="Result",
                border_style="red",
                box=box.DOUBLE
            )
            self.console.print(status_panel)
            return 2
        elif warnings > 0:
            status_panel = Panel(
                "[bold yellow]⚠️ Validation PASSED with warnings[/bold yellow]\n\n"
                "Spyglass is functional but some optional features may not work.\n"
                "Review the warnings above if you need those features.",
                title="Result",
                border_style="yellow",
                box=box.DOUBLE
            )
            self.console.print(status_panel)
            return 1
        else:
            status_panel = Panel(
                "[bold green]✅ Validation PASSED[/bold green]\n\n"
                "Spyglass is properly installed and configured!\n"
                "You can start with the tutorials in the notebooks directory.",
                title="Result",
                border_style="green",
                box=box.DOUBLE
            )
            self.console.print(status_panel)
            return 0

    def _show_detailed_results(self):
        """Show detailed results in an expandable tree."""
        # Group results by category
        categories = {}
        for result in self.results:
            # Determine category from dependency configs or result name
            category = "Other"
            for dep in DEPENDENCIES:
                if dep.name == result.name:
                    category = dep.category
                    break

            if "Prerequisites" in result.name or result.name in ["Python version", "Operating System", "Package Manager"]:
                category = "Prerequisites"
            elif "Config" in result.name or "Directory" in result.name:
                category = "Configuration"
            elif "Database" in result.name or "Tables" in result.name:
                category = "Database"

            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        # Create tree
        tree = Tree("📋 Detailed Results")

        for category, results in categories.items():
            # Determine category status
            passed_count = sum(1 for r in results if r.passed)
            total_count = len(results)

            if passed_count == total_count:
                category_icon = "[green]✅[/green]"
            elif any(r.severity == Severity.ERROR for r in results if not r.passed):
                category_icon = "[red]❌[/red]"
            else:
                category_icon = "[yellow]⚠️[/yellow]"

            category_branch = tree.add(f"{category_icon} [bold]{category}[/bold] ({passed_count}/{total_count})")

            for result in results:
                category_branch.add(str(result))

        self.console.print(tree)
        self.console.print()

    def get_summary_stats(self) -> Dict[str, int]:
        """Get validation summary statistics."""
        from collections import Counter
        stats = Counter(total=len(self.results))

        for result in self.results:
            if result.passed:
                stats['passed'] += 1
            else:
                if result.severity == Severity.ERROR:
                    stats['errors'] += 1
                elif result.severity == Severity.WARNING:
                    stats['warnings'] += 1

        return dict(stats)


def main() -> None:
    """Execute the rich validation script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate Spyglass installation and configuration (Rich UI Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_spyglass_rich.py                    # Basic validation
  python validate_spyglass_rich.py -v                 # Verbose output
  python validate_spyglass_rich.py --config-file ./my_config.json  # Custom config
        """
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show all checks, not just failures"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to DataJoint config file (overrides default search)"
    )

    args = parser.parse_args()

    try:
        validator = RichSpyglassValidator(verbose=args.verbose, config_file=args.config_file)
        exit_code = validator.run_all_checks()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]Validation interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Validation failed with error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()