#!/usr/bin/env python
"""
Spyglass Quickstart Setup Script (Rich UI Version)

A comprehensive setup script with enhanced Rich UI that automates the Spyglass
installation process with beautiful progress bars, styled output, and interactive elements.

This is a demonstration version showing how Rich can enhance the user experience.

Usage:
    python test_quickstart_rich.py [options]

Requirements:
    pip install rich

Features:
    - Animated progress bars for long operations
    - Styled console output with colors and formatting
    - Interactive menus with keyboard navigation
    - Live status updates during installation
    - Beautiful tables for system information
    - Spinners for background operations
"""

import sys
import platform
import subprocess
import shutil
import argparse
import time
import json
import getpass
from pathlib import Path
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass, replace
from enum import Enum
from contextlib import contextmanager

# Rich imports - graceful fallback if not available
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.tree import Tree
    from rich.text import Text
    from rich.live import Live
    from rich.status import Status
    from rich.layout import Layout
    from rich.align import Align
    from rich.columns import Columns
    from rich.rule import Rule
    from rich import box
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    print("❌ Rich is not installed. Please install it with: pip install rich")
    print("   This script requires Rich for enhanced UI features.")
    print("   Use the standard quickstart.py script instead, or install Rich:")
    print("   pip install rich")
    sys.exit(1)

# Import shared utilities
from common import (
    SpyglassSetupError, SystemRequirementError,
    EnvironmentCreationError, DatabaseSetupError,
    MenuChoice, DatabaseChoice, ConfigLocationChoice, PipelineChoice
)

# Rich console instance
console = Console()

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


class RichUserInterface:
    """Rich-enhanced user interface for the quickstart script."""

    def __init__(self, auto_yes: bool = False):
        self.auto_yes = auto_yes
        self.console = console

    def print_banner(self):
        """Display a beautiful banner."""
        banner_text = """
[bold blue]╔═══════════════════════════════════════╗[/bold blue]
[bold blue]║     [bold white]Spyglass Quickstart Installer[/bold white]    ║[/bold blue]
[bold blue]║           [dim]Rich Enhanced Version[/dim]         ║[/bold blue]
[bold blue]╚═══════════════════════════════════════╝[/bold blue]
        """
        panel = Panel(
            Align.center(banner_text),
            box=box.DOUBLE,
            style="bold blue",
            padding=(1, 2)
        )
        self.console.print(panel)
        self.console.print()

    def print_section_header(self, title: str, description: str = ""):
        """Print a styled section header."""
        rule = Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan")
        self.console.print(rule)
        if description:
            self.console.print(f"[dim]{description}[/dim]")
            self.console.print()

    def print_system_info(self, info: SystemInfo):
        """Display system information in a beautiful table."""
        table = Table(title="System Information", box=box.ROUNDED)
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Status", style="bold")

        table.add_row("Operating System", info.os_name, "✅ Detected")
        table.add_row("Architecture", info.arch, "✅ Compatible" if info.arch in ["x86_64", "arm64"] else "⚠️ Unknown")

        python_ver = f"{info.python_version[0]}.{info.python_version[1]}.{info.python_version[2]}"
        table.add_row("Python Version", python_ver, "✅ Compatible" if info.python_version >= (3, 9) else "❌ Too Old")

        conda_status = "✅ Found" if info.conda_cmd else "❌ Not Found"
        conda_name = info.conda_cmd or "Not Available"
        table.add_row("Package Manager", conda_name, conda_status)

        if info.is_m1:
            table.add_row("Apple Silicon", "M1/M2/M3 Detected", "🚀 Optimized")

        self.console.print(table)
        self.console.print()

    def select_install_type(self) -> Tuple[InstallType, Optional[Pipeline]]:
        """Let user select installation type with rich interface."""
        if self.auto_yes:
            self.console.print("[yellow]Auto-accepting minimal installation (--yes mode)[/yellow]")
            return InstallType.MINIMAL, None

        choices = [
            ("[bold green]1[/bold green]", "Minimal", "Core dependencies only", "🚀 Fastest (~5-10 min)"),
            ("[bold blue]2[/bold blue]", "Full", "All optional dependencies", "📦 Complete (~15-30 min)"),
            ("[bold magenta]3[/bold magenta]", "Pipeline", "Specific analysis pipeline", "🎯 Targeted (~10-20 min)")
        ]

        table = Table(title="Choose Installation Type", box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("Choice", style="bold", width=8)
        table.add_column("Type", style="bold", width=12)
        table.add_column("Description", width=30)
        table.add_column("Duration", style="dim", width=20)

        for choice, type_name, desc, duration in choices:
            table.add_row(choice, type_name, desc, duration)

        self.console.print(table)
        self.console.print()

        choice = Prompt.ask(
            "[bold cyan]Enter your choice[/bold cyan]",
            choices=["1", "2", "3"],
            default="1"
        )

        if choice == "1":
            return InstallType.MINIMAL, None
        elif choice == "2":
            return InstallType.FULL, None
        else:
            return InstallType.FULL, self._select_pipeline()

    def _select_pipeline(self) -> Pipeline:
        """Select specific pipeline with rich interface."""
        pipelines = [
            ("1", "DeepLabCut", "Pose estimation and behavior analysis", "🐭"),
            ("2", "Keypoint-Moseq (CPU)", "Behavioral sequence analysis", "💻"),
            ("3", "Keypoint-Moseq (GPU)", "GPU-accelerated behavioral analysis", "🚀"),
            ("4", "LFP Analysis", "Local field potential processing", "📈"),
            ("5", "Decoding", "Neural population decoding", "🧠")
        ]

        table = Table(title="Select Pipeline", box=box.ROUNDED)
        table.add_column("Choice", style="bold cyan", width=8)
        table.add_column("Pipeline", style="bold", width=25)
        table.add_column("Description", width=35)
        table.add_column("", width=5)

        for choice, name, desc, emoji in pipelines:
            table.add_row(f"[bold]{choice}[/bold]", name, desc, emoji)

        self.console.print(table)
        self.console.print()

        choice = Prompt.ask(
            "[bold cyan]Select pipeline[/bold cyan]",
            choices=["1", "2", "3", "4", "5"],
            default="1"
        )

        pipeline_map = {
            "1": Pipeline.DLC,
            "2": Pipeline.MOSEQ_CPU,
            "3": Pipeline.MOSEQ_GPU,
            "4": Pipeline.LFP,
            "5": Pipeline.DECODING
        }

        return pipeline_map[choice]

    def confirm_environment_update(self, env_name: str) -> bool:
        """Rich confirmation dialog for environment updates."""
        if self.auto_yes:
            self.console.print(f"[yellow]Auto-accepting environment update for '{env_name}' (--yes mode)[/yellow]")
            return True

        panel = Panel(
            f"[yellow]Environment '[bold]{env_name}[/bold]' already exists[/yellow]\n\n"
            "Would you like to update it with the latest packages?\n"
            "[dim]This will preserve your existing packages and add new ones.[/dim]",
            title="Environment Exists",
            border_style="yellow",
            box=box.ROUNDED
        )
        self.console.print(panel)

        return Confirm.ask("[bold cyan]Update environment?[/bold cyan]", default=False)

    def get_database_credentials(self) -> Tuple[str, int, str, str]:
        """Get database credentials with rich interface."""
        panel = Panel(
            "[cyan]Enter database connection details[/cyan]\n"
            "[dim]These will be used to connect to your existing database.[/dim]",
            title="Database Configuration",
            border_style="cyan",
            box=box.ROUNDED
        )
        self.console.print(panel)

        host = Prompt.ask("[bold]Database host[/bold]", default="localhost")
        port = IntPrompt.ask("[bold]Database port[/bold]", default=3306)
        user = Prompt.ask("[bold]Database user[/bold]", default="root")

        # Use rich's hidden input for password
        password = Prompt.ask("[bold]Database password[/bold]", password=True)

        return host, port, user, password

    def select_database_option(self) -> DatabaseChoice:
        """Select database setup option with rich interface."""
        if self.auto_yes:
            self.console.print("[yellow]Auto-selecting Docker database (--yes mode)[/yellow]")
            return DatabaseChoice.DOCKER

        choices = [
            ("1", "🐳 Local Docker Database", "Recommended for beginners", "Sets up MySQL in Docker"),
            ("2", "🔗 Existing Database", "Connect to existing MySQL", "Requires database credentials"),
            ("3", "⏭️  Skip Database Setup", "Configure manually later", "You'll need to set up database yourself")
        ]

        table = Table(title="Database Setup Options", box=box.ROUNDED)
        table.add_column("Choice", style="bold", width=8)
        table.add_column("Option", style="bold", width=30)
        table.add_column("Best For", style="dim", width=25)
        table.add_column("Description", width=35)

        for choice, option, best_for, desc in choices:
            table.add_row(f"[bold cyan]{choice}[/bold cyan]", option, best_for, desc)

        self.console.print(table)

        choice = Prompt.ask(
            "[bold cyan]Choose database setup[/bold cyan]",
            choices=["1", "2", "3"],
            default="1"
        )

        choice_map = {
            "1": DatabaseChoice.DOCKER,
            "2": DatabaseChoice.EXISTING,
            "3": DatabaseChoice.SKIP
        }

        return choice_map[choice]

    def show_progress_with_live_updates(self, title: str, steps: List[str]):
        """Show progress with live updates using Rich."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:

            task = progress.add_task(title, total=len(steps))

            for i, step in enumerate(steps):
                progress.update(task, description=f"[cyan]{step}[/cyan]")

                # Simulate work (replace with actual operations)
                time.sleep(1)

                progress.advance(task)

    def show_environment_creation_progress(self, env_file: str):
        """Show environment creation with rich progress."""
        steps = [
            "Reading environment file",
            "Resolving dependencies",
            "Downloading packages",
            "Installing packages",
            "Configuring environment"
        ]

        self.show_progress_with_live_updates(f"Creating environment from {env_file}", steps)

    def show_installation_summary(self, config: SetupConfig, success: bool = True):
        """Display installation summary with rich formatting."""
        if success:
            title = "[bold green]🎉 Installation Complete![/bold green]"
            border_style = "green"
        else:
            title = "[bold red]❌ Installation Failed[/bold red]"
            border_style = "red"

        # Create summary content
        summary_items = [
            f"[bold]Installation Type:[/bold] {config.install_type.value.title()}",
            f"[bold]Environment:[/bold] {config.env_name}",
            f"[bold]Base Directory:[/bold] {config.base_dir}",
            f"[bold]Database Setup:[/bold] {'Configured' if config.setup_database else 'Skipped'}",
        ]

        if config.pipeline:
            summary_items.insert(1, f"[bold]Pipeline:[/bold] {config.pipeline.value.upper()}")

        summary_text = "\n".join(summary_items)

        if success:
            next_steps = """
[bold cyan]Next Steps:[/bold cyan]

1. [bold]Activate environment:[/bold]
   [dim]conda activate spyglass[/dim]

2. [bold]Test installation:[/bold]
   [dim]python -c "import spyglass; print('✅ Success!')"[/dim]

3. [bold]Explore tutorials:[/bold]
   [dim]cd notebooks && jupyter notebook[/dim]
            """
            content = summary_text + next_steps
        else:
            content = summary_text + "\n\n[red]Please check the errors above and try again.[/red]"

        panel = Panel(
            content,
            title=title,
            border_style=border_style,
            box=box.DOUBLE,
            padding=(1, 2)
        )

        self.console.print(panel)

    def print_success(self, message: str):
        """Print success message with rich styling."""
        self.console.print(f"[bold green]✅ {message}[/bold green]")

    def print_info(self, message: str):
        """Print info message with rich styling."""
        self.console.print(f"[bold blue]ℹ {message}[/bold blue]")

    def print_warning(self, message: str):
        """Print warning message with rich styling."""
        self.console.print(f"[bold yellow]⚠ {message}[/bold yellow]")

    def print_error(self, message: str):
        """Print error message with rich styling."""
        self.console.print(f"[bold red]❌ {message}[/bold red]")


class SystemDetector:
    """System detection and validation."""

    def __init__(self, ui: RichUserInterface):
        self.ui = ui

    def detect_system(self) -> Optional[SystemInfo]:
        """Detect system information."""
        try:
            # Get OS information
            system = platform.system()
            machine = platform.machine()

            # Normalize OS name
            os_name = {
                "Darwin": "macOS",
                "Linux": "Linux",
                "Windows": "Windows"
            }.get(system, system)

            # Check for Apple Silicon
            is_m1 = system == "Darwin" and machine == "arm64"

            # Get Python version
            python_version = sys.version_info[:3]

            # Find conda command
            conda_cmd = self._find_conda_command()

            return SystemInfo(
                os_name=os_name,
                arch=machine,
                is_m1=is_m1,
                python_version=python_version,
                conda_cmd=conda_cmd
            )

        except Exception as e:
            self.ui.print_error(f"Failed to detect system: {e}")
            return None

    def _find_conda_command(self) -> Optional[str]:
        """Find available conda command (prefer mamba)."""
        for cmd in ["mamba", "conda"]:
            if shutil.which(cmd):
                return cmd
        return None

    def validate_system(self, info: SystemInfo) -> bool:
        """Validate system requirements."""
        issues = []

        # Check Python version
        if info.python_version < (3, 9):
            issues.append(f"Python {info.python_version[0]}.{info.python_version[1]} is too old (need ≥3.9)")

        # Check conda
        if not info.conda_cmd:
            issues.append("No conda/mamba found - please install Miniconda or Mambaforge")

        # Check OS support
        if info.os_name not in ["macOS", "Linux"]:
            issues.append(f"Operating system '{info.os_name}' is not fully supported")

        if issues:
            for issue in issues:
                self.ui.print_error(issue)
            return False

        return True


def main():
    """Main entry point for the rich quickstart script."""
    parser = argparse.ArgumentParser(
        description="Spyglass Quickstart Installer (Rich UI Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_quickstart_rich.py                    # Interactive installation
  python test_quickstart_rich.py --minimal          # Minimal installation
  python test_quickstart_rich.py --full --yes       # Full automated installation
  python test_quickstart_rich.py --pipeline=dlc     # DeepLabCut pipeline
        """
    )

    # Installation type options
    install_group = parser.add_mutually_exclusive_group()
    install_group.add_argument("--minimal", action="store_true", help="Install minimal dependencies")
    install_group.add_argument("--full", action="store_true", help="Install all dependencies")
    install_group.add_argument("--pipeline", choices=["dlc", "moseq-cpu", "moseq-gpu", "lfp", "decoding"],
                              help="Install specific pipeline")

    # Setup options
    parser.add_argument("--no-database", action="store_true", help="Skip database setup")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    parser.add_argument("--yes", action="store_true", help="Auto-accept all prompts")
    parser.add_argument("--base-dir", type=str, help="Base directory for data")
    parser.add_argument("--env-name", type=str, default="spyglass", help="Conda environment name")

    args = parser.parse_args()

    # Create UI
    ui = RichUserInterface(auto_yes=args.yes)

    try:
        # Show banner
        ui.print_banner()

        # System detection
        ui.print_section_header("System Detection", "Analyzing your system configuration")

        detector = SystemDetector(ui)
        with console.status("[bold green]Detecting system configuration...", spinner="dots"):
            time.sleep(2)  # Simulate detection time
            system_info = detector.detect_system()

        if not system_info:
            ui.print_error("Failed to detect system information")
            return 1

        ui.print_system_info(system_info)

        if not detector.validate_system(system_info):
            ui.print_error("System requirements not met")
            return 1

        # Installation type selection
        ui.print_section_header("Installation Configuration")

        if args.minimal:
            install_type, pipeline = InstallType.MINIMAL, None
            ui.print_info("Using minimal installation (from command line)")
        elif args.full:
            install_type, pipeline = InstallType.FULL, None
            ui.print_info("Using full installation (from command line)")
        elif args.pipeline:
            install_type = InstallType.FULL
            pipeline_map = {
                "dlc": Pipeline.DLC,
                "moseq-cpu": Pipeline.MOSEQ_CPU,
                "moseq-gpu": Pipeline.MOSEQ_GPU,
                "lfp": Pipeline.LFP,
                "decoding": Pipeline.DECODING
            }
            pipeline = pipeline_map[args.pipeline]
            ui.print_info(f"Using {args.pipeline.upper()} pipeline (from command line)")
        else:
            install_type, pipeline = ui.select_install_type()

        # Create configuration
        config = SetupConfig(
            install_type=install_type,
            pipeline=pipeline,
            setup_database=not args.no_database,
            run_validation=not args.no_validate,
            base_dir=Path(args.base_dir) if args.base_dir else Path.home() / "spyglass_data",
            env_name=args.env_name,
            auto_yes=args.yes
        )

        # Environment creation demo
        ui.print_section_header("Environment Creation")
        env_file = "environment.yml" if install_type == InstallType.FULL else "environment-min.yml"

        if ui.confirm_environment_update(config.env_name):
            ui.show_environment_creation_progress(env_file)
            ui.print_success(f"Environment '{config.env_name}' created successfully")

        # Database setup demo
        if config.setup_database:
            ui.print_section_header("Database Configuration")
            db_choice = ui.select_database_option()

            if db_choice == DatabaseChoice.EXISTING:
                credentials = ui.get_database_credentials()
                ui.print_info(f"Database configured for {credentials[0]}:{credentials[1]}")
            elif db_choice == DatabaseChoice.DOCKER:
                with console.status("[bold blue]Setting up Docker database...", spinner="bouncingBar"):
                    time.sleep(3)  # Simulate Docker setup
                ui.print_success("Docker database started successfully")
            else:
                ui.print_info("Database setup skipped")

        # Validation demo
        if config.run_validation:
            ui.print_section_header("Validation")
            with console.status("[bold green]Running validation checks...", spinner="arrow3"):
                time.sleep(2)  # Simulate validation
            ui.print_success("All validation checks passed!")

        # Success summary
        ui.show_installation_summary(config, success=True)

        return 0

    except KeyboardInterrupt:
        ui.print_error("\nSetup interrupted by user")
        return 130
    except Exception as e:
        ui.print_error(f"Setup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())