#!/usr/bin/env python
"""
Demo script to test rich functionality without requiring actual installation.

This script demonstrates the Rich UI components used in the enhanced scripts.

Usage:
    python demo_rich.py          # Interactive mode (waits for Enter between demos)
    python demo_rich.py --auto   # Auto mode (runs all demos continuously)

Requirements:
    pip install rich
"""

import time
import sys
from pathlib import Path

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
except ImportError:
    print("❌ Rich is not installed. Please install it with: pip install rich")
    sys.exit(1)

console = Console()


def demo_banner():
    """Demonstrate rich banner."""
    console.print("\n[bold cyan]🎨 Rich UI Demo - Banner Example[/bold cyan]\n")

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
    console.print(panel)


def demo_system_info():
    """Demonstrate system information table."""
    console.print("\n[bold cyan]📊 System Information Table Example[/bold cyan]\n")

    table = Table(title="System Information", box=box.ROUNDED)
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    table.add_column("Status", style="bold")

    table.add_row("Operating System", "macOS", "✅ Detected")
    table.add_row("Architecture", "arm64", "✅ Compatible")
    table.add_row("Python Version", "3.10.18", "✅ Compatible")
    table.add_row("Package Manager", "conda", "✅ Found")
    table.add_row("Apple Silicon", "M1/M2/M3 Detected", "🚀 Optimized")

    console.print(table)


def demo_installation_menu():
    """Demonstrate installation type selection."""
    console.print("\n[bold cyan]🎯 Installation Menu Example[/bold cyan]\n")

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

    console.print(table)


def demo_progress_bars():
    """Demonstrate various progress bar styles."""
    console.print("\n[bold cyan]⏳ Progress Bar Examples[/bold cyan]\n")

    # Standard progress bar
    console.print("[bold]Standard Progress Bar:[/bold]")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Processing...", total=100)
        for i in range(100):
            time.sleep(0.02)
            progress.advance(task)

    console.print()

    # Spinner with progress
    console.print("[bold]Spinner with Progress:[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[green]Installing packages...", total=50)
        for i in range(50):
            time.sleep(0.05)
            progress.advance(task)


def demo_status_indicators():
    """Demonstrate status indicators and spinners."""
    console.print("\n[bold cyan]🔄 Status Indicators Example[/bold cyan]\n")

    with console.status("[bold green]Detecting system configuration...", spinner="dots"):
        time.sleep(2)
    console.print("[green]✅ System detection complete[/green]")

    with console.status("[bold blue]Setting up Docker database...", spinner="bouncingBar"):
        time.sleep(2)
    console.print("[blue]✅ Docker setup complete[/blue]")

    with console.status("[bold yellow]Running validation checks...", spinner="arrow3"):
        time.sleep(2)
    console.print("[yellow]✅ Validation complete[/yellow]")


def demo_validation_tree():
    """Demonstrate validation results tree."""
    console.print("\n[bold cyan]🌳 Validation Results Tree Example[/bold cyan]\n")

    tree = Tree("📋 Validation Results")

    # Prerequisites
    prereq_branch = tree.add("[green]✅[/green] [bold]Prerequisites[/bold] (3/3)")
    prereq_branch.add("[bold green]✓[/bold green] Python version: [green]Python 3.10.18[/green]")
    prereq_branch.add("[bold green]✓[/bold green] Operating System: [green]macOS[/green]")
    prereq_branch.add("[bold green]✓[/bold green] Package Manager: [green]conda found[/green]")

    # Installation
    install_branch = tree.add("[green]✅[/green] [bold]Spyglass Installation[/bold] (6/6)")
    install_branch.add("[bold green]✓[/bold green] Spyglass Import: [green]Version 0.5.6[/green]")
    install_branch.add("[bold green]✓[/bold green] DataJoint: [green]Version 0.14.6[/green]")
    install_branch.add("[bold green]✓[/bold green] PyNWB: [green]Version 2.8.3[/green]")
    install_branch.add("[bold green]✓[/bold green] Pandas: [green]Version 2.3.2[/green]")
    install_branch.add("[bold green]✓[/bold green] NumPy: [green]Version 1.26.4[/green]")
    install_branch.add("[bold green]✓[/bold green] Matplotlib: [green]Version 3.10.6[/green]")

    # Configuration
    config_branch = tree.add("[yellow]⚠️[/yellow] [bold]Configuration[/bold] (4/5)")
    config_branch.add("[bold yellow]⚠[/bold yellow] Multiple Config Files: [yellow]Found 3 config files[/yellow]")
    config_branch.add("[bold green]✓[/bold green] DataJoint Config: [green]Using config file[/green]")
    config_branch.add("[bold green]✓[/bold green] Spyglass Config: [green]spyglass_dirs found[/green]")
    config_branch.add("[bold green]✓[/bold green] Base Directory: [green]Found at /Users/user/spyglass_data[/green]")

    # Optional Dependencies
    optional_branch = tree.add("[green]✅[/green] [bold]Optional Dependencies[/bold] (5/7)")
    optional_branch.add("[bold green]✓[/bold green] Spike Sorting: [green]Version 0.99.1[/green]")
    optional_branch.add("[bold green]✓[/bold green] LFP Analysis: [green]Version 0.2.2[/green]")
    optional_branch.add("[bold blue]ℹ[/bold blue] DeepLabCut: [blue]Not installed (optional)[/blue]")
    optional_branch.add("[bold green]✓[/bold green] Visualization: [green]Version 0.3.1[/green]")
    optional_branch.add("[bold blue]ℹ[/bold blue] Data Sharing: [blue]Not installed (optional)[/blue]")

    console.print(tree)


def demo_summary_panel():
    """Demonstrate summary panels."""
    console.print("\n[bold cyan]📋 Summary Panel Examples[/bold cyan]\n")

    # Success panel
    success_content = """
[bold]Installation Type:[/bold] Full
[bold]Environment:[/bold] spyglass
[bold]Base Directory:[/bold] /Users/user/spyglass_data
[bold]Database Setup:[/bold] Configured

[bold cyan]Next Steps:[/bold cyan]

1. [bold]Activate environment:[/bold]
   [dim]conda activate spyglass[/dim]

2. [bold]Test installation:[/bold]
   [dim]python -c "import spyglass; print('✅ Success!')"[/dim]

3. [bold]Explore tutorials:[/bold]
   [dim]cd notebooks && jupyter notebook[/dim]
    """

    success_panel = Panel(
        success_content,
        title="[bold green]🎉 Installation Complete![/bold green]",
        border_style="green",
        box=box.DOUBLE,
        padding=(1, 2)
    )
    console.print(success_panel)

    console.print()

    # Validation summary
    validation_content = """
Total checks: 27
  [green]Passed: 24[/green]
  [yellow]Warnings: 1[/yellow]

[bold green]✅ Validation PASSED[/bold green]

Spyglass is properly installed and configured!
You can start with the tutorials in the notebooks directory.
    """

    validation_panel = Panel(
        validation_content,
        title="Validation Summary",
        border_style="green",
        box=box.ROUNDED
    )
    console.print(validation_panel)


def demo_interactive_prompts():
    """Demonstrate interactive prompts (optional - requires user input)."""
    console.print("\n[bold cyan]💬 Interactive Prompts Example[/bold cyan]\n")
    console.print("[dim]This section demonstrates interactive prompts.[/dim]")
    console.print("[dim]Run the actual rich scripts to see them in action![/dim]\n")

    # Show what the prompts would look like
    examples = [
        "❯ Enter your choice [1/2/3] (1): ",
        "❯ Database host (localhost): ",
        "❯ Update environment? [y/N]: ",
        "❯ Database password: ••••••••"
    ]

    for example in examples:
        console.print(f"[bold cyan]{example}[/bold cyan]")
        time.sleep(0.5)


def main():
    """Run the Rich UI demonstration."""
    import sys

    console.print("[bold magenta]🎨 Spyglass Rich UI Demonstration[/bold magenta]")
    console.print("[dim]This demo shows the enhanced UI components available in the Rich versions.[/dim]\n")

    # Check if running interactively
    interactive = sys.stdin.isatty() and "--auto" not in sys.argv

    demos = [
        ("Banner", demo_banner),
        ("System Information", demo_system_info),
        ("Installation Menu", demo_installation_menu),
        ("Progress Bars", demo_progress_bars),
        ("Status Indicators", demo_status_indicators),
        ("Validation Tree", demo_validation_tree),
        ("Summary Panels", demo_summary_panel),
        ("Interactive Prompts", demo_interactive_prompts)
    ]

    for name, demo_func in demos:
        console.print(f"\n[bold yellow]═══ {name} Demo ═══[/bold yellow]")
        try:
            demo_func()
        except KeyboardInterrupt:
            console.print("\n[yellow]Demo interrupted by user[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Demo error: {e}[/red]")

        if name != demos[-1][0] and interactive:  # Don't pause after last demo or in non-interactive mode
            console.print("\n[dim]Press Enter to continue to next demo...[/dim]")
            try:
                input()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Demo interrupted by user[/yellow]")
                break
        elif not interactive and name != demos[-1][0]:
            # Small pause for auto mode
            time.sleep(1)

    console.print(f"\n[bold green]🎉 Demo Complete![/bold green]")
    console.print("[dim]To see the full rich experience, try:[/dim]")
    console.print("[cyan]  python quickstart_rich.py[/cyan]")
    console.print("[cyan]  python validate_spyglass_rich.py -v[/cyan]")


if __name__ == "__main__":
    main()