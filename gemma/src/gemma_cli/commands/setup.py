"""Setup and configuration commands for gemma-cli."""

import asyncio
from pathlib import Path
from typing import Any

import click
import toml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from ..onboarding import (
    OnboardingWizard,
    check_system_requirements,
    display_health_check_results,
)
from ..onboarding.checks import run_comprehensive_checks
from ..onboarding.tutorial import InteractiveTutorial, run_quick_start

console = Console()


@click.command()
@click.option(
    "--force",
    is_flag=True,
    help="Force re-run onboarding even if already configured",
)
@click.option(
    "--config-path",
    type=click.Path(path_type=Path),
    help="Path to save configuration (default: ~/.gemma_cli/config.toml)",
)
@click.pass_context
def init(ctx: click.Context, force: bool, config_path: Path | None) -> None:
    """
    Initialize gemma-cli (first-run setup).

    Runs an interactive wizard to configure:
    - Model selection
    - Redis connection
    - Performance profile
    - UI preferences
    - Optional features
    """
    config_path = config_path or Path.home() / ".gemma_cli" / "config.toml"

    # Check if already configured
    if config_path.exists() and not force:
        console.print(
            f"[yellow]Configuration already exists at {config_path}[/yellow]"
        )
        console.print("[dim]Use --force to reconfigure[/dim]")
        return

    console.print("[bold cyan]Starting gemma-cli setup...[/bold cyan]\n")

    # Run onboarding wizard
    wizard = OnboardingWizard(config_path=config_path)
    asyncio.run(wizard.run())


@click.command()
@click.option(
    "--verbose",
    is_flag=True,
    help="Show detailed diagnostic information",
)
@click.pass_context
def health(ctx: click.Context, verbose: bool) -> None:
    """
    Run health checks on current configuration.

    Validates:
    - System requirements
    - Redis connection
    - Model files
    - Environment variables
    """
    console.print("[bold cyan]Running Health Checks...[/bold cyan]\n")

    if verbose:
        # Run comprehensive checks
        results = asyncio.run(run_comprehensive_checks())

        # Display system checks
        display_health_check_results(results["system"])

        # Display Redis diagnostics
        console.print("\n[bold]Redis Diagnostics:[/bold]")
        redis_diag = results["redis_diagnostics"]

        if redis_diag["redis_installed"]:
            console.print("[green]✓ Redis library installed[/green]")

            if redis_diag["localhost_reachable"]:
                console.print("[green]✓ Redis server reachable[/green]")

                for port, info in redis_diag["common_ports"].items():
                    if info["success"]:
                        console.print(
                            f"  [green]✓ Port {port}: {info['message']}[/green]"
                        )
            else:
                console.print("[yellow]⚠ Redis server not reachable[/yellow]")
                console.print("\n[bold]Suggestions:[/bold]")
                for suggestion in redis_diag["suggestions"]:
                    console.print(f"  • {suggestion}")
        else:
            console.print("[red]✗ Redis library not installed[/red]")

        # Display environment variables
        console.print("\n[bold]Environment Variables:[/bold]")
        for var, (is_set, value) in results["environment_vars"].items():
            if is_set:
                # Truncate long paths
                display_value = value if len(value) < 60 else value[:57] + "..."
                console.print(f"  [green]✓ {var}[/green] = {display_value}")
            else:
                console.print(f"  [dim]• {var}[/dim] - {value}")

        # Display platform info
        console.print("\n[bold]Platform Information:[/bold]")
        plat = results["platform_info"]
        console.print(
            f"  System: {plat['system']} {plat['release']} ({plat['machine']})"
        )

        # Display disk space
        console.print("\n[bold]Disk Space:[/bold]")
        disk_ok, disk_msg = results["disk_space"]
        if disk_ok:
            console.print(f"  [green]✓ {disk_msg}[/green]")
        else:
            console.print(f"  [yellow]⚠ {disk_msg}[/yellow]")

    else:
        # Run basic checks only
        checks = asyncio.run(check_system_requirements())
        all_passed = display_health_check_results(checks)

        if all_passed:
            console.print("\n[bold green]All checks passed! ✓[/bold green]")
        else:
            console.print(
                "\n[yellow]Some checks failed. Run with --verbose for details.[/yellow]"
            )


@click.command()
@click.option(
    "--quick",
    is_flag=True,
    help="Run quick-start guide instead of full tutorial",
)
@click.pass_context
def tutorial(ctx: click.Context, quick: bool) -> None:
    """
    Run interactive tutorial for new users.

    Learn about:
    - Basic chat interaction
    - Memory system and RAG
    - MCP tools integration
    - Advanced features
    """
    if quick:
        asyncio.run(run_quick_start())
    else:
        tutorial_obj = InteractiveTutorial()
        asyncio.run(tutorial_obj.run())


@click.command()
@click.option(
    "--full",
    is_flag=True,
    help="Full reset (delete all configuration and data)",
)
@click.option(
    "--keep-models",
    is_flag=True,
    help="Keep model configuration when resetting",
)
@click.pass_context
def reset(ctx: click.Context, full: bool, keep_models: bool) -> None:
    """
    Reset configuration to defaults.

    Use --full to delete all configuration and data.
    Use --keep-models to preserve model paths during reset.
    """
    config_path = Path.home() / ".gemma_cli" / "config.toml"

    if not config_path.exists():
        console.print("[yellow]No configuration found to reset.[/yellow]")
        return

    # Confirm with user
    warning_text = "[bold red]Warning:[/bold red] This will "
    if full:
        warning_text += "delete all configuration and data."
    else:
        warning_text += "reset configuration to defaults."

    if keep_models:
        warning_text += "\nModel configuration will be preserved."

    console.print(Panel(warning_text, border_style="red"))

    if not Confirm.ask("\nAre you sure you want to continue?", default=False):
        console.print("[yellow]Reset cancelled.[/yellow]")
        return

    # Preserve model config if requested
    model_config = None
    if keep_models:
        try:
            with open(config_path, encoding="utf-8") as f:
                current_config = toml.load(f)
                model_config = current_config.get("gemma")
        except Exception as e:
            console.print(f"[yellow]Could not read model config: {e}[/yellow]")

    # Delete configuration
    try:
        config_path.unlink()
        console.print(f"[green]✓ Deleted {config_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error deleting config: {e}[/red]")
        return

    # Delete additional data if full reset
    if full:
        data_dirs = [
            Path.home() / ".gemma_cli",
            Path.home() / ".gemma_conversations",
        ]

        for data_dir in data_dirs:
            if data_dir.exists():
                try:
                    import shutil

                    shutil.rmtree(data_dir)
                    console.print(f"[green]✓ Deleted {data_dir}[/green]")
                except Exception as e:
                    console.print(f"[yellow]Could not delete {data_dir}: {e}[/yellow]")

    # Re-run onboarding
    console.print("\n[bold cyan]Running setup wizard...[/bold cyan]\n")
    wizard = OnboardingWizard(config_path=config_path)

    # Pre-populate model config if preserved
    if model_config:
        wizard.config["gemma"] = model_config

    asyncio.run(wizard.run())


@click.command()
@click.option(
    "--show",
    is_flag=True,
    help="Show current configuration",
)
@click.option(
    "--edit",
    is_flag=True,
    help="Open configuration in editor",
)
@click.option(
    "--validate",
    is_flag=True,
    help="Validate configuration file",
)
@click.pass_context
def config(ctx: click.Context, show: bool, edit: bool, validate: bool) -> None:
    """
    Manage gemma-cli configuration.

    View, edit, or validate the configuration file.
    """
    config_path = Path.home() / ".gemma_cli" / "config.toml"

    if not config_path.exists():
        console.print(
            "[yellow]No configuration found. Run: gemma-cli init[/yellow]"
        )
        return

    if show:
        # Display configuration
        try:
            with open(config_path, encoding="utf-8") as f:
                config_data = toml.load(f)

            console.print(f"[bold cyan]Configuration: {config_path}[/bold cyan]\n")

            # Pretty print configuration
            import json

            console.print(json.dumps(config_data, indent=2))

        except Exception as e:
            console.print(f"[red]Error reading configuration: {e}[/red]")

    elif edit:
        # Open in editor
        import os

        editor = os.environ.get("EDITOR", "notepad" if os.name == "nt" else "nano")

        console.print(f"[cyan]Opening {config_path} in {editor}...[/cyan]")

        try:
            import subprocess

            subprocess.run([editor, str(config_path)], check=True)
            console.print("[green]✓ Configuration updated[/green]")
        except Exception as e:
            console.print(f"[red]Error opening editor: {e}[/red]")

    elif validate:
        # Validate configuration
        try:
            with open(config_path, encoding="utf-8") as f:
                toml.load(f)

            console.print("[green]✓ Configuration is valid[/green]")

            # Additional validation (optional)
            console.print("\n[bold]Validation Details:[/bold]")
            console.print("  [green]✓ TOML syntax correct[/green]")

            # Could add more specific validation here
            # e.g., check required fields, validate paths, etc.

        except toml.TomlDecodeError as e:
            console.print(f"[red]✗ Invalid TOML syntax: {e}[/red]")
        except Exception as e:
            console.print(f"[red]✗ Validation error: {e}[/red]")

    else:
        # Default: show config location
        console.print(f"[cyan]Configuration: {config_path}[/cyan]")

        if config_path.exists():
            size_kb = config_path.stat().st_size / 1024
            console.print(f"[dim]Size: {size_kb:.1f} KB[/dim]")
        else:
            console.print("[yellow]Configuration not found[/yellow]")

        console.print("\n[bold]Options:[/bold]")
        console.print("  --show      Display configuration")
        console.print("  --edit      Edit in text editor")
        console.print("  --validate  Validate syntax")


# Group all setup commands
@click.group(name="setup")
def setup_group() -> None:
    """Setup and configuration commands."""
    pass


# Register commands with group
setup_group.add_command(init)
setup_group.add_command(health)
setup_group.add_command(tutorial)
setup_group.add_command(reset)
setup_group.add_command(config)
