"""Main CLI entry point using Click framework.

A modern, extensible CLI for Gemma LLM with subcommands for chat, memory,
MCP integration, and configuration management.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel

# Import command groups
from gemma_cli.commands.chat import chat_group
from gemma_cli.commands.memory import memory_group
from gemma_cli.commands.mcp import mcp_group
from gemma_cli.commands.config import config_group
from gemma_cli.commands.model import model_group

# Initialize Rich console for all output
console = Console()

# ASCII Art Header
GEMMA_HEADER = """
╔══════════════════════════════════════════════════════════════╗
║              Gemma CLI - Modern Terminal Interface           ║
║               Powered by Google Gemma Foundation Models       ║
╚══════════════════════════════════════════════════════════════╝
"""


class AsyncioGroup(click.Group):
    """Click Group that supports async command handlers."""

    def invoke(self, ctx: click.Context) -> None:
        """Override invoke to handle async commands."""
        result = super().invoke(ctx)

        # If the result is a coroutine, run it with asyncio
        if asyncio.iscoroutine(result):
            try:
                asyncio.run(result)
            except KeyboardInterrupt:
                console.print("\n[yellow]Operation cancelled by user[/yellow]")
                ctx.exit(0)
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
                if ctx.obj.get("DEBUG"):
                    console.print_exception()
                ctx.exit(1)


@click.group(cls=AsyncioGroup)
@click.version_option(version="2.0.0", prog_name="Gemma CLI")
@click.option(
    "--debug/--no-debug",
    default=False,
    envvar="GEMMA_DEBUG",
    help="Enable debug mode with verbose output",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default=Path("config/config.toml"),
    envvar="GEMMA_CONFIG",
    help="Path to configuration file",
)
@click.option(
    "--profile",
    type=str,
    default=None,
    envvar="GEMMA_PROFILE",
    help="Performance profile to use (fast, balanced, quality)",
)
@click.pass_context
def cli(ctx: click.Context, debug: bool, config: Path, profile: Optional[str]) -> None:
    """Gemma CLI - Modern terminal interface for Gemma LLM.

    A comprehensive CLI for interacting with Google's Gemma foundation models,
    featuring multi-tier memory, RAG capabilities, and MCP integration.

    \b
    Quick Start:
      gemma chat interactive        # Start chat session
      gemma memory stats            # View memory statistics
      gemma mcp status              # Check MCP server status
      gemma config show             # Display configuration

    \b
    Examples:
      gemma chat ask "Explain transformers"
      gemma memory recall "machine learning" --tier long_term --limit 5
      gemma model list              # Show available models

    For detailed help on any command:
      gemma <command> --help
    """
    # Initialize context object for passing state between commands
    ctx.ensure_object(dict)

    ctx.obj["DEBUG"] = debug
    ctx.obj["CONFIG_PATH"] = config
    ctx.obj["PROFILE"] = profile
    ctx.obj["CONSOLE"] = console

    # Display header on first run (not for subcommands like --help)
    if ctx.invoked_subcommand is None and not ctx.resilient_parsing:
        console.print(GEMMA_HEADER, style="cyan bold")


# Register command groups
cli.add_command(chat_group, name="chat")
cli.add_command(memory_group, name="memory")
cli.add_command(mcp_group, name="mcp")
cli.add_command(config_group, name="config")
cli.add_command(model_group, name="model")


# Shell completion support
@cli.command(hidden=True)
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def completion(shell: str) -> None:
    """Generate shell completion script.

    \b
    Installation:
      bash:  eval "$(gemma completion bash)"
      zsh:   eval "$(gemma completion zsh)"
      fish:  gemma completion fish | source
    """
    import subprocess

    completion_script = subprocess.run(
        ["click-completion-command", shell],
        capture_output=True,
        text=True,
    )

    if completion_script.returncode == 0:
        console.print(completion_script.stdout)
    else:
        console.print("[red]Failed to generate completion script[/red]")


@cli.command()
@click.option(
    "--format",
    type=click.Choice(["text", "json", "markdown"]),
    default="text",
    help="Output format",
)
@click.pass_context
def info(ctx: click.Context, format: str) -> None:
    """Display system and environment information."""
    import platform
    import json
    from gemma_cli.utils.system import get_system_info

    console = ctx.obj["CONSOLE"]

    info_data = get_system_info()

    if format == "json":
        console.print_json(data=info_data)
    elif format == "markdown":
        # Generate markdown table
        console.print("# Gemma CLI System Information\n")
        for section, values in info_data.items():
            console.print(f"## {section.replace('_', ' ').title()}\n")
            for key, value in values.items():
                console.print(f"- **{key}**: {value}")
            console.print()
    else:
        # Rich formatted text
        from rich.table import Table

        table = Table(title="System Information", show_header=True)
        table.add_column("Category", style="cyan")
        table.add_column("Property", style="green")
        table.add_column("Value", style="white")

        for section, values in info_data.items():
            section_name = section.replace("_", " ").title()
            for i, (key, value) in enumerate(values.items()):
                table.add_row(
                    section_name if i == 0 else "",
                    key,
                    str(value)
                )

        console.print(table)


@cli.command()
@click.pass_context
async def health(ctx: click.Context) -> None:
    """Perform comprehensive health check of all systems.

    Checks:
    - Configuration validity
    - Model availability
    - Memory system connectivity
    - MCP server status
    """
    from gemma_cli.utils.health import run_health_check

    console = ctx.obj["CONSOLE"]

    with console.status("[cyan]Running health checks...") as status:
        results = await run_health_check(ctx.obj)

    from rich.table import Table

    table = Table(title="Health Check Results", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Details", style="dim")

    all_healthy = True

    for component, result in results.items():
        status = "[green]✓ Healthy[/green]" if result["healthy"] else "[red]✗ Unhealthy[/red]"
        details = result.get("details", "")
        table.add_row(component, status, details)

        if not result["healthy"]:
            all_healthy = False

    console.print(table)

    if all_healthy:
        console.print("\n[bold green]All systems operational[/bold green] ✓")
        ctx.exit(0)
    else:
        console.print("\n[bold red]Some systems require attention[/bold red] ✗")
        ctx.exit(1)


def main() -> None:
    """Entry point for the CLI application."""
    try:
        cli(obj={}, auto_envvar_prefix="GEMMA")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {str(e)}")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
