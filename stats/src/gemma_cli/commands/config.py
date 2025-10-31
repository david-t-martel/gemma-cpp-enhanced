"""Configuration management commands."""

from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

console = Console()


@click.group(name="config")
def config_group() -> None:
    """Configuration file management."""
    pass


@config_group.command(name="show")
@click.option("--format", type=click.Choice(["toml", "json", "table"]), default="table")
@click.pass_context
def show(ctx: click.Context, format: str) -> None:
    """Show current configuration.

    \b
    Examples:
      gemma config show
      gemma config show --format json
      gemma config show --format toml
    """
    from gemma_cli.utils.config import load_config

    config_path = ctx.obj["CONFIG_PATH"]

    try:
        config = load_config(config_path)
    except Exception as e:
        console.print(f"[red]Failed to load configuration: {e}[/red]")
        raise click.Abort()

    if format == "json":
        import json
        console.print_json(data=config)
    elif format == "toml":
        with open(config_path, "r") as f:
            syntax = Syntax(f.read(), "toml", theme="monokai", line_numbers=True)
            console.print(syntax)
    else:
        # Table format
        table = Table(title=f"Configuration ({config_path})", show_header=True)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")

        def add_config_rows(data: dict, prefix: str = "") -> None:
            for key, value in data.items():
                full_key = f"{prefix}{key}" if prefix else key
                if isinstance(value, dict):
                    add_config_rows(value, f"{full_key}.")
                else:
                    table.add_row(full_key, str(value))

        add_config_rows(config)
        console.print(table)


@config_group.command(name="set")
@click.argument("key", required=True)
@click.argument("value", required=True)
@click.pass_context
def set_value(ctx: click.Context, key: str, value: str) -> None:
    """Set configuration value.

    \b
    Examples:
      gemma config set model.temperature 0.8
      gemma config set redis.port 6380
    """
    console.print(f"[yellow]Setting {key}={value} not yet implemented[/yellow]")
    console.print("[dim]Edit config/config.toml directly for now[/dim]")


@config_group.command(name="validate")
@click.pass_context
def validate(ctx: click.Context) -> None:
    """Validate configuration file.

    Checks for:
    - Valid TOML syntax
    - Required fields present
    - Valid paths and URLs
    - Reasonable parameter values
    """
    from gemma_cli.utils.config import validate_config

    config_path = ctx.obj["CONFIG_PATH"]

    with console.status("[cyan]Validating configuration..."):
        errors, warnings = validate_config(config_path)

    if not errors and not warnings:
        console.print("[green]✓ Configuration is valid[/green]")
        return

    if warnings:
        console.print(f"\n[yellow]Warnings ({len(warnings)}):[/yellow]")
        for warning in warnings:
            console.print(f"  ⚠ {warning}")

    if errors:
        console.print(f"\n[red]Errors ({len(errors)}):[/red]")
        for error in errors:
            console.print(f"  ✗ {error}")
        raise click.Abort()


@config_group.command(name="init")
@click.option("--force", is_flag=True, help="Overwrite existing configuration")
@click.pass_context
def init(ctx: click.Context, force: bool) -> None:
    """Initialize default configuration file.

    \b
    Example:
      gemma config init
      gemma config init --force  # Overwrite existing
    """
    config_path = Path("config/config.toml")

    if config_path.exists() and not force:
        console.print(f"[yellow]Configuration already exists: {config_path}[/yellow]")
        console.print("[dim]Use --force to overwrite[/dim]")
        raise click.Abort()

    # Create default config
    default_config = """
# Gemma CLI Configuration

[model]
default_model = "4b"
model_path = "C:/codedev/llm/.models/gemma-3-gemmaCpp-3.0-4b-it-sfp-v1/4b-it-sfp.sbs"
tokenizer_path = "C:/codedev/llm/.models/gemma-3-gemmaCpp-3.0-4b-it-sfp-v1/tokenizer.spm"
gemma_executable = "C:/codedev/llm/gemma/build-avx2-sycl/bin/RELEASE/gemma.exe"

[generation]
max_tokens = 2048
temperature = 0.7
max_context = 8192

[rag]
enabled = true
redis_url = "redis://localhost:6379"
prefer_backend = "mcp"  # mcp, ffi, or python

[mcp]
host = "localhost"
port = 8765
timeout = 30

[system]
system_prompt = "You are a helpful AI assistant. Provide clear, concise, and accurate responses."
"""

    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        f.write(default_config.strip())

    console.print(f"[green]✓ Configuration initialized: {config_path}[/green]")
    console.print("[dim]Edit the file to customize settings[/dim]")
