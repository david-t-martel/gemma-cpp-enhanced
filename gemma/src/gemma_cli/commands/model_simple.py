"""Simplified model management commands.

This module provides streamlined model management focused on direct CLI usage:
- Priority 1: --model CLI argument (direct path)
- Priority 2: Model name from detected/configured models
- Priority 3: default_model from config

Commands:
- model detect: Scan filesystem for models, save to detected_models.json
- model list: Show all models (detected + configured + default)
- model add: Add a model to configured_models
- model remove: Remove a configured model
- model set-default: Set default model in config
"""

import json
import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gemma_cli.config.settings import (
    ConfigManager,
    ConfiguredModel,
    DetectedModel,
    load_config,
    load_detected_models,
    save_detected_models,
)

logger = logging.getLogger(__name__)


@click.group()
def model() -> None:
    """Simplified model management commands."""
    pass


@model.command()
@click.option(
    "--path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory to search for models (default: common model directories)",
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Search subdirectories recursively",
)
@click.pass_context
def detect(ctx: click.Context, path: Optional[Path], recursive: bool) -> None:
    """Auto-detect models and save to detected_models.json.

    Scans filesystem for .sbs model files and matching .spm tokenizers.
    Results are persisted to ~/.gemma_cli/detected_models.json for fast access.

    Args:
        path: Directory to search (default: common model directories)
        recursive: Whether to search subdirectories
    """
    # Get console from context
    console = ctx.obj["console"]

    try:
        # Determine search paths
        if path:
            search_paths = [path]
        else:
            # Use common model directories
            search_paths = [
                Path("C:/codedev/llm/.models"),
                Path("/c/codedev/llm/.models"),
                Path.home() / ".cache" / "gemma",
                Path.home() / "models",
                Path.home() / ".gemma_cli" / "models",
            ]
            # Filter to existing paths
            search_paths = [p for p in search_paths if p.exists()]

        if not search_paths:
            console.print("[yellow]No model directories found to search[/yellow]")
            return

        console.print(f"[cyan]Scanning for models in:[/cyan]")
        for sp in search_paths:
            console.print(f"  • {sp}")

        # Find models
        detected_models = {}
        for search_path in search_paths:
            _scan_directory(search_path, recursive, detected_models)

        if not detected_models:
            console.print("\n[yellow]No models detected[/yellow]")
            console.print("[dim]Model files should have .sbs extension with matching .spm tokenizer[/dim]")
            return

        # Save results
        save_detected_models(detected_models)

        # Display results
        console.print(f"\n[green]✓ Detected {len(detected_models)} model(s)[/green]")
        console.print(f"[dim]Saved to: ~/.gemma_cli/detected_models.json[/dim]\n")

        _display_detected_table(detected_models)

    except Exception as e:
        logger.exception(f"Detection failed: {e}")
        raise click.Abort()


def _scan_directory(
    search_path: Path, recursive: bool, detected_models: dict[str, DetectedModel]
) -> None:
    """Scan a directory for model files.

    Args:
        search_path: Directory to scan
        recursive: Whether to scan recursively
        detected_models: Dictionary to populate with detected models (mutated)
    """
    # Find .sbs files
    pattern = "**/*.sbs" if recursive else "*.sbs"
    weight_files = list(search_path.glob(pattern))

    for weight_file in weight_files:
        # Look for tokenizer in same directory
        tokenizer_file = weight_file.parent / "tokenizer.spm"

        # Extract model name from filename
        name = weight_file.stem.replace(".sbs", "")

        # Determine format from filename
        format_str = "unknown"
        if "sfp" in name.lower():
            format_str = "sfp"
        elif "bf16" in name.lower():
            format_str = "bf16"
        elif "f32" in name.lower():
            format_str = "f32"
        elif "nuq" in name.lower():
            format_str = "nuq"

        # Calculate size
        size_gb = weight_file.stat().st_size / (1024**3)

        model = DetectedModel(
            name=name,
            weights_path=str(weight_file.resolve()),
            tokenizer_path=str(tokenizer_file.resolve()) if tokenizer_file.exists() else None,
            format=format_str,
            size_gb=round(size_gb, 2),
        )

        detected_models[name] = model


def _display_detected_table(models: dict[str, DetectedModel]) -> None:
    """Display detected models in a table.

    Args:
        models: Dictionary of detected models
    """
    table = Table(title="Detected Models", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Size", justify="right")
    table.add_column("Format", justify="center")
    table.add_column("Tokenizer", justify="center")
    table.add_column("Path", style="dim")

    for name, model in sorted(models.items()):
        size_str = f"{model.size_gb:.2f} GB"
        tok_status = "✓" if model.tokenizer_path else "✗"
        tok_style = "green" if model.tokenizer_path else "red"

        table.add_row(
            name,
            size_str,
            model.format.upper(),
            f"[{tok_style}]{tok_status}[/{tok_style}]",
            str(Path(model.weights_path).parent),
        )

    console.print(table)


@model.command()
@click.option(
    "--format",
    type=click.Choice(["table", "simple"], case_sensitive=False),
    default="table",
    help="Output format",
)
@click.pass_context
def list(ctx: click.Context, format: str) -> None:
    """List all available models (detected + configured + default).

    Shows models from three sources:
    1. Detected models (from 'model detect')
    2. Configured models (from 'model add')
    3. Default model from config

    Args:
        format: Output format (table or simple)
    """
    # Get console from context
    console = ctx.obj["console"]

    try:
        # Load models from all sources
        detected = load_detected_models()
        settings = load_config()
        configured = settings.configured_models
        default_model = settings.gemma.default_model

        # Merge all models
        all_models = {}

        # Add detected
        for name, model in detected.items():
            all_models[name] = {
                "source": "detected",
                "weights": model.weights_path,
                "tokenizer": model.tokenizer_path,
                "format": model.format,
                "size_gb": model.size_gb,
            }

        # Add configured (override detected if same name)
        for name, model in configured.items():
            # Calculate size if not already in detected
            if name not in all_models:
                weights_path = Path(model.weights_path)
                size_gb = weights_path.stat().st_size / (1024**3) if weights_path.exists() else 0.0
                format_str = "unknown"
                if "sfp" in name.lower():
                    format_str = "sfp"

                all_models[name] = {
                    "source": "configured",
                    "weights": model.weights_path,
                    "tokenizer": model.tokenizer_path,
                    "format": format_str,
                    "size_gb": round(size_gb, 2),
                }

        if not all_models:
            console.print("[yellow]No models found[/yellow]")
            console.print("\n[dim]Run 'gemma-cli model detect' to find models[/dim]")
            return

        # Determine which is default
        default_name = None
        if default_model:
            # Check if default_model is a name or a path
            default_path = Path(default_model)
            if default_path.exists():
                # It's a path - try to find matching model by path
                for name, info in all_models.items():
                    if Path(info["weights"]).samefile(default_path):
                        default_name = name
                        break
            else:
                # It's a name
                if default_model in all_models:
                    default_name = default_model

        # Display
        if format == "table":
            _display_models_table(all_models, default_name)
        else:
            _display_models_simple(all_models, default_name)

        # Show summary
        console.print(f"\n[dim]Total: {len(all_models)} model(s)[/dim]")
        if default_name:
            console.print(f"[dim]Default: {default_name}[/dim]")
        else:
            console.print("[dim]Default: [yellow]None set[/yellow][/dim]")

    except Exception as e:
        logger.exception(f"Failed to list models: {e}")
        raise click.Abort()


def _display_models_table(models: dict, default_name: Optional[str]) -> None:
    """Display models in table format."""
    table = Table(title="Available Models", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Size", justify="right")
    table.add_column("Format", justify="center")
    table.add_column("Source", justify="center")
    table.add_column("Tokenizer", justify="center")

    for name, info in sorted(models.items()):
        display_name = f"{name} [bold](default)[/bold]" if name == default_name else name
        size_str = f"{info['size_gb']:.2f} GB"
        tok_status = "✓" if info["tokenizer"] else "✗"
        tok_style = "green" if info["tokenizer"] else "red"

        table.add_row(
            display_name,
            size_str,
            info["format"].upper(),
            info["source"],
            f"[{tok_style}]{tok_status}[/{tok_style}]",
        )

    console.print(table)


def _display_models_simple(models: dict, default_name: Optional[str]) -> None:
    """Display models in simple list format."""
    for name, info in sorted(models.items()):
        default_marker = " (default)" if name == default_name else ""
        console.print(f"{name}{default_marker} - {info['size_gb']:.2f} GB {info['format']}")


@model.command()
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.option("--name", help="Friendly name for the model (default: filename)")
@click.option("--tokenizer", type=click.Path(exists=True, path_type=Path), help="Tokenizer path (default: auto-detect in same directory)")
@click.pass_context
def add(ctx: click.Context, model_path: Path, name: Optional[str], tokenizer: Optional[Path]) -> None:
    """Add a model to configured models.

    Args:
        model_path: Path to .sbs weights file
        name: Friendly name (default: derived from filename)
        tokenizer: Optional tokenizer path (auto-detected if not provided)
    """
    # Get console from context
    console = ctx.obj["console"]

    try:
        # Validate model file
        if not model_path.suffix == ".sbs":
            console.print("[red]Error: Model file must have .sbs extension[/red]")
            raise click.Abort()

        # Determine name
        if not name:
            name = model_path.stem

        # Find tokenizer
        if tokenizer is None:
            tokenizer_path = model_path.parent / "tokenizer.spm"
            if not tokenizer_path.exists():
                console.print("[yellow]Warning: No tokenizer found in same directory[/yellow]")
                tokenizer_path_str = None
            else:
                tokenizer_path_str = str(tokenizer_path.resolve())
        else:
            tokenizer_path_str = str(tokenizer.resolve())

        # Load current config
        config_manager = ConfigManager()
        settings = config_manager.load()

        # Add model
        settings.configured_models[name] = ConfiguredModel(
            name=name,
            weights_path=str(model_path.resolve()),
            tokenizer_path=tokenizer_path_str,
        )

        # Save config
        config_manager.save(settings)

        console.print(f"[green]✓ Added model: {name}[/green]")
        console.print(f"[dim]Weights: {model_path}[/dim]")
        if tokenizer_path_str:
            console.print(f"[dim]Tokenizer: {tokenizer_path_str}[/dim]")

    except Exception as e:
        logger.exception(f"Failed to add model: {e}")
        raise click.Abort()


@model.command()
@click.argument("name")
@click.confirmation_option(prompt="Are you sure you want to remove this model?")
@click.pass_context
def remove(ctx: click.Context, name: str) -> None:
    """Remove a configured model.

    Args:
        name: Model name to remove
    """
    # Get console from context
    console = ctx.obj["console"]

    try:
        config_manager = ConfigManager()
        settings = config_manager.load()

        if name not in settings.configured_models:
            console.print(f"[red]Error: Model '{name}' not found in configured models[/red]")
            console.print("[dim]Use 'gemma-cli model list' to see available models[/dim]")
            raise click.Abort()

        # Remove model
        del settings.configured_models[name]

        # Save config
        config_manager.save(settings)

        console.print(f"[green]✓ Removed model: {name}[/green]")

    except Exception as e:
        logger.exception(f"Failed to remove model: {e}")
        raise click.Abort()


@model.command()
@click.argument("name")
@click.pass_context
def set_default(ctx: click.Context, name: str) -> None:
    """Set the default model.

    The default model is used when no --model flag is provided to chat/ask commands.

    Args:
        name: Model name to set as default
    """
    # Get console from context
    console = ctx.obj["console"]

    try:
        # Check if model exists
        detected = load_detected_models()
        settings = load_config()

        model_exists = name in detected or name in settings.configured_models

        if not model_exists:
            console.print(f"[red]Error: Model '{name}' not found[/red]")
            console.print("[dim]Use 'gemma-cli model list' to see available models[/dim]")
            raise click.Abort()

        # Resolve to actual path
        if name in detected:
            model_path = detected[name].weights_path
        else:
            model_path = settings.configured_models[name].weights_path

        # Update config
        config_manager = ConfigManager()
        settings = config_manager.load()
        settings.gemma.default_model = model_path

        # Also set tokenizer if available
        if name in detected and detected[name].tokenizer_path:
            settings.gemma.default_tokenizer = detected[name].tokenizer_path
        elif name in settings.configured_models and settings.configured_models[name].tokenizer_path:
            settings.gemma.default_tokenizer = settings.configured_models[name].tokenizer_path

        config_manager.save(settings)

        console.print(f"[green]✓ Set default model: {name}[/green]")
        console.print(f"[dim]Path: {model_path}[/dim]")

    except Exception as e:
        logger.exception(f"Failed to set default model: {e}")
        raise click.Abort()
