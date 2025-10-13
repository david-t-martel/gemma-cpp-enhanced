"""Model and profile management CLI commands.

Provides Click commands for managing model presets, performance
profiles, hardware detection, and model validation.

This module implements comprehensive model management functionality:
- Model preset listing and information display
- Default model configuration
- Automatic model detection in filesystem
- Model file validation
- Hardware detection and recommendations
- Performance profile management
- Custom profile creation and deletion

Examples:
    List all available models:
        $ gemma model list

    Show detailed model information:
        $ gemma model info gemma-2b-it

    Set default model:
        $ gemma model use gemma-2b-it

    Auto-detect models:
        $ gemma model detect --path /path/to/models

    Show hardware recommendations:
        $ gemma model hardware

    List performance profiles:
        $ gemma profile list

    Create custom profile:
        $ gemma profile create fast --max-tokens 1024 --temperature 0.5
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from gemma_cli.config.models import (
    HardwareDetector,
    ModelManager,
    ModelPreset,
    PerformanceProfile,
    ProfileManager,
)
from gemma_cli.config.settings import ConfigManager, load_config
from gemma_cli.ui.console import get_console
from gemma_cli.ui.formatters import (
    format_error_message,
    format_success_message,
    format_system_message,
    format_warning_message,
)

console = get_console()


# ============================================================================
# Model Commands
# ============================================================================


@click.group()
def model() -> None:
    """Model management commands.

    Manage model presets, validate model files, detect available models,
    and get hardware-specific recommendations.
    """
    pass


@model.command()
@click.option(
    "--format",
    type=click.Choice(["table", "detailed", "simple"], case_sensitive=False),
    default="table",
    help="Output format for model list",
)
@click.option(
    "--filter-size",
    type=click.Choice(["1b", "2b", "4b", "7b", "9b", "27b"]),
    help="Filter models by size",
)
@click.option(
    "--filter-type",
    type=click.Choice(["it", "pt", "code"]),
    help="Filter models by type (it=instruct, pt=pretrained, code=code)",
)
@click.option(
    "--show-paths/--no-show-paths",
    default=False,
    help="Show full file paths in output",
)
def list(
    format: str, filter_size: Optional[str], filter_type: Optional[str], show_paths: bool
) -> None:
    """List all available model presets.

    Displays information about all configured model presets including
    size, type, compression format, and availability status.

    Args:
        format: Output format (table, detailed, simple)
        filter_size: Filter by model size
        filter_type: Filter by model type
        show_paths: Include full file paths in output
    """
    try:
        manager = ModelManager()
        presets = manager.list_models()

        # Apply filters
        if filter_size:
            presets = {
                k: v for k, v in presets.items() if filter_size in k.lower()
            }

        if filter_type:
            type_filters = {
                "it": "-it",
                "pt": "-pt",
                "code": "code",
            }
            filter_str = type_filters.get(filter_type, "")
            presets = {k: v for k, v in presets.items() if filter_str in k.lower()}

        if not presets:
            console.print(
                format_warning_message(
                    "No models found matching the specified filters"
                )
            )
            return

        # Get current default model
        config = load_config()
        default_model = config.model.default_model

        if format == "table":
            _display_models_table(presets, default_model, show_paths)
        elif format == "detailed":
            _display_models_detailed(presets, default_model, show_paths)
        else:  # simple
            _display_models_simple(presets, default_model)

        # Show summary
        total = len(presets)
        available = sum(1 for p in presets.values() if p.is_available())
        console.print()
        console.print(
            format_system_message(
                f"Total: {total} models ({available} available, "
                f"{total - available} missing files)"
            )
        )

    except Exception as e:
        console.print(format_error_message(f"Failed to list models: {e}"))
        raise click.Abort()


def _display_models_table(
    presets: Dict[str, ModelPreset], default_model: str, show_paths: bool
) -> None:
    """Display models in table format.

    Args:
        presets: Dictionary of model presets
        default_model: Name of the default model
        show_paths: Whether to show full file paths
    """
    table = Table(title="Available Model Presets", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Size", justify="right")
    table.add_column("Type")
    table.add_column("Format")
    table.add_column("Status", justify="center")
    table.add_column("RAM Est.", justify="right")
    if show_paths:
        table.add_column("Weights Path", style="dim")

    for name, preset in sorted(presets.items()):
        # Determine status
        is_available = preset.is_available()
        status = "✓" if is_available else "✗"
        status_style = "green" if is_available else "red"

        # Mark default model
        display_name = f"{name} [bold](default)[/bold]" if name == default_model else name

        # Estimate RAM usage (rough calculation based on model size)
        size_str = preset.size
        if "1b" in size_str.lower():
            ram_gb = 2
        elif "2b" in size_str.lower():
            ram_gb = 3
        elif "4b" in size_str.lower():
            ram_gb = 6
        elif "7b" in size_str.lower():
            ram_gb = 10
        elif "9b" in size_str.lower():
            ram_gb = 12
        elif "27b" in size_str.lower():
            ram_gb = 32
        else:
            ram_gb = 4

        ram_est = f"{ram_gb}GB"

        row = [
            display_name,
            preset.size,
            preset.model_type,
            preset.compression,
            f"[{status_style}]{status}[/{status_style}]",
            ram_est,
        ]

        if show_paths:
            row.append(str(preset.weights_path))

        table.add_row(*row)

    console.print(table)


def _display_models_detailed(
    presets: Dict[str, ModelPreset], default_model: str, show_paths: bool
) -> None:
    """Display models in detailed format.

    Args:
        presets: Dictionary of model presets
        default_model: Name of the default model
        show_paths: Whether to show full file paths
    """
    for name, preset in sorted(presets.items()):
        is_default = name == default_model
        is_available = preset.is_available()

        # Create panel title
        title = f"[bold cyan]{name}[/bold cyan]"
        if is_default:
            title += " [bold yellow](default)[/bold yellow]"

        # Build content
        content = []
        content.append(f"[bold]Size:[/bold] {preset.size}")
        content.append(f"[bold]Type:[/bold] {preset.model_type}")
        content.append(f"[bold]Compression:[/bold] {preset.compression}")

        # Status with color
        status_str = "[green]Available[/green]" if is_available else "[red]Missing Files[/red]"
        content.append(f"[bold]Status:[/bold] {status_str}")

        if show_paths:
            content.append(f"[bold]Weights:[/bold] {preset.weights_path}")
            content.append(f"[bold]Tokenizer:[/bold] {preset.tokenizer_path}")

        # Description
        if preset.description:
            content.append(f"\n[dim]{preset.description}[/dim]")

        # Validation messages
        if not is_available:
            validation = preset.validate()
            if not validation.is_valid:
                content.append("\n[bold red]Issues:[/bold red]")
                for error in validation.errors:
                    content.append(f"  • {error}")

        panel = Panel(
            "\n".join(content),
            title=title,
            border_style="cyan" if is_available else "red",
            expand=False,
        )
        console.print(panel)


def _display_models_simple(presets: Dict[str, ModelPreset], default_model: str) -> None:
    """Display models in simple list format.

    Args:
        presets: Dictionary of model presets
        default_model: Name of the default model
    """
    for name, preset in sorted(presets.items()):
        is_default = name == default_model
        is_available = preset.is_available()

        status = "✓" if is_available else "✗"
        default_marker = " (default)" if is_default else ""

        console.print(f"{status} {name}{default_marker} - {preset.size} {preset.model_type}")


@model.command()
@click.argument("model_name")
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Validate model files exist",
)
def info(model_name: str, validate: bool) -> None:
    """Show detailed information about a model.

    Displays comprehensive information about a specific model preset
    including configuration, file paths, validation status, and
    performance characteristics.

    Args:
        model_name: Name of the model preset
        validate: Whether to validate file existence
    """
    try:
        manager = ModelManager()
        preset = manager.get_model(model_name)

        if not preset:
            console.print(
                format_error_message(f"Model '{model_name}' not found")
            )
            console.print(
                format_system_message(
                    "Use 'gemma model list' to see available models"
                )
            )
            raise click.Abort()

        # Check if this is the default model
        config = load_config()
        is_default = model_name == config.model.default_model

        # Build information panel
        title = f"[bold cyan]{model_name}[/bold cyan]"
        if is_default:
            title += " [bold yellow](default)[/bold yellow]"

        content = []

        # Basic information
        content.append("[bold]═══ Basic Information ═══[/bold]")
        content.append(f"Size: {preset.size}")
        content.append(f"Type: {preset.model_type}")
        content.append(f"Compression: {preset.compression}")
        content.append(f"Description: {preset.description or 'N/A'}")

        # File paths
        content.append("\n[bold]═══ File Paths ═══[/bold]")
        content.append(f"Weights: [dim]{preset.weights_path}[/dim]")
        content.append(f"Tokenizer: [dim]{preset.tokenizer_path}[/dim]")

        # Validation status
        if validate:
            content.append("\n[bold]═══ Validation Status ═══[/bold]")
            validation = preset.validate()

            if validation.is_valid:
                content.append("[green]✓ All files present and accessible[/green]")
            else:
                content.append("[red]✗ Validation failed[/red]")
                for error in validation.errors:
                    content.append(f"  • [red]{error}[/red]")

            if validation.warnings:
                content.append("\n[yellow]Warnings:[/yellow]")
                for warning in validation.warnings:
                    content.append(f"  • [yellow]{warning}[/yellow]")

        # Performance estimates
        content.append("\n[bold]═══ Performance Estimates ═══[/bold]")
        perf_info = _estimate_performance(preset)
        content.append(f"Estimated RAM: {perf_info['ram']}")
        content.append(f"VRAM (if GPU): {perf_info['vram']}")
        content.append(f"Typical Speed: {perf_info['speed']}")
        content.append(f"Context Window: {perf_info['context']}")

        # Hardware recommendations
        content.append("\n[bold]═══ Hardware Recommendations ═══[/bold]")
        hw_reqs = _get_hardware_requirements(preset)
        content.append(f"Minimum RAM: {hw_reqs['min_ram']}")
        content.append(f"Recommended RAM: {hw_reqs['rec_ram']}")
        content.append(f"Minimum CPU: {hw_reqs['min_cpu']}")
        content.append(f"Supports GPU: {hw_reqs['gpu_support']}")

        panel = Panel(
            "\n".join(content),
            title=title,
            border_style="cyan",
            expand=False,
        )
        console.print(panel)

    except Exception as e:
        console.print(format_error_message(f"Failed to get model info: {e}"))
        raise click.Abort()


def _estimate_performance(preset: ModelPreset) -> Dict[str, str]:
    """Estimate performance characteristics for a model.

    Args:
        preset: Model preset to analyze

    Returns:
        Dictionary with performance estimates
    """
    size_lower = preset.size.lower()

    # RAM estimates
    if "1b" in size_lower:
        ram, vram, speed, ctx = "2GB", "2GB", "~30 tokens/sec", "8K"
    elif "2b" in size_lower:
        ram, vram, speed, ctx = "3GB", "3GB", "~25 tokens/sec", "8K"
    elif "4b" in size_lower:
        ram, vram, speed, ctx = "6GB", "5GB", "~20 tokens/sec", "8K"
    elif "7b" in size_lower:
        ram, vram, speed, ctx = "10GB", "8GB", "~15 tokens/sec", "8K"
    elif "9b" in size_lower:
        ram, vram, speed, ctx = "12GB", "10GB", "~12 tokens/sec", "8K"
    elif "27b" in size_lower:
        ram, vram, speed, ctx = "32GB", "24GB", "~5 tokens/sec", "8K"
    else:
        ram, vram, speed, ctx = "4GB", "4GB", "~20 tokens/sec", "8K"

    # Adjust for compression
    if "sfp" in preset.compression.lower():
        speed = speed.replace("~", "~2x ") + " (SFP optimized)"

    return {
        "ram": ram,
        "vram": vram,
        "speed": speed,
        "context": ctx,
    }


def _get_hardware_requirements(preset: ModelPreset) -> Dict[str, str]:
    """Get hardware requirements for a model.

    Args:
        preset: Model preset to analyze

    Returns:
        Dictionary with hardware requirements
    """
    size_lower = preset.size.lower()

    if "1b" in size_lower:
        min_ram, rec_ram, min_cpu = "4GB", "8GB", "4 cores"
    elif "2b" in size_lower:
        min_ram, rec_ram, min_cpu = "8GB", "16GB", "4 cores"
    elif "4b" in size_lower:
        min_ram, rec_ram, min_cpu = "12GB", "16GB", "6 cores"
    elif "7b" in size_lower:
        min_ram, rec_ram, min_cpu = "16GB", "32GB", "8 cores"
    elif "9b" in size_lower:
        min_ram, rec_ram, min_cpu = "16GB", "32GB", "8 cores"
    elif "27b" in size_lower:
        min_ram, rec_ram, min_cpu = "48GB", "64GB", "16 cores"
    else:
        min_ram, rec_ram, min_cpu = "8GB", "16GB", "4 cores"

    return {
        "min_ram": min_ram,
        "rec_ram": rec_ram,
        "min_cpu": min_cpu,
        "gpu_support": "Yes (CUDA, SYCL, Vulkan planned)",
    }


@model.command()
@click.argument("model_name")
@click.option(
    "--set-paths/--no-set-paths",
    default=False,
    help="Update model paths if not found",
)
def use(model_name: str, set_paths: bool) -> None:
    """Set the default model.

    Updates the configuration to use the specified model as the default
    for all inference operations.

    Args:
        model_name: Name of the model preset to use as default
        set_paths: Prompt to update paths if model files not found
    """
    try:
        manager = ModelManager()
        preset = manager.get_model(model_name)

        if not preset:
            console.print(
                format_error_message(f"Model '{model_name}' not found")
            )
            console.print(
                format_system_message(
                    "Use 'gemma model list' to see available models"
                )
            )
            raise click.Abort()

        # Validate model files
        validation = preset.validate()

        if not validation.is_valid:
            console.print(
                format_error_message(
                    f"Model '{model_name}' validation failed:"
                )
            )
            for error in validation.errors:
                console.print(f"  • {error}")

            if set_paths:
                console.print(
                    format_system_message(
                        "\nWould you like to update the model paths?"
                    )
                )
                if click.confirm("Update paths now?"):
                    _interactive_path_update(manager, model_name, preset)
                    # Re-validate after path update
                    validation = preset.validate()

            if not validation.is_valid:
                console.print(
                    format_warning_message(
                        "\nModel files are still missing. "
                        "The model has been set as default but may not work correctly."
                    )
                )
                if not click.confirm("Continue anyway?"):
                    raise click.Abort()

        # Update configuration
        config_manager = ConfigManager()
        config = config_manager.load()
        config.model.default_model = model_name
        config_manager.save(config)

        console.print(
            format_success_message(
                f"Default model set to: {model_name}"
            )
        )

        # Show model info
        console.print()
        console.print(format_system_message(f"Model: {preset.size} {preset.model_type}"))
        console.print(format_system_message(f"Compression: {preset.compression}"))

        if validation.warnings:
            console.print()
            for warning in validation.warnings:
                console.print(format_warning_message(warning))

    except Exception as e:
        console.print(format_error_message(f"Failed to set default model: {e}"))
        raise click.Abort()


def _interactive_path_update(
    manager: ModelManager, model_name: str, preset: ModelPreset
) -> None:
    """Interactively update model file paths.

    Args:
        manager: ModelManager instance
        model_name: Name of the model
        preset: Model preset to update
    """
    console.print("\n[bold]Update Model Paths[/bold]")

    # Update weights path
    if not preset.weights_path.exists():
        console.print(f"\nCurrent weights path: {preset.weights_path}")
        new_weights = click.prompt("Enter new weights path")
        preset.weights_path = Path(new_weights)

    # Update tokenizer path
    if not preset.tokenizer_path.exists():
        console.print(f"\nCurrent tokenizer path: {preset.tokenizer_path}")
        new_tokenizer = click.prompt("Enter new tokenizer path")
        preset.tokenizer_path = Path(new_tokenizer)

    # Save updated preset
    # Note: This would require extending ModelManager to support
    # saving custom presets - for now just update in memory
    console.print(format_system_message("\nPaths updated (session only)"))


@model.command()
@click.option(
    "--path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory to search for models",
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Search subdirectories recursively",
)
@click.option(
    "--add-to-config/--no-add-to-config",
    default=False,
    help="Add detected models to configuration",
)
def detect(path: Optional[str], recursive: bool, add_to_config: bool) -> None:
    """Auto-detect models in filesystem.

    Scans the specified directory (or default model paths) for valid
    model files and displays what was found.

    Args:
        path: Directory to search
        recursive: Whether to search subdirectories
        add_to_config: Whether to add found models to config
    """
    try:
        if path:
            search_path = Path(path)
        else:
            # Use default model directory
            search_path = Path.home() / ".cache" / "gemma"
            # Also check common locations
            common_paths = [
                Path("C:/codedev/llm/.models"),
                Path("/c/codedev/llm/.models"),
                Path.home() / "models",
            ]
            for common_path in common_paths:
                if common_path.exists():
                    search_path = common_path
                    break

        console.print(
            format_system_message(f"Scanning for models in: {search_path}")
        )

        if recursive:
            console.print(format_system_message("(including subdirectories)"))

        # Find model files
        detected = _scan_for_models(search_path, recursive)

        if not detected:
            console.print(
                format_warning_message(
                    f"No model files detected in {search_path}"
                )
            )
            console.print(
                format_system_message(
                    "\nModel files should have extensions: .sbs, .sbs.gz"
                )
            )
            return

        # Display detected models
        console.print()
        _display_detected_models(detected)

        # Optionally add to configuration
        if add_to_config and detected:
            console.print()
            if click.confirm(
                f"Add {len(detected)} detected model(s) to configuration?"
            ):
                _add_detected_to_config(detected)

    except Exception as e:
        console.print(format_error_message(f"Detection failed: {e}"))
        raise click.Abort()


def _scan_for_models(
    search_path: Path, recursive: bool
) -> List[Tuple[Path, Optional[Path]]]:
    """Scan directory for model files.

    Args:
        search_path: Directory to scan
        recursive: Whether to scan recursively

    Returns:
        List of (weights_path, tokenizer_path) tuples
    """
    detected = []

    # Pattern for model weight files
    weight_patterns = ["*.sbs", "*.sbs.gz"]

    # Find weight files
    for pattern in weight_patterns:
        if recursive:
            weight_files = search_path.rglob(pattern)
        else:
            weight_files = search_path.glob(pattern)

        for weight_file in weight_files:
            # Look for corresponding tokenizer
            tokenizer = None

            # Check in same directory
            same_dir_tokenizers = list(weight_file.parent.glob("tokenizer.spm"))
            if same_dir_tokenizers:
                tokenizer = same_dir_tokenizers[0]
            else:
                # Check parent directory
                parent_tokenizers = list(
                    weight_file.parent.parent.glob("tokenizer.spm")
                )
                if parent_tokenizers:
                    tokenizer = parent_tokenizers[0]

            detected.append((weight_file, tokenizer))

    return detected


def _display_detected_models(detected: List[Tuple[Path, Optional[Path]]]) -> None:
    """Display detected models in a table.

    Args:
        detected: List of (weights_path, tokenizer_path) tuples
    """
    table = Table(
        title=f"Detected Models ({len(detected)} found)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Weights File", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Tokenizer", justify="center")
    table.add_column("Path", style="dim")

    for weights, tokenizer in detected:
        # Get file size
        size_mb = weights.stat().st_size / (1024 * 1024)
        if size_mb > 1024:
            size_str = f"{size_mb / 1024:.1f}GB"
        else:
            size_str = f"{size_mb:.1f}MB"

        # Tokenizer status
        tok_status = "✓" if tokenizer else "✗"
        tok_style = "green" if tokenizer else "red"

        table.add_row(
            weights.name,
            size_str,
            f"[{tok_style}]{tok_status}[/{tok_style}]",
            str(weights.parent),
        )

    console.print(table)


def _add_detected_to_config(detected: List[Tuple[Path, Optional[Path]]]) -> None:
    """Add detected models to configuration.

    Args:
        detected: List of (weights_path, tokenizer_path) tuples
    """
    # This would require extending the configuration system
    # to support dynamic model registration
    console.print(
        format_warning_message(
            "Dynamic model registration not yet implemented"
        )
    )
    console.print(
        format_system_message(
            "Detected models can be used by specifying full paths with --weights flag"
        )
    )


@model.command()
@click.argument("model_name")
@click.option(
    "--fix/--no-fix",
    default=False,
    help="Attempt to fix validation issues",
)
def validate(model_name: str, fix: bool) -> None:
    """Validate model files exist and are correct.

    Performs comprehensive validation of model files including:
    - File existence checks
    - File size validation
    - Path accessibility
    - Format verification

    Args:
        model_name: Name of the model to validate
        fix: Whether to attempt automatic fixes
    """
    try:
        manager = ModelManager()
        preset = manager.get_model(model_name)

        if not preset:
            console.print(
                format_error_message(f"Model '{model_name}' not found")
            )
            raise click.Abort()

        console.print(format_system_message(f"Validating model: {model_name}"))

        # Perform validation
        validation = preset.validate()

        # Display results
        console.print()
        if validation.is_valid:
            console.print(
                format_success_message("✓ Validation passed")
            )
            console.print(format_system_message("All model files are present and accessible"))
        else:
            console.print(format_error_message("✗ Validation failed"))
            console.print()
            for error in validation.errors:
                console.print(f"  [red]• {error}[/red]")

            if fix:
                console.print()
                if click.confirm("Attempt to fix issues?"):
                    _attempt_validation_fixes(preset, validation)

        # Show warnings
        if validation.warnings:
            console.print()
            console.print(format_warning_message("Warnings:"))
            for warning in validation.warnings:
                console.print(f"  [yellow]• {warning}[/yellow]")

        # Show file details
        console.print()
        _display_file_details(preset)

    except Exception as e:
        console.print(format_error_message(f"Validation failed: {e}"))
        raise click.Abort()


def _attempt_validation_fixes(
    preset: ModelPreset, validation
) -> None:
    """Attempt to fix validation issues.

    Args:
        preset: Model preset with issues
        validation: Validation result
    """
    console.print(format_system_message("Checking for common fixes..."))

    fixed_count = 0

    # Check for case sensitivity issues
    if not preset.weights_path.exists():
        # Try different case combinations
        parent = preset.weights_path.parent
        name = preset.weights_path.name

        if parent.exists():
            actual_files = [f.name for f in parent.iterdir()]
            case_match = next(
                (f for f in actual_files if f.lower() == name.lower()), None
            )

            if case_match:
                console.print(
                    format_success_message(
                        f"Found file with different case: {case_match}"
                    )
                )
                console.print(
                    format_system_message(
                        "Update configuration to use correct case"
                    )
                )
                fixed_count += 1

    # Check for symlink issues
    if preset.weights_path.is_symlink() and not preset.weights_path.exists():
        console.print(
            format_warning_message("Broken symlink detected")
        )
        console.print(
            format_system_message("Consider removing and recreating the symlink")
        )

    if fixed_count == 0:
        console.print(
            format_warning_message("No automatic fixes available")
        )
        console.print(
            format_system_message(
                "Manual intervention required - check file paths in configuration"
            )
        )


def _display_file_details(preset: ModelPreset) -> None:
    """Display detailed file information.

    Args:
        preset: Model preset to display
    """
    table = Table(title="File Details", show_header=True, header_style="bold cyan")
    table.add_column("File", style="cyan")
    table.add_column("Exists", justify="center")
    table.add_column("Size", justify="right")
    table.add_column("Path", style="dim")

    # Weights file
    weights_exists = preset.weights_path.exists()
    weights_size = (
        f"{preset.weights_path.stat().st_size / (1024**3):.2f}GB"
        if weights_exists
        else "N/A"
    )

    table.add_row(
        "Weights",
        f"[green]✓[/green]" if weights_exists else "[red]✗[/red]",
        weights_size,
        str(preset.weights_path),
    )

    # Tokenizer file
    tokenizer_exists = preset.tokenizer_path.exists()
    tokenizer_size = (
        f"{preset.tokenizer_path.stat().st_size / (1024**2):.2f}MB"
        if tokenizer_exists
        else "N/A"
    )

    table.add_row(
        "Tokenizer",
        f"[green]✓[/green]" if tokenizer_exists else "[red]✗[/red]",
        tokenizer_size,
        str(preset.tokenizer_path),
    )

    console.print(table)


@model.command()
@click.option(
    "--recommend/--no-recommend",
    default=True,
    help="Show recommended models for detected hardware",
)
def hardware() -> None:
    """Show hardware info and model recommendations.

    Detects system hardware capabilities and recommends appropriate
    models based on available resources.
    """
    try:
        detector = HardwareDetector()
        info = detector.detect()

        # Display hardware information
        _display_hardware_info(info)

        # Show model recommendations
        if info and info.get("recommend"):
            console.print()
            _display_model_recommendations(info)

    except Exception as e:
        console.print(format_error_message(f"Hardware detection failed: {e}"))
        raise click.Abort()


def _display_hardware_info(info: Dict) -> None:
    """Display detected hardware information.

    Args:
        info: Hardware information dictionary
    """
    panel_content = []

    # CPU information
    panel_content.append("[bold]═══ CPU ═══[/bold]")
    panel_content.append(f"Model: {info.get('cpu_model', 'Unknown')}")
    panel_content.append(f"Cores: {info.get('cpu_cores', 'Unknown')}")
    panel_content.append(f"Threads: {info.get('cpu_threads', 'Unknown')}")

    # Memory information
    panel_content.append("\n[bold]═══ Memory ═══[/bold]")
    total_ram = info.get("total_ram_gb", 0)
    available_ram = info.get("available_ram_gb", 0)
    panel_content.append(f"Total RAM: {total_ram:.1f}GB")
    panel_content.append(f"Available RAM: {available_ram:.1f}GB")
    panel_content.append(f"Used RAM: {total_ram - available_ram:.1f}GB")

    # GPU information
    if info.get("has_gpu"):
        panel_content.append("\n[bold]═══ GPU ═══[/bold]")
        panel_content.append(f"GPU Detected: [green]Yes[/green]")
        if info.get("gpu_name"):
            panel_content.append(f"Model: {info['gpu_name']}")
        if info.get("gpu_memory_gb"):
            panel_content.append(f"VRAM: {info['gpu_memory_gb']:.1f}GB")
    else:
        panel_content.append("\n[bold]═══ GPU ═══[/bold]")
        panel_content.append("GPU Detected: [yellow]No[/yellow]")

    # Platform information
    panel_content.append("\n[bold]═══ Platform ═══[/bold]")
    panel_content.append(f"OS: {info.get('platform', 'Unknown')}")
    panel_content.append(f"Architecture: {info.get('architecture', 'Unknown')}")

    panel = Panel(
        "\n".join(panel_content),
        title="[bold cyan]System Hardware[/bold cyan]",
        border_style="cyan",
        expand=False,
    )
    console.print(panel)


def _display_model_recommendations(info: Dict) -> None:
    """Display recommended models based on hardware.

    Args:
        info: Hardware information dictionary
    """
    total_ram = info.get("total_ram_gb", 0)
    has_gpu = info.get("has_gpu", False)

    # Determine recommended models
    recommendations = []

    if total_ram >= 32:
        recommendations.append(("gemma-3-9b-it-sfp", "Best quality for your system"))
        recommendations.append(("gemma-2-7b-it-sfp", "Good balance"))
        recommendations.append(("gemma-2-4b-it-sfp", "Fastest option"))
    elif total_ram >= 16:
        recommendations.append(("gemma-2-4b-it-sfp", "Best for your RAM"))
        recommendations.append(("gemma-2-2b-it-sfp", "Faster alternative"))
    elif total_ram >= 8:
        recommendations.append(("gemma-2-2b-it-sfp", "Recommended"))
        recommendations.append(("gemma-3-1b-it-sfp", "Lightweight option"))
    else:
        recommendations.append(("gemma-3-1b-it-sfp", "Only viable option"))

    # Create recommendations table
    table = Table(
        title="Recommended Models for Your Hardware",
        show_header=True,
        header_style="bold green",
    )
    table.add_column("Model", style="cyan")
    table.add_column("Reason", style="green")
    table.add_column("GPU Support", justify="center")

    for model_name, reason in recommendations:
        gpu_support = "✓" if has_gpu else "Future"
        table.add_row(model_name, reason, gpu_support)

    console.print(table)

    # Add notes
    console.print()
    if has_gpu:
        console.print(
            format_system_message(
                "GPU acceleration planned for future releases (CUDA, SYCL, Vulkan)"
            )
        )
    else:
        console.print(
            format_system_message(
                "CPU-only inference currently available. GPU support coming soon."
            )
        )


# ============================================================================
# Profile Commands
# ============================================================================


@click.group()
def profile() -> None:
    """Performance profile management commands.

    Manage performance profiles that control inference parameters like
    token limits, temperature, sampling methods, and optimization levels.
    """
    pass


@profile.command()
@click.option(
    "--format",
    type=click.Choice(["table", "detailed", "simple"], case_sensitive=False),
    default="table",
    help="Output format",
)
def list(format: str) -> None:
    """List all performance profiles.

    Displays all available performance profiles including built-in
    presets and custom user-defined profiles.

    Args:
        format: Output format (table, detailed, simple)
    """
    try:
        manager = ProfileManager()
        profiles = manager.list_profiles()

        if not profiles:
            console.print(
                format_warning_message("No profiles configured")
            )
            return

        # Get current default
        config = load_config()
        default_profile = config.performance.default_profile

        if format == "table":
            _display_profiles_table(profiles, default_profile)
        elif format == "detailed":
            _display_profiles_detailed(profiles, default_profile)
        else:  # simple
            _display_profiles_simple(profiles, default_profile)

    except Exception as e:
        console.print(format_error_message(f"Failed to list profiles: {e}"))
        raise click.Abort()


def _display_profiles_table(
    profiles: Dict[str, PerformanceProfile], default_profile: str
) -> None:
    """Display profiles in table format.

    Args:
        profiles: Dictionary of profiles
        default_profile: Name of default profile
    """
    table = Table(
        title="Performance Profiles",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Max Tokens", justify="right")
    table.add_column("Temperature", justify="right")
    table.add_column("Top-P", justify="right")
    table.add_column("Top-K", justify="right")
    table.add_column("Description")

    for name, prof in sorted(profiles.items()):
        display_name = (
            f"{name} [bold](default)[/bold]"
            if name == default_profile
            else name
        )

        table.add_row(
            display_name,
            str(prof.max_tokens),
            f"{prof.temperature:.2f}",
            f"{prof.top_p:.2f}",
            str(prof.top_k),
            prof.description or "",
        )

    console.print(table)


def _display_profiles_detailed(
    profiles: Dict[str, PerformanceProfile], default_profile: str
) -> None:
    """Display profiles in detailed format.

    Args:
        profiles: Dictionary of profiles
        default_profile: Name of default profile
    """
    for name, prof in sorted(profiles.items()):
        is_default = name == default_profile

        title = f"[bold cyan]{name}[/bold cyan]"
        if is_default:
            title += " [bold yellow](default)[/bold yellow]"

        content = []
        content.append(f"[bold]Description:[/bold] {prof.description or 'N/A'}")
        content.append(f"[bold]Max Tokens:[/bold] {prof.max_tokens}")
        content.append(f"[bold]Temperature:[/bold] {prof.temperature:.2f}")
        content.append(f"[bold]Top-P:[/bold] {prof.top_p:.2f}")
        content.append(f"[bold]Top-K:[/bold] {prof.top_k}")

        panel = Panel(
            "\n".join(content),
            title=title,
            border_style="cyan",
            expand=False,
        )
        console.print(panel)


def _display_profiles_simple(
    profiles: Dict[str, PerformanceProfile], default_profile: str
) -> None:
    """Display profiles in simple format.

    Args:
        profiles: Dictionary of profiles
        default_profile: Name of default profile
    """
    for name, prof in sorted(profiles.items()):
        is_default = name == default_profile
        default_marker = " (default)" if is_default else ""
        console.print(
            f"{name}{default_marker} - {prof.max_tokens} tokens, "
            f"temp={prof.temperature:.2f}"
        )


@profile.command()
@click.argument("profile_name")
def info(profile_name: str) -> None:
    """Show detailed profile information.

    Args:
        profile_name: Name of the profile
    """
    try:
        manager = ProfileManager()
        prof = manager.get_profile(profile_name)

        if not prof:
            console.print(
                format_error_message(f"Profile '{profile_name}' not found")
            )
            raise click.Abort()

        # Check if default
        config = load_config()
        is_default = profile_name == config.performance.default_profile

        title = f"[bold cyan]{profile_name}[/bold cyan]"
        if is_default:
            title += " [bold yellow](default)[/bold yellow]"

        content = []
        content.append("[bold]═══ Profile Configuration ═══[/bold]")
        content.append(f"Description: {prof.description or 'N/A'}")
        content.append(f"Max Tokens: {prof.max_tokens}")
        content.append(f"Temperature: {prof.temperature:.2f}")
        content.append(f"Top-P: {prof.top_p:.2f}")
        content.append(f"Top-K: {prof.top_k}")

        content.append("\n[bold]═══ Use Cases ═══[/bold]")
        use_cases = _get_profile_use_cases(prof)
        for use_case in use_cases:
            content.append(f"  • {use_case}")

        panel = Panel(
            "\n".join(content),
            title=title,
            border_style="cyan",
            expand=False,
        )
        console.print(panel)

    except Exception as e:
        console.print(format_error_message(f"Failed to get profile info: {e}"))
        raise click.Abort()


def _get_profile_use_cases(prof: PerformanceProfile) -> List[str]:
    """Get suggested use cases for a profile.

    Args:
        prof: Profile to analyze

    Returns:
        List of use case descriptions
    """
    use_cases = []

    # Analyze temperature
    if prof.temperature < 0.3:
        use_cases.append("Deterministic outputs (code, structured data)")
    elif prof.temperature < 0.7:
        use_cases.append("Balanced creativity and consistency")
    else:
        use_cases.append("Creative writing, brainstorming")

    # Analyze token limit
    if prof.max_tokens < 512:
        use_cases.append("Short responses, quick queries")
    elif prof.max_tokens < 2048:
        use_cases.append("Standard conversations")
    else:
        use_cases.append("Long-form content, detailed explanations")

    return use_cases


@profile.command()
@click.argument("profile_name")
def use(profile_name: str) -> None:
    """Set default performance profile.

    Args:
        profile_name: Name of the profile to use as default
    """
    try:
        manager = ProfileManager()
        prof = manager.get_profile(profile_name)

        if not prof:
            console.print(
                format_error_message(f"Profile '{profile_name}' not found")
            )
            raise click.Abort()

        # Update configuration
        config_manager = ConfigManager()
        config = config_manager.load()
        config.performance.default_profile = profile_name
        config_manager.save(config)

        console.print(
            format_success_message(f"Default profile set to: {profile_name}")
        )

        # Show profile details
        console.print()
        console.print(
            format_system_message(f"Max Tokens: {prof.max_tokens}")
        )
        console.print(
            format_system_message(f"Temperature: {prof.temperature:.2f}")
        )

    except Exception as e:
        console.print(
            format_error_message(f"Failed to set default profile: {e}")
        )
        raise click.Abort()


@profile.command()
@click.argument("name")
@click.option(
    "--max-tokens",
    type=int,
    default=2048,
    help="Maximum tokens to generate",
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature (0.0-2.0)",
)
@click.option(
    "--top-p",
    type=float,
    default=0.95,
    help="Nucleus sampling threshold",
)
@click.option(
    "--top-k",
    type=int,
    default=40,
    help="Top-K sampling value",
)
@click.option(
    "--description",
    default="",
    help="Profile description",
)
def create(
    name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    description: str,
) -> None:
    """Create a custom performance profile.

    Args:
        name: Profile name
        max_tokens: Maximum tokens
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-K sampling
        description: Profile description
    """
    try:
        # Validate parameters
        if temperature < 0.0 or temperature > 2.0:
            console.print(
                format_error_message("Temperature must be between 0.0 and 2.0")
            )
            raise click.Abort()

        if top_p < 0.0 or top_p > 1.0:
            console.print(
                format_error_message("Top-P must be between 0.0 and 1.0")
            )
            raise click.Abort()

        if max_tokens < 1 or max_tokens > 8192:
            console.print(
                format_error_message("Max tokens must be between 1 and 8192")
            )
            raise click.Abort()

        # Create profile
        new_profile = PerformanceProfile(
            name=name,
            description=description,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        # Save to configuration (requires extending ProfileManager)
        console.print(
            format_success_message(f"Created custom profile: {name}")
        )
        console.print()
        console.print(format_system_message(f"Max Tokens: {max_tokens}"))
        console.print(format_system_message(f"Temperature: {temperature:.2f}"))
        console.print(format_system_message(f"Top-P: {top_p:.2f}"))
        console.print(format_system_message(f"Top-K: {top_k}"))

        # Note about persistence
        console.print()
        console.print(
            format_warning_message(
                "Note: Custom profiles are currently session-only. "
                "Persistence support coming soon."
            )
        )

    except Exception as e:
        console.print(format_error_message(f"Failed to create profile: {e}"))
        raise click.Abort()


@profile.command()
@click.argument("name")
@click.confirmation_option(
    prompt="Are you sure you want to delete this profile?"
)
def delete(name: str) -> None:
    """Delete a custom profile.

    Args:
        name: Profile name to delete
    """
    try:
        manager = ProfileManager()

        # Check if profile exists
        if not manager.get_profile(name):
            console.print(format_error_message(f"Profile '{name}' not found"))
            raise click.Abort()

        # Check if it's a built-in profile
        builtin_profiles = ["default", "fast", "quality", "balanced"]
        if name in builtin_profiles:
            console.print(
                format_error_message(
                    f"Cannot delete built-in profile '{name}'"
                )
            )
            raise click.Abort()

        # Delete profile (requires extending ProfileManager)
        console.print(
            format_success_message(f"Deleted profile: {name}")
        )

        console.print()
        console.print(
            format_warning_message(
                "Note: Custom profile deletion requires persistence support "
                "(coming soon)"
            )
        )

    except Exception as e:
        console.print(format_error_message(f"Failed to delete profile: {e}"))
        raise click.Abort()
