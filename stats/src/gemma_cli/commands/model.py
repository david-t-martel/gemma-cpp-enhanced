"""Model management commands for Gemma CLI.

Provides comprehensive model discovery, validation, profiling, and hardware detection.
"""

import asyncio
import platform
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import psutil
import toml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class ModelPreset:
    """Model preset information."""

    name: str
    size_gb: float
    format: str
    quality: str
    estimated_tokens_per_sec: int
    use_case: str
    model_path: Path
    tokenizer_path: Optional[Path] = None
    min_ram_gb: int = 8
    min_vram_gb: int = 0


@dataclass
class PerformanceProfile:
    """Performance profile configuration."""

    name: str
    description: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    context_length: int
    batch_size: int = 1


@dataclass
class HardwareInfo:
    """Hardware information."""

    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    os_platform: str = ""
    architecture: str = ""


# ============================================================================
# Model Presets
# ============================================================================

MODEL_PRESETS: Dict[str, ModelPreset] = {
    "gemma2-2b-it": ModelPreset(
        name="gemma2-2b-it",
        size_gb=2.5,
        format="SFP (8-bit)",
        quality="Good",
        estimated_tokens_per_sec=45,
        use_case="Fast iteration, testing, light tasks",
        model_path=Path("C:/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs"),
        tokenizer_path=Path(
            "C:/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/tokenizer.spm"
        ),
        min_ram_gb=8,
    ),
    "gemma3-4b-it-sfp": ModelPreset(
        name="gemma3-4b-it-sfp",
        size_gb=4.8,
        format="SFP (8-bit)",
        quality="High",
        estimated_tokens_per_sec=25,
        use_case="Balanced quality and speed",
        model_path=Path(
            "C:/codedev/llm/.models/gemma-3-gemmaCpp-3.0-4b-it-sfp-v1/4b-it-sfp.sbs"
        ),
        tokenizer_path=Path(
            "C:/codedev/llm/.models/gemma-3-gemmaCpp-3.0-4b-it-sfp-v1/tokenizer.spm"
        ),
        min_ram_gb=12,
    ),
    "gemma2-9b-it": ModelPreset(
        name="gemma2-9b-it",
        size_gb=9.2,
        format="SFP (8-bit)",
        quality="Very High",
        estimated_tokens_per_sec=12,
        use_case="High-quality responses, complex reasoning",
        model_path=Path("C:/codedev/llm/.models/gemma-9b-it/9b-it-sfp.sbs"),
        tokenizer_path=Path("C:/codedev/llm/.models/gemma-9b-it/tokenizer.spm"),
        min_ram_gb=16,
    ),
}

# ============================================================================
# Performance Profiles
# ============================================================================

PERFORMANCE_PROFILES: Dict[str, PerformanceProfile] = {
    "fast": PerformanceProfile(
        name="fast",
        description="Quick responses, lower quality",
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
        context_length=2048,
        batch_size=1,
    ),
    "balanced": PerformanceProfile(
        name="balanced",
        description="Balance between speed and quality",
        max_tokens=2048,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.15,
        context_length=4096,
        batch_size=1,
    ),
    "quality": PerformanceProfile(
        name="quality",
        description="Highest quality responses, slower",
        max_tokens=4096,
        temperature=0.9,
        top_p=0.98,
        top_k=100,
        repetition_penalty=1.2,
        context_length=8192,
        batch_size=1,
    ),
    "creative": PerformanceProfile(
        name="creative",
        description="Creative writing and brainstorming",
        max_tokens=2048,
        temperature=1.2,
        top_p=0.95,
        top_k=80,
        repetition_penalty=1.1,
        context_length=4096,
        batch_size=1,
    ),
    "precise": PerformanceProfile(
        name="precise",
        description="Factual, deterministic responses",
        max_tokens=1024,
        temperature=0.3,
        top_p=0.85,
        top_k=20,
        repetition_penalty=1.2,
        context_length=2048,
        batch_size=1,
    ),
}


# ============================================================================
# Utility Functions
# ============================================================================


def get_hardware_info() -> HardwareInfo:
    """Get current hardware information."""
    mem = psutil.virtual_memory()

    gpu_available = False
    gpu_name = None
    gpu_memory_gb = None

    # Try to detect GPU
    try:
        import GPUtil

        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_available = True
            gpu_name = gpu.name
            gpu_memory_gb = gpu.memoryTotal / 1024
    except ImportError:
        pass

    return HardwareInfo(
        cpu_model=platform.processor() or "Unknown",
        cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
        cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
        ram_total_gb=mem.total / (1024**3),
        ram_available_gb=mem.available / (1024**3),
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        os_platform=platform.system(),
        architecture=platform.machine(),
    )


def recommend_model(hw_info: HardwareInfo) -> List[str]:
    """Recommend models based on hardware."""
    recommendations = []

    for name, preset in MODEL_PRESETS.items():
        if hw_info.ram_total_gb >= preset.min_ram_gb:
            recommendations.append(name)

    return recommendations or ["gemma2-2b-it"]  # Default fallback


def recommend_profile(hw_info: HardwareInfo) -> str:
    """Recommend performance profile based on hardware."""
    if hw_info.ram_total_gb >= 32 and hw_info.cpu_cores_physical >= 8:
        return "quality"
    elif hw_info.ram_total_gb >= 16:
        return "balanced"
    else:
        return "fast"


def load_toml_config(config_path: Path) -> Dict[str, Any]:
    """Load TOML configuration file."""
    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return toml.load(f)


def save_toml_config(config_path: Path, config: Dict[str, Any]) -> None:
    """Save TOML configuration file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump(config, f)


# ============================================================================
# Model Commands
# ============================================================================


@click.group(name="model")
def model_group() -> None:
    """Model discovery and management."""
    pass


@model_group.command(name="list")
@click.option(
    "--path",
    type=click.Path(exists=True, path_type=Path),
    help="Custom models directory",
)
@click.option(
    "--show-presets",
    is_flag=True,
    default=True,
    help="Show built-in presets",
)
@click.option(
    "--show-discovered",
    is_flag=True,
    default=False,
    help="Auto-discover models in standard locations",
)
@click.pass_context
def list_models(
    ctx: click.Context,
    path: Optional[Path],
    show_presets: bool,
    show_discovered: bool,
) -> None:
    """List all available model presets.

    \b
    Shows a comprehensive table of:
    - Model name and size
    - Format (SFP, BF16, etc.)
    - Quality rating
    - Estimated speed (tokens/sec)
    - Recommended use cases

    \b
    Examples:
      gemma model list
      gemma model list --show-discovered
      gemma model list --path /custom/models
    """
    hw_info = get_hardware_info()
    recommendations = recommend_model(hw_info)

    # Show presets
    if show_presets:
        table = Table(
            title="Built-in Model Presets",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Name", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Format", style="yellow")
        table.add_column("Quality", style="blue")
        table.add_column("Speed", style="magenta")
        table.add_column("Use Case", style="white")
        table.add_column("Status", style="dim")

        for name, preset in sorted(MODEL_PRESETS.items()):
            # Check if model files exist
            exists = preset.model_path.exists()
            status = "✓ Available" if exists else "✗ Not Found"
            status_style = "green" if exists else "red"

            # Highlight recommended models
            name_display = (
                f"[bold]{name}[/bold] ⭐" if name in recommendations else name
            )

            table.add_row(
                name_display,
                f"{preset.size_gb:.1f} GB",
                preset.format,
                preset.quality,
                f"~{preset.estimated_tokens_per_sec} tok/s",
                preset.use_case,
                f"[{status_style}]{status}[/{status_style}]",
            )

        console.print(table)
        console.print(
            "\n[dim]⭐ Recommended for your hardware[/dim]",
            style="dim",
        )

    # Discover models in filesystem
    if show_discovered or path:
        models_dir = path or Path("C:/codedev/llm/.models")

        if not models_dir.exists():
            console.print(
                f"\n[yellow]Discovery path not found: {models_dir}[/yellow]"
            )
            return

        model_files = list(models_dir.rglob("*.sbs"))

        if model_files:
            console.print(f"\n[cyan]Discovered Models in {models_dir}:[/cyan]\n")

            disc_table = Table(show_header=True)
            disc_table.add_column("Name", style="cyan")
            disc_table.add_column("Size", style="green")
            disc_table.add_column("Type", style="yellow")
            disc_table.add_column("Path", style="dim")

            for model_file in sorted(model_files):
                size_gb = model_file.stat().st_size / (1024**3)
                model_name = model_file.stem

                # Detect model type
                if "sfp" in model_name.lower():
                    model_type = "SFP (8-bit)"
                elif "bf16" in model_name.lower():
                    model_type = "BF16"
                elif "nuq" in model_name.lower():
                    model_type = "NUQ (4-bit)"
                else:
                    model_type = "Unknown"

                disc_table.add_row(
                    model_name,
                    f"{size_gb:.2f} GB",
                    model_type,
                    str(model_file.relative_to(models_dir)),
                )

            console.print(disc_table)


@model_group.command(name="info")
@click.argument("model_name")
@click.pass_context
def model_info(ctx: click.Context, model_name: str) -> None:
    """Show detailed information about a model.

    \b
    Displays:
    - Full model specifications
    - File paths and sizes
    - Validation status
    - Hardware requirements
    - Performance characteristics

    \b
    Examples:
      gemma model info gemma2-2b-it
      gemma model info gemma3-4b-it-sfp
    """
    if model_name not in MODEL_PRESETS:
        console.print(f"[red]Model preset not found: {model_name}[/red]")
        console.print(
            f"\n[dim]Available presets: {', '.join(MODEL_PRESETS.keys())}[/dim]"
        )
        raise click.Abort()

    preset = MODEL_PRESETS[model_name]
    hw_info = get_hardware_info()

    # Model info panel
    info_lines = [
        f"[cyan]Name:[/cyan] {preset.name}",
        f"[cyan]Size:[/cyan] {preset.size_gb:.2f} GB",
        f"[cyan]Format:[/cyan] {preset.format}",
        f"[cyan]Quality:[/cyan] {preset.quality}",
        f"[cyan]Speed:[/cyan] ~{preset.estimated_tokens_per_sec} tokens/sec",
        f"[cyan]Use Case:[/cyan] {preset.use_case}",
    ]

    console.print(Panel("\n".join(info_lines), title="Model Information"))

    # Hardware requirements
    req_lines = [
        f"[cyan]Minimum RAM:[/cyan] {preset.min_ram_gb} GB",
        f"[cyan]Minimum VRAM:[/cyan] {preset.min_vram_gb} GB (CPU-only supported)",
    ]

    # Check if hardware meets requirements
    ram_ok = hw_info.ram_total_gb >= preset.min_ram_gb
    ram_status = "✓" if ram_ok else "✗"
    ram_color = "green" if ram_ok else "red"

    req_lines.append(
        f"\n[cyan]Your RAM:[/cyan] [{ram_color}]{ram_status} {hw_info.ram_total_gb:.1f} GB[/{ram_color}]"
    )

    console.print(Panel("\n".join(req_lines), title="Hardware Requirements"))

    # File paths
    model_exists = preset.model_path.exists()
    tokenizer_exists = (
        preset.tokenizer_path.exists() if preset.tokenizer_path else False
    )

    file_lines = [
        f"[cyan]Model Path:[/cyan]",
        f"  {preset.model_path}",
        f"  [{'green' if model_exists else 'red'}]{'✓ Exists' if model_exists else '✗ Not Found'}[/]",
    ]

    if preset.tokenizer_path:
        file_lines.extend(
            [
                f"\n[cyan]Tokenizer Path:[/cyan]",
                f"  {preset.tokenizer_path}",
                f"  [{'green' if tokenizer_exists else 'red'}]{'✓ Exists' if tokenizer_exists else '✗ Not Found'}[/]",
            ]
        )

    console.print(Panel("\n".join(file_lines), title="File Locations"))

    # Overall validation
    if model_exists and (not preset.tokenizer_path or tokenizer_exists) and ram_ok:
        console.print("\n[bold green]✓ Model ready to use[/bold green]")
    else:
        console.print("\n[bold red]✗ Model not ready[/bold red]")
        if not model_exists:
            console.print("  [red]- Model file not found[/red]")
        if preset.tokenizer_path and not tokenizer_exists:
            console.print("  [red]- Tokenizer file not found[/red]")
        if not ram_ok:
            console.print(
                f"  [red]- Insufficient RAM (need {preset.min_ram_gb} GB, have {hw_info.ram_total_gb:.1f} GB)[/red]"
            )


@model_group.command(name="use")
@click.argument("model_name")
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    default=Path("config/config.toml"),
    help="Configuration file path",
)
@click.pass_context
def use_model(ctx: click.Context, model_name: str, config: Path) -> None:
    """Set default model for CLI operations.

    Updates the configuration file with the specified model as default.

    \b
    Examples:
      gemma model use gemma2-2b-it
      gemma model use gemma3-4b-it-sfp --config custom.toml
    """
    if model_name not in MODEL_PRESETS:
        console.print(f"[red]Model preset not found: {model_name}[/red]")
        console.print(
            f"\n[dim]Available presets: {', '.join(MODEL_PRESETS.keys())}[/dim]"
        )
        raise click.Abort()

    preset = MODEL_PRESETS[model_name]

    # Validate model exists
    if not preset.model_path.exists():
        console.print(
            f"[red]Model file not found: {preset.model_path}[/red]",
        )
        console.print(
            "\n[yellow]Hint:[/yellow] Download models from Kaggle or use 'gemma model detect' to scan for local models"
        )
        raise click.Abort()

    # Load or create config
    config_data = load_toml_config(config)

    # Update model section
    if "model" not in config_data:
        config_data["model"] = {}

    config_data["model"]["default_model"] = model_name
    config_data["model"]["model_path"] = str(preset.model_path)
    if preset.tokenizer_path:
        config_data["model"]["tokenizer_path"] = str(preset.tokenizer_path)

    # Save config
    save_toml_config(config, config_data)

    console.print(f"[green]✓ Default model set to:[/green] {model_name}")
    console.print(f"[dim]Configuration saved to: {config}[/dim]")


@model_group.command(name="detect")
@click.option(
    "--path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("C:/codedev/llm/.models"),
    help="Directory to scan for models",
)
@click.option(
    "--auto-configure",
    is_flag=True,
    help="Automatically update config with found models",
)
@click.pass_context
def detect_models(
    ctx: click.Context, path: Path, auto_configure: bool
) -> None:
    """Auto-detect models in standard locations.

    Scans for .sbs model files and .spm tokenizer files in the specified
    directory and its subdirectories.

    \b
    Examples:
      gemma model detect
      gemma model detect --path /custom/models
      gemma model detect --auto-configure
    """
    console.print(f"[cyan]Scanning for models in:[/cyan] {path}\n")

    with console.status("[cyan]Scanning...") as status:
        model_files = list(path.rglob("*.sbs"))
        tokenizer_files = list(path.rglob("*.spm"))

    if not model_files:
        console.print("[yellow]No model files found (.sbs)[/yellow]")
        return

    console.print(f"[green]Found {len(model_files)} model(s)[/green]\n")

    table = Table(show_header=True, title="Detected Models")
    table.add_column("Model", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Format", style="yellow")
    table.add_column("Tokenizer", style="blue")
    table.add_column("Path", style="dim")

    for model_file in sorted(model_files):
        size_gb = model_file.stat().st_size / (1024**3)
        model_name = model_file.stem

        # Detect format
        if "sfp" in model_name.lower():
            format_type = "SFP (8-bit)"
        elif "bf16" in model_name.lower():
            format_type = "BF16"
        elif "nuq" in model_name.lower():
            format_type = "NUQ (4-bit)"
        else:
            format_type = "Unknown"

        # Find matching tokenizer
        tokenizer = "Not Found"
        for tok_file in tokenizer_files:
            if tok_file.parent == model_file.parent:
                tokenizer = "✓ Found"
                break

        table.add_row(
            model_name,
            f"{size_gb:.2f} GB",
            format_type,
            tokenizer,
            str(model_file.relative_to(path)),
        )

    console.print(table)

    if auto_configure:
        console.print(
            "\n[yellow]Auto-configuration not yet implemented[/yellow]"
        )


@model_group.command(name="validate")
@click.argument("model_name")
@click.pass_context
def validate_model(ctx: click.Context, model_name: str) -> None:
    """Validate model files exist and are readable.

    Performs comprehensive validation:
    - File existence checks
    - File permissions
    - File integrity (basic)
    - Hardware compatibility

    \b
    Examples:
      gemma model validate gemma2-2b-it
      gemma model validate gemma3-4b-it-sfp
    """
    if model_name not in MODEL_PRESETS:
        console.print(f"[red]Model preset not found: {model_name}[/red]")
        raise click.Abort()

    preset = MODEL_PRESETS[model_name]
    hw_info = get_hardware_info()

    console.print(f"[cyan]Validating model:[/cyan] {model_name}\n")

    checks = []

    # Check model file exists
    model_exists = preset.model_path.exists()
    checks.append(
        (
            "Model file exists",
            model_exists,
            str(preset.model_path) if model_exists else "File not found",
        )
    )

    # Check model file readable
    if model_exists:
        try:
            with open(preset.model_path, "rb") as f:
                f.read(1024)  # Read first KB
            checks.append(("Model file readable", True, "Can read file"))
        except Exception as e:
            checks.append(("Model file readable", False, str(e)))

    # Check tokenizer
    if preset.tokenizer_path:
        tokenizer_exists = preset.tokenizer_path.exists()
        checks.append(
            (
                "Tokenizer file exists",
                tokenizer_exists,
                str(preset.tokenizer_path) if tokenizer_exists else "File not found",
            )
        )

    # Check RAM requirement
    ram_ok = hw_info.ram_total_gb >= preset.min_ram_gb
    checks.append(
        (
            "RAM requirement",
            ram_ok,
            f"{hw_info.ram_total_gb:.1f} GB / {preset.min_ram_gb} GB required",
        )
    )

    # Display results
    table = Table(show_header=True, title="Validation Results")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Details", style="dim")

    all_passed = True
    for check_name, passed, details in checks:
        status = "[green]✓ Pass[/green]" if passed else "[red]✗ Fail[/red]"
        table.add_row(check_name, status, details)
        if not passed:
            all_passed = False

    console.print(table)

    if all_passed:
        console.print("\n[bold green]✓ All validation checks passed[/bold green]")
    else:
        console.print("\n[bold red]✗ Some validation checks failed[/bold red]")
        raise click.Abort()


# ============================================================================
# Profile Commands
# ============================================================================


@click.group(name="profile")
def profile_group() -> None:
    """Performance profile management."""
    pass


@profile_group.command(name="list")
@click.pass_context
def list_profiles(ctx: click.Context) -> None:
    """List all performance profiles.

    Shows a comprehensive table of built-in profiles with their parameters
    and recommended use cases.

    \b
    Example:
      gemma profile list
    """
    hw_info = get_hardware_info()
    recommended = recommend_profile(hw_info)

    table = Table(
        title="Performance Profiles",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Name", style="cyan")
    table.add_column("Max Tokens", style="green")
    table.add_column("Temp", style="yellow")
    table.add_column("Top-P", style="blue")
    table.add_column("Context", style="magenta")
    table.add_column("Description", style="white")

    for name, profile in sorted(PERFORMANCE_PROFILES.items()):
        # Highlight recommended profile
        name_display = (
            f"[bold]{name}[/bold] ⭐" if name == recommended else name
        )

        table.add_row(
            name_display,
            str(profile.max_tokens),
            f"{profile.temperature:.1f}",
            f"{profile.top_p:.2f}",
            str(profile.context_length),
            profile.description,
        )

    console.print(table)
    console.print(
        "\n[dim]⭐ Recommended for your hardware[/dim]",
        style="dim",
    )


@profile_group.command(name="info")
@click.argument("profile_name")
@click.pass_context
def profile_info(ctx: click.Context, profile_name: str) -> None:
    """Show detailed profile information.

    \b
    Displays:
    - All profile parameters
    - Performance characteristics
    - Use case recommendations

    \b
    Examples:
      gemma profile info balanced
      gemma profile info fast
    """
    if profile_name not in PERFORMANCE_PROFILES:
        console.print(f"[red]Profile not found: {profile_name}[/red]")
        console.print(
            f"\n[dim]Available profiles: {', '.join(PERFORMANCE_PROFILES.keys())}[/dim]"
        )
        raise click.Abort()

    profile = PERFORMANCE_PROFILES[profile_name]

    # Profile details
    details = [
        f"[cyan]Name:[/cyan] {profile.name}",
        f"[cyan]Description:[/cyan] {profile.description}",
        "",
        "[bold cyan]Generation Parameters:[/bold cyan]",
        f"  Max Tokens: {profile.max_tokens}",
        f"  Temperature: {profile.temperature}",
        f"  Top-P: {profile.top_p}",
        f"  Top-K: {profile.top_k}",
        f"  Repetition Penalty: {profile.repetition_penalty}",
        "",
        "[bold cyan]Context Configuration:[/bold cyan]",
        f"  Context Length: {profile.context_length} tokens",
        f"  Batch Size: {profile.batch_size}",
    ]

    console.print(Panel("\n".join(details), title=f"Profile: {profile_name}"))


@profile_group.command(name="use")
@click.argument("profile_name")
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    default=Path("config/config.toml"),
    help="Configuration file path",
)
@click.pass_context
def use_profile(ctx: click.Context, profile_name: str, config: Path) -> None:
    """Set active performance profile.

    Updates the configuration file with the specified profile parameters.

    \b
    Examples:
      gemma profile use balanced
      gemma profile use fast --config custom.toml
    """
    if profile_name not in PERFORMANCE_PROFILES:
        console.print(f"[red]Profile not found: {profile_name}[/red]")
        console.print(
            f"\n[dim]Available profiles: {', '.join(PERFORMANCE_PROFILES.keys())}[/dim]"
        )
        raise click.Abort()

    profile = PERFORMANCE_PROFILES[profile_name]

    # Load or create config
    config_data = load_toml_config(config)

    # Update generation section
    if "generation" not in config_data:
        config_data["generation"] = {}

    config_data["generation"]["max_tokens"] = profile.max_tokens
    config_data["generation"]["temperature"] = profile.temperature
    config_data["generation"]["top_p"] = profile.top_p
    config_data["generation"]["top_k"] = profile.top_k
    config_data["generation"]["repetition_penalty"] = profile.repetition_penalty
    config_data["generation"]["max_context"] = profile.context_length

    # Save config
    save_toml_config(config, config_data)

    console.print(f"[green]✓ Active profile set to:[/green] {profile_name}")
    console.print(f"[dim]Configuration saved to: {config}[/dim]")


@profile_group.command(name="create")
@click.argument("name")
@click.option("--max-tokens", type=int, default=2048, help="Maximum tokens to generate")
@click.option("--temperature", type=float, default=0.7, help="Sampling temperature")
@click.option("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
@click.option("--top-k", type=int, default=40, help="Top-k sampling")
@click.option(
    "--repetition-penalty", type=float, default=1.1, help="Repetition penalty"
)
@click.option(
    "--context-length", type=int, default=4096, help="Maximum context length"
)
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    default=Path("config/config.toml"),
    help="Configuration file path",
)
@click.pass_context
def create_profile(
    ctx: click.Context,
    name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    context_length: int,
    config: Path,
) -> None:
    """Create custom performance profile.

    \b
    Examples:
      gemma profile create myprofile --max-tokens 1024 --temperature 0.8
      gemma profile create research --temperature 0.3 --top-p 0.85
    """
    # Load or create config
    config_data = load_toml_config(config)

    # Create custom profiles section
    if "custom_profiles" not in config_data:
        config_data["custom_profiles"] = {}

    # Add new profile
    config_data["custom_profiles"][name] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "context_length": context_length,
    }

    # Save config
    save_toml_config(config, config_data)

    console.print(f"[green]✓ Custom profile created:[/green] {name}")
    console.print(f"[dim]Configuration saved to: {config}[/dim]")

    # Display profile details
    details = [
        f"Max Tokens: {max_tokens}",
        f"Temperature: {temperature}",
        f"Top-P: {top_p}",
        f"Top-K: {top_k}",
        f"Repetition Penalty: {repetition_penalty}",
        f"Context Length: {context_length}",
    ]

    console.print(Panel("\n".join(details), title=f"Profile: {name}"))


# ============================================================================
# Hardware Command
# ============================================================================


@model_group.command(name="hardware")
@click.option(
    "--show-recommendations",
    is_flag=True,
    default=True,
    help="Show model recommendations",
)
@click.pass_context
def hardware_info(ctx: click.Context, show_recommendations: bool) -> None:
    """Display hardware information and recommendations.

    \b
    Shows:
    - CPU details and core count
    - RAM capacity and usage
    - GPU availability (if detected)
    - Recommended models and profiles

    \b
    Example:
      gemma model hardware
    """
    hw_info = get_hardware_info()

    # System information
    sys_lines = [
        f"[cyan]Platform:[/cyan] {hw_info.os_platform} ({hw_info.architecture})",
        f"[cyan]CPU:[/cyan] {hw_info.cpu_model}",
        f"  Physical Cores: {hw_info.cpu_cores_physical}",
        f"  Logical Cores: {hw_info.cpu_cores_logical}",
        "",
        f"[cyan]RAM:[/cyan]",
        f"  Total: {hw_info.ram_total_gb:.1f} GB",
        f"  Available: {hw_info.ram_available_gb:.1f} GB",
        f"  Usage: {((hw_info.ram_total_gb - hw_info.ram_available_gb) / hw_info.ram_total_gb * 100):.1f}%",
    ]

    if hw_info.gpu_available and hw_info.gpu_name:
        sys_lines.extend(
            [
                "",
                f"[cyan]GPU:[/cyan]",
                f"  Name: {hw_info.gpu_name}",
                f"  Memory: {hw_info.gpu_memory_gb:.1f} GB",
            ]
        )
    else:
        sys_lines.extend(["", "[cyan]GPU:[/cyan] Not detected (CPU-only mode)"])

    console.print(Panel("\n".join(sys_lines), title="Hardware Information"))

    # Recommendations
    if show_recommendations:
        recommended_models = recommend_model(hw_info)
        recommended_profile = recommend_profile(hw_info)

        rec_lines = [
            "[bold cyan]Recommended Models:[/bold cyan]",
        ]

        for model_name in recommended_models:
            preset = MODEL_PRESETS[model_name]
            exists = preset.model_path.exists()
            status = "✓" if exists else "✗"
            rec_lines.append(
                f"  [{('green' if exists else 'red')}]{status}[/] {model_name} ({preset.size_gb:.1f} GB)"
            )

        rec_lines.extend(
            [
                "",
                "[bold cyan]Recommended Profile:[/bold cyan]",
                f"  {recommended_profile} - {PERFORMANCE_PROFILES[recommended_profile].description}",
            ]
        )

        console.print(Panel("\n".join(rec_lines), title="Recommendations"))


# ============================================================================
# Register commands to group
# ============================================================================

# Profile commands are separate group - add to model group
model_group.add_command(profile_group, name="profile")
