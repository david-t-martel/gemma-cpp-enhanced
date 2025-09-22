"""Configuration management commands for Gemma chatbot.

This module provides commands to manage configuration files, settings,
environment variables, and system setup.
"""

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Annotated
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from ..shared.config.settings import Settings
from .utils import get_console
from .utils import handle_exceptions

# Create config subcommand app
config_app = typer.Typer(
    name="config", help="⚙️ Configuration management commands", rich_markup_mode="rich"
)

console = get_console()


@config_app.command("show")
@handle_exceptions(console)
def show_config(
    section: Annotated[
        str | None, typer.Option("--section", "-s", help="Show specific configuration section")
    ] = None,
    format_output: Annotated[
        str, typer.Option("--format", "-f", help="Output format: json, yaml, table")
    ] = "table",
    include_defaults: Annotated[
        bool, typer.Option("--include-defaults", help="Include default values")
    ] = False,
    file_path: Annotated[
        Path | None, typer.Option("--file", help="Configuration file to show")
    ] = None,
) -> None:
    """Show current configuration."""
    asyncio.run(_show_configuration(section, format_output, include_defaults, file_path))


async def _show_configuration(
    section: str | None,
    format_output: str,
    include_defaults: bool,
    file_path: Path | None,
) -> None:
    """Show configuration implementation."""
    try:
        if file_path:
            # Show specific file
            if not file_path.exists():
                console.print(f"❌ Configuration file not found: {file_path}", style="red")
                return

            with open(file_path) as f:
                if file_path.suffix.lower() in [".yaml", ".yml"]:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)

            console.print(f"📖 Configuration from: {file_path}", style="blue")
        else:
            # Show current settings
            settings = Settings()
            config_data = settings.model_dump(exclude_defaults=not include_defaults)

            if not include_defaults:
                # Remove fields with default values
                config_data = _remove_default_values(config_data)

        # Filter by section if specified
        if section:
            if section in config_data:
                config_data = {section: config_data[section]}
            else:
                console.print(f"❌ Configuration section '{section}' not found", style="red")
                available_sections = list(config_data.keys())
                console.print(
                    f"Available sections: {', '.join(available_sections)}", style="yellow"
                )
                return

        # Display configuration
        if format_output == "json":
            json_str = json.dumps(config_data, indent=2, default=str)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
            console.print(syntax)

        elif format_output == "yaml":
            yaml_str = yaml.dump(config_data, default_flow_style=False, indent=2)
            syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
            console.print(syntax)

        else:  # table format
            _display_config_table(config_data, console)

    except Exception as e:
        console.print(f"❌ Error showing configuration: {e}", style="red")


def _remove_default_values(config_data: dict[str, Any]) -> dict[str, Any]:
    """Remove default values from configuration."""
    # This would need to be implemented with knowledge of default values
    # For now, return as-is
    return config_data


def _display_config_table(config_data: dict[str, Any], console: Console, prefix: str = "") -> None:
    """Display configuration as a table."""
    if not config_data:
        console.print("📭 No configuration data", style="yellow")
        return

    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    table.add_column("Type", style="dim")

    def add_config_rows(data: dict[str, Any], parent_key: str = ""):
        for key, value in data.items():
            full_key = f"{parent_key}.{key}" if parent_key else key

            if isinstance(value, dict):
                add_config_rows(value, full_key)
            else:
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."

                table.add_row(full_key, value_str, type(value).__name__)

    add_config_rows(config_data)
    console.print(table)


@config_app.command("set")
@handle_exceptions(console)
def set_config(
    key: Annotated[str, typer.Argument(help="Configuration key (e.g., 'model.temperature')")],
    value: Annotated[str, typer.Argument(help="Configuration value")],
    config_file: Annotated[
        Path | None, typer.Option("--file", "-f", help="Configuration file to modify")
    ] = None,
    create_backup: Annotated[
        bool, typer.Option("--backup/--no-backup", help="Create backup before modifying")
    ] = True,
    value_type: Annotated[
        str | None, typer.Option("--type", "-t", help="Value type: str, int, float, bool")
    ] = None,
) -> None:
    """Set a configuration value."""
    asyncio.run(_set_configuration(key, value, config_file, create_backup, value_type))


async def _set_configuration(
    key: str,
    value: str,
    config_file: Path | None,
    create_backup: bool,
    value_type: str | None,
) -> None:
    """Set configuration implementation."""
    try:
        # Determine config file
        if not config_file:
            # Use default settings file
            from ..shared.config.settings import get_config_path

            config_file = get_config_path()

        # Create config file if it doesn't exist
        if not config_file.exists():
            config_file.parent.mkdir(parents=True, exist_ok=True)
            initial_config = {}
            with open(config_file, "w") as f:
                json.dump(initial_config, f, indent=2)
            console.print(f"📄 Created new config file: {config_file}", style="green")

        # Create backup if requested
        if create_backup:
            backup_file = config_file.with_suffix(
                f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy2(config_file, backup_file)
            console.print(f"💾 Backup created: {backup_file}", style="blue")

        # Load existing config
        with open(config_file) as f:
            if config_file.suffix.lower() in [".yaml", ".yml"]:
                config_data = yaml.safe_load(f) or {}
            else:
                config_data = json.load(f)

        # Convert value to appropriate type
        typed_value = _convert_value_type(value, value_type)

        # Set nested key
        _set_nested_key(config_data, key, typed_value)

        # Save updated config
        with open(config_file, "w") as f:
            if config_file.suffix.lower() in [".yaml", ".yml"]:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            else:
                json.dump(config_data, f, indent=2, default=str)

        console.print(f"✅ Configuration updated: {key} = {typed_value}", style="green")
        console.print(f"📁 Config file: {config_file}", style="dim")

    except Exception as e:
        console.print(f"❌ Error setting configuration: {e}", style="red")


def _convert_value_type(value: str, value_type: str | None) -> Any:
    """Convert string value to appropriate type."""
    if not value_type:
        # Try to infer type
        if value.lower() in ["true", "false"]:
            return value.lower() == "true"
        elif value.isdigit():
            return int(value)
        elif value.replace(".", "", 1).isdigit():
            return float(value)
        else:
            return value

    if value_type == "bool":
        return value.lower() in ["true", "1", "yes", "on"]
    elif value_type == "int":
        return int(value)
    elif value_type == "float":
        return float(value)
    else:  # str
        return value


def _set_nested_key(data: dict[str, Any], key: str, value: Any) -> None:
    """Set a nested key in dictionary."""
    keys = key.split(".")
    current = data

    # Navigate to the nested location
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    # Set the final value
    current[keys[-1]] = value


@config_app.command("unset")
@handle_exceptions(console)
def unset_config(
    key: Annotated[str, typer.Argument(help="Configuration key to remove")],
    config_file: Annotated[
        Path | None, typer.Option("--file", "-f", help="Configuration file to modify")
    ] = None,
    create_backup: Annotated[
        bool, typer.Option("--backup/--no-backup", help="Create backup before modifying")
    ] = True,
) -> None:
    """Remove a configuration value."""
    asyncio.run(_unset_configuration(key, config_file, create_backup))


async def _unset_configuration(key: str, config_file: Path | None, create_backup: bool) -> None:
    """Remove configuration implementation."""
    try:
        # Determine config file
        if not config_file:
            from ..shared.config.settings import get_config_path

            config_file = get_config_path()

        if not config_file.exists():
            console.print(f"❌ Configuration file not found: {config_file}", style="red")
            return

        # Create backup if requested
        if create_backup:
            backup_file = config_file.with_suffix(
                f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy2(config_file, backup_file)
            console.print(f"💾 Backup created: {backup_file}", style="blue")

        # Load config
        with open(config_file) as f:
            if config_file.suffix.lower() in [".yaml", ".yml"]:
                config_data = yaml.safe_load(f) or {}
            else:
                config_data = json.load(f)

        # Remove nested key
        if _unset_nested_key(config_data, key):
            # Save updated config
            with open(config_file, "w") as f:
                if config_file.suffix.lower() in [".yaml", ".yml"]:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_data, f, indent=2, default=str)

            console.print(f"✅ Configuration key removed: {key}", style="green")
        else:
            console.print(f"❌ Configuration key not found: {key}", style="red")

    except Exception as e:
        console.print(f"❌ Error removing configuration: {e}", style="red")


def _unset_nested_key(data: dict[str, Any], key: str) -> bool:
    """Remove a nested key from dictionary."""
    keys = key.split(".")
    current = data

    # Navigate to the parent of the key to remove
    try:
        for k in keys[:-1]:
            current = current[k]

        # Remove the final key
        if keys[-1] in current:
            del current[keys[-1]]
            return True
        else:
            return False

    except (KeyError, TypeError):
        return False


@config_app.command("reset")
@handle_exceptions(console)
def reset_config(
    section: Annotated[
        str | None, typer.Option("--section", "-s", help="Reset specific section only")
    ] = None,
    config_file: Annotated[
        Path | None, typer.Option("--file", "-f", help="Configuration file to reset")
    ] = None,
    create_backup: Annotated[
        bool, typer.Option("--backup/--no-backup", help="Create backup before resetting")
    ] = True,
    force: Annotated[bool, typer.Option("--force", help="Skip confirmation prompt")] = False,
) -> None:
    """Reset configuration to defaults."""
    asyncio.run(_reset_configuration(section, config_file, create_backup, force))


async def _reset_configuration(
    section: str | None,
    config_file: Path | None,
    create_backup: bool,
    force: bool,
) -> None:
    """Reset configuration implementation."""
    try:
        # Determine config file
        if not config_file:
            from ..shared.config.settings import get_config_path

            config_file = get_config_path()

        if not config_file.exists():
            console.print(f"❌ Configuration file not found: {config_file}", style="red")
            return

        # Confirm action
        if not force:
            if section:
                if not Confirm.ask(f"Reset section '{section}' to defaults?", default=False):
                    console.print("❌ Reset cancelled", style="yellow")
                    return
            elif not Confirm.ask("Reset entire configuration to defaults?", default=False):
                console.print("❌ Reset cancelled", style="yellow")
                return

        # Create backup if requested
        if create_backup:
            backup_file = config_file.with_suffix(
                f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy2(config_file, backup_file)
            console.print(f"💾 Backup created: {backup_file}", style="blue")

        if section:
            # Reset specific section
            with open(config_file) as f:
                if config_file.suffix.lower() in [".yaml", ".yml"]:
                    config_data = yaml.safe_load(f) or {}
                else:
                    config_data = json.load(f)

            # Get default settings
            default_settings = Settings()
            default_data = default_settings.model_dump()

            if section in default_data:
                config_data[section] = default_data[section]

                # Save updated config
                with open(config_file, "w") as f:
                    if config_file.suffix.lower() in [".yaml", ".yml"]:
                        yaml.dump(config_data, f, default_flow_style=False, indent=2)
                    else:
                        json.dump(config_data, f, indent=2, default=str)

                console.print(f"✅ Section '{section}' reset to defaults", style="green")
            else:
                console.print(f"❌ Unknown section: {section}", style="red")

        else:
            # Reset entire configuration
            default_settings = Settings()
            default_data = default_settings.model_dump()

            with open(config_file, "w") as f:
                if config_file.suffix.lower() in [".yaml", ".yml"]:
                    yaml.dump(default_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(default_data, f, indent=2, default=str)

            console.print("✅ Configuration reset to defaults", style="green")

    except Exception as e:
        console.print(f"❌ Error resetting configuration: {e}", style="red")


@config_app.command("validate")
@handle_exceptions(console)
def validate_config(
    config_file: Annotated[
        Path | None, typer.Option("--file", "-f", help="Configuration file to validate")
    ] = None,
    schema_file: Annotated[
        Path | None, typer.Option("--schema", help="JSON schema file for validation")
    ] = None,
    fix_issues: Annotated[
        bool, typer.Option("--fix", help="Automatically fix common issues")
    ] = False,
) -> None:
    """Validate configuration file."""
    asyncio.run(_validate_configuration(config_file, schema_file, fix_issues))


async def _validate_configuration(
    config_file: Path | None,
    schema_file: Path | None,
    fix_issues: bool,
) -> None:
    """Validate configuration implementation."""
    try:
        # Determine config file
        if not config_file:
            from ..shared.config.settings import get_config_path

            config_file = get_config_path()

        if not config_file.exists():
            console.print(f"❌ Configuration file not found: {config_file}", style="red")
            return

        console.print(f"🔍 Validating configuration: {config_file}", style="blue")

        # Load and validate settings
        validation_errors = []
        warnings = []

        try:
            settings = Settings(_env_file=None)  # Load from file
            console.print("✅ Configuration syntax is valid", style="green")
        except Exception as e:
            validation_errors.append(f"Invalid configuration syntax: {e}")

        # Additional validation checks
        if not validation_errors:
            # Validate model settings
            if hasattr(settings, "model") and settings.model:
                if not settings.model.name:
                    validation_errors.append("Model name is required")
                if settings.model.temperature < 0 or settings.model.temperature > 2:
                    warnings.append("Temperature should be between 0 and 2")
                if settings.model.max_length <= 0:
                    validation_errors.append("Max length must be positive")

            # Validate performance settings
            if hasattr(settings, "performance") and settings.performance:
                valid_precisions = ["float32", "float16", "bfloat16", "int8", "int4"]
                if settings.performance.precision not in valid_precisions:
                    validation_errors.append(f"Invalid precision: {settings.performance.precision}")

        # Schema validation if provided
        if schema_file and schema_file.exists():
            try:
                import jsonschema

                with open(config_file) as f:
                    config_data = json.load(f)

                with open(schema_file) as f:
                    schema = json.load(f)

                jsonschema.validate(config_data, schema)
                console.print("✅ Schema validation passed", style="green")

            except ImportError:
                console.print(
                    "⚠️ jsonschema not installed, skipping schema validation", style="yellow"
                )
            except jsonschema.ValidationError as e:
                validation_errors.append(f"Schema validation failed: {e.message}")

        # Display results
        if validation_errors:
            console.print(
                f"\n❌ Validation failed with {len(validation_errors)} error(s):", style="red bold"
            )
            for i, error in enumerate(validation_errors, 1):
                console.print(f"  {i}. {error}", style="red")

            if fix_issues:
                console.print("\n🔧 Attempting to fix issues...", style="yellow")
                # Implement automatic fixes here
                await _auto_fix_config_issues(config_file, validation_errors)

        else:
            console.print("✅ Configuration is valid", style="green bold")

        if warnings:
            console.print(f"\n⚠️ {len(warnings)} warning(s):", style="yellow bold")
            for i, warning in enumerate(warnings, 1):
                console.print(f"  {i}. {warning}", style="yellow")

    except Exception as e:
        console.print(f"❌ Error validating configuration: {e}", style="red")


async def _auto_fix_config_issues(config_file: Path, errors: list[str]) -> None:
    """Automatically fix common configuration issues."""
    # This would implement automatic fixes for common issues
    console.print("🚧 Auto-fix functionality not yet implemented", style="yellow")


@config_app.command("init")
@handle_exceptions(console)
def init_config(
    config_dir: Annotated[
        Path | None, typer.Option("--dir", "-d", help="Configuration directory")
    ] = None,
    template: Annotated[
        str,
        typer.Option(
            "--template", "-t", help="Configuration template: basic, development, production"
        ),
    ] = "basic",
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite existing configuration")
    ] = False,
) -> None:
    """Initialize configuration with templates."""
    asyncio.run(_init_configuration(config_dir, template, overwrite))


async def _init_configuration(config_dir: Path | None, template: str, overwrite: bool) -> None:
    """Initialize configuration implementation."""
    try:
        # Determine config directory
        if not config_dir:
            config_dir = Path.home() / ".gemma"

        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.yaml"

        # Check if config already exists
        if config_file.exists() and not overwrite:
            console.print(f"❌ Configuration already exists: {config_file}", style="red")
            console.print("Use --overwrite to replace existing configuration", style="yellow")
            return

        console.print(f"🔧 Initializing {template} configuration...", style="blue")

        # Generate configuration template
        config_data = _generate_config_template(template)

        # Save configuration
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)

        console.print(f"✅ Configuration initialized: {config_file}", style="green")

        # Create additional files
        await _create_additional_config_files(config_dir, template)

        # Show configuration structure
        console.print("\n📁 Configuration directory structure:")
        _show_config_tree(config_dir, console)

    except Exception as e:
        console.print(f"❌ Error initializing configuration: {e}", style="red")


def _generate_config_template(template: str) -> dict[str, Any]:
    """Generate configuration template based on type."""
    from ..shared.config.agent_configs import DEFAULT_MODELS
    from ..shared.config.model_configs import get_default_model

    default_model = get_default_model("default")
    base_config = {
        "model": {
            "name": default_model.hf_model_id,
            "temperature": default_model.default_temperature,
            "max_length": default_model.context_length,
            "top_p": default_model.default_top_p,
            "top_k": default_model.default_top_k,
            "repetition_penalty": default_model.default_repetition_penalty,
            "do_sample": True,
        },
        "performance": {
            "device": "auto",
            "precision": "float16",
            "batch_size": 1,
            "use_cache": True,
            "use_flash_attention": True,
            "use_bettertransformer": True,
            "use_torch_compile": False,
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "cors": True,
            "log_level": "info",
        },
    }

    if template == "development":
        base_config["server"]["reload"] = True
        base_config["server"]["log_level"] = "debug"
        base_config["performance"]["precision"] = "float32"

    elif template == "production":
        base_config["server"]["workers"] = 4
        base_config["server"]["access_log"] = True
        base_config["performance"]["use_torch_compile"] = True
        base_config["logging"] = {
            "level": "info",
            "file": "gemma_server.log",
            "rotation": "daily",
            "retention": "7 days",
        }

    return base_config


async def _create_additional_config_files(config_dir: Path, template: str) -> None:
    """Create additional configuration files."""
    # Create .env file
    env_file = config_dir / ".env"
    env_content = """# Gemma Chatbot Environment Variables
# HUGGINGFACE_HUB_TOKEN=your_token_here
# GEMMA_API_KEY=your_api_key_here
# GEMMA_LOG_LEVEL=info
"""

    with open(env_file, "w") as f:
        f.write(env_content)

    # Create logging config
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler", "level": "INFO", "formatter": "standard"}
        },
        "root": {"level": "INFO", "handlers": ["console"]},
    }

    logging_file = config_dir / "logging.yaml"
    with open(logging_file, "w") as f:
        yaml.dump(logging_config, f, default_flow_style=False, indent=2)


def _show_config_tree(config_dir: Path, console: Console) -> None:
    """Show configuration directory tree."""
    tree = Tree(f"[bold blue]{config_dir}[/bold blue]")

    for item in sorted(config_dir.iterdir()):
        if item.is_file():
            size = item.stat().st_size
            size_str = f" [dim]({_format_bytes(size)})[/dim]"
            tree.add(f"[green]{item.name}[/green]{size_str}")
        elif item.is_dir():
            tree.add(f"[cyan]{item.name}/[/cyan]")

    console.print(tree)


def _format_bytes(size: int) -> str:
    """Format byte size as human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"


@config_app.command("export")
@handle_exceptions(console)
def export_config(
    output_file: Annotated[Path, typer.Argument(help="Output file path")],
    config_file: Annotated[
        Path | None, typer.Option("--file", "-f", help="Configuration file to export")
    ] = None,
    format_output: Annotated[
        str, typer.Option("--format", help="Output format: json, yaml")
    ] = "yaml",
    include_secrets: Annotated[
        bool, typer.Option("--include-secrets", help="Include sensitive values")
    ] = False,
) -> None:
    """Export configuration to a file."""
    asyncio.run(_export_configuration(output_file, config_file, format_output, include_secrets))


async def _export_configuration(
    output_file: Path,
    config_file: Path | None,
    format_output: str,
    include_secrets: bool,
) -> None:
    """Export configuration implementation."""
    try:
        # Load configuration
        if config_file:
            if not config_file.exists():
                console.print(f"❌ Configuration file not found: {config_file}", style="red")
                return

            with open(config_file) as f:
                if config_file.suffix.lower() in [".yaml", ".yml"]:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
        else:
            settings = Settings()
            config_data = settings.model_dump()

        # Remove secrets if not requested
        if not include_secrets:
            config_data = _remove_secrets(config_data)

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Export configuration
        with open(output_file, "w") as f:
            if format_output == "json":
                json.dump(config_data, f, indent=2, default=str)
            else:  # yaml
                yaml.dump(config_data, f, default_flow_style=False, indent=2)

        console.print(f"✅ Configuration exported to: {output_file}", style="green")

    except Exception as e:
        console.print(f"❌ Error exporting configuration: {e}", style="red")


def _remove_secrets(config_data: dict[str, Any]) -> dict[str, Any]:
    """Remove sensitive values from configuration."""
    sensitive_keys = ["api_key", "token", "password", "secret", "key"]

    def remove_sensitive(data):
        if isinstance(data, dict):
            return {
                k: (
                    "***hidden***"
                    if any(sensitive in k.lower() for sensitive in sensitive_keys)
                    else remove_sensitive(v)
                )
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [remove_sensitive(item) for item in data]
        else:
            return data

    return remove_sensitive(config_data)
