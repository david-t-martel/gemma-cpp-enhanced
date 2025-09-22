"""Main CLI entry point for Gemma chatbot.

This module provides the main Typer application that coordinates all CLI commands
including chat, training, serving, and configuration management.
"""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel
from rich.text import Text

from .chat import chat_app
from .config import config_app
from .serve import serve_app
from .train import train_app
from .utils import get_console
from .utils import handle_exceptions
from .utils import setup_logging
from .utils import show_banner
from .utils import validate_environment

# Create the main Typer application
app = typer.Typer(
    name="gemma-cli",
    help="🤖 Gemma Chatbot CLI - A comprehensive interface for Google Gemma models",
    rich_markup_mode="rich",
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Add subcommands
app.add_typer(chat_app, name="chat", help="💬 Interactive chat commands")
app.add_typer(train_app, name="train", help="🎓 Training and fine-tuning commands")
app.add_typer(serve_app, name="serve", help="🚀 Server management commands")
app.add_typer(config_app, name="config", help="⚙️  Configuration management commands")

console = get_console()


@app.command()
@handle_exceptions(console)
def version(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed version information")
    ] = False,
) -> None:
    """Show version information."""
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as get_version

        try:
            version_str = get_version("gemma-chatbot")
        except PackageNotFoundError:
            version_str = "development"
    except ImportError:
        version_str = "unknown"

    if verbose:
        import platform
        import sys

        import torch
        import transformers

        info_text = Text()
        info_text.append("Gemma Chatbot CLI\n", style="bold cyan")
        info_text.append(f"Version: {version_str}\n", style="white")
        info_text.append(f"Python: {sys.version.split()[0]}\n", style="dim")
        info_text.append(f"Platform: {platform.platform()}\n", style="dim")
        info_text.append(f"PyTorch: {torch.__version__}\n", style="dim")
        info_text.append(f"Transformers: {transformers.__version__}\n", style="dim")

        if torch.cuda.is_available():
            info_text.append(f"CUDA: {torch.version.cuda}\n", style="green")
            info_text.append(f"GPU Count: {torch.cuda.device_count()}\n", style="green")
        else:
            info_text.append("CUDA: Not available\n", style="yellow")

        panel = Panel(info_text, title="System Information", border_style="cyan", padding=(1, 2))
        console.print(panel)
    else:
        console.print(f"Gemma Chatbot CLI v{version_str}", style="bold cyan")


@app.command()
@handle_exceptions(console)
def status() -> None:
    """Show system and environment status."""
    from .utils import check_system_status

    status_info = check_system_status()

    # Create status display
    status_text = Text()

    # Environment status
    env_status = "✅ Ready" if status_info["environment"]["ready"] else "❌ Issues detected"
    status_text.append(
        f"Environment: {env_status}\n",
        style="green" if status_info["environment"]["ready"] else "red",
    )

    # GPU status
    gpu_status = "✅ Available" if status_info["gpu"]["available"] else "⚠️  Not available"
    status_text.append(
        f"GPU: {gpu_status}", style="green" if status_info["gpu"]["available"] else "yellow"
    )

    if status_info["gpu"]["available"]:
        status_text.append(f" ({status_info['gpu']['device_count']} device(s))\n")

        # Memory info if available
        if "memory" in status_info["gpu"]:
            mem_info = status_info["gpu"]["memory"]
            status_text.append(
                f"  Memory: {mem_info['used']:.1f}GB / {mem_info['total']:.1f}GB\n", style="dim"
            )
    else:
        status_text.append("\n")

    # Models status
    model_count = status_info["models"]["available_count"]
    status_text.append(f"Available Models: {model_count}\n", style="blue")

    # Server status (if running)
    if "server" in status_info and status_info["server"]["running"]:
        status_text.append("Server: ✅ Running", style="green")
        status_text.append(f" (Port: {status_info['server']['port']})\n", style="dim")
    else:
        status_text.append("Server: ⏹️  Stopped\n", style="yellow")

    # Configuration status
    config_status = "✅ Valid" if status_info["config"]["valid"] else "⚠️  Issues"
    status_text.append(
        f"Configuration: {config_status}\n",
        style="green" if status_info["config"]["valid"] else "yellow",
    )

    panel = Panel(status_text, title="System Status", border_style="blue", padding=(1, 2))
    console.print(panel)

    # Show any warnings or issues
    if status_info["warnings"]:
        console.print("\n⚠️  Warnings:", style="yellow bold")
        for warning in status_info["warnings"]:
            console.print(f"  • {warning}", style="yellow")

    if status_info["errors"]:
        console.print("\n❌ Errors:", style="red bold")
        for error in status_info["errors"]:
            console.print(f"  • {error}", style="red")


@app.command("quick-chat")
@handle_exceptions(console)
def quick_chat(
    message: Annotated[str, typer.Argument(help="Message to send to the chatbot")],
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Model to use for generation")
    ] = None,
    temperature: Annotated[
        float, typer.Option("--temperature", "-t", help="Generation temperature", min=0.0, max=2.0)
    ] = 0.7,
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", help="Maximum tokens to generate", min=1)
    ] = 512,
    stream: Annotated[bool, typer.Option("--stream", "-s", help="Enable streaming output")] = False,
) -> None:
    """Send a quick message to the chatbot without entering interactive mode."""
    from .chat import quick_generate

    console.print("🤖 [bold blue]Generating response...[/bold blue]")

    # Run the quick generation
    asyncio.run(
        quick_generate(
            message=message,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            console=console,
        )
    )


@app.command()
@handle_exceptions(console)
def models(
    list_all: Annotated[
        bool, typer.Option("--all", "-a", help="List all available models (including remote)")
    ] = False,
    download: Annotated[
        str | None, typer.Option("--download", "-d", help="Download a specific model")
    ] = None,
    info: Annotated[
        str | None, typer.Option("--info", "-i", help="Show detailed information about a model")
    ] = None,
) -> None:
    """Manage available models."""
    from .utils import download_model
    from .utils import get_model_info
    from .utils import list_models

    if download:
        console.print(f"📥 Downloading model: [bold]{download}[/bold]")
        success = asyncio.run(download_model(download, console))
        if success:
            console.print("✅ Model downloaded successfully!", style="green")
        else:
            console.print("❌ Failed to download model", style="red")
            raise typer.Exit(1)

    elif info:
        console.print(f"📋 Model information for: [bold]{info}[/bold]")
        model_info = asyncio.run(get_model_info(info))

        if model_info:
            from .utils import display_model_info

            display_model_info(model_info, console)
        else:
            console.print(f"❌ Model '{info}' not found", style="red")
            raise typer.Exit(1)

    else:
        console.print("📚 Available models:", style="bold blue")
        models_list = asyncio.run(list_models(include_remote=list_all))

        if not models_list:
            console.print("No models found. Use --download to download a model.", style="yellow")
            return

        from .utils import display_models_table

        display_models_table(models_list, console)


@app.callback()
def main(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
    log_file: Annotated[Path | None, typer.Option("--log-file", help="Log file path")] = None,
    no_banner: Annotated[bool, typer.Option("--no-banner", help="Skip the startup banner")] = False,
) -> None:
    """Gemma Chatbot CLI - A comprehensive interface for Google Gemma models.

    This CLI provides commands for interactive chat, model training, server management,
    and configuration. Get started with 'gemma-cli chat' for interactive mode or
    'gemma-cli quick-chat "your message"' for a one-off response.
    """
    # Setup logging
    setup_logging(verbose=verbose, log_file=log_file)

    # Validate environment
    validation_result = validate_environment()
    if not validation_result["success"] and validation_result["critical_errors"]:
        console.print("❌ Critical environment issues detected:", style="red bold")
        for error in validation_result["critical_errors"]:
            console.print(f"  • {error}", style="red")
        console.print("\nPlease resolve these issues before continuing.", style="yellow")
        raise typer.Exit(1)

    # Show banner unless disabled
    if not no_banner:
        show_banner(console)

        # Show warnings if any
        if validation_result["warnings"]:
            console.print("⚠️  Warnings:", style="yellow bold")
            for warning in validation_result["warnings"]:
                console.print(f"  • {warning}", style="yellow")
            console.print()


if __name__ == "__main__":
    app()
