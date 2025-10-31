"""Main CLI entry point for Gemma CLI.

This module provides the main Typer application that coordinates all CLI commands
including chat, and configuration management.
"""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel
from rich.text import Text

from .utils import get_console, setup_logging, handle_exceptions, show_banner
from .cli.chat import chat_app
from .cli.serve import serve_app
from .cli.rag import rag_app

# Create the main Typer application
app = typer.Typer(
    name="gemma-cli",
    help="🤖 Gemma CLI - A comprehensive interface for Google Gemma models",
    rich_markup_mode="rich",
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(chat_app, name="chat", help="💬 Interactive chat commands")
app.add_typer(serve_app, name="serve", help="🚀 Server management commands")
app.add_typer(rag_app, name="rag", help="🚀 RAG commands")

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
            version_str = get_version("gemma-cli")
        except PackageNotFoundError:
            version_str = "development"
    except ImportError:
        version_str = "unknown"

    if verbose:
        import platform
        import sys

        info_text = Text()
        info_text.append("Gemma CLI\n", style="bold cyan")
        info_text.append(f"Version: {version_str}\n", style="white")
        info_text.append(f"Python: {sys.version.split()[0]}\n", style="dim")
        info_text.append(f"Platform: {platform.platform()}\n", style="dim")

        panel = Panel(info_text, title="System Information", border_style="cyan", padding=(1, 2))
        console.print(panel)
    else:
        console.print(f"Gemma CLI v{version_str}", style="bold cyan")


@app.command()
@handle_exceptions(console)
def status() -> None:
    """Show system and environment status."""
    version_str = "development"
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as get_version
        try:
            version_str = get_version("gemma-cli")
        except PackageNotFoundError:
            pass
    except ImportError:
        pass
    console.print(f"Gemma CLI v{version_str}", style="bold cyan")


@app.callback()
def main(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
    log_file: Annotated[Path | None, typer.Option("--log-file", help="Log file path")] = None,
    no_banner: Annotated[bool, typer.Option("--no-banner", help="Skip the startup banner")] = False,
) -> None:
    """Gemma CLI - A comprehensive interface for Google Gemma models.

    This CLI provides commands for interactive chat, model training, server management,
    and configuration. Get started with 'gemma-cli chat' for interactive mode or
    'gemma-cli quick-chat "your message"' for a one-off response.
    """
    setup_logging(verbose=verbose, log_file=log_file)
    if not no_banner:
        show_banner(console)


if __name__ == "__main__":
    app()
