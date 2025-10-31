"""CLI utilities and helpers for Gemma CLI.

This module provides common utilities, formatting functions, error handling,
and helper functions used across all CLI commands.
"""

import functools
import logging
import platform
import sys
from collections.abc import Callable
from typing import Any
from typing import TypeVar

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text

# Type variable for decorators
F = TypeVar("F", bound=Callable[..., Any])

# Global console instance
_console: Console | None = None


def get_console() -> Console:
    """Get the global Rich console instance."""
    global _console
    if _console is None:
        _console = Console(
            color_system="auto",
            force_terminal=True,
            legacy_windows=False,
        )
    return _console


def setup_logging(verbose: bool = False, log_file: str | None = None) -> None:
    """Setup logging configuration with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    handlers = [RichHandler(console=get_console(), rich_tracebacks=True)]

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, format="%(message)s")


def handle_exceptions(console: Console) -> Callable[[F], F]:
    """Decorator to handle exceptions in CLI commands."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                console.print("\n⏸️  Operation interrupted by user", style="yellow")
                raise typer.Exit(1)
            except typer.Exit:
                # Re-raise typer exits
                raise
            except Exception as e:
                console.print(f"❌ Unexpected error: {e}", style="red")
                if "--verbose" in sys.argv or "-v" in sys.argv:
                    import traceback

                    console.print(traceback.format_exc(), style="dim red")
                raise typer.Exit(1)

        return wrapper

    return decorator


def show_banner(console: Console) -> None:
    """Show the application banner."""
    banner_text = Text()
    banner_text.append("🤖 ", style="bold blue")
    banner_text.append("Gemma CLI", style="bold cyan")
    banner_text.append(" - Your AI Assistant\n", style="bold blue")
    banner_text.append("Powered by Google Gemma", style="dim")

    panel = Panel(banner_text, border_style="blue", padding=(1, 2))
    console.print(panel)
    console.print()
