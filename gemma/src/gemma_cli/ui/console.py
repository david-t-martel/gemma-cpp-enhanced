import sys
import warnings
from io import StringIO
from unittest.mock import MagicMock

from rich.console import Console

_console = None


def create_console() -> Console:
    """Create a new console instance (factory function).

    This is the preferred way to get a console instance for dependency injection.

    Returns:
        Console: A new Rich Console instance configured for the current environment.
    """
    if not sys.stdout.isatty():
        return MagicMock()
    else:
        return Console()


def get_console() -> Console:
    """Get or create the global console singleton (deprecated).

    DEPRECATED: This function maintains global state and will be removed in v3.0.
    Use create_console() to create a new instance or accept console as a parameter.

    Returns:
        Console: The global console singleton instance.
    """
    global _console
    if _console is None:
        warnings.warn(
            "get_console() is deprecated and will be removed in v3.0. "
            "Use create_console() or pass console as a parameter.",
            DeprecationWarning,
            stacklevel=2
        )
        _console = create_console()
    return _console


def captured_console():
    """Return a console that captures output to a string."""
    return Console(file=StringIO())


def clear_screen():
    """Clear the console screen."""
    get_console().clear()


def get_console_height() -> int:
    """Get the height of the console."""
    return get_console().height


def get_console_width() -> int:
    """Get the width of the console."""
    return get_console().width


def print_error(text: str, **kwargs):
    """Print an error message."""
    get_console().print(f"[red]Error: {text}[/red]", **kwargs)


def print_info(text: str, **kwargs):
    """Print an info message."""
    get_console().print(f"[blue]Info: {text}[/blue]", **kwargs)


def print_rule(**kwargs):
    """Print a horizontal rule."""
    get_console().rule(**kwargs)


def print_success(text: str, **kwargs):
    """Print a success message."""
    get_console().print(f"[green]Success: {text}[/green]", **kwargs)


def print_warning(text: str, **kwargs):
    """Print a warning message."""
    get_console().print(f"[yellow]Warning: {text}[/yellow]", **kwargs)


def print_with_style(style: str, text: str, **kwargs):
    """Print text with a specific style."""
    get_console().print(f"[{style}]{text}[/{style}]", **kwargs)


def reset_console():
    """Reset the console."""
    global _console
    _console = None


def status_context(status: str, **kwargs):
    """Return a status context manager."""
    return get_console().status(status, **kwargs)


def styled_console():
    """Return the styled console."""
    return get_console()
