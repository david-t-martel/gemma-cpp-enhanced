"""Global Rich console instance with custom configuration.

This module provides a singleton console instance used throughout the CLI
for consistent output rendering and styling.
"""

from contextlib import contextmanager
from typing import Optional, Iterator
from rich.console import Console
from rich.theme import Theme

from .theme import get_theme, ThemeVariant


# Global console instance
_console: Optional[Console] = None
_current_theme: ThemeVariant = "dark"


def get_console(
    force_terminal: Optional[bool] = None,
    force_jupyter: Optional[bool] = None,
    force_interactive: Optional[bool] = None,
    width: Optional[int] = None,
    theme: Optional[ThemeVariant] = None,
) -> Console:
    """Get or create the global console instance.

    This function returns a singleton console instance. On first call,
    it creates and configures the console. Subsequent calls return the
    same instance unless parameters are provided.

    Args:
        force_terminal: Force terminal mode even if not detected
        force_jupyter: Force Jupyter rendering mode
        force_interactive: Force interactive mode (disables recording)
        width: Console width in characters (None = auto-detect)
        theme: Theme variant to use ('dark' or 'light')

    Returns:
        Configured Rich Console instance
    """
    global _console, _current_theme

    # If parameters provided, recreate console
    if any(
        param is not None
        for param in [force_terminal, force_jupyter, force_interactive, width, theme]
    ):
        _current_theme = theme or _current_theme
        _console = _create_console(
            force_terminal=force_terminal,
            force_jupyter=force_jupyter,
            force_interactive=force_interactive,
            width=width,
            theme=_current_theme,
        )
    elif _console is None:
        # First time initialization
        _console = _create_console(theme=_current_theme)

    return _console


def _create_console(
    force_terminal: Optional[bool] = None,
    force_jupyter: Optional[bool] = None,
    force_interactive: Optional[bool] = None,
    width: Optional[int] = None,
    theme: ThemeVariant = "dark",
) -> Console:
    """Create a new console instance with specified configuration.

    Args:
        force_terminal: Force terminal mode
        force_jupyter: Force Jupyter rendering
        force_interactive: Force interactive mode
        width: Console width in characters
        theme: Theme variant to use

    Returns:
        Configured Rich Console instance
    """
    theme_obj: Theme = get_theme(theme)

    return Console(
        theme=theme_obj,
        force_terminal=force_terminal,
        force_jupyter=force_jupyter,
        force_interactive=force_interactive,
        width=width,
        color_system="truecolor",  # Enable 24-bit color
        legacy_windows=False,  # Use modern Windows terminal
        safe_box=True,  # Use safe box characters for compatibility
        highlight=True,  # Enable automatic syntax highlighting
        markup=True,  # Enable Rich markup in strings
        emoji=True,  # Enable emoji support
        soft_wrap=True,  # Enable soft wrapping
        tab_size=4,  # Tab width
    )


def reset_console() -> None:
    """Reset the global console instance.

    Forces creation of a new console on next get_console() call.
    Useful for testing or when console configuration needs to change.
    """
    global _console
    _console = None


def set_theme(theme: ThemeVariant) -> None:
    """Change the global console theme.

    Args:
        theme: Theme variant to use ('dark' or 'light')
    """
    global _current_theme
    _current_theme = theme
    reset_console()


def get_current_theme() -> ThemeVariant:
    """Get the current theme variant.

    Returns:
        Current theme variant ('dark' or 'light')
    """
    return _current_theme


@contextmanager
def scoped_style(style: str) -> Iterator[Console]:
    """Context manager for scoped styling.

    Temporarily applies a style to all console output within the context.

    Args:
        style: Rich style string or semantic style name

    Yields:
        Console instance with style applied

    Example:
        >>> with scoped_style("bold red"):
        ...     console.print("This is bold and red")
        >>> console.print("This is normal")
    """
    console = get_console()
    console.push_style(style)
    try:
        yield console
    finally:
        console.pop_style()


@contextmanager
def capture_output() -> Iterator[tuple[Console, callable]]:
    """Context manager for capturing console output.

    Useful for testing or logging console output.

    Yields:
        Tuple of (console, get_output_fn) where get_output_fn returns captured text

    Example:
        >>> with capture_output() as (console, get_output):
        ...     console.print("Hello, World!")
        ...     output = get_output()
        >>> assert "Hello" in output
    """
    console = Console(
        theme=get_theme(_current_theme),
        record=True,
        force_terminal=True,
    )

    def get_output() -> str:
        return console.export_text()

    yield console, get_output


@contextmanager
def pager_mode() -> Iterator[Console]:
    """Context manager for paged output.

    Automatically paginates long output using the system pager.

    Yields:
        Console instance configured for paging

    Example:
        >>> with pager_mode() as console:
        ...     console.print("\\n".join(f"Line {i}" for i in range(1000)))
    """
    console = get_console()
    with console.pager():
        yield console


@contextmanager
def status_spinner(message: str, spinner: str = "dots") -> Iterator[Console]:
    """Context manager for status spinner.

    Shows an animated spinner with a status message during long operations.

    Args:
        message: Status message to display
        spinner: Spinner style (see Rich documentation for options)

    Yields:
        Console instance

    Example:
        >>> with status_spinner("Loading model..."):
        ...     load_model()
    """
    console = get_console()
    with console.status(message, spinner=spinner):
        yield console


def print_header(title: str, subtitle: Optional[str] = None) -> None:
    """Print a formatted header.

    Args:
        title: Main header title
        subtitle: Optional subtitle
    """
    console = get_console()
    console.rule(f"[title]{title}[/title]", style="panel.border")
    if subtitle:
        console.print(f"[subtitle]{subtitle}[/subtitle]", justify="center")
        console.print()


def print_separator(char: str = "─") -> None:
    """Print a visual separator line.

    Args:
        char: Character to use for separator
    """
    console = get_console()
    console.print(char * console.width, style="dim")


def clear_screen() -> None:
    """Clear the console screen."""
    console = get_console()
    console.clear()
