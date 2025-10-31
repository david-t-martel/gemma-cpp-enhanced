"""Reusable Rich UI components for gemma-cli.

This module provides a collection of pre-configured Rich components
with consistent styling and behavior across the application.
"""

from typing import Any, Dict, List, Optional

from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from gemma_cli.ui.theme import COLORS


def create_panel(
    content: Any,
    title: str = "",
    border_style: str = "blue",
    padding: tuple = (1, 2),
    expand: bool = False,
    subtitle: str = "",
) -> Panel:
    """Create a styled panel with content.

    Args:
        content: Content to display in the panel (can be any Rich renderable)
        title: Panel title
        border_style: Rich style for the border
        padding: Tuple of (vertical, horizontal) padding
        expand: Whether to expand panel to fill width
        subtitle: Panel subtitle (displayed at bottom)

    Returns:
        Panel: Configured Rich Panel instance
    """
    return Panel(
        content,
        title=title,
        border_style=border_style,
        padding=padding,
        expand=expand,
        subtitle=subtitle,
    )


def create_table(
    headers: List[str],
    rows: List[List[str]],
    title: Optional[str] = None,
    show_header: bool = True,
    header_style: str = "bold magenta",
    border_style: str = "bright_black",
    expand: bool = False,
    caption: Optional[str] = None,
) -> Table:
    """Create a formatted table.

    Args:
        headers: List of column header strings
        rows: List of row data (each row is a list of strings)
        title: Optional table title
        show_header: Whether to show header row
        header_style: Rich style for header text
        border_style: Rich style for table borders
        expand: Whether to expand table to fill width
        caption: Optional table caption (displayed at bottom)

    Returns:
        Table: Configured Rich Table instance
    """
    table = Table(
        title=title,
        show_header=show_header,
        header_style=header_style,
        border_style=border_style,
        expand=expand,
        caption=caption,
        caption_style="dim italic",
    )

    # Add columns
    for header in headers:
        table.add_column(header)

    # Add rows
    for row in rows:
        table.add_row(*row)

    return table


def create_grid_table(
    padding: tuple = (0, 2), expand: bool = False
) -> Table:
    """Create a borderless grid table for layouts.

    Grid tables are useful for creating layouts without visible borders.

    Args:
        padding: Tuple of (vertical, horizontal) padding between cells
        expand: Whether to expand table to fill width

    Returns:
        Table: Configured Rich Table with no borders
    """
    return Table.grid(padding=padding, expand=expand)


def create_progress() -> Progress:
    """Create styled progress bar with spinner and ETA.

    Returns:
        Progress: Configured Rich Progress instance with multiple columns
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style=COLORS["primary"], finished_style=COLORS["success"]),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        expand=True,
    )


def create_simple_progress() -> Progress:
    """Create simple progress bar without time estimates.

    Returns:
        Progress: Configured Rich Progress instance with basic columns
    """
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style=COLORS["primary"], finished_style=COLORS["success"]),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        expand=True,
    )


def create_download_progress() -> Progress:
    """Create progress bar optimized for downloads with time tracking.

    Returns:
        Progress: Configured Rich Progress instance for downloads
    """
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TextColumn("•"),
        TextColumn("{task.completed}/{task.total} MB"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        expand=True,
    )


def create_tree(label: str, data: Dict[str, Any], guide_style: str = "dim") -> Tree:
    """Create a tree view for hierarchical data.

    Args:
        label: Root label for the tree
        data: Dictionary to display as tree structure
        guide_style: Rich style for tree guide lines

    Returns:
        Tree: Configured Rich Tree instance
    """
    tree = Tree(label, guide_style=guide_style)

    def add_branch(parent: Tree, items: Dict[str, Any]) -> None:
        """Recursively add branches to tree."""
        for key, value in items.items():
            if isinstance(value, dict):
                branch = parent.add(f"[bold cyan]{key}[/bold cyan]")
                add_branch(branch, value)
            elif isinstance(value, list):
                branch = parent.add(f"[bold cyan]{key}[/bold cyan]")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        sub_branch = branch.add(f"[dim]Item {i}[/dim]")
                        add_branch(sub_branch, item)
                    else:
                        branch.add(f"[green]{item}[/green]")
            else:
                parent.add(f"[cyan]{key}[/cyan]: [green]{value}[/green]")

    add_branch(tree, data)
    return tree


def create_syntax(
    code: str,
    language: str = "python",
    theme: str = "monokai",
    line_numbers: bool = True,
    word_wrap: bool = False,
    line_range: Optional[tuple] = None,
) -> Syntax:
    """Create syntax-highlighted code block.

    Args:
        code: Source code to highlight
        language: Programming language for syntax highlighting
        theme: Pygments theme name
        line_numbers: Whether to show line numbers
        word_wrap: Whether to wrap long lines
        line_range: Optional tuple of (start_line, end_line) to display

    Returns:
        Syntax: Configured Rich Syntax instance
    """
    return Syntax(
        code,
        language,
        theme=theme,
        line_numbers=line_numbers,
        word_wrap=word_wrap,
        line_range=line_range,
        code_width=None,
    )


def create_markdown(text: str, code_theme: str = "monokai") -> Markdown:
    """Create markdown-rendered content.

    Args:
        text: Markdown text to render
        code_theme: Pygments theme for code blocks

    Returns:
        Markdown: Configured Rich Markdown instance
    """
    return Markdown(text, code_theme=code_theme)


def create_key_value_table(data: Dict[str, Any], title: Optional[str] = None) -> Table:
    """Create a two-column table for key-value pairs.

    Args:
        data: Dictionary of key-value pairs
        title: Optional table title

    Returns:
        Table: Configured Rich Table with key-value pairs
    """
    table = Table(
        title=title,
        show_header=False,
        border_style="bright_black",
        padding=(0, 1),
    )

    table.add_column("Key", style="cyan bold", no_wrap=True)
    table.add_column("Value", style="green")

    for key, value in data.items():
        table.add_row(str(key), str(value))

    return table


def create_status_table(statuses: Dict[str, str], title: str = "Status") -> Table:
    """Create a status table with colored indicators.

    Args:
        statuses: Dictionary mapping labels to status values
        title: Table title

    Returns:
        Table: Configured Rich Table with status indicators
    """
    table = create_table(
        headers=["Component", "Status"],
        rows=[],
        title=title,
        border_style="blue",
    )

    # Map status values to colors
    status_colors = {
        "ok": "green",
        "online": "green",
        "ready": "green",
        "active": "green",
        "warning": "yellow",
        "pending": "yellow",
        "error": "red",
        "offline": "red",
        "failed": "red",
        "disabled": "dim",
    }

    for label, status in statuses.items():
        status_lower = status.lower()
        color = status_colors.get(status_lower, "white")
        table.add_row(label, f"[{color}]{status}[/{color}]")

    return table


def create_info_panel(
    title: str, items: Dict[str, str], border_style: str = "blue"
) -> Panel:
    """Create an info panel with key-value pairs.

    Args:
        title: Panel title
        items: Dictionary of information items
        border_style: Rich style for the border

    Returns:
        Panel: Configured Rich Panel with info table inside
    """
    table = create_key_value_table(items)
    return create_panel(table, title=title, border_style=border_style)


def create_error_panel(error: str, details: Optional[str] = None) -> Panel:
    """Create an error panel with optional details.

    Args:
        error: Error message
        details: Optional detailed error information

    Returns:
        Panel: Configured Rich Panel for error display
    """
    content = f"[bold red]{error}[/bold red]"
    if details:
        content += f"\n\n[dim]{details}[/dim]"

    return create_panel(
        content, title="[bold red]Error[/bold red]", border_style="red", padding=(1, 2)
    )


def create_success_panel(message: str, details: Optional[str] = None) -> Panel:
    """Create a success panel with optional details.

    Args:
        message: Success message
        details: Optional additional information

    Returns:
        Panel: Configured Rich Panel for success display
    """
    content = f"[bold green]✓[/bold green] {message}"
    if details:
        content += f"\n\n[dim]{details}[/dim]"

    return create_panel(
        content,
        title="[bold green]Success[/bold green]",
        border_style="green",
        padding=(1, 2),
    )


def create_warning_panel(message: str, details: Optional[str] = None) -> Panel:
    """Create a warning panel with optional details.

    Args:
        message: Warning message
        details: Optional additional information

    Returns:
        Panel: Configured Rich Panel for warning display
    """
    content = f"[bold yellow]⚠[/bold yellow] {message}"
    if details:
        content += f"\n\n[dim]{details}[/dim]"

    return create_panel(
        content,
        title="[bold yellow]Warning[/bold yellow]",
        border_style="yellow",
        padding=(1, 2),
    )
