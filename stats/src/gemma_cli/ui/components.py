"""Reusable Rich UI components.

This module provides factory functions for creating common UI components
with consistent styling and behavior.
"""

from typing import Optional, List, Dict, Any, Sequence
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from rich.tree import Tree
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.text import Text
from rich.align import Align
from rich.columns import Columns

from .console import get_console


def create_panel(
    content: Any,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style: str = "panel.border",
    border_style: str = "panel.border",
    title_align: str = "left",
    width: Optional[int] = None,
    expand: bool = True,
) -> Panel:
    """Create a Rich panel with consistent styling.

    Args:
        content: Content to display in panel (string, Text, or renderable)
        title: Optional panel title
        subtitle: Optional panel subtitle
        style: Panel content style
        border_style: Border style
        title_align: Title alignment ('left', 'center', 'right')
        width: Panel width (None = auto)
        expand: Whether panel should expand to fill width

    Returns:
        Configured Rich Panel
    """
    return Panel(
        content,
        title=title,
        subtitle=subtitle,
        style=style,
        border_style=border_style,
        title_align=title_align,
        width=width,
        expand=expand,
    )


def create_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    title: Optional[str] = None,
    caption: Optional[str] = None,
    show_header: bool = True,
    show_lines: bool = False,
    highlight: bool = True,
    header_style: str = "table.header",
    border_style: str = "table.border",
    row_styles: Optional[List[str]] = None,
) -> Table:
    """Create a Rich table with consistent styling.

    Args:
        headers: Column header names
        rows: Table rows (list of lists)
        title: Optional table title
        caption: Optional table caption
        show_header: Whether to show header row
        show_lines: Whether to show lines between rows
        highlight: Enable cell highlighting
        header_style: Header row style
        border_style: Border style
        row_styles: Optional list of styles to alternate for rows

    Returns:
        Configured Rich Table
    """
    table = Table(
        title=title,
        caption=caption,
        show_header=show_header,
        show_lines=show_lines,
        highlight=highlight,
        header_style=header_style,
        border_style=border_style,
        row_styles=row_styles or ["", "dim"],
    )

    # Add columns
    for header in headers:
        table.add_column(header)

    # Add rows
    for row in rows:
        table.add_row(*[str(cell) for cell in row])

    return table


def create_progress_bar(
    show_spinner: bool = True,
    show_percentage: bool = True,
    show_time_remaining: bool = True,
    show_elapsed: bool = False,
    console_override: Optional[Any] = None,
) -> Progress:
    """Create a Rich progress bar with standard configuration.

    Args:
        show_spinner: Show animated spinner
        show_percentage: Show percentage complete
        show_time_remaining: Show estimated time remaining
        show_elapsed: Show elapsed time
        console_override: Optional console instance to use

    Returns:
        Configured Rich Progress instance
    """
    columns = []

    if show_spinner:
        columns.append(SpinnerColumn())

    columns.append(TextColumn("[progress.description]{task.description}"))
    columns.append(BarColumn(bar_width=40))

    if show_percentage:
        columns.append(TaskProgressColumn())

    if show_time_remaining:
        columns.append(TimeRemainingColumn())

    if show_elapsed:
        columns.append(TimeElapsedColumn())

    return Progress(
        *columns,
        console=console_override or get_console(),
    )


def create_tree_view(
    label: str,
    data: Dict[str, Any],
    guide_style: str = "dim",
) -> Tree:
    """Create a Rich tree view from nested dictionary.

    Args:
        label: Root tree label
        data: Nested dictionary to display
        guide_style: Style for tree guide lines

    Returns:
        Configured Rich Tree
    """
    tree = Tree(label, guide_style=guide_style)
    _add_dict_to_tree(tree, data)
    return tree


def _add_dict_to_tree(tree: Tree, data: Dict[str, Any]) -> None:
    """Recursively add dictionary items to tree.

    Args:
        tree: Tree node to add to
        data: Dictionary data to add
    """
    for key, value in data.items():
        if isinstance(value, dict):
            branch = tree.add(f"[label]{key}[/label]")
            _add_dict_to_tree(branch, value)
        elif isinstance(value, list):
            branch = tree.add(f"[label]{key}[/label] ([value]{len(value)} items[/value])")
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    sub_branch = branch.add(f"[dim]Item {i}[/dim]")
                    _add_dict_to_tree(sub_branch, item)
                else:
                    branch.add(f"[value]{item}[/value]")
        else:
            tree.add(f"[label]{key}[/label]: [value]{value}[/value]")


def create_syntax_highlighted_code(
    code: str,
    language: str = "python",
    theme: str = "monokai",
    line_numbers: bool = True,
    word_wrap: bool = False,
    highlight_lines: Optional[set[int]] = None,
) -> Syntax:
    """Create syntax-highlighted code display.

    Args:
        code: Source code to highlight
        language: Programming language (e.g., 'python', 'javascript')
        theme: Syntax highlighting theme
        line_numbers: Show line numbers
        word_wrap: Enable word wrapping
        highlight_lines: Set of line numbers to highlight

    Returns:
        Rich Syntax object
    """
    return Syntax(
        code,
        language,
        theme=theme,
        line_numbers=line_numbers,
        word_wrap=word_wrap,
        highlight_lines=highlight_lines or set(),
    )


def create_markdown_panel(
    markdown_text: str,
    title: Optional[str] = None,
    code_theme: str = "monokai",
) -> Panel:
    """Create a panel with rendered Markdown content.

    Args:
        markdown_text: Markdown-formatted text
        title: Optional panel title
        code_theme: Theme for code blocks

    Returns:
        Panel containing rendered Markdown
    """
    markdown = Markdown(markdown_text, code_theme=code_theme)
    return create_panel(markdown, title=title)


def create_key_value_display(
    data: Dict[str, Any],
    title: Optional[str] = None,
    key_style: str = "label",
    value_style: str = "value",
) -> Panel:
    """Create a key-value pair display panel.

    Args:
        data: Dictionary of key-value pairs
        title: Optional panel title
        key_style: Style for keys
        value_style: Style for values

    Returns:
        Panel with formatted key-value pairs
    """
    lines = []
    max_key_len = max(len(str(k)) for k in data.keys()) if data else 0

    for key, value in data.items():
        padded_key = str(key).ljust(max_key_len)
        lines.append(f"[{key_style}]{padded_key}[/{key_style}]: [{value_style}]{value}[/{value_style}]")

    content = "\n".join(lines)
    return create_panel(content, title=title)


def create_centered_text(
    text: str,
    style: Optional[str] = None,
    vertical: Optional[str] = None,
) -> Align:
    """Create centered text.

    Args:
        text: Text to center
        style: Optional text style
        vertical: Vertical alignment ('top', 'middle', 'bottom')

    Returns:
        Aligned text renderable
    """
    text_obj = Text(text, style=style) if style else Text(text)
    return Align.center(text_obj, vertical=vertical)


def create_columns(
    renderables: Sequence[Any],
    equal: bool = True,
    expand: bool = False,
    padding: int = 1,
) -> Columns:
    """Create columns layout for side-by-side content.

    Args:
        renderables: List of renderables to display in columns
        equal: Make all columns equal width
        expand: Expand columns to fill available space
        padding: Padding between columns

    Returns:
        Columns layout
    """
    return Columns(
        renderables,
        equal=equal,
        expand=expand,
        padding=padding,
    )


def create_divider(
    text: Optional[str] = None,
    style: str = "dim",
    align: str = "center",
) -> Any:
    """Create a divider line with optional text.

    Args:
        text: Optional text to display in divider
        style: Divider style
        align: Text alignment ('left', 'center', 'right')

    Returns:
        Renderable divider
    """
    console = get_console()
    if text:
        return Align(Text(f" {text} ", style=style), align=align)
    else:
        return Text("─" * console.width, style=style)
