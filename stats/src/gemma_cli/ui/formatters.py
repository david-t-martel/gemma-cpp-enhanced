"""Message formatting utilities.

This module provides consistent formatting for different types of messages
and data structures in the CLI.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from rich.text import Text
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.style import Style

from .components import create_panel, create_table
from .theme import get_semantic_style


def _color_to_hex(style: Style) -> str:
    """Convert Rich Style color to hex string.

    Args:
        style: Rich Style object

    Returns:
        Hex color string (e.g., '#00d9ff')
    """
    if style.color and hasattr(style.color, 'triplet'):
        return '#%02x%02x%02x' % style.color.triplet
    return 'white'


def format_user_message(
    content: str,
    timestamp: Optional[datetime] = None,
    include_panel: bool = True,
) -> Any:
    """Format a user message with consistent styling.

    Args:
        content: Message content
        timestamp: Optional timestamp
        include_panel: Whether to wrap in a panel

    Returns:
        Formatted renderable (Panel or Text)
    """
    style = get_semantic_style("user")

    color_hex = _color_to_hex(style)

    if timestamp:
        time_str = timestamp.strftime("%H:%M:%S")
        title = f"User [{color_hex}]{time_str}[/{color_hex}]"
    else:
        title = "User"

    if include_panel:
        return create_panel(
            content,
            title=title,
            border_style=color_hex,
        )
    else:
        text = Text(content, style=style)
        return text


def format_assistant_message(
    content: str,
    timestamp: Optional[datetime] = None,
    include_panel: bool = True,
    render_markdown: bool = True,
) -> Any:
    """Format an assistant message with consistent styling.

    Args:
        content: Message content
        timestamp: Optional timestamp
        include_panel: Whether to wrap in a panel
        render_markdown: Whether to render content as Markdown

    Returns:
        Formatted renderable (Panel, Markdown, or Text)
    """
    style = get_semantic_style("assistant")
    color_hex = _color_to_hex(style)

    if timestamp:
        time_str = timestamp.strftime("%H:%M:%S")
        title = f"Assistant [{color_hex}]{time_str}[/{color_hex}]"
    else:
        title = "Assistant"

    if render_markdown:
        content_renderable = Markdown(content, code_theme="monokai")
    else:
        content_renderable = Text(content, style=style)

    if include_panel:
        return create_panel(
            content_renderable,
            title=title,
            border_style=color_hex,
        )
    else:
        return content_renderable


def format_system_message(
    content: str,
    timestamp: Optional[datetime] = None,
    level: str = "info",
) -> Panel:
    """Format a system message with consistent styling.

    Args:
        content: Message content
        timestamp: Optional timestamp
        level: Message level ('info', 'warning', 'error', 'success')

    Returns:
        Formatted Panel
    """
    style = get_semantic_style(level)
    color_hex = _color_to_hex(style)

    if timestamp:
        time_str = timestamp.strftime("%H:%M:%S")
        title = f"System [{color_hex}]{time_str}[/{color_hex}]"
    else:
        title = f"System - {level.upper()}"

    text = Text(content, style=style)

    return create_panel(
        text,
        title=title,
        border_style=color_hex,
    )


def format_error_message(
    error: str,
    suggestion: Optional[str] = None,
    traceback: Optional[str] = None,
) -> Panel:
    """Format an error message with optional suggestion.

    Args:
        error: Error message
        suggestion: Optional suggestion for fixing the error
        traceback: Optional traceback information

    Returns:
        Formatted Panel
    """
    style = get_semantic_style("error")

    lines = [Text(error, style=style)]

    if suggestion:
        lines.append(Text())  # Empty line
        lines.append(Text("Suggestion:", style="bold"))
        lines.append(Text(suggestion, style="info"))

    if traceback:
        lines.append(Text())  # Empty line
        lines.append(Text("Traceback:", style="dim"))
        lines.append(Text(traceback, style="dim"))

    content = Text("\n").join(lines)

    return create_panel(
        content,
        title="Error",
        border_style=_color_to_hex(style),
    )


def format_success_message(
    message: str,
    details: Optional[str] = None,
) -> Panel:
    """Format a success message.

    Args:
        message: Success message
        details: Optional additional details

    Returns:
        Formatted Panel
    """
    style = get_semantic_style("success")

    if details:
        content = Text(f"{message}\n\n{details}", style=style)
    else:
        content = Text(message, style=style)

    return create_panel(
        content,
        title="Success",
        border_style=_color_to_hex(style),
    )


def format_memory_entry(
    entry: Dict[str, Any],
    tier: Optional[str] = None,
    show_metadata: bool = True,
) -> Panel:
    """Format a memory entry for display.

    Args:
        entry: Memory entry dictionary
        tier: Memory tier name
        show_metadata: Whether to show metadata

    Returns:
        Formatted Panel
    """
    title = f"Memory Entry"
    if tier:
        from .theme import get_memory_tier_style
        tier_style = get_memory_tier_style(tier)
        color_hex = _color_to_hex(tier_style)
        title += f" - [{color_hex}]{tier.upper()}[/{color_hex}]"

    content_parts = []

    # Main content
    if "content" in entry:
        content_parts.append(entry["content"])

    # Metadata
    if show_metadata:
        content_parts.append("")  # Empty line
        metadata_lines = []

        if "timestamp" in entry:
            ts = entry["timestamp"]
            if isinstance(ts, datetime):
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            else:
                ts_str = str(ts)
            metadata_lines.append(f"[dim]Timestamp:[/dim] {ts_str}")

        if "importance" in entry:
            metadata_lines.append(f"[dim]Importance:[/dim] {entry['importance']}")

        if "access_count" in entry:
            metadata_lines.append(f"[dim]Access Count:[/dim] {entry['access_count']}")

        if metadata_lines:
            content_parts.extend(metadata_lines)

    content = "\n".join(content_parts)

    return create_panel(content, title=title)


def format_conversation_history(
    messages: List[Dict[str, Any]],
    max_messages: Optional[int] = None,
    show_timestamps: bool = True,
) -> Table:
    """Format conversation history as a table.

    Args:
        messages: List of message dictionaries
        max_messages: Maximum number of messages to display
        show_timestamps: Whether to show timestamps

    Returns:
        Formatted Table
    """
    if max_messages:
        messages = messages[-max_messages:]

    headers = ["Role", "Content"]
    if show_timestamps:
        headers.insert(0, "Time")

    rows = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Truncate long messages
        if len(content) > 100:
            content = content[:97] + "..."

        row_data = []

        if show_timestamps:
            timestamp = msg.get("timestamp")
            if timestamp:
                if isinstance(timestamp, datetime):
                    time_str = timestamp.strftime("%H:%M:%S")
                else:
                    time_str = str(timestamp)
            else:
                time_str = "-"
            row_data.append(time_str)

        # Style role based on type
        if role == "user":
            styled_role = "[user]User[/user]"
        elif role == "assistant":
            styled_role = "[assistant]Assistant[/assistant]"
        else:
            styled_role = "[system]System[/system]"

        row_data.extend([styled_role, content])
        rows.append(row_data)

    return create_table(
        headers=headers,
        rows=rows,
        title="Conversation History",
        show_lines=True,
    )


def format_model_info(model_info: Dict[str, Any]) -> Panel:
    """Format model information for display.

    Args:
        model_info: Model information dictionary

    Returns:
        Formatted Panel
    """
    lines = []

    if "name" in model_info:
        lines.append(f"[label]Model:[/label] [value]{model_info['name']}[/value]")

    if "type" in model_info:
        lines.append(f"[label]Type:[/label] [value]{model_info['type']}[/value]")

    if "size" in model_info:
        lines.append(f"[label]Size:[/label] [value]{model_info['size']}[/value]")

    if "context_length" in model_info:
        lines.append(f"[label]Context Length:[/label] [value]{model_info['context_length']}[/value]")

    if "parameters" in model_info:
        params = model_info["parameters"]
        if isinstance(params, dict):
            lines.append("\n[label]Parameters:[/label]")
            for key, value in params.items():
                lines.append(f"  [dim]{key}:[/dim] {value}")
        else:
            lines.append(f"[label]Parameters:[/label] [value]{params}[/value]")

    content = "\n".join(lines)

    return create_panel(content, title="Model Information")


def format_statistics(stats: Dict[str, Any]) -> Table:
    """Format statistics as a table.

    Args:
        stats: Statistics dictionary

    Returns:
        Formatted Table
    """
    rows = []
    for key, value in stats.items():
        # Format key (convert underscores to spaces, title case)
        display_key = key.replace("_", " ").title()

        # Format value based on type
        if isinstance(value, float):
            display_value = f"{value:.2f}"
        elif isinstance(value, int):
            display_value = f"{value:,}"
        else:
            display_value = str(value)

        rows.append([display_key, display_value])

    return create_table(
        headers=["Metric", "Value"],
        rows=rows,
        title="Statistics",
        show_lines=False,
    )
