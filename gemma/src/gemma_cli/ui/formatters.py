"""Message formatting utilities for gemma-cli.

This module provides formatters for different types of messages and data
structures, ensuring consistent presentation across the UI.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from gemma_cli.rag.memory import MemoryEntry
from gemma_cli.ui.components import create_table
from gemma_cli.ui.theme import MESSAGE_STYLES, get_color_for_memory_tier


def format_user_message(
    content: str, timestamp: Optional[datetime] = None, username: str = "You"
) -> Panel:
    """Format user message with timestamp.

    Args:
        content: Message content
        timestamp: Optional message timestamp
        username: Username to display (default: "You")

    Returns:
        Panel: Formatted user message panel
    """
    text = Text()
    text.append(f"{username}: ", style=MESSAGE_STYLES["user"])
    text.append(content)

    subtitle = ""
    if timestamp:
        subtitle = timestamp.strftime("%H:%M:%S")

    return Panel(
        text,
        border_style="cyan",
        subtitle=subtitle,
        padding=(0, 2),
        expand=False,
    )


def format_assistant_message(
    content: str, metadata: Optional[Dict[str, Any]] = None, model_name: str = "Gemma"
) -> Panel:
    """Format assistant message with optional metadata.

    Args:
        content: Message content (supports markdown)
        metadata: Optional metadata (tokens, time_ms, etc.)
        model_name: Model name to display

    Returns:
        Panel: Formatted assistant message panel
    """
    # Use Markdown for rich formatting
    md = Markdown(content)

    # Build subtitle from metadata
    subtitle_parts = []
    if metadata:
        if "tokens" in metadata:
            subtitle_parts.append(f"Tokens: {metadata['tokens']}")
        if "time_ms" in metadata:
            subtitle_parts.append(f"Time: {metadata['time_ms']:.0f}ms")
        if "model" in metadata:
            model_name = metadata["model"]

    subtitle = " | ".join(subtitle_parts) if subtitle_parts else ""

    return Panel(
        md,
        title=f"[bold green]{model_name}[/bold green]",
        border_style="green",
        subtitle=subtitle,
        padding=(1, 2),
        expand=False,
    )


def format_system_message(
    content: str, message_type: str = "info", title: Optional[str] = None
) -> Panel:
    """Format system message with type indicator.

    Args:
        content: Message content
        message_type: Type of message (info, warning, error, success)
        title: Optional custom title

    Returns:
        Panel: Formatted system message panel
    """
    # Icon mapping
    icons = {
        "info": "ℹ",
        "warning": "⚠",
        "error": "✗",
        "success": "✓",
    }

    # Style mapping
    styles = {
        "info": "blue",
        "warning": "yellow",
        "error": "red",
        "success": "green",
    }

    icon = icons.get(message_type, "•")
    style = styles.get(message_type, "white")

    if not title:
        title = message_type.capitalize()

    text = Text()
    text.append(f"{icon} ", style=f"bold {style}")
    text.append(content, style=style)

    return Panel(
        text,
        title=f"[bold {style}]{title}[/bold {style}]",
        border_style=style,
        padding=(0, 2),
        expand=False,
    )


def format_error_message(error: str, suggestion: Optional[str] = None) -> Panel:
    """Format error message with optional suggestion.

    Args:
        error: Error message
        suggestion: Optional suggestion for fixing the error

    Returns:
        Panel: Formatted error panel
    """
    text = Text()
    text.append("✗ ", style="bold red")
    text.append(error, style="red")

    if suggestion:
        text.append("\n\n")
        text.append("💡 Suggestion: ", style="bold yellow")
        text.append(suggestion, style="yellow")

    return Panel(
        text,
        title="[bold red]Error[/bold red]",
        border_style="red",
        padding=(1, 2),
        expand=False,
    )


def format_success_message(message: str) -> Panel:
    """Format success message.

    Args:
        message: Success message text

    Returns:
        Panel: Formatted success panel
    """
    text = Text()
    text.append("✓ ", style="bold green")
    text.append(message, style="green")

    return Panel(
        text,
        title="[bold green]Success[/bold green]",
        border_style="green",
        padding=(1, 2),
        expand=False,
    )


def format_warning_message(message: str, details: Optional[str] = None) -> Panel:
    """Format warning message with optional details.

    Args:
        message: Warning message text
        details: Optional additional details

    Returns:
        Panel: Formatted warning panel
    """
    text = Text()
    text.append("⚠ ", style="bold yellow")
    text.append(message, style="yellow")

    if details:
        text.append("\n\n")
        text.append(details, style="dim yellow")

    return Panel(
        text,
        title="[bold yellow]Warning[/bold yellow]",
        border_style="yellow",
        padding=(1, 2),
        expand=False,
    )


def format_memory_entry(entry: MemoryEntry) -> Table:
    """Format memory entry for display.

    Args:
        entry: MemoryEntry instance to format

    Returns:
        Table: Formatted memory entry table
    """
    color = get_color_for_memory_tier(entry.memory_type)

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()

    # Basic info
    table.add_row(
        "ID:", f"[dim]{entry.id[:8]}...[/dim]"
    )
    table.add_row(
        "Type:", f"[{color}]{entry.memory_type.replace('_', ' ').title()}[/{color}]"
    )
    table.add_row("Content:", entry.content)
    table.add_row("Importance:", f"[yellow]{entry.importance:.2f}[/yellow]")
    table.add_row("Access Count:", f"[cyan]{entry.access_count}[/cyan]")

    # Timestamps
    created = entry.created_at.strftime("%Y-%m-%d %H:%M:%S")
    accessed = entry.last_accessed.strftime("%Y-%m-%d %H:%M:%S")
    table.add_row("Created:", f"[dim]{created}[/dim]")
    table.add_row("Last Accessed:", f"[dim]{accessed}[/dim]")

    # Tags if present
    if entry.tags:
        tags_str = ", ".join(f"[cyan]#{tag}[/cyan]" for tag in entry.tags)
        table.add_row("Tags:", tags_str)

    # Metadata if present
    if entry.metadata:
        metadata_str = ", ".join(
            f"[magenta]{k}[/magenta]: {v}" for k, v in entry.metadata.items()
        )
        table.add_row("Metadata:", metadata_str)

    return Panel(
        table,
        title="[bold]Memory Entry[/bold]",
        border_style=color,
        padding=(1, 2),
    )


def format_conversation_history(
    messages: List[Dict[str, str]], max_messages: int = 10
) -> Table:
    """Format conversation history as table.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        max_messages: Maximum number of messages to display

    Returns:
        Table: Formatted conversation history table
    """
    table = create_table(
        headers=["Role", "Content", "Time"],
        rows=[],
        title="Conversation History",
        border_style="blue",
    )

    # Show only the most recent messages
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages

    for msg in recent_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")

        # Truncate long messages
        if len(content) > 100:
            content = content[:97] + "..."

        # Color code by role
        role_styles = {
            "user": "cyan",
            "assistant": "green",
            "system": "yellow",
        }
        role_style = role_styles.get(role, "white")

        table.add_row(
            f"[{role_style}]{role.capitalize()}[/{role_style}]",
            content,
            f"[dim]{timestamp}[/dim]",
        )

    return table


def format_model_info(info: Dict[str, Any]) -> Panel:
    """Format model information panel.

    Args:
        info: Dictionary of model information

    Returns:
        Panel: Formatted model info panel
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()

    for key, value in info.items():
        # Format key nicely
        display_key = key.replace("_", " ").title()
        table.add_row(f"{display_key}:", str(value))

    return Panel(
        table,
        title="[bold cyan]Model Information[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )


def format_statistics(stats: Dict[str, Any]) -> Table:
    """Format statistics as a formatted table.

    Args:
        stats: Dictionary of statistics

    Returns:
        Table: Formatted statistics table
    """
    table = create_table(
        headers=["Metric", "Value"],
        rows=[],
        title="Statistics",
        header_style="bold magenta",
        border_style="magenta",
    )

    for key, value in stats.items():
        # Format key
        display_key = key.replace("_", " ").title()

        # Format value based on type
        if isinstance(value, float):
            display_value = f"{value:.2f}"
        elif isinstance(value, int):
            display_value = f"{value:,}"
        else:
            display_value = str(value)

        table.add_row(display_key, f"[green]{display_value}[/green]")

    return table


def format_token_usage(
    prompt_tokens: int, completion_tokens: int, total_tokens: int
) -> Text:
    """Format token usage information.

    Args:
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        total_tokens: Total number of tokens

    Returns:
        Text: Formatted token usage text
    """
    text = Text()
    text.append("Tokens: ", style="dim")
    text.append(f"{prompt_tokens}", style="cyan")
    text.append(" prompt + ", style="dim")
    text.append(f"{completion_tokens}", style="green")
    text.append(" completion = ", style="dim")
    text.append(f"{total_tokens}", style="bold yellow")
    text.append(" total", style="dim")

    return text


def format_timing_info(elapsed_ms: float, tokens_per_second: Optional[float] = None) -> Text:
    """Format timing information.

    Args:
        elapsed_ms: Elapsed time in milliseconds
        tokens_per_second: Optional tokens per second metric

    Returns:
        Text: Formatted timing text
    """
    text = Text()
    text.append("Time: ", style="dim")
    text.append(f"{elapsed_ms:.0f}ms", style="magenta")

    if tokens_per_second:
        text.append(" (", style="dim")
        text.append(f"{tokens_per_second:.1f} tok/s", style="cyan")
        text.append(")", style="dim")

    return text


def format_progress_message(current: int, total: int, item_name: str = "items") -> str:
    """Format progress message for status displays.

    Args:
        current: Current progress value
        total: Total value
        item_name: Name of items being processed

    Returns:
        str: Formatted progress message
    """
    percentage = (current / total * 100) if total > 0 else 0
    return f"Processing {item_name}: {current}/{total} ({percentage:.1f}%)"


def format_memory_stats(stats: Dict[str, int]) -> Table:
    """Format memory tier statistics.

    Args:
        stats: Dictionary mapping tier names to entry counts

    Returns:
        Table: Formatted memory statistics table
    """
    table = create_table(
        headers=["Memory Tier", "Entries", "Capacity", "Usage %"],
        rows=[],
        title="Memory Statistics",
        header_style="bold cyan",
        border_style="cyan",
    )

    # Tier capacities (should match MemoryDashboard)
    capacities = {
        "working": 15,
        "short_term": 100,
        "long_term": 10000,
        "episodic": 5000,
        "semantic": 50000,
    }

    for tier, count in stats.items():
        capacity = capacities.get(tier, 0)
        percentage = (count / capacity * 100) if capacity > 0 else 0

        # Color code based on usage
        if percentage < 50:
            color = "green"
        elif percentage < 80:
            color = "yellow"
        else:
            color = "red"

        tier_color = get_color_for_memory_tier(tier)

        table.add_row(
            f"[{tier_color}]{tier.replace('_', ' ').title()}[/{tier_color}]",
            f"{count:,}",
            f"{capacity:,}",
            f"[{color}]{percentage:.1f}%[/{color}]",
        )

    return table
