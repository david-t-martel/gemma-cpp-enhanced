"""Complex Rich widgets (dashboard, status bar, etc.).

This module provides complex interactive widgets for the CLI interface.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import BarColumn, Progress, TextColumn
from rich.align import Align
from rich.console import Group
from rich.live import Live
from rich.style import Style

from .console import get_console
from .components import create_panel, create_table, create_centered_text
from .theme import get_memory_tier_style, Colors


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


class MemoryDashboard:
    """Dashboard widget showing 5-tier memory usage."""

    def __init__(self, memory_stats: Optional[Dict[str, Dict[str, Any]]] = None):
        """Initialize memory dashboard.

        Args:
            memory_stats: Dictionary of memory tier statistics
                Format: {tier_name: {"used": int, "capacity": int, "items": int}}
        """
        self.memory_stats = memory_stats or {}
        self.tiers = ["working", "short", "long", "episodic", "semantic"]

    def update_stats(self, memory_stats: Dict[str, Dict[str, Any]]) -> None:
        """Update memory statistics.

        Args:
            memory_stats: New memory statistics
        """
        self.memory_stats = memory_stats

    def render(self) -> Panel:
        """Render the memory dashboard.

        Returns:
            Panel containing memory visualization
        """
        table = Table(
            show_header=True,
            header_style="bold",
            show_lines=True,
            expand=True,
        )

        table.add_column("Tier", style="bold", width=12)
        table.add_column("Usage", width=40)
        table.add_column("Items", justify="right", width=10)
        table.add_column("Utilization", justify="right", width=12)

        for tier in self.tiers:
            tier_stats = self.memory_stats.get(tier, {})
            used = tier_stats.get("used", 0)
            capacity = tier_stats.get("capacity", 100)
            items = tier_stats.get("items", 0)

            # Calculate utilization percentage
            utilization = (used / capacity * 100) if capacity > 0 else 0

            # Get tier style
            tier_style = get_memory_tier_style(tier)
            tier_color_hex = _color_to_hex(tier_style)
            tier_label = Text(tier.upper(), style=tier_style)

            # Create progress bar
            bar_width = 30
            filled = int(bar_width * utilization / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            usage_text = Text(bar, style=tier_color_hex)

            # Format utilization with color
            if utilization < 50:
                util_style = "metric_good"
            elif utilization < 80:
                util_style = "metric_warn"
            else:
                util_style = "metric_bad"

            util_text = Text(f"{utilization:.1f}%", style=util_style)
            items_text = Text(str(items))

            table.add_row(tier_label, usage_text, items_text, util_text)

        return create_panel(
            table,
            title="Memory System Dashboard",
            border_style=Colors.PRIMARY,
        )


class StatusBar:
    """Bottom status bar showing model, memory, and token information."""

    def __init__(
        self,
        model_name: str = "Unknown",
        memory_usage: int = 0,
        total_memory: int = 100,
        tokens_used: int = 0,
        max_tokens: int = 8192,
    ):
        """Initialize status bar.

        Args:
            model_name: Name of active model
            memory_usage: Current memory usage
            total_memory: Total available memory
            tokens_used: Tokens used in current context
            max_tokens: Maximum token limit
        """
        self.model_name = model_name
        self.memory_usage = memory_usage
        self.total_memory = total_memory
        self.tokens_used = tokens_used
        self.max_tokens = max_tokens

    def update(
        self,
        model_name: Optional[str] = None,
        memory_usage: Optional[int] = None,
        total_memory: Optional[int] = None,
        tokens_used: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """Update status bar values.

        Args:
            model_name: New model name
            memory_usage: New memory usage
            total_memory: New total memory
            tokens_used: New token count
            max_tokens: New max token limit
        """
        if model_name is not None:
            self.model_name = model_name
        if memory_usage is not None:
            self.memory_usage = memory_usage
        if total_memory is not None:
            self.total_memory = total_memory
        if tokens_used is not None:
            self.tokens_used = tokens_used
        if max_tokens is not None:
            self.max_tokens = max_tokens

    def render(self) -> Text:
        """Render the status bar.

        Returns:
            Formatted status bar text
        """
        # Memory percentage
        mem_pct = (self.memory_usage / self.total_memory * 100) if self.total_memory > 0 else 0

        # Token percentage
        token_pct = (self.tokens_used / self.max_tokens * 100) if self.max_tokens > 0 else 0

        # Build status text
        parts = [
            Text("Model: ", style="dim"),
            Text(self.model_name, style="value"),
            Text(" | ", style="dim"),
            Text("Memory: ", style="dim"),
            Text(f"{mem_pct:.1f}%", style=self._get_metric_style(mem_pct)),
            Text(f" ({self.memory_usage}/{self.total_memory})", style="dim"),
            Text(" | ", style="dim"),
            Text("Tokens: ", style="dim"),
            Text(f"{token_pct:.1f}%", style=self._get_metric_style(token_pct)),
            Text(f" ({self.tokens_used}/{self.max_tokens})", style="dim"),
        ]

        return Text.assemble(*parts)

    @staticmethod
    def _get_metric_style(percentage: float) -> str:
        """Get style based on percentage.

        Args:
            percentage: Percentage value

        Returns:
            Style name
        """
        if percentage < 50:
            return "metric_good"
        elif percentage < 80:
            return "metric_warn"
        else:
            return "metric_bad"


class StartupBanner:
    """Animated splash screen for CLI startup."""

    def __init__(self, app_name: str = "Gemma CLI", version: str = "1.0.0"):
        """Initialize startup banner.

        Args:
            app_name: Application name
            version: Application version
        """
        self.app_name = app_name
        self.version = version

    def render(self) -> Panel:
        """Render the startup banner.

        Returns:
            Formatted banner panel
        """
        title_art = f"""
╔═══════════════════════════════════════════╗
║                                           ║
║         {self.app_name.upper():^30}        ║
║         {f"v{self.version}":^30}        ║
║                                           ║
╚═══════════════════════════════════════════╝
        """

        subtitle = Text("High-Performance LLM CLI with Memory", style="dim italic")

        content = Group(
            Text(title_art, style=Colors.PRIMARY, justify="center"),
            Align.center(subtitle),
        )

        return create_panel(
            content,
            border_style=Colors.PRIMARY,
            expand=False,
        )


class CommandPalette:
    """Interactive command list widget."""

    def __init__(self, commands: Dict[str, str]):
        """Initialize command palette.

        Args:
            commands: Dictionary of command names to descriptions
        """
        self.commands = commands

    def render(self) -> Panel:
        """Render the command palette.

        Returns:
            Formatted command list panel
        """
        table = Table(
            show_header=True,
            header_style="table.header",
            show_lines=False,
            expand=True,
        )

        table.add_column("Command", style="command", width=20)
        table.add_column("Description", style="dim")

        for cmd, desc in sorted(self.commands.items()):
            table.add_row(f"/{cmd}", desc)

        return create_panel(
            table,
            title="Available Commands",
            border_style=Colors.INFO,
        )


class ModelSelector:
    """Interactive model picker widget."""

    def __init__(self, models: List[Dict[str, Any]]):
        """Initialize model selector.

        Args:
            models: List of model dictionaries with 'name', 'size', 'description'
        """
        self.models = models
        self.selected_index = 0

    def render(self, highlight_selected: bool = True) -> Panel:
        """Render the model selector.

        Args:
            highlight_selected: Whether to highlight the selected model

        Returns:
            Formatted model selection panel
        """
        table = Table(
            show_header=True,
            header_style="table.header",
            show_lines=True,
            expand=True,
        )

        table.add_column("", width=3)
        table.add_column("Model", style="bold", width=30)
        table.add_column("Size", justify="right", width=12)
        table.add_column("Description", style="dim")

        for i, model in enumerate(self.models):
            # Selection indicator
            indicator = "→" if i == self.selected_index and highlight_selected else " "

            # Extract model info
            name = model.get("name", "Unknown")
            size = model.get("size", "-")
            description = model.get("description", "No description")

            # Style selected row differently
            if i == self.selected_index and highlight_selected:
                name = f"[{Colors.PRIMARY}]{name}[/{Colors.PRIMARY}]"

            table.add_row(indicator, name, size, description)

        return create_panel(
            table,
            title="Select Model",
            subtitle="Use ↑/↓ to navigate, Enter to select",
            border_style=Colors.INFO,
        )

    def select_next(self) -> None:
        """Move selection to next model."""
        self.selected_index = (self.selected_index + 1) % len(self.models)

    def select_previous(self) -> None:
        """Move selection to previous model."""
        self.selected_index = (self.selected_index - 1) % len(self.models)

    def get_selected_model(self) -> Dict[str, Any]:
        """Get the currently selected model.

        Returns:
            Selected model dictionary
        """
        return self.models[self.selected_index]


class LoadingSpinner:
    """Animated loading spinner for long operations."""

    def __init__(self, message: str = "Loading..."):
        """Initialize loading spinner.

        Args:
            message: Loading message to display
        """
        self.message = message
        self.console = get_console()

    def __enter__(self) -> "LoadingSpinner":
        """Enter context manager."""
        self.status = self.console.status(self.message, spinner="dots")
        self.status.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        self.status.__exit__(exc_type, exc_val, exc_tb)

    def update(self, message: str) -> None:
        """Update loading message.

        Args:
            message: New message to display
        """
        self.message = message
        if hasattr(self, "status"):
            self.status.update(message)


class ConversationView:
    """Scrollable conversation history view."""

    def __init__(self, messages: Optional[List[Dict[str, Any]]] = None):
        """Initialize conversation view.

        Args:
            messages: List of message dictionaries
        """
        self.messages = messages or []

    def add_message(self, role: str, content: str, timestamp: Optional[datetime] = None) -> None:
        """Add a message to the conversation.

        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            timestamp: Optional timestamp
        """
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": timestamp or datetime.now(),
        })

    def render(self, max_messages: Optional[int] = None) -> Panel:
        """Render the conversation view.

        Args:
            max_messages: Maximum number of recent messages to show

        Returns:
            Formatted conversation panel
        """
        from .formatters import format_conversation_history

        if not self.messages:
            return create_panel(
                Text("No messages yet", style="dim italic"),
                title="Conversation",
            )

        table = format_conversation_history(
            self.messages,
            max_messages=max_messages,
        )

        return create_panel(
            table,
            title=f"Conversation ({len(self.messages)} messages)",
            border_style=Colors.PRIMARY,
        )
