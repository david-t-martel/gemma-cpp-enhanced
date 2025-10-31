"""Complex Rich widgets for gemma-cli.

This module provides advanced UI widgets including dashboards, status bars,
and interactive components with live updates.
"""

import time
from typing import Any, Dict, List, Optional

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from gemma_cli.rag.memory import MemoryTier
from gemma_cli.ui.console import get_console
from gemma_cli.ui.theme import COLORS, MEMORY_TIER_COLORS


class MemoryDashboard:
    """Interactive dashboard showing 5-tier memory usage.

    This widget displays real-time memory usage across all memory tiers
    with visual progress bars and capacity information.
    """

    def __init__(self, console: Optional[Console] = None):
        """Initialize memory dashboard.

        Args:
            console: Optional Rich Console instance (uses singleton if None)
        """
        self.console = console or get_console()
        self._tier_capacities = {
            MemoryTier.WORKING: 15,
            MemoryTier.SHORT_TERM: 100,
            MemoryTier.LONG_TERM: 10000,
            MemoryTier.EPISODIC: 5000,
            MemoryTier.SEMANTIC: 50000,
        }

    def render(self, stats: Dict[str, int]) -> Panel:
        """Render memory dashboard with progress bars.

        Args:
            stats: Dictionary mapping tier names to entry counts

        Returns:
            Panel: Rendered memory dashboard panel
        """
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold", width=15)
        table.add_column(width=35)
        table.add_column(justify="right", width=15)

        tiers = [
            (MemoryTier.WORKING, self._tier_capacities[MemoryTier.WORKING]),
            (MemoryTier.SHORT_TERM, self._tier_capacities[MemoryTier.SHORT_TERM]),
            (MemoryTier.LONG_TERM, self._tier_capacities[MemoryTier.LONG_TERM]),
            (MemoryTier.EPISODIC, self._tier_capacities[MemoryTier.EPISODIC]),
            (MemoryTier.SEMANTIC, self._tier_capacities[MemoryTier.SEMANTIC]),
        ]

        for tier_name, capacity in tiers:
            count = stats.get(tier_name, 0)
            percentage = (count / capacity) * 100 if capacity > 0 else 0
            color = MEMORY_TIER_COLORS[tier_name]

            # Create visual bar
            bar_length = 30
            filled = int(bar_length * percentage / 100)
            bar = f"[{color}]{'█' * filled}{' ' * (bar_length - filled)}[/{color}]"

            # Format tier name
            display_name = tier_name.replace("_", " ").title()

            # Format count
            count_str = f"{count:,}/{capacity:,}"

            table.add_row(display_name, bar, count_str)

        return Panel(
            table,
            title="[bold cyan]Memory Usage[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )

    def display(self, stats: Dict[str, int]) -> None:
        """Display the dashboard to console.

        Args:
            stats: Dictionary mapping tier names to entry counts
        """
        self.console.print(self.render(stats))


class StatusBar:
    """Persistent status bar showing system information.

    Displays model name, memory tier, token usage, and response time
    in a compact status bar format.
    """

    def __init__(self, console: Optional[Console] = None):
        """Initialize status bar.

        Args:
            console: Optional Rich Console instance (uses singleton if None)
        """
        self.console = console or get_console()
        self.model = "Unknown"
        self.memory_tier = "working"
        self.tokens_used = 0
        self.response_time = 0
        self.context_size = 0
        self.max_context = 8192

    def update(
        self,
        model: Optional[str] = None,
        memory_tier: Optional[str] = None,
        tokens_used: Optional[int] = None,
        response_time: Optional[float] = None,
        context_size: Optional[int] = None,
    ) -> None:
        """Update status bar values.

        Args:
            model: Model name
            memory_tier: Active memory tier
            tokens_used: Number of tokens used
            response_time: Response time in milliseconds
            context_size: Current context size
        """
        if model is not None:
            self.model = model
        if memory_tier is not None:
            self.memory_tier = memory_tier
        if tokens_used is not None:
            self.tokens_used = tokens_used
        if response_time is not None:
            self.response_time = response_time
        if context_size is not None:
            self.context_size = context_size

    def render(self) -> Text:
        """Render status bar.

        Returns:
            Text: Formatted status bar text
        """
        text = Text()

        # Model
        text.append("Model: ", style="dim")
        text.append(self.model, style="cyan bold")

        # Memory tier
        text.append(" | Memory: ", style="dim")
        tier_color = MEMORY_TIER_COLORS.get(self.memory_tier, "white")
        text.append(self.memory_tier.replace("_", " ").title(), style=tier_color)

        # Tokens
        text.append(" | Tokens: ", style="dim")
        text.append(f"{self.tokens_used:,}", style="green")

        # Context usage
        if self.max_context > 0:
            context_pct = (self.context_size / self.max_context) * 100
            text.append(" | Context: ", style="dim")
            context_color = "green" if context_pct < 70 else "yellow" if context_pct < 90 else "red"
            text.append(f"{self.context_size}/{self.max_context} ({context_pct:.0f}%)", style=context_color)

        # Response time
        text.append(" | Time: ", style="dim")
        text.append(f"{self.response_time:.0f}ms", style="magenta")

        return text

    def display(self) -> None:
        """Display the status bar to console."""
        self.console.print(self.render())


class StartupBanner:
    """Animated startup banner with system checks.

    Displays an animated banner during application startup with
    system initialization progress.
    """

    def __init__(self, console: Optional[Console] = None):
        """Initialize startup banner.

        Args:
            console: Optional Rich Console instance (uses singleton if None)
        """
        self.console = console or get_console()

    @staticmethod
    def _create_banner_text() -> Text:
        """Create ASCII art banner text.

        Returns:
            Text: Formatted banner text
        """
        banner = Text()
        banner.append("""
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   ╔═══╗ ╔═══╗ ╔═══╗ ╔═══╗ ╔═══╗   ╔═══╗ ╔╗    ╔══╗  ║
║   ║       ║   ║     ║ ║  ║ ║ ║  ║   ║     ║║    ║  ║  ║
║   ║ ╔═╗ ╠═══╣ ║ ╔╗ ║ ║  ║ ╠═══╣   ║     ║║    ║  ║  ║
║   ║   ║ ║   ║ ║   ║ ║ ║  ║ ║   ║   ║     ║║    ║  ║  ║
║   ╚═══╝ ╚═══╝ ╚═══╝ ╚═══╝ ╚═══╝   ╚═══╝ ╚═══╝ ╚══╝  ║
║                                                       ║
║              High-Performance AI Inference            ║
║                   with 5-Tier RAG                     ║
╚═══════════════════════════════════════════════════════╝
""", style=COLORS["primary"])
        return banner

    def show(self, checks: Optional[List[Dict[str, Any]]] = None) -> None:
        """Display animated startup banner with optional system checks.

        Args:
            checks: Optional list of system checks to display
                   Each check should have 'name' and 'status' keys
        """
        banner = self._create_banner_text()
        self.console.print(Align.center(banner))

        if checks:
            self.console.print()
            check_table = Table.grid(padding=(0, 2))
            check_table.add_column(width=30)
            check_table.add_column(width=10)

            for check in checks:
                name = check.get("name", "Unknown")
                status = check.get("status", "unknown")

                # Status indicators
                if status == "ok":
                    indicator = "[green]✓ OK[/green]"
                elif status == "warning":
                    indicator = "[yellow]⚠ WARNING[/yellow]"
                elif status == "error":
                    indicator = "[red]✗ ERROR[/red]"
                else:
                    indicator = "[dim]• CHECKING[/dim]"

                check_table.add_row(name, indicator)

            self.console.print(check_table)
            self.console.print()


class CommandPalette:
    """Interactive command list with descriptions.

    Displays available commands in an organized, searchable format.
    """

    def __init__(self, commands: Dict[str, str], console: Optional[Console] = None):
        """Initialize command palette.

        Args:
            commands: Dictionary mapping command names to descriptions
            console: Optional Rich Console instance (uses singleton if None)
        """
        self.commands = commands
        self.console = console or get_console()

    def render(self, filter_text: Optional[str] = None) -> Table:
        """Render command palette as table.

        Args:
            filter_text: Optional text to filter commands

        Returns:
            Table: Formatted command table
        """
        table = Table(
            title="[bold cyan]Available Commands[/bold cyan]",
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
            expand=False,
        )

        table.add_column("Command", style="cyan bold", no_wrap=True)
        table.add_column("Description", style="white")

        # Filter and sort commands
        filtered_commands = self.commands.items()
        if filter_text:
            filter_lower = filter_text.lower()
            filtered_commands = [
                (cmd, desc)
                for cmd, desc in self.commands.items()
                if filter_lower in cmd.lower() or filter_lower in desc.lower()
            ]

        # Sort alphabetically
        filtered_commands = sorted(filtered_commands)

        # Add rows
        for command, description in filtered_commands:
            table.add_row(f"/{command}", description)

        return table

    def display(self, filter_text: Optional[str] = None) -> None:
        """Display the command palette to console.

        Args:
            filter_text: Optional text to filter commands
        """
        self.console.print(self.render(filter_text))


class LiveDashboard:
    """Live-updating dashboard with multiple panels.

    Combines multiple widgets into a unified dashboard with automatic
    refresh using Rich's Live display.
    """

    def __init__(self, console: Optional[Console] = None, refresh_rate: int = 4):
        """Initialize live dashboard.

        Args:
            console: Optional Rich Console instance (uses singleton if None)
            refresh_rate: Refresh rate in updates per second
        """
        self.console = console or get_console()
        self.refresh_rate = refresh_rate
        self.layout = Layout()
        self._setup_layout()

    def _setup_layout(self) -> None:
        """Setup the dashboard layout structure."""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )

        self.layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1),
        )

    def update_header(self, content: Any) -> None:
        """Update header section.

        Args:
            content: Rich renderable content for header
        """
        self.layout["header"].update(content)

    def update_left_panel(self, content: Any) -> None:
        """Update left main panel.

        Args:
            content: Rich renderable content for left panel
        """
        self.layout["left"].update(content)

    def update_right_panel(self, content: Any) -> None:
        """Update right main panel.

        Args:
            content: Rich renderable content for right panel
        """
        self.layout["right"].update(content)

    def update_footer(self, content: Any) -> None:
        """Update footer section.

        Args:
            content: Rich renderable content for footer
        """
        self.layout["footer"].update(content)

    def start(self) -> Live:
        """Start live display.

        Returns:
            Live: Rich Live context manager
        """
        return Live(
            self.layout,
            console=self.console,
            refresh_per_second=self.refresh_rate,
            screen=False,
        )


class ProgressTracker:
    """Multi-stage progress tracker with visual indicators.

    Tracks progress across multiple stages with individual progress bars
    and overall completion status.
    """

    def __init__(self, stages: List[str], console: Optional[Console] = None):
        """Initialize progress tracker.

        Args:
            stages: List of stage names
            console: Optional Rich Console instance (uses singleton if None)
        """
        self.stages = stages
        self.console = console or get_console()
        self.current_stage = 0
        self.stage_progress: Dict[int, float] = {i: 0.0 for i in range(len(stages))}
        self.completed_stages: set = set()

    def render(self) -> Panel:
        """Render progress tracker.

        Returns:
            Panel: Formatted progress tracker panel
        """
        table = Table.grid(padding=(0, 2))
        table.add_column(width=20)
        table.add_column(width=35)
        table.add_column(width=10, justify="right")

        for i, stage in enumerate(self.stages):
            # Status icon
            if i in self.completed_stages:
                icon = "[green]✓[/green]"
            elif i == self.current_stage:
                icon = "[yellow]⏵[/yellow]"
            else:
                icon = "[dim]○[/dim]"

            # Progress bar
            progress = self.stage_progress.get(i, 0.0)
            bar_length = 30
            filled = int(bar_length * progress / 100)

            if i in self.completed_stages:
                bar = f"[green]{'█' * bar_length}[/green]"
            elif i == self.current_stage:
                bar = f"[yellow]{'█' * filled}{' ' * (bar_length - filled)}[/yellow]"
            else:
                bar = f"[dim]{' ' * bar_length}[/dim]"

            # Stage name styling
            if i in self.completed_stages:
                stage_style = "green"
            elif i == self.current_stage:
                stage_style = "yellow bold"
            else:
                stage_style = "dim"

            table.add_row(
                f"{icon} [{stage_style}]{stage}[/{stage_style}]",
                bar,
                f"{progress:.0f}%",
            )

        # Overall progress
        overall_progress = (len(self.completed_stages) / len(self.stages)) * 100
        title = f"[bold cyan]Progress: {overall_progress:.0f}%[/bold cyan]"

        return Panel(table, title=title, border_style="cyan", padding=(1, 2))

    def update_stage(self, stage_index: int, progress: float) -> None:
        """Update progress for a specific stage.

        Args:
            stage_index: Index of the stage to update
            progress: Progress value (0-100)
        """
        self.stage_progress[stage_index] = min(100.0, max(0.0, progress))
        if progress >= 100:
            self.completed_stages.add(stage_index)

    def next_stage(self) -> None:
        """Move to the next stage."""
        if self.current_stage < len(self.stages) - 1:
            self.completed_stages.add(self.current_stage)
            self.current_stage += 1

    def display(self) -> None:
        """Display the progress tracker to console."""
        self.console.print(self.render())
