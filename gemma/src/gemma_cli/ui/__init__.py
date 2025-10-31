"""Gemma CLI UI system - Rich terminal components.

This package provides a comprehensive set of Rich-based UI components
for the gemma-cli terminal interface, including themes, formatters,
and interactive widgets.
"""

# Theme system
from gemma_cli.ui.theme import (
    COLORS,
    MEMORY_TIER_COLORS,
    MESSAGE_STYLES,
    PROGRESS_STYLES,
    TABLE_STYLES,
    create_theme,
    get_color_for_memory_tier,
    get_style_for_message_type,
    get_theme,
)

# Console factory and utilities
from gemma_cli.ui.console import (
    captured_console,
    clear_screen,
    create_console,  # NEW: Factory function for dependency injection
    get_console,  # DEPRECATED: Use create_console() instead
    get_console_height,
    get_console_width,
    print_error,
    print_info,
    print_rule,
    print_success,
    print_warning,
    print_with_style,
    reset_console,
    status_context,
    styled_console,
)

# Reusable components
from gemma_cli.ui.components import (
    create_download_progress,
    create_error_panel,
    create_grid_table,
    create_info_panel,
    create_key_value_table,
    create_markdown,
    create_panel,
    create_progress,
    create_simple_progress,
    create_status_table,
    create_success_panel,
    create_syntax,
    create_table,
    create_tree,
    create_warning_panel,
)

# Message formatters
from gemma_cli.ui.formatters import (
    format_assistant_message,
    format_conversation_history,
    format_error_message,
    format_memory_entry,
    format_memory_stats,
    format_model_info,
    format_progress_message,
    format_statistics,
    format_system_message,
    format_timing_info,
    format_token_usage,
    format_user_message,
)

# Complex widgets
from gemma_cli.ui.widgets import (
    CommandPalette,
    LiveDashboard,
    MemoryDashboard,
    ProgressTracker,
    StartupBanner,
    StatusBar,
)

__all__ = [
    # Theme
    "get_theme",
    "create_theme",
    "COLORS",
    "MESSAGE_STYLES",
    "MEMORY_TIER_COLORS",
    "PROGRESS_STYLES",
    "TABLE_STYLES",
    "get_style_for_message_type",
    "get_color_for_memory_tier",
    # Console
    "create_console",  # NEW: Recommended way to create console instances
    "get_console",  # DEPRECATED: Use create_console() instead
    "reset_console",
    "styled_console",
    "captured_console",
    "print_with_style",
    "print_error",
    "print_success",
    "print_warning",
    "print_info",
    "clear_screen",
    "print_rule",
    "status_context",
    "get_console_width",
    "get_console_height",
    # Components
    "create_panel",
    "create_table",
    "create_grid_table",
    "create_progress",
    "create_simple_progress",
    "create_download_progress",
    "create_tree",
    "create_syntax",
    "create_markdown",
    "create_key_value_table",
    "create_status_table",
    "create_info_panel",
    "create_error_panel",
    "create_success_panel",
    "create_warning_panel",
    # Formatters
    "format_user_message",
    "format_assistant_message",
    "format_system_message",
    "format_error_message",
    "format_memory_entry",
    "format_conversation_history",
    "format_model_info",
    "format_statistics",
    "format_token_usage",
    "format_timing_info",
    "format_progress_message",
    "format_memory_stats",
    # Widgets
    "MemoryDashboard",
    "StatusBar",
    "StartupBanner",
    "CommandPalette",
    "LiveDashboard",
    "ProgressTracker",
]

__version__ = "2.0.0"
__author__ = "Gemma CLI Team"
__description__ = "Rich terminal UI components for gemma-cli"


# Convenience function for quick testing
def demo() -> None:
    """Run a quick demo of UI components."""
    from gemma_cli.ui.console import get_console

    console = get_console()

    # Demo startup banner
    banner = StartupBanner(console)
    checks = [
        {"name": "Model Loading", "status": "ok"},
        {"name": "Memory System", "status": "ok"},
        {"name": "RAG Backend", "status": "warning"},
    ]
    banner.show(checks)

    # Demo memory dashboard
    console.print()
    dashboard = MemoryDashboard(console)
    stats = {
        "working": 8,
        "short_term": 45,
        "long_term": 1200,
        "episodic": 350,
        "semantic": 8500,
    }
    dashboard.display(stats)

    # Demo status bar
    console.print()
    status = StatusBar(console)
    status.update(
        model="gemma-2b-it",
        memory_tier="short_term",
        tokens_used=1234,
        response_time=156.7,
        context_size=2048,
    )
    status.display()

    # Demo message formatting
    console.print()
    console.print(format_user_message("How does the 5-tier memory system work?"))
    console.print()
    console.print(
        format_assistant_message(
            "The 5-tier memory system uses **hierarchical storage** with different "
            "retention policies for each tier:\n\n"
            "1. **Working** - Immediate context (15 entries)\n"
            "2. **Short-term** - Recent interactions (100 entries)\n"
            "3. **Long-term** - Consolidated knowledge (10K entries)\n"
            "4. **Episodic** - Event sequences (5K entries)\n"
            "5. **Semantic** - Graph relationships (50K entries)",
            metadata={"tokens": 247, "time_ms": 156.7},
        )
    )

    console.print()
    console.print(format_success_panel("UI Demo completed successfully!"))


if __name__ == "__main__":
    # Run demo when module is executed directly
    demo()
