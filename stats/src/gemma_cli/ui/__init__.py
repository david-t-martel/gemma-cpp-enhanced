"""Rich terminal UI components for gemma-cli.

This package provides a comprehensive Rich-based UI system for the CLI,
including themes, components, formatters, and complex widgets.
"""

# Theme system
from .theme import (
    Colors,
    ThemeVariant,
    get_theme,
    get_semantic_style,
    get_memory_tier_style,
)

# Console management
from .console import (
    get_console,
    reset_console,
    set_theme,
    get_current_theme,
    scoped_style,
    capture_output,
    pager_mode,
    status_spinner,
    print_header,
    print_separator,
    clear_screen,
)

# Reusable components
from .components import (
    create_panel,
    create_table,
    create_progress_bar,
    create_tree_view,
    create_syntax_highlighted_code,
    create_markdown_panel,
    create_key_value_display,
    create_centered_text,
    create_columns,
    create_divider,
)

# Message formatters
from .formatters import (
    format_user_message,
    format_assistant_message,
    format_system_message,
    format_error_message,
    format_success_message,
    format_memory_entry,
    format_conversation_history,
    format_model_info,
    format_statistics,
)

# Complex widgets
from .widgets import (
    MemoryDashboard,
    StatusBar,
    StartupBanner,
    CommandPalette,
    ModelSelector,
    LoadingSpinner,
    ConversationView,
)


__all__ = [
    # Theme
    "Colors",
    "ThemeVariant",
    "get_theme",
    "get_semantic_style",
    "get_memory_tier_style",
    # Console
    "get_console",
    "reset_console",
    "set_theme",
    "get_current_theme",
    "scoped_style",
    "capture_output",
    "pager_mode",
    "status_spinner",
    "print_header",
    "print_separator",
    "clear_screen",
    # Components
    "create_panel",
    "create_table",
    "create_progress_bar",
    "create_tree_view",
    "create_syntax_highlighted_code",
    "create_markdown_panel",
    "create_key_value_display",
    "create_centered_text",
    "create_columns",
    "create_divider",
    # Formatters
    "format_user_message",
    "format_assistant_message",
    "format_system_message",
    "format_error_message",
    "format_success_message",
    "format_memory_entry",
    "format_conversation_history",
    "format_model_info",
    "format_statistics",
    # Widgets
    "MemoryDashboard",
    "StatusBar",
    "StartupBanner",
    "CommandPalette",
    "ModelSelector",
    "LoadingSpinner",
    "ConversationView",
]
