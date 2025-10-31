"""Rich theme configuration and color schemes for gemma-cli.

This module provides a comprehensive theme system for the terminal UI,
including color palettes, semantic styles, and theme creation utilities.
"""

from typing import Dict

from rich.style import Style
from rich.theme import Theme

# Primary color palette based on design system
COLORS = {
    # Primary colors
    "primary": "#00d4aa",  # Teal
    "secondary": "#6366f1",  # Indigo
    "accent": "#f59e0b",  # Amber
    # Status colors
    "success": "#10b981",  # Green
    "warning": "#f59e0b",  # Amber
    "error": "#ef4444",  # Red
    "info": "#3b82f6",  # Blue
    # UI elements
    "border": "#4b5563",  # Gray
    "muted": "#9ca3af",  # Light gray
    "text": "#e5e7eb",  # Almost white
    "background": "#1f2937",  # Dark gray
    # Light theme variants
    "light_border": "#d1d5db",
    "light_muted": "#6b7280",
    "light_text": "#1f2937",
    "light_background": "#f9fafb",
}

# Semantic styles for different message types
MESSAGE_STYLES = {
    "user": Style(color="cyan", bold=True),
    "assistant": Style(color="green"),
    "system": Style(color="yellow", italic=True),
    "error": Style(color="red", bold=True),
    "success": Style(color="green", bold=True),
    "warning": Style(color="#f59e0b", bold=True),
    "info": Style(color="blue"),
}

# Memory tier colors for 5-tier RAG system
MEMORY_TIER_COLORS = {
    "working": "bright_cyan",
    "short_term": "bright_blue",
    "long_term": "bright_green",
    "episodic": "bright_magenta",
    "semantic": "bright_yellow",
}

# Progress bar styles
PROGRESS_STYLES = {
    "bar.complete": Style(color=COLORS["primary"]),
    "bar.finished": Style(color=COLORS["success"]),
    "bar.pulse": Style(color=COLORS["accent"]),
    "progress.description": Style(color=COLORS["text"]),
    "progress.percentage": Style(color=COLORS["primary"], bold=True),
    "progress.remaining": Style(color=COLORS["muted"]),
}

# Table styles
TABLE_STYLES = {
    "table.header": Style(color=COLORS["primary"], bold=True),
    "table.border": Style(color=COLORS["border"]),
    "table.caption": Style(color=COLORS["muted"], italic=True),
}


def create_dark_theme() -> Theme:
    """Create dark theme optimized for terminal use.

    Returns:
        Theme: Rich Theme object with dark color scheme
    """
    return Theme(
        {
            # Basic text styles
            "info": Style(color=COLORS["info"]),
            "warning": Style(color=COLORS["warning"], bold=True),
            "error": Style(color=COLORS["error"], bold=True),
            "success": Style(color=COLORS["success"], bold=True),
            # Message role styles
            "user": MESSAGE_STYLES["user"],
            "assistant": MESSAGE_STYLES["assistant"],
            "system": MESSAGE_STYLES["system"],
            # UI component styles
            "panel.border": Style(color=COLORS["border"]),
            "panel.title": Style(color=COLORS["primary"], bold=True),
            "panel.subtitle": Style(color=COLORS["muted"]),
            # Progress bar styles
            **PROGRESS_STYLES,
            # Table styles
            **TABLE_STYLES,
            # Memory tier styles
            "memory.working": Style(color=MEMORY_TIER_COLORS["working"], bold=True),
            "memory.short_term": Style(color=MEMORY_TIER_COLORS["short_term"]),
            "memory.long_term": Style(color=MEMORY_TIER_COLORS["long_term"]),
            "memory.episodic": Style(color=MEMORY_TIER_COLORS["episodic"]),
            "memory.semantic": Style(color=MEMORY_TIER_COLORS["semantic"]),
            # Status indicators
            "status.active": Style(color=COLORS["primary"], blink=True),
            "status.idle": Style(color=COLORS["muted"]),
            # Code and syntax
            "code": Style(color=COLORS["accent"], bgcolor="#1a1a1a"),
            "code.keyword": Style(color="#ff79c6", bold=True),
            "code.string": Style(color="#f1fa8c"),
            "code.comment": Style(color="#6272a4", italic=True),
            # Links and references
            "link": Style(color=COLORS["secondary"], underline=True),
            "repr.url": Style(color=COLORS["secondary"], underline=True),
            # Numbers and values
            "repr.number": Style(color=COLORS["accent"]),
            "repr.bool_true": Style(color=COLORS["success"]),
            "repr.bool_false": Style(color=COLORS["error"]),
            "repr.none": Style(color=COLORS["muted"], italic=True),
            # Special text
            "dim": Style(color=COLORS["muted"]),
            "bold": Style(bold=True),
            "italic": Style(italic=True),
        }
    )


def create_light_theme() -> Theme:
    """Create light theme for bright environments.

    Returns:
        Theme: Rich Theme object with light color scheme
    """
    return Theme(
        {
            # Basic text styles (adjusted for light background)
            "info": Style(color="#1e40af"),
            "warning": Style(color="#b45309", bold=True),
            "error": Style(color="#dc2626", bold=True),
            "success": Style(color="#15803d", bold=True),
            # Message role styles
            "user": Style(color="#0e7490", bold=True),
            "assistant": Style(color="#15803d"),
            "system": Style(color="#ca8a04", italic=True),
            # UI component styles
            "panel.border": Style(color=COLORS["light_border"]),
            "panel.title": Style(color="#0891b2", bold=True),
            "panel.subtitle": Style(color=COLORS["light_muted"]),
            # Progress bar styles (adjusted)
            "bar.complete": Style(color="#0891b2"),
            "bar.finished": Style(color="#15803d"),
            "bar.pulse": Style(color="#b45309"),
            "progress.description": Style(color=COLORS["light_text"]),
            "progress.percentage": Style(color="#0891b2", bold=True),
            "progress.remaining": Style(color=COLORS["light_muted"]),
            # Table styles
            "table.header": Style(color="#0891b2", bold=True),
            "table.border": Style(color=COLORS["light_border"]),
            "table.caption": Style(color=COLORS["light_muted"], italic=True),
            # Memory tier styles (darker for visibility)
            "memory.working": Style(color="#0e7490", bold=True),
            "memory.short_term": Style(color="#1e40af"),
            "memory.long_term": Style(color="#15803d"),
            "memory.episodic": Style(color="#a21caf"),
            "memory.semantic": Style(color="#ca8a04"),
            # Status indicators
            "status.active": Style(color="#0891b2", blink=True),
            "status.idle": Style(color=COLORS["light_muted"]),
            # Code and syntax
            "code": Style(color="#b45309", bgcolor="#f5f5f5"),
            "code.keyword": Style(color="#be185d", bold=True),
            "code.string": Style(color="#15803d"),
            "code.comment": Style(color="#6b7280", italic=True),
            # Links and references
            "link": Style(color="#4f46e5", underline=True),
            "repr.url": Style(color="#4f46e5", underline=True),
            # Numbers and values
            "repr.number": Style(color="#b45309"),
            "repr.bool_true": Style(color="#15803d"),
            "repr.bool_false": Style(color="#dc2626"),
            "repr.none": Style(color=COLORS["light_muted"], italic=True),
            # Special text
            "dim": Style(color=COLORS["light_muted"]),
            "bold": Style(bold=True),
            "italic": Style(italic=True),
        }
    )


def create_theme(name: str = "dark") -> Theme:
    """Create Rich theme with semantic styles.

    Args:
        name: Theme name ("dark" or "light")

    Returns:
        Theme: Rich Theme object

    Raises:
        ValueError: If theme name is not recognized
    """
    if name == "dark":
        return create_dark_theme()
    elif name == "light":
        return create_light_theme()
    else:
        raise ValueError(f"Unknown theme name: {name}. Use 'dark' or 'light'.")


def get_theme(name: str = "dark") -> Theme:
    """Get theme by name (dark or light).

    This is an alias for create_theme() for backward compatibility.

    Args:
        name: Theme name ("dark" or "light")

    Returns:
        Theme: Rich Theme object
    """
    return create_theme(name)


def get_style_for_message_type(message_type: str) -> Style:
    """Get style for a specific message type.

    Args:
        message_type: Type of message (user, assistant, system, error, etc.)

    Returns:
        Style: Rich Style object for the message type
    """
    return MESSAGE_STYLES.get(message_type, Style())


def get_color_for_memory_tier(tier: str) -> str:
    """Get color name for a memory tier.

    Args:
        tier: Memory tier name (working, short_term, etc.)

    Returns:
        str: Rich color name
    """
    return MEMORY_TIER_COLORS.get(tier, "white")
