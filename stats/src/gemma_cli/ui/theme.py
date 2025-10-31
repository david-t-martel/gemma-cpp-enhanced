"""Rich theme configuration and color schemes.

This module provides a centralized theme system for the gemma-cli UI,
including color palettes, semantic styles, and theme variants.
"""

from typing import Dict, Literal
from rich.theme import Theme
from rich.style import Style


# Color Palette - Base Colors
class Colors:
    """Base color palette for the UI."""

    # Primary colors
    PRIMARY = "#00D9FF"  # Cyan
    SECONDARY = "#FF00FF"  # Magenta
    ACCENT = "#FFD700"  # Gold

    # Semantic colors
    SUCCESS = "#00FF00"  # Green
    ERROR = "#FF0000"  # Red
    WARNING = "#FFA500"  # Orange
    INFO = "#00D9FF"  # Cyan

    # Grayscale
    WHITE = "#FFFFFF"
    GRAY_LIGHT = "#D0D0D0"
    GRAY = "#808080"
    GRAY_DARK = "#404040"
    BLACK = "#000000"

    # Special
    HIGHLIGHT = "#FFFF00"  # Yellow
    DIM = "#606060"  # Dark gray


# Semantic Style Definitions
SEMANTIC_STYLES: Dict[str, Style] = {
    # Status styles
    "success": Style(color=Colors.SUCCESS, bold=True),
    "error": Style(color=Colors.ERROR, bold=True),
    "warning": Style(color=Colors.WARNING, bold=True),
    "info": Style(color=Colors.INFO, bold=False),

    # Message role styles
    "user": Style(color=Colors.PRIMARY, bold=True),
    "assistant": Style(color=Colors.SECONDARY, bold=False),
    "system": Style(color=Colors.ACCENT, italic=True),

    # UI component styles
    "title": Style(color=Colors.PRIMARY, bold=True, underline=True),
    "subtitle": Style(color=Colors.GRAY_LIGHT, bold=True),
    "label": Style(color=Colors.GRAY_LIGHT, bold=False),
    "value": Style(color=Colors.WHITE, bold=True),
    "dim": Style(color=Colors.DIM, dim=True),

    # Code and syntax
    "code": Style(color=Colors.ACCENT, bgcolor=Colors.GRAY_DARK),
    "command": Style(color=Colors.PRIMARY, bold=True),
    "path": Style(color=Colors.INFO, underline=True),

    # Memory tiers
    "memory_working": Style(color="#FF6B6B", bold=True),  # Red
    "memory_short": Style(color="#FFA500", bold=True),  # Orange
    "memory_long": Style(color="#FFD700", bold=True),  # Gold
    "memory_episodic": Style(color="#00D9FF", bold=True),  # Cyan
    "memory_semantic": Style(color="#00FF00", bold=True),  # Green

    # Progress and metrics
    "progress_bar": Style(color=Colors.PRIMARY, bold=True),
    "progress_complete": Style(color=Colors.SUCCESS, bold=True),
    "metric_good": Style(color=Colors.SUCCESS, bold=False),
    "metric_warn": Style(color=Colors.WARNING, bold=False),
    "metric_bad": Style(color=Colors.ERROR, bold=False),
}


# Theme Definitions
DARK_THEME = Theme({
    # Base styles
    "repr.number": Style(color=Colors.ACCENT),
    "repr.str": Style(color=Colors.SUCCESS),
    "repr.bool_true": Style(color=Colors.SUCCESS, bold=True),
    "repr.bool_false": Style(color=Colors.ERROR, bold=True),
    "repr.none": Style(color=Colors.DIM, italic=True),

    # Logging levels
    "logging.level.debug": Style(color=Colors.DIM),
    "logging.level.info": Style(color=Colors.INFO),
    "logging.level.warning": Style(color=Colors.WARNING, bold=True),
    "logging.level.error": Style(color=Colors.ERROR, bold=True),
    "logging.level.critical": Style(color=Colors.ERROR, bold=True, reverse=True),

    # Semantic styles
    **SEMANTIC_STYLES,

    # Panel borders
    "panel.border": Style(color=Colors.PRIMARY),
    "panel.title": Style(color=Colors.PRIMARY, bold=True),

    # Table styles
    "table.header": Style(color=Colors.PRIMARY, bold=True),
    "table.border": Style(color=Colors.GRAY),

    # Markdown styles
    "markdown.h1": Style(color=Colors.PRIMARY, bold=True, underline=True),
    "markdown.h2": Style(color=Colors.SECONDARY, bold=True),
    "markdown.h3": Style(color=Colors.ACCENT, bold=True),
    "markdown.code": Style(color=Colors.ACCENT, bgcolor=Colors.GRAY_DARK),
    "markdown.link": Style(color=Colors.INFO, underline=True),
})


LIGHT_THEME = Theme({
    # Base styles
    "repr.number": Style(color="#B8860B"),  # Dark goldenrod
    "repr.str": Style(color="#008000"),  # Dark green
    "repr.bool_true": Style(color="#008000", bold=True),
    "repr.bool_false": Style(color="#8B0000", bold=True),
    "repr.none": Style(color="#808080", italic=True),

    # Logging levels
    "logging.level.debug": Style(color="#808080"),
    "logging.level.info": Style(color="#0000FF"),
    "logging.level.warning": Style(color="#FF8C00", bold=True),
    "logging.level.error": Style(color="#DC143C", bold=True),
    "logging.level.critical": Style(color="#8B0000", bold=True, reverse=True),

    # Semantic styles (adapted for light background)
    "success": Style(color="#008000", bold=True),
    "error": Style(color="#DC143C", bold=True),
    "warning": Style(color="#FF8C00", bold=True),
    "info": Style(color="#0000FF", bold=False),
    "user": Style(color="#0000CD", bold=True),
    "assistant": Style(color="#8B008B", bold=False),
    "system": Style(color="#B8860B", italic=True),

    # Panel and table borders
    "panel.border": Style(color="#0000CD"),
    "panel.title": Style(color="#0000CD", bold=True),
    "table.header": Style(color="#0000CD", bold=True),
    "table.border": Style(color="#808080"),
})


ThemeVariant = Literal["dark", "light"]


def get_theme(variant: ThemeVariant = "dark") -> Theme:
    """Get a Rich theme by variant name.

    Args:
        variant: Theme variant to use ('dark' or 'light')

    Returns:
        Rich Theme object configured with the specified variant

    Raises:
        ValueError: If variant is not 'dark' or 'light'
    """
    if variant == "dark":
        return DARK_THEME
    elif variant == "light":
        return LIGHT_THEME
    else:
        raise ValueError(f"Unknown theme variant: {variant}. Use 'dark' or 'light'.")


def get_semantic_style(name: str) -> Style:
    """Get a semantic style by name.

    Args:
        name: Semantic style name (e.g., 'success', 'error', 'user')

    Returns:
        Rich Style object for the semantic name

    Raises:
        KeyError: If semantic style name is not defined
    """
    if name not in SEMANTIC_STYLES:
        raise KeyError(
            f"Unknown semantic style: {name}. "
            f"Available: {', '.join(SEMANTIC_STYLES.keys())}"
        )
    return SEMANTIC_STYLES[name]


def get_memory_tier_style(tier: str) -> Style:
    """Get style for a specific memory tier.

    Args:
        tier: Memory tier name ('working', 'short', 'long', 'episodic', 'semantic')

    Returns:
        Rich Style object for the memory tier

    Raises:
        KeyError: If memory tier is not defined
    """
    style_key = f"memory_{tier}"
    if style_key not in SEMANTIC_STYLES:
        raise KeyError(
            f"Unknown memory tier: {tier}. "
            f"Available: working, short, long, episodic, semantic"
        )
    return SEMANTIC_STYLES[style_key]
