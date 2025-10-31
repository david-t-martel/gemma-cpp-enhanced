"""Test script to verify gemma_cli.ui implementation."""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing gemma_cli.ui imports...")
print("-" * 60)

try:
    # Test theme imports
    from gemma_cli.ui.theme import get_theme, COLORS, MEMORY_TIER_COLORS
    print("✓ theme.py imports successful")

    # Test console imports
    from gemma_cli.ui.console import get_console, print_success
    print("✓ console.py imports successful")

    # Test components imports
    from gemma_cli.ui.components import create_panel, create_table
    print("✓ components.py imports successful")

    # Test formatters imports
    from gemma_cli.ui.formatters import format_user_message, format_assistant_message
    print("✓ formatters.py imports successful")

    # Test widgets imports
    from gemma_cli.ui.widgets import MemoryDashboard, StatusBar
    print("✓ widgets.py imports successful")

    # Test main package imports
    from gemma_cli.ui import get_console, MemoryDashboard
    print("✓ __init__.py imports successful")

    print("-" * 60)
    print("All imports successful!")
    print()

    # Quick functionality test
    print("Testing basic functionality...")
    console = get_console()
    print("✓ Console singleton created")

    theme = get_theme("dark")
    print("✓ Dark theme created")

    dashboard = MemoryDashboard(console)
    print("✓ MemoryDashboard instantiated")

    status = StatusBar(console)
    print("✓ StatusBar instantiated")

    print("-" * 60)
    print("SUCCESS: All tests passed!")

except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
