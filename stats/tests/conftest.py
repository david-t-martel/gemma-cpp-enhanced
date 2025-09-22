"""Pytest configuration for the test suite."""

from pathlib import Path
import sys

# Add the src directory to Python path so imports work
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Also add project root for relative imports
sys.path.insert(0, str(project_root))
