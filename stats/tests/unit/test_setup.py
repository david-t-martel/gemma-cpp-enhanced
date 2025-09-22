"""Basic tests to validate project setup and configuration."""

from pathlib import Path
import sys

import pytest


def test_python_version():
    """Test that we're running Python 3.11+."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info}"


def test_project_structure():
    """Test that basic project structure exists."""
    project_root = Path(__file__).parent.parent

    # Check key directories exist
    assert (project_root / "src").exists()
    assert (project_root / "tests").exists()
    assert (project_root / "pyproject.toml").exists()
    assert (project_root / ".python-version").exists()
    assert (project_root / "ruff.toml").exists()
    assert (project_root / ".gitignore").exists()
    assert (project_root / "Makefile").exists()


def test_imports():
    """Test that main modules can be imported."""
    # Test domain imports
    from src.application.agents.orchestrator import AgentOrchestrator

    # Skip sandbox imports to avoid docker dependency in basic tests
    # Test application imports
    from src.application.inference.service import InferenceService
    from src.domain.interfaces.llm import LLMProtocol
    from src.domain.models.chat import ChatMessage, ChatSession
    from src.domain.tools.base import BaseTool, ToolRegistry

    # Test infrastructure imports
    from src.infrastructure.llm.base import BaseLLM
    from src.infrastructure.mcp.server import McpServer

    # Test shared imports
    from src.shared.config.settings import Settings, get_settings
    from src.shared.exceptions import ChatbotException

    # Verify classes are available
    assert LLMProtocol is not None
    assert ChatMessage is not None
    assert BaseTool is not None
    assert Settings is not None
    assert ChatbotException is not None
    assert True  # All imports successful


@pytest.mark.asyncio
async def test_tool_registry():
    """Test basic tool registry functionality."""
    from src.domain.tools.base import ToolRegistry

    registry = ToolRegistry()
    assert len(registry.list_tools()) == 0

    # Registry should be empty initially
    assert registry.get_tool("nonexistent") is None


def test_settings_load():
    """Test that settings can be loaded."""
    from src.shared.config.settings import get_settings

    settings = get_settings()
    assert settings is not None
    assert hasattr(settings, "model")
    assert hasattr(settings, "performance")
    assert hasattr(settings, "cache")
