"""Pytest configuration for terminal UI tests.

This module provides fixtures for testing terminal-based CLI applications
with Rich UI components, snapshot generation, and video recording.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator, Optional
from io import StringIO
from contextlib import contextmanager
import json
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.terminal_theme import TerminalTheme
import pyte


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_output_dir(request) -> Path:
    """Create output directory for test artifacts."""
    test_name = request.node.name
    output_dir = Path(__file__).parent / "screenshots" / test_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def console_capture() -> Generator[tuple[Console, StringIO], None, None]:
    """Capture Rich console output."""
    buffer = StringIO()
    console = Console(
        file=buffer,
        force_terminal=True,
        force_interactive=False,
        color_system="truecolor",
        width=120,
        height=40,
        legacy_windows=False,
    )
    yield console, buffer
    buffer.close()


@pytest.fixture
def terminal_emulator() -> Generator[pyte.Screen, None, None]:
    """Create terminal emulator for snapshot testing."""
    screen = pyte.Screen(120, 40)
    stream = pyte.Stream(screen)
    screen.stream = stream  # Store stream reference
    yield screen


@pytest.fixture
def snapshot_recorder(test_output_dir):
    """Provide snapshot recording utilities."""
    from tests.playwright.utils.terminal_recorder import TerminalRecorder

    recorder = TerminalRecorder(test_output_dir)
    yield recorder
    # Cleanup recordings older than 7 days
    recorder.cleanup_old_recordings(days=7)


@pytest.fixture
def cli_runner():
    """Provide CLI command runner with output capture."""
    from tests.playwright.utils.cli_runner import AsyncCLIRunner

    runner = AsyncCLIRunner()
    yield runner


@pytest.fixture
def memory_state():
    """Provide isolated memory state for tests."""
    # Use temporary Redis database or in-memory mock
    import tempfile
    from pathlib import Path

    temp_dir = Path(tempfile.mkdtemp(prefix="gemma_test_"))
    yield {
        "redis_db": 15,  # Test database
        "data_dir": temp_dir,
    }

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch):
    """Reset environment variables for each test."""
    # Set test-specific environment
    monkeypatch.setenv("GEMMA_TEST_MODE", "1")
    monkeypatch.setenv("GEMMA_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("NO_COLOR", "0")  # Enable colors

    # Use test config
    test_config = Path(__file__).parent / "test_config.json"
    if test_config.exists():
        monkeypatch.setenv("GEMMA_CONFIG", str(test_config))


@pytest.fixture
def mock_model_inference():
    """Mock model inference for faster tests."""
    from unittest.mock import Mock, AsyncMock

    mock = AsyncMock()
    mock.generate = AsyncMock(return_value={
        "text": "This is a test response from the mocked model.",
        "tokens": 42,
        "time": 1.23,
    })
    return mock


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "ui: mark test as UI/visual test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "snapshot: mark test as snapshot test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Auto-mark tests based on filename
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "test_" in item.nodeid and "ui" in item.nodeid:
            item.add_marker(pytest.mark.ui)


# Assertion helpers
def assert_rich_output_contains(console_output: str, expected: str):
    """Assert Rich console output contains expected string."""
    # Strip ANSI codes if needed for comparison
    from rich.console import Console
    temp_console = Console()
    plain_text = temp_console._render_buffer(console_output, strip_styles=True)
    assert expected in plain_text, f"Expected '{expected}' not found in output"


def assert_color_present(console_output: str, color: str):
    """Assert specific color is present in ANSI output."""
    color_codes = {
        "red": "\x1b[31m",
        "green": "\x1b[32m",
        "cyan": "\x1b[36m",
        "yellow": "\x1b[33m",
        "blue": "\x1b[34m",
        "magenta": "\x1b[35m",
    }
    assert color_codes.get(color, "") in console_output, \
        f"Color '{color}' not found in output"


# Session-level setup/teardown
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before all tests."""
    # Create necessary directories
    base_dir = Path(__file__).parent
    (base_dir / "screenshots").mkdir(exist_ok=True)
    (base_dir / "videos").mkdir(exist_ok=True)
    (base_dir / "recordings").mkdir(exist_ok=True)

    # Create test config
    test_config = {
        "model_path": "mock://gemma-2b-it",
        "redis_db": 15,
        "log_level": "DEBUG",
    }
    (base_dir / "test_config.json").write_text(json.dumps(test_config, indent=2))

    yield

    # Cleanup
    # Keep artifacts for debugging, only clean on success


@contextmanager
def capture_terminal_output():
    """Context manager to capture terminal output."""
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    try:
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        yield stdout_buffer, stderr_buffer
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
