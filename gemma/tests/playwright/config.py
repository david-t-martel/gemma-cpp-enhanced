"""Configuration for terminal UI tests."""

from pathlib import Path
from typing import Dict, Any, List
import os


# Base directories
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent.parent
GEMMA_CLI_PATH = PROJECT_ROOT / "gemma-cli.py"

# Output directories
SCREENSHOTS_DIR = TEST_DIR / "screenshots"
VIDEOS_DIR = TEST_DIR / "videos"
RECORDINGS_DIR = TEST_DIR / "recordings"
SNAPSHOTS_DIR = TEST_DIR / "snapshots"

# Ensure directories exist
for directory in [SCREENSHOTS_DIR, VIDEOS_DIR, RECORDINGS_DIR, SNAPSHOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# Test configuration
TEST_CONFIG: Dict[str, Any] = {
    # Timeouts
    "default_timeout": 30.0,  # 30 seconds
    "slow_timeout": 60.0,  # 1 minute for slow tests
    "command_timeout": 10.0,  # 10 seconds for quick commands

    # Terminal settings
    "terminal_width": 120,
    "terminal_height": 40,

    # Recording settings
    "frame_rate": 2,  # Capture 2 frames per second
    "video_quality": "high",

    # Snapshot settings
    "snapshot_format": "svg",  # svg, png, html
    "snapshot_theme": "monokai",  # monokai, dimmed, light

    # Test data
    "test_model": "mock://gemma-2b-it",
    "redis_test_db": 15,  # Use database 15 for tests

    # Directories
    "screenshots_dir": SCREENSHOTS_DIR,
    "videos_dir": VIDEOS_DIR,
    "recordings_dir": RECORDINGS_DIR,
    "snapshots_dir": SNAPSHOTS_DIR,

    # Cleanup settings
    "cleanup_old_artifacts": True,
    "artifact_retention_days": 7,

    # Debug settings
    "verbose_output": False,
    "save_debug_logs": True,
}


# Commands to test
TEST_COMMANDS: Dict[str, List[str]] = {
    "version": ["python", str(GEMMA_CLI_PATH), "--version"],
    "help": ["python", str(GEMMA_CLI_PATH), "--help"],
    "chat_help": ["python", str(GEMMA_CLI_PATH), "chat", "--help"],
    "memory_help": ["python", str(GEMMA_CLI_PATH), "memory", "--help"],

    "chat_interactive": [
        "python", str(GEMMA_CLI_PATH), "chat",
        "--model", "mock://gemma-2b-it"
    ],

    "memory_stats": ["python", str(GEMMA_CLI_PATH), "memory", "stats"],
    "memory_search": [
        "python", str(GEMMA_CLI_PATH), "memory", "search", "test"
    ],

    "health_check": ["python", str(GEMMA_CLI_PATH), "health", "--verbose"],
}


# Test scenarios with interactions
TEST_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "basic_chat": {
        "command": TEST_COMMANDS["chat_interactive"],
        "interactions": [
            (1.0, "Hello, how are you?"),
            (2.0, "/exit"),
        ],
        "timeout": 15.0,
        "expected_output": ["hello", "message"],
    },

    "memory_store": {
        "command": [
            "python", str(GEMMA_CLI_PATH), "memory", "store",
            "Test memory entry"
        ],
        "timeout": 10.0,
        "expected_output": ["stored", "success", "memory"],
    },

    "chat_with_history": {
        "command": TEST_COMMANDS["chat_interactive"],
        "interactions": [
            (1.0, "First message"),
            (2.0, "Second message"),
            (1.0, "/history"),
            (2.0, "/exit"),
        ],
        "timeout": 20.0,
        "expected_output": ["history", "message"],
    },

    "error_invalid_model": {
        "command": [
            "python", str(GEMMA_CLI_PATH), "chat",
            "--model", "/nonexistent/model.sbs"
        ],
        "timeout": 5.0,
        "expected_output": ["error", "not found", "invalid"],
        "expect_failure": True,
    },
}


# Environment variables for tests
TEST_ENV: Dict[str, str] = {
    "GEMMA_TEST_MODE": "1",
    "GEMMA_LOG_LEVEL": "DEBUG",
    "NO_COLOR": "0",  # Enable colors
    "FORCE_COLOR": "1",
    "PYTHONUNBUFFERED": "1",  # Unbuffered output
    "GEMMA_REDIS_DB": str(TEST_CONFIG["redis_test_db"]),
}


# Model paths for testing
MODEL_PATHS: Dict[str, Path] = {
    "gemma-2b": Path("C:/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs"),
    "gemma-4b": Path(
        "C:/codedev/llm/.models/gemma-3-gemmaCpp-3.0-4b-it-sfp-v1/4b-it-sfp.sbs"
    ),
    "tokenizer-2b": Path(
        "C:/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/tokenizer.spm"
    ),
    "tokenizer-4b": Path(
        "C:/codedev/llm/.models/gemma-3-gemmaCpp-3.0-4b-it-sfp-v1/tokenizer.spm"
    ),
}


# Snapshot comparison settings
COMPARISON_CONFIG: Dict[str, Any] = {
    "threshold": 0.95,  # 95% similarity required
    "ignore_timestamps": True,
    "ignore_dynamic_content": True,
    "highlight_differences": True,
}


# Performance benchmarks
PERFORMANCE_BENCHMARKS: Dict[str, float] = {
    "startup_time": 5.0,  # Max 5 seconds to start
    "help_display_time": 2.0,  # Max 2 seconds to show help
    "memory_stats_time": 10.0,  # Max 10 seconds for stats
    "chat_response_time": 30.0,  # Max 30 seconds for response
}


def get_test_output_dir(test_name: str) -> Path:
    """Get output directory for specific test.

    Args:
        test_name: Name of the test

    Returns:
        Path to test output directory
    """
    output_dir = SCREENSHOTS_DIR / test_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_test_command(command_name: str) -> List[str]:
    """Get command list for testing.

    Args:
        command_name: Name of command from TEST_COMMANDS

    Returns:
        List of command parts

    Raises:
        KeyError: If command name not found
    """
    if command_name not in TEST_COMMANDS:
        raise KeyError(f"Unknown command: {command_name}")
    return TEST_COMMANDS[command_name].copy()


def get_test_scenario(scenario_name: str) -> Dict[str, Any]:
    """Get test scenario configuration.

    Args:
        scenario_name: Name of scenario from TEST_SCENARIOS

    Returns:
        Dictionary with scenario configuration

    Raises:
        KeyError: If scenario name not found
    """
    if scenario_name not in TEST_SCENARIOS:
        raise KeyError(f"Unknown scenario: {scenario_name}")
    return TEST_SCENARIOS[scenario_name].copy()


def get_model_path(model_name: str) -> Path:
    """Get path to model weights.

    Args:
        model_name: Model identifier (e.g., 'gemma-2b')

    Returns:
        Path to model file

    Raises:
        KeyError: If model name not found
        FileNotFoundError: If model file doesn't exist
    """
    if model_name not in MODEL_PATHS:
        raise KeyError(f"Unknown model: {model_name}")

    model_path = MODEL_PATHS[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return model_path


def merge_env(additional_env: Dict[str, str] = None) -> Dict[str, str]:
    """Merge test environment with additional variables.

    Args:
        additional_env: Additional environment variables

    Returns:
        Merged environment dictionary
    """
    env = {**os.environ, **TEST_ENV}
    if additional_env:
        env.update(additional_env)
    return env


# Color codes for output validation
ANSI_COLORS: Dict[str, List[str]] = {
    "red": ["\x1b[31m", "\x1b[91m", "\x1b[1;31m"],
    "green": ["\x1b[32m", "\x1b[92m", "\x1b[1;32m"],
    "yellow": ["\x1b[33m", "\x1b[93m", "\x1b[1;33m"],
    "blue": ["\x1b[34m", "\x1b[94m", "\x1b[1;34m"],
    "magenta": ["\x1b[35m", "\x1b[95m", "\x1b[1;35m"],
    "cyan": ["\x1b[36m", "\x1b[96m", "\x1b[1;36m"],
    "reset": ["\x1b[0m"],
}


def has_color(text: str, color: str) -> bool:
    """Check if text contains specific color code.

    Args:
        text: Text to check
        color: Color name (red, green, cyan, etc.)

    Returns:
        True if color code found
    """
    if color not in ANSI_COLORS:
        return False
    return any(code in text for code in ANSI_COLORS[color])


# Test markers configuration
PYTEST_MARKERS: Dict[str, str] = {
    "ui": "UI/visual test requiring snapshot capture",
    "slow": "Test takes longer than 10 seconds",
    "integration": "Integration test with multiple components",
    "snapshot": "Test generates comparison snapshots",
    "memory": "Test uses Redis memory system",
    "chat": "Test interactive chat interface",
}
