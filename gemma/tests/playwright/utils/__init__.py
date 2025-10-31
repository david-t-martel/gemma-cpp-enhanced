"""Terminal UI testing utilities for gemma-cli.

This package provides tools for automated testing of terminal-based
user interfaces with snapshot generation and visual validation.
"""

from .terminal_recorder import TerminalRecorder
from .cli_runner import AsyncCLIRunner, CLIResult

__all__ = [
    "TerminalRecorder",
    "AsyncCLIRunner",
    "CLIResult",
]
