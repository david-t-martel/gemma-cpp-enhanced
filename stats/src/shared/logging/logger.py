"""Core logging utilities for standardized logging across the project.

This module provides a centralized logging configuration that ensures
consistent formatting, levels, and behavior across all components.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .config import LoggingConfig
from .config import get_logging_config


class LogLevel(str, Enum):
    """Standard log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Available log formats."""

    STANDARD = "standard"
    DETAILED = "detailed"
    JSON = "json"
    CONSOLE = "console"


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
                "message",
                "asctime",
            }:
                log_entry[key] = value

        return json.dumps(log_entry)


def _get_formatter(format_type: LogFormat) -> logging.Formatter:
    """Get formatter based on type."""
    formatters = {
        LogFormat.STANDARD: logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
        LogFormat.DETAILED: logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(module)s:%(funcName)s:%(lineno)d - %(message)s"
        ),
        LogFormat.JSON: JSONFormatter(),
        LogFormat.CONSOLE: logging.Formatter("%(levelname)-8s | %(name)-20s | %(message)s"),
    }
    return formatters[format_type]


def setup_logging(
    level: str | LogLevel = LogLevel.INFO,
    format_type: LogFormat = LogFormat.STANDARD,
    log_file: Path | None = None,
    console: bool = True,
    json_output: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    config: LoggingConfig | None = None,
) -> None:
    """Set up centralized logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Log format type
        log_file: Optional log file path
        console: Whether to log to console
        json_output: Whether to use JSON formatting
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        config: Optional logging configuration object
    """
    if config is None:
        config = get_logging_config()

    # Override with provided parameters
    if isinstance(level, str):
        level = LogLevel(level.upper())

    if json_output:
        format_type = LogFormat.JSON

    # Get root logger
    root_logger = logging.getLogger()

    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.setLevel(getattr(logging, level.value))

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = _get_formatter(LogFormat.JSON if json_output else LogFormat.CONSOLE)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, level.value))
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file or config.log_file:
        file_path = log_file or config.log_file
        if file_path:
            # Ensure log directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            file_formatter = _get_formatter(format_type)
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(getattr(logging, level.value))
            root_logger.addHandler(file_handler)

    # Configure third-party loggers
    _configure_third_party_loggers(level)


def _configure_third_party_loggers(level: LogLevel) -> None:
    """Configure log levels for third-party libraries."""
    third_party_configs = {
        "httpx": LogLevel.WARNING,
        "urllib3": LogLevel.WARNING,
        "requests": LogLevel.WARNING,
        "transformers": LogLevel.WARNING,
        "torch": LogLevel.WARNING,
        "asyncio": LogLevel.WARNING,
        "matplotlib": LogLevel.WARNING,
        "PIL": LogLevel.WARNING,
        "websockets": LogLevel.WARNING,
    }

    for logger_name, log_level in third_party_configs.items():
        logging.getLogger(logger_name).setLevel(getattr(logging, log_level.value))


def get_logger(
    name: str | None = None,
    level: str | LogLevel | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> logging.Logger:
    """Get a logger with standardized configuration.

    Args:
        name: Logger name (defaults to caller's module name)
        level: Optional override for logger level
        extra_fields: Extra fields to include in log records

    Returns:
        Configured logger instance
    """
    if name is None:
        # Get caller's module name
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_module = frame.f_back.f_globals.get("__name__", "unknown")
            name = caller_module
        else:
            name = "unknown"

    logger = logging.getLogger(name)

    if level:
        if isinstance(level, str):
            level = LogLevel(level.upper())
        logger.setLevel(getattr(logging, level.value))

    # Add extra fields if provided
    if extra_fields:
        original_handle = logger.handle

        def handle_with_extra(record):
            for key, value in extra_fields.items():
                setattr(record, key, value)
            return original_handle(record)

        logger.handle = handle_with_extra

    return logger


def get_structured_logger(
    name: str | None = None,
    context: dict[str, Any] | None = None,
) -> logging.Logger:
    """Get a logger configured for structured logging.

    Args:
        name: Logger name
        context: Context fields to include in all log records

    Returns:
        Logger configured for structured logging
    """
    return get_logger(name=name, extra_fields=context)


def update_log_level(level: str | LogLevel) -> None:
    """Update the global log level.

    Args:
        level: New log level
    """
    if isinstance(level, str):
        level = LogLevel(level.upper())

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.value))

    # Update all handlers
    for handler in root_logger.handlers:
        handler.setLevel(getattr(logging, level.value))


def is_logging_configured() -> bool:
    """Check if logging has been configured.

    Returns:
        True if logging is configured, False otherwise
    """
    root_logger = logging.getLogger()
    return bool(root_logger.handlers)


def get_log_file_path() -> Path | None:
    """Get the current log file path if configured.

    Returns:
        Path to log file or None if not configured
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            return Path(handler.baseFilename)
    return None


# Auto-configure logging if not already done and we're not in a test environment
if not is_logging_configured() and "pytest" not in sys.modules:
    # Use environment variables or defaults
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    json_logging = os.environ.get("LOG_FORMAT", "").lower() == "json"

    setup_logging(level=LogLevel(log_level.upper()), json_output=json_logging, console=True)
