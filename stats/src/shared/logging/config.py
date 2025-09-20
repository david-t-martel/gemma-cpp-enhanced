"""Logging configuration management.

This module handles loading and managing logging configuration
from various sources including environment variables and config files.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel
from pydantic import Field


class LoggingConfig(BaseModel):
    """Logging configuration model."""

    level: str = Field(default="INFO", description="Default logging level")
    format_type: str = Field(default="standard", description="Log format type")
    console_enabled: bool = Field(default=True, description="Enable console logging")
    json_format: bool = Field(default=False, description="Use JSON formatting")
    log_file: Path | None = Field(default=None, description="Log file path")
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Max log file size")
    backup_count: int = Field(default=5, description="Number of backup files")
    third_party_level: str = Field(default="WARNING", description="Third-party logger level")

    # Performance logging options
    performance_logging: bool = Field(default=False, description="Enable performance logging")
    function_call_logging: bool = Field(default=False, description="Log function calls")

    # Structured logging options
    include_context: bool = Field(default=True, description="Include context in logs")
    include_process_info: bool = Field(default=True, description="Include process info")

    # Environment-specific settings
    environment: str = Field(default="development", description="Environment name")
    debug_mode: bool = Field(default=False, description="Enable debug mode")

    @classmethod
    def from_environment(cls) -> LoggingConfig:
        """Create config from environment variables."""
        return cls(
            level=os.environ.get("LOG_LEVEL", "INFO"),
            format_type=os.environ.get("LOG_FORMAT", "standard"),
            console_enabled=os.environ.get("LOG_CONSOLE", "true").lower() == "true",
            json_format=os.environ.get("LOG_JSON", "false").lower() == "true",
            log_file=Path(log_file) if (log_file := os.environ.get("LOG_FILE")) else None,
            max_file_size=int(os.environ.get("LOG_MAX_SIZE", "10485760")),  # 10MB
            backup_count=int(os.environ.get("LOG_BACKUP_COUNT", "5")),
            third_party_level=os.environ.get("LOG_THIRD_PARTY_LEVEL", "WARNING"),
            performance_logging=os.environ.get("LOG_PERFORMANCE", "false").lower() == "true",
            function_call_logging=os.environ.get("LOG_FUNCTION_CALLS", "false").lower() == "true",
            environment=os.environ.get("ENVIRONMENT", "development"),
            debug_mode=os.environ.get("DEBUG", "false").lower() == "true",
        )

    @classmethod
    def from_file(cls, config_path: Path) -> LoggingConfig:
        """Load config from file."""
        if config_path.suffix.lower() == ".json":
            import json

            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)
        elif config_path.suffix.lower() in (".yml", ".yaml"):
            try:
                import yaml

                with open(config_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required to load YAML config files")
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        return cls(**data.get("logging", data))

    def to_dict(self) -> dict[str, any]:
        """Convert to dictionary."""
        return self.model_dump()

    def update_from_env(self) -> LoggingConfig:
        """Update current config with environment variables."""
        env_config = self.from_environment()

        # Update only non-default values from environment
        for field_name, field_info in self.model_fields.items():
            env_value = getattr(env_config, field_name)
            if env_value != field_info.default:
                setattr(self, field_name, env_value)

        return self


# Global configuration instance
_logging_config: LoggingConfig | None = None


def get_logging_config() -> LoggingConfig:
    """Get the current logging configuration.

    Returns:
        Current logging configuration
    """
    global _logging_config

    if _logging_config is None:
        # Try to load from config file first
        config_paths = [
            Path("logging.json"),
            Path("config/logging.json"),
            Path(".config/logging.json"),
            Path("logging.yaml"),
            Path("config/logging.yaml"),
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    _logging_config = LoggingConfig.from_file(config_path)
                    break
                except Exception:
                    continue  # Try next config file

        # Fall back to environment variables
        if _logging_config is None:
            _logging_config = LoggingConfig.from_environment()

    return _logging_config


def update_logging_config(config: LoggingConfig) -> None:
    """Update the global logging configuration.

    Args:
        config: New logging configuration
    """
    global _logging_config
    _logging_config = config


def reset_logging_config() -> None:
    """Reset logging configuration to defaults."""
    global _logging_config
    _logging_config = None


def get_project_log_dir() -> Path:
    """Get the default project log directory.

    Returns:
        Path to project log directory
    """
    # Try to find project root
    current_dir = Path.cwd()
    for parent in [current_dir, *list(current_dir.parents)]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent / "logs"

    # Fall back to current directory
    return current_dir / "logs"


def get_default_log_file() -> Path:
    """Get default log file path.

    Returns:
        Default log file path
    """
    log_dir = get_project_log_dir()
    return log_dir / "gemma-stats.log"


def setup_development_logging() -> LoggingConfig:
    """Set up development-friendly logging configuration.

    Returns:
        Development logging configuration
    """
    config = LoggingConfig(
        level="DEBUG",
        format_type="detailed",
        console_enabled=True,
        json_format=False,
        log_file=get_default_log_file(),
        performance_logging=True,
        function_call_logging=True,
        debug_mode=True,
        environment="development",
    )

    update_logging_config(config)
    return config


def setup_production_logging() -> LoggingConfig:
    """Set up production-friendly logging configuration.

    Returns:
        Production logging configuration
    """
    config = LoggingConfig(
        level="INFO",
        format_type="json",
        console_enabled=True,
        json_format=True,
        log_file=get_default_log_file(),
        performance_logging=False,
        function_call_logging=False,
        debug_mode=False,
        environment="production",
        third_party_level="ERROR",  # Reduce third-party noise in production
    )

    update_logging_config(config)
    return config
