"""
Configuration management using Builder pattern.
Single Responsibility: Managing configuration only.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Configuration:
    """Immutable configuration object."""

    # Model settings
    model_path: str
    tokenizer_path: Optional[str] = None
    gemma_executable: str = "/mnt/c/codedev/llm/gemma/gemma.cpp/build_wsl/gemma"

    # Generation settings
    max_tokens: int = 2048
    temperature: float = 0.7
    max_context: int = 8192

    # Memory settings
    enable_memory: bool = True
    memory_backend: str = "redis"  # redis, inmemory
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # Server settings
    debug: bool = False
    log_level: str = "INFO"
    metrics_enabled: bool = True

    # Transport settings
    transport_configs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration values."""
        if not self.model_path:
            raise ValueError("model_path is required")

        model_file = Path(self.model_path)
        if not model_file.exists():
            raise ValueError(f"Model file not found: {self.model_path}")

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

        if self.max_context <= 0:
            raise ValueError("max_context must be positive")


class ConfigurationBuilder:
    """Builder for creating Configuration objects."""

    def __init__(self):
        self._config = {}

    def with_model(self, path: str, tokenizer: Optional[str] = None) -> "ConfigurationBuilder":
        """Set model configuration."""
        self._config["model_path"] = path
        if tokenizer:
            self._config["tokenizer_path"] = tokenizer
        return self

    def with_gemma_executable(self, path: str) -> "ConfigurationBuilder":
        """Set gemma executable path."""
        self._config["gemma_executable"] = path
        return self

    def with_generation_params(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_context: Optional[int] = None,
    ) -> "ConfigurationBuilder":
        """Set generation parameters."""
        if max_tokens is not None:
            self._config["max_tokens"] = max_tokens
        if temperature is not None:
            self._config["temperature"] = temperature
        if max_context is not None:
            self._config["max_context"] = max_context
        return self

    def with_memory_backend(self, backend: str = "redis", **kwargs) -> "ConfigurationBuilder":
        """Set memory backend configuration."""
        self._config["memory_backend"] = backend
        self._config["enable_memory"] = True

        if backend == "redis":
            self._config["redis_host"] = kwargs.get("host", "localhost")
            self._config["redis_port"] = kwargs.get("port", 6379)
            self._config["redis_db"] = kwargs.get("db", 0)
        elif backend == "inmemory":
            self._config["enable_memory"] = True
        elif backend == "disabled":
            self._config["enable_memory"] = False

        return self

    def with_debug(self, enabled: bool = True) -> "ConfigurationBuilder":
        """Set debug mode."""
        self._config["debug"] = enabled
        self._config["log_level"] = "DEBUG" if enabled else "INFO"
        return self

    def with_metrics(self, enabled: bool = True) -> "ConfigurationBuilder":
        """Enable or disable metrics."""
        self._config["metrics_enabled"] = enabled
        return self

    def with_transport_config(self, transport: str, **config) -> "ConfigurationBuilder":
        """Add transport-specific configuration."""
        if "transport_configs" not in self._config:
            self._config["transport_configs"] = {}
        self._config["transport_configs"][transport] = config
        return self

    def build(self) -> Configuration:
        """Build the configuration object."""
        if "model_path" not in self._config:
            raise ValueError("model_path is required")

        return Configuration(**self._config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> Configuration:
        """Create configuration from dictionary."""
        builder = cls()

        # Model settings
        if "model_path" in config_dict:
            builder.with_model(config_dict["model_path"], config_dict.get("tokenizer_path"))

        if "gemma_executable" in config_dict:
            builder.with_gemma_executable(config_dict["gemma_executable"])

        # Generation settings
        builder.with_generation_params(
            max_tokens=config_dict.get("max_tokens"),
            temperature=config_dict.get("temperature"),
            max_context=config_dict.get("max_context"),
        )

        # Memory settings
        if "memory_backend" in config_dict:
            backend = config_dict["memory_backend"]
            builder.with_memory_backend(
                backend,
                host=config_dict.get("redis_host"),
                port=config_dict.get("redis_port"),
                db=config_dict.get("redis_db"),
            )

        # Debug settings
        if "debug" in config_dict:
            builder.with_debug(config_dict["debug"])

        # Metrics
        if "metrics_enabled" in config_dict:
            builder.with_metrics(config_dict["metrics_enabled"])

        # Transport configs
        if "transport_configs" in config_dict:
            for transport, config in config_dict["transport_configs"].items():
                builder.with_transport_config(transport, **config)

        return builder.build()

    @classmethod
    def from_args(cls, args) -> Configuration:
        """Create configuration from command-line arguments."""
        builder = cls()

        # Required model path
        builder.with_model(args.model, args.tokenizer)

        if hasattr(args, "gemma_executable"):
            builder.with_gemma_executable(args.gemma_executable)

        # Generation params
        builder.with_generation_params(
            max_tokens=getattr(args, "max_tokens", None),
            temperature=getattr(args, "temperature", None),
            max_context=getattr(args, "max_context", None),
        )

        # Memory backend
        if hasattr(args, "no_redis") and args.no_redis:
            builder.with_memory_backend("inmemory")
        else:
            builder.with_memory_backend(
                "redis",
                host=getattr(args, "redis_host", "localhost"),
                port=getattr(args, "redis_port", 6379),
                db=getattr(args, "redis_db", 0),
            )

        # Debug mode
        if hasattr(args, "debug"):
            builder.with_debug(args.debug)

        # Transport configs
        if hasattr(args, "mode"):
            if args.mode in ["http", "all"]:
                builder.with_transport_config(
                    "http",
                    host=getattr(args, "host", "localhost"),
                    port=getattr(args, "port", 8080),
                )

            if args.mode in ["websocket", "all"]:
                ws_port = getattr(args, "ws_port", None) or (getattr(args, "port", 8080) + 1)
                builder.with_transport_config(
                    "websocket", host=getattr(args, "host", "localhost"), port=ws_port
                )

        return builder.build()
