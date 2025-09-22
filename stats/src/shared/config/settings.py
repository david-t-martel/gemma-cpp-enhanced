"""Configuration management for the LLM chatbot framework.

This module provides Pydantic-based configuration management with environment
variable support, validation, and type safety.
"""

import os
import warnings
from enum import Enum
from pathlib import Path

import torch
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from pydantic_settings import BaseSettings

from .redis_config import RedisConfig
from .redis_config import create_redis_config


class LogLevel(str, Enum):
    """Available logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DeviceType(str, Enum):
    """Available compute devices."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"


class ModelConfig(BaseModel):
    """Configuration for model parameters."""

    name: str = Field(..., description="Model name or path")
    max_length: int = Field(2048, ge=1, le=32768, description="Maximum sequence length")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    top_k: int = Field(50, ge=1, description="Top-k sampling")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    do_sample: bool = Field(True, description="Whether to use sampling")
    batch_size: int = Field(1, ge=1, description="Inference batch size")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate model name is not empty."""
        if not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class PerformanceConfig(BaseModel):
    """Configuration for performance optimization."""

    device: DeviceType = DeviceType.AUTO
    precision: str = Field("float16", description="Model precision (float32, float16, bfloat16)")
    use_flash_attention: bool = Field(False, description="Enable Flash Attention if available")
    use_bettertransformer: bool = Field(False, description="Enable BetterTransformer optimization")
    use_torch_compile: bool = Field(False, description="Enable torch.compile optimization")
    memory_fraction: float = Field(0.9, ge=0.1, le=1.0, description="GPU memory fraction to use")
    offload_to_cpu: bool = Field(False, description="Offload unused layers to CPU")

    @field_validator("precision")
    @classmethod
    def validate_precision(cls, v: str) -> str:
        """Validate precision format."""
        valid_precisions = {"float32", "float16", "bfloat16", "int8", "int4"}
        if v not in valid_precisions:
            raise ValueError(f"Precision must be one of: {valid_precisions}")
        return v


class CacheConfig(BaseModel):
    """Configuration for caching."""

    enabled: bool = Field(True, description="Enable response caching")
    ttl_seconds: int = Field(3600, ge=0, description="Cache TTL in seconds")
    max_size_mb: int = Field(100, ge=1, description="Maximum cache size in MB")
    cache_dir: Path = Field(Path("cache"), description="Cache directory path")


class SecurityConfig(BaseModel):
    """Configuration for security settings."""

    max_request_size_mb: int = Field(10, ge=1, description="Maximum request size in MB")
    rate_limit_per_minute: int = Field(60, ge=1, description="Rate limit per minute")
    allowed_origins: list[str] = Field(
        default_factory=list,  # Empty by default - must be explicitly configured
        description="CORS allowed origins - MUST be configured for production",
    )
    api_key_required: bool = Field(
        True, description="Require API key authentication - ALWAYS True in production"
    )
    api_keys: list[str] = Field(
        default_factory=list, description="Valid API keys (loaded from environment)"
    )
    enable_rate_limiting: bool = Field(True, description="Enable rate limiting")
    max_prompt_length: int = Field(4096, ge=1, description="Maximum prompt length in characters")
    block_sensitive_patterns: bool = Field(
        True, description="Block prompts with sensitive patterns"
    )
    enable_request_validation: bool = Field(True, description="Enable request validation")
    jwt_secret: str | None = Field(None, description="JWT secret for token-based auth")
    jwt_algorithm: str = Field("HS256", description="JWT algorithm")
    jwt_expiry_hours: int = Field(24, ge=1, description="JWT token expiry in hours")

    @field_validator("allowed_origins")
    @classmethod
    def validate_origins(cls, v: list[str]) -> list[str]:
        """Validate CORS origins - disallow wildcard and enforce configuration."""

        # Check for wildcard - absolutely forbidden
        if "*" in v or any("*" in origin for origin in v):
            raise ValueError(
                "Wildcard (*) in allowed_origins is forbidden for security. "
                "Please specify exact origins."
            )

        # In production, require explicit configuration
        if os.getenv("GEMMA_ENVIRONMENT", "").lower() == "production":
            if not v:
                # Load from environment variable if not set
                env_origins = os.getenv("GEMMA_ALLOWED_ORIGINS", "")
                if env_origins:
                    v = [origin.strip() for origin in env_origins.split(",") if origin.strip()]

                if not v:
                    raise ValueError(
                        "CORS allowed_origins must be configured in production. "
                        "Set GEMMA_ALLOWED_ORIGINS environment variable."
                    )
        elif not v:
            # Development mode - use safe defaults
            v = [
                "http://localhost:3000",
                "http://localhost:8000",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8000",
            ]
            warnings.warn(
                f"No CORS origins configured. Using development defaults: {v}",
                UserWarning,
                stacklevel=2,
            )

        return v

    @field_validator("api_key_required")
    @classmethod
    def validate_api_key_required(cls, v: bool) -> bool:
        """Ensure API key is required in production."""

        if os.getenv("GEMMA_ENVIRONMENT", "").lower() == "production" and not v:
            raise ValueError("API key authentication must be enabled in production")
        return v

    @model_validator(mode="after")
    def validate_api_keys_configured(self) -> "SecurityConfig":
        """Ensure API keys are configured when authentication is required."""

        if self.api_key_required:
            # Try to load API keys from environment
            if not self.api_keys:
                env_keys = os.getenv("GEMMA_API_KEYS", "")
                if env_keys:
                    self.api_keys = [k.strip() for k in env_keys.split(",") if k.strip()]

            # In production, require at least one API key
            if os.getenv("GEMMA_ENVIRONMENT", "").lower() == "production" and not self.api_keys:
                raise ValueError(
                    "At least one API key must be configured when authentication is required in production. "
                    "Set GEMMA_API_KEYS environment variable."
                )
            elif not self.api_keys:
                # Development mode - generate a temporary key
                import secrets

                # Generate a simple temporary key without using validators module
                temp_key = f"sk-dev-{secrets.token_urlsafe(32)}"
                self.api_keys = [temp_key]
                warnings.warn(
                    f"No API keys configured. Generated temporary key for development: {temp_key}",
                    UserWarning,
                    stacklevel=2,
                )

        return self


class ServerConfig(BaseModel):
    """Configuration for HTTP server."""

    host: str = Field("0.0.0.0", description="Server host")
    port: int = Field(8000, ge=1024, le=65535, description="Server port")
    workers: int = Field(1, ge=1, description="Number of worker processes")
    reload: bool = Field(False, description="Enable auto-reload in development")
    access_log: bool = Field(True, description="Enable access logging")


class Settings(BaseSettings):
    """Main application settings.

    This class uses Pydantic Settings to automatically load configuration
    from environment variables with the GEMMA_ prefix.
    """

    # Application metadata
    app_name: str = Field("Gemma Chatbot", description="Application name")
    version: str = Field("0.1.0", description="Application version")
    environment: str = Field("development", description="Environment (development, production)")

    # Logging configuration
    log_level: LogLevel = LogLevel.INFO
    log_format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format string"
    )
    log_file: Path | None = Field(None, description="Log file path (None for stdout only)")

    # Model configuration
    model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name="google/gemma-2b-it",
            max_length=2048,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
            batch_size=1,
        )
    )

    # Performance configuration
    performance: PerformanceConfig = Field(default_factory=lambda: PerformanceConfig())

    # Cache configuration
    cache: CacheConfig = Field(default_factory=lambda: CacheConfig())

    # Security configuration
    security: SecurityConfig = Field(default_factory=lambda: SecurityConfig())

    # Server configuration
    server: ServerConfig = Field(default_factory=lambda: ServerConfig())

    # Redis configuration
    redis: RedisConfig = Field(
        default_factory=create_redis_config, description="Redis configuration"
    )

    # Paths
    data_dir: Path = Field(Path("data"), description="Data directory")
    models_dir: Path = Field(Path("models"), description="Models cache directory")

    # Timeouts (in seconds)
    inference_timeout: int = Field(300, ge=1, description="Inference timeout in seconds")
    model_load_timeout: int = Field(600, ge=1, description="Model load timeout in seconds")

    class Config:
        env_prefix = "GEMMA_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields from .env

    @field_validator("data_dir", "models_dir")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Create directories if they don't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("cache")
    @classmethod
    def create_cache_dir(cls, v: CacheConfig) -> CacheConfig:
        """Create cache directory if it doesn't exist."""
        v.cache_dir.mkdir(parents=True, exist_ok=True)
        return v

    def get_device(self) -> str:
        """Get the actual device to use based on availability."""
        if self.performance.device == DeviceType.AUTO:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.performance.device.value

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    def get_model_path(self) -> Path:
        """Get the full path for model storage."""
        return self.models_dir / self.model.name.replace("/", "_")


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance.

    This function implements a singleton pattern to ensure configuration
    is loaded only once throughout the application lifecycle.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment/files.

    Useful for testing or when configuration changes need to be picked up
    without restarting the application.
    """
    global _settings
    _settings = Settings()
    return _settings
