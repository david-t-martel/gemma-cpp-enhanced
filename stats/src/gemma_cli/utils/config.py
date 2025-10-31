"""Configuration loading and validation utilities."""

import tomllib
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field, ValidationError


class ModelConfig(BaseModel):
    """Model configuration."""
    default_model: str = "4b"
    model_path: str
    tokenizer_path: str | None = None
    gemma_executable: str


class GenerationConfig(BaseModel):
    """Generation parameters."""
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_context: int = Field(default=8192, ge=512, le=131072)


class RAGConfig(BaseModel):
    """RAG system configuration."""
    enabled: bool = True
    redis_url: str = "redis://localhost:6379"
    prefer_backend: str | None = None


class MCPConfig(BaseModel):
    """MCP server configuration."""
    host: str = "localhost"
    port: int = Field(default=8765, ge=1, le=65535)
    timeout: int = Field(default=30, ge=1, le=300)


class SystemConfig(BaseModel):
    """System configuration."""
    system_prompt: str = "You are a helpful AI assistant."


class GemmaConfig(BaseModel):
    """Complete configuration model."""
    model: ModelConfig
    generation: GenerationConfig
    rag: RAGConfig
    mcp: MCPConfig
    system: SystemConfig


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from TOML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If configuration is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    return config


def validate_config(config_path: Path) -> Tuple[List[str], List[str]]:
    """Validate configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of (errors, warnings)
    """
    errors = []
    warnings = []

    try:
        config = load_config(config_path)

        # Validate with Pydantic
        try:
            GemmaConfig(**config)
        except ValidationError as e:
            for error in e.errors():
                field = ".".join(str(x) for x in error["loc"])
                errors.append(f"{field}: {error['msg']}")

        # Check file paths
        model_path = Path(config.get("model", {}).get("model_path", ""))
        if not model_path.exists():
            warnings.append(f"Model file not found: {model_path}")

        tokenizer_path = config.get("model", {}).get("tokenizer_path")
        if tokenizer_path:
            tokenizer_path = Path(tokenizer_path)
            if not tokenizer_path.exists():
                warnings.append(f"Tokenizer file not found: {tokenizer_path}")

        gemma_exe = Path(config.get("model", {}).get("gemma_executable", ""))
        if not gemma_exe.exists():
            warnings.append(f"Gemma executable not found: {gemma_exe}")

    except FileNotFoundError as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(f"Configuration parsing error: {e}")

    return errors, warnings
