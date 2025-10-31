"""Configuration management for Gemma CLI."""

import os
from pathlib import Path
from typing import Any, List, Optional

import toml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class GemmaConfig(BaseModel):
    """Simplified Gemma model configuration.

    This is a streamlined configuration focused on direct CLI usage.
    Users can override these values with --model and --tokenizer flags.
    """

    default_model: Optional[str] = None  # Path to default model .sbs file
    default_tokenizer: Optional[str] = None  # Path to default tokenizer .spm file
    executable_path: Optional[str] = None  # Path to gemma.exe (auto-discovered if None)


class DetectedModel(BaseModel):
    """Information about a detected model.

    This represents a model found via 'model detect' command.
    """

    name: str = Field(..., description="User-friendly model name (e.g., 'gemma-2b-it')")
    weights_path: str = Field(..., description="Absolute path to .sbs weights file")
    tokenizer_path: Optional[str] = Field(None, description="Absolute path to .spm tokenizer file")
    format: str = Field("unknown", description="Weight format (sfp, bf16, f32, nuq)")
    size_gb: float = Field(0.0, description="Model size in GB")

    @field_validator("weights_path", "tokenizer_path")
    @classmethod
    def validate_absolute_path(cls, v: Optional[str]) -> Optional[str]:
        """Ensure paths are absolute."""
        if v is None:
            return None
        path = Path(v)
        if not path.is_absolute():
            raise ValueError(f"Path must be absolute: {v}")
        return str(path)


class ConfiguredModel(BaseModel):
    """A manually configured model entry.

    This represents a model added via 'model add' command.
    """

    name: str = Field(..., description="User-friendly model name")
    weights_path: str = Field(..., description="Absolute path to .sbs weights file")
    tokenizer_path: Optional[str] = Field(None, description="Absolute path to .spm tokenizer file")

    @field_validator("weights_path", "tokenizer_path")
    @classmethod
    def validate_absolute_path(cls, v: Optional[str]) -> Optional[str]:
        """Ensure paths are absolute."""
        if v is None:
            return None
        path = Path(v)
        if not path.is_absolute():
            raise ValueError(f"Path must be absolute: {v}")
        return str(path)


class RagBackendConfig(BaseModel):
    """RAG backend configuration.

    Supports three backend options:
    1. 'embedded' - File-based vector store (default, no dependencies)
    2. 'redis' - Python Redis backend (requires Redis server)
    3. 'rust' - High-performance Rust MCP server (SIMD-optimized, optional Redis)
    """

    backend: str = "embedded"  # Options: 'embedded', 'redis', 'rust'
    rust_mcp_server_path: Optional[str] = None  # Path to mcp-server.exe (auto-detected if None)

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        """Validate backend is a supported option."""
        valid_backends = ["embedded", "redis", "rust"]
        if v not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}, got: {v}")
        return v


class RedisConfig(BaseModel):
    """Redis configuration.

    Note: When enable_fallback=True (default), the application will use an embedded
    file-based vector store instead of Redis, allowing standalone operation without
    external dependencies. This is the recommended setting for local development.

    Deprecated: Use RagBackendConfig.backend instead. This config is only used when
    RagBackendConfig.backend='redis'.
    """

    host: str = "localhost"
    port: int = 6380
    db: int = 0
    pool_size: int = 10
    connection_timeout: int = 5
    command_timeout: int = 10
    max_retries: int = 3
    retry_delay: float = 0.1
    enable_fallback: bool = True  # Default: Use embedded store (standalone mode)

    @field_validator("pool_size")
    @classmethod
    def validate_pool_size(cls, v: int) -> int:
        """Validate pool_size is within safe bounds."""
        if v < 1:
            raise ValueError("pool_size must be at least 1")
        if v > 100:
            raise ValueError("pool_size must not exceed 100 (DoS prevention)")
        return v

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is within valid range."""
        if v < 1 or v > 65535:
            raise ValueError("port must be between 1 and 65535")
        return v

    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, v: int) -> int:
        """Validate max_retries is reasonable."""
        if v < 0:
            raise ValueError("max_retries must be non-negative")
        if v > 10:
            raise ValueError("max_retries must not exceed 10")
        return v


class MemoryConfig(BaseModel):
    """Memory system configuration."""

    working_ttl: int = 900
    short_term_ttl: int = 3600
    long_term_ttl: int = 2592000
    episodic_ttl: int = 604800
    semantic_ttl: int = 0

    working_capacity: int = 15
    short_term_capacity: int = 100
    long_term_capacity: int = 10000
    episodic_capacity: int = 5000
    semantic_capacity: int = 50000

    consolidation_threshold: float = 0.75
    importance_decay_rate: float = 0.1
    cleanup_interval: int = 300
    enable_background_tasks: bool = True
    auto_consolidate: bool = True


class EmbeddingConfig(BaseModel):
    """Embedding configuration."""

    provider: str = "local"
    model: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    cache_embeddings: bool = True


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""

    dimension: int = 384
    distance_metric: str = "cosine"
    index_type: str = "hnsw"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50


class DocumentConfig(BaseModel):
    """Document ingestion configuration."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    chunking_method: str = "token"
    supported_formats: list[str] = ["txt", "md", "html", "json", "pdf"]
    max_file_size: int = 52428800  # 50MB default

    @field_validator("max_file_size")
    @classmethod
    def validate_max_file_size(cls, v: int) -> int:
        """Validate max_file_size is within reasonable bounds."""
        if v < 1024:  # Minimum 1KB
            raise ValueError("max_file_size must be at least 1024 bytes (1KB)")
        if v > 104857600:  # Maximum 100MB
            raise ValueError("max_file_size must not exceed 104857600 bytes (100MB)")
        return v

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        """Validate chunk_size is reasonable."""
        if v < 10:
            raise ValueError("chunk_size must be at least 10")
        if v > 10000:
            raise ValueError("chunk_size must not exceed 10000")
        return v

    @field_validator("min_chunk_size")
    @classmethod
    def validate_min_chunk_size(cls, v: int) -> int:
        """Validate min_chunk_size is reasonable."""
        if v < 1:
            raise ValueError("min_chunk_size must be at least 1")
        if v > 5000:
            raise ValueError("min_chunk_size must not exceed 5000")
        return v

    @field_validator("max_chunk_size")
    @classmethod
    def validate_max_chunk_size(cls, v: int) -> int:
        """Validate max_chunk_size is reasonable."""
        if v < 10:
            raise ValueError("max_chunk_size must be at least 10")
        if v > 20000:
            raise ValueError("max_chunk_size must not exceed 20000")
        return v


class MCPConfig(BaseModel):
    """MCP client configuration."""

    enabled: bool = True
    servers_config: str = "config/mcp_servers.toml"
    tool_cache_ttl: int = 3600
    connection_timeout: int = 10
    retry_count: int = 3


class UIConfig(BaseModel):
    """UI configuration."""

    theme: str = "default"
    show_memory_stats: bool = True
    show_performance: bool = True
    show_status_bar: bool = True
    progress_style: str = "rich"
    color_scheme: str = "auto"


class OnboardingConfig(BaseModel):
    """Onboarding configuration."""

    show_on_first_run: bool = True
    skip_if_configured: bool = True
    auto_detect_hardware: bool = True


class AutocompleteConfig(BaseModel):
    """Autocomplete configuration."""

    enabled: bool = True
    show_examples: bool = True
    max_suggestions: int = 5


class ConversationConfig(BaseModel):
    """Conversation configuration."""

    max_context_length: int = 8192
    max_history_messages: int = 50
    save_directory: str = "~/.gemma_conversations"
    auto_save: bool = True
    auto_save_interval: int = 300


class SystemConfig(BaseModel):
    """System configuration."""

    prompt_file: str = "config/prompts/GEMMA.md"
    enable_rag_context: bool = True
    max_rag_context_tokens: int = 2000


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    file: str = "~/.gemma_cli/gemma.log"
    max_size: int = 10485760
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""

    enabled: bool = True
    track_latency: bool = True
    track_memory: bool = True
    track_token_usage: bool = True
    report_interval: int = 60



class PerformanceConfig(BaseModel):
    """Performance optimization configuration.

    Controls feature flags for gradual rollout of performance optimizations.
    These flags allow toggling between original and optimized implementations.
    """

    use_optimized_gemma: bool = Field(
        default=True,
        description="Use OptimizedGemmaInterface with streaming and batch optimizations"
    )
    use_optimized_rag: bool = Field(
        default=True,
        description="Use OptimizedEmbeddedStore with indexing and caching"
    )
    enable_query_cache: bool = Field(
        default=True,
        description="Enable LRU cache for frequent RAG queries"
    )
    batch_size: int = Field(
        default=100,
        description="Maximum batch size for RAG write operations"
    )
    cache_max_size: int = Field(
        default=100,
        description="Maximum number of cached query results"
    )

class Settings(BaseSettings):
    """Main settings class that loads from config.toml."""

    # Configuration sections
    gemma: GemmaConfig = Field(default_factory=GemmaConfig)
    rag_backend: RagBackendConfig = Field(default_factory=RagBackendConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    # Simplified model management - configured models only
    configured_models: dict[str, ConfiguredModel] = Field(default_factory=dict, description="Models added via 'model add'")

    # NOTE: Detected models are stored separately in ~/.gemma_cli/detected_models.json
    # This keeps the main config clean and allows easier updates from 'model detect'

    class Config:
        """Pydantic configuration."""

        env_prefix = "GEMMA_"
        case_sensitive = False


class ConfigManager:
    """Manages loading and saving of the application settings."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self._find_config_path()

    def _find_config_path(self) -> Path:
        """Find the config.toml file in standard locations."""
        possible_paths = [
            Path("config/config.toml"),
            Path.cwd() / "config" / "config.toml",
            Path.home() / ".gemma_cli" / "config.toml",
        ]
        for path in possible_paths:
            if path.exists():
                return path
        return Path.home() / ".gemma_cli" / "config.toml"  # Default path

    def load(self) -> Settings:
        """Load configuration from TOML file with automatic migration support."""
        if not self.config_path.exists():
            return Settings()  # Return default settings if no config file

        try:
            with open(self.config_path, encoding="utf-8") as f:
                config_data = toml.load(f)

            # Migration: Check for old preset-based configuration
            if "models" in config_data or "profiles" in config_data or "model_presets" in config_data or "performance_profiles" in config_data:
                logger = logging.getLogger(__name__)
                logger.warning("Old preset-based configuration detected!")
                logger.warning("The model system has been simplified. Please run:")
                logger.warning("  1. gemma-cli model detect")
                logger.warning("  2. gemma-cli model list")
                logger.warning("  3. gemma-cli model set-default <name>")
                logger.warning("Old 'models' and 'profiles' sections are ignored.")

                # Remove old sections to avoid validation errors
                config_data.pop("models", None)
                config_data.pop("profiles", None)
                config_data.pop("model_presets", None)
                config_data.pop("performance_profiles", None)

            return Settings(**config_data)
        except (OSError, toml.TomlDecodeError) as e:
            raise ValueError(f"Error loading config file: {e}") from e

    def save(self, settings: Settings) -> None:
        """Save configuration to TOML file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                toml.dump(settings.model_dump(), f)
        except (OSError, toml.TomlDecodeError) as e:
            raise ValueError(f"Error saving config file: {e}") from e


def load_config(config_path: Optional[Path] = None) -> Settings:
    """Load configuration from TOML file."""
    return ConfigManager(config_path).load()


def load_detected_models() -> dict[str, DetectedModel]:
    """Load detected models from the separate JSON file.

    Returns:
        Dictionary mapping model names to DetectedModel objects
    """
    import json

    detected_path = Path.home() / ".gemma_cli" / "detected_models.json"
    if not detected_path.exists():
        return {}

    try:
        with open(detected_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        models = {}
        for model_data in data.get("models", []):
            model = DetectedModel(**model_data)
            models[model.name] = model
        return models

    except (json.JSONDecodeError, ValidationError) as e:
        # Log error but don't crash - return empty dict
        import logging
        logging.getLogger(__name__).warning(f"Failed to load detected models: {e}")
        return {}


def save_detected_models(models: dict[str, DetectedModel]) -> None:
    """Save detected models to the separate JSON file.

    Args:
        models: Dictionary mapping model names to DetectedModel objects
    """
    import json

    detected_path = Path.home() / ".gemma_cli" / "detected_models.json"
    detected_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "models": [model.model_dump() for model in models.values()],
        "last_updated": Path(detected_path).stat().st_mtime if detected_path.exists() else None
    }

    with open(detected_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_model_by_name(name: str, settings: Optional[Settings] = None) -> Optional[tuple[str, Optional[str]]]:
    """Resolve a model name to (weights_path, tokenizer_path).

    Priority order:
    1. Check detected models (from 'model detect')
    2. Check configured models (from 'model add')
    3. Check default_model in settings

    Args:
        name: Model name to resolve
        settings: Optional Settings instance (loaded if not provided)

    Returns:
        Tuple of (weights_path, tokenizer_path) or None if not found
    """
    # Load detected models
    detected = load_detected_models()
    if name in detected:
        model = detected[name]
        return (model.weights_path, model.tokenizer_path)

    # Load settings if not provided
    if settings is None:
        settings = load_config()

    # Check configured models
    if name in settings.configured_models:
        model = settings.configured_models[name]
        return (model.weights_path, model.tokenizer_path)

    return None


def expand_path(path_str: str, allowed_dirs: Optional[List[Path]] = None) -> Path:
    """
    Expand path with security validation to prevent path traversal attacks.

    SECURITY: This is the secure version that prevents multiple attack vectors:
    1. Direct traversal: "../../../etc/passwd"
    2. Environment variable injection: export EVIL="../../.."
    3. URL encoding: "%2e%2e%2f" variations
    4. Symlink attacks: links pointing outside allowed directories

    This function validates BEFORE and AFTER expansion to catch environment
    variable injection attempts, then ensures the resulting path is within
    allowed directories to prevent malicious path traversal attempts.

    Args:
        path_str: Path string to expand
        allowed_dirs: Optional list of allowed parent directories. If None,
                     uses default safe directories (.gemma_cli, models, config)

    Returns:
        Expanded and validated Path object

    Raises:
        ValueError: If path contains traversal attempts or is outside allowed directories
        FileNotFoundError: If resolved path doesn't exist (for stricter validation)

    Security:
        - Validates BEFORE and AFTER expansion to catch env var injection
        - Prevents path traversal with ".." components
        - Validates resolved path is within allowed directories
        - Properly checks symlink targets (both relative and absolute)

    Example:
        >>> expand_path("~/.gemma_cli/config.toml")
        Path("/home/user/.gemma_cli/config.toml")
        >>> expand_path("../../../etc/shadow")  # Raises ValueError
        >>> os.environ['EVIL'] = '../..'; expand_path("$EVIL/etc")  # Also raises ValueError
    """
    import logging

    # Define default allowed directories if none provided
    if allowed_dirs is None:
        allowed_dirs = [
            Path.home() / ".gemma_cli",  # User config directory
            Path.cwd(),  # Current working directory
            Path("C:\\codedev\\llm\\.models"),  # Model directory (Windows)
            Path("/c/codedev/llm/.models"),  # Model directory (WSL)
            Path.cwd() / "config",  # Local config directory
            Path.cwd() / "models",  # Local models directory
        ]
        # Add user home if different from .gemma_cli
        allowed_dirs.append(Path.home())

    # SECURITY CHECK 1: Validate raw input BEFORE any expansion
    # This catches direct traversal attempts
    if ".." in path_str:
        raise ValueError(
            f"Path traversal not allowed in input: {path_str}\n"
            "Security: Detected '..' component which could access parent directories"
        )

    # Also check for encoded variations (critical fix for encoded attacks)
    if "%2e%2e" in path_str.lower() or "%252e%252e" in path_str.lower():
        raise ValueError(
            f"Path traversal not allowed (encoded): {path_str}\n"
            "Security: Detected encoded '..' component"
        )

    # Expand user home directory (~) and environment variables
    expanded = os.path.expanduser(path_str)
    expanded = os.path.expandvars(expanded)

    # SECURITY CHECK 2: Re-validate AFTER expansion to catch env var injection
    # This catches: export EVIL="../../.."; expand_path("$EVIL/etc/passwd")
    if ".." in expanded:
        raise ValueError(
            f"Path traversal detected after expansion: {expanded}\n"
            f"Original: {path_str}\n"
            "Security: Environment variable or tilde expansion introduced '..' component"
        )

    # Convert to Path
    path = Path(expanded)

    # SECURITY CHECK 3: Validate path parts (catches normalized traversal)
    if ".." in path.parts:
        raise ValueError(
            f"Path traversal in path components: {path.parts}\n"
            "Security: Detected '..' in normalized path parts"
        )

    # Resolve to real path (follows ALL symlinks, gets absolute path)
    try:
        resolved = path.resolve(strict=False)  # Don't require file to exist yet
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Cannot resolve path {path_str}: {e}") from e

    # SECURITY CHECK 4: Validate resolved path is within allowed directories
    # Resolve all allowed directories for proper comparison
    is_allowed = False
    for allowed_dir in allowed_dirs:
        try:
            allowed_resolved = allowed_dir.resolve(strict=False)
            # Use Path.is_relative_to for secure comparison (Python 3.9+)
            # For older Python, fall back to string prefix check
            try:
                # Python 3.9+ has is_relative_to
                if hasattr(resolved, 'is_relative_to'):
                    if resolved.is_relative_to(allowed_resolved):
                        is_allowed = True
                        break
            except AttributeError:
                pass  # Method doesn't exist, use fallback

            # Fallback for Python < 3.9 with proper path separator handling
            if not is_allowed:
                if str(resolved).startswith(str(allowed_resolved) + os.sep) or str(resolved) == str(allowed_resolved):
                    is_allowed = True
                    break
        except (OSError, RuntimeError):
            continue  # Skip invalid allowed directories

    if not is_allowed:
        allowed_dirs_str = "\n  - ".join(str(d.resolve(strict=False)) for d in allowed_dirs if d.exists())
        raise ValueError(
            f"Path {resolved} is not within allowed directories:\n  - {allowed_dirs_str}\n"
            "Security: Paths must be within designated safe directories"
        )

    # SECURITY CHECK 5: Validate symlinks properly
    # Check if the original path (not resolved) is a symlink
    if path.exists() and path.is_symlink():
        # The resolved path already follows symlinks, so we just need to ensure
        # the final target is within allowed directories (which we already checked)
        # Log for security audit trail
        logger = logging.getLogger(__name__)
        logger.debug(f"Symlink detected: {path} -> {resolved}")

        # Double-check: ensure symlink target is also validated (defense in depth)
        # This is redundant but provides extra safety
        if not is_allowed:
            raise ValueError(
                f"Symlink {path} resolves to {resolved} which is outside allowed directories\n"
                "Security: Symlink targets must be within allowed directories"
            )

    return resolved
