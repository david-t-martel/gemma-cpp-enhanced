"""Configuration management for Gemma CLI."""

import os
from pathlib import Path
from typing import Any, List, Optional

import toml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class GemmaConfig(BaseModel):
    """Gemma model configuration."""

    default_model: str
    default_tokenizer: str
    executable: str


class ModelPreset(BaseModel):
    """Model preset configuration."""

    name: str
    weights: str
    tokenizer: str
    format: str
    size_gb: float
    avg_tokens_per_sec: int
    quality: str
    use_case: str


class PerformanceProfile(BaseModel):
    """Performance profile configuration."""

    max_tokens: int
    temperature: float
    top_p: float
    description: str


class RedisConfig(BaseModel):
    """Redis configuration."""

    host: str = "localhost"
    port: int = 6380
    db: int = 0
    pool_size: int = 10
    connection_timeout: int = 5
    command_timeout: int = 10
    max_retries: int = 3
    retry_delay: float = 0.1
    enable_fallback: bool = True

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


class Settings(BaseSettings):
    """Main settings class that loads from config.toml."""

    # Configuration sections
    gemma: Optional[GemmaConfig] = None
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

    # Model presets and performance profiles
    models: dict[str, ModelPreset] = Field(default_factory=dict)
    profiles: dict[str, PerformanceProfile] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        env_prefix = "GEMMA_"
        case_sensitive = False


def load_config(config_path: Optional[Path] = None) -> Settings:
    """
    Load configuration from TOML file.

    Args:
        config_path: Path to config.toml file. If None, uses default location.

    Returns:
        Settings instance with loaded configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    if config_path is None:
        # Try to find config.toml in standard locations
        possible_paths = [
            Path("config/config.toml"),
            Path.cwd() / "config" / "config.toml",
            Path.home() / ".gemma_cli" / "config.toml",
        ]

        for path in possible_paths:
            if path.exists():
                config_path = path
                break

    if config_path is None or not config_path.exists():
        raise FileNotFoundError(
            "Config file not found. Please create config/config.toml or ~/.gemma_cli/config.toml"
        )

    # Load TOML file
    try:
        with open(config_path, encoding="utf-8") as f:
            config_data = toml.load(f)
    except (OSError, toml.TomlDecodeError) as e:
        raise ValueError(f"Error loading config file: {e}") from e

    # Parse configuration
    settings = Settings(**config_data)

    return settings


def get_model_preset(settings: Settings, model_name: str) -> Optional[ModelPreset]:
    """
    Get model preset by name.

    Args:
        settings: Settings instance
        model_name: Name of model preset

    Returns:
        ModelPreset if found, None otherwise
    """
    return settings.models.get(model_name)


def get_performance_profile(
    settings: Settings, profile_name: str
) -> Optional[PerformanceProfile]:
    """
    Get performance profile by name.

    Args:
        settings: Settings instance
        profile_name: Name of performance profile

    Returns:
        PerformanceProfile if found, None otherwise
    """
    return settings.profiles.get(profile_name)


def expand_path(path_str: str, allowed_dirs: Optional[List[Path]] = None) -> Path:
    """
    Expand path with security validation to prevent path traversal attacks.

    This function expands ~ and environment variables, then validates the
    resulting path is within allowed directories to prevent malicious
    path traversal attempts (e.g., "../../../etc/shadow").

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
        - Prevents path traversal with ".." components
        - Validates resolved path is within allowed directories
        - Checks for symlink attacks by resolving to real path

    Example:
        >>> expand_path("~/.gemma_cli/config.toml")
        Path("/home/user/.gemma_cli/config.toml")
        >>> expand_path("../../../etc/shadow")  # Raises ValueError
    """
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

    # Expand user home directory (~) and environment variables
    expanded = os.path.expanduser(path_str)
    expanded = os.path.expandvars(expanded)

    # Convert to Path and resolve to absolute path
    path = Path(expanded)

    # Security check 1: Detect path traversal attempts
    # Check both in original string and normalized path
    if ".." in str(path_str) or ".." in path.parts:
        raise ValueError(
            f"Path traversal not allowed: {path_str}\n"
            "Security: Detected '..' component which could access parent directories"
        )

    # Resolve to real path (follows symlinks, gets absolute path)
    try:
        resolved = path.resolve(strict=False)  # Don't require file to exist yet
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Cannot resolve path {path_str}: {e}") from e

    # Security check 2: Validate path is within allowed directories
    is_allowed = False
    for allowed_dir in allowed_dirs:
        try:
            allowed_resolved = allowed_dir.resolve(strict=False)
            # Check if resolved path is relative to allowed directory
            if str(resolved).startswith(str(allowed_resolved)):
                is_allowed = True
                break
        except (OSError, RuntimeError):
            continue  # Skip invalid allowed directories

    if not is_allowed:
        allowed_dirs_str = "\n  - ".join(str(d) for d in allowed_dirs)
        raise ValueError(
            f"Path {resolved} is not within allowed directories:\n  - {allowed_dirs_str}\n"
            "Security: Paths must be within designated safe directories"
        )

    # Security check 3: Additional validation for symlinks
    if path.is_symlink():
        # Verify symlink target is also within allowed directories
        target = path.readlink()
        if target.is_absolute():
            try:
                expand_path(str(target), allowed_dirs)  # Recursive validation
            except ValueError as e:
                raise ValueError(
                    f"Symlink target {target} failed security validation: {e}"
                ) from e

    return resolved
