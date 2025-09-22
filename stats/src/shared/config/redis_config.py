"""Redis configuration management for the LLM chatbot framework.

This module provides centralized Redis configuration with environment variable
support, connection validation, and fallback mechanisms.
"""

import logging
import os
import warnings
from enum import Enum
from typing import Any

import redis
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from redis.exceptions import ConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError

logger = logging.getLogger(__name__)


class RedisConnectionMode(str, Enum):
    """Redis connection modes."""

    STANDALONE = "standalone"
    CLUSTER = "cluster"
    SENTINEL = "sentinel"


class RedisSSLConfig(BaseModel):
    """Redis SSL/TLS configuration."""

    enabled: bool = Field(False, description="Enable SSL/TLS connection")
    cert_file: str | None = Field(None, description="Path to SSL certificate file")
    key_file: str | None = Field(None, description="Path to SSL private key file")
    ca_certs: str | None = Field(None, description="Path to CA certificates file")
    verify_mode: str = Field("required", description="SSL verification mode")
    check_hostname: bool = Field(True, description="Verify SSL hostname")


class RedisPoolConfig(BaseModel):
    """Redis connection pool configuration."""

    max_connections: int = Field(20, ge=1, le=1000, description="Maximum pool connections")
    min_connections: int = Field(1, ge=1, description="Minimum pool connections")
    retry_on_timeout: bool = Field(True, description="Retry on connection timeout")
    retry_on_error: list[str] = Field(
        default_factory=lambda: ["ConnectionError", "TimeoutError"],
        description="Error types to retry on",
    )
    health_check_interval: int = Field(30, ge=1, description="Health check interval in seconds")
    socket_keepalive: bool = Field(True, description="Enable TCP keepalive")
    socket_keepalive_options: dict[str, int] = Field(
        default_factory=lambda: {"TCP_KEEPIDLE": 1, "TCP_KEEPINTVL": 3, "TCP_KEEPCNT": 5},
        description="TCP keepalive options",
    )


class RedisTimeoutConfig(BaseModel):
    """Redis timeout configuration."""

    connection_timeout: float = Field(5.0, ge=0.1, description="Connection timeout in seconds")
    socket_timeout: float = Field(10.0, ge=0.1, description="Socket timeout in seconds")
    command_timeout: float = Field(30.0, ge=0.1, description="Command timeout in seconds")
    blocking_timeout: float = Field(0.1, ge=0.0, description="Blocking command timeout")


class RedisRetryConfig(BaseModel):
    """Redis retry configuration."""

    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(0.1, ge=0.0, description="Initial retry delay in seconds")
    exponential_backoff: bool = Field(True, description="Use exponential backoff")
    max_retry_delay: float = Field(2.0, ge=0.1, description="Maximum retry delay")
    jitter: bool = Field(True, description="Add random jitter to retry delays")


class RedisFallbackConfig(BaseModel):
    """Redis fallback configuration."""

    enabled: bool = Field(True, description="Enable fallback to in-memory cache")
    fallback_type: str = Field("memory", description="Fallback cache type")
    max_fallback_size: int = Field(1000, ge=1, description="Maximum fallback cache entries")
    fallback_ttl: int = Field(300, ge=1, description="Fallback cache TTL in seconds")
    warn_on_fallback: bool = Field(True, description="Warn when falling back")


class RedisConfig(BaseModel):
    """Comprehensive Redis configuration."""

    # Connection settings
    host: str = Field("localhost", description="Redis server hostname")
    port: int = Field(6380, ge=1, le=65535, description="Redis server port (Windows default: 6380)")
    db: int = Field(0, ge=0, le=15, description="Redis database number")
    username: str | None = Field(None, description="Redis username (Redis 6.0+)")
    password: str | None = Field(None, description="Redis password")

    # Connection mode and clustering
    mode: RedisConnectionMode = Field(RedisConnectionMode.STANDALONE, description="Connection mode")
    cluster_nodes: list[str] = Field(default_factory=list, description="Cluster node addresses")
    sentinel_hosts: list[str] = Field(default_factory=list, description="Sentinel host addresses")
    sentinel_service: str | None = Field(None, description="Sentinel service name")

    # Sub-configurations
    ssl: RedisSSLConfig = Field(default_factory=RedisSSLConfig, description="SSL configuration")
    pool: RedisPoolConfig = Field(
        default_factory=RedisPoolConfig, description="Connection pool config"
    )
    timeouts: RedisTimeoutConfig = Field(
        default_factory=RedisTimeoutConfig, description="Timeout config"
    )
    retry: RedisRetryConfig = Field(default_factory=RedisRetryConfig, description="Retry config")
    fallback: RedisFallbackConfig = Field(
        default_factory=RedisFallbackConfig, description="Fallback config"
    )

    # Advanced settings
    encoding: str = Field("utf-8", description="String encoding for Redis operations")
    decode_responses: bool = Field(True, description="Automatically decode string responses")
    compression: bool = Field(False, description="Enable data compression")
    key_prefix: str = Field("llm_stats:", description="Key prefix for all Redis operations")

    # Environment detection
    auto_detect_environment: bool = Field(
        True, description="Auto-detect Windows/Docker environment"
    )

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate Redis port with Windows-specific defaults."""
        if v == 6379:
            # Windows users often run Redis on 6380 to avoid conflicts
            if os.name == "nt":  # Windows
                warnings.warn(
                    "Port 6379 detected on Windows. Consider using 6380 for Windows Redis servers. "
                    "Set REDIS_PORT=6380 environment variable.",
                    UserWarning,
                    stacklevel=2,
                )
        return v

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate Redis hostname."""
        if not v.strip():
            raise ValueError("Redis host cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_configuration(self) -> "RedisConfig":
        """Validate overall Redis configuration."""
        # Cluster mode validation
        if self.mode == RedisConnectionMode.CLUSTER and not self.cluster_nodes:
            raise ValueError("Cluster nodes must be specified for cluster mode")

        # Sentinel mode validation
        if self.mode == RedisConnectionMode.SENTINEL:
            if not self.sentinel_hosts or not self.sentinel_service:
                raise ValueError("Sentinel hosts and service name required for sentinel mode")

        # SSL validation
        if self.ssl.enabled:
            if self.ssl.cert_file and not os.path.exists(self.ssl.cert_file):
                warnings.warn(f"SSL certificate file not found: {self.ssl.cert_file}", stacklevel=2)

        # Pool configuration validation
        if self.pool.min_connections > self.pool.max_connections:
            raise ValueError("Minimum connections cannot exceed maximum connections")

        return self

    def get_redis_url(self) -> str:
        """Generate Redis connection URL."""
        scheme = "rediss" if self.ssl.enabled else "redis"

        # Build auth part
        auth_part = ""
        if self.username and self.password:
            auth_part = f"{self.username}:{self.password}@"
        elif self.password:
            auth_part = f":{self.password}@"

        # Build URL
        url = f"{scheme}://{auth_part}{self.host}:{self.port}/{self.db}"

        return url

    def get_connection_kwargs(self) -> dict[str, Any]:
        """Generate Redis connection keyword arguments."""
        kwargs = {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "encoding": self.encoding,
            "decode_responses": self.decode_responses,
            "socket_timeout": self.timeouts.socket_timeout,
            "socket_connect_timeout": self.timeouts.connection_timeout,
            "socket_keepalive": self.pool.socket_keepalive,
            "socket_keepalive_options": self.pool.socket_keepalive_options,
            "retry_on_timeout": self.pool.retry_on_timeout,
            "health_check_interval": self.pool.health_check_interval,
            "max_connections": self.pool.max_connections,
        }

        # Add authentication
        if self.username:
            kwargs["username"] = self.username
        if self.password:
            kwargs["password"] = self.password

        # Add SSL settings
        if self.ssl.enabled:
            kwargs["ssl"] = True
            if self.ssl.cert_file:
                kwargs["ssl_certfile"] = self.ssl.cert_file
            if self.ssl.key_file:
                kwargs["ssl_keyfile"] = self.ssl.key_file
            if self.ssl.ca_certs:
                kwargs["ssl_ca_certs"] = self.ssl.ca_certs
            kwargs["ssl_check_hostname"] = self.ssl.check_hostname

        return kwargs

    def create_connection_pool(self) -> redis.ConnectionPool:
        """Create Redis connection pool."""
        kwargs = self.get_connection_kwargs()

        if self.mode == RedisConnectionMode.CLUSTER:
            # Redis cluster not implemented in this example
            raise NotImplementedError("Redis cluster mode not yet implemented")
        elif self.mode == RedisConnectionMode.SENTINEL:
            # Redis sentinel not implemented in this example
            raise NotImplementedError("Redis sentinel mode not yet implemented")
        else:
            # Standalone mode
            return redis.ConnectionPool(**kwargs)

    def create_client(self) -> redis.Redis:
        """Create Redis client with full configuration."""
        pool = self.create_connection_pool()
        return redis.Redis(connection_pool=pool)

    def test_connection(self) -> bool:
        """Test Redis connection."""
        try:
            client = self.create_client()
            client.ping()
            logger.info(f"Redis connection successful: {self.host}:{self.port}")
            return True
        except (ConnectionError, TimeoutError, RedisError) as e:
            logger.warning(f"Redis connection failed: {e}")
            return False

    @classmethod
    def from_environment(cls) -> "RedisConfig":
        """Create Redis configuration from environment variables."""
        # Extract environment variables with REDIS_ prefix
        env_config = {}

        # Basic connection settings
        if redis_url := os.getenv("REDIS_URL"):
            # Parse Redis URL
            import urllib.parse

            parsed = urllib.parse.urlparse(redis_url)
            env_config.update(
                {
                    "host": parsed.hostname or "localhost",
                    "port": parsed.port or 6380,
                    "db": int(parsed.path.lstrip("/")) if parsed.path and parsed.path != "/" else 0,
                    "username": parsed.username,
                    "password": parsed.password,
                }
            )
        else:
            # Individual settings
            env_config.update(
                {
                    "host": os.getenv("REDIS_HOST", "localhost"),
                    "port": int(os.getenv("REDIS_PORT", "6380")),
                    "db": int(os.getenv("REDIS_DB", "0")),
                    "username": os.getenv("REDIS_USERNAME"),
                    "password": os.getenv("REDIS_PASSWORD"),
                }
            )

        # SSL settings
        ssl_config = {}
        if os.getenv("REDIS_SSL_ENABLED", "").lower() in ("true", "1", "yes"):
            ssl_config.update(
                {
                    "enabled": True,
                    "cert_file": os.getenv("REDIS_SSL_CERT_FILE"),
                    "key_file": os.getenv("REDIS_SSL_KEY_FILE"),
                    "ca_certs": os.getenv("REDIS_SSL_CA_CERTS"),
                    "verify_mode": os.getenv("REDIS_SSL_VERIFY_MODE", "required"),
                    "check_hostname": os.getenv("REDIS_SSL_CHECK_HOSTNAME", "true").lower()
                    in ("true", "1", "yes"),
                }
            )

        # Pool settings
        pool_config = {}
        if max_conn := os.getenv("REDIS_MAX_CONNECTIONS"):
            pool_config["max_connections"] = int(max_conn)
        if min_conn := os.getenv("REDIS_MIN_CONNECTIONS"):
            pool_config["min_connections"] = int(min_conn)

        # Timeout settings
        timeout_config = {}
        if conn_timeout := os.getenv("REDIS_CONNECTION_TIMEOUT"):
            timeout_config["connection_timeout"] = float(conn_timeout)
        if socket_timeout := os.getenv("REDIS_SOCKET_TIMEOUT"):
            timeout_config["socket_timeout"] = float(socket_timeout)

        # Retry settings
        retry_config = {}
        if max_retries := os.getenv("REDIS_MAX_RETRIES"):
            retry_config["max_retries"] = int(max_retries)
        if retry_delay := os.getenv("REDIS_RETRY_DELAY"):
            retry_config["retry_delay"] = float(retry_delay)

        # Fallback settings
        fallback_config = {}
        if os.getenv("REDIS_FALLBACK_ENABLED", "").lower() in ("false", "0", "no"):
            fallback_config["enabled"] = False

        # Build nested configuration
        config_dict = env_config.copy()
        if ssl_config:
            config_dict["ssl"] = ssl_config
        if pool_config:
            config_dict["pool"] = pool_config
        if timeout_config:
            config_dict["timeouts"] = timeout_config
        if retry_config:
            config_dict["retry"] = retry_config
        if fallback_config:
            config_dict["fallback"] = fallback_config

        # Additional settings
        if key_prefix := os.getenv("REDIS_KEY_PREFIX"):
            config_dict["key_prefix"] = key_prefix
        if encoding := os.getenv("REDIS_ENCODING"):
            config_dict["encoding"] = encoding

        return cls(**config_dict)

    @classmethod
    def get_default_windows_config(cls) -> "RedisConfig":
        """Get default configuration optimized for Windows."""
        return cls(
            host="localhost",
            port=6380,  # Windows Redis default
            pool=RedisPoolConfig(
                max_connections=10, health_check_interval=60, socket_keepalive=True
            ),
            timeouts=RedisTimeoutConfig(connection_timeout=10.0, socket_timeout=15.0),
            fallback=RedisFallbackConfig(enabled=True, warn_on_fallback=True),
        )

    @classmethod
    def get_default_docker_config(cls) -> "RedisConfig":
        """Get default configuration optimized for Docker."""
        return cls(
            host="redis",  # Docker service name
            port=6379,  # Standard Redis port in containers
            pool=RedisPoolConfig(max_connections=20, health_check_interval=30),
            timeouts=RedisTimeoutConfig(connection_timeout=5.0, socket_timeout=10.0),
            fallback=RedisFallbackConfig(enabled=False),  # Docker should have reliable Redis
        )


def create_redis_config() -> RedisConfig:
    """Create Redis configuration with environment detection."""
    # Check if we're in Docker
    if os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER"):
        logger.info("Docker environment detected, using Docker Redis configuration")
        base_config = RedisConfig.get_default_docker_config()
    # Check if we're on Windows
    elif os.name == "nt":
        logger.info("Windows environment detected, using Windows Redis configuration")
        base_config = RedisConfig.get_default_windows_config()
    else:
        # Linux/Unix defaults
        logger.info("Unix environment detected, using standard Redis configuration")
        base_config = RedisConfig()

    # Override with environment variables
    env_config = RedisConfig.from_environment()

    # Merge configurations (environment takes precedence)
    merged_config = base_config.model_copy()
    for field_name in env_config.model_fields:
        env_value = getattr(env_config, field_name)
        if env_value != getattr(RedisConfig(), field_name):  # Not default value
            setattr(merged_config, field_name, env_value)

    return merged_config


def get_redis_client(config: RedisConfig | None = None) -> redis.Redis:
    """Get configured Redis client with fallback support."""
    if config is None:
        config = create_redis_config()

    # Test connection first
    if config.test_connection():
        return config.create_client()

    # Handle fallback
    if config.fallback.enabled:
        if config.fallback.warn_on_fallback:
            warnings.warn(
                f"Redis connection failed, falling back to {config.fallback.fallback_type} cache",
                UserWarning,
                stacklevel=2,
            )

        # Return a mock Redis client or in-memory cache
        # This would need to be implemented based on requirements
        logger.warning("Redis fallback not fully implemented, returning None")
        return None
    else:
        raise ConnectionError(f"Cannot connect to Redis at {config.host}:{config.port}")


# Global configuration instance
_redis_config: RedisConfig | None = None


def get_redis_config() -> RedisConfig:
    """Get the global Redis configuration instance."""
    global _redis_config
    if _redis_config is None:
        _redis_config = create_redis_config()
    return _redis_config


def reload_redis_config() -> RedisConfig:
    """Reload Redis configuration from environment."""
    global _redis_config
    _redis_config = create_redis_config()
    return _redis_config
