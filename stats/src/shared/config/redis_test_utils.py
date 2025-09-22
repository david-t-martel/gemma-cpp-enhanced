"""Redis test utilities for consistent test configuration across the project."""

import os
import warnings
from typing import Any

import pytest

try:
    import redis
    from redis.exceptions import ConnectionError
    from redis.exceptions import TimeoutError
except ImportError:
    redis = None


def get_test_redis_config() -> dict[str, Any]:
    """Get Redis configuration optimized for testing."""
    # Use environment variables or safe test defaults
    config = {
        "host": os.getenv("REDIS_TEST_HOST", "localhost"),
        "port": int(os.getenv("REDIS_TEST_PORT", "6380")),  # Default to Windows port
        "db": int(os.getenv("REDIS_TEST_DB", "1")),  # Use DB 1 for tests to avoid conflicts
        "decode_responses": True,
        "socket_timeout": 5.0,
        "socket_connect_timeout": 5.0,
        "retry_on_timeout": True,
        "health_check_interval": 30,
    }

    # Add password if provided
    if password := os.getenv("REDIS_TEST_PASSWORD"):
        config["password"] = password

    return config


def create_test_redis_client() -> redis.Redis | None:
    """Create a Redis client for testing with proper error handling."""
    if redis is None:
        warnings.warn("Redis not available, skipping Redis-dependent tests", stacklevel=2)
        return None

    config = get_test_redis_config()

    try:
        client = redis.Redis(**config)
        client.ping()  # Test connection
        return client
    except (ConnectionError, TimeoutError, Exception) as e:
        warnings.warn(f"Redis connection failed: {e}. Skipping Redis tests.", stacklevel=2)
        return None


def requires_redis():
    """Pytest decorator to skip tests if Redis is not available."""

    def decorator(func):
        return pytest.mark.skipif(
            create_test_redis_client() is None,
            reason="Redis server not available or connection failed",
        )(func)

    return decorator


def cleanup_test_redis(client: redis.Redis, key_pattern: str = "test:*"):
    """Clean up test keys from Redis."""
    if client is None:
        return

    try:
        keys = client.keys(key_pattern)
        if keys:
            client.delete(*keys)
    except Exception as e:
        warnings.warn(f"Failed to cleanup Redis test keys: {e}", stacklevel=2)


class RedisTestFixture:
    """Test fixture for Redis operations."""

    def __init__(self, key_prefix: str = "test:"):
        self.client = create_test_redis_client()
        self.key_prefix = key_prefix
        self.keys_created = set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """Clean up all keys created during testing."""
        if self.client and self.keys_created:
            try:
                self.client.delete(*self.keys_created)
            except Exception as e:
                warnings.warn(f"Failed to cleanup Redis test keys: {e}", stacklevel=2)

    def set(self, key: str, value: Any, **kwargs) -> bool:
        """Set a value in Redis and track the key for cleanup."""
        if self.client is None:
            return False

        full_key = f"{self.key_prefix}{key}"
        self.keys_created.add(full_key)
        return self.client.set(full_key, value, **kwargs)

    def get(self, key: str) -> Any:
        """Get a value from Redis."""
        if self.client is None:
            return None

        full_key = f"{self.key_prefix}{key}"
        return self.client.get(full_key)

    def delete(self, key: str) -> int:
        """Delete a key from Redis."""
        if self.client is None:
            return 0

        full_key = f"{self.key_prefix}{key}"
        self.keys_created.discard(full_key)
        return self.client.delete(full_key)

    def ping(self) -> bool:
        """Test Redis connection."""
        if self.client is None:
            return False

        try:
            return self.client.ping()
        except Exception:
            return False


# Global test settings
TEST_REDIS_CONFIG = {
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6380",  # Windows-friendly default
    "REDIS_DB": "1",  # Test database
    "REDIS_CONNECTION_TIMEOUT": "5",
    "REDIS_SOCKET_TIMEOUT": "5",
    "REDIS_MAX_RETRIES": "3",
    "REDIS_ENABLE_FALLBACK": "true",
}


def setup_test_environment():
    """Setup environment variables for Redis testing."""
    for key, value in TEST_REDIS_CONFIG.items():
        if key not in os.environ:
            os.environ[key] = value


def pytest_configure(config):
    """Pytest hook to configure Redis testing."""
    setup_test_environment()
