"""Shared pytest fixtures for Gemma CLI tests.

This module provides common fixtures for testing the Gemma CLI application,
including mock interfaces, Redis connections, temporary directories, and
configuration data.

Fixtures are organized by category:
- Interface mocks: Mock Gemma inference and subprocess calls
- Database mocks: Mock Redis connections and clients
- Configuration: Temporary config directories and sample configs
- File system: Model files, tokenizers, and temporary paths
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio
from redis.asyncio import Redis

# Import Gemma CLI components for type hints
try:
    from gemma_cli.config.models import ModelPreset, PerformanceProfile
    from gemma_cli.config.settings import (
        EmbeddingConfig,
        GemmaConfig,
        MemoryConfig,
        RedisConfig,
        Settings,
    )
    from gemma_cli.core.gemma import GemmaInterface
except ImportError as e:
    # Fallback types if imports fail
    print(f"Warning: Failed to import Gemma CLI types: {e}")
    ModelPreset = type("ModelPreset", (), {})  # type: ignore
    PerformanceProfile = type("PerformanceProfile", (), {})  # type: ignore
    EmbeddingConfig = type("EmbeddingConfig", (), {})  # type: ignore
    GemmaConfig = type("GemmaConfig", (), {})  # type: ignore
    MemoryConfig = type("MemoryConfig", (), {})  # type: ignore
    RedisConfig = type("RedisConfig", (), {})  # type: ignore
    Settings = type("Settings", (), {})  # type: ignore
    GemmaInterface = type("GemmaInterface", (), {})  # type: ignore


# =============================================================================
# Session-scoped fixtures (run once per test session)
# =============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """
    Create an event loop for the entire test session.

    This fixture ensures async tests work properly by providing
    a single event loop that persists across all tests.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Interface Mocks
# =============================================================================


@pytest_asyncio.fixture
async def mock_gemma_interface() -> AsyncMock:
    """
    Mock GemmaInterface for testing without actual model inference.

    Returns:
        AsyncMock configured with typical GemmaInterface behavior:
        - generate_response: Returns sample text response
        - generate_stream: Yields text chunks
        - is_running: Returns True
        - shutdown: Completes without error

    Example:
        async def test_generation(mock_gemma_interface):
            response = await mock_gemma_interface.generate_response("test")
            assert "sample response" in response
    """
    mock = AsyncMock(spec=GemmaInterface)

    # Configure generate_response
    mock.generate_response.return_value = "This is a sample response from the model."

    # Configure generate_stream to yield chunks
    async def mock_stream(*args, **kwargs):
        chunks = ["This ", "is ", "a ", "streaming ", "response."]
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.01)  # Simulate processing time

    mock.generate_stream.return_value = mock_stream()

    # Configure status methods
    mock.is_running.return_value = True
    mock.shutdown.return_value = None

    # Set properties
    mock.model_path = "C:/test/models/gemma-2b-it.sbs"
    mock.tokenizer_path = "C:/test/models/tokenizer.spm"
    mock.max_tokens = 2048
    mock.temperature = 0.7

    return mock


@pytest.fixture
def mock_subprocess_call() -> Generator[Mock, None, None]:
    """
    Mock subprocess calls to prevent actual process execution.

    Returns:
        Mock object that can be used to verify subprocess calls

    Example:
        def test_subprocess(mock_subprocess_call):
            # Your test code that calls subprocess
            mock_subprocess_call.assert_called_once()
    """
    with patch("subprocess.Popen") as mock_popen:
        mock_process = Mock()
        mock_process.communicate.return_value = (b"stdout output", b"")
        mock_process.returncode = 0
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        yield mock_popen


# =============================================================================
# Database Mocks
# =============================================================================


@pytest_asyncio.fixture
async def mock_redis() -> AsyncGenerator[AsyncMock, None]:
    """
    Mock Redis client for testing RAG and memory features.

    Returns:
        AsyncMock configured with typical Redis operations:
        - get/set/delete: Basic key-value operations
        - hget/hset: Hash operations
        - zadd/zrange: Sorted set operations
        - ping: Health check

    Example:
        async def test_redis(mock_redis):
            await mock_redis.set("key", "value")
            result = await mock_redis.get("key")
            assert result == "value"
    """
    mock = AsyncMock(spec=Redis)

    # In-memory storage for testing
    storage: Dict[str, Any] = {}
    hash_storage: Dict[str, Dict[str, Any]] = {}
    sorted_sets: Dict[str, List[tuple]] = {}

    # Configure basic operations
    async def mock_get(key: str) -> Optional[str]:
        return storage.get(key)

    async def mock_set(key: str, value: Any, **kwargs) -> bool:
        storage[key] = value
        return True

    async def mock_delete(*keys: str) -> int:
        count = 0
        for key in keys:
            if key in storage:
                del storage[key]
                count += 1
        return count

    # Configure hash operations
    async def mock_hget(name: str, key: str) -> Optional[Any]:
        return hash_storage.get(name, {}).get(key)

    async def mock_hset(name: str, key: str, value: Any) -> int:
        if name not in hash_storage:
            hash_storage[name] = {}
        hash_storage[name][key] = value
        return 1

    async def mock_hgetall(name: str) -> Dict[str, Any]:
        return hash_storage.get(name, {})

    # Configure sorted set operations
    async def mock_zadd(name: str, mapping: Dict[str, float]) -> int:
        if name not in sorted_sets:
            sorted_sets[name] = []
        for member, score in mapping.items():
            sorted_sets[name].append((member, score))
        sorted_sets[name].sort(key=lambda x: x[1])
        return len(mapping)

    async def mock_zrange(name: str, start: int, end: int, withscores: bool = False) -> List:
        items = sorted_sets.get(name, [])
        result = items[start:end + 1 if end >= 0 else None]
        if withscores:
            return result
        return [item[0] for item in result]

    # Configure health check
    async def mock_ping() -> bool:
        return True

    # Assign mock implementations
    mock.get.side_effect = mock_get
    mock.set.side_effect = mock_set
    mock.delete.side_effect = mock_delete
    mock.hget.side_effect = mock_hget
    mock.hset.side_effect = mock_hset
    mock.hgetall.side_effect = mock_hgetall
    mock.zadd.side_effect = mock_zadd
    mock.zrange.side_effect = mock_zrange
    mock.ping.side_effect = mock_ping

    # Configure connection methods
    mock.close.return_value = None

    yield mock

    # Cleanup
    storage.clear()
    hash_storage.clear()
    sorted_sets.clear()


@pytest.fixture
def mock_redis_unavailable() -> Mock:
    """
    Mock Redis client that simulates connection failure.

    Returns:
        Mock that raises ConnectionError on operations

    Example:
        def test_redis_fallback(mock_redis_unavailable):
            with pytest.raises(ConnectionError):
                await mock_redis_unavailable.ping()
    """
    mock = Mock(spec=Redis)
    mock.ping.side_effect = ConnectionError("Redis connection refused")
    mock.get.side_effect = ConnectionError("Redis connection refused")
    mock.set.side_effect = ConnectionError("Redis connection refused")
    return mock


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Create a temporary directory for configuration files.

    Returns:
        Path to temporary directory (automatically cleaned up)

    Example:
        def test_config(temp_config_dir):
            config_file = temp_config_dir / "config.toml"
            config_file.write_text("[section]\\nkey = 'value'")
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    yield config_dir
    # Cleanup handled by tmp_path fixture


@pytest.fixture
def sample_model_preset() -> ModelPreset:
    """
    Create a sample ModelPreset for testing.

    Returns:
        ModelPreset with typical configuration values

    Example:
        def test_preset(sample_model_preset):
            assert sample_model_preset.name == "gemma-2b-fast"
    """
    return ModelPreset(
        name="gemma-2b-fast",
        weights="C:/test/models/gemma-2b-it.sbs",
        tokenizer="C:/test/models/tokenizer.spm",
        format="sfp",
        size_gb=2.5,
        avg_tokens_per_sec=50,
        quality="fast",
        use_case="development",
        context_length=2048,
        min_ram_gb=4,
    )


@pytest.fixture
def sample_performance_profile() -> PerformanceProfile:
    """
    Create a sample PerformanceProfile for testing.

    Returns:
        PerformanceProfile with balanced settings

    Example:
        def test_profile(sample_performance_profile):
            assert sample_performance_profile.name == "balanced"
            assert sample_performance_profile.batch_size == 32
    """
    return PerformanceProfile(
        name="balanced",
        description="Balanced performance and quality",
        batch_size=32,
        cache_size=1000,
        max_concurrent_requests=10,
        timeout_seconds=30.0,
        enable_caching=True,
        enable_batching=True,
    )


@pytest.fixture
def sample_settings(
    temp_config_dir: Path,
    sample_model_preset: ModelPreset,
    sample_performance_profile: PerformanceProfile,
) -> Settings:
    """
    Create a complete Settings object for testing.

    Returns:
        Settings with all subsystems configured

    Example:
        def test_settings(sample_settings):
            assert sample_settings.gemma.default_model.endswith(".sbs")
            assert sample_settings.redis.host == "localhost"
    """
    return Settings(
        gemma=GemmaConfig(
            default_model="C:/test/models/gemma-2b-it.sbs",
            default_tokenizer="C:/test/models/tokenizer.spm",
            executable="C:/test/gemma.exe",
        ),
        redis=RedisConfig(
            host="localhost",
            port=6379,
            db=0,
            pool_size=10,
        ),
        memory=MemoryConfig(
            working_ttl=900,
            short_term_ttl=3600,
            long_term_ttl=2592000,
        ),
        embedding=EmbeddingConfig(
            provider="local",
            model="all-MiniLM-L6-v2",
            dimension=384,
        ),
    )


# =============================================================================
# File System Fixtures
# =============================================================================


@pytest.fixture
def mock_model_file(tmp_path: Path) -> Path:
    """
    Create a mock model file (.sbs) for testing.

    Returns:
        Path to temporary mock model file

    Example:
        def test_model_loading(mock_model_file):
            assert mock_model_file.exists()
            assert mock_model_file.suffix == ".sbs"
    """
    model_file = tmp_path / "test-model.sbs"
    # Create a minimal mock file (1MB)
    model_file.write_bytes(b"\x00" * (1024 * 1024))
    return model_file


@pytest.fixture
def mock_tokenizer_file(tmp_path: Path) -> Path:
    """
    Create a mock tokenizer file (.spm) for testing.

    Returns:
        Path to temporary mock tokenizer file

    Example:
        def test_tokenizer(mock_tokenizer_file):
            assert mock_tokenizer_file.exists()
            assert mock_tokenizer_file.suffix == ".spm"
    """
    tokenizer_file = tmp_path / "tokenizer.spm"
    # Create a minimal mock file
    tokenizer_file.write_bytes(b"mock tokenizer data")
    return tokenizer_file


@pytest.fixture
def mock_model_directory(
    tmp_path: Path, mock_model_file: Path, mock_tokenizer_file: Path
) -> Path:
    """
    Create a complete mock model directory structure.

    Returns:
        Path to directory containing model and tokenizer files

    Example:
        def test_model_directory(mock_model_directory):
            models = list(mock_model_directory.glob("*.sbs"))
            assert len(models) == 1
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Copy mock files to model directory
    import shutil
    shutil.copy(mock_model_file, model_dir / "model.sbs")
    shutil.copy(mock_tokenizer_file, model_dir / "tokenizer.spm")

    return model_dir


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_prompts() -> List[str]:
    """
    Provide sample prompts for testing generation.

    Returns:
        List of diverse test prompts
    """
    return [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate factorial.",
        "Translate 'Hello, world!' to Spanish.",
        "What are the benefits of exercise?",
    ]


@pytest.fixture
def sample_responses() -> List[str]:
    """
    Provide sample model responses for testing.

    Returns:
        List of realistic model responses
    """
    return [
        "The capital of France is Paris, which is located in the north-central part of the country.",
        "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to perform computations.",
        "Here's a factorial function:\n\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        "In Spanish, 'Hello, world!' is '¡Hola, mundo!'",
        "Regular exercise improves cardiovascular health, strengthens muscles, and enhances mental well-being.",
    ]


@pytest.fixture
def sample_conversation_history() -> List[Dict[str, str]]:
    """
    Provide sample conversation history for testing context.

    Returns:
        List of conversation turns with role and content
    """
    return [
        {"role": "user", "content": "Hello! Can you help me?"},
        {"role": "assistant", "content": "Of course! I'm here to help. What do you need?"},
        {"role": "user", "content": "I need to understand Python decorators."},
        {
            "role": "assistant",
            "content": "Python decorators are functions that modify the behavior of other functions.",
        },
    ]


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def capture_logs(caplog):
    """
    Capture log output for testing logging behavior.

    Example:
        def test_logging(capture_logs):
            logger.info("test message")
            assert "test message" in capture_logs.text
    """
    import logging

    caplog.set_level(logging.DEBUG)
    return caplog


@pytest.fixture(autouse=True)
def reset_environment():
    """
    Reset environment variables before each test.

    This fixture automatically runs before each test to ensure
    a clean environment state.
    """
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


# =============================================================================
# Performance Testing Fixtures
# =============================================================================


@pytest.fixture
def benchmark_timer():
    """
    Simple timer for performance testing.

    Returns:
        Context manager that measures execution time

    Example:
        def test_performance(benchmark_timer):
            with benchmark_timer as timer:
                # Code to benchmark
                pass
            assert timer.elapsed < 1.0  # Should complete in 1 second
    """
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start_time

    return Timer()
