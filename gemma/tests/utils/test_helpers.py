"""Test utilities and helper functions.

This module provides reusable utilities for testing the Gemma CLI application.
Functions are organized by category:
- File operations: Create mock models, tokenizers, configs
- Validation: Assert functions for responses and data structures
- Subprocess: Mock subprocess calls and process management
- Data generation: Create realistic test data
"""

import json
import os
import random
import string
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import Mock, patch

import pytest


# =============================================================================
# File Creation Helpers
# =============================================================================


def create_mock_model_file(
    path: Union[str, Path],
    size_mb: int = 100,
    header: Optional[bytes] = None,
) -> Path:
    """
    Create a mock model file (.sbs) for testing.

    Args:
        path: Path where the mock model should be created
        size_mb: Size of the model file in megabytes
        header: Optional header bytes to write at the beginning

    Returns:
        Path to the created model file

    Example:
        model_path = create_mock_model_file("test_model.sbs", size_mb=50)
        assert model_path.stat().st_size == 50 * 1024 * 1024
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure .sbs extension
    if path.suffix != ".sbs":
        path = path.with_suffix(".sbs")

    # Create file with specified size
    size_bytes = size_mb * 1024 * 1024

    with open(path, "wb") as f:
        # Write header if provided
        if header:
            f.write(header)
            size_bytes -= len(header)

        # Write zeros for the rest
        chunk_size = 1024 * 1024  # 1MB chunks
        remaining = size_bytes
        while remaining > 0:
            write_size = min(chunk_size, remaining)
            f.write(b"\x00" * write_size)
            remaining -= write_size

    return path


def create_mock_tokenizer(
    path: Union[str, Path],
    vocab_size: int = 32000,
    include_special_tokens: bool = True,
) -> Path:
    """
    Create a mock tokenizer file (.spm) for testing.

    Args:
        path: Path where the mock tokenizer should be created
        vocab_size: Number of tokens in vocabulary
        include_special_tokens: Whether to include special tokens

    Returns:
        Path to the created tokenizer file

    Example:
        tokenizer_path = create_mock_tokenizer("tokenizer.spm", vocab_size=1000)
        assert tokenizer_path.exists()
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure .spm extension
    if path.suffix != ".spm":
        path = path.with_suffix(".spm")

    # Create mock sentencepiece model data
    # This is a simplified representation - real .spm files use protobuf
    mock_data = b"sentencepiece_model_v1\n"

    if include_special_tokens:
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
        for token in special_tokens:
            mock_data += f"{token}\n".encode("utf-8")

    # Add mock vocabulary
    for i in range(vocab_size):
        mock_data += f"token_{i}\n".encode("utf-8")

    path.write_bytes(mock_data)
    return path


def create_mock_config_file(
    path: Union[str, Path],
    config_data: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Create a mock configuration file (TOML or JSON).

    Args:
        path: Path where config should be created
        config_data: Configuration data to write

    Returns:
        Path to the created config file

    Example:
        config = {"gemma": {"model": "test.sbs"}}
        config_path = create_mock_config_file("config.toml", config)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if config_data is None:
        config_data = get_default_test_config()

    # Write based on file extension
    if path.suffix == ".toml":
        import toml
        with open(path, "w", encoding="utf-8") as f:
            toml.dump(config_data, f)
    elif path.suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    return path


def create_complete_test_environment(base_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    Create a complete test environment with models, configs, etc.

    Args:
        base_dir: Base directory for test environment

    Returns:
        Dictionary mapping component names to their paths

    Example:
        env = create_complete_test_environment("/tmp/test_env")
        assert env["model"].exists()
        assert env["tokenizer"].exists()
        assert env["config"].exists()
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    env = {}

    # Create models directory
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Create model files
    env["model"] = create_mock_model_file(models_dir / "gemma-2b-it.sbs", size_mb=50)
    env["tokenizer"] = create_mock_tokenizer(models_dir / "tokenizer.spm")

    # Create config directory
    config_dir = base_dir / "config"
    config_dir.mkdir(exist_ok=True)
    env["config"] = create_mock_config_file(
        config_dir / "config.toml",
        {
            "gemma": {
                "default_model": str(env["model"]),
                "default_tokenizer": str(env["tokenizer"]),
            }
        },
    )

    # Create cache directory
    env["cache"] = base_dir / "cache"
    env["cache"].mkdir(exist_ok=True)

    # Create logs directory
    env["logs"] = base_dir / "logs"
    env["logs"].mkdir(exist_ok=True)

    return env


# =============================================================================
# Validation Helpers
# =============================================================================


def assert_valid_response(
    response: str,
    min_length: int = 10,
    max_length: int = 10000,
    required_keywords: Optional[List[str]] = None,
    forbidden_keywords: Optional[List[str]] = None,
) -> None:
    """
    Assert that a model response is valid.

    Args:
        response: Response text to validate
        min_length: Minimum acceptable length
        max_length: Maximum acceptable length
        required_keywords: Keywords that must appear in response
        forbidden_keywords: Keywords that must not appear in response

    Raises:
        AssertionError: If validation fails

    Example:
        assert_valid_response(
            response="The capital is Paris.",
            min_length=5,
            required_keywords=["Paris"]
        )
    """
    # Check type
    assert isinstance(response, str), f"Response must be string, got {type(response)}"

    # Check length
    assert (
        min_length <= len(response) <= max_length
    ), f"Response length {len(response)} outside range [{min_length}, {max_length}]"

    # Check not empty or whitespace only
    assert response.strip(), "Response is empty or whitespace-only"

    # Check required keywords
    if required_keywords:
        for keyword in required_keywords:
            assert (
                keyword.lower() in response.lower()
            ), f"Required keyword '{keyword}' not found in response"

    # Check forbidden keywords
    if forbidden_keywords:
        for keyword in forbidden_keywords:
            assert (
                keyword.lower() not in response.lower()
            ), f"Forbidden keyword '{keyword}' found in response"


def assert_valid_config(config: Dict[str, Any], required_sections: Optional[List[str]] = None) -> None:
    """
    Assert that a configuration dictionary is valid.

    Args:
        config: Configuration dictionary to validate
        required_sections: List of required top-level sections

    Raises:
        AssertionError: If validation fails

    Example:
        config = {"gemma": {...}, "redis": {...}}
        assert_valid_config(config, required_sections=["gemma", "redis"])
    """
    assert isinstance(config, dict), f"Config must be dict, got {type(config)}"
    assert config, "Config is empty"

    if required_sections:
        for section in required_sections:
            assert section in config, f"Required section '{section}' not found in config"


def assert_valid_model_path(path: Union[str, Path], must_exist: bool = False) -> None:
    """
    Assert that a model path is valid.

    Args:
        path: Path to validate
        must_exist: Whether the file must actually exist

    Raises:
        AssertionError: If validation fails

    Example:
        assert_valid_model_path("model.sbs", must_exist=True)
    """
    path = Path(path)

    # Check extension
    assert path.suffix == ".sbs", f"Model must have .sbs extension, got {path.suffix}"

    # Check existence if required
    if must_exist:
        assert path.exists(), f"Model file does not exist: {path}"
        assert path.is_file(), f"Model path is not a file: {path}"
        assert path.stat().st_size > 0, f"Model file is empty: {path}"


def assert_valid_prompt(prompt: str, max_length: int = 50000) -> None:
    """
    Assert that a prompt is valid and safe.

    Args:
        prompt: Prompt text to validate
        max_length: Maximum allowed length

    Raises:
        AssertionError: If validation fails

    Example:
        assert_valid_prompt("Tell me about Python")
    """
    assert isinstance(prompt, str), f"Prompt must be string, got {type(prompt)}"
    assert prompt.strip(), "Prompt is empty or whitespace-only"
    assert len(prompt) <= max_length, f"Prompt exceeds max length: {len(prompt)} > {max_length}"

    # Check for forbidden characters
    forbidden_chars = ["\x00", "\x1b"]
    for char in forbidden_chars:
        assert char not in prompt, f"Prompt contains forbidden character: {repr(char)}"


# =============================================================================
# Subprocess Mocking
# =============================================================================


def mock_subprocess_call(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
    pid: int = 12345,
) -> Mock:
    """
    Create a mock subprocess.Popen object.

    Args:
        stdout: Standard output to return
        stderr: Standard error to return
        returncode: Process return code
        pid: Process ID

    Returns:
        Mock Popen object configured with specified behavior

    Example:
        with patch("subprocess.Popen", mock_subprocess_call(stdout="output")):
            # Your subprocess code here
            pass
    """
    mock_process = Mock(spec=subprocess.Popen)
    mock_process.communicate.return_value = (stdout.encode(), stderr.encode())
    mock_process.returncode = returncode
    mock_process.pid = pid
    mock_process.poll.return_value = returncode
    return mock_process


def mock_gemma_executable(
    response: str = "This is a test response.",
    delay_seconds: float = 0.0,
) -> Mock:
    """
    Create a mock for the Gemma executable subprocess.

    Args:
        response: Text response to return
        delay_seconds: Simulated processing delay

    Returns:
        Mock configured to simulate Gemma inference

    Example:
        mock = mock_gemma_executable(response="Hello!")
        # Use mock in your tests
    """
    import time

    def communicate_with_delay(*args, **kwargs):
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        return (response.encode(), b"")

    mock = Mock(spec=subprocess.Popen)
    mock.communicate = communicate_with_delay
    mock.returncode = 0
    mock.pid = random.randint(1000, 9999)
    return mock


# =============================================================================
# Data Generation
# =============================================================================


def generate_random_prompt(
    min_words: int = 5,
    max_words: int = 50,
    topic: Optional[str] = None,
) -> str:
    """
    Generate a random prompt for testing.

    Args:
        min_words: Minimum number of words
        max_words: Maximum number of words
        topic: Optional topic to focus on

    Returns:
        Random prompt string

    Example:
        prompt = generate_random_prompt(min_words=10, topic="Python")
    """
    words = [
        "explain",
        "describe",
        "what",
        "how",
        "why",
        "tell",
        "me",
        "about",
        "the",
        "is",
        "are",
        "can",
        "you",
        "please",
        "help",
    ]

    if topic:
        words.extend(topic.split())

    num_words = random.randint(min_words, max_words)
    prompt_words = [random.choice(words) for _ in range(num_words)]

    # Capitalize first word
    prompt_words[0] = prompt_words[0].capitalize()

    # Add punctuation
    prompt = " ".join(prompt_words)
    if not prompt.endswith(("?", ".", "!")):
        prompt += "?"

    return prompt


def generate_random_response(
    min_words: int = 20,
    max_words: int = 200,
) -> str:
    """
    Generate a random response for testing.

    Args:
        min_words: Minimum number of words
        max_words: Maximum number of words

    Returns:
        Random response string

    Example:
        response = generate_random_response(min_words=50)
    """
    sentences = [
        "This is an interesting topic.",
        "Let me explain further.",
        "There are several key points to consider.",
        "First, we should understand the basics.",
        "Additionally, it's important to note that.",
        "In conclusion, this demonstrates that.",
        "The evidence suggests that.",
        "Research has shown that.",
    ]

    num_words = random.randint(min_words, max_words)
    response_text = []
    current_words = 0

    while current_words < num_words:
        sentence = random.choice(sentences)
        response_text.append(sentence)
        current_words += len(sentence.split())

    return " ".join(response_text)


def get_default_test_config() -> Dict[str, Any]:
    """
    Get default configuration for testing.

    Returns:
        Dictionary with standard test configuration

    Example:
        config = get_default_test_config()
        assert "gemma" in config
    """
    return {
        "gemma": {
            "default_model": "C:/test/models/gemma-2b-it.sbs",
            "default_tokenizer": "C:/test/models/tokenizer.spm",
            "executable": "C:/test/gemma.exe",
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "pool_size": 10,
        },
        "memory": {
            "working_ttl": 900,
            "short_term_ttl": 3600,
            "long_term_ttl": 2592000,
        },
        "embedding": {
            "provider": "local",
            "model": "all-MiniLM-L6-v2",
            "dimension": 384,
        },
        "ui": {
            "theme": "default",
            "show_timestamps": True,
            "syntax_highlighting": True,
        },
    }


# =============================================================================
# Context Managers
# =============================================================================


class MockEnvironment:
    """
    Context manager for temporarily mocking environment variables.

    Example:
        with MockEnvironment(REDIS_HOST="localhost", REDIS_PORT="6379"):
            # Code that uses environment variables
            pass
    """

    def __init__(self, **env_vars):
        self.env_vars = env_vars
        self.original_env = {}

    def __enter__(self):
        for key, value in self.env_vars.items():
            self.original_env[key] = os.environ.get(key)
            os.environ[key] = str(value)
        return self

    def __exit__(self, *args):
        for key, original_value in self.original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


class TemporaryDirectory:
    """
    Context manager for temporary directory with automatic cleanup.

    Example:
        with TemporaryDirectory() as tmpdir:
            test_file = tmpdir / "test.txt"
            test_file.write_text("test")
    """

    def __init__(self, prefix: str = "gemma_test_"):
        self.prefix = prefix
        self.path = None

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix=self.prefix))
        return self.path

    def __exit__(self, *args):
        if self.path and self.path.exists():
            import shutil
            shutil.rmtree(self.path, ignore_errors=True)


# =============================================================================
# Pytest Markers
# =============================================================================


def requires_redis(func):
    """
    Decorator to skip tests if Redis is not available.

    Example:
        @requires_redis
        async def test_redis_feature(mock_redis):
            # Test code that requires Redis
            pass
    """
    return pytest.mark.skipif(
        not _is_redis_available(),
        reason="Redis server not available",
    )(func)


def requires_model_files(func):
    """
    Decorator to skip tests if model files are not available.

    Example:
        @requires_model_files
        def test_inference():
            # Test code that requires actual model files
            pass
    """
    return pytest.mark.skipif(
        not Path("C:/codedev/llm/.models").exists(),
        reason="Model files not available",
    )(func)


def _is_redis_available() -> bool:
    """Check if Redis is available for testing."""
    try:
        import redis
        client = redis.Redis(host="localhost", port=6379, socket_timeout=1)
        client.ping()
        return True
    except Exception:
        return False


# =============================================================================
# Comparison Helpers
# =============================================================================


def compare_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> List[str]:
    """
    Compare two configuration dictionaries and return differences.

    Args:
        config1: First configuration
        config2: Second configuration

    Returns:
        List of difference descriptions (empty if identical)

    Example:
        diffs = compare_configs(config1, config2)
        assert len(diffs) == 0, f"Configs differ: {diffs}"
    """
    differences = []

    def _compare_recursive(d1, d2, path=""):
        if type(d1) != type(d2):
            differences.append(f"{path}: type mismatch ({type(d1)} vs {type(d2)})")
            return

        if isinstance(d1, dict):
            all_keys = set(d1.keys()) | set(d2.keys())
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key
                if key not in d1:
                    differences.append(f"{new_path}: missing in first config")
                elif key not in d2:
                    differences.append(f"{new_path}: missing in second config")
                else:
                    _compare_recursive(d1[key], d2[key], new_path)
        elif d1 != d2:
            differences.append(f"{path}: value mismatch ({d1} vs {d2})")

    _compare_recursive(config1, config2)
    return differences
