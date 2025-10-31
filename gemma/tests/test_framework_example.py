"""Example tests demonstrating the pytest framework.

This file shows how to use the fixtures and utilities provided by
the test framework. Use this as a reference for writing new tests.
"""

import asyncio
from pathlib import Path
from typing import Dict, List

import pytest

from tests.utils.fixtures import (
    SAMPLE_PROMPTS,
    SAMPLE_RESPONSES,
    get_sample_config,
    get_sample_conversation,
)
from tests.utils.test_helpers import (
    assert_valid_config,
    assert_valid_prompt,
    assert_valid_response,
    create_mock_model_file,
    create_mock_tokenizer,
    generate_random_prompt,
)


# =============================================================================
# Unit Tests (fast, isolated)
# =============================================================================


@pytest.mark.unit
def test_sample_config_fixture(sample_model_preset):
    """Test that sample_model_preset fixture works correctly."""
    assert sample_model_preset.name == "gemma-2b-fast"
    assert sample_model_preset.temperature == 0.7
    assert sample_model_preset.max_tokens == 2048


@pytest.mark.unit
def test_sample_performance_profile(sample_performance_profile):
    """Test that sample_performance_profile fixture works correctly."""
    assert sample_performance_profile.name == "balanced"
    assert sample_performance_profile.batch_size == 32
    assert sample_performance_profile.enable_caching is True


@pytest.mark.unit
def test_sample_settings(sample_settings):
    """Test that sample_settings fixture provides complete configuration."""
    assert sample_settings.gemma.default_model.endswith(".sbs")
    assert sample_settings.redis.host == "localhost"
    assert sample_settings.redis.port == 6379
    assert sample_settings.memory.working_ttl == 900


@pytest.mark.unit
def test_temp_config_dir_fixture(temp_config_dir):
    """Test that temp_config_dir fixture creates usable directory."""
    assert temp_config_dir.exists()
    assert temp_config_dir.is_dir()

    # Test writing to the directory
    test_file = temp_config_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()
    assert test_file.read_text() == "test content"


@pytest.mark.unit
def test_sample_prompts_fixture(sample_prompts):
    """Test that sample_prompts fixture provides diverse prompts."""
    assert len(sample_prompts) == 5
    for prompt in sample_prompts:
        assert isinstance(prompt, str)
        assert len(prompt) > 0


@pytest.mark.unit
def test_sample_responses_fixture(sample_responses):
    """Test that sample_responses fixture provides valid responses."""
    assert len(sample_responses) == 5
    for response in sample_responses:
        assert isinstance(response, str)
        assert len(response) > 10  # Should be substantive


# =============================================================================
# Async Tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
async def test_mock_gemma_interface(mock_gemma_interface):
    """Test that mock_gemma_interface works for async operations."""
    # Test generate_response
    response = await mock_gemma_interface.generate_response("test prompt")
    assert isinstance(response, str)
    assert len(response) > 0

    # Verify mock was called
    mock_gemma_interface.generate_response.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_mock_redis(mock_redis):
    """Test that mock_redis provides basic functionality."""
    # Test basic operations
    await mock_redis.set("test_key", "test_value")
    value = await mock_redis.get("test_key")
    assert value == "test_value"

    # Test hash operations
    await mock_redis.hset("test_hash", "field1", "value1")
    hash_value = await mock_redis.hget("test_hash", "field1")
    assert hash_value == "value1"

    # Test sorted sets
    await mock_redis.zadd("test_zset", {"member1": 1.0, "member2": 2.0})
    members = await mock_redis.zrange("test_zset", 0, -1)
    assert len(members) == 2


@pytest.mark.asyncio
@pytest.mark.unit
async def test_mock_redis_unavailable(mock_redis_unavailable):
    """Test that mock_redis_unavailable simulates connection failure."""
    with pytest.raises(ConnectionError):
        await mock_redis_unavailable.ping()


# =============================================================================
# File System Tests
# =============================================================================


@pytest.mark.unit
def test_mock_model_file(mock_model_file):
    """Test that mock_model_file fixture creates valid model file."""
    assert mock_model_file.exists()
    assert mock_model_file.suffix == ".sbs"
    assert mock_model_file.stat().st_size > 0


@pytest.mark.unit
def test_mock_tokenizer_file(mock_tokenizer_file):
    """Test that mock_tokenizer_file fixture creates valid tokenizer."""
    assert mock_tokenizer_file.exists()
    assert mock_tokenizer_file.suffix == ".spm"
    assert mock_tokenizer_file.stat().st_size > 0


@pytest.mark.unit
def test_mock_model_directory(mock_model_directory):
    """Test that mock_model_directory contains all necessary files."""
    assert mock_model_directory.exists()
    assert mock_model_directory.is_dir()

    # Check for model and tokenizer
    models = list(mock_model_directory.glob("*.sbs"))
    tokenizers = list(mock_model_directory.glob("*.spm"))

    assert len(models) == 1
    assert len(tokenizers) == 1


# =============================================================================
# Helper Function Tests
# =============================================================================


@pytest.mark.unit
def test_create_mock_model_file(tmp_path):
    """Test create_mock_model_file helper function."""
    model_path = tmp_path / "test_model.sbs"
    created_path = create_mock_model_file(model_path, size_mb=10)

    assert created_path.exists()
    assert created_path.suffix == ".sbs"
    assert created_path.stat().st_size == 10 * 1024 * 1024


@pytest.mark.unit
def test_create_mock_tokenizer(tmp_path):
    """Test create_mock_tokenizer helper function."""
    tokenizer_path = tmp_path / "test_tokenizer.spm"
    created_path = create_mock_tokenizer(tokenizer_path, vocab_size=1000)

    assert created_path.exists()
    assert created_path.suffix == ".spm"


@pytest.mark.unit
def test_assert_valid_response():
    """Test assert_valid_response validation function."""
    # Valid response should pass
    assert_valid_response("This is a valid response.", min_length=5)

    # Empty response should fail
    with pytest.raises(AssertionError):
        assert_valid_response("", min_length=5)

    # Too short should fail
    with pytest.raises(AssertionError):
        assert_valid_response("Hi", min_length=10)

    # Required keyword check
    assert_valid_response(
        "Paris is the capital of France.",
        required_keywords=["Paris", "France"],
    )

    # Forbidden keyword check
    with pytest.raises(AssertionError):
        assert_valid_response(
            "This contains badword.",
            forbidden_keywords=["badword"],
        )


@pytest.mark.unit
def test_assert_valid_prompt():
    """Test assert_valid_prompt validation function."""
    # Valid prompt should pass
    assert_valid_prompt("What is Python?")

    # Empty prompt should fail
    with pytest.raises(AssertionError):
        assert_valid_prompt("")

    # Too long should fail
    with pytest.raises(AssertionError):
        assert_valid_prompt("x" * 100000, max_length=1000)

    # Forbidden characters should fail
    with pytest.raises(AssertionError):
        assert_valid_prompt("test\x00null")


@pytest.mark.unit
def test_assert_valid_config():
    """Test assert_valid_config validation function."""
    # Valid config should pass
    config = {"gemma": {"model": "test.sbs"}, "redis": {"host": "localhost"}}
    assert_valid_config(config, required_sections=["gemma", "redis"])

    # Empty config should fail
    with pytest.raises(AssertionError):
        assert_valid_config({})

    # Missing required section should fail
    with pytest.raises(AssertionError):
        assert_valid_config({"gemma": {}}, required_sections=["redis"])


@pytest.mark.unit
def test_generate_random_prompt():
    """Test generate_random_prompt utility."""
    prompt = generate_random_prompt(min_words=10, max_words=20)

    assert isinstance(prompt, str)
    word_count = len(prompt.split())
    assert 10 <= word_count <= 20
    assert prompt[0].isupper()  # First letter capitalized


# =============================================================================
# Data Fixture Tests
# =============================================================================


@pytest.mark.unit
def test_sample_configs_fixture():
    """Test that SAMPLE_CONFIGS provides valid configurations."""
    minimal_config = get_sample_config("minimal")
    assert "gemma" in minimal_config
    assert_valid_config(minimal_config, required_sections=["gemma"])

    dev_config = get_sample_config("development")
    assert "gemma" in dev_config
    assert "redis" in dev_config


@pytest.mark.unit
def test_sample_prompts_categories():
    """Test that SAMPLE_PROMPTS provides various categories."""
    from tests.utils.fixtures import SAMPLE_PROMPTS

    assert "simple" in SAMPLE_PROMPTS
    assert "factual" in SAMPLE_PROMPTS
    assert "technical" in SAMPLE_PROMPTS
    assert "coding" in SAMPLE_PROMPTS

    for category, prompts in SAMPLE_PROMPTS.items():
        assert len(prompts) > 0
        for prompt in prompts:
            assert_valid_prompt(prompt)


@pytest.mark.unit
def test_sample_conversation():
    """Test that sample conversations are well-formed."""
    conversation = get_sample_conversation(0)

    assert "title" in conversation
    assert "messages" in conversation
    assert len(conversation["messages"]) > 0

    for message in conversation["messages"]:
        assert "role" in message
        assert "content" in message
        assert message["role"] in ["user", "assistant"]


# =============================================================================
# Subprocess Mocking Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.subprocess
def test_mock_subprocess_call(mock_subprocess_call):
    """Test that subprocess mocking works."""
    import subprocess

    # This should use the mock instead of actually running
    result = subprocess.Popen(["echo", "test"])
    stdout, stderr = result.communicate()

    # Mock returns empty by default
    assert result.returncode == 0
    assert result.pid == 12345


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.performance
def test_benchmark_timer(benchmark_timer):
    """Test that benchmark_timer measures execution time."""
    import time

    with benchmark_timer as timer:
        time.sleep(0.1)

    assert timer.elapsed >= 0.1
    assert timer.elapsed < 0.2  # Should not take much longer


# =============================================================================
# Conversation History Tests
# =============================================================================


@pytest.mark.unit
def test_sample_conversation_history(sample_conversation_history):
    """Test that sample_conversation_history provides valid data."""
    assert len(sample_conversation_history) > 0

    for turn in sample_conversation_history:
        assert "role" in turn
        assert "content" in turn
        assert turn["role"] in ["user", "assistant"]
        assert len(turn["content"]) > 0


# =============================================================================
# Integration Example (commented out - requires actual components)
# =============================================================================


@pytest.mark.skip(reason="Example only - requires actual implementation")
@pytest.mark.integration
async def test_full_inference_pipeline(
    mock_gemma_interface,
    mock_redis,
    sample_model_preset,
):
    """Example integration test (requires actual components)."""
    # This demonstrates how to combine fixtures for integration testing
    prompt = "What is Python?"

    # Store in Redis
    await mock_redis.set("last_prompt", prompt)

    # Generate response
    response = await mock_gemma_interface.generate_response(prompt)

    # Validate
    assert_valid_response(response)

    # Retrieve from Redis
    stored_prompt = await mock_redis.get("last_prompt")
    assert stored_prompt == prompt


# =============================================================================
# Logging Tests
# =============================================================================


@pytest.mark.unit
def test_capture_logs(capture_logs):
    """Test that capture_logs fixture works."""
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Test log message")

    assert "Test log message" in capture_logs.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
