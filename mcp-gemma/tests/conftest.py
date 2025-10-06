#!/usr/bin/env python3
"""
Pytest configuration and fixtures for MCP Gemma testing.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_config():
    """Provide a test configuration for MCP Gemma server."""
    return {
        "gemma": {
            "executable_path": "/c/codedev/llm/gemma/gemma.cpp/build_wsl/gemma",
            "model_path": "/c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs",
            "tokenizer_path": "/c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/tokenizer.spm",
            "max_tokens": 512,
            "temperature": 0.7,
            "timeout": 30.0
        },
        "server": {
            "name": "mcp-gemma-test",
            "version": "1.0.0-test",
            "host": "localhost",
            "port": 8080,
            "max_connections": 10
        },
        "memory": {
            "backend": "memory",  # Use in-memory for tests
            "conversation_history_limit": 100
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }

@pytest.fixture
def test_config_minimal():
    """Minimal test configuration for basic functionality tests."""
    return {
        "gemma": {
            "executable_path": "/bin/echo",  # Mock executable for basic tests
            "model_path": "/tmp/fake_model.sbs",
            "tokenizer_path": "/tmp/fake_tokenizer.spm",
            "max_tokens": 10,
            "temperature": 0.5,
            "timeout": 5.0
        },
        "server": {
            "name": "mcp-gemma-minimal",
            "version": "1.0.0-test"
        }
    }

@pytest.fixture
def temp_config_file(test_config):
    """Create a temporary configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f, indent=2)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)

@pytest.fixture
def mock_model_files(tmp_path):
    """Create mock model files for testing."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    # Create fake model file
    model_file = model_dir / "test_model.sbs"
    model_file.write_bytes(b"fake_model_data" * 100)

    # Create fake tokenizer file
    tokenizer_file = model_dir / "tokenizer.spm"
    tokenizer_file.write_bytes(b"fake_tokenizer_data" * 50)

    return {
        "model_path": str(model_file),
        "tokenizer_path": str(tokenizer_file),
        "model_dir": str(model_dir)
    }

@pytest.fixture
def mock_gemma_executable(tmp_path):
    """Create a mock gemma executable for testing."""
    executable = tmp_path / "gemma_mock"
    executable.write_text("""#!/bin/bash
# Mock gemma executable for testing
echo "Mock Gemma v1.0"
echo "Loading model: $4"
echo "Response: This is a test response from mock gemma."
""")
    executable.chmod(0o755)
    return str(executable)

@pytest.fixture(scope="session")
def gemma_executable_exists():
    """Check if the real gemma executable exists."""
    gemma_path = Path("/c/codedev/llm/gemma/gemma.cpp/build_wsl/gemma")
    return gemma_path.exists() and gemma_path.is_file()

@pytest.fixture(scope="session")
def model_files_exist():
    """Check if model files exist."""
    model_path = Path("/c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs")
    tokenizer_path = Path("/c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/tokenizer.spm")
    return model_path.exists() and tokenizer_path.exists()

@pytest.fixture
def sample_mcp_request():
    """Sample MCP request for testing."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "chat",
            "arguments": {
                "message": "Hello, how are you?",
                "max_tokens": 50,
                "temperature": 0.7
            }
        }
    }

@pytest.fixture
def sample_conversation_data():
    """Sample conversation data for testing."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "I don't have access to current weather data."}
    ]

@pytest.mark.asyncio
async def pytest_configure(config):
    """Configure pytest for async testing."""
    pass

def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on requirements."""
    for item in items:
        # Mark tests that require real gemma executable
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark tests that require model files
        if "model" in item.nodeid or "inference" in item.nodeid:
            item.add_marker(pytest.mark.model_required)

        # Mark performance tests
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.performance)