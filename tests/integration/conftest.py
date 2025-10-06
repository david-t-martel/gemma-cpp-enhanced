"""
Shared pytest fixtures for integration tests
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncIterator, Iterator, Dict, Any
from unittest.mock import MagicMock, AsyncMock, patch
import json
import tempfile

import pytest
import pytest_asyncio
import fakeredis
from fakeredis import FakeRedis, FakeAsyncRedis

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "stats"))

# Import after path setup
from src.agent.core import Agent, AgentConfig
from src.agent.react_agent import ReActAgent
from src.agent.gemma_agent import GemmaAgent, load_model
from src.agent.tools import Tool, ToolRegistry


# ==================== Redis Fixtures ====================

@pytest.fixture
def redis_client() -> FakeRedis:
    """Provides a synchronous fake Redis client for testing."""
    client = fakeredis.FakeRedis(decode_responses=True)
    # Pre-populate with test data if needed
    client.set("test:key", "test_value")
    yield client
    client.flushall()


@pytest_asyncio.fixture
async def async_redis_client() -> AsyncIterator[FakeAsyncRedis]:
    """Provides an async fake Redis client for testing."""
    async with fakeredis.FakeAsyncRedis(decode_responses=True) as client:
        # Pre-populate with test data
        await client.set("test:async:key", "test_async_value")
        yield client
        await client.flushall()


@pytest.fixture
def redis_server():
    """Provides a fake Redis server instance for connection testing."""
    server = fakeredis.FakeServer()
    server.connected = True  # Ensure server is "connected"
    return server


@pytest.fixture
def redis_with_error():
    """Provides a Redis client that simulates connection errors."""
    server = fakeredis.FakeServer()
    server.connected = False  # Simulate connection error
    return fakeredis.FakeRedis(server=server)


# ==================== Agent Fixtures ====================

@pytest.fixture
def agent_config() -> AgentConfig:
    """Provides a test agent configuration."""
    return AgentConfig(
        name="test_agent",
        model="test_model",
        temperature=0.7,
        max_tokens=1000,
        enable_planning=False,
        enable_reflection=False,
        enable_self_critique=False,
        lightweight_mode=True,
        debug_mode=True
    )


@pytest.fixture
def mock_gemma_model():
    """Provides a mocked Gemma model for testing without actual inference."""
    model = MagicMock()
    model.generate = MagicMock(return_value="Mocked response from Gemma model")
    model.generate_async = AsyncMock(return_value="Async mocked response")
    return model


@pytest.fixture
def react_agent(agent_config, mock_gemma_model) -> ReActAgent:
    """Provides a ReAct agent with mocked model for testing."""
    with patch('src.agent.react_agent.load_model', return_value=mock_gemma_model):
        agent = ReActAgent(agent_config)
        agent.model = mock_gemma_model
        return agent


@pytest_asyncio.fixture
async def async_react_agent(agent_config, mock_gemma_model) -> ReActAgent:
    """Provides an async ReAct agent for testing."""
    with patch('src.agent.react_agent.load_model', return_value=mock_gemma_model):
        agent = ReActAgent(agent_config)
        agent.model = mock_gemma_model
        return agent


# ==================== Tool Registry Fixtures ====================

@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Provides a test tool registry with sample tools."""
    registry = ToolRegistry()

    # Add sample synchronous tool
    @registry.register("calculator")
    def calculator(expression: str) -> str:
        """Evaluates a mathematical expression."""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    # Add sample async tool
    @registry.register("async_search")
    async def async_search(query: str) -> str:
        """Performs an async search operation."""
        await asyncio.sleep(0.1)  # Simulate async work
        return f"Search results for: {query}"

    # Add sample tool with Redis
    @registry.register("redis_tool")
    def redis_tool(key: str, value: str = None) -> str:
        """Interacts with Redis."""
        # This would normally use the Redis client
        if value:
            return f"Set {key}={value}"
        return f"Get {key}"

    return registry


# ==================== MCP Server Fixtures ====================

@pytest.fixture
def mcp_server_config() -> Dict[str, Any]:
    """Provides MCP server configuration for testing."""
    return {
        "mcpServers": {
            "test_server": {
                "command": "python",
                "args": ["-m", "test_mcp_server"],
                "env": {
                    "API_KEY": "test_key",
                    "DEBUG": "true"
                }
            },
            "memory_server": {
                "command": "python",
                "args": ["-m", "mcp_memory_server"],
                "env": {
                    "REDIS_URL": "redis://localhost:6379"
                }
            }
        }
    }


@pytest.fixture
def mock_mcp_client():
    """Provides a mocked MCP client for testing."""
    client = MagicMock()
    client.call_tool = AsyncMock(return_value={"status": "success", "result": "test"})
    client.list_tools = AsyncMock(return_value=["tool1", "tool2", "tool3"])
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    return client


# ==================== Memory System Fixtures ====================

@pytest.fixture
def memory_config() -> Dict[str, Any]:
    """Provides memory system configuration."""
    return {
        "tiers": {
            "working": {"capacity": 10, "ttl": 300},
            "short_term": {"capacity": 100, "ttl": 3600},
            "long_term": {"capacity": 10000, "ttl": 86400},
            "episodic": {"capacity": 1000, "ttl": None},
            "semantic": {"capacity": 5000, "ttl": None}
        },
        "consolidation": {
            "enabled": True,
            "interval": 60,
            "threshold": 0.8
        },
        "vector_dim": 384,
        "similarity_threshold": 0.7
    }


@pytest.fixture
def sample_memories() -> list:
    """Provides sample memory entries for testing."""
    return [
        {
            "id": "mem_001",
            "content": "User asked about Python testing",
            "tier": "working",
            "timestamp": 1700000000,
            "vector": [0.1] * 384,
            "metadata": {"source": "chat", "importance": 0.8}
        },
        {
            "id": "mem_002",
            "content": "Discussed Redis integration patterns",
            "tier": "short_term",
            "timestamp": 1700000100,
            "vector": [0.2] * 384,
            "metadata": {"source": "technical", "importance": 0.9}
        },
        {
            "id": "mem_003",
            "content": "Explained MCP protocol basics",
            "tier": "long_term",
            "timestamp": 1700000200,
            "vector": [0.3] * 384,
            "metadata": {"source": "documentation", "importance": 0.7}
        }
    ]


# ==================== File System Fixtures ====================

@pytest.fixture
def temp_project_dir():
    """Provides a temporary project directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)

        # Create basic project structure
        (project_path / "src").mkdir()
        (project_path / "tests").mkdir()
        (project_path / "data").mkdir()
        (project_path / "models").mkdir()

        # Create sample files
        (project_path / "src" / "main.py").write_text("# Main application file")
        (project_path / "tests" / "test_main.py").write_text("# Test file")
        (project_path / "data" / "sample.json").write_text('{"key": "value"}')

        yield project_path


# ==================== Performance Testing Fixtures ====================

@pytest.fixture
def performance_monitor():
    """Provides a performance monitoring utility."""
    import time
    import psutil
    import tracemalloc

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.metrics = {}

        def start(self):
            """Start monitoring."""
            self.start_time = time.perf_counter()
            tracemalloc.start()
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        def stop(self):
            """Stop monitoring and return metrics."""
            if not self.start_time:
                return {}

            end_time = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            self.metrics = {
                "duration_seconds": end_time - self.start_time,
                "memory_used_mb": end_memory - self.start_memory,
                "peak_memory_mb": peak / 1024 / 1024,
                "cpu_percent": psutil.cpu_percent(interval=0.1)
            }

            return self.metrics

        def assert_performance(self, max_duration=None, max_memory=None):
            """Assert performance metrics are within limits."""
            if max_duration and self.metrics.get("duration_seconds", 0) > max_duration:
                pytest.fail(f"Duration {self.metrics['duration_seconds']:.2f}s exceeds limit {max_duration}s")
            if max_memory and self.metrics.get("memory_used_mb", 0) > max_memory:
                pytest.fail(f"Memory {self.metrics['memory_used_mb']:.2f}MB exceeds limit {max_memory}MB")

    return PerformanceMonitor()


# ==================== Async Helpers ====================

@pytest.fixture
def event_loop():
    """Provides an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def async_timeout():
    """Provides a timeout context for async operations."""
    return 10.0  # 10 seconds default timeout


# ==================== Test Data Fixtures ====================

@pytest.fixture
def sample_documents():
    """Provides sample documents for RAG testing."""
    return [
        {
            "id": "doc1",
            "title": "Python Testing Best Practices",
            "content": "Testing is crucial for maintaining code quality...",
            "metadata": {"category": "testing", "language": "python"}
        },
        {
            "id": "doc2",
            "title": "Redis Performance Optimization",
            "content": "Redis can handle millions of operations per second...",
            "metadata": {"category": "database", "technology": "redis"}
        },
        {
            "id": "doc3",
            "title": "MCP Protocol Specification",
            "content": "The Model Context Protocol enables communication...",
            "metadata": {"category": "protocol", "version": "1.0"}
        }
    ]


@pytest.fixture
def sample_queries():
    """Provides sample queries for testing."""
    return [
        "How do I write unit tests in Python?",
        "What are the best practices for Redis caching?",
        "Explain the MCP protocol architecture",
        "How to optimize Gemma model inference?",
        "What is RAG and how does it work?"
    ]


# ==================== Cleanup Fixtures ====================

@pytest.fixture(autouse=True)
def cleanup_env():
    """Automatically cleanup environment after each test."""
    yield
    # Clear any test environment variables
    for key in list(os.environ.keys()):
        if key.startswith("TEST_"):
            del os.environ[key]


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    # Reset tool registry if it's a singleton
    if hasattr(ToolRegistry, '_instance'):
        ToolRegistry._instance = None
    yield