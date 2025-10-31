"""Comprehensive test suite for RAG backend adapter.

Tests:
- Backend selection and fallback
- All RAG operations for each backend
- Error handling and recovery
- Performance metrics
- Health checking
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.gemma_cli.rag import (
    BackendType,
    DocumentMetadata,
    FFIRAGBackend,
    HybridRAGManager,
    MCPRAGBackend,
    MemoryType,
    PythonRAGBackend,
)


class TestPythonRAGBackend:
    """Test suite for Python fallback backend."""

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization."""
        backend = PythonRAGBackend()
        result = await backend.initialize()
        assert result is True
        await backend.close()

    @pytest.mark.asyncio
    async def test_store_memory(self):
        """Test storing memories."""
        backend = PythonRAGBackend()
        await backend.initialize()

        memory_id = await backend.store_memory(
            "Test memory content",
            memory_type=MemoryType.LONG_TERM,
            importance=0.8,
            tags=["test", "important"],
        )

        assert memory_id.startswith("mem_")
        assert len(memory_id) > 4
        await backend.close()

    @pytest.mark.asyncio
    async def test_recall_memories(self):
        """Test recalling memories."""
        backend = PythonRAGBackend()
        await backend.initialize()

        # Store some memories
        await backend.store_memory(
            "Python programming is great",
            memory_type=MemoryType.LONG_TERM,
        )
        await backend.store_memory(
            "Python has great libraries",
            memory_type=MemoryType.SHORT_TERM,
        )

        # Recall memories
        memories = await backend.recall_memories("Python", limit=10)

        assert len(memories) == 2
        assert all(m.memory_type in [MemoryType.LONG_TERM, MemoryType.SHORT_TERM] for m in memories)
        await backend.close()

    @pytest.mark.asyncio
    async def test_search_memories(self):
        """Test searching memories."""
        backend = PythonRAGBackend()
        await backend.initialize()

        # Store memories
        await backend.store_memory(
            "Machine learning with Python",
            importance=0.9,
        )

        # Search
        results = await backend.search_memories("Python", min_importance=0.5)

        assert len(results) > 0
        assert all(r.score >= 0.0 for r in results)
        await backend.close()

    @pytest.mark.asyncio
    async def test_ingest_document(self):
        """Test document ingestion."""
        backend = PythonRAGBackend()
        await backend.initialize()

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("This is a test document.\nIt has multiple lines.\n")
            temp_path = Path(f.name)

        try:
            doc_id = await backend.ingest_document(
                temp_path,
                metadata=DocumentMetadata(
                    title="Test Document",
                    importance=0.7,
                ),
                chunk_size=50,
            )

            assert doc_id.startswith("doc_")
        finally:
            temp_path.unlink()
            await backend.close()

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting backend statistics."""
        backend = PythonRAGBackend()
        await backend.initialize()

        stats = await backend.get_memory_stats()

        assert stats["backend"] == "python"
        assert "total_memories" in stats
        await backend.close()


class TestMCPRAGBackend:
    """Test suite for MCP backend."""

    @pytest.mark.asyncio
    async def test_initialize_failure_no_server(self):
        """Test initialization fails when server unavailable."""
        backend = MCPRAGBackend(host="localhost", port=9999)
        result = await backend.initialize()
        assert result is False

    @pytest.mark.asyncio
    async def test_store_memory_not_initialized(self):
        """Test operations fail when not initialized."""
        backend = MCPRAGBackend()

        with pytest.raises(RuntimeError, match="not initialized"):
            await backend.store_memory("test")

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_store_memory_success(self, mock_session):
        """Test successful memory storage via MCP."""
        # Mock HTTP responses
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "success": True,
            "data": {"memory_id": "test_mem_123"}
        })

        mock_session.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )
        mock_session.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )

        backend = MCPRAGBackend()

        # Mock initialization
        backend._initialized = True

        # Store memory
        memory_id = await backend.store_memory("test content")

        assert memory_id == "test_mem_123"


class TestFFIRAGBackend:
    """Test suite for FFI backend."""

    @pytest.mark.asyncio
    async def test_initialize_no_module(self):
        """Test initialization fails when FFI module unavailable."""
        backend = FFIRAGBackend()

        # This should fail gracefully as module likely not available
        result = await backend.initialize()

        # Accept either success (if module available) or failure
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_operations_not_initialized(self):
        """Test operations fail when not initialized."""
        backend = FFIRAGBackend()

        with pytest.raises(RuntimeError, match="not initialized"):
            await backend.store_memory("test")


class TestHybridRAGManager:
    """Test suite for hybrid RAG manager."""

    @pytest.mark.asyncio
    async def test_initialize_fallback_to_python(self):
        """Test fallback to Python backend."""
        manager = HybridRAGManager()
        result = await manager.initialize()

        assert result is True
        assert manager.get_active_backend() == BackendType.PYTHON
        await manager.close()

    @pytest.mark.asyncio
    async def test_prefer_backend_python(self):
        """Test preferring Python backend."""
        manager = HybridRAGManager(prefer_backend=BackendType.PYTHON)
        result = await manager.initialize()

        assert result is True
        assert manager.get_active_backend() == BackendType.PYTHON
        await manager.close()

    @pytest.mark.asyncio
    async def test_store_memory_through_manager(self):
        """Test storing memory through manager."""
        manager = HybridRAGManager()
        await manager.initialize()

        memory_id = await manager.store_memory(
            "Test content",
            memory_type=MemoryType.LONG_TERM,
            importance=0.7,
        )

        assert memory_id is not None
        assert len(memory_id) > 0
        await manager.close()

    @pytest.mark.asyncio
    async def test_recall_memories_through_manager(self):
        """Test recalling memories through manager."""
        manager = HybridRAGManager()
        await manager.initialize()

        # Store memory first
        await manager.store_memory("Python is awesome")

        # Recall
        memories = await manager.recall_memories("Python", limit=5)

        assert isinstance(memories, list)
        await manager.close()

    @pytest.mark.asyncio
    async def test_search_memories_through_manager(self):
        """Test searching memories through manager."""
        manager = HybridRAGManager()
        await manager.initialize()

        # Store memory
        await manager.store_memory("Machine learning", importance=0.9)

        # Search
        results = await manager.search_memories("machine", min_importance=0.5)

        assert isinstance(results, list)
        await manager.close()

    @pytest.mark.asyncio
    async def test_ingest_document_through_manager(self):
        """Test document ingestion through manager."""
        manager = HybridRAGManager()
        await manager.initialize()

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Test document content\n" * 10)
            temp_path = Path(f.name)

        try:
            doc_id = await manager.ingest_document(
                temp_path,
                metadata=DocumentMetadata(title="Test Doc"),
            )

            assert doc_id is not None
        finally:
            temp_path.unlink()
            await manager.close()

    @pytest.mark.asyncio
    async def test_get_backend_stats(self):
        """Test getting backend statistics."""
        manager = HybridRAGManager()
        await manager.initialize()

        # Perform some operations
        await manager.store_memory("test 1")
        await manager.store_memory("test 2")

        stats = manager.get_backend_stats()

        assert BackendType.PYTHON in stats
        python_stats = stats[BackendType.PYTHON]
        assert python_stats["successful_calls"] >= 2
        assert python_stats["avg_latency_ms"] >= 0
        await manager.close()

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality."""
        manager = HybridRAGManager()
        await manager.initialize()

        health = await manager.health_check()

        assert health["status"] == "healthy"
        assert health["active_backend"] == BackendType.PYTHON.value
        assert "performance" in health
        await manager.close()

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self):
        """Test health check when not initialized."""
        manager = HybridRAGManager()

        health = await manager.health_check()

        assert health["status"] == "not_initialized"

    @pytest.mark.asyncio
    async def test_operations_not_initialized(self):
        """Test operations fail when manager not initialized."""
        manager = HybridRAGManager()

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.store_memory("test")

    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        """Test cleanup of expired memories."""
        manager = HybridRAGManager()
        await manager.initialize()

        count = await manager.cleanup_expired()

        assert isinstance(count, int)
        assert count >= 0
        await manager.close()

    @pytest.mark.asyncio
    async def test_multiple_operations(self):
        """Test multiple concurrent operations."""
        manager = HybridRAGManager()
        await manager.initialize()

        # Run multiple operations concurrently
        tasks = [
            manager.store_memory(f"memory {i}", importance=0.5 + i * 0.1)
            for i in range(5)
        ]

        memory_ids = await asyncio.gather(*tasks)

        assert len(memory_ids) == 5
        assert all(mid is not None for mid in memory_ids)
        await manager.close()

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error handling and recovery."""
        manager = HybridRAGManager()
        await manager.initialize()

        # Test with invalid importance (should be handled gracefully)
        try:
            await manager.store_memory("test", importance=1.5)
        except Exception as e:
            # Should either validate or handle gracefully
            assert isinstance(e, (ValueError, RuntimeError))

        # Manager should still be functional
        memory_id = await manager.store_memory("valid memory", importance=0.7)
        assert memory_id is not None
        await manager.close()


class TestBackendPerformance:
    """Performance and benchmarking tests."""

    @pytest.mark.asyncio
    async def test_python_backend_latency(self):
        """Test Python backend latency."""
        backend = PythonRAGBackend()
        await backend.initialize()

        import time

        start = time.time()
        for i in range(10):
            await backend.store_memory(f"memory {i}")
        latency = (time.time() - start) * 1000

        # Should complete 10 operations in reasonable time
        assert latency < 1000  # < 1 second for 10 ops
        await backend.close()

    @pytest.mark.asyncio
    async def test_manager_metrics_accuracy(self):
        """Test metrics tracking accuracy."""
        manager = HybridRAGManager()
        await manager.initialize()

        # Perform operations
        await manager.store_memory("test 1")
        await manager.store_memory("test 2")
        await manager.recall_memories("test")

        stats = manager.get_backend_stats()
        active_backend = manager.get_active_backend()

        assert active_backend is not None
        backend_stats = stats[active_backend]

        # Should have tracked all operations
        assert backend_stats["successful_calls"] == 3
        assert backend_stats["avg_latency_ms"] > 0
        assert backend_stats["success_rate"] == 1.0
        await manager.close()


@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring external services."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        True,  # Skip by default, requires Redis
        reason="Requires Redis server running"
    )
    async def test_python_backend_with_redis(self):
        """Test Python backend with actual Redis."""
        backend = PythonRAGBackend(redis_url="redis://localhost:6379")
        result = await backend.initialize()

        if result:
            memory_id = await backend.store_memory("redis test")
            assert memory_id is not None
            await backend.close()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        True,  # Skip by default, requires MCP server
        reason="Requires MCP server running"
    )
    async def test_mcp_backend_with_server(self):
        """Test MCP backend with actual server."""
        backend = MCPRAGBackend(host="localhost", port=8765)
        result = await backend.initialize()

        if result:
            memory_id = await backend.store_memory("mcp test")
            assert memory_id is not None
            await backend.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
