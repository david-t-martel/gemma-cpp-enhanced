"""Tests for Rust RAG MCP client integration."""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from gemma_cli.rag.rust_rag_client import (
    RustRagClient,
    ServerNotRunningError,
    ServerStartupError,
    CommunicationError,
)
from gemma_cli.rag.hybrid_rag import (
    HybridRAGManager,
    IngestDocumentParams,
    SearchParams,
    StoreMemoryParams,
    RecallMemoriesParams,
)


class TestRustRagClient:
    """Test suite for RustRagClient."""

    def test_find_binary_with_env_var(self, monkeypatch):
        """Test that environment variable takes precedence for binary path."""
        test_path = "C:/test/path/mcp-server.exe"
        monkeypatch.setenv("RAG_REDIS_MCP_SERVER", test_path)

        with patch("pathlib.Path.exists", return_value=True):
            client = RustRagClient()
            assert client.mcp_server_path == test_path

    def test_find_binary_not_found(self):
        """Test ServerStartupError when binary not found."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(ServerStartupError, match="MCP server binary not found"):
                RustRagClient()

    def test_find_binary_in_default_locations(self):
        """Test finding binary in default locations."""

        def mock_exists(self):
            # Only return True for the first default path
            return str(self) == str(
                Path("C:/codedev/llm/rag-redis/target/release/mcp-server.exe").resolve()
            )

        with patch.object(Path, "exists", mock_exists):
            client = RustRagClient()
            assert "mcp-server.exe" in client.mcp_server_path

    @pytest.mark.asyncio
    async def test_start_server_already_running(self):
        """Test that starting an already running server logs warning."""
        client = RustRagClient(mcp_server_path="C:/fake/path.exe")

        # Mock process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process running
        client.process = mock_process

        await client.start()  # Should not raise, just log warning

    @pytest.mark.asyncio
    async def test_stop_server_gracefully(self):
        """Test graceful server shutdown."""
        client = RustRagClient(mcp_server_path="C:/fake/path.exe")

        # Mock process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Initially running
        client.process = mock_process

        # Mock wait to return immediately (graceful shutdown)
        async def mock_wait():
            mock_process.poll.return_value = 0  # Process terminated
            return 0

        with patch("asyncio.to_thread", side_effect=lambda f, *args: mock_wait()):
            await client.stop()

        mock_process.terminate.assert_called_once()
        assert client.process is None
        assert not client.initialized

    @pytest.mark.asyncio
    async def test_send_request_server_not_running(self):
        """Test that sending request to stopped server raises error."""
        client = RustRagClient(mcp_server_path="C:/fake/path.exe")
        client.process = None

        with pytest.raises(ServerNotRunningError):
            await client._send_request("test_method")

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test MCP initialization handshake."""
        client = RustRagClient(mcp_server_path="C:/fake/path.exe")

        # Mock running process
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        client.process = mock_process

        # Mock successful response
        mock_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "serverInfo": {"name": "rag-redis-mcp-server", "version": "0.1.0"},
                "capabilities": {},
            },
        }

        with patch.object(client, "_send_request", return_value=mock_response["result"]):
            result = await client.initialize()

        assert client.initialized
        assert result["serverInfo"]["name"] == "rag-redis-mcp-server"

    @pytest.mark.asyncio
    async def test_is_running(self):
        """Test is_running check."""
        client = RustRagClient(mcp_server_path="C:/fake/path.exe")

        # No process
        assert not client.is_running()

        # Running process
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        client.process = mock_process
        assert client.is_running()

        # Stopped process
        mock_process.poll.return_value = 0
        assert not client.is_running()


class TestHybridRAGManagerRustBackend:
    """Test suite for HybridRAGManager with Rust backend."""

    def test_init_rust_backend(self):
        """Test initialization with Rust backend."""
        with patch("gemma_cli.rag.rust_rag_client.RustRagClient"):
            manager = HybridRAGManager(backend="rust")

        assert manager.backend_type == "rust"
        assert manager.rust_client is not None
        assert manager.python_backend is None

    def test_init_embedded_backend(self):
        """Test initialization with embedded backend (default)."""
        manager = HybridRAGManager(backend="embedded")

        assert manager.backend_type == "embedded"
        assert manager.python_backend is not None
        assert manager.rust_client is None

    def test_init_redis_backend(self):
        """Test initialization with Redis backend."""
        manager = HybridRAGManager(backend="redis")

        assert manager.backend_type == "redis"
        assert manager.python_backend is not None
        assert manager.rust_client is None

    def test_backward_compatibility(self):
        """Test backward compatibility with use_embedded_store parameter."""
        # use_embedded_store=True should use embedded backend
        manager = HybridRAGManager(use_embedded_store=True)
        assert manager.backend_type == "embedded"

        # use_embedded_store=False should use redis backend
        manager = HybridRAGManager(use_embedded_store=False)
        assert manager.backend_type == "redis"

    @pytest.mark.asyncio
    async def test_initialize_rust_backend_success(self):
        """Test successful Rust backend initialization."""
        with patch("gemma_cli.rag.rust_rag_client.RustRagClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.start = asyncio.coroutine(lambda: None)
            mock_client.initialize = asyncio.coroutine(lambda: {"status": "ok"})
            mock_client_class.return_value = mock_client

            manager = HybridRAGManager(backend="rust")
            result = await manager.initialize()

            assert result is True
            mock_client.start.assert_called_once()
            mock_client.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_rust_backend_fallback(self):
        """Test fallback to embedded backend when Rust fails."""
        with patch("gemma_cli.rag.rust_rag_client.RustRagClient") as mock_client_class:
            # Mock Rust client that fails to start
            mock_client = MagicMock()
            mock_client.start = asyncio.coroutine(
                lambda: (_ for _ in ()).throw(ServerStartupError("Binary not found"))
            )
            mock_client_class.return_value = mock_client

            manager = HybridRAGManager(backend="rust")

            # Should fallback to embedded and succeed
            with patch.object(
                manager.python_backend, "initialize", return_value=asyncio.coroutine(lambda: True)()
            ):
                result = await manager.initialize()

            assert manager.backend_type == "embedded"  # Switched to fallback
            assert manager.python_backend is not None
            assert result is True


@pytest.mark.integration
@pytest.mark.skipif(
    not Path("C:/codedev/llm/rag-redis/target/release/mcp-server.exe").exists(),
    reason="Rust MCP server not built",
)
class TestRustRagClientIntegration:
    """Integration tests requiring actual Rust MCP server binary."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, tmp_path):
        """Test full client lifecycle: start, initialize, operations, stop."""
        async with RustRagClient() as client:
            # Should be running and initialized
            assert client.is_running()
            assert client.initialized

            # Test health check
            health = await client.health_check()
            assert isinstance(health, dict)

            # Test list tools
            tools = await client.list_tools()
            assert isinstance(tools, list)
            assert len(tools) > 0

            # Test ingest document
            test_doc = tmp_path / "test.txt"
            test_doc.write_text("This is a test document for RAG ingestion.")

            params = IngestDocumentParams(
                file_path=str(test_doc), memory_type="long_term", chunk_size=100
            )
            result = await client.ingest_document(params)
            assert isinstance(result, dict)

            # Test search
            results = await client.search("test document", limit=5)
            assert isinstance(results, list)

        # After context exit, server should be stopped
        assert not client.is_running()

    @pytest.mark.asyncio
    async def test_memory_operations(self):
        """Test memory store and recall operations."""
        async with RustRagClient() as client:
            # Store memory
            params = StoreMemoryParams(
                content="Test memory entry", memory_type="short_term", importance=0.8
            )
            memory_id = await client.store_memory(params)
            assert memory_id is not None

            # Recall memory
            results = await client.recall_memory("Test memory", limit=5)
            assert isinstance(results, list)
            if len(results) > 0:
                assert "content" in results[0]

            # Get stats
            stats = await client.get_memory_stats()
            assert isinstance(stats, dict)


@pytest.mark.integration
@pytest.mark.skipif(
    not Path("C:/codedev/llm/rag-redis/target/release/mcp-server.exe").exists(),
    reason="Rust MCP server not built",
)
class TestHybridRAGManagerIntegration:
    """Integration tests for HybridRAGManager with Rust backend."""

    @pytest.mark.asyncio
    async def test_end_to_end_rag_workflow(self, tmp_path):
        """Test complete RAG workflow with Rust backend."""
        manager = HybridRAGManager(backend="rust")

        try:
            # Initialize
            await manager.initialize()
            assert manager.backend_type == "rust"

            # Ingest document
            test_doc = tmp_path / "knowledge.txt"
            test_doc.write_text(
                "Python is a high-level programming language. "
                "Rust provides memory safety without garbage collection."
            )

            ingest_params = IngestDocumentParams(
                file_path=str(test_doc), memory_type="long_term", chunk_size=100
            )
            chunks = await manager.ingest_document(ingest_params)
            assert chunks >= 0

            # Search memories
            search_params = SearchParams(
                query="programming language", memory_type=None, min_importance=0.0
            )
            results = await manager.search_memories(search_params)
            assert isinstance(results, list)

            # Store memory
            store_params = StoreMemoryParams(
                content="Integration test memory", memory_type="short_term", importance=0.9
            )
            memory_id = await manager.store_memory(store_params)
            assert memory_id is not None

            # Recall memories
            recall_params = RecallMemoriesParams(query="integration test", memory_type=None, limit=5)
            recalled = await manager.recall_memories(recall_params)
            assert isinstance(recalled, list)

            # Get stats
            stats = await manager.get_memory_stats()
            assert isinstance(stats, dict)

        finally:
            await manager.close()
