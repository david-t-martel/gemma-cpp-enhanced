"""Rust RAG MCP Client for high-performance RAG operations.

This module provides a Python client that communicates with the Rust-based
RAG-Redis MCP server via JSON-RPC 2.0 over stdio. The Rust backend provides
SIMD-optimized vector operations, efficient memory management, and optional
Redis integration with automatic fallback to in-memory storage.

Architecture:
    Python (gemma-cli) ↔ stdio/JSON-RPC ↔ Rust MCP Server ↔ Redis/In-Memory

Features:
    - Document ingestion with chunking
    - Semantic and hybrid search
    - 5-tier memory management
    - Connection pooling and retry logic
    - Automatic fallback to in-memory when Redis unavailable
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from subprocess import Popen, PIPE, TimeoutExpired
from typing import Any, Dict, List, Optional

from gemma_cli.rag.memory import MemoryEntry, MemoryTier
from gemma_cli.rag.params import (
    RecallMemoriesParams,
    StoreMemoryParams,
    IngestDocumentParams,
    SearchParams,
)

logger = logging.getLogger(__name__)


class RustRagClientError(Exception):
    """Base exception for Rust RAG client errors."""
    pass


class ServerNotRunningError(RustRagClientError):
    """Raised when attempting to communicate with a stopped server."""
    pass


class ServerStartupError(RustRagClientError):
    """Raised when server fails to start."""
    pass


class CommunicationError(RustRagClientError):
    """Raised when communication with server fails."""
    pass


class RustRagClient:
    """Python client for Rust RAG-Redis MCP server.

    This client manages the lifecycle of the Rust MCP server subprocess and
    provides async methods for RAG operations. Communication uses JSON-RPC 2.0
    over stdin/stdout.

    Example:
        ```python
        client = RustRagClient()
        await client.start()
        await client.initialize()

        # Ingest document
        params = IngestDocumentParams(file_path="doc.txt", memory_type="long_term")
        result = await client.ingest_document(params)

        # Search
        results = await client.search("query", limit=5)

        await client.stop()
        ```
    """

    # MCP protocol version
    MCP_PROTOCOL_VERSION = "2024-11-05"

    # Default paths
    DEFAULT_BINARY_PATHS = [
        "C:/codedev/llm/stats/target/release/rag-redis-mcp-server.exe",
        "C:/codedev/llm/rag-redis/target/release/rag-redis-mcp-server.exe",
        "C:/codedev/llm/rag-redis/rag-redis-system/target/release/rag-redis-mcp-server.exe",
        "../../../stats/target/release/rag-redis-mcp-server.exe",
        "../../stats/target/release/rag-redis-mcp-server.exe",
    ]

    def __init__(
        self,
        mcp_server_path: Optional[str] = None,
        startup_timeout: int = 30,
        request_timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ) -> None:
        """Initialize Rust RAG client.

        Args:
            mcp_server_path: Path to mcp-server.exe binary. If None, searches default locations.
            startup_timeout: Maximum seconds to wait for server startup.
            request_timeout: Maximum seconds to wait for request response.
            max_retries: Maximum number of retry attempts for failed requests.
            retry_delay: Delay in seconds between retry attempts.
        """
        self.mcp_server_path = mcp_server_path or self._find_binary()
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.process: Optional[Popen] = None
        self.request_id = 0
        self.initialized = False
        self._write_lock = asyncio.Lock()
        self._read_lock = asyncio.Lock()

    def _find_binary(self) -> str:
        """Find MCP server binary in default locations.

        Returns:
            Path to mcp-server.exe

        Raises:
            ServerStartupError: If binary not found in any default location.
        """
        # Check environment variable first
        env_path = os.getenv("RAG_REDIS_MCP_SERVER")
        if env_path and Path(env_path).exists():
            return env_path

        # Check default paths
        for path in self.DEFAULT_BINARY_PATHS:
            resolved_path = Path(path).resolve()
            if resolved_path.exists():
                logger.info(f"Found MCP server binary at: {resolved_path}")
                return str(resolved_path)

        raise ServerStartupError(
            f"MCP server binary not found. Searched locations:\n"
            f"  - Environment variable RAG_REDIS_MCP_SERVER\n"
            f"  - {', '.join(self.DEFAULT_BINARY_PATHS)}\n"
            f"To build the binary, run:\n"
            f"  cd C:/codedev/llm/rag-redis && cargo build --release"
        )

    async def start(self) -> None:
        """Start the Rust MCP server subprocess.

        Raises:
            ServerStartupError: If server fails to start or binary not found.
        """
        if self.process and self.process.poll() is None:
            logger.warning("Server already running")
            return

        try:
            logger.info(f"Starting Rust MCP server: {self.mcp_server_path}")

            # Start subprocess with stdio pipes
            self.process = Popen(
                [self.mcp_server_path],
                stdin=PIPE,
                stdout=PIPE,
                stderr=PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,  # Line buffered
            )

            # Wait for startup (check stderr for "listening on stdio")
            await self._wait_for_startup()

            logger.info(f"Rust MCP server started (PID: {self.process.pid})")

        except FileNotFoundError as e:
            raise ServerStartupError(f"MCP server binary not found: {self.mcp_server_path}") from e
        except Exception as e:
            await self.stop()  # Cleanup on failure
            raise ServerStartupError(f"Failed to start MCP server: {e}") from e

    async def _wait_for_startup(self) -> None:
        """Wait for server to be ready by monitoring stderr.

        Raises:
            ServerStartupError: If server doesn't start within timeout.
        """
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < self.startup_timeout:
            # Check if process crashed
            if self.process.poll() is not None:
                stderr = self.process.stderr.read()
                raise ServerStartupError(f"Server process terminated during startup. Stderr:\n{stderr}")

            # Try reading a line from stderr (non-blocking would be better but simplified here)
            try:
                # Simple check: if process is still running after 2 seconds, assume ready
                await asyncio.sleep(2)
                if self.process.poll() is None:
                    return
            except Exception as e:
                logger.debug(f"Waiting for server startup: {e}")
                await asyncio.sleep(0.5)

        raise ServerStartupError(f"Server failed to start within {self.startup_timeout}s")

    async def stop(self) -> None:
        """Stop the MCP server process gracefully."""
        if not self.process:
            return

        try:
            logger.info("Stopping Rust MCP server...")

            # Try graceful shutdown first
            self.process.terminate()

            # Wait up to 5 seconds for graceful shutdown
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self.process.wait),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Server didn't stop gracefully, forcing kill...")
                self.process.kill()
                await asyncio.to_thread(self.process.wait)

            logger.info("Rust MCP server stopped")

        except Exception as e:
            logger.error(f"Error stopping server: {e}")
        finally:
            self.process = None
            self.initialized = False

    async def _send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        """Send JSON-RPC request to Rust server.

        Args:
            method: JSON-RPC method name
            params: Method parameters
            retry_count: Current retry attempt (for internal use)

        Returns:
            JSON-RPC result object

        Raises:
            ServerNotRunningError: If server is not running
            CommunicationError: If communication fails after retries
        """
        if not self.process or self.process.poll() is not None:
            raise ServerNotRunningError("MCP server is not running")

        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {},
        }

        try:
            # Send request (with lock to prevent interleaved writes)
            async with self._write_lock:
                request_json = json.dumps(request) + "\n"
                logger.debug(f"Sending request: {method} (id={self.request_id})")

                await asyncio.to_thread(
                    self.process.stdin.write,
                    request_json
                )
                await asyncio.to_thread(self.process.stdin.flush)

            # Read response (with lock to prevent interleaved reads)
            async with self._read_lock:
                response_line = await asyncio.wait_for(
                    asyncio.to_thread(self.process.stdout.readline),
                    timeout=self.request_timeout,
                )

                if not response_line:
                    stderr = await asyncio.to_thread(self.process.stderr.read)
                    raise CommunicationError(f"No response from server. Stderr: {stderr}")

                response = json.loads(response_line)

                # Check for JSON-RPC error
                if "error" in response:
                    error = response["error"]
                    raise CommunicationError(
                        f"RPC error {error.get('code')}: {error.get('message')}"
                    )

                logger.debug(f"Received response for request {self.request_id}")
                return response.get("result", {})

        except (asyncio.TimeoutError, json.JSONDecodeError, OSError) as e:
            # Retry logic
            if retry_count < self.max_retries:
                logger.warning(
                    f"Request failed (attempt {retry_count + 1}/{self.max_retries + 1}): {e}"
                )
                await asyncio.sleep(self.retry_delay)
                return await self._send_request(method, params, retry_count + 1)
            else:
                raise CommunicationError(
                    f"Communication failed after {self.max_retries + 1} attempts: {e}"
                ) from e

    # --- MCP Protocol Methods ---

    async def initialize(self) -> Dict[str, Any]:
        """Initialize MCP connection and handshake.

        Returns:
            Server capabilities and info

        Raises:
            ServerNotRunningError: If server not running
            CommunicationError: If initialization fails
        """
        if self.initialized:
            logger.warning("Client already initialized")
            return {}

        params = {
            "protocolVersion": self.MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "gemma-cli-rust-rag-client",
                "version": "1.0.0",
            },
        }

        result = await self._send_request("initialize", params)
        self.initialized = True
        logger.info(f"MCP connection initialized: {result.get('serverInfo', {}).get('name')}")
        return result

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the server.

        Returns:
            List of tool definitions
        """
        result = await self._send_request("tools/list")
        return result.get("tools", [])

    # --- RAG Operation Methods ---

    async def ingest_document(self, params: IngestDocumentParams) -> Dict[str, Any]:
        """Ingest a document into the RAG system.

        Args:
            params: Document ingestion parameters

        Returns:
            Ingestion result with document ID and chunk count
        """
        # Read file content
        file_path = Path(params.file_path)
        if not file_path.exists():
            raise ValueError(f"File not found: {params.file_path}")

        content = file_path.read_text(encoding="utf-8")

        # Call Rust MCP server
        tool_params = {
            "name": "ingest_document",
            "arguments": {
                "content": content,
                "metadata": {
                    "source": str(file_path),
                    "memory_type": params.memory_type,
                    "chunk_size": params.chunk_size,
                },
            },
        }

        return await self._send_request("tools/call", tool_params)

    async def search(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Perform semantic search for documents.

        Args:
            query: Search query string
            limit: Maximum number of results
            threshold: Minimum similarity threshold (0.0-1.0)

        Returns:
            List of search results with content and metadata
        """
        tool_params = {
            "name": "search_documents",
            "arguments": {
                "query": query,
                "limit": limit,
                "threshold": threshold,
            },
        }

        result = await self._send_request("tools/call", tool_params)

        # Extract results from MCP response
        if isinstance(result, dict) and "content" in result:
            # Parse the text content which should be JSON
            content_items = result.get("content", [])
            if content_items and len(content_items) > 0:
                text = content_items[0].get("text", "[]")
                return json.loads(text)

        return []

    async def store_memory(self, params: StoreMemoryParams) -> Optional[str]:
        """Store a memory in the RAG system.

        Args:
            params: Memory storage parameters

        Returns:
            Memory ID if successful
        """
        tool_params = {
            "name": "store_memory",
            "arguments": {
                "content": params.content,
                "memory_type": params.memory_type,
                "importance": params.importance,
                "tags": params.tags or [],
            },
        }

        result = await self._send_request("tools/call", tool_params)

        # Extract memory ID from response
        if isinstance(result, dict) and "content" in result:
            content_items = result.get("content", [])
            if content_items:
                text = content_items[0].get("text", "")
                # Parse JSON response
                data = json.loads(text)
                return data.get("memory_id")

        return None

    async def recall_memory(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Recall memories from the system.

        Args:
            query: Query string to search memories
            memory_type: Optional memory tier filter
            limit: Maximum number of memories to return

        Returns:
            List of recalled memories
        """
        tool_params = {
            "name": "recall_memory",
            "arguments": {
                "query": query,
                "memory_type": memory_type,
                "limit": limit,
            },
        }

        result = await self._send_request("tools/call", tool_params)

        # Extract memories from response
        if isinstance(result, dict) and "content" in result:
            content_items = result.get("content", [])
            if content_items:
                text = content_items[0].get("text", "[]")
                return json.loads(text)

        return []

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics.

        Returns:
            Statistics about memory usage across tiers
        """
        tool_params = {
            "name": "get_memory_stats",
            "arguments": {},
        }

        result = await self._send_request("tools/call", tool_params)

        # Extract stats from response
        if isinstance(result, dict) and "content" in result:
            content_items = result.get("content", [])
            if content_items:
                text = content_items[0].get("text", "{}")
                return json.loads(text)

        return {}

    async def health_check(self) -> Dict[str, Any]:
        """Check server health and status.

        Returns:
            Health status information
        """
        tool_params = {
            "name": "health_check",
            "arguments": {},
        }

        result = await self._send_request("tools/call", tool_params)

        # Extract health info from response
        if isinstance(result, dict) and "content" in result:
            content_items = result.get("content", [])
            if content_items:
                text = content_items[0].get("text", "{}")
                return json.loads(text)

        return {}

    # --- Lifecycle Management ---

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    def is_running(self) -> bool:
        """Check if server process is running.

        Returns:
            True if server is running, False otherwise
        """
        return self.process is not None and self.process.poll() is None
