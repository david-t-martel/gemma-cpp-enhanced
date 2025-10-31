"""Hybrid RAG backend adapter with intelligent backend selection.

This module provides a unified interface for RAG operations with automatic
fallback between MCP, FFI (PyO3/Rust), and Python implementations.

Backend priority:
1. MCP client (best performance + isolation, Rust backend via network)
2. FFI bindings (direct Rust integration via PyO3, zero-copy)
3. Python fallback (pure Python, always available)

The adapter automatically selects the best available backend and provides
consistent error handling and performance metrics.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field

# Import shared utilities
try:
    from src.shared.logging import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)


class BackendType(str, Enum):
    """Available RAG backend types."""

    MCP = "mcp"  # MCP client to Rust backend
    FFI = "ffi"  # PyO3 FFI bindings to Rust
    PYTHON = "python"  # Pure Python implementation


class MemoryType(str, Enum):
    """Memory tier types matching RAG system."""

    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class BackendStats:
    """Performance statistics for a backend."""

    backend_type: BackendType
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency_ms: float = 0.0
    last_error: Optional[str] = None
    last_success: Optional[datetime] = None
    initialization_time_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_latency_ms / self.successful_calls

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successful_calls + self.failed_calls
        if total == 0:
            return 0.0
        return self.successful_calls / total

    def record_success(self, latency_ms: float) -> None:
        """Record a successful operation."""
        self.successful_calls += 1
        self.total_latency_ms += latency_ms
        self.last_success = datetime.now()

    def record_failure(self, error: str) -> None:
        """Record a failed operation."""
        self.failed_calls += 1
        self.last_error = error


class MemoryItem(BaseModel):
    """Memory item representation."""

    id: str
    content: str
    memory_type: MemoryType
    importance: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Search result representation."""

    id: str
    content: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    """Document metadata for ingestion."""

    title: Optional[str] = None
    source: Optional[str] = None
    doc_type: str = "text"
    tags: List[str] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class RAGBackend(Protocol):
    """Protocol defining the RAG backend interface.

    All backend implementations must provide these methods with consistent
    signatures and semantics.
    """

    async def initialize(self) -> bool:
        """Initialize the backend.

        Returns:
            True if initialization successful, False otherwise
        """
        ...

    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.LONG_TERM,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Store a memory in the RAG system.

        Args:
            content: Memory content
            memory_type: Type of memory tier
            importance: Importance score (0-1)
            tags: Optional tags for categorization

        Returns:
            Memory ID

        Raises:
            RuntimeError: If storage fails
        """
        ...

    async def recall_memories(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 5,
    ) -> List[MemoryItem]:
        """Recall memories matching a query.

        Args:
            query: Search query
            memory_type: Optional memory type filter
            limit: Maximum number of results

        Returns:
            List of memory items
        """
        ...

    async def search_memories(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        min_importance: float = 0.0,
    ) -> List[SearchResult]:
        """Search memories with similarity scoring.

        Args:
            query: Search query
            memory_type: Optional memory type filter
            min_importance: Minimum importance threshold

        Returns:
            List of search results with scores
        """
        ...

    async def ingest_document(
        self,
        file_path: Path,
        metadata: Optional[DocumentMetadata] = None,
        chunk_size: int = 512,
    ) -> str:
        """Ingest a document into the RAG system.

        Args:
            file_path: Path to document
            metadata: Optional document metadata
            chunk_size: Size of text chunks for processing

        Returns:
            Document ID

        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If ingestion fails
        """
        ...

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics.

        Returns:
            Dictionary with statistics
        """
        ...

    async def cleanup_expired(self) -> int:
        """Clean up expired memories.

        Returns:
            Number of memories cleaned up
        """
        ...

    async def close(self) -> None:
        """Close backend and release resources."""
        ...


class MCPRAGBackend:
    """MCP client backend for RAG operations.

    Connects to a Rust-based RAG-Redis MCP server via HTTP/stdio.
    Provides best performance and isolation.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        timeout: int = 30,
    ):
        """Initialize MCP backend.

        Args:
            host: MCP server host
            port: MCP server port
            timeout: Request timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self._initialized = False
        self._client: Optional[Any] = None

    async def initialize(self) -> bool:
        """Initialize MCP client connection."""
        try:
            import aiohttp

            # Test connection
            url = f"http://{self.host}:{self.port}/health"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        self._initialized = True
                        logger.info(
                            f"MCP backend initialized: {self.host}:{self.port}"
                        )
                        return True
                    else:
                        logger.warning(
                            f"MCP server returned status {response.status}"
                        )
                        return False

        except Exception as e:
            logger.warning(f"Failed to initialize MCP backend: {e}")
            return False

    async def _call_mcp_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call an MCP tool with parameters.

        Args:
            tool_name: Name of the MCP tool
            params: Tool parameters

        Returns:
            Tool result data

        Raises:
            RuntimeError: If call fails
        """
        import aiohttp

        url = f"http://{self.host}:{self.port}/tools/{tool_name}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=params,
                    timeout=self.timeout,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("success"):
                            return result.get("data", {})
                        else:
                            raise RuntimeError(
                                f"MCP tool failed: {result.get('error')}"
                            )
                    else:
                        raise RuntimeError(
                            f"MCP server error: {response.status}"
                        )

        except asyncio.TimeoutError:
            raise RuntimeError(f"MCP call timeout after {self.timeout}s")
        except Exception as e:
            raise RuntimeError(f"MCP call failed: {e}")

    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.LONG_TERM,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Store memory via MCP."""
        if not self._initialized:
            raise RuntimeError("MCP backend not initialized")

        result = await self._call_mcp_tool(
            "rag_store_memory",
            {
                "content": content,
                "memory_type": memory_type.value,
                "importance": importance,
                "tags": tags or [],
            },
        )

        return result.get("memory_id", "")

    async def recall_memories(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 5,
    ) -> List[MemoryItem]:
        """Recall memories via MCP."""
        if not self._initialized:
            raise RuntimeError("MCP backend not initialized")

        result = await self._call_mcp_tool(
            "rag_recall_memory",
            {
                "query": query,
                "memory_type": memory_type.value if memory_type else None,
                "limit": limit,
            },
        )

        memories = result.get("memories", [])
        return [
            MemoryItem(
                id=m["id"],
                content=m["content"],
                memory_type=MemoryType(m["memory_type"]),
                importance=m.get("importance", 0.5),
                created_at=datetime.fromisoformat(m["created_at"]),
                tags=m.get("tags", []),
                metadata=m.get("metadata", {}),
            )
            for m in memories
        ]

    async def search_memories(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        min_importance: float = 0.0,
    ) -> List[SearchResult]:
        """Search memories via MCP."""
        if not self._initialized:
            raise RuntimeError("MCP backend not initialized")

        result = await self._call_mcp_tool(
            "rag_search",
            {
                "query": query,
                "memory_type": memory_type.value if memory_type else None,
                "min_importance": min_importance,
            },
        )

        results = result.get("results", [])
        return [
            SearchResult(
                id=r["id"],
                content=r["content"],
                score=r["score"],
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]

    async def ingest_document(
        self,
        file_path: Path,
        metadata: Optional[DocumentMetadata] = None,
        chunk_size: int = 512,
    ) -> str:
        """Ingest document via MCP."""
        if not self._initialized:
            raise RuntimeError("MCP backend not initialized")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        meta = metadata or DocumentMetadata()

        result = await self._call_mcp_tool(
            "rag_ingest_document",
            {
                "content": content,
                "title": meta.title or file_path.name,
                "source": meta.source or str(file_path),
                "doc_type": meta.doc_type,
                "tags": meta.tags,
                "importance": meta.importance,
                "chunk_size": chunk_size,
            },
        )

        return result.get("document_id", "")

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory stats via MCP."""
        if not self._initialized:
            raise RuntimeError("MCP backend not initialized")

        result = await self._call_mcp_tool("rag_get_stats", {})
        return result

    async def cleanup_expired(self) -> int:
        """Cleanup expired memories via MCP."""
        if not self._initialized:
            raise RuntimeError("MCP backend not initialized")

        result = await self._call_mcp_tool("rag_cleanup_expired", {})
        return result.get("cleaned_count", 0)

    async def close(self) -> None:
        """Close MCP client."""
        self._initialized = False
        logger.info("MCP backend closed")


class FFIRAGBackend:
    """FFI backend using PyO3 bindings to Rust.

    Provides direct FFI calls to rag-redis-system Rust library.
    Offers best performance for local operations with zero-copy data transfer.
    """

    def __init__(self):
        """Initialize FFI backend."""
        self._initialized = False
        self._rag_system: Optional[Any] = None

    async def initialize(self) -> bool:
        """Initialize FFI bindings."""
        try:
            # Try to import Rust FFI module
            import rag_redis_system

            # Initialize Rust RAG system
            config = rag_redis_system.Config.default()
            self._rag_system = await rag_redis_system.RagSystem.new(config)

            self._initialized = True
            logger.info("FFI backend initialized with Rust bindings")
            return True

        except ImportError:
            logger.warning("rag_redis_system module not available")
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize FFI backend: {e}")
            return False

    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.LONG_TERM,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Store memory via FFI."""
        if not self._initialized or not self._rag_system:
            raise RuntimeError("FFI backend not initialized")

        memory_id = await self._rag_system.handle_store_memory(
            content,
            memory_type.value,
            importance,
        )

        return memory_id

    async def recall_memories(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 5,
    ) -> List[MemoryItem]:
        """Recall memories via FFI."""
        if not self._initialized or not self._rag_system:
            raise RuntimeError("FFI backend not initialized")

        memories = await self._rag_system.handle_recall_memory(
            query,
            memory_type.value if memory_type else None,
            limit,
        )

        return [
            MemoryItem(
                id=m["id"],
                content=m["content"],
                memory_type=MemoryType(m["memory_type"]),
                importance=m.get("importance", 0.5),
                created_at=datetime.fromisoformat(m["created_at"]),
                tags=m.get("tags", []),
                metadata=m.get("metadata", {}),
            )
            for m in memories
        ]

    async def search_memories(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        min_importance: float = 0.0,
    ) -> List[SearchResult]:
        """Search memories via FFI."""
        if not self._initialized or not self._rag_system:
            raise RuntimeError("FFI backend not initialized")

        results = await self._rag_system.handle_search(
            query,
            limit=10,
            threshold=0.7,
        )

        # Filter by memory type and importance
        filtered = []
        for r in results:
            if memory_type and r.get("memory_type") != memory_type.value:
                continue
            if r.get("importance", 0.0) < min_importance:
                continue

            filtered.append(
                SearchResult(
                    id=r["id"],
                    content=r["content"],
                    score=r["score"],
                    metadata=r.get("metadata", {}),
                )
            )

        return filtered

    async def ingest_document(
        self,
        file_path: Path,
        metadata: Optional[DocumentMetadata] = None,
        chunk_size: int = 512,
    ) -> str:
        """Ingest document via FFI."""
        if not self._initialized or not self._rag_system:
            raise RuntimeError("FFI backend not initialized")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        meta = metadata or DocumentMetadata()

        doc_metadata = {
            "title": meta.title or file_path.name,
            "source": meta.source or str(file_path),
            "type": meta.doc_type,
            "tags": meta.tags,
            "importance": meta.importance,
        }

        document_id = await self._rag_system.handle_ingest_document(
            content,
            doc_metadata,
        )

        return document_id

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory stats via FFI."""
        if not self._initialized or not self._rag_system:
            raise RuntimeError("FFI backend not initialized")

        # FFI backend stats
        return {
            "backend": "ffi",
            "status": "active",
        }

    async def cleanup_expired(self) -> int:
        """Cleanup expired memories via FFI."""
        if not self._initialized or not self._rag_system:
            raise RuntimeError("FFI backend not initialized")

        # Call Rust cleanup if available
        # For now return 0
        return 0

    async def close(self) -> None:
        """Close FFI backend."""
        self._rag_system = None
        self._initialized = False
        logger.info("FFI backend closed")


class PythonRAGBackend:
    """Pure Python fallback backend.

    Implements RAG operations in pure Python with Redis.
    Always available but with lower performance than MCP/FFI.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Python backend.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self._initialized = False
        self._redis: Optional[Any] = None
        self._memories: Dict[str, MemoryItem] = {}

    async def initialize(self) -> bool:
        """Initialize Python backend."""
        try:
            import redis.asyncio as aioredis

            # Connect to Redis
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )

            # Test connection
            await self._redis.ping()

            self._initialized = True
            logger.info("Python backend initialized with Redis")
            return True

        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            # Use in-memory fallback (no Redis)
            self._redis = None
            self._initialized = True
            logger.info("Python backend using in-memory storage (no Redis)")
            return True

    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.LONG_TERM,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Store memory in Python backend."""
        import hashlib

        memory_id = f"mem_{hashlib.md5(content.encode()).hexdigest()[:12]}"

        memory = MemoryItem(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            created_at=datetime.now(),
            tags=tags or [],
        )

        self._memories[memory_id] = memory

        # Store in Redis if available
        if self._redis:
            await self._redis.hset(
                f"memory:{memory_id}",
                mapping={
                    "content": content,
                    "type": memory_type.value,
                    "importance": importance,
                    "created_at": datetime.now().isoformat(),
                },
            )

        return memory_id

    async def recall_memories(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 5,
    ) -> List[MemoryItem]:
        """Recall memories from Python backend."""
        # Simple keyword matching for fallback
        results = []

        query_lower = query.lower()

        for memory in self._memories.values():
            if memory_type and memory.memory_type != memory_type:
                continue

            if query_lower in memory.content.lower():
                results.append(memory)

            if len(results) >= limit:
                break

        return results

    async def search_memories(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        min_importance: float = 0.0,
    ) -> List[SearchResult]:
        """Search memories in Python backend."""
        results = []

        query_lower = query.lower()

        for memory in self._memories.values():
            if memory_type and memory.memory_type != memory_type:
                continue
            if memory.importance < min_importance:
                continue

            # Simple scoring based on keyword presence
            score = 0.5
            if query_lower in memory.content.lower():
                score = 0.9

            results.append(
                SearchResult(
                    id=memory.id,
                    content=memory.content,
                    score=score,
                    metadata={"type": memory.memory_type.value},
                )
            )

        return sorted(results, key=lambda x: x.score, reverse=True)[:10]

    async def ingest_document(
        self,
        file_path: Path,
        metadata: Optional[DocumentMetadata] = None,
        chunk_size: int = 512,
    ) -> str:
        """Ingest document in Python backend."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read and chunk document
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Simple chunking
        chunks = [
            content[i:i + chunk_size]
            for i in range(0, len(content), chunk_size)
        ]

        meta = metadata or DocumentMetadata()

        # Store each chunk as a memory
        doc_id = f"doc_{hash(content) % 100000}"
        for i, chunk in enumerate(chunks):
            await self.store_memory(
                chunk,
                memory_type=MemoryType.LONG_TERM,
                importance=meta.importance,
                tags=[f"doc:{doc_id}", f"chunk:{i}"] + meta.tags,
            )

        return doc_id

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory stats from Python backend."""
        return {
            "backend": "python",
            "total_memories": len(self._memories),
            "redis_connected": self._redis is not None,
        }

    async def cleanup_expired(self) -> int:
        """Cleanup expired memories (no-op for Python backend)."""
        return 0

    async def close(self) -> None:
        """Close Python backend."""
        if self._redis:
            await self._redis.close()

        self._initialized = False
        logger.info("Python backend closed")


class HybridRAGManager:
    """Hybrid RAG manager with intelligent backend selection.

    Automatically selects the best available backend with fallback chain:
    MCP -> FFI -> Python

    Provides unified interface with consistent error handling and metrics.
    """

    def __init__(
        self,
        prefer_backend: Optional[BackendType] = None,
        mcp_host: str = "localhost",
        mcp_port: int = 8765,
        redis_url: str = "redis://localhost:6379",
    ):
        """Initialize hybrid RAG manager.

        Args:
            prefer_backend: Preferred backend (or None for auto-select)
            mcp_host: MCP server host
            mcp_port: MCP server port
            redis_url: Redis URL for Python backend
        """
        self.prefer_backend = prefer_backend
        self.mcp_host = mcp_host
        self.mcp_port = mcp_port
        self.redis_url = redis_url

        self._backend: Optional[RAGBackend] = None
        self._active_backend_type: Optional[BackendType] = None
        self._stats: Dict[BackendType, BackendStats] = {
            BackendType.MCP: BackendStats(BackendType.MCP),
            BackendType.FFI: BackendStats(BackendType.FFI),
            BackendType.PYTHON: BackendStats(BackendType.PYTHON),
        }

    async def initialize(self) -> bool:
        """Initialize the hybrid RAG manager.

        Tries backends in priority order:
        1. Preferred backend (if specified)
        2. MCP
        3. FFI
        4. Python (always succeeds)

        Returns:
            True if initialization successful
        """
        start_time = time.time()

        # Try preferred backend first
        if self.prefer_backend:
            if await self._try_backend(self.prefer_backend):
                init_time = (time.time() - start_time) * 1000
                self._stats[self.prefer_backend].initialization_time_ms = init_time
                logger.info(
                    f"Using preferred backend: {self.prefer_backend.value} "
                    f"(init: {init_time:.1f}ms)"
                )
                return True

        # Fallback chain: MCP -> FFI -> Python
        for backend_type in [BackendType.MCP, BackendType.FFI, BackendType.PYTHON]:
            if await self._try_backend(backend_type):
                init_time = (time.time() - start_time) * 1000
                self._stats[backend_type].initialization_time_ms = init_time
                logger.info(
                    f"Using backend: {backend_type.value} (init: {init_time:.1f}ms)"
                )
                return True

        logger.error("Failed to initialize any RAG backend")
        return False

    async def _try_backend(self, backend_type: BackendType) -> bool:
        """Try to initialize a specific backend.

        Args:
            backend_type: Type of backend to try

        Returns:
            True if successful
        """
        try:
            if backend_type == BackendType.MCP:
                backend = MCPRAGBackend(self.mcp_host, self.mcp_port)
            elif backend_type == BackendType.FFI:
                backend = FFIRAGBackend()
            elif backend_type == BackendType.PYTHON:
                backend = PythonRAGBackend(self.redis_url)
            else:
                return False

            if await backend.initialize():
                self._backend = backend
                self._active_backend_type = backend_type
                return True

            return False

        except Exception as e:
            logger.debug(f"Failed to initialize {backend_type.value}: {e}")
            return False

    def _measure_operation(self, backend_type: BackendType):
        """Context manager to measure operation time."""
        class OperationTimer:
            def __init__(self, manager: "HybridRAGManager", btype: BackendType):
                self.manager = manager
                self.backend_type = btype
                self.start_time = 0.0

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                latency = (time.time() - self.start_time) * 1000

                if exc_type is None:
                    self.manager._stats[self.backend_type].record_success(latency)
                else:
                    error_msg = str(exc_val) if exc_val else "Unknown error"
                    self.manager._stats[self.backend_type].record_failure(error_msg)

        return OperationTimer(self, backend_type)

    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.LONG_TERM,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Store memory via active backend."""
        if not self._backend or not self._active_backend_type:
            raise RuntimeError("RAG manager not initialized")

        with self._measure_operation(self._active_backend_type):
            return await self._backend.store_memory(
                content,
                memory_type,
                importance,
                tags,
            )

    async def recall_memories(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 5,
    ) -> List[MemoryItem]:
        """Recall memories via active backend."""
        if not self._backend or not self._active_backend_type:
            raise RuntimeError("RAG manager not initialized")

        with self._measure_operation(self._active_backend_type):
            return await self._backend.recall_memories(query, memory_type, limit)

    async def search_memories(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        min_importance: float = 0.0,
    ) -> List[SearchResult]:
        """Search memories via active backend."""
        if not self._backend or not self._active_backend_type:
            raise RuntimeError("RAG manager not initialized")

        with self._measure_operation(self._active_backend_type):
            return await self._backend.search_memories(
                query,
                memory_type,
                min_importance,
            )

    async def ingest_document(
        self,
        file_path: Path,
        metadata: Optional[DocumentMetadata] = None,
        chunk_size: int = 512,
    ) -> str:
        """Ingest document via active backend."""
        if not self._backend or not self._active_backend_type:
            raise RuntimeError("RAG manager not initialized")

        with self._measure_operation(self._active_backend_type):
            return await self._backend.ingest_document(file_path, metadata, chunk_size)

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory stats via active backend."""
        if not self._backend or not self._active_backend_type:
            raise RuntimeError("RAG manager not initialized")

        with self._measure_operation(self._active_backend_type):
            return await self._backend.get_memory_stats()

    async def cleanup_expired(self) -> int:
        """Cleanup expired memories via active backend."""
        if not self._backend or not self._active_backend_type:
            raise RuntimeError("RAG manager not initialized")

        with self._measure_operation(self._active_backend_type):
            return await self._backend.cleanup_expired()

    async def close(self) -> None:
        """Close active backend."""
        if self._backend:
            await self._backend.close()
            self._backend = None
            self._active_backend_type = None

    def get_active_backend(self) -> Optional[BackendType]:
        """Get currently active backend type."""
        return self._active_backend_type

    def get_backend_stats(self) -> Dict[BackendType, Dict[str, Any]]:
        """Get performance statistics for all backends."""
        return {
            btype: {
                "backend": btype.value,
                "successful_calls": stats.successful_calls,
                "failed_calls": stats.failed_calls,
                "avg_latency_ms": stats.avg_latency_ms,
                "success_rate": stats.success_rate,
                "last_error": stats.last_error,
                "last_success": (
                    stats.last_success.isoformat() if stats.last_success else None
                ),
                "initialization_time_ms": stats.initialization_time_ms,
            }
            for btype, stats in self._stats.items()
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on active backend."""
        if not self._backend or not self._active_backend_type:
            return {"status": "not_initialized"}

        try:
            stats = await self.get_memory_stats()
            backend_stats = self._stats[self._active_backend_type]

            return {
                "status": "healthy",
                "active_backend": self._active_backend_type.value,
                "backend_stats": stats,
                "performance": {
                    "avg_latency_ms": backend_stats.avg_latency_ms,
                    "success_rate": backend_stats.success_rate,
                    "total_calls": (
                        backend_stats.successful_calls + backend_stats.failed_calls
                    ),
                },
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "active_backend": self._active_backend_type.value,
                "error": str(e),
            }


__all__ = [
    "BackendType",
    "MemoryType",
    "BackendStats",
    "MemoryItem",
    "SearchResult",
    "DocumentMetadata",
    "RAGBackend",
    "MCPRAGBackend",
    "FFIRAGBackend",
    "PythonRAGBackend",
    "HybridRAGManager",
]
