# MCP edit_block demonstration
"""Hybrid RAG Manager for Gemma CLI."""

import logging
from typing import Any, Optional, List
from pydantic import BaseModel, Field, PositiveInt, NonNegativeFloat

from gemma_cli.rag.memory import MemoryEntry
from gemma_cli.rag.params import RecallMemoriesParams, StoreMemoryParams, IngestDocumentParams, SearchParams
from gemma_cli.rag.python_backend import PythonRAGBackend

logger = logging.getLogger(__name__)


class HybridRAGManager:
    """Manages RAG operations with support for multiple backends.

    Supports three backend options:
    1. 'embedded' - File-based vector store (default, no dependencies)
    2. 'redis' - Python Redis backend (requires Redis server)
    3. 'rust' - High-performance Rust MCP server (SIMD-optimized)
    """

    def __init__(
        self,
        backend: str = "embedded",
        use_embedded_store: Optional[bool] = None,  # Deprecated, for backward compatibility
        rust_mcp_server_path: Optional[str] = None,
        use_optimized_rag: bool = True,
    ) -> None:
        """Initialize RAG manager with specified backend.

        Args:
            backend: Backend type ('embedded', 'redis', or 'rust')
            use_embedded_store: (Deprecated) If provided, overrides backend setting
            rust_mcp_server_path: Path to Rust MCP server binary (for 'rust' backend)
            use_optimized_rag: Use optimized RAG stores for better performance
        """
        # Handle backward compatibility
        if use_embedded_store is not None:
            logger.warning(
                "use_embedded_store parameter is deprecated. Use backend='embedded' or backend='redis' instead."
            )
            backend = "embedded" if use_embedded_store else "redis"

        self.backend_type = backend
        self.rust_mcp_server_path = rust_mcp_server_path
        self.use_optimized_rag = use_optimized_rag

        # Initialize appropriate backend
        if backend == "rust":
            # Import here to avoid circular dependency
            from gemma_cli.rag.rust_rag_client import RustRagClient

            self.rust_client = RustRagClient(mcp_server_path=rust_mcp_server_path)
            self.python_backend = None
            logger.info("Using Rust MCP backend for RAG operations")
        else:
            # Use Python backend (embedded or redis)
            use_embedded = backend == "embedded"
            self.python_backend = PythonRAGBackend(use_embedded_store=use_embedded, use_optimized_rag=self.use_optimized_rag)
            self.rust_client = None
            logger.info(f"Using Python backend ({backend}) for RAG operations")

    async def initialize(self) -> bool:
        """Initialize the RAG manager and its backends."""
        if self.backend_type == "rust":
            try:
                await self.rust_client.start()
                await self.rust_client.initialize()
                logger.info("Rust MCP backend initialized successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize Rust backend: {e}")
                logger.warning("Falling back to embedded Python backend")

                # Fallback to embedded Python backend
                self.backend_type = "embedded"
                self.python_backend = PythonRAGBackend(use_embedded_store=True, use_optimized_rag=self.use_optimized_rag)
                self.rust_client = None
                return await self.python_backend.initialize()
        else:
            return await self.python_backend.initialize()

    async def recall_memories(self, params: RecallMemoriesParams) -> list[MemoryEntry]:
        """Recall memories from the RAG system using structured parameters."""
        if self.backend_type == "rust":
            # Use Rust backend
            results = await self.rust_client.recall_memory(
                query=params.query,
                memory_type=params.memory_type,
                limit=params.limit,
            )
            # Convert Rust results to MemoryEntry objects
            return [self._rust_result_to_memory_entry(r) for r in results]
        else:
            return await self.python_backend.recall_memories(params=params)

    async def store_memory(self, params: StoreMemoryParams) -> Optional[str]:
        """Store memory in the RAG system using structured parameters."""
        if self.backend_type == "rust":
            # Use Rust backend
            memory_id = await self.rust_client.store_memory(params)
            return memory_id
        else:
            return await self.python_backend.store_memory(params=params)

    async def get_memory_stats(self) -> dict[str, Any]:
        """Get statistics about memory usage across tiers."""
        if self.backend_type == "rust":
            return await self.rust_client.get_memory_stats()
        else:
            return await self.python_backend.get_memory_stats()

    async def ingest_document(self, params: IngestDocumentParams) -> int:
        """
        Ingest a document into the memory system by chunking using structured parameters.
        """
        if self.backend_type == "rust":
            result = await self.rust_client.ingest_document(params)
            # Extract chunk count from result
            if isinstance(result, dict) and "content" in result:
                content = result.get("content", [])
                if content:
                    import json
                    data = json.loads(content[0].get("text", "{}"))
                    return data.get("chunks_created", 0)
            return 0
        else:
            return await self.python_backend.ingest_document(params=params)

    async def search_memories(self, params: SearchParams) -> list[MemoryEntry]:
        """
        Search memories by content and importance using structured parameters.
        """
        if self.backend_type == "rust":
            # Use Rust backend's search
            results = await self.rust_client.search(
                query=params.query,
                limit=5,
                threshold=params.min_importance,
            )
            # Convert Rust results to MemoryEntry objects
            return [self._rust_result_to_memory_entry(r) for r in results]
        else:
            return await self.python_backend.search_memories(params=params)

    async def close(self) -> None:
        """Close RAG manager and its backends."""
        if self.backend_type == "rust":
            await self.rust_client.stop()
        else:
            await self.python_backend.close()

    def _rust_result_to_memory_entry(self, rust_result: dict) -> MemoryEntry:
        """Convert Rust backend result to MemoryEntry.

        Args:
            rust_result: Dictionary from Rust backend

        Returns:
            MemoryEntry object
        """
        from gemma_cli.rag.memory import MemoryTier
        import datetime

        # Map memory type string to MemoryTier enum
        tier_map = {
            "working": MemoryTier.WORKING,
            "short_term": MemoryTier.SHORT_TERM,
            "long_term": MemoryTier.LONG_TERM,
            "episodic": MemoryTier.EPISODIC,
            "semantic": MemoryTier.SEMANTIC,
        }

        memory_type_str = rust_result.get("memory_type", "long_term")
        tier = tier_map.get(memory_type_str, MemoryTier.LONG_TERM)

        return MemoryEntry(
            content=rust_result.get("content", ""),
            tier=tier,
            importance=rust_result.get("importance", 0.5),
            timestamp=datetime.datetime.fromisoformat(
                rust_result.get("timestamp", datetime.datetime.now().isoformat())
            ),
            memory_id=rust_result.get("id"),
            tags=rust_result.get("tags", []),
        )
