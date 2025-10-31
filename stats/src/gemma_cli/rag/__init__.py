"""RAG (Retrieval-Augmented Generation) integration module.

This module provides hybrid RAG backend support with intelligent selection
between MCP, FFI, and Python implementations.

Example usage:
    ```python
    from src.gemma_cli.rag import HybridRAGManager, MemoryType

    # Initialize with automatic backend selection
    rag_manager = HybridRAGManager()
    await rag_manager.initialize()

    # Store a memory
    memory_id = await rag_manager.store_memory(
        "Important information about the project",
        memory_type=MemoryType.LONG_TERM,
        importance=0.8,
    )

    # Recall memories
    memories = await rag_manager.recall_memories(
        "project information",
        limit=5,
    )

    # Cleanup
    await rag_manager.close()
    ```

Backend selection:
    The HybridRAGManager automatically selects the best available backend:
    1. MCP (localhost:8765) - Best performance, Rust backend
    2. FFI (PyO3) - Direct Rust bindings, zero-copy
    3. Python - Pure Python fallback, always available
"""

from .adapter import (
    BackendStats,
    BackendType,
    DocumentMetadata,
    FFIRAGBackend,
    HybridRAGManager,
    MCPRAGBackend,
    MemoryItem,
    MemoryType,
    PythonRAGBackend,
    RAGBackend,
    SearchResult,
)

__all__ = [
    "HybridRAGManager",
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
]

__version__ = "0.1.0"
