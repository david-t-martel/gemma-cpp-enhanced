"""RAG (Retrieval-Augmented Generation) memory system."""

from gemma_cli.rag.memory import MemoryEntry, MemoryTier

# Remove circular import - these modules import from each other
# from gemma_cli.rag.optimizations import (
#     BatchEmbedder,
#     MemoryConsolidator,
#     PerformanceMonitor,
#     QueryOptimizer,
# )
# from gemma_cli.rag.python_backend import PythonRAGBackend

__all__ = [
    "MemoryEntry",
    "MemoryTier",
    # "PythonRAGBackend",
    # "BatchEmbedder",
    # "MemoryConsolidator",
    # "PerformanceMonitor",
    # "QueryOptimizer",
]
