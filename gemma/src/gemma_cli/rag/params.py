"""Parameter classes for RAG operations to avoid circular imports."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from gemma_cli.rag.memory import MemoryTier


@dataclass
class RecallMemoriesParams:
    """Parameters for recall_memories operation."""
    query: str
    tier: MemoryTier
    max_results: int = 10
    threshold: float = 0.5


@dataclass
class StoreMemoryParams:
    """Parameters for store_memory operation."""
    content: str
    tier: MemoryTier
    metadata: Optional[Dict] = None


@dataclass
class IngestDocumentParams:
    """Parameters for document ingestion."""
    file_path: str
    memory_type: MemoryTier
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class SearchParams:
    """Parameters for search operation."""
    query: str
    tier: Optional[MemoryTier] = None
    max_results: int = 10
    threshold: float = 0.5