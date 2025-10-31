# src/gemma_cli/rag/embedded_vector_store.py

"""
Embedded, local vector store implementation for RAG memory.

This module provides an alternative to Redis for RAG memory, allowing the
application to be fully standalone without external dependencies. It uses
JSON file persistence for data storage and simple in-memory operations.

This is the DEFAULT storage backend for Gemma CLI, enabling local-first
operation without requiring Redis installation.

Features:
- File-based persistence (JSON format)
- In-memory search operations
- No external dependencies required
- Automatic initialization
- Compatible with all RAG operations

Limitations:
- Keyword-based search only (no true semantic search without embedding model)
- Performance scales linearly with dataset size
- Single-process only (no distributed access)

For production deployments with large datasets or distributed access,
consider using Redis backend by setting redis.enable_fallback=False.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import List, Optional, Any

import numpy as np
import numpy.typing as npt
import aiofiles

from gemma_cli.rag.memory import MemoryEntry
from gemma_cli.rag.hybrid_rag import RecallMemoriesParams, StoreMemoryParams, IngestDocumentParams, SearchParams

logger = logging.getLogger(__name__)

class EmbeddedVectorStore:
    """
    An embedded, local vector store implementation for RAG memory.
    Uses a JSON file for persistence and provides basic in-memory search.
    """
    STORE_FILE = Path.home() / ".gemma_cli" / "embedded_store.json"

    def __init__(self):
        self.store: List[MemoryEntry] = []
        self.initialized = False
        logger.info("EmbeddedVectorStore initialized (in-memory/JSON persistence).")

    async def initialize(self) -> bool:
        """
        Initialize the embedded vector store by loading from file.
        """
        if self.initialized:
            return True
        
        logger.info(f"EmbeddedVectorStore: Initializing from {self.STORE_FILE}...")
        self.STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        if self.STORE_FILE.exists():
            try:
                async with aiofiles.open(self.STORE_FILE, mode="r", encoding="utf-8") as f:
                    data = json.loads(await f.read())
                    self.store = [MemoryEntry.from_dict(entry_dict) for entry_dict in data]
                logger.info(f"EmbeddedVectorStore: Loaded {len(self.store)} entries from {self.STORE_FILE}.")
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"EmbeddedVectorStore: Error loading from file: {e}")
                self.store = [] # Reset store on error
        else:
            logger.info("EmbeddedVectorStore: No existing store file found, starting fresh.")
        
        self.initialized = True
        return True

    async def recall_memories(self, params: RecallMemoriesParams) -> List[MemoryEntry]:
        """
        Recall memories from the embedded store based on semantic similarity.
        """
        if not self.initialized:
            await self.initialize()

        logger.debug(f"EmbeddedVectorStore: Recalling memories for query: {params.query}")
        
        # For a basic implementation, we'll simulate semantic search.
        # In a real scenario, this would involve embedding the query and comparing with stored embeddings.
        # For now, we'll do a simple keyword match and prioritize by importance.
        
        query_lower = params.query.lower()
        results: List[MemoryEntry] = []

        for entry in self.store:
            if params.memory_type and entry.memory_type != params.memory_type:
                continue
            
            # Simple keyword match for now
            if query_lower in entry.content.lower():
                # Simulate a similarity score based on importance
                entry.similarity_score = entry.importance # type: ignore
                results.append(entry)
        
        results.sort(key=lambda x: (x.similarity_score * x.importance), reverse=True) # type: ignore
        return results[:params.limit]

    async def store_memory(self, params: StoreMemoryParams) -> Optional[str]:
        """
        Store memory in the embedded store.
        """
        if not self.initialized:
            await self.initialize()

        logger.debug(f"EmbeddedVectorStore: Storing memory: {params.content[:50]}...")
        entry = MemoryEntry(params.content, params.memory_type, params.importance)
        if params.tags:
            entry.add_tags(*params.tags)
        
        # TODO: [RAG Backend] Generate and store actual embeddings here if an embedder is available.
        # For now, embedding will be None.

        self.store.append(entry)
        logger.info(f"EmbeddedVectorStore: Stored memory with ID {entry.id[:8]}...")
        return entry.id

    async def ingest_document(self, params: IngestDocumentParams) -> int:
        """
        Ingest a document into the embedded store by chunking.
        """
        if not self.initialized:
            await self.initialize()

        logger.info(f"EmbeddedVectorStore: Ingesting document: {params.file_path}")
        
        path = Path(params.file_path)
        if not path.exists():
            logger.warning(f"EmbeddedVectorStore: Document not found: {params.file_path}")
            return 0
        
        # Simple text chunking (can be improved)
        try:
            async with aiofiles.open(path, encoding="utf-8") as f:
                content = await f.read()
            
            chunks = [content[i:i + params.chunk_size] for i in range(0, len(content), params.chunk_size)]
            
            stored_count = 0
            for i, chunk in enumerate(chunks):
                store_params = StoreMemoryParams(
                    content=chunk,
                    memory_type=params.memory_type,
                    importance=0.5, # Default importance for ingested docs
                    tags=[f"document:{path.name}", f"chunk:{i}"]
                )
                entry_id = await self.store_memory(params=store_params)
                if entry_id:
                    stored_count += 1
            
            logger.info(f"EmbeddedVectorStore: Ingested {stored_count} chunks from {params.file_path}.")
            return stored_count
        except (OSError, UnicodeDecodeError) as e:
            logger.error(f"EmbeddedVectorStore: Error ingesting document {params.file_path}: {e}")
            return 0

    async def search_memories(self, params: SearchParams) -> List[MemoryEntry]:
        """
        Search memories in the embedded store by content and importance (keyword-based).
        """
        if not self.initialized:
            await self.initialize()

        logger.debug(f"EmbeddedVectorStore: Searching memories for query: {params.query}")
        
        query_lower = params.query.lower()
        results: List[MemoryEntry] = []

        for entry in self.store:
            if params.memory_type and entry.memory_type != params.memory_type:
                continue
            
            if entry.importance >= params.min_importance and query_lower in entry.content.lower():
                results.append(entry)
        
        results.sort(key=lambda x: (x.importance, x.last_accessed), reverse=True)
        return results

    async def get_memory_stats(self) -> dict[str, Any]:
        """
        Get statistics about memory usage.
        """
        if not self.initialized:
            await self.initialize()

        logger.debug("EmbeddedVectorStore: Getting memory stats.")
        
        stats = {
            "total": len(self.store),
            "redis_memory": 0 # Not applicable for embedded store
        }
        for tier in ["working", "short_term", "long_term", "episodic", "semantic"]:
            stats[tier] = sum(1 for entry in self.store if entry.memory_type == tier)
        
        return stats

    async def close(self) -> None:
        """
        Close the embedded store and save to file.
        """
        if not self.initialized:
            return

        logger.info(f"EmbeddedVectorStore: Closing and saving to {self.STORE_FILE}.")
        try:
            data = [entry.to_dict() for entry in self.store]
            async with aiofiles.open(self.STORE_FILE, mode="w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))
            logger.info("EmbeddedVectorStore: Saved successfully.")
        except (json.JSONEncodeError, OSError) as e:
            logger.error(f"EmbeddedVectorStore: Error saving to file: {e}")
        
        self.initialized = False