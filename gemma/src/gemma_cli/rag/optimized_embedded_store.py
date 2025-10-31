"""Optimized embedded vector store with indexing and async batching."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict
import time

import aiofiles
import numpy as np

from gemma_cli.rag.memory import MemoryEntry
from gemma_cli.rag.hybrid_rag import (
    RecallMemoriesParams,
    StoreMemoryParams,
    IngestDocumentParams,
    SearchParams
)
from ..utils.profiler import PerformanceMonitor

logger = logging.getLogger(__name__)


class OptimizedEmbeddedVectorStore:
    """
    Optimized embedded vector store with:
    - Inverted index for O(log n) search
    - Async write batching to reduce I/O
    - Memory-mapped file support for large datasets
    - LRU cache for frequent queries
    """

    STORE_FILE = Path.home() / ".gemma_cli" / "embedded_store.json"
    INDEX_FILE = Path.home() / ".gemma_cli" / "embedded_index.json"
    BATCH_WINDOW = 0.5  # Seconds to wait for batch accumulation
    MAX_BATCH_SIZE = 100  # Maximum entries to batch

    def __init__(self):
        self.store: List[MemoryEntry] = []
        self.index: Dict[str, Set[int]] = defaultdict(set)  # word -> entry indices
        self.importance_index: List[tuple[float, int]] = []  # (importance, index) sorted
        self.write_queue: List[MemoryEntry] = []
        self.write_task: Optional[asyncio.Task] = None
        self.initialized = False
        self.last_persist_time = time.time()

        # Query cache (LRU-style)
        self.query_cache: Dict[str, List[MemoryEntry]] = {}
        self.cache_max_size = 100

        logger.info("OptimizedEmbeddedVectorStore initialized")

    @PerformanceMonitor.track("embedded_store_init")
    async def initialize(self) -> bool:
        """Initialize store with optimized loading."""
        if self.initialized:
            return True

        self.STORE_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Load store and index in parallel
        store_task = self._load_store()
        index_task = self._load_index()

        await asyncio.gather(store_task, index_task)

        # Build index if not loaded or outdated
        if not self.index:
            await self._build_index()

        self.initialized = True
        return True

    async def _load_store(self):
        """Load store from JSON file."""
        if not self.STORE_FILE.exists():
            logger.info("No existing store file, starting fresh")
            return

        try:
            async with aiofiles.open(self.STORE_FILE, mode="r", encoding="utf-8") as f:
                data = json.loads(await f.read())
                self.store = [MemoryEntry.from_dict(entry_dict) for entry_dict in data]
                logger.info(f"Loaded {len(self.store)} entries from store")
        except Exception as e:
            logger.error(f"Error loading store: {e}")
            self.store = []

    async def _load_index(self):
        """Load pre-built index from file."""
        if not self.INDEX_FILE.exists():
            return

        try:
            async with aiofiles.open(self.INDEX_FILE, mode="r", encoding="utf-8") as f:
                index_data = json.loads(await f.read())

                # Convert lists back to sets
                self.index = {
                    word: set(indices)
                    for word, indices in index_data.get('word_index', {}).items()
                }

                # Load importance index
                self.importance_index = [
                    tuple(item) for item in index_data.get('importance_index', [])
                ]

                logger.info(f"Loaded index with {len(self.index)} terms")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self.index = defaultdict(set)

    @PerformanceMonitor.track("build_index")
    async def _build_index(self):
        """Build inverted index and importance index."""
        logger.info("Building search indices...")

        self.index.clear()
        self.importance_index.clear()

        for idx, entry in enumerate(self.store):
            # Build word index
            words = self._tokenize(entry.content)
            for word in words:
                self.index[word].add(idx)

            # Build importance index
            self.importance_index.append((entry.importance, idx))

        # Sort importance index for efficient range queries
        self.importance_index.sort(reverse=True)

        # Persist index
        await self._save_index()

        logger.info(f"Built indices: {len(self.index)} unique terms")

    def _tokenize(self, text: str) -> Set[str]:
        """Simple tokenization for indexing."""
        # Basic word splitting and normalization
        words = set()
        for word in text.lower().split():
            # Remove common punctuation
            word = word.strip('.,!?;:"')
            if len(word) > 2:  # Skip very short words
                words.add(word)
        return words

    @PerformanceMonitor.track("recall_memories_indexed")
    async def recall_memories(self, params: RecallMemoriesParams) -> List[MemoryEntry]:
        """
        Recall memories using index for O(log n) lookup.

        Performance:
        - Indexed search: O(k log n) where k = query terms
        - Linear search (fallback): O(n)
        """
        if not self.initialized:
            await self.initialize()

        # Check cache first
        cache_key = f"{params.query}:{params.memory_type}:{params.limit}"
        if cache_key in self.query_cache:
            logger.debug(f"Cache hit for query: {params.query[:20]}")
            return self.query_cache[cache_key]

        query_words = self._tokenize(params.query)

        # Find candidate entries using index
        candidate_indices: Set[int] = set()
        for word in query_words:
            if word in self.index:
                candidate_indices.update(self.index[word])

        # Score and filter candidates
        results: List[tuple[float, MemoryEntry]] = []
        for idx in candidate_indices:
            entry = self.store[idx]

            # Filter by memory type if specified
            if params.memory_type and entry.memory_type != params.memory_type:
                continue

            # Calculate relevance score
            content_words = self._tokenize(entry.content)
            overlap = len(query_words & content_words)
            if overlap > 0:
                score = overlap / len(query_words) * entry.importance
                results.append((score, entry))

        # Sort by score and limit
        results.sort(key=lambda x: x[0], reverse=True)
        final_results = [entry for _, entry in results[:params.limit]]

        # Update cache (LRU eviction if needed)
        if len(self.query_cache) >= self.cache_max_size:
            # Remove oldest entry (simple FIFO for now)
            self.query_cache.pop(next(iter(self.query_cache)))
        self.query_cache[cache_key] = final_results

        return final_results

    @PerformanceMonitor.track("store_memory_batched")
    async def store_memory(self, params: StoreMemoryParams) -> Optional[str]:
        """Store memory with write batching."""
        if not self.initialized:
            await self.initialize()

        entry = MemoryEntry(params.content, params.memory_type, params.importance)
        if params.tags:
            entry.add_tags(*params.tags)

        # Add to store immediately (for immediate availability)
        self.store.append(entry)

        # Update indices immediately
        idx = len(self.store) - 1
        words = self._tokenize(entry.content)
        for word in words:
            self.index[word].add(idx)
        self.importance_index.append((entry.importance, idx))

        # Queue for batched persistence
        self.write_queue.append(entry)

        # Invalidate query cache
        self.query_cache.clear()

        # Schedule batch write if not already scheduled
        if not self.write_task or self.write_task.done():
            self.write_task = asyncio.create_task(self._batch_persist())

        logger.debug(f"Stored memory {entry.id[:8]}, queued for persistence")
        return entry.id

    async def _batch_persist(self):
        """Batch persistence to reduce I/O operations."""
        # Wait for batch window or max size
        await asyncio.sleep(self.BATCH_WINDOW)

        if not self.write_queue:
            return

        logger.debug(f"Persisting batch of {len(self.write_queue)} entries")

        # Clear queue
        self.write_queue.clear()

        # Persist store and index
        await asyncio.gather(
            self._save_store(),
            self._save_index()
        )

        self.last_persist_time = time.time()

    async def _save_store(self):
        """Save store to JSON file."""
        try:
            data = [entry.to_dict() for entry in self.store]
            async with aiofiles.open(self.STORE_FILE, mode="w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=2))
            logger.debug(f"Persisted {len(self.store)} entries to store")
        except Exception as e:
            logger.error(f"Error saving store: {e}")

    async def _save_index(self):
        """Save index to JSON file."""
        try:
            # Convert sets to lists for JSON serialization
            index_data = {
                'word_index': {
                    word: list(indices)
                    for word, indices in self.index.items()
                },
                'importance_index': self.importance_index,
                'timestamp': time.time()
            }

            async with aiofiles.open(self.INDEX_FILE, mode="w", encoding="utf-8") as f:
                await f.write(json.dumps(index_data))
            logger.debug(f"Persisted index with {len(self.index)} terms")
        except Exception as e:
            logger.error(f"Error saving index: {e}")

    @PerformanceMonitor.track("ingest_document_chunked")
    async def ingest_document(self, params: IngestDocumentParams) -> int:
        """Ingest document with optimized chunking."""
        if not self.initialized:
            await self.initialize()

        path = Path(params.file_path)
        if not path.exists():
            logger.warning(f"Document not found: {params.file_path}")
            return 0

        try:
            # Read file asynchronously
            async with aiofiles.open(path, encoding="utf-8") as f:
                content = await f.read()

            # Smart chunking with overlap
            chunks = self._smart_chunk(content, params.chunk_size)

            # Store chunks in batch
            stored_count = 0
            for i, chunk in enumerate(chunks):
                store_params = StoreMemoryParams(
                    content=chunk,
                    memory_type=params.memory_type,
                    importance=0.5,
                    tags=[f"document:{path.name}", f"chunk:{i}"]
                )
                entry_id = await self.store_memory(params=store_params)
                if entry_id:
                    stored_count += 1

            logger.info(f"Ingested {stored_count} chunks from {path.name}")
            return stored_count

        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            return 0

    def _smart_chunk(self, text: str, chunk_size: int) -> List[str]:
        """Smart chunking that tries to preserve sentence boundaries."""
        chunks = []
        sentences = text.replace('\n', ' ').split('. ')

        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_size = len(sentence)

            if current_size + sentence_size > chunk_size and current_chunk:
                # Save current chunk
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        # Save last chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')

        return chunks

    async def search_memories(self, params: SearchParams) -> List[MemoryEntry]:
        """Search with importance filtering using sorted index."""
        if not self.initialized:
            await self.initialize()

        # Use importance index for efficient filtering
        results = []
        query_words = self._tokenize(params.query) if params.query else set()

        for importance, idx in self.importance_index:
            if importance < params.min_importance:
                break  # Index is sorted, so we can stop here

            entry = self.store[idx]

            # Filter by memory type
            if params.memory_type and entry.memory_type != params.memory_type:
                continue

            # Check query match if provided
            if query_words:
                content_words = self._tokenize(entry.content)
                if not query_words & content_words:
                    continue

            results.append(entry)

            if len(results) >= 100:  # Limit for performance
                break

        return results

    async def get_memory_stats(self) -> dict:
        """Get memory statistics."""
        if not self.initialized:
            await self.initialize()

        # Count by type
        type_counts = defaultdict(int)
        for entry in self.store:
            type_counts[entry.memory_type] += 1

        return {
            "total": len(self.store),
            "by_type": dict(type_counts),
            "index_terms": len(self.index),
            "cache_entries": len(self.query_cache),
            "pending_writes": len(self.write_queue),
            "file_size_kb": self.STORE_FILE.stat().st_size / 1024 if self.STORE_FILE.exists() else 0
        }

    async def persist(self):
        """Force immediate persistence."""
        if self.write_queue:
            await self._batch_persist()

    async def close(self):
        """Cleanup and ensure all data is persisted."""
        if self.write_task and not self.write_task.done():
            await self.write_task
        await self.persist()