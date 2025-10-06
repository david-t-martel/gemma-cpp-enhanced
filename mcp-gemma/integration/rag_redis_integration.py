"""
RAG-Redis integration module for MCP Gemma server.

This module provides enhanced memory management by integrating with the
existing RAG-Redis system for vector-based similarity search and
multi-tier memory management.
"""

import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add paths for imports
STATS_PATH = Path(__file__).parent.parent.parent / "stats"
RAG_REDIS_PATH = STATS_PATH / "rag-redis-system"
sys.path.insert(0, str(STATS_PATH))
sys.path.insert(0, str(RAG_REDIS_PATH))

try:
    import numpy as np
    import redis
    from sentence_transformers import SentenceTransformer

    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    logging.warning("Redis or sentence-transformers not available")

try:
    # Try to import from existing RAG-Redis system
    from src.rag.memory_tiers import MemoryTiers
    from src.rag.vector_store import VectorStore

    HAS_RAG_SYSTEM = True
except ImportError:
    HAS_RAG_SYSTEM = False
    logging.warning("RAG system not available")


class RAGRedisMemoryHandler:
    """Enhanced memory handler using RAG-Redis system."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        embedding_model: str = "all-MiniLM-L6-v2",
        memory_prefix: str = "gemma_mcp",
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.memory_prefix = memory_prefix
        self.logger = logging.getLogger(__name__)

        # Initialize Redis client
        self.redis_client = None
        if HAS_REDIS:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host, port=redis_port, db=redis_db, decode_responses=True
                )
                self.redis_client.ping()
                self.logger.info("Redis connection established")
            except Exception as e:
                self.logger.error(f"Redis connection failed: {e}")
                self.redis_client = None

        # Initialize embedding model
        self.embedding_model = None
        if HAS_REDIS:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                self.logger.info(f"Embedding model loaded: {embedding_model}")
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {e}")

        # Initialize memory tiers
        self.memory_tiers = None
        if HAS_RAG_SYSTEM and self.redis_client:
            try:
                self.memory_tiers = MemoryTiers(
                    redis_client=self.redis_client, prefix=memory_prefix
                )
                self.logger.info("Memory tiers initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize memory tiers: {e}")

        # Vector store for similarity search
        self.vector_store = None
        if HAS_RAG_SYSTEM and self.redis_client and self.embedding_model:
            try:
                self.vector_store = VectorStore(
                    redis_client=self.redis_client,
                    embedding_model=self.embedding_model,
                    prefix=f"{memory_prefix}:vectors",
                )
                self.logger.info("Vector store initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize vector store: {e}")

    def is_available(self) -> bool:
        """Check if RAG-Redis integration is available."""
        return self.redis_client is not None and self.embedding_model is not None and HAS_REDIS

    async def store_memory(
        self, key: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store memory with vector embeddings and tier management."""
        if not self.is_available():
            raise Exception("RAG-Redis integration not available")

        memory_id = str(uuid.uuid4())
        timestamp = time.time()

        memory_data = {
            "id": memory_id,
            "key": key,
            "content": content,
            "metadata": metadata or {},
            "timestamp": timestamp,
            "access_count": 0,
            "last_accessed": timestamp,
        }

        try:
            # Store in Redis
            full_key = f"{self.memory_prefix}:memory:{key}"
            self.redis_client.set(full_key, json.dumps(memory_data))

            # Generate and store embeddings for vector search
            if self.vector_store:
                embedding = self.embedding_model.encode(content)
                await self.vector_store.store(
                    memory_id,
                    embedding,
                    {"key": key, "content_preview": content[:200], "timestamp": timestamp},
                )

            # Add to appropriate memory tier
            if self.memory_tiers:
                await self.memory_tiers.add_to_working_memory(memory_id, memory_data)

            # Update search index
            await self._update_search_index(key, content, metadata)

            self.logger.debug(f"Stored memory: {key} -> {memory_id}")
            return memory_id

        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}")
            raise

    async def retrieve_memory(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory and update access patterns."""
        if not self.is_available():
            raise Exception("RAG-Redis integration not available")

        try:
            full_key = f"{self.memory_prefix}:memory:{key}"
            memory_data = self.redis_client.get(full_key)

            if not memory_data:
                return None

            data = json.loads(memory_data)

            # Update access patterns
            data["access_count"] = data.get("access_count", 0) + 1
            data["last_accessed"] = time.time()

            # Update in Redis
            self.redis_client.set(full_key, json.dumps(data))

            # Update memory tiers based on access pattern
            if self.memory_tiers:
                await self.memory_tiers.promote_memory(data["id"], data)

            return data

        except Exception as e:
            self.logger.error(f"Failed to retrieve memory: {e}")
            return None

    async def search_memory(
        self, query: str, limit: int = 10, similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search memory using vector similarity and text matching."""
        if not self.is_available():
            raise Exception("RAG-Redis integration not available")

        results = []

        try:
            # Vector-based similarity search
            if self.vector_store:
                query_embedding = self.embedding_model.encode(query)
                vector_results = await self.vector_store.search(
                    query_embedding, limit=limit, threshold=similarity_threshold
                )

                for result in vector_results:
                    memory_key = result.get("metadata", {}).get("key")
                    if memory_key:
                        memory_data = await self.retrieve_memory(memory_key)
                        if memory_data:
                            results.append(
                                {
                                    **memory_data,
                                    "similarity_score": result.get("score", 0.0),
                                    "search_type": "vector",
                                }
                            )

            # Text-based search as fallback/supplement
            if len(results) < limit:
                text_results = await self._text_search(query, limit - len(results))
                for result in text_results:
                    # Avoid duplicates
                    if not any(r["key"] == result["key"] for r in results):
                        result["search_type"] = "text"
                        results.append(result)

            # Sort by relevance (similarity score or text relevance)
            results.sort(
                key=lambda x: x.get("similarity_score", x.get("relevance_score", 0)), reverse=True
            )

            return results[:limit]

        except Exception as e:
            self.logger.error(f"Failed to search memory: {e}")
            return []

    async def _text_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback text-based search."""
        pattern = f"{self.memory_prefix}:memory:*"
        keys = self.redis_client.keys(pattern)

        results = []
        query_words = set(query.lower().split())

        for key in keys[: limit * 2]:  # Search more than needed to rank
            try:
                memory_data = self.redis_client.get(key)
                if memory_data:
                    data = json.loads(memory_data)
                    content = data.get("content", "").lower()

                    # Calculate text relevance
                    content_words = set(content.split())
                    common_words = query_words.intersection(content_words)

                    if common_words:
                        relevance = len(common_words) / len(query_words)
                        data["relevance_score"] = relevance
                        results.append(data)

            except Exception as e:
                self.logger.error(f"Error in text search for key {key}: {e}")
                continue

        # Sort by relevance and return top results
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return results[:limit]

    async def _update_search_index(
        self, key: str, content: str, metadata: Optional[Dict[str, Any]]
    ):
        """Update search index for efficient text-based queries."""
        try:
            # Extract keywords
            keywords = self._extract_keywords(content)

            # Store keywords in Redis sets for fast lookup
            for keyword in keywords:
                index_key = f"{self.memory_prefix}:index:{keyword}"
                self.redis_client.sadd(index_key, key)

            # Also index metadata if available
            if metadata:
                for meta_key, meta_value in metadata.items():
                    if isinstance(meta_value, str):
                        meta_keywords = self._extract_keywords(meta_value)
                        for keyword in meta_keywords:
                            index_key = f"{self.memory_prefix}:meta_index:{keyword}"
                            self.redis_client.sadd(index_key, key)

        except Exception as e:
            self.logger.error(f"Failed to update search index: {e}")

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for indexing."""
        import re

        # Simple keyword extraction - could be enhanced with NLP
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

        # Remove common stop words
        stop_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "this",
            "that",
            "these",
            "those",
            "was",
            "were",
            "been",
            "have",
            "has",
            "had",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "must",
        }

        keywords = [word for word in words if word not in stop_words]
        return list(set(keywords))[:50]  # Limit to 50 unique keywords

    async def consolidate_memory(self) -> Dict[str, Any]:
        """Perform memory consolidation using tier management."""
        if not self.memory_tiers:
            return {"status": "not_available"}

        try:
            result = await self.memory_tiers.consolidate()
            self.logger.info(f"Memory consolidation completed: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Memory consolidation failed: {e}")
            return {"status": "error", "error": str(e)}

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {"total_memories": 0, "tier_distribution": {}, "index_size": 0, "vector_count": 0}

        try:
            # Count total memories
            pattern = f"{self.memory_prefix}:memory:*"
            keys = self.redis_client.keys(pattern)
            stats["total_memories"] = len(keys)

            # Get tier distribution if available
            if self.memory_tiers:
                tier_stats = await self.memory_tiers.get_stats()
                stats["tier_distribution"] = tier_stats

            # Count index entries
            index_pattern = f"{self.memory_prefix}:index:*"
            index_keys = self.redis_client.keys(index_pattern)
            stats["index_size"] = len(index_keys)

            # Count vectors if available
            if self.vector_store:
                vector_stats = await self.vector_store.get_stats()
                stats["vector_count"] = vector_stats.get("count", 0)

        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            stats["error"] = str(e)

        return stats

    async def clear_memory(self, tier: Optional[str] = None) -> int:
        """Clear memory entries, optionally by tier."""
        try:
            if tier and self.memory_tiers:
                return await self.memory_tiers.clear_tier(tier)
            else:
                # Clear all memories
                pattern = f"{self.memory_prefix}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    return self.redis_client.delete(*keys)
                return 0

        except Exception as e:
            self.logger.error(f"Failed to clear memory: {e}")
            return 0

    async def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory storage and access patterns."""
        result = {"consolidated": 0, "optimized_vectors": 0, "rebuilt_indexes": 0}

        try:
            # Consolidate memory tiers
            if self.memory_tiers:
                consolidation_result = await self.consolidate_memory()
                result["consolidated"] = consolidation_result.get("moved_items", 0)

            # Optimize vector store
            if self.vector_store:
                optimization_result = await self.vector_store.optimize()
                result["optimized_vectors"] = optimization_result.get("optimized", 0)

            # Rebuild search indexes
            rebuild_result = await self._rebuild_search_index()
            result["rebuilt_indexes"] = rebuild_result

            return result

        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            result["error"] = str(e)
            return result

    async def _rebuild_search_index(self) -> int:
        """Rebuild the search index from existing memories."""
        try:
            # Clear existing indexes
            index_pattern = f"{self.memory_prefix}:index:*"
            index_keys = self.redis_client.keys(index_pattern)
            if index_keys:
                self.redis_client.delete(*index_keys)

            # Rebuild from existing memories
            pattern = f"{self.memory_prefix}:memory:*"
            keys = self.redis_client.keys(pattern)
            rebuilt_count = 0

            for key in keys:
                try:
                    memory_data = self.redis_client.get(key)
                    if memory_data:
                        data = json.loads(memory_data)
                        await self._update_search_index(
                            data.get("key", ""), data.get("content", ""), data.get("metadata", {})
                        )
                        rebuilt_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to rebuild index for {key}: {e}")

            return rebuilt_count

        except Exception as e:
            self.logger.error(f"Failed to rebuild search index: {e}")
            return 0
