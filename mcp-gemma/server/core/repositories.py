"""
Repository implementations following the Repository pattern.
Separates data access logic from business logic.
"""

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis

from .contracts import IMemoryRepository, IModelRepository, MemoryEntry, ModelInfo


class FileModelRepository(IModelRepository):
    """Repository for discovering models from filesystem."""

    def __init__(self, models_directory: str = "/c/codedev/llm/.models"):
        self.models_dir = Path(models_directory)
        self._cache = None
        self._cache_time = 0
        self._cache_ttl = 60  # Cache for 60 seconds

    def find_all(self) -> List[ModelInfo]:
        """Find all available models."""
        current_time = time.time()

        # Use cache if valid
        if self._cache and (current_time - self._cache_time) < self._cache_ttl:
            return self._cache

        models = []

        if self.models_dir.exists():
            # Search for different model formats
            patterns = ["*.sbs", "*.bin", "*.safetensors", "*.gguf"]

            for pattern in patterns:
                for model_file in self.models_dir.glob(pattern):
                    # Look for corresponding tokenizer
                    tokenizer_path = None
                    tokenizer_file = model_file.with_suffix(".spm")
                    if tokenizer_file.exists():
                        tokenizer_path = str(tokenizer_file)

                    models.append(
                        ModelInfo(
                            path=str(model_file),
                            name=model_file.stem,
                            size=model_file.stat().st_size,
                            type=model_file.suffix[1:],  # Remove the dot
                            tokenizer_path=tokenizer_path,
                        )
                    )

        # Update cache
        self._cache = models
        self._cache_time = current_time

        return models

    def find_by_name(self, name: str) -> Optional[ModelInfo]:
        """Find a model by name."""
        models = self.find_all()
        for model in models:
            if model.name == name:
                return model
        return None

    def find_by_path(self, path: str) -> Optional[ModelInfo]:
        """Find a model by path."""
        model_path = Path(path)
        if not model_path.exists():
            return None

        tokenizer_path = None
        tokenizer_file = model_path.with_suffix(".spm")
        if tokenizer_file.exists():
            tokenizer_path = str(tokenizer_file)

        return ModelInfo(
            path=str(model_path),
            name=model_path.stem,
            size=model_path.stat().st_size,
            type=model_path.suffix[1:],
            tokenizer_path=tokenizer_path,
        )


class RedisMemoryRepository(IMemoryRepository):
    """Repository for storing memory in Redis."""

    def __init__(self, redis_client: redis.Redis, prefix: str = "gemma_mcp"):
        self.client = redis_client
        self.prefix = prefix

    async def save(self, entry: MemoryEntry) -> None:
        """Save a memory entry."""
        key = f"{self.prefix}:memory:{entry.key}"
        data = {
            "content": entry.content,
            "metadata": entry.metadata,
            "timestamp": entry.timestamp,
            "id": entry.id,
        }
        self.client.set(key, json.dumps(data))

        # Update search index
        await self._update_search_index(entry)

    async def find_by_key(self, key: str) -> Optional[MemoryEntry]:
        """Find entry by key."""
        full_key = f"{self.prefix}:memory:{key}"
        data = self.client.get(full_key)

        if not data:
            return None

        entry_data = json.loads(data)
        return MemoryEntry(
            key=key,
            content=entry_data["content"],
            metadata=entry_data["metadata"],
            timestamp=entry_data["timestamp"],
            id=entry_data["id"],
        )

    async def find_by_query(self, query: str, limit: int) -> List[MemoryEntry]:
        """Find entries by query."""
        pattern = f"{self.prefix}:memory:*"
        keys = self.client.keys(pattern)

        results = []
        query_lower = query.lower()

        for redis_key in keys:
            if len(results) >= limit:
                break

            data = self.client.get(redis_key)
            if data:
                entry_data = json.loads(data)

                # Simple text matching
                if query_lower in entry_data["content"].lower():
                    key = redis_key.replace(f"{self.prefix}:memory:", "")
                    results.append(
                        MemoryEntry(
                            key=key,
                            content=entry_data["content"],
                            metadata=entry_data["metadata"],
                            timestamp=entry_data["timestamp"],
                            id=entry_data["id"],
                        )
                    )

        # Sort by relevance (simple implementation)
        results.sort(key=lambda x: self._calculate_relevance(query, x.content), reverse=True)
        return results

    async def delete_by_key(self, key: str) -> bool:
        """Delete entry by key."""
        full_key = f"{self.prefix}:memory:{key}"
        result = self.client.delete(full_key)
        return result > 0

    async def find_all_keys(self) -> List[str]:
        """Find all keys."""
        pattern = f"{self.prefix}:memory:*"
        keys = self.client.keys(pattern)
        return [key.replace(f"{self.prefix}:memory:", "") for key in keys]

    async def _update_search_index(self, entry: MemoryEntry):
        """Update search index for efficient querying."""
        # Extract keywords
        keywords = self._extract_keywords(entry.content)
        index_key = f"{self.prefix}:search_index"

        # Add to keyword sets
        for keyword in keywords:
            self.client.sadd(f"{index_key}:{keyword}", entry.key)

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content."""
        words = re.findall(r"\b[a-zA-Z]{3,}\b", content.lower())
        return list(set(words))[:50]  # Limit to 50 unique keywords

    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.0

        # Jaccard similarity
        intersection = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))

        return intersection / union if union > 0 else 0.0


class InMemoryRepository(IMemoryRepository):
    """In-memory repository for when Redis is not available."""

    def __init__(self):
        self.storage: Dict[str, MemoryEntry] = {}

    async def save(self, entry: MemoryEntry) -> None:
        """Save a memory entry."""
        self.storage[entry.key] = entry

    async def find_by_key(self, key: str) -> Optional[MemoryEntry]:
        """Find entry by key."""
        return self.storage.get(key)

    async def find_by_query(self, query: str, limit: int) -> List[MemoryEntry]:
        """Find entries by query."""
        results = []
        query_lower = query.lower()

        for entry in self.storage.values():
            if len(results) >= limit:
                break

            if query_lower in entry.content.lower():
                results.append(entry)

        # Sort by relevance
        results.sort(key=lambda x: self._calculate_relevance(query, x.content), reverse=True)
        return results

    async def delete_by_key(self, key: str) -> bool:
        """Delete entry by key."""
        if key in self.storage:
            del self.storage[key]
            return True
        return False

    async def find_all_keys(self) -> List[str]:
        """Find all keys."""
        return list(self.storage.keys())

    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.0

        intersection = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))

        return intersection / union if union > 0 else 0.0


class MemoryRepositoryFactory:
    """Factory for creating memory repositories."""

    @staticmethod
    def create(backend: str, **kwargs) -> IMemoryRepository:
        """Create a memory repository based on backend type."""
        if backend == "redis":
            redis_client = redis.Redis(
                host=kwargs.get("host", "localhost"),
                port=kwargs.get("port", 6379),
                db=kwargs.get("db", 0),
                decode_responses=True,
            )
            return RedisMemoryRepository(redis_client, kwargs.get("prefix", "gemma_mcp"))

        elif backend == "inmemory":
            return InMemoryRepository()

        else:
            raise ValueError(f"Unknown memory backend: {backend}")
