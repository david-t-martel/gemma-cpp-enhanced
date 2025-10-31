"""Memory entry and tier definitions for RAG system."""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional
import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

class MemoryTier:
    """Represents memory tier types with TTL and capacity settings."""

    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class MemoryEntry:
    """Memory entry with content, metadata, and importance scoring."""

    def __init__(
        self, content: str, memory_type: str, importance: float = 0.5
    ) -> None:
        """
        Initialize a memory entry.

        Args:
            content: The text content to store
            memory_type: Memory tier type (from MemoryTier constants)
            importance: Importance score between 0.0 and 1.0
        """
        self.id = str(uuid.uuid4())
        self.content = content
        self.memory_type = memory_type
        self.importance = max(0.0, min(1.0, importance))  # Clamp to [0, 1]
        self.created_at = datetime.now(timezone.utc)
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count = 0
        self.tags: set[str] = set()
        self.metadata: dict[str, Any] = {}
        self.embedding: Optional[npt.NDArray[np.float32]] = None
        logger.debug(f"MemoryEntry created: id={self.id[:8]}..., type={memory_type}, importance={importance}")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for Redis storage.

        Returns:
            Dictionary representation of the memory entry
        """
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        """
        Create from dictionary loaded from Redis.

        Args:
            data: Dictionary representation of memory entry

        Returns:
            MemoryEntry instance
        """
        entry = cls(data["content"], data["memory_type"], data["importance"])
        entry.id = data["id"]
        entry.created_at = datetime.fromisoformat(data["created_at"])
        entry.last_accessed = datetime.fromisoformat(data["last_accessed"])
        entry.access_count = data["access_count"]
        entry.tags = set(data.get("tags", []))
        entry.metadata = data.get("metadata", {})

        if data.get("embedding"):
            entry.embedding = np.array(data["embedding"], dtype=np.float32)
        logger.debug(f"MemoryEntry loaded from dict: id={entry.id[:8]}..., type={entry.memory_type}")
        return entry

    def update_access(self) -> None:
        """Update access statistics."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1

    def add_tags(self, *tags: str) -> None:
        """
        Add tags to the entry.

        Args:
            *tags: Variable number of tag strings to add
        """
        self.tags.update(tags)
        logger.debug(f"Tags added to MemoryEntry {self.id[:8]}...: {tags}. All tags: {self.tags}")

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata to the entry.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
        logger.debug(f"Metadata added to MemoryEntry {self.id[:8]}...: {key}={value}")

    def calculate_relevance(self, time_decay_factor: float = 0.1) -> float:
        """
        Calculate relevance score based on importance, recency, and access frequency.

        Args:
            time_decay_factor: Factor for time-based decay (0.0 to 1.0)

        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Time-based decay
        age_seconds = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        age_days = age_seconds / 86400.0
        time_decay = max(0.0, 1.0 - (time_decay_factor * age_days))

        # Access frequency boost
        access_boost = min(1.0, self.access_count / 10.0)

        # Combined relevance
        relevance = (self.importance * 0.5) + (time_decay * 0.3) + (access_boost * 0.2)
        logger.debug(f"Relevance calculated for MemoryEntry {self.id[:8]}...: {relevance:.4f}")

        return max(0.0, min(1.0, relevance))
