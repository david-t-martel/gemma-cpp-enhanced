"""Performance optimization utilities for the RAG system.

This module provides high-performance batch operations, memory consolidation,
and performance monitoring for the RAG memory backend.
"""

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from gemma_cli.rag.memory import MemoryEntry, MemoryTier
from gemma_cli.rag.python_backend import PythonRAGBackend


class BatchEmbedder:
    """Efficient batch embedding generator with caching and queue-based processing.

    Processes multiple texts in single model calls for 10x performance improvement
    over sequential embedding generation.

    Attributes:
        model: Embedding model (SentenceTransformer or compatible)
        batch_size: Number of texts to process per batch
        cache_size: Maximum number of cached embeddings
        queue_timeout: Maximum time to wait for batch to fill (seconds)
    """

    def __init__(
        self,
        model: Any,
        batch_size: int = 32,
        cache_size: int = 1000,
        queue_timeout: float = 0.1,
    ) -> None:
        """
        Initialize batch embedder.

        Args:
            model: Embedding model with encode() method
            batch_size: Number of texts to process per batch (default 32)
            cache_size: Maximum number of cached embeddings (default 1000)
            queue_timeout: Maximum time to wait for batch to fill in seconds (default 0.1)
        """
        self.model = model
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.queue_timeout = queue_timeout

        # Embedding cache (LRU-style with deque for O(1) operations)
        self._cache: dict[str, npt.NDArray[np.float32]] = {}
        self._cache_keys: deque[str] = deque(maxlen=cache_size)

        # Background processing queue
        self._queue: list[tuple[str, asyncio.Future[npt.NDArray[np.float32]]]] = []
        self._queue_lock = asyncio.Lock()
        self._processor_task: Optional[asyncio.Task[None]] = None
        self._running = False

        # Statistics
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "batch_count": 0,
            "total_embeddings": 0,
            "total_time": 0.0,
            "avg_batch_size": 0.0,
        }

    async def embed_text(self, text: str) -> npt.NDArray[np.float32]:
        """
        Embed a single text with caching.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]

        self._stats["cache_misses"] += 1

        # If background processor is running, queue the request
        if self._running:
            future: asyncio.Future[npt.NDArray[np.float32]] = asyncio.Future()
            async with self._queue_lock:
                self._queue.append((text, future))
            return await future

        # Otherwise, process immediately
        embedding = self._generate_embedding(text)
        self._update_cache(cache_key, embedding)
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[npt.NDArray[np.float32]]:
        """
        Embed multiple texts in a single batch operation.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        start_time = time.perf_counter()

        # Check cache for each text
        results: list[Optional[npt.NDArray[np.float32]]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
                self._stats["cache_hits"] += 1
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
                self._stats["cache_misses"] += 1

        # Generate embeddings for uncached texts
        if uncached_texts:
            embeddings = self._generate_batch_embeddings(uncached_texts)

            for idx, embedding in zip(uncached_indices, embeddings):
                results[idx] = embedding
                cache_key = self._get_cache_key(texts[idx])
                self._update_cache(cache_key, embedding)

        # Update statistics
        elapsed = time.perf_counter() - start_time
        self._stats["batch_count"] += 1
        self._stats["total_embeddings"] += len(texts)
        self._stats["total_time"] += elapsed
        self._stats["avg_batch_size"] = self._stats["total_embeddings"] / self._stats["batch_count"]

        return [r for r in results if r is not None]  # Type guard

    async def start_background_processor(self) -> None:
        """Start background batch processing task."""
        if self._running:
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_queue())

    async def stop_background_processor(self) -> None:
        """Stop background batch processing task."""
        if not self._running:
            return

        self._running = False
        if self._processor_task:
            await self._processor_task
            self._processor_task = None

    async def _process_queue(self) -> None:
        """Background task that processes queued embedding requests in batches."""
        while self._running:
            try:
                # Wait for queue to fill or timeout
                await asyncio.sleep(self.queue_timeout)

                async with self._queue_lock:
                    if not self._queue:
                        continue

                    # Extract batch
                    batch = self._queue[: self.batch_size]
                    self._queue = self._queue[self.batch_size :]

                if not batch:
                    continue

                # Process batch
                texts = [text for text, _ in batch]
                futures = [future for _, future in batch]

                try:
                    embeddings = self._generate_batch_embeddings(texts)

                    # Fulfill futures and update cache
                    for text, future, embedding in zip(texts, futures, embeddings):
                        if not future.done():
                            future.set_result(embedding)
                        cache_key = self._get_cache_key(text)
                        self._update_cache(cache_key, embedding)

                except Exception as e:
                    # Propagate error to all futures
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in batch processor: {e}")

    def _generate_embedding(self, text: str) -> npt.NDArray[np.float32]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if hasattr(self.model, "encode"):
            return self.model.encode([text])[0].astype(np.float32)
        else:
            # Fallback for models without encode method
            raise NotImplementedError("Model must have encode() method")

    def _generate_batch_embeddings(self, texts: list[str]) -> list[npt.NDArray[np.float32]]:
        """
        Generate embeddings for multiple texts in a single call.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if hasattr(self.model, "encode"):
            embeddings = self.model.encode(texts)
            return [emb.astype(np.float32) for emb in embeddings]
        else:
            raise NotImplementedError("Model must have encode() method")

    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text.

        Args:
            text: Input text

        Returns:
            Cache key string
        """
        # Use hash of text for efficient lookup
        return str(hash(text))

    def _update_cache(self, key: str, embedding: npt.NDArray[np.float32]) -> None:
        """
        Update cache with new embedding (LRU eviction).

        Args:
            key: Cache key
            embedding: Embedding vector to cache
        """
        # Remove oldest if at capacity
        if len(self._cache) >= self.cache_size and key not in self._cache:
            if self._cache_keys:
                oldest_key = self._cache_keys[0]
                self._cache.pop(oldest_key, None)

        # Add to cache
        self._cache[key] = embedding
        if key in self._cache_keys:
            self._cache_keys.remove(key)
        self._cache_keys.append(key)

    def get_stats(self) -> dict[str, Any]:
        """
        Get embedding statistics.

        Returns:
            Dictionary with statistics
        """
        cache_hit_rate = 0.0
        total_requests = self._stats["cache_hits"] + self._stats["cache_misses"]
        if total_requests > 0:
            cache_hit_rate = self._stats["cache_hits"] / total_requests

        avg_time_per_embedding = 0.0
        if self._stats["total_embeddings"] > 0:
            avg_time_per_embedding = self._stats["total_time"] / self._stats["total_embeddings"]

        return {
            "cache_size": len(self._cache),
            "cache_capacity": self.cache_size,
            "cache_hit_rate": cache_hit_rate,
            "total_embeddings": self._stats["total_embeddings"],
            "batch_count": self._stats["batch_count"],
            "avg_batch_size": self._stats["avg_batch_size"],
            "avg_time_per_embedding_ms": avg_time_per_embedding * 1000,
            "queue_length": len(self._queue),
        }


class MemoryConsolidator:
    """Automatic memory tier consolidation based on importance and access patterns.

    Analyzes memory entries across tiers and promotes important/frequently accessed
    content to longer-lived tiers.

    Attributes:
        rag_backend: RAG backend instance
        promotion_threshold: Minimum relevance score for promotion (0.0 to 1.0)
        time_decay_factor: Factor for time-based decay
        consolidation_interval: Interval between consolidation runs (seconds)
    """

    def __init__(
        self,
        rag_backend: PythonRAGBackend,
        promotion_threshold: float = 0.75,
        time_decay_factor: float = 0.1,
        consolidation_interval: int = 300,
    ) -> None:
        """
        Initialize memory consolidator.

        Args:
            rag_backend: RAG backend instance
            promotion_threshold: Minimum relevance score for promotion (default 0.75)
            time_decay_factor: Factor for time-based decay (default 0.1)
            consolidation_interval: Seconds between consolidation runs (default 300)
        """
        self.rag_backend = rag_backend
        self.promotion_threshold = promotion_threshold
        self.time_decay_factor = time_decay_factor
        self.consolidation_interval = consolidation_interval

        # Tier promotion hierarchy
        self._tier_hierarchy = [
            MemoryTier.WORKING,
            MemoryTier.SHORT_TERM,
            MemoryTier.LONG_TERM,
            MemoryTier.EPISODIC,
            MemoryTier.SEMANTIC,
        ]

        # Background task
        self._consolidation_task: Optional[asyncio.Task[None]] = None
        self._running = False

        # Statistics
        self._stats = {
            "total_consolidations": 0,
            "total_promotions": 0,
            "promotions_by_tier": defaultdict(int),
            "last_consolidation": None,
            "avg_consolidation_time": 0.0,
        }

    async def run_consolidation(self) -> int:
        """
        Run a single consolidation pass across all tiers.

        Returns:
            Number of entries promoted
        """
        start_time = time.perf_counter()
        total_promoted = 0

        try:
            # Process each tier (except permanent semantic tier)
            for tier in self._tier_hierarchy[:-1]:
                candidates = await self.analyze_candidates(tier)

                for entry in candidates:
                    relevance = entry.calculate_relevance(self.time_decay_factor)

                    if relevance >= self.promotion_threshold:
                        next_tier = self._get_next_tier(tier)
                        if next_tier:
                            await self.promote_memory(entry.id, next_tier)
                            total_promoted += 1
                            self._stats["promotions_by_tier"][f"{tier}->{next_tier}"] += 1

            # Update statistics
            elapsed = time.perf_counter() - start_time
            self._stats["total_consolidations"] += 1
            self._stats["total_promotions"] += total_promoted
            self._stats["last_consolidation"] = datetime.utcnow().isoformat()

            # Update rolling average
            prev_avg = self._stats["avg_consolidation_time"]
            count = self._stats["total_consolidations"]
            self._stats["avg_consolidation_time"] = (prev_avg * (count - 1) + elapsed) / count

            print(f"Memory consolidation complete: {total_promoted} entries promoted in {elapsed:.2f}s")

        except Exception as e:
            print(f"Error during consolidation: {e}")

        return total_promoted

    async def start_background_task(self, interval: Optional[int] = None) -> None:
        """
        Start background consolidation task.

        Args:
            interval: Consolidation interval in seconds (uses default if None)
        """
        if self._running:
            return

        if interval is not None:
            self.consolidation_interval = interval

        self._running = True
        self._consolidation_task = asyncio.create_task(self._consolidation_loop())

    async def stop_background_task(self) -> None:
        """Stop background consolidation task."""
        if not self._running:
            return

        self._running = False
        if self._consolidation_task:
            await self._consolidation_task
            self._consolidation_task = None

    async def _consolidation_loop(self) -> None:
        """Background task that runs consolidation periodically."""
        while self._running:
            try:
                await asyncio.sleep(self.consolidation_interval)
                await self.run_consolidation()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in consolidation loop: {e}")

    async def analyze_candidates(self, tier: str) -> list[MemoryEntry]:
        """
        Analyze entries in a tier for promotion candidates.

        Args:
            tier: Memory tier to analyze

        Returns:
            List of memory entries that are candidates for promotion
        """
        if not self.rag_backend.async_redis_client:
            return []

        try:
            candidates: list[MemoryEntry] = []
            pattern = self.rag_backend.get_redis_key(tier)
            keys = await self.rag_backend._scan_keys(pattern)

            if keys:
                # Batch retrieve all entries
                async with self.rag_backend.async_redis_client.pipeline() as pipe:
                    for key in keys:
                        pipe.get(key)
                    values = await pipe.execute()

                # Parse and filter entries
                for data_bytes in values:
                    if data_bytes:
                        try:
                            import json

                            entry_dict = json.loads(data_bytes.decode())
                            entry = MemoryEntry.from_dict(entry_dict)

                            # Calculate relevance score
                            relevance = entry.calculate_relevance(self.time_decay_factor)

                            # Filter by threshold and access patterns
                            if relevance >= self.promotion_threshold or entry.access_count >= 5:
                                candidates.append(entry)

                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue

            return candidates

        except Exception as e:
            print(f"Error analyzing candidates in {tier}: {e}")
            return []

    async def promote_memory(self, entry_id: str, to_tier: str) -> bool:
        """
        Promote a memory entry to a higher tier.

        Args:
            entry_id: Entry ID to promote
            to_tier: Target tier

        Returns:
            True if promotion successful, False otherwise
        """
        if not self.rag_backend.async_redis_client:
            return False

        try:
            import json

            # Find entry in current tier
            for tier in self._tier_hierarchy:
                key = self.rag_backend.get_redis_key(tier, entry_id)
                data_bytes = await self.rag_backend.async_redis_client.get(key)

                if data_bytes:
                    # Parse entry
                    entry_dict = json.loads(data_bytes.decode())
                    entry = MemoryEntry.from_dict(entry_dict)

                    # Update tier
                    entry.memory_type = to_tier

                    # Store in new tier with new TTL
                    new_key = self.rag_backend.get_redis_key(to_tier, entry.id)
                    new_data = json.dumps(entry.to_dict())

                    config = self.rag_backend.TIER_CONFIG.get(
                        to_tier, self.rag_backend.TIER_CONFIG[MemoryTier.SHORT_TERM]
                    )

                    if config["ttl"]:
                        await self.rag_backend.async_redis_client.setex(
                            new_key, config["ttl"], new_data
                        )
                    else:
                        await self.rag_backend.async_redis_client.set(new_key, new_data)

                    # Delete from old tier
                    await self.rag_backend.async_redis_client.delete(key)

                    print(f"Promoted memory {entry_id[:8]} from {tier} to {to_tier}")
                    return True

            print(f"Memory entry {entry_id} not found")
            return False

        except Exception as e:
            print(f"Error promoting memory: {e}")
            return False

    def _get_next_tier(self, current_tier: str) -> Optional[str]:
        """
        Get next tier in hierarchy.

        Args:
            current_tier: Current tier

        Returns:
            Next tier name or None if already at top
        """
        try:
            idx = self._tier_hierarchy.index(current_tier)
            if idx < len(self._tier_hierarchy) - 1:
                return self._tier_hierarchy[idx + 1]
        except ValueError:
            pass
        return None

    def get_stats(self) -> dict[str, Any]:
        """
        Get consolidation statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_consolidations": self._stats["total_consolidations"],
            "total_promotions": self._stats["total_promotions"],
            "promotions_by_tier": dict(self._stats["promotions_by_tier"]),
            "last_consolidation": self._stats["last_consolidation"],
            "avg_consolidation_time_ms": self._stats["avg_consolidation_time"] * 1000,
            "running": self._running,
            "promotion_threshold": self.promotion_threshold,
        }


class PerformanceMonitor:
    """Performance monitoring for RAG operations.

    Tracks operation latencies, counts, memory usage, and connection pool statistics.
    Provides detailed performance reports and summaries.

    Attributes:
        enable_detailed: Enable detailed per-operation tracking
        track_percentiles: Calculate latency percentiles
    """

    def __init__(self, enable_detailed: bool = True, track_percentiles: bool = True) -> None:
        """
        Initialize performance monitor.

        Args:
            enable_detailed: Enable detailed per-operation tracking (default True)
            track_percentiles: Calculate latency percentiles (default True)
        """
        self.enable_detailed = enable_detailed
        self.track_percentiles = track_percentiles

        # Operation tracking
        self._operation_counts: dict[str, int] = defaultdict(int)
        self._operation_times: dict[str, list[float]] = defaultdict(list)
        self._operation_errors: dict[str, int] = defaultdict(int)

        # Metric tracking
        self._metrics: dict[str, list[float]] = defaultdict(list)

        # System metrics
        self._memory_usage: list[float] = []
        self._start_time = time.perf_counter()

        # Rolling window for recent metrics (last 1000 operations)
        self._max_history = 1000

    async def track_operation(self, op_name: str, duration: float) -> None:
        """
        Track an operation's execution time.

        Args:
            op_name: Operation name
            duration: Execution duration in seconds
        """
        self._operation_counts[op_name] += 1
        if self.enable_detailed:
            self._operation_times[op_name].append(duration)
            # Keep only recent history
            if len(self._operation_times[op_name]) > self._max_history:
                self._operation_times[op_name] = self._operation_times[op_name][-self._max_history :]

    def record_metric(self, metric: str, value: float) -> None:
        """
        Record a custom metric value.

        Args:
            metric: Metric name
            value: Metric value
        """
        self._metrics[metric].append(value)
        # Keep only recent history
        if len(self._metrics[metric]) > self._max_history:
            self._metrics[metric] = self._metrics[metric][-self._max_history :]

    def record_error(self, op_name: str) -> None:
        """
        Record an operation error.

        Args:
            op_name: Operation name that errored
        """
        self._operation_errors[op_name] += 1

    async def get_report(self) -> dict[str, Any]:
        """
        Get detailed performance report.

        Returns:
            Dictionary with performance metrics
        """
        report: dict[str, Any] = {
            "uptime_seconds": time.perf_counter() - self._start_time,
            "operations": {},
            "metrics": {},
            "errors": dict(self._operation_errors),
        }

        # Operation statistics
        for op_name, count in self._operation_counts.items():
            times = self._operation_times.get(op_name, [])
            if times:
                stats = {
                    "count": count,
                    "total_time_ms": sum(times) * 1000,
                    "avg_time_ms": (sum(times) / len(times)) * 1000,
                    "min_time_ms": min(times) * 1000,
                    "max_time_ms": max(times) * 1000,
                }

                # Calculate percentiles if enabled
                if self.track_percentiles and len(times) >= 10:
                    sorted_times = sorted(times)
                    stats["p50_ms"] = sorted_times[len(sorted_times) // 2] * 1000
                    stats["p95_ms"] = sorted_times[int(len(sorted_times) * 0.95)] * 1000
                    stats["p99_ms"] = sorted_times[int(len(sorted_times) * 0.99)] * 1000

                report["operations"][op_name] = stats
            else:
                report["operations"][op_name] = {"count": count}

        # Custom metrics
        for metric, values in self._metrics.items():
            if values:
                report["metrics"][metric] = {
                    "count": len(values),
                    "current": values[-1],
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }

        return report

    async def get_summary(self) -> str:
        """
        Get human-readable performance summary.

        Returns:
            Formatted summary string
        """
        report = await self.get_report()
        uptime = report["uptime_seconds"]

        lines = [
            "=== Performance Summary ===",
            f"Uptime: {uptime:.1f}s",
            "",
            "Top Operations:",
        ]

        # Sort operations by count
        sorted_ops = sorted(
            report["operations"].items(), key=lambda x: x[1].get("count", 0), reverse=True
        )

        for op_name, stats in sorted_ops[:10]:
            count = stats.get("count", 0)
            avg_time = stats.get("avg_time_ms", 0)
            lines.append(f"  {op_name}: {count} calls, {avg_time:.2f}ms avg")

        if report["errors"]:
            lines.append("")
            lines.append("Errors:")
            for op_name, count in report["errors"].items():
                lines.append(f"  {op_name}: {count} errors")

        return "\n".join(lines)

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._operation_counts.clear()
        self._operation_times.clear()
        self._operation_errors.clear()
        self._metrics.clear()
        self._memory_usage.clear()
        self._start_time = time.perf_counter()


class QueryOptimizer:
    """Query optimization with deduplication, caching, and prefetching.

    Attributes:
        cache_ttl: Time-to-live for cached results in seconds
        enable_prefetch: Enable smart prefetching
    """

    def __init__(self, cache_ttl: int = 300, enable_prefetch: bool = True) -> None:
        """
        Initialize query optimizer.

        Args:
            cache_ttl: Time-to-live for cached results in seconds (default 300)
            enable_prefetch: Enable smart prefetching (default True)
        """
        self.cache_ttl = cache_ttl
        self.enable_prefetch = enable_prefetch

        # Query result cache
        self._cache: dict[str, tuple[Any, float]] = {}

        # Query deduplication (in-flight requests)
        self._in_flight: dict[str, asyncio.Future[Any]] = {}

        # Statistics
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "dedup_hits": 0,
            "total_queries": 0,
        }

    async def execute_query(
        self, query_key: str, query_fn: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """
        Execute query with caching and deduplication.

        Args:
            query_key: Unique key for query
            query_fn: Async function to execute
            *args: Positional arguments for query_fn
            **kwargs: Keyword arguments for query_fn

        Returns:
            Query result
        """
        self._stats["total_queries"] += 1

        # Check cache
        if query_key in self._cache:
            result, timestamp = self._cache[query_key]
            if time.time() - timestamp < self.cache_ttl:
                self._stats["cache_hits"] += 1
                return result
            else:
                del self._cache[query_key]

        # Check if query is already in flight (deduplication)
        if query_key in self._in_flight:
            self._stats["dedup_hits"] += 1
            return await self._in_flight[query_key]

        # Execute query
        future: asyncio.Future[Any] = asyncio.Future()
        self._in_flight[query_key] = future

        try:
            result = await query_fn(*args, **kwargs)
            self._cache[query_key] = (result, time.time())
            self._stats["cache_misses"] += 1
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            del self._in_flight[query_key]

    def invalidate_cache(self, query_key: Optional[str] = None) -> None:
        """
        Invalidate cached queries.

        Args:
            query_key: Specific query to invalidate (None = all)
        """
        if query_key:
            self._cache.pop(query_key, None)
        else:
            self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """
        Get query optimization statistics.

        Returns:
            Dictionary with statistics
        """
        total_requests = self._stats["cache_hits"] + self._stats["cache_misses"]
        cache_hit_rate = 0.0
        if total_requests > 0:
            cache_hit_rate = self._stats["cache_hits"] / total_requests

        return {
            "total_queries": self._stats["total_queries"],
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "cache_hit_rate": cache_hit_rate,
            "dedup_hits": self._stats["dedup_hits"],
            "cache_size": len(self._cache),
            "in_flight_queries": len(self._in_flight),
        }
