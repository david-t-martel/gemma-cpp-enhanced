"""Unit tests for RAG performance optimizations."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from gemma_cli.rag.optimizations import (
    BatchEmbedder,
    MemoryConsolidator,
    PerformanceMonitor,
    QueryOptimizer,
)


class TestBatchEmbedder:
    """Tests for BatchEmbedder class."""

    @pytest.fixture
    def mock_model(self):
        """Create mock embedding model."""
        model = MagicMock()
        model.encode = MagicMock(
            side_effect=lambda texts: [np.random.rand(384).astype(np.float32) for _ in texts]
        )
        return model

    @pytest.fixture
    def embedder(self, mock_model):
        """Create BatchEmbedder instance."""
        return BatchEmbedder(mock_model, batch_size=4, cache_size=10)

    @pytest.mark.asyncio
    async def test_single_embedding(self, embedder):
        """Test single text embedding."""
        text = "test query"
        embedding = await embedder.embed_text(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert embedding.shape == (384,)

    @pytest.mark.asyncio
    async def test_embedding_cache(self, embedder):
        """Test embedding caching."""
        text = "cached query"

        # First call (cache miss)
        embedding1 = await embedder.embed_text(text)
        stats1 = embedder.get_stats()
        assert stats1["cache_misses"] == 1

        # Second call (cache hit)
        embedding2 = await embedder.embed_text(text)
        stats2 = embedder.get_stats()
        assert stats2["cache_hits"] == 1

        # Results should be identical
        np.testing.assert_array_equal(embedding1, embedding2)

    @pytest.mark.asyncio
    async def test_batch_embedding(self, embedder):
        """Test batch embedding."""
        texts = ["text 1", "text 2", "text 3", "text 4"]
        embeddings = await embedder.embed_batch(texts)

        assert len(embeddings) == len(texts)
        for emb in embeddings:
            assert isinstance(emb, np.ndarray)
            assert emb.dtype == np.float32
            assert emb.shape == (384,)

    @pytest.mark.asyncio
    async def test_batch_with_cache(self, embedder):
        """Test batch embedding with partial cache hits."""
        # Cache some texts
        await embedder.embed_text("text 1")
        await embedder.embed_text("text 2")

        # Batch with mixed cached/uncached
        texts = ["text 1", "text 2", "text 3", "text 4"]
        embeddings = await embedder.embed_batch(texts)

        stats = embedder.get_stats()
        assert stats["cache_hits"] == 2  # text 1, text 2
        assert len(embeddings) == len(texts)

    @pytest.mark.asyncio
    async def test_background_processor(self, embedder):
        """Test background batch processor."""
        await embedder.start_background_processor()

        # Queue multiple requests
        tasks = [embedder.embed_text(f"background text {i}") for i in range(10)]
        embeddings = await asyncio.gather(*tasks)

        assert len(embeddings) == 10
        for emb in embeddings:
            assert isinstance(emb, np.ndarray)

        await embedder.stop_background_processor()

    def test_get_stats(self, embedder):
        """Test statistics reporting."""
        stats = embedder.get_stats()

        assert "cache_size" in stats
        assert "cache_hit_rate" in stats
        assert "total_embeddings" in stats
        assert "avg_time_per_embedding_ms" in stats


class TestMemoryConsolidator:
    """Tests for MemoryConsolidator class."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock RAG backend."""
        backend = MagicMock()
        backend.async_redis_client = MagicMock()
        backend.TIER_CONFIG = {
            "working": {"ttl": 900, "max_size": 15},
            "short_term": {"ttl": 3600, "max_size": 100},
            "long_term": {"ttl": 2592000, "max_size": 10000},
        }
        backend.get_redis_key = MagicMock(return_value="test:key")
        backend._scan_keys = AsyncMock(return_value=[])
        return backend

    @pytest.fixture
    def consolidator(self, mock_backend):
        """Create MemoryConsolidator instance."""
        return MemoryConsolidator(mock_backend, promotion_threshold=0.7)

    @pytest.mark.asyncio
    async def test_analyze_candidates(self, consolidator, mock_backend):
        """Test candidate analysis."""
        mock_backend._scan_keys = AsyncMock(return_value=[])
        candidates = await consolidator.analyze_candidates("working")

        assert isinstance(candidates, list)

    @pytest.mark.asyncio
    async def test_run_consolidation(self, consolidator):
        """Test consolidation run."""
        promoted = await consolidator.run_consolidation()

        assert isinstance(promoted, int)
        assert promoted >= 0

    @pytest.mark.asyncio
    async def test_background_task(self, consolidator):
        """Test background consolidation task."""
        await consolidator.start_background_task(interval=1)
        assert consolidator._running

        await asyncio.sleep(0.1)

        await consolidator.stop_background_task()
        assert not consolidator._running

    def test_get_stats(self, consolidator):
        """Test statistics reporting."""
        stats = consolidator.get_stats()

        assert "total_consolidations" in stats
        assert "total_promotions" in stats
        assert "avg_consolidation_time_ms" in stats
        assert "running" in stats


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create PerformanceMonitor instance."""
        return PerformanceMonitor(enable_detailed=True, track_percentiles=True)

    @pytest.mark.asyncio
    async def test_track_operation(self, monitor):
        """Test operation tracking."""
        await monitor.track_operation("test_op", 0.1)
        await monitor.track_operation("test_op", 0.2)

        report = await monitor.get_report()
        assert "test_op" in report["operations"]
        assert report["operations"]["test_op"]["count"] == 2

    def test_record_metric(self, monitor):
        """Test metric recording."""
        monitor.record_metric("test_metric", 100.0)
        monitor.record_metric("test_metric", 200.0)

        report = asyncio.run(monitor.get_report())
        assert "test_metric" in report["metrics"]
        assert report["metrics"]["test_metric"]["count"] == 2

    def test_record_error(self, monitor):
        """Test error recording."""
        monitor.record_error("failing_op")
        monitor.record_error("failing_op")

        report = asyncio.run(monitor.get_report())
        assert "failing_op" in report["errors"]
        assert report["errors"]["failing_op"] == 2

    @pytest.mark.asyncio
    async def test_get_report(self, monitor):
        """Test detailed report generation."""
        await monitor.track_operation("op1", 0.05)
        await monitor.track_operation("op2", 0.10)
        monitor.record_metric("metric1", 50.0)

        report = await monitor.get_report()

        assert "uptime_seconds" in report
        assert "operations" in report
        assert "metrics" in report
        assert "errors" in report

    @pytest.mark.asyncio
    async def test_get_summary(self, monitor):
        """Test summary generation."""
        await monitor.track_operation("test_op", 0.1)
        summary = await monitor.get_summary()

        assert isinstance(summary, str)
        assert "Performance Summary" in summary

    def test_reset_stats(self, monitor):
        """Test statistics reset."""
        asyncio.run(monitor.track_operation("test", 0.1))
        monitor.reset_stats()

        report = asyncio.run(monitor.get_report())
        assert len(report["operations"]) == 0


class TestQueryOptimizer:
    """Tests for QueryOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create QueryOptimizer instance."""
        return QueryOptimizer(cache_ttl=60, enable_prefetch=True)

    @pytest.mark.asyncio
    async def test_execute_query_cache_miss(self, optimizer):
        """Test query execution with cache miss."""

        async def mock_query():
            await asyncio.sleep(0.01)
            return "result"

        result = await optimizer.execute_query("test_key", mock_query)
        assert result == "result"

        stats = optimizer.get_stats()
        assert stats["cache_misses"] == 1

    @pytest.mark.asyncio
    async def test_execute_query_cache_hit(self, optimizer):
        """Test query execution with cache hit."""

        async def mock_query():
            return "result"

        # First call (cache miss)
        await optimizer.execute_query("test_key", mock_query)

        # Second call (cache hit)
        result = await optimizer.execute_query("test_key", mock_query)
        assert result == "result"

        stats = optimizer.get_stats()
        assert stats["cache_hits"] == 1

    @pytest.mark.asyncio
    async def test_query_deduplication(self, optimizer):
        """Test deduplication of in-flight queries."""

        async def slow_query():
            await asyncio.sleep(0.1)
            return "result"

        # Execute multiple concurrent identical queries
        tasks = [optimizer.execute_query("dedup_key", slow_query) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        assert all(r == "result" for r in results)
        stats = optimizer.get_stats()
        # Some should be deduplicated
        assert stats["dedup_hits"] > 0

    def test_invalidate_cache(self, optimizer):
        """Test cache invalidation."""

        async def setup():
            async def mock_query():
                return "result"

            await optimizer.execute_query("key1", mock_query)
            await optimizer.execute_query("key2", mock_query)

        asyncio.run(setup())

        # Invalidate specific key
        optimizer.invalidate_cache("key1")
        stats = optimizer.get_stats()
        assert stats["cache_size"] == 1

        # Invalidate all
        optimizer.invalidate_cache()
        stats = optimizer.get_stats()
        assert stats["cache_size"] == 0

    def test_get_stats(self, optimizer):
        """Test statistics reporting."""

        async def setup():
            async def mock_query():
                return "result"

            await optimizer.execute_query("test", mock_query)

        asyncio.run(setup())
        stats = optimizer.get_stats()

        assert "total_queries" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "cache_hit_rate" in stats
        assert "dedup_hits" in stats
