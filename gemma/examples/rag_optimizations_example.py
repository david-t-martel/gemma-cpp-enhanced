#!/usr/bin/env python3
"""Example demonstrating RAG performance optimizations.

This script shows how to use BatchEmbedder, MemoryConsolidator, and PerformanceMonitor
to optimize RAG system performance.
"""

import asyncio
import time
from pathlib import Path

# Add project root to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemma_cli.rag import (
    BatchEmbedder,
    MemoryConsolidator,
    MemoryTier,
    PerformanceMonitor,
    PythonRAGBackend,
    QueryOptimizer,
)


async def demo_batch_embedder():
    """Demonstrate batch embedding with caching."""
    print("\n=== Batch Embedder Demo ===\n")

    # Check if sentence-transformers is available
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedder = BatchEmbedder(model, batch_size=16, cache_size=100)

        # Single embedding (cold cache)
        text = "This is a test query about machine learning."
        start = time.perf_counter()
        embedding = await embedder.embed_text(text)
        elapsed = time.perf_counter() - start
        print(f"Single embedding (cold): {elapsed*1000:.2f}ms, shape: {embedding.shape}")

        # Same embedding (cache hit)
        start = time.perf_counter()
        embedding = await embedder.embed_text(text)
        elapsed = time.perf_counter() - start
        print(f"Single embedding (cached): {elapsed*1000:.2f}ms, shape: {embedding.shape}")

        # Batch embeddings
        texts = [
            "Machine learning is fascinating",
            "Deep learning uses neural networks",
            "Natural language processing",
            "Computer vision applications",
            "Reinforcement learning agents",
            "Transformer models are powerful",
            "BERT and GPT architectures",
            "Fine-tuning pre-trained models",
        ]

        start = time.perf_counter()
        embeddings = await embedder.embed_batch(texts)
        elapsed = time.perf_counter() - start
        print(f"\nBatch embedding ({len(texts)} texts): {elapsed*1000:.2f}ms")
        print(f"Average per text: {elapsed*1000/len(texts):.2f}ms")

        # Show statistics
        stats = embedder.get_stats()
        print(f"\nEmbedder Statistics:")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
        print(f"  Total embeddings: {stats['total_embeddings']}")
        print(f"  Average batch size: {stats['avg_batch_size']:.1f}")
        print(f"  Average time per embedding: {stats['avg_time_per_embedding_ms']:.2f}ms")

        # Background processor demo
        print("\n--- Background Processor Demo ---")
        await embedder.start_background_processor()

        # Queue multiple requests
        tasks = [embedder.embed_text(f"Background query {i}") for i in range(20)]
        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

        print(f"20 background embeddings: {elapsed*1000:.2f}ms")
        print(f"Average per embedding: {elapsed*1000/len(results):.2f}ms")

        await embedder.stop_background_processor()

    except ImportError:
        print("sentence-transformers not installed. Skipping batch embedder demo.")
        print("Install with: pip install sentence-transformers")


async def demo_memory_consolidator():
    """Demonstrate memory tier consolidation."""
    print("\n\n=== Memory Consolidator Demo ===\n")

    # Initialize RAG backend
    backend = PythonRAGBackend(redis_host="localhost", redis_port=6380)
    initialized = await backend.initialize()

    if not initialized:
        print("Redis not available. Skipping consolidator demo.")
        print("Start Redis with: redis-server --port 6380")
        return

    # Create consolidator
    consolidator = MemoryConsolidator(
        backend, promotion_threshold=0.7, time_decay_factor=0.05, consolidation_interval=10
    )

    # Add some test memories
    print("Adding test memories...")
    for i in range(10):
        importance = 0.5 + (i / 20)  # Varying importance
        await backend.store_memory(
            f"Test memory {i}: important content about topic {i}",
            MemoryTier.WORKING,
            importance=importance,
            tags=[f"test{i}", "demo"],
        )

    # Analyze candidates
    print("\nAnalyzing promotion candidates...")
    candidates = await consolidator.analyze_candidates(MemoryTier.WORKING)
    print(f"Found {len(candidates)} candidates for promotion")

    for entry in candidates[:3]:
        relevance = entry.calculate_relevance(0.05)
        print(f"  {entry.id[:8]}: importance={entry.importance:.2f}, relevance={relevance:.2f}")

    # Run consolidation
    print("\nRunning consolidation...")
    promoted = await consolidator.run_consolidation()
    print(f"Promoted {promoted} entries")

    # Show statistics
    stats = consolidator.get_stats()
    print(f"\nConsolidator Statistics:")
    print(f"  Total consolidations: {stats['total_consolidations']}")
    print(f"  Total promotions: {stats['total_promotions']}")
    print(f"  Average time: {stats['avg_consolidation_time_ms']:.2f}ms")
    print(f"  Promotions by tier: {stats['promotions_by_tier']}")

    # Background consolidation demo
    print("\n--- Background Consolidation Demo ---")
    print("Starting background consolidation (10 second interval)...")
    await consolidator.start_background_task(interval=10)

    # Add more memories over time
    for i in range(5):
        await backend.store_memory(
            f"Background memory {i}",
            MemoryTier.WORKING,
            importance=0.8,
            tags=["background"],
        )
        await asyncio.sleep(2)

    # Wait for consolidation
    await asyncio.sleep(12)

    await consolidator.stop_background_task()

    # Final stats
    final_stats = await backend.get_memory_stats()
    print(f"\nFinal Memory Stats:")
    for tier, count in final_stats.items():
        if tier != "total" and tier != "redis_memory":
            print(f"  {tier}: {count} entries")

    await backend.close()


async def demo_performance_monitor():
    """Demonstrate performance monitoring."""
    print("\n\n=== Performance Monitor Demo ===\n")

    monitor = PerformanceMonitor(enable_detailed=True, track_percentiles=True)

    # Simulate various operations
    operations = [
        ("store_memory", 0.002),
        ("recall_memories", 0.015),
        ("embed_text", 0.005),
        ("consolidate", 0.100),
        ("store_memory", 0.003),
        ("recall_memories", 0.012),
        ("embed_text", 0.004),
        ("recall_memories", 0.018),
        ("store_memory", 0.002),
        ("embed_text", 0.006),
    ]

    print("Tracking operations...")
    for op_name, duration in operations:
        await monitor.track_operation(op_name, duration)
        # Simulate error rate
        if duration > 0.05:
            monitor.record_error(op_name)

    # Record custom metrics
    monitor.record_metric("cache_hit_rate", 0.85)
    monitor.record_metric("memory_usage_mb", 128.5)
    monitor.record_metric("redis_connections", 5)

    # Get detailed report
    report = await monitor.get_report()
    print("\nDetailed Performance Report:")
    print(f"Uptime: {report['uptime_seconds']:.2f}s")
    print(f"\nOperations:")
    for op_name, stats in report["operations"].items():
        print(f"  {op_name}:")
        print(f"    Count: {stats['count']}")
        print(f"    Average: {stats.get('avg_time_ms', 0):.3f}ms")
        if "p95_ms" in stats:
            print(f"    P95: {stats['p95_ms']:.3f}ms")

    print(f"\nCustom Metrics:")
    for metric, values in report["metrics"].items():
        print(f"  {metric}: {values['current']:.2f} (avg: {values['avg']:.2f})")

    if report["errors"]:
        print(f"\nErrors:")
        for op_name, count in report["errors"].items():
            print(f"  {op_name}: {count}")

    # Get summary
    summary = await monitor.get_summary()
    print(f"\n{summary}")


async def demo_query_optimizer():
    """Demonstrate query optimization."""
    print("\n\n=== Query Optimizer Demo ===\n")

    optimizer = QueryOptimizer(cache_ttl=60, enable_prefetch=True)

    # Simulate expensive query function
    async def expensive_query(query: str) -> list[str]:
        await asyncio.sleep(0.1)  # Simulate I/O
        return [f"Result for {query}", f"Another result for {query}"]

    # First query (cache miss)
    print("Executing query (cold)...")
    start = time.perf_counter()
    result1 = await optimizer.execute_query("test_query_1", expensive_query, "machine learning")
    elapsed1 = time.perf_counter() - start
    print(f"Cold query: {elapsed1*1000:.2f}ms, results: {len(result1)}")

    # Same query (cache hit)
    print("\nExecuting same query (cached)...")
    start = time.perf_counter()
    result2 = await optimizer.execute_query("test_query_1", expensive_query, "machine learning")
    elapsed2 = time.perf_counter() - start
    print(f"Cached query: {elapsed2*1000:.2f}ms, results: {len(result2)}")
    print(f"Speedup: {elapsed1/elapsed2:.1f}x")

    # Deduplication test (concurrent identical queries)
    print("\nTesting query deduplication...")
    start = time.perf_counter()
    tasks = [
        optimizer.execute_query("dedup_query", expensive_query, "deep learning")
        for _ in range(5)
    ]
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start
    print(f"5 concurrent identical queries: {elapsed*1000:.2f}ms")
    print(f"Without deduplication would take: {0.1*5*1000:.2f}ms")
    print(f"Speedup: {0.5/elapsed:.1f}x")

    # Show statistics
    stats = optimizer.get_stats()
    print(f"\nOptimizer Statistics:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Deduplication hits: {stats['dedup_hits']}")


async def demo_integrated_performance():
    """Demonstrate integrated performance optimizations."""
    print("\n\n=== Integrated Performance Demo ===\n")

    # Initialize components
    backend = PythonRAGBackend(redis_host="localhost", redis_port=6380)
    initialized = await backend.initialize()

    if not initialized:
        print("Redis not available. Skipping integrated demo.")
        return

    monitor = PerformanceMonitor()
    query_optimizer = QueryOptimizer(cache_ttl=300)

    print("Running integrated workload...")

    # Store memories with monitoring
    for i in range(20):
        start = time.perf_counter()
        await backend.store_memory(
            f"Document {i}: This is important content about topic {i}",
            MemoryTier.LONG_TERM,
            importance=0.6 + (i / 50),
        )
        elapsed = time.perf_counter() - start
        await monitor.track_operation("store_memory", elapsed)

    # Optimized recall with caching
    async def cached_recall(query: str) -> list:
        start = time.perf_counter()
        results = await backend.recall_memories(query, limit=5)
        elapsed = time.perf_counter() - start
        await monitor.track_operation("recall_memories", elapsed)
        return results

    # Execute queries with caching
    queries = [
        "important content",
        "topic information",
        "important content",  # Duplicate
        "document details",
        "important content",  # Duplicate
    ]

    print(f"\nExecuting {len(queries)} queries (with duplicates)...")
    for query in queries:
        await query_optimizer.execute_query(f"recall:{query}", cached_recall, query)

    # Show combined statistics
    print("\n--- Performance Report ---")
    summary = await monitor.get_summary()
    print(summary)

    print("\n--- Query Optimization ---")
    opt_stats = query_optimizer.get_stats()
    print(f"Cache hit rate: {opt_stats['cache_hit_rate']:.2%}")
    print(f"Queries served from cache: {opt_stats['cache_hits']}")

    await backend.close()


async def main():
    """Run all demos."""
    print("=" * 70)
    print("RAG Performance Optimization Demonstrations")
    print("=" * 70)

    try:
        await demo_batch_embedder()
        await demo_memory_consolidator()
        await demo_performance_monitor()
        await demo_query_optimizer()
        await demo_integrated_performance()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
