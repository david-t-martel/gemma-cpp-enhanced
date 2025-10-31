"""Example usage of the Hybrid RAG Manager.

This script demonstrates all the key features of the RAG adapter:
- Automatic backend selection
- Memory operations (store, recall, search)
- Document ingestion
- Performance monitoring
- Health checking
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.gemma_cli.rag import (
    DocumentMetadata,
    HybridRAGManager,
    MemoryType,
)


async def basic_usage_example():
    """Basic usage example."""
    print("=== Basic Usage Example ===\n")

    # Initialize manager with automatic backend selection
    manager = HybridRAGManager()
    await manager.initialize()

    print(f"Active backend: {manager.get_active_backend().value}\n")

    # Store some memories
    print("Storing memories...")
    mem1_id = await manager.store_memory(
        "Python is a high-level programming language",
        memory_type=MemoryType.LONG_TERM,
        importance=0.8,
        tags=["python", "programming"],
    )
    print(f"  Stored memory 1: {mem1_id}")

    mem2_id = await manager.store_memory(
        "Machine learning uses statistical algorithms",
        memory_type=MemoryType.LONG_TERM,
        importance=0.9,
        tags=["ml", "ai"],
    )
    print(f"  Stored memory 2: {mem2_id}\n")

    # Recall memories
    print("Recalling memories about 'Python'...")
    memories = await manager.recall_memories("Python", limit=5)
    for i, memory in enumerate(memories, 1):
        print(f"  {i}. {memory.content[:50]}... (importance: {memory.importance})")
    print()

    # Search with scoring
    print("Searching memories with similarity scoring...")
    results = await manager.search_memories("machine learning", min_importance=0.5)
    for i, result in enumerate(results, 1):
        print(f"  {i}. Score: {result.score:.2f} - {result.content[:50]}...")
    print()

    # Get stats
    print("Backend statistics:")
    stats = manager.get_backend_stats()
    for backend, stat in stats.items():
        if stat["successful_calls"] > 0:
            print(f"  {backend.value}:")
            print(f"    Successful calls: {stat['successful_calls']}")
            print(f"    Avg latency: {stat['avg_latency_ms']:.2f}ms")
            print(f"    Success rate: {stat['success_rate']:.1%}")
    print()

    # Cleanup
    await manager.close()
    print("Manager closed\n")


async def document_ingestion_example():
    """Document ingestion example."""
    print("=== Document Ingestion Example ===\n")

    manager = HybridRAGManager()
    await manager.initialize()

    # Create a temporary document
    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        suffix=".txt",
        encoding="utf-8",
    ) as f:
        content = """
        The Hybrid RAG Manager is a sophisticated backend adapter that provides
        intelligent selection between MCP, FFI, and Python implementations.

        It automatically chooses the best available backend with a fallback chain:
        1. MCP client (best performance + isolation)
        2. FFI bindings (direct Rust integration via PyO3)
        3. Python fallback (pure Python, always available)

        The manager provides consistent error handling, performance metrics,
        and health monitoring across all backend implementations.
        """
        f.write(content)
        temp_path = Path(f.name)

    try:
        print(f"Ingesting document: {temp_path.name}")

        doc_id = await manager.ingest_document(
            temp_path,
            metadata=DocumentMetadata(
                title="RAG Manager Documentation",
                source="example_script",
                doc_type="markdown",
                tags=["docs", "rag", "hybrid"],
                importance=0.9,
            ),
            chunk_size=200,
        )

        print(f"  Document ingested: {doc_id}\n")

        # Search the ingested document
        print("Searching ingested content...")
        results = await manager.search_memories("backend selection", min_importance=0.7)
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. Score: {result.score:.2f}")
            print(f"     {result.content[:80]}...\n")

    finally:
        temp_path.unlink()
        await manager.close()


async def health_monitoring_example():
    """Health monitoring example."""
    print("=== Health Monitoring Example ===\n")

    manager = HybridRAGManager()
    await manager.initialize()

    # Perform some operations
    for i in range(5):
        await manager.store_memory(f"Test memory {i}", importance=0.5)

    # Check health
    print("Health check:")
    health = await manager.health_check()
    print(f"  Status: {health['status']}")
    print(f"  Active backend: {health['active_backend']}")
    print(f"  Avg latency: {health['performance']['avg_latency_ms']:.2f}ms")
    print(f"  Success rate: {health['performance']['success_rate']:.1%}")
    print(f"  Total calls: {health['performance']['total_calls']}")
    print()

    await manager.close()


async def backend_preference_example():
    """Backend preference example."""
    print("=== Backend Preference Example ===\n")

    # Try to prefer MCP backend (will fallback if unavailable)
    from src.gemma_cli.rag.adapter import BackendType

    print("Attempting to use MCP backend...")
    manager = HybridRAGManager(prefer_backend=BackendType.MCP)
    await manager.initialize()

    active = manager.get_active_backend()
    print(f"  Active backend: {active.value}")

    if active == BackendType.MCP:
        print("  MCP backend is available!")
    else:
        print(f"  Fell back to {active.value} backend")
    print()

    await manager.close()


async def memory_tiers_example():
    """Memory tiers example."""
    print("=== Memory Tiers Example ===\n")

    manager = HybridRAGManager()
    await manager.initialize()

    print("Storing memories in different tiers...")

    # Working memory - immediate context
    await manager.store_memory(
        "Current task: implementing RAG adapter",
        memory_type=MemoryType.WORKING,
        importance=0.9,
    )
    print("  [OK] Working memory")

    # Short-term - recent interactions
    await manager.store_memory(
        "User asked about backend selection",
        memory_type=MemoryType.SHORT_TERM,
        importance=0.7,
    )
    print("  [OK] Short-term memory")

    # Long-term - consolidated facts
    await manager.store_memory(
        "RAG stands for Retrieval-Augmented Generation",
        memory_type=MemoryType.LONG_TERM,
        importance=0.8,
    )
    print("  [OK] Long-term memory")

    # Episodic - event sequences
    await manager.store_memory(
        "User session started at 10:00 AM",
        memory_type=MemoryType.EPISODIC,
        importance=0.6,
    )
    print("  [OK] Episodic memory")

    # Semantic - concept relationships
    await manager.store_memory(
        "MCP provides better performance than Python",
        memory_type=MemoryType.SEMANTIC,
        importance=0.7,
    )
    print("  [OK] Semantic memory\n")

    # Query specific memory type
    print("Recalling long-term memories...")
    memories = await manager.recall_memories(
        "RAG",
        memory_type=MemoryType.LONG_TERM,
    )
    for memory in memories:
        print(f"  - {memory.content}")
    print()

    await manager.close()


async def main():
    """Run all examples."""
    try:
        await basic_usage_example()
        await document_ingestion_example()
        await health_monitoring_example()
        await backend_preference_example()
        await memory_tiers_example()

        print("=== All examples completed successfully! ===")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
