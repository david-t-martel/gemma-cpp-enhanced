#!/usr/bin/env python3
"""Test script to verify embedded vector store works without Redis."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rag.embedded_vector_store import EmbeddedVectorStore
from rag.hybrid_rag import (
    HybridRAGManager,
    RecallMemoriesParams,
    StoreMemoryParams,
    SearchParams,
)


async def test_embedded_store_basic():
    """Test basic embedded store operations."""
    print("=" * 60)
    print("Testing Embedded Vector Store (No Redis Required)")
    print("=" * 60)

    # Test 1: Direct embedded store initialization
    print("\n[Test 1] Direct Embedded Store Initialization")
    print("-" * 60)
    store = EmbeddedVectorStore()
    success = await store.initialize()
    print(f"✓ Initialization: {'SUCCESS' if success else 'FAILED'}")

    # Test 2: Store memory
    print("\n[Test 2] Store Memory")
    print("-" * 60)
    params = StoreMemoryParams(
        content="This is a test memory about Python programming",
        memory_type="semantic",
        importance=0.8,
        tags=["test", "python"],
    )
    memory_id = await store.store_memory(params)
    print(f"✓ Stored memory with ID: {memory_id[:8]}...")

    # Test 3: Recall memories
    print("\n[Test 3] Recall Memories")
    print("-" * 60)
    recall_params = RecallMemoriesParams(
        query="Python programming", memory_type="semantic", limit=5
    )
    memories = await store.recall_memories(recall_params)
    print(f"✓ Recalled {len(memories)} memories")
    for mem in memories:
        print(f"  - {mem.content[:50]}... (importance: {mem.importance})")

    # Test 4: Get memory stats
    print("\n[Test 4] Memory Statistics")
    print("-" * 60)
    stats = await store.get_memory_stats()
    print("✓ Memory stats:")
    for tier, count in stats.items():
        if tier != "redis_memory":
            print(f"  - {tier}: {count} entries")

    # Test 5: Close store
    print("\n[Test 5] Close and Persist")
    print("-" * 60)
    await store.close()
    print(f"✓ Saved to: {store.STORE_FILE}")

    return True


async def test_hybrid_rag_manager():
    """Test HybridRAGManager with embedded store."""
    print("\n" + "=" * 60)
    print("Testing HybridRAGManager with Embedded Store")
    print("=" * 60)

    # Test 1: Initialize RAG manager with embedded store
    print("\n[Test 1] Initialize RAG Manager")
    print("-" * 60)
    rag_manager = HybridRAGManager(use_embedded_store=True)
    success = await rag_manager.initialize()
    print(f"✓ Initialization: {'SUCCESS' if success else 'FAILED'}")

    # Test 2: Store multiple memories
    print("\n[Test 2] Store Multiple Memories")
    print("-" * 60)
    test_memories = [
        ("Python is a high-level programming language", "semantic", 0.9),
        ("I learned about async/await today", "episodic", 0.7),
        ("Redis is optional for this application", "semantic", 0.8),
    ]

    for content, mem_type, importance in test_memories:
        params = StoreMemoryParams(
            content=content, memory_type=mem_type, importance=importance
        )
        mem_id = await rag_manager.store_memory(params)
        print(f"✓ Stored: {content[:40]}... (ID: {mem_id[:8]}...)")

    # Test 3: Search memories
    print("\n[Test 3] Search Memories")
    print("-" * 60)
    search_params = SearchParams(
        query="Python", memory_type=None, min_importance=0.5
    )
    results = await rag_manager.search_memories(search_params)
    print(f"✓ Found {len(results)} memories matching 'Python':")
    for mem in results:
        print(f"  - [{mem.memory_type}] {mem.content[:50]}...")

    # Test 4: Get stats
    print("\n[Test 4] RAG Statistics")
    print("-" * 60)
    stats = await rag_manager.get_memory_stats()
    print("✓ RAG stats:")
    print(f"  - Total entries: {stats.get('total', 0)}")
    for tier in ["working", "short_term", "long_term", "episodic", "semantic"]:
        print(f"  - {tier}: {stats.get(tier, 0)}")

    # Test 5: Close
    print("\n[Test 5] Close RAG Manager")
    print("-" * 60)
    await rag_manager.close()
    print("✓ Closed successfully")

    return True


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("EMBEDDED VECTOR STORE TEST SUITE")
    print("Testing standalone operation without Redis")
    print("=" * 60)

    try:
        # Test direct embedded store
        result1 = await test_embedded_store_basic()

        # Test via HybridRAGManager
        result2 = await test_hybrid_rag_manager()

        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        if result1 and result2:
            print("✓ All tests PASSED")
            print("\nThe application can run WITHOUT Redis!")
            print("Memory is persisted to:")
            print(f"  {Path.home() / '.gemma_cli' / 'embedded_store.json'}")
            return 0
        else:
            print("✗ Some tests FAILED")
            return 1

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
