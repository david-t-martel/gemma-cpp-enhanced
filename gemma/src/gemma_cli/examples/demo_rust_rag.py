#!/usr/bin/env python3
"""Demo script for Rust RAG backend integration.

This script demonstrates how to use the high-performance Rust MCP server
as a backend for RAG operations in Gemma CLI.

Usage:
    uv run python examples/demo_rust_rag.py
"""

import asyncio
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gemma_cli.rag.hybrid_rag import (
    HybridRAGManager,
    IngestDocumentParams,
    SearchParams,
    StoreMemoryParams,
    RecallMemoriesParams,
)
from gemma_cli.config.settings import load_config


async def demo_rust_backend():
    """Demonstrate Rust RAG backend capabilities."""
    print("=" * 60)
    print("Rust RAG Backend Demo")
    print("=" * 60)

    # Load configuration
    config = load_config()
    backend_type = config.rag_backend.backend
    rust_path = config.rag_backend.rust_mcp_server_path

    print(f"\nConfiguration:")
    print(f"  Backend: {backend_type}")
    print(f"  Rust server path: {rust_path or 'auto-detect'}")

    # Initialize RAG manager with Rust backend
    print("\n1. Initializing Rust RAG backend...")
    manager = HybridRAGManager(backend="rust", rust_mcp_server_path=rust_path)

    try:
        success = await manager.initialize()
        if not success:
            print("   ❌ Failed to initialize Rust backend")
            return

        print(f"   ✓ Backend initialized: {manager.backend_type}")

        # Create a temporary test document
        print("\n2. Creating test document...")
        with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                """
# Rust Programming Language

Rust is a systems programming language that runs blazingly fast, prevents
segfaults, and guarantees thread safety. It accomplishes these goals without
requiring a garbage collector or runtime.

Key features:
- Zero-cost abstractions
- Move semantics
- Guaranteed memory safety
- Threads without data races
- Trait-based generics
- Pattern matching
- Type inference

Rust is ideal for:
- System programming
- WebAssembly
- Embedded systems
- High-performance applications
"""
            )
            test_doc_path = f.name

        print(f"   ✓ Created: {test_doc_path}")

        # Ingest document
        print("\n3. Ingesting document into RAG system...")
        ingest_params = IngestDocumentParams(
            file_path=test_doc_path, memory_type="long_term", chunk_size=200
        )
        chunks = await manager.ingest_document(ingest_params)
        print(f"   ✓ Ingested {chunks} chunks")

        # Search for information
        print("\n4. Performing semantic search...")
        queries = [
            "What are the key features of Rust?",
            "Is Rust suitable for embedded systems?",
            "Does Rust have a garbage collector?",
        ]

        for query in queries:
            print(f"\n   Query: '{query}'")
            search_params = SearchParams(query=query, memory_type=None, min_importance=0.0)

            results = await manager.search_memories(search_params)

            if results:
                print(f"   Found {len(results)} results:")
                for i, result in enumerate(results[:2], 1):
                    content_preview = result.content[:150]
                    print(f"     [{i}] {content_preview}...")
            else:
                print("   No results found")

        # Store custom memory
        print("\n5. Storing custom memory...")
        store_params = StoreMemoryParams(
            content="Rust backend integration was tested on " + str(asyncio.get_event_loop().time()),
            memory_type="episodic",
            importance=0.95,
            tags=["test", "integration", "rust"],
        )
        memory_id = await manager.store_memory(store_params)
        print(f"   ✓ Stored memory with ID: {memory_id}")

        # Recall memories
        print("\n6. Recalling memories...")
        recall_params = RecallMemoriesParams(
            query="integration test", memory_type="episodic", limit=5
        )
        recalled = await manager.recall_memories(recall_params)
        print(f"   ✓ Recalled {len(recalled)} memories")
        for mem in recalled[:3]:
            print(f"     - {mem.content[:100]}...")

        # Get system stats
        print("\n7. Retrieving memory statistics...")
        stats = await manager.get_memory_stats()
        print("   Memory tiers:")
        for tier, info in stats.items():
            if isinstance(info, dict):
                count = info.get("count", 0)
                print(f"     {tier}: {count} entries")

        # Cleanup
        print("\n8. Cleaning up...")
        Path(test_doc_path).unlink(missing_ok=True)
        print("   ✓ Removed test document")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Close RAG manager
        await manager.close()
        print("\n   ✓ Rust backend stopped")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


async def compare_backends():
    """Compare performance between embedded and Rust backends."""
    import time

    print("\n" + "=" * 60)
    print("Backend Performance Comparison")
    print("=" * 60)

    # Create test document
    with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test document for performance comparison. " * 100)
        test_doc_path = f.name

    test_params = IngestDocumentParams(
        file_path=test_doc_path, memory_type="long_term", chunk_size=100
    )

    # Test embedded backend
    print("\n1. Testing embedded backend...")
    embedded_manager = HybridRAGManager(backend="embedded")
    await embedded_manager.initialize()

    start = time.perf_counter()
    await embedded_manager.ingest_document(test_params)
    embedded_time = time.perf_counter() - start

    await embedded_manager.close()
    print(f"   Time: {embedded_time:.4f}s")

    # Test Rust backend
    print("\n2. Testing Rust backend...")
    rust_manager = HybridRAGManager(backend="rust")

    try:
        await rust_manager.initialize()

        start = time.perf_counter()
        await rust_manager.ingest_document(test_params)
        rust_time = time.perf_counter() - start

        print(f"   Time: {rust_time:.4f}s")

        # Calculate speedup
        if rust_time > 0:
            speedup = embedded_time / rust_time
            print(f"\n   Speedup: {speedup:.2f}x faster")
        else:
            print("\n   Rust backend significantly faster!")

    except Exception as e:
        print(f"   ⚠ Rust backend not available: {e}")
    finally:
        await rust_manager.close()

    # Cleanup
    Path(test_doc_path).unlink(missing_ok=True)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Rust RAG backend demo")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare performance with embedded backend",
    )
    args = parser.parse_args()

    if args.compare:
        asyncio.run(compare_backends())
    else:
        asyncio.run(demo_rust_backend())


if __name__ == "__main__":
    main()
