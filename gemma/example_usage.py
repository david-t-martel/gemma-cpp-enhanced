#!/usr/bin/env python3
"""
Example usage demonstration for the enhanced Gemma CLI with RAG.
This script shows how to use the RAG system programmatically.
"""

import asyncio
import sys
from pathlib import Path

# Add the gemma directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def demo_rag_system():
    """Demonstrate RAG system capabilities."""
    try:
        # Import the enhanced gemma-cli module
        import importlib.util
        spec = importlib.util.spec_from_file_location("gemma_cli", "gemma-cli.py")
        gemma_cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gemma_cli)

        print("=== Enhanced Gemma CLI - RAG System Demo ===\n")

        # Check if Redis dependencies are available
        if not gemma_cli.REDIS_AVAILABLE:
            print("❌ Redis dependencies not available")
            print("Install with: pip install redis numpy sentence-transformers")
            return

        print("✅ Redis dependencies available")

        # Initialize RAG system
        print("\n--- Initializing RAG System ---")
        rag_system = gemma_cli.RAGRedisManager()

        if not await rag_system.initialize():
            print("❌ Failed to initialize RAG system")
            print("Make sure Redis is running: redis-server")
            return

        print("✅ RAG system initialized successfully")

        # Demonstrate memory storage
        print("\n--- Storing Sample Data ---")
        sample_data = [
            ("Python is a high-level programming language", gemma_cli.MemoryTier.SEMANTIC, 0.8),
            ("Machine learning uses algorithms to find patterns", gemma_cli.MemoryTier.LONG_TERM, 0.7),
            ("Redis is an in-memory data structure store", gemma_cli.MemoryTier.LONG_TERM, 0.6),
            ("Current conversation about RAG system", gemma_cli.MemoryTier.WORKING, 0.5),
        ]

        stored_ids = []
        for content, tier, importance in sample_data:
            entry_id = await rag_system.store_memory(content, tier, importance)
            if entry_id:
                stored_ids.append(entry_id)
                print(f"✅ Stored: {content[:50]}... in {tier}")

        # Demonstrate memory recall
        print("\n--- Recalling Memories ---")
        queries = [
            "programming language",
            "machine learning",
            "data storage"
        ]

        for query in queries:
            print(f"\nQuery: '{query}'")
            memories = await rag_system.recall_memories(query, limit=2)

            if memories:
                for i, memory in enumerate(memories, 1):
                    score = getattr(memory, 'similarity_score', 0)
                    print(f"  {i}. [{memory.memory_type}] (score: {score:.3f})")
                    print(f"     {memory.content[:80]}...")
            else:
                print("  No relevant memories found")

        # Demonstrate search
        print("\n--- Searching Memories ---")
        search_results = await rag_system.search_memories("algorithm", min_importance=0.5)

        if search_results:
            print(f"Found {len(search_results)} results for 'algorithm':")
            for memory in search_results:
                print(f"  - [{memory.memory_type}] {memory.content}")
        else:
            print("No search results found")

        # Show memory statistics
        print("\n--- Memory Statistics ---")
        stats = await rag_system.get_memory_stats()
        print(f"Total entries: {stats.get('total', 0)}")
        for tier in gemma_cli.RAGRedisManager.TIER_CONFIG.keys():
            count = stats.get(tier, 0)
            print(f"  {tier}: {count} entries")

        # Cleanup
        print("\n--- Cleanup ---")
        cleaned = await rag_system.cleanup_expired()
        print(f"Cleaned up {cleaned} expired entries")

        print("\n🎉 Demo completed successfully!")
        print("\nTo use the full CLI with RAG:")
        print("python gemma-cli.py --model your_model.sbs --enable-rag")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

def demo_basic_usage():
    """Show basic usage without Redis."""
    print("=== Basic Gemma CLI Usage ===\n")

    print("1. Basic chat without RAG:")
    print("   python gemma-cli.py --model C:\\path\\to\\model.sbs")

    print("\n2. With RAG enabled:")
    print("   python gemma-cli.py --model C:\\path\\to\\model.sbs --enable-rag")

    print("\n3. Full configuration:")
    print("   python gemma-cli.py \\")
    print("       --model C:\\path\\to\\model.sbs \\")
    print("       --tokenizer C:\\path\\to\\tokenizer.spm \\")
    print("       --enable-rag \\")
    print("       --max-tokens 2048 \\")
    print("       --temperature 0.7")

    print("\n4. Available RAG commands in chat:")
    print("   /store 'Important information' long_term 0.8")
    print("   /recall 'query about something'")
    print("   /search 'keyword' semantic 0.5")
    print("   /ingest C:\\docs\\document.txt")
    print("   /memory_stats")
    print("   /cleanup")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Gemma CLI Demo")
    parser.add_argument("--basic", action="store_true", help="Show basic usage only")
    args = parser.parse_args()

    if args.basic:
        demo_basic_usage()
    else:
        asyncio.run(demo_rag_system())