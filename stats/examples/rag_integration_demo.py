#!/usr/bin/env python3
"""RAG-Redis System Integration Demo"""

import sys
from pathlib import Path

# Add the stats directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("RAG-Redis System Integration Demo")
    print("=" * 50)
    
    try:
        import gemma_extensions
        print(f"✓ Rust extensions loaded: v{gemma_extensions.get_version()}")
        
        # Test RAG module
        if hasattr(gemma_extensions, 'rag'):
            print("✓ RAG integration module available")
            
            # Demonstrate memory types
            print("\n=== Memory Tier System ===")
            working = gemma_extensions.rag.MemoryType.working()
            short_term = gemma_extensions.rag.MemoryType.short_term()
            long_term = gemma_extensions.rag.MemoryType.long_term()
            episodic = gemma_extensions.rag.MemoryType.episodic()
            semantic = gemma_extensions.rag.MemoryType.semantic()
            
            memory_types = [working, short_term, long_term, episodic, semantic]
            for mt in memory_types:
                print(f"  - {mt.name.replace('_', ' ').title()}")
            
            # Demonstrate RAG system
            print("\n=== RAG System Demo ===")
            rag_system = gemma_extensions.rag.RagSystem()
            
            if rag_system.test_connection():
                print("✓ RAG system connection successful")
                
                # Test search
                results = rag_system.search("machine learning", 3)
                print(f"✓ Search returned {len(results)} results")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result}")
            
            print("\n🎉 Demo completed successfully!")
            return True
        else:
            print("✗ RAG integration module not found")
            return False
            
    except ImportError as e:
        print(f"✗ Failed to import Rust extensions: {e}")
        print("Build extensions with: uv run maturin develop")
        return False
    except Exception as e:
        print(f"✗ Demo error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
