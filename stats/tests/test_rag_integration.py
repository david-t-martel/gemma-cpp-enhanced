#!/usr/bin/env python3
"""Integration tests for RAG-Redis system."""

import unittest
import sys
from pathlib import Path

# Add the stats directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestRagIntegration(unittest.TestCase):
    """Test RAG-Redis integration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            import gemma_extensions
            self.rag_available = hasattr(gemma_extensions, 'rag')
            if self.rag_available:
                self.rag_system = gemma_extensions.rag.RagSystem()
        except ImportError:
            self.rag_available = False
    
    def test_rust_extensions_import(self):
        """Test that Rust extensions can be imported."""
        try:
            import gemma_extensions
            version = gemma_extensions.get_version()
            self.assertIsInstance(version, str)
            self.assertTrue(len(version) > 0)
        except ImportError:
            self.fail("Could not import gemma_extensions")
    
    def test_rag_module_available(self):
        """Test that RAG module is available."""
        if not self.rag_available:
            self.skipTest("RAG module not available")
        
        import gemma_extensions
        self.assertTrue(hasattr(gemma_extensions, 'rag'))
    
    def test_memory_types(self):
        """Test memory type creation and properties."""
        if not self.rag_available:
            self.skipTest("RAG module not available")
        
        import gemma_extensions
        
        # Test all memory types
        working = gemma_extensions.rag.MemoryType.working()
        self.assertEqual(working.name, "working")
        
        short_term = gemma_extensions.rag.MemoryType.short_term()
        self.assertEqual(short_term.name, "short_term")
        
        long_term = gemma_extensions.rag.MemoryType.long_term()
        self.assertEqual(long_term.name, "long_term")
        
        episodic = gemma_extensions.rag.MemoryType.episodic()
        self.assertEqual(episodic.name, "episodic")
        
        semantic = gemma_extensions.rag.MemoryType.semantic()
        self.assertEqual(semantic.name, "semantic")
    
    def test_rag_system_creation(self):
        """Test RAG system instantiation."""
        if not self.rag_available:
            self.skipTest("RAG module not available")
        
        import gemma_extensions
        
        # Test default creation
        rag_system = gemma_extensions.rag.RagSystem()
        self.assertIsNotNone(rag_system)
        
        # Test with custom Redis URL
        rag_system_custom = gemma_extensions.rag.RagSystem("redis://localhost:6379")
        self.assertIsNotNone(rag_system_custom)
    
    def test_connection_test(self):
        """Test connection testing functionality."""
        if not self.rag_available:
            self.skipTest("RAG module not available")
        
        result = self.rag_system.test_connection()
        self.assertIsInstance(result, bool)
    
    def test_search_functionality(self):
        """Test search functionality."""
        if not self.rag_available:
            self.skipTest("RAG module not available")
        
        # Test basic search
        results = self.rag_system.search("test query")
        self.assertIsInstance(results, list)
        
        # Test search with limit
        results_limited = self.rag_system.search("test query", 2)
        self.assertIsInstance(results_limited, list)
        self.assertLessEqual(len(results_limited), 2)
        
        # Test search result properties
        if results:
            result = results[0]
            self.assertTrue(hasattr(result, 'id'))
            self.assertTrue(hasattr(result, 'content'))
            self.assertTrue(hasattr(result, 'score'))
            self.assertTrue(hasattr(result, 'metadata'))
    
    def test_search_result_creation(self):
        """Test search result object creation."""
        if not self.rag_available:
            self.skipTest("RAG module not available")
        
        import gemma_extensions
        
        metadata = {"source": "test", "type": "document"}
        result = gemma_extensions.rag.SearchResult(
            "test_id", 
            "test content", 
            0.95, 
            metadata
        )
        
        self.assertEqual(result.id, "test_id")
        self.assertEqual(result.content, "test content")
        self.assertEqual(result.score, 0.95)
        self.assertEqual(result.metadata, metadata)

class TestRedisIntegration(unittest.TestCase):
    """Test Redis integration if available."""
    
    def setUp(self):
        """Set up Redis test fixtures."""
        try:
            import redis
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            self.redis_available = True
        except Exception:
            self.redis_available = False
    
    def test_redis_connection(self):
        """Test Redis connection if available."""
        if not self.redis_available:
            self.skipTest("Redis not available")
        
        self.assertTrue(self.redis_client.ping())
    
    def test_redis_basic_operations(self):
        """Test basic Redis operations."""
        if not self.redis_available:
            self.skipTest("Redis not available")
        
        # Test set/get
        test_key = "rag:test:key"
        test_value = "test value"
        
        self.redis_client.set(test_key, test_value, ex=60)
        retrieved_value = self.redis_client.get(test_key)
        self.assertEqual(retrieved_value, test_value)
        
        # Cleanup
        self.redis_client.delete(test_key)

if __name__ == '__main__':
    unittest.main()
