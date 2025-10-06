#!/usr/bin/env python3
"""
Simple test to verify the enhanced gemma-cli.py can be imported and basic functionality works.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all imports work."""
    try:
        print("Testing basic imports...")
        import json
        import hashlib
        import uuid
        from datetime import datetime, timedelta
        from pathlib import Path
        print("✓ Basic imports successful")

        try:
            import redis
            import numpy as np
            print("✓ Redis and numpy imports successful")
            redis_available = True
        except ImportError:
            print("⚠ Redis/numpy not available (optional)")
            redis_available = False

        try:
            from sentence_transformers import SentenceTransformer
            print("✓ SentenceTransformers import successful")
        except ImportError:
            print("⚠ SentenceTransformers not available (optional)")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_memory_tier():
    """Test MemoryTier class."""
    try:
        # Import our enhanced CLI module
        import importlib.util
        spec = importlib.util.spec_from_file_location("gemma_cli", "gemma-cli.py")
        gemma_cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gemma_cli)

        print("Testing MemoryTier class...")
        assert hasattr(gemma_cli.MemoryTier, 'WORKING')
        assert hasattr(gemma_cli.MemoryTier, 'SHORT_TERM')
        assert hasattr(gemma_cli.MemoryTier, 'LONG_TERM')
        assert hasattr(gemma_cli.MemoryTier, 'EPISODIC')
        assert hasattr(gemma_cli.MemoryTier, 'SEMANTIC')
        print("✓ MemoryTier class structure correct")

        print("Testing MemoryEntry class...")
        entry = gemma_cli.MemoryEntry("test content", gemma_cli.MemoryTier.SHORT_TERM, 0.5)
        assert entry.content == "test content"
        assert entry.memory_type == gemma_cli.MemoryTier.SHORT_TERM
        assert entry.importance == 0.5
        print("✓ MemoryEntry creation successful")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_gemma_interface():
    """Test GemmaInterface path validation."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("gemma_cli", "gemma-cli.py")
        gemma_cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gemma_cli)

        print("Testing GemmaInterface executable validation...")

        # Test with a dummy path that doesn't exist
        try:
            interface = gemma_cli.GemmaInterface(
                model_path="dummy_model.sbs",
                gemma_executable="nonexistent.exe"
            )
            print("✗ Should have raised FileNotFoundError")
            return False
        except FileNotFoundError:
            print("✓ Correctly validates executable existence")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Testing Enhanced Gemma CLI ===\n")

    tests = [
        ("Basic Imports", test_imports),
        ("Memory System", test_memory_tier),
        ("Gemma Interface", test_gemma_interface)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")

    print(f"\n=== Results: {passed}/{total} tests passed ===")

    if passed == total:
        print("🎉 All tests passed! The enhanced CLI is ready to use.")
    else:
        print("⚠ Some tests failed. Review the issues above.")

if __name__ == "__main__":
    main()