#!/usr/bin/env python
"""Simple performance test for Gemma CLI."""

import time
import tracemalloc
import json
from pathlib import Path

def test_imports():
    """Test import times."""
    results = {}

    # Test core imports
    start = time.perf_counter()
    import cli
    results['cli'] = time.perf_counter() - start

    start = time.perf_counter()
    from config.settings import load_config
    results['config.settings'] = time.perf_counter() - start

    start = time.perf_counter()
    from core.gemma import GemmaInterface
    results['core.gemma'] = time.perf_counter() - start

    start = time.perf_counter()
    from rag.hybrid_rag import HybridRAGManager
    results['rag.hybrid_rag'] = time.perf_counter() - start

    return results

def test_config_loading():
    """Test configuration loading performance."""
    from config.settings import load_config

    config_path = Path.home() / ".gemma_cli" / "config.toml"

    start = time.perf_counter()
    if config_path.exists():
        config = load_config(config_path)
    else:
        config = None
    load_time = time.perf_counter() - start

    return {'config_load_time': load_time, 'config_exists': config is not None}

def test_memory_usage():
    """Test memory usage during startup."""
    tracemalloc.start()

    # Import main modules
    import cli
    from config.settings import load_config
    from core.gemma import GemmaInterface
    from rag.hybrid_rag import HybridRAGManager

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        'current_mb': current / 1024 / 1024,
        'peak_mb': peak / 1024 / 1024
    }

def main():
    """Run all performance tests."""
    print("=" * 60)
    print("GEMMA CLI PERFORMANCE TEST")
    print("=" * 60)

    # Test imports
    print("\n📦 Testing import times...")
    import_results = test_imports()
    for module, time_taken in sorted(import_results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {module}: {time_taken*1000:.1f}ms")

    # Test config loading
    print("\n⚙️ Testing config loading...")
    config_results = test_config_loading()
    if config_results['config_exists']:
        print(f"  Config load time: {config_results['config_load_time']*1000:.1f}ms")
    else:
        print("  No config file found")

    # Test memory usage
    print("\n💾 Testing memory usage...")
    memory_results = test_memory_usage()
    print(f"  Current: {memory_results['current_mb']:.1f}MB")
    print(f"  Peak: {memory_results['peak_mb']:.1f}MB")

    # Save results
    all_results = {
        'imports': import_results,
        'config': config_results,
        'memory': memory_results
    }

    with open('performance_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Results saved to performance_test_results.json")
    print("=" * 60)

if __name__ == "__main__":
    main()