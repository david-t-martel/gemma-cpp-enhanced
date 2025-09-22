#!/usr/bin/env python3
"""
Demo script showing how to use Gemma Rust Extensions.

This script demonstrates the key features of the high-performance
Rust extensions for tokenization, tensor operations, and caching.
"""

import asyncio
import sys
import time

try:
    import gemma_extensions as ge
except ImportError:
    print("ERROR: Gemma extensions not available. Build them with: maturin develop")
    sys.exit(1)


def demo_basic_info():
    """Demonstrate basic extension information."""
    print("=== Basic Information ===")
    print(f"Version: {ge.get_version()}")
    print(f"SIMD Support: {ge.check_simd_support()}")

    build_info = ge.get_build_info()
    print("Build Info:")
    for key, value in build_info.items():
        print(f"  {key}: {value}")

    # Warmup extensions
    warmup_result = ge.warmup()
    print(f"Warmup: {warmup_result}")
    print()


def demo_tokenization():
    """Demonstrate fast tokenization."""
    print("=== Tokenization Demo ===")

    # Create tokenizer configuration
    config = ge.TokenizerConfig()
    config.vocab_size = 32000
    config.use_simd = True
    config.parallel = True
    print(f"Tokenizer config: vocab_size={config.vocab_size}, simd={config.use_simd}")

    # Create tokenizer
    tokenizer = ge.FastTokenizer(config)

    # Single text tokenization
    text = "Hello, this is a test sentence for tokenization."
    start_time = time.time()
    tokens = tokenizer.encode(text)
    encode_time = time.time() - start_time

    print(f"Text: '{text}'")
    print(f"Tokens: {tokens[:10]}... (showing first 10)")
    print(f"Encode time: {encode_time * 1000:.2f}ms")

    # Decode back
    start_time = time.time()
    decoded = tokenizer.decode(tokens)
    decode_time = time.time() - start_time
    print(f"Decoded: '{decoded[:50]}...'")
    print(f"Decode time: {decode_time * 1000:.2f}ms")

    # Batch tokenization
    texts = [
        "First example sentence.",
        "Second example with more words.",
        "Third sentence for batch processing.",
        "Final sentence in the batch.",
    ]

    start_time = time.time()
    ge.batch_encode(tokenizer, texts, parallel=True)
    batch_time = time.time() - start_time

    print(f"Batch tokenization of {len(texts)} texts: {batch_time * 1000:.2f}ms")
    print(f"Average per text: {batch_time * 1000 / len(texts):.2f}ms")

    # Show statistics
    stats = tokenizer.get_stats()
    print("Tokenizer Statistics:")
    for key, value in stats.items():
        if key.endswith("_percent"):
            print(f"  {key}: {value:.1f}%")
        else:
            print(f"  {key}: {value}")
    print()


def demo_tensor_operations():
    """Demonstrate high-performance tensor operations."""
    print("=== Tensor Operations Demo ===")

    # Create tensor operations
    tensor_ops = ge.TensorOperations()

    # SIMD dot product
    a = [1.0, 2.0, 3.0, 4.0, 5.0] * 1000  # 5000 elements
    b = [2.0, 3.0, 4.0, 5.0, 6.0] * 1000  # 5000 elements

    start_time = time.time()
    dot_product = ge.simd_dot_product(a, b)
    simd_time = time.time() - start_time

    print(f"SIMD dot product of {len(a)} elements: {dot_product}")
    print(f"SIMD time: {simd_time * 1000:.2f}ms")

    # Compare with Python
    start_time = time.time()
    python_dot = sum(x * y for x, y in zip(a, b, strict=False))
    python_time = time.time() - start_time

    print(f"Python dot product: {python_dot}")
    print(f"Python time: {python_time * 1000:.2f}ms")
    print(f"Speedup: {python_time / simd_time:.1f}x")

    # Vector addition
    start_time = time.time()
    vector_sum = ge.simd_vector_add(a, b)
    vector_time = time.time() - start_time

    print(f"SIMD vector addition time: {vector_time * 1000:.2f}ms")
    print(f"Result sum: {sum(vector_sum)}")

    # Matrix multiplication
    size = 64
    matrix_a = [0.1 * i for i in range(size * size)]
    matrix_b = [0.2 * i for i in range(size * size)]

    start_time = time.time()
    tensor_ops.matmul(matrix_a, matrix_b, size, size, size)
    matmul_time = time.time() - start_time

    print(f"Matrix multiplication ({size}x{size}): {matmul_time * 1000:.2f}ms")

    # Activation functions
    input_data = [x * 0.1 - 2.5 for x in range(1000)]  # Range from -2.5 to 97.5

    start_time = time.time()
    tensor_ops.gelu(input_data)
    gelu_time = time.time() - start_time

    start_time = time.time()
    tensor_ops.silu(input_data)
    silu_time = time.time() - start_time

    print(f"GELU activation ({len(input_data)} elements): {gelu_time * 1000:.2f}ms")
    print(f"SiLU activation ({len(input_data)} elements): {silu_time * 1000:.2f}ms")

    # Softmax
    softmax_input = [x * 0.1 for x in range(100)]
    start_time = time.time()
    softmax_result = ge.simd_softmax(softmax_input)
    softmax_time = time.time() - start_time

    print(f"Softmax ({len(softmax_input)} elements): {softmax_time * 1000:.2f}ms")
    print(f"Softmax sum: {sum(softmax_result):.6f} (should be ~1.0)")
    print()


def demo_caching():
    """Demonstrate high-performance caching."""
    print("=== Caching Demo ===")

    # Create LRU cache
    cache = ge.LRUCache(capacity=1000)

    # Basic operations
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")

    print(f"Get key1: {cache.get('key1')}")
    print(f"Get nonexistent: {cache.get('nonexistent')}")

    # Bulk operations
    start_time = time.time()
    for i in range(500):
        cache.put(f"bulk_key_{i}", f"bulk_value_{i}")
    put_time = time.time() - start_time

    print(f"Bulk put 500 items: {put_time * 1000:.2f}ms")
    print(f"Average per put: {put_time * 1000 / 500:.3f}ms")

    # Bulk get operations
    start_time = time.time()
    hits = 0
    for i in range(500):
        result = cache.get(f"bulk_key_{i}")
        if result is not None:
            hits += 1
    get_time = time.time() - start_time

    print(f"Bulk get 500 items: {get_time * 1000:.2f}ms")
    print(f"Cache hits: {hits}/500")
    print(f"Average per get: {get_time * 1000 / 500:.3f}ms")

    # Cache statistics
    stats = cache.stats()
    print("Cache Statistics:")
    print(f"  Size: {stats.size}/{stats.capacity}")
    print(f"  Hits: {stats.hits}, Misses: {stats.misses}")
    print(f"  Hit Rate: {stats.hit_rate():.1f}%")
    print(f"  Memory Usage: {stats.memory_usage} bytes")
    print(f"  Average Entry Size: {stats.avg_entry_size():.1f} bytes")

    # TTL cache demo
    ttl_cache = ge.LRUCache.with_ttl(capacity=100, ttl_seconds=2)
    ttl_cache.put("temp_key", "temp_value")
    print(f"TTL cache get (immediate): {ttl_cache.get('temp_key')}")

    time.sleep(1)
    print(f"TTL cache get (after 1s): {ttl_cache.get('temp_key')}")

    # Cache manager demo
    manager = ge.CacheManager(default_capacity=100)
    manager.create_cache("users", 200)
    manager.create_cache("sessions", 300)

    cache_names = manager.list_caches()
    print(f"Managed caches: {cache_names}")

    # Get all statistics
    all_stats = manager.get_all_stats()
    print("Manager Statistics:")
    for cache_name, cache_stats in all_stats.items():
        print(
            f"  {cache_name}: {cache_stats['size']} items, {cache_stats['hit_rate']:.1f}% hit rate"
        )
    print()


async def demo_async_operations():
    """Demonstrate async cache operations."""
    print("=== Async Operations Demo ===")

    cache = ge.LRUCache(capacity=100)

    # Async put/get operations
    await cache.put_async("async_key1", "async_value1")
    await cache.put_async("async_key2", "async_value2")

    value1 = await cache.get_async("async_key1")
    value2 = await cache.get_async("async_key2")
    nonexistent = await cache.get_async("nonexistent")

    print(f"Async get async_key1: {value1}")
    print(f"Async get async_key2: {value2}")
    print(f"Async get nonexistent: {nonexistent}")

    # Concurrent async operations
    start_time = time.time()
    tasks = []
    for i in range(50):
        tasks.append(cache.put_async(f"async_bulk_{i}", f"async_value_{i}"))

    await asyncio.gather(*tasks)
    async_put_time = time.time() - start_time

    print(f"Async bulk put 50 items: {async_put_time * 1000:.2f}ms")

    # Concurrent get
    start_time = time.time()
    get_tasks = []
    for i in range(50):
        get_tasks.append(cache.get_async(f"async_bulk_{i}"))

    results = await asyncio.gather(*get_tasks)
    async_get_time = time.time() - start_time

    successful_gets = sum(1 for r in results if r is not None)
    print(f"Async bulk get 50 items: {async_get_time * 1000:.2f}ms")
    print(f"Successful gets: {successful_gets}/50")
    print()


def demo_benchmarks():
    """Run and display benchmark results."""
    print("=== Benchmark Results ===")

    print("Running comprehensive benchmarks...")
    start_time = time.time()
    benchmark_results = ge.benchmark_operations()
    benchmark_time = time.time() - start_time

    print(f"Benchmark suite completed in {benchmark_time:.2f}s")
    print("Results:")
    for operation, time_taken in benchmark_results.items():
        operations_per_second = 1000 / time_taken if time_taken > 0 else float("inf")
        print(f"  {operation}: {time_taken:.4f}s ({operations_per_second:.0f} ops/sec)")
    print()


def main():
    """Run all demos."""
    print("Gemma Rust Extensions Demo")
    print("=" * 50)

    try:
        # Basic information
        demo_basic_info()

        # Tokenization
        demo_tokenization()

        # Tensor operations
        demo_tensor_operations()

        # Caching
        demo_caching()

        # Async operations
        print("Running async demo...")
        asyncio.run(demo_async_operations())

        # Benchmarks
        demo_benchmarks()

        print("All demos completed successfully!")

    except Exception as e:
        print(f"ERROR during demo: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
