"""
Specialized Performance Benchmark Suite for RAG-Redis System

Focuses on detailed performance metrics:
- SIMD vector operations benchmarking
- Redis connection pool performance
- Memory usage optimization validation
- Throughput and latency measurements
- Concurrent operation testing
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gc
import json
import logging
import multiprocessing
import os
from pathlib import Path
from statistics import mean, median, stdev
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil
import pytest
import redis

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PROJECT_ROOT = Path(__file__).parent.parent
RAG_REDIS_DIR = PROJECT_ROOT / "rag-redis"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkResults:
    """Store and analyze benchmark results"""

    def __init__(self):
        self.results = {}
        self.start_time = time.time()

    def add_benchmark(self, category: str, name: str, metrics: dict[str, Any]):
        """Add benchmark results"""
        if category not in self.results:
            self.results[category] = {}

        self.results[category][name] = {
            **metrics,
            "timestamp": time.time(),
            "elapsed_since_start": time.time() - self.start_time,
        }

    def get_summary(self) -> dict[str, Any]:
        """Get benchmark summary"""
        return {
            "total_categories": len(self.results),
            "total_benchmarks": sum(len(cat) for cat in self.results.values()),
            "duration_seconds": time.time() - self.start_time,
            "results": self.results,
        }

    def save_results(self, filepath: Path):
        """Save results to JSON file"""
        with open(filepath, "w") as f:
            json.dump(self.get_summary(), f, indent=2, default=str)


@pytest.fixture(scope="session")
def benchmark_results():
    """Shared benchmark results collector"""
    return BenchmarkResults()


@pytest.fixture(scope="session")
def redis_pool():
    """Redis connection pool for testing"""
    # Import centralized Redis test configuration
    from src.shared.config.redis_test_utils import get_test_redis_config

    try:
        config = get_test_redis_config()
        config["db"] = 1  # Use different DB for benchmarks
        config["max_connections"] = 20

        pool = redis.ConnectionPool(**config)
        client = redis.Redis(connection_pool=pool)
        client.ping()
        client.flushdb()
        yield pool
        client.flushdb()
    except redis.ConnectionError:
        pytest.skip("Redis server not available")


class TestSIMDOptimizations:
    """Test SIMD optimization performance"""

    def test_vector_similarity_performance(self, benchmark_results):
        """Benchmark vector similarity computations"""
        logger.info("Benchmarking vector similarity performance...")

        # Test different vector sizes and dimensions
        test_configs = [
            (1000, 128),  # Small vectors
            (1000, 384),  # Medium vectors (common embedding size)
            (10000, 384),  # Large dataset
            (1000, 768),  # Large dimension vectors
        ]

        results = {}

        for num_vectors, dim in test_configs:
            config_name = f"{num_vectors}x{dim}"
            logger.info(f"Testing configuration: {config_name}")

            # Generate random vectors
            vectors_a = np.random.rand(num_vectors, dim).astype(np.float32)
            vectors_b = np.random.rand(num_vectors, dim).astype(np.float32)

            # Benchmark different similarity methods
            methods = {
                "dot_product": lambda a, b: np.sum(a * b, axis=1),
                "cosine_similarity": lambda a, b: np.sum(a * b, axis=1)
                / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)),
                "euclidean_distance": lambda a, b: np.linalg.norm(a - b, axis=1),
            }

            method_results = {}

            for method_name, method_func in methods.items():
                times = []
                # Run multiple iterations for stable measurement
                for _ in range(5):
                    start_time = time.perf_counter()
                    result = method_func(vectors_a, vectors_b)
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)  # Convert to ms

                method_results[method_name] = {
                    "mean_time_ms": mean(times),
                    "median_time_ms": median(times),
                    "std_time_ms": stdev(times) if len(times) > 1 else 0,
                    "min_time_ms": min(times),
                    "max_time_ms": max(times),
                    "throughput_ops_per_sec": num_vectors / (mean(times) / 1000),
                }

            results[config_name] = method_results

        benchmark_results.add_benchmark("simd_optimizations", "vector_similarity", results)
        logger.info("Vector similarity benchmarking completed")

    def test_memory_aligned_operations(self, benchmark_results):
        """Test memory-aligned vector operations performance"""
        logger.info("Testing memory-aligned operations...")

        vector_size = 10000
        dimension = 384

        # Test aligned vs unaligned memory access
        # Aligned arrays
        aligned_array = np.empty((vector_size, dimension), dtype=np.float32, order="C")
        aligned_array.fill(1.0)

        # Create reference array for operations
        ref_array = np.random.rand(vector_size, dimension).astype(np.float32)

        operations = {
            "element_wise_multiply": lambda a, b: a * b,
            "matrix_vector_multiply": lambda a, b: np.dot(a, b.T),
            "reduce_sum": lambda a, b: np.sum(a, axis=1),
            "reduce_max": lambda a, b: np.max(a, axis=1),
        }

        results = {}

        for op_name, op_func in operations.items():
            times = []
            for _ in range(10):
                start_time = time.perf_counter()
                if op_name == "matrix_vector_multiply":
                    # Use smaller subset for matrix multiplication
                    result = op_func(aligned_array[:100], ref_array[:100])
                else:
                    result = op_func(aligned_array, ref_array)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)

            results[op_name] = {
                "mean_time_ms": mean(times),
                "std_time_ms": stdev(times) if len(times) > 1 else 0,
                "throughput_elements_per_ms": vector_size / mean(times),
            }

        benchmark_results.add_benchmark("simd_optimizations", "memory_aligned_ops", results)
        logger.info("Memory-aligned operations testing completed")


class TestRedisPerformance:
    """Test Redis operations performance"""

    def test_connection_pool_performance(self, benchmark_results, redis_pool):
        """Test Redis connection pool performance under load"""
        logger.info("Testing Redis connection pool performance...")

        def redis_operation_batch(pool, batch_id: int, operations_per_batch: int):
            """Perform a batch of Redis operations"""
            client = redis.Redis(connection_pool=pool)
            start_time = time.perf_counter()

            for i in range(operations_per_batch):
                key = f"batch_{batch_id}_key_{i}"
                value = f"batch_{batch_id}_value_{i}"
                client.set(key, value)
                retrieved = client.get(key)
                assert retrieved.decode() == value

            end_time = time.perf_counter()
            return end_time - start_time

        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        operations_per_batch = 100

        results = {}

        for concurrency in concurrency_levels:
            logger.info(f"Testing concurrency level: {concurrency}")

            start_time = time.perf_counter()

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = []
                for batch_id in range(concurrency):
                    future = executor.submit(
                        redis_operation_batch, redis_pool, batch_id, operations_per_batch
                    )
                    futures.append(future)

                batch_times = [future.result() for future in futures]

            total_time = time.perf_counter() - start_time
            total_operations = concurrency * operations_per_batch * 2  # set + get

            results[f"concurrency_{concurrency}"] = {
                "total_time_seconds": total_time,
                "total_operations": total_operations,
                "operations_per_second": total_operations / total_time,
                "mean_batch_time": mean(batch_times),
                "std_batch_time": stdev(batch_times) if len(batch_times) > 1 else 0,
                "concurrent_efficiency": (sum(batch_times) / total_time) * 100,
            }

            # Cleanup
            client = redis.Redis(connection_pool=redis_pool)
            for batch_id in range(concurrency):
                for i in range(operations_per_batch):
                    client.delete(f"batch_{batch_id}_key_{i}")

        benchmark_results.add_benchmark("redis_performance", "connection_pool", results)
        logger.info("Redis connection pool performance testing completed")

    def test_bulk_operations_performance(self, benchmark_results, redis_pool):
        """Test Redis bulk operations performance"""
        logger.info("Testing Redis bulk operations performance...")

        client = redis.Redis(connection_pool=redis_pool)

        # Test different bulk operation sizes
        bulk_sizes = [100, 1000, 5000, 10000]

        results = {}

        for bulk_size in bulk_sizes:
            logger.info(f"Testing bulk size: {bulk_size}")

            # Generate test data
            test_data = {f"bulk_key_{i}": f"bulk_value_{i}" for i in range(bulk_size)}

            # Test pipeline operations
            start_time = time.perf_counter()
            pipe = client.pipeline()
            for key, value in test_data.items():
                pipe.set(key, value)
            pipe.execute()
            pipeline_set_time = time.perf_counter() - start_time

            # Test pipeline retrieval
            start_time = time.perf_counter()
            pipe = client.pipeline()
            for key in test_data:
                pipe.get(key)
            retrieved_values = pipe.execute()
            pipeline_get_time = time.perf_counter() - start_time

            # Test mset/mget operations
            start_time = time.perf_counter()
            client.mset(test_data)
            mset_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            retrieved_mget = client.mget(list(test_data.keys()))
            mget_time = time.perf_counter() - start_time

            results[f"bulk_size_{bulk_size}"] = {
                "pipeline_set_time_ms": pipeline_set_time * 1000,
                "pipeline_get_time_ms": pipeline_get_time * 1000,
                "pipeline_set_ops_per_sec": bulk_size / pipeline_set_time,
                "pipeline_get_ops_per_sec": bulk_size / pipeline_get_time,
                "mset_time_ms": mset_time * 1000,
                "mget_time_ms": mget_time * 1000,
                "mset_ops_per_sec": bulk_size / mset_time,
                "mget_ops_per_sec": bulk_size / mget_time,
                "data_integrity_check": len(retrieved_values) == bulk_size,
            }

            # Cleanup
            pipe = client.pipeline()
            for key in test_data:
                pipe.delete(key)
            pipe.execute()

        benchmark_results.add_benchmark("redis_performance", "bulk_operations", results)
        logger.info("Redis bulk operations performance testing completed")

    def test_vector_storage_performance(self, benchmark_results, redis_pool):
        """Test performance of storing and retrieving vectors in Redis"""
        logger.info("Testing Redis vector storage performance...")

        client = redis.Redis(connection_pool=redis_pool)

        # Test configurations
        test_configs = [
            (100, 384),  # Small dataset
            (1000, 384),  # Medium dataset
            (5000, 384),  # Large dataset
        ]

        results = {}

        for num_vectors, vector_dim in test_configs:
            config_name = f"{num_vectors}_vectors_{vector_dim}d"
            logger.info(f"Testing configuration: {config_name}")

            # Generate test vectors
            vectors = np.random.rand(num_vectors, vector_dim).astype(np.float32)

            # Test JSON serialization approach
            start_time = time.perf_counter()
            pipe = client.pipeline()
            for i, vector in enumerate(vectors):
                vector_data = {
                    "id": i,
                    "vector": vector.tolist(),
                    "metadata": {"source": f"test_doc_{i}", "timestamp": time.time()},
                }
                pipe.set(f"json_vector:{i}", json.dumps(vector_data))
            pipe.execute()
            json_store_time = time.perf_counter() - start_time

            # Test JSON retrieval
            start_time = time.perf_counter()
            pipe = client.pipeline()
            for i in range(num_vectors):
                pipe.get(f"json_vector:{i}")
            json_results = pipe.execute()
            json_retrieve_time = time.perf_counter() - start_time

            # Test binary serialization (NumPy)
            start_time = time.perf_counter()
            pipe = client.pipeline()
            for i, vector in enumerate(vectors):
                vector_bytes = vector.tobytes()
                pipe.set(f"binary_vector:{i}", vector_bytes)
            pipe.execute()
            binary_store_time = time.perf_counter() - start_time

            # Test binary retrieval
            start_time = time.perf_counter()
            pipe = client.pipeline()
            for i in range(num_vectors):
                pipe.get(f"binary_vector:{i}")
            binary_results = pipe.execute()
            binary_retrieve_time = time.perf_counter() - start_time

            # Calculate storage efficiency
            json_sample = json.dumps(
                {
                    "id": 0,
                    "vector": vectors[0].tolist(),
                    "metadata": {"source": "test", "timestamp": 0},
                }
            )
            json_size = len(json_sample.encode())
            binary_size = vectors[0].nbytes

            results[config_name] = {
                "num_vectors": num_vectors,
                "vector_dimension": vector_dim,
                "json_store_time_ms": json_store_time * 1000,
                "json_retrieve_time_ms": json_retrieve_time * 1000,
                "json_store_ops_per_sec": num_vectors / json_store_time,
                "json_retrieve_ops_per_sec": num_vectors / json_retrieve_time,
                "binary_store_time_ms": binary_store_time * 1000,
                "binary_retrieve_time_ms": binary_retrieve_time * 1000,
                "binary_store_ops_per_sec": num_vectors / binary_store_time,
                "binary_retrieve_ops_per_sec": num_vectors / binary_retrieve_time,
                "json_bytes_per_vector": json_size,
                "binary_bytes_per_vector": binary_size,
                "storage_efficiency_ratio": binary_size / json_size,
            }

            # Cleanup
            pipe = client.pipeline()
            for i in range(num_vectors):
                pipe.delete(f"json_vector:{i}")
                pipe.delete(f"binary_vector:{i}")
            pipe.execute()

        benchmark_results.add_benchmark("redis_performance", "vector_storage", results)
        logger.info("Redis vector storage performance testing completed")


class TestMemoryOptimization:
    """Test memory usage optimization"""

    def test_memory_usage_tracking(self, benchmark_results):
        """Track memory usage during various operations"""
        logger.info("Testing memory usage tracking...")

        process = psutil.Process()

        def get_memory_info():
            """Get current memory usage"""
            info = process.memory_info()
            return {
                "rss_mb": info.rss / 1024 / 1024,
                "vms_mb": info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
            }

        results = {}

        # Baseline memory
        gc.collect()
        baseline_memory = get_memory_info()
        results["baseline"] = baseline_memory

        # Large numpy array allocation
        large_array = np.random.rand(10000, 1000).astype(np.float32)
        after_allocation_memory = get_memory_info()
        results["after_large_allocation"] = after_allocation_memory

        # Memory usage with Redis operations
        try:
            import redis

            from src.shared.config.redis_test_utils import get_test_redis_config

            config = get_test_redis_config()
            config["db"] = 2  # Use separate DB for memory tests
            client = redis.Redis(**config)
            client.ping()

            # Store large amount of data
            for i in range(1000):
                client.set(f"memory_test_{i}", "x" * 1000)

            after_redis_memory = get_memory_info()
            results["after_redis_operations"] = after_redis_memory

            # Cleanup Redis data
            for i in range(1000):
                client.delete(f"memory_test_{i}")

        except redis.ConnectionError:
            logger.warning("Redis not available for memory testing")
            results["after_redis_operations"] = "redis_unavailable"

        # Cleanup large array
        del large_array
        gc.collect()
        after_cleanup_memory = get_memory_info()
        results["after_cleanup"] = after_cleanup_memory

        # Calculate memory deltas
        if "after_redis_operations" != "redis_unavailable":
            results["memory_deltas"] = {
                "allocation_impact_mb": after_allocation_memory["rss_mb"]
                - baseline_memory["rss_mb"],
                "redis_impact_mb": after_redis_memory["rss_mb"] - after_allocation_memory["rss_mb"],
                "cleanup_recovered_mb": after_redis_memory["rss_mb"]
                - after_cleanup_memory["rss_mb"],
            }

        benchmark_results.add_benchmark("memory_optimization", "memory_tracking", results)
        logger.info("Memory usage tracking completed")

    def test_garbage_collection_performance(self, benchmark_results):
        """Test garbage collection performance impact"""
        logger.info("Testing garbage collection performance...")

        results = {}

        # Test with different object creation patterns
        test_scenarios = {
            "small_objects": lambda: [{"id": i, "data": f"item_{i}"} for i in range(10000)],
            "large_objects": lambda: [
                {"id": i, "data": "x" * 1000, "array": list(range(100))} for i in range(1000)
            ],
            "numpy_arrays": lambda: [np.random.rand(100, 100) for _ in range(100)],
        }

        for scenario_name, scenario_func in test_scenarios.items():
            gc.collect()  # Start clean

            # Measure object creation time
            start_time = time.perf_counter()
            objects = scenario_func()
            creation_time = time.perf_counter() - start_time

            # Measure garbage collection time
            start_time = time.perf_counter()
            del objects
            gc.collect()
            gc_time = time.perf_counter() - start_time

            results[scenario_name] = {
                "creation_time_ms": creation_time * 1000,
                "gc_time_ms": gc_time * 1000,
                "total_time_ms": (creation_time + gc_time) * 1000,
                "gc_overhead_percent": (gc_time / (creation_time + gc_time)) * 100,
            }

        benchmark_results.add_benchmark("memory_optimization", "garbage_collection", results)
        logger.info("Garbage collection performance testing completed")


class TestThroughputLatency:
    """Test system throughput and latency characteristics"""

    def test_concurrent_request_throughput(self, benchmark_results):
        """Test throughput under concurrent load"""
        logger.info("Testing concurrent request throughput...")

        def cpu_intensive_task(task_id: int, iterations: int = 10000) -> dict[str, Any]:
            """Simulate CPU intensive task"""
            start_time = time.perf_counter()

            # Compute intensive operation
            result = sum(i * i for i in range(iterations))

            end_time = time.perf_counter()
            return {"task_id": task_id, "result": result, "duration": end_time - start_time}

        # Test different concurrency levels
        concurrency_levels = [1, 2, 4, 8, 16]
        tasks_per_level = 50

        results = {}

        for concurrency in concurrency_levels:
            logger.info(f"Testing concurrency level: {concurrency}")

            start_time = time.perf_counter()

            with ProcessPoolExecutor(max_workers=concurrency) as executor:
                futures = []
                for task_id in range(tasks_per_level):
                    future = executor.submit(cpu_intensive_task, task_id)
                    futures.append(future)

                task_results = [future.result() for future in futures]

            total_time = time.perf_counter() - start_time

            # Analyze results
            task_durations = [r["duration"] for r in task_results]

            results[f"concurrency_{concurrency}"] = {
                "total_time_seconds": total_time,
                "total_tasks": tasks_per_level,
                "tasks_per_second": tasks_per_level / total_time,
                "mean_task_duration": mean(task_durations),
                "median_task_duration": median(task_durations),
                "std_task_duration": stdev(task_durations) if len(task_durations) > 1 else 0,
                "cpu_cores_used": concurrency,
                "efficiency_ratio": (sum(task_durations) / total_time) / concurrency * 100,
            }

        benchmark_results.add_benchmark("throughput_latency", "concurrent_requests", results)
        logger.info("Concurrent request throughput testing completed")

    def test_latency_percentiles(self, benchmark_results):
        """Test latency distribution and percentiles"""
        logger.info("Testing latency percentiles...")

        def timed_operation(operation_type: str) -> float:
            """Perform timed operation and return duration"""
            start_time = time.perf_counter()

            if operation_type == "cpu_light":
                result = sum(range(1000))
            elif operation_type == "cpu_medium":
                result = sum(i * i for i in range(10000))
            elif operation_type == "cpu_heavy":
                result = sum(i * i * i for i in range(50000))
            elif operation_type == "memory_access":
                arr = np.random.rand(10000)
                result = np.sum(arr)
            else:
                time.sleep(0.001)  # I/O simulation
                result = 1

            return time.perf_counter() - start_time

        operation_types = ["cpu_light", "cpu_medium", "cpu_heavy", "memory_access", "io_simulation"]
        iterations = 1000

        results = {}

        for operation_type in operation_types:
            logger.info(f"Testing operation type: {operation_type}")

            durations = []
            for _ in range(iterations):
                duration = timed_operation(operation_type)
                durations.append(duration * 1000)  # Convert to milliseconds

            durations.sort()

            # Calculate percentiles
            percentiles = [50, 75, 90, 95, 99, 99.9]
            percentile_values = {}

            for p in percentiles:
                index = int((p / 100) * len(durations))
                if index >= len(durations):
                    index = len(durations) - 1
                percentile_values[f"p{p}"] = durations[index]

            results[operation_type] = {
                "iterations": iterations,
                "mean_ms": mean(durations),
                "median_ms": median(durations),
                "std_ms": stdev(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                **percentile_values,
            }

        benchmark_results.add_benchmark("throughput_latency", "latency_percentiles", results)
        logger.info("Latency percentiles testing completed")


def run_performance_benchmarks():
    """Run all performance benchmarks"""
    logger.info("Starting performance benchmark suite...")

    # Create results directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / "test_results" / f"performance_benchmarks_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run pytest for performance tests
    pytest_args = [
        str(__file__),
        "-v",
        "-s",  # Show print statements
        "--tb=short",
        f"--junitxml={results_dir}/performance_results.xml",
    ]

    exit_code = pytest.main(pytest_args)

    logger.info(f"Performance benchmarks completed with exit code: {exit_code}")
    logger.info(f"Results will be saved to: {results_dir}")

    return exit_code


if __name__ == "__main__":
    exit_code = run_performance_benchmarks()
    sys.exit(exit_code)
