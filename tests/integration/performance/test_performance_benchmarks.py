"""
Performance Benchmark Tests for LLM System

Comprehensive performance testing including:
- Inference latency benchmarks
- Concurrent request handling
- Memory usage profiling
- Redis operation throughput
- End-to-end pipeline performance
"""

import asyncio
import pytest
import time
import psutil
import tracemalloc
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import numpy as np
from unittest.mock import MagicMock, patch

# Performance thresholds (adjust based on hardware)
THRESHOLDS = {
    'inference_latency_p95_ms': 500,  # 95th percentile latency
    'inference_latency_p99_ms': 1000,  # 99th percentile latency
    'requests_per_second': 10,  # Minimum RPS
    'memory_growth_mb': 100,  # Max memory growth during test
    'redis_ops_per_second': 1000,  # Minimum Redis operations
    'rag_search_latency_ms': 100,  # RAG search latency
    'agent_response_time_ms': 2000,  # Full agent response time
}


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    duration_seconds: float
    operations: int
    latencies_ms: List[float]
    memory_start_mb: float
    memory_end_mb: float
    memory_peak_mb: float
    cpu_percent: float
    errors: int = 0
    metadata: Dict[str, Any] = None

    @property
    def throughput(self) -> float:
        """Operations per second."""
        return self.operations / self.duration_seconds if self.duration_seconds > 0 else 0

    @property
    def latency_p50(self) -> float:
        """50th percentile latency."""
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0

    @property
    def latency_p95(self) -> float:
        """95th percentile latency."""
        if not self.latencies_ms:
            return 0
        sorted_latencies = sorted(self.latencies_ms)
        index = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]

    @property
    def latency_p99(self) -> float:
        """99th percentile latency."""
        if not self.latencies_ms:
            return 0
        sorted_latencies = sorted(self.latencies_ms)
        index = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]

    @property
    def memory_growth_mb(self) -> float:
        """Memory growth during test."""
        return self.memory_end_mb - self.memory_start_mb

    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting."""
        return {
            'name': self.name,
            'duration_seconds': round(self.duration_seconds, 2),
            'operations': self.operations,
            'throughput_ops_per_sec': round(self.throughput, 2),
            'latency_p50_ms': round(self.latency_p50, 2),
            'latency_p95_ms': round(self.latency_p95, 2),
            'latency_p99_ms': round(self.latency_p99, 2),
            'memory_start_mb': round(self.memory_start_mb, 2),
            'memory_end_mb': round(self.memory_end_mb, 2),
            'memory_peak_mb': round(self.memory_peak_mb, 2),
            'memory_growth_mb': round(self.memory_growth_mb, 2),
            'cpu_percent': round(self.cpu_percent, 2),
            'errors': self.errors,
            'metadata': self.metadata or {}
        }


class PerformanceBenchmark:
    """Base class for performance benchmarks."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def start_monitoring(self):
        """Start performance monitoring."""
        tracemalloc.start()
        process = psutil.Process()
        return {
            'start_time': time.perf_counter(),
            'start_memory': process.memory_info().rss / 1024 / 1024,
            'process': process
        }

    def stop_monitoring(self, monitor_data: Dict) -> Dict:
        """Stop performance monitoring."""
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        process = monitor_data['process']
        return {
            'duration': time.perf_counter() - monitor_data['start_time'],
            'memory_start': monitor_data['start_memory'],
            'memory_end': process.memory_info().rss / 1024 / 1024,
            'memory_peak': peak / 1024 / 1024,
            'cpu_percent': process.cpu_percent(interval=0.1)
        }

    async def run_benchmark(self, name: str, func, iterations: int = 100) -> BenchmarkResult:
        """Run a benchmark and collect metrics."""
        monitor = self.start_monitoring()
        latencies = []
        errors = 0

        for _ in range(iterations):
            start = time.perf_counter()
            try:
                if asyncio.iscoroutinefunction(func):
                    await func()
                else:
                    func()
                latencies.append((time.perf_counter() - start) * 1000)  # Convert to ms
            except Exception:
                errors += 1

        metrics = self.stop_monitoring(monitor)

        result = BenchmarkResult(
            name=name,
            duration_seconds=metrics['duration'],
            operations=iterations,
            latencies_ms=latencies,
            memory_start_mb=metrics['memory_start'],
            memory_end_mb=metrics['memory_end'],
            memory_peak_mb=metrics['memory_peak'],
            cpu_percent=metrics['cpu_percent'],
            errors=errors
        )

        self.results.append(result)
        return result


class TestInferencePerformance(PerformanceBenchmark):
    """Benchmark inference performance."""

    @pytest.mark.asyncio
    async def test_single_inference_latency(self, mock_gemma_model):
        """Benchmark single inference latency."""
        # Mock model with controlled latency
        async def mock_generate(prompt: str, max_tokens: int = 100):
            await asyncio.sleep(0.05)  # Simulate 50ms inference
            return "Generated response " * 10

        mock_gemma_model.generate = mock_generate

        # Benchmark
        async def single_inference():
            return await mock_gemma_model.generate("Test prompt", max_tokens=100)

        result = await self.run_benchmark(
            "single_inference",
            single_inference,
            iterations=100
        )

        # Assertions
        assert result.latency_p95 < THRESHOLDS['inference_latency_p95_ms']
        assert result.latency_p99 < THRESHOLDS['inference_latency_p99_ms']
        assert result.errors == 0

    @pytest.mark.asyncio
    async def test_batch_inference_throughput(self, mock_gemma_model):
        """Benchmark batch inference throughput."""
        batch_size = 10

        async def batch_inference():
            prompts = [f"Prompt {i}" for i in range(batch_size)]
            tasks = [mock_gemma_model.generate(p) for p in prompts]
            return await asyncio.gather(*tasks)

        result = await self.run_benchmark(
            "batch_inference",
            batch_inference,
            iterations=20
        )

        # Calculate effective throughput
        effective_throughput = (result.operations * batch_size) / result.duration_seconds

        assert effective_throughput > THRESHOLDS['requests_per_second']
        assert result.memory_growth_mb < THRESHOLDS['memory_growth_mb']

    @pytest.mark.asyncio
    async def test_streaming_inference(self):
        """Benchmark streaming inference performance."""
        async def stream_tokens(prompt: str):
            """Simulate streaming token generation."""
            tokens = prompt.split()
            for token in tokens:
                await asyncio.sleep(0.01)  # 10ms per token
                yield token

        async def streaming_inference():
            tokens = []
            async for token in stream_tokens("This is a test prompt"):
                tokens.append(token)
            return ' '.join(tokens)

        result = await self.run_benchmark(
            "streaming_inference",
            streaming_inference,
            iterations=50
        )

        # Streaming should have predictable latency
        latency_variance = statistics.variance(result.latencies_ms) if len(result.latencies_ms) > 1 else 0
        assert latency_variance < 100  # Low variance expected

    def test_model_loading_time(self):
        """Benchmark model loading time."""
        def mock_load_model(model_path: str):
            # Simulate model loading
            time.sleep(0.1)  # 100ms load time
            return MagicMock()

        monitor = self.start_monitoring()

        # Load multiple model variants
        models = []
        for i in range(3):
            start = time.perf_counter()
            model = mock_load_model(f"model_{i}.bin")
            load_time_ms = (time.perf_counter() - start) * 1000
            models.append(model)
            assert load_time_ms < 200  # Should load in under 200ms

        metrics = self.stop_monitoring(monitor)
        assert metrics['memory_peak'] < 500  # Model loading should be memory efficient


class TestConcurrencyPerformance(PerformanceBenchmark):
    """Benchmark concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_agent_requests(self, react_agent):
        """Test multiple concurrent agent requests."""
        num_concurrent = 20

        async def agent_request(idx: int):
            return await react_agent.arun(f"Query {idx}")

        # Run concurrent requests
        start = time.perf_counter()
        tasks = [agent_request(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.perf_counter() - start

        # Calculate metrics
        successful = sum(1 for r in results if not isinstance(r, Exception))
        throughput = successful / duration

        assert throughput > 5  # Should handle at least 5 req/s
        assert successful >= num_concurrent * 0.95  # 95% success rate

    @pytest.mark.asyncio
    async def test_connection_pool_efficiency(self, async_redis_client):
        """Test Redis connection pool under load."""
        num_connections = 50
        operations_per_connection = 100

        async def redis_operations(conn_id: int):
            latencies = []
            for op_id in range(operations_per_connection):
                start = time.perf_counter()
                key = f"bench:conn{conn_id}:op{op_id}"
                await async_redis_client.set(key, f"value_{op_id}")
                await async_redis_client.get(key)
                latencies.append((time.perf_counter() - start) * 1000)
            return latencies

        # Run concurrent Redis operations
        start_time = time.perf_counter()
        tasks = [redis_operations(i) for i in range(num_connections)]
        all_latencies = await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - start_time

        # Flatten latencies and calculate stats
        flat_latencies = [lat for conn_lats in all_latencies for lat in conn_lats]
        p95_latency = sorted(flat_latencies)[int(len(flat_latencies) * 0.95)]

        total_ops = num_connections * operations_per_connection * 2  # set + get
        throughput = total_ops / total_duration

        assert throughput > THRESHOLDS['redis_ops_per_second']
        assert p95_latency < 10  # Redis ops should be < 10ms at p95

    def test_thread_pool_scaling(self):
        """Test thread pool scaling for CPU-bound tasks."""
        def cpu_bound_task(n: int) -> int:
            """Simulate CPU-intensive work."""
            result = 0
            for i in range(n * 1000):
                result += i ** 2
            return result

        # Test different thread pool sizes
        pool_sizes = [1, 2, 4, 8]
        scaling_results = {}

        for pool_size in pool_sizes:
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=pool_size) as executor:
                futures = [executor.submit(cpu_bound_task, 100) for _ in range(16)]
                results = [f.result() for f in as_completed(futures)]
            duration = time.perf_counter() - start
            scaling_results[pool_size] = duration

        # Verify scaling (should improve up to CPU count)
        cpu_count = psutil.cpu_count()
        if cpu_count >= 4:
            # 4 threads should be faster than 1 thread
            assert scaling_results[4] < scaling_results[1] * 0.5


class TestMemoryPerformance(PerformanceBenchmark):
    """Benchmark memory usage and efficiency."""

    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, react_agent):
        """Test for memory leaks during extended operation."""
        iterations = 100
        memory_samples = []

        process = psutil.Process()

        for i in range(iterations):
            if i % 10 == 0:
                memory_samples.append(process.memory_info().rss / 1024 / 1024)

            # Perform operations that might leak
            await react_agent.arun(f"Process item {i}")

            # Small delay to allow GC
            if i % 20 == 0:
                await asyncio.sleep(0.1)

        # Analyze memory trend
        memory_growth = memory_samples[-1] - memory_samples[0]
        avg_growth_per_10 = memory_growth / (len(memory_samples) - 1)

        # Should not grow more than 1MB per 10 operations
        assert avg_growth_per_10 < 1.0

    @pytest.mark.asyncio
    async def test_large_context_handling(self):
        """Test memory efficiency with large contexts."""
        # Create large documents
        large_doc = "Lorem ipsum " * 10000  # ~100KB
        num_docs = 50

        memory_before = psutil.Process().memory_info().rss / 1024 / 1024

        # Process large documents
        processed = []
        for i in range(num_docs):
            # Simulate processing
            doc_tokens = large_doc.split()
            processed.append(len(doc_tokens))

        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before

        # Should handle 5MB of text with < 100MB memory overhead
        assert memory_used < 100

    @pytest.mark.asyncio
    async def test_vector_operations_memory(self):
        """Test memory efficiency of vector operations."""
        vector_dim = 768  # Common embedding dimension
        num_vectors = 10000

        memory_before = psutil.Process().memory_info().rss / 1024 / 1024

        # Create and manipulate vectors
        vectors = np.random.randn(num_vectors, vector_dim).astype(np.float32)

        # Perform common operations
        normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        similarities = np.dot(normalized[:100], normalized.T)

        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before

        # Theoretical size: 10000 * 768 * 4 bytes = ~30MB
        # Should use < 100MB with overhead
        assert memory_used < 100


class TestRAGPerformance(PerformanceBenchmark):
    """Benchmark RAG system performance."""

    @pytest.mark.asyncio
    async def test_document_ingestion_rate(self, async_redis_client):
        """Test document ingestion performance."""
        num_documents = 1000

        documents = [
            {
                'id': f'doc_{i}',
                'content': f'Document content {i} ' + 'text ' * 100,
                'embedding': [0.1] * 384
            }
            for i in range(num_documents)
        ]

        async def ingest_document(doc: Dict):
            # Store document
            await async_redis_client.hset(
                f"doc:{doc['id']}",
                mapping={'content': doc['content']}
            )
            # Store embedding
            await async_redis_client.set(
                f"embedding:{doc['id']}",
                json.dumps(doc['embedding'])
            )
            # Add to index
            await async_redis_client.zadd("doc:index", {doc['id']: 1.0})

        # Benchmark ingestion
        start = time.perf_counter()
        tasks = [ingest_document(doc) for doc in documents]
        await asyncio.gather(*tasks)
        duration = time.perf_counter() - start

        ingestion_rate = num_documents / duration

        assert ingestion_rate > 100  # Should ingest > 100 docs/sec
        assert duration < 30  # Should complete in < 30 seconds

    @pytest.mark.asyncio
    async def test_similarity_search_performance(self):
        """Test vector similarity search performance."""
        num_vectors = 5000
        vector_dim = 384
        num_searches = 100

        # Create vector database
        vectors = np.random.randn(num_vectors, vector_dim).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        def similarity_search(query_vector: np.ndarray, top_k: int = 10):
            similarities = np.dot(vectors, query_vector)
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            return top_indices[np.argsort(similarities[top_indices])][::-1]

        # Benchmark searches
        search_latencies = []
        for _ in range(num_searches):
            query = np.random.randn(vector_dim).astype(np.float32)
            query = query / np.linalg.norm(query)

            start = time.perf_counter()
            results = similarity_search(query)
            search_latencies.append((time.perf_counter() - start) * 1000)

        p95_latency = sorted(search_latencies)[int(num_searches * 0.95)]

        assert p95_latency < THRESHOLDS['rag_search_latency_ms']
        assert statistics.mean(search_latencies) < 50  # Average < 50ms


class TestEndToEndPerformance(PerformanceBenchmark):
    """Benchmark end-to-end system performance."""

    @pytest.mark.asyncio
    async def test_full_pipeline_latency(self, react_agent, async_redis_client):
        """Test complete request pipeline latency."""

        async def full_pipeline(query: str):
            # 1. Store query in Redis
            await async_redis_client.lpush("queries", query)

            # 2. Agent processing
            response = await react_agent.arun(query)

            # 3. Store response
            await async_redis_client.set(f"response:{hash(query)}", str(response))

            return response

        # Benchmark different query types
        queries = [
            "Simple factual question",
            "Complex reasoning task that requires multiple steps",
            "Code generation request for Python function",
        ]

        latencies = []
        for query in queries * 10:  # Run each query 10 times
            start = time.perf_counter()
            await full_pipeline(query)
            latencies.append((time.perf_counter() - start) * 1000)

        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        assert p95_latency < THRESHOLDS['agent_response_time_ms']
        assert statistics.mean(latencies) < 1000  # Average < 1 second

    @pytest.mark.asyncio
    async def test_system_saturation_point(self):
        """Find system saturation point for concurrent requests."""
        async def mock_request(idx: int):
            await asyncio.sleep(0.1)  # Simulate processing
            return f"Response {idx}"

        # Test increasing concurrency levels
        concurrency_levels = [1, 5, 10, 20, 50, 100]
        saturation_metrics = []

        for concurrency in concurrency_levels:
            start = time.perf_counter()
            tasks = [mock_request(i) for i in range(concurrency)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.perf_counter() - start

            successful = sum(1 for r in results if not isinstance(r, Exception))
            throughput = successful / duration

            saturation_metrics.append({
                'concurrency': concurrency,
                'throughput': throughput,
                'latency': duration * 1000 / concurrency,
                'success_rate': successful / concurrency
            })

        # Find saturation point (where throughput stops increasing)
        max_throughput = max(m['throughput'] for m in saturation_metrics)
        saturation_point = None

        for i, metric in enumerate(saturation_metrics[1:], 1):
            if metric['throughput'] < saturation_metrics[i-1]['throughput'] * 0.95:
                saturation_point = saturation_metrics[i-1]['concurrency']
                break

        # System should handle at least 20 concurrent requests efficiently
        assert saturation_point is None or saturation_point >= 20


def generate_performance_report(results: List[BenchmarkResult]) -> str:
    """Generate a performance report from benchmark results."""
    report = ["=" * 80]
    report.append("PERFORMANCE BENCHMARK REPORT")
    report.append("=" * 80)
    report.append("")

    for result in results:
        report.append(f"Benchmark: {result.name}")
        report.append("-" * 40)
        report.append(f"Duration: {result.duration_seconds:.2f}s")
        report.append(f"Operations: {result.operations}")
        report.append(f"Throughput: {result.throughput:.2f} ops/sec")
        report.append(f"Latency P50: {result.latency_p50:.2f}ms")
        report.append(f"Latency P95: {result.latency_p95:.2f}ms")
        report.append(f"Latency P99: {result.latency_p99:.2f}ms")
        report.append(f"Memory Growth: {result.memory_growth_mb:.2f}MB")
        report.append(f"Peak Memory: {result.memory_peak_mb:.2f}MB")
        report.append(f"CPU Usage: {result.cpu_percent:.1f}%")
        if result.errors > 0:
            report.append(f"Errors: {result.errors}")
        report.append("")

    # Summary statistics
    report.append("=" * 80)
    report.append("SUMMARY")
    report.append("-" * 40)

    if results:
        avg_throughput = statistics.mean(r.throughput for r in results)
        avg_latency_p95 = statistics.mean(r.latency_p95 for r in results)
        total_memory = sum(r.memory_growth_mb for r in results)
        max_cpu = max(r.cpu_percent for r in results)

        report.append(f"Average Throughput: {avg_throughput:.2f} ops/sec")
        report.append(f"Average P95 Latency: {avg_latency_p95:.2f}ms")
        report.append(f"Total Memory Growth: {total_memory:.2f}MB")
        report.append(f"Peak CPU Usage: {max_cpu:.1f}%")

    report.append("=" * 80)
    return "\n".join(report)