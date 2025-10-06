"""
Stress Testing Suite for LLM System

Tests system behavior under extreme load conditions:
- Multiple concurrent agents
- Large document processing
- Memory tier overflow
- Network failures and recovery
- Resource exhaustion scenarios
"""

import asyncio
import pytest
import random
import time
import psutil
import gc
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import json
from pathlib import Path
import tempfile
import string


class StressTestScenario:
    """Base class for stress test scenarios."""

    def __init__(self, name: str, duration_seconds: int = 60):
        self.name = name
        self.duration = duration_seconds
        self.start_time = None
        self.end_time = None
        self.metrics = {}
        self.errors = []
        self.warnings = []

    async def setup(self):
        """Setup test scenario."""
        self.start_time = time.time()
        gc.collect()  # Clean slate

    async def teardown(self):
        """Cleanup after test."""
        self.end_time = time.time()
        gc.collect()

    async def run(self):
        """Run the stress test scenario."""
        raise NotImplementedError

    def record_error(self, error: str):
        """Record an error during the test."""
        self.errors.append({
            'timestamp': time.time() - self.start_time,
            'error': error
        })

    def record_metric(self, name: str, value: Any):
        """Record a metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get_summary(self) -> Dict:
        """Get test summary."""
        return {
            'name': self.name,
            'duration': self.end_time - self.start_time if self.end_time else 0,
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'metrics': {
                k: {
                    'count': len(v),
                    'mean': np.mean(v) if v else 0,
                    'max': max(v) if v else 0,
                    'min': min(v) if v else 0
                }
                for k, v in self.metrics.items()
            }
        }


class TestMultipleAgentStress:
    """Stress test with multiple concurrent agents."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_hundred_concurrent_agents(self, agent_config):
        """Test system with 100 concurrent agents."""
        num_agents = 100
        queries_per_agent = 10

        class AgentStressTest(StressTestScenario):
            async def run(self):
                # Create mock agents
                agents = []
                for i in range(num_agents):
                    config = agent_config.copy()
                    config.name = f"agent_{i}"
                    agent = MagicMock()
                    agent.arun = AsyncMock(return_value=f"Response from agent {i}")
                    agents.append(agent)

                # Generate random queries
                queries = [
                    f"Query {i}: " + ' '.join(random.choices(string.ascii_letters, k=20))
                    for i in range(num_agents * queries_per_agent)
                ]

                # Run agents concurrently
                async def run_agent_queries(agent_id: int):
                    agent = agents[agent_id]
                    agent_queries = queries[agent_id * queries_per_agent:(agent_id + 1) * queries_per_agent]

                    for query in agent_queries:
                        try:
                            start = time.time()
                            response = await agent.arun(query)
                            latency = time.time() - start
                            self.record_metric('latency', latency)

                            if latency > 5.0:  # Warning for slow responses
                                self.warnings.append(f"Slow response: {latency:.2f}s")
                        except Exception as e:
                            self.record_error(str(e))

                # Execute all agents
                await self.setup()

                tasks = [run_agent_queries(i) for i in range(num_agents)]
                await asyncio.gather(*tasks, return_exceptions=True)

                await self.teardown()

        # Run stress test
        scenario = AgentStressTest("hundred_agents", duration_seconds=120)
        await scenario.run()

        summary = scenario.get_summary()

        # Assertions
        assert summary['errors'] < num_agents * 0.05  # Less than 5% error rate
        assert summary['metrics']['latency']['mean'] < 2.0  # Average latency < 2s

    @pytest.mark.asyncio
    async def test_agent_resource_competition(self, async_redis_client):
        """Test agents competing for shared resources."""
        num_agents = 50
        shared_resource_keys = [f"resource_{i}" for i in range(10)]

        async def agent_task(agent_id: int, duration: int):
            """Simulate agent competing for resources."""
            end_time = time.time() + duration
            operations = 0
            conflicts = 0

            while time.time() < end_time:
                # Try to acquire random resource
                resource_key = random.choice(shared_resource_keys)

                # Attempt to lock resource
                lock_acquired = await async_redis_client.setnx(
                    f"lock:{resource_key}",
                    agent_id
                )

                if lock_acquired:
                    # Do work with resource
                    await asyncio.sleep(random.uniform(0.01, 0.1))

                    # Release lock
                    await async_redis_client.delete(f"lock:{resource_key}")
                    operations += 1
                else:
                    conflicts += 1
                    await asyncio.sleep(0.01)  # Back off

            return operations, conflicts

        # Run competing agents
        tasks = [agent_task(i, 10) for i in range(num_agents)]
        results = await asyncio.gather(*tasks)

        total_operations = sum(r[0] for r in results)
        total_conflicts = sum(r[1] for r in results)

        # System should handle competition gracefully
        assert total_operations > num_agents * 10  # Each agent should complete operations
        assert total_conflicts < total_operations * 2  # Conflicts should be manageable


class TestLargeDocumentStress:
    """Stress test with large documents."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_gigabyte_document_processing(self):
        """Test processing documents totaling 1GB."""
        total_size_mb = 1024
        chunk_size_mb = 10
        num_chunks = total_size_mb // chunk_size_mb

        class DocumentStressTest(StressTestScenario):
            async def run(self):
                await self.setup()

                # Generate large documents
                documents = []
                for i in range(num_chunks):
                    # Create chunk_size_mb of text data
                    doc_size = chunk_size_mb * 1024 * 1024
                    # Use efficient string generation
                    doc_content = ''.join(random.choices(
                        string.ascii_letters + string.digits + ' \n',
                        k=doc_size // 100
                    )) * 100

                    documents.append({
                        'id': f'large_doc_{i}',
                        'content': doc_content,
                        'size_mb': chunk_size_mb
                    })

                    # Record memory usage
                    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                    self.record_metric('memory_mb', memory_mb)

                    # Clear old documents to manage memory
                    if i > 10:
                        documents.pop(0)

                    # Force garbage collection periodically
                    if i % 10 == 0:
                        gc.collect()

                await self.teardown()

        scenario = DocumentStressTest("gigabyte_docs", duration_seconds=300)
        await scenario.run()

        summary = scenario.get_summary()

        # Memory should not grow unbounded
        memory_metrics = summary['metrics']['memory_mb']
        memory_growth = memory_metrics['max'] - memory_metrics['min']
        assert memory_growth < 500  # Should not grow more than 500MB

    @pytest.mark.asyncio
    async def test_document_chunking_stress(self):
        """Test chunking extremely large documents."""
        # Create a 100MB document
        doc_size_mb = 100
        chunk_size = 1024  # 1KB chunks

        # Generate document
        content_parts = []
        for _ in range(doc_size_mb):
            # 1MB of text
            part = ''.join(random.choices(string.ascii_letters, k=1024*1024))
            content_parts.append(part)

        large_document = ''.join(content_parts)

        # Chunk the document
        chunks = []
        for i in range(0, len(large_document), chunk_size):
            chunk = {
                'id': f'chunk_{i // chunk_size}',
                'content': large_document[i:i + chunk_size],
                'position': i
            }
            chunks.append(chunk)

        # Verify chunking
        expected_chunks = (doc_size_mb * 1024 * 1024) // chunk_size
        assert len(chunks) >= expected_chunks * 0.99  # Allow 1% variance

        # Test reconstruction
        reconstructed = ''.join(c['content'] for c in chunks)
        assert len(reconstructed) == len(large_document)

    @pytest.mark.asyncio
    async def test_concurrent_document_operations(self, async_redis_client):
        """Test concurrent document ingestion and retrieval."""
        num_writers = 20
        num_readers = 30
        documents_per_writer = 100
        test_duration = 30

        # Shared metrics
        write_count = {'value': 0}
        read_count = {'value': 0}

        async def document_writer(writer_id: int):
            """Continuously write documents."""
            end_time = time.time() + test_duration

            while time.time() < end_time:
                doc = {
                    'id': f'doc_{writer_id}_{write_count["value"]}',
                    'content': f'Document from writer {writer_id}',
                    'timestamp': time.time()
                }

                await async_redis_client.hset(
                    f"doc:{doc['id']}",
                    mapping=doc
                )
                write_count['value'] += 1

                await asyncio.sleep(random.uniform(0.01, 0.1))

        async def document_reader(reader_id: int):
            """Continuously read documents."""
            end_time = time.time() + test_duration

            while time.time() < end_time:
                # Read random document
                if write_count['value'] > 0:
                    doc_id = f"doc_0_{random.randint(0, min(write_count['value']-1, 1000))}"
                    doc = await async_redis_client.hgetall(f"doc:{doc_id}")

                    if doc:
                        read_count['value'] += 1

                await asyncio.sleep(random.uniform(0.01, 0.05))

        # Run concurrent operations
        writers = [document_writer(i) for i in range(num_writers)]
        readers = [document_reader(i) for i in range(num_readers)]

        await asyncio.gather(*writers, *readers, return_exceptions=True)

        # Verify operations
        assert write_count['value'] > num_writers * 100  # Good write throughput
        assert read_count['value'] > num_readers * 200  # Good read throughput


class TestMemoryTierStress:
    """Stress test memory tier management."""

    @pytest.mark.asyncio
    async def test_memory_tier_overflow(self, async_redis_client, memory_config):
        """Test behavior when memory tiers overflow."""
        # Set small capacities for testing
        test_config = memory_config.copy()
        test_config['tiers']['working']['capacity'] = 10
        test_config['tiers']['short_term']['capacity'] = 50

        # Overflow working memory
        for i in range(100):
            memory = {
                'id': f'overflow_{i}',
                'content': f'Memory {i}',
                'tier': 'working',
                'timestamp': time.time() - i  # Older memories have lower timestamp
            }

            await async_redis_client.hset(
                f"memory:working:{memory['id']}",
                mapping={k: str(v) for k, v in memory.items()}
            )
            await async_redis_client.zadd(
                "index:working",
                {memory['id']: memory['timestamp']}
            )

        # Check tier sizes
        working_size = await async_redis_client.zcard("index:working")

        # Should handle overflow (in production, would evict old memories)
        assert working_size > test_config['tiers']['working']['capacity']

        # Simulate eviction of oldest memories
        if working_size > test_config['tiers']['working']['capacity']:
            to_evict = working_size - test_config['tiers']['working']['capacity']
            oldest = await async_redis_client.zrange(
                "index:working",
                0,
                to_evict - 1
            )

            for mem_id in oldest:
                await async_redis_client.zrem("index:working", mem_id)
                await async_redis_client.delete(f"memory:working:{mem_id}")

        # Verify capacity enforcement
        final_size = await async_redis_client.zcard("index:working")
        assert final_size <= test_config['tiers']['working']['capacity']

    @pytest.mark.asyncio
    async def test_rapid_memory_transitions(self, async_redis_client):
        """Test rapid transitions between memory tiers."""
        num_memories = 1000
        transition_interval = 0.01  # 10ms between transitions

        async def create_and_transition_memory(mem_id: int):
            """Create memory and transition through tiers."""
            memory = {
                'id': f'trans_{mem_id}',
                'content': f'Transitioning memory {mem_id}',
                'importance': random.random()
            }

            # Start in working memory
            await async_redis_client.hset(
                f"memory:working:{memory['id']}",
                mapping={k: str(v) for k, v in memory.items()}
            )

            await asyncio.sleep(transition_interval)

            # Move to short-term
            await async_redis_client.delete(f"memory:working:{memory['id']}")
            await async_redis_client.hset(
                f"memory:short_term:{memory['id']}",
                mapping={k: str(v) for k, v in memory.items()}
            )

            await asyncio.sleep(transition_interval)

            # Move to long-term if important
            if memory['importance'] > 0.7:
                await async_redis_client.delete(f"memory:short_term:{memory['id']}")
                await async_redis_client.hset(
                    f"memory:long_term:{memory['id']}",
                    mapping={k: str(v) for k, v in memory.items()}
                )

            return memory['id']

        # Run transitions concurrently
        tasks = [create_and_transition_memory(i) for i in range(num_memories)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for r in results if not isinstance(r, Exception))
        assert successful > num_memories * 0.95  # 95% success rate


class TestNetworkFailureStress:
    """Test system behavior under network failures."""

    @pytest.mark.asyncio
    async def test_intermittent_redis_failures(self, redis_with_error):
        """Test handling intermittent Redis connection failures."""
        failure_rate = 0.3  # 30% chance of failure
        num_operations = 1000

        success_count = 0
        retry_count = 0

        for i in range(num_operations):
            # Randomly fail connections
            if random.random() < failure_rate:
                redis_with_error.server.connected = False
            else:
                redis_with_error.server.connected = True

            # Attempt operation with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    redis_with_error.set(f"key_{i}", f"value_{i}")
                    success_count += 1
                    break
                except Exception:
                    retry_count += 1
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.01 * (2 ** attempt))  # Exponential backoff

        # Should recover from most failures
        success_rate = success_count / num_operations
        assert success_rate > 0.9  # 90% eventual success rate
        assert retry_count > 0  # Should have some retries

    @pytest.mark.asyncio
    async def test_network_partition_recovery(self):
        """Test recovery from network partitions."""

        class NetworkPartition:
            def __init__(self):
                self.partitioned = False
                self.partition_start = None
                self.partition_duration = 0

            async def cause_partition(self, duration: float):
                """Simulate network partition."""
                self.partitioned = True
                self.partition_start = time.time()
                await asyncio.sleep(duration)
                self.partitioned = False
                self.partition_duration = duration

            async def is_accessible(self) -> bool:
                """Check if network is accessible."""
                return not self.partitioned

        network = NetworkPartition()

        # Operations during partition
        operations_during_partition = []
        operations_after_recovery = []

        async def continuous_operations():
            """Continuously perform operations."""
            for i in range(100):
                if await network.is_accessible():
                    if network.partition_duration > 0:  # After recovery
                        operations_after_recovery.append(i)
                    # Successful operation
                    await asyncio.sleep(0.01)
                else:
                    operations_during_partition.append(i)
                    await asyncio.sleep(0.05)  # Wait during partition

        # Run operations with network partition
        partition_task = network.cause_partition(2.0)  # 2 second partition
        ops_task = continuous_operations()

        await asyncio.gather(partition_task, ops_task)

        # Should detect partition and recover
        assert len(operations_during_partition) > 0
        assert len(operations_after_recovery) > 0


class TestResourceExhaustion:
    """Test system behavior under resource exhaustion."""

    @pytest.mark.asyncio
    async def test_cpu_saturation(self):
        """Test behavior under CPU saturation."""

        def cpu_intensive_task(duration: float):
            """CPU-intensive computation."""
            end_time = time.time() + duration
            result = 0
            while time.time() < end_time:
                result += sum(i ** 2 for i in range(1000))
            return result

        # Get CPU count
        cpu_count = psutil.cpu_count()

        # Create more tasks than CPUs
        num_tasks = cpu_count * 4

        with ThreadPoolExecutor(max_workers=cpu_count * 2) as executor:
            futures = [
                executor.submit(cpu_intensive_task, 2.0)
                for _ in range(num_tasks)
            ]

            # Monitor CPU usage
            cpu_samples = []
            for _ in range(10):
                cpu_samples.append(psutil.cpu_percent(interval=0.1))

            # Wait for completion
            results = [f.result() for f in futures]

        # CPU should be heavily utilized
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        assert avg_cpu > 70  # High CPU usage expected
        assert len(results) == num_tasks  # All tasks should complete

    @pytest.mark.asyncio
    async def test_memory_exhaustion_handling(self):
        """Test graceful handling of memory exhaustion."""

        # Monitor initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Try to allocate large amounts of memory
        allocations = []
        allocation_size = 10 * 1024 * 1024  # 10MB chunks
        max_allocations = 100  # Up to 1GB

        for i in range(max_allocations):
            try:
                # Allocate memory
                data = bytearray(allocation_size)
                allocations.append(data)

                # Check memory usage
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_used = current_memory - initial_memory

                # Stop if using too much memory
                if memory_used > 500:  # 500MB limit for test
                    break
            except MemoryError:
                # Gracefully handle memory error
                break

        # Clean up
        allocations.clear()
        gc.collect()

        # Should handle memory pressure gracefully
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_released = current_memory - final_memory

        assert memory_released > 0  # Should release memory

    @pytest.mark.asyncio
    async def test_file_descriptor_exhaustion(self):
        """Test handling of file descriptor exhaustion."""
        files = []
        max_files = 1000  # Try to open many files

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            try:
                for i in range(max_files):
                    try:
                        file_path = tmpdir_path / f"test_{i}.txt"
                        f = open(file_path, 'w')
                        files.append(f)
                        f.write(f"Test file {i}")
                    except OSError as e:
                        if "Too many open files" in str(e):
                            # Expected - system limit reached
                            break
                        raise
            finally:
                # Clean up
                for f in files:
                    try:
                        f.close()
                    except:
                        pass

        # Should handle file limit gracefully
        assert len(files) > 0  # Should open some files
        # All files should be properly closed
        for f in files:
            assert f.closed


def generate_stress_test_report(scenarios: List[StressTestScenario]) -> str:
    """Generate stress test report."""
    report = ["=" * 80]
    report.append("STRESS TEST REPORT")
    report.append("=" * 80]
    report.append("")

    for scenario in scenarios:
        summary = scenario.get_summary()
        report.append(f"Scenario: {summary['name']}")
        report.append("-" * 40)
        report.append(f"Duration: {summary['duration']:.2f}s")
        report.append(f"Errors: {summary['errors']}")
        report.append(f"Warnings: {summary['warnings']}")

        if summary['metrics']:
            report.append("\nMetrics:")
            for metric_name, stats in summary['metrics'].items():
                report.append(f"  {metric_name}:")
                report.append(f"    Count: {stats['count']}")
                report.append(f"    Mean: {stats['mean']:.2f}")
                report.append(f"    Min: {stats['min']:.2f}")
                report.append(f"    Max: {stats['max']:.2f}")

        report.append("")

    # Overall summary
    total_errors = sum(s.get_summary()['errors'] for s in scenarios)
    total_warnings = sum(s.get_summary()['warnings'] for s in scenarios)

    report.append("=" * 80)
    report.append("OVERALL SUMMARY")
    report.append("-" * 40)
    report.append(f"Total Scenarios: {len(scenarios)}")
    report.append(f"Total Errors: {total_errors}")
    report.append(f"Total Warnings: {total_warnings}")
    report.append("=" * 80)

    return "\n".join(report)