#!/usr/bin/env python3
"""
Comprehensive 5-tier memory system test script for Redis-based RAG system.

Tests the complete memory flow:
Working Memory → Short-term Memory → Long-term Memory → Episodic Memory → Semantic Memory

This script validates:
1. Redis connectivity and authentication
2. Database tier allocation and isolation
3. Memory consolidation between tiers
4. Connection pooling and failover
5. Performance under load
6. Data integrity across tiers
"""

import redis
import json
import time
import logging
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import random
import uuid


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('memory_system_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """Represents a memory item across all tiers"""
    id: str
    content: str
    vector: List[float]
    timestamp: str
    importance_score: float
    access_count: int
    tier: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        return cls(**data)


class MemoryTier:
    """Represents a single memory tier with its characteristics"""

    def __init__(self, name: str, db_index: int, max_items: int, retention_hours: int):
        self.name = name
        self.db_index = db_index
        self.max_items = max_items
        self.retention_hours = retention_hours
        self.redis_client = None

    def connect(self, host='127.0.0.1', port=6379, password='testpass123'):
        """Connect to Redis database for this tier"""
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                password=password,
                db=self.db_index,
                decode_responses=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to {self.name} tier (DB {self.db_index})")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.name} tier: {e}")
            return False

    def add_item(self, item: MemoryItem) -> bool:
        """Add an item to this memory tier"""
        try:
            # Store as hash with all fields
            key = f"memory:{item.id}"
            self.redis_client.hset(key, mapping={
                'id': item.id,
                'content': item.content,
                'vector': json.dumps(item.vector),
                'timestamp': item.timestamp,
                'importance_score': item.importance_score,
                'access_count': item.access_count,
                'tier': item.tier,
                'metadata': json.dumps(item.metadata)
            })

            # Set expiration if configured
            if self.retention_hours > 0:
                self.redis_client.expire(key, self.retention_hours * 3600)

            # Add to sorted set for importance-based retrieval
            self.redis_client.zadd(f"tier:{self.name}:by_importance", {item.id: item.importance_score})

            # Add to sorted set for timestamp-based retrieval
            timestamp_score = int(datetime.fromisoformat(item.timestamp).timestamp())
            self.redis_client.zadd(f"tier:{self.name}:by_time", {item.id: timestamp_score})

            logger.debug(f"Added item {item.id} to {self.name} tier")
            return True

        except Exception as e:
            logger.error(f"Failed to add item to {self.name} tier: {e}")
            return False

    def get_item(self, item_id: str) -> Optional[MemoryItem]:
        """Retrieve an item from this tier"""
        try:
            key = f"memory:{item_id}"
            data = self.redis_client.hgetall(key)

            if not data:
                return None

            # Increment access count
            self.redis_client.hincrby(key, 'access_count', 1)

            # Parse complex fields
            data['vector'] = json.loads(data['vector'])
            data['importance_score'] = float(data['importance_score'])
            data['access_count'] = int(data['access_count']) + 1
            data['metadata'] = json.loads(data['metadata'])

            return MemoryItem.from_dict(data)

        except Exception as e:
            logger.error(f"Failed to get item from {self.name} tier: {e}")
            return None

    def get_top_items(self, count: int) -> List[MemoryItem]:
        """Get top items by importance score"""
        try:
            item_ids = self.redis_client.zrevrange(f"tier:{self.name}:by_importance", 0, count - 1)
            items = []

            for item_id in item_ids:
                item = self.get_item(item_id)
                if item:
                    items.append(item)

            return items

        except Exception as e:
            logger.error(f"Failed to get top items from {self.name} tier: {e}")
            return []

    def count_items(self) -> int:
        """Count total items in this tier"""
        try:
            return self.redis_client.zcard(f"tier:{self.name}:by_importance")
        except Exception as e:
            logger.error(f"Failed to count items in {self.name} tier: {e}")
            return 0

    def clear_tier(self):
        """Clear all items from this tier"""
        try:
            self.redis_client.flushdb()
            logger.info(f"Cleared {self.name} tier")
        except Exception as e:
            logger.error(f"Failed to clear {self.name} tier: {e}")


class MemorySystem:
    """5-tier memory system management"""

    def __init__(self):
        self.tiers = {
            'working': MemoryTier('working', 0, 10, 1),  # 1 hour retention, DB 0
            'short_term': MemoryTier('short_term', 1, 100, 24),  # 24 hours retention, DB 1
            'long_term': MemoryTier('long_term', 2, 10000, 0),  # No expiration, DB 2
            'episodic': MemoryTier('episodic', 3, 1000, 0),  # No expiration, DB 3
            'semantic': MemoryTier('semantic', 4, 5000, 0)  # No expiration, DB 4
        }
        self.consolidation_thread = None
        self.consolidation_active = False

    def connect_all_tiers(self) -> bool:
        """Connect to all memory tiers"""
        success = True
        for tier in self.tiers.values():
            if not tier.connect():
                success = False
        return success

    def add_to_working_memory(self, content: str, vector: List[float] = None,
                             importance: float = 0.5, metadata: Dict[str, Any] = None) -> str:
        """Add new content to working memory"""
        if vector is None:
            # Generate dummy vector for testing
            vector = np.random.random(384).tolist()

        if metadata is None:
            metadata = {}

        item_id = str(uuid.uuid4())
        item = MemoryItem(
            id=item_id,
            content=content,
            vector=vector,
            timestamp=datetime.now().isoformat(),
            importance_score=importance,
            access_count=0,
            tier='working',
            metadata=metadata
        )

        if self.tiers['working'].add_item(item):
            logger.info(f"Added item {item_id} to working memory")
            return item_id
        else:
            raise Exception("Failed to add item to working memory")

    def start_consolidation(self):
        """Start background consolidation process"""
        self.consolidation_active = True
        self.consolidation_thread = threading.Thread(target=self._consolidation_worker)
        self.consolidation_thread.daemon = True
        self.consolidation_thread.start()
        logger.info("Started memory consolidation process")

    def stop_consolidation(self):
        """Stop background consolidation process"""
        self.consolidation_active = False
        if self.consolidation_thread:
            self.consolidation_thread.join(timeout=5)
        logger.info("Stopped memory consolidation process")

    def _consolidation_worker(self):
        """Background worker for memory consolidation"""
        while self.consolidation_active:
            try:
                # Working → Short-term
                self._consolidate_working_to_short_term()

                # Short-term → Long-term
                self._consolidate_short_term_to_long_term()

                # Long-term → Episodic/Semantic
                self._consolidate_long_term_to_higher_order()

                time.sleep(10)  # Consolidate every 10 seconds for testing

            except Exception as e:
                logger.error(f"Error in consolidation worker: {e}")
                time.sleep(30)

    def _consolidate_working_to_short_term(self):
        """Move high-importance items from working to short-term memory"""
        working_items = self.tiers['working'].get_top_items(50)

        for item in working_items:
            if item.importance_score > 0.7 or item.access_count > 3:
                # Move to short-term
                item.tier = 'short_term'
                if self.tiers['short_term'].add_item(item):
                    # Remove from working memory
                    self.tiers['working'].redis_client.delete(f"memory:{item.id}")
                    logger.debug(f"Consolidated item {item.id} to short-term memory")

    def _consolidate_short_term_to_long_term(self):
        """Move persistent items from short-term to long-term memory"""
        short_term_items = self.tiers['short_term'].get_top_items(200)

        for item in short_term_items:
            # Items accessed multiple times or with high importance
            if item.access_count > 5 or item.importance_score > 0.8:
                item.tier = 'long_term'
                if self.tiers['long_term'].add_item(item):
                    self.tiers['short_term'].redis_client.delete(f"memory:{item.id}")
                    logger.debug(f"Consolidated item {item.id} to long-term memory")

    def _consolidate_long_term_to_higher_order(self):
        """Organize long-term memories into episodic and semantic"""
        long_term_items = self.tiers['long_term'].get_top_items(500)

        for item in long_term_items:
            if 'event' in item.metadata or 'sequence' in item.metadata:
                # Move to episodic memory
                item.tier = 'episodic'
                if self.tiers['episodic'].add_item(item):
                    logger.debug(f"Consolidated item {item.id} to episodic memory")
            elif item.importance_score > 0.9:
                # Move to semantic memory for concept storage
                item.tier = 'semantic'
                if self.tiers['semantic'].add_item(item):
                    logger.debug(f"Consolidated item {item.id} to semantic memory")

    def get_memory_stats(self) -> Dict[str, int]:
        """Get count of items in each tier"""
        stats = {}
        for name, tier in self.tiers.items():
            stats[name] = tier.count_items()
        return stats

    def clear_all_tiers(self):
        """Clear all memory tiers"""
        for tier in self.tiers.values():
            tier.clear_tier()
        logger.info("Cleared all memory tiers")


def test_redis_connectivity():
    """Test basic Redis connectivity and authentication"""
    logger.info("Testing Redis connectivity...")

    try:
        client = redis.Redis(host='127.0.0.1', port=6379, password='testpass123')
        response = client.ping()

        if response:
            logger.info("✓ Redis connectivity test PASSED")
            return True
        else:
            logger.error("✗ Redis connectivity test FAILED")
            return False

    except Exception as e:
        logger.error(f"✗ Redis connectivity test FAILED: {e}")
        return False


def test_tier_isolation():
    """Test that memory tiers are properly isolated"""
    logger.info("Testing tier isolation...")

    try:
        # Create connections to different databases
        clients = {}
        for i in range(5):
            clients[i] = redis.Redis(
                host='127.0.0.1',
                port=6379,
                password='testpass123',
                db=i,
                decode_responses=True
            )

        # Add unique data to each database
        test_data = {}
        for db_index, client in clients.items():
            key = f"test_tier_{db_index}"
            value = f"data_for_tier_{db_index}"
            client.set(key, value)
            test_data[db_index] = (key, value)

        # Verify isolation - each DB should only see its own data
        isolation_success = True
        for db_index, client in clients.items():
            # Should find its own data
            own_key, own_value = test_data[db_index]
            if client.get(own_key) != own_value:
                logger.error(f"DB {db_index} cannot find its own data")
                isolation_success = False

            # Should not find other DBs' data
            for other_db, (other_key, _) in test_data.items():
                if other_db != db_index:
                    if client.exists(other_key):
                        logger.error(f"DB {db_index} can see data from DB {other_db}")
                        isolation_success = False

        # Clean up
        for client in clients.values():
            client.flushdb()

        if isolation_success:
            logger.info("✓ Tier isolation test PASSED")
            return True
        else:
            logger.error("✗ Tier isolation test FAILED")
            return False

    except Exception as e:
        logger.error(f"✗ Tier isolation test FAILED: {e}")
        return False


def test_memory_flow():
    """Test the complete memory flow through all tiers"""
    logger.info("Testing complete memory flow...")

    try:
        memory_system = MemorySystem()

        # Connect all tiers
        if not memory_system.connect_all_tiers():
            logger.error("✗ Failed to connect to all memory tiers")
            return False

        # Clear all tiers first
        memory_system.clear_all_tiers()

        # Start consolidation
        memory_system.start_consolidation()

        # Add test items with varying importance
        test_items = [
            ("Low importance item", 0.3, {}),
            ("Medium importance item", 0.6, {}),
            ("High importance item", 0.8, {}),
            ("Critical importance item", 0.95, {}),
            ("Event item", 0.7, {"event": "user_login", "sequence": 1}),
            ("Concept item", 0.9, {"concept": "machine_learning", "category": "ai"}),
        ]

        item_ids = []
        for content, importance, metadata in test_items:
            item_id = memory_system.add_to_working_memory(content, None, importance, metadata)
            item_ids.append(item_id)

        # Wait for consolidation to occur
        logger.info("Waiting for memory consolidation...")
        time.sleep(30)  # Wait 30 seconds for consolidation

        # Check memory distribution
        stats = memory_system.get_memory_stats()
        logger.info(f"Memory distribution: {stats}")

        # Stop consolidation
        memory_system.stop_consolidation()

        # Verify items have moved to appropriate tiers
        success = True
        if stats['working'] >= len(test_items):
            logger.warning("Items may not have consolidated from working memory")
            success = False

        if stats['short_term'] == 0 and stats['long_term'] == 0:
            logger.error("No items consolidated to higher tiers")
            success = False

        if success:
            logger.info("✓ Memory flow test PASSED")
            return True
        else:
            logger.error("✗ Memory flow test FAILED")
            return False

    except Exception as e:
        logger.error(f"✗ Memory flow test FAILED: {e}")
        return False


def test_connection_pooling():
    """Test connection pooling and concurrent access"""
    logger.info("Testing connection pooling and concurrent access...")

    def worker_task(worker_id: int, iterations: int) -> Dict[str, Any]:
        """Worker function for concurrent testing"""
        try:
            memory_system = MemorySystem()
            if not memory_system.connect_all_tiers():
                return {"worker_id": worker_id, "success": False, "error": "Connection failed"}

            success_count = 0
            for i in range(iterations):
                try:
                    content = f"Worker {worker_id} - Item {i}"
                    importance = random.uniform(0.1, 1.0)
                    memory_system.add_to_working_memory(content, None, importance)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Worker {worker_id} failed on iteration {i}: {e}")

            return {
                "worker_id": worker_id,
                "success": True,
                "success_count": success_count,
                "total_iterations": iterations
            }

        except Exception as e:
            return {"worker_id": worker_id, "success": False, "error": str(e)}

    try:
        # Create multiple concurrent workers
        num_workers = 10
        iterations_per_worker = 50

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(num_workers):
                future = executor.submit(worker_task, worker_id, iterations_per_worker)
                futures.append(future)

            # Collect results
            results = []
            for future in as_completed(futures):
                result = future.result(timeout=30)
                results.append(result)

        # Analyze results
        successful_workers = [r for r in results if r["success"]]
        total_operations = sum(r["success_count"] for r in successful_workers)

        logger.info(f"Concurrent test results: {len(successful_workers)}/{num_workers} workers successful")
        logger.info(f"Total successful operations: {total_operations}")

        if len(successful_workers) >= num_workers * 0.8:  # 80% success rate
            logger.info("✓ Connection pooling test PASSED")
            return True
        else:
            logger.error("✗ Connection pooling test FAILED")
            return False

    except Exception as e:
        logger.error(f"✗ Connection pooling test FAILED: {e}")
        return False


def test_failover_resilience():
    """Test system resilience to Redis connection issues"""
    logger.info("Testing failover resilience...")

    try:
        memory_system = MemorySystem()

        # First, test normal operation
        if not memory_system.connect_all_tiers():
            logger.error("✗ Failed to connect initially")
            return False

        # Add some items
        normal_item = memory_system.add_to_working_memory("Normal operation test", None, 0.5)

        # Simulate connection issues by creating bad connections
        # Note: In a real test, you might temporarily stop Redis or change credentials
        logger.info("Testing graceful handling of connection errors...")

        # Try to connect with wrong credentials
        bad_tier = MemoryTier('bad_test', 0, 10, 1)
        success = bad_tier.connect(password='wrongpassword')

        if success:
            logger.error("✗ Bad connection succeeded when it should have failed")
            return False

        # Verify the good connections still work
        test_item = memory_system.add_to_working_memory("Post-error test", None, 0.7)

        if test_item:
            logger.info("✓ System remained operational after connection error")
            logger.info("✓ Failover resilience test PASSED")
            return True
        else:
            logger.error("✗ System failed after connection error")
            return False

    except Exception as e:
        logger.error(f"✗ Failover resilience test FAILED: {e}")
        return False


def test_performance_benchmarks():
    """Run performance benchmarks on the memory system"""
    logger.info("Running performance benchmarks...")

    try:
        memory_system = MemorySystem()
        if not memory_system.connect_all_tiers():
            logger.error("✗ Failed to connect for performance test")
            return False

        memory_system.clear_all_tiers()

        # Benchmark 1: Write performance
        write_start = time.time()
        write_count = 1000

        for i in range(write_count):
            content = f"Performance test item {i}"
            importance = random.uniform(0.1, 1.0)
            memory_system.add_to_working_memory(content, None, importance)

        write_end = time.time()
        write_duration = write_end - write_start
        writes_per_second = write_count / write_duration

        logger.info(f"Write performance: {writes_per_second:.2f} items/second")

        # Benchmark 2: Read performance
        read_start = time.time()
        read_count = 500

        for i in range(read_count):
            working_items = memory_system.tiers['working'].get_top_items(10)

        read_end = time.time()
        read_duration = read_end - read_start
        reads_per_second = read_count / read_duration

        logger.info(f"Read performance: {reads_per_second:.2f} operations/second")

        # Performance thresholds (adjust based on system capabilities)
        write_threshold = 100  # items/second
        read_threshold = 200   # operations/second

        if writes_per_second >= write_threshold and reads_per_second >= read_threshold:
            logger.info("✓ Performance benchmarks PASSED")
            return True
        else:
            logger.warning(f"✗ Performance below thresholds: W={writes_per_second:.2f}/{write_threshold}, R={reads_per_second:.2f}/{read_threshold}")
            return False

    except Exception as e:
        logger.error(f"✗ Performance benchmarks FAILED: {e}")
        return False


def run_comprehensive_test_suite():
    """Run the complete memory system test suite"""
    logger.info("=" * 60)
    logger.info("STARTING COMPREHENSIVE MEMORY SYSTEM TEST SUITE")
    logger.info("=" * 60)

    test_results = {}

    # Test 1: Redis Connectivity
    test_results['connectivity'] = test_redis_connectivity()

    # Test 2: Tier Isolation
    test_results['tier_isolation'] = test_tier_isolation()

    # Test 3: Memory Flow
    test_results['memory_flow'] = test_memory_flow()

    # Test 4: Connection Pooling
    test_results['connection_pooling'] = test_connection_pooling()

    # Test 5: Failover Resilience
    test_results['failover_resilience'] = test_failover_resilience()

    # Test 6: Performance Benchmarks
    test_results['performance'] = test_performance_benchmarks()

    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUITE SUMMARY")
    logger.info("=" * 60)

    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)

    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name.upper()}: {status}")

    success_rate = (passed_tests / total_tests) * 100
    logger.info(f"\nOVERALL RESULT: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")

    if success_rate >= 80:
        logger.info("✓ MEMORY SYSTEM TEST SUITE: PASSED")
        return True
    else:
        logger.error("✗ MEMORY SYSTEM TEST SUITE: FAILED")
        return False


if __name__ == "__main__":
    # Ensure Redis is running
    logger.info("Memory System Test Suite")
    logger.info("Make sure Redis server is running on 127.0.0.1:6379 with password 'testpass123'")

    try:
        success = run_comprehensive_test_suite()
        exit_code = 0 if success else 1
        exit(exit_code)

    except KeyboardInterrupt:
        logger.info("Test suite interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Test suite failed with unexpected error: {e}")
        exit(1)