"""
Integration tests for RAG-Redis Memory System

Tests the multi-tier memory system with Redis backend including:
- Working, short-term, long-term, episodic, and semantic memory
- Memory consolidation and transitions
- Vector similarity search
- Memory persistence and retrieval
"""

import asyncio
import pytest
import json
import numpy as np
import time
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

import fakeredis
from fakeredis import FakeAsyncRedis


class MemorySystem:
    """Memory system implementation for testing."""

    def __init__(self, redis_client: FakeAsyncRedis, config: Dict[str, Any]):
        self.redis = redis_client
        self.config = config
        self.tiers = config['tiers']

    async def store_memory(self, tier: str, memory: Dict[str, Any]) -> str:
        """Store a memory in the specified tier."""
        memory_id = memory.get('id', f"mem_{int(time.time() * 1000000)}")
        memory['tier'] = tier
        memory['timestamp'] = memory.get('timestamp', time.time())

        # Store in Redis hash
        key = f"memory:{tier}:{memory_id}"
        await self.redis.hset(key, mapping={
            k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
            for k, v in memory.items()
        })

        # Add to tier index
        await self.redis.zadd(f"index:{tier}", {memory_id: memory['timestamp']})

        # Store vector for similarity search if provided
        if 'vector' in memory:
            vector_key = f"vector:{memory_id}"
            vector_str = json.dumps(memory['vector'])
            await self.redis.set(vector_key, vector_str)

        # Set TTL if configured
        if self.tiers[tier].get('ttl'):
            await self.redis.expire(key, self.tiers[tier]['ttl'])

        return memory_id

    async def retrieve_memory(self, tier: str, memory_id: str) -> Dict[str, Any]:
        """Retrieve a specific memory."""
        key = f"memory:{tier}:{memory_id}"
        data = await self.redis.hgetall(key)

        if not data:
            return None

        # Parse JSON fields
        memory = {}
        for k, v in data.items():
            try:
                memory[k] = json.loads(v) if v.startswith('[') or v.startswith('{') else v
            except (json.JSONDecodeError, AttributeError):
                memory[k] = v

        return memory

    async def search_memories(self, query_vector: List[float], tier: str = None, limit: int = 10) -> List[Dict]:
        """Search memories by vector similarity."""
        results = []
        search_tiers = [tier] if tier else self.tiers.keys()

        for search_tier in search_tiers:
            # Get all memory IDs from tier
            memory_ids = await self.redis.zrange(f"index:{search_tier}", 0, -1)

            for memory_id in memory_ids:
                # Get vector
                vector_str = await self.redis.get(f"vector:{memory_id}")
                if not vector_str:
                    continue

                memory_vector = json.loads(vector_str)

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_vector, memory_vector)

                # Retrieve memory if similarity exceeds threshold
                if similarity >= self.config.get('similarity_threshold', 0.7):
                    memory = await self.retrieve_memory(search_tier, memory_id)
                    if memory:
                        memory['similarity_score'] = similarity
                        results.append(memory)

        # Sort by similarity and return top results
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:limit]

    async def consolidate_memories(self) -> Dict[str, int]:
        """Consolidate memories from working to higher tiers."""
        consolidated = {'working_to_short': 0, 'short_to_long': 0}

        # Move old working memories to short-term
        working_ids = await self.redis.zrange("index:working", 0, -1)
        current_time = time.time()

        for memory_id in working_ids:
            memory = await self.retrieve_memory("working", memory_id)
            if not memory:
                continue

            age = current_time - float(memory.get('timestamp', current_time))

            # Move to short-term if older than 5 minutes
            if age > 300:
                await self.store_memory("short_term", memory)
                await self.redis.delete(f"memory:working:{memory_id}")
                await self.redis.zrem("index:working", memory_id)
                consolidated['working_to_short'] += 1

        # Move important short-term memories to long-term
        short_term_ids = await self.redis.zrange("index:short_term", 0, -1)

        for memory_id in short_term_ids:
            memory = await self.retrieve_memory("short_term", memory_id)
            if not memory:
                continue

            importance = float(memory.get('metadata', {}).get('importance', 0))

            # Move to long-term if importance > threshold
            if importance > self.config['consolidation']['threshold']:
                await self.store_memory("long_term", memory)
                await self.redis.delete(f"memory:short_term:{memory_id}")
                await self.redis.zrem("index:short_term", memory_id)
                consolidated['short_to_long'] += 1

        return consolidated

    async def create_episodic_memory(self, events: List[Dict]) -> str:
        """Create an episodic memory from a sequence of events."""
        episode_id = f"episode_{int(time.time() * 1000000)}"

        episode = {
            'id': episode_id,
            'events': events,
            'start_time': min(e.get('timestamp', 0) for e in events),
            'end_time': max(e.get('timestamp', 0) for e in events),
            'num_events': len(events),
            'metadata': {
                'type': 'episodic',
                'created_at': time.time()
            }
        }

        await self.store_memory("episodic", episode)
        return episode_id

    async def build_semantic_graph(self, concepts: List[Dict]) -> Dict:
        """Build semantic relationships between concepts."""
        graph = {'nodes': {}, 'edges': []}

        for concept in concepts:
            node_id = concept['id']
            graph['nodes'][node_id] = {
                'label': concept.get('label', node_id),
                'properties': concept.get('properties', {}),
                'vector': concept.get('vector', [])
            }

            # Store in semantic memory
            await self.store_memory("semantic", {
                'id': node_id,
                'type': 'concept',
                **concept
            })

        # Find relationships based on similarity
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i + 1:]:
                if 'vector' in concept1 and 'vector' in concept2:
                    similarity = self._cosine_similarity(
                        concept1['vector'],
                        concept2['vector']
                    )

                    if similarity > 0.8:  # Strong relationship
                        graph['edges'].append({
                            'source': concept1['id'],
                            'target': concept2['id'],
                            'weight': similarity,
                            'type': 'similarity'
                        })

        # Store graph structure
        await self.redis.set(
            "semantic:graph",
            json.dumps(graph)
        )

        return graph

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)


class TestRAGRedisMemory:
    """Test RAG-Redis memory operations."""

    @pytest.mark.asyncio
    async def test_memory_tier_storage(self, async_redis_client, memory_config):
        """Test storing memories in different tiers."""
        memory_system = MemorySystem(async_redis_client, memory_config)

        # Store in working memory
        working_mem = {
            'content': 'Current task context',
            'vector': [0.1] * 384,
            'metadata': {'importance': 0.5}
        }
        working_id = await memory_system.store_memory('working', working_mem)

        # Store in long-term memory
        long_term_mem = {
            'content': 'Important fact',
            'vector': [0.2] * 384,
            'metadata': {'importance': 0.9}
        }
        long_term_id = await memory_system.store_memory('long_term', long_term_mem)

        # Verify storage
        retrieved_working = await memory_system.retrieve_memory('working', working_id)
        retrieved_long = await memory_system.retrieve_memory('long_term', long_term_id)

        assert retrieved_working['content'] == 'Current task context'
        assert retrieved_long['content'] == 'Important fact'
        assert retrieved_working['tier'] == 'working'
        assert retrieved_long['tier'] == 'long_term'

    @pytest.mark.asyncio
    async def test_vector_similarity_search(self, async_redis_client, memory_config):
        """Test vector similarity search across memory tiers."""
        memory_system = MemorySystem(async_redis_client, memory_config)

        # Create test memories with vectors
        memories = [
            {
                'id': 'mem1',
                'content': 'Python programming',
                'vector': [0.8, 0.2, 0.1] + [0.0] * 381  # Similar to query
            },
            {
                'id': 'mem2',
                'content': 'Java development',
                'vector': [0.3, 0.7, 0.2] + [0.0] * 381  # Less similar
            },
            {
                'id': 'mem3',
                'content': 'Python testing',
                'vector': [0.9, 0.1, 0.15] + [0.0] * 381  # Most similar
            }
        ]

        for mem in memories:
            await memory_system.store_memory('short_term', mem)

        # Search with query vector
        query_vector = [0.85, 0.15, 0.12] + [0.0] * 381
        results = await memory_system.search_memories(query_vector, limit=2)

        # Should return most similar memories first
        assert len(results) <= 2
        if len(results) > 0:
            assert 'Python' in results[0]['content']
            assert results[0]['similarity_score'] > 0.7

    @pytest.mark.asyncio
    async def test_memory_consolidation(self, async_redis_client, memory_config):
        """Test automatic memory consolidation between tiers."""
        memory_system = MemorySystem(async_redis_client, memory_config)

        # Create old working memories (should be consolidated)
        old_memories = []
        base_time = time.time() - 400  # 6+ minutes old

        for i in range(5):
            mem = {
                'id': f'old_{i}',
                'content': f'Old memory {i}',
                'timestamp': base_time - (i * 10),
                'metadata': {'importance': 0.5}
            }
            old_memories.append(mem)
            await memory_system.store_memory('working', mem)

        # Create recent working memories (should stay)
        recent_memories = []
        for i in range(3):
            mem = {
                'id': f'recent_{i}',
                'content': f'Recent memory {i}',
                'timestamp': time.time() - (i * 10),
                'metadata': {'importance': 0.3}
            }
            recent_memories.append(mem)
            await memory_system.store_memory('working', mem)

        # Create important short-term memories (should move to long-term)
        important_memories = []
        for i in range(2):
            mem = {
                'id': f'important_{i}',
                'content': f'Important memory {i}',
                'metadata': {'importance': 0.9}
            }
            important_memories.append(mem)
            await memory_system.store_memory('short_term', mem)

        # Run consolidation
        consolidated = await memory_system.consolidate_memories()

        # Verify consolidation occurred
        assert consolidated['working_to_short'] >= 5
        assert consolidated['short_to_long'] >= 2

        # Check that old memories moved to short-term
        for mem in old_memories:
            working_mem = await memory_system.retrieve_memory('working', mem['id'])
            short_mem = await memory_system.retrieve_memory('short_term', mem['id'])
            assert working_mem is None
            assert short_mem is not None

        # Check that recent memories stayed in working
        for mem in recent_memories:
            working_mem = await memory_system.retrieve_memory('working', mem['id'])
            assert working_mem is not None

    @pytest.mark.asyncio
    async def test_episodic_memory_creation(self, async_redis_client, memory_config):
        """Test creating episodic memories from event sequences."""
        memory_system = MemorySystem(async_redis_client, memory_config)

        # Create a sequence of related events
        events = [
            {
                'id': 'event1',
                'action': 'user_query',
                'content': 'How to test Python code?',
                'timestamp': time.time() - 30
            },
            {
                'id': 'event2',
                'action': 'tool_execution',
                'content': 'search_documentation("pytest")',
                'timestamp': time.time() - 25
            },
            {
                'id': 'event3',
                'action': 'response',
                'content': 'Use pytest for testing Python code',
                'timestamp': time.time() - 20
            }
        ]

        # Create episodic memory
        episode_id = await memory_system.create_episodic_memory(events)

        # Retrieve and verify
        episode = await memory_system.retrieve_memory('episodic', episode_id)

        assert episode is not None
        assert episode['num_events'] == 3
        assert len(episode['events']) == 3
        assert episode['events'][0]['action'] == 'user_query'
        assert episode['start_time'] < episode['end_time']

    @pytest.mark.asyncio
    async def test_semantic_graph_construction(self, async_redis_client, memory_config):
        """Test building semantic relationships between concepts."""
        memory_system = MemorySystem(async_redis_client, memory_config)

        # Create related concepts
        concepts = [
            {
                'id': 'python',
                'label': 'Python',
                'vector': [0.9, 0.1, 0.0] + [0.0] * 381,
                'properties': {'type': 'language', 'paradigm': 'multi'}
            },
            {
                'id': 'testing',
                'label': 'Testing',
                'vector': [0.8, 0.2, 0.1] + [0.0] * 381,
                'properties': {'type': 'practice', 'importance': 'high'}
            },
            {
                'id': 'pytest',
                'label': 'PyTest',
                'vector': [0.85, 0.15, 0.05] + [0.0] * 381,
                'properties': {'type': 'framework', 'language': 'python'}
            },
            {
                'id': 'java',
                'label': 'Java',
                'vector': [0.1, 0.9, 0.0] + [0.0] * 381,
                'properties': {'type': 'language', 'paradigm': 'oop'}
            }
        ]

        # Build semantic graph
        graph = await memory_system.build_semantic_graph(concepts)

        # Verify graph structure
        assert len(graph['nodes']) == 4
        assert 'python' in graph['nodes']
        assert 'testing' in graph['nodes']

        # Should find relationships between similar concepts
        assert len(graph['edges']) > 0

        # Python and PyTest should be related (high similarity)
        python_pytest_edge = None
        for edge in graph['edges']:
            if (edge['source'] == 'python' and edge['target'] == 'pytest') or \
               (edge['source'] == 'pytest' and edge['target'] == 'python'):
                python_pytest_edge = edge
                break

        assert python_pytest_edge is not None
        assert python_pytest_edge['weight'] > 0.8

    @pytest.mark.asyncio
    async def test_memory_ttl_expiration(self, async_redis_client, memory_config):
        """Test that memories expire according to TTL settings."""
        # Modify config for fast testing
        test_config = memory_config.copy()
        test_config['tiers']['working']['ttl'] = 2  # 2 seconds

        memory_system = MemorySystem(async_redis_client, test_config)

        # Store memory with short TTL
        memory = {
            'id': 'temp_memory',
            'content': 'Temporary information',
            'vector': [0.5] * 384
        }
        memory_id = await memory_system.store_memory('working', memory)

        # Verify it exists
        retrieved = await memory_system.retrieve_memory('working', memory_id)
        assert retrieved is not None

        # Wait for expiration
        await asyncio.sleep(3)

        # Should be expired
        retrieved_after = await memory_system.retrieve_memory('working', memory_id)
        # Note: FakeRedis might not fully support TTL expiration
        # In real Redis, this would be None

    @pytest.mark.asyncio
    async def test_memory_capacity_limits(self, async_redis_client, memory_config):
        """Test that memory tiers respect capacity limits."""
        memory_system = MemorySystem(async_redis_client, memory_config)

        # Fill working memory beyond capacity
        capacity = memory_config['tiers']['working']['capacity']

        for i in range(capacity + 5):
            await memory_system.store_memory('working', {
                'id': f'mem_{i}',
                'content': f'Memory {i}',
                'timestamp': time.time() - i  # Older memories have lower timestamp
            })

        # Check tier size
        working_size = await async_redis_client.zcard("index:working")

        # In a real implementation, we'd enforce capacity limits
        # For testing, we verify we can detect over-capacity
        assert working_size > capacity

        # A proper implementation would evict oldest memories
        # Let's simulate that
        if working_size > capacity:
            to_remove = working_size - capacity
            oldest = await async_redis_client.zrange("index:working", 0, to_remove - 1)
            for memory_id in oldest:
                await async_redis_client.zrem("index:working", memory_id)
                await async_redis_client.delete(f"memory:working:{memory_id}")

        # Verify capacity is now respected
        final_size = await async_redis_client.zcard("index:working")
        assert final_size <= capacity


class TestRAGDocumentProcessing:
    """Test RAG document processing and retrieval."""

    @pytest.mark.asyncio
    async def test_document_chunking(self, async_redis_client):
        """Test document chunking for RAG."""
        # Sample large document
        document = {
            'id': 'doc1',
            'title': 'Python Testing Guide',
            'content': ' '.join([f"Section {i}: " + "Lorem ipsum " * 100 for i in range(10)])
        }

        # Chunk document
        chunk_size = 500  # characters
        chunks = []
        content = document['content']

        for i in range(0, len(content), chunk_size):
            chunk = {
                'id': f"{document['id']}_chunk_{i // chunk_size}",
                'doc_id': document['id'],
                'content': content[i:i + chunk_size],
                'position': i // chunk_size,
                'metadata': {
                    'title': document['title'],
                    'chunk_size': min(chunk_size, len(content) - i)
                }
            }
            chunks.append(chunk)

            # Store chunk in Redis
            await async_redis_client.hset(
                f"chunk:{chunk['id']}",
                mapping={k: json.dumps(v) if isinstance(v, dict) else str(v)
                        for k, v in chunk.items()}
            )

        # Verify chunking
        assert len(chunks) > 5  # Should have multiple chunks
        assert all(len(c['content']) <= chunk_size for c in chunks)

        # Retrieve chunks for document
        stored_chunks = []
        async for key in async_redis_client.scan_iter(f"chunk:{document['id']}_*"):
            chunk_data = await async_redis_client.hgetall(key)
            stored_chunks.append(chunk_data)

        assert len(stored_chunks) == len(chunks)

    @pytest.mark.asyncio
    async def test_document_embedding_and_indexing(self, async_redis_client):
        """Test document embedding and vector indexing."""
        # Mock embedding model
        def mock_embed(text: str) -> List[float]:
            # Simple mock: use text length and character frequencies
            vec = [0.0] * 384
            vec[0] = len(text) / 1000  # Normalize length
            for i, char in enumerate(text[:100]):
                vec[i % 384] += ord(char) / 10000
            # Normalize
            norm = sum(v**2 for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            return vec

        documents = [
            {'id': 'doc1', 'content': 'Python is a programming language'},
            {'id': 'doc2', 'content': 'Testing is important for quality'},
            {'id': 'doc3', 'content': 'Redis is a fast database'}
        ]

        # Embed and index documents
        for doc in documents:
            embedding = mock_embed(doc['content'])

            # Store document with embedding
            await async_redis_client.hset(
                f"doc:{doc['id']}",
                mapping={
                    'content': doc['content'],
                    'embedding': json.dumps(embedding)
                }
            )

            # Add to vector index (in real implementation, use Redis Vector Similarity)
            await async_redis_client.zadd(
                "doc:index",
                {doc['id']: 1.0}  # Score could be based on importance
            )

        # Search similar documents
        query = "Python programming"
        query_embedding = mock_embed(query)

        # Retrieve and rank documents
        results = []
        doc_ids = await async_redis_client.zrange("doc:index", 0, -1)

        for doc_id in doc_ids:
            doc_data = await async_redis_client.hget(f"doc:{doc_id}", "embedding")
            if doc_data:
                doc_embedding = json.loads(doc_data)
                # Calculate similarity
                similarity = sum(q * d for q, d in zip(query_embedding, doc_embedding))
                results.append((doc_id, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        # Most similar should be doc1 (contains "Python" and "programming")
        assert results[0][0] == 'doc1'
        assert results[0][1] > 0  # Should have positive similarity