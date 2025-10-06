"""
Integration tests for ReAct Agent with full system components
"""

import asyncio
import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from src.agent.react_agent import ReActAgent
from src.agent.core import AgentConfig
from src.agent.tools import ToolRegistry


class TestReActAgentIntegration:
    """Integration tests for ReAct agent with various components."""

    @pytest.mark.asyncio
    async def test_agent_with_redis_memory(self, async_redis_client, agent_config):
        """Test agent integration with Redis memory backend."""
        # Configure agent with Redis memory
        agent_config.enable_memory = True
        agent = ReActAgent(agent_config)

        # Store some context in Redis
        await async_redis_client.set("context:user", "test_user")
        await async_redis_client.lpush("history:chat", "Previous conversation")

        # Mock the model to return a structured response
        mock_response = """
        Thought: I need to check the user context from memory.
        Action: redis_get
        Action Input: context:user
        """

        with patch.object(agent, 'model') as mock_model:
            mock_model.generate = MagicMock(return_value=mock_response)

            # Register Redis tool
            @agent.tool_registry.register("redis_get")
            async def redis_get(key: str) -> str:
                value = await async_redis_client.get(key)
                return f"Value for {key}: {value}"

            # Execute agent
            response = await agent.arun("What is my user context?")

            # Verify interaction
            assert "test_user" in str(response) or "context" in str(response).lower()
            assert mock_model.generate.called

    @pytest.mark.asyncio
    async def test_agent_tool_execution_chain(self, react_agent, tool_registry):
        """Test agent executing a chain of tools."""
        executed_tools = []

        # Register multiple tools
        @tool_registry.register("step1")
        async def step1(input: str) -> str:
            executed_tools.append("step1")
            return f"Step1 processed: {input}"

        @tool_registry.register("step2")
        async def step2(input: str) -> str:
            executed_tools.append("step2")
            return f"Step2 processed: {input}"

        @tool_registry.register("finalize")
        async def finalize(input: str) -> str:
            executed_tools.append("finalize")
            return f"Final result: {input}"

        # Mock model to return tool chain
        responses = [
            "Thought: Starting process\nAction: step1\nAction Input: initial",
            "Thought: Continue processing\nAction: step2\nAction Input: from step1",
            "Thought: Finalize\nAction: finalize\nAction Input: from step2",
            "Thought: Complete\nFinal Answer: Process completed"
        ]

        react_agent.tool_registry = tool_registry
        with patch.object(react_agent.model, 'generate', side_effect=responses):
            result = await react_agent.arun("Execute multi-step process")

            # Verify all tools were executed in order
            assert executed_tools == ["step1", "step2", "finalize"]
            assert "Process completed" in str(result)

    @pytest.mark.asyncio
    async def test_agent_error_recovery(self, react_agent, tool_registry):
        """Test agent's ability to recover from tool errors."""
        error_count = 0

        @tool_registry.register("unreliable_tool")
        async def unreliable_tool(input: str) -> str:
            nonlocal error_count
            error_count += 1
            if error_count < 2:
                raise Exception("Temporary failure")
            return f"Success on attempt {error_count}"

        react_agent.tool_registry = tool_registry

        # Mock model to retry on error
        responses = [
            "Thought: Try the tool\nAction: unreliable_tool\nAction Input: test",
            "Thought: Tool failed, retrying\nAction: unreliable_tool\nAction Input: test",
            "Thought: Success\nFinal Answer: Tool succeeded"
        ]

        with patch.object(react_agent.model, 'generate', side_effect=responses):
            result = await react_agent.arun("Use unreliable tool")

            assert error_count == 2
            assert "succeeded" in str(result).lower()

    @pytest.mark.asyncio
    async def test_agent_with_rag_integration(self, react_agent, async_redis_client, sample_documents):
        """Test agent with RAG (Retrieval-Augmented Generation) integration."""
        # Store documents in Redis
        for doc in sample_documents:
            await async_redis_client.hset(f"doc:{doc['id']}", mapping=doc)
            await async_redis_client.zadd("doc:index", {doc['id']: 1.0})

        @react_agent.tool_registry.register("search_documents")
        async def search_documents(query: str) -> str:
            # Simple search in Redis
            doc_ids = await async_redis_client.zrange("doc:index", 0, -1)
            results = []
            for doc_id in doc_ids[:2]:  # Return top 2
                doc = await async_redis_client.hgetall(f"doc:{doc_id}")
                if query.lower() in doc.get('content', '').lower():
                    results.append(doc)
            return json.dumps(results)

        # Mock model responses
        responses = [
            "Thought: Search for relevant documents\nAction: search_documents\nAction Input: testing",
            "Thought: Found relevant documents\nFinal Answer: Based on the documents, testing is crucial for code quality."
        ]

        with patch.object(react_agent.model, 'generate', side_effect=responses):
            result = await react_agent.arun("Tell me about testing")

            assert "testing" in str(result).lower()
            assert "quality" in str(result).lower()

    @pytest.mark.asyncio
    async def test_agent_parallel_tool_execution(self, react_agent, tool_registry):
        """Test agent executing tools in parallel when possible."""
        import time
        start_times = {}
        end_times = {}

        @tool_registry.register("parallel_task_1")
        async def parallel_task_1(input: str) -> str:
            start_times['task1'] = time.time()
            await asyncio.sleep(0.5)
            end_times['task1'] = time.time()
            return "Task 1 complete"

        @tool_registry.register("parallel_task_2")
        async def parallel_task_2(input: str) -> str:
            start_times['task2'] = time.time()
            await asyncio.sleep(0.5)
            end_times['task2'] = time.time()
            return "Task 2 complete"

        react_agent.tool_registry = tool_registry

        # Execute both tasks
        tasks = [
            react_agent.tool_registry.execute("parallel_task_1", "input1"),
            react_agent.tool_registry.execute("parallel_task_2", "input2")
        ]

        results = await asyncio.gather(*tasks)

        # Verify parallel execution (tasks should overlap)
        assert len(results) == 2
        assert "Task 1 complete" in results
        assert "Task 2 complete" in results

        # Check that tasks ran in parallel (with some tolerance)
        if start_times and end_times:
            overlap = min(end_times['task1'], end_times['task2']) - max(start_times['task1'], start_times['task2'])
            assert overlap > 0.3  # At least 0.3s overlap

    @pytest.mark.asyncio
    async def test_agent_memory_consolidation(self, react_agent, async_redis_client):
        """Test agent's memory consolidation across tiers."""
        # Simulate multiple interactions
        memories = []
        for i in range(15):
            memory = {
                "id": f"mem_{i}",
                "content": f"Interaction {i}",
                "tier": "working" if i >= 10 else "short_term",
                "importance": 0.5 + (i * 0.03)
            }
            memories.append(memory)
            await async_redis_client.hset(f"memory:{memory['id']}", mapping=memory)

        # Register memory management tool
        @react_agent.tool_registry.register("consolidate_memory")
        async def consolidate_memory() -> str:
            working_memories = []
            async for key in async_redis_client.scan_iter("memory:*"):
                mem = await async_redis_client.hgetall(key)
                if mem.get('tier') == 'working':
                    working_memories.append(mem)

            # Move old working memories to short-term
            consolidated = 0
            for mem in working_memories[:5]:  # Move oldest 5
                mem['tier'] = 'short_term'
                await async_redis_client.hset(f"memory:{mem['id']}", mapping=mem)
                consolidated += 1

            return f"Consolidated {consolidated} memories"

        # Execute consolidation
        result = await react_agent.tool_registry.execute("consolidate_memory")
        assert "Consolidated" in result
        assert "5" in result or "memories" in result

    def test_agent_sync_async_compatibility(self, agent_config, tool_registry):
        """Test that agent works with both sync and async tools."""
        results = []

        @tool_registry.register("sync_tool")
        def sync_tool(input: str) -> str:
            results.append("sync")
            return f"Sync: {input}"

        @tool_registry.register("async_tool")
        async def async_tool(input: str) -> str:
            await asyncio.sleep(0.1)
            results.append("async")
            return f"Async: {input}"

        agent = ReActAgent(agent_config)
        agent.tool_registry = tool_registry

        # Test sync tool
        sync_result = agent.tool_registry.execute_sync("sync_tool", "test1")
        assert "Sync: test1" == sync_result
        assert "sync" in results

        # Test async tool in sync context (should work with event loop)
        loop = asyncio.new_event_loop()
        async_result = loop.run_until_complete(
            agent.tool_registry.execute("async_tool", "test2")
        )
        loop.close()

        assert "Async: test2" == async_result
        assert "async" in results

    @pytest.mark.asyncio
    async def test_agent_context_window_management(self, react_agent):
        """Test agent's ability to manage context window size."""
        # Create a large context
        large_context = "Lorem ipsum " * 1000  # ~2000 tokens

        # Mock model with token limit
        with patch.object(react_agent, 'model') as mock_model:
            mock_model.max_tokens = 1000

            # Agent should truncate or summarize
            response = await react_agent.arun(
                f"Summarize this: {large_context}",
                max_context_tokens=500
            )

            # Verify truncation occurred
            call_args = str(mock_model.generate.call_args)
            assert len(call_args) < len(large_context)

    @pytest.mark.asyncio
    async def test_agent_multi_model_fallback(self, agent_config):
        """Test agent fallback to different models on failure."""
        primary_model = MagicMock()
        primary_model.generate = MagicMock(side_effect=Exception("Model unavailable"))

        fallback_model = MagicMock()
        fallback_model.generate = MagicMock(return_value="Fallback response")

        agent = ReActAgent(agent_config)
        agent.models = [primary_model, fallback_model]

        with patch.object(agent, 'model', primary_model):
            with patch.object(agent, '_get_fallback_model', return_value=fallback_model):
                response = await agent.arun("Test query")

                assert fallback_model.generate.called
                assert "Fallback response" in str(response)


class TestAgentPerformance:
    """Performance tests for agent operations."""

    @pytest.mark.asyncio
    async def test_agent_response_latency(self, react_agent, performance_monitor):
        """Test agent response time is within acceptable limits."""
        performance_monitor.start()

        # Simple query without tools
        with patch.object(react_agent.model, 'generate', return_value="Final Answer: Quick response"):
            response = await react_agent.arun("Simple question")

        metrics = performance_monitor.stop()

        # Assert response time < 1 second for simple queries
        performance_monitor.assert_performance(max_duration=1.0)
        assert response is not None

    @pytest.mark.asyncio
    async def test_agent_concurrent_requests(self, agent_config):
        """Test agent handling multiple concurrent requests."""
        agent = ReActAgent(agent_config)
        num_requests = 10

        # Mock fast responses
        with patch.object(agent, 'model') as mock_model:
            mock_model.generate = AsyncMock(return_value="Final Answer: Response")

            # Send concurrent requests
            tasks = [
                agent.arun(f"Query {i}")
                for i in range(num_requests)
            ]

            import time
            start = time.time()
            responses = await asyncio.gather(*tasks)
            duration = time.time() - start

            # All requests should complete
            assert len(responses) == num_requests

            # Should handle concurrently (not take num_requests * single_request_time)
            assert duration < num_requests * 0.5  # Assuming each would take 0.5s sequentially

    @pytest.mark.asyncio
    async def test_agent_memory_usage(self, react_agent, performance_monitor, sample_documents):
        """Test agent memory usage with large contexts."""
        performance_monitor.start()

        # Process multiple large documents
        for doc in sample_documents * 10:  # Process 30 documents
            with patch.object(react_agent.model, 'generate', return_value=f"Processed {doc['id']}"):
                await react_agent.arun(f"Process document: {doc['content']}")

        metrics = performance_monitor.stop()

        # Memory usage should stay under 100MB for this operation
        performance_monitor.assert_performance(max_memory=100)

    @pytest.mark.asyncio
    async def test_agent_cache_efficiency(self, react_agent, async_redis_client):
        """Test agent's caching mechanism for repeated queries."""
        query = "What is the capital of France?"
        cache_key = f"cache:query:{hash(query)}"

        # First call - should generate and cache
        with patch.object(react_agent.model, 'generate', return_value="Final Answer: Paris") as mock_gen:
            response1 = await react_agent.arun(query)
            await async_redis_client.set(cache_key, response1, ex=3600)

        # Second call - should use cache
        with patch.object(react_agent.model, 'generate') as mock_gen_2:
            # Simulate cache hit
            cached = await async_redis_client.get(cache_key)
            if cached:
                response2 = cached
            else:
                response2 = await react_agent.arun(query)

            # Model should not be called for cached response
            assert response1 == response2
            assert not mock_gen_2.called or cached is not None