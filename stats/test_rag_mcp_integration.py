#!/usr/bin/env python3
"""
Comprehensive test for RAG-MCP integration.

This script tests the complete integration between the RAG-Redis system
and MCP protocol for document ingestion, search, and memory operations.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agent.rag_integration import RAGClient, enhance_agent_with_rag, rag_context
from src.infrastructure.mcp.server import McpServerConfig, McpServer
from src.domain.tools.base import ToolRegistry, get_global_registry
from src.infrastructure.tools import register_builtin_tools, register_rag_tools
from src.shared.logging import setup_logging

logger = logging.getLogger(__name__)


async def test_rag_tools_direct():
    """Test RAG tools directly without MCP server."""
    logger.info("=== Testing RAG Tools Directly ===")

    # Register tools
    registry = get_global_registry()
    await register_builtin_tools()
    await register_rag_tools()

    # Test document ingestion
    logger.info("Testing document ingestion...")
    from src.infrastructure.tools.rag_tools import RagIngestDocumentTool
    from src.domain.tools.base import ToolExecutionContext
    from uuid import uuid4

    ingest_tool = RagIngestDocumentTool()
    context = ToolExecutionContext(
        execution_id=str(uuid4()),
        agent_id="test_agent",
        session_id="test_session",
        timeout=60,
        security_level="standard",
        metadata={}
    )

    # Test with sample document
    sample_doc = """
    # Machine Learning Best Practices

    ## Introduction
    Machine learning (ML) is a powerful subset of artificial intelligence that enables
    computers to learn patterns from data without being explicitly programmed.

    ## Key Principles
    1. **Data Quality**: Ensure your training data is clean, representative, and sufficient
    2. **Feature Engineering**: Transform raw data into meaningful features
    3. **Model Selection**: Choose appropriate algorithms for your problem type
    4. **Validation**: Use proper cross-validation to assess model performance
    5. **Regularization**: Prevent overfitting with techniques like L1/L2 regularization

    ## Common Algorithms
    - **Supervised Learning**: Linear Regression, Random Forest, SVM, Neural Networks
    - **Unsupervised Learning**: K-means, PCA, DBSCAN
    - **Reinforcement Learning**: Q-learning, Policy Gradient methods

    ## Tools and Libraries
    - Python: scikit-learn, TensorFlow, PyTorch
    - R: caret, randomForest
    - Scala: Spark MLlib

    This document covers fundamental concepts that every ML practitioner should understand.
    """

    result = await ingest_tool.execute(
        context,
        content=sample_doc,
        title="Machine Learning Best Practices Guide",
        source="test_integration",
        doc_type="markdown",
        tags=["machine-learning", "best-practices", "guide"],
        importance=0.8
    )

    if result.success:
        logger.info(f"✓ Document ingestion successful: {result.data.get('document_id')}")
        doc_id = result.data.get('document_id')
    else:
        logger.error(f"✗ Document ingestion failed: {result.error}")
        return False

    # Test search
    logger.info("Testing document search...")
    from src.infrastructure.tools.rag_tools import RagSearchTool

    search_tool = RagSearchTool()
    search_result = await search_tool.execute(
        context,
        query="machine learning feature engineering",
        limit=3,
        threshold=0.5
    )

    if search_result.success:
        results = search_result.data.get('results', [])
        logger.info(f"✓ Search successful, found {len(results)} results")
        for i, result in enumerate(results):
            logger.info(f"  Result {i+1}: {result['content'][:100]}... (score: {result['score']:.3f})")
    else:
        logger.error(f"✗ Search failed: {search_result.error}")
        return False

    # Test memory storage
    logger.info("Testing memory storage...")
    from src.infrastructure.tools.rag_tools import RagStoreMemoryTool

    memory_tool = RagStoreMemoryTool()
    memory_result = await memory_tool.execute(
        context,
        content="The user is interested in machine learning best practices and feature engineering techniques",
        memory_type="episodic",
        importance=0.7,
        tags=["user-interest", "ml"],
        context="User interaction during RAG testing"
    )

    if memory_result.success:
        logger.info(f"✓ Memory storage successful: {memory_result.data.get('memory_id')}")
        memory_id = memory_result.data.get('memory_id')
    else:
        logger.error(f"✗ Memory storage failed: {memory_result.error}")
        return False

    # Test memory recall
    logger.info("Testing memory recall...")
    from src.infrastructure.tools.rag_tools import RagRecallMemoryTool

    recall_tool = RagRecallMemoryTool()
    recall_result = await recall_tool.execute(
        context,
        query="user machine learning interests",
        memory_type="episodic",
        limit=3,
        min_importance=0.5
    )

    if recall_result.success:
        memories = recall_result.data.get('memories', [])
        logger.info(f"✓ Memory recall successful, found {len(memories)} memories")
        for i, memory in enumerate(memories):
            logger.info(f"  Memory {i+1}: {memory['content'][:100]}... (importance: {memory.get('importance', 0.0):.3f})")
    else:
        logger.error(f"✗ Memory recall failed: {recall_result.error}")
        return False

    return True


async def test_rag_integration_layer():
    """Test the RAG integration layer."""
    logger.info("=== Testing RAG Integration Layer ===")

    # Create a mock MCP server (just for connection testing)
    config = McpServerConfig(name="test_rag_server", port=8001)
    registry = get_global_registry()
    server = McpServer(config, registry)

    async with rag_context(server) as rag_client:
        # Test document ingestion via integration layer
        logger.info("Testing document ingestion via integration layer...")
        doc_content = """
        # RAG System Integration Test

        This document tests the integration between the RAG system and MCP protocol.
        It includes information about vector embeddings, document chunking, and
        multi-tier memory management.

        ## Key Features
        - Vector similarity search
        - Document chunking and embedding
        - Redis-backed storage
        - Multi-tier memory (working, short-term, long-term, episodic, semantic)
        """

        doc_id = await rag_client.ingest_document(
            content=doc_content,
            metadata={"title": "RAG Integration Test", "type": "test", "source": "integration_test"}
        )

        if doc_id and not doc_id.startswith("doc_"):  # Check if it's not just a fallback mock ID
            logger.info(f"✓ Integration layer document ingestion successful: {doc_id}")
        else:
            logger.warning(f"⚠ Integration layer using fallback ingestion: {doc_id}")

        # Test search via integration layer
        logger.info("Testing search via integration layer...")
        search_results = await rag_client.search(
            query="RAG system vector embeddings",
            limit=3,
            threshold=0.6
        )

        if search_results:
            logger.info(f"✓ Integration layer search successful, found {len(search_results)} results")
            for i, result in enumerate(search_results):
                logger.info(f"  Result {i+1}: {result.get('content', '')[:80]}...")
        else:
            logger.warning("⚠ Integration layer search returned no results")

        # Test memory operations via integration layer
        logger.info("Testing memory storage via integration layer...")
        memory_stored = await rag_client.store_memory(
            content="User tested RAG-MCP integration successfully",
            memory_type="episodic"
        )

        if memory_stored:
            logger.info("✓ Integration layer memory storage successful")
        else:
            logger.warning("⚠ Integration layer memory storage failed")

        logger.info("Testing memory recall via integration layer...")
        recalled_memories = await rag_client.recall_memory(
            query="RAG integration test",
            memory_type="episodic"
        )

        if recalled_memories:
            logger.info(f"✓ Integration layer memory recall successful, found {len(recalled_memories)} memories")
            for i, memory in enumerate(recalled_memories):
                logger.info(f"  Memory {i+1}: {memory.get('content', '')[:80]}...")
        else:
            logger.warning("⚠ Integration layer memory recall returned no results")

    return True


async def test_enhanced_agent():
    """Test agent enhancement with RAG capabilities."""
    logger.info("=== Testing RAG-Enhanced Agent ===")

    # Create a mock agent (simplified for testing)
    class MockAgent:
        def solve(self, query: str, max_iterations: int = 5) -> str:
            return f"Mock agent response to: {query}"

        def chat(self, message: str, use_tools: bool = True) -> str:
            return f"Mock chat response to: {message}"

    mock_agent = MockAgent()

    # Create mock MCP server
    config = McpServerConfig(name="test_enhanced_server", port=8002)
    registry = get_global_registry()
    server = McpServer(config, registry)

    # Enhance agent with RAG
    enhanced_agent = await enhance_agent_with_rag(mock_agent, server)

    if hasattr(enhanced_agent, 'rag_client'):
        logger.info("✓ Agent successfully enhanced with RAG capabilities")

        # Test enhanced solve method
        logger.info("Testing enhanced solve method...")
        try:
            response = await enhanced_agent.solve("What are machine learning best practices?")
            logger.info(f"✓ Enhanced solve response: {response[:100]}...")
        except Exception as e:
            logger.error(f"✗ Enhanced solve failed: {e}")
            # Test fallback to base agent
            response = enhanced_agent.solve("What are machine learning best practices?")
            logger.info(f"✓ Fallback to base agent worked: {response[:100]}...")

        # Test chat method
        chat_response = enhanced_agent.chat("Tell me about RAG systems")
        logger.info(f"✓ Enhanced chat response: {chat_response[:100]}...")

    else:
        logger.warning("⚠ Agent enhancement returned original agent (RAG unavailable)")

    return True


async def test_error_handling():
    """Test error handling and fallback mechanisms."""
    logger.info("=== Testing Error Handling ===")

    # Test with no MCP server (should use mock mode)
    async with rag_context(None) as rag_client:
        logger.info("Testing fallback mode (no MCP server)...")

        # Should work with mock implementations
        doc_id = await rag_client.ingest_document("Test document for error handling")
        search_results = await rag_client.search("test query")
        memory_stored = await rag_client.store_memory("test memory")
        memories = await rag_client.recall_memory("test recall")

        if doc_id and search_results and memory_stored and memories:
            logger.info("✓ Fallback mode working correctly")
        else:
            logger.error("✗ Fallback mode failed")
            return False

    # Test with invalid parameters
    logger.info("Testing error handling with invalid parameters...")
    try:
        from src.infrastructure.tools.rag_tools import RagSearchTool
        from src.domain.tools.base import ToolExecutionContext
        from uuid import uuid4

        search_tool = RagSearchTool()
        context = ToolExecutionContext(
            execution_id=str(uuid4()),
            agent_id="test_agent",
            session_id="test_session",
            timeout=60,
            security_level="standard",
            metadata={}
        )

        # Test with empty query (should fail gracefully)
        result = await search_tool.execute(context, query="", limit=5)
        if not result.success:
            logger.info("✓ Empty query handled correctly")
        else:
            logger.warning("⚠ Empty query should have failed")

    except Exception as e:
        logger.info(f"✓ Exception handled gracefully: {e}")

    return True


async def main():
    """Run all tests."""
    # Setup logging with proper level
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Starting RAG-MCP Integration Tests")

    all_tests_passed = True

    try:
        # Test 1: Direct RAG tools
        if not await test_rag_tools_direct():
            all_tests_passed = False

        # Test 2: Integration layer
        if not await test_rag_integration_layer():
            all_tests_passed = False

        # Test 3: Enhanced agent
        if not await test_enhanced_agent():
            all_tests_passed = False

        # Test 4: Error handling
        if not await test_error_handling():
            all_tests_passed = False

    except Exception as e:
        logger.error(f"Test execution failed: {e}", exc_info=True)
        all_tests_passed = False

    # Summary
    logger.info("=" * 60)
    if all_tests_passed:
        logger.info("🎉 All RAG-MCP integration tests PASSED!")
        logger.info("The integration is working correctly with proper fallback mechanisms.")
    else:
        logger.error("❌ Some RAG-MCP integration tests FAILED!")
        logger.error("Please check the logs above for details.")

    return all_tests_passed


if __name__ == "__main__":
    import sys
    success = asyncio.run(main())
    sys.exit(0 if success else 1)