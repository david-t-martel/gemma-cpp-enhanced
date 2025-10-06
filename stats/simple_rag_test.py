#!/usr/bin/env python3
"""
Simple test for RAG-MCP integration without complex dependencies.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


async def test_rag_integration_basic():
    """Test the RAG integration layer without complex setup."""
    logger.info("=== Testing RAG Integration Layer ===")

    try:
        # Import the RAG integration
        from src.agent.rag_integration import RAGClient, rag_context

        # Test with no MCP server (should use mock mode)
        async with rag_context(None) as rag_client:
            logger.info("Testing RAG client creation...")

            # Test document ingestion via integration layer
            logger.info("Testing document ingestion...")
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

            if doc_id:
                logger.info(f"✓ Document ingestion successful: {doc_id}")
            else:
                logger.error("✗ Document ingestion failed")
                return False

            # Test search via integration layer
            logger.info("Testing search...")
            search_results = await rag_client.search(
                query="RAG system vector embeddings",
                limit=3,
                threshold=0.6
            )

            if search_results:
                logger.info(f"✓ Search successful, found {len(search_results)} results")
                for i, result in enumerate(search_results):
                    content = result.get('content', '')[:80]
                    logger.info(f"  Result {i+1}: {content}...")
            else:
                logger.error("✗ Search returned no results")
                return False

            # Test memory operations via integration layer
            logger.info("Testing memory storage...")
            memory_stored = await rag_client.store_memory(
                content="User tested RAG-MCP integration successfully",
                memory_type="episodic"
            )

            if memory_stored:
                logger.info("✓ Memory storage successful")
            else:
                logger.error("✗ Memory storage failed")
                return False

            logger.info("Testing memory recall...")
            recalled_memories = await rag_client.recall_memory(
                query="RAG integration test",
                memory_type="episodic"
            )

            if recalled_memories:
                logger.info(f"✓ Memory recall successful, found {len(recalled_memories)} memories")
                for i, memory in enumerate(recalled_memories):
                    content = memory.get('content', '')[:80]
                    logger.info(f"  Memory {i+1}: {content}...")
            else:
                logger.error("✗ Memory recall returned no results")
                return False

            return True

    except Exception as e:
        logger.error(f"RAG integration test failed: {e}", exc_info=True)
        return False


async def test_rag_tools_direct():
    """Test RAG tools directly in isolation."""
    logger.info("=== Testing RAG Tools Directly ===")

    try:
        # Test document ingestion tool
        logger.info("Testing RAG ingestion tool...")
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
        Machine learning is a powerful subset of AI that learns patterns from data.

        ## Key Principles
        1. Data Quality: Ensure clean, representative data
        2. Feature Engineering: Transform raw data meaningfully
        3. Model Selection: Choose appropriate algorithms
        4. Validation: Use cross-validation for assessment
        5. Regularization: Prevent overfitting

        This document covers fundamental ML concepts.
        """

        result = await ingest_tool.execute(
            context,
            content=sample_doc,
            title="ML Best Practices Guide",
            source="test_integration",
            doc_type="markdown",
            tags=["machine-learning", "best-practices"],
            importance=0.8
        )

        if result.success:
            logger.info(f"✓ Document ingestion successful: {result.data.get('document_id')}")
            doc_id = result.data.get('document_id')
        else:
            logger.error(f"✗ Document ingestion failed: {result.error}")
            return False

        # Test search
        logger.info("Testing RAG search tool...")
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
                content = result['content'][:100]
                score = result['score']
                logger.info(f"  Result {i+1}: {content}... (score: {score:.3f})")
        else:
            logger.error(f"✗ Search failed: {search_result.error}")
            return False

        # Test memory storage
        logger.info("Testing RAG memory storage tool...")
        from src.infrastructure.tools.rag_tools import RagStoreMemoryTool

        memory_tool = RagStoreMemoryTool()
        memory_result = await memory_tool.execute(
            context,
            content="The user is interested in machine learning best practices",
            memory_type="episodic",
            importance=0.7,
            tags=["user-interest", "ml"],
            context="User interaction during RAG testing"
        )

        if memory_result.success:
            logger.info(f"✓ Memory storage successful: {memory_result.data.get('memory_id')}")
        else:
            logger.error(f"✗ Memory storage failed: {memory_result.error}")
            return False

        # Test memory recall
        logger.info("Testing RAG memory recall tool...")
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
                content = memory['content'][:100]
                importance = memory.get('importance', 0.0)
                logger.info(f"  Memory {i+1}: {content}... (importance: {importance:.3f})")
        else:
            logger.error(f"✗ Memory recall failed: {recall_result.error}")
            return False

        return True

    except Exception as e:
        logger.error(f"RAG tools test failed: {e}", exc_info=True)
        return False


async def main():
    """Run simple tests."""
    logger.info("Starting Simple RAG-MCP Integration Tests")

    all_tests_passed = True

    try:
        # Test 1: Integration layer
        if not await test_rag_integration_basic():
            all_tests_passed = False

        # Test 2: Direct tools
        if not await test_rag_tools_direct():
            all_tests_passed = False

    except Exception as e:
        logger.error(f"Test execution failed: {e}", exc_info=True)
        all_tests_passed = False

    # Summary
    logger.info("=" * 60)
    if all_tests_passed:
        logger.info("SUCCESS: All RAG-MCP integration tests PASSED!")
        logger.info("The integration is working correctly with mock implementations.")
    else:
        logger.error("FAILURE: Some RAG-MCP integration tests FAILED!")
        logger.error("Please check the logs above for details.")

    return all_tests_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)