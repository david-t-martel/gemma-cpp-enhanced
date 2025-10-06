#!/usr/bin/env python3
"""
RAG-MCP Integration Usage Demo

This script demonstrates how to use the RAG-MCP integration for:
1. Document ingestion
2. Vector-based search
3. Multi-tier memory storage and recall
4. Agent enhancement with RAG capabilities
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


async def demo_rag_client():
    """Demonstrate RAG client usage."""
    print("=" * 60)
    print("RAG-MCP Integration Demo")
    print("=" * 60)

    # Import RAG integration components
    from src.agent.rag_integration import rag_context

    # Create RAG client (using mock mode since Rust components may not be available)
    async with rag_context(None) as rag_client:
        print("\n1. Document Ingestion Demo")
        print("-" * 30)

        # Sample documents to ingest
        documents = [
            {
                "content": """
                # Python Best Practices Guide

                Python is a powerful, readable programming language. Here are key best practices:

                ## Code Style
                - Follow PEP 8 style guidelines
                - Use meaningful variable names
                - Write docstrings for functions and classes
                - Keep functions small and focused

                ## Performance
                - Use list comprehensions when appropriate
                - Leverage built-in functions and libraries
                - Profile before optimizing
                - Consider using generators for large datasets

                ## Testing
                - Write unit tests with pytest
                - Aim for high test coverage
                - Use mocking for external dependencies
                - Test edge cases and error conditions

                This guide helps developers write maintainable Python code.
                """,
                "metadata": {
                    "title": "Python Best Practices Guide",
                    "type": "documentation",
                    "source": "internal_guide",
                    "tags": ["python", "best-practices", "coding"]
                }
            },
            {
                "content": """
                # Machine Learning Pipeline Design

                Effective ML pipelines are crucial for production systems. Key components:

                ## Data Pipeline
                - Data validation and quality checks
                - Feature engineering and transformation
                - Data versioning and lineage tracking
                - Automated data preprocessing

                ## Model Pipeline
                - Experiment tracking and versioning
                - Automated model training and validation
                - Model registry for deployment
                - A/B testing for model comparison

                ## Monitoring
                - Data drift detection
                - Model performance monitoring
                - Alert systems for anomalies
                - Feedback loops for continuous improvement

                A well-designed pipeline ensures reliable ML systems.
                """,
                "metadata": {
                    "title": "ML Pipeline Design Guide",
                    "type": "documentation",
                    "source": "ml_team",
                    "tags": ["machine-learning", "mlops", "pipeline"]
                }
            }
        ]

        # Ingest documents
        doc_ids = []
        for i, doc in enumerate(documents):
            print(f"  Ingesting document {i+1}: {doc['metadata']['title']}")
            doc_id = await rag_client.ingest_document(
                content=doc["content"],
                metadata=doc["metadata"]
            )
            doc_ids.append(doc_id)
            print(f"    -> Document ID: {doc_id}")

        print(f"\n  Successfully ingested {len(doc_ids)} documents")

        print("\n2. Vector Search Demo")
        print("-" * 25)

        # Perform various searches
        search_queries = [
            "Python code style and formatting",
            "machine learning model deployment",
            "testing and quality assurance",
            "data pipeline monitoring"
        ]

        for query in search_queries:
            print(f"\n  Query: '{query}'")
            results = await rag_client.search(query, limit=2, threshold=0.3)

            if results:
                for j, result in enumerate(results):
                    content_preview = result.get('content', '')[:150] + "..."
                    score = result.get('metadata', {}).get('score', 'N/A')
                    print(f"    Result {j+1} (score: {score}): {content_preview}")
            else:
                print("    No results found")

        print("\n3. Memory Operations Demo")
        print("-" * 30)

        # Store different types of memories
        memories_to_store = [
            {
                "content": "User asked about Python best practices and code formatting",
                "type": "episodic",
                "context": "User interaction focused on Python development"
            },
            {
                "content": "System successfully processed ML pipeline documentation",
                "type": "semantic",
                "context": "Document processing completed without errors"
            },
            {
                "content": "Current session involves RAG system demonstration",
                "type": "working",
                "context": "Active demonstration session"
            }
        ]

        print("  Storing memories:")
        for memory in memories_to_store:
            success = await rag_client.store_memory(
                content=memory["content"],
                memory_type=memory["type"]
            )
            status = "✓" if success else "✗"
            print(f"    {status} {memory['type']}: {memory['content'][:60]}...")

        # Recall memories
        print("\n  Recalling memories:")
        recall_queries = [
            "Python development session",
            "ML pipeline processing",
            "demonstration session"
        ]

        for query in recall_queries:
            print(f"\n    Query: '{query}'")
            memories = await rag_client.recall_memory(query, limit=2)

            if memories:
                for k, memory in enumerate(memories):
                    content = memory.get('content', '')[:100] + "..."
                    mem_type = memory.get('metadata', {}).get('type', 'unknown')
                    print(f"      Memory {k+1} ({mem_type}): {content}")
            else:
                print("      No memories found")

        print("\n4. Integration Summary")
        print("-" * 25)
        print("  The RAG-MCP integration provides:")
        print("  • Document ingestion with metadata")
        print("  • Vector-based semantic search")
        print("  • Multi-tier memory management")
        print("  • Seamless fallback to mock implementations")
        print("  • MCP protocol compatibility")


async def demo_enhanced_agent():
    """Demonstrate agent enhancement with RAG."""
    print("\n" + "=" * 60)
    print("RAG-Enhanced Agent Demo")
    print("=" * 60)

    from src.agent.rag_integration import enhance_agent_with_rag

    # Simple mock agent
    class MockAgent:
        def solve(self, query: str, max_iterations: int = 5) -> str:
            return f"Base agent response to: {query}"

    # Enhance the agent with RAG
    base_agent = MockAgent()
    enhanced_agent = await enhance_agent_with_rag(base_agent)

    print("\n  Testing enhanced agent capabilities:")

    # Test queries that would benefit from RAG context
    test_queries = [
        "What are the key Python coding best practices?",
        "How should I design an ML pipeline for production?",
        "What testing strategies should I use for Python code?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n  Query {i}: {query}")
        try:
            # The enhanced agent will search for relevant context before responding
            response = await enhanced_agent.solve(query)
            print(f"    Response: {response[:200]}...")
        except Exception as e:
            # Fallback to base agent if RAG fails
            response = enhanced_agent.solve(query)
            print(f"    Fallback response: {response}")


def main():
    """Run the demo."""
    try:
        print("Starting RAG-MCP Integration Demo...")

        # Run the demos
        asyncio.run(demo_rag_client())
        asyncio.run(demo_enhanced_agent())

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)

        print("\nNext Steps:")
        print("1. Build the Rust RAG-Redis components for full functionality")
        print("2. Configure Redis for persistent storage")
        print("3. Set up embedding models for better semantic search")
        print("4. Integrate with your agent framework")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)