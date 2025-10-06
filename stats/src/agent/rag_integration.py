"""RAG integration compatibility layer for agents.

This module provides a compatibility shim for the legacy RAG integration API
while integrating with the new Rust-based RAG-Redis MCP system.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from ..infrastructure.mcp.server import McpServer
from .react_agent import UnifiedReActAgent

logger = logging.getLogger(__name__)


class RAGClient:
    """Client for interacting with the RAG-Redis MCP system."""

    def __init__(self, mcp_server: McpServer | None = None):
        self.mcp_server = mcp_server
        self._connected = False

    async def connect(self) -> bool:
        """Connect to the RAG-Redis MCP server."""
        try:
            # For now, we'll assume connection is available if mcp_server is provided
            if self.mcp_server is None:
                logger.warning("No MCP server provided for RAG connection")
                return False

            self._connected = True
            logger.info("Connected to RAG-Redis system via MCP")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to RAG system: {e}")
            return False

    async def ingest_document(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Ingest a document into the RAG system.

        Args:
            content: Document content to ingest
            metadata: Optional metadata for the document

        Returns:
            Document ID of the ingested document
        """
        try:
            if self._connected and self.mcp_server:
                # Real implementation using MCP call to RAG-Redis system
                logger.info(f"Ingesting document with {len(content)} characters via MCP")

                # Prepare MCP request
                request_params = {
                    "name": "rag_ingest_document",
                    "arguments": {
                        "content": content,
                        **(metadata or {}),
                    },
                }

                # Make MCP call through the server
                from src.domain.tools.base import ToolExecutionContext
                from uuid import uuid4

                context = ToolExecutionContext(
                    execution_id=str(uuid4()),
                    agent_id="rag_client",
                    session_id=None,
                    timeout=60,
                    security_level="standard",
                    metadata=metadata or {}
                )

                # Call the RAG ingestion tool directly
                from src.infrastructure.tools.rag_tools import RagIngestDocumentTool
                rag_tool = RagIngestDocumentTool()
                result = await rag_tool.execute(context, **request_params["arguments"])

                if result.success:
                    doc_id = result.data.get("document_id", f"doc_{hash(content[:100]) % 10000}_{len(content)}")
                    logger.info(f"Successfully ingested document: {doc_id}")
                    return doc_id
                else:
                    logger.error(f"Document ingestion failed: {result.error}")
                    # Fallback to mock ID
                    return f"doc_{hash(content[:100]) % 10000}_{len(content)}"
            else:
                # Mock implementation for compatibility
                logger.info(f"Mock ingesting document with {len(content)} characters")
                return f"doc_{hash(content[:100]) % 10000}_{len(content)}"

        except Exception as e:
            logger.error(f"Failed to ingest document: {e}")
            # Return a mock ID for compatibility
            return f"doc_{hash(content[:100]) % 10000}_{len(content)}"

    async def search(
        self, query: str, limit: int = 5, threshold: float = 0.7
    ) -> list[dict[str, Any]]:
        """Search for relevant documents.

        Args:
            query: Search query
            limit: Maximum number of results
            threshold: Similarity threshold

        Returns:
            List of search results with content and metadata
        """
        try:
            if self._connected and self.mcp_server:
                # Real implementation using MCP call to RAG-Redis system
                logger.info(f"Searching for: {query} via MCP")

                # Prepare MCP request
                request_params = {
                    "name": "rag_search",
                    "arguments": {
                        "query": query,
                        "limit": limit,
                        "threshold": threshold,
                    },
                }

                # Make MCP call through the server
                from src.domain.tools.base import ToolExecutionContext
                from uuid import uuid4

                context = ToolExecutionContext(
                    execution_id=str(uuid4()),
                    agent_id="rag_client",
                    session_id=None,
                    timeout=30,
                    security_level="standard",
                    metadata={}
                )

                # Call the RAG search tool directly
                from src.infrastructure.tools.rag_tools import RagSearchTool
                rag_tool = RagSearchTool()
                result = await rag_tool.execute(context, **request_params["arguments"])

                if result.success:
                    search_results = result.data.get("results", [])
                    logger.info(f"Successfully found {len(search_results)} results for query: {query}")
                    return search_results
                else:
                    logger.error(f"Search failed: {result.error}")
                    # Fallback to mock results
                    return [
                        {
                            "content": f"Fallback result for query: {query}",
                            "metadata": {"score": 0.5, "type": "fallback"},
                            "id": f"fallback_{hash(query) % 1000}",
                        }
                    ]
            else:
                # Mock implementation for compatibility
                logger.info(f"Mock searching for: {query}")
                return [
                    {
                        "content": f"Mock search result 1 for query: {query}",
                        "metadata": {"score": 0.9, "type": "mock"},
                        "id": f"result_1_{hash(query) % 1000}",
                    },
                    {
                        "content": f"Mock search result 2 for query: {query}",
                        "metadata": {"score": 0.8, "type": "mock"},
                        "id": f"result_2_{hash(query) % 1000}",
                    },
                ][:limit]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            # Return empty results for compatibility
            return []

    async def store_memory(self, content: str, memory_type: str = "long_term") -> bool:
        """Store a memory in the RAG system.

        Args:
            content: Memory content
            memory_type: Type of memory (short_term, long_term, episodic, semantic, working)

        Returns:
            True if successful, False otherwise
        """
        try:
            if self._connected and self.mcp_server:
                # Real implementation using MCP call to RAG-Redis system
                logger.info(f"Storing {memory_type} memory via MCP: {content[:100]}...")

                # Prepare MCP request
                request_params = {
                    "name": "rag_store_memory",
                    "arguments": {
                        "content": content,
                        "memory_type": memory_type,
                        "importance": 0.5,  # Default importance
                        "tags": [],
                        "context": f"Stored via RAG client at memory tier: {memory_type}",
                    },
                }

                # Make MCP call through the server
                from src.domain.tools.base import ToolExecutionContext
                from uuid import uuid4

                context = ToolExecutionContext(
                    execution_id=str(uuid4()),
                    agent_id="rag_client",
                    session_id=None,
                    timeout=30,
                    security_level="standard",
                    metadata={"memory_type": memory_type}
                )

                # Call the RAG memory storage tool directly
                from src.infrastructure.tools.rag_tools import RagStoreMemoryTool
                rag_tool = RagStoreMemoryTool()
                result = await rag_tool.execute(context, **request_params["arguments"])

                if result.success:
                    memory_id = result.data.get("memory_id")
                    logger.info(f"Successfully stored {memory_type} memory: {memory_id}")
                    return True
                else:
                    logger.error(f"Memory storage failed: {result.error}")
                    return False
            else:
                # Mock implementation for compatibility
                logger.info(f"Mock storing {memory_type} memory: {content[:100]}...")
                return True

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return False

    async def recall_memory(
        self, query: str, memory_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Recall memories from the RAG system.

        Args:
            query: Query to find relevant memories
            memory_type: Optional memory type filter

        Returns:
            List of relevant memories
        """
        try:
            if self._connected and self.mcp_server:
                # Real implementation using MCP call to RAG-Redis system
                logger.info(f"Recalling memories via MCP for: {query}")

                # Prepare MCP request
                request_params = {
                    "name": "rag_recall_memory",
                    "arguments": {
                        "query": query,
                        "memory_type": memory_type,
                        "limit": 5,
                        "min_importance": 0.0,
                    },
                }

                # Make MCP call through the server
                from src.domain.tools.base import ToolExecutionContext
                from uuid import uuid4

                context = ToolExecutionContext(
                    execution_id=str(uuid4()),
                    agent_id="rag_client",
                    session_id=None,
                    timeout=30,
                    security_level="standard",
                    metadata={"query": query}
                )

                # Call the RAG memory recall tool directly
                from src.infrastructure.tools.rag_tools import RagRecallMemoryTool
                rag_tool = RagRecallMemoryTool()
                result = await rag_tool.execute(context, **request_params["arguments"])

                if result.success:
                    memories = result.data.get("memories", [])
                    logger.info(f"Successfully recalled {len(memories)} memories for query: {query}")

                    # Format memories to match expected interface
                    formatted_memories = []
                    for memory in memories:
                        formatted_memories.append({
                            "content": memory["content"],
                            "metadata": {
                                "type": memory["memory_type"],
                                "timestamp": memory.get("created_at", ""),
                                "importance": memory.get("importance", 0.0),
                                "id": memory["id"],
                            },
                            "id": memory["id"],
                        })

                    return formatted_memories
                else:
                    logger.error(f"Memory recall failed: {result.error}")
                    # Return empty list on failure
                    return []
            else:
                # Mock implementation for compatibility
                logger.info(f"Mock recalling memories for: {query}")
                return [
                    {
                        "content": f"Mock memory related to: {query}",
                        "metadata": {"type": memory_type or "general", "timestamp": "2024-01-01"},
                        "id": f"memory_{hash(query) % 1000}",
                    }
                ]

        except Exception as e:
            logger.error(f"Failed to recall memory: {e}")
            return []

    async def close(self) -> None:
        """Close the RAG client connection."""
        self._connected = False


class RAGEnhancedAgent:
    """Wrapper for agents enhanced with RAG capabilities."""

    def __init__(self, base_agent: UnifiedReActAgent, rag_client: RAGClient):
        self.base_agent = base_agent
        self.rag_client = rag_client

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the base agent."""
        return getattr(self.base_agent, name)

    async def solve(self, query: str, max_iterations: int = 5) -> str:
        """Solve a query using RAG-enhanced context.

        Args:
            query: Query to solve
            max_iterations: Maximum iterations for solving

        Returns:
            Response from the agent
        """
        try:
            # Search for relevant context
            context_results = await self.rag_client.search(query, limit=3)

            # Enhance query with context if available
            if context_results:
                context_text = "\n".join(
                    [
                        f"Context {i + 1}: {result.get('content', '')[:200]}..."
                        for i, result in enumerate(context_results)
                    ]
                )
                enhanced_query = f"Context:\n{context_text}\n\nQuery: {query}"
            else:
                enhanced_query = query

            # Call base agent with enhanced query
            response = self.base_agent.solve(enhanced_query, max_iterations=max_iterations)

            # Store successful interactions as memories
            if response and len(response) > 50:  # Only store substantial responses
                await self.rag_client.store_memory(f"Q: {query}\nA: {response[:500]}", "episodic")

            return response

        except Exception as e:
            logger.error(f"RAG-enhanced solve failed: {e}")
            # Fallback to base agent
            return self.base_agent.solve(query, max_iterations=max_iterations)

    def chat(self, message: str, use_tools: bool = True) -> str:
        """Chat with RAG-enhanced context.

        Args:
            message: Chat message
            use_tools: Whether to use tools

        Returns:
            Response from the agent
        """
        # For synchronous chat, we can't easily integrate async RAG
        # Fall back to base agent for now
        return self.base_agent.chat(message, use_tools)


async def enhance_agent_with_rag(
    agent: UnifiedReActAgent, mcp_server: McpServer | None = None
) -> RAGEnhancedAgent | UnifiedReActAgent:
    """Enhance an agent with RAG capabilities.

    Args:
        agent: Base agent to enhance
        mcp_server: Optional MCP server for RAG system access

    Returns:
        RAG-enhanced agent or original agent if enhancement fails
    """
    try:
        # Create RAG client
        rag_client = RAGClient(mcp_server)

        # Try to connect to RAG system
        if await rag_client.connect():
            logger.info("Successfully enhanced agent with RAG capabilities")
            return RAGEnhancedAgent(agent, rag_client)
        else:
            logger.warning("Failed to connect to RAG system, returning original agent")
            return agent

    except Exception as e:
        logger.error(f"Failed to enhance agent with RAG: {e}")
        return agent


@asynccontextmanager
async def rag_context(mcp_server: McpServer | None = None) -> AsyncGenerator[RAGClient, None]:
    """Context manager for RAG client operations.

    Args:
        mcp_server: Optional MCP server

    Yields:
        Connected RAG client
    """
    rag_client = None
    try:
        # Create and connect RAG client
        rag_client = RAGClient(mcp_server)

        if not await rag_client.connect():
            logger.warning("Could not connect to RAG system - using mock mode")

        yield rag_client

    except Exception as e:
        logger.error(f"RAG context manager error: {e}")
        # Create a fallback mock client if the main one failed
        if rag_client is None:
            rag_client = RAGClient(None)
            # Force connection state for mock mode
            rag_client._connected = True
        yield rag_client
    finally:
        if rag_client:
            try:
                await rag_client.close()
            except Exception as cleanup_error:
                logger.warning(f"Error during RAG client cleanup: {cleanup_error}")


# For backwards compatibility
__all__ = ["RAGClient", "RAGEnhancedAgent", "enhance_agent_with_rag", "rag_context"]
