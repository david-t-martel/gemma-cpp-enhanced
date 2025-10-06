"""RAG-specific MCP tools for document ingestion and memory management."""

import asyncio
import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.domain.tools.base import BaseTool, ToolExecutionContext, ToolResult, tool
from src.domain.tools.schemas import (
    EnhancedToolSchema,
    ParameterSchema,
    SecurityLevel,
    ToolCategory,
    ToolType,
)
from src.shared.logging import get_logger

logger = get_logger(__name__)


class DocumentIngestionResult(BaseModel):
    """Result of document ingestion."""

    document_id: str
    chunks_created: int
    processing_time: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Vector search result."""

    id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryResult(BaseModel):
    """Memory storage/recall result."""

    id: str
    content: str
    memory_type: str
    importance: float
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


@tool(
    name="rag_ingest_document",
    description="Ingest a document into the RAG-Redis system",
    category=ToolCategory.DATA_PROCESSING,
)
class RagIngestDocumentTool(BaseTool):
    """Tool for ingesting documents into RAG system."""

    @property
    def schema(self) -> EnhancedToolSchema:
        return EnhancedToolSchema(
            name="rag_ingest_document",
            description="Ingest a document into the RAG-Redis system with embeddings and chunking",
            category=ToolCategory.DATA_PROCESSING,
            type=ToolType.BUILTIN,
            parameters=[
                ParameterSchema(
                    name="content",
                    type="string",
                    description="Document content to ingest",
                    required=True,
                    min_length=10,
                    max_length=1_000_000,  # 1MB max
                ),
                ParameterSchema(
                    name="title",
                    type="string",
                    description="Document title",
                    required=False,
                ),
                ParameterSchema(
                    name="source",
                    type="string",
                    description="Document source/origin",
                    required=False,
                ),
                ParameterSchema(
                    name="doc_type",
                    type="string",
                    description="Document type",
                    required=False,
                    default="text",
                    enum=["text", "markdown", "code", "pdf", "web", "email"],
                ),
                ParameterSchema(
                    name="tags",
                    type="array",
                    description="Document tags/categories",
                    required=False,
                ),
                ParameterSchema(
                    name="importance",
                    type="number",
                    description="Document importance score (0-1)",
                    required=False,
                    default=0.5,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ],
            security=SecurityLevel(level="standard", sandbox_required=False),
            idempotent=False,
            cacheable=False,
        )

    async def execute(self, context: ToolExecutionContext, **kwargs) -> ToolResult:
        """Execute document ingestion via RAG system."""
        try:
            content = kwargs["content"]
            title = kwargs.get("title", "Untitled Document")
            source = kwargs.get("source", "unknown")
            doc_type = kwargs.get("doc_type", "text")
            tags = kwargs.get("tags", [])
            importance = kwargs.get("importance", 0.5)

            start_time = asyncio.get_event_loop().time()

            # Prepare metadata for the RAG system
            metadata = {
                "title": title,
                "source": source,
                "type": doc_type,
                "tags": tags,
                "importance": importance,
                "context_id": context.execution_id,
                "agent_id": context.agent_id,
            }

            # Call RAG system via Rust/Python integration
            try:
                # Try to import the RAG system integration
                # For now, we'll create a mock implementation since Rust components may not be built
                try:
                    from rag_redis_system import RagSystem, Config
                    rag_available = True
                except ImportError:
                    logger.warning("RAG-Redis system not available, using mock implementation")
                    rag_available = False

                if rag_available:
                    # TODO: This should come from application config
                    config = Config.default()
                    rag_system = await RagSystem.new(config)

                    # Ingest document
                    document_id = await rag_system.handle_ingest_document(content, metadata)
                else:
                    # Mock implementation for testing
                    document_id = f"mock_doc_{abs(hash(content[:200])) % 100000}"
                    logger.info(f"Mock ingestion: created document {document_id}")

                execution_time = asyncio.get_event_loop().time() - start_time

                result_data = {
                    "document_id": document_id,
                    "content_length": len(content),
                    "processing_time": execution_time,
                    "metadata": metadata,
                    "chunks_estimated": max(1, len(content) // 512),  # Rough estimate
                    "mock_mode": not rag_available,
                }

                logger.info(f"Successfully ingested document {document_id} with {len(content)} characters")

                return ToolResult(
                    success=True,
                    data=result_data,
                    execution_time=execution_time,
                    context=context,
                )

            except Exception as e:
                logger.error(f"Document ingestion failed: {e}")
                return ToolResult(
                    success=False,
                    error=f"Document ingestion failed: {str(e)}",
                    execution_time=0,
                    context=context,
                )

        except Exception as e:
            logger.error(f"Document ingestion error: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=0,
                context=context,
            )


@tool(
    name="rag_search",
    description="Search for relevant documents in RAG system",
    category=ToolCategory.DATA_PROCESSING,
)
class RagSearchTool(BaseTool):
    """Tool for searching documents in RAG system."""

    @property
    def schema(self) -> EnhancedToolSchema:
        return EnhancedToolSchema(
            name="rag_search",
            description="Search for relevant documents using vector similarity",
            category=ToolCategory.DATA_PROCESSING,
            type=ToolType.BUILTIN,
            parameters=[
                ParameterSchema(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                    min_length=1,
                    max_length=1000,
                ),
                ParameterSchema(
                    name="limit",
                    type="integer",
                    description="Maximum number of results",
                    required=False,
                    default=5,
                    minimum=1,
                    maximum=50,
                ),
                ParameterSchema(
                    name="threshold",
                    type="number",
                    description="Minimum similarity threshold",
                    required=False,
                    default=0.7,
                    minimum=0.0,
                    maximum=1.0,
                ),
                ParameterSchema(
                    name="doc_types",
                    type="array",
                    description="Filter by document types",
                    required=False,
                ),
            ],
            security=SecurityLevel(level="standard", sandbox_required=False),
            idempotent=True,
            cacheable=True,
        )

    async def execute(self, context: ToolExecutionContext, **kwargs) -> ToolResult:
        """Execute vector search via RAG system."""
        try:
            query = kwargs["query"]
            limit = kwargs.get("limit", 5)
            threshold = kwargs.get("threshold", 0.7)
            doc_types = kwargs.get("doc_types", [])

            start_time = asyncio.get_event_loop().time()

            try:
                # Try to import the RAG system integration
                try:
                    from rag_redis_system import RagSystem, Config
                    rag_available = True
                except ImportError:
                    logger.warning("RAG-Redis system not available, using mock search")
                    rag_available = False

                if rag_available:
                    # TODO: This should come from application config
                    config = Config.default()
                    rag_system = await RagSystem.new(config)

                    # Perform search
                    search_results = await rag_system.handle_search(query, limit, threshold)

                    # Filter by document types if specified
                    if doc_types:
                        search_results = [
                            result
                            for result in search_results
                            if result.get("metadata", {}).get("type") in doc_types
                        ]

                    # Format results
                    formatted_results = []
                    for result in search_results:
                        formatted_results.append({
                            "id": result["id"],
                            "content": result["content"],
                            "score": result["score"],
                            "metadata": result.get("metadata", {}),
                        })
                else:
                    # Mock search implementation
                    formatted_results = []
                    mock_results = [
                        f"Mock search result 1 for '{query}': This is a simulated document that matches your query about {query}.",
                        f"Mock search result 2 for '{query}': Another relevant document containing information about {query} and related topics.",
                        f"Mock search result 3 for '{query}': Additional content that would be retrieved by vector search for {query}."
                    ]

                    for i, content in enumerate(mock_results[:limit]):
                        formatted_results.append({
                            "id": f"mock_search_{abs(hash(query + str(i))) % 10000}",
                            "content": content,
                            "score": max(0.5, 0.9 - i * 0.1),  # Decreasing scores
                            "metadata": {"type": "mock", "source": "mock_search"},
                        })

                execution_time = asyncio.get_event_loop().time() - start_time

                result_data = {
                    "query": query,
                    "results": formatted_results,
                    "total_results": len(formatted_results),
                    "search_time": execution_time,
                    "threshold_used": threshold,
                    "mock_mode": not rag_available,
                }

                logger.info(f"RAG search for '{query}' returned {len(formatted_results)} results")

                return ToolResult(
                    success=True,
                    data=result_data,
                    execution_time=execution_time,
                    context=context,
                )

            except Exception as e:
                logger.error(f"Search failed: {e}")
                return ToolResult(
                    success=False,
                    error=f"Search failed: {str(e)}",
                    execution_time=0,
                    context=context,
                )

        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=0,
                context=context,
            )


@tool(
    name="rag_store_memory",
    description="Store a memory in the RAG system",
    category=ToolCategory.DATA_PROCESSING,
)
class RagStoreMemoryTool(BaseTool):
    """Tool for storing memories in RAG system."""

    @property
    def schema(self) -> EnhancedToolSchema:
        return EnhancedToolSchema(
            name="rag_store_memory",
            description="Store a memory in the multi-tier RAG memory system",
            category=ToolCategory.DATA_PROCESSING,
            type=ToolType.BUILTIN,
            parameters=[
                ParameterSchema(
                    name="content",
                    type="string",
                    description="Memory content to store",
                    required=True,
                    min_length=1,
                    max_length=10000,
                ),
                ParameterSchema(
                    name="memory_type",
                    type="string",
                    description="Type of memory",
                    required=False,
                    default="long_term",
                    enum=["working", "short_term", "long_term", "episodic", "semantic"],
                ),
                ParameterSchema(
                    name="importance",
                    type="number",
                    description="Memory importance score (0-1)",
                    required=False,
                    default=0.5,
                    minimum=0.0,
                    maximum=1.0,
                ),
                ParameterSchema(
                    name="tags",
                    type="array",
                    description="Memory tags",
                    required=False,
                ),
                ParameterSchema(
                    name="context",
                    type="string",
                    description="Additional context for the memory",
                    required=False,
                ),
            ],
            security=SecurityLevel(level="standard", sandbox_required=False),
            idempotent=False,
            cacheable=False,
        )

    async def execute(self, context: ToolExecutionContext, **kwargs) -> ToolResult:
        """Execute memory storage via RAG system."""
        try:
            content = kwargs["content"]
            memory_type = kwargs.get("memory_type", "long_term")
            importance = kwargs.get("importance", 0.5)
            tags = kwargs.get("tags", [])
            memory_context = kwargs.get("context", "")

            start_time = asyncio.get_event_loop().time()

            try:
                # Try to import the RAG system integration
                try:
                    from rag_redis_system import RagSystem, Config
                    rag_available = True
                except ImportError:
                    logger.warning("RAG-Redis system not available, using mock memory storage")
                    rag_available = False

                if rag_available:
                    # TODO: This should come from application config
                    config = Config.default()
                    rag_system = await RagSystem.new(config)

                    # Store memory with enhanced content
                    enhanced_content = content
                    if memory_context:
                        enhanced_content = f"{content}\n\nContext: {memory_context}"
                    if tags:
                        enhanced_content = f"{enhanced_content}\n\nTags: {', '.join(tags)}"

                    memory_id = await rag_system.handle_store_memory(
                        enhanced_content, memory_type, importance
                    )
                else:
                    # Mock memory storage
                    memory_id = f"mock_memory_{abs(hash(content + memory_type)) % 100000}"
                    logger.info(f"Mock memory storage: created {memory_type} memory {memory_id}")

                execution_time = asyncio.get_event_loop().time() - start_time

                result_data = {
                    "memory_id": memory_id,
                    "memory_type": memory_type,
                    "content_length": len(content),
                    "importance": importance,
                    "tags": tags,
                    "storage_time": execution_time,
                    "mock_mode": not rag_available,
                }

                logger.info(f"Successfully stored {memory_type} memory {memory_id}")

                return ToolResult(
                    success=True,
                    data=result_data,
                    execution_time=execution_time,
                    context=context,
                )

            except Exception as e:
                logger.error(f"Memory storage failed: {e}")
                return ToolResult(
                    success=False,
                    error=f"Memory storage failed: {str(e)}",
                    execution_time=0,
                    context=context,
                )

        except Exception as e:
            logger.error(f"Memory storage error: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=0,
                context=context,
            )


@tool(
    name="rag_recall_memory",
    description="Recall memories from RAG system",
    category=ToolCategory.DATA_PROCESSING,
)
class RagRecallMemoryTool(BaseTool):
    """Tool for recalling memories from RAG system."""

    @property
    def schema(self) -> EnhancedToolSchema:
        return EnhancedToolSchema(
            name="rag_recall_memory",
            description="Recall relevant memories from the multi-tier memory system",
            category=ToolCategory.DATA_PROCESSING,
            type=ToolType.BUILTIN,
            parameters=[
                ParameterSchema(
                    name="query",
                    type="string",
                    description="Query to find relevant memories",
                    required=True,
                    min_length=1,
                    max_length=1000,
                ),
                ParameterSchema(
                    name="memory_type",
                    type="string",
                    description="Filter by memory type",
                    required=False,
                    enum=["working", "short_term", "long_term", "episodic", "semantic"],
                ),
                ParameterSchema(
                    name="limit",
                    type="integer",
                    description="Maximum number of memories to recall",
                    required=False,
                    default=5,
                    minimum=1,
                    maximum=20,
                ),
                ParameterSchema(
                    name="min_importance",
                    type="number",
                    description="Minimum importance threshold",
                    required=False,
                    default=0.0,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ],
            security=SecurityLevel(level="standard", sandbox_required=False),
            idempotent=True,
            cacheable=True,
        )

    async def execute(self, context: ToolExecutionContext, **kwargs) -> ToolResult:
        """Execute memory recall via RAG system."""
        try:
            query = kwargs["query"]
            memory_type = kwargs.get("memory_type")
            limit = kwargs.get("limit", 5)
            min_importance = kwargs.get("min_importance", 0.0)

            start_time = asyncio.get_event_loop().time()

            try:
                # Try to import the RAG system integration
                try:
                    from rag_redis_system import RagSystem, Config
                    rag_available = True
                except ImportError:
                    logger.warning("RAG-Redis system not available, using mock memory recall")
                    rag_available = False

                if rag_available:
                    # TODO: This should come from application config
                    config = Config.default()
                    rag_system = await RagSystem.new(config)

                    # Recall memories
                    memory_results = await rag_system.handle_recall_memory(
                        query, memory_type, limit
                    )

                    # Filter by importance if specified
                    if min_importance > 0.0:
                        memory_results = [
                            memory
                            for memory in memory_results
                            if memory.get("importance", 0.0) >= min_importance
                        ]

                    # Format results
                    formatted_memories = []
                    for memory in memory_results:
                        formatted_memories.append({
                            "id": memory["id"],
                            "content": memory["content"],
                            "memory_type": memory["memory_type"],
                            "importance": memory.get("importance", 0.0),
                            "created_at": memory.get("created_at"),
                            "metadata": memory.get("metadata", {}),
                        })
                else:
                    # Mock memory recall
                    from datetime import datetime
                    formatted_memories = []
                    mock_memories = [
                        f"Mock {memory_type or 'general'} memory 1 related to '{query}': This is a simulated memory that would be retrieved based on semantic similarity.",
                        f"Mock {memory_type or 'general'} memory 2 about '{query}': Another relevant memory containing contextual information about {query}.",
                    ]

                    for i, content in enumerate(mock_memories[:limit]):
                        if min_importance <= 0.6:  # Mock importance threshold
                            formatted_memories.append({
                                "id": f"mock_mem_{abs(hash(query + str(i))) % 10000}",
                                "content": content,
                                "memory_type": memory_type or "general",
                                "importance": max(min_importance, 0.6 + i * 0.1),
                                "created_at": datetime.now().isoformat(),
                                "metadata": {"type": "mock", "source": "mock_recall"},
                            })

                execution_time = asyncio.get_event_loop().time() - start_time

                result_data = {
                    "query": query,
                    "memories": formatted_memories,
                    "total_memories": len(formatted_memories),
                    "memory_type_filter": memory_type,
                    "recall_time": execution_time,
                    "mock_mode": not rag_available,
                }

                logger.info(f"Recalled {len(formatted_memories)} memories for query: '{query}'")

                return ToolResult(
                    success=True,
                    data=result_data,
                    execution_time=execution_time,
                    context=context,
                )

            except Exception as e:
                logger.error(f"Memory recall failed: {e}")
                return ToolResult(
                    success=False,
                    error=f"Memory recall failed: {str(e)}",
                    execution_time=0,
                    context=context,
                )

        except Exception as e:
            logger.error(f"Memory recall error: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=0,
                context=context,
            )


# Helper function to register RAG tools
async def register_rag_tools():
    """Register all RAG tools with the global registry."""
    from src.domain.tools.base import get_global_registry

    registry = get_global_registry()

    tools = [
        RagIngestDocumentTool(),
        RagSearchTool(),
        RagStoreMemoryTool(),
        RagRecallMemoryTool(),
    ]

    for tool in tools:
        await registry.register(tool)

    logger.info(f"Registered {len(tools)} RAG tools")
    return registry