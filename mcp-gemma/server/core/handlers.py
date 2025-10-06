"""
Request handlers implementing Chain of Responsibility pattern.
Each handler has a single responsibility and can be composed.
"""

import json
from typing import Any, Dict, List, Optional

from mcp.types import Tool

from .contracts import (
    GenerationRequest,
    IGenerationService,
    IMemoryService,
    IMetricsService,
    IModelService,
    IRequestHandler,
    MemoryEntry,
    ModelInfo,
)


class GenerationHandler(IRequestHandler):
    """Handler for text generation requests."""

    def __init__(
        self,
        generation_service: IGenerationService,
        metrics_service: Optional[IMetricsService] = None,
    ):
        self.generation_service = generation_service
        self.metrics_service = metrics_service

    def get_tools(self) -> List[Tool]:
        """Get tools provided by this handler."""
        return [
            Tool(
                name="generate_text",
                description="Generate text using the Gemma model",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to generate text from",
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens to generate",
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature (0.0-2.0)",
                        },
                        "stream": {
                            "type": "boolean",
                            "description": "Enable streaming response",
                            "default": False,
                        },
                        "context": {
                            "type": "array",
                            "description": "Conversation context",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["prompt"],
                },
            ),
        ]

    def can_handle(self, method: str) -> bool:
        """Check if this handler can handle the method."""
        return method == "generate_text"

    async def handle(self, method: str, params: Dict[str, Any]) -> Any:
        """Handle the request."""
        if method == "generate_text":
            request = GenerationRequest(
                prompt=params["prompt"],
                max_tokens=params.get("max_tokens"),
                temperature=params.get("temperature"),
                stream=params.get("stream", False),
                context=params.get("context"),
            )

            if request.stream:
                # For streaming, collect all chunks and return
                chunks = []
                async for chunk in self.generation_service.generate_stream(request):
                    chunks.append(chunk)
                return "".join(chunks)
            else:
                response = await self.generation_service.generate(request)
                return response.text

        raise ValueError(f"Cannot handle method: {method}")


class ModelHandler(IRequestHandler):
    """Handler for model management requests."""

    def __init__(self, model_service: IModelService):
        self.model_service = model_service

    def get_tools(self) -> List[Tool]:
        """Get tools provided by this handler."""
        return [
            Tool(
                name="switch_model",
                description="Switch to a different model",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_path": {
                            "type": "string",
                            "description": "Path to the model file",
                        },
                        "tokenizer_path": {
                            "type": "string",
                            "description": "Path to the tokenizer file (optional)",
                        },
                    },
                    "required": ["model_path"],
                },
            ),
            Tool(
                name="list_models",
                description="List available models",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            Tool(
                name="get_current_model",
                description="Get information about the current model",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
        ]

    def can_handle(self, method: str) -> bool:
        """Check if this handler can handle the method."""
        return method in ["switch_model", "list_models", "get_current_model"]

    async def handle(self, method: str, params: Dict[str, Any]) -> Any:
        """Handle the request."""
        if method == "switch_model":
            model_info = ModelInfo(
                path=params["model_path"],
                name=params["model_path"].split("/")[-1].split(".")[0],
                size=0,  # Will be determined by repository
                type="unknown",
                tokenizer_path=params.get("tokenizer_path"),
            )
            await self.model_service.switch_model(model_info)
            return f"Switched to model: {model_info.name}"

        elif method == "list_models":
            models = self.model_service.list_available_models()
            return json.dumps(
                [{"name": m.name, "path": m.path, "size": m.size, "type": m.type} for m in models],
                indent=2,
            )

        elif method == "get_current_model":
            model = self.model_service.get_current_model()
            return json.dumps(
                {
                    "name": model.name,
                    "path": model.path,
                    "type": model.type,
                    "tokenizer_path": model.tokenizer_path,
                },
                indent=2,
            )

        raise ValueError(f"Cannot handle method: {method}")


class MemoryHandler(IRequestHandler):
    """Handler for memory management requests."""

    def __init__(self, memory_service: IMemoryService):
        self.memory_service = memory_service

    def get_tools(self) -> List[Tool]:
        """Get tools provided by this handler."""
        return [
            Tool(
                name="store_memory",
                description="Store information in memory for later retrieval",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Memory key identifier",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to store",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional metadata",
                        },
                    },
                    "required": ["key", "content"],
                },
            ),
            Tool(
                name="retrieve_memory",
                description="Retrieve stored memory by key",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Memory key identifier",
                        }
                    },
                    "required": ["key"],
                },
            ),
            Tool(
                name="search_memory",
                description="Search memory by content similarity",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results to return",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="delete_memory",
                description="Delete a memory entry",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Memory key to delete",
                        }
                    },
                    "required": ["key"],
                },
            ),
            Tool(
                name="list_memory_keys",
                description="List all memory keys",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
        ]

    def can_handle(self, method: str) -> bool:
        """Check if this handler can handle the method."""
        return method in [
            "store_memory",
            "retrieve_memory",
            "search_memory",
            "delete_memory",
            "list_memory_keys",
        ]

    async def handle(self, method: str, params: Dict[str, Any]) -> Any:
        """Handle the request."""
        if method == "store_memory":
            entry = MemoryEntry(
                key=params["key"],
                content=params["content"],
                metadata=params.get("metadata", {}),
                timestamp=0,  # Will be set by service
                id="",  # Will be generated by service
            )
            entry_id = await self.memory_service.store(entry)
            return f"Memory stored with key: {params['key']}, id: {entry_id}"

        elif method == "retrieve_memory":
            entry = await self.memory_service.retrieve(params["key"])
            if entry:
                return json.dumps(
                    {
                        "key": entry.key,
                        "content": entry.content,
                        "metadata": entry.metadata,
                        "timestamp": entry.timestamp,
                        "id": entry.id,
                    },
                    indent=2,
                )
            else:
                return json.dumps({"error": f"No memory found for key: {params['key']}"})

        elif method == "search_memory":
            entries = await self.memory_service.search(params["query"], params.get("limit", 10))
            return json.dumps(
                {
                    "query": params["query"],
                    "results": [
                        {
                            "key": e.key,
                            "content": e.content,
                            "metadata": e.metadata,
                            "timestamp": e.timestamp,
                        }
                        for e in entries
                    ],
                    "total": len(entries),
                },
                indent=2,
            )

        elif method == "delete_memory":
            success = await self.memory_service.delete(params["key"])
            if success:
                return f"Memory entry deleted: {params['key']}"
            else:
                return f"Memory entry not found: {params['key']}"

        elif method == "list_memory_keys":
            keys = await self.memory_service.list_keys()
            return json.dumps({"keys": keys, "total": len(keys)}, indent=2)

        raise ValueError(f"Cannot handle method: {method}")


class MetricsHandler(IRequestHandler):
    """Handler for metrics and monitoring requests."""

    def __init__(self, metrics_service: IMetricsService):
        self.metrics_service = metrics_service

    def get_tools(self) -> List[Tool]:
        """Get tools provided by this handler."""
        return [
            Tool(
                name="get_metrics",
                description="Get server performance metrics",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            Tool(
                name="health_check",
                description="Perform server health check",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            Tool(
                name="reset_metrics",
                description="Reset performance metrics",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
        ]

    def can_handle(self, method: str) -> bool:
        """Check if this handler can handle the method."""
        return method in ["get_metrics", "health_check", "reset_metrics"]

    async def handle(self, method: str, params: Dict[str, Any]) -> Any:
        """Handle the request."""
        if method == "get_metrics":
            snapshot = self.metrics_service.get_metrics()
            return json.dumps(
                {
                    "total_requests": snapshot.total_requests,
                    "total_tokens": snapshot.total_tokens,
                    "average_response_time": snapshot.avg_response_time,
                    "requests_per_minute": snapshot.requests_per_minute,
                    "uptime_seconds": snapshot.uptime_seconds,
                    "timestamp": snapshot.timestamp,
                },
                indent=2,
            )

        elif method == "health_check":
            health = await self.metrics_service.health_check()
            return json.dumps(health, indent=2)

        elif method == "reset_metrics":
            self.metrics_service.reset_metrics()
            return "Metrics reset successfully"

        raise ValueError(f"Cannot handle method: {method}")
